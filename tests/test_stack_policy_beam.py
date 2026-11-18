import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from wandering_light import stack_beam
from wandering_light import stack_policy as policy
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve, step_all
from wandering_light.stack_machine import FALSE, MARK, NIL, TRUE, Int, Lst, Sym
from wandering_light.stack_pbe import BY_NAME, OP_BY_NAME, build_task, draw_task
from wandering_light.stack_pbe_eval import grade

TINY = PolicyArchitecture(4, 6, 8)


@pytest.fixture(scope="module")
def sum_task():
    behaviour = BY_NAME["sum"]
    return build_task(behaviour, draw_task(behaviour, random.Random(17), 3, held_out=2))


def test_value_vocabulary_and_caps():
    names = tuple(op.name for op in stack_beam.OPS)
    assert names == policy.OP_NAMES
    assert {op: i for i, op in enumerate(names)} == policy.OP_INDEX
    ints = tuple(Int(i) for i in range(-64, 65))
    symbols = tuple(Sym(op) for op in names)
    assert policy.encode_stack(symbols + ints, 200) == list(range(9, policy.VOCAB - 10))
    digits = policy.VOCAB - 10
    stack = (TRUE, FALSE, MARK, Lst((NIL, Int(65), Int(-907), Sym("dup"))))
    tokens = [
        policy.TRUE_TOKEN,
        policy.FALSE_TOKEN,
        policy.MARK_TOKEN,
        policy.OPEN,
        policy.OPEN,
        policy.CLOSE,
        policy.BIG,
        digits + 6,
        digits + 5,
        policy.BIG,
        policy.NEG,
        digits + 9,
        digits,
        digits + 7,
        9 + names.index("dup"),
        policy.CLOSE,
    ]
    assert policy.encode_stack(stack, 100) == tokens
    for cap in range(1, len(tokens) + 1):
        assert policy.encode_stack(stack, cap) == tokens[-cap:]
    assert policy.encode_stack(stack, 0) == []
    assert policy.encode_stack((), 100) == []
    assert policy.encode_targets([ints])[0] == policy.encode_stack(ints, 64)
    long = symbols * 8
    assert policy.encode_item(long, tokens) == [
        *policy.encode_stack(long, 160),
        policy.SEP,
        *tokens,
    ]


def test_replay_rows_use_each_shown_example(sum_task):
    task = replace(sum_task, held_out_inputs=("(",), held_out_outputs=("(",))
    rows = policy.witness_rows([task, task])
    targets = policy.encode_targets(task.outputs)
    assert len(rows) == 2 * task.length
    for index, row in enumerate(rows):
        task_id, step = divmod(index, task.length)
        assert (row.task, row.step, row.remaining, row.label) == (
            task_id,
            step,
            task.length - step,
            policy.OP_INDEX[task.program[step]],
        )
        assert len(row.items) == task.examples
        assert row.items.dtype == np.int16
    # After "0 [ add", each shown stack still has a mark and a captured symbol.
    step, product, name = list(policy.replay_witness(task))[3]
    assert (step, name) == (3, "]")
    for i, stack in enumerate(product):
        assert stack[-2:] == (MARK, Sym("add"))
        expected = policy.encode_item(stack, targets[i])
        np.testing.assert_array_equal(rows[step].items[i, : len(expected)], expected)
        assert (rows[step].items[i, len(expected) :] == policy.PAD).all()
    with pytest.raises(ValueError, match="example 1"):
        list(
            policy.replay_witness(
                replace(task, inputs=("[1 2]", "[]"), program=("add",))
            )
        )


def test_validation_is_a_skeleton_split(sum_task):
    groups = [replace(sum_task, skeleton=f"shape-{i}") for i in range(1000)]
    held = [policy.validation_skeleton(task) for task in groups]
    assert 60 < sum(held) < 140
    for task, decision in zip(groups, held, strict=True):
        assert (
            policy.validation_skeleton(replace(task, behaviour="different")) == decision
        )
        assert (
            policy.validation_skeleton(
                replace(task, behaviour=task.skeleton, skeleton="")
            )
            == decision
        )


def test_collation_pooling_and_legal_normalization():
    a = policy.item_array([[7, 1], [5, 1, 9]])
    b = policy.item_array([[6, 1, 3, 2, 9]])
    model = policy.StackPairPolicy(TINY)
    scorer = policy.PolicyScorer(model)
    assert model.embedding.num_embeddings == policy.VOCAB
    assert model.embedding.padding_idx == policy.PAD
    assert not model.training
    sequences, counts, mask = policy.collate([a, b], "cpu")
    assert sequences.tolist() == [[7, 1, 0, 0, 0], [5, 1, 9, 0, 0], [6, 1, 3, 2, 9]]
    assert sequences.dtype == torch.long
    assert counts.tolist() == [2, 1]
    assert mask.tolist() == [[True, True], [True, False]]
    legal = np.zeros((2, len(policy.OP_NAMES)), dtype=bool)
    legal[0, [1, 3]] = True
    logp = scorer.log_probs([a, b], legal)
    assert np.isneginf(logp[~legal]).all()
    np.testing.assert_allclose(np.exp(logp[0]).sum(), 1, atol=1e-6)
    logits = model(sequences, counts, mask)
    expected = logits[0, [1, 3]].log_softmax(0).detach().numpy()
    np.testing.assert_allclose(logp[0, legal[0]], expected)
    np.testing.assert_allclose(
        scorer.log_probs([a[::-1]], legal[:1]), logp[:1], atol=1e-6
    )


def test_deterministic_training_restores_best_epoch():
    block = policy.item_array([[9, policy.SEP, 10]])
    row = policy.Row(block, 0, 0, 0, 1)
    # Contradict the training label on validation so later epochs degrade.
    train, validation = [row] * 3, [replace(row, label=1)] * 3
    kwargs = {
        "learning_rate": 0.02,
        "batch_size": 2,
        "validation_batch_size": 2,
        "seed": 6,
        "device": "cpu",
    }
    callbacks = []
    model, state = policy.train_policy(
        train, validation, TINY, epochs=3, epoch_callback=callbacks.append, **kwargs
    )
    first, first_state = policy.train_policy(
        train, validation, TINY, epochs=1, **kwargs
    )
    assert state["best_epoch"] == 1
    assert not model.training
    assert callbacks == state["history"] and len(callbacks) == 3
    assert state["best_validation_cross_entropy"] == min(
        row["validation_cross_entropy"] for row in callbacks
    )
    assert first_state["history"] == callbacks[:1]
    assert (
        state["parameters"]
        == sum(p.numel() for p in model.parameters())
        == policy.parameter_count(TINY)
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, first.state_dict()[key])
    for empty_train, empty_validation in (([], validation), (train, [])):
        with pytest.raises(ValueError, match="non-empty"):
            policy.train_policy(empty_train, empty_validation, TINY, epochs=1, **kwargs)


@pytest.mark.parametrize(
    "url", [None, "https://wandb.ai/abhishekraok-na/policy/runs/123"]
)
def test_weights_only_checkpoint(tmp_path, url):
    model = policy.StackPairPolicy(TINY)
    path = tmp_path / "checkpoint"
    kwargs = {
        "trainer_state": {"best_epoch": 2},
        "manifest": {"split": "v4"},
        "wandb_url": url,
    }
    assert policy.save_policy(path, model, **kwargs) == path
    names = {
        policy.MODEL_FILE,
        policy.CONFIG_FILE,
        policy.TRAINER_STATE_FILE,
        policy.MANIFEST_FILE,
    }
    if url:
        names.add(policy.WANDB_LINK_FILE)
        assert f"URL={url}" in (path / policy.WANDB_LINK_FILE).read_text().splitlines()
    assert {file.name for file in path.iterdir()} == names
    assert json.loads((path / policy.MANIFEST_FILE).read_text()) == {
        "split": "v4",
        "wandb_url": url,
    }
    assert (
        json.loads((path / policy.TRAINER_STATE_FILE).read_text())
        == kwargs["trainer_state"]
    )
    with pytest.raises(FileExistsError):
        policy.save_policy(path, model, **kwargs)
    loaded = policy.load_policy(path)
    assert loaded.architecture == TINY
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    config_path = path / policy.CONFIG_FILE
    config = json.loads(config_path.read_text())
    for key, bad in (
        ("schema_version", -1),
        ("basis", "full-52"),
        ("ordered_actions", list(reversed(policy.OP_NAMES))),
    ):
        config_path.write_text(json.dumps({**config, key: bad}))
        with pytest.raises(ValueError):
            policy.load_policy(path)


def test_partial_product_cost_includes_refused_stack():
    body = Lst((Sym("dup"), Sym("add")))
    ok = (Lst((Int(2), Int(3))), body)
    moved, fuel, steps = step_all((ok,), OP_BY_NAME["map"])
    assert moved == ((Lst((Int(4), Int(6))),),)
    assert (fuel, steps) == (5, 1)
    assert step_all((ok, (), ok), OP_BY_NAME["map"]) == (None, 5, 2)


class PreferredOps:
    def __init__(self, names=()):
        self.names = iter(names)
        self.batches = []

    def log_probs(self, blocks, legal):
        self.batches.append(legal.copy())
        assert len(blocks) == len(legal)
        logits = np.zeros(legal.shape)
        name = next(self.names, None)
        if name:
            logits[:, policy.OP_INDEX[name]] = 10
        logits[~legal] = -np.inf
        return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)


def test_beam_fits_a_quotation_without_witness_access(sum_task):
    scorer = PreferredOps(sum_task.program)
    task = replace(
        sum_task, program=(), held_out_inputs=("(",), held_out_outputs=("(",)
    )
    result = beam_solve(task, scorer, beam_width=1, max_depth=5)
    assert result.termination == "fit" and result.depth == result.expansions == 5
    assert result.program == sum_task.program
    assert grade(result.program, sum_task).solved
    assert result.seconds >= 0
    assert len(scorer.batches) == 4
    assert scorer.batches[2].all()  # Capturing under an open quotation.
    product = tuple(policy.parse_stack(s, basis="mark-28") for s in task.inputs)
    executions = fuel = 0
    for depth, name in enumerate(result.program, 1):
        for op in stack_beam.OPS:
            _, cost, steps = step_all(product, op)
            executions += steps
            fuel += cost
            if depth == result.depth and op.name == name:
                break
        product = step_all(product, OP_BY_NAME[name])[0]
    assert (result.executions, result.fuel) == (executions, fuel)


def test_budget_and_max_depth(sum_task):
    scorer = PreferredOps()
    limited = beam_solve(sum_task, scorer, budget=1)
    assert (limited.program, limited.termination, limited.depth) == (None, "budget", 1)
    assert (limited.expansions, limited.executions, limited.fuel) == (1, 3, 3)
    zero = beam_solve(sum_task, scorer, max_depth=0)
    assert (zero.termination, zero.depth, zero.executions, zero.expansions) == (
        "max_depth",
        0,
        0,
        0,
    )
    exhausted = beam_solve(sum_task, scorer, max_depth=2, beam_width=3)
    assert (exhausted.termination, exhausted.depth, exhausted.expansions) == (
        "max_depth",
        2,
        4,
    )
    assert [len(batch) for batch in scorer.batches] == [1, 3]


@pytest.mark.parametrize("mode", ["close", "both"])
def test_guard_precondition_applies_to_every_target(sum_task, mode):
    task = replace(sum_task, outputs=("[0]", "[0]", "[(MARK)]"))
    with pytest.raises(ValueError, match="mark-free"):
        beam_solve(task, PreferredOps(), guard=mode)
    assert (
        beam_solve(task, PreferredOps(), guard="body", max_depth=1).termination
        == "max_depth"
    )


def test_guard_prunes_after_legal_scoring(sum_task, monkeypatch):
    # A single available '[' can step the product but cannot close at depth 1.
    task = replace(sum_task, inputs=("[]",), outputs=("[42]",))
    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["["],))
    scorer = PreferredOps()
    result = beam_solve(task, scorer, guard="close", max_depth=1)
    assert (result.termination, result.depth, result.fuel) == (
        "frontier_exhausted",
        0,
        1,
    )
    assert scorer.batches[0].tolist() == [[True]]
    assert beam_solve(task, scorer, guard="off", max_depth=1).termination == "max_depth"


@pytest.mark.parametrize("op", ["add", "swap"])
def test_frontier_exhaustion_on_refusal_or_seen_product(sum_task, monkeypatch, op):
    scorer = PreferredOps()
    stack = "[]" if op == "add" else "[3 3]"
    task = replace(sum_task, inputs=(stack,), outputs=("[42]",))
    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME[op],))
    result = beam_solve(task, scorer)
    assert (result.termination, result.depth, result.expansions, result.executions) == (
        "frontier_exhausted",
        0,
        1,
        1,
    )
    assert len(scorer.batches) == (0 if op == "add" else 1)


def test_beam_ranks_paths_before_deduplicating_and_truncating(sum_task):
    task = replace(sum_task, inputs=("[1]",), outputs=("[1 1 false true]",))

    class ForkScorer:
        def log_probs(self, blocks, legal):
            # '1' and 'dup' reach the same product at the root. At the fork,
            # both continuations of 'dup' outrank the best child of '0'.
            rows = (
                [{"dup": 0.75, "0": 0.20, "1": 0.05}]
                if len(blocks) == 1
                else [{"true": 0.6, "false": 0.4}, {"true": 1.0}]
            )
            scores = np.full(legal.shape, -np.inf)
            for i, row in enumerate(rows):
                for name, probability in row.items():
                    scores[i, policy.OP_INDEX[name]] = np.log(probability)
            return scores

    result = beam_solve(task, ForkScorer(), beam_width=2, max_depth=3)
    assert result.program == ("dup", "false", "true")
