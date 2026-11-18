import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn.functional import cross_entropy

from wandering_light import stack_beam
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve, step_all
from wandering_light.stack_machine import Int, parse_stack
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS, Task
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import (
    BIG,
    CLOSE,
    CONFIG_FILE,
    FALSE_TOKEN,
    MANIFEST_FILE,
    MARK_TOKEN,
    MODEL_FILE,
    NEG,
    OP_INDEX,
    OP_NAMES,
    OPEN,
    PAD,
    SEP,
    SMALL_INT,
    STATE_CAP,
    TARGET_CAP,
    TRAINER_STATE_FILE,
    TRUE_TOKEN,
    VOCAB,
    WANDB_LINK_FILE,
    PolicyScorer,
    StackPairPolicy,
    collate,
    encode_item,
    encode_stack,
    encode_targets,
    item_array,
    load_policy,
    replay_witness,
    save_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

ARCHITECTURE = PolicyArchitecture(byte_embedding=4, conv_channels=8, hidden=6)


@pytest.fixture(scope="module", autouse=True)
def small_torch_batches():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def stack(text):
    return parse_stack(text, basis=BASIS)


@pytest.fixture
def sum_task():
    return Task(
        behaviour="sum",
        summary="Sum the list.",
        split="train",
        task_hash_split="train",
        inputs=("[ (2 3) ]", "[ (-2 8 1) ]"),
        outputs=("[ 5 ]", "[ 7 ]"),
        held_out_inputs=("[ (3 7 1) ]",),
        held_out_outputs=("[ 11 ]",),
        program=("0", "[", "add", "]", "fold"),
        length=5,
        examples=2,
        held_out=1,
        fuel=15,
        tags=(),
        skeleton="_ [ _ ] fold",
    )


def test_tokens_and_truncation():
    assert tuple(op.name for op in OPS) == OP_NAMES
    assert {op.name: index for index, op in enumerate(OPS)} == OP_INDEX
    integer_start = 9 + len(OPS)
    assert integer_start + 2 * SMALL_INT + 1 + 10 == VOCAB
    tokens = [
        integer_start,
        integer_start + 128,
        BIG,
        VOCAB - 4,
        VOCAB - 5,
        BIG,
        NEG,
        VOCAB - 3,
        VOCAB - 4,
        TRUE_TOKEN,
        FALSE_TOKEN,
        MARK_TOKEN,
        OPEN,
        BIG + 1 + OP_INDEX["add"],
        OPEN,
        CLOSE,
        CLOSE,
    ]
    values = stack("[-64 64 65 -76 true false MARK (add nil)]")
    assert encode_stack(values, 100) == tokens
    assert encode_stack(values, 8) == tokens[-8:]
    assert encode_stack(values, 0) == []
    assert encode_stack((Int(10**5000 + 27),), 2) == [VOCAB - 8, VOCAB - 3]
    body = stack("[ MARK " + "add " * STATE_CAP + "]")
    target = (Int(1),) * (TARGET_CAP + 1)
    (target_tokens,) = encode_targets((target,))
    assert len(target_tokens) == TARGET_CAP
    assert encode_item(body, target_tokens) == [
        *([BIG + 1 + OP_INDEX["add"]] * STATE_CAP),
        SEP,
        *target_tokens,
    ]


def test_padding_set_pooling_and_legal_normalization():
    block = item_array([[SEP], [OPEN, CLOSE, SEP]])
    other = item_array([[TRUE_TOKEN, SEP, OPEN, CLOSE]])
    sequences, counts, item_mask = collate([block, other], "cpu")
    assert block.dtype == np.int16
    assert sequences.tolist() == [
        [SEP, PAD, PAD, PAD],
        [OPEN, CLOSE, SEP, PAD],
        [TRUE_TOKEN, SEP, OPEN, CLOSE],
    ]
    assert counts.tolist() == [2, 1]
    assert item_mask.tolist() == [[True, True], [True, False]]
    model = StackPairPolicy(ARCHITECTURE)
    assert model.embedding.num_embeddings == VOCAB
    assert model.embedding.padding_idx == PAD
    with torch.no_grad():
        model.embedding.weight[PAD].fill_(123)
        logits = model(*collate([block], "cpu"))
        torch.testing.assert_close(
            logits, model(*collate([block[::-1], other], "cpu"))[:1]
        )

    legal = np.zeros((3, len(OPS)), dtype=bool)
    legal[0, :2] = True
    legal[1, 1] = True
    scorer = PolicyScorer(model)
    result = scorer.log_probs([block, other, other], legal)
    assert not model.training
    assert np.isneginf(result[~legal]).all()
    np.testing.assert_allclose(
        result[0, :2], logits[0, :2].log_softmax(0).numpy(), atol=1e-6
    )
    np.testing.assert_allclose(np.exp(result).sum(axis=1), [1, 1, 0], atol=1e-6)


def test_replay_rows_and_skeleton_split(sum_task):
    replayed = list(replay_witness(sum_task))
    assert replayed[2] == (
        2,
        (stack("[(2 3) 0 MARK]"), stack("[(-2 8 1) 0 MARK]")),
        "add",
    )
    rows = witness_rows([sum_task, sum_task])
    assert [(row.task, row.step, row.remaining) for row in rows] == [
        (task, step, sum_task.length - step) for task in range(2) for step in range(5)
    ]
    assert [OP_NAMES[row.label] for row in rows[:5]] == list(sum_task.program)
    targets = encode_targets(tuple(map(stack, sum_task.outputs)))
    np.testing.assert_array_equal(
        rows[2].items,
        item_array(
            [
                encode_item(stack("[(2 3) 0 MARK]"), targets[0]),
                encode_item(stack("[(-2 8 1) 0 MARK]"), targets[1]),
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"witness step 0.*add:"):
        list(replay_witness(replace(sum_task, program=("add",))))

    assert validation_skeleton(sum_task) == validation_skeleton(
        replace(sum_task, behaviour="other")
    )
    digest = hashlib.sha256(sum_task.behaviour.encode()).digest()
    assert validation_skeleton(replace(sum_task, skeleton="")) == (
        int.from_bytes(digest, "big") % 10 == 0
    )


def test_train_restores_best_weights_and_checkpoint_round_trips(sum_task, tmp_path):
    row = witness_rows([sum_task])[0]
    validation = [replace(row, label=OP_INDEX["add"])]
    options = {
        "epochs": 3,
        "learning_rate": 0.1,
        "batch_size": 1,
        "validation_batch_size": 1,
        "seed": 31,
        "device": "cpu",
    }
    history = []
    trained, state = train_policy(
        [row], validation, ARCHITECTURE, epoch_callback=history.append, **options
    )
    again, repeated_state = train_policy([row], validation, ARCHITECTURE, **options)
    assert history == state["history"]
    assert state == repeated_state
    assert 1 <= state["best_epoch"] < options["epochs"]
    assert state["parameters"] == trained.parameter_count
    inputs = collate([row.items], "cpu")
    with torch.inference_mode():
        expected = trained(*inputs)
        torch.testing.assert_close(again(*inputs), expected, atol=0, rtol=0)
        loss = cross_entropy(expected, torch.tensor([validation[0].label])).item()
    assert loss == pytest.approx(state["best_validation_cross_entropy"])
    assert loss == pytest.approx(
        min(epoch["validation_cross_entropy"] for epoch in history)
    )
    with pytest.raises(ValueError, match="non-empty"):
        train_policy([row], [], ARCHITECTURE, **options)

    url = "https://wandb.ai/abhishekraok-na/project/runs/seed31"
    checkpoint = save_policy(
        tmp_path / "run",
        trained,
        trainer_state=state,
        manifest={"corpus": "stack_pbe_v4"},
        wandb_url=url,
    )
    assert {file.name for file in checkpoint.iterdir()} == {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINER_STATE_FILE,
        MANIFEST_FILE,
        WANDB_LINK_FILE,
    }
    assert json.loads((checkpoint / TRAINER_STATE_FILE).read_text()) == state
    assert json.loads((checkpoint / MANIFEST_FILE).read_text()) == {
        "corpus": "stack_pbe_v4",
        "wandb_url": url,
    }
    assert f"URL={url}" in (checkpoint / WANDB_LINK_FILE).read_text()
    loaded = load_policy(checkpoint)
    assert loaded.architecture == ARCHITECTURE
    with torch.inference_mode():
        torch.testing.assert_close(loaded(*inputs), expected, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_policy(
            checkpoint, trained, trainer_state=state, manifest={}, wandb_url=None
        )
    another = save_policy(
        tmp_path / "local", trained, trainer_state={}, manifest={}, wandb_url=None
    )
    assert not (another / WANDB_LINK_FILE).exists()

    config_file = checkpoint / CONFIG_FILE
    config = json.loads(config_file.read_text())
    for key, value in [
        ("schema_version", -1),
        ("basis", "full-52"),
        ("ordered_actions", list(reversed(OP_NAMES))),
    ]:
        config_file.write_text(json.dumps({**config, key: value}))
        with pytest.raises(ValueError, match="Unsupported stack policy"):
            load_policy(checkpoint)


def test_step_all_costs_and_budget(sum_task):
    product = (stack("[(2 3) (dup add)]"), stack("[1]"), stack("[(8) (dup add)]"))
    assert step_all(product, OP_BY_NAME["map"]) == (None, 5, 2)
    assert step_all(product[::2], OP_BY_NAME["map"]) == (
        (stack("[(4 6)]"), stack("[(16)]")),
        8,
        2,
    )

    # The first attempt pushes 0 on both stacks before the budget is rechecked.
    result = beam_solve(sum_task, None, budget=1)
    assert (result.program, result.termination, result.depth) == (None, "budget", 1)
    assert (result.executions, result.fuel, result.expansions) == (2, 2, 1)
    zero = beam_solve(sum_task, None, budget=0)
    assert (zero.termination, zero.depth, zero.executions, zero.expansions) == (
        "budget",
        1,
        0,
        0,
    )


def test_narrow_beam_fits_a_quotation_witness(sum_task):
    rows = witness_rows([sum_task])

    class WitnessScorer:
        calls = 0

        def log_probs(self, blocks, legal):
            assert len(blocks) == 1
            row = rows[self.calls]
            np.testing.assert_array_equal(blocks[0], row.items)
            assert legal[0, row.label]
            if self.calls == 0:
                assert not legal[0, OP_INDEX["add"]]
                assert not legal[0, OP_INDEX["]"]]
            self.calls += 1
            result = np.full(legal.shape, -np.inf)
            result[0, row.label] = 0
            return result

    scorer = WitnessScorer()
    result = beam_solve(sum_task, scorer, beam_width=1, max_depth=sum_task.length)
    assert result.program == sum_task.program
    assert grade(result.program, sum_task).solved
    assert (result.termination, result.depth, result.expansions) == ("fit", 5, 5)
    assert scorer.calls == 4
    assert result.executions > result.fuel > 0
    assert result.seconds >= 0


def test_beam_ranks_cumulative_scores_before_deduplication(sum_task, monkeypatch):
    task = replace(sum_task, inputs=("[0]",), outputs=("[9]",))
    monkeypatch.setattr(stack_beam, "OPS", OPS[:2])
    graph = {
        (0, "0"): 1,
        (0, "1"): 2,
        (1, "0"): 3,
        (1, "1"): 4,
        (2, "0"): 3,
        (2, "1"): 5,
        (3, "1"): 9,
    }
    attempts = []

    def step(product, op):
        start = product[0][0].value
        attempts.append((start, op.name))
        end = graph.get((start, op.name))
        if end is None:
            return None, 0, 1
        return ((Int(end),),), 1, 1

    class GraphScorer:
        def log_probs(self, blocks, legal):
            assert legal.all()
            weights = [[0.6, 0.4]] if len(blocks) == 1 else [[0.2, 0.8], [0.9, 0.1]]
            return np.log(weights)

    monkeypatch.setattr(stack_beam, "step_all", step)
    result = beam_solve(task, GraphScorer(), beam_width=2, max_depth=3, guard="off")
    assert result.program == ("1", "0", "1")
    assert attempts[-4:] == [(4, "0"), (4, "1"), (3, "0"), (3, "1")]
    assert (result.expansions, result.executions, result.depth) == (5, 10, 3)


def test_beam_guard_and_depth_terminations(sum_task, monkeypatch):
    class UniformScorer:
        def log_probs(self, blocks, legal):
            return np.where(legal, -np.log(legal.sum(axis=1, keepdims=True)), -np.inf)

    marked = replace(sum_task, outputs=("[ (2 3) MARK ]", "[ (-2 8 1) MARK ]"))
    for guard in ("close", "both"):
        with pytest.raises(ValueError, match="mark-free"):
            beam_solve(marked, None, guard=guard)
    assert beam_solve(marked, None, guard="body").program == ("[",)

    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["["],))
    scorer = UniformScorer()
    pruned = beam_solve(sum_task, scorer, max_depth=1)
    assert (pruned.termination, pruned.depth, pruned.executions) == (
        "frontier_exhausted",
        1,
        2,
    )
    bounded = beam_solve(sum_task, scorer, max_depth=1, guard="off")
    assert (bounded.termination, bounded.depth, bounded.executions) == (
        "max_depth",
        1,
        2,
    )

    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["add"],))
    refused = beam_solve(sum_task, None)
    assert (refused.termination, refused.depth, refused.executions, refused.fuel) == (
        "frontier_exhausted",
        0,
        1,
        0,
    )
