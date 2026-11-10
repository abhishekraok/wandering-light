"""The mark-28 token policy (`stack_policy`) and its beam search (`stack_beam`)."""

import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from wandering_light import stack_beam
from wandering_light.rollout_policy import AlignedPairPolicy, PolicyArchitecture
from wandering_light.stack_beam import BeamResult, beam_solve, step_all
from wandering_light.stack_machine import (
    FALSE,
    MARK,
    NIL,
    TRUE,
    Int,
    Lst,
    Sym,
    parse_stack,
)
from wandering_light.stack_pbe import BY_NAME, OP_BY_NAME, OPS, build_task, draw_task
from wandering_light.stack_policy import (
    BIG,
    CLOSE,
    CONFIG_FILE,
    DIGIT_BASE,
    FALSE_TOKEN,
    INT_BASE,
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
    SYMBOL_BASE,
    TARGET_CAP,
    TRAINER_STATE_FILE,
    TRUE_TOKEN,
    VOCAB,
    WANDB_LINK_FILE,
    PolicyScorer,
    Row,
    StackPairPolicy,
    collate,
    encode_item,
    encode_stack,
    encode_targets,
    item_array,
    load_policy,
    parameter_count,
    replay_witness,
    save_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

TINY = PolicyArchitecture(byte_embedding=8, conv_channels=12, hidden=16)


def small(value: int) -> int:
    return INT_BASE + value + SMALL_INT


@pytest.fixture(scope="module")
def sum_task():
    rng = random.Random(4)
    return build_task(BY_NAME["sum"], draw_task(BY_NAME["sum"], rng, 4, held_out=6))


@pytest.fixture(scope="module")
def add_task():
    rng = random.Random(6)
    return build_task(
        BY_NAME["add_pair"], draw_task(BY_NAME["add_pair"], rng, 4, held_out=6)
    )


@pytest.fixture(scope="module")
def tasks(sum_task, add_task):
    rng = random.Random(9)
    extra = [
        build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 3, held_out=2))
        for name in ("product", "double_each", "reverse")
    ]
    return [sum_task, add_task, *extra]


def manual_task(inputs, outputs, program=("add",)):
    """A Task built by hand, for states the catalogue never produces."""
    base = build_task(
        BY_NAME["add_pair"], draw_task(BY_NAME["add_pair"], random.Random(0), 3)
    )
    return replace(
        base,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        examples=len(inputs),
        program=tuple(program),
        length=len(program),
    )


# ---------------------------------------------------------------------------
# Actions and vocabulary
# ---------------------------------------------------------------------------


def test_actions_are_mark_28_in_order():
    assert tuple(op.name for op in OPS) == OP_NAMES
    assert len(OP_NAMES) == 28
    assert OP_NAMES[-2:] == ("[", "]")
    assert all(OP_NAMES[index] == name for name, index in OP_INDEX.items())


def test_vocabulary_layout():
    assert (tuple(range(9))) == (
        PAD,
        SEP,
        OPEN,
        CLOSE,
        MARK_TOKEN,
        TRUE_TOKEN,
        FALSE_TOKEN,
        NEG,
        BIG,
    )
    assert SYMBOL_BASE == 9
    assert 9 + len(OP_NAMES) == INT_BASE
    assert DIGIT_BASE == INT_BASE + 2 * SMALL_INT + 1
    assert VOCAB == DIGIT_BASE + 10
    assert (STATE_CAP, TARGET_CAP, SMALL_INT) == (160, 64, 64)


def test_small_and_big_integers():
    assert encode_stack((Int(0), Int(64), Int(-64)), 10) == [
        small(0),
        small(64),
        small(-64),
    ]
    assert encode_stack((Int(65),), 10) == [BIG, DIGIT_BASE + 6, DIGIT_BASE + 5]
    assert encode_stack((Int(-1203),), 10) == [
        BIG,
        NEG,
        DIGIT_BASE + 1,
        DIGIT_BASE + 2,
        DIGIT_BASE + 0,
        DIGIT_BASE + 3,
    ]


def test_bools_marks_symbols_and_nested_lists():
    stack = (TRUE, FALSE, MARK, Sym("add"), NIL, Lst((Int(1), Lst((Sym("dup"),)))))
    assert encode_stack(stack, 100) == [
        TRUE_TOKEN,
        FALSE_TOKEN,
        MARK_TOKEN,
        SYMBOL_BASE + OP_INDEX["add"],
        OPEN,
        CLOSE,
        OPEN,
        small(1),
        OPEN,
        SYMBOL_BASE + OP_INDEX["dup"],
        CLOSE,
        CLOSE,
    ]


def test_an_open_quotation_body_is_one_token_per_op():
    stack = parse_stack("[ (1 2) MARK dup add swap ]", basis="mark-28")
    tokens = encode_stack(stack, 100)
    assert tokens[-4:] == [
        MARK_TOKEN,
        *(SYMBOL_BASE + OP_INDEX[name] for name in ("dup", "add", "swap")),
    ]


def test_unknown_symbol_is_refused():
    with pytest.raises(ValueError, match="not a mark-28 op"):
        encode_stack((Sym("plus"),), 10)


def test_over_cap_keeps_the_top_of_the_stack():
    stack = tuple(Int(n) for n in range(10))
    assert encode_stack(stack, 3) == [small(7), small(8), small(9)]
    assert len(encode_stack(tuple(Int(1) for _ in range(500)), STATE_CAP)) == STATE_CAP


def test_items_and_targets():
    outputs = [(Int(3),), (Lst(tuple(Int(1) for _ in range(100))),)]
    targets = encode_targets(outputs)
    assert targets[0] == [small(3)]
    assert len(targets[1]) == TARGET_CAP
    assert targets[1][-1] == CLOSE
    item = encode_item((Int(5), Int(6)), targets[0])
    assert item == [small(5), small(6), SEP, small(3)]


def test_item_array_and_collate():
    first = item_array([[10, 11, SEP, 12], [10, SEP, 13]])
    assert first.dtype == np.int16
    assert first.tolist() == [[10, 11, SEP, 12], [10, SEP, 13, PAD]]
    second = item_array([[SEP, 20, 21, 22, 23, 24]])
    sequences, counts, mask = collate([first, second], "cpu")
    assert sequences.dtype == torch.long
    assert sequences.shape == (3, 6)
    assert sequences[0].tolist() == [10, 11, SEP, 12, PAD, PAD]
    assert counts.tolist() == [2, 1]
    assert mask.tolist() == [[True, True], [True, False]]


# ---------------------------------------------------------------------------
# Model and scorer
# ---------------------------------------------------------------------------


def test_policy_is_an_aligned_pair_policy_over_the_token_vocabulary():
    torch.manual_seed(0)
    model = StackPairPolicy(TINY)
    assert isinstance(model, AlignedPairPolicy)
    assert model.action_count == len(OP_NAMES)
    assert model.embedding.num_embeddings == VOCAB
    assert model.embedding.padding_idx == PAD
    assert model.parameter_count() == parameter_count(TINY)
    assert model.parameter_count() == sum(p.numel() for p in model.parameters())


def test_padding_width_does_not_change_logits():
    torch.manual_seed(0)
    model = StackPairPolicy(TINY).eval()
    block = item_array([[small(1), small(2), SEP, small(3)], [small(4), SEP, small(4)]])
    wide = np.full((2, 40), PAD, dtype=np.int16)
    wide[:, : block.shape[1]] = block
    with torch.inference_mode():
        narrow_logits = model(*collate([block]))
        wide_logits = model(*collate([wide]))
    torch.testing.assert_close(narrow_logits, wide_logits)


def test_item_order_does_not_change_logits():
    torch.manual_seed(0)
    model = StackPairPolicy(TINY).eval()
    items = [[small(1), SEP, small(3)], [small(4), small(5), SEP, small(9)]]
    with torch.inference_mode():
        forward = model(*collate([item_array(items)]))
        backward = model(*collate([item_array(items[::-1])]))
    torch.testing.assert_close(forward, backward)


def test_scorer_normalises_over_legal_ops_only():
    torch.manual_seed(0)
    model = StackPairPolicy(TINY).train()
    scorer = PolicyScorer(model, device="cpu")
    assert not scorer.model.training
    blocks = [
        item_array([[small(1), SEP, small(2)]]),
        item_array([[small(3), SEP, small(4)], [small(5), SEP, small(6)]]),
        item_array([[SEP, small(0)]]),
    ]
    legal = np.zeros((3, len(OP_NAMES)), dtype=bool)
    legal[0, [OP_INDEX["dup"], OP_INDEX["add"], OP_INDEX["["]]] = True
    legal[1, :] = True
    scores = scorer.log_probs(blocks, legal)
    assert scores.shape == (3, len(OP_NAMES))
    assert np.all(np.isneginf(scores[~legal]))
    for row in (0, 1):
        assert np.isclose(np.logaddexp.reduce(scores[row, legal[row]]), 0.0)
    assert np.all(np.isneginf(scores[2]))
    with pytest.raises(ValueError):
        scorer.log_probs(blocks, legal[:2])


# ---------------------------------------------------------------------------
# Witness rows
# ---------------------------------------------------------------------------


def test_replay_follows_the_witness_to_the_outputs(sum_task):
    steps = list(replay_witness(sum_task))
    assert [name for _, _, name in steps] == list(sum_task.program)
    assert [step for step, _, _ in steps] == list(range(len(sum_task.program)))
    inputs = tuple(parse_stack(text, basis="mark-28") for text in sum_task.inputs)
    assert steps[0][1] == inputs
    _, last, name = steps[-1]
    moved, _, _ = step_all(last, OP_BY_NAME[name])
    assert moved == tuple(
        parse_stack(text, basis="mark-28") for text in sum_task.outputs
    )


def test_replay_raises_when_the_machine_refuses(sum_task):
    broken = replace(sum_task, program=("add",), length=1)
    with pytest.raises(ValueError, match="refused"):
        list(replay_witness(broken))


def test_witness_rows(tasks):
    rows = witness_rows(tasks)
    assert len(rows) == sum(len(task.program) for task in tasks)
    first = [row for row in rows if row.task == 0]
    sum_task = tasks[0]
    assert [row.label for row in first] == [OP_INDEX[n] for n in sum_task.program]
    assert [row.step for row in first] == list(range(sum_task.length))
    assert [row.remaining for row in first] == [
        sum_task.length - step for step in range(sum_task.length)
    ]
    for row in rows:
        assert isinstance(row, Row)
        assert row.items.dtype == np.int16
        assert row.items.shape[0] == tasks[row.task].examples
        assert not row.items.flags.writeable
        assert np.all((row.items == SEP).sum(axis=1) == 1)
    with pytest.raises(AttributeError):
        rows[0].label = 3


def test_validation_skeleton_holds_out_about_one_in_ten():
    held = [
        validation_skeleton(
            build_task(b, draw_task(b, random.Random(1), 3, held_out=1))
        )
        for b in list(BY_NAME.values())[:120]
    ]
    assert 0 < sum(held) < 30


def test_validation_skeleton_is_keyed_on_skeleton(sum_task):
    renamed = replace(sum_task, behaviour="something_else")
    assert validation_skeleton(renamed) == validation_skeleton(sum_task)
    assert all(validation_skeleton(sum_task, out_of=1) for _ in range(2))
    with pytest.raises(ValueError):
        validation_skeleton(sum_task, out_of=0)


# ---------------------------------------------------------------------------
# Training and checkpoints
# ---------------------------------------------------------------------------


def _train(rows, seed=3, epochs=3, callback=None):
    return train_policy(
        rows[:-4],
        rows[-4:],
        TINY,
        epochs=epochs,
        learning_rate=3e-3,
        batch_size=4,
        validation_batch_size=3,
        seed=seed,
        device="cpu",
        epoch_callback=callback,
    )


def test_training_is_deterministic_and_restores_the_best_epoch(tasks):
    rows = witness_rows(tasks)
    seen = []
    model, state = _train(rows, callback=seen.append)
    again, again_state = _train(rows)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, again.state_dict()[name], rtol=0, atol=0)
    assert state["history"] == [
        {key: row[key] for key in row} for row in again_state["history"]
    ]
    assert seen == state["history"]
    assert len(state["history"]) == 3
    losses = [row["validation_cross_entropy"] for row in state["history"]]
    assert state["best_epoch"] == 1 + losses.index(min(losses))
    assert state["best_validation_cross_entropy"] == min(losses)
    assert state["parameters"] == parameter_count(TINY)
    assert not model.training
    json.dumps(state)


def test_training_lowers_cross_entropy(tasks):
    rows = witness_rows(tasks)
    _, state = train_policy(
        rows,
        rows,
        TINY,
        epochs=15,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=64,
        seed=0,
        device="cpu",
    )
    history = state["history"]
    assert history[-1]["train_cross_entropy"] < history[0]["train_cross_entropy"]
    assert state["best_validation_cross_entropy"] < np.log(len(OP_NAMES))


def test_training_refuses_empty_rows(tasks):
    rows = witness_rows(tasks)
    kwargs = {
        "epochs": 1,
        "learning_rate": 1e-3,
        "batch_size": 2,
        "validation_batch_size": 2,
        "seed": 0,
        "device": "cpu",
    }
    with pytest.raises(ValueError):
        train_policy([], rows, TINY, **kwargs)
    with pytest.raises(ValueError):
        train_policy(rows, [], TINY, **kwargs)


def test_training_leaves_global_determinism_as_it_was(tasks):
    before = torch.are_deterministic_algorithms_enabled()
    _train(witness_rows(tasks), epochs=1)
    assert torch.are_deterministic_algorithms_enabled() == before


def test_checkpoint_round_trip(tmp_path, tasks):
    torch.manual_seed(1)
    model = StackPairPolicy(TINY).eval()
    url = "https://wandb.ai/abhishekraok-na/stack-policy/runs/abc123"
    path = save_policy(
        tmp_path / "ckpt",
        model,
        trainer_state={"best_epoch": 2},
        manifest={"corpus": "stack_pbe_v4"},
        wandb_url=url,
    )
    names = {p.name for p in path.iterdir()}
    assert names == {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINER_STATE_FILE,
        MANIFEST_FILE,
        WANDB_LINK_FILE,
    }
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["basis"] == "mark-28"
    assert config["ordered_actions"] == list(OP_NAMES)
    assert config["vocabulary"]["size"] == VOCAB
    assert config["architecture"]["hidden"] == TINY.hidden
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    assert manifest == {"corpus": "stack_pbe_v4", "wandb_url": url}
    assert json.loads((path / TRAINER_STATE_FILE).read_text()) == {"best_epoch": 2}
    assert url in (path / WANDB_LINK_FILE).read_text()

    loaded = load_policy(path)
    assert loaded.architecture == TINY
    assert not loaded.training
    blocks = [row.items for row in witness_rows(tasks)[:6]]
    with torch.inference_mode():
        torch.testing.assert_close(model(*collate(blocks)), loaded(*collate(blocks)))

    with pytest.raises(FileExistsError):
        save_policy(path, model, trainer_state={}, manifest={}, wandb_url=None)


def test_checkpoint_without_url_has_no_link_file(tmp_path):
    path = save_policy(
        tmp_path / "ckpt",
        StackPairPolicy(TINY),
        trainer_state={},
        manifest={},
        wandb_url=None,
    )
    assert not (path / WANDB_LINK_FILE).exists()
    assert json.loads((path / MANIFEST_FILE).read_text()) == {"wandb_url": None}
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 99),
        ("basis", "executable-26"),
        ("ordered_actions", list(reversed(OP_NAMES))),
    ],
)
def test_load_refuses_a_foreign_config(tmp_path, field, value):
    path = save_policy(
        tmp_path / "ckpt",
        StackPairPolicy(TINY),
        trainer_state={},
        manifest={},
        wandb_url=None,
    )
    config = json.loads((path / CONFIG_FILE).read_text())
    config[field] = value
    (path / CONFIG_FILE).write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_policy(path)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


class UniformScorer:
    def __init__(self):
        self.calls = []

    def log_probs(self, blocks, legal):
        self.calls.append((list(blocks), np.array(legal)))
        legal = np.asarray(legal, dtype=bool)
        counts = legal.sum(axis=1, keepdims=True)
        out = np.where(legal, -np.log(np.maximum(counts, 1)), -np.inf)
        return out


class ScriptedScorer(UniformScorer):
    """Prefers op ``script[layer]``; one call per layer, so the call count is it."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)

    def log_probs(self, blocks, legal):
        layer = len(self.calls)
        super().log_probs(blocks, legal)
        out = np.where(np.asarray(legal, dtype=bool), -10.0, -np.inf)
        if layer < len(self.script):
            out[:, OP_INDEX[self.script[layer]]] = np.where(
                legal[:, OP_INDEX[self.script[layer]]], 0.0, -np.inf
            )
        return out


def test_step_all_reports_the_cost_of_a_refusal():
    product = (
        (Int(1), Int(2)),
        (Lst(()), Int(2)),
        (Int(3), Int(4)),
    )
    moved, fuel, stepped = step_all(product, OP_BY_NAME["add"])
    assert moved is None
    assert stepped == 2
    assert fuel == 1
    moved, fuel, stepped = step_all(product[::2], OP_BY_NAME["add"])
    assert moved == ((Int(3),), (Int(7),))
    assert (fuel, stepped) == (2, 2)


def test_beam_finds_a_one_op_program(add_task):
    torch.manual_seed(0)
    found = beam_solve(add_task, PolicyScorer(StackPairPolicy(TINY)))
    assert isinstance(found, BeamResult)
    assert found.program == ("add",)
    assert found.termination == "fit"
    assert found.depth == 1
    assert found.expansions == 1
    assert 1 <= found.executions <= len(OPS) * add_task.examples
    assert found.fuel >= add_task.examples
    assert found.seconds >= 0


def test_a_guiding_policy_reaches_the_witness(sum_task):
    scorer = ScriptedScorer(sum_task.program)
    found = beam_solve(sum_task, scorer, beam_width=2, max_depth=8)
    assert found.termination == "fit"
    assert found.program == sum_task.program
    assert found.depth == len(sum_task.program)
    # One scoring call per completed layer; the fit lands in the last one.
    assert len(scorer.calls) == len(sum_task.program) - 1
    assert found.expansions <= 2 * len(sum_task.program)


def test_one_scorer_call_per_layer_and_beam_width_respected(sum_task):
    scorer = UniformScorer()
    found = beam_solve(sum_task, scorer, beam_width=3, max_depth=4)
    assert found.termination == "max_depth"
    assert found.program is None
    assert found.depth == 4
    assert len(scorer.calls) == 4
    assert [len(blocks) for blocks, _ in scorer.calls] == [1, 3, 3, 3]
    assert found.expansions == 1 + 3 + 3 + 3
    for blocks, legal in scorer.calls:
        assert legal.shape == (len(blocks), len(OP_NAMES))
        assert all(block.shape[0] == sum_task.examples for block in blocks)


def test_budget_caps_executions(sum_task):
    found = beam_solve(sum_task, UniformScorer(), budget=50)
    assert found.termination == "budget"
    assert found.program is None
    assert found.executions <= 50
    assert found.executions > 50 - sum_task.examples
    assert found.depth == 1
    assert beam_solve(sum_task, UniformScorer(), budget=0).executions == 0

    # Charged for every attempt, kept or refused: a whole first layer costs
    # what a one-layer search reports. The cap is hard -- an attempt that
    # could step more stacks than are left is not made -- so finishing the
    # layer needs headroom of one attempt (one stack per example).
    layer = beam_solve(sum_task, UniformScorer(), max_depth=1)
    assert layer.termination == "max_depth"
    assert layer.executions > len(OPS)
    exact = beam_solve(sum_task, UniformScorer(), budget=layer.executions)
    assert exact.depth == 1
    assert exact.executions <= layer.executions
    headroom = layer.executions + sum_task.examples
    past = beam_solve(sum_task, UniformScorer(), budget=headroom + 20)
    assert past.termination == "budget"
    assert past.depth == 2
    assert layer.executions < past.executions <= headroom + 20


def test_no_successors_is_frontier_exhausted(monkeypatch):
    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["add"],))
    task = manual_task(["[ (1 2) ]", "[ (3) ]"], ["[ 1 ]", "[ 3 ]"])
    found = beam_solve(task, UniformScorer())
    assert found.termination == "frontier_exhausted"
    assert found.depth == 0
    assert found.expansions == 1


def test_only_seen_successors_is_frontier_exhausted(monkeypatch):
    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["swap"],))
    task = manual_task(["[ 1 2 ]", "[ 3 4 ]"], ["[ 9 ]", "[ 9 ]"])
    scorer = UniformScorer()
    found = beam_solve(task, scorer)
    assert found.termination == "frontier_exhausted"
    assert found.depth == 1
    assert len(scorer.calls) == 2


def test_close_guard_needs_mark_free_targets():
    task = manual_task(["[ 1 2 ]", "[ 3 4 ]"], ["[ MARK ]", "[ 1 MARK ]"])
    for guard in ("close", "both"):
        with pytest.raises(ValueError, match="mark-free"):
            beam_solve(task, UniformScorer(), guard=guard)
    beam_solve(task, UniformScorer(), guard="off", max_depth=1)


def test_close_guard_prunes_marks_that_cannot_close(sum_task):
    def marks_in_last_call(guard):
        scorer = ScriptedScorer(["["] * 3)
        beam_solve(sum_task, scorer, beam_width=1, max_depth=3, guard=guard)
        blocks, _ = scorer.calls[2]
        return int((blocks[0][0] == MARK_TOKEN).sum())

    assert marks_in_last_call("off") == 2
    assert marks_in_last_call("close") == 1


def test_beam_never_reads_the_held_out_examples(sum_task):
    blinded = replace(sum_task, held_out_inputs=(), held_out_outputs=(), held_out=0)
    a = beam_solve(sum_task, UniformScorer(), beam_width=8, max_depth=5)
    b = beam_solve(blinded, UniformScorer(), beam_width=8, max_depth=5)
    assert (a.program, a.termination, a.executions) == (
        b.program,
        b.termination,
        b.executions,
    )
