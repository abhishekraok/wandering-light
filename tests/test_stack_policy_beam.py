"""The stack-token policy and the beam search that spends its scores.

Three things here are worth more than the rest. The encoder is checked against
a naive reference, because the real one walks the stack backwards and stops
early. The search's cost counters are recomputed independently, because they
are what a compute-optimal comparison is made of. And the guard's `remaining`
is exercised at a depth cap with no slack, where being off by one loses the
witness.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from wandering_light import stack_beam, stack_policy
from wandering_light.rollout_policy import AlignedPairPolicy, PolicyArchitecture
from wandering_light.stack_beam import BeamResult, beam_solve, step_all
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    MARK,
    MARK_OPS,
    Bool,
    Int,
    Lst,
    Sym,
    parse_stack,
    value_key,
)
from wandering_light.stack_pbe import BASIS, BY_NAME, OP_BY_NAME, build_task, draw_task
from wandering_light.stack_pbe_eval import grade
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
    replay_witness,
    save_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

TINY = PolicyArchitecture(byte_embedding=8, conv_channels=12, hidden=16)


def make_task(name: str, seed: int, examples: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, examples, 6))


@pytest.fixture(scope="module")
def sum_task():
    return make_task("sum", 4)


@pytest.fixture(scope="module")
def add_task():
    return make_task("add_pair", 6)


@pytest.fixture(scope="module")
def maximum_task():
    return make_task("maximum", 9)


def stack(text: str):
    return parse_stack(text, basis=BASIS)


def small(number: int) -> int:
    return INT_BASE + SMALL_INT + number


def symbol(name: str) -> int:
    return SYMBOL_BASE + OP_INDEX[name]


def digits(text: str) -> list[int]:
    return [DIGIT_BASE + int(character) for character in text]


def product_of(task, program):
    """The shown inputs after `program`, by the search's own step."""
    product = tuple(stack(text) for text in task.inputs)
    for name in program:
        product, _fuel, _stepped = step_all(product, OP_BY_NAME[name])
        assert product is not None
    return product


def block_of(task, program) -> np.ndarray:
    targets = encode_targets([stack(text) for text in task.outputs])
    return item_array(
        [
            encode_item(state, target)
            for state, target in zip(product_of(task, program), targets, strict=True)
        ]
    )


class UniformScorer:
    """Every legal op equally likely: a beam with no opinion."""

    def __init__(self):
        self.calls: list[int] = []

    def log_probs(self, blocks, legal):
        self.calls.append(len(blocks))
        legal = np.asarray(legal, dtype=bool)
        counts = legal.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            return np.where(legal, -np.log(np.maximum(counts, 1)), -np.inf)


class WitnessScorer:
    """Puts nearly all its mass on the witness's next op, layer by layer.

    With a beam of one this walks the witness, which makes the search's cost
    on that task a number the test can work out for itself.
    """

    def __init__(self, program):
        self.program = program
        self.layer = 0

    def log_probs(self, blocks, legal):
        scores = np.where(np.asarray(legal, dtype=bool), -20.0, -np.inf)
        scores[:, OP_INDEX[self.program[self.layer]]] = -1e-6
        self.layer += 1
        return scores


class TableScorer:
    """Scores from a per-layer table of {(node, op name): log-prob}."""

    def __init__(self, layers, default=-50.0):
        self.layers = layers
        self.default = default
        self.blocks: list[list[np.ndarray]] = []

    def log_probs(self, blocks, legal):
        table = (
            self.layers[len(self.blocks)] if len(self.blocks) < len(self.layers) else {}
        )
        self.blocks.append(list(blocks))
        scores = np.where(np.asarray(legal, dtype=bool), self.default, -np.inf)
        for (node, name), value in table.items():
            scores[node, OP_INDEX[name]] = value
        return scores


# ---------------------------------------------------------------------------
# Vocabulary and encoding
# ---------------------------------------------------------------------------


def test_actions_are_the_mark_28_ops_in_basis_order():
    assert tuple(op.name for op in MARK_OPS) == OP_NAMES
    assert len(OP_NAMES) == 28
    assert all(OP_NAMES[index] == name for name, index in OP_INDEX.items())
    assert len(OP_INDEX) == len(OP_NAMES)


def test_vocabulary_layout_has_no_gaps_or_overlaps():
    reserved = (PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG)
    assert reserved == tuple(range(9))
    assert SMALL_INT == 64
    assert (STATE_CAP, TARGET_CAP) == (160, 64)
    structural = set(range(9))
    symbols = {symbol(name) for name in OP_NAMES}
    integers = {small(n) for n in range(-SMALL_INT, SMALL_INT + 1)}
    digit_tokens = set(digits("0123456789"))
    groups = [structural, symbols, integers, digit_tokens]
    assert sum(map(len, groups)) == 9 + 28 + 129 + 10 == VOCAB
    assert set().union(*groups) == set(range(VOCAB))


def test_each_kind_of_value_is_one_token():
    assert encode_stack((Int(0),), 8) == [small(0)]
    assert encode_stack((Int(-64),), 8) == [small(-64)] == [INT_BASE]
    assert encode_stack((Int(64),), 8) == [small(64)]
    assert encode_stack((Bool(True), Bool(False)), 8) == [TRUE_TOKEN, FALSE_TOKEN]
    assert encode_stack((MARK,), 8) == [MARK_TOKEN]
    assert encode_stack((Sym("fold"),), 8) == [symbol("fold")]
    assert encode_stack((), 8) == []


def test_true_and_one_do_not_share_a_token():
    """`Bool` and `Int` are kept apart by the machine; the encoder must agree."""
    assert encode_stack((Bool(True),), 4) != encode_stack((Int(1),), 4)
    assert encode_stack((Bool(False),), 4) != encode_stack((Int(0),), 4)


def test_integers_past_the_small_range_are_spelled_in_digits():
    assert encode_stack((Int(65),), 8) == [BIG, *digits("65")]
    assert encode_stack((Int(-65),), 8) == [BIG, NEG, *digits("65")]
    assert encode_stack((Int(12345),), 8) == [BIG, *digits("12345")]
    assert encode_stack((Int(-1000),), 8) == [BIG, NEG, *digits("1000")]


def test_lists_are_bracketed_and_nest():
    assert encode_stack(stack("[ nil ]"), 8) == [OPEN, CLOSE]
    assert encode_stack(stack("[ 3 (1 (2 add)) ]"), 16) == [
        small(3),
        OPEN,
        small(1),
        OPEN,
        small(2),
        symbol("add"),
        CLOSE,
        CLOSE,
    ]


def test_an_open_quotation_body_costs_one_token_per_op():
    """The reason this is not the byte encoder: a body is as long as it has ops."""
    state = product_of(
        replace(make_task("maximum", 9), inputs=("[ (4 9 2) ]",)),
        ("uncons", "[", "[", "over", "lt?", "]", "[", "drop"),
    )[0]
    assert encode_stack(state, STATE_CAP) == [
        OPEN,
        small(9),
        small(2),
        CLOSE,
        small(4),
        MARK_TOKEN,
        OPEN,
        symbol("over"),
        symbol("lt?"),
        CLOSE,
        MARK_TOKEN,
        symbol("drop"),
    ]


def test_truncation_keeps_the_top_of_the_stack():
    values = tuple(Int(n) for n in range(10))
    assert encode_stack(values, 3) == [small(7), small(8), small(9)]
    assert encode_stack(values, 10) == [small(n) for n in range(10)]
    assert encode_stack(values, 0) == []
    # A cut may land inside a list or a number; what survives is still the top.
    assert encode_stack(stack("[ 5 (1 2 3) ]"), 3) == [small(2), small(3), CLOSE]
    assert encode_stack((Int(-98765), Int(1)), 3) == [*digits("65"), small(1)]
    with pytest.raises(ValueError, match="cap"):
        encode_stack(values, -1)


@pytest.fixture
def default_int_digit_limit():
    """Pin the interpreter's int-to-str limit; another module may have lifted it."""
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    yield
    sys.set_int_max_str_digits(previous)


def test_an_integer_too_long_to_print_still_encodes(default_int_digit_limit):
    """`dup mul` squares per pair of ops; a search reaches this, and `str` refuses it."""
    huge = 7 ** (2**14)
    with pytest.raises(ValueError):
        str(huge)
    low = f"{huge % 10**12:012d}"
    assert encode_stack((Int(huge),), 12) == digits(low)
    assert encode_stack((Int(-huge), Int(2)), 12) == [*digits(low[1:]), small(2)]


def test_a_symbol_outside_the_basis_is_refused_not_guessed():
    with pytest.raises(ValueError, match="plus"):
        encode_stack((Sym("plus"),), 8)
    with pytest.raises(TypeError):
        encode_stack(("add",), 8)


def _reference_tokens(value) -> list[int]:
    if isinstance(value, Lst):
        return [
            OPEN,
            *(t for item in value.items for t in _reference_tokens(item)),
            CLOSE,
        ]
    if isinstance(value, Int):
        if abs(value.value) <= SMALL_INT:
            return [small(value.value)]
        sign = [NEG] if value.value < 0 else []
        return [BIG, *sign, *digits(str(abs(value.value)))]
    if isinstance(value, Bool):
        return [TRUE_TOKEN if value.value else FALSE_TOKEN]
    if isinstance(value, Sym):
        return [symbol(value.name)]
    assert value == MARK
    return [MARK_TOKEN]


def _random_value(rng: random.Random, depth: int):
    kind = rng.randrange(7 if depth < 3 else 5)
    if kind == 0:
        return Int(rng.randint(-70, 70))
    if kind == 1:
        return Int(rng.choice([-1, 1]) * rng.randint(65, 10**30))
    if kind == 2:
        return Bool(rng.random() < 0.5)
    if kind == 3:
        return Sym(rng.choice(OP_NAMES))
    if kind == 4:
        return MARK
    return Lst(tuple(_random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))))


def test_encoder_matches_a_naive_reference_at_every_cap():
    """The real walk goes top-down and stops early; this one does neither."""
    rng = random.Random(20260901)
    for _ in range(300):
        values = tuple(_random_value(rng, 0) for _ in range(rng.randint(0, 6)))
        full = [token for value in values for token in _reference_tokens(value)]
        for cap in (0, 1, 2, 3, 7, 20, TARGET_CAP, STATE_CAP, len(full), len(full) + 1):
            expected = full[len(full) - cap :] if len(full) > cap else full
            assert encode_stack(values, cap) == expected


def test_item_is_state_then_separator_then_target():
    target = encode_stack(stack("[ 7 ]"), TARGET_CAP)
    assert encode_item(stack("[ (3 4) 0 ]"), target) == [
        OPEN,
        small(3),
        small(4),
        CLOSE,
        small(0),
        SEP,
        small(7),
    ]


def test_state_and_target_are_capped_separately():
    long = tuple(Int(n % 50) for n in range(500))
    targets = encode_targets([long, stack("[ 1 ]")])
    assert [len(target) for target in targets] == [TARGET_CAP, 1]
    assert targets[0] == encode_stack(long, TARGET_CAP)
    item = encode_item(long, targets[0])
    assert len(item) == STATE_CAP + 1 + TARGET_CAP
    assert item[STATE_CAP] == SEP
    assert item[:STATE_CAP] == encode_stack(long, STATE_CAP)


def test_item_array_pads_one_node_into_an_int16_block():
    block = item_array([[5, 6, 7], [8]])
    assert block.dtype == np.int16
    assert block.tolist() == [[5, 6, 7], [8, PAD, PAD]]
    assert np.iinfo(np.int16).max >= VOCAB
    with pytest.raises(ValueError):
        item_array([])


def test_collate_follows_the_pair_batch_contract():
    blocks = [item_array([[5, 6, 7], [8]]), item_array([[9, 10, 11, 12]])]
    sequences, counts, item_mask = collate(blocks, "cpu")
    assert sequences.dtype == torch.long
    assert sequences.tolist() == [[5, 6, 7, PAD], [8, PAD, PAD, PAD], [9, 10, 11, 12]]
    assert counts.dtype == torch.long
    assert counts.tolist() == [2, 1]
    assert item_mask.dtype == torch.bool
    assert item_mask.tolist() == [[True, True], [True, False]]
    with pytest.raises(ValueError):
        collate([], "cpu")


# ---------------------------------------------------------------------------
# Model and scorer
# ---------------------------------------------------------------------------


def test_policy_is_the_aligned_pair_encoder_with_a_token_embedding():
    torch.manual_seed(0)
    model = StackPairPolicy(TINY)
    assert isinstance(model, AlignedPairPolicy)
    assert model.architecture == TINY
    assert model.action_count == len(OP_NAMES)
    assert model.embedding.num_embeddings == VOCAB
    assert model.embedding.padding_idx == PAD
    assert not model.embedding.weight[PAD].any()
    assert model.head[-1].out_features == len(OP_NAMES)


def test_parameter_count_is_the_architecture_formula_with_this_vocabulary():
    model = StackPairPolicy(TINY)
    byte_table = 258 * TINY.byte_embedding
    expected = TINY.parameter_count(len(OP_NAMES)) - byte_table + VOCAB * 8
    assert model.parameter_count == expected
    assert model.parameter_count == sum(p.numel() for p in model.parameters())


def test_padding_is_absent_not_merely_small(sum_task):
    """The same node scores the same however wide its batch was padded."""
    torch.manual_seed(1)
    model = StackPairPolicy(TINY).eval()
    narrow = block_of(sum_task, ())
    wide = block_of(sum_task, ("0", "[", "add", "add", "add", "add", "add"))
    assert wide.shape[1] > narrow.shape[1]
    with torch.inference_mode():
        alone = model(*collate([narrow], "cpu"))
        padded = model(*collate([narrow, wide], "cpu"))[:1]
    assert torch.allclose(alone, padded, atol=1e-6)


def test_examples_are_pooled_as_a_set(sum_task):
    torch.manual_seed(2)
    model = StackPairPolicy(TINY).eval()
    block = block_of(sum_task, ("0",))
    with torch.inference_mode():
        forward = model(*collate([block], "cpu"))
        backward = model(*collate([block[::-1].copy()], "cpu"))
    assert torch.allclose(forward, backward, atol=1e-6)


def test_scorer_normalises_over_the_legal_ops_only(sum_task):
    torch.manual_seed(3)
    model = StackPairPolicy(TINY)
    model.train()
    scorer = PolicyScorer(model)
    assert not scorer.model.training
    blocks = [block_of(sum_task, ()), block_of(sum_task, ("0",))]
    legal = np.zeros((2, len(OP_NAMES)), dtype=bool)
    legal[0, [0, 3, 9]] = True
    legal[1, 5:] = True
    scores = scorer.log_probs(blocks, legal)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2, len(OP_NAMES))
    assert np.isneginf(scores[~legal]).all()
    assert np.isfinite(scores[legal]).all()
    assert np.allclose(np.exp(scores).sum(axis=1), 1.0)
    with torch.inference_mode():
        logits = model(*collate(blocks, "cpu")).double()
    expected = torch.log_softmax(
        logits.masked_fill(~torch.from_numpy(legal), -math.inf), dim=1
    )
    assert np.allclose(scores[legal], expected.numpy()[legal], atol=1e-9)


def test_scorer_gives_a_dead_end_minus_infinity_rather_than_nan(sum_task):
    scorer = PolicyScorer(StackPairPolicy(TINY))
    blocks = [block_of(sum_task, ()), block_of(sum_task, ())]
    legal = np.zeros((2, len(OP_NAMES)), dtype=bool)
    legal[1, 4] = True
    with np.errstate(all="raise"):
        scores = scorer.log_probs(blocks, legal)
    assert np.isneginf(scores[0]).all()
    assert scores[1, 4] == 0.0
    assert not np.isnan(scores).any()


def test_scorer_checks_the_mask_against_the_batch(sum_task):
    scorer = PolicyScorer(StackPairPolicy(TINY))
    block = block_of(sum_task, ())
    with pytest.raises(ValueError, match="legal"):
        scorer.log_probs([block], np.ones((2, len(OP_NAMES)), dtype=bool))
    with pytest.raises(ValueError, match="legal"):
        scorer.log_probs([block], np.ones((1, 3), dtype=bool))
    empty = scorer.log_probs([], np.zeros((0, len(OP_NAMES)), dtype=bool))
    assert empty.shape == (0, len(OP_NAMES))


# ---------------------------------------------------------------------------
# Witness rows
# ---------------------------------------------------------------------------


def test_replay_yields_each_prefix_state_and_the_op_that_follows(sum_task):
    steps = list(replay_witness(sum_task))
    assert [step for step, _product, _name in steps] == list(range(sum_task.length))
    assert tuple(name for _step, _product, name in steps) == sum_task.program
    assert steps[0][1] == tuple(stack(text) for text in sum_task.inputs)
    for step, product, _name in steps:
        assert product == product_of(sum_task, sum_task.program[:step])
    # The step after the last one is the shown outputs: the labels lead there.
    _step, last, name = steps[-1]
    final, _fuel, _stepped = step_all(last, OP_BY_NAME[name])
    assert final == tuple(stack(text) for text in sum_task.outputs)


def test_replay_reads_the_shown_examples_only(sum_task):
    blinded = replace(sum_task, held_out_inputs=(), held_out_outputs=(), held_out=0)
    assert list(replay_witness(blinded)) == list(replay_witness(sum_task))
    broken = replace(sum_task, held_out_inputs=tuple("[ 1 ]" for _ in range(6)))
    assert list(replay_witness(broken)) == list(replay_witness(sum_task))


def test_replay_raises_when_the_machine_refuses(sum_task):
    with pytest.raises(ValueError, match="refused at step 1"):
        list(replay_witness(replace(sum_task, program=("0", "add"))))
    with pytest.raises(ValueError, match="plus"):
        list(replay_witness(replace(sum_task, program=("0", "plus"))))
    with pytest.raises(ValueError):
        witness_rows([replace(sum_task, program=("0", "add"))])


def test_witness_rows_are_one_per_prefix(sum_task, add_task, maximum_task):
    tasks = [sum_task, add_task, maximum_task]
    rows = witness_rows(tasks)
    assert len(rows) == sum(task.length for task in tasks) == 5 + 1 + 16
    cursor = 0
    for index, task in enumerate(tasks):
        for step, name in enumerate(task.program):
            row = rows[cursor]
            cursor += 1
            assert isinstance(row, Row)
            assert (row.task, row.step) == (index, step)
            assert row.label == OP_INDEX[name]
            assert row.remaining == task.length - step
            assert row.items.dtype == np.int16
            assert len(row.items) == task.examples == len(task.inputs)
            assert np.array_equal(row.items, block_of(task, task.program[:step]))
    assert rows[0].remaining == sum_task.length
    assert rows[sum_task.length - 1].remaining == 1
    assert witness_rows([]) == []


def test_row_is_frozen_and_slotted(add_task):
    row = witness_rows([add_task])[0]
    with pytest.raises(FrozenInstanceError):
        row.label = 3
    assert not hasattr(row, "__dict__")
    assert Row.__slots__ == ("items", "label", "task", "step", "remaining")


def test_every_item_ends_in_its_own_example_target(sum_task):
    row = witness_rows([sum_task])[0]
    for item, given, wanted in zip(
        row.items.tolist(), sum_task.inputs, sum_task.outputs, strict=True
    ):
        tokens = [token for token in item if token != PAD]
        assert tokens == [
            *encode_stack(stack(given), STATE_CAP),
            SEP,
            *encode_stack(stack(wanted), TARGET_CAP),
        ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sum_rows():
    """Witness rows for two five-op folds: enough to learn, quick to fit."""
    tasks = [make_task("sum", seed) for seed in range(12)]
    tasks += [make_task("reverse", seed) for seed in range(12)]
    return witness_rows(tasks[2:]), witness_rows(tasks[:2])


def fit(rows, **overrides):
    train, validation = rows
    settings = {
        "epochs": 8,
        "learning_rate": 0.01,
        "batch_size": 16,
        "validation_batch_size": 7,
        "seed": 5,
        "device": "cpu",
    }
    return train_policy(train, validation, TINY, **{**settings, **overrides})


def validation_cross_entropy(model, rows) -> float:
    with torch.inference_mode():
        logits = model(*collate([row.items for row in rows], "cpu"))
        labels = torch.tensor([row.label for row in rows])
        return torch.nn.functional.cross_entropy(logits, labels).item()


def test_training_is_deterministic_for_one_seed(sum_rows):
    first_model, first_state = fit(sum_rows)
    second_model, second_state = fit(sum_rows)
    assert first_state == second_state
    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name])
    _other, other_state = fit(sum_rows, seed=6)
    assert other_state["history"] != first_state["history"]


def test_training_learns_and_reports_every_epoch(sum_rows):
    seen = []
    model, state = fit(sum_rows, epoch_callback=seen.append)
    history = state["history"]
    assert seen == history
    assert [row["epoch"] for row in history] == list(range(1, 9))
    assert history[-1]["train_cross_entropy"] < history[0]["train_cross_entropy"]
    # Well under the log(28) of a policy that has learned nothing.
    assert state["best_validation_cross_entropy"] < math.log(len(OP_NAMES)) / 2
    assert state["parameters"] == model.parameter_count
    assert isinstance(model, StackPairPolicy)
    assert not model.training
    json.dumps(state)


def test_training_returns_the_lowest_validation_epoch_not_the_last(sum_rows):
    """Validate against labels the training set contradicts, so fitting hurts.

    Every epoch then scores worse than the one before, the best epoch is an
    early one whatever the arithmetic does, and the weights handed back have
    to be that epoch's rather than the final ones.
    """
    train, _validation = sum_rows
    wrong = [replace(row, label=(row.label + 1) % len(OP_NAMES)) for row in train]
    model, state = fit((train, wrong))
    losses = [row["validation_cross_entropy"] for row in state["history"]]
    assert state["best_validation_cross_entropy"] == min(losses)
    assert state["best_epoch"] == losses.index(min(losses)) + 1
    assert state["best_epoch"] < len(losses)
    assert losses[-1] > 2 * min(losses)
    assert validation_cross_entropy(model, wrong) == pytest.approx(
        min(losses), rel=1e-4
    )


def test_training_refuses_an_empty_side(sum_rows):
    train, validation = sum_rows
    with pytest.raises(ValueError):
        fit(([], validation))
    with pytest.raises(ValueError):
        fit((train, []))
    with pytest.raises(ValueError):
        fit(sum_rows, epochs=0)


def test_training_leaves_the_determinism_switch_as_it_found_it(sum_rows):
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        fit(sum_rows, epochs=1)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)


def test_validation_skeleton_holds_out_whole_skeletons():
    tasks = [make_task(name, 0) for name in sorted(BY_NAME)[:120]]
    by_skeleton: dict[str, set[bool]] = {}
    for task in tasks:
        by_skeleton.setdefault(task.skeleton, set()).add(validation_skeleton(task))
    assert all(len(verdicts) == 1 for verdicts in by_skeleton.values())
    held = sum(True in verdicts for verdicts in by_skeleton.values())
    assert 0.03 < held / len(by_skeleton) < 0.25
    # Deterministic, and independent of the data a task happened to draw.
    assert validation_skeleton(make_task("sum", 1)) == validation_skeleton(
        make_task("sum", 2)
    )


def test_validation_skeleton_rate_follows_out_of_and_falls_back_to_behaviour(sum_task):
    tasks = [make_task(name, 0) for name in sorted(BY_NAME)[:60]]
    assert all(validation_skeleton(task, out_of=1) for task in tasks)
    assert sum(validation_skeleton(task, out_of=2) for task in tasks) > sum(
        validation_skeleton(task, out_of=20) for task in tasks
    )
    bare = replace(sum_task, skeleton="", behaviour="_ [ _ ] fold")
    assert validation_skeleton(bare) == validation_skeleton(sum_task)
    with pytest.raises(ValueError):
        validation_skeleton(sum_task, out_of=0)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

URL = "https://wandb.ai/abhishekraok-na/stack-policy/runs/abc123"


@pytest.fixture
def saved(tmp_path, sum_task):
    torch.manual_seed(7)
    model = StackPairPolicy(TINY)
    path = save_policy(
        tmp_path / "runs" / "policy",
        model,
        trainer_state={"best_epoch": 3, "history": []},
        manifest={"corpus": "stack_pbe_v4", "seed": 7},
        wandb_url=URL,
    )
    return path, model


def test_save_then_load_scores_identically(saved, sum_task):
    path, model = saved
    loaded = load_policy(path)
    assert isinstance(loaded, StackPairPolicy)
    assert loaded.architecture == TINY
    assert not loaded.training
    inputs = collate([block_of(sum_task, ()), block_of(sum_task, ("0", "["))], "cpu")
    with torch.inference_mode():
        assert torch.equal(loaded(*inputs), model.eval()(*inputs))


def test_checkpoint_is_weights_config_state_manifest_and_link(saved, tmp_path):
    path, _model = saved
    assert path == tmp_path / "runs" / "policy"
    assert sorted(entry.name for entry in path.iterdir()) == sorted(
        [MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE, MANIFEST_FILE, WANDB_LINK_FILE]
    )
    assert (MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE) == (
        "model.safetensors",
        "policy_config.json",
        "trainer_state.json",
    )
    assert (MANIFEST_FILE, WANDB_LINK_FILE) == (
        "training_manifest.json",
        "wandb_run.url",
    )
    # Nothing left beside it by the staged write.
    assert [entry.name for entry in path.parent.iterdir()] == ["policy"]

    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["schema_version"] == stack_policy.SCHEMA_VERSION
    assert config["basis"] == "mark-28"
    assert config["ordered_actions"] == list(OP_NAMES)
    assert config["vocabulary"]["size"] == VOCAB
    dims = config["architecture"]
    assert (dims["byte_embedding"], dims["conv_channels"], dims["hidden"]) == (
        8,
        12,
        16,
    )

    assert json.loads((path / TRAINER_STATE_FILE).read_text()) == {
        "best_epoch": 3,
        "history": [],
    }
    assert json.loads((path / MANIFEST_FILE).read_text()) == {
        "corpus": "stack_pbe_v4",
        "seed": 7,
        "wandb_url": URL,
    }
    assert f"URL={URL}" in (path / WANDB_LINK_FILE).read_text().splitlines()


def test_save_without_a_run_writes_no_link(tmp_path):
    manifest = {"corpus": "stack_pbe_v4"}
    path = save_policy(
        tmp_path / "policy",
        StackPairPolicy(TINY),
        trainer_state={},
        manifest=manifest,
        wandb_url=None,
    )
    assert not (path / WANDB_LINK_FILE).exists()
    assert json.loads((path / MANIFEST_FILE).read_text())["wandb_url"] is None
    assert manifest == {"corpus": "stack_pbe_v4"}
    assert isinstance(load_policy(path), StackPairPolicy)


def test_save_refuses_an_existing_directory(saved, tmp_path):
    path, model = saved
    before = (path / MODEL_FILE).read_bytes()
    with pytest.raises(FileExistsError):
        save_policy(path, model, trainer_state={}, manifest={}, wandb_url=None)
    assert (path / MODEL_FILE).read_bytes() == before
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        save_policy(empty, model, trainer_state={}, manifest={}, wandb_url=None)
    assert list(empty.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 99),
        ("basis", "executable-26"),
        ("ordered_actions", list(reversed(OP_NAMES))),
        ("ordered_actions", list(OP_NAMES[:-1])),
    ],
)
def test_load_refuses_a_config_it_would_misread(saved, field, value):
    path, _model = saved
    config = json.loads((path / CONFIG_FILE).read_text())
    config[field] = value
    (path / CONFIG_FILE).write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_policy(path)


# ---------------------------------------------------------------------------
# step_all
# ---------------------------------------------------------------------------


def test_step_all_reports_what_the_attempt_cost():
    product = (stack("[ (1 2) ]"), stack("[ nil ]"), stack("[ (3) ]"))
    moved, fuel, stepped = step_all(product, OP_BY_NAME["dup"])
    assert moved == tuple((*one, *one) for one in product)
    assert (fuel, stepped) == (3, 3)
    # The first example accepts `uncons`, the second refuses it, the third is
    # never asked: two stacks stepped, one unit of fuel spent.
    assert step_all(product, OP_BY_NAME["uncons"]) == (None, 1, 2)
    assert step_all(product, OP_BY_NAME["add"]) == (None, 0, 1)


def test_step_all_charges_the_fuel_a_combinator_spends():
    product = (stack("[ (1 2) 0 (add) ]"), stack("[ (1 2 4) 0 (add) ]"))
    moved, fuel, stepped = step_all(product, OP_BY_NAME["fold"])
    assert moved == (stack("[ 3 ]"), stack("[ 7 ]"))
    assert stepped == 2
    assert fuel > stepped
    assert fuel < DEFAULT_FUEL_LIMIT


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def walk_cost(task, program):
    """Executions and fuel for a width-one beam that follows `program`."""
    executions = fuel = 0
    for step, name in enumerate(program):
        product = product_of(task, program[:step])
        last = step == len(program) - 1
        for op in MARK_OPS:
            _moved, spent, stepped = step_all(product, op)
            executions += stepped
            fuel += spent
            if last and op.name == name:
                break
    return executions, fuel


def test_beam_finds_a_one_op_program(add_task):
    scorer = UniformScorer()
    result = beam_solve(add_task, scorer)
    assert isinstance(result, BeamResult)
    assert result.program == ("add",)
    assert result.termination == "fit"
    assert result.depth == 1
    assert result.expansions == 1
    assert (result.executions, result.fuel) == walk_cost(add_task, ("add",))
    assert result.seconds >= 0
    assert scorer.calls == []  # a hit in the first layer never needs a score
    assert grade(result.program, add_task).solved


def test_result_is_frozen(add_task):
    result = beam_solve(add_task, UniformScorer())
    with pytest.raises(FrozenInstanceError):
        result.depth = 0


def test_a_width_one_beam_follows_the_policy_to_the_witness(sum_task):
    result = beam_solve(sum_task, WitnessScorer(sum_task.program), beam_width=1)
    assert result.program == sum_task.program
    assert result.termination == "fit"
    assert result.depth == 5
    assert result.expansions == 5
    assert (result.executions, result.fuel) == walk_cost(sum_task, sum_task.program)
    assert grade(result.program, sum_task).solved


def test_cost_counts_refused_attempts_and_the_fuel_combinators_spend(sum_task):
    result = beam_solve(sum_task, WitnessScorer(sum_task.program), beam_width=1)
    # Five layers of 28 ops would be 140 attempts on a single example. The
    # search stops at `fold` in the last layer, and every refusal still costs.
    attempts = 4 * len(MARK_OPS) + OP_INDEX["fold"] + 1
    assert attempts <= result.executions <= attempts * sum_task.examples
    assert result.executions > attempts  # more than one example was stepped
    assert result.fuel != result.executions


def test_guard_keeps_the_witness_at_a_depth_cap_with_no_slack(maximum_task):
    """`remaining` off by one would prune the last op before a closing `]`."""
    program = maximum_task.program
    for guard in ("off", "body", "close", "both"):
        result = beam_solve(
            maximum_task,
            WitnessScorer(program),
            beam_width=1,
            max_depth=len(program),
            guard=guard,
        )
        assert result.program == program, guard
        assert (result.termination, result.depth) == ("fit", 16)


def test_close_guard_lets_a_mark_live_until_its_last_chance(add_task):
    """`[ add ]` at a cap of three: the mark survives with exactly one op left."""
    quoting = replace(
        add_task,
        inputs=("[ 3 ]", "[ 4 ]"),
        outputs=("[ 3 (add) ]", "[ 4 (add) ]"),
        program=("[", "add", "]"),
        length=3,
    )
    for guard in ("close", "both"):
        result = beam_solve(
            quoting,
            WitnessScorer(quoting.program),
            beam_width=1,
            max_depth=3,
            guard=guard,
        )
        assert (result.program, result.termination) == (quoting.program, "fit")
    # One op fewer and the fit is out of reach; the search says so and stops.
    short = beam_solve(
        quoting, WitnessScorer(quoting.program), beam_width=1, max_depth=2
    )
    assert (short.program, short.termination, short.depth) == (None, "max_depth", 2)


def test_beam_never_reads_the_held_out_examples(sum_task):
    blinded = replace(sum_task, held_out_inputs=(), held_out_outputs=(), held_out=0)
    seen = beam_solve(sum_task, UniformScorer(), beam_width=8, max_depth=4)
    blind = beam_solve(blinded, UniformScorer(), beam_width=8, max_depth=4)
    assert replace(seen, seconds=0.0) == replace(blind, seconds=0.0)


def test_beam_keeps_at_most_its_width_and_scores_each_layer_once(sum_task):
    scorer = UniformScorer()
    result = beam_solve(sum_task, scorer, beam_width=3, max_depth=4)
    assert result.program is None
    assert (result.termination, result.depth) == ("max_depth", 4)
    assert scorer.calls == [1, 3, 3, 3]
    assert result.expansions == 1 + 3 + 3 + 3
    assert beam_solve(sum_task, scorer, max_depth=0).termination == "max_depth"


def test_successors_rank_by_parent_score_plus_log_prob(add_task):
    """Greedy per-layer ranking would put `0`'s children first; the sum does not."""
    task = replace(add_task, outputs=tuple("[ 99 98 97 ]" for _ in add_task.outputs))
    scorer = TableScorer(
        [
            {(0, "0"): -0.1, (0, "1"): -3.0},
            # Node 0 is `0` and node 1 is `1`, in score order.
            {(0, "dup"): -5.0, (1, "dup"): -0.5, (1, "nil"): -4.0},
        ]
    )
    beam_solve(task, scorer, beam_width=2, max_depth=3, guard="off")
    first, second, third = scorer.blocks
    assert len(first) == 1
    assert [block.tolist() for block in second] == [
        block_of(task, ("0",)).tolist(),
        block_of(task, ("1",)).tolist(),
    ]
    # -3.0 + -0.5 beats -0.1 + -5.0, which beats -3.0 + -4.0.
    assert [block.tolist() for block in third] == [
        block_of(task, ("1", "dup")).tolist(),
        block_of(task, ("0", "dup")).tolist(),
    ]


def test_a_state_already_held_is_not_kept_again(add_task):
    """`dup drop` is the root again, so the slot goes to the next best."""
    task = replace(add_task, outputs=tuple("[ 99 98 97 ]" for _ in add_task.outputs))
    scorer = TableScorer(
        [{(0, "dup"): -0.1}, {(0, "drop"): -0.1, (0, "over"): -1.0}, {}]
    )
    beam_solve(task, scorer, beam_width=1, max_depth=3, guard="off")
    assert scorer.blocks[2][0].tolist() == block_of(task, ("dup", "over")).tolist()


def test_a_state_dropped_for_width_may_be_held_later(add_task):
    """Only states a beam has held are closed; `nil` loses its slot, not its future."""
    task = replace(add_task, outputs=tuple("[ 99 98 97 ]" for _ in add_task.outputs))
    assert product_of(task, ("[", "]")) == product_of(task, ("nil",))
    scorer = TableScorer([{(0, "["): -0.1, (0, "nil"): -0.2}, {(0, "]"): -0.1}, {}])
    beam_solve(task, scorer, beam_width=1, max_depth=3, guard="off")
    assert scorer.blocks[2][0].tolist() == block_of(task, ("nil",)).tolist()


def test_two_parents_reaching_one_product_keep_it_once(add_task):
    """`dup sub` and `0 mul` both leave `a 0`; the better-scored path holds it."""
    task = replace(add_task, outputs=tuple("[ 99 98 97 ]" for _ in add_task.outputs))
    assert product_of(task, ("dup", "sub")) == product_of(task, ("0", "mul"))
    scorer = TableScorer(
        [
            {(0, "dup"): -0.1, (0, "0"): -0.2},
            {(0, "sub"): -0.1, (1, "mul"): -0.1, (1, "1"): -1.0},
            {},
        ]
    )
    beam_solve(task, scorer, beam_width=2, max_depth=3, guard="off")
    assert [block.tolist() for block in scorer.blocks[2]] == [
        block_of(task, ("dup", "sub")).tolist(),
        block_of(task, ("0", "1")).tolist(),
    ]


def test_budget_is_checked_before_every_attempt(sum_task):
    free = beam_solve(sum_task, WitnessScorer(sum_task.program), beam_width=1)
    assert free.termination == "fit"

    nothing = beam_solve(sum_task, WitnessScorer(sum_task.program), budget=0)
    assert (nothing.program, nothing.termination) == (None, "budget")
    assert (nothing.executions, nothing.fuel, nothing.depth) == (0, 0, 1)

    # One short of what the fit needs: the last attempt is never made.
    short = beam_solve(
        sum_task,
        WitnessScorer(sum_task.program),
        beam_width=1,
        budget=free.executions - sum_task.examples,
    )
    assert (short.program, short.termination) == (None, "budget")
    assert short.depth == 5
    assert short.executions == free.executions - sum_task.examples
    assert short.expansions == 5

    # A budget that runs out mid-attempt lets that attempt finish, no more.
    exact = beam_solve(
        sum_task,
        WitnessScorer(sum_task.program),
        beam_width=1,
        budget=free.executions - sum_task.examples + 1,
    )
    assert exact.termination == "fit"
    assert replace(exact, seconds=0.0) == replace(free, seconds=0.0)

    early = beam_solve(sum_task, UniformScorer(), beam_width=4, budget=10)
    assert early.termination == "budget"
    assert 10 <= early.executions < 10 + sum_task.examples
    assert (early.depth, early.expansions) == (1, 1)


def test_frontier_exhausted_when_no_op_applies(add_task, monkeypatch):
    """Unreachable on mark-28, where a literal is always legal; kept honest anyway."""
    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["fold"], OP_BY_NAME["cat"]))
    result = beam_solve(add_task, UniformScorer(), guard="off")
    assert (result.program, result.termination) == (None, "frontier_exhausted")
    assert (result.depth, result.expansions) == (0, 1)
    assert result.executions == 2  # each op refused by the first example


def test_frontier_exhausted_when_every_successor_was_already_held(
    add_task, monkeypatch
):
    monkeypatch.setattr(stack_beam, "OPS", (OP_BY_NAME["swap"],))
    task = replace(
        add_task,
        inputs=("[ 1 2 ]", "[ 3 4 ]"),
        outputs=("[ 9 ]", "[ 9 ]"),
    )
    scorer = UniformScorer()
    result = beam_solve(task, scorer, guard="off")
    assert (result.program, result.termination) == (None, "frontier_exhausted")
    # `swap` is kept once; swapping back is the root again.
    assert (result.depth, result.expansions, result.executions) == (1, 2, 4)
    assert scorer.calls == [1, 1]


def test_close_guard_refuses_a_mark_bearing_target(add_task):
    marked = replace(add_task, outputs=tuple("[ 1 MARK ]" for _ in add_task.outputs))
    for guard in ("close", "both"):
        with pytest.raises(ValueError, match="mark-free"):
            beam_solve(marked, UniformScorer(), max_depth=2, guard=guard)
    with pytest.raises(ValueError, match="mark-free"):
        beam_solve(marked, UniformScorer(), max_depth=2)  # the default is both
    for guard in ("body", "off"):
        result = beam_solve(
            marked, UniformScorer(), beam_width=4, max_depth=2, guard=guard
        )
        assert result.termination == "max_depth"
    with pytest.raises(ValueError, match="guard"):
        beam_solve(add_task, UniformScorer(), guard="sometimes")


def test_guard_drops_states_that_cannot_close_in_time(sum_task):
    """With one op left, a state holding two marks is not worth a beam slot."""
    scorer = TableScorer([{(0, "["): -0.1}, {(0, "["): -0.1, (0, "0"): -1.0}, {}])
    beam_solve(sum_task, scorer, beam_width=1, max_depth=3, guard="close")
    assert scorer.blocks[2][0].tolist() == block_of(sum_task, ("[", "0")).tolist()

    scorer = TableScorer([{(0, "["): -0.1}, {(0, "["): -0.1, (0, "0"): -1.0}, {}])
    beam_solve(sum_task, scorer, beam_width=1, max_depth=3, guard="off")
    assert scorer.blocks[2][0].tolist() == block_of(sum_task, ("[", "[")).tolist()


def test_a_trained_policy_solves_what_a_blind_beam_of_that_width_cannot(sum_rows):
    """End to end: witness rows -> weights -> scorer -> a fit on unseen data."""
    model, _state = fit(sum_rows, epochs=20, learning_rate=0.02, batch_size=8)
    scorer = PolicyScorer(model)
    for name in ("sum", "reverse"):
        task = make_task(name, 1000)
        blind = beam_solve(task, UniformScorer(), beam_width=2, max_depth=8)
        assert blind.program is None
        guided = beam_solve(task, scorer, beam_width=2, max_depth=8)
        assert guided.termination == "fit"
        assert guided.depth == 5
        assert guided.expansions <= 1 + 2 * 4
        assert grade(guided.program, task).solved


def test_results_agree_on_what_a_state_is(sum_task):
    """The hit is structural equality with the shown outputs, as in `bfs_solve`."""
    result = beam_solve(sum_task, WitnessScorer(sum_task.program), beam_width=1)
    final = product_of(sum_task, result.program)
    wanted = tuple(stack(text) for text in sum_task.outputs)
    assert [[value_key(v) for v in one] for one in final] == [
        [value_key(v) for v in one] for one in wanted
    ]
