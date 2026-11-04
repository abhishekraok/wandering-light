"""The `difference` item layout: flags of the full stacks after a second SEP.

`machine` is every published checkpoint's layout and must stay byte-identical;
`difference` is the inverse-dynamics probe's items with the task's shown outputs
as the target. The layout is fixed per model, recorded in its checkpoint, and
every block a search or a training row builds goes through `node_encoder`.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve, block_encoder, segment_beam_solve
from wandering_light.stack_machine import Int, Lst, Sym, parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_policy import (
    ADDED,
    CHANGED,
    CONFIG_FILE,
    DIFFERENCE,
    EMBEDDING_ROWS,
    ENCODINGS,
    EQUAL,
    FLAG_BASE,
    FLAG_TOP,
    MACHINE,
    OP_NAMES,
    PAD,
    REMOVED,
    SEP,
    STATE_CAP,
    VOCAB,
    PolicyScorer,
    StackPairPolicy,
    encode_item,
    encode_node,
    encode_stack,
    encode_targets,
    flags,
    item_array,
    load_policy,
    node_encoder,
    policy_config,
    save_policy,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def stacks(*texts):
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def machine_block(product, outputs):
    """Today's block, built the way every search built it before layouts."""
    targets = encode_targets(outputs)
    return item_array(
        [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
    )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_constants():
    assert ENCODINGS == (MACHINE, DIFFERENCE) == ("machine", "difference")
    assert FLAG_TOP == 8
    assert (EQUAL, CHANGED, ADDED, REMOVED) == (0, 1, 2, 3)
    assert FLAG_BASE == VOCAB
    assert EMBEDDING_ROWS == {MACHINE: VOCAB, DIFFERENCE: VOCAB + 4 * FLAG_TOP}


def test_flags_compare_positions_from_the_top():
    (state,) = stacks("1 2 3")
    (target,) = stacks("9 2 3")  # top two kept, the bottom changed
    assert flags(state, target) == [
        FLAG_BASE + 4 * 0 + EQUAL,
        FLAG_BASE + 4 * 1 + EQUAL,
        FLAG_BASE + 4 * 2 + CHANGED,
    ]
    (taller,) = stacks("7 1 2 3")  # the target has one more item underneath
    assert flags(state, taller)[3] == FLAG_BASE + 4 * 3 + ADDED
    assert flags(taller, state)[3] == FLAG_BASE + 4 * 3 + REMOVED
    assert flags((), ()) == []
    assert flags(state, state) == [FLAG_BASE + 4 * k + EQUAL for k in range(3)]


def test_flags_stop_at_the_top_positions_and_skip_absent_ones():
    (state,) = stacks(" ".join(str(i) for i in range(20)))
    (target,) = stacks("5")
    out = flags(state, target)
    assert len(out) == FLAG_TOP
    assert out[0] == FLAG_BASE + CHANGED  # 19 against 5
    assert out[1:] == [FLAG_BASE + 4 * k + REMOVED for k in range(1, FLAG_TOP)]
    assert flags((), target) == [FLAG_BASE + ADDED]
    assert max(out) < EMBEDDING_ROWS[DIFFERENCE]


def test_flags_are_structural_over_values():
    state = (Lst((Int(1), Int(2))), Sym("add"))
    assert flags(state, (Lst((Int(1), Int(2))), Sym("add"))) == [
        FLAG_BASE + EQUAL,
        FLAG_BASE + 4 + EQUAL,
    ]
    assert (
        flags(state, (Lst((Int(1), Int(3))), Sym("add")))[1] == FLAG_BASE + 4 + CHANGED
    )
    assert flags(state, (Lst((Int(1), Int(2))), Sym("mul")))[0] == FLAG_BASE + CHANGED


def test_flags_match_the_inverse_dynamics_probe():
    probe = pytest.importorskip("experiments.inverse_dynamics")
    assert (probe.TOP, probe.FLAG_BASE) == (FLAG_TOP, FLAG_BASE)
    assert (probe.EQUAL, probe.CHANGED, probe.ADDED, probe.REMOVED) == (
        EQUAL,
        CHANGED,
        ADDED,
        REMOVED,
    )
    rng = random.Random(0)
    for _ in range(200):
        values = [Int(rng.randint(0, 3)) for _ in range(rng.randint(0, 12))]
        other = [
            v if rng.random() < 0.6 else Int(rng.randint(0, 3))
            for v in values[rng.randint(0, len(values)) :]
        ]
        rng.shuffle(other)
        assert flags(tuple(values), tuple(other)) == probe.flags(values, other)
        assert flags(tuple(other), tuple(values)) == probe.flags(other, values)


def test_flags_read_the_full_stacks_not_the_truncated_tokens():
    """A state wider than STATE_CAP loses its bottom tokens, never its flags."""
    tall = tuple(Int(i % 50) for i in range(STATE_CAP + 40))
    assert len(encode_stack(tall, STATE_CAP)) == STATE_CAP
    (short,) = stacks("1")
    out = flags(tall, short)
    assert out[0] == FLAG_BASE + CHANGED and len(out) == FLAG_TOP
    assert flags(tall, tall) == [FLAG_BASE + 4 * k + EQUAL for k in range(FLAG_TOP)]
    # And in a block: the item is the truncated tokens, SEP, then the flags.
    block = encode_node((tall,), (tall,), DIFFERENCE)
    row = [int(t) for t in block[0] if t != PAD]
    assert row[STATE_CAP] == SEP and row[-FLAG_TOP - 1] == SEP
    assert row[-FLAG_TOP:] == flags(tall, tall)


# ---------------------------------------------------------------------------
# Node blocks
# ---------------------------------------------------------------------------


def test_machine_blocks_are_byte_identical_to_today():
    task = task_for("sum_positive")
    product = stacks(*task.inputs)
    outputs = stacks(*task.outputs)
    expected = machine_block(product, outputs)
    for block in (
        node_encoder(outputs)(product),
        node_encoder(outputs, MACHINE)(product),
        encode_node(product, outputs),
        PolicyScorer(StackPairPolicy(TINY)).encode(product, outputs),
    ):
        assert block.dtype == np.int16
        assert block.shape == expected.shape
        assert block.tobytes() == expected.tobytes()


def test_difference_items_are_machine_items_then_sep_then_flags():
    task = task_for("sum_positive")
    product = stacks(*task.inputs)
    outputs = stacks(*task.outputs)
    targets = encode_targets(outputs)
    block = node_encoder(outputs, DIFFERENCE)(product)
    assert block.dtype == np.int16 and block.shape[0] == task.examples
    for row, state, target, output in zip(
        block, product, targets, outputs, strict=True
    ):
        tokens = [int(t) for t in row if t != PAD]
        assert tokens == [*encode_item(state, target), SEP, *flags(state, output)]
        assert tokens.count(SEP) == 2
    assert np.array_equal(block, encode_node(product, outputs, DIFFERENCE))
    # The flags are the only tokens past VOCAB, and they sit in the flag rows.
    assert (block >= VOCAB).sum() == sum(
        len(flags(s, o)) for s, o in zip(product, outputs, strict=True)
    )
    assert block.max() < EMBEDDING_ROWS[DIFFERENCE]


def test_the_encoder_encodes_the_targets_once_and_is_reusable():
    task = task_for("sum_positive")
    outputs = stacks(*task.outputs)
    inputs = stacks(*task.inputs)
    encode = node_encoder(outputs, DIFFERENCE)
    assert np.array_equal(encode(inputs), encode(inputs))
    assert np.array_equal(encode(outputs), encode(outputs))
    # At the target every flag is EQUAL.
    at_target = [int(t) for t in encode(outputs)[0] if t >= VOCAB]
    assert all((t - FLAG_BASE) % 4 == EQUAL for t in at_target)


def test_an_unknown_encoding_is_refused_everywhere():
    outputs = stacks("1")
    message = "unknown encoding 'bytes'; choose one of"
    with pytest.raises(ValueError, match=message):
        node_encoder(outputs, "bytes")
    with pytest.raises(ValueError, match=message):
        encode_node(outputs, outputs, "bytes")
    with pytest.raises(ValueError, match=message):
        StackPairPolicy(TINY, encoding="bytes")
    with pytest.raises(ValueError, match=message):
        policy_config(TINY, encoding="bytes")
    with pytest.raises(ValueError, match=message):
        witness_rows([task_for("sum")], "bytes")


# ---------------------------------------------------------------------------
# Model, scorer, rows
# ---------------------------------------------------------------------------


def test_the_model_sizes_its_embedding_by_layout_with_flag_rows_last():
    machine = StackPairPolicy(TINY)
    assert machine.encoding == MACHINE
    assert machine.embedding.num_embeddings == VOCAB
    difference = StackPairPolicy(TINY, encoding=DIFFERENCE)
    assert difference.encoding == DIFFERENCE
    assert difference.embedding.num_embeddings == EMBEDDING_ROWS[DIFFERENCE]
    # Nothing else about the model changes: the same parameters, the same
    # shapes, except the embedding's extra rows.
    a = dict(machine.named_parameters())
    b = dict(difference.named_parameters())
    assert a.keys() == b.keys()
    for name, parameter in a.items():
        if name == "embedding.weight":
            assert b[name].shape == (EMBEDDING_ROWS[DIFFERENCE], TINY.byte_embedding)
            assert parameter.shape == (VOCAB, TINY.byte_embedding)
        else:
            assert parameter.shape == b[name].shape
    assert (
        difference.parameter_count - machine.parameter_count
        == 4 * FLAG_TOP * TINY.byte_embedding
    )


def test_the_scorer_reports_and_encodes_in_its_model_layout():
    task = task_for("sum_positive")
    product, outputs = stacks(*task.inputs), stacks(*task.outputs)
    for encoding in ENCODINGS:
        scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding), "cpu")
        assert scorer.encoding == encoding
        block = scorer.encode(product, outputs)
        assert np.array_equal(block, node_encoder(outputs, encoding)(product))
        assert np.array_equal(scorer.encoder(outputs)(product), block)
        assert np.array_equal(block_encoder(scorer, outputs)(product), block)
        legal = np.ones((1, len(OP_NAMES)), bool)
        rows = scorer.log_probs([block], legal)
        assert rows.shape == (1, len(OP_NAMES))
        assert np.isclose(np.exp(rows).sum(), 1.0)

    class Bare:
        def __init__(self, model):
            self.model = model

    bare = Bare(StackPairPolicy(TINY, encoding=DIFFERENCE))
    assert PolicyScorer.encoding.fget(Bare(object())) == MACHINE
    assert PolicyScorer.encoding.fget(bare) == DIFFERENCE


def test_a_machine_model_cannot_read_a_difference_block():
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    outputs = stacks("1 2")
    block = node_encoder(outputs, DIFFERENCE)(stacks("2 2"))
    with pytest.raises((IndexError, RuntimeError)):
        scorer.log_probs([block], np.ones((1, len(OP_NAMES)), bool))


def test_witness_rows_build_items_in_the_layout():
    tasks = [task_for("sum"), task_for("sum_positive", 5)]
    machine = witness_rows(tasks)
    same = witness_rows(tasks, MACHINE)
    difference = witness_rows(tasks, DIFFERENCE)
    assert len(machine) == len(same) == len(difference) == sum(t.length for t in tasks)
    for a, b, c in zip(machine, same, difference, strict=True):
        assert (a.label, a.task, a.step, a.remaining) == (
            b.label,
            b.task,
            b.step,
            b.remaining,
        )
        assert (a.label, a.task, a.step, a.remaining) == (
            c.label,
            c.task,
            c.step,
            c.remaining,
        )
        assert a.items.tobytes() == b.items.tobytes() and a.items.shape == b.items.shape
        assert c.items.shape[0] == a.items.shape[0]
        assert c.items.shape[1] > a.items.shape[1]
        for row_a, row_c in zip(a.items, c.items, strict=True):
            tokens_a = [int(t) for t in row_a if t != PAD]
            tokens_c = [int(t) for t in row_c if t != PAD]
            assert tokens_c[: len(tokens_a)] == tokens_a
            assert tokens_c[len(tokens_a)] == SEP
            assert all(t >= FLAG_BASE for t in tokens_c[len(tokens_a) + 1 :])


# ---------------------------------------------------------------------------
# Training, checkpoints, search
# ---------------------------------------------------------------------------


def train_tiny(encoding, **overrides):
    rows = witness_rows([task_for("sum"), task_for("sum", 5)], encoding)
    knobs = {
        "epochs": 2,
        "learning_rate": 1e-2,
        "batch_size": 4,
        "validation_batch_size": 8,
        "seed": 0,
        "device": "cpu",
        "encoding": encoding,
    }
    return train_policy(rows, rows, TINY, **{**knobs, **overrides})


def test_training_records_the_layout_and_round_trips_through_a_checkpoint(tmp_path):
    for encoding in ENCODINGS:
        model, state = train_tiny(encoding)
        assert model.encoding == encoding
        assert state["encoding"] == encoding
        assert model.embedding.num_embeddings == EMBEDDING_ROWS[encoding]
        path = save_policy(
            tmp_path / encoding,
            model,
            trainer_state=state,
            manifest={"arm": "tiny"},
            wandb_url=None,
        )
        config = json.loads((path / CONFIG_FILE).read_text())
        assert config["encoding"] == encoding
        assert config["vocabulary"]["size"] == EMBEDDING_ROWS[encoding]
        loaded = load_policy(path)
        assert loaded.encoding == encoding
        assert loaded.embedding.num_embeddings == EMBEDDING_ROWS[encoding]
        task = task_for("sum_positive")
        product, outputs = stacks(*task.inputs), stacks(*task.outputs)
        before = PolicyScorer(model).log_probs(
            [PolicyScorer(model).encode(product, outputs)],
            np.ones((1, len(OP_NAMES)), bool),
        )
        after = PolicyScorer(loaded).log_probs(
            [PolicyScorer(loaded).encode(product, outputs)],
            np.ones((1, len(OP_NAMES)), bool),
        )
        assert np.allclose(before, after)


def test_a_config_without_the_field_loads_as_machine(tmp_path):
    model, state = train_tiny(MACHINE)
    path = save_policy(
        tmp_path / "old", model, trainer_state=state, manifest={}, wandb_url=None
    )
    config = json.loads((path / CONFIG_FILE).read_text())
    del config["encoding"]
    (path / CONFIG_FILE).write_text(json.dumps(config))
    loaded = load_policy(path)
    assert loaded.encoding == MACHINE
    assert PolicyScorer(loaded).encoding == MACHINE
    assert loaded.embedding.num_embeddings == VOCAB


def test_policy_config_records_the_layout_and_is_otherwise_unchanged():
    machine = policy_config(TINY)
    assert machine == policy_config(TINY, OP_NAMES, MACHINE)
    assert machine["encoding"] == MACHINE
    assert machine["vocabulary"]["size"] == VOCAB
    difference = policy_config(TINY, encoding=DIFFERENCE)
    assert difference["encoding"] == DIFFERENCE
    assert difference["vocabulary"]["size"] == EMBEDDING_ROWS[DIFFERENCE]
    stripped = {k: v for k, v in machine.items() if k not in ("encoding", "vocabulary")}
    assert stripped == {
        k: v for k, v in difference.items() if k not in ("encoding", "vocabulary")
    }
    assert {**machine["vocabulary"], "size": None} == {
        **difference["vocabulary"],
        "size": None,
    }


def test_a_difference_model_loads_from_its_checkpoint_and_searches(tmp_path):
    model, state = train_tiny(DIFFERENCE)
    path = save_policy(
        tmp_path / "d", model, trainer_state=state, manifest={}, wandb_url=None
    )
    scorer = PolicyScorer(load_policy(path), "cpu")
    assert scorer.encoding == DIFFERENCE
    task = task_for("add_pair", 6)
    flat = beam_solve(task, scorer, beam_width=64, max_depth=4, guard="both")
    assert flat.termination == "fit" and flat.program == ("add",)
    segment = segment_beam_solve(task, scorer, beam_width=4, roots=2, segment_depth=2)
    assert segment.termination == "fit" and segment.program == ("add",)


def test_the_flat_beam_feeds_the_scorer_its_own_layout():
    task = task_for("sum_positive")
    for encoding in ENCODINGS:
        scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding), "cpu")
        seen = []
        original = scorer.log_probs

        def spy(blocks, legal, _original=original, _seen=seen):
            _seen.extend(blocks)
            return _original(blocks, legal)

        scorer.log_probs = spy
        beam_solve(task, scorer, beam_width=2, max_depth=2, guard="both")
        assert seen
        outputs = stacks(*task.outputs)
        inputs = stacks(*task.inputs)
        assert np.array_equal(seen[0], node_encoder(outputs, encoding)(inputs))
        seps = 2 if encoding == DIFFERENCE else 1
        assert all(int((row == SEP).sum()) == seps for block in seen for row in block)
