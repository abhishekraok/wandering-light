"""The stack policy's `difference` item layout, fixed per model (#inverse-dynamics)."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.inverse_dynamics as idm
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve, block_encoder, step_all_quiet
from wandering_light.stack_machine import Bool, Int, Lst, parse_stack
from wandering_light.stack_pbe import (
    BASIS,
    BY_NAME,
    CATALOGUE,
    OP_BY_NAME,
    OPS,
    build_task,
    draw_task,
)
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
    REMOVED,
    SEP,
    TARGET_CAP,
    VOCAB,
    PolicyScorer,
    StackPairPolicy,
    encode_item,
    encode_node,
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
UNKNOWN = r"unknown encoding 'bogus'; choose one of \('machine', 'difference'\)"


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def witness_pairs(limit: int = 40):
    """``(product, outputs)`` at every witness prefix of the first tasks that draw."""
    pairs = []
    for behaviour in CATALOGUE:
        try:
            task = build_task(behaviour, draw_task(behaviour, random.Random(0), 3, 4))
        except Exception:  # a sampler that cannot draw three distinct examples
            continue
        outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
        product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
        for name in task.program:
            pairs.append((product, outputs))
            product = step_all_quiet(product, OP_BY_NAME[name])[0]
        pairs.append((product, outputs))
        if len(pairs) >= limit * 10:
            break
    return pairs


def machine_block(product, outputs):
    """The block every search built before layouts existed."""
    targets = encode_targets(outputs)
    return item_array(
        [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
    )


def test_constants():
    assert (MACHINE, DIFFERENCE) == ("machine", "difference")
    assert ENCODINGS == (MACHINE, DIFFERENCE)
    assert FLAG_TOP == 8
    assert (EQUAL, CHANGED, ADDED, REMOVED) == (0, 1, 2, 3)
    assert FLAG_BASE == VOCAB
    assert EMBEDDING_ROWS == {MACHINE: VOCAB, DIFFERENCE: VOCAB + 4 * FLAG_TOP}
    # The probe's constants are the same numbers.
    assert (idm.TOP, idm.FLAG_BASE) == (FLAG_TOP, FLAG_BASE)
    assert (idm.EQUAL, idm.CHANGED, idm.ADDED, idm.REMOVED) == (0, 1, 2, 3)


def test_flags_by_position_from_the_top():
    state = (Int(1), Int(2), Int(3))
    target = (Int(9), Int(2), Int(3), Int(4))
    # top 0: 3 vs 4 changed; 1: 2 vs 3 changed; 2: 1 vs 2 changed; 3: only target.
    assert flags(state, target) == [
        FLAG_BASE + 4 * 0 + CHANGED,
        FLAG_BASE + 4 * 1 + CHANGED,
        FLAG_BASE + 4 * 2 + CHANGED,
        FLAG_BASE + 4 * 3 + ADDED,
    ]
    assert flags((Int(5), Int(7)), (Int(7),)) == [
        FLAG_BASE + 0 + EQUAL,
        FLAG_BASE + 4 + REMOVED,
    ]
    assert flags((), ()) == []
    assert flags((), (Bool(True),)) == [FLAG_BASE + ADDED]
    # Only the top FLAG_TOP positions; both stacks deeper still gives FLAG_TOP.
    deep = tuple(Int(i) for i in range(FLAG_TOP + 5))
    assert flags(deep, deep) == [FLAG_BASE + 4 * k + EQUAL for k in range(FLAG_TOP)]
    assert all(FLAG_BASE <= t < EMBEDDING_ROWS[DIFFERENCE] for t in flags(deep, ()))


def test_flags_compare_the_full_stacks_not_their_truncated_tokens():
    # Two lists that differ only at their bottom element, past TARGET_CAP tokens:
    # their capped tokens agree, their values do not.
    long = list(range(TARGET_CAP + 10))
    a = (Lst(tuple(Int(v) for v in long)),)
    b = (Lst(tuple(Int(v) for v in [99, *long[1:]])),)
    assert encode_targets([a]) == encode_targets([b])
    assert flags(a, b) == [FLAG_BASE + CHANGED]
    assert flags(a, a) == [FLAG_BASE + EQUAL]


def test_flags_match_the_inverse_dynamics_probe():
    checked = 0
    for product, outputs in witness_pairs():
        for state, target in zip(product, outputs, strict=True):
            assert flags(state, target) == idm.flags(state, target)
            assert flags(target, state) == idm.flags(target, state)
            checked += 1
        triple = {"s": product, "t": outputs}
        # The whole block is the probe's `difference` item with the outputs as s'.
        np.testing.assert_array_equal(
            node_encoder(outputs, DIFFERENCE)(product),
            idm.sequence_items(triple, "difference"),
        )
        np.testing.assert_array_equal(
            node_encoder(outputs, MACHINE)(product),
            idm.sequence_items(triple, "current"),
        )
    assert checked > 100


def test_machine_blocks_are_byte_identical_to_today():
    for product, outputs in witness_pairs():
        block = node_encoder(outputs)(product)
        expected = machine_block(product, outputs)
        assert block.dtype == expected.dtype == np.int16
        assert block.shape == expected.shape
        assert block.tobytes() == expected.tobytes()
        assert encode_node(product, outputs).tobytes() == expected.tobytes()
        assert (block < VOCAB).all()


def test_difference_items_append_sep_and_flags():
    for product, outputs in witness_pairs(10):
        block = node_encoder(outputs, DIFFERENCE)(product)
        assert block.dtype == np.int16 and block.shape[0] == len(product)
        targets = encode_targets(outputs)
        items = [
            [*encode_item(s, t), SEP, *flags(s, stack)]
            for s, t, stack in zip(product, targets, outputs, strict=True)
        ]
        assert block.tobytes() == item_array(items).tobytes()
        assert encode_node(product, outputs, DIFFERENCE).tobytes() == block.tobytes()
        # Every example has at least one stack non-empty, so every item flags.
        assert ((block >= FLAG_BASE).sum(axis=1) >= 1).all()


def test_unknown_encodings_are_refused():
    outputs = ((Int(1),),)
    with pytest.raises(ValueError, match=UNKNOWN):
        node_encoder(outputs, "bogus")
    with pytest.raises(ValueError, match=UNKNOWN):
        encode_node(outputs, outputs, "bogus")
    with pytest.raises(ValueError, match=UNKNOWN):
        StackPairPolicy(TINY, encoding="bogus")
    with pytest.raises(ValueError, match=UNKNOWN):
        policy_config(TINY, encoding="bogus")
    with pytest.raises(ValueError, match=UNKNOWN):
        witness_rows([task_for("sum")], "bogus")


def test_the_difference_model_adds_flag_rows_last_and_nothing_else():
    torch.manual_seed(0)
    machine = StackPairPolicy(TINY)
    torch.manual_seed(0)
    difference = StackPairPolicy(TINY, encoding=DIFFERENCE)
    assert (machine.encoding, difference.encoding) == (MACHINE, DIFFERENCE)
    assert machine.embedding.num_embeddings == VOCAB
    assert difference.embedding.num_embeddings == EMBEDDING_ROWS[DIFFERENCE]
    assert (
        difference.parameter_count - machine.parameter_count
        == 4 * FLAG_TOP * TINY.byte_embedding
    )
    shapes = {k: v.shape for k, v in machine.state_dict().items()}
    for key, value in difference.state_dict().items():
        if key != "embedding.weight":
            assert value.shape == shapes[key], key


def test_scorer_encodes_in_its_models_layout():
    task = task_for("sum")
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    for encoding in ENCODINGS:
        scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding), "cpu")
        assert scorer.encoding == encoding
        expected = encode_node(inputs, outputs, encoding)
        assert scorer.encoder(outputs)(inputs).tobytes() == expected.tobytes()
        assert scorer.encode(inputs, outputs).tobytes() == expected.tobytes()
        assert block_encoder(scorer, outputs)(inputs).tobytes() == expected.tobytes()
        legal = np.ones((1, len(OPS)), dtype=bool)
        rows = scorer.log_probs([scorer.encode(inputs, outputs)], legal)
        assert rows.shape == (1, len(OPS))
        assert np.isclose(np.exp(rows).sum(), 1.0)

    class Legacy(torch.nn.Module):
        """A model from before layouts: no ``encoding`` attribute."""

    assert PolicyScorer(Legacy(), "cpu").encoding == MACHINE


def test_a_machine_sized_model_cannot_read_difference_blocks():
    """The point of the per-model layout: the flags only fit their own embedding."""
    task = task_for("sum")
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    with pytest.raises(IndexError):
        scorer.logits([encode_node(inputs, outputs, DIFFERENCE)])


def test_a_duck_typed_scorer_reads_machine_blocks():
    class Flat:
        def log_probs(self, blocks, legal):
            return np.where(legal, 0.0, -np.inf)

    for product, outputs in witness_pairs(3):
        block = block_encoder(Flat(), outputs)(product)
        assert block.tobytes() == machine_block(product, outputs).tobytes()


def test_witness_rows_in_each_layout():
    tasks = [task_for("sum"), task_for("sum", 5)]
    machine = witness_rows(tasks)
    for a, b in zip(machine, witness_rows(tasks, MACHINE), strict=True):
        assert a.items.tobytes() == b.items.tobytes() and a.label == b.label
    difference = witness_rows(tasks, DIFFERENCE)
    assert len(machine) == len(difference)
    for index, task in enumerate(tasks):
        outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
        product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
        rows = [
            (m, d) for m, d in zip(machine, difference, strict=True) if m.task == index
        ]
        for (m, d), name in zip(rows, task.program, strict=True):
            assert (m.label, m.task, m.step, m.remaining) == (
                d.label,
                d.task,
                d.step,
                d.remaining,
            )
            assert m.items.tobytes() == machine_block(product, outputs).tobytes()
            assert (
                d.items.tobytes() == encode_node(product, outputs, DIFFERENCE).tobytes()
            )
            product = step_all_quiet(product, OP_BY_NAME[name])[0]


def _train(rows, encoding):
    return train_policy(
        rows,
        rows,
        TINY,
        epochs=2,
        learning_rate=1e-2,
        batch_size=4,
        validation_batch_size=8,
        seed=0,
        device="cpu",
        encoding=encoding,
    )


def test_train_save_and_load_keep_the_layout(tmp_path):
    tasks = [task_for("sum"), task_for("sum", 5)]
    model, state = _train(witness_rows(tasks, DIFFERENCE), DIFFERENCE)
    assert model.encoding == DIFFERENCE and state["encoding"] == DIFFERENCE
    path = save_policy(
        tmp_path / "difference",
        model,
        trainer_state=state,
        manifest={"arm": "tiny"},
        wandb_url=None,
    )
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["encoding"] == DIFFERENCE
    assert config["vocabulary"]["size"] == EMBEDDING_ROWS[DIFFERENCE]
    assert json.loads((path / "trainer_state.json").read_text())["encoding"] == (
        DIFFERENCE
    )
    loaded = load_policy(path)
    assert loaded.encoding == DIFFERENCE
    task = tasks[0]
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    block = encode_node(inputs, outputs, DIFFERENCE)
    np.testing.assert_allclose(
        PolicyScorer(loaded).logits([block]), PolicyScorer(model).logits([block])
    )


def test_machine_config_and_a_config_without_the_field(tmp_path):
    config = policy_config(TINY)
    assert config["encoding"] == MACHINE
    assert config["vocabulary"]["size"] == VOCAB
    model, state = _train(witness_rows([task_for("sum")]), MACHINE)
    assert state["encoding"] == MACHINE
    path = save_policy(
        tmp_path / "machine",
        model,
        trainer_state=state,
        manifest={"arm": "tiny"},
        wandb_url=None,
    )
    # A checkpoint published before the field existed.
    written = json.loads((path / CONFIG_FILE).read_text())
    del written["encoding"]
    (path / CONFIG_FILE).write_text(json.dumps(written))
    loaded = load_policy(path)
    assert loaded.encoding == MACHINE
    assert loaded.embedding.num_embeddings == VOCAB
    assert PolicyScorer(loaded).encoding == MACHINE


def test_machine_training_is_unchanged_by_the_layout_argument():
    rows = witness_rows([task_for("sum"), task_for("sum", 5)])
    default, _ = train_policy(
        rows,
        rows,
        TINY,
        epochs=2,
        learning_rate=1e-2,
        batch_size=4,
        validation_batch_size=8,
        seed=0,
        device="cpu",
    )
    explicit, _ = _train(rows, MACHINE)
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key]), key


def test_a_trained_difference_policy_guides_the_flat_beam():
    """Fit to its own prefixes, as the machine test is; the beam must read flags."""
    task = task_for("sum")
    rows = witness_rows([task, task_for("sum", 5)], DIFFERENCE)
    model, _ = train_policy(
        rows,
        rows,
        TINY,
        epochs=40,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=8,
        seed=0,
        device="cpu",
        encoding=DIFFERENCE,
    )
    result = beam_solve(
        task, PolicyScorer(model, "cpu"), beam_width=4, max_depth=task.length
    )
    assert result.termination == "fit", result
