"""Stack-policy item layouts and their checkpoint compatibility."""

import json
import random

import numpy as np
import pytest

from experiments import inverse_dynamics
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_machine import Bool, Int, Lst, parse_stack
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
    replay_witness,
    save_policy,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)


@pytest.mark.parametrize(
    ("state", "target", "statuses"),
    [
        ((), (), []),
        ((Int(1),), (Int(2), Int(1)), [EQUAL, ADDED]),
        ((Int(2), Int(1)), (Bool(True),), [CHANGED, REMOVED]),
        (
            (Int(1),) + (Lst((Int(3),)),) * FLAG_TOP,
            (Int(2),) + (Lst((Int(3),)),) * FLAG_TOP,
            [EQUAL] * FLAG_TOP,
        ),
    ],
)
def test_flags_match_the_inverse_dynamics_probe(state, target, statuses):
    expected = [FLAG_BASE + 4 * k + status for k, status in enumerate(statuses)]
    assert flags(state, target) == expected == inverse_dynamics.flags(state, target)


def test_difference_appends_flags_to_the_unchanged_machine_items():
    tail = (Int(0),) * STATE_CAP
    long_value = Lst((Int(1), *tail))
    changed_value = Lst((Int(2), *tail))
    assert encode_stack((long_value,), STATE_CAP) == encode_stack(
        (changed_value,), STATE_CAP
    )
    product = ((long_value,), ())
    targets = ((changed_value,), ())
    items = [
        encode_item(s, t) for s, t in zip(product, encode_targets(targets), strict=True)
    ]
    machine = encode_node(product, targets)
    expected = item_array(items)
    assert machine.shape == expected.shape
    assert machine.dtype == expected.dtype == np.int16
    assert machine.tobytes() == expected.tobytes()

    difference = node_encoder(targets, DIFFERENCE)(product)
    assert difference.dtype == np.int16
    np.testing.assert_array_equal(
        difference, item_array([items[0] + [SEP, FLAG_BASE + CHANGED], [SEP, SEP]])
    )
    np.testing.assert_array_equal(
        difference,
        inverse_dynamics.sequence_items({"s": product, "t": targets}, DIFFERENCE),
    )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_checkpoint_round_trips_the_encoding_and_logits(tmp_path, encoding):
    model = StackPairPolicy(TINY, encoding=encoding)
    rows = VOCAB if encoding == MACHINE else VOCAB + 4 * FLAG_TOP
    assert model.embedding.num_embeddings == EMBEDDING_ROWS[encoding] == rows
    scorer = PolicyScorer(model)
    product, targets = ((Int(1),),), ((Int(2),),)
    block = scorer.encode(product, targets)
    np.testing.assert_array_equal(block, encode_node(product, targets, encoding))
    logits = scorer.logits([block])

    checkpoint = save_policy(
        tmp_path / encoding, model, trainer_state={}, manifest={}, wandb_url=None
    )
    config_path = checkpoint / CONFIG_FILE
    config = json.loads(config_path.read_text())
    assert config["encoding"] == encoding
    assert config["vocabulary"]["size"] == rows

    def check_loaded():
        loaded = PolicyScorer(load_policy(checkpoint))
        assert loaded.encoding == encoding
        assert loaded.model.embedding.num_embeddings == rows
        loaded_block = loaded.encoder(targets)(product)
        np.testing.assert_array_equal(loaded_block, block)
        np.testing.assert_array_equal(loaded.logits([loaded_block]), logits)

    check_loaded()
    if encoding == MACHINE:
        del config["encoding"]
        config_path.write_text(json.dumps(config))
        check_loaded()


def test_scorer_defaults_to_machine_when_model_has_no_encoding():
    model = StackPairPolicy(TINY)
    del model.encoding
    scorer = PolicyScorer(model)
    targets = ((Int(2),),)
    assert scorer.encoding == MACHINE
    np.testing.assert_array_equal(
        scorer.encode(((Int(1),),), targets), encode_node(((Int(1),),), targets)
    )


def test_difference_training_rows_and_scorer_use_the_shown_outputs():
    behaviour = BY_NAME["sum"]
    task = build_task(behaviour, draw_task(behaviour, random.Random(4), 3, held_out=2))
    targets = tuple(parse_stack(output, basis=BASIS) for output in task.outputs)
    rows = witness_rows([task], DIFFERENCE)
    model, state = train_policy(
        rows,
        rows,
        TINY,
        epochs=1,
        learning_rate=0.001,
        batch_size=len(rows),
        validation_batch_size=len(rows),
        seed=0,
        device="cpu",
        encoding=DIFFERENCE,
    )
    scorer = PolicyScorer(model)
    assert state["encoding"] == scorer.encoding == DIFFERENCE
    for row, (step, product, name) in zip(rows, replay_witness(task), strict=True):
        assert (row.step, OP_NAMES[row.label]) == (step, name)
        expected = inverse_dynamics.sequence_items(
            {"s": product, "t": targets}, DIFFERENCE
        )
        np.testing.assert_array_equal(row.items, expected)
        np.testing.assert_array_equal(row.items, scorer.encode(product, targets))

    legal = np.ones((len(rows), len(OP_NAMES)), dtype=bool)
    log_probs = scorer.log_probs([row.items for row in rows], legal)
    assert scorer.last_features is not None
    np.testing.assert_allclose(np.exp(log_probs).sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("factory", "args"),
    [(node_encoder, ((),)), (StackPairPolicy, (TINY,)), (policy_config, (TINY,))],
)
def test_unknown_encodings_are_rejected(factory, args):
    with pytest.raises(ValueError) as error:
        factory(*args, encoding="unknown")
    assert str(error.value) == f"unknown encoding 'unknown'; choose one of {ENCODINGS}"
