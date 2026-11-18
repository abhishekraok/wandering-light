"""Stack policy item layouts across witness replay, scoring, and checkpoints."""

import json
import random

import numpy as np
import pytest

from experiments import inverse_dynamics as probe
from wandering_light import stack_policy
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_machine import Int, Lst, parse_stack
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
    TARGET_CAP,
    TRAINER_STATE_FILE,
    VOCAB,
    PolicyScorer,
    StackPairPolicy,
    encode_node,
    flags,
    load_policy,
    node_encoder,
    replay_witness,
    save_policy,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(4, 8, 12)


def test_flags_compare_from_the_top_and_stop_at_eight_values():
    state = (Int(1), Int(2), Int(3))
    target = (Int(4), Int(3))
    assert flags(state, target) == [
        FLAG_BASE + EQUAL,
        FLAG_BASE + 4 + CHANGED,
        FLAG_BASE + 8 + REMOVED,
    ]
    assert flags(target, state)[2] == FLAG_BASE + 8 + ADDED
    assert flags((), ()) == []
    deep = (Int(0),) * (FLAG_TOP + 1)
    assert flags((), deep) == [FLAG_BASE + 4 * k + ADDED for k in range(FLAG_TOP)]


def test_flags_see_a_difference_outside_both_token_windows():
    suffix = (Int(0),) * STATE_CAP
    state = ((Lst((Int(1), *suffix)),),)
    target = ((Lst((Int(2), *suffix)),),)
    machine = encode_node(target, target)
    np.testing.assert_array_equal(encode_node(state, target), machine)
    difference = encode_node(state, target, DIFFERENCE)
    np.testing.assert_array_equal(difference[:, :-2], machine)
    assert difference[0, -2:].tolist() == [SEP, FLAG_BASE + CHANGED]


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_node_encoder_tokenizes_each_target_only_once(mocker, encoding):
    outputs = ((Int(1),), (Int(2),))
    tokenize = mocker.spy(stack_policy, "encode_stack")
    encode = node_encoder(outputs, encoding)
    encode(((), ()))
    encode(outputs)
    assert [
        stack for (stack, cap), _ in tokenize.call_args_list if cap == TARGET_CAP
    ] == list(outputs)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_witness_and_scorer_blocks_match_probe_items_for_shown_outputs(encoding):
    behaviour = BY_NAME["sum"]
    task = build_task(behaviour, draw_task(behaviour, random.Random(4), 4, held_out=6))
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding))
    encode = scorer.encoder(outputs)
    rows = witness_rows([task], encoding)
    for row, (_, product, _) in zip(rows, replay_witness(task), strict=True):
        expected = probe.sequence_items({"s": product, "t": outputs}, encoding)
        for block in (row.items, encode(product), scorer.encode(product, outputs)):
            assert block.dtype == np.int16
            assert block.shape == expected.shape
            assert block.tobytes() == expected.tobytes()


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_trained_layout_survives_checkpoint_round_trip(tmp_path, encoding):
    behaviour = BY_NAME["add_pair"]
    task = build_task(behaviour, draw_task(behaviour, random.Random(6), 4, held_out=6))
    rows = witness_rows([task], encoding)
    model, state = train_policy(
        rows,
        rows,
        TINY,
        encoding=encoding,
        epochs=1,
        learning_rate=0.01,
        batch_size=1,
        validation_batch_size=1,
        seed=0,
        device="cpu",
    )
    assert model.embedding.num_embeddings == EMBEDDING_ROWS[encoding]
    path = save_policy(
        tmp_path / "policy", model, trainer_state=state, manifest={}, wandb_url=None
    )
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["encoding"] == encoding
    assert config["vocabulary"]["size"] == VOCAB + (
        4 * FLAG_TOP if encoding == DIFFERENCE else 0
    )
    assert json.loads((path / TRAINER_STATE_FILE).read_text())["encoding"] == encoding
    blocks = [rows[0].items]
    expected = PolicyScorer(model).logits(blocks)
    loaded = PolicyScorer(load_policy(path))
    assert loaded.encoding == encoding
    np.testing.assert_array_equal(loaded.logits(blocks), expected)
    if encoding == MACHINE:
        del config["encoding"]
        (path / CONFIG_FILE).write_text(json.dumps(config))
        legacy = PolicyScorer(load_policy(path))
        assert legacy.encoding == MACHINE
        np.testing.assert_array_equal(legacy.logits(blocks), expected)


def test_model_without_an_encoding_uses_machine_blocks():
    model = StackPairPolicy(TINY)
    del model.encoding
    scorer = PolicyScorer(model)
    assert scorer.encoding == MACHINE
    product = ((Int(1),), ())
    outputs = ((), (Int(2),))
    np.testing.assert_array_equal(
        scorer.encode(product, outputs), encode_node(product, outputs)
    )


def test_unknown_encoding_is_refused_by_encoder_and_model():
    for call in (
        lambda: node_encoder((), "other"),
        lambda: StackPairPolicy(TINY, OP_NAMES, "other"),
    ):
        with pytest.raises(ValueError) as error:
            call()
        assert (
            str(error.value) == f"unknown encoding 'other'; choose one of {ENCODINGS}"
        )
