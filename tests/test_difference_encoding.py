"""The `difference` item layout of the stack policy, beside today's `machine`."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve, block_encoder
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, parse_stack
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
    PAD,
    REMOVED,
    SEP,
    TARGET_CAP,
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


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def outputs_of(task):
    return tuple(parse_stack(text, basis=BASIS) for text in task.outputs)


def random_stack(rng: random.Random, depth: int) -> tuple:
    pool = [
        lambda: Int(rng.randint(-3, 3)),
        lambda: Int(rng.choice([100, -1234])),
        lambda: Bool(rng.random() < 0.5),
        lambda: Sym(rng.choice(["add", "dup", "swap"])),
        Mark,
        lambda: Lst(tuple(Int(rng.randint(0, 2)) for _ in range(rng.randint(0, 3)))),
    ]
    return tuple(rng.choice(pool)() for _ in range(depth))


def old_machine_block(product, outputs) -> np.ndarray:
    """The block every search built before layouts existed."""
    targets = encode_targets(outputs)
    return item_array(
        [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
    )


def test_constants():
    assert ENCODINGS == (MACHINE, DIFFERENCE) == ("machine", "difference")
    assert (EQUAL, CHANGED, ADDED, REMOVED) == (0, 1, 2, 3)
    assert FLAG_BASE == VOCAB and FLAG_TOP == 8
    assert EMBEDDING_ROWS == {MACHINE: VOCAB, DIFFERENCE: VOCAB + 32}


def test_flags_compare_from_the_top():
    a, b, c = Int(1), Int(2), Int(3)
    # top: kept; below it: changed; then the state has an extra item.
    assert flags((c, b, a), (b, a)) == [
        FLAG_BASE + 0 * 4 + EQUAL,
        FLAG_BASE + 1 * 4 + EQUAL,
        FLAG_BASE + 2 * 4 + REMOVED,
    ]
    assert flags((a,), (b, c)) == [FLAG_BASE + CHANGED, FLAG_BASE + 4 + ADDED]
    assert flags((), ()) == []
    deep = tuple(Int(i) for i in range(20))
    assert flags(deep, deep) == [FLAG_BASE + 4 * k + EQUAL for k in range(FLAG_TOP)]


def test_flags_compare_full_stacks_not_truncated_tokens():
    # Two lists whose target tokens agree after truncation still differ.
    long = Lst(tuple(Int(i % 5) for i in range(TARGET_CAP + 10)))
    other = Lst((Int(4), *long.items[1:]))
    assert encode_stack((long,), TARGET_CAP) == encode_stack((other,), TARGET_CAP)
    assert flags((long,), (other,)) == [FLAG_BASE + CHANGED]


def test_flags_match_the_inverse_dynamics_probe():
    inverse = pytest.importorskip("experiments.inverse_dynamics")
    assert inverse.FLAG_BASE == FLAG_BASE and inverse.TOP == FLAG_TOP
    rng = random.Random(0)
    for _ in range(300):
        s = random_stack(rng, rng.randint(0, 12))
        t = random_stack(rng, rng.randint(0, 12))
        if rng.random() < 0.5:  # share a bottom so EQUAL appears often
            t = s[: rng.randint(0, len(s))] + t[: rng.randint(0, 2)]
        assert flags(s, t) == inverse.flags(s, t)


def test_machine_blocks_are_byte_identical_to_before():
    rng = random.Random(1)
    for name in ("sum", "sum_positive"):
        task = task_for(name)
        outputs = outputs_of(task)
        encode = node_encoder(outputs)
        for _step, states, _op in replay_witness(task):
            want = old_machine_block(states, outputs)
            for got in (encode(states), encode_node(states, outputs, MACHINE)):
                assert got.dtype == np.int16
                assert got.shape == want.shape and got.tobytes() == want.tobytes()
        product = tuple(random_stack(rng, 30) for _ in outputs)
        assert (
            encode(product).tobytes() == old_machine_block(product, outputs).tobytes()
        )


def test_difference_items_append_sep_and_flags():
    task = task_for("sum_positive")
    outputs = outputs_of(task)
    targets = encode_targets(outputs)
    encode = node_encoder(outputs, DIFFERENCE)
    for _step, states, _op in replay_witness(task):
        items = [
            [*encode_item(s, t), SEP, *flags(s, o)]
            for s, t, o in zip(states, targets, outputs, strict=True)
        ]
        block = encode(states)
        assert block.dtype == np.int16
        assert block.tobytes() == item_array(items).tobytes()
        assert block.max() < EMBEDDING_ROWS[DIFFERENCE]
        assert encode_node(states, outputs, DIFFERENCE).tobytes() == block.tobytes()


def test_unknown_encoding_is_refused_everywhere():
    message = "unknown encoding 'bytes'; choose one of"
    task = task_for("sum")
    with pytest.raises(ValueError, match=message):
        node_encoder(outputs_of(task), "bytes")
    with pytest.raises(ValueError, match=message):
        StackPairPolicy(TINY, encoding="bytes")
    with pytest.raises(ValueError, match=message):
        witness_rows([task], "bytes")
    with pytest.raises(ValueError, match=message):
        policy_config(TINY, encoding="bytes")


def test_model_embedding_grows_only_by_the_flag_rows():
    machine = StackPairPolicy(TINY)
    difference = StackPairPolicy(TINY, encoding=DIFFERENCE)
    assert machine.encoding == MACHINE and difference.encoding == DIFFERENCE
    assert machine.embedding.num_embeddings == VOCAB
    assert difference.embedding.num_embeddings == VOCAB + 4 * FLAG_TOP
    shapes = {k: v.shape for k, v in machine.state_dict().items()}
    other = {k: v.shape for k, v in difference.state_dict().items()}
    assert shapes.keys() == other.keys()
    assert {k for k in shapes if shapes[k] != other[k]} == {"embedding.weight"}


def test_policy_config_records_the_encoding():
    machine = policy_config(TINY)
    assert machine["encoding"] == MACHINE and machine["vocabulary"]["size"] == VOCAB
    difference = policy_config(TINY, encoding=DIFFERENCE)
    assert difference["encoding"] == DIFFERENCE
    assert difference["vocabulary"]["size"] == EMBEDDING_ROWS[DIFFERENCE]
    rest = {k: v for k, v in machine.items() if k not in ("encoding", "vocabulary")}
    assert rest == {
        k: v for k, v in difference.items() if k not in ("encoding", "vocabulary")
    }


def test_scorer_builds_blocks_in_the_models_layout():
    task = task_for("sum_positive")
    outputs = outputs_of(task)
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    for encoding in ENCODINGS:
        scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding))
        assert scorer.encoding == encoding
        want = node_encoder(outputs, encoding)(inputs)
        assert scorer.encode(inputs, outputs).tobytes() == want.tobytes()
        assert scorer.encoder(outputs)(inputs).tobytes() == want.tobytes()
        assert block_encoder(scorer, outputs)(inputs).tobytes() == want.tobytes()
        legal = np.ones((1, 28), dtype=bool)
        rows = scorer.log_probs([want], legal)
        assert rows.shape == (1, 28) and np.isfinite(rows).all()


def test_scorer_of_a_model_without_an_encoding_reads_machine():
    model = StackPairPolicy(TINY)
    del model.encoding
    assert PolicyScorer(model).encoding == MACHINE


def test_witness_rows_use_the_requested_layout():
    tasks = [task_for("sum"), task_for("sum_positive")]
    machine = witness_rows(tasks)
    difference = witness_rows(tasks, DIFFERENCE)
    assert [(r.label, r.task, r.step) for r in machine] == [
        (r.label, r.task, r.step) for r in difference
    ]
    for m, d in zip(machine, difference, strict=True):
        assert m.items.dtype == d.items.dtype == np.int16
        assert m.items.max() < VOCAB <= d.items.max()
        # The difference item begins with the machine item, unchanged.
        for k, length in enumerate((m.items != PAD).sum(axis=1)):
            assert (d.items[k, :length] == m.items[k, :length]).all()
            assert d.items[k, length] == SEP


def train_tiny(encoding, tasks):
    rows = witness_rows(tasks, encoding)
    return train_policy(
        rows,
        rows,
        TINY,
        epochs=40,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=8,
        seed=0,
        device="cpu",
        encoding=encoding,
    )


def test_difference_policy_trains_saves_loads_and_searches(tmp_path):
    task = task_for("sum")
    model, state = train_tiny(DIFFERENCE, [task, task_for("sum", 5)])
    assert model.encoding == DIFFERENCE and state["encoding"] == DIFFERENCE
    checkpoint = save_policy(
        tmp_path / "difference",
        model,
        trainer_state=state,
        manifest={"test": True},
        wandb_url=None,
    )
    config = json.loads((checkpoint / CONFIG_FILE).read_text())
    assert config["encoding"] == DIFFERENCE
    assert config["vocabulary"]["size"] == EMBEDDING_ROWS[DIFFERENCE]
    loaded = load_policy(checkpoint)
    assert loaded.encoding == DIFFERENCE
    for key, value in model.state_dict().items():
        assert (loaded.state_dict()[key] == value).all()
    result = beam_solve(
        task, PolicyScorer(loaded, "cpu"), beam_width=4, max_depth=task.length
    )
    assert result.termination == "fit", result


def test_a_config_without_an_encoding_loads_as_machine(tmp_path):
    model = StackPairPolicy(TINY)
    checkpoint = save_policy(
        tmp_path / "machine",
        model,
        trainer_state={},
        manifest={},
        wandb_url=None,
    )
    config = json.loads((checkpoint / CONFIG_FILE).read_text())
    assert config["encoding"] == MACHINE and config["vocabulary"]["size"] == VOCAB
    del config["encoding"]
    (checkpoint / CONFIG_FILE).write_text(json.dumps(config))
    loaded = load_policy(checkpoint)
    assert loaded.encoding == MACHINE
    assert loaded.embedding.num_embeddings == VOCAB
