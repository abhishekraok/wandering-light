"""The stack policy's two item layouts and where each is fixed (difference-encoding).

``machine`` is what every published checkpoint reads and must not move by a
byte; ``difference`` is the inverse-dynamics probe's (#166) item with the task's
shown outputs as the target. One function builds a node's block for the
searches, the training rows and `PolicyScorer`, and a checkpoint records which
layout its model reads.
"""

import json
import random

import numpy as np
import pytest
import torch

import experiments.inverse_dynamics as idp
from wandering_light import stack_policy
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import (
    BeamObserver,
    beam_solve,
    block_encoder,
    segment_beam_solve,
)
from wandering_light.stack_machine import Int, Lst, parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_pbe_eval import grade
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
    replay_witness,
    save_policy,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)
BEHAVIOURS = ("sum", "sum_positive", "negate_each", "add_pair")
TRAINING = {
    "learning_rate": 1e-2,
    "batch_size": 8,
    "validation_batch_size": 8,
    "seed": 0,
    "device": "cpu",
}


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def outputs_of(task):
    return tuple(parse_stack(text, basis=BASIS) for text in task.outputs)


def witness_nodes():
    """``(product, outputs)`` at every witness prefix of a few behaviours."""
    for name in BEHAVIOURS:
        for seed in (4, 5):
            task = task_for(name, seed)
            outputs = outputs_of(task)
            for _step, states, _name in replay_witness(task):
                yield states, outputs


def legacy_block(product, outputs):
    """A node's block as every search and `witness_rows` spelled it before."""
    targets = encode_targets(outputs)
    return item_array(
        [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
    )


def same_block(left, right) -> bool:
    return (left.shape, left.dtype, left.tobytes()) == (
        right.shape,
        right.dtype,
        right.tobytes(),
    )


def unknown(encoding="bogus"):
    return pytest.raises(
        ValueError,
        match=rf"^unknown encoding {encoding!r}; choose one of \('machine', 'difference'\)$",
    )


# ---------------------------------------------------------------------------
# Constants and flags
# ---------------------------------------------------------------------------


def test_the_layout_constants():
    assert (MACHINE, DIFFERENCE) == ("machine", "difference")
    assert ENCODINGS == (MACHINE, DIFFERENCE)
    assert FLAG_TOP == 8
    assert (EQUAL, CHANGED, ADDED, REMOVED) == (0, 1, 2, 3)
    assert FLAG_BASE == VOCAB
    assert EMBEDDING_ROWS == {MACHINE: VOCAB, DIFFERENCE: VOCAB + 4 * FLAG_TOP}
    # The probe this layout comes from must agree on every number in it.
    assert (idp.TOP, idp.FLAG_BASE) == (FLAG_TOP, FLAG_BASE)
    assert (idp.EQUAL, idp.CHANGED, idp.ADDED, idp.REMOVED) == (0, 1, 2, 3)


def test_flags_say_kept_changed_added_or_removed_from_the_top():
    def read(state, target):
        return [divmod(token - FLAG_BASE, 4) for token in flags(state, target)]

    one, two, three = Int(1), Int(2), Int(3)
    assert read((one, two, three), (one, three, two)) == [
        (0, CHANGED),
        (1, CHANGED),
        (2, EQUAL),
    ]
    # Aligned from the top, so a shorter target removes the *bottom* position.
    assert read((one, two, three), (two, three)) == [
        (0, EQUAL),
        (1, EQUAL),
        (2, REMOVED),
    ]
    assert read((three,), (one, two, three)) == [(0, EQUAL), (1, ADDED), (2, ADDED)]
    assert read((), (one,)) == [(0, ADDED)]
    assert read((one,), ()) == [(0, REMOVED)]
    assert flags((), ()) == []  # a position neither stack reaches has no token
    # Equality is by value, lists included.
    assert read((Lst((one, two)),), (Lst((one, two)),)) == [(0, EQUAL)]
    assert read((Lst((one, two)),), (Lst((two, one)),)) == [(0, CHANGED)]


def test_flags_stop_at_flag_top_and_stay_in_the_embedding():
    deep = tuple(Int(i) for i in range(FLAG_TOP + 5))
    moved = (Int(99), *deep[1:])  # differs only below the positions compared
    assert flags(deep, moved) == flags(deep, deep)
    assert [divmod(t - FLAG_BASE, 4) for t in flags(deep, deep)] == [
        (k, EQUAL) for k in range(FLAG_TOP)
    ]
    every = {
        token
        for state in ((), deep, deep[:3])
        for target in ((), deep, deep[1:], tuple(reversed(deep)))
        for token in flags(state, target)
    }
    assert every
    assert min(every) >= FLAG_BASE
    assert max(every) < EMBEDDING_ROWS[DIFFERENCE]
    # ... and the last flag token there can be fits an item block's int16.
    assert EMBEDDING_ROWS[DIFFERENCE] - 1 <= np.iinfo(np.int16).max


def test_flags_match_the_inverse_dynamics_probe():
    checked = 0
    for product, outputs in witness_nodes():
        for state, target in zip(product, outputs, strict=True):
            assert flags(state, target) == idp.flags(state, target)
            assert flags(target, state) == idp.flags(target, state)
            checked += 1
    deep = tuple(Int(i) for i in range(FLAG_TOP + 3))
    for state, target in (((), ()), ((), deep), (deep, ()), (deep, deep[2:])):
        assert flags(state, target) == idp.flags(state, target)
        checked += 1
    assert checked > 100


def test_flags_compare_the_stacks_not_their_truncated_tokens():
    """Two states whose tokens are equal after the cap, flagged differently."""
    tail = tuple(Int(i % 50) for i in range(STATE_CAP + 40))
    kept = (Lst((Int(1), *tail)), Int(7))
    other = (Lst((Int(2), *tail)), Int(7))
    assert kept != other
    assert encode_stack(kept, STATE_CAP) == encode_stack(other, STATE_CAP)
    target = (kept,)
    assert same_block(
        encode_node((kept,), target, MACHINE), encode_node((other,), target, MACHINE)
    )
    assert flags(kept, kept) == [FLAG_BASE + EQUAL, FLAG_BASE + 4 + EQUAL]
    assert flags(other, kept) == [FLAG_BASE + EQUAL, FLAG_BASE + 4 + CHANGED]
    same = encode_node((kept,), target, DIFFERENCE)
    differs = encode_node((other,), target, DIFFERENCE)
    assert same.shape == differs.shape
    assert (same != differs).sum() == 1
    assert same[0, -2:].tolist() == flags(kept, kept)
    assert differs[0, -2:].tolist() == flags(other, kept)


# ---------------------------------------------------------------------------
# One function builds a node's block
# ---------------------------------------------------------------------------


def test_machine_blocks_are_byte_identical_to_the_layout_before():
    checked = 0
    for product, outputs in witness_nodes():
        legacy = legacy_block(product, outputs)
        assert legacy.dtype == np.int16
        assert same_block(node_encoder(outputs)(product), legacy)
        assert same_block(node_encoder(outputs, MACHINE)(product), legacy)
        assert same_block(encode_node(product, outputs), legacy)
        assert same_block(encode_node(product, outputs, MACHINE), legacy)
        # ... which is the probe's `current` arm with the outputs as s'.
        triple = {"s": product, "t": outputs}
        assert same_block(idp.sequence_items(triple, "current"), legacy)
        assert legacy.max() < VOCAB
        checked += 1
    assert checked > 40


def test_difference_items_are_the_machine_item_then_sep_and_the_flags():
    checked = 0
    for product, outputs in witness_nodes():
        block = encode_node(product, outputs, DIFFERENCE)
        assert block.dtype == np.int16
        assert block.shape[0] == len(product)
        targets = encode_targets(outputs)
        for row, state, tokens, target in zip(
            block, product, targets, outputs, strict=True
        ):
            item = [*encode_item(state, tokens), SEP, *flags(state, target)]
            assert row[: len(item)].tolist() == item
            assert (row[len(item) :] == PAD).all()
        assert same_block(block, node_encoder(outputs, DIFFERENCE)(product))
        # The probe's `difference` arm, token for token.
        triple = {"s": product, "t": outputs}
        assert same_block(block, idp.sequence_items(triple, "difference"))
        checked += 1
    assert checked > 40


def test_an_empty_state_and_target_still_carry_both_separators():
    assert encode_node(((),), ((),), MACHINE).tolist() == [[SEP]]
    assert encode_node(((),), ((),), DIFFERENCE).tolist() == [[SEP, SEP]]


def test_node_encoder_encodes_the_targets_once(monkeypatch):
    task = task_for("sum_positive")
    outputs = outputs_of(task)
    calls = []
    real = stack_policy.encode_targets

    def counting(stacks):
        calls.append(len(stacks))
        return real(stacks)

    monkeypatch.setattr(stack_policy, "encode_targets", counting)
    for encoding in ENCODINGS:
        calls.clear()
        encode = node_encoder(outputs, encoding)
        assert calls == [task.examples]
        for _step, states, _name in replay_witness(task):
            encode(states)
        assert calls == [task.examples]


def test_a_block_has_one_item_per_example_or_none():
    task = task_for("sum")
    outputs = outputs_of(task)
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    for encoding in ENCODINGS:
        encode = node_encoder(outputs, encoding)
        assert encode(product).shape[0] == task.examples
        with pytest.raises(ValueError, match="zip"):
            encode(product[:-1])
        with pytest.raises(ValueError, match="zip"):
            encode((*product, product[0]))


def test_an_unknown_encoding_is_refused_everywhere_it_is_named():
    task = task_for("sum")
    outputs = outputs_of(task)
    rows = witness_rows([task])
    with unknown():
        node_encoder(outputs, "bogus")  # when made, not when first called
    with unknown():
        encode_node(outputs, outputs, "bogus")
    with unknown():
        StackPairPolicy(TINY, encoding="bogus")
    with unknown():
        witness_rows([task], "bogus")
    with unknown():
        witness_rows([], "bogus")
    with unknown():
        train_policy(rows, rows, TINY, epochs=1, encoding="bogus", **TRAINING)
    with unknown():
        policy_config(TINY, encoding="bogus")
    with unknown(None):
        StackPairPolicy(TINY, encoding=None)


# ---------------------------------------------------------------------------
# The model and its scorer
# ---------------------------------------------------------------------------


def test_the_model_stores_its_encoding_and_sizes_its_embedding_by_it():
    default = StackPairPolicy(TINY)
    assert default.encoding == MACHINE
    assert default.embedding.num_embeddings == VOCAB
    for encoding in ENCODINGS:
        model = StackPairPolicy(TINY, OP_NAMES, encoding)
        assert model.encoding == encoding
        assert model.embedding.num_embeddings == EMBEDDING_ROWS[encoding]
        assert model.embedding.embedding_dim == TINY.byte_embedding
        assert model.embedding.padding_idx == PAD
    extra = 4 * FLAG_TOP * TINY.byte_embedding
    difference = StackPairPolicy(TINY, encoding=DIFFERENCE)
    assert difference.parameter_count == default.parameter_count + extra


def test_only_the_embedding_differs_between_the_two_models():
    torch.manual_seed(0)
    before = StackPairPolicy(TINY).state_dict()
    torch.manual_seed(0)
    machine = StackPairPolicy(TINY, encoding=MACHINE).state_dict()
    torch.manual_seed(0)
    difference = StackPairPolicy(TINY, encoding=DIFFERENCE).state_dict()
    assert list(before) == list(machine) == list(difference)
    for name, tensor in before.items():
        assert torch.equal(tensor, machine[name])
        if name == "embedding.weight":
            assert difference[name].shape == (EMBEDDING_ROWS[DIFFERENCE], 8)
        else:
            assert torch.equal(tensor, difference[name])


def test_the_scorer_reads_its_models_encoding():
    task = task_for("sum_positive")
    outputs = outputs_of(task)
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    for encoding in ENCODINGS:
        scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding), "cpu")
        assert scorer.encoding == encoding
        block = encode_node(product, outputs, encoding)
        assert same_block(scorer.encode(product, outputs), block)
        assert same_block(scorer.encoder(outputs)(product), block)
        assert same_block(block_encoder(scorer, outputs)(product), block)
        legal = np.ones((1, len(OP_NAMES)), bool)
        rows = scorer.log_probs([scorer.encode(product, outputs)], legal)
        assert rows.shape == (1, len(OP_NAMES))
        assert np.isclose(np.exp(rows).sum(), 1.0)
    assert same_block(
        PolicyScorer(StackPairPolicy(TINY), "cpu").encode(product, outputs),
        legacy_block(product, outputs),
    )


def test_a_model_that_names_no_encoding_is_scored_as_machine():
    model = StackPairPolicy(TINY)
    del model.encoding
    scorer = PolicyScorer(model, "cpu")
    assert scorer.encoding == MACHINE
    task = task_for("sum")
    outputs = outputs_of(task)
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    assert same_block(scorer.encode(product, outputs), legacy_block(product, outputs))


def test_log_probs_still_scores_through_logits():
    """A caller that wraps ``logits`` (the debugger does) still sees every row."""
    scorer = PolicyScorer(StackPairPolicy(TINY, encoding=DIFFERENCE), "cpu")
    task = task_for("sum")
    block = scorer.encode(
        tuple(parse_stack(text, basis=BASIS) for text in task.inputs),
        outputs_of(task),
    )
    seen = []
    real = scorer.logits

    def logits(blocks):
        seen.append(list(blocks))
        return real(blocks)

    scorer.logits = logits
    legal = np.zeros((1, len(OP_NAMES)), bool)
    legal[0, :3] = True
    rows = scorer.log_probs([block], legal)
    assert len(seen) == 1 and seen[0][0] is block
    raw = real([block])[0, :3]
    assert np.allclose(rows[0, :3], raw - np.log(np.exp(raw).sum()), atol=1e-5)
    assert np.isneginf(rows[0, 3:]).all()


def test_a_machine_model_cannot_read_a_flag_token():
    """The loud half of a mismatch: a flag indexes past a `machine` embedding."""
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    task = task_for("sum")
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    block = encode_node(product, outputs_of(task), DIFFERENCE)
    with pytest.raises(IndexError):
        scorer.logits([block])


# ---------------------------------------------------------------------------
# Training rows and training
# ---------------------------------------------------------------------------


def test_witness_rows_default_to_the_machine_layout_unchanged():
    tasks = [task_for(name, seed) for name in BEHAVIOURS for seed in (4, 5)]
    default, machine = witness_rows(tasks), witness_rows(tasks, MACHINE)
    assert len(default) == len(machine) == sum(task.length for task in tasks)
    rows = iter(default)
    for index, task in enumerate(tasks):
        outputs = outputs_of(task)
        for step, states, name in replay_witness(task):
            row = next(rows)
            assert same_block(row.items, legacy_block(states, outputs))
            assert (row.label, row.task, row.step, row.remaining) == (
                OP_NAMES.index(name),
                index,
                step,
                task.length - step,
            )
    for left, right in zip(default, machine, strict=True):
        assert same_block(left.items, right.items)
        assert (left.label, left.task, left.step, left.remaining) == (
            right.label,
            right.task,
            right.step,
            right.remaining,
        )


def test_witness_rows_build_their_items_in_the_encoding_asked_for():
    tasks = [task_for("sum_positive"), task_for("negate_each", 5)]
    machine, difference = witness_rows(tasks), witness_rows(tasks, DIFFERENCE)
    assert len(machine) == len(difference)
    rows = iter(difference)
    for task in tasks:
        outputs = outputs_of(task)
        for _step, states, _name in replay_witness(task):
            assert same_block(
                next(rows).items, encode_node(states, outputs, DIFFERENCE)
            )
    for plain, flagged in zip(machine, difference, strict=True):
        assert (plain.label, plain.task, plain.step, plain.remaining) == (
            flagged.label,
            flagged.task,
            flagged.step,
            flagged.remaining,
        )
        assert flagged.items.shape[0] == plain.items.shape[0]
        assert flagged.items.max() >= FLAG_BASE > plain.items.max()


def test_train_policy_builds_the_model_in_the_encoding_and_records_it():
    tasks = [task_for("sum"), task_for("sum", 5)]
    rows = witness_rows(tasks)
    default, default_state = train_policy(rows, rows, TINY, epochs=2, **TRAINING)
    named, named_state = train_policy(
        rows, rows, TINY, epochs=2, encoding=MACHINE, **TRAINING
    )
    assert default.encoding == named.encoding == MACHINE
    assert default_state["encoding"] == named_state["encoding"] == MACHINE
    for name, tensor in default.state_dict().items():
        assert torch.equal(tensor, named.state_dict()[name])
    flagged = witness_rows(tasks, DIFFERENCE)
    model, state = train_policy(
        flagged, flagged, TINY, epochs=2, encoding=DIFFERENCE, **TRAINING
    )
    assert model.encoding == state["encoding"] == DIFFERENCE
    assert model.embedding.num_embeddings == EMBEDDING_ROWS[DIFFERENCE]
    assert state["parameters"] == model.parameter_count
    assert state["parameters"] == default_state["parameters"] + 4 * FLAG_TOP * 8
    json.dumps(state)  # the trainer state is written beside the weights


@pytest.fixture(scope="module")
def trained_difference():
    task = task_for("sum")
    rows = witness_rows([task, task_for("sum", 5)], DIFFERENCE)
    model, state = train_policy(
        rows, rows, TINY, epochs=40, encoding=DIFFERENCE, **TRAINING
    )
    return task, model, state


def test_a_trained_difference_policy_guides_a_narrow_beam_to_the_witness(
    trained_difference,
):
    """The layout it trained on is the layout the beam feeds it."""
    task, model, _state = trained_difference
    scorer = PolicyScorer(model, "cpu")
    result = beam_solve(task, scorer, beam_width=4, max_depth=task.length)
    assert result.termination == "fit", result
    assert grade(result.program, task).solved


# ---------------------------------------------------------------------------
# The searches build their blocks in the scorer's layout
# ---------------------------------------------------------------------------


class Recorder(BeamObserver):
    def __init__(self):
        self.layers = []

    def layer_started(self, depth, beam):
        self.layers.append([product for product, _program, _score in beam])


def spy_on_logits(scorer):
    seen = []
    real = scorer.logits

    def logits(blocks):
        seen.append(list(blocks))
        return real(blocks)

    scorer.logits = logits
    return seen


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_the_flat_beam_scores_the_blocks_the_scorers_encoder_builds(encoding):
    task = task_for("sum_positive")
    outputs = outputs_of(task)
    torch.manual_seed(0)
    scorer = PolicyScorer(StackPairPolicy(TINY, encoding=encoding), "cpu")
    seen = spy_on_logits(scorer)
    made = []
    real = scorer.encoder

    def encoder(target_stacks):
        made.append(target_stacks)
        return real(target_stacks)

    scorer.encoder = encoder
    watch = Recorder()
    beam_solve(task, scorer, beam_width=3, max_depth=4, observer=watch)
    assert made == [outputs]  # once per search
    # A layer that ends the search is started but not scored.
    assert 1 <= len(seen) <= len(watch.layers)
    assert sum(len(blocks) for blocks in seen) > 1
    for blocks, products in zip(seen, watch.layers, strict=False):
        assert len(blocks) == len(products)
        for block, product in zip(blocks, products, strict=True):
            assert same_block(block, encode_node(product, outputs, encoding))
            if encoding == MACHINE:
                assert same_block(block, legacy_block(product, outputs))


def test_the_segment_beam_feeds_a_difference_model_difference_blocks():
    task = task_for("sum_positive")
    scorer = PolicyScorer(StackPairPolicy(TINY, encoding=DIFFERENCE), "cpu")
    seen = spy_on_logits(scorer)
    result = segment_beam_solve(
        task, scorer, beam_width=2, roots=2, segment_depth=3, max_segments=2
    )
    assert result.termination in ("max_segments", "frontier_exhausted", "fit")
    blocks = [block for call in seen for block in call]
    assert blocks
    for block in blocks:
        assert block.shape[0] == task.examples
        assert ((block == SEP).sum(axis=1) == 2).all()
        assert (block.max(axis=1) >= FLAG_BASE).all()


def test_a_scorer_with_only_log_probs_is_given_machine_blocks():
    task = task_for("sum_positive")
    outputs = outputs_of(task)
    watch = Recorder()

    class Flat:
        def __init__(self):
            self.calls = []

        def log_probs(self, blocks, legal):
            self.calls.append(list(blocks))
            return np.where(legal, 0.0, -np.inf)

    scorer = Flat()
    beam_solve(task, scorer, beam_width=3, max_depth=3, observer=watch)
    assert 1 <= len(scorer.calls) <= len(watch.layers)
    for blocks, products in zip(scorer.calls, watch.layers, strict=False):
        assert len(blocks) == len(products)
        for block, product in zip(blocks, products, strict=True):
            assert same_block(block, legacy_block(product, outputs))


# ---------------------------------------------------------------------------
# Checkpoints record the layout
# ---------------------------------------------------------------------------


def test_policy_config_records_the_encoding_and_the_embedding_rows():
    default = policy_config(TINY)
    assert default == policy_config(TINY, OP_NAMES, MACHINE)
    assert default["encoding"] == MACHINE
    assert default["vocabulary"]["size"] == VOCAB
    difference = policy_config(TINY, OP_NAMES, DIFFERENCE)
    assert difference["encoding"] == DIFFERENCE
    assert difference["vocabulary"]["size"] == EMBEDDING_ROWS[DIFFERENCE]
    # Nothing else in the config depends on the layout.
    for config in (default, difference):
        del config["encoding"]
        del config["vocabulary"]["size"]
    assert default == difference
    assert set(default) == {
        "schema_version",
        "architecture",
        "vocabulary",
        "basis",
        "ordered_actions",
    }


def saved(tmp_path, model, name="policy"):
    return save_policy(
        tmp_path / name,
        model,
        trainer_state={"encoding": model.encoding},
        manifest={"arm": "tiny"},
        wandb_url=None,
    )


def probe_node():
    task = task_for("sum_positive")
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    return product, outputs_of(task)


def probe_block(encoding):
    return encode_node(*probe_node(), encoding)


def test_a_difference_checkpoint_round_trips_in_its_encoding(
    tmp_path, trained_difference
):
    _task, model, _state = trained_difference
    path = saved(tmp_path, model)
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["encoding"] == DIFFERENCE
    assert config["vocabulary"]["size"] == EMBEDDING_ROWS[DIFFERENCE]
    loaded = load_policy(path)
    assert loaded.encoding == DIFFERENCE
    assert loaded.embedding.num_embeddings == EMBEDDING_ROWS[DIFFERENCE]
    assert PolicyScorer(loaded).encoding == DIFFERENCE
    block = probe_block(DIFFERENCE)
    assert np.array_equal(
        PolicyScorer(loaded).logits([block]), PolicyScorer(model).logits([block])
    )


def test_a_machine_checkpoint_records_machine_and_round_trips(tmp_path):
    torch.manual_seed(0)
    model = StackPairPolicy(TINY).eval()
    path = saved(tmp_path, model)
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["encoding"] == MACHINE
    assert config["vocabulary"]["size"] == VOCAB
    loaded = load_policy(path)
    assert loaded.encoding == MACHINE
    block = probe_block(MACHINE)
    assert np.array_equal(
        PolicyScorer(loaded).logits([block]), PolicyScorer(model).logits([block])
    )


def test_a_config_without_the_field_loads_as_machine(tmp_path):
    """Every published checkpoint: written before the layout was recorded."""
    torch.manual_seed(0)
    model = StackPairPolicy(TINY).eval()
    path = saved(tmp_path, model)
    config = json.loads((path / CONFIG_FILE).read_text())
    del config["encoding"]
    (path / CONFIG_FILE).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    loaded = load_policy(path)
    assert loaded.encoding == MACHINE
    assert loaded.embedding.num_embeddings == VOCAB
    scorer = PolicyScorer(loaded)
    assert scorer.encoding == MACHINE
    block = probe_block(MACHINE)
    assert same_block(block, legacy_block(*probe_node()))
    assert np.array_equal(scorer.logits([block]), PolicyScorer(model).logits([block]))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


def test_a_checkpoint_cannot_be_read_in_another_layout_than_it_was_saved_in(
    tmp_path, trained_difference
):
    _task, model, _state = trained_difference
    for name, encoding in (("relabelled", MACHINE), ("legacy", None)):
        path = saved(tmp_path, model, name)
        config = json.loads((path / CONFIG_FILE).read_text())
        if encoding is None:
            del config["encoding"]
        else:
            config["encoding"] = encoding
        (path / CONFIG_FILE).write_text(json.dumps(config))
        with pytest.raises(RuntimeError, match=r"embedding\.weight"):
            load_policy(path)
    path = saved(tmp_path, model, "unknown")
    config = json.loads((path / CONFIG_FILE).read_text())
    config["encoding"] = "bogus"
    (path / CONFIG_FILE).write_text(json.dumps(config))
    with unknown():
        load_policy(path)
