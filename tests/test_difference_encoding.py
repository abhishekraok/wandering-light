"""The `difference` policy encoding and the `capped` training arm (short-segment-campaign, #168)."""

import dataclasses
import json
import random

import numpy as np
import pytest
import torch

import experiments.compute_optimal_mark28 as co
import experiments.inverse_dynamics as idp
import experiments.staged_arms as sa
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import (
    beam_solve,
    block_encoder,
    segment_beam_solve,
    segment_witness_ranks,
)
from wandering_light.stack_machine import Int, Lst, parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_policy import (
    ADDED,
    CHANGED,
    CONFIG_FILE,
    DIFFERENCE,
    EQUAL,
    FLAG_BASE,
    MACHINE,
    OP_NAMES,
    REMOVED,
    VOCAB,
    PolicyScorer,
    StackPairPolicy,
    encode_item,
    encode_node,
    encode_targets,
    flags,
    item_array,
    load_policy,
    replay_witness,
    save_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def outputs_of(task):
    return tuple(parse_stack(text, basis=BASIS) for text in task.outputs)


def witness_products(names=("sum", "length", "square_each", "keep_above")):
    """Every (product, outputs) along several witnesses, lists and scalars mixed."""
    out = []
    for name in names:
        task = task_for(name)
        out += [(states, outputs_of(task)) for _, states, _ in replay_witness(task)]
    return out


def old_block(product, outputs):
    """The machine block exactly as the beams built it before #168."""
    targets = encode_targets(outputs)
    return item_array(
        [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
    )


def same_block(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# Flags and blocks
# ---------------------------------------------------------------------------


def decode(tokens):
    return [((t - FLAG_BASE) // 4, (t - FLAG_BASE) % 4) for t in tokens]


def test_flags_say_kept_changed_added_removed_from_the_top():
    s = (Int(9), Int(1), Int(2), Int(3))
    assert decode(flags(s, (Int(1), Int(3), Int(2)))) == [
        (0, CHANGED),
        (1, CHANGED),
        (2, EQUAL),
        (3, REMOVED),
    ]
    assert decode(flags((Int(1),), (Int(0), Int(1)))) == [(0, EQUAL), (1, ADDED)]
    assert flags((), ()) == []
    # Only the top eight positions are compared.
    deep = tuple(Int(i) for i in range(20))
    assert decode(flags(deep, deep)) == [(k, EQUAL) for k in range(8)]
    assert all(FLAG_BASE <= t < FLAG_BASE + 32 for t in flags(deep, ()))


def test_flags_are_the_ones_inverse_dynamics_validated():
    lists = (Lst((Int(1), Int(2))), Lst(()), Int(3))
    pairs = [(product, outputs) for product, outputs in witness_products()]
    pairs += [((lists,), ((Int(3), Lst(())),))]
    for product, outputs in pairs:
        for s, t in zip(product, outputs, strict=True):
            assert flags(s, t) == idp.flags(s, t)


def test_machine_blocks_are_byte_identical_to_the_old_path():
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")

    class Stub:  # a duck-typed scorer, as the beam tests write them
        def log_probs(self, blocks, legal):
            return np.where(legal, 0.0, -np.inf)

    for product, outputs in witness_products():
        old = old_block(product, outputs)
        assert same_block(encode_node(product, outputs), old)
        assert same_block(encode_node(product, outputs, MACHINE), old)
        assert same_block(scorer.encode(product, outputs), old)
        assert same_block(block_encoder(scorer, outputs)(product), old)
        assert same_block(block_encoder(Stub(), outputs)(product), old)


def test_machine_training_rows_are_unchanged():
    tasks = [task_for(name) for name in ("sum", "length", "square_each")]
    new, explicit = witness_rows(tasks), witness_rows(tasks, MACHINE)
    old = []
    for index, task in enumerate(tasks):
        outputs = outputs_of(task)
        for step, states, name in replay_witness(task):
            old.append((old_block(states, outputs), OP_NAMES.index(name), index, step))
    assert len(new) == len(explicit) == len(old)
    for a, b, (block, label, task, step) in zip(new, explicit, old, strict=True):
        assert same_block(a.items, block) and same_block(b.items, block)
        assert (a.label, a.task, a.step) == (label, task, step)


def test_difference_blocks_are_the_probes_items_with_the_task_outputs_as_target():
    for product, outputs in witness_products():
        block = encode_node(product, outputs, DIFFERENCE)
        probe = idp.sequence_items({"s": product, "t": outputs}, "difference")
        assert same_block(block, probe)
        machine = old_block(product, outputs)
        # The machine item is a prefix of each difference item.
        assert (
            block[:, : machine.shape[1]][machine != 0] == machine[machine != 0]
        ).all()
        assert block.max() >= FLAG_BASE


def test_difference_rows_carry_flags_and_the_same_labels():
    tasks = [task_for("sum"), task_for("length")]
    machine, difference = witness_rows(tasks), witness_rows(tasks, DIFFERENCE)
    assert [(r.label, r.task, r.step) for r in machine] == [
        (r.label, r.task, r.step) for r in difference
    ]
    assert all(r.items.max() >= FLAG_BASE for r in difference)
    assert all(r.items.max() < VOCAB for r in machine)


def test_unknown_encoding_is_refused():
    with pytest.raises(ValueError, match="unknown encoding"):
        StackPairPolicy(TINY, encoding="aligned")
    with pytest.raises(ValueError, match="unknown encoding"):
        encode_node(((Int(1),),), ((Int(1),),), "aligned")


# ---------------------------------------------------------------------------
# Models and checkpoints
# ---------------------------------------------------------------------------


def test_difference_policy_widens_only_the_embedding():
    """#168's parameter counts: 655,836 for `1m`, 659,932 with the flag rows."""
    one_m = PolicyArchitecture(128, 320, 640)
    assert StackPairPolicy(one_m).parameter_count == 655_836
    assert StackPairPolicy(one_m, encoding=DIFFERENCE).parameter_count == 659_932
    tiny = StackPairPolicy(TINY, encoding=DIFFERENCE)
    assert tiny.embedding.num_embeddings == VOCAB + 32
    assert StackPairPolicy(TINY).embedding.num_embeddings == VOCAB


def logits_on(model, block):
    return PolicyScorer(model, "cpu").logits([block])


def test_difference_checkpoint_round_trips_its_encoding(tmp_path):
    torch.manual_seed(0)
    model = StackPairPolicy(TINY, encoding=DIFFERENCE)
    path = save_policy(
        tmp_path / "difference", model, trainer_state={}, manifest={}, wandb_url=None
    )
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config["encoding"] == DIFFERENCE
    assert config["vocabulary"]["size"] == VOCAB + 32
    loaded = load_policy(path)
    assert loaded.encoding == DIFFERENCE
    assert PolicyScorer(loaded).encoding == DIFFERENCE
    product, outputs = witness_products(("sum",))[1]
    block = encode_node(product, outputs, DIFFERENCE)
    np.testing.assert_array_equal(logits_on(model, block), logits_on(loaded, block))


def test_a_config_without_encoding_loads_as_machine(tmp_path):
    torch.manual_seed(0)
    model = StackPairPolicy(TINY)
    path = save_policy(
        tmp_path / "old", model, trainer_state={}, manifest={}, wandb_url=None
    )
    config = json.loads((path / CONFIG_FILE).read_text())
    assert config.pop("encoding") == MACHINE
    (path / CONFIG_FILE).write_text(json.dumps(config))
    loaded = load_policy(path)
    assert loaded.encoding == MACHINE
    assert loaded.embedding.num_embeddings == VOCAB
    product, outputs = witness_products(("sum",))[1]
    block = old_block(product, outputs)
    np.testing.assert_array_equal(logits_on(model, block), logits_on(loaded, block))


class Recording(PolicyScorer):
    """A real scorer that remembers the largest token any block held."""

    peak = 0

    def logits(self, blocks):
        self.peak = max(self.peak, *(int(block.max()) for block in blocks))
        return super().logits(blocks)


def test_a_difference_policy_runs_in_both_beams():
    torch.manual_seed(0)
    task = task_for("sum")
    scorer = Recording(StackPairPolicy(TINY, encoding=DIFFERENCE), "cpu")
    flat = beam_solve(task, scorer, beam_width=4, max_depth=6)
    assert flat.termination in ("fit", "max_depth", "frontier_exhausted")
    assert scorer.peak >= FLAG_BASE
    knobs = {"beam_width": 4, "roots": 2, "segment_depth": 6, "max_segments": 2}
    scorer.peak = 0
    segment = segment_beam_solve(task, scorer, **knobs)
    assert segment.termination in ("fit", "max_segments", "frontier_exhausted")
    assert scorer.peak >= FLAG_BASE
    rows = segment_witness_ranks(task, scorer, **knobs)
    assert len(rows) == 1  # `fold` is the witness's one top-level application


def test_a_machine_policy_never_sees_a_flag_token():
    torch.manual_seed(0)
    scorer = Recording(StackPairPolicy(TINY), "cpu")
    beam_solve(task_for("sum"), scorer, beam_width=4, max_depth=6)
    segment_beam_solve(
        task_for("sum"), scorer, beam_width=4, roots=2, segment_depth=6, max_segments=2
    )
    assert 0 < scorer.peak < VOCAB


# ---------------------------------------------------------------------------
# staged_arms: the capped arm and --encoding
# ---------------------------------------------------------------------------

FIT = ("sum", "length", "square_each")
HELD = "keep_above"  # its skeleton is in `validation_skeleton`'s tenth
CAPPED_SUM = "dup drop 0 [ add ] fold"


def arm_tasks():
    tasks = [
        dataclasses.replace(task_for(name, seed), split="train")
        for name in (*FIT, HELD)
        for seed in (4, 5)
    ]
    assert {t.behaviour for t in tasks if sa.validation_skeleton(t)} == {HELD}
    return tasks


@pytest.fixture
def capped_file(tmp_path, monkeypatch):
    path = tmp_path / "train_capped.jsonl"
    path.write_text(
        json.dumps({"behaviour": "sum", "program": CAPPED_SUM, "status": "capped"})
        + "\n"
    )
    monkeypatch.setattr(sa, "CAPPED_FILE", path)
    return path


def test_the_arm_list_includes_capped():
    assert "capped" in sa.ARMS
    assert sa.ARMS[:4] == ("fused", "refused", "staged", "both")


def test_capped_without_its_side_file_fails_clearly(tmp_path, monkeypatch):
    missing = tmp_path / "train_capped.jsonl"
    monkeypatch.setattr(sa, "CAPPED_FILE", missing)
    with pytest.raises(FileNotFoundError, match=r"train_capped\.jsonl.*does not exist"):
        sa.arm_rows("capped", arm_tasks(), keep_contrived=False)
    with pytest.raises(FileNotFoundError, match="short-segment-campaign"):
        sa.check_arms(["fused", "capped"])
    sa.check_arms(["fused", "staged"])  # the other arms never read it
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sa.main(["rows", "--arms", "capped", "--run-dir", str(tmp_path / "run")])


def test_capped_side_file_lines_are_checked(tmp_path):
    path = tmp_path / "bad.jsonl"
    line = json.dumps({"behaviour": "sum", "program": CAPPED_SUM})
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="listed twice"):
        sa.capped_programs(path)
    path.write_text(json.dumps({"behaviour": "sum", "program": ""}) + "\n")
    with pytest.raises(ValueError, match="needs a behaviour and a program"):
        sa.capped_programs(path)


def test_capped_rows_replay_the_listed_program_at_the_fused_row_count(capped_file):
    tasks = arm_tasks()
    fused, held, _ = sa.arm_rows("fused", tasks, keep_contrived=False)
    fit, capped_held, facts = sa.arm_rows("capped", tasks, keep_contrived=False)
    assert len(fit) == len(fused) == facts["rows_per_epoch"]
    assert facts["treated_behaviours"] == 1 and facts["treated_tasks"] == 2
    assert facts["side_file"] == str(capped_file)
    assert len(capped_held) == len(held)
    fit_tasks = [t for t in tasks if not sa.validation_skeleton(t)]
    program = CAPPED_SUM.split()
    seen = set()
    for row in fit:
        task = fit_tasks[row.task]
        own = program if task.behaviour == "sum" else list(task.program)
        assert OP_NAMES[row.label] == own[row.step]
        assert row.remaining == len(own) - row.step
        seen.add(task.behaviour)
    assert "sum" in seen


def test_encoding_changes_the_tokens_not_the_draw(capped_file):
    tasks = arm_tasks()
    machine, _, _ = sa.arm_rows("capped", tasks, False)
    difference, held, _ = sa.arm_rows("capped", tasks, False, DIFFERENCE)
    assert [(r.task, r.step, r.label) for r in machine] == [
        (r.task, r.step, r.label) for r in difference
    ]
    assert all(r.items.max() >= FLAG_BASE for r in difference)
    assert all(r.items.max() >= FLAG_BASE for r in held)


def test_train_records_the_encoding_and_refuses_to_mix(
    capped_file, tmp_path, monkeypatch
):
    monkeypatch.setattr(co, "architecture", lambda arm: TINY)
    run_dir = tmp_path / "run"
    lines = []
    sa.train(
        run_dir, arm_tasks(), "cpu", ["capped"], [0], False, lines.append, DIFFERENCE
    )
    snapshot = sa.hs.policy_dir(run_dir, "capped", 0, sa.EPOCHS)
    assert load_policy(snapshot).encoding == DIFFERENCE
    manifest = json.loads((snapshot / "training_manifest.json").read_text())
    assert manifest["encoding"] == DIFFERENCE
    assert manifest["side_file"] == str(capped_file)
    state = json.loads((run_dir / "training" / "capped-s0.json").read_text())
    assert state["encoding"] == DIFFERENCE
    # The same encoding skips; the other one is an error, not a silent skip.
    sa.train(
        run_dir, arm_tasks(), "cpu", ["capped"], [0], False, lines.append, DIFFERENCE
    )
    assert any("skip train capped" in line for line in lines)
    with pytest.raises(ValueError, match="separate --run-dir"):
        sa.train(run_dir, arm_tasks(), "cpu", ["capped"], [0], False, lines.append)
