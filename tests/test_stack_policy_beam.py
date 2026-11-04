"""The stack-token policy and the policy-guided beam search beside ``bfs_solve``."""

import json
import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from wandering_light import stack_policy as sp
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import BeamResult, beam_solve, step_all
from wandering_light.stack_machine import (
    MARK,
    Bool,
    Int,
    Lst,
    Sym,
)
from wandering_light.stack_pbe import BY_NAME, OP_BY_NAME, OPS, build_task, draw_task

TINY = PolicyArchitecture(byte_embedding=8, conv_channels=8, hidden=8)


def _task(name, seed, examples=4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, examples, 4))


@pytest.fixture(scope="module")
def tasks():
    return [_task("sum", 4), _task("add_pair", 6), _task("sum", 9, examples=3)]


class _Uniform:
    """A scorer with no opinion: uniform over the legal ops."""

    def __init__(self):
        self.calls = 0

    def log_probs(self, blocks, legal):
        self.calls += 1
        legal = np.asarray(legal, dtype=bool)
        counts = legal.sum(axis=1, keepdims=True).clip(min=1)
        return np.where(legal, -np.log(counts), -np.inf)


class _Oracle:
    """Prefers the witness's next op on the matching program prefix."""

    def __init__(self, program):
        self.program = program

    def log_probs(self, blocks, legal):
        out = np.where(legal, -10.0, -np.inf)
        for name in set(self.program):
            index = sp.OP_INDEX[name]
            out[:, index] = np.where(legal[:, index], -0.1, -np.inf)
        return out


# --- vocabulary and encoding ------------------------------------------------


def test_vocabulary_layout():
    assert tuple(op.name for op in OPS) == sp.OP_NAMES
    assert len(sp.OP_NAMES) == 28
    assert sp.OP_INDEX["["] == sp.OP_NAMES.index("[")
    assert sp.VOCAB == 9 + 28 + 2 * sp.SMALL_INT + 1 + 10
    assert (sp.PAD, sp.SEP, sp.BIG) == (0, 1, 8)


def test_encode_stack_values():
    small = sp.encode_stack((Int(3), Int(-64), Int(64)), 99)
    assert len(small) == 3 and len(set(small)) == 3
    assert sp.encode_stack((Int(-205),), 99) == [
        sp.BIG,
        sp.NEG,
        sp.DIGIT_BASE + 2,
        sp.DIGIT_BASE + 0,
        sp.DIGIT_BASE + 5,
    ]
    assert sp.encode_stack((Int(65),), 99)[0] == sp.BIG
    assert sp.encode_stack((Bool(True), Bool(False), MARK), 99) == [
        sp.TRUE_TOKEN,
        sp.FALSE_TOKEN,
        sp.MARK_TOKEN,
    ]
    assert sp.encode_stack((Sym("add"),), 99) == [9 + sp.OP_INDEX["add"]]
    nested = sp.encode_stack((Lst((Int(1), Lst(()))),), 99)
    assert nested == [
        sp.OPEN,
        sp.encode_stack((Int(1),), 9)[0],
        sp.OPEN,
        sp.CLOSE,
        sp.CLOSE,
    ]
    # Bools are never confused with the ints 0 and 1.
    assert sp.encode_stack((Bool(True),), 9) != sp.encode_stack((Int(1),), 9)


def test_encode_stack_keeps_the_top():
    stack = tuple(Int(i) for i in range(10))
    assert sp.encode_stack(stack, 4) == sp.encode_stack(stack[-4:], 4)
    with pytest.raises(ValueError):
        sp.encode_stack((Sym("nope"),), 9)


def test_item_and_collate_contract():
    targets = sp.encode_targets([(Int(1),), (Lst((Int(2),)),)])
    items = [sp.encode_item((Int(5),), targets[0]), sp.encode_item((), targets[1])]
    assert items[0][1] == sp.SEP and items[1][0] == sp.SEP
    block = sp.item_array(items)
    assert block.dtype == np.int16 and block.shape == (2, 4)
    assert block[0, -1] == sp.PAD
    other = sp.item_array([[sp.SEP]])
    sequences, counts, mask = sp.collate([block, other])
    assert sequences.shape == (3, 4) and sequences.dtype == torch.long
    assert counts.tolist() == [2, 1]
    assert mask.tolist() == [[True, True], [True, False]]


# --- model and scorer ---------------------------------------------------------


def test_model_shape_and_padding_is_absent():
    torch.manual_seed(0)
    model = sp.StackPairPolicy(TINY).eval()
    assert model.parameter_count() == sum(p.numel() for p in model.parameters())
    block = sp.item_array([[sp.SEP, 20, 30]])
    wide = np.concatenate([block, np.zeros((1, 7), dtype=np.int16)], axis=1)
    with torch.no_grad():
        a = model(*sp.collate([block]))
        b = model(*sp.collate([wide]))
    assert a.shape == (1, 28)
    assert torch.allclose(a, b, atol=1e-5)


def test_scorer_normalises_over_legal_only():
    torch.manual_seed(0)
    scorer = sp.PolicyScorer(sp.StackPairPolicy(TINY))
    blocks = [sp.item_array([[sp.SEP, 20]]), sp.item_array([[sp.SEP, 21]])]
    legal = np.zeros((2, 28), dtype=bool)
    legal[0, [0, 3, 5]] = True
    out = scorer.log_probs(blocks, legal)
    assert np.isneginf(out[~legal]).all()
    assert math.isclose(np.exp(out[0, legal[0]]).sum(), 1.0, rel_tol=1e-5)
    assert np.isneginf(out[1]).all()
    assert not scorer.model.training


# --- rows ---------------------------------------------------------------------


def test_witness_rows(tasks):
    rows = sp.witness_rows(tasks)
    assert len(rows) == sum(len(t.program) for t in tasks)
    first = [r for r in rows if r.task == 0]
    assert [r.step for r in first] == list(range(len(tasks[0].program)))
    assert [sp.OP_NAMES[r.label] for r in first] == list(tasks[0].program)
    assert all(r.remaining == tasks[r.task].length - r.step for r in rows)
    assert all(len(r.items) == tasks[r.task].examples for r in rows)
    with pytest.raises(AttributeError):
        rows[0].__dict__  # noqa: B018 - slots


def test_replay_witness_refusal_raises(tasks):
    broken = replace(tasks[0], program=("add", "add", "add", "add"))
    with pytest.raises(ValueError):
        list(sp.replay_witness(broken))


def test_validation_skeleton_is_stable(tasks):
    assert sp.validation_skeleton(tasks[0]) == sp.validation_skeleton(tasks[2])
    held = sum(
        sp.validation_skeleton(replace(tasks[0], skeleton=f"s{i}")) for i in range(1000)
    )
    assert 60 < held < 140


# --- training and checkpoints --------------------------------------------------


def test_train_is_deterministic_and_keeps_the_best_epoch(tasks):
    rows = sp.witness_rows(tasks)
    seen = []
    kwargs = {
        "epochs": 3,
        "learning_rate": 1e-2,
        "batch_size": 4,
        "validation_batch_size": 8,
        "seed": 7,
        "device": "cpu",
    }
    model, state = sp.train_policy(
        rows, rows[:3], TINY, epoch_callback=seen.append, **kwargs
    )
    again, state_again = sp.train_policy(rows, rows[:3], TINY, **kwargs)
    assert state["history"] == state_again["history"]
    assert len(seen) == 3 and seen == state["history"]
    losses = [h["validation_cross_entropy"] for h in state["history"]]
    assert state["best_epoch"] == 1 + int(np.argmin(losses))
    assert state["best_validation_cross_entropy"] == min(losses)
    assert state["parameters"] == model.parameter_count()
    for a, b in zip(
        model.state_dict().values(), again.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
    with pytest.raises(ValueError):
        sp.train_policy([], rows, TINY, **kwargs)
    with pytest.raises(ValueError):
        sp.train_policy(rows, [], TINY, **kwargs)


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(1)
    model = sp.StackPairPolicy(TINY)
    url = "https://wandb.ai/abhishekraok-na/project/runs/abc"
    path = sp.save_policy(
        tmp_path / "ckpt",
        model,
        trainer_state={"best_epoch": 1},
        manifest={"status": "complete"},
        wandb_url=url,
    )
    names = {p.name for p in path.iterdir()}
    assert names == {
        sp.MODEL_FILE,
        sp.CONFIG_FILE,
        sp.TRAINER_STATE_FILE,
        sp.MANIFEST_FILE,
        sp.WANDB_LINK_FILE,
    }
    manifest = json.loads((path / sp.MANIFEST_FILE).read_text())
    assert manifest["wandb_url"] == url
    config = json.loads((path / sp.CONFIG_FILE).read_text())
    assert config["actions"] == list(sp.OP_NAMES) and config["basis"] == "mark-28"
    loaded = sp.load_policy(path)
    for a, b in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
    with pytest.raises(FileExistsError):
        sp.save_policy(path, model, trainer_state={}, manifest={}, wandb_url=None)

    bare = sp.save_policy(
        tmp_path / "bare", model, trainer_state={}, manifest={}, wandb_url=None
    )
    assert not (bare / sp.WANDB_LINK_FILE).exists()

    for field, value in [
        ("schema_version", 99),
        ("basis", "executable-26"),
        ("actions", list(reversed(sp.OP_NAMES))),
    ]:
        bad = json.loads((bare / sp.CONFIG_FILE).read_text())
        bad[field] = value
        (bare / sp.CONFIG_FILE).write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            sp.load_policy(bare)
        (bare / sp.CONFIG_FILE).write_text(json.dumps(config))


# --- beam search ---------------------------------------------------------------


def test_step_all_counts_the_refused_stack():
    product = ((Int(1), Int(2)), (Int(3),), (Int(4), Int(5)))
    moved, fuel, stepped = step_all(product, OP_BY_NAME["add"])
    assert moved is None and stepped == 2 and fuel == 1
    moved, fuel, stepped = step_all(product, OP_BY_NAME["dup"])
    assert moved is not None and stepped == 3 and fuel == 3


def test_beam_finds_short_witnesses(tasks):
    add = tasks[1]
    result = beam_solve(add, _Uniform(), beam_width=8, max_depth=4)
    assert isinstance(result, BeamResult)
    assert result.termination == "fit" and result.program is not None
    assert result.depth == len(result.program)
    assert result.executions > 0 and result.expansions >= 1

    oracle = beam_solve(tasks[0], _Oracle(tasks[0].program), beam_width=32, max_depth=6)
    assert oracle.termination == "fit"
    assert oracle.depth == len(oracle.program) <= len(tasks[0].program)


def test_beam_with_the_trained_policy_runs(tasks):
    torch.manual_seed(0)
    scorer = sp.PolicyScorer(sp.StackPairPolicy(TINY))
    result = beam_solve(tasks[1], scorer, beam_width=4, max_depth=3)
    assert result.termination in ("fit", "max_depth", "frontier_exhausted")


def test_beam_budget_and_depth(tasks):
    result = beam_solve(tasks[0], _Uniform(), beam_width=4, max_depth=3, budget=10)
    assert result.termination == "budget" and result.program is None
    assert result.executions >= 10 and result.depth == 1

    capped = beam_solve(tasks[0], _Uniform(), beam_width=2, max_depth=2)
    assert capped.termination == "max_depth" and capped.depth == 2
    assert capped.expansions == 1 + 2

    zero = beam_solve(tasks[0], _Uniform(), max_depth=0)
    assert zero.termination == "max_depth" and zero.executions == 0


def test_beam_frontier_exhausted(tasks, monkeypatch):
    import wandering_light.stack_beam as beam

    one = replace(tasks[0], inputs=("[ 1 ]",) * 3, outputs=("[ 7 ]",) * 3)
    # `drop` empties every stack, then refuses everywhere: no successors.
    monkeypatch.setattr(beam, "OPS", (OP_BY_NAME["drop"],))
    result = beam_solve(one, _Uniform(), max_depth=8, guard="off")
    assert result.termination == "frontier_exhausted" and result.depth == 1

    # `swap` twice returns to a seen state, so the next beam is empty.
    two = replace(tasks[0], inputs=("[ 1 2 ]",) * 3, outputs=("[ 7 ]",) * 3)
    monkeypatch.setattr(beam, "OPS", (OP_BY_NAME["swap"],))
    result = beam_solve(two, _Uniform(), max_depth=8, guard="off")
    assert result.termination == "frontier_exhausted" and result.depth == 1
    assert result.expansions == 2


def test_beam_close_guard_needs_mark_free_outputs(tasks):
    marked = replace(tasks[0], outputs=tuple("[ MARK ]" for _ in tasks[0].outputs))
    with pytest.raises(ValueError):
        beam_solve(marked, _Uniform(), guard="both")
    with pytest.raises(ValueError):
        beam_solve(marked, _Uniform(), guard="close")
    beam_solve(marked, _Uniform(), guard="off", max_depth=1)


def test_beam_guard_blocks_unclosable_quotations(tasks):
    """Opening a quotation with one op left can never close, so it is pruned."""

    result = beam_solve(tasks[0], _Uniform(), beam_width=64, max_depth=2)
    off = beam_solve(tasks[0], _Uniform(), beam_width=64, max_depth=2, guard="off")
    assert result.termination == off.termination == "max_depth"
    assert result.executions <= off.executions
