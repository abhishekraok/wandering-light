"""The segment-reset beam: closed states, `segment_beam_solve`, the K diagnostic."""

import gc
import random

import numpy as np
import pytest

from wandering_light import solve
from wandering_light.quotation_guard import RUNS_CODE
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import (
    OP_INDEX,
    _prepare,
    _run_segment,
    _Tally,
    closed_cuts,
    segment_beam_solve,
    segment_witness_ranks,
    step_all_quiet,
)
from wandering_light.stack_machine import Mark
from wandering_light.stack_pbe import BY_NAME, OP_BY_NAME, OPS, build_task, draw_task
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import (
    OP_NAMES,
    PolicyScorer,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


class Seeded:
    """Seeded logits, no model: the search only calls ``log_probs``."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.calls = 0

    def log_probs(self, blocks, legal):
        self.calls += 1
        rng = np.random.default_rng(self.seed + len(blocks))
        raw = rng.standard_normal((len(blocks), len(OP_NAMES)))
        masked = np.where(legal, raw, -np.inf)
        peak = masked.max(axis=1, keepdims=True)
        shifted = masked - np.where(np.isfinite(peak), peak, 0.0)
        with np.errstate(divide="ignore"):
            return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def replay(task, program=None):
    inputs, _, _ = _prepare(task, "both")
    states = [inputs]
    for name in task.program if program is None else program:
        states.append(step_all_quiet(states[-1], OP_BY_NAME[name])[0])
    return states


@pytest.fixture(scope="module")
def sum_scorer():
    """A tiny policy fit to `sum`'s own witness prefixes, as the beam tests do."""
    rows = witness_rows([task_for("sum"), task_for("sum", 5)])
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
    )
    return PolicyScorer(model, "cpu")


def test_op_index_inverts_ops():
    assert len(OP_INDEX) == len(OPS)
    assert all(OPS[OP_INDEX[op.name]] is op for op in OPS)


def test_closed_cuts_skip_combinators_run_under_a_mark():
    task = task_for("prefix_sums")
    states = replay(task)
    cuts = closed_cuts(states, task.program)
    # The inner ``ifte`` runs inside the fold's open body, so it closes nothing.
    assert "ifte" in task.program[: cuts[0]]
    assert cuts == [
        t + 1
        for t, name in enumerate(task.program)
        if name in RUNS_CODE
        and not any(isinstance(v, Mark) for s in states[t] for v in s)
    ]
    assert cuts == [20, 25]
    assert closed_cuts(replay(task_for("sum")), task_for("sum").program) == [5]
    assert closed_cuts(replay(task_for("add_pair", 6)), ("add",)) == []


def test_prepare_refuses_marked_targets_under_the_close_guard():
    task = task_for("sum")
    inputs, outputs, targets = _prepare(task, "both")
    assert len(inputs) == len(outputs) == len(targets) == task.examples
    marked = task.__class__(**{**task.__dict__, "outputs": ("[ MARK ]",) * 4})
    with pytest.raises(ValueError, match="mark-free targets"):
        _prepare(marked, "close")


def test_a_one_op_witness_is_a_fit_in_the_first_layer():
    task = task_for("add_pair", 6)
    result = segment_beam_solve(task, Seeded(), beam_width=8)
    assert result.termination == "fit"
    assert result.program == ("add",)
    assert result.depth == 1
    assert result.expansions == 1
    assert grade(result.program, task).solved


def test_a_trained_policy_reaches_the_witness_through_a_segment(sum_scorer):
    task = task_for("sum")
    result = segment_beam_solve(task, sum_scorer, beam_width=4, segment_depth=6)
    assert result.termination == "fit", result
    assert grade(result.program, task).solved
    assert result.depth == len(result.program)


def test_nothing_closed_is_frontier_exhausted():
    """``max_pair`` starts on two ints, where no combinator can run in one op."""
    result = segment_beam_solve(task_for("max_pair"), Seeded(), segment_depth=1)
    assert result.termination == "frontier_exhausted"
    assert result.expansions == 1
    assert result.depth == 1  # the deepest program any expansion produced


def test_segments_run_out_as_max_segments():
    result = segment_beam_solve(
        task_for("negate_each"), Seeded(), segment_depth=3, max_segments=1
    )
    assert result.termination == "max_segments"
    assert result.depth == 3


def test_budget_is_checked_before_every_attempt():
    task = task_for("sum")
    result = segment_beam_solve(task, Seeded(), budget=50)
    assert result.termination == "budget"
    assert result.program is None
    assert 50 <= result.executions < 50 + task.examples


def test_closed_states_take_no_beam_slot_and_the_last_layer_drops_open_ones():
    task = task_for("negate_each")
    inputs, outputs, targets = _prepare(task, "both")
    root = (inputs, (), 0.0)
    tally = _Tally()
    scorer = Seeded()
    stop, program, closed = _run_segment(
        [root],
        outputs,
        targets,
        scorer,
        tally,
        beam_width=1,
        segment_depth=3,
        horizon=36,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert (stop, program) == (None, None)
    # Beam 1: one node per layer, three layers, one scoring call per layer.
    assert tally.expansions == 3
    assert scorer.calls == 3
    assert tally.deepest == 3
    assert closed
    for product, (score, path) in closed.items():
        assert path[-1] in RUNS_CODE
        states = replay(task, path)
        assert states[-1] == product
        assert not any(isinstance(v, Mark) for s in states[-2] for v in s)
        assert np.isfinite(score)
    # ``append`` starts on two lists, where ``i`` and ``dip`` both close in one
    # op: two closed states recorded from a beam with one slot, which the last
    # (only) layer leaves empty of open successors.
    task = task_for("append")
    inputs, outputs, targets = _prepare(task, "both")
    one_layer = _Tally()
    _, _, first = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        Seeded(),
        one_layer,
        beam_width=1,
        segment_depth=1,
        horizon=36,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert one_layer.expansions == 1
    assert sorted(path for _, path in first.values()) == [("dip",), ("i",)]


def test_each_root_has_its_own_visited_set():
    task = task_for("negate_each")
    inputs, outputs, targets = _prepare(task, "both")
    root = (inputs, (), 0.0)
    kwargs = {
        "beam_width": 4,
        "segment_depth": 2,
        "horizon": 36,
        "guard": "both",
        "budget": None,
        "max_state_atoms": None,
        "stop_at_fit": False,
    }
    alone = _Tally()
    _run_segment([root], outputs, targets, Seeded(), alone, **kwargs)
    twice = _Tally()
    _run_segment([root, root], outputs, targets, Seeded(), twice, **kwargs)
    assert twice.expansions == 2 * alone.expansions
    assert twice.executions == 2 * alone.executions


def test_the_search_is_reproducible_and_restores_the_collector():
    task = task_for("reverse", 9)
    was = gc.isenabled()
    first = segment_beam_solve(task, Seeded(3), beam_width=4, segment_depth=4)
    second = segment_beam_solve(task, Seeded(3), beam_width=4, segment_depth=4)
    assert gc.isenabled() is was
    for name in ("program", "termination", "executions", "fuel", "expansions", "depth"):
        assert getattr(first, name) == getattr(second, name), name


def test_witness_ranks_report_one_row_per_closed_cut(sum_scorer):
    task = task_for("sum")
    rows = segment_witness_ranks(task, sum_scorer, beam_width=4, segment_depth=6)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "segment",
        "ops",
        "roots",
        "forced",
        "closed",
        "rank",
        "score",
        "best",
        "executions",
        "complete",
    }
    assert (row["segment"], row["ops"], row["roots"], row["forced"]) == (
        0,
        5,
        1,
        False,
    )
    assert row["complete"]
    assert row["rank"] is not None and 1 <= row["rank"] <= row["closed"]
    assert row["best"] >= row["score"]


def test_witness_ranks_match_the_search_they_diagnose():
    """With no fit in reach, the diagnostic's first segment is the search's."""
    task = task_for("negate_each")
    kwargs = {"beam_width": 4, "segment_depth": 3, "max_segments": 1}
    search = segment_beam_solve(task, Seeded(1), **kwargs)
    assert search.termination == "max_segments"
    rows = segment_witness_ranks(task, Seeded(1), **kwargs)
    assert len(rows) == 1
    assert rows[0]["executions"] == search.executions
    # The witness's closed state is six ops away, beyond a three-op segment.
    assert rows[0]["ops"] == 6
    assert rows[0]["rank"] is None and rows[0]["score"] is None


def test_a_lost_witness_is_forced_back_as_a_root():
    task = task_for("prefix_sums")
    rows = segment_witness_ranks(
        task, Seeded(2), beam_width=2, roots=2, segment_depth=2
    )
    assert [r["ops"] for r in rows] == [20, 5]
    assert rows[0]["rank"] is None
    assert rows[1]["forced"]
    assert rows[1]["roots"] == min(2, rows[0]["closed"]) + 1
    assert rows[1]["executions"] > rows[0]["executions"]


def test_witness_ranks_stop_at_the_budget():
    task = task_for("prefix_sums")
    rows = segment_witness_ranks(task, Seeded(), segment_depth=2, budget=40)
    assert len(rows) == 1
    assert rows[0]["complete"] is False


def test_a_refused_witness_step_is_named():
    with pytest.raises(ValueError, match=r"sum: the program refuses step 0 \(add\)"):
        segment_witness_ranks(task_for("sum"), Seeded(), ("add",))


def test_system_chooses_the_search_and_best_is_flat():
    assert solve.SEARCHES == ("flat", "segment")
    assert solve.BEST.search == "flat"
    assert "search" not in solve.BEST.describe()
    base = {
        "name": "tiny",
        "repo_id": "nobody/nothing",
        "revision": "0" * 40,
        "beam_width": 4,
        "max_depth": 3,
        "guard": "both",
        "library": False,
        "measured": "a test",
    }
    with pytest.raises(ValueError, match="unknown search 'deep'"):
        solve.System(**base, search="deep")
    segment = solve.System(**base, search="segment", roots=3, segment_depth=5)
    assert segment.describe().endswith(
        ", search segment: 3 roots, 12 segments of up to 5 ops, the beam per root"
    )


def test_solve_tasks_runs_the_segment_search(monkeypatch):
    calls = []
    real = segment_beam_solve

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(solve, "segment_beam_solve", spy)
    monkeypatch.setattr(
        solve, "beam_solve", lambda *a, **k: pytest.fail("flat search ran")
    )
    system = solve.System(
        name="tiny",
        repo_id="nobody/nothing",
        revision="0" * 40,
        beam_width=8,
        max_depth=1,
        guard="both",
        library=False,
        measured="a test",
        search="segment",
        roots=2,
        segment_depth=3,
        max_segments=2,
    )
    rows = solve.solve_tasks([task_for("add_pair", 6)], Seeded(), system)
    assert rows[0].termination == "fit"
    assert rows[0].solved
    assert calls == [
        {
            "beam_width": 8,
            "roots": 2,
            "segment_depth": 3,
            "max_segments": 2,
            "guard": "both",
            "max_state_atoms": None,
        }
    ]
