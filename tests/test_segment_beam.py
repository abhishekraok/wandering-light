"""The segment-reset beam, its witness diagnostic, and the `System` switch (#160).

A *closed state* is what a top-level combinator application leaves. The
segment beam resets at those states and nothing here touches the flat beam:
`beam_solve` and `BEST` are asserted unchanged, not re-tested.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest
import torch

from wandering_light import solve
from wandering_light.quotation_guard import RUNS_CODE
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import (
    OP_INDEX,
    _prepare,
    _run_segment,
    _Tally,
    beam_solve,
    closed_cuts,
    segment_beam_solve,
    segment_witness_ranks,
    step_all_quiet,
)
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import (
    BASIS,
    BY_NAME,
    OP_BY_NAME,
    OPS,
    build_task,
    draw_task,
)
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import (
    OP_NAMES,
    PolicyScorer,
    StackPairPolicy,
    encode_item,
    item_array,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def products(task):
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    return inputs, outputs


def replay(product, program):
    """``states[t]`` is the product before ``program[t]``."""
    states = [product]
    for name in program:
        moved, _fuel, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        assert moved is not None, name
        states.append(moved)
    return states


def untrained() -> PolicyScorer:
    torch.manual_seed(160)
    return PolicyScorer(StackPairPolicy(TINY), "cpu")


class _Flat:
    """Every legal op scores the same, so ranks follow generation order."""

    def log_probs(self, blocks, legal):
        return np.zeros(legal.shape)


class _Likes:
    """Scores one op above the rest; nothing is normalised."""

    def __init__(self, name: str, bonus: float = 1.0):
        self.name, self.bonus = name, bonus

    def log_probs(self, blocks, legal):
        out = np.zeros(legal.shape)
        out[:, OP_INDEX[self.name]] = self.bonus
        return out


# ---------------------------------------------------------------------------
# Closed states
# ---------------------------------------------------------------------------


def test_op_index_names_each_op_by_its_position():
    assert {op.name: index for index, op in enumerate(OPS)} == OP_INDEX
    assert {name: index for index, name in enumerate(OP_NAMES)} == OP_INDEX
    assert all(name in OP_INDEX for name in RUNS_CODE)


def test_closed_cuts_fall_after_top_level_combinators_only():
    """A combinator under an open mark is captured, so it closes nothing."""
    inputs, _ = products(task_for("add_pair", 6))
    program = ("[", "[", "]", "i", "]", "i")
    states = replay(inputs, program)
    assert closed_cuts(states, program) == [6]
    assert states[-1] == inputs
    assert closed_cuts(states[:1], ()) == []
    for name, cuts in (("sum", [5]), ("drop_last", [5, 12]), ("add_pair", [])):
        task = task_for(name)
        states = replay(products(task)[0], task.program)
        assert closed_cuts(states, task.program) == cuts, name
        for cut in cuts:
            assert task.program[cut - 1] in RUNS_CODE


def test_tally_starts_at_zero_and_is_mutable():
    tally = _Tally()
    assert (tally.executions, tally.fuel, tally.expansions, tally.deepest) == (
        0,
        0,
        0,
        0,
    )
    tally.executions += 3
    tally.deepest = 7
    assert (tally.executions, tally.deepest) == (3, 7)
    with pytest.raises(AttributeError):
        tally.anything = 1


def test_prepare_parses_the_task_and_refuses_marked_targets_under_the_close_guard():
    task = task_for("sum")
    inputs, outputs, targets = _prepare(task, "both")
    assert (inputs, outputs) == products(task)
    assert len(targets) == task.examples
    marked = dataclasses.replace(task, outputs=("[ MARK ]",) * task.examples)
    for guard in ("close", "both"):
        with pytest.raises(ValueError, match="mark-free targets"):
            _prepare(marked, guard)
    for guard in ("off", "body"):
        assert (
            _prepare(marked, guard)[1][0][0] == parse_stack("[ MARK ]", basis=BASIS)[0]
        )


# ---------------------------------------------------------------------------
# One segment
# ---------------------------------------------------------------------------


def test_closed_successors_are_recorded_whatever_their_rank_and_take_no_slot():
    """`nil dip`, `nil i` and `quote i` all run back to the inputs of `add_pair`.

    With beam 1 the open beam holds one node, yet every closed state the layer
    reaches is recorded. Under a flat scorer the three routes tie and the first
    generated, `nil dip`, is kept; a scorer that likes `i` lifts `nil i` above
    it, and `quote i` at the same score does not replace it.
    """
    task = task_for("add_pair", 6)
    inputs, outputs, targets = _prepare(task, "both")
    tally = _Tally()
    stop, program, closed = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Flat(),
        tally,
        beam_width=64,
        segment_depth=2,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert (stop, program) == (None, None)
    assert closed[inputs] == (0.0, ("nil", "dip"))
    assert all(route[-1] in RUNS_CODE for _score, route in closed.values())
    # Layer 1 expands the root, layer 2 every open successor of it.
    legal_at_root = sum(step_all_quiet(inputs, op)[0] is not None for op in OPS)
    assert tally.expansions == 1 + legal_at_root - sum(
        op.name in RUNS_CODE and step_all_quiet(inputs, op)[0] is not None for op in OPS
    )
    assert tally.deepest == 2

    narrow = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Likes("i"),
        _Tally(),
        beam_width=1,
        segment_depth=2,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )[2]
    # Beam 1 keeps only `0` after layer 1 (first in generation order), so the
    # routes through `nil` and `quote` never form: `0 i` is illegal and
    # nothing closes.
    assert inputs not in narrow
    wide = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Likes("i"),
        _Tally(),
        beam_width=64,
        segment_depth=2,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )[2]
    assert wide[inputs] == (1.0, ("nil", "i"))


def test_the_layer_is_scored_as_the_flat_beam_scores_it():
    """Each closed state's score is the policy's masked log-probability of its op."""
    task = task_for("add_pair", 6)
    inputs, outputs, targets = _prepare(task, "both")
    scorer = untrained()
    nil = step_all_quiet(inputs, OP_BY_NAME["nil"])[0]
    legal = np.array([[step_all_quiet(nil, op)[0] is not None for op in OPS]])
    block = item_array([encode_item(s, t) for s, t in zip(nil, targets, strict=True)])
    row = scorer.log_probs([block], legal)[0]
    _stop, _program, closed = _run_segment(
        [(nil, ("nil",), -0.5)],
        outputs,
        targets,
        scorer,
        _Tally(),
        beam_width=64,
        segment_depth=1,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert inputs in closed
    best = max(("dip", "i"), key=lambda name: row[OP_INDEX[name]])
    score, route = closed[inputs]
    assert route == ("nil", best)
    assert score == pytest.approx(-0.5 + row[OP_INDEX[best]])


def test_each_root_has_its_own_beam_and_visited_set():
    """Two roots at the same product both expand it; one root would dedup it."""
    task = task_for("add_pair", 6)
    inputs, outputs, targets = _prepare(task, "both")
    one = _Tally()
    _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Flat(),
        one,
        beam_width=4,
        segment_depth=2,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    two = _Tally()
    _run_segment(
        [(inputs, (), 0.0), (inputs, ("nil", "i"), -1.0)],
        outputs,
        targets,
        _Flat(),
        two,
        beam_width=4,
        segment_depth=2,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert (two.expansions, two.executions, two.fuel) == (
        2 * one.expansions,
        2 * one.executions,
        2 * one.fuel,
    )
    assert two.deepest == 4  # the second root's programs are two ops longer


def test_a_fit_stops_the_segment_only_when_asked():
    task = task_for("add_pair", 6)
    inputs, outputs, targets = _prepare(task, "both")
    stop, program, _closed = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Flat(),
        _Tally(),
        beam_width=64,
        segment_depth=3,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=True,
    )
    assert (stop, program) == ("fit", ("add",))
    tally = _Tally()
    stop, program, _closed = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Flat(),
        tally,
        beam_width=64,
        segment_depth=3,
        horizon=24,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert (stop, program) == (None, None)
    assert tally.deepest == 3


def test_the_budget_trips_before_an_attempt_and_the_tally_keeps_what_was_paid():
    task = task_for("add_pair", 6)
    inputs, outputs, targets = _prepare(task, "both")
    tally = _Tally(executions=5)
    stop, program, closed = _run_segment(
        [(inputs, (), 0.0)],
        outputs,
        targets,
        _Flat(),
        tally,
        beam_width=64,
        segment_depth=3,
        horizon=24,
        guard="both",
        budget=5,
        max_state_atoms=None,
        stop_at_fit=True,
    )
    assert (stop, program, closed) == ("budget", None, {})
    assert (tally.executions, tally.expansions) == (5, 1)


# ---------------------------------------------------------------------------
# The whole search
# ---------------------------------------------------------------------------


def test_segment_beam_finds_a_one_op_witness_under_any_policy():
    task = task_for("add_pair", 6)
    result = segment_beam_solve(task, untrained(), beam_width=64)
    assert (result.termination, result.program, result.depth) == ("fit", ("add",), 1)
    assert grade(result.program, task).solved
    assert result.expansions == 1
    assert 0 < result.executions <= len(OPS) * task.examples


def test_budget_is_checked_before_every_attempt():
    task = task_for("add_pair", 6)
    scorer = untrained()
    none = segment_beam_solve(task, scorer, budget=0)
    assert (none.program, none.termination, none.executions) == (None, "budget", 0)
    inputs, _ = products(task)
    expected = 0
    for op in OPS:
        expected += step_all_quiet(inputs, op)[2]
        if expected >= 1:
            break
    one = segment_beam_solve(task, scorer, budget=1)
    assert (one.termination, one.executions) == ("budget", expected)
    assert one.depth == 1  # one legal successor was produced before the trip


def test_one_segment_of_one_layer_pays_what_the_flat_beam_pays_at_depth_one():
    task = task_for("sum")
    scorer = untrained()
    flat = beam_solve(task, scorer, beam_width=8, max_depth=1)
    segmented = segment_beam_solve(
        task, scorer, beam_width=8, roots=4, segment_depth=1, max_segments=1
    )
    assert (segmented.executions, segmented.fuel, segmented.expansions) == (
        flat.executions,
        flat.fuel,
        flat.expansions,
    )
    assert segmented.program is None
    assert segmented.termination in ("max_segments", "frontier_exhausted")
    assert segmented.depth == 1


def test_segments_restart_from_closed_states():
    """`drop_last` is reverse, `uncons drop`, reverse: two segments of 5 and 7 ops.

    A policy fit to the witness carries beam 8 through both when every
    segment restarts from the closed state the first one reached, and the
    program reported is the whole route from the inputs.
    """
    task = task_for("drop_last")
    assert closed_cuts(replay(products(task)[0], task.program), task.program) == [
        5,
        12,
    ]
    rows = witness_rows([task, task_for("drop_last", 5)])
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
    scorer = PolicyScorer(model, "cpu")
    result = segment_beam_solve(
        task, scorer, beam_width=8, roots=4, segment_depth=7, max_segments=3
    )
    assert result.termination == "fit", result
    assert grade(result.program, task).solved
    assert result.depth == len(result.program)
    inputs, outputs = products(task)
    assert replay(inputs, result.program)[-1] == outputs
    # The route passes through at least one closed state, since no seven-op
    # program takes these inputs to their outputs within one segment.
    states = replay(inputs, result.program)
    assert closed_cuts(states, result.program)
    assert len(result.program) > 7


def test_frontier_exhausted_when_nothing_closes():
    """A flat scorer with beam 1 walks `0 0 0 ...` and never runs a combinator."""
    task = task_for("add_pair", 6)
    marked = dataclasses.replace(task, outputs=("[ 5 5 5 ]",) * task.examples)
    result = segment_beam_solve(
        marked, _Flat(), beam_width=1, roots=2, segment_depth=3, max_segments=4
    )
    assert (result.program, result.termination) == (None, "frontier_exhausted")
    assert result.expansions == 3
    assert result.depth == 3


def test_max_segments_is_reached_when_closed_states_keep_coming():
    """From a list, `i` spills it and `quote dip` reorders, so every segment closes something new."""
    task = task_for("sum")
    unreachable = dataclasses.replace(task, outputs=("[ (9 9 9) ]",) * task.examples)
    result = segment_beam_solve(
        unreachable, _Flat(), beam_width=64, roots=2, segment_depth=2, max_segments=3
    )
    assert (result.program, result.termination) == (None, "max_segments")
    assert 3 <= result.depth <= 6
    # Fewer segments spend less, and no closed state is expanded twice.
    shorter = segment_beam_solve(
        unreachable, _Flat(), beam_width=64, roots=2, segment_depth=2, max_segments=1
    )
    assert shorter.termination == "max_segments"
    assert shorter.executions < result.executions
    # Three segments of two layers expand at most 1 + 2 * 64 + 2 * (2 + 64) nodes.
    assert result.expansions <= 1 + 64 + 2 * (2 + 2 * 64)


def test_the_cyclic_collector_is_restored():
    import gc

    task = task_for("add_pair", 6)
    assert gc.isenabled()
    segment_beam_solve(task, _Flat(), beam_width=4, segment_depth=2, max_segments=1)
    assert gc.isenabled()
    gc.disable()
    try:
        segment_beam_solve(task, _Flat(), beam_width=4, segment_depth=2, max_segments=1)
        assert not gc.isenabled()
    finally:
        gc.enable()


# ---------------------------------------------------------------------------
# The witness diagnostic
# ---------------------------------------------------------------------------

ROW_KEYS = {
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


def test_witness_ranks_follow_the_cuts_and_rank_a_witness_the_policy_knows():
    task = task_for("drop_last")
    rows = witness_rows([task, task_for("drop_last", 5)])
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
    scorer = PolicyScorer(model, "cpu")
    report = segment_witness_ranks(
        task, scorer, beam_width=8, roots=4, segment_depth=7, max_segments=3
    )
    assert [set(row) for row in report] == [ROW_KEYS, ROW_KEYS]
    assert [row["segment"] for row in report] == [0, 1]
    assert [row["ops"] for row in report] == [5, 7]
    assert report[0]["roots"] == 1 and report[0]["forced"] is False
    for row in report:
        assert row["complete"] is True
        assert row["rank"] is not None and 1 <= row["rank"] <= row["closed"]
        assert row["score"] <= row["best"]
        if row["rank"] == 1:
            assert row["score"] == row["best"]
    assert report[1]["executions"] > report[0]["executions"]
    assert report[1]["forced"] is False  # the search reached the reversed list
    assert report[1]["roots"] <= 4


def test_witness_ranks_force_the_witness_root_when_the_search_misses_it():
    """Beam 1 under a flat scorer closes nothing, so every later root is forced."""
    task = task_for("drop_last")
    report = segment_witness_ranks(
        task, _Flat(), beam_width=1, roots=4, segment_depth=7, max_segments=12
    )
    assert len(report) == 2
    # The one closed state is `i` spilling the input list; it is not the
    # witness's reversed list, so the witness root is forced in segment 1.
    assert report[0] == {
        "segment": 0,
        "ops": 5,
        "roots": 1,
        "forced": False,
        "closed": 1,
        "rank": None,
        "score": None,
        "best": 0.0,
        "executions": report[0]["executions"],
        "complete": True,
    }
    assert report[1]["forced"] is True
    assert report[1]["roots"] == 2  # the spilled list and the forced root
    assert report[1]["rank"] is None
    assert report[1]["executions"] > report[0]["executions"] > 0


def test_max_segments_caps_the_rows_and_a_tail_after_the_last_cut_is_no_segment():
    task = task_for("drop_last")
    assert segment_witness_ranks(task, _Flat(), beam_width=1, max_segments=1) == [
        segment_witness_ranks(task, _Flat(), beam_width=1, max_segments=12)[0]
    ]
    assert segment_witness_ranks(task_for("add_pair", 6), _Flat()) == []
    # `fold dup drop`: the ops after the last cut belong to no segment.
    task = task_for("sum")
    report = segment_witness_ranks(
        task, _Flat(), program=(*task.program, "dup", "drop"), beam_width=1
    )
    assert [row["ops"] for row in report] == [5]


def test_witness_ranks_refuse_a_program_the_machine_refuses():
    task = task_for("sum")
    with pytest.raises(ValueError, match=r"^sum: the program refuses step 0 \(add\)$"):
        segment_witness_ranks(task, _Flat(), program=("add",))
    with pytest.raises(ValueError, match=r"refuses step 1 \(i\)"):
        segment_witness_ranks(task, _Flat(), program=("0", "i", "[", "]", "i"))


def test_a_budget_marks_the_row_incomplete_and_stops_the_rows():
    task = task_for("drop_last")
    report = segment_witness_ranks(task, _Flat(), beam_width=4, budget=10)
    assert len(report) == 1
    assert report[0]["complete"] is False
    assert report[0]["executions"] >= 10


def test_a_forced_root_carries_the_witness_score_plus_the_policy_over_the_segment():
    """Under a scorer that pays 2.0 per `fold`, beam 1 walks `0 0 0 ...` and misses
    every witness state, so each later root is forced. The program is `drop_last`
    plus a final `i`, three segments: the forced root of segment 2 carries the
    two folds walked so far, 4.0, and its one-op segment `i` closes at 4.0."""
    task = task_for("drop_last")
    program = (*task.program, "i")
    inputs, _, _targets = _prepare(task, "both")
    states = replay(inputs, program)
    assert closed_cuts(states, program) == [5, 12, 13]
    report = segment_witness_ranks(
        task, _Likes("fold", bonus=2.0), program, beam_width=1, roots=4, segment_depth=7
    )
    assert [row["ops"] for row in report] == [5, 7, 1]
    assert [row["forced"] for row in report] == [False, True, True]
    # Segment 0 closes only `i` on the input list (score 0); the witness's
    # reversed list is not among them.
    assert report[0]["rank"] is None and report[0]["best"] == pytest.approx(0.0)
    assert report[1]["roots"] == 2 and report[1]["rank"] is None
    assert report[2]["roots"] == 2
    assert (report[2]["rank"], report[2]["closed"]) == (1, 1)
    assert report[2]["score"] == pytest.approx(4.0)
    assert report[2]["best"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# `System`
# ---------------------------------------------------------------------------


def system(**overrides) -> solve.System:
    fields = {
        "name": "tiny",
        "repo_id": "nobody/nothing",
        "revision": "0" * 40,
        "beam_width": 4,
        "max_depth": 6,
        "guard": "both",
        "library": True,
        "measured": "a test",
    }
    return solve.System(**{**fields, **overrides})


def test_best_is_the_flat_search_and_describes_itself_as_before():
    assert solve.SEARCHES == ("flat", "segment")
    assert solve.BEST.search == "flat"
    assert (solve.BEST.roots, solve.BEST.segment_depth, solve.BEST.max_segments) == (
        16,
        12,
        12,
    )
    assert "search" not in solve.BEST.describe()
    assert system().describe() == system(search="flat").describe()


def test_a_segment_system_says_so_and_an_unknown_search_is_refused():
    described = system(search="segment", roots=3, segment_depth=5, max_segments=2)
    assert described.describe() == (
        system().describe()
        + ", search segment: 3 roots, 2 segments of up to 5 ops, the beam per root"
    )
    with pytest.raises(
        ValueError, match=r"unknown search 'bogus'; choose one of \('flat', 'segment'\)"
    ):
        system(search="bogus")


def test_solve_tasks_runs_the_segment_search_and_still_replays_the_library():
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7)]
    scorer = untrained()
    rows = solve.solve_tasks(tasks, scorer, system(search="segment", max_depth=0))
    assert [row.source for row in rows] == ["search", "library"]
    assert [row.termination for row in rows] == ["fit", "fit"]
    assert all(row.solved for row in rows)
    assert rows[0].program == ("add",) and rows[0].depth == 1
    # `max_depth` is not read by the segment search: depth 0 would have
    # stopped the flat beam before its first layer.
    flat = solve.solve_tasks(tasks[:1], scorer, system(search="flat", max_depth=0))
    assert flat[0].program is None and flat[0].termination == "max_depth"
