"""Segment boundaries, independent beams, and witness diagnostics."""

import gc
from dataclasses import replace

import numpy as np
import pytest

from wandering_light import solve, stack_beam
from wandering_light.stack_beam import (
    OP_INDEX,
    _run_segment,
    _Tally,
    closed_cuts,
    segment_beam_solve,
    segment_witness_ranks,
    step_all_quiet,
)
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS, Task
from wandering_light.stack_policy import encode_targets


class FixedScores:
    def __init__(self, **scores):
        self.scores = np.full(len(OPS), -1.0, dtype=np.float32)
        for name, score in scores.items():
            self.scores[OP_INDEX[name]] = score
        self.calls = []

    def log_probs(self, blocks, legal):
        assert len(blocks) == len(legal)
        self.calls.append(legal.copy())
        return np.where(legal, self.scores, -np.inf)


def product(*stacks):
    return tuple(parse_stack(stack, basis=BASIS) for stack in stacks)


def make_task(
    inputs=("( ( 89 ) )", "( ( 97 ) )"), outputs=("89", "97"), program=("i", "i")
):
    return Task(
        behaviour="test_segments",
        summary="",
        split="eval",
        task_hash_split="eval",
        inputs=inputs,
        outputs=outputs,
        held_out_inputs=inputs,
        held_out_outputs=outputs,
        program=program,
        length=len(program),
        examples=len(inputs),
        held_out=len(inputs),
        fuel=0,
        tags=(),
    )


def layer_cost(state):
    costs = [step_all_quiet(state, op)[1:] for op in OPS]
    fuel, executions = (sum(values) for values in zip(*costs, strict=True))
    return executions, fuel


def test_cuts_follow_top_level_applications_and_check_every_stack():
    program = ("[", "[", "i", "]", "drop", "]", "i", "i", "1")
    states = [product("( 89 )")]
    for name in program:
        moved, _, _ = step_all_quiet(states[-1], OP_BY_NAME[name])
        assert moved is not None
        states.append(moved)
    assert closed_cuts(states, program) == [7, 8]
    assert closed_cuts([product("( 89 )", "MARK")], ("i",)) == []
    assert {op.name: index for index, op in enumerate(OPS)} == OP_INDEX


def test_separate_root_beams_keep_and_merge_all_closed_successors(monkeypatch):
    root_a, root_b, shared, discarded, tied, entry_a, entry_b, merged, final = (
        product(str(i)) for i in range(10, 19)
    )
    moves = {
        root_a: {"0": shared, "1": discarded, "dip": tied, "i": entry_a, "map": merged},
        root_b: {"0": shared, "1": discarded, "dip": tied, "i": entry_b, "map": merged},
        shared: {"dip": merged, "i": final},
    }
    expanded = []

    def step(state, op):
        if op is OPS[0]:
            expanded.append(state)
        successor = moves[state].get(op.name)
        return (successor, 2, 1) if successor is not None else (None, 0, 0)

    monkeypatch.setattr(stack_beam, "step_all_quiet", step)
    scorer = FixedScores(dip=-2, i=-3, map=-8)
    tally = _Tally()
    outputs = product("999")
    stop, program, closed = _run_segment(
        [(root_a, ("dup",), 0.0), (root_b, ("swap",), 0.0)],
        outputs,
        encode_targets(outputs),
        scorer,
        tally,
        beam_width=1,
        segment_depth=2,
        horizon=4,
        guard="off",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=True,
    )
    assert (stop, program) == (None, None)
    assert expanded == [root_a, root_b, shared, shared]
    assert len(scorer.calls) == 4
    assert list(closed.items()) == [
        (tied, (-2.0, ("dup", "dip"))),
        (entry_a, (-3.0, ("dup", "i"))),
        (merged, (-3.0, ("dup", "0", "dip"))),
        (entry_b, (-3.0, ("swap", "i"))),
        (final, (-4.0, ("dup", "0", "i"))),
    ]
    assert tally == _Tally(executions=14, fuel=28, expansions=4, deepest=3)


def test_search_resets_and_can_fit_on_a_final_open_successor():
    task = make_task(outputs=("( 89 ) 1", "( 97 ) 1"), program=("i", "1"))
    scorer = FixedScores(i=-100)
    result = segment_beam_solve(
        task, scorer, beam_width=1, roots=1, segment_depth=1, max_segments=2
    )
    assert (result.termination, result.program, result.depth) == (
        "fit",
        task.program,
        2,
    )
    assert result.expansions == 2
    assert len(scorer.calls) == 1


def test_search_limits_and_cumulative_costs():
    task = make_task()
    executions, fuel = layer_cost(product(*task.inputs))
    options = {"beam_width": 1, "roots": 1, "segment_depth": 1}
    limited = segment_beam_solve(task, FixedScores(), max_segments=1, **options)
    assert (limited.termination, limited.program, limited.depth) == (
        "max_segments",
        None,
        1,
    )
    assert (limited.executions, limited.fuel, limited.expansions) == (
        executions,
        fuel,
        1,
    )

    exhausted = segment_beam_solve(task, FixedScores(), max_state_atoms=3, **options)
    assert (exhausted.termination, exhausted.depth) == ("frontier_exhausted", 1)

    stopped = segment_beam_solve(task, FixedScores(), budget=executions + 1, **options)
    assert (stopped.termination, stopped.program, stopped.depth) == ("budget", None, 2)
    assert (stopped.executions, stopped.fuel, stopped.expansions) == (
        executions + 2,
        fuel + 2,
        2,
    )

    zero = segment_beam_solve(task, FixedScores(), budget=0, **options)
    assert (zero.termination, zero.executions, zero.fuel, zero.depth) == (
        "budget",
        0,
        0,
        0,
    )


def test_close_guard_counts_remaining_ops_from_the_whole_program(monkeypatch):
    initial = product("MARK")
    expanded = []

    def step(state, op):
        if op is OPS[0]:
            expanded.append(state)
        return step_all_quiet(state, op)

    monkeypatch.setattr(stack_beam, "step_all_quiet", step)
    outputs = product("999")
    _run_segment(
        [(initial, ("[",), 0.0)],
        outputs,
        encode_targets(outputs),
        FixedScores(**{"[": -0.1, "0": -0.2}),
        _Tally(),
        beam_width=1,
        segment_depth=2,
        horizon=3,
        guard="close",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert expanded == [initial, product("MARK 0")]


def test_close_guard_and_collector_restoration():
    task = make_task(outputs=("( ( 89 ) ) MARK", "( ( 97 ) ) MARK"), program=("[",))
    collecting = gc.isenabled()
    with pytest.raises(ValueError, match="the close guard needs mark-free targets"):
        segment_beam_solve(task, FixedScores())
    assert gc.isenabled() is collecting
    gc.disable()
    try:
        fit = segment_beam_solve(task, FixedScores(), guard="off", segment_depth=1)
        assert fit.program == ("[",)
        assert not gc.isenabled()
    finally:
        if collecting:
            gc.enable()


def test_witness_ranks_include_final_fit_and_report_partial_budget():
    task = make_task()
    first_cost, _ = layer_cost(product(*task.inputs))
    second_cost, _ = layer_cost(product("( 89 )", "( 97 )"))
    options = {"beam_width": 1, "roots": 1, "segment_depth": 1}
    rows = segment_witness_ranks(task, FixedScores(), **options)
    assert rows == [
        {
            "segment": 1,
            "ops": 1,
            "roots": 1,
            "forced": False,
            "closed": 1,
            "rank": 1,
            "score": -1.0,
            "best": -1.0,
            "executions": first_cost,
            "complete": True,
        },
        {
            "segment": 2,
            "ops": 1,
            "roots": 1,
            "forced": False,
            "closed": 1,
            "rank": 1,
            "score": -2.0,
            "best": -2.0,
            "executions": first_cost + second_cost,
            "complete": True,
        },
    ]
    limited = segment_witness_ranks(task, FixedScores(), max_segments=1, **options)
    assert limited == rows[:1]
    stopped = segment_witness_ranks(
        task, FixedScores(), budget=first_cost + 1, **options
    )
    assert stopped == [
        rows[0],
        {
            "segment": 2,
            "ops": 1,
            "roots": 1,
            "forced": False,
            "closed": 0,
            "rank": None,
            "score": None,
            "best": None,
            "executions": first_cost + 2,
            "complete": False,
        },
    ]


def test_missing_witness_is_forced_with_its_policy_score_and_tail_is_ignored():
    task = make_task(
        inputs=("( ( 89 ) ( 99 ) )",),
        outputs=("( ( 89 ) ( 99 ) ) ( 89 ) 99 1",),
        program=("dup", "i", "i", "1"),
    )
    scorer = FixedScores(dup=-4, i=-2)
    rows = segment_witness_ranks(task, scorer, roots=1, segment_depth=1)
    assert [
        (row["segment"], row["ops"], row["roots"], row["forced"]) for row in rows
    ] == [
        (1, 2, 1, False),
        (2, 1, 2, True),
    ]
    assert (rows[0]["rank"], rows[0]["score"], rows[0]["best"]) == (None, None, -2.0)
    assert rows[1]["rank"] > 1
    assert rows[1]["score"] == -8.0


def test_ranked_witness_outside_the_frontier_is_forced_with_its_ranked_score():
    task = make_task(
        inputs=("( 89 ) ( 99 )",), outputs=("99 89",), program=("dip", "i")
    )
    rows = segment_witness_ranks(task, FixedScores(dip=-4), roots=1, segment_depth=1)
    assert (rows[0]["rank"], rows[0]["score"], rows[0]["best"]) == (2, -4.0, -1.0)
    assert (rows[1]["forced"], rows[1]["roots"], rows[1]["score"]) == (True, 2, -5.0)


def test_witness_override_and_refused_step():
    task = make_task()
    assert segment_witness_ranks(task, FixedScores(), ()) == []
    with pytest.raises(
        ValueError, match=r"test_segments: the program refuses step 1 \(add\)"
    ):
        segment_witness_ranks(task, FixedScores(), ("i", "add"))


def test_system_selects_segment_search_even_with_no_flat_depth(monkeypatch):
    monkeypatch.setattr(solve.gc, "freeze", lambda: None)
    assert solve.BEST.search == "flat"
    with pytest.raises(
        ValueError, match=r"unknown search 'bad'; choose one of \('flat', 'segment'\)"
    ):
        replace(solve.BEST, search="bad")
    system = replace(
        solve.BEST,
        search="segment",
        max_depth=0,
        roots=1,
        max_segments=2,
        segment_depth=1,
    )
    assert system.describe().endswith(
        ", search segment: 1 roots, 2 segments of up to 1 ops, the beam per root"
    )
    task = make_task()
    [row] = solve.solve_tasks([task], FixedScores(), system)
    assert (row.termination, row.depth, row.program, row.solved) == (
        "fit",
        2,
        task.program,
        True,
    )
