"""Closed states start independent beams, with a shared budget across resets."""

import gc
import math
from dataclasses import replace

import numpy as np
import pytest

from wandering_light import solve, stack_beam
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
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, OPS, Task
from wandering_light.stack_policy import encode_targets


class Uniform:
    """Uniform over legal ops, so tied paths expose generation and merge order."""

    def __init__(self):
        self.batch_sizes = []

    def log_probs(self, blocks, legal):
        assert legal.shape == (len(blocks), len(OPS))
        self.batch_sizes.append(len(blocks))
        return np.where(legal, -np.log(legal.sum(axis=1, keepdims=True)), -np.inf)


def task_for(given="3 ((1 add))", wanted="4", names=("i", "i")):
    return Task(
        behaviour="test",
        summary="",
        split="eval",
        task_hash_split="eval",
        inputs=(given,),
        outputs=(wanted,),
        held_out_inputs=(given,),
        held_out_outputs=(wanted,),
        program=names,
        length=len(names),
        examples=1,
        held_out=1,
        fuel=0,
        tags=(),
    )


def product(*texts):
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def attempts(state):
    return [step_all_quiet(state, op) for op in OPS]


def test_closed_cuts_check_the_parent_and_every_example():
    clear = product("nil", "nil")
    marked = product("nil", "MARK")
    program = ("i", "[", "map", "]", "dip", "1")
    states = [clear, clear, marked, marked, clear, clear, clear]
    assert closed_cuts(states, program) == [1, 5]
    assert [OP_INDEX[op.name] for op in OPS] == list(range(len(OPS)))


def test_prepare_parses_both_products_and_checks_all_targets():
    task = replace(task_for(), inputs=("nil", "1"), outputs=("2", "MARK"))
    for guard in ("both", "close"):
        with pytest.raises(
            ValueError, match=r"^the close guard needs mark-free targets$"
        ):
            _prepare(task, guard)
    inputs, outputs, targets = _prepare(task, "body")
    assert inputs == product("nil", "1")
    assert outputs == product("2", "MARK")
    assert targets == encode_targets(outputs)


@pytest.mark.parametrize(("later_score", "winner"), [(-1.0, ()), (0.0, ("i",))])
def test_each_root_has_its_own_beam_and_visited_set(later_score, winner):
    # Two routes to the same root. Both must expand the same two open products
    # on layer two. A closed state reached by several ops keeps its first tie.
    state = product("3 nil")
    roots = [(state, (), -1.0), (state, ("i",), later_score)]
    scorer = Uniform()
    tally = _Tally()
    stop, program, closed = _run_segment(
        roots,
        product("42"),
        encode_targets(product("42")),
        scorer,
        tally,
        beam_width=2,
        segment_depth=2,
        horizon=5,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert (stop, program) == (None, None)
    assert scorer.batch_sizes == [1, 1, 2, 2]
    assert tally.expansions == 6
    assert tally.deepest == 3
    score, path = closed[product("3")]
    assert path == (*winner, "dip")
    assert score == pytest.approx(
        max(-1.0, later_score)
        - math.log(sum(moved is not None for moved, _, _ in attempts(state)))
    )


def test_guard_uses_the_remaining_whole_program_horizon(monkeypatch):
    inputs, outputs, targets = _prepare(task_for(), "both")
    seen = []

    def block(stack, remaining, mode):
        seen.append((remaining, mode))
        return "close" if stack == product("3 (1 add)")[0] else None

    monkeypatch.setattr(stack_beam, "blocked", block)
    _, _, closed = _run_segment(
        [(inputs, ("dup", "drop"), 0.0)],
        outputs,
        targets,
        Uniform(),
        _Tally(),
        beam_width=1,
        segment_depth=1,
        horizon=7,
        guard="both",
        budget=None,
        max_state_atoms=None,
        stop_at_fit=False,
    )
    assert set(seen) == {(4, "both")}
    assert list(closed) == [product("(1 add) 3")]


def test_reset_keeps_low_ranked_closed_paths_and_charges_all_attempts():
    task = task_for()
    state = product(*task.inputs)
    first = attempts(state)
    settings = {"segment_depth": 1, "beam_width": 1, "roots": 2}
    one = segment_beam_solve(task, Uniform(), max_segments=1, **settings)
    assert (one.termination, one.depth, one.expansions) == ("max_segments", 1, 1)
    assert one.executions == sum(stepped for _, _, stepped in first)
    assert one.fuel == sum(spent for _, spent, _ in first)
    solved = segment_beam_solve(task, Uniform(), max_segments=2, **settings)
    assert (solved.termination, solved.program, solved.depth) == ("fit", ("i", "i"), 2)
    assert solved.expansions == 3
    # Spend one execution at the first root of segment two.
    limited = segment_beam_solve(task, Uniform(), budget=one.executions + 1, **settings)
    assert (limited.termination, limited.program) == ("budget", None)
    assert (limited.executions, limited.depth, limited.expansions) == (
        one.executions + 1,
        2,
        2,
    )
    capped = segment_beam_solve(task, Uniform(), max_state_atoms=1, **settings)
    assert (capped.termination, capped.depth) == ("frontier_exhausted", 1)


def test_budget_and_fit_precede_scoring_even_for_open_oversize_outputs():
    task = replace(task_for(), inputs=("1", "2"), outputs=("1 0", "2 0"))
    scorer = Uniform()
    for budget, depth, executions in ((0, 0, 0), (1, 1, 2)):
        miss = segment_beam_solve(
            replace(task, outputs=("4", "5")), scorer, budget=budget
        )
        assert (miss.termination, miss.depth, miss.executions) == (
            "budget",
            depth,
            executions,
        )
    hit = segment_beam_solve(task, scorer, budget=1, max_state_atoms=0)
    assert (hit.program, hit.termination, hit.executions) == (("0",), "fit", 2)
    assert not scorer.batch_sizes


def test_captured_combinator_does_not_seed_the_next_segment():
    task = task_for("MARK", "42")
    result = segment_beam_solve(task, Uniform(), segment_depth=1, guard="off")
    assert (result.termination, result.depth, result.expansions) == (
        "frontier_exhausted",
        1,
        1,
    )


def test_highest_scoring_closed_successor_leaves_the_open_beam_slot_free():
    class PreferCombinator:
        def log_probs(self, blocks, legal):
            weights = np.ones(len(OPS))
            weights[OP_INDEX["i"]] = 100
            weights[OP_INDEX["nil"]] = 50
            masked = np.where(legal, weights, 0)
            return np.where(
                legal, np.log(weights / masked.sum(axis=1, keepdims=True)), -np.inf
            )

    task = task_for(wanted="3 ((1 add)) nil 0")
    settings = {"segment_depth": 2, "beam_width": 1, "max_segments": 1}
    hit = segment_beam_solve(task, PreferCombinator(), **settings)
    assert hit.program == ("nil", "0")
    # The best closed path ('i', 'i') must wait for another segment.
    miss = segment_beam_solve(task_for(), PreferCombinator(), **settings)
    assert miss.termination == "max_segments"


def test_witness_uses_ranked_path_outside_root_cutoff_and_skips_the_tail():
    task = task_for(names=("dup", "drop", "i", "i", "1"))
    rows = segment_witness_ranks(task, Uniform(), roots=1, segment_depth=1)
    first, second = rows
    first_attempts = attempts(product(*task.inputs))
    score = -math.log(sum(m is not None for m, _, _ in first_attempts))
    assert first == {
        "segment": 1,
        "ops": 3,
        "roots": 1,
        "forced": False,
        "closed": 2,
        "rank": 2,
        "score": pytest.approx(score),
        "best": pytest.approx(score),
        "executions": sum(n for _, _, n in first_attempts),
        "complete": True,
    }
    assert second["forced"] and second["roots"] == 2 and second["ops"] == 1
    assert second["segment"] == 2 and second["rank"] == 1 and second["complete"]
    # The first 'i' alone reached the intermediate state at a better score.
    # Its continuation inherits this ranked score, even though we forced its root.
    legal = sum(m is not None for m, _, _ in attempts(product("3 (1 add)")))
    assert second["score"] == pytest.approx(score - math.log(legal))
    assert second["executions"] > first["executions"]
    assert segment_witness_ranks(task, Uniform(), program=("dup", "drop")) == []


def test_missing_witness_root_inherits_its_own_masked_log_probability():
    task = task_for()
    witness = ("1", "swap", "i", "i")
    first, second = segment_witness_ranks(
        task, Uniform(), witness, roots=1, segment_depth=1
    )
    assert first["rank"] is None and first["score"] is None
    assert second["forced"] and second["roots"] == 2 and second["rank"] is not None
    states = ["3 ((1 add))", "3 ((1 add)) 1", "3 1 ((1 add))", "3 1 (1 add)"]
    expected = -sum(
        math.log(sum(m is not None for m, _, _ in attempts(product(s)))) for s in states
    )
    assert second["score"] == pytest.approx(expected)


def test_witness_excludes_expanded_roots_and_reports_partial_budget():
    task = task_for(names=("nil", "i", "i", "i"))
    rows = segment_witness_ranks(task, Uniform(), segment_depth=2, max_segments=1)
    assert len(rows) == 1 and rows[0]["rank"] is None  # nil i returns the inputs

    task = task_for()
    cost = sum(n for _, _, n in attempts(product(*task.inputs)))
    rows = segment_witness_ranks(
        task, Uniform(), roots=1, segment_depth=1, budget=cost + 1
    )
    assert [row["complete"] for row in rows] == [True, False]
    assert rows[-1]["executions"] == cost + 1
    assert rows[-1]["closed"] == 0 and rows[-1]["best"] is None
    # With a second layer in this segment, the scored first layer survives
    # interruption and the witness still has a rank in the partial results.
    (partial,) = segment_witness_ranks(
        task, Uniform(), segment_depth=2, budget=cost + 1
    )
    assert not partial["complete"] and partial["closed"] == 2 and partial["rank"] == 2

    with pytest.raises(ValueError, match=r"^test: the program refuses step 2 \(map\)$"):
        segment_witness_ranks(task, Uniform(), ("i", "i", "map"), max_segments=1)


def test_system_dispatch_preserves_library_and_ignores_flat_depth():
    assert solve.BEST.search == "flat"
    assert ", search" not in solve.BEST.describe()
    with pytest.raises(ValueError) as caught:
        replace(solve.BEST, search="unknown")
    assert (
        str(caught.value)
        == "unknown search 'unknown'; choose one of ('flat', 'segment')"
    )
    system = replace(
        solve.BEST,
        search="segment",
        segment_depth=1,
        max_segments=2,
        max_depth=0,
        beam_width=1,
        roots=2,
    )
    assert system.describe().endswith(
        ", search segment: 2 roots, 2 segments of up to 1 ops, the beam per root"
    )
    first, second = solve.solve_tasks([task_for(), task_for()], Uniform(), system)
    assert first.solved and first.program == ("i", "i") and first.source == "search"
    assert (
        second.solved and second.program == first.program and second.source == "library"
    )


@pytest.mark.parametrize("collecting", [True, False])
def test_segment_restores_collector_if_scoring_raises(collecting):
    class Fail:
        def log_probs(self, blocks, legal):
            assert not gc.isenabled()
            raise RuntimeError("scorer failed")

    original = gc.isenabled()
    try:
        (gc.enable if collecting else gc.disable)()
        with pytest.raises(RuntimeError, match="scorer failed"):
            segment_beam_solve(task_for(), Fail())
        assert gc.isenabled() == collecting
    finally:
        (gc.enable if original else gc.disable)()
