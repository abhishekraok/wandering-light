"""The segment-reset beam and its K diagnostic (short-segment-campaign, #168)."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.staged_transfer import RUNNERS, step_kinds
from wandering_light import solve
from wandering_light.quotation_guard import RUNS_CODE
from wandering_light.stack_beam import (
    OP_INDEX,
    _prepare,
    _run_segment,
    _Tally,
    beam_solve,
    block_encoder,
    closed_cuts,
    segment_beam_solve,
    segment_witness_ranks,
    step_all_quiet,
)
from wandering_light.stack_pbe import (
    BY_NAME,
    CATALOGUE,
    OP_BY_NAME,
    build_task,
    draw_task,
)
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import DIFFERENCE, MACHINE, SEP, node_encoder


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def states_along(task, program):
    inputs, _outputs = _prepare(task, "both")
    states = [inputs]
    for name in program:
        states.append(step_all_quiet(states[-1], OP_BY_NAME[name])[0])
    return states


class WitnessScorer:
    """``bonus`` on the witness's next op at each witness state; uniform elsewhere.

    A state is recognised by its encoded block, which is what a real scorer
    sees, so the stub cannot read anything the policy could not.
    """

    def __init__(self, task, bonus: float = 5.0, encoding=MACHINE):
        _inputs, outputs = _prepare(task, "both")
        encode = node_encoder(outputs, encoding)
        self.bonus = bonus
        self.want = {}
        states = states_along(task, task.program)[:-1]
        for state, name in zip(states, task.program, strict=True):
            block = encode(state)
            self.want[block.shape, block.tobytes()] = OP_INDEX[name]

    def log_probs(self, blocks, legal):
        logits = np.zeros(legal.shape)
        for row, block in enumerate(blocks):
            want = self.want.get((block.shape, block.tobytes()))
            if want is not None:
                logits[row, want] = self.bonus
        logits = np.where(legal, logits, -np.inf)
        peak = logits.max(axis=1, keepdims=True)
        shifted = logits - np.where(np.isfinite(peak), peak, 0.0)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


# `[ 0 swap lt? ] filter 0 [ add ] fold`: two top-level combinator applications,
# closed after the filter (6 ops) and after the fold (11).
TWO = "sum_positive"
SMALL = {"beam_width": 2, "roots": 2, "segment_depth": 6, "max_segments": 3}


@pytest.mark.parametrize(
    "search", [beam_solve, segment_beam_solve, segment_witness_ranks]
)
def test_search_uses_one_difference_encoder_for_every_scored_node(search):
    task = task_for(TWO)
    _, outputs = _prepare(task, "both")

    class DifferenceScorer(WitnessScorer):
        preparations = 0

        def encoder(self, targets):
            assert targets == outputs
            self.preparations += 1
            return node_encoder(targets, DIFFERENCE)

        def log_probs(self, blocks, legal):
            for block in blocks:
                assert np.all((block == SEP).sum(axis=1) == 2)
            return super().log_probs(blocks, legal)

    scorer = DifferenceScorer(task, encoding=DIFFERENCE)
    if search is beam_solve:
        assert (
            search(task, scorer, beam_width=2, max_depth=task.length).program
            == task.program
        )
    elif search is segment_beam_solve:
        assert search(task, scorer, **SMALL).program == task.program
    else:
        assert [row["rank"] for row in search(task, scorer, **SMALL)] == [1, 1]
    assert scorer.preparations == 1


def test_the_cut_is_step_kinds_run_label():
    assert frozenset(RUNNERS) == RUNS_CODE
    checked = 0
    for behaviour in CATALOGUE:
        try:
            task = build_task(behaviour, draw_task(behaviour, random.Random(0), 3, 4))
        except Exception:  # a sampler that cannot draw three distinct examples
            continue
        runs = [step + 1 for step, k in enumerate(step_kinds(task)) if k == "run"]
        assert closed_cuts(states_along(task, task.program), task.program) == runs
        checked += 1
    assert checked > 150


def test_a_two_segment_witness_is_found_across_its_closed_state():
    task = task_for(TWO)
    assert closed_cuts(states_along(task, task.program), task.program) == [6, 11]
    result = segment_beam_solve(task, WitnessScorer(task), **SMALL)
    assert result.termination == "fit"
    # The fit is the whole program from the inputs, not the last segment's ops.
    assert result.program == task.program
    assert result.depth == len(task.program) == 11
    assert grade(result.program, task).solved
    # One root in the first segment, two in the second, each a beam of <= 2.
    assert 0 < result.expansions <= 6 * 1 * 2 + 6 * 2 * 2


def test_a_segment_too_short_for_the_witness_cannot_close_it():
    task = task_for(TWO)
    short = segment_beam_solve(
        task, WitnessScorer(task), **{**SMALL, "segment_depth": 5, "max_segments": 1}
    )
    assert short.program is None
    assert short.termination in ("max_segments", "frontier_exhausted")
    assert short.depth <= 5


def test_merge_keeps_the_best_score_whatever_the_root_order():
    """Two roots on one product close the same states; the better root's paths win."""
    task = task_for(TWO)
    inputs, outputs = _prepare(task, "both")
    good = (inputs, (), 0.0)
    worse = (inputs, ("dup", "drop"), -3.0)
    knobs = {
        "beam_width": 4,
        "segment_depth": 6,
        "horizon": 18,
        "guard": "both",
        "budget": None,
        "max_state_atoms": None,
        "stop_at_fit": False,
    }
    scorer = WitnessScorer(task)
    encode = block_encoder(scorer, outputs)
    alone = _run_segment([good], outputs, encode, scorer, _Tally(), **knobs)[2]
    for roots in ([good, worse], [worse, good]):
        stop, _program, closed = _run_segment(
            roots, outputs, encode, scorer, _Tally(), **knobs
        )
        assert stop is None
        assert closed == alone
        assert closed and all(
            program[:2] != ("dup", "drop") for _, program in closed.values()
        )


def test_budget_caps_executions_over_the_whole_search():
    task = task_for(TWO)
    scorer = WitnessScorer(task)
    none = segment_beam_solve(task, scorer, **SMALL, budget=0)
    assert (none.program, none.termination, none.executions) == (None, "budget", 0)
    first = segment_beam_solve(task, scorer, **{**SMALL, "max_segments": 1})
    assert first.termination == "max_segments"
    full = segment_beam_solve(task, scorer, **SMALL)
    assert full.termination == "fit" and full.executions > first.executions + 10
    # A budget the first segment fits inside trips in the second.
    budget = first.executions + 10
    cut = segment_beam_solve(task, scorer, **SMALL, budget=budget)
    assert cut.program is None and cut.termination == "budget"
    assert budget <= cut.executions < budget + task.examples


def test_a_search_with_nothing_to_close_is_frontier_exhausted():
    """Only `i` can close in one op from a list, and the size cap refuses what it makes."""
    task = task_for(TWO)
    result = segment_beam_solve(
        task, WitnessScorer(task), **{**SMALL, "segment_depth": 1}, max_state_atoms=1
    )
    assert result.program is None
    assert result.termination == "frontier_exhausted"


def test_witness_ranks_one_row_per_segment_and_rank_one_when_preferred():
    task = task_for(TWO)
    rows = segment_witness_ranks(task, WitnessScorer(task), **SMALL)
    assert [row["segment"] for row in rows] == [0, 1]
    assert [row["ops"] for row in rows] == [6, 5]
    assert [row["rank"] for row in rows] == [1, 1]
    assert not any(row["forced"] for row in rows)
    assert all(row["closed"] >= 1 and row["complete"] for row in rows)
    assert all(row["score"] == row["best"] for row in rows)


def test_witness_ranks_force_the_witness_back_in_when_the_search_loses_it():
    task = task_for(TWO)
    scorer = WitnessScorer(task, bonus=-20.0)  # the witness op is the worst choice
    knobs = {**SMALL, "beam_width": 1, "roots": 1}
    rows = segment_witness_ranks(task, scorer, **knobs)
    assert len(rows) == 2
    assert rows[0]["rank"] is None and not rows[0]["forced"]
    assert rows[1]["forced"] and rows[1]["roots"] <= 2
    # Given the witness's own closed state as a root, a width-1 beam that ranks
    # every witness op last still cannot close the second segment.
    assert rows[1]["rank"] is None
    preferred = segment_witness_ranks(task, WitnessScorer(task), **knobs)
    assert [row["rank"] for row in preferred] == [1, 1]


def test_witness_ranks_take_another_program_and_stop_at_its_last_cut():
    task = task_for(TWO)
    # The same behaviour with trailing glue after the last cut: no third row.
    program = (*task.program, "dup", "drop")
    rows = segment_witness_ranks(task, WitnessScorer(task), program, **SMALL)
    assert [row["ops"] for row in rows] == [6, 5]
    with pytest.raises(ValueError, match="refuses"):
        segment_witness_ranks(task, WitnessScorer(task), ("add",), **SMALL)


def test_solve_tasks_runs_the_segment_search_when_the_system_asks():
    first, second = task_for(TWO, 4), task_for(TWO, 5)
    system = solve.System(
        name="tiny-segment",
        repo_id="nobody/nothing",
        revision="0" * 40,
        beam_width=2,
        max_depth=4,  # unused by the segment search; the flat beam could not reach 11
        guard="both",
        library=True,
        measured="a test",
        search="segment",
        roots=2,
        segment_depth=6,
        max_segments=3,
    )
    rows = solve.solve_tasks([first, second], WitnessScorer(first), system)
    assert [row.source for row in rows] == ["search", "library"]
    assert rows[0].program == first.program and rows[0].depth == 11
    assert all(row.solved for row in rows)
    assert "search segment: 2 roots, 3 segments of up to 6 ops" in system.describe()


def test_best_keeps_the_flat_search():
    assert solve.BEST.search == "flat"
    assert "search" not in solve.BEST.describe()
    with pytest.raises(ValueError, match="unknown search"):
        solve.System("x", "r", "0" * 40, 1, 1, "both", True, "m", search="bogus")


def test_witness_rank_counts_a_closed_state_already_expanded():
    # Codex's reproducer: `nil i` closes back on its own input, which the
    # search expanded as the first root; the rank must still see it.
    from types import SimpleNamespace

    from wandering_light.stack_beam import segment_witness_ranks

    task = SimpleNamespace(
        behaviour="identity_via_i", inputs=["3"], outputs=["3"], program=("nil", "i")
    )

    class Flat:
        def log_probs(self, blocks, legal):
            import numpy as np

            return np.where(legal, 0.0, -np.inf)

    rows = segment_witness_ranks(task, Flat(), beam_width=28, segment_depth=2)
    assert rows[0]["rank"] is not None
    assert rows[0]["eligible_rank"] is None
