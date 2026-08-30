"""Grading a candidate against a PBE task.

The behaviour that matters is the two-number one: a program is scored on the
examples it saw and on the examples it did not, and a program that passes the
first and fails the second must be reported as a failure. Every other check
here exists to stop that one from being reported by accident.
"""

import random

import pytest

from tests.test_stack_pbe import CORPUS
from wandering_light.stack_pbe import (
    BY_NAME,
    Drawn,
    build_task,
    draw_task,
    generate,
    load,
)
from wandering_light.stack_pbe_eval import Score, best, grade, summarise


@pytest.fixture(scope="module")
def tasks():
    return load(str(CORPUS))


@pytest.fixture(scope="module")
def sum_task():
    rng = random.Random(4)
    return build_task(BY_NAME["sum"], draw_task(BY_NAME["sum"], rng, 4, held_out=6))


def test_the_authored_witness_solves_every_task(tasks):
    scores = [grade(task.program, task) for task in tasks]
    report = summarise(scores)
    assert report.solved == report.tasks
    assert report.memorised == 0


def test_an_alternative_witness_also_counts_as_solved(sum_task):
    """Grading is on behaviour, not on matching the recorded program text."""
    assert grade("uncons [ add ] fold", sum_task).solved


def test_a_wrong_program_fails_both_halves(sum_task):
    score = grade("1 [ mul ] fold", sum_task)
    assert score.parsed and not score.shown and not score.held_out


def test_an_illegal_program_is_not_legal(sum_task):
    score = grade("add add add", sum_task)
    assert score.parsed and not score.legal and not score.solved


def test_an_unknown_op_does_not_raise(sum_task):
    score = grade("frobnicate", sum_task)
    assert not score.parsed and not score.solved


def test_an_empty_candidate_loses(sum_task):
    assert not grade("", sum_task).parsed
    assert not best([], sum_task).parsed


def test_a_program_fitting_only_the_shown_examples_is_memorised():
    """The failure mode, end to end, on a task built to contain it.

    `uncons drop i` returns the last element of a two-element list by accident:
    it drops the head and runs the one-element tail as a program. Shown three
    length-two lists it is indistinguishable from `last`; on a longer list it
    leaves the tail instead.
    """
    behaviour = BY_NAME["last"]
    shown = [([9, 1],), ([-2, -2],), ([4, -7],)]
    hidden = [([1, 2, 3],), ([4, 5, 6, 7],)]
    task = build_task(
        behaviour,
        Drawn(
            shown=shown,
            shown_results=[behaviour.reference(*a) for a in shown],
            held_out=hidden,
            held_out_results=[behaviour.reference(*a) for a in hidden],
        ),
    )
    score = grade("uncons drop i", task)
    assert score.shown
    assert not score.held_out
    assert score.memorised
    assert not score.solved

    report = summarise([score])
    assert report.solve_rate == 0.0
    assert report.shown_rate == 1.0
    assert report.overstatement == 1.0


def test_best_prefers_a_solution_over_a_memorisation(sum_task):
    """A run that took `any(shown)` would score the memoriser as a hit."""
    memoriser = "drop 0"
    picked = best([memoriser, sum_task.program], sum_task)
    assert picked.solved
    picked_reordered = best([sum_task.program, memoriser], sum_task)
    assert picked_reordered.solved


def test_report_separates_the_two_numbers():
    scores = [
        Score(("a",), parsed=True, legal=True, shown=True, held_out=True),
        Score(("b",), parsed=True, legal=True, shown=True, held_out=False),
        Score(("c",), parsed=True, legal=False, shown=False, held_out=False),
        Score((), parsed=False, legal=False, shown=False, held_out=False),
    ]
    report = summarise(scores)
    assert (report.tasks, report.solved, report.memorised) == (4, 1, 1)
    assert report.shown_fit == 2
    assert report.illegal == 2
    assert report.solve_rate == 0.25
    assert report.shown_rate == 0.5
    assert report.overstatement == 0.25
    assert "1 memorised" in str(report)


def test_grading_never_reads_the_held_out_examples_to_decide_shown():
    """Sanity: the two halves are scored independently."""
    made, _stats = generate(per_behaviour=1, seed=9)
    for task in made:
        score = grade(task.program, task)
        assert score.shown and score.held_out
