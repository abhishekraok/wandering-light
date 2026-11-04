"""The held-out grader: a solve and a memorisation must come apart.

The tasks here are built by hand so the shown and hidden halves can be made
to disagree on purpose -- the corpus generator takes care never to produce
that, which is exactly why a test cannot rely on it.
"""

from __future__ import annotations

import random

import pytest

from wandering_light.stack_pbe import CATALOGUE, Task, build_task, draw_task, load
from wandering_light.stack_pbe_eval import Report, Score, best, grade, summarise

CORPUS = "wandering_light/evals/data/stack_pbe_v3.jsonl.gz"


def make_task(
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    held_out_inputs: tuple[str, ...],
    held_out_outputs: tuple[str, ...],
    program: str = "dup add",
) -> Task:
    return Task(
        behaviour="hand_written",
        summary="a task whose halves are chosen by the test",
        split="eval",
        task_hash_split="eval",
        inputs=inputs,
        outputs=outputs,
        held_out_inputs=held_out_inputs,
        held_out_outputs=held_out_outputs,
        program=tuple(program.split()),
        length=len(program.split()),
        examples=len(inputs),
        held_out=len(held_out_inputs),
        fuel=0,
        tags=(),
    )


@pytest.fixture
def doubling() -> Task:
    """Doubling, with shown examples that squaring also fits."""
    return make_task(
        inputs=("[ 0 ]", "[ 2 ]"),
        outputs=("[ 0 ]", "[ 4 ]"),
        held_out_inputs=("[ 3 ]", "[ -1 ]"),
        held_out_outputs=("[ 6 ]", "[ -2 ]"),
    )


@pytest.fixture
def catalogue_task() -> Task:
    behaviour = next(b for b in CATALOGUE if b.name == "sum")
    return build_task(behaviour, draw_task(behaviour, random.Random(3), examples=4))


# -- Score ------------------------------------------------------------------


def test_score_properties():
    assert Score(("add",), True, True, True, True).solved
    assert not Score(("add",), True, True, True, True).memorised
    memorised = Score(("add",), True, True, True, False)
    assert memorised.memorised and not memorised.solved
    assert not Score(("add",), True, True, False, True).solved
    assert not Score(("add",), True, True, False, True).memorised


def test_score_is_frozen():
    score = Score(("add",), True, True, True, True)
    with pytest.raises(AttributeError):
        score.shown = False  # type: ignore[misc]


# -- grade ------------------------------------------------------------------


def test_witness_solves(doubling, catalogue_task):
    score = grade("dup add", doubling)
    assert score == Score(("dup", "add"), True, True, True, True)
    assert score.solved
    assert grade(catalogue_task.program, catalogue_task).solved


def test_string_and_sequence_agree(doubling):
    assert grade("dup add", doubling) == grade(["dup", "add"], doubling)
    assert grade("  dup   add ", doubling).program == ("dup", "add")


def test_memorisation_is_caught(doubling):
    score = grade("dup mul", doubling)
    assert score.parsed and score.legal
    assert score.shown and not score.held_out
    assert score.memorised and not score.solved


def test_empty_program_scores_all_false(doubling):
    for empty in ("", "   ", (), []):
        score = grade(empty, doubling)
        assert score == Score(())
        assert not (score.parsed or score.legal or score.shown or score.held_out)


def test_unknown_token_is_unparsed_and_does_not_raise(doubling):
    score = grade("dup frobnicate", doubling)
    assert score.program == ("dup", "frobnicate")
    assert score == Score(("dup", "frobnicate"))
    assert grade("dup ad d", doubling) == Score(("dup", "ad", "d"))


def test_refused_run_is_illegal_and_fails_that_half(doubling):
    # `add` needs two values and every input stack has one.
    score = grade("add", doubling)
    assert score.parsed
    assert not score.legal
    assert not score.shown and not score.held_out


def test_legal_wrong_program(doubling):
    score = grade("dup", doubling)
    assert score.parsed and score.legal
    assert not score.shown and not score.held_out


def test_halves_are_scored_independently():
    # Integer division fits the shown half and refuses on the hidden one,
    # where the divisor is zero: still a shown fit, still not legal.
    task = make_task(
        inputs=("[ 4 2 ]", "[ 9 3 ]"),
        outputs=("[ 2 ]", "[ 3 ]"),
        held_out_inputs=("[ 5 0 ]",),
        held_out_outputs=("[ 0 ]",),
        program="div",
    )
    score = grade("div", task)
    assert score.shown
    assert not score.held_out
    assert not score.legal
    assert score.memorised

    # And the other way round: refuse on the shown half, fit the hidden one.
    flipped = make_task(
        inputs=("[ 5 0 ]",),
        outputs=("[ 0 ]",),
        held_out_inputs=("[ 4 2 ]", "[ 9 3 ]"),
        held_out_outputs=("[ 2 ]", "[ 3 ]"),
        program="div",
    )
    score = grade("div", flipped)
    assert not score.shown
    assert score.held_out
    assert not score.legal


def test_program_must_leave_exactly_the_output_stack(doubling):
    # Right value on top, but an extra value underneath.
    score = grade("dup dup add", doubling)
    assert score.legal
    assert not score.shown and not score.held_out


def test_combinators_are_graded_by_running_them():
    task = make_task(
        inputs=("[ (1 2) ]", "[ (3) ]"),
        outputs=("[ (2 4) ]", "[ (6) ]"),
        held_out_inputs=("[ (0 -1) ]",),
        held_out_outputs=("[ (0 -2) ]",),
        program="[ dup add ] map",
    )
    assert grade("[ dup add ] map", task).solved
    assert grade("[ dup mul ] map", task) == Score(
        ("[", "dup", "mul", "]", "map"), True, True, False, False
    )
    # An unclosed quotation leaves a MARK on the stack: legal, wrong.
    dangling = grade("[ dup add", task)
    assert dangling.legal and not dangling.shown


# -- best -------------------------------------------------------------------


def test_best_prefers_a_solve_over_a_memorisation_in_either_order(doubling):
    solve, memorise = "dup add", "dup mul"
    assert best([memorise, solve], doubling).program == ("dup", "add")
    assert best([solve, memorise], doubling).program == ("dup", "add")
    assert best([memorise, solve], doubling).solved


def test_best_ranks_shown_then_legal_then_parsed(doubling):
    assert best(["add", "dup mul"], doubling).program == ("dup", "mul")
    assert best(["dup", "add"], doubling).program == ("dup",)  # legal beats illegal
    assert best(["frob", "add"], doubling).program == ("add",)  # parsed beats not
    assert best(["add", "frob"], doubling).program == ("add",)


def test_best_of_nothing_is_the_unparsed_empty_program(doubling):
    score = best([], doubling)
    assert score == Score(())
    assert not score.parsed


def test_best_accepts_sequences_and_strings(doubling):
    assert best([("dup", "add"), "dup mul"], doubling).solved


# -- Report -----------------------------------------------------------------


def test_report_rates_and_str():
    report = Report(tasks=10, solved=4, memorised=2, shown_fit=6, illegal=1)
    assert report.solve_rate == pytest.approx(0.4)
    assert report.shown_rate == pytest.approx(0.6)
    assert report.overstatement == pytest.approx(0.2)
    assert str(report) == (
        "4/10 solved (40.0%); shown-only would say 60.0% (+20.0%, 2 memorised)"
    )


def test_empty_report_is_all_zero():
    report = summarise([])
    assert report == Report()
    assert report.solve_rate == 0.0
    assert report.shown_rate == 0.0
    assert report.overstatement == 0.0
    assert str(report) == (
        "0/0 solved (0.0%); shown-only would say 0.0% (+0.0%, 0 memorised)"
    )


def test_summarise_counts(doubling):
    scores = [
        grade("dup add", doubling),  # solved
        grade("dup mul", doubling),  # memorised: shown fit, held-out miss
        grade("dup", doubling),  # legal, wrong
        grade("add", doubling),  # illegal
        grade("", doubling),  # unparsed -- also not legal
    ]
    report = summarise(scores)
    assert report == Report(tasks=5, solved=1, memorised=1, shown_fit=2, illegal=2)
    assert report.solve_rate == pytest.approx(0.2)
    assert report.shown_rate == pytest.approx(0.4)
    assert report.overstatement == pytest.approx(0.2)


def test_summarise_accepts_a_generator(doubling):
    report = summarise(grade(p, doubling) for p in ("dup add", "dup add"))
    assert report.tasks == 2 and report.solved == 2


# -- the corpus -------------------------------------------------------------


def test_every_corpus_witness_solves_its_task():
    tasks = load(CORPUS)
    assert len(tasks) == 1600
    scores = [grade(task.program, task) for task in tasks]
    report = summarise(scores)
    assert report.solved == report.tasks == 1600
    assert report.memorised == 0 and report.illegal == 0
    assert report.overstatement == 0.0
