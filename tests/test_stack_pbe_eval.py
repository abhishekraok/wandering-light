"""The held-out grader: each half is scored on its own, and nothing raises."""

import dataclasses

import pytest

from wandering_light.stack_pbe import BY_NAME, generate
from wandering_light.stack_pbe_eval import Report, Score, best, grade, summarise


def make_task(inputs, outputs, held_in, held_out):
    """A real task record with hand-written data, for grader edge cases."""
    (task,), _ = generate(1, 0, catalogue=[BY_NAME["add_pair"]])
    return dataclasses.replace(
        task,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        held_out_inputs=tuple(held_in),
        held_out_outputs=tuple(held_out),
    )


# `add` is the rule; `drop 1 add` fits these shown pairs and nothing else.
ADD = make_task(
    ["[ 1 1 ]", "[ 0 1 ]"],
    ["[ 2 ]", "[ 1 ]"],
    ["[ 3 4 ]", "[ 5 1 ]"],
    ["[ 7 ]", "[ 6 ]"],
)


def test_rule_solves():
    score = grade("add", ADD)
    assert score == Score(("add",), parsed=True, legal=True, shown=True, held_out=True)
    assert score.solved
    assert not score.memorised


def test_special_case_is_memorised_not_solved():
    score = grade("drop 1 add", ADD)
    assert score.parsed and score.legal and score.shown
    assert not score.held_out
    assert score.memorised
    assert not score.solved


def test_string_and_sequence_agree():
    assert grade("drop  1\tadd", ADD) == grade(["drop", "1", "add"], ADD)
    assert grade(("add",), ADD).program == ("add",)


@pytest.mark.parametrize(
    "candidate",
    ["", "   ", [], (), "add foo", "[add]", "plus", ["add", "MARK"], [None], [3]],
)
def test_unparseable_scores_all_false(candidate):
    score = grade(candidate, ADD)
    assert not (score.parsed or score.legal or score.shown or score.held_out)
    assert not score.solved and not score.memorised


def test_wrong_but_legal_program():
    score = grade("sub", ADD)
    assert score.parsed and score.legal
    assert not score.shown and not score.held_out


def test_extra_values_left_on_stack_do_not_fit():
    score = grade("add dup", ADD)
    assert score.legal and not score.shown and not score.held_out


def test_refusal_on_held_out_fails_only_that_half():
    task = make_task(["[ 6 2 ]", "[ 9 3 ]"], ["[ 3 ]", "[ 3 ]"], ["[ 1 0 ]"], ["[ 0 ]"])
    score = grade("div", task)
    assert score.parsed and score.shown
    assert not score.legal and not score.held_out
    assert score.memorised


def test_refusal_on_shown_fails_only_that_half():
    task = make_task(["[ 4 0 ]", "[ 6 3 ]"], ["[ 0 ]", "[ 2 ]"], ["[ 8 2 ]"], ["[ 4 ]"])
    score = grade("div", task)
    assert score.parsed and score.held_out
    assert not score.legal and not score.shown
    assert not score.solved and not score.memorised


def test_unclosed_quotation_is_legal_but_wrong():
    score = grade("[ add", ADD)
    assert score.parsed and score.legal
    assert not score.shown


def test_quotation_programs_run():
    (task,), _ = generate(1, 3, catalogue=[BY_NAME["sum"]])
    assert grade("0 [ add ] fold", task).solved
    assert not grade("0 [ mul ] fold", task).shown


def test_every_witness_solves_its_generated_tasks():
    tasks, _ = generate(2, 11)
    for task in tasks:
        assert grade(task.program, task).solved, task.behaviour


def test_best_prefers_solve_over_memorisation_in_either_order():
    for order in (["drop 1 add", "add"], ["add", "drop 1 add"]):
        assert best(order, ADD).program == ("add",)
    assert best(["foo", "sub", "drop 1 add"], ADD).program == ("drop", "1", "add")
    assert best(["foo", "sub"], ADD).program == ("sub",)
    assert best(["foo"], ADD).program == ("foo",)


def test_best_of_nothing_is_unparsed():
    assert best([], ADD) == Score(())
    assert not best(iter(()), ADD).parsed


def test_summarise_and_str():
    scores = [
        grade("add", ADD),
        grade("drop 1 add", ADD),
        grade("sub", ADD),
        grade("bogus", ADD),
    ]
    report = summarise(scores)
    assert report == Report(tasks=4, solved=1, memorised=1, shown_fit=2, illegal=1)
    assert report.solve_rate == 0.25
    assert report.shown_rate == 0.5
    assert report.overstatement == 0.25
    assert str(report) == (
        "1/4 solved (25.0%); shown-only would say 50.0% (+25.0%, 1 memorised)"
    )


def test_empty_report():
    report = summarise([])
    assert report == Report(0, 0, 0, 0, 0)
    assert (report.solve_rate, report.shown_rate, report.overstatement) == (0, 0, 0)
    assert (
        str(report)
        == "0/0 solved (0.0%); shown-only would say 0.0% (+0.0%, 0 memorised)"
    )
