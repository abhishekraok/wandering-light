from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import (
    ActionScorer,
    RolloutDepthTrace,
    RolloutPredictor,
    RolloutSearchResult,
)
from wandering_light.typed_list import TypedList


def function(name, code, input_type="int", output_type="int"):
    return FunctionDef(
        name=name,
        input_type=f"builtins.{input_type}",
        output_type=f"builtins.{output_type}",
        code=code,
    )


class RecordingScorer(ActionScorer):
    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append(problems)
        # Reverse insertion order to distinguish scorer order from basis order.
        return [
            {
                func.name: self.scores.get(state[0], {}).get(func.name, 0)
                for func in reversed(available_functions)
            }
            for state, _ in problems
        ]


@pytest.mark.parametrize(
    ("name", "minimum"), [("beam_width", 1), ("max_depth", 0), ("budget", 0)]
)
@pytest.mark.parametrize("invalid", [False, True, 1.0, "2", None, -1])
def test_limits_require_integers_in_range(name, minimum, invalid):
    with pytest.raises(ValueError, match=rf"^{name} must be an integer >= {minimum}$"):
        RolloutPredictor(RecordingScorer(), **{name: invalid})


def test_zero_width_is_invalid():
    with pytest.raises(ValueError, match="beam_width must be an integer >= 1"):
        RolloutPredictor(RecordingScorer(), beam_width=0)


@pytest.mark.parametrize(
    ("target", "empty_basis", "budget", "termination"),
    [
        (0, True, 0, "solved"),
        (1, True, 0, "frontier_exhausted"),
        (1, False, 0, "budget"),
        (1, False, 1, "max_depth"),
    ],
)
def test_preconditions_return_before_scoring(target, empty_basis, budget, termination):
    scorer = RecordingScorer()
    basis = FunctionDefSet([] if empty_basis else [function("inc", "return x + 1")])
    result = RolloutPredictor(scorer, budget=budget, max_depth=0).search_with_cost(
        TypedList([0]), TypedList([target]), basis
    )
    assert result == RolloutSearchResult(FunctionDefList(), 0, termination)
    assert result.trace == ()
    assert scorer.calls == []


@pytest.mark.parametrize(("convert_score", "executions"), [(-0.5, 3), (-2, 5)])
def test_attempt_order_uses_masked_log_softmax_and_parent_probability(
    convert_score, executions
):
    inc = function("inc", "return x + 1")
    convert = function("convert", "return str(x)", output_type="str")
    finish = function("finish", "return 'done'", "str", "str")
    basis = FunctionDefSet([inc, convert, finish])
    scorer = RecordingScorer(
        {
            0: {"convert": convert_score},
            1: {"inc": 1e300, "convert": 1e300},
            "0": {"inc": 1e300, "finish": -1e300},
        }
    )
    start, target = TypedList([0]), TypedList(["done"])
    result = RolloutPredictor(scorer, budget=executions).search_with_cost(
        start, target, basis
    )

    # The integer parent splits its mass equally. The string parent has only
    # one compatible action, so it carries all its mass into the candidate.
    assert result == RolloutSearchResult(
        FunctionDefList([convert, finish]), executions, "solved"
    )
    assert scorer.calls == [
        [(start, target)],
        [(TypedList([1]), target), (TypedList(["0"]), target)],
    ]
    assert result.trace == (RolloutDepthTrace(1, (("inc",), ("convert",)), 2),)


def test_width_one_changes_action_after_observing_the_intermediate_value():
    inc = function("inc", "return x + 1")
    triple = function("triple", "return x * 3")
    scorer = RecordingScorer({0: {"inc": 4}, 1: {"triple": 4}})
    predictor = RolloutPredictor(scorer, beam_width=1, budget=3)
    basis = FunctionDefSet([inc, triple])
    start, target = TypedList([0]), TypedList([3])

    result = predictor.search_with_cost(start, target, basis)
    assert result == RolloutSearchResult(FunctionDefList([inc, triple]), 3, "solved")
    assert result.trace == (RolloutDepthTrace(1, (("inc",),), 2),)
    assert scorer.calls == [[(start, target)], [(TypedList([1]), target)]]

    unsolved = predictor.search_with_cost(start, TypedList([4]), basis)
    assert unsolved == RolloutSearchResult(FunctionDefList(), 3, "budget")
    assert unsolved.trace == result.trace


def test_traces_deduplicate_within_each_layer_then_prune_in_attempt_order():
    basis = FunctionDefSet(
        [
            function("alias", "return x + 1"),
            function("stay", "return x"),
            function("inc", "return x + 1"),
            function("two", "return x + 2"),
        ]
    )
    scorer = RecordingScorer({0: {"alias": -1}, 1: {"alias": -1}})
    result = RolloutPredictor(scorer, beam_width=2, max_depth=2).search_with_cost(
        TypedList([0]), TypedList([99]), basis
    )
    assert result == RolloutSearchResult(FunctionDefList(), 12, "max_depth")
    assert result.trace == (
        RolloutDepthTrace(1, (("stay",), ("inc",)), 3),
        RolloutDepthTrace(2, (("stay", "stay"), ("stay", "inc")), 4),
    )


def test_search_key_preserves_distinguishable_signed_zero_successors():
    positive = function("positive", "return 0.0", output_type="float")
    negative = function("negative", "return -0.0", output_type="float")
    stringify = function("stringify", "return str(x)", "float", "str")
    result = RolloutPredictor(RecordingScorer()).search_with_cost(
        TypedList([0]),
        TypedList(["-0.0"]),
        FunctionDefSet([positive, negative, stringify]),
    )
    assert result == RolloutSearchResult(
        FunctionDefList([negative, stringify]), 4, "solved"
    )
    assert result.trace == (RolloutDepthTrace(1, (("positive",), ("negative",)), 2),)


def test_layer_with_no_compatible_transitions_stops_after_scoring():
    stringify = function("stringify", "return str(x)", output_type="str")
    scorer = RecordingScorer()
    start, target = TypedList([1]), TypedList([2])
    result = RolloutPredictor(scorer).search_with_cost(
        start, target, FunctionDefSet([stringify])
    )
    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert scorer.calls == [[(start, target)], [(TypedList(["1"]), target)]]
    assert result.trace == (RolloutDepthTrace(1, (("stringify",),), 1),)


@pytest.mark.parametrize(
    ("budget", "termination"), [(1, "budget"), (2, "frontier_exhausted")]
)
def test_layer_with_only_a_raising_action(budget, termination):
    basis = FunctionDefSet([function("fail", "raise ValueError()")])
    result = RolloutPredictor(RecordingScorer(), budget=budget).search_with_cost(
        TypedList([0]), TypedList([1]), basis
    )
    assert result == RolloutSearchResult(FunctionDefList(), 1, termination)
    assert result.trace == (() if budget == 1 else (RolloutDepthTrace(1, (), 0),))


@pytest.mark.parametrize(("target", "termination"), [(1, "solved"), (2, "budget")])
def test_last_execution_after_a_failure(target, termination):
    inc = function("inc", "return x + 1")
    basis = FunctionDefSet([function("fail", "raise ValueError()"), inc])
    result = RolloutPredictor(RecordingScorer(), budget=2).search_with_cost(
        TypedList([0]), TypedList([target]), basis
    )
    path = FunctionDefList([inc] if target == 1 else [])
    assert result == RolloutSearchResult(path, 2, termination)
    assert result.trace == ()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "action scorer returned 0 rows; expected 1"),
        (({"inc": 1},), "action scorer returned 1 rows; expected 1"),
        (None, "rows; expected 1"),
        ([None], "wrong function names"),
        ([{}], "wrong function names"),
        ([{"typo": 0}], "wrong function names"),
        ([{"inc": 0, "extra": 1}], "wrong function names"),
    ],
)
def test_invalid_scorer_output(rows, message):
    scorer = Mock(spec=ActionScorer)
    scorer.score_actions_batch.return_value = rows
    with pytest.raises(ValueError, match=message):
        RolloutPredictor(scorer).search_with_cost(
            TypedList([0]),
            TypedList([1]),
            FunctionDefSet([function("inc", "return x + 1")]),
        )


@pytest.mark.parametrize(
    "score", [True, False, None, "1", 1j, float("nan"), float("inf"), -float("inf")]
)
def test_validate_all_scores_even_for_incompatible_actions(score):
    basis = FunctionDefSet(
        [
            function("inc", "return x + 1"),
            function("upper", "return x.upper()", "str", "str"),
        ]
    )
    scorer = RecordingScorer({0: {"upper": score}})
    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        RolloutPredictor(scorer).search_with_cost(TypedList([0]), TypedList([1]), basis)
    assert basis[0].usage_count == 0


def test_batch_problems_have_independent_budgets():
    inc = function("inc", "return x + 1")
    basis = FunctionDefSet([inc])
    predictor = RolloutPredictor(RecordingScorer(), budget=2)
    assert predictor.predict_functions_batch([], basis) == []
    results = predictor.predict_functions_batch(
        [(TypedList([0]), TypedList([target])) for target in (3, 2, 1)], basis
    )
    assert results == [
        FunctionDefList(),
        FunctionDefList([inc, inc]),
        FunctionDefList([inc]),
    ]


def test_result_and_trace_are_frozen_and_trace_is_excluded_from_result_equality():
    entry = RolloutDepthTrace(1, (("inc",),), 1)
    result = RolloutSearchResult(FunctionDefList(), 1, "max_depth")
    assert result == replace(result, trace=(entry,))
    with pytest.raises(FrozenInstanceError):
        result.executions = 2
    with pytest.raises(FrozenInstanceError):
        entry.depth = 2
