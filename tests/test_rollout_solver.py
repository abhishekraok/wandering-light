from dataclasses import FrozenInstanceError

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import (
    ActionScorer,
    RolloutDepthTrace,
    RolloutPredictor,
    RolloutSearchResult,
)
from wandering_light.typed_list import TypedList


def make_function(name, code, input_type="int", output_type="int"):
    return FunctionDef(
        name=name,
        input_type=f"builtins.{input_type}",
        output_type=f"builtins.{output_type}",
        code=code,
    )


class RecordingScorer(ActionScorer):
    def __init__(self, score_batch):
        self.score_batch = score_batch
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append((problems, available_functions))
        return self.score_batch(problems, available_functions)


def uniform_scores(problems, functions):
    return [
        dict.fromkeys((function.name for function in functions), 0) for _ in problems
    ]


def test_scores_materialized_beam_in_order_and_normalizes_only_compatible_actions():
    increment = make_function("increment", "return x + 1")
    stringify = make_function("stringify", "return str(x)", output_type="str")
    length = make_function("length", "return len(x) + 1", input_type="str")
    basis = FunctionDefSet([increment, stringify, length])
    root, target = TypedList([7]), TypedList([2])

    def score_batch(problems, functions):
        return [
            {
                "length": -1e308 if state.item_type is str else 1e308,
                "stringify": 1e308,
                "increment": 1e308,
            }
            for state, _ in problems
        ]

    scorer = RecordingScorer(score_batch)
    result = RolloutPredictor(scorer, beam_width=2, budget=3).search_with_cost(
        root, target, basis
    )

    assert scorer.calls == [
        ([(root, target)], basis),
        ([(TypedList([8]), target), (TypedList(["7"]), target)], basis),
    ]
    assert result == RolloutSearchResult(
        FunctionDefList([stringify, length]), 3, "solved"
    )
    assert result.trace == (RolloutDepthTrace(1, (("increment",), ("stringify",)), 2),)
    assert [function.usage_count for function in basis] == [1, 1, 1]


def test_attempt_order_accumulates_parent_and_local_log_probabilities():
    small = make_function("small", "return x + 1")
    big = make_function("big", "return x + 10")
    basis = FunctionDefSet([small, big])

    def score_batch(problems, functions):
        scores_by_state = {
            0: {"small": 2, "big": 0},
            1: {"small": 0, "big": 0},
            10: {"small": -100, "big": 100},
        }
        return [scores_by_state[state[0]] for state, _ in problems]

    result = RolloutPredictor(RecordingScorer(score_batch), budget=3).search_with_cost(
        TypedList([0]), TypedList([2]), basis
    )

    assert result == RolloutSearchResult(FunctionDefList([small, small]), 3, "solved")


def test_ties_follow_beam_and_basis_order_and_dedup_only_within_each_layer():
    first = make_function("zeta", "return x + 1")
    duplicate = make_function("alpha", "return x + 1")
    back = make_function("back", "return x - 1")
    stay = make_function("stay", "return x")
    basis = FunctionDefSet([first, duplicate, back, stay])

    scorer = RecordingScorer(uniform_scores)
    result = RolloutPredictor(scorer, beam_width=2, max_depth=2).search_with_cost(
        TypedList([0]), TypedList([99]), basis
    )

    assert result == RolloutSearchResult(FunctionDefList(), 12, "max_depth")
    assert result.trace == (
        RolloutDepthTrace(1, (("zeta",), ("back",)), 3),
        RolloutDepthTrace(2, (("zeta", "zeta"), ("zeta", "back")), 5),
    )


def test_keeps_the_higher_scoring_path_to_a_duplicate_state():
    low = make_function("low", "return x + 1")
    high = make_function("high", "return x + 1")
    scorer = RecordingScorer(lambda problems, functions: [{"low": -1000, "high": 1000}])

    result = RolloutPredictor(scorer, beam_width=1, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([2]), FunctionDefSet([low, high])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 2, "max_depth")
    assert result.trace == (RolloutDepthTrace(1, (("high",),), 1),)


def test_deduplicates_using_search_key_so_both_signed_zero_states_survive():
    positive = make_function("positive", "return 0.0", output_type="float")
    negative = make_function("negative", "return -0.0", output_type="float")
    stringify = make_function("stringify", "return str(x)", "float", "str")
    basis = FunctionDefSet([positive, negative, stringify])

    result = RolloutPredictor(RecordingScorer(uniform_scores)).search_with_cost(
        TypedList([0]), TypedList(["-0.0"]), basis
    )

    assert result == RolloutSearchResult(
        FunctionDefList([negative, stringify]), 4, "solved"
    )
    assert result.trace == (RolloutDepthTrace(1, (("positive",), ("negative",)), 2),)


@pytest.mark.parametrize(
    ("name", "value", "minimum"),
    [
        ("beam_width", 0, 1),
        ("beam_width", 1.5, 1),
        ("beam_width", True, 1),
        ("max_depth", -1, 0),
        ("max_depth", 0.0, 0),
        ("max_depth", False, 0),
        ("budget", -1, 0),
        ("budget", "1", 0),
        ("budget", False, 0),
    ],
)
def test_validates_limits(name, value, minimum):
    with pytest.raises(ValueError, match=rf"^{name} must be an integer >= {minimum}$"):
        RolloutPredictor(RecordingScorer(uniform_scores), **{name: value})


@pytest.mark.parametrize(
    ("target", "has_functions", "budget", "max_depth", "termination"),
    [
        (0, False, 0, 0, "solved"),
        (1, False, 0, 0, "frontier_exhausted"),
        (1, True, 0, 0, "budget"),
        (1, True, 1, 0, "max_depth"),
    ],
)
def test_early_terminations_are_ordered_and_do_not_call_scorer(
    target, has_functions, budget, max_depth, termination
):
    increment = make_function("increment", "return x + 1")
    basis = FunctionDefSet([increment] if has_functions else [])
    scorer = RecordingScorer(uniform_scores)
    predictor = RolloutPredictor(scorer, budget=budget, max_depth=max_depth)

    result = predictor.search_with_cost(TypedList([0]), TypedList([target]), basis)

    assert result == RolloutSearchResult(FunctionDefList(), 0, termination)
    assert result.trace == ()
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (None, "returned 0 rows; expected 1"),
        (({"increment": 0},), "returned 1 rows; expected 1"),
        ([], "returned 0 rows; expected 1"),
        ([{"increment": 0}] * 2, "returned 2 rows; expected 1"),
        ([None], "wrong function names"),
        ([{}], "wrong function names"),
        ([{"increment": 0, "unexpected": 0}], "wrong function names"),
        ([{"increment": True}], "non-finite or non-numeric"),
        ([{"increment": "0"}], "non-finite or non-numeric"),
        ([{"increment": 1j}], "non-finite or non-numeric"),
        ([{"increment": float("nan")}], "non-finite or non-numeric"),
        ([{"increment": float("inf")}], "non-finite or non-numeric"),
        ([{"increment": -float("inf")}], "non-finite or non-numeric"),
    ],
)
def test_rejects_invalid_batch_before_execution(rows, message):
    increment = make_function("increment", "return x + 1")
    scorer = RecordingScorer(lambda problems, functions: rows)

    with pytest.raises(ValueError, match=message):
        RolloutPredictor(scorer).search_with_cost(
            TypedList([0]), TypedList([1]), FunctionDefSet([increment])
        )

    assert increment.usage_count == 0


def test_validates_even_incompatible_scores():
    increment = make_function("increment", "return x + 1")
    upper = make_function("upper", "return x.upper()", "str", "str")
    scorer = RecordingScorer(
        lambda problems, functions: [{"increment": 0, "upper": float("nan")}]
    )

    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        RolloutPredictor(scorer).search_with_cost(
            TypedList([0]), TypedList([1]), FunctionDefSet([increment, upper])
        )

    assert increment.usage_count == 0


def test_no_compatible_transitions_still_scores_but_does_not_trace():
    upper = make_function("upper", "return x.upper()", "str", "str")
    basis = FunctionDefSet([upper])
    scorer = RecordingScorer(uniform_scores)
    root, target = TypedList([0]), TypedList([1])

    result = RolloutPredictor(scorer).search_with_cost(root, target, basis)

    assert result == RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
    assert result.trace == ()
    assert scorer.calls == [([(root, target)], basis)]


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_raising_attempt_counts_and_last_affordable_solution_wins(budget):
    fail = make_function("fail", "return x / 0")
    increment = make_function("increment", "return x + 1")
    basis = FunctionDefSet([fail, increment])

    result = RolloutPredictor(
        RecordingScorer(uniform_scores), budget=budget
    ).search_with_cost(TypedList([0]), TypedList([1]), basis)

    if budget == 1:
        assert result == RolloutSearchResult(FunctionDefList(), 1, "budget")
    else:
        assert result == RolloutSearchResult(FunctionDefList([increment]), 2, "solved")
    assert result.trace == ()


def test_a_complete_layer_of_raising_attempts_traces_an_empty_beam():
    fail = make_function("fail", "return x / 0")
    scorer = RecordingScorer(uniform_scores)

    result = RolloutPredictor(scorer).search_with_cost(
        TypedList([0]), TypedList([1]), FunctionDefSet([fail])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert result.trace == (RolloutDepthTrace(1, (), 0),)
    assert len(scorer.calls) == 1


def test_budget_stop_preserves_only_previous_complete_layers():
    increment = make_function("increment", "return x + 1")
    double = make_function("double", "return x * 2")
    basis = FunctionDefSet([increment, double])

    result = RolloutPredictor(
        RecordingScorer(uniform_scores), budget=3
    ).search_with_cost(TypedList([0]), TypedList([99]), basis)

    assert result == RolloutSearchResult(FunctionDefList(), 3, "budget")
    assert result.trace == (RolloutDepthTrace(1, (("increment",), ("double",)), 2),)


def test_batch_predictions_have_independent_budgets():
    increment = make_function("increment", "return x + 1")
    basis = FunctionDefSet([increment])
    scorer = RecordingScorer(uniform_scores)
    predictor = RolloutPredictor(scorer, budget=2)
    root = TypedList([0])

    results = predictor.predict_functions_batch(
        [(root, TypedList([3])), (root, TypedList([2])), (root, TypedList([1]))], basis
    )

    assert results == [
        FunctionDefList(),
        FunctionDefList([increment, increment]),
        FunctionDefList([increment]),
    ]
    assert predictor.predict_functions_batch([], basis) == []
    assert len(scorer.calls) == 5


def test_result_and_depth_trace_are_frozen_and_trace_is_excluded_from_equality():
    layer = RolloutDepthTrace(1, (("increment",),), 1)
    result = RolloutSearchResult(FunctionDefList(), 1, "max_depth", (layer,))

    assert result == RolloutSearchResult(FunctionDefList(), 1, "max_depth")
    with pytest.raises(FrozenInstanceError):
        result.executions = 2
    with pytest.raises(FrozenInstanceError):
        layer.depth = 2
