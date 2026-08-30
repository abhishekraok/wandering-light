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


def make_function(name: str, code: str, input_type: str = "builtins.int"):
    return FunctionDef(
        name=name,
        input_type=input_type,
        output_type="builtins.int" if input_type == "builtins.int" else input_type,
        code=code,
    )


class RecordingScorer(ActionScorer):
    def __init__(self, score):
        self.score = score
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append(problems)
        return [
            self.score(state, target, available_functions) for state, target in problems
        ]


def uniform_scores(_state, _target, functions):
    return {function.name: 0.0 for function in functions}


def test_zero_distance_is_solved_without_scoring_or_execution():
    scorer = RecordingScorer(uniform_scores)
    value = TypedList([1])

    result = RolloutPredictor(scorer).search_with_cost(value, value, FunctionDefSet())

    assert result == RolloutSearchResult(FunctionDefList(), 0, "solved")
    assert scorer.calls == []


def test_policy_observes_the_materialized_state_at_every_step():
    increment = make_function("increment", "return x + 1")
    double = make_function("double", "return x * 2")
    functions = FunctionDefSet([increment, double])

    def score(state, _target, _functions):
        if state == TypedList([2]):
            return {"increment": 2.0, "double": 0.0}
        assert state == TypedList([3])
        return {"increment": 0.0, "double": 2.0}

    scorer = RecordingScorer(score)
    result = RolloutPredictor(
        scorer, beam_width=1, max_depth=2, budget=10
    ).search_with_cost(TypedList([2]), TypedList([6]), functions)

    assert result.function_defs == FunctionDefList([increment, double])
    assert result.executions == 3
    assert result.termination == "solved"
    assert [[state.items for state, _ in call] for call in scorer.calls] == [
        [[2]],
        [[3]],
    ]


def test_wider_beam_can_recover_a_lower_ranked_first_action():
    increment = make_function("increment", "return x + 1")
    double = make_function("double", "return x * 2")
    functions = FunctionDefSet([increment, double])

    def score(state, _target, _functions):
        if state == TypedList([2]):
            return {"increment": 2.0, "double": 1.0}
        if state == TypedList([4]):
            return {"increment": 0.0, "double": 10.0}
        return {"increment": 0.0, "double": 0.0}

    greedy = RolloutPredictor(
        RecordingScorer(score), beam_width=1, max_depth=2, budget=10
    ).search_with_cost(TypedList([2]), TypedList([8]), functions)
    beam = RolloutPredictor(
        RecordingScorer(score), beam_width=2, max_depth=2, budget=10
    ).search_with_cost(TypedList([2]), TypedList([8]), functions)

    assert greedy.termination == "max_depth"
    assert beam.function_defs == FunctionDefList([double, double])
    assert beam.termination == "solved"


def test_trace_shows_lower_ranked_witness_prefix_surviving_only_in_wider_beam():
    increment = make_function("increment", "return x + 1")
    double = make_function("double", "return x * 2")
    functions = FunctionDefSet([increment, double])

    def scorer():
        return RecordingScorer(
            lambda _state, _target, _functions: {"increment": 2.0, "double": 1.0}
        )

    narrow = RolloutPredictor(
        scorer(), beam_width=1, max_depth=1, budget=10
    ).search_with_cost(TypedList([2]), TypedList([8]), functions)
    wide = RolloutPredictor(
        scorer(), beam_width=2, max_depth=1, budget=10
    ).search_with_cost(TypedList([2]), TypedList([8]), functions)

    assert narrow.trace[0].retained_paths == (("increment",),)
    assert wide.trace[0].retained_paths == (("increment",), ("double",))
    assert ("double",) not in narrow.trace[0].retained_paths
    assert ("double",) in wide.trace[0].retained_paths


def test_scores_are_normalized_per_node_before_paths_are_compared():
    to_a = make_function("to_a", "return 1")
    to_b = make_function("to_b", "return 10")
    finish_a = make_function("finish_a", "return 99 if x == 1 else 100")
    finish_b = make_function("finish_b", "return 99 if x == 10 else 100")
    functions = FunctionDefSet([to_a, to_b, finish_a, finish_b])

    def score(state, _target, _functions):
        if state == TypedList([0]):
            return {"to_a": 3.0, "to_b": 1.0, "finish_a": -10.0, "finish_b": -10.0}
        if state == TypedList([1]):
            return {"to_a": 0.0, "to_b": 0.0, "finish_a": 3.0, "finish_b": 0.0}
        if state == TypedList([10]):
            # The additive shift must not make this lower-probability parent win.
            return {
                "to_a": 100.0,
                "to_b": 100.0,
                "finish_a": 100.0,
                "finish_b": 103.0,
            }
        raise AssertionError(f"unexpected beam state: {state}")

    result = RolloutPredictor(
        RecordingScorer(score), beam_width=2, max_depth=2, budget=20
    ).search_with_cost(TypedList([0]), TypedList([99]), functions)

    assert result.function_defs == FunctionDefList([to_a, finish_a])
    assert result.executions == 5


def test_policy_order_controls_execution_order_with_basis_order_as_tiebreak():
    low = make_function("low", "return x + 1")
    high = make_function("high", "return x + 2")
    functions = FunctionDefSet([low, high])
    scorer = RecordingScorer(
        lambda _state, _target, _functions: {"low": -1.0, "high": 1.0}
    )

    result = RolloutPredictor(scorer, budget=2, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([2]), functions
    )

    assert result.function_defs == FunctionDefList([high])
    assert result.executions == 1
    assert low.usage_count == 0


def test_type_mask_excludes_an_incompatible_high_scoring_action():
    upper = FunctionDef(
        name="upper",
        input_type="builtins.str",
        output_type="builtins.str",
        code="return x.upper()",
    )
    increment = make_function("increment", "return x + 1")
    functions = FunctionDefSet([upper, increment])
    scorer = RecordingScorer(
        lambda _state, _target, _functions: {"upper": 100.0, "increment": -100.0}
    )

    result = RolloutPredictor(scorer, budget=1, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([1]), functions
    )

    assert result.function_defs == FunctionDefList([increment])
    assert result.executions == 1
    assert upper.usage_count == 0


def test_within_layer_structural_dedup_does_not_consume_two_beam_slots():
    to_one = make_function("to_one", "return 1")
    to_one_alias = make_function("to_one_alias", "return 1")
    to_two = make_function("to_two", "return 2")
    finish = make_function("finish", "return 3 if x == 2 else 99")
    functions = FunctionDefSet([to_one, to_one_alias, to_two, finish])

    def score(state, _target, _functions):
        if state == TypedList([0]):
            return {
                "to_one": 3.0,
                "to_one_alias": 2.0,
                "to_two": 1.0,
                "finish": 0.0,
            }
        if state == TypedList([2]):
            return {
                "to_one": 0.0,
                "to_one_alias": 0.0,
                "to_two": 0.0,
                "finish": 10.0,
            }
        return {function.name: 0.0 for function in _functions}

    scorer = RecordingScorer(score)
    result = RolloutPredictor(
        scorer, beam_width=2, max_depth=2, budget=20
    ).search_with_cost(TypedList([0]), TypedList([3]), functions)

    assert {state.items[0] for state, _target in scorer.calls[1]} == {1, 2}
    assert result.function_defs == FunctionDefList([to_two, finish])
    assert result.termination == "solved"


def test_trace_counts_unique_successors_before_pruning_and_records_kept_paths():
    to_one = make_function("to_one", "return 1")
    to_one_alias = make_function("to_one_alias", "return 1")
    to_two = make_function("to_two", "return 2")
    functions = FunctionDefSet([to_one, to_one_alias, to_two])
    scorer = RecordingScorer(
        lambda _state, _target, _functions: {
            "to_one": 3.0,
            "to_one_alias": 2.0,
            "to_two": 1.0,
        }
    )

    result = RolloutPredictor(
        scorer, beam_width=2, max_depth=1, budget=10
    ).search_with_cost(TypedList([0]), TypedList([99]), functions)

    assert result.trace == (
        RolloutDepthTrace(
            depth=1,
            retained_paths=(("to_one",), ("to_two",)),
            unique_successors=2,
        ),
    )


def test_raising_transition_is_charged_and_can_exhaust_the_budget():
    exploding = make_function("exploding", "return x + None")
    increment = make_function("increment", "return x + 1")
    functions = FunctionDefSet([exploding, increment])
    scorer = RecordingScorer(
        lambda _state, _target, _functions: {"exploding": 1.0, "increment": 0.0}
    )

    result = RolloutPredictor(scorer, budget=1, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([1]), functions
    )

    assert result == RolloutSearchResult(FunctionDefList(), 1, "budget")


def test_solution_on_the_last_execution_wins_over_budget_termination():
    exploding = make_function("exploding", "return x + None")
    increment = make_function("increment", "return x + 1")
    functions = FunctionDefSet([exploding, increment])
    scorer = RecordingScorer(
        lambda _state, _target, _functions: {"exploding": 1.0, "increment": 0.0}
    )

    result = RolloutPredictor(scorer, budget=2, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([1]), functions
    )

    assert result == RolloutSearchResult(FunctionDefList([increment]), 2, "solved")


def test_max_depth_and_dead_frontier_are_distinct_terminations():
    increment = make_function("increment", "return x + 1")
    exploding = make_function("exploding", "return x + None")

    depth = RolloutPredictor(
        RecordingScorer(uniform_scores), budget=10, max_depth=1
    ).search_with_cost(TypedList([0]), TypedList([2]), FunctionDefSet([increment]))
    dead = RolloutPredictor(
        RecordingScorer(uniform_scores), budget=10, max_depth=2
    ).search_with_cost(TypedList([0]), TypedList([2]), FunctionDefSet([exploding]))

    assert depth == RolloutSearchResult(FunctionDefList(), 1, "max_depth")
    assert dead == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")


def test_no_type_compatible_actions_exhausts_the_frontier_without_execution():
    upper = FunctionDef(
        name="upper",
        input_type="builtins.str",
        output_type="builtins.str",
        code="return x.upper()",
    )

    result = RolloutPredictor(
        RecordingScorer(uniform_scores), budget=10, max_depth=1
    ).search_with_cost(TypedList([0]), TypedList([1]), FunctionDefSet([upper]))

    assert result == RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beam_width": 0}, "beam_width"),
        ({"beam_width": True}, "beam_width"),
        ({"max_depth": -1}, "max_depth"),
        ({"budget": -1}, "budget"),
    ],
)
def test_constructor_rejects_invalid_search_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RolloutPredictor(RecordingScorer(uniform_scores), **kwargs)


def test_zero_budget_and_zero_depth_stop_before_scoring():
    increment = make_function("increment", "return x + 1")
    functions = FunctionDefSet([increment])
    budget_scorer = RecordingScorer(uniform_scores)
    depth_scorer = RecordingScorer(uniform_scores)

    budget = RolloutPredictor(budget_scorer, budget=0).search_with_cost(
        TypedList([0]), TypedList([1]), functions
    )
    depth = RolloutPredictor(depth_scorer, max_depth=0).search_with_cost(
        TypedList([0]), TypedList([1]), functions
    )

    assert budget.termination == "budget"
    assert depth.termination == "max_depth"
    assert budget_scorer.calls == depth_scorer.calls == []


class FixedOutputScorer(ActionScorer):
    def __init__(self, output):
        self.output = output

    def score_actions_batch(self, problems, available_functions):
        return self.output


def test_scorer_batch_length_is_validated():
    increment = make_function("increment", "return x + 1")

    with pytest.raises(ValueError, match="returned 0 rows; expected 1"):
        RolloutPredictor(FixedOutputScorer([])).search_with_cost(
            TypedList([0]), TypedList([1]), FunctionDefSet([increment])
        )


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"increment": 0.0, "unknown": 1.0},
    ],
)
def test_scorer_function_names_are_validated(row):
    increment = make_function("increment", "return x + 1")

    with pytest.raises(ValueError, match="wrong function names"):
        RolloutPredictor(FixedOutputScorer([row])).search_with_cost(
            TypedList([0]), TypedList([1]), FunctionDefSet([increment])
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), "0"])
def test_scorer_scores_must_be_finite_numbers(score):
    increment = make_function("increment", "return x + 1")

    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        RolloutPredictor(FixedOutputScorer([{"increment": score}])).search_with_cost(
            TypedList([0]), TypedList([1]), FunctionDefSet([increment])
        )


def test_batch_predictions_have_independent_execution_budgets(monkeypatch):
    increment = make_function("increment", "return x + 1")
    functions = FunctionDefSet([increment])
    predictor = RolloutPredictor(
        RecordingScorer(uniform_scores), beam_width=1, max_depth=2, budget=2
    )
    recorded_results = []
    search_with_cost = predictor.search_with_cost

    def record_search(input_list, output_list, available_functions):
        result = search_with_cost(input_list, output_list, available_functions)
        recorded_results.append(result)
        return result

    monkeypatch.setattr(predictor, "search_with_cost", record_search)
    predictions = predictor.predict_functions_batch(
        [
            (TypedList([0]), TypedList([1])),
            (TypedList([0]), TypedList([2])),
        ],
        functions,
    )

    assert predictions == [
        FunctionDefList([increment]),
        FunctionDefList([increment, increment]),
    ]
    assert [result.executions for result in recorded_results] == [1, 2]


def test_batch_predictions_report_comparable_cost_and_search_config():
    increment = make_function("increment", "return x + 1")
    predictor = RolloutPredictor(
        RecordingScorer(uniform_scores), beam_width=2, max_depth=3, budget=7
    )

    prediction = predictor.predict_functions_batch_with_cost(
        [(TypedList([0]), TypedList([1]))],
        FunctionDefSet([increment]),
    )[0]

    assert prediction.function_defs == FunctionDefList([increment])
    assert prediction.cost.charged_executions == 1
    assert prediction.cost.candidate_programs is None
    assert prediction.cost.termination == "solved"
    assert prediction.cost.seconds is not None
    assert prediction.cost.seconds >= 0.0
    assert predictor.cost_metadata() == {
        "cost_unit": "basis_function_executions",
        "cost_definition": predictor.cost_definition,
        "budget_unit": "basis_function_executions",
        "budget_value": 7,
        "beam_width": 2,
        "max_depth": 3,
    }


def test_rollout_result_is_frozen():
    result = RolloutSearchResult(FunctionDefList(), 0, "solved")

    with pytest.raises(FrozenInstanceError):
        result.executions = 1
