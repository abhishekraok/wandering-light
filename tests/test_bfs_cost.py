from dataclasses import FrozenInstanceError

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import BFSPredictor, BFSSearchResult, BFSSolve
from wandering_light.typed_list import TypedList


def make_function(name: str, input_type: str, output_type: str, code: str):
    return FunctionDef(
        name=name,
        input_type=input_type,
        output_type=output_type,
        code=code,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"budget": -1}, "budget"), ({"max_depth": -1}, "max_depth")],
)
def test_search_rejects_negative_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BFSPredictor(**kwargs)


def test_zero_distance_is_solved_without_an_execution():
    value = TypedList([1])

    result = BFSPredictor().search_with_cost(value, value, FunctionDefSet())

    assert result == BFSSearchResult(FunctionDefList(), 0, "solved")


def test_search_counts_each_compatible_function_tried():
    upper = make_function("upper", "builtins.str", "builtins.str", "return x.upper()")
    identity = make_function("identity", "builtins.int", "builtins.int", "return x")
    increment = make_function(
        "increment", "builtins.int", "builtins.int", "return x + 1"
    )
    functions = FunctionDefSet([upper, identity, increment])

    result = BFSPredictor(budget=10, max_depth=2).search_with_cost(
        TypedList([0]), TypedList([1]), functions
    )

    assert result.function_defs == FunctionDefList([increment])
    assert result.executions == 2
    assert result.termination == "solved"


def test_raising_transition_counts_and_can_exhaust_the_budget():
    exploding = make_function(
        "exploding", "builtins.int", "builtins.int", "return x + None"
    )
    increment = make_function(
        "increment", "builtins.int", "builtins.int", "return x + 1"
    )

    result = BFSPredictor(budget=1, max_depth=1).search_with_cost(
        TypedList([0]),
        TypedList([1]),
        FunctionDefSet([exploding, increment]),
    )

    assert result.function_defs == FunctionDefList()
    assert result.executions == 1
    assert result.termination == "budget"


def test_solution_on_the_last_execution_wins_over_budget_termination():
    increment = make_function(
        "increment", "builtins.int", "builtins.int", "return x + 1"
    )

    result = BFSPredictor(budget=1, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([1]), FunctionDefSet([increment])
    )

    assert result.function_defs == FunctionDefList([increment])
    assert result.executions == 1
    assert result.termination == "solved"


def test_search_reports_max_depth_when_deeper_states_remain():
    increment = make_function(
        "increment", "builtins.int", "builtins.int", "return x + 1"
    )

    result = BFSPredictor(budget=10, max_depth=1).search_with_cost(
        TypedList([0]), TypedList([2]), FunctionDefSet([increment])
    )

    assert result.function_defs == FunctionDefList()
    assert result.executions == 1
    assert result.termination == "max_depth"


def test_search_reports_frontier_exhaustion_when_no_transitions_survive():
    exploding = make_function(
        "exploding", "builtins.int", "builtins.int", "return x + None"
    )

    result = BFSPredictor(budget=10, max_depth=2).search_with_cost(
        TypedList([0]), TypedList([1]), FunctionDefSet([exploding])
    )

    assert result.function_defs == FunctionDefList()
    assert result.executions == 1
    assert result.termination == "frontier_exhausted"


def test_result_is_frozen():
    value = TypedList([1])
    result = BFSPredictor().search_with_cost(value, value, FunctionDefSet())

    with pytest.raises(FrozenInstanceError):
        result.executions = 1


def test_repeated_and_batch_searches_have_independent_execution_counts(monkeypatch):
    increment = make_function(
        "increment", "builtins.int", "builtins.int", "return x + 1"
    )
    functions = FunctionDefSet([increment])
    predictor = BFSPredictor(budget=10, max_depth=2)
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
            (TypedList([0]), TypedList([1])),
        ],
        functions,
    )

    assert [result.executions for result in recorded_results] == [1, 2, 1]
    assert predictions == [
        FunctionDefList([increment]),
        FunctionDefList([increment, increment]),
        FunctionDefList([increment]),
    ]


def test_solver_result_preserves_bfs_cost_without_charging_verification():
    identity = make_function("identity", "builtins.int", "builtins.int", "return x")
    increment = make_function(
        "increment", "builtins.int", "builtins.int", "return x + 1"
    )
    solver = BFSSolve(budget=10, max_depth=1)

    result = solver.solve(
        TypedList([0]),
        TypedList([1]),
        FunctionDefSet([identity, increment]),
    )

    assert result.success
    assert result.cost.charged_executions == 2
    assert result.cost.candidate_programs is None
    assert result.cost.termination == "solved"
    assert solver.cost_metadata()["budget_unit"] == "basis_function_executions"
    assert solver.cost_metadata()["max_depth"] == 1
