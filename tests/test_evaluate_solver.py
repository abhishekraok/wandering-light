from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import BFSSolve, RandomSolve
from wandering_light.typed_list import TypedList

f1 = FunctionDef(
    name="inc",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 1",
)
available_functions = FunctionDefList([f1])


def test_evaluate_trivial():
    tl = TypedList([1, 2, 3])
    input_lists = [tl]
    solver = RandomSolve(budget=1, path_length=0)
    result = EvaluateSolver.evaluate_using_random_walk(
        solver,
        input_lists,
        num_samples=5,
        available_functions=available_functions,
        path_length=0,
    )
    assert result.success_count == 5
    assert result.success_rate == 1.0
    assert result.avg_solution_length == 0.0
    assert result.failures == []


def test_evaluate_bfs_normal():
    tl = TypedList([1, 2])
    input_lists = [tl]
    solver = BFSSolve(budget=16)
    result = EvaluateSolver.evaluate_using_random_walk(
        solver,
        input_lists,
        num_samples=4,
        available_functions=available_functions,
        path_length=2,
    )
    assert result.success_count == 4
    assert result.success_rate == 1.0
    assert result.avg_solution_length == 2.0
    assert result.failures == []


def test_evaluate_bfs_with_nested_dictionary_states():
    add_value = FunctionDef(
        name="add_value",
        input_type="builtins.dict",
        output_type="builtins.dict",
        code=(
            'return {"nested": {"values": [*x["nested"]["values"], 2]}, '
            '"name": x["name"]}'
        ),
    )
    input_lists = [
        TypedList([{"name": "example", "nested": {"values": [1]}}], item_type=dict)
    ]

    result = EvaluateSolver.evaluate_using_random_walk(
        BFSSolve(budget=4, max_depth=1),
        input_lists,
        num_samples=2,
        available_functions=FunctionDefSet([add_value]),
        path_length=1,
    )

    assert result.success_count == 2
    assert result.success_rate == 1.0
    assert result.failures == []
