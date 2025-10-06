from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.function_def import FunctionDef, FunctionDefList
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
