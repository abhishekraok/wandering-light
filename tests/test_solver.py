from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import BFSPredictor, BFSSolve, RandomSolve
from wandering_light.typed_list import TypedList


def make_function(name, input_type, output_type, code):
    return FunctionDef(
        name=name,
        input_type=input_type,
        output_type=output_type,
        code=code,
    )


def test_random_solver_trivial():
    tl = TypedList([1, 2, 3])
    available_functions = FunctionDefList([])
    solver = RandomSolve(budget=5, path_length=1)
    result = solver.solve(tl, tl, available_functions)
    assert result.success
    traj = result.trajectory
    assert traj.function_defs == FunctionDefList([])
    assert traj.input == tl
    assert traj.output == tl


def test_random_solver_single_step():
    tl = TypedList([1, 2])
    target = TypedList([2, 3])
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    available_functions = FunctionDefList([f1])
    solver = RandomSolve(budget=5, path_length=1)
    result = solver.solve(tl, target, available_functions)
    assert result.success
    traj = result.trajectory
    assert len(traj.function_defs) == 1
    assert traj.function_defs[0].name == f1.name
    assert traj.output == target
    assert f1.usage_count == 1


def test_random_solver_multi_step():
    tl = TypedList([1, 2])
    target = TypedList([3, 4])
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    available_functions = FunctionDefList([f1])
    solver = RandomSolve(budget=5, path_length=2)
    result = solver.solve(tl, target, available_functions)
    assert result.success
    traj = result.trajectory
    assert len(traj.function_defs) == 2
    assert all(f.name == f1.name for f in traj.function_defs)
    assert traj.output == target
    assert f1.usage_count == 2


def test_random_solver_no_solution():
    tl = TypedList([1, 2])
    target = TypedList([2, 3])
    available_functions = FunctionDefList([])
    solver = RandomSolve(budget=3, path_length=1)
    result = solver.solve(tl, target, available_functions)
    assert not result.success
    assert "No solution found" in result.error_msg


# BFS solver tests
def test_bfs_solver_multi_step():
    tl = TypedList([1, 2, 3])
    target = TypedList([3, 8, 15])
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    f2 = make_function("square", "builtins.int", "builtins.int", "return x * x")
    f3 = make_function("dec", "builtins.int", "builtins.int", "return x - 1")
    available_functions = FunctionDefList([f1, f2, f3])
    solver = BFSSolve(budget=64)
    result = solver.solve(tl, target, available_functions)
    assert result.success
    traj = result.trajectory
    assert len(traj.function_defs) == 3
    assert [f.name for f in traj.function_defs] == [f1.name, f2.name, f3.name]
    assert traj.output == target


def test_bfs_solver_respects_budget():
    tl = TypedList([0])
    target = TypedList([10])
    f_inc = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    f_dec = make_function("dec", "builtins.int", "builtins.int", "return x - 1")
    available_functions = FunctionDefList([f_inc, f_dec])

    solver = BFSSolve(budget=5)
    result = solver.solve(tl, target, available_functions)

    assert not result.success
    assert "No solution found" in result.error_msg
    # Ensure no individual function exceeded the budget
    assert f_inc.usage_count <= 5
    assert f_dec.usage_count <= 5


# ================================
# Tests for corrected BFS behavior
# ================================


def test_bfs_incremental_execution():
    """
    Test that the corrected BFS implementation executes functions incrementally
    rather than re-executing full trajectories from scratch.
    """
    input_list = TypedList([0])
    target = TypedList([2])  # Need 2 increments: 0 -> 1 -> 2

    f_inc = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    available_functions = FunctionDefSet([f_inc])

    predictor = BFSPredictor(budget=10, max_depth=3)

    # Track individual execute calls to verify incremental execution
    execute_calls = []

    class ExecutorWrapper:
        def __init__(self, original_executor):
            self.original_executor = original_executor

        def execute(self, func, typed_list):
            execute_calls.append((func.name, typed_list.items.copy()))
            return self.original_executor.execute(func, typed_list)

    # Mock the Executor to track execute calls
    original_executor_class = predictor._bfs_search.__globals__["Executor"]

    class MockExecutorClass:
        def __init__(self, available_functions):
            real_executor = original_executor_class(available_functions)
            self.wrapped = ExecutorWrapper(real_executor)

        def execute(self, func, typed_list):
            return self.wrapped.execute(func, typed_list)

    predictor._bfs_search.__globals__["Executor"] = MockExecutorClass

    try:
        result = predictor._bfs_search(input_list, target, available_functions)

        # Verify that execution was incremental:
        # Should be: inc([0]) -> [1], then inc([1]) -> [2]
        # Not: inc([0]) -> [1], then re-execute inc,inc([0]) -> [2]
        assert len(execute_calls) == 2
        assert execute_calls[0] == ("inc", [0])  # First call: inc on [0]
        assert execute_calls[1] == (
            "inc",
            [1],
        )  # Second call: inc on [1] (incremental!)

        # Verify correct result
        assert len(result.functions) == 2
        assert all(f.name == "inc" for f in result.functions)

    finally:
        # Restore original Executor
        predictor._bfs_search.__globals__["Executor"] = original_executor_class


def test_bfs_visited_state_tracking():
    """
    Test that the corrected BFS implementation tracks visited states
    to avoid redundant exploration.
    """
    input_list = TypedList([2])
    target = TypedList([4])  # Can reach via multiple paths

    f_inc = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    f_dec = make_function("dec", "builtins.int", "builtins.int", "return x - 1")
    available_functions = FunctionDefSet([f_inc, f_dec])

    predictor = BFSPredictor(budget=20, max_depth=5)

    # Count total function executions
    original_usage_inc = f_inc.usage_count
    original_usage_dec = f_dec.usage_count

    result = predictor._bfs_search(input_list, target, available_functions)

    total_executions = (
        f_inc.usage_count - original_usage_inc + f_dec.usage_count - original_usage_dec
    )

    # With visited tracking, should explore efficiently
    # The optimal path is [inc, inc] (2 -> 3 -> 4)
    # With proper visited tracking, total executions should be reasonable
    # (much less than without visited tracking)
    assert total_executions <= 10  # Should be efficient
    assert len(result.functions) == 2  # Optimal solution length
    assert all(f.name == "inc" for f in result.functions)


def test_bfs_type_compatibility_filtering():
    """
    Test that the corrected BFS implementation only tries type-compatible functions.
    """
    input_list = TypedList([1])  # int type
    target = TypedList([2])

    # Mix compatible and incompatible functions
    f_inc = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    f_upper = make_function("upper", "builtins.str", "builtins.str", "return x.upper()")
    f_len = make_function("len", "builtins.list", "builtins.int", "return len(x)")

    available_functions = FunctionDefSet([f_inc, f_upper, f_len])

    predictor = BFSPredictor(budget=10, max_depth=2)

    # Track which functions are actually executed
    original_usage_inc = f_inc.usage_count
    original_usage_upper = f_upper.usage_count
    original_usage_len = f_len.usage_count

    result = predictor._bfs_search(input_list, target, available_functions)

    # Only the compatible function (inc) should be executed
    inc_executions = f_inc.usage_count - original_usage_inc
    upper_executions = f_upper.usage_count - original_usage_upper
    len_executions = f_len.usage_count - original_usage_len

    assert inc_executions > 0  # Compatible function should be tried
    assert upper_executions == 0  # Incompatible function should not be tried
    assert len_executions == 0  # Incompatible function should not be tried

    # Should find the solution using only the compatible function
    assert len(result.functions) == 1
    assert result.functions[0].name == "inc"


def test_bfs_optimal_path_selection():
    """
    Test that BFS finds the shortest path when multiple solutions exist.
    This is a core property of breadth-first search.
    """
    input_list = TypedList([0])
    target = TypedList([3])

    # Multiple ways to reach target:
    # - [add_three]: 1 step (optimal)
    # - [inc, inc, inc]: 3 steps
    # - Other longer combinations
    f_inc = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    f_dec = make_function("dec", "builtins.int", "builtins.int", "return x - 1")
    f_add_three = make_function(
        "add_three", "builtins.int", "builtins.int", "return x + 3"
    )

    available_functions = FunctionDefSet([f_inc, f_dec, f_add_three])

    predictor = BFSPredictor(budget=20, max_depth=5)
    result = predictor._bfs_search(input_list, target, available_functions)

    # Should find the optimal (shortest) solution
    assert len(result.functions) == 1
    assert result.functions[0].name == "add_three"


def test_bfs_efficiency_improvement():
    """
    Test that the corrected implementation is more efficient than the buggy version
    by using fewer function executions for the same problem.
    """
    input_list = TypedList([1])
    target = TypedList([4])  # Need 3 increments

    f_inc = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    available_functions = FunctionDefSet([f_inc])

    predictor = BFSPredictor(budget=15, max_depth=4)

    original_usage = f_inc.usage_count
    result = predictor._bfs_search(input_list, target, available_functions)
    total_executions = f_inc.usage_count - original_usage

    # The corrected implementation should be efficient:
    # With incremental execution: 3 executions (1->2, 2->3, 3->4)
    # With visited tracking: minimal redundant exploration
    # Total should be small and reasonable
    assert total_executions <= 6  # Should be efficient

    # Should still find correct solution
    assert len(result.functions) == 3
    assert all(f.name == "inc" for f in result.functions)
