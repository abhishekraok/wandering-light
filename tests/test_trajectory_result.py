"""Tests for TrajectoryResult and the updated execute_trajectory behavior."""

import pytest

from wandering_light.executor import Executor, TrajectoryResult
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList


@pytest.fixture
def basic_functions():
    """Basic functions for testing."""
    increment_fn = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    double_fn = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )
    str_upper_fn = FunctionDef(
        name="str_upper",
        input_type="builtins.str",
        output_type="builtins.str",
        code="return x.upper()",
    )
    return [increment_fn, double_fn, str_upper_fn]


@pytest.fixture
def executor_with_functions(basic_functions):
    """Executor with basic functions."""
    return Executor(basic_functions)


class TestTrajectoryResult:
    """Test the TrajectoryResult class."""

    def test_success_result(self):
        """Test creating a success result."""
        # Create a dummy trajectory for testing
        tl = TypedList([1, 2, 3])
        spec = TrajectorySpec(tl, FunctionDefList([]))
        from wandering_light.trajectory import Trajectory

        trajectory = Trajectory(spec, tl)

        result = TrajectoryResult.success_result(trajectory)

        assert result.success is True
        assert result.trajectory == trajectory
        assert result.error_msg is None
        assert result.failed_at_step is None

    def test_failure_result(self):
        """Test creating a failure result."""
        error_msg = "Type mismatch error"
        failed_step = 2

        result = TrajectoryResult.failure_result(error_msg, failed_step)

        assert result.success is False
        assert result.trajectory is None
        assert result.error_msg == error_msg
        assert result.failed_at_step == failed_step

    def test_failure_result_no_step(self):
        """Test creating a failure result without specifying failed step."""
        error_msg = "General error"

        result = TrajectoryResult.failure_result(error_msg)

        assert result.success is False
        assert result.trajectory is None
        assert result.error_msg == error_msg
        assert result.failed_at_step is None


class TestExecuteTrajectorySuccess:
    """Test successful trajectory execution."""

    def test_empty_trajectory(self, executor_with_functions):
        """Test executing an empty trajectory."""
        tl = TypedList([1, 2, 3])
        spec = TrajectorySpec(tl, FunctionDefList([]))

        result = executor_with_functions.execute_trajectory(spec)

        assert result.success is True
        assert result.trajectory is not None
        assert result.trajectory.output == tl
        assert result.error_msg is None
        assert result.failed_at_step is None

    def test_single_function_success(self, executor_with_functions, basic_functions):
        """Test executing a single function successfully."""
        tl = TypedList([1, 2, 3])
        increment_fn = basic_functions[0]  # increment function
        spec = TrajectorySpec(tl, FunctionDefList([increment_fn]))

        result = executor_with_functions.execute_trajectory(spec)

        assert result.success is True
        assert result.trajectory is not None
        assert result.trajectory.output.items == [2, 3, 4]
        assert result.error_msg is None
        assert result.failed_at_step is None

    def test_multi_function_success(self, executor_with_functions, basic_functions):
        """Test executing multiple functions successfully."""
        tl = TypedList([1, 2, 3])
        increment_fn, double_fn = basic_functions[:2]
        spec = TrajectorySpec(tl, FunctionDefList([increment_fn, double_fn]))

        result = executor_with_functions.execute_trajectory(spec)

        assert result.success is True
        assert result.trajectory is not None
        assert result.trajectory.output.items == [4, 6, 8]  # (1+1)*2, (2+1)*2, (3+1)*2
        assert result.error_msg is None
        assert result.failed_at_step is None


class TestExecuteTrajectoryFailure:
    """Test trajectory execution failures."""

    def test_type_mismatch_first_function(
        self, executor_with_functions, basic_functions
    ):
        """Test type mismatch on the first function."""
        tl = TypedList(["hello", "world"])  # String input
        increment_fn = basic_functions[0]  # Expects int input
        spec = TrajectorySpec(tl, FunctionDefList([increment_fn]))

        result = executor_with_functions.execute_trajectory(spec)

        assert result.success is False
        assert result.trajectory is None
        assert "Type mismatch at function 'increment'" in result.error_msg
        assert result.failed_at_step == 0

    def test_type_mismatch_second_function(
        self, executor_with_functions, basic_functions
    ):
        """Test type mismatch on the second function."""
        tl = TypedList([1, 2, 3])
        increment_fn = basic_functions[0]  # int -> int
        str_upper_fn = basic_functions[2]  # str -> str, but gets int input
        spec = TrajectorySpec(tl, FunctionDefList([increment_fn, str_upper_fn]))

        result = executor_with_functions.execute_trajectory(spec)

        assert result.success is False
        assert result.trajectory is None
        assert "Type mismatch at function 'str_upper'" in result.error_msg
        assert result.failed_at_step == 1

    def test_function_runtime_error(self, executor_with_functions):
        """Test function that raises a runtime error."""
        error_fn = FunctionDef(
            name="error_function",
            input_type="builtins.int",
            output_type="builtins.int",
            code="raise ValueError('Intentional error')",
        )
        executor = Executor([error_fn])

        tl = TypedList([1, 2, 3])
        spec = TrajectorySpec(tl, FunctionDefList([error_fn]))

        result = executor.execute_trajectory(spec)

        assert result.success is False
        assert result.trajectory is None
        assert "Function 'error_function' failed" in result.error_msg
        assert "Intentional error" in result.error_msg
        assert result.failed_at_step == 0

    def test_function_error_second_step(self, executor_with_functions, basic_functions):
        """Test function error on the second step."""
        error_fn = FunctionDef(
            name="error_function",
            input_type="builtins.int",
            output_type="builtins.int",
            code="raise RuntimeError('Second step error')",
        )
        executor = Executor([*basic_functions, error_fn])

        tl = TypedList([1, 2, 3])
        increment_fn = basic_functions[0]
        spec = TrajectorySpec(tl, FunctionDefList([increment_fn, error_fn]))

        result = executor.execute_trajectory(spec)

        assert result.success is False
        assert result.trajectory is None
        assert "Function 'error_function' failed" in result.error_msg
        assert "Second step error" in result.error_msg
        assert result.failed_at_step == 1


class TestIntegrationWithSolver:
    """Test integration with solver components."""

    def test_solver_with_trajectory_result(self, basic_functions):
        """Test that solver works with new TrajectoryResult."""
        from wandering_light.solver import BFSPredictor, TrajectorySolver

        predictor = BFSPredictor(budget=10, max_depth=2)
        solver = TrajectorySolver(predictor)

        function_set = FunctionDefSet(basic_functions)
        input_list = TypedList([1])
        output_list = TypedList([4])  # increment then double: (1+1)*2 = 4

        result = solver.solve(input_list, output_list, function_set)

        assert result.success is True
        assert result.trajectory is not None
        assert result.trajectory.output == output_list

    def test_solver_with_impossible_problem(self, basic_functions):
        """Test solver gracefully handles impossible problems."""
        from wandering_light.solver import BFSPredictor, TrajectorySolver

        predictor = BFSPredictor(budget=10, max_depth=2)
        solver = TrajectorySolver(predictor)

        function_set = FunctionDefSet(basic_functions)
        input_list = TypedList([1])
        output_list = TypedList([999])  # Impossible with basic operations

        result = solver.solve(input_list, output_list, function_set)

        assert result.success is False
        # Should handle gracefully without crashing
