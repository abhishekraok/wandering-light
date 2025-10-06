import os
import tempfile

from wandering_light.evals.evaluate_solver import (
    EvaluateSolver,
    evaluate_solver_from_file,
)
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import BFSSolve, RandomSolve
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList


class TestEvaluateSolverFile:
    """Test suite for file-based solver evaluation functionality."""

    def test_evaluate_using_trajectory_specs(self):
        """Test evaluation using TrajectorySpecList directly."""
        # Create simple functions
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        double_func = FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        )

        # Create trajectory specs
        specs = [
            TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func])),
            TrajectorySpec(TypedList([3, 4]), FunctionDefList([double_func])),
            TrajectorySpec(TypedList([5, 6]), FunctionDefList([inc_func, double_func])),
        ]

        spec_list = TrajectorySpecList(specs)

        # Test with BFS solver (should succeed on simple cases)
        solver = BFSSolve(budget=50)
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver,
            spec_list,
            available_functions=FunctionDefSet([inc_func, double_func]),
        )

        assert result.total_samples == 3
        assert result.success_count == 3
        assert result.success_rate == 1.0
        assert result.avg_solution_length > 1.0

        # Test detailed results have correct function tracking
        assert len(result.detailed_results) == 3

        # Check first spec: [inc]
        assert result.detailed_results[0].golden_functions == ["inc"]
        assert result.detailed_results[0].predicted_functions is not None
        assert result.detailed_results[0].success

        # Check second spec: [double]
        assert result.detailed_results[1].golden_functions == ["double"]
        assert result.detailed_results[1].predicted_functions is not None
        assert result.detailed_results[1].success

        # Check third spec: [inc, double]
        assert result.detailed_results[2].golden_functions == ["inc", "double"]
        assert result.detailed_results[2].predicted_functions is not None
        assert result.detailed_results[2].success

    def test_evaluate_using_trajectory_specs_failure(self):
        """Test evaluation with a solver that fails to find solutions."""
        # Create functions
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        double_func = FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        )

        # Create trajectory specs that require multiple steps
        specs = [
            TrajectorySpec(
                TypedList([1, 2]), FunctionDefList([inc_func, double_func, inc_func])
            ),
            TrajectorySpec(
                TypedList([3, 4]), FunctionDefList([double_func, inc_func, double_func])
            ),
        ]

        spec_list = TrajectorySpecList(specs)

        # Test with Random solver with very low budget (should fail)
        solver = RandomSolve(budget=1)  # Too low budget to find solutions
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver,
            spec_list,
            available_functions=FunctionDefSet([inc_func, double_func]),
        )

        assert result.total_samples == 2
        assert result.success_count == 0
        assert result.success_rate == 0.0
        assert result.avg_solution_length == 0.0  # No successful solutions
        assert len(result.detailed_results) == 2
        assert not result.detailed_results[0].success
        assert not result.detailed_results[1].success
        assert result.detailed_results[0].actual_output is not None
        assert result.detailed_results[1].actual_output is not None
        assert result.detailed_results[0].expected_output is not None
        assert result.detailed_results[1].expected_output is not None
        assert (
            result.detailed_results[0].actual_output
            != result.detailed_results[0].expected_output
        )
        assert (
            result.detailed_results[1].actual_output
            != result.detailed_results[1].expected_output
        )

        # Test that golden_functions are populated correctly with ground truth
        assert result.detailed_results[0].golden_functions == ["inc", "double", "inc"]
        assert result.detailed_results[1].golden_functions == [
            "double",
            "inc",
            "double",
        ]

        # Test that predicted_functions are populated (solver tried something)
        assert result.detailed_results[0].predicted_functions is not None
        assert result.detailed_results[1].predicted_functions is not None

    def test_evaluate_using_trajectory_specs_with_sampling(self):
        """Test evaluation with num_samples parameter."""
        # Create multiple specs
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        specs = []
        for i in range(10):
            specs.append(
                TrajectorySpec(TypedList([i, i + 1]), FunctionDefList([inc_func]))
            )

        spec_list = TrajectorySpecList(specs)

        # Test with limited samples
        solver = BFSSolve(budget=20)
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver,
            spec_list,
            num_samples=3,
            available_functions=FunctionDefSet([inc_func]),
        )

        assert result.total_samples == 3
        assert len(spec_list) == 10  # Original list unchanged

    def test_evaluate_solver_from_file(self):
        """Test evaluation from serialized file."""
        # Create test data
        inc_func = FunctionDef(
            name="test_inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        specs = [
            TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func])),
            TrajectorySpec(TypedList([10, 20]), FunctionDefList([inc_func])),
        ]

        spec_list = TrajectorySpecList(specs)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name

        try:
            spec_list.to_py_file(temp_path, "test_eval_data")

            # Test file-based evaluation
            result = evaluate_solver_from_file(
                eval_file=temp_path,
                solver_name="bfs",
                num_samples=2,
                budget=20,
                variable_name="test_eval_data",
            )

            assert result is not None
            assert result.total_samples == 2

        finally:
            os.unlink(temp_path)

    def test_evaluate_solver_from_file_error_handling(self):
        """Test error handling in file-based evaluation."""
        # Test with non-existent file
        result = evaluate_solver_from_file(
            eval_file="non_existent_file.py", solver_name="bfs"
        )

        assert result is None

    def test_function_extraction_from_specs(self):
        """Test that functions are correctly extracted from trajectory specs."""
        # Create specs with overlapping functions
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )
        double_func = FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        )

        specs = [
            TrajectorySpec(TypedList([1]), FunctionDefList([inc_func])),
            TrajectorySpec(TypedList([2]), FunctionDefList([double_func])),
            TrajectorySpec(
                TypedList([3]), FunctionDefList([inc_func, double_func])
            ),  # Reuses inc_func
        ]

        spec_list = TrajectorySpecList(specs)
        solver = BFSSolve(budget=10)

        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver,
            spec_list,
            available_functions=FunctionDefSet([inc_func, double_func]),
        )

        # Should work even with overlapping functions
        assert result.total_samples == 3

    def test_empty_trajectory_specs_list(self):
        """Test evaluation with empty trajectory specs list."""
        empty_list = TrajectorySpecList()
        solver = BFSSolve(budget=10)

        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver, empty_list, available_functions=FunctionDefSet([])
        )

        assert result.total_samples == 0
        assert result.success_count == 0
        assert result.success_rate == 0.0
        assert result.avg_solution_length == 0.0
        assert len(result.failures) == 0

    def test_detailed_result_function_fields(self):
        """Test that predicted_functions and golden_functions are correctly populated."""
        # Create functions with distinct names
        add_one = FunctionDef(
            name="add_one",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        multiply_by_three = FunctionDef(
            name="multiply_by_three",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 3",
        )

        # Create a spec with a specific function sequence
        spec = TrajectorySpec(
            TypedList([2]), FunctionDefList([add_one, multiply_by_three])
        )
        spec_list = TrajectorySpecList([spec])

        # Use BFS solver with sufficient budget to find the solution
        solver = BFSSolve(budget=100)
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver,
            spec_list,
            available_functions=FunctionDefSet([add_one, multiply_by_three]),
        )

        # Verify basic success
        assert result.total_samples == 1
        assert result.success_count == 1
        assert len(result.detailed_results) == 1

        detailed = result.detailed_results[0]

        # Test golden_functions matches the ground truth spec
        assert detailed.golden_functions == ["add_one", "multiply_by_three"]

        # Test predicted_functions is populated and contains valid function names
        assert detailed.predicted_functions is not None
        assert len(detailed.predicted_functions) > 0
        assert all(
            func in ["add_one", "multiply_by_three"]
            for func in detailed.predicted_functions
        )

        # Test that both fields are lists of strings
        assert isinstance(detailed.golden_functions, list)
        assert isinstance(detailed.predicted_functions, list)
        assert all(isinstance(f, str) for f in detailed.golden_functions)
        assert all(isinstance(f, str) for f in detailed.predicted_functions)

    def test_detailed_result_error_cases(self):
        """Test that function fields handle error cases correctly."""
        # Create a function that will cause execution errors
        error_func = FunctionDef(
            name="divide_by_zero",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x / 0",  # This will cause a division by zero error
        )

        spec = TrajectorySpec(TypedList([5]), FunctionDefList([error_func]))
        spec_list = TrajectorySpecList([spec])

        solver = BFSSolve(budget=10)
        # TODO: this should raise an error in the future
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver, spec_list, available_functions=FunctionDefSet([error_func])
        )

        assert result.total_samples == 0
        assert len(result.detailed_results) == 0
