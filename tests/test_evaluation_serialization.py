import json
import os

from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.solver import BFSSolve, RandomSolve
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList


class TestEvaluationSerialization:
    """Test suite for evaluation result serialization and inspection."""

    def test_serialize_eval_result(self):
        """Test using the built-in save functionality of EvalResult."""
        # Create sample functions
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
        specs = TrajectorySpecList()
        specs.append(TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func])))
        specs.append(TrajectorySpec(TypedList([3, 4]), FunctionDefList([double_func])))

        # Run evaluation with built-in save functionality
        solver = BFSSolve(budget=10)
        output_dir = "test_outputs"
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver,
            specs,
            available_functions=FunctionDefList([inc_func, double_func]),
            save_results=True,
            output_dir=output_dir,
        )

        # Test the save_to_file method directly
        output_file = os.path.join(output_dir, "test_eval_result.json")
        result.save_to_file(output_file)

        # Verify file exists and contains expected data
        assert os.path.exists(output_file)
        with open(output_file) as f:
            loaded_data = json.load(f)
            assert "total_samples" in loaded_data
            assert "success_count" in loaded_data
            assert "success_rate" in loaded_data
            assert "avg_solution_length" in loaded_data
            assert "detailed_results" in loaded_data
            assert len(loaded_data["detailed_results"]) == len(specs)

    def test_serialize_multiple_solvers(self):
        """Test serializing results from multiple solvers using built-in functionality."""
        # Create sample functions
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        specs = TrajectorySpecList()
        specs.append(TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func])))

        solvers = {"random": RandomSolve(budget=10), "bfs": BFSSolve(budget=10)}

        output_dir = "test_outputs"
        results = {}
        for name, solver in solvers.items():
            result = EvaluateSolver.evaluate_using_trajectory_specs(
                solver,
                specs,
                available_functions=FunctionDefList([inc_func]),
                save_results=True,
                output_dir=output_dir,
            )
            results[name] = result

        # Test manual saving using to_dict() method
        combined_results = {}
        for name, result in results.items():
            result_dict = result.to_dict()
            result_dict["solver_name"] = name
            combined_results[name] = result_dict

        # Save combined results using built-in JSON functionality
        output_file = os.path.join(output_dir, "multi_solver_results.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(combined_results, f, indent=2)

        # Verify file contains results from both solvers
        with open(output_file) as f:
            loaded_data = json.load(f)
            assert "random" in loaded_data
            assert "bfs" in loaded_data
            assert loaded_data["random"]["solver_name"] == "random"
            assert loaded_data["bfs"]["solver_name"] == "bfs"

    def test_eval_result_to_dict_method(self):
        """Test the to_dict() method for proper serialization."""
        # Create sample functions and specs
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        specs = TrajectorySpecList()
        specs.append(TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func])))

        # Run evaluation
        solver = BFSSolve(budget=10)
        result = EvaluateSolver.evaluate_using_trajectory_specs(
            solver, specs, available_functions=FunctionDefList([inc_func])
        )

        # Test to_dict() method
        result_dict = result.to_dict()

        # Verify all expected fields are present and serializable
        assert isinstance(result_dict, dict)
        assert "total_samples" in result_dict
        assert "success_count" in result_dict
        assert "success_rate" in result_dict
        assert "avg_solution_length" in result_dict
        assert "failures" in result_dict
        assert "detailed_results" in result_dict

        # Ensure it's JSON serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None

        # Verify failures are properly converted to strings
        for failure in result_dict["failures"]:
            assert isinstance(failure, list | tuple)
            assert len(failure) == 2
            assert isinstance(failure[0], str)  # TrajectorySpec as string
            assert isinstance(failure[1], str)  # Exception as string
