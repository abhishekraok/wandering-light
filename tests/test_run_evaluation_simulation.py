"""
Test to simulate run_evaluation.py as faithfully as possible.
Only mocks the LLM response to output ground truth, then manually calculates
average solution length to verify the metric calculation is correct.
"""

# Import what we need to simulate run_evaluation.py
from wandering_light.common_functions import basic_fns
from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.executor import Executor
from wandering_light.solver import (
    TokenGenerator,
    TokenGeneratorSolver,
)
from wandering_light.trajectory import TrajectorySpecList


class GroundTruthTokenGenerator(TokenGenerator):
    """Mock token generator that always outputs the ground truth solution."""

    def __init__(self):
        self.llm_io_history = []
        self.specs_and_solutions = {}  # Will store {spec_id: ground_truth_functions}
        self.current_spec_index = 0
        self.trajectory_specs = None

    def set_trajectory_specs(self, trajectory_specs: TrajectorySpecList):
        """Set the trajectory specs so we can return ground truth for each."""
        self.trajectory_specs = trajectory_specs
        # Pre-compute ground truth for each spec
        for i, spec in enumerate(trajectory_specs.specs):
            function_names = [f.name for f in spec.function_defs]
            self.specs_and_solutions[i] = ",".join(function_names)

    def generate(self, prompt: str) -> str:
        """Return the ground truth solution for the current spec."""
        # Extract the spec from the prompt if possible, or use sequential ordering
        if self.current_spec_index in self.specs_and_solutions:
            solution = self.specs_and_solutions[self.current_spec_index]
        else:
            # Fallback - shouldn't happen in our test
            solution = "identity_int"

        self.llm_io_history.append((prompt, solution))
        self.current_spec_index += 1
        return solution


class TestRunEvaluationSimulation:
    """Test suite that simulates run_evaluation.py as faithfully as possible."""

    def test_average_solution_length_calculation_simulation(self):
        """
        Simulate run_evaluation.py with mocked LLM that outputs ground truth,
        then manually calculate average solution length and verify it matches.
        """
        # Use the actual eval file as specified by user
        eval_file = "wandering_light/evals/data/random_inputs.py"

        # Load the real evaluation data (exactly like run_evaluation.py does)
        trajectories, available_functions = load_eval_data_as_trajectories(eval_file)
        trajectory_specs = trajectories.to_spec_list()

        # Add basic functions (exactly like run_evaluation.py does)
        available_functions.extend(basic_fns)

        num_samples = len(trajectory_specs.specs)
        specs_subset = trajectory_specs.specs
        trajectory_specs_subset = TrajectorySpecList(specs_subset)

        print(f"Testing with {len(specs_subset)} trajectory specifications")
        print(f"Available functions: {len(available_functions)} total")

        # Create our ground truth token generator
        mock_token_generator = GroundTruthTokenGenerator()
        mock_token_generator.set_trajectory_specs(trajectory_specs_subset)

        # Create solver with mocked LLM (this is the ONLY thing we mock)
        solver = TokenGeneratorSolver(
            token_generator=mock_token_generator,
            budget=1,  # Only need 1 attempt since we're returning ground truth
        )

        # Run evaluation using the REAL evaluation infrastructure
        result = EvaluateSolver.evaluate_using_trajectories(
            solver,
            trajectories,
            available_functions=available_functions,
            num_samples=num_samples,
            save_results=False,
        )

        # Manually calculate expected metrics to verify correctness
        expected_successes = 0
        expected_total_length = 0
        manual_detailed_results = []

        # Create executor to verify ground truth
        executor = Executor(available_functions)

        for i, spec in enumerate(specs_subset):
            try:
                # Execute ground truth to get expected output
                ground_truth_result = executor.execute_trajectory(spec)
                if not ground_truth_result.success:
                    raise ValueError(
                        f"Ground truth execution failed: {ground_truth_result.error_msg}"
                    )
                expected_output = ground_truth_result.trajectory.output

                # The mock should have returned the ground truth function names
                expected_function_names = [f.name for f in spec.function_defs]
                expected_solution_length = len(spec.function_defs)

                # Since our mock returns ground truth, this should always succeed
                expected_successes += 1
                expected_total_length += expected_solution_length

                manual_detailed_results.append(
                    {
                        "spec_index": i,
                        "input": str(spec.input),
                        "expected_output": str(expected_output),
                        "ground_truth_functions": expected_function_names,
                        "solution_length": expected_solution_length,
                        "should_succeed": True,
                    }
                )

            except Exception as e:
                print(f"Error with spec {i}: {e}")
                manual_detailed_results.append(
                    {
                        "spec_index": i,
                        "input": str(spec.input),
                        "error": str(e),
                        "should_succeed": False,
                    }
                )

        expected_avg_length = (
            expected_total_length / expected_successes
            if expected_successes > 0
            else 0.0
        )

        # Print manual calculation details
        print("\n=== Manual Calculation ===")
        print(f"Expected successes: {expected_successes}")
        print(f"Expected total length: {expected_total_length}")
        print(f"Expected average length: {expected_avg_length:.4f}")

        # Print some example solutions
        print("\n=== Example Solutions ===")
        for i, detail in enumerate(manual_detailed_results[:5]):
            if detail.get("should_succeed", False):
                print(
                    f"Spec {i}: {detail['solution_length']} functions -> {detail['ground_truth_functions']}"
                )

        # Print actual results from evaluation
        print("\n=== Actual Evaluation Results ===")
        print(f"Total samples: {result.total_samples}")
        print(f"Success count: {result.success_count}")
        print(f"Success rate: {result.success_rate:.4f}")
        print(f"Average solution length: {result.avg_solution_length:.4f}")
        print(f"Failures: {len(result.failures)}")

        # Verify the calculations match
        assert result.total_samples == num_samples, (
            f"Expected {num_samples} samples, got {result.total_samples}"
        )
        assert result.success_count == expected_successes, (
            f"Expected {expected_successes} successes, got {result.success_count}"
        )

        # This is the key assertion - verify average solution length calculation
        assert abs(result.avg_solution_length - expected_avg_length) < 0.0001, (
            f"Average solution length mismatch: expected {expected_avg_length:.4f}, got {result.avg_solution_length:.4f}"
        )

        # Verify success rate
        expected_success_rate = expected_successes / num_samples
        assert abs(result.success_rate - expected_success_rate) < 0.0001, (
            f"Success rate mismatch: expected {expected_success_rate:.4f}, got {result.success_rate:.4f}"
        )

        # Verify detailed results
        assert len(result.detailed_results) == num_samples, (
            f"Expected {num_samples} detailed results, got {len(result.detailed_results)}"
        )

        # Check that successful detailed results have correct solution lengths
        successful_detailed = [dr for dr in result.detailed_results if dr.success]
        manual_successful = [
            dr for dr in manual_detailed_results if dr.get("should_succeed", False)
        ]

        assert len(successful_detailed) == len(manual_successful), (
            f"Mismatch in successful results count: {len(successful_detailed)} vs {len(manual_successful)}"
        )

        # Verify individual solution lengths
        for i, (actual_detail, manual_detail) in enumerate(
            zip(successful_detailed, manual_successful, strict=False)
        ):
            assert actual_detail.solution_length == manual_detail["solution_length"], (
                f"Solution length mismatch for result {i}: expected {manual_detail['solution_length']}, got {actual_detail.solution_length}"
            )

        print(
            "\n✅ All metrics verified! Average solution length calculation is correct."
        )
        print(
            f"✅ Mock LLM successfully output ground truth for all {expected_successes} successful cases."
        )

        # Additional verification: check that the mock was called correctly
        assert len(mock_token_generator.llm_io_history) == expected_successes, (
            f"Expected {expected_successes} LLM calls, got {len(mock_token_generator.llm_io_history)}"
        )

        # Verify that the mock returned the correct ground truth for each call
        for i, (_prompt, response) in enumerate(mock_token_generator.llm_io_history):
            expected_response = manual_detailed_results[i]["ground_truth_functions"]
            expected_response_str = ",".join(expected_response)
            assert response == expected_response_str, (
                f"LLM response {i} mismatch: expected '{expected_response_str}', got '{response}'"
            )

        print("✅ All LLM responses verified as correct ground truth!")
