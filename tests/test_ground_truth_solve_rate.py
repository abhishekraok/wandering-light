"""
Test to verify GroundTruthTokenGenerator can solve each evaluation sample.
This test focuses on understanding why we don't get 100% solve rate.
"""

# Import what we need
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.executor import Executor
from wandering_light.solver import TokenGenerator, TokenGeneratorSolver
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
        if self.current_spec_index in self.specs_and_solutions:
            solution = self.specs_and_solutions[self.current_spec_index]
        else:
            # Fallback - shouldn't happen in our test
            solution = "identity_int"

        self.llm_io_history.append((prompt, solution))
        self.current_spec_index += 1
        return solution


class TestGroundTruthSolver:
    """Test suite to verify GroundTruthTokenGenerator solve rate."""

    def test_ground_truth_solve_rate(self):
        """
        Test each eval sample with GroundTruthTokenGenerator and show failures.
        """
        # Load evaluation data using the new trajectory-based method
        eval_file = "wandering_light/evals/data/random_inputs.py"
        trajectories, available_functions = load_eval_data_as_trajectories(eval_file)
        trajectory_specs = trajectories.to_spec_list()

        print(f"Testing {len(trajectories)} trajectories")
        print(f"Available functions: {len(available_functions)}")

        # Create ground truth token generator
        mock_token_generator = GroundTruthTokenGenerator()
        mock_token_generator.set_trajectory_specs(trajectory_specs)

        # Create solver with ground truth LLM
        solver = TokenGeneratorSolver(
            token_generator=mock_token_generator,
            budget=1,  # Only need 1 attempt since we're returning ground truth
        )

        # Create executor for ground truth validation
        executor = Executor(available_functions)

        # Track results
        successes = 0
        failures = []

        # Test each trajectory spec individually
        for i, spec in enumerate(trajectory_specs.specs):
            try:
                # Execute ground truth to get expected output
                ground_truth_result = executor.execute_trajectory(spec)
                if not ground_truth_result.success:
                    raise ValueError(
                        f"Ground truth execution failed: {ground_truth_result.error_msg}"
                    )
                expected_output = ground_truth_result.trajectory.output

                # Try to solve with our ground truth solver
                result = solver.solve(spec.input, expected_output, available_functions)

                if result.success and result.trajectory is not None:
                    successes += 1
                    if i < 5:  # Show first few successes
                        print(f"✅ Spec {i}: SUCCESS")
                else:
                    # Record failure details
                    actual_output = (
                        result.trajectory.output if result.trajectory else "None"
                    )
                    failures.append(
                        {
                            "spec_index": i,
                            "input": str(spec.input),
                            "expected_output": str(expected_output),
                            "actual_output": str(actual_output),
                            "expected_functions": [f.name for f in spec.function_defs],
                            "actual_functions": (
                                [f.name for f in result.trajectory.function_defs]
                                if result.trajectory
                                else []
                            ),
                            "error": result.error_msg,
                            "ground_truth_response": mock_token_generator.specs_and_solutions[
                                i
                            ],
                        }
                    )

            except Exception as e:
                # Ground truth execution failed
                failures.append(
                    {
                        "spec_index": i,
                        "input": str(spec.input),
                        "expected_output": "EXECUTION_FAILED",
                        "actual_output": "EXECUTION_FAILED",
                        "expected_functions": [f.name for f in spec.function_defs],
                        "actual_functions": [],
                        "error": str(e),
                        "ground_truth_response": f"GT_EXEC_FAILED: {e}",
                    }
                )

        # Print results summary
        total_specs = len(trajectory_specs.specs)
        success_rate = successes / total_specs

        print("\n=== RESULTS ===")
        print(f"Total specs: {total_specs}")
        print(f"Successes: {successes}")
        print(f"Failures: {len(failures)}")
        print(f"Success rate: {success_rate:.2%}")

        # Show failure details
        if failures:
            print("\n=== FAILURE ANALYSIS ===")
            for i, failure in enumerate(failures[:10]):  # Show first 10 failures
                print(f"\nFailure {i + 1} (Spec {failure['spec_index']}):")
                print(f"  Input: {failure['input']}")
                print(f"  Expected functions: {failure['expected_functions']}")
                print(f"  Ground truth response: '{failure['ground_truth_response']}'")
                print(f"  Actual functions: {failure['actual_functions']}")
                print(f"  Expected output: {failure['expected_output']}")
                print(f"  Actual output: {failure['actual_output']}")
                print(f"  Error: {failure['error']}")

        # If we don't get 100%, investigate why
        if success_rate < 1.0:
            print("\n=== WHY NOT 100%? ===")

            # Check if it's ground truth execution failures
            gt_exec_failures = [
                f for f in failures if "GT_EXEC_FAILED" in f["ground_truth_response"]
            ]
            if gt_exec_failures:
                print(f"Ground truth execution failures: {len(gt_exec_failures)}")
                for f in gt_exec_failures[:3]:
                    print(f"  - Spec {f['spec_index']}: {f['error']}")

            # Check if it's solver execution failures
            solver_failures = [
                f
                for f in failures
                if "GT_EXEC_FAILED" not in f["ground_truth_response"]
            ]
            if solver_failures:
                print(
                    f"Solver failures (even with ground truth): {len(solver_failures)}"
                )
                for f in solver_failures[:3]:
                    print(f"  - Spec {f['spec_index']}: {f['error']}")

        # Assert that we achieve 100% success rate with ground truth
        assert success_rate == 1.0, (
            f"Expected 100% success rate with ground truth, got {success_rate:.2%}. Failures: {len(failures)}"
        )
