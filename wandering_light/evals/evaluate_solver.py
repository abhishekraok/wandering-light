import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import fire

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.solver import TrajectorySolver, get_solver_by_name
from wandering_light.trajectory import (
    Trajectory,
    TrajectoryList,
    TrajectorySpec,
    TrajectorySpecList,
)
from wandering_light.typed_list import TypedList


@dataclass
class DetailedEvalResult:
    """Detailed result for a single evaluation sample."""

    input: str
    expected_output: str | None = None
    success: bool = False
    actual_output: str | None = None
    predicted_functions: list[str] | None = None
    golden_functions: list[str] | None = None
    solution_length: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization compatibility."""
        return asdict(self)


@dataclass
class EvalResult:
    total_samples: int
    success_count: int
    success_rate: float
    avg_solution_length: float
    failures: list[tuple[TrajectorySpec, Exception]]
    detailed_results: list[DetailedEvalResult] = field(default_factory=list)

    def __repr__(self):
        return (
            f"EvalResult(total={self.total_samples}, successes={self.success_count}, "
            f"rate={self.success_rate:.2f}, avg_length={self.avg_solution_length:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluation result to a dictionary for serialization."""
        result_dict = asdict(self)
        # Convert TrajectorySpec and Exception objects to strings
        result_dict["failures"] = [
            (str(spec), str(error)) for spec, error in self.failures
        ]
        # Convert detailed_results to list of dicts for JSON serialization
        result_dict["detailed_results"] = [
            detail.to_dict() for detail in self.detailed_results
        ]
        return result_dict

    def save_to_file(self, output_file: str):
        """Save the evaluation result to a JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EvaluateSolver:
    @staticmethod
    def evaluate_using_trajectory_specs(
        solver: TrajectorySolver,
        trajectory_specs: TrajectorySpecList,
        available_functions: FunctionDefSet,
        num_samples: int | None = None,
        save_results: bool = False,
        output_dir: str = "results/solver/",
    ) -> EvalResult:
        """
        Evaluate a solver on given trajectory specifications using batch processing.

        Args:
            solver: an instance of TrajectorySolver to evaluate.
            trajectory_specs: precomputed TrajectorySpecList with input/output pairs
            available_functions: List of available functions for solving
            num_samples: number of samples to evaluate (None = all)
            save_results: whether to save detailed results to a file
            output_dir: directory to save results if save_results is True

        Returns:
            EvalResult with stats on successes, avg length, and failures.
        """
        executor = Executor(available_functions)

        # Select samples to evaluate
        specs_to_evaluate = trajectory_specs.specs
        if num_samples is not None and num_samples < len(specs_to_evaluate):
            specs_to_evaluate = random.sample(specs_to_evaluate, num_samples)

        # Prepare batch problems by executing ground truth trajectories
        trajectories = []
        spec_to_ground_truth = {}
        valid_specs = []

        for spec in specs_to_evaluate:
            # For ground truth, we use execute_trajectory which handles failures gracefully
            result = executor.execute_trajectory(spec)
            if result.success:
                true_output = result.trajectory.output
                trajectories.append(Trajectory(spec=spec, output=true_output))
                spec_to_ground_truth[id(spec)] = true_output
                valid_specs.append(spec)
            else:
                # Skip specs that fail during ground truth computation
                continue

        return EvaluateSolver.evaluate_using_trajectories(
            solver,
            TrajectoryList(trajectories),
            available_functions,
            num_samples,
            save_results,
            output_dir,
        )

    @staticmethod
    def evaluate_using_trajectories(
        solver: TrajectorySolver,
        trajectories: "TrajectoryList",
        available_functions: FunctionDefSet,
        num_samples: int | None = None,
        save_results: bool = False,
        output_dir: str = "results",
    ) -> EvalResult:
        """
        Evaluate a solver on given pre-computed trajectories using batch processing.

        This is more efficient than evaluate_using_trajectory_specs since trajectories
        already have computed outputs, eliminating the need to re-execute specs.

        Args:
            solver: an instance of TrajectorySolver to evaluate.
            trajectories: pre-computed TrajectoryList with input/output pairs
            available_functions: List of available functions for solving
            num_samples: number of samples to evaluate (None = all)
            save_results: whether to save detailed results to a file
            output_dir: directory to save results if save_results is True

        Returns:
            EvalResult with stats on successes, avg length, and failures.
        """
        # Select samples to evaluate
        trajs_to_evaluate = trajectories.trajectories
        if num_samples is not None and num_samples < len(trajs_to_evaluate):
            trajs_to_evaluate = random.sample(trajs_to_evaluate, num_samples)
        if num_samples is not None and num_samples > len(trajs_to_evaluate):
            raise ValueError(
                f"num_samples {num_samples} is greater than the number of trajectories {len(trajs_to_evaluate)}"
            )

        # Prepare batch problems using pre-computed outputs (no execution needed!)
        problems = []
        traj_to_ground_truth = {}

        for traj in trajs_to_evaluate:
            # No execution needed - trajectories already have computed outputs
            true_output = traj.output
            problems.append((traj.input, true_output))
            traj_to_ground_truth[id(traj)] = true_output

        # Batch solve all problems at once
        if problems:
            batch_results = solver.solve_batch(problems, available_functions)
        else:
            batch_results = []

        # Process results
        successes = 0
        total_length = 0
        failures: list[tuple[TrajectorySpec, Exception]] = []
        detailed_results: list[DetailedEvalResult] = []

        for traj, result in zip(trajs_to_evaluate, batch_results, strict=False):
            true_output = traj_to_ground_truth[id(traj)]

            detailed_result = DetailedEvalResult(
                input=str(traj.input),
                expected_output=str(true_output),
                golden_functions=[f.name for f in traj.function_defs],
            )

            try:
                # Record detailed result
                detailed_result.success = result.success
                detailed_result.error = result.error_msg

                if result.success and result.trajectory is not None:
                    successes += 1
                    total_length += len(result.trajectory.function_defs)
                else:
                    failures.append(
                        (
                            traj.to_spec(),
                            ValueError(result.error_msg or "Unknown error"),
                        )
                    )

                # Always populate trajectory details if available (for both success and failure)
                if result.trajectory is not None:
                    detailed_result.actual_output = str(result.trajectory.output)
                    detailed_result.predicted_functions = [
                        f.name for f in result.trajectory.function_defs
                    ]
                    detailed_result.solution_length = len(
                        result.trajectory.function_defs
                    )
            except Exception as e:
                failures.append((traj.to_spec(), e))
                detailed_result.error = str(e)
                detailed_result.actual_output = None
                detailed_result.predicted_functions = None
                detailed_result.solution_length = None

            detailed_results.append(detailed_result)

        total_evaluated = len(trajs_to_evaluate)
        success_rate = successes / total_evaluated if total_evaluated > 0 else 0.0
        avg_length = total_length / successes if successes > 0 else 0.0

        eval_result = EvalResult(
            total_samples=total_evaluated,
            success_count=successes,
            success_rate=success_rate,
            avg_solution_length=avg_length,
            failures=failures,
            detailed_results=detailed_results,
        )

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            solver_name = solver.__class__.__name__
            output_file = os.path.join(output_dir, f"{solver_name}_{timestamp}.json")
            eval_result.save_to_file(output_file)
            print(f"Saved detailed evaluation results to {output_file}")

        return eval_result

    @staticmethod
    def evaluate_using_random_walk(
        solver: TrajectorySolver,
        input_lists: list[TypedList],
        num_samples: int,
        available_functions: FunctionDefSet,
        path_length: int,
    ) -> EvalResult:
        """
        Evaluate a solver on randomly generated trajectories.

        Args:
            solver: an instance of TrajectorySolver to evaluate.
            input_lists: list of possible TypedList inputs to choose from.
            num_samples: number of random trajectories to evaluate.
            available_functions: List of available functions.
            path_length: length of random walk to generate each sample.

        Returns:
            EvalResult with stats on successes, avg length, and failures.
        """
        successes = 0
        total_length = 0
        failures: list[tuple[TrajectorySpec, Exception]] = []
        executor = Executor(available_functions)

        for _ in range(num_samples):
            input_list = random.choice(input_lists)
            spec = TrajectorySpec.create_random_walk(
                input_list, path_length, available_functions.functions
            )
            if len(spec.function_defs) != path_length:
                raise ValueError(
                    f"Generated spec length {len(spec.function_defs)} < requested {path_length}"
                )
            # ground truth execution
            result = executor.execute_trajectory(spec)
            if result.success:
                true_output = result.trajectory.output
            else:
                failures.append((spec, ValueError(result.error_msg)))
                continue
            # solver attempt
            result = solver.solve(input_list, true_output, available_functions)
            # record success or failure from MaybeTrajectory
            if result.success and result.trajectory is not None:
                successes += 1
                total_length += len(result.trajectory.function_defs)
            else:
                failures.append((spec, ValueError(result.error_msg)))

        success_rate = successes / num_samples if num_samples > 0 else 0.0
        avg_length = total_length / successes if successes > 0 else 0.0
        return EvalResult(
            total_samples=num_samples,
            success_count=successes,
            success_rate=success_rate,
            avg_solution_length=avg_length,
            failures=failures,
        )


def evaluate_solver_from_file(
    eval_file: str,
    solver_name: str = "random",
    num_samples: int | None = None,
    budget: int = 10,
    variable_name: str = "eval_trajectory_specs",
):
    """
    Evaluate a solver using trajectory specs loaded from a .py file.

    Args:
        eval_file: Path to the .py file containing trajectory specifications
        solver_name: Name of the solver ("random", "bfs")
        num_samples: Number of samples to evaluate (None = all)
        budget: Budget for the solver
        variable_name: Variable name in the eval file
    """
    print(f"Loading evaluation data from {eval_file}...")

    # Load trajectory specs from file
    try:
        trajectory_specs = TrajectorySpecList.from_py_file(eval_file, variable_name)
        print(f"Loaded {len(trajectory_specs)} trajectory specifications")
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return

    # Create solver
    solver = get_solver_by_name(solver_name, budget=budget)

    # Evaluate
    print(f"Evaluating {solver_name} solver...")
    if num_samples:
        print(f"Using {num_samples} random samples from the dataset")
    else:
        print(f"Using all {len(trajectory_specs)} samples")

    # Get available functions from the trajectory specs
    available_functions = FunctionDefSet()
    for spec in trajectory_specs.specs:
        available_functions.extend(spec.function_defs)

    result = EvaluateSolver.evaluate_using_trajectory_specs(
        solver, trajectory_specs, available_functions, num_samples
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Solver: {solver_name}")
    print(f"Budget: {budget}")
    print(f"Total samples: {result.total_samples}")
    print(f"Successes: {result.success_count}")
    print(f"Success rate: {result.success_rate:.2%}")
    print(f"Average solution length: {result.avg_solution_length:.2f}")
    print(f"Failures: {len(result.failures)}")

    if result.failures and len(result.failures) <= 5:
        print("\nFirst few failures:")
        for i, (spec, error) in enumerate(result.failures[:5]):
            funcs = " -> ".join(f.name for f in spec.function_defs)
            print(f"  [{i + 1}] {spec.input} -> {funcs}: {error}")

    # Note: This standalone function saves LLM logs separately from main evaluation runs
    solver.save(
        f"results/standalone_llm_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    )
    return result


# Common evaluation harness
def evaluate_solver_name(
    solver_name: str = "random",
    num_samples: int = 10,
    path_length: int = 4,
    budget: int = 10,
):
    """
    Run evaluation for RandomSolve and BFSSolve on sample functions and inputs.
    Prints summary of EvalResult for each solver.
    """
    # prepare sample functions
    available_functions = [
        FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        ),
        FunctionDef(
            name="dec",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x - 1",
        ),
        FunctionDef(
            name="square",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * x",
        ),
        FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        ),
        FunctionDef(
            name="halve",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x // 2",
        ),
    ]

    # sample input lists
    input_lists = [
        TypedList(list(range(1, 5))),
        TypedList(list(range(10, 14))),
    ]

    # instantiate solver
    solver = get_solver_by_name(solver_name, budget, path_length)
    # evaluate random solver
    print(f"Evaluating {solver_name}: samples={num_samples}, path_length={path_length}")
    result = EvaluateSolver.evaluate_using_random_walk(
        solver, input_lists, num_samples, available_functions, path_length
    )
    print(f"{solver_name} result:", result)


if __name__ == "__main__":
    fire.Fire(
        {
            "evaluate_solver_from_file": evaluate_solver_from_file,
            "evaluate_solver_name": evaluate_solver_name,
        },
        serialize=False,
    )
