import json
import os
import platform
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import fire

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.solver import (
    WALL_CLOCK_COST_DEFINITION,
    WALL_CLOCK_COST_UNIT,
    PredictionCost,
    TrajectorySolver,
    get_solver_by_name,
)
from wandering_light.trajectory import (
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
    charged_executions: int | None = None
    candidate_programs: int | None = None
    cost_termination: str | None = None
    seconds: float | None = None

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
    cost_unit: str | None = None
    cost_definition: str | None = None
    budget_unit: str | None = None
    budget_value: int | None = None
    costed_samples: int = 0
    uncosted_samples: int = 0
    total_charged_executions: int | None = None
    mean_charged_executions: float | None = None
    total_candidate_programs: int | None = None
    mean_candidate_programs: float | None = None
    primary_cost_unit: str | None = None
    primary_cost_definition: str | None = None
    total_wall_clock_seconds: float | None = None
    mean_wall_clock_seconds: float | None = None
    evaluation_host: dict[str, Any] = field(default_factory=dict)
    solver_config: dict[str, Any] = field(default_factory=dict)

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


def evaluation_host() -> dict[str, Any]:
    """Identify the host so wall-clock costs are only compared within it."""
    return {
        "node": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _wall_clock_summary(
    total_seconds: float | None,
    sample_count: int,
) -> dict[str, Any]:
    """Primary comparable cost: wall clock charges inference and execution."""
    return {
        "primary_cost_unit": WALL_CLOCK_COST_UNIT,
        "primary_cost_definition": WALL_CLOCK_COST_DEFINITION,
        "total_wall_clock_seconds": total_seconds,
        "mean_wall_clock_seconds": (
            total_seconds / sample_count
            if total_seconds is not None and sample_count
            else None
        ),
        "evaluation_host": evaluation_host(),
    }


def _cost_summary(
    solver: TrajectorySolver,
    costs: list[PredictionCost | None],
) -> dict[str, Any]:
    metadata = solver.cost_metadata()
    reported = [cost for cost in costs if cost is not None]
    complete = len(reported) == len(costs)
    executions = [cost.charged_executions for cost in reported]
    candidate_counts = [
        cost.candidate_programs
        for cost in reported
        if cost.candidate_programs is not None
    ]
    return {
        "cost_unit": metadata.get("cost_unit"),
        "cost_definition": metadata.get("cost_definition"),
        "budget_unit": metadata.get("budget_unit"),
        "budget_value": metadata.get("budget_value"),
        "costed_samples": len(reported),
        "uncosted_samples": len(costs) - len(reported),
        "total_charged_executions": sum(executions) if complete else None,
        "mean_charged_executions": (
            sum(executions) / len(executions)
            if complete and executions
            else (0.0 if complete else None)
        ),
        "total_candidate_programs": (
            sum(candidate_counts)
            if complete and len(candidate_counts) == len(reported)
            else None
        ),
        "mean_candidate_programs": (
            sum(candidate_counts) / len(candidate_counts)
            if complete and len(candidate_counts) == len(reported) and candidate_counts
            else None
        ),
        "solver_config": metadata,
    }


class EvaluateSolver:
    @staticmethod
    def evaluate_using_trajectory_specs(
        solver: TrajectorySolver,
        trajectory_specs: TrajectorySpecList,
        available_functions: FunctionDefSet,
        num_samples: int | None = None,
        save_results: bool = False,
        output_dir: str = "results/solver/",
        sample_seed: int | None = None,
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
        trajectories = TrajectoryList.from_trajectory_specs(
            trajectory_specs,
            available_functions,
        )
        return EvaluateSolver.evaluate_using_trajectories(
            solver,
            trajectories,
            available_functions,
            num_samples,
            save_results,
            output_dir,
            sample_seed,
        )

    @staticmethod
    def evaluate_using_trajectories(
        solver: TrajectorySolver,
        trajectories: "TrajectoryList",
        available_functions: FunctionDefSet,
        num_samples: int | None = None,
        save_results: bool = False,
        output_dir: str = "results",
        sample_seed: int | None = None,
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
            sampled_indices = sorted(
                random.Random(sample_seed).sample(
                    range(len(trajs_to_evaluate)),
                    num_samples,
                )
            )
            trajs_to_evaluate = [trajs_to_evaluate[index] for index in sampled_indices]
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

        # Batch solve all problems at once. Wall clock covers the whole call,
        # so policy inference is charged alongside basis executions.
        solve_started = time.perf_counter()
        if problems:
            batch_results = solver.solve_batch(problems, available_functions)
        else:
            batch_results = []
        wall_clock_seconds = time.perf_counter() - solve_started
        if len(batch_results) != len(problems):
            raise ValueError(
                f"solver returned {len(batch_results)} results for "
                f"{len(problems)} problems"
            )

        # Process results
        successes = 0
        total_length = 0
        failures: list[tuple[TrajectorySpec, Exception]] = []
        detailed_results: list[DetailedEvalResult] = []

        for traj, result in zip(trajs_to_evaluate, batch_results, strict=True):
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
                if result.cost is not None:
                    detailed_result.charged_executions = result.cost.charged_executions
                    detailed_result.candidate_programs = result.cost.candidate_programs
                    detailed_result.cost_termination = result.cost.termination
                    detailed_result.seconds = result.cost.seconds

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
            **_cost_summary(solver, [result.cost for result in batch_results]),
            **_wall_clock_summary(wall_clock_seconds, total_evaluated),
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
        prediction_costs: list[PredictionCost | None] = []
        wall_clock_seconds = 0.0
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
                prediction_costs.append(None)
                continue
            # solver attempt
            solve_started = time.perf_counter()
            result = solver.solve(input_list, true_output, available_functions)
            wall_clock_seconds += time.perf_counter() - solve_started
            prediction_costs.append(result.cost)
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
            **_cost_summary(solver, prediction_costs),
            **_wall_clock_summary(wall_clock_seconds, num_samples),
        )


def evaluate_solver_from_file(
    eval_file: str,
    solver_name: str = "random",
    num_samples: int | None = None,
    budget: int = 10,
    variable_name: str = "eval_trajectory_specs",
    trusted_legacy_python: bool = False,
    path_length: int = 4,
    random_path_lengths: list[int] | None = None,
    bfs_max_depth: int | None = None,
    seed: int | None = None,
):
    """
    Evaluate a solver using trajectory specs loaded from a .py file.

    Args:
        eval_file: Path to the .py file containing trajectory specifications
        solver_name: Name of the solver ("random", "bfs")
        num_samples: Number of samples to evaluate (None = all)
        budget: Budget for the solver
        variable_name: Variable name in the eval file
        trusted_legacy_python: Explicit opt-in because ``.py`` data is executed
        path_length: Backward-compatible fixed random length/BFS depth default
        random_path_lengths: Uniform random-program length choices
        bfs_max_depth: BFS depth cap independent of random path sampling
        seed: Random solver and cohort-sampling seed
    """
    print(f"Loading evaluation data from {eval_file}...")

    if not trusted_legacy_python:
        print(
            "Error loading evaluation data: legacy .py files execute Python code; "
            "pass trusted_legacy_python=True only for a reviewed local fixture"
        )
        return None

    # Load trajectory specs from an explicitly trusted file.
    try:
        trajectory_specs = TrajectorySpecList.from_py_file(
            eval_file,
            variable_name,
            trusted_legacy_python=True,
        )
        print(f"Loaded {len(trajectory_specs)} trajectory specifications")
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return

    # Create solver
    solver = get_solver_by_name(
        solver_name,
        budget=budget,
        path_length=path_length,
        path_lengths=random_path_lengths,
        max_depth=bfs_max_depth,
        seed=seed,
    )

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
        solver,
        trajectory_specs,
        available_functions,
        num_samples,
        sample_seed=seed,
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
    if result.mean_wall_clock_seconds is not None:
        print(
            f"Mean wall clock: {result.mean_wall_clock_seconds:.4f} s/sample "
            f"({result.total_wall_clock_seconds:.2f} s total)"
        )

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
