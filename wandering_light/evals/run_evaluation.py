import json
import os
from datetime import datetime
from pathlib import Path

import fire

from wandering_light.basis_set import (
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.constants import DEFAULT_SOLVER_BASIS_SET
from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.solver import get_solver_by_name
from wandering_light.trajectory import TrajectoryList, TrajectorySpecList


def is_packaged_legacy_eval_file(eval_file: str) -> bool:
    """Return whether ``eval_file`` is a checked-in evaluator fixture."""
    candidate = Path(eval_file).resolve()
    trusted_root = Path(__file__).resolve().parent / "data"
    return (
        candidate.is_file()
        and candidate.suffix == ".py"
        and candidate.is_relative_to(trusted_root.resolve())
    )


def load_eval_data_as_trajectories(
    eval_file: str,
    variable_name: str = "eval_trajectory_specs",
    basis_set_id: str = DEFAULT_SOLVER_BASIS_SET,
    trusted_legacy_python: bool = False,
) -> tuple[TrajectoryList, FunctionDefSet]:
    """
    Load evaluation data and pre-compute trajectories for efficient evaluation.

    This is more efficient than load_eval_data for repeated evaluations since
    trajectory outputs are computed once instead of every evaluation.
    """
    basis_set = load_basis_set(basis_set_id)
    require_reproducible_basis_runtime(basis_set)
    if not trusted_legacy_python:
        raise ValueError(
            "Legacy .py evaluation files execute Python code. Pass "
            "trusted_legacy_python=True only for a reviewed local fixture; use "
            "BasisTaskRecord JSONL for portable experiment data."
        )
    trajectory_specs = TrajectorySpecList.from_py_file(
        eval_file,
        variable_name,
        trusted_legacy_python=True,
    )
    available_functions = basis_set.as_function_set()
    rebound_specs = []
    for spec in trajectory_specs.specs:
        rebound = []
        for embedded in spec.function_defs:
            registered = available_functions.name_to_function.get(embedded.name)
            if registered is None:
                raise ValueError(
                    f"Evaluation function {embedded.name!r} is absent from basis "
                    f"{basis_set.basis_set_id!r}"
                )
            if registered != embedded:
                raise ValueError(
                    f"Evaluation function {embedded.name!r} does not match basis "
                    f"{basis_set.basis_set_id!r}; select the dataset's exact basis"
                )
            rebound.append(registered)
        rebound_specs.append(type(spec)(spec.input, FunctionDefList(rebound)))

    # Pre-compute trajectories to avoid re-execution during evaluation
    trajectories = TrajectoryList.from_trajectory_specs(
        TrajectorySpecList(rebound_specs), available_functions
    )

    print(
        f"Pre-computed {len(trajectories)} trajectories from {len(trajectory_specs)} specs"
    )

    return trajectories, available_functions


def run_evaluation(
    eval_file: str,
    solver_names: list[str] | None = None,
    num_samples: int | None = None,
    budget: int = 1,
    output_dir: str = "results",
    variable_name: str = "eval_trajectory_specs",
    model_name: str = "checkpoints/latest",
    command: str = "",
    basis_set_id: str = DEFAULT_SOLVER_BASIS_SET,
    trusted_legacy_python: bool = False,
):
    """
    Run evaluation for multiple solvers and save detailed results.

    Args:
        eval_file: Path to the .py file containing trajectory specifications
        solver_names: List of solver names to evaluate
        num_samples: Number of samples to evaluate (None = all)
        budget: Budget for each solver
        output_dir: Directory to save results
        variable_name: Variable name in the eval file containing trajectory specs
        model_name: Name or the path of the saved HF model to use if solver is local_trained
    """
    if solver_names is None:
        solver_names = ["random", "bfs"]
    print(f"Loading evaluation data from {eval_file}...")

    try:
        trajectories, available_functions = load_eval_data_as_trajectories(
            eval_file,
            variable_name,
            basis_set_id,
            trusted_legacy_python=trusted_legacy_python,
        )
        print(f"Loaded {len(trajectories)} trajectories")
        print(f"Found {len(available_functions)} unique functions")
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return

    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Run evaluation for each solver
    results = {}
    for solver_name in solver_names:
        print(f"\nEvaluating {solver_name} solver...")
        solver = get_solver_by_name(solver_name, budget=budget, model_name=model_name)

        result = EvaluateSolver.evaluate_using_trajectories(
            solver,
            trajectories,
            available_functions=available_functions,
            num_samples=num_samples,
            save_results=True,
            output_dir=run_dir,
        )

        results[solver_name] = result.to_dict()
        solver.save(os.path.join(run_dir, "llm_input_output", solver_name))

    # Save summary of all results
    first_function = next(iter(available_functions))
    resolved_basis_set_id = first_function.metadata["basis_set_id"]
    resolved_basis_set_digest = first_function.metadata["basis_set_digest"]
    summary = {
        "timestamp": timestamp,
        "eval_file": eval_file,
        "num_samples": num_samples,
        "budget": budget,
        "basis_set_id": resolved_basis_set_id,
        "basis_set_digest": resolved_basis_set_digest,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "results": results,
    }
    if command:
        summary["command"] = command

    summary_file = os.path.join(run_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {run_dir}")
    print(f"Summary file: {summary_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for solver_name, result in results.items():
        print(f"\n{solver_name}:")
        print(f"  Success Rate: {result['success_rate']:.2%}")
        print(f"  Average Solution Length: {result['avg_solution_length']:.2f}")
        print(f"  Total Samples: {result['total_samples']}")
        print(f"  Successes: {result['success_count']}")
        print(f"  Failures: {len(result['failures'])}")


if __name__ == "__main__":
    import sys

    cmd = "python -m evals.run_evaluation " + " ".join(sys.argv[1:])

    fire.Fire(lambda **kwargs: run_evaluation(command=cmd, **kwargs))
