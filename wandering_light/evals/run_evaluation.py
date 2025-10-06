import json
import os
from datetime import datetime

import fire

from wandering_light.common_functions import basic_fns
from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.function_def import FunctionDefSet
from wandering_light.solver import get_solver_by_name
from wandering_light.trajectory import TrajectoryList, TrajectorySpecList


def load_eval_data_as_trajectories(
    eval_file: str, variable_name: str = "eval_trajectory_specs"
) -> tuple[TrajectoryList, FunctionDefSet]:
    """
    Load evaluation data and pre-compute trajectories for efficient evaluation.

    This is more efficient than load_eval_data for repeated evaluations since
    trajectory outputs are computed once instead of every evaluation.
    """
    trajectory_specs = TrajectorySpecList.from_py_file(eval_file, variable_name)

    available_functions = FunctionDefSet()
    for spec in trajectory_specs.specs:
        available_functions.extend(spec.function_defs)
    # Merge available functions with basic functions
    available_functions.extend(basic_fns)

    # Pre-compute trajectories to avoid re-execution during evaluation
    trajectories = TrajectoryList.from_trajectory_specs(
        trajectory_specs, available_functions
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
            eval_file, variable_name
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
    summary = {
        "timestamp": timestamp,
        "eval_file": eval_file,
        "num_samples": num_samples,
        "budget": budget,
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
