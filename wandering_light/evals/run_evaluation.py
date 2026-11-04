import hashlib
import json
import os
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import fire

from wandering_light.basis_set import (
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.constants import DEFAULT_SOLVER_BASIS_SET
from wandering_light.evals.evaluate_solver import EvalResult, EvaluateSolver
from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.solver import get_solver_by_name
from wandering_light.trajectory import TrajectoryList, TrajectorySpecList

DEFAULT_BASELINE_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
EVALUATION_SCHEMA_VERSION = 1


def _normalize_int_values(
    values: int | str | Sequence[int] | None,
    *,
    default: Sequence[int],
    name: str,
) -> tuple[int, ...]:
    if values is None:
        normalized = tuple(default)
    elif isinstance(values, int):
        normalized = (values,)
    elif isinstance(values, str):
        normalized = tuple(
            int(value.strip())
            for value in values.strip("[]() ").split(",")
            if value.strip()
        )
    else:
        normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must contain at least one integer")
    return normalized


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    seed_effective: bool = True,
) -> dict[str, Any]:
    """Percentile bootstrap over whole seed-level measurements."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    result = {
        "level": confidence,
        "method": "percentile-bootstrap-over-seed-means",
        "unit": "seed",
        "n": len(values),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "lower": None,
        "upper": None,
    }
    if len(values) < 2 or not seed_effective:
        return result

    rng = random.Random(seed)
    sample_size = len(values)
    bootstrap_means = sorted(
        sum(rng.choice(values) for _ in range(sample_size)) / sample_size
        for _ in range(samples)
    )
    alpha = 1 - confidence
    result["lower"] = _quantile(bootstrap_means, alpha / 2)
    result["upper"] = _quantile(bootstrap_means, 1 - alpha / 2)
    return result


def _freeze_cohort(
    trajectories: TrajectoryList,
    *,
    num_samples: int | None,
    seed: int,
) -> tuple[TrajectoryList, dict[str, Any]]:
    source_size = len(trajectories)
    if source_size == 0:
        raise ValueError("evaluation corpus must contain at least one trajectory")
    sample_size = source_size if num_samples is None else num_samples
    if sample_size <= 0 or sample_size > source_size:
        raise ValueError(f"num_samples {sample_size} is outside [1, {source_size}]")
    if sample_size == source_size:
        source_indices = list(range(source_size))
    else:
        source_indices = sorted(
            random.Random(seed).sample(range(source_size), sample_size)
        )

    selected = [trajectories[index] for index in source_indices]
    task_ids = []
    for source_index, trajectory in zip(source_indices, selected, strict=True):
        payload = json.dumps(
            {
                "source_index": source_index,
                "input": trajectory.input.to_string(),
                "output": trajectory.output.to_string(),
                "golden_functions": [
                    function.name for function in trajectory.function_defs
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        task_ids.append(hashlib.sha256(payload).hexdigest())
    task_ids_digest = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
    return TrajectoryList(selected), {
        "source_size": source_size,
        "size": sample_size,
        "seed": seed,
        "source_indices": source_indices,
        "task_ids": task_ids,
        "task_ids_sha256": f"sha256:{task_ids_digest}",
    }


def _aggregate_seed_results(
    seed_results: Sequence[tuple[int, EvalResult, str]],
    *,
    seed_effective: bool,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not seed_results:
        raise ValueError("at least one seed result is required")
    cohort_sizes = {result.total_samples for _, result, _ in seed_results}
    if len(cohort_sizes) != 1:
        raise ValueError("seed evaluations used different cohort sizes")
    hosts = [result.evaluation_host for _, result, _ in seed_results]
    if any(host != hosts[0] for host in hosts[1:]):
        raise ValueError(
            "seed evaluations ran on different hosts; wall-clock costs are "
            f"not comparable: {hosts}"
        )

    results = [result for _, result, _ in seed_results]
    seeds = [seed for seed, _, _ in seed_results]
    total_samples = sum(result.total_samples for result in results)
    success_count = sum(result.success_count for result in results)
    total_solution_length = sum(
        result.avg_solution_length * result.success_count for result in results
    )
    costed_samples = sum(result.costed_samples for result in results)
    uncosted_samples = sum(result.uncosted_samples for result in results)
    execution_totals = [result.total_charged_executions for result in results]
    complete_costs = all(value is not None for value in execution_totals)
    total_executions = sum(execution_totals) if complete_costs else None
    candidate_totals = [result.total_candidate_programs for result in results]
    total_candidates = (
        sum(candidate_totals)
        if all(value is not None for value in candidate_totals)
        else None
    )

    wall_clock_totals = [result.total_wall_clock_seconds for result in results]
    complete_wall_clock = all(value is not None for value in wall_clock_totals)
    total_wall_clock = sum(wall_clock_totals) if complete_wall_clock else None

    def consistent(field: str):
        values = {getattr(result, field) for result in results}
        if len(values) != 1:
            raise ValueError(f"seed results disagree on {field}: {values}")
        return values.pop()

    failures = []
    detailed_results = []
    compact_seed_results = []
    for seed, result, artifact in seed_results:
        result_dict = result.to_dict()
        failures.extend(
            [spec, f"seed {seed}: {error}"] for spec, error in result_dict["failures"]
        )
        detailed_results.extend(
            {**detail, "seed": seed} for detail in result_dict["detailed_results"]
        )
        compact_seed_results.append(
            {
                "seed": seed,
                "artifact": artifact,
                **{
                    key: value
                    for key, value in result_dict.items()
                    if key not in {"failures", "detailed_results"}
                },
            }
        )

    success_rate_ci = bootstrap_mean_confidence_interval(
        [result.success_rate for result in results],
        confidence=confidence,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        seed_effective=seed_effective,
    )
    execution_ci = bootstrap_mean_confidence_interval(
        [
            result.mean_charged_executions
            for result in results
            if result.mean_charged_executions is not None
        ],
        confidence=confidence,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
        seed_effective=seed_effective and complete_costs,
    )
    wall_clock_ci = bootstrap_mean_confidence_interval(
        [
            result.mean_wall_clock_seconds
            for result in results
            if result.mean_wall_clock_seconds is not None
        ],
        confidence=confidence,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
        seed_effective=seed_effective and complete_wall_clock,
    )
    if not seed_effective:
        reason = "solver is deterministic or its generator is not seed-controlled"
        success_rate_ci["reason"] = reason
        execution_ci["reason"] = reason
        wall_clock_ci["reason"] = reason

    solver_config = dict(results[0].solver_config)
    solver_config.pop("seed", None)
    solver_config["seeds"] = seeds
    return {
        "total_samples": total_samples,
        "cohort_size": cohort_sizes.pop(),
        "success_count": success_count,
        "success_rate": success_count / total_samples if total_samples else 0.0,
        "success_rate_ci": success_rate_ci,
        "avg_solution_length": (
            total_solution_length / success_count if success_count else 0.0
        ),
        "failures": failures,
        "detailed_results": detailed_results,
        "cost_unit": consistent("cost_unit"),
        "cost_definition": consistent("cost_definition"),
        "budget_unit": consistent("budget_unit"),
        "budget_value": consistent("budget_value"),
        "costed_samples": costed_samples,
        "uncosted_samples": uncosted_samples,
        "total_charged_executions": total_executions,
        "mean_charged_executions": (
            total_executions / costed_samples if costed_samples else 0.0
        )
        if total_executions is not None
        else None,
        "mean_charged_executions_ci": execution_ci,
        "total_candidate_programs": total_candidates,
        "mean_candidate_programs": (
            total_candidates / costed_samples
            if total_candidates is not None and costed_samples
            else None
        ),
        "primary_cost_unit": consistent("primary_cost_unit"),
        "primary_cost_definition": consistent("primary_cost_definition"),
        "total_wall_clock_seconds": total_wall_clock,
        "mean_wall_clock_seconds": (
            total_wall_clock / total_samples
            if total_wall_clock is not None and total_samples
            else None
        ),
        "mean_wall_clock_seconds_ci": wall_clock_ci,
        "evaluation_host": dict(hosts[0]),
        "solver_config": solver_config,
        "seed_effective": seed_effective,
        "seeds": seeds,
        "seed_results": compact_seed_results,
    }


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
    path_length: int = 4,
    random_path_lengths: int | str | Sequence[int] | None = None,
    bfs_max_depth: int | None = None,
    seeds: int | str | Sequence[int] | None = None,
    cohort_seed: int = 0,
    confidence: float = 0.95,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """
    Run evaluation for multiple solvers and save detailed results.

    Args:
        eval_file: Path to the .py file containing trajectory specifications
        solver_names: List of solver names to evaluate
        num_samples: Number of samples to evaluate (None = all)
        budget: Solver-native budget. It counts candidate programs for random/LLM
            solvers and basis-function executions for BFS; every result records
            this unit and the common observed execution cost.
        output_dir: Directory to save results
        variable_name: Variable name in the eval file containing trajectory specs
        model_name: Name or the path of the saved HF model to use if solver is local_trained
        path_length: Backward-compatible fixed random length and BFS depth default
        random_path_lengths: Uniformly sampled random-program lengths
        bfs_max_depth: BFS depth cap, independent of random path lengths
        seeds: Replicate seeds for stochastic, seed-controlled solvers
        cohort_seed: Seed used once to freeze a shared task cohort
    """
    if solver_names is None:
        solver_names = ["random", "bfs"]
    if not solver_names or len(set(solver_names)) != len(solver_names):
        raise ValueError("solver_names must be a nonempty list of unique names")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    seed_values = _normalize_int_values(
        seeds,
        default=DEFAULT_BASELINE_SEEDS,
        name="seeds",
    )
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    sampled_path_lengths = _normalize_int_values(
        random_path_lengths,
        default=(path_length,),
        name="random_path_lengths",
    )
    if any(length < 0 for length in sampled_path_lengths):
        raise ValueError("random path lengths must be non-negative")
    if bfs_max_depth is None:
        bfs_max_depth = path_length
    if bfs_max_depth < 0:
        raise ValueError("bfs_max_depth must be non-negative")

    print(f"Loading evaluation data from {eval_file}...")
    trajectories, available_functions = load_eval_data_as_trajectories(
        eval_file,
        variable_name,
        basis_set_id,
        trusted_legacy_python=(
            trusted_legacy_python or is_packaged_legacy_eval_file(eval_file)
        ),
    )
    frozen_trajectories, cohort = _freeze_cohort(
        trajectories,
        num_samples=num_samples,
        seed=cohort_seed,
    )
    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Frozen shared cohort of {len(frozen_trajectories)} trajectories")
    print(f"Found {len(available_functions)} unique functions")

    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each solver
    results: dict[str, Any] = {}
    for solver_name in solver_names:
        print(f"\nEvaluating {solver_name} solver...")
        seed_effective = solver_name == "random"
        solver_seeds = seed_values if seed_effective else seed_values[:1]
        per_seed: list[tuple[int, EvalResult, str]] = []
        for seed in solver_seeds:
            print(f"  seed={seed}")
            solver = get_solver_by_name(
                solver_name,
                budget=budget,
                path_length=path_length,
                model_name=model_name,
                path_lengths=sampled_path_lengths,
                max_depth=bfs_max_depth,
                seed=seed,
            )
            result = EvaluateSolver.evaluate_using_trajectories(
                solver,
                frozen_trajectories,
                available_functions=available_functions,
            )

            result_file = run_dir / solver_name / f"seed-{seed}.json"
            result.save_to_file(str(result_file))
            solver.save(
                str(run_dir / "solver_artifacts" / solver_name / f"seed-{seed}")
            )
            per_seed.append(
                (
                    seed,
                    result,
                    result_file.relative_to(run_dir).as_posix(),
                )
            )

        results[solver_name] = _aggregate_seed_results(
            per_seed,
            seed_effective=seed_effective,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )

    # Save summary of all results
    first_function = next(iter(available_functions))
    resolved_basis_set_id = first_function.metadata["basis_set_id"]
    resolved_basis_set_digest = first_function.metadata["basis_set_digest"]
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "timestamp": timestamp,
        "eval_file": eval_file,
        "requested_num_samples": num_samples,
        "num_samples": cohort["size"],
        "cohort": cohort,
        "budget": budget,
        "budget_deprecated": (
            "solver-native; inspect each result's budget_value and budget_unit"
        ),
        "seeds": list(seed_values),
        "random_path_lengths": list(sampled_path_lengths),
        "bfs_max_depth": bfs_max_depth,
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "basis_set_id": resolved_basis_set_id,
        "basis_set_digest": resolved_basis_set_digest,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "results": results,
    }
    if command:
        summary["command"] = command

    summary_file = run_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {run_dir}")
    print(f"Summary file: {summary_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for solver_name, result in results.items():
        print(f"\n{solver_name}:")
        print(f"  Success Rate: {result['success_rate']:.2%}")
        interval = result["success_rate_ci"]
        if interval["lower"] is not None:
            print(
                f"  {interval['level']:.0%} seed CI: "
                f"[{interval['lower']:.2%}, {interval['upper']:.2%}]"
            )
        print(f"  Average Solution Length: {result['avg_solution_length']:.2f}")
        print(f"  Cohort Size: {result['cohort_size']}")
        print(f"  Seed-Task Trials: {result['total_samples']}")
        print(f"  Successes: {result['success_count']}")
        print(f"  Failures: {len(result['failures'])}")
        if result["mean_charged_executions"] is None:
            print("  Mean Cost: unreported")
        else:
            print(
                "  Mean Cost: "
                f"{result['mean_charged_executions']:.2f} {result['cost_unit']}"
            )
        if result["mean_wall_clock_seconds"] is None:
            print("  Mean Wall Clock: unreported")
        else:
            print(f"  Mean Wall Clock: {result['mean_wall_clock_seconds']:.4f} s")
        print(f"  Budget: {result['budget_value']} {result['budget_unit']}")

    return summary


if __name__ == "__main__":
    import sys

    cmd = "python -m wandering_light.evals.run_evaluation " + " ".join(sys.argv[1:])

    def run_cli(**kwargs):
        run_evaluation(command=cmd, **kwargs)

    fire.Fire(run_cli)
