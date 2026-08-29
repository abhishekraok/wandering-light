"""Locked beam-rollout evaluation on the deep-corpus-v1 #50 cohort.

Validation evaluates four independent beam searches, then ``freeze`` records the
chosen width and the complete protocol before the held-out split is scored.  Test
evaluation is single-use and requires the exact 96-task #50 cohort plus its
preserved per-task BFS table; no distance-8 search is repeated.

    python experiments/evaluate_rollout_policy.py evaluate \
        --checkpoint CHECKPOINT --output-dir OUTPUT [--dry-run]
    python experiments/evaluate_rollout_policy.py freeze \
        --validation-summary OUTPUT/summary.json --output selection-lock.json
    python experiments/evaluate_rollout_policy.py evaluate --split test \
        --checkpoint CHECKPOINT --output-dir OUTPUT \
        --selection-lock selection-lock.json --bfs-reference bfs_tasks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.evaluate_deep_corpus_sft import (
    DEFAULT_WANDB_PROJECT,
    PERSONAL_WANDB_ENTITY,
    _cohort_identity,
    _corpus_identity,
    _validate_wandb_url,
    configure_unbounded_int_serialization,
    select_balanced_cohort,
    validate_personal_wandb_entity,
    verify_and_load_split,
    write_jsonl_gz,
)
from wandering_light.executor import Executor
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.function_def import FunctionDefSet
    from wandering_light.solver import ActionScorer, RolloutPredictor


SCHEMA_VERSION = 1
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v1")
LOCKED_WIDTHS = (1, 8, 10, 64)
LOCKED_SEED = 20260816
LOCKED_TASKS_PER_DISTANCE = 12
LOCKED_MAX_DEPTH = 8
LOCKED_EXECUTION_BUDGET = 100_000
EXPECTED_TEST_COHORT_DIGEST = (
    "sha256:00932c5217071249cd7698dd0d7b5aa87132bf9adcfd10e259a1c3ccc4741598"
)
EXPECTED_BFS_REFERENCE_SHA256 = (
    "sha256:1470a2147a0cdc59c21be9964bf4b90f5a7a88856c612befdfe84632844bd5a7"
)
BFS_REFERENCE_URL = (
    "https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v1/runs/txxxodnx"
)
BFS_REFERENCE_COLUMNS = ("task_id", "distance", "success", "executions", "seconds")
COST_DEFINITION = (
    "One execution for every type-compatible basis function attempted, including "
    "transitions whose function raises. Each beam width is searched independently; "
    "a task stops charging when it solves or terminates."
)
TERMINATIONS = frozenset({"solved", "budget", "max_depth", "frontier_exhausted"})


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: Path
    output_dir: Path
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    split: str = "validation"
    selection_lock: Path | None = None
    bfs_reference: Path | None = None
    widths: tuple[int, ...] = LOCKED_WIDTHS
    seed: int = LOCKED_SEED
    tasks_per_distance: int = LOCKED_TASKS_PER_DISTANCE
    max_depth: int = LOCKED_MAX_DEPTH
    execution_budget: int = LOCKED_EXECUTION_BUDGET
    device: str = "cpu"
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_run_name: str | None = None
    dry_run: bool = False
    progress: bool = True


PredictorFactory = Callable[["ActionScorer", int, int, int], "RolloutPredictor"]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_config(config: EvaluationConfig) -> None:
    validate_personal_wandb_entity(config.wandb_entity)
    if config.split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    locked = {
        "widths": (config.widths, LOCKED_WIDTHS),
        "seed": (config.seed, LOCKED_SEED),
        "tasks_per_distance": (
            config.tasks_per_distance,
            LOCKED_TASKS_PER_DISTANCE,
        ),
        "max_depth": (config.max_depth, LOCKED_MAX_DEPTH),
        "execution_budget": (
            config.execution_budget,
            LOCKED_EXECUTION_BUDGET,
        ),
    }
    drift = [name for name, (actual, expected) in locked.items() if actual != expected]
    if drift:
        raise ValueError(f"Locked rollout protocol changed: {', '.join(drift)}")
    if not config.device:
        raise ValueError("device must not be empty")
    if not config.wandb_project:
        raise ValueError("wandb_project must not be empty")
    if config.split == "validation":
        if config.selection_lock is not None:
            raise ValueError("--selection-lock is only valid with --split test")
        if config.bfs_reference is not None:
            raise ValueError("The preserved BFS table belongs only to the test cohort")
    else:
        if config.selection_lock is None:
            raise ValueError("Held-out test scoring requires --selection-lock")
        if config.bfs_reference is None:
            raise ValueError("Held-out test scoring requires --bfs-reference")
        if config.dry_run:
            raise ValueError(
                "Held-out test dry runs are forbidden; dry-run validation instead"
            )
    if not config.dry_run and not config.wandb_run_name:
        raise ValueError("An actual evaluation requires --wandb-run-name")


def _evaluation_protocol(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "widths": list(config.widths),
        "seed": config.seed,
        "tasks_per_distance": config.tasks_per_distance,
        "max_depth": config.max_depth,
        "execution_budget": config.execution_budget,
        "device": config.device,
        "wandb_entity": config.wandb_entity,
        "wandb_project": config.wandb_project,
        "selection_rule": (
            "Lexicographic: distance-5-8 solved, distance-7-8 solved, all-distance "
            "solved, lower distance-5-8 mean executions, then narrower beam."
        ),
        "cost_definition": COST_DEFINITION,
    }


def _source_identity() -> dict[str, Any]:
    runner = Path(__file__).resolve()
    repo_root = runner.parents[1]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode()),
        "tracked_diff_sha256": _sha256_bytes(diff.encode()),
        "runner": {
            "path": runner.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(runner),
        },
    }


def _checkpoint_identity(
    checkpoint: Path,
    *,
    corpus_manifest_digest: str,
    basis_set_id: str,
    basis_set_digest: str,
) -> dict[str, Any]:
    """Verify the loadable, weights-only policy identity without loading weights."""

    from wandering_light.rollout_policy import (
        CONFIG_FILE,
        MODEL_FILE,
        TRAINER_STATE_FILE,
        TRAINING_MANIFEST_FILE,
        WANDB_LINK_FILE,
        _read_wandb_url,
        assert_weights_only_checkpoint,
    )

    checkpoint = Path(checkpoint)
    assert_weights_only_checkpoint(checkpoint, require_complete=True)
    config = _read_json(checkpoint / CONFIG_FILE, label="policy config")
    manifest = _read_json(
        checkpoint / TRAINING_MANIFEST_FILE, label="training manifest"
    )
    if manifest.get("status") != "complete":
        raise ValueError("Rollout policy training manifest is not complete")
    policy_basis = config.get("basis_set")
    if not isinstance(policy_basis, dict) or (
        policy_basis.get("basis_set_id"),
        policy_basis.get("basis_set_digest"),
    ) != (basis_set_id, basis_set_digest):
        raise ValueError("Policy checkpoint basis differs from the corpus basis")

    source_corpus = manifest.get("source_corpus")
    if not isinstance(source_corpus, dict):
        raise ValueError("Rollout policy manifest has no source_corpus identity")
    if source_corpus.get("manifest_digest") != corpus_manifest_digest:
        raise ValueError("Policy checkpoint was trained on a different corpus")
    if (
        source_corpus.get("basis_set_id", basis_set_id),
        source_corpus.get("basis_set_digest", basis_set_digest),
    ) != (basis_set_id, basis_set_digest):
        raise ValueError("Policy training manifest has different basis provenance")
    task_roles = source_corpus.get("task_roles")
    if not isinstance(task_roles, dict) or "held-out-evaluation-only" not in str(
        task_roles.get("test", {}).get("role", "")
    ):
        raise ValueError("Policy training manifest does not keep test held out")
    prefixes = manifest.get("prefix_examples")
    if not isinstance(prefixes, dict) or prefixes.get("test") != 0:
        raise ValueError("Policy training manifest materialized held-out prefixes")

    wandb_lineage = manifest.get("wandb")
    if (
        not isinstance(wandb_lineage, dict)
        or wandb_lineage.get("entity") != PERSONAL_WANDB_ENTITY
    ):
        raise ValueError("Policy checkpoint has no personal W&B lineage")
    training_project = wandb_lineage.get("project")
    training_url = wandb_lineage.get("url")
    if not isinstance(training_project, str) or not isinstance(training_url, str):
        raise ValueError("Policy checkpoint has incomplete W&B lineage")
    _validate_wandb_url(training_url, project=training_project)
    if _read_wandb_url(checkpoint / WANDB_LINK_FILE) != training_url:
        raise ValueError("Policy W&B link differs from its training manifest")

    required = {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINING_MANIFEST_FILE,
        TRAINER_STATE_FILE,
        WANDB_LINK_FILE,
    }
    files = sorted(path for path in checkpoint.iterdir() if path.is_file())
    actual_names = {path.name for path in files}
    if actual_names != required:
        raise ValueError(
            "Rollout policy checkpoint must contain exactly its five weights-only "
            f"lineage files; missing={sorted(required - actual_names)}, "
            f"unexpected={sorted(actual_names - required)}"
        )
    recorded_artifacts = manifest.get("artifacts")
    if not isinstance(recorded_artifacts, dict):
        raise ValueError("Policy training manifest has no artifact hashes")
    for name in (MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE, WANDB_LINK_FILE):
        if recorded_artifacts.get(name) != _sha256_file(checkpoint / name):
            raise ValueError(f"Policy checkpoint artifact hash mismatch: {name}")
    artifacts = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    return {
        "path": str(checkpoint.resolve()),
        "artifact_files": artifacts,
        "training_manifest_sha256": _sha256_file(checkpoint / TRAINING_MANIFEST_FILE),
        "policy_config_sha256": _sha256_file(checkpoint / CONFIG_FILE),
        "weights": next(row for row in artifacts if row["name"] == MODEL_FILE),
        "training": manifest,
    }


def _checkpoint_signature(identity: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    try:
        return tuple(
            sorted(
                (str(row["name"]), str(row["sha256"]))
                for row in identity["artifact_files"]
            )
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint identity has no artifact signature") from error


def _load_scorer(checkpoint: Path, device: str) -> ActionScorer:
    from wandering_light.rollout_policy import RolloutPolicy

    return RolloutPolicy.load(checkpoint, device=device)


def _default_predictor_factory(
    scorer: ActionScorer, width: int, max_depth: int, budget: int
) -> RolloutPredictor:
    from wandering_light.solver import RolloutPredictor

    return RolloutPredictor(
        scorer,
        beam_width=width,
        max_depth=max_depth,
        budget=budget,
    )


def _json_diagnostics(
    result: Any, witness_function_names: Sequence[str]
) -> dict[str, Any] | None:
    """Retain the optional core trace and mark witness-prefix beam survival."""

    trace_missing = object()
    trace = getattr(result, "trace", trace_missing)
    if trace is not trace_missing and trace is not None:
        depths = []
        first_pruned = None
        for row in trace:
            retained = [list(path) for path in row.retained_paths]
            witness_prefix = list(witness_function_names[: row.depth])
            survived = witness_prefix in retained
            if not survived and first_pruned is None:
                first_pruned = row.depth
            depths.append(
                {
                    "depth": row.depth,
                    "retained_paths": retained,
                    "unique_successors": row.unique_successors,
                    "witness_prefix_retained": survived,
                }
            )
        termination = getattr(result, "termination", None)
        right_censored = termination in {"solved", "budget"}
        return {
            "trace_supported": True,
            "depths": depths,
            "observed_completed_layers": len(depths),
            "right_censored": right_censored,
            "censoring_reason": termination if right_censored else None,
            "witness_first_pruned_depth": first_pruned,
            "witness_survived_all_observed_depths": first_pruned is None,
        }

    for name in ("diagnostics", "beam_diagnostics", "oracle_prefix_survival"):
        value = getattr(result, name, None)
        if value is None:
            continue
        if isinstance(value, Mapping):
            candidate: Any = dict(value)
        else:
            candidate = {name: value}
        try:
            return json.loads(_canonical_json(candidate))
        except (TypeError, ValueError):
            return {"unserializable": repr(candidate)}
    return None


def evaluate_rollout_records(
    records: Sequence[BasisTaskRecord],
    *,
    scorer: ActionScorer,
    available_functions: FunctionDefSet,
    widths: Sequence[int] = LOCKED_WIDTHS,
    max_depth: int = LOCKED_MAX_DEPTH,
    execution_budget: int = LOCKED_EXECUTION_BUDGET,
    predictor_factory: PredictorFactory = _default_predictor_factory,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate each width from scratch and retain one auditable row per task."""

    executor = Executor(available_functions)
    rows: list[dict[str, Any]] = []
    for width in widths:
        predictor = predictor_factory(scorer, width, max_depth, execution_budget)
        for index, record in enumerate(records, start=1):
            started = time.perf_counter()
            result = predictor.search_with_cost(
                record.input_value,
                record.output_value,
                available_functions,
            )
            seconds = time.perf_counter() - started
            termination = result.termination
            executions = result.executions
            if termination not in TERMINATIONS:
                raise ValueError(
                    f"Rollout returned unknown termination: {termination!r}"
                )
            if type(executions) is not int or not 0 <= executions <= execution_budget:
                raise ValueError(
                    f"Rollout returned invalid execution cost: {executions!r}"
                )
            solved = termination == "solved"
            names = [function.name for function in result.function_defs]
            exact_match = tuple(names) == tuple(record.witness_function_names)
            if solved:
                execution = executor.execute_trajectory(
                    TrajectorySpec(record.input_value, result.function_defs)
                )
                if (
                    not execution.success
                    or execution.trajectory is None
                    or execution.trajectory.output != record.output_value
                ):
                    raise ValueError(
                        f"Rollout reported a false solution for task {record.task_id}"
                    )
            row = {
                "task_id": record.task_id,
                "split": record.split,
                "certified_distance": record.metadata["certified_distance"],
                "root_digest": record.metadata["root_digest"],
                "beam_width": width,
                "success": solved,
                "executions": executions,
                "termination": termination,
                "solution_length": len(result.function_defs) if solved else None,
                "solution_function_names": names if solved else [],
                "witness_exact_match": solved and exact_match,
                "seconds": seconds,
                "diagnostics": _json_diagnostics(result, record.witness_function_names),
            }
            rows.append(row)
            if progress:
                print(
                    f"[rollout b={width}] {index}/{len(records)} "
                    f"d={row['certified_distance']} {termination} "
                    f"executions={executions:,}",
                    flush=True,
                )
    return rows


def _group_records(
    records: Sequence[BasisTaskRecord],
) -> dict[str, list[BasisTaskRecord]]:
    groups = {
        str(distance): [
            record
            for record in records
            if record.metadata["certified_distance"] == distance
        ]
        for distance in range(1, 9)
    }
    groups["5-8"] = [
        record for record in records if 5 <= record.metadata["certified_distance"] <= 8
    ]
    groups["7-8"] = [
        record for record in records if record.metadata["certified_distance"] in {7, 8}
    ]
    groups["all"] = list(records)
    return groups


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    costs = [int(row["executions"]) for row in rows]
    solved_rows = [row for row in rows if row["success"]]
    hit_costs = [int(row["executions"]) for row in solved_rows]
    lengths = [int(row["solution_length"]) for row in solved_rows]
    diagnostics = [row["diagnostics"] for row in rows if row.get("diagnostics")]
    trace_diagnostics = [
        row for row in diagnostics if row.get("trace_supported") is True
    ]
    first_pruned = Counter(
        str(row["witness_first_pruned_depth"])
        for row in diagnostics
        if row.get("witness_first_pruned_depth") is not None
    )
    traced_depths = sorted(
        {
            int(depth["depth"])
            for diagnostic in trace_diagnostics
            for depth in diagnostic.get("depths", ())
        }
    )
    trace_by_depth = {}
    for depth in traced_depths:
        cells = [
            cell
            for diagnostic in trace_diagnostics
            for cell in diagnostic.get("depths", ())
            if cell["depth"] == depth
        ]
        trace_by_depth[str(depth)] = {
            "tasks_observed": len(cells),
            "witness_prefix_retained": sum(
                bool(cell["witness_prefix_retained"]) for cell in cells
            ),
            "witness_prefix_survival_rate": sum(
                bool(cell["witness_prefix_retained"]) for cell in cells
            )
            / len(cells),
            "mean_unique_successors": statistics.fmean(
                int(cell["unique_successors"]) for cell in cells
            ),
        }
    return {
        "tasks": len(rows),
        "solved": len(solved_rows),
        "solve_rate": len(solved_rows) / len(rows) if rows else 0.0,
        "exact_witness_matches": sum(bool(row["witness_exact_match"]) for row in rows),
        "total_executions": sum(costs),
        "mean_executions": statistics.fmean(costs) if costs else 0.0,
        "median_executions": statistics.median(costs) if costs else 0.0,
        "max_executions": max(costs, default=0),
        "mean_hit_executions": statistics.fmean(hit_costs) if hit_costs else None,
        "mean_solution_length": statistics.fmean(lengths) if lengths else None,
        "termination_counts": dict(
            sorted(Counter(row["termination"] for row in rows).items())
        ),
        "mean_seconds": statistics.fmean(float(row["seconds"]) for row in rows)
        if rows
        else 0.0,
        "diagnostic_rows": len(diagnostics),
        "trace_supported_rows": len(trace_diagnostics),
        "right_censored_trace_rows": sum(
            bool(row.get("right_censored")) for row in trace_diagnostics
        ),
        "witness_first_pruned_depth_counts": dict(sorted(first_pruned.items())),
        "trace_by_depth": trace_by_depth,
    }


def aggregate_rollout_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    records: Sequence[BasisTaskRecord],
    widths: Sequence[int] = LOCKED_WIDTHS,
) -> tuple[dict[str, Any], dict[int, dict[str, Mapping[str, Any]]]]:
    """Validate the width/task cross product and aggregate fixed task groups."""

    expected = {record.task_id: record for record in records}
    actual: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["beam_width"]), str(row["task_id"]))
        if key in actual:
            raise ValueError(f"Duplicate rollout row for width/task {key}")
        record = expected.get(key[1])
        if record is None or (
            row["split"] != record.split
            or row["certified_distance"] != record.metadata["certified_distance"]
        ):
            raise ValueError(f"Rollout row does not match cohort task {key[1]}")
        actual[key] = row
    expected_keys = {(width, task_id) for width in widths for task_id in expected}
    if set(actual) != expected_keys:
        raise ValueError(
            "Rollout rows do not contain the exact width/task cross product"
        )

    groups = _group_records(records)
    per_task = {
        width: {task_id: actual[(width, task_id)] for task_id in expected}
        for width in widths
    }
    by_width: dict[str, Any] = {}
    for width in widths:
        by_width[str(width)] = {
            group: _aggregate_rows(
                [per_task[width][record.task_id] for record in group_records]
            )
            for group, group_records in groups.items()
        }
    return {"widths": list(widths), "by_width": by_width}, per_task


def load_bfs_reference(
    path: Path, records: Sequence[BasisTaskRecord]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and authenticate the preserved #50 W&B table for this exact cohort."""

    digest = _sha256_file(path)
    if digest != EXPECTED_BFS_REFERENCE_SHA256:
        raise ValueError(
            f"BFS reference hash differs from the preserved #50 table: {digest}"
        )
    payload = _read_json(path, label="BFS reference")
    if tuple(payload.get("columns", ())) != BFS_REFERENCE_COLUMNS:
        raise ValueError("BFS reference has unexpected columns")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("BFS reference has no data rows")

    rows = []
    for values in data:
        if not isinstance(values, list) or len(values) != len(BFS_REFERENCE_COLUMNS):
            raise ValueError("BFS reference contains a malformed row")
        raw = dict(zip(BFS_REFERENCE_COLUMNS, values, strict=True))
        if (
            not isinstance(raw["task_id"], str)
            or type(raw["distance"]) is not int
            or raw["success"] is not True
            or type(raw["executions"]) is not int
            or raw["executions"] < 0
            or not isinstance(raw["seconds"], (int, float))
            or isinstance(raw["seconds"], bool)
            or raw["seconds"] < 0
        ):
            raise ValueError("BFS reference contains invalid field values")
        rows.append(
            {
                "task_id": raw["task_id"],
                "certified_distance": raw["distance"],
                "success": True,
                "executions": raw["executions"],
                "seconds": float(raw["seconds"]),
            }
        )

    expected = {record.task_id: record for record in records}
    actual = {row["task_id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("BFS reference task IDs do not match the exact test cohort")
    for task_id, row in actual.items():
        if (
            row["certified_distance"]
            != expected[task_id].metadata["certified_distance"]
        ):
            raise ValueError(f"BFS reference distance mismatch for task {task_id}")

    groups = _group_records(records)
    aggregate = {
        group: {
            "tasks": len(group_records),
            "solved": len(group_records),
            "solve_rate": 1.0,
            "total_executions": sum(
                actual[record.task_id]["executions"] for record in group_records
            ),
            "mean_executions": statistics.fmean(
                actual[record.task_id]["executions"] for record in group_records
            ),
        }
        for group, group_records in groups.items()
    }
    identity = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "source_url": BFS_REFERENCE_URL,
        "cohort_task_ids_digest": EXPECTED_TEST_COHORT_DIGEST,
        "rows": len(rows),
        "aggregate": aggregate,
    }
    return rows, identity


def matched_bfs_comparison(
    *,
    rollout: Mapping[str, Any],
    per_task: Mapping[int, Mapping[str, Mapping[str, Any]]],
    bfs_rows: Sequence[Mapping[str, Any]],
    records: Sequence[BasisTaskRecord],
    widths: Sequence[int] = LOCKED_WIDTHS,
) -> dict[str, Any]:
    """Compare actual adaptive rollout spend with capped per-task BFS spend."""

    bfs_by_id = {str(row["task_id"]): row for row in bfs_rows}
    result: dict[str, Any] = {}
    for group, group_records in _group_records(records).items():
        ids = [record.task_id for record in group_records]
        bfs_costs = sorted(int(bfs_by_id[task_id]["executions"]) for task_id in ids)
        cells = {}
        for width in widths:
            aggregate = rollout["by_width"][str(width)][group]
            rollout_total = int(aggregate["total_executions"])
            rollout_mean = float(aggregate["mean_executions"])
            solved_target = int(aggregate["solved"])

            candidate_budgets = [0, *sorted(set(bfs_costs))]
            curve = []
            for budget in candidate_budgets:
                solved = sum(cost <= budget for cost in bfs_costs)
                spent = sum(min(cost, budget) for cost in bfs_costs)
                curve.append((budget, solved, spent))
            affordable = [
                point for point in curve if point[2] / len(ids) <= rollout_mean
            ]
            bfs_at_cost = max(affordable, key=lambda point: (point[1], point[2]))
            if solved_target:
                matched_budget = bfs_costs[solved_target - 1]
                bfs_spend = sum(min(cost, matched_budget) for cost in bfs_costs)
                bfs_solved = sum(cost <= matched_budget for cost in bfs_costs)
            else:
                matched_budget = bfs_spend = bfs_solved = 0
            cells[str(width)] = {
                "rollout_solved": solved_target,
                "rollout_tasks": len(ids),
                "rollout_total_executions": rollout_total,
                "rollout_mean_executions": rollout_mean,
                "bfs_uniform_budget_at_or_below_rollout_mean_spend": bfs_at_cost[0],
                "bfs_solved_at_rollout_mean_spend": bfs_at_cost[1],
                "bfs_mean_executions_at_rollout_mean_spend": bfs_at_cost[2] / len(ids),
                "bfs_uniform_budget_for_at_least_rollout_solve_count": matched_budget,
                "bfs_solved_at_matched_budget": bfs_solved,
                "bfs_total_executions_for_at_least_rollout_solve_count": bfs_spend,
                "bfs_to_rollout_total_execution_ratio": (
                    bfs_spend / rollout_total
                    if solved_target and rollout_total
                    else None
                ),
                "rollout_cheaper_than_bfs_at_equal_or_higher_solve_count": (
                    bool(solved_target)
                    and bfs_solved >= solved_target
                    and rollout_total < bfs_spend
                ),
            }
        result[group] = cells
    return result


def _selection_key(width: int, rollout: Mapping[str, Any]) -> tuple[float, ...]:
    rows = rollout["by_width"][str(width)]
    primary = rows["5-8"]
    far = rows["7-8"]
    return (
        float(primary["solved"]),
        float(far["solved"]),
        float(rows["all"]["solved"]),
        -float(primary["mean_executions"]),
        -float(width),
    )


def select_validation_width(summary: Mapping[str, Any]) -> int:
    if (
        summary.get("status") != "complete"
        or summary.get("corpus", {}).get("split") != "validation"
    ):
        raise ValueError("Width selection requires one complete validation summary")
    protocol = summary.get("protocol", {})
    if tuple(protocol.get("widths", ())) != LOCKED_WIDTHS:
        raise ValueError("Validation summary does not contain the locked widths")
    rollout = summary.get("rollout")
    if not isinstance(rollout, Mapping):
        raise ValueError("Validation summary has no rollout metrics")
    return max(LOCKED_WIDTHS, key=lambda width: _selection_key(width, rollout))


def build_selection_lock(
    validation_summary: Mapping[str, Any],
    *,
    validation_summary_sha256: str,
    locked_at_utc: str | None = None,
) -> dict[str, Any]:
    champion = select_validation_width(validation_summary)
    wandb_evidence = validation_summary.get("wandb")
    if (
        not isinstance(wandb_evidence, Mapping)
        or wandb_evidence.get("entity") != PERSONAL_WANDB_ENTITY
        or not isinstance(wandb_evidence.get("project"), str)
        or not isinstance(wandb_evidence.get("url"), str)
    ):
        raise ValueError("Validation summary has no personal W&B run evidence")
    _validate_wandb_url(
        str(wandb_evidence["url"]), project=str(wandb_evidence["project"])
    )
    source = validation_summary.get("source_code", {}).get("runner", {})
    if not isinstance(source.get("path"), str) or not isinstance(
        source.get("sha256"), str
    ):
        raise ValueError("Validation summary has no frozen evaluator identity")
    checkpoint = validation_summary.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Validation summary has no checkpoint identity")
    protocol = validation_summary["protocol"]
    cohort = validation_summary["cohort"]
    corpus = validation_summary["corpus"]
    locked_at = locked_at_utc or datetime.now(UTC).isoformat()
    _parse_timestamp(locked_at, label="Selection lock")
    candidates = [
        {
            "beam_width": width,
            "selection_key": list(_selection_key(width, validation_summary["rollout"])),
            "distance_5_8": validation_summary["rollout"]["by_width"][str(width)][
                "5-8"
            ],
            "distance_7_8": validation_summary["rollout"]["by_width"][str(width)][
                "7-8"
            ],
        }
        for width in LOCKED_WIDTHS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "locked-before-held-out-test-scoring",
        "locked_at_utc": locked_at,
        "discipline": {
            "selection_data": "validation summary only",
            "held_out_test_integrity_read_allowed": True,
            "held_out_test_policy_scored": False,
            "held_out_test_outcomes_used_for_selection": False,
        },
        "frozen_evaluator": dict(source),
        "validation_summary_sha256": validation_summary_sha256,
        "validation_identity": {
            "corpus_manifest_digest": corpus["manifest_digest"],
            "basis_set_id": corpus["basis_set_id"],
            "basis_set_digest": corpus["basis_set_digest"],
            "cohort_task_ids_digest": cohort["task_ids_digest"],
            "cohort_protocol": {
                key: cohort[key]
                for key in ("selection", "seed", "tasks", "by_certified_distance")
            },
            "evaluation_protocol": dict(protocol),
        },
        "checkpoint_signature": [
            {"name": name, "sha256": digest}
            for name, digest in _checkpoint_signature(checkpoint)
        ],
        "checkpoint_training_manifest_sha256": checkpoint["training_manifest_sha256"],
        "candidates": candidates,
        "champion": {
            "beam_width": champion,
            "selection_key": list(
                _selection_key(champion, validation_summary["rollout"])
            ),
        },
    }


def create_selection_lock(
    validation_summary_path: Path, *, output: Path
) -> dict[str, Any]:
    summary = _read_json(validation_summary_path, label="validation summary")
    lock = build_selection_lock(
        summary,
        validation_summary_sha256=_sha256_file(validation_summary_path),
    )
    try:
        _write_json(output, lock, exclusive=True)
    except FileExistsError as error:
        raise FileExistsError(
            f"Selection lock already exists; refusing to overwrite: {output}"
        ) from error
    return lock


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} has no timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} has an invalid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp must include a UTC offset")
    return parsed


def validate_test_selection_lock(
    path: Path,
    *,
    config: EvaluationConfig,
    checkpoint_identity: Mapping[str, Any],
    corpus_identity: Mapping[str, Any],
    cohort_identity: Mapping[str, Any],
) -> dict[str, Any]:
    lock = _read_json(path, label="selection lock")
    if lock.get("status") != "locked-before-held-out-test-scoring":
        raise ValueError("Selection lock is not ready for held-out scoring")
    discipline = lock.get("discipline", {})
    if (
        discipline.get("selection_data") != "validation summary only"
        or discipline.get("held_out_test_policy_scored") is not False
        or discipline.get("held_out_test_outcomes_used_for_selection") is not False
    ):
        raise ValueError("Selection lock has invalid held-out discipline")
    frozen = lock.get("frozen_evaluator", {})
    if frozen.get("sha256") != _sha256_file(Path(__file__).resolve()):
        raise ValueError("Selection lock does not freeze this evaluator")
    locked_signature = tuple(
        sorted(
            (str(row["name"]), str(row["sha256"]))
            for row in lock.get("checkpoint_signature", ())
        )
    )
    if locked_signature != _checkpoint_signature(checkpoint_identity):
        raise ValueError("Test checkpoint is not the validation-locked checkpoint")
    if lock.get("checkpoint_training_manifest_sha256") != checkpoint_identity.get(
        "training_manifest_sha256"
    ):
        raise ValueError("Test checkpoint training manifest differs from the lock")

    identity = lock.get("validation_identity", {})
    cohort_protocol = {
        key: cohort_identity[key]
        for key in ("selection", "seed", "tasks", "by_certified_distance")
    }
    if (
        identity.get("corpus_manifest_digest") != corpus_identity.get("manifest_digest")
        or identity.get("basis_set_id") != corpus_identity.get("basis_set_id")
        or identity.get("basis_set_digest") != corpus_identity.get("basis_set_digest")
        or identity.get("cohort_protocol") != cohort_protocol
        or identity.get("evaluation_protocol") != _evaluation_protocol(config)
    ):
        raise ValueError(
            "Held-out configuration differs from the frozen validation protocol"
        )
    locked_at = _parse_timestamp(lock.get("locked_at_utc"), label="Selection lock")
    started = datetime.now(UTC)
    if locked_at >= started:
        raise ValueError("Selection lock timestamp is not in the past")
    return {
        "selection_lock_path": str(path.resolve()),
        "selection_lock_sha256": _sha256_file(path),
        "selection_locked_at_utc": lock["locked_at_utc"],
        "selection_lock_validated_at_utc": started.isoformat(),
        "frozen_evaluator_sha256": frozen["sha256"],
        "validation_cohort_task_ids_digest": identity["cohort_task_ids_digest"],
        "champion_beam_width": lock["champion"]["beam_width"],
    }


def consume_test_selection_lock(
    path: Path,
    *,
    access: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    receipt_path = path.with_name(path.name + ".test-consumed.json")
    receipt = {
        "status": "held-out-test-scoring-started",
        "selection_lock_sha256": access["selection_lock_sha256"],
        "test_scoring_started_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_output_dir": str(output_dir.resolve()),
    }
    try:
        _write_json(receipt_path, receipt, exclusive=True)
    except FileExistsError as error:
        raise RuntimeError(
            f"Selection lock has already been consumed: {receipt_path}"
        ) from error
    return {
        "path": str(receipt_path.resolve()),
        "sha256": _sha256_file(receipt_path),
        **receipt,
    }


def _log_wandb_run(
    *,
    config: EvaluationConfig,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    """Publish aggregate and per-task evidence under the personal W&B entity."""

    import wandb

    run = wandb.init(
        entity=PERSONAL_WANDB_ENTITY,
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode="online",
        tags=["deep-corpus-v1", "issue-55", "rollout", config.split],
        config={
            "schema_version": SCHEMA_VERSION,
            "split": config.split,
            "protocol": summary["protocol"],
            "corpus": summary["corpus"],
            "cohort": summary["cohort"],
            "checkpoint_signature": summary["checkpoint"]["artifact_files"],
            "bfs_reference": summary.get("bfs_reference"),
        },
    )
    try:
        url = _validate_wandb_url(str(run.url), project=config.wandb_project)
        with_wandb = dict(summary)
        with_wandb["wandb"] = {
            "entity": PERSONAL_WANDB_ENTITY,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "url": url,
        }
        _write_json(output_dir / "summary.json", with_wandb)

        aggregate_table = []
        matched = summary.get("matched_compute")
        for width in config.widths:
            for group in [*(str(distance) for distance in range(1, 9)), "5-8", "7-8"]:
                cell = summary["rollout"]["by_width"][str(width)][group]
                comparison = matched[group][str(width)] if matched else {}
                aggregate_table.append(
                    [
                        width,
                        group,
                        cell["tasks"],
                        cell["solved"],
                        cell["solve_rate"],
                        cell["mean_executions"],
                        comparison.get("bfs_solved_at_rollout_mean_spend"),
                        comparison.get("bfs_to_rollout_total_execution_ratio"),
                    ]
                )
                run.log(
                    {
                        "eval/beam_width": width,
                        "eval/group": group,
                        "eval/solve_rate": cell["solve_rate"],
                        "eval/mean_executions": cell["mean_executions"],
                    }
                )
                for depth, trace in cell["trace_by_depth"].items():
                    run.log(
                        {
                            "trace/beam_width": width,
                            "trace/group": group,
                            "trace/depth": int(depth),
                            "trace/witness_prefix_survival_rate": trace[
                                "witness_prefix_survival_rate"
                            ],
                            "trace/mean_unique_successors": trace[
                                "mean_unique_successors"
                            ],
                        }
                    )

        run.log(
            {
                "rollout_aggregate": wandb.Table(
                    columns=[
                        "beam_width",
                        "group",
                        "tasks",
                        "solved",
                        "solve_rate",
                        "mean_executions",
                        "bfs_solved_at_rollout_mean_spend",
                        "bfs_to_rollout_execution_ratio",
                    ],
                    data=aggregate_table,
                ),
                "rollout_tasks": wandb.Table(
                    columns=[
                        "task_id",
                        "distance",
                        "beam_width",
                        "success",
                        "executions",
                        "termination",
                        "solution_length",
                        "witness_exact_match",
                    ],
                    data=[
                        [
                            row["task_id"],
                            row["certified_distance"],
                            row["beam_width"],
                            row["success"],
                            row["executions"],
                            row["termination"],
                            row["solution_length"],
                            row["witness_exact_match"],
                        ]
                        for row in rows
                    ],
                ),
            }
        )
        selected = str(summary["primary"]["validation_selected_beam_width"])
        selected_5_8 = summary["rollout"]["by_width"][selected]["5-8"]
        selected_7_8 = summary["rollout"]["by_width"][selected]["7-8"]
        run.summary.update(
            {
                "validation_selected_beam_width": int(selected),
                "distance_5_8_solved": selected_5_8["solved"],
                "distance_5_8_tasks": selected_5_8["tasks"],
                "distance_7_8_solved": selected_7_8["solved"],
                "distance_7_8_tasks": selected_7_8["tasks"],
                "distance_5_8_mean_executions": selected_5_8["mean_executions"],
            }
        )
        artifact = wandb.Artifact(
            name=f"rollout-policy-{config.split}-evaluation-{run.id}",
            type="evaluation",
            metadata={
                "corpus_manifest_digest": summary["corpus"]["manifest_digest"],
                "basis_set_digest": summary["corpus"]["basis_set_digest"],
                "cohort_task_ids_digest": summary["cohort"]["task_ids_digest"],
                "checkpoint_weights_sha256": summary["checkpoint"]["weights"]["sha256"],
            },
        )
        for name in ("summary.json", "cohort.json", "rollout_results.jsonl.gz"):
            artifact.add_file(str(output_dir / name), name=name)
        run.log_artifact(artifact)
        return url
    finally:
        run.finish()


def _assert_cohort_contract(split: str, cohort: Mapping[str, Any]) -> None:
    expected_histogram = {str(distance): 12 for distance in range(1, 9)}
    if (
        cohort.get("seed") != LOCKED_SEED
        or cohort.get("tasks") != 96
        or cohort.get("by_certified_distance") != expected_histogram
    ):
        raise ValueError(
            "Selected cohort differs from the locked 12-per-distance protocol"
        )
    if split == "test" and cohort.get("task_ids_digest") != EXPECTED_TEST_COHORT_DIGEST:
        raise ValueError(
            "Selected test tasks differ from the exact #50 held-out cohort"
        )


def run_evaluation(
    config: EvaluationConfig,
    *,
    scorer_loader: Callable[[Path, str], ActionScorer] = _load_scorer,
    predictor_factory: PredictorFactory = _default_predictor_factory,
) -> dict[str, Any]:
    """Verify provenance, score fixed widths, aggregate, and persist one run."""

    _validate_config(config)
    configure_unbounded_int_serialization()
    manifest, basis, records, verification = verify_and_load_split(
        config.corpus_dir, config.split
    )
    cohort = select_balanced_cohort(
        records,
        tasks_per_distance=config.tasks_per_distance,
        seed=config.seed,
    )
    corpus_identity = _corpus_identity(config.corpus_dir, manifest, config.split)
    cohort_identity = _cohort_identity(cohort, config.seed)
    _assert_cohort_contract(config.split, cohort_identity)
    checkpoint_identity = _checkpoint_identity(
        config.checkpoint,
        corpus_manifest_digest=manifest["manifest_digest"],
        basis_set_id=manifest["basis_set_id"],
        basis_set_digest=manifest["basis_set_digest"],
    )
    source_identity = _source_identity()
    access = None
    if config.split == "test" and config.selection_lock is not None:
        access = validate_test_selection_lock(
            config.selection_lock,
            config=config,
            checkpoint_identity=checkpoint_identity,
            corpus_identity=corpus_identity,
            cohort_identity=cohort_identity,
        )
    bfs_rows: list[dict[str, Any]] = []
    bfs_identity = None
    if config.bfs_reference is not None:
        # Authenticate the exact task-level reference before consuming the lock
        # or invoking the policy on a held-out target.
        bfs_rows, bfs_identity = load_bfs_reference(config.bfs_reference, cohort)

    preview = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry-run",
        "corpus": corpus_identity,
        "cohort": cohort_identity,
        "checkpoint": checkpoint_identity,
        "protocol": _evaluation_protocol(config),
        "source_code": source_identity,
        "verification": verification,
        "test_access": access,
        "bfs_reference": bfs_identity,
    }
    if config.dry_run:
        return preview
    if config.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a fresh path: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    if access is not None and config.selection_lock is not None:
        access["consumption_receipt"] = consume_test_selection_lock(
            config.selection_lock,
            access=access,
            output_dir=config.output_dir,
        )
    _write_json(config.output_dir / "cohort.json", cohort_identity)

    scorer = scorer_loader(config.checkpoint, config.device)
    available_functions = basis.as_function_set()
    started = time.perf_counter()
    rows = evaluate_rollout_records(
        cohort,
        scorer=scorer,
        available_functions=available_functions,
        widths=config.widths,
        max_depth=config.max_depth,
        execution_budget=config.execution_budget,
        predictor_factory=predictor_factory,
        progress=config.progress,
    )
    runtime = time.perf_counter() - started
    write_jsonl_gz(rows, config.output_dir / "rollout_results.jsonl.gz")
    rollout, per_task = aggregate_rollout_rows(
        rows,
        records=cohort,
        widths=config.widths,
    )

    matched = None
    if bfs_rows:
        matched = matched_bfs_comparison(
            rollout=rollout,
            per_task=per_task,
            bfs_rows=bfs_rows,
            records=cohort,
            widths=config.widths,
        )

    selected_width = (
        int(access["champion_beam_width"])
        if access is not None
        else max(config.widths, key=lambda width: _selection_key(width, rollout))
    )
    summary = {
        **preview,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "test_access": access,
        "runtime": {
            "seconds": runtime,
            "searches": len(rows),
            "searches_per_second": len(rows) / runtime if runtime else None,
            "device": config.device,
        },
        "rollout": rollout,
        "matched_compute": matched,
        "primary": {
            "question": "Does materialized-state rollout solve certified distance 5-8?",
            "validation_selected_beam_width": selected_width,
            "selected_width_distance_5_8": rollout["by_width"][str(selected_width)][
                "5-8"
            ],
            "selected_width_distance_7_8": rollout["by_width"][str(selected_width)][
                "7-8"
            ],
            "by_width": {
                str(width): rollout["by_width"][str(width)]["5-8"]
                for width in config.widths
            },
            "one_shot_reference": {
                "opt_125m_solved_per_distance_5_8": [0, 0, 0, 0],
                "smollm2_360m_solved_per_distance_5_8": [0, 0, 0, 0],
            },
        },
        "diagnostics": {
            "core_trace_available": any(
                row["diagnostics"] and row["diagnostics"].get("trace_supported") is True
                for row in rows
            ),
            "note": (
                "Completed-layer core traces are retained per task and aggregated "
                "by depth. Each depth rate is conditional on tasks that completed "
                "that layer; earlier solutions and mid-layer budget exits are "
                "right-censored. First-pruned counts likewise cover observed "
                "pruning only, not a complete failure-time distribution. Survival "
                "is against the recorded witness prefix, which is optimal but not "
                "the complete optimal-action set inward. "
                "Path survival can also understate state survival when structural "
                "deduplication keeps another path to the same state."
            ),
        },
    }
    _write_json(config.output_dir / "summary.json", summary)
    wandb_url = _log_wandb_run(
        config=config,
        summary=summary,
        rows=rows,
        output_dir=config.output_dir,
    )
    summary["wandb"] = {
        "entity": PERSONAL_WANDB_ENTITY,
        "project": config.wandb_project,
        "run_name": config.wandb_run_name,
        "url": wandb_url,
    }
    _write_json(config.output_dir / "summary.json", summary)
    (config.output_dir / "wandb_run.url").write_text(
        f"[InternetShortcut]\nURL={wandb_url}\n", encoding="utf-8"
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> tuple[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    evaluate.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluate.add_argument("--selection-lock", type=Path)
    evaluate.add_argument("--bfs-reference", type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--wandb-entity", default=PERSONAL_WANDB_ENTITY)
    evaluate.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    evaluate.add_argument("--wandb-run-name")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--quiet", action="store_true")

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--validation-summary", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "freeze":
        return "freeze", (arguments.validation_summary, arguments.output)
    return "evaluate", EvaluationConfig(
        checkpoint=arguments.checkpoint,
        output_dir=arguments.output_dir,
        corpus_dir=arguments.corpus_dir,
        split=arguments.split,
        selection_lock=arguments.selection_lock,
        bfs_reference=arguments.bfs_reference,
        device=arguments.device,
        wandb_entity=arguments.wandb_entity,
        wandb_project=arguments.wandb_project,
        wandb_run_name=arguments.wandb_run_name,
        dry_run=arguments.dry_run,
        progress=not arguments.quiet,
    )


def main(argv: Sequence[str] | None = None) -> None:
    command, value = _parse_args(argv)
    if command == "freeze":
        summary, output = value
        result = create_selection_lock(summary, output=output)
    else:
        result = run_evaluation(value)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
