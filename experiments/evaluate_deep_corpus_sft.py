"""Matched-execution evaluation for the deep-corpus-v1 SFT experiment.

The evaluator deliberately keeps model sampling, program parsing, execution,
and aggregation separate.  A single ordered draw of 64 completions supplies
the nested solve@1, solve@8, and solve@64 results.  Every syntactically valid
program is executed through :class:`wandering_light.executor.Executor`; exact
match to the stored witness is recorded alongside, but never used as the
solver correctness criterion.

The BFS arm runs on the exact same task IDs and consumes the explicit
``BFSSearchResult.executions`` counter.  Both arms therefore report cost in the
unit required by issue #50: one execution per basis function applied (or tried,
for BFS).  Model generation FLOPs and wall time are retained as diagnostics,
not silently mixed into that unit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from experiments.generate_deep_corpus import load_corpus, verify_corpus
from experiments.train_deep_corpus_sft import (
    PRETRAINED_MODEL_LOADER,
    assert_weights_only_artifact,
    resolve_base_model,
    training_row_for_record,
)
from wandering_light.basis_set import BasisSet, load_basis_set
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.solver import BFSPredictor
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from wandering_light.basis_dataset import BasisTaskRecord


SCHEMA_VERSION = 1
PERSONAL_WANDB_ENTITY = "abhishekraok-na"
DEFAULT_WANDB_PROJECT = "wandering-light-deep-corpus-v1"
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v1")
DEFAULT_SAMPLE_COUNTS = (1, 8, 64)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
MODEL_COST_DEFINITION = (
    "Each strictly parsed sampled program is charged its full function sequence "
    "length; malformed text is charged zero and cannot solve. Sampling stops for "
    "cost accounting at the first execution-correct program in each prefix."
)


@dataclass(frozen=True)
class ParsedProgram:
    """Strict parse result that never confuses malformed text with emptiness."""

    valid: bool
    names: tuple[str, ...]
    functions: FunctionDefList
    error: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for one validation or held-out test evaluation."""

    checkpoint: Path
    output_dir: Path
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    split: str = "validation"
    selection_lock: Path | None = None
    tasks_per_distance: int = 12
    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS
    seed: int = 20260816
    generation_batch_size: int = 32
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    bfs_budget: int = 20_000_000
    run_bfs: bool = True
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_run_name: str | None = None
    dry_run: bool = False


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


def configure_deterministic_cuda_environment() -> str:
    """Configure cuBLAS before deterministic CUDA generation starts.

    PyTorch's deterministic-algorithm mode requires one of two cuBLAS workspace
    layouts.  Establish the larger documented layout when the environment is
    absent or invalid so an inherited shell setting cannot make a run fail only
    after the checkpoint has been loaded onto the GPU.
    """

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in _SUPPORTED_CUBLAS_WORKSPACE_CONFIGS:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    return os.environ["CUBLAS_WORKSPACE_CONFIG"]


def configure_unbounded_int_serialization() -> int:
    """Allow valid DSL outputs to contain integers beyond Python's text guard.

    The evaluator serializes the actual output for diagnostics. Python 3.11+
    otherwise raises for integers longer than the process limit (normally 4,300
    digits), even though execution itself succeeded. This is deterministic and
    changes only diagnostic JSON conversion, not solver semantics.
    """

    setter = getattr(sys, "set_int_max_str_digits", None)
    if setter is not None:
        setter(0)
    getter = getattr(sys, "get_int_max_str_digits", None)
    return int(getter()) if getter is not None else 0


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def write_jsonl_gz(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    """Write deterministic compressed rows for auditable raw predictions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8") as text_file,
    ):
        for row in rows:
            text_file.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as text_file:
        return [json.loads(line) for line in text_file]


def validate_personal_wandb_entity(entity: str) -> str:
    if entity.strip().lower() != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            "Deep-corpus W&B writes are restricted to the personal entity "
            f"{PERSONAL_WANDB_ENTITY!r}; got {entity!r}"
        )
    return PERSONAL_WANDB_ENTITY


def _validate_wandb_url(url: str, *, project: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() not in {"wandb.ai", "www.wandb.ai"}
        or len(parts) != 4
        or parts[0].lower() != PERSONAL_WANDB_ENTITY
        or parts[1] != project
        or parts[2] != "runs"
        or not parts[3]
    ):
        raise ValueError(
            "W&B returned a run URL outside the required personal entity/project: "
            f"{url!r}"
        )
    return url


def _validate_config(config: EvaluationConfig) -> None:
    validate_personal_wandb_entity(config.wandb_entity)
    if config.split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if config.split == "test" and config.selection_lock is None:
        raise ValueError(
            "Held-out test scoring requires a validation champion --selection-lock"
        )
    if config.split == "validation" and config.selection_lock is not None:
        raise ValueError("--selection-lock is only valid with --split test")
    if config.split == "test" and config.dry_run:
        raise ValueError(
            "A held-out test dry run would expose test labels without producing the "
            "locked evaluation; dry-run the validation split instead"
        )
    if config.split == "test" and (
        not config.run_bfs or config.bfs_budget != 20_000_000
    ):
        raise ValueError(
            "Held-out test requires matched BFS with the locked 20,000,000-"
            "execution safety budget"
        )
    if config.tasks_per_distance <= 0:
        raise ValueError("tasks_per_distance must be positive")
    if config.sample_counts != DEFAULT_SAMPLE_COUNTS:
        raise ValueError(
            "Issue #50 evaluation requires the nested sample counts (1, 8, 64)"
        )
    if config.generation_batch_size <= 0 or config.max_new_tokens <= 0:
        raise ValueError("generation_batch_size and max_new_tokens must be positive")
    if config.temperature <= 0 or not 0 < config.top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")
    if config.bfs_budget <= 0:
        raise ValueError("bfs_budget must be positive")
    if not config.dry_run and not config.wandb_run_name:
        raise ValueError("An actual evaluation requires --wandb-run-name")


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
    selection_lock_path: Path,
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    """Verify the validation-selected checkpoint before any test scoring starts."""

    try:
        payload = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read validation selection lock {selection_lock_path}: {error}"
        ) from error
    if payload.get("status") != "locked-before-held-out-test-scoring":
        raise ValueError("Validation selection lock is not ready for test scoring")
    discipline = payload.get("discipline", {})
    if (
        discipline.get("held_out_test_model_scored") is not False
        or discipline.get("held_out_test_outcomes_used_for_selection") is not False
        or discipline.get("selection_data") != "validation summaries only"
    ):
        raise ValueError("Validation selection lock has invalid test-score discipline")

    evaluator_sha256 = _sha256_file(Path(__file__).resolve())
    if payload.get("frozen_evaluator", {}).get("sha256") != evaluator_sha256:
        raise ValueError("Validation selection lock does not freeze this evaluator")

    assert_weights_only_artifact(checkpoint, require_complete=True)
    actual_signature = tuple(
        sorted(
            (path.name, _sha256_file(path))
            for path in checkpoint.iterdir()
            if path.is_file()
        )
    )
    champion = payload.get("champion", {})
    try:
        locked_signature = tuple(
            sorted(
                (row["name"], row["sha256"]) for row in champion["checkpoint_signature"]
            )
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Validation selection lock has no checkpoint signature"
        ) from error
    if locked_signature != actual_signature:
        raise ValueError("Test checkpoint is not the validation-locked champion")
    training_manifest_path = checkpoint / "training_manifest.json"
    if champion.get("checkpoint_training_manifest_sha256") != _sha256_file(
        training_manifest_path
    ):
        raise ValueError("Validation lock has a different training manifest")

    locked_at = _parse_timestamp(
        payload.get("locked_at_utc"), label="Validation selection lock"
    )
    access_started = datetime.now(UTC)
    if locked_at >= access_started:
        raise ValueError("Validation selection lock timestamp is not in the past")
    return {
        "selection_lock_path": str(selection_lock_path.resolve()),
        "selection_lock_sha256": _sha256_file(selection_lock_path),
        "selection_locked_at_utc": payload["locked_at_utc"],
        "selection_lock_validated_at_utc": access_started.isoformat(),
        "frozen_evaluator_sha256": evaluator_sha256,
        "locked_validation_identity": payload.get("validation_identity"),
        "discipline": (
            "Corpus integrity reads are allowed before selection; model-scored test "
            "outcomes begin only after the validation champion is locked."
        ),
    }


def _generation_protocol_from_config(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "samples_per_task": max(config.sample_counts),
        "sample_counts": list(config.sample_counts),
        "seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_new_tokens": config.max_new_tokens,
        "batch_size": config.generation_batch_size,
        "cost_definition": MODEL_COST_DEFINITION,
    }


def validate_locked_test_protocol(
    test_access: Mapping[str, Any],
    *,
    config: EvaluationConfig,
    corpus_identity: Mapping[str, Any],
    cohort_identity: Mapping[str, Any],
) -> None:
    """Reject any post-lock corpus, cohort, or sampling drift before test scoring."""

    identity = test_access.get("locked_validation_identity")
    if not isinstance(identity, dict):
        raise ValueError("Validation selection lock has no frozen protocol identity")
    cohort_protocol = {
        key: cohort_identity.get(key)
        for key in ("seed", "selection", "tasks", "by_certified_distance")
    }
    if (
        identity.get("corpus_manifest_digest") != corpus_identity.get("manifest_digest")
        or identity.get("basis_set_digest") != corpus_identity.get("basis_set_digest")
        or identity.get("cohort_protocol") != cohort_protocol
        or identity.get("generation_protocol")
        != _generation_protocol_from_config(config)
    ):
        raise ValueError(
            "Held-out test configuration differs from the validation-locked protocol"
        )


def consume_test_selection_lock(
    selection_lock_path: Path,
    *,
    test_access: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Create an exclusive receipt so the normal pipeline scores a lock once."""

    receipt_path = selection_lock_path.with_name(
        selection_lock_path.name + ".test-consumed.json"
    )
    receipt = {
        "status": "held-out-test-scoring-started",
        "selection_lock_sha256": test_access["selection_lock_sha256"],
        "test_scoring_started_at_utc": test_access["test_scoring_started_at_utc"],
        "evaluation_output_dir": str(output_dir.resolve()),
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as receipt_file:
            json.dump(receipt, receipt_file, indent=2, sort_keys=True)
            receipt_file.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            "Validation selection lock has already been consumed for test scoring: "
            f"{receipt_path}"
        ) from error
    return {
        "path": str(receipt_path.resolve()),
        "sha256": _sha256_file(receipt_path),
        **receipt,
    }


def ensure_corpus(corpus_dir: Path) -> Path:
    """Fetch missing payload files from the manifest's immutable Hub revision."""

    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deep-corpus manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [
        corpus_dir / metadata["path"]
        for metadata in manifest["splits"].values()
        if not (corpus_dir / metadata["path"]).is_file()
    ]
    if missing:
        fetch_corpus(manifest_path, destination=corpus_dir)
    return corpus_dir


def verify_and_load_split(
    corpus_dir: Path, split: str
) -> tuple[dict[str, Any], BasisSet, list[BasisTaskRecord], dict[str, Any]]:
    """Verify the entire root split before exposing one requested split."""

    ensure_corpus(corpus_dir)
    verification = verify_corpus(corpus_dir, recertify=0)
    if not verification["ok"]:
        raise ValueError(
            f"Deep corpus failed witness or root-split verification: {verification}"
        )
    manifest, records = load_corpus(corpus_dir, splits=(split,))
    basis = load_basis_set(manifest["basis_set_id"])
    if manifest["basis_set_digest"] != basis.digest:
        raise ValueError("Corpus basis digest does not match the loaded basis")
    if any(record.split != split for record in records):
        raise ValueError("Requested corpus split contains a differently labeled task")
    return manifest, basis, records, verification


def select_balanced_cohort(
    records: Sequence[BasisTaskRecord],
    *,
    tasks_per_distance: int,
    seed: int,
) -> list[BasisTaskRecord]:
    """Select one deterministic, balanced cohort without changing split membership."""

    by_distance: defaultdict[int, list[BasisTaskRecord]] = defaultdict(list)
    for record in records:
        distance = record.metadata.get("certified_distance")
        if isinstance(distance, bool) or not isinstance(distance, int):
            raise ValueError(f"Task {record.task_id} has no integer certified distance")
        by_distance[distance].append(record)
    expected_distances = set(range(1, 9))
    if set(by_distance) != expected_distances:
        raise ValueError(
            "Deep-corpus cohort must cover certified distances 1 through 8; "
            f"got {sorted(by_distance)}"
        )

    selected: list[BasisTaskRecord] = []
    for distance in sorted(by_distance):
        pool = by_distance[distance]
        if len(pool) < tasks_per_distance:
            raise ValueError(
                f"Distance {distance} has {len(pool)} tasks, fewer than requested "
                f"{tasks_per_distance}"
            )
        # A distance-specific integer seed avoids dependence on iteration over other
        # buckets while retaining the source split exactly as published.
        rng = random.Random(seed ^ (distance * 0x9E3779B1))
        chosen = rng.sample(pool, tasks_per_distance)
        selected.extend(sorted(chosen, key=lambda record: record.source_index))
    return selected


def parse_program(response: str, available_functions: FunctionDefSet) -> ParsedProgram:
    """Strictly parse one comma-separated basis program with diagnostics."""

    if not isinstance(response, str):
        return ParsedProgram(False, (), FunctionDefList(), "response is not a string")
    stripped = response.strip()
    if not stripped:
        return ParsedProgram(False, (), FunctionDefList(), "empty response")
    raw_names = stripped.split(",")
    names = tuple(name.strip() for name in raw_names)
    if any(not name for name in names):
        return ParsedProgram(
            False,
            names,
            FunctionDefList(),
            "empty function name in comma-separated response",
        )
    unknown = [
        name for name in names if name not in available_functions.name_to_function
    ]
    if unknown:
        return ParsedProgram(
            False,
            names,
            FunctionDefList(),
            f"unknown function names: {unknown}",
        )
    functions = FunctionDefList(
        [available_functions.name_to_function[name] for name in names]
    )
    return ParsedProgram(True, names, functions)


def evaluate_candidate(
    record: BasisTaskRecord,
    *,
    sample_index: int,
    response: str,
    available_functions: FunctionDefSet,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Parse and execute one sample, retaining exact and semantic correctness."""

    configure_unbounded_int_serialization()
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    parsed = parse_program(response, available_functions)
    exact_match = parsed.valid and parsed.names == record.witness_function_names
    row: dict[str, Any] = {
        "task_id": record.task_id,
        "split": record.split,
        "certified_distance": record.metadata["certified_distance"],
        "root_digest": record.metadata["root_digest"],
        "sample_index": sample_index,
        "raw_response": response,
        "parse_valid": parsed.valid,
        "parse_error": parsed.error,
        "parsed_function_names": list(parsed.names),
        # Issue #50 defines a sampled program's cost as its full program length.
        # Malformed text is not an executable basis program and therefore costs 0;
        # parse validity remains explicit so this cannot masquerade as a solution.
        "program_length": len(parsed.functions) if parsed.valid else 0,
        "charged_executions": len(parsed.functions) if parsed.valid else 0,
        "exact_match": exact_match,
        "executor_success": False,
        "execution_correct": False,
        "failed_at_step": None,
        "execution_error": None,
        "actual_output": None,
    }
    if not parsed.valid:
        return row

    executor = executor or Executor(available_functions)
    result = executor.execute_trajectory(
        TrajectorySpec(record.input_value, parsed.functions)
    )
    row["executor_success"] = bool(result.success)
    row["failed_at_step"] = result.failed_at_step
    row["execution_error"] = result.error_msg
    if result.success and result.trajectory is not None:
        row["actual_output"] = result.trajectory.output.to_string()
        row["execution_correct"] = result.trajectory.output == record.output_value
    return row


def _task_prefix_metrics(
    rows: Sequence[Mapping[str, Any]], sample_counts: Sequence[int]
) -> dict[int, dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["sample_index"])
    expected_indices = list(range(max(sample_counts)))
    actual_indices = [row["sample_index"] for row in ordered]
    if actual_indices != expected_indices:
        raise ValueError(
            "Candidate samples must be one complete ordered prefix; "
            f"expected {expected_indices}, got {actual_indices}"
        )

    metrics: dict[int, dict[str, Any]] = {}
    for sample_count in sample_counts:
        prefix = ordered[:sample_count]
        cumulative_executions = 0
        hit_cost: int | None = None
        hit_sample: int | None = None
        for row in prefix:
            if hit_cost is None:
                cumulative_executions += int(row["charged_executions"])
                if row["execution_correct"]:
                    hit_cost = cumulative_executions
                    hit_sample = int(row["sample_index"]) + 1
        metrics[sample_count] = {
            "execution_correct": hit_cost is not None,
            "exact_match": any(bool(row["exact_match"]) for row in prefix),
            "executions_spent": cumulative_executions,
            "execution_hit_cost": hit_cost,
            "execution_hit_sample": hit_sample,
            "samples_available": sample_count,
        }
    return metrics


def _mean(values: Sequence[int | float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate_task_metrics(
    task_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tasks = len(task_metrics)
    solved = sum(bool(row["execution_correct"]) for row in task_metrics)
    exact = sum(bool(row["exact_match"]) for row in task_metrics)
    costs = [int(row["executions_spent"]) for row in task_metrics]
    hit_costs = [
        int(row["execution_hit_cost"])
        for row in task_metrics
        if row["execution_hit_cost"] is not None
    ]
    return {
        "tasks": tasks,
        "solved": solved,
        "solve_rate": solved / tasks if tasks else 0.0,
        "exact_matches": exact,
        "exact_match_rate": exact / tasks if tasks else 0.0,
        "total_executions_spent": sum(costs),
        "mean_executions_spent": _mean(costs),
        "median_executions_spent": statistics.median(costs) if costs else 0.0,
        "mean_hit_executions": _mean(hit_costs),
        "max_hit_executions": max(hit_costs, default=None),
    }


def aggregate_model_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_records: Sequence[BasisTaskRecord],
    sample_counts: Sequence[int] = DEFAULT_SAMPLE_COUNTS,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    """Aggregate nested prefixes while proving task/sample alignment."""

    expected_by_id = {record.task_id: record for record in expected_records}
    if len(expected_by_id) != len(expected_records):
        raise ValueError("Expected cohort contains duplicate task IDs")
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        task_id = row["task_id"]
        record = expected_by_id.get(task_id)
        if record is None:
            raise ValueError(f"Candidate row references unexpected task {task_id}")
        if (
            row["split"] != record.split
            or row["certified_distance"] != record.metadata["certified_distance"]
        ):
            raise ValueError(f"Candidate metadata does not match task {task_id}")
        grouped[task_id].append(row)
    if set(grouped) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(grouped))
        raise ValueError(f"Candidate rows are missing cohort tasks: {missing}")

    per_task = {
        task_id: _task_prefix_metrics(grouped[task_id], sample_counts)
        for task_id in expected_by_id
    }
    by_distance: dict[str, Any] = {}
    for distance in range(1, 9):
        task_ids = [
            record.task_id
            for record in expected_records
            if record.metadata["certified_distance"] == distance
        ]
        by_distance[str(distance)] = {
            str(sample_count): _aggregate_task_metrics(
                [per_task[task_id][sample_count] for task_id in task_ids]
            )
            for sample_count in sample_counts
        }
    long_task_ids = [
        record.task_id
        for record in expected_records
        if record.metadata["certified_distance"] in {7, 8}
    ]
    combined = {
        str(sample_count): _aggregate_task_metrics(
            [per_task[task_id][sample_count] for task_id in long_task_ids]
        )
        for sample_count in sample_counts
    }
    return {
        "sample_counts": list(sample_counts),
        "by_certified_distance": by_distance,
        "distance_7_8": combined,
    }, per_task


def evaluate_bfs_records(
    records: Sequence[BasisTaskRecord],
    *,
    available_functions: FunctionDefSet,
    budget: int,
    max_depth: int = 8,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run cost-instrumented BFS on the exact model cohort."""

    predictor = BFSPredictor(budget=budget, max_depth=max_depth)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        result = predictor.search_with_cost(
            record.input_value, record.output_value, available_functions
        )
        elapsed = time.perf_counter() - started
        solved = result.termination == "solved"
        row = {
            "task_id": record.task_id,
            "split": record.split,
            "certified_distance": record.metadata["certified_distance"],
            "root_digest": record.metadata["root_digest"],
            "success": solved,
            "executions": result.executions,
            "solution_length": len(result.function_defs) if solved else None,
            "solution_function_names": [
                function.name for function in result.function_defs
            ],
            "termination": result.termination,
            "seconds": elapsed,
        }
        rows.append(row)
        if progress:
            print(
                f"[bfs] {index}/{len(records)} d={row['certified_distance']} "
                f"termination={row['termination']} executions={row['executions']:,} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )
    return rows


def validate_bfs_rows(
    bfs_rows: Sequence[Mapping[str, Any]], records: Sequence[BasisTaskRecord]
) -> dict[str, Any]:
    expected_by_id = {record.task_id: record for record in records}
    actual_by_id = {row["task_id"]: row for row in bfs_rows}
    if len(actual_by_id) != len(bfs_rows) or set(actual_by_id) != set(expected_by_id):
        raise ValueError("BFS rows do not match the model cohort task IDs exactly")
    for task_id, row in actual_by_id.items():
        record = expected_by_id[task_id]
        if row["certified_distance"] != record.metadata["certified_distance"]:
            raise ValueError(f"BFS distance mismatch for task {task_id}")
        if not row["success"]:
            raise ValueError(
                f"BFS did not solve certified distance-{row['certified_distance']} "
                f"task {task_id}: termination={row['termination']!r}. This is not "
                "an exhaustive baseline; raise the budget before reporting."
            )

    by_distance: dict[str, Any] = {}
    for distance in range(1, 9):
        subset = [row for row in bfs_rows if row["certified_distance"] == distance]
        executions = [int(row["executions"]) for row in subset]
        by_distance[str(distance)] = {
            "tasks": len(subset),
            "solved": sum(bool(row["success"]) for row in subset),
            "solve_rate": _mean([bool(row["success"]) for row in subset]),
            "mean_executions": _mean(executions),
            "median_executions": statistics.median(executions) if executions else 0,
            "max_executions": max(executions, default=0),
            "mean_seconds": _mean([float(row["seconds"]) for row in subset]),
        }
    return {"by_certified_distance": by_distance}


def _bfs_curve_for_group(
    bfs_rows: Sequence[Mapping[str, Any]], task_ids: set[str]
) -> list[dict[str, Any]]:
    subset = [row for row in bfs_rows if row["task_id"] in task_ids]
    hit_costs = sorted({int(row["executions"]) for row in subset if row["success"]})
    curve = [
        {
            "method": "bfs",
            "budget": 0,
            "mean_executions_spent": 0.0,
            "solve_rate": 0.0,
            "solved": 0,
            "tasks": len(subset),
        }
    ]
    for budget in hit_costs:
        solved = sum(
            bool(row["success"]) and int(row["executions"]) <= budget for row in subset
        )
        spent = [min(int(row["executions"]), budget) for row in subset]
        curve.append(
            {
                "method": "bfs",
                "budget": budget,
                "mean_executions_spent": _mean(spent),
                "solve_rate": solved / len(subset),
                "solved": solved,
                "tasks": len(subset),
            }
        )
    return curve


def matched_compute_comparison(
    *,
    per_task: Mapping[str, Mapping[int, Mapping[str, Any]]],
    bfs_rows: Sequence[Mapping[str, Any]],
    records: Sequence[BasisTaskRecord],
    sample_counts: Sequence[int],
) -> dict[str, Any]:
    """Put model prefix points and BFS's cost CDF on one execution axis."""

    groups: dict[str, list[BasisTaskRecord]] = {
        str(distance): [
            record
            for record in records
            if record.metadata["certified_distance"] == distance
        ]
        for distance in range(1, 9)
    }
    groups["7-8"] = [
        record for record in records if record.metadata["certified_distance"] in {7, 8}
    ]

    result: dict[str, Any] = {}
    for group_name, group_records in groups.items():
        task_ids = {record.task_id for record in group_records}
        group_bfs = [row for row in bfs_rows if row["task_id"] in task_ids]
        bfs_curve = _bfs_curve_for_group(bfs_rows, task_ids)
        model_points: list[dict[str, Any]] = []
        matched_rows: dict[str, Any] = {}
        for sample_count in sample_counts:
            task_rows = [per_task[task_id][sample_count] for task_id in task_ids]
            aggregate = _aggregate_task_metrics(task_rows)
            # A uniform per-task BFS budget does not equal its actual spend: tasks
            # found early stop early. Compare total/mean executions by selecting
            # the furthest BFS curve point whose measured capped spend is no more
            # than the model's adaptive sampling spend.
            eligible_bfs_points = [
                point
                for point in bfs_curve
                if point["mean_executions_spent"] <= aggregate["mean_executions_spent"]
            ]
            bfs_at_model_cost = max(
                eligible_bfs_points,
                key=lambda point: (
                    point["solved"],
                    point["mean_executions_spent"],
                ),
            )
            solved_target = int(aggregate["solved"])
            if solved_target:
                sorted_bfs_hits = sorted(
                    int(row["executions"]) for row in group_bfs if row["success"]
                )
                matched_budget = sorted_bfs_hits[solved_target - 1]
                bfs_spend = sum(
                    min(int(row["executions"]), matched_budget) for row in group_bfs
                )
                bfs_solved = sum(
                    bool(row["success"]) and int(row["executions"]) <= matched_budget
                    for row in group_bfs
                )
            else:
                matched_budget = 0
                bfs_spend = 0
                bfs_solved = 0
            model_spend = int(aggregate["total_executions_spent"])
            ratio = (
                bfs_spend / model_spend
                if model_spend > 0 and solved_target > 0
                else None
            )
            model_points.append(
                {
                    "method": "model",
                    "samples": sample_count,
                    "mean_executions_spent": aggregate["mean_executions_spent"],
                    "solve_rate": aggregate["solve_rate"],
                    "solved": aggregate["solved"],
                    "tasks": aggregate["tasks"],
                }
            )
            matched_rows[str(sample_count)] = {
                **aggregate,
                "bfs_uniform_budget_at_or_below_model_mean_execution_spend": (
                    bfs_at_model_cost["budget"]
                ),
                "bfs_mean_executions_at_model_mean_execution_spend": (
                    bfs_at_model_cost["mean_executions_spent"]
                ),
                "bfs_solved_at_model_mean_execution_spend": bfs_at_model_cost["solved"],
                "bfs_solve_rate_at_model_mean_execution_spend": bfs_at_model_cost[
                    "solve_rate"
                ],
                "bfs_uniform_budget_for_at_least_model_solve_count": matched_budget,
                "bfs_solved_at_matched_rate_budget": bfs_solved,
                "bfs_total_executions_for_at_least_model_solve_count": bfs_spend,
                "bfs_to_model_total_execution_ratio": ratio,
                "model_cheaper_than_bfs_at_equal_or_higher_solve_count": (
                    bool(solved_target)
                    and bfs_solved >= solved_target
                    and model_spend < bfs_spend
                ),
            }
        result[group_name] = {
            "model_points": model_points,
            "bfs_curve": bfs_curve,
            "matched": matched_rows,
        }
    return result


def _checkpoint_identity(
    checkpoint: Path,
    manifest_digest: str,
    basis: BasisSet,
    *,
    project: str,
) -> dict[str, Any]:
    assert_weights_only_artifact(checkpoint, require_complete=True)
    training_manifest_path = checkpoint / "training_manifest.json"
    if not training_manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint has no training_manifest.json: {training_manifest_path}"
        )
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("status") != "complete":
        raise ValueError("Checkpoint training manifest is not complete")
    base_model = training_manifest.get("base_model")
    if not isinstance(base_model, dict) or base_model.get("source") != "huggingface":
        raise ValueError("Checkpoint has no immutable Hugging Face base identity")
    try:
        resolve_base_model(base_model["repo_id"], base_model["revision"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Checkpoint has an invalid pretrained base identity"
        ) from error
    # Artifacts produced before this field was added are accepted because their
    # frozen runner source still proves the same from_pretrained call.
    if base_model.get("weight_initialization", "pretrained") != "pretrained":
        raise ValueError("Checkpoint was not initialized from pretrained weights")
    if (
        base_model.get("model_loader", PRETRAINED_MODEL_LOADER)
        != PRETRAINED_MODEL_LOADER
    ):
        raise ValueError("Checkpoint records an unsupported pretrained model loader")
    if training_manifest.get("basis_set") != basis.identity_dict():
        raise ValueError(
            "Checkpoint basis identity does not match the evaluation corpus"
        )
    source = training_manifest.get("source_corpus", {})
    if source.get("manifest_digest") != manifest_digest:
        raise ValueError("Checkpoint was trained from a different corpus manifest")
    task_roles = source.get("task_roles")
    expected_roles = {
        "discovery": "training-only",
        "validation": "trainer-evaluation-and-tuning-only",
        "test": "held-out-evaluation-only; never passed to trainer",
    }
    if not isinstance(task_roles, dict) or set(task_roles) != set(expected_roles):
        raise ValueError(
            "Checkpoint training provenance must identify discovery, validation, "
            "and held-out test roles"
        )
    for split, expected_role in expected_roles.items():
        role = task_roles[split]
        if (
            not isinstance(role, dict)
            or role.get("split") != split
            or role.get("role") != expected_role
            or not isinstance(role.get("records"), int)
            or role["records"] <= 0
        ):
            raise ValueError(f"Checkpoint has invalid {split!r} task-role provenance")
    hyperparameters = training_manifest.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict) or (
        hyperparameters.get("train_split"),
        hyperparameters.get("validation_split"),
        hyperparameters.get("held_out_split"),
    ) != ("discovery", "validation", "test"):
        raise ValueError("Checkpoint split hyperparameters permit data leakage")
    tokenization = training_manifest.get("tokenization")
    if not isinstance(tokenization, dict):
        raise ValueError("Checkpoint has no token-length provenance")
    for role in ("train", "validation"):
        audit = tokenization.get(role)
        if (
            not isinstance(audit, dict)
            or audit.get("complete_labels_only") is not True
            or audit.get("selected_records")
            != audit.get("retained_records", 0) + audit.get("excluded_records", 0)
        ):
            raise ValueError(
                f"Checkpoint has invalid complete-label tokenization for {role}"
            )
    wandb_identity = training_manifest.get("wandb", {})
    if not isinstance(wandb_identity, dict) or (
        wandb_identity.get("entity"),
        wandb_identity.get("project"),
    ) != (PERSONAL_WANDB_ENTITY, project):
        raise ValueError("Checkpoint W&B lineage is outside the required project")
    wandb_url = wandb_identity.get("url")
    if not isinstance(wandb_url, str):
        raise ValueError("Checkpoint training manifest has no W&B run URL")
    _validate_wandb_url(wandb_url, project=project)
    weights = sorted(
        path
        for path in checkpoint.iterdir()
        if path.suffix == ".safetensors"
        or (path.suffix == ".bin" and path.name.startswith("pytorch_model"))
    )
    if not weights:
        raise ValueError(f"Checkpoint has no model weights: {checkpoint}")
    artifact_files = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    ]
    return {
        "path": str(checkpoint.resolve()),
        "training_manifest_sha256": _sha256_file(training_manifest_path),
        "training_wandb_url": wandb_url,
        "weights": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in weights
        ],
        "artifact_files": artifact_files,
        "training": training_manifest,
    }


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = run("rev-parse", "HEAD")
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError(f"Git returned an invalid commit: {commit!r}")
    status = run("status", "--porcelain", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode()),
        "tracked_diff_sha256": _sha256_bytes(diff.encode()),
        "runner": {
            "path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }


def generate_model_candidates(
    records: Sequence[BasisTaskRecord],
    *,
    checkpoint: Path,
    available_functions: FunctionDefSet,
    samples_per_task: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate ordered samples and execute each under the frozen basis."""

    configure_deterministic_cuda_environment()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Deep-corpus model evaluation requires a CUDA GPU")
    set_seed(seed, deterministic=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    prompt_max_length = model.config.max_position_embeddings - max_new_tokens
    if prompt_max_length <= 0:
        raise ValueError(
            "max_new_tokens must be smaller than the model context window; "
            f"got {max_new_tokens} for {model.config.max_position_embeddings}"
        )
    executor = Executor(available_functions)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for record_index, record in enumerate(records, start=1):
        training_row = training_row_for_record(record, available_functions)
        prompt = training_row["prompt"]
        responses: list[str] = []
        while len(responses) < samples_per_task:
            count = min(batch_size, samples_per_task - len(responses))
            prompts = [prompt] * count
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
            ).to(model.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            generated = outputs[:, input_width:]
            responses.extend(
                tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        for sample_index, response in enumerate(responses):
            rows.append(
                evaluate_candidate(
                    record,
                    sample_index=sample_index,
                    response=response,
                    available_functions=available_functions,
                    executor=executor,
                )
            )
        if progress:
            solved = sum(row["execution_correct"] for row in rows[-samples_per_task:])
            print(
                f"[model] {record_index}/{len(records)} "
                f"d={record.metadata['certified_distance']} "
                f"successful_samples={solved}/{samples_per_task}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return rows, {
        "seconds": elapsed,
        "samples": len(rows),
        "samples_per_second": len(rows) / elapsed if elapsed else None,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _corpus_identity(
    corpus_dir: Path, manifest: Mapping[str, Any], split: str
) -> dict[str, Any]:
    split_metadata = manifest["splits"][split]
    split_path = corpus_dir / split_metadata["path"]
    return {
        "root": str(corpus_dir.resolve()),
        "manifest_path": str((corpus_dir / "manifest.json").resolve()),
        "manifest_file_sha256": _sha256_file(corpus_dir / "manifest.json"),
        "manifest_digest": manifest["manifest_digest"],
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "hub": dict(manifest.get("hub", {})),
        "split": split,
        "split_file": {
            "path": str(split_path.resolve()),
            "bytes": split_path.stat().st_size,
            "sha256": _sha256_file(split_path),
            "records": split_metadata["size"],
        },
    }


def _cohort_identity(records: Sequence[BasisTaskRecord], seed: int) -> dict[str, Any]:
    task_ids = [record.task_id for record in records]
    return {
        "selection": "distance-stratified-within-frozen-root-split",
        "seed": seed,
        "tasks": len(records),
        "task_ids": task_ids,
        "task_ids_digest": _sha256_bytes(_canonical_json(task_ids).encode()),
        "by_certified_distance": {
            str(distance): sum(
                record.metadata["certified_distance"] == distance for record in records
            )
            for distance in range(1, 9)
        },
    }


def _log_wandb_run(
    *,
    config: EvaluationConfig,
    summary: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    bfs_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    import wandb

    run = wandb.init(
        entity=PERSONAL_WANDB_ENTITY,
        project=config.wandb_project,
        name=config.wandb_run_name,
        tags=["deep-corpus-v1", "issue-50", "evaluation", config.split],
        config={
            "schema_version": SCHEMA_VERSION,
            "split": config.split,
            "sample_counts": list(config.sample_counts),
            "seed": config.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "bfs_budget": config.bfs_budget,
            "corpus": summary["corpus"],
            "cohort": summary["cohort"],
            "checkpoint": summary["checkpoint"],
        },
    )
    try:
        url = _validate_wandb_url(str(run.url), project=config.wandb_project)
        summary_with_wandb = dict(summary)
        summary_with_wandb["wandb"] = {
            "entity": PERSONAL_WANDB_ENTITY,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "url": url,
        }
        # Persist the exact document uploaded as an artifact, including its run URL.
        _write_json(output_dir / "summary.json", summary_with_wandb)

        aggregate_rows = []
        matched = summary.get("matched_compute")
        has_bfs = isinstance(matched, dict)
        model = summary["model"]

        def group_row(group: str, samples: int) -> Mapping[str, Any]:
            if has_bfs:
                return matched[group]["matched"][str(samples)]
            if group == "7-8":
                return model["distance_7_8"][str(samples)]
            return model["by_certified_distance"][group][str(samples)]

        for group in [*(str(distance) for distance in range(1, 9)), "7-8"]:
            for samples in config.sample_counts:
                row = group_row(group, samples)
                aggregate_rows.append(
                    [
                        group,
                        samples,
                        row["solve_rate"],
                        row["exact_match_rate"],
                        row["mean_executions_spent"],
                        row.get("bfs_solve_rate_at_model_mean_execution_spend"),
                        row.get("bfs_to_model_total_execution_ratio"),
                    ]
                )
                metrics = {
                    "eval/group": group,
                    "eval/samples": samples,
                    "eval/model_solve_rate": row["solve_rate"],
                    "eval/model_exact_match_rate": row["exact_match_rate"],
                    "eval/model_mean_executions": row["mean_executions_spent"],
                }
                if has_bfs:
                    metrics.update(
                        {
                            "eval/bfs_solve_rate_at_model_mean_executions": row[
                                "bfs_solve_rate_at_model_mean_execution_spend"
                            ]
                        }
                    )
                run.log(metrics)

        max_samples = max(config.sample_counts)
        for distance in range(1, 9):
            row = group_row(str(distance), max_samples)
            metrics = {
                "by_distance/certified_distance": distance,
                "by_distance/model_solve_rate": row["solve_rate"],
                "by_distance/model_exact_match_rate": row["exact_match_rate"],
            }
            if has_bfs:
                metrics.update(
                    {
                        "by_distance/bfs_solve_rate_at_model_cost": row[
                            "bfs_solve_rate_at_model_mean_execution_spend"
                        ]
                    }
                )
            run.log(metrics)
        for samples in config.sample_counts:
            row = group_row("7-8", samples)
            run.log(
                {
                    "solve_at_k/k": samples,
                    "solve_at_k/execution_rate": row["solve_rate"],
                    "solve_at_k/exact_match_rate": row["exact_match_rate"],
                    "solve_at_k/mean_executions": row["mean_executions_spent"],
                }
            )
        for group, prefix in (("8", "distance_8"), ("7-8", "distance_7_8")):
            model_points = (
                matched[group]["model_points"]
                if has_bfs
                else [
                    {
                        "mean_executions_spent": group_row(group, samples)[
                            "mean_executions_spent"
                        ],
                        "solve_rate": group_row(group, samples)["solve_rate"],
                    }
                    for samples in config.sample_counts
                ]
            )
            for point in model_points:
                run.log(
                    {
                        f"{prefix}_cost/mean_executions": point[
                            "mean_executions_spent"
                        ],
                        f"{prefix}_cost/model_solve_rate": point["solve_rate"],
                    }
                )
            if has_bfs:
                for point in matched[group]["bfs_curve"]:
                    run.log(
                        {
                            f"{prefix}_cost/mean_executions": point[
                                "mean_executions_spent"
                            ],
                            f"{prefix}_cost/bfs_solve_rate": point["solve_rate"],
                        }
                    )
        run.log(
            {
                "evaluation_aggregate": wandb.Table(
                    columns=[
                        "group",
                        "samples",
                        "model_solve_rate",
                        "model_exact_match_rate",
                        "model_mean_executions",
                        "bfs_solve_rate_at_model_mean_executions",
                        "bfs_to_model_execution_ratio_at_equal_solve_count",
                    ],
                    data=aggregate_rows,
                ),
                "model_candidates": wandb.Table(
                    columns=[
                        "task_id",
                        "distance",
                        "sample_index",
                        "raw_response",
                        "parse_valid",
                        "program_length",
                        "execution_correct",
                        "exact_match",
                    ],
                    data=[
                        [
                            row["task_id"],
                            row["certified_distance"],
                            row["sample_index"],
                            row["raw_response"],
                            row["parse_valid"],
                            row["program_length"],
                            row["execution_correct"],
                            row["exact_match"],
                        ]
                        for row in candidate_rows
                    ],
                ),
                "bfs_tasks": wandb.Table(
                    columns=[
                        "task_id",
                        "distance",
                        "success",
                        "executions",
                        "seconds",
                    ],
                    data=[
                        [
                            row["task_id"],
                            row["certified_distance"],
                            row["success"],
                            row["executions"],
                            row["seconds"],
                        ]
                        for row in bfs_rows
                    ],
                ),
            }
        )
        primary = summary.get("primary", {})
        run.summary.update(
            {
                "summary_path": str((output_dir / "summary.json").resolve()),
                "distance_7_8_solve_at_64": primary.get("solve_rate"),
                "distance_7_8_exact_match_at_64": primary.get("exact_match_rate"),
                "distance_7_8_mean_executions_at_64": primary.get(
                    "mean_executions_spent"
                ),
                "distance_7_8_bfs_to_model_execution_ratio": primary.get(
                    "bfs_to_model_total_execution_ratio"
                ),
                "kill_criterion_triggered": summary.get("verdict", {}).get(
                    "kill_criterion_triggered"
                ),
            }
        )
        artifact = wandb.Artifact(
            name=f"deep-corpus-{config.split}-evaluation-{run.id}",
            type="evaluation",
            metadata={
                "corpus_manifest_digest": summary["corpus"]["manifest_digest"],
                "basis_set_digest": summary["corpus"]["basis_set_digest"],
                "cohort_task_ids_digest": summary["cohort"]["task_ids_digest"],
            },
        )
        for name in (
            "summary.json",
            "model_candidates.jsonl.gz",
            "bfs_results.jsonl.gz",
        ):
            path = output_dir / name
            if path.is_file():
                artifact.add_file(str(path), name=name)
        run.log_artifact(artifact)
        return url
    finally:
        run.finish()


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Verify provenance, evaluate, aggregate, persist, and log one run."""

    _validate_config(config)
    int_max_str_digits = configure_unbounded_int_serialization()
    test_access = (
        validate_test_selection_lock(
            config.selection_lock,
            checkpoint=config.checkpoint,
        )
        if config.split == "test" and config.selection_lock is not None
        else None
    )
    if config.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a fresh path: {config.output_dir}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    source_code = _git_identity(repo_root)
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
    checkpoint_identity = _checkpoint_identity(
        config.checkpoint,
        manifest["manifest_digest"],
        basis,
        project=config.wandb_project,
    )
    if test_access is not None:
        validate_locked_test_protocol(
            test_access,
            config=config,
            corpus_identity=corpus_identity,
            cohort_identity=cohort_identity,
        )
    available_functions = basis.as_function_set()

    if config.dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "dry-run",
            "corpus": corpus_identity,
            "cohort": cohort_identity,
            "checkpoint": checkpoint_identity,
            "source_code": source_code,
            "verification": verification,
            "test_access": test_access,
            "runtime_safety": {"int_max_str_digits": int_max_str_digits},
            "preview": [
                training_row_for_record(record, available_functions)
                for record in cohort[:8]
            ],
        }

    config.output_dir.mkdir(parents=True)
    if test_access is not None and config.selection_lock is not None:
        test_access["test_scoring_started_at_utc"] = datetime.now(UTC).isoformat()
        test_access["consumption_receipt"] = consume_test_selection_lock(
            config.selection_lock,
            test_access=test_access,
            output_dir=config.output_dir,
        )
    _write_json(config.output_dir / "cohort.json", cohort_identity)
    candidate_rows, generation_runtime = generate_model_candidates(
        cohort,
        checkpoint=config.checkpoint,
        available_functions=available_functions,
        samples_per_task=max(config.sample_counts),
        batch_size=config.generation_batch_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
    )
    write_jsonl_gz(candidate_rows, config.output_dir / "model_candidates.jsonl.gz")
    model_summary, per_task = aggregate_model_candidates(
        candidate_rows,
        expected_records=cohort,
        sample_counts=config.sample_counts,
    )

    bfs_rows: list[dict[str, Any]] = []
    bfs_summary: dict[str, Any] | None = None
    matched: dict[str, Any] | None = None
    if config.run_bfs:
        bfs_rows = evaluate_bfs_records(
            cohort,
            available_functions=available_functions,
            budget=config.bfs_budget,
            max_depth=8,
            progress=True,
        )
        write_jsonl_gz(bfs_rows, config.output_dir / "bfs_results.jsonl.gz")
        bfs_summary = validate_bfs_rows(bfs_rows, cohort)
        matched = matched_compute_comparison(
            per_task=per_task,
            bfs_rows=bfs_rows,
            records=cohort,
            sample_counts=config.sample_counts,
        )

    primary = (
        matched["7-8"]["matched"][str(max(config.sample_counts))]
        if matched is not None
        else model_summary["distance_7_8"][str(max(config.sample_counts))]
    )
    cheaper = primary.get("model_cheaper_than_bfs_at_equal_or_higher_solve_count")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "corpus": corpus_identity,
        "cohort": cohort_identity,
        "checkpoint": checkpoint_identity,
        "source_code": source_code,
        "verification": verification,
        "test_access": test_access,
        "runtime_safety": {"int_max_str_digits": int_max_str_digits},
        "generation": {
            **_generation_protocol_from_config(config),
            "runtime": generation_runtime,
        },
        "bfs_configuration": (
            {
                "budget": config.bfs_budget,
                "max_depth": 8,
                "cost_definition": (
                    "One execution for every type-compatible basis function tried, "
                    "including transitions whose function raises."
                ),
            }
            if config.run_bfs
            else None
        ),
        "model": model_summary,
        "bfs": bfs_summary,
        "matched_compute": matched,
        "primary": primary,
        "verdict": {
            "scope": f"{config.split} certified distance 7-8 at solve@{max(config.sample_counts)}",
            "model_cheaper_than_bfs_at_equal_solve_rate": cheaper,
            "kill_criterion_triggered": (
                (not bool(primary["solved"])) or cheaper is False
                if config.split == "test" and matched is not None
                else None
            ),
            "note": (
                "The kill criterion is evaluated only on the selected test cohort "
                "with a complete matched-task BFS result."
            ),
        },
    }
    _write_json(config.output_dir / "summary.json", summary)
    wandb_url = _log_wandb_run(
        config=config,
        summary=summary,
        candidate_rows=candidate_rows,
        bfs_rows=bfs_rows,
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


def _parse_sample_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("sample counts must be integers") from error
    if not counts:
        raise argparse.ArgumentTypeError("sample counts must not be empty")
    return counts


def _parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--selection-lock",
        type=Path,
        help="Validation champion lock required before --split test scoring",
    )
    parser.add_argument("--tasks-per-distance", type=int, default=12)
    parser.add_argument(
        "--sample-counts", type=_parse_sample_counts, default=DEFAULT_SAMPLE_COUNTS
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--bfs-budget", type=int, default=20_000_000)
    parser.add_argument("--skip-bfs", action="store_true")
    parser.add_argument("--wandb-entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    return EvaluationConfig(
        checkpoint=arguments.checkpoint,
        output_dir=arguments.output_dir,
        corpus_dir=arguments.corpus_dir,
        split=arguments.split,
        selection_lock=arguments.selection_lock,
        tasks_per_distance=arguments.tasks_per_distance,
        sample_counts=arguments.sample_counts,
        seed=arguments.seed,
        generation_batch_size=arguments.generation_batch_size,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        bfs_budget=arguments.bfs_budget,
        run_bfs=not arguments.skip_bfs,
        wandb_entity=arguments.wandb_entity,
        wandb_project=arguments.wandb_project,
        wandb_run_name=arguments.wandb_run_name,
        dry_run=arguments.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_args(argv)
    summary = run_evaluation(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
