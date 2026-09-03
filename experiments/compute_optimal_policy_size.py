"""Freeze, run, and aggregate the compute-optimal policy-size experiment.

The protocol is validation-only. One timed, uninstrumented search produces each
task's wall-clock and execution cost. All 16 timing cells must complete and seal
before any untimed observer replay or oracle annotation can begin. Each replay
must reproduce the exact result before its scored visits are retained. Successful
task costs therefore define the complete solve-rate-vs-log-budget curves without
rerunning the search at a grid of artificial budgets.

Usage::

    python experiments/compute_optimal_policy_size.py freeze \
        --output checkpoints/compute-optimal-policy-size/protocol.json
    python experiments/compute_optimal_policy_size.py time \
        --protocol checkpoints/compute-optimal-policy-size/protocol.json \
        --checkpoint CHECKPOINT --arm 90k \
        --seed 0 --device cpu --output-dir OUTPUT
    python experiments/compute_optimal_policy_size.py seal-timings \
        --protocol checkpoints/compute-optimal-policy-size/protocol.json \
        --timing OUTPUT/timing_summary.json ... \
        --output timing_matrix.json
    python experiments/compute_optimal_policy_size.py annotate \
        --protocol checkpoints/compute-optimal-policy-size/protocol.json \
        --checkpoint CHECKPOINT --arm 90k \
        --seed 0 --device cpu --timing-summary OUTPUT/timing_summary.json \
        --timing-matrix timing_matrix.json --output-dir ANNOTATED
    python experiments/compute_optimal_policy_size.py aggregate \
        --protocol checkpoints/compute-optimal-policy-size/protocol.json \
        --evaluation OUTPUT/summary.json ... \
        --output aggregate.json
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wandering_light.evals import deep_corpus_protocol as _artifact_primitives
from wandering_light.shortest_path_data import (
    read_jsonl_gz as _read_jsonl_gz,
)
from wandering_light.shortest_path_data import (
    write_jsonl_gz as _write_jsonl_gz,
)

_canonical_copy = _artifact_primitives.canonical_copy
_canonical_json = _artifact_primitives.canonical_json
_file_identity = _artifact_primitives.file_identity
_sha256_bytes = _artifact_primitives.sha256_bytes
_sha256_file = _artifact_primitives.sha256_file
_sha256_json = _artifact_primitives.sha256_json

if TYPE_CHECKING:
    from wandering_light.function_def import FunctionDefSet


SCHEMA_VERSION = 1
STUDY_ID = "compute-optimal-policy-size-v1"
PROTOCOL_STATUS = "frozen-validation-only"
EVALUATION_STATUS = "complete"
TIMING_STATUS = "timing-complete"
TIMING_MATRIX_STATUS = "timing-matrix-sealed"
AGGREGATE_STATUS = "complete"

DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v2")
DEFAULT_REFERENCE_CORPUS_DIRS = (Path("wandering_light/training/data/deep_corpus_v1"),)
TRAINING_SEEDS = (0, 1)
DEVICES = ("cpu", "cuda")
COHORT_SELECTION_SEED = 20_260_826
EVALUATION_TASK_ORDER_SEED = 20_260_829
EVALUATION_CELL_ORDER_SEED = 20_260_831
TASKS_PER_DISTANCE = 12
CERTIFIED_DISTANCES = tuple(range(1, 9))
BEAM_WIDTH = 64
MAX_DEPTH = 8
EXECUTION_BUDGET = 100_000

# The range is frozen before any policy is evaluated. AUC is the exact integral
# over log2(seconds), not a sampled grid.
LOG2_WALL_SECONDS_MIN = -7.0
LOG2_WALL_SECONDS_MAX = 4.0
WALL_SECONDS_MIN = 2**LOG2_WALL_SECONDS_MIN
WALL_SECONDS_MAX = 2**LOG2_WALL_SECONDS_MAX
EXECUTIONS_MIN = 1

ARM_IDS = ("90k", "1m", "10m")
EVALUATION_CELL_COUNT = len(ARM_IDS) * len(TRAINING_SEEDS) * len(DEVICES)
ARM_LABELS = {"90k": "90K", "1m": "1M", "10m": "10M"}
EXPECTED_PARAMETER_COUNTS = {
    "90k": 90_374,
    "1m": 724_022,
    "10m": 6_504_310,
}
FROZEN_ACTION_COUNT = 118

TRAINING_EPOCHS = 12
TRAINING_DEVICE = "cuda"
EFFECTIVE_TRAIN_BATCH_SIZE = 256
CHECKPOINT_SELECTION = "minimum validation cross-entropy"

ORACLE_MAX_DEPTH = 6
ORACLE_MAX_STATES = 20_000
ORACLE_MAX_EXECUTIONS = 100_000
OPTIMAL_BITS_EPSILON = 0.01
VISITS_PER_TASK_DEPTH = 2
MIN_H_CERTIFIED_COVERAGE = 0.8
VISIT_SAMPLING_SEED = 20_260_830

TASK_ROWS_FILE = "task_results.jsonl.gz"
VISIT_ROWS_FILE = "scored_visits.jsonl.gz"
SUMMARY_FILE = "summary.json"
TIMED_ROWS_FILE = "timed_task_results.jsonl.gz"
TIMING_SUMMARY_FILE = "timing_summary.json"
PROTOCOL_DIGEST_FIELD = "protocol_sha256"
EVALUATION_DIGEST_FIELD = "evaluation_sha256"
TIMING_DIGEST_FIELD = "timing_sha256"
TIMING_MATRIX_DIGEST_FIELD = "timing_matrix_sha256"
AGGREGATE_DIGEST_FIELD = "aggregate_sha256"
ORACLE_CACHE_DIGEST_FIELD = "oracle_cache_sha256"
ORACLE_CACHE_FILE = "oracle_labels.json"
ORACLE_CACHE_SAVE_EVERY_TASKS = 12
EVALUATION_LOCK_FILE = ".compute-optimal-policy-size.lock"

_SELECTIVE_CORPUS_PASSES = (
    "manifest_identity",
    "witness_replay",
    "record_provenance",
    "selected_split_summaries",
    "selected_split_isolation",
    "reference_overlap",
)
_SELECTIVE_CORPUS_SKIPS = (
    "corpus_global_summaries",
    "target_contract",
    "distance_recertification",
)
_SELECTIVE_CORPUS_SKIP_REASONS = {
    "corpus_global_summaries": "not every manifest split was loaded",
    "target_contract": "target contract is corpus-global",
    "distance_recertification": "recertify=0",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_SOURCE_FILES = tuple(
    path.relative_to(_REPO_ROOT).as_posix()
    for source_root in ("experiments", "wandering_light")
    for path in sorted((_REPO_ROOT / source_root).rglob("*.py"))
)


@dataclass(frozen=True)
class _PolicySizeArmConfig:
    arm_id: str
    nominal_label: str
    byte_embedding: int
    conv_channels: int
    hidden: int
    parameters: int
    learning_rate: float
    train_batch_size: int
    gradient_accumulation_steps: int
    validation_batch_size: int

    def protocol_identity(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "nominal_label": self.nominal_label,
            "architecture": {
                "byte_embedding": self.byte_embedding,
                "conv_channels": self.conv_channels,
                "hidden": self.hidden,
            },
            "parameters": self.parameters,
        }

    def batching(self) -> dict[str, int]:
        return {
            "train_batch_size": self.train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "validation_batch_size": self.validation_batch_size,
        }


@functools.cache
def _frozen_arm_configs(
    action_count: int = FROZEN_ACTION_COUNT,
) -> tuple[_PolicySizeArmConfig, ...]:
    """Resolve immutable arm identities and batching from the trainer."""

    from experiments.train_rollout_policy import (
        POLICY_SIZE_ARMS,
        POLICY_SIZE_BATCHING,
        POLICY_SIZE_LEARNING_RATES,
    )

    # The trainer may define more arms than the sweep evaluates (40m was dropped
    # after the tuned 10m arm overfit at epoch 7); every sweep arm must exist there
    # in the same relative order.
    for label, table in (
        ("arms", POLICY_SIZE_ARMS),
        ("batching", POLICY_SIZE_BATCHING),
        ("learning rates", POLICY_SIZE_LEARNING_RATES),
    ):
        if tuple(arm_id for arm_id in table if arm_id in ARM_IDS) != ARM_IDS:
            raise ValueError(
                f"Trainer policy-size {label} do not cover the frozen sweep arms"
            )
    arms: list[_PolicySizeArmConfig] = []
    for arm_id in ARM_IDS:
        architecture = POLICY_SIZE_ARMS[arm_id]
        parameters = architecture.parameter_count(action_count)
        expected = EXPECTED_PARAMETER_COUNTS[arm_id]
        if action_count != FROZEN_ACTION_COUNT or parameters != expected:
            raise ValueError(
                f"Arm {arm_id} has {parameters:,} parameters for {action_count} "
                f"actions, expected {expected:,} for {FROZEN_ACTION_COUNT} actions"
            )
        batching = POLICY_SIZE_BATCHING[arm_id]
        if (
            not isinstance(batching, tuple)
            or len(batching) != 3
            or any(type(value) is not int or value <= 0 for value in batching)
        ):
            raise ValueError(f"Trainer policy-size batching is invalid for {arm_id}")
        train_batch_size, accumulation_steps, validation_batch_size = batching
        learning_rate = POLICY_SIZE_LEARNING_RATES[arm_id]
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, int | float)
            or not math.isfinite(float(learning_rate))
            or learning_rate <= 0
        ):
            raise ValueError(
                f"Trainer policy-size learning rate is invalid for {arm_id}"
            )
        if train_batch_size * accumulation_steps != EFFECTIVE_TRAIN_BATCH_SIZE:
            raise ValueError(
                f"Trainer effective batch size differs from the sweep for {arm_id}"
            )
        arms.append(
            _PolicySizeArmConfig(
                arm_id=arm_id,
                nominal_label=ARM_LABELS[arm_id],
                byte_embedding=architecture.byte_embedding,
                conv_channels=architecture.conv_channels,
                hidden=architecture.hidden,
                parameters=parameters,
                learning_rate=float(learning_rate),
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=accumulation_steps,
                validation_batch_size=validation_batch_size,
            )
        )
    return tuple(arms)


def _frozen_arms(action_count: int = FROZEN_ACTION_COUNT) -> list[dict[str, Any]]:
    """Materialize fresh JSON identities from the cached immutable arm configs."""

    return [arm.protocol_identity() for arm in _frozen_arm_configs(action_count)]


@functools.cache
def _arm_config_by_id(
    arm_id: str, action_count: int = FROZEN_ACTION_COUNT
) -> _PolicySizeArmConfig:
    try:
        return next(
            arm for arm in _frozen_arm_configs(action_count) if arm.arm_id == arm_id
        )
    except StopIteration as error:
        raise ValueError(f"Unknown architecture arm {arm_id!r}") from error


def _arm_by_id(arm_id: str) -> dict[str, Any]:
    return _arm_config_by_id(arm_id).protocol_identity()


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: Path
    checkpoint: Path
    arm_id: str
    training_seed: int
    device: str
    output_dir: Path
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    oracle_cache: Path | None = None
    timing_summary: Path | None = None
    timing_matrix: Path | None = None
    progress: bool = True


@contextlib.contextmanager
def _exclusive_protocol_lock(protocol: Path):
    """Reject overlapping cells and serialize the shared oracle cache."""

    if not protocol.is_file():
        raise FileNotFoundError(f"Frozen protocol is missing: {protocol}")
    lock_path = protocol.with_name(EVALUATION_LOCK_FILE)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another compute-optimal evaluation holds {lock_path}"
            ) from error
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _serialized_evaluation(function: Callable[..., dict[str, Any]]):
    @functools.wraps(function)
    def wrapped(config: EvaluationConfig, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with _exclusive_protocol_lock(config.protocol):
            return function(config, *args, **kwargs)

    return wrapped


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result = _canonical_copy(result)
    result[field] = _sha256_json(result)
    return result


def _verify_seal(value: Mapping[str, Any], field: str, *, label: str) -> dict[str, Any]:
    result = _canonical_copy(dict(value))
    actual = result.pop(field, None)
    expected = _sha256_json(result)
    if actual != expected:
        raise ValueError(f"{label} digest mismatch: {actual!r} != {expected!r}")
    result[field] = actual
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _write_json(
    path: Path, value: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
        )
        try:
            staging = staging_dir / path.name
            _write_json(staging, value)
            os.link(staging, path)
        finally:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
        return
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, allow_nan=False, indent=2, sort_keys=True)
        output_file.write("\n")


def _validated_jsonl_rows(rows: Sequence[Any], *, label: str) -> list[dict[str, Any]]:
    """Require the public JSONL reader's values to be artifact row objects."""

    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} contains a non-object row")
    return [dict(row) for row in rows]


def _artifact_identity(path: Path) -> dict[str, Any]:
    return _file_identity(path, root=path.parent)


def study_source_identity() -> dict[str, Any]:
    """Pin the code that defines training, search, labels, and aggregation."""

    repo_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Could not identify the compute-optimal source commit"
        ) from error
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"Git returned an invalid HEAD commit: {commit!r}")
    files = [
        _file_identity(repo_root / relative, root=repo_root)
        for relative in STUDY_SOURCE_FILES
    ]
    return {
        "head_commit": commit,
        "files": files,
        "files_sha256": _sha256_json(files),
    }


def require_clean_study_checkout() -> None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not verify a clean study checkout") from error
    if completed.stdout:
        raise RuntimeError(
            "Freeze requires a clean checkout so all checkpoints can match its HEAD"
        )


def evaluation_host() -> dict[str, Any]:
    """Fingerprint every host property used to compare wall-clock results."""

    import torch

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                }
            )
    return {
        "node": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
    }


def _timing_environment() -> dict[str, Any]:
    load_average = None
    if hasattr(os, "getloadavg"):
        load_average = [float(value) for value in os.getloadavg()]
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "load_average_1m_5m_15m": load_average,
        "cpu_affinity": affinity,
    }


def _cell_ordinal(protocol: Mapping[str, Any], cell: Mapping[str, Any]) -> int:
    requested = {
        "arm_id": cell["arm_id"],
        "training_seed": cell["training_seed"],
        "device": cell["device"],
    }
    matches = [
        int(row["ordinal"])
        for row in protocol["evaluation_cell_order"]["cells"]
        if {key: row[key] for key in requested} == requested
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cell is absent or duplicated in the frozen order: {requested}"
        )
    return matches[0]


def _require_selective_corpus_verification(
    verification: Mapping[str, Any],
    *,
    manifest_digest: Any,
    selected_splits: Sequence[str],
    manifest_splits: Sequence[str],
    required_references: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if any(not isinstance(reference, Mapping) for reference in required_references):
        raise ValueError(f"{label} has an invalid required-reference contract")
    selected = list(selected_splits)
    expected_references = _canonical_copy(
        [dict(reference) for reference in required_references]
    )
    expected_scope = {
        "kind": "selected-splits",
        "selected_splits": selected,
        "manifest_splits": sorted(manifest_splits),
    }
    expected_statuses = {
        **dict.fromkeys(_SELECTIVE_CORPUS_PASSES, "passed"),
        **dict.fromkeys(_SELECTIVE_CORPUS_SKIPS, "skipped"),
    }
    checks = verification.get("checks")
    actual_statuses = (
        {
            str(check): evidence.get("status")
            if isinstance(evidence, Mapping)
            else None
            for check, evidence in checks.items()
        }
        if isinstance(checks, Mapping)
        else {}
    )
    manifest_identity = (
        checks.get("manifest_identity") if isinstance(checks, Mapping) else None
    )
    reference_overlap = (
        checks.get("reference_overlap") if isinstance(checks, Mapping) else None
    )
    evidence_consistent = (
        isinstance(manifest_identity, Mapping)
        and manifest_identity.get("manifest_digest")
        == verification.get("manifest_digest")
        and manifest_identity.get("failures", []) == []
        and isinstance(reference_overlap, Mapping)
        and reference_overlap.get("required_references") == expected_references
        and reference_overlap.get("verified_references") == expected_references
        and all(
            isinstance(evidence := checks.get(check), Mapping)
            and evidence.get("failures") == []
            for check in _SELECTIVE_CORPUS_PASSES
            if check != "manifest_identity"
        )
        and all(
            isinstance(evidence := checks.get(check), Mapping)
            and evidence.get("reason") == reason
            and evidence.get("failures", []) == []
            for check, reason in _SELECTIVE_CORPUS_SKIP_REASONS.items()
        )
    )
    if (
        verification.get("manifest_digest") != manifest_digest
        or verification.get("scope") != expected_scope
        or actual_statuses != expected_statuses
        or not evidence_consistent
        or verification.get("ok") is not True
        or verification.get("verified_splits") != selected
        or verification.get("full_corpus_verified") is not None
        or verification.get("target_contract_failures") is not None
    ):
        raise ValueError(f"{label} lacks exact selected-split verification evidence")


@functools.cache
def _load_validation_corpus_cached(
    corpus_dir: Path,
) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    """Parse and verify one resolved validation corpus path exactly once."""

    from experiments.generate_deep_corpus import (
        load_corpus,
        load_required_reference_identities,
        required_reference_corpora,
        verify_loaded_corpus,
    )
    from wandering_light.basis_set import load_basis_set
    from wandering_light.corpus_hub import corpus_split_files, fetch_corpus

    manifest_path = corpus_dir / "manifest.json"
    manifest_header = _read_json(manifest_path, label="deep-corpus-v2 manifest")
    [(split, validation_name)] = corpus_split_files(
        manifest_header,
        splits=("validation",),
    )
    if split != "validation":
        raise ValueError("Manifest selected an unexpected validation split")
    validation_path = corpus_dir / validation_name
    if not validation_path.is_file():
        fetch_corpus(
            manifest_path,
            destination=corpus_dir,
            splits=("validation",),
            include_details=False,
        )
    manifest, records = load_corpus(corpus_dir, splits=("validation",))
    required_references = required_reference_corpora(manifest)
    reference_identities = load_required_reference_identities(
        manifest,
        DEFAULT_REFERENCE_CORPUS_DIRS,
    )
    verification = verify_loaded_corpus(
        manifest,
        records,
        splits=("validation",),
        recertify=0,
        reference_identities=reference_identities,
    )
    _require_selective_corpus_verification(
        verification,
        manifest_digest=manifest.get("manifest_digest"),
        selected_splits=("validation",),
        manifest_splits=tuple(manifest.get("splits", ())),
        required_references=required_references,
        label="Validation corpus",
    )
    basis = load_basis_set(manifest["basis_set_id"])
    if basis.digest != manifest["basis_set_digest"]:
        raise ValueError("Loaded basis differs from the corpus manifest")
    return _canonical_copy(manifest), basis, tuple(records)


def _load_validation_corpus(
    corpus_dir: Path,
) -> tuple[dict[str, Any], FunctionDefSet, list[Any]]:
    """Return cached verified data with a fresh executable palette per caller."""

    manifest, basis, records = _load_validation_corpus_cached(corpus_dir.resolve())
    return _canonical_copy(manifest), basis.as_function_set(), list(records)


def _validate_cohort_records(
    records: Sequence[Any],
    cohort: Mapping[str, Any],
    task_order: Mapping[str, Any],
) -> list[Any]:
    from wandering_light.evals.deep_corpus_protocol import validate_frozen_cohort

    validated = validate_frozen_cohort(cohort, records=records)
    if validated.get("split") != "validation":
        raise ValueError("Compute-optimal evaluation is validation-only")
    by_id = {record.task_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Validation corpus contains duplicate task IDs")
    expected_order = _evaluation_task_order(validated)
    if task_order != expected_order:
        raise ValueError("Evaluation task order differs from the frozen protocol")
    try:
        selected = [by_id[task_id] for task_id in expected_order["task_ids"]]
    except KeyError as error:
        raise ValueError(
            f"Frozen validation task is missing: {error.args[0]}"
        ) from error
    return selected


def _evaluation_task_order(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Decorrelate certified distance from a frozen timing order."""

    task_ids = cohort.get("task_ids")
    if not isinstance(task_ids, list) or any(
        not isinstance(task_id, str) for task_id in task_ids
    ):
        raise ValueError("Frozen cohort has no task IDs for evaluation ordering")
    ordered = sorted(
        task_ids,
        key=lambda task_id: (
            _sha256_bytes(
                _canonical_json(
                    {
                        "seed": EVALUATION_TASK_ORDER_SEED,
                        "task_id": task_id,
                    }
                ).encode("utf-8")
            ),
            task_id,
        ),
    )
    return {
        "method": "sha256-rank",
        "seed": EVALUATION_TASK_ORDER_SEED,
        "task_ids": ordered,
        "task_ids_sha256": _sha256_bytes(_canonical_json(ordered).encode("utf-8")),
    }


def _evaluation_cell_order() -> dict[str, Any]:
    """Pair devices, hash-randomize model/seed pairs, and alternate device first."""

    pairs = [
        {"arm_id": arm_id, "training_seed": seed}
        for arm_id in ARM_IDS
        for seed in TRAINING_SEEDS
    ]
    pairs.sort(
        key=lambda cell: (
            _sha256_bytes(
                _canonical_json(
                    {
                        "seed": EVALUATION_CELL_ORDER_SEED,
                        **cell,
                    }
                ).encode("utf-8")
            ),
            cell["arm_id"],
            cell["training_seed"],
        )
    )
    cells = []
    for pair_index, pair in enumerate(pairs):
        device_order = DEVICES if pair_index % 2 == 0 else tuple(reversed(DEVICES))
        for device in device_order:
            cells.append(
                {
                    "ordinal": len(cells),
                    **pair,
                    "device": device,
                }
            )
    return {
        "method": "sha256-ranked-arm-seed-pairs; adjacent-devices; alternating-first",
        "seed": EVALUATION_CELL_ORDER_SEED,
        "cells": cells,
        "cells_sha256": _sha256_bytes(_canonical_json(cells).encode("utf-8")),
    }


def _protocol_body(
    *,
    manifest: Mapping[str, Any],
    cohort: Mapping[str, Any],
    host: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.generate_deep_corpus import required_reference_corpora

    validation = manifest.get("splits", {}).get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Deep-corpus-v2 manifest has no validation split")
    required_references = required_reference_corpora(manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": PROTOCOL_STATUS,
        "discipline": {
            "evaluation_split": "validation",
            "held_out_model_scoring": False,
            "cohort_shared_across_all_cells": True,
            "paired_training_seeds": True,
            "evaluation_cells_sequential": True,
            "global_phase_barrier": [
                f"all-{EVALUATION_CELL_COUNT}-cell-timing",
                "sealed-timing-matrix",
                "observer-replay-and-oracle-annotation",
            ],
            "evaluation_lock": {
                "file": EVALUATION_LOCK_FILE,
                "mode": "exclusive-nonblocking-whole-cell",
            },
        },
        "corpus": {
            "name": "deep-corpus-v2",
            "manifest_digest": manifest.get("manifest_digest"),
            "basis_set_id": manifest.get("basis_set_id"),
            "basis_set_digest": manifest.get("basis_set_digest"),
            "reference_corpora_excluded": _canonical_copy(required_references),
            "action_count": FROZEN_ACTION_COUNT,
            "validation": {
                key: validation.get(key)
                for key in (
                    "path",
                    "sha256",
                    "size",
                    "roots",
                    "root_digests",
                    "root_digest_set_sha256",
                    "task_id_set_sha256",
                    "by_certified_distance",
                )
                if key in validation
            },
        },
        "cohort": dict(cohort),
        "evaluation_task_order": _evaluation_task_order(cohort),
        "evaluation_cell_order": _evaluation_cell_order(),
        "arms": _frozen_arms(),
        "training_seeds": list(TRAINING_SEEDS),
        "devices": list(DEVICES),
        "training": {
            "source_corpus": "deep-corpus-v2",
            "materialized_splits": ["discovery", "validation"],
            "held_out_prefix_examples": 0,
            "epochs": TRAINING_EPOCHS,
            "per_arm_learning_rates": {
                arm.arm_id: arm.learning_rate for arm in _frozen_arm_configs()
            },
            "device": TRAINING_DEVICE,
            "effective_train_batch_size": EFFECTIVE_TRAIN_BATCH_SIZE,
            "per_arm_batching": {
                arm.arm_id: arm.batching() for arm in _frozen_arm_configs()
            },
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "weights_only": True,
        },
        "search": {
            "beam_width": BEAM_WIDTH,
            "max_depth": MAX_DEPTH,
            "execution_budget": EXECUTION_BUDGET,
            "cost_unit": "basis_function_executions",
            "cost_definition": (
                "one per type-compatible basis function attempted, including "
                "an attempt whose function raises"
            ),
        },
        "metrics": {
            "primary": "wall_clock_seconds",
            "wall_clock": {
                "unit": "wall_clock_seconds",
                "log2_min": LOG2_WALL_SECONDS_MIN,
                "log2_max": LOG2_WALL_SECONDS_MAX,
                "min_seconds": WALL_SECONDS_MIN,
                "max_seconds": WALL_SECONDS_MAX,
            },
            "executions": {
                "unit": "basis_function_executions",
                "min_executions": EXECUTIONS_MIN,
                "max_executions": EXECUTION_BUDGET,
            },
            "auc": (
                "exact integral of the empirical solve-rate step function over "
                "log2 budget; misses contribute zero area"
            ),
            "in_situ_bits": {
                "epsilon": OPTIMAL_BITS_EPSILON,
                "definition": (
                    "-log2 of epsilon-smoothed policy mass on the certified set "
                    "of all shortest first actions after the production type mask"
                ),
                "population": "states scored by the untimed production-search replay",
                "sample": {
                    "method": "lowest sha256 rank within task and rollout depth",
                    "seed": VISIT_SAMPLING_SEED,
                    "maximum_per_task_depth": VISITS_PER_TASK_DEPTH,
                },
                "oracle": {
                    "max_depth": ORACLE_MAX_DEPTH,
                    "max_states": ORACLE_MAX_STATES,
                    "max_executions": ORACLE_MAX_EXECUTIONS,
                    "state_identity": "TypedList.search_key",
                    "target_identity": "TypedList.canonical_key",
                },
                "incomplete_label_policy": "exclude and report explicit censoring",
                "headline_min_certified_coverage": MIN_H_CERTIFIED_COVERAGE,
                "headline_coverage_gate": (
                    "each weighting uses its own inverse-probability coverage"
                ),
                "cache": {
                    "scope": "shared across all sequential evaluation cells",
                    "default_path": ORACLE_CACHE_FILE,
                    "save_every_tasks": ORACLE_CACHE_SAVE_EVERY_TASKS,
                },
                "weightings": {
                    "state-weighted": (
                        "inverse-sampling-probability weighting over scored states"
                    ),
                    "task-balanced": (
                        "state-weighted mean within task, then equal weight per task"
                    ),
                },
            },
        },
        "evaluation_host": dict(host),
        "source": dict(source),
    }


def validate_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    from wandering_light.evals.deep_corpus_protocol import validate_frozen_cohort

    protocol = _verify_seal(value, PROTOCOL_DIGEST_FIELD, label="protocol")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("study_id") != STUDY_ID
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Unsupported compute-optimal protocol")
    if protocol.get("arms") != _frozen_arms():
        raise ValueError("Protocol architecture arms differ from the frozen sweep")
    if tuple(protocol.get("training_seeds", ())) != TRAINING_SEEDS:
        raise ValueError("Protocol training seeds differ from the paired sweep")
    if tuple(protocol.get("devices", ())) != DEVICES:
        raise ValueError("Protocol must evaluate both CPU and CUDA")
    if protocol.get("evaluation_cell_order") != _evaluation_cell_order():
        raise ValueError("Protocol cell order differs from the frozen sweep")
    expected_search = {
        "beam_width": BEAM_WIDTH,
        "max_depth": MAX_DEPTH,
        "execution_budget": EXECUTION_BUDGET,
        "cost_unit": "basis_function_executions",
        "cost_definition": (
            "one per type-compatible basis function attempted, including "
            "an attempt whose function raises"
        ),
    }
    if protocol.get("search") != expected_search:
        raise ValueError("Protocol search settings differ from the frozen sweep")
    expected_training = _protocol_body(
        manifest={"splits": {"validation": {}}},
        cohort={"task_ids": []},
        host={},
        source={},
    )["training"]
    if protocol.get("training") != expected_training:
        raise ValueError("Protocol training settings differ from the frozen sweep")
    expected_metrics = _protocol_body(
        manifest={"splits": {"validation": {}}},
        cohort={"task_ids": []},
        host={},
        source={},
    )["metrics"]
    if protocol.get("metrics") != expected_metrics:
        raise ValueError("Protocol metric range differs from the frozen sweep")
    discipline = protocol.get("discipline", {})
    if (
        discipline.get("evaluation_split") != "validation"
        or discipline.get("held_out_model_scoring") is not False
    ):
        raise ValueError("Protocol is not validation-only")
    cohort = validate_frozen_cohort(protocol.get("cohort", {}))
    if (
        cohort["split"] != "validation"
        or cohort["size"] != TASKS_PER_DISTANCE * len(CERTIFIED_DISTANCES)
        or any(
            cohort["distance_histogram"].get(str(distance)) != TASKS_PER_DISTANCE
            for distance in CERTIFIED_DISTANCES
        )
    ):
        raise ValueError(
            "Protocol cohort is not the common 12-per-distance validation set"
        )
    if protocol.get("evaluation_task_order") != _evaluation_task_order(cohort):
        raise ValueError("Protocol task order differs from the frozen sweep")
    if (
        not isinstance(protocol.get("evaluation_host"), Mapping)
        or not protocol["evaluation_host"]
    ):
        raise ValueError("Protocol has no evaluation-host fingerprint")
    if protocol.get("corpus", {}).get("action_count") != FROZEN_ACTION_COUNT:
        raise ValueError("Protocol basis action count differs from the frozen sweep")
    source = protocol.get("source")
    current_source = study_source_identity()
    if (
        not isinstance(source, Mapping)
        or source.get("files") != current_source["files"]
        or source.get("files_sha256") != current_source["files_sha256"]
        or not isinstance(source.get("head_commit"), str)
    ):
        raise ValueError("Protocol source differs from this evaluator runtime")
    return protocol


def load_protocol(path: Path) -> dict[str, Any]:
    return validate_protocol(_read_json(path, label="protocol"))


def _load_oracle_cache(
    path: Path,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = _verify_seal(
        _read_json(path, label="oracle cache"),
        ORACLE_CACHE_DIGEST_FIELD,
        label="oracle cache",
    )
    expected_oracle = protocol["metrics"]["in_situ_bits"]["oracle"]
    if (
        cache.get("schema_version") != SCHEMA_VERSION
        or cache.get("study_id") != STUDY_ID
        or cache.get("protocol_sha256") != protocol[PROTOCOL_DIGEST_FIELD]
        or cache.get("basis_set_digest") != protocol["corpus"]["basis_set_digest"]
        or cache.get("oracle") != expected_oracle
    ):
        raise ValueError("Oracle cache belongs to another frozen protocol")
    entries = cache.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("Oracle cache has no entry mapping")
    validated = {}
    for digest, raw_entry in entries.items():
        if not isinstance(digest, str) or not isinstance(raw_entry, Mapping):
            raise ValueError("Oracle cache contains a malformed entry")
        query = raw_entry.get("query")
        result = raw_entry.get("result")
        if (
            not isinstance(query, Mapping)
            or not isinstance(result, Mapping)
            or digest != _sha256_bytes(_canonical_json(query).encode("utf-8"))
        ):
            raise ValueError(f"Oracle cache entry digest mismatch: {digest!r}")
        validated[digest] = {
            "query": dict(query),
            "result": dict(result),
        }
    return validated


def _save_oracle_cache(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = _seal(
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "protocol_sha256": protocol[PROTOCOL_DIGEST_FIELD],
            "basis_set_digest": protocol["corpus"]["basis_set_digest"],
            "oracle": protocol["metrics"]["in_situ_bits"]["oracle"],
            "entries": dict(sorted(entries.items())),
        },
        ORACLE_CACHE_DIGEST_FIELD,
    )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
    )
    try:
        staging = staging_dir / path.name
        _write_json(staging, value)
        staging.replace(path)
    finally:
        import shutil

        shutil.rmtree(staging_dir, ignore_errors=True)


def freeze_protocol(
    corpus_dir: Path,
    *,
    output: Path,
    validation_loader: Callable[
        [Path], tuple[dict[str, Any], FunctionDefSet, list[Any]]
    ] = _load_validation_corpus,
    host_loader: Callable[[], Mapping[str, Any]] = evaluation_host,
    source_loader: Callable[[], Mapping[str, Any]] = study_source_identity,
    clean_source_checker: Callable[[], None] = require_clean_study_checkout,
) -> dict[str, Any]:
    """Freeze the one validation cohort and every comparison dimension."""

    from wandering_light.evals.deep_corpus_protocol import freeze_cohort

    clean_source_checker()
    _require_gitignored_operational_path(output)
    if output.exists():
        raise FileExistsError(
            f"Protocol already exists; refusing to overwrite: {output}"
        )
    manifest, functions, records = validation_loader(corpus_dir)
    if any(record.split != "validation" for record in records):
        raise ValueError("Validation loader exposed a record from another split")
    action_count = len(functions.functions)
    _frozen_arms(action_count)
    cohort = freeze_cohort(
        records,
        split="validation",
        selection_seed=COHORT_SELECTION_SEED,
    )
    protocol = _seal(
        _protocol_body(
            manifest=manifest,
            cohort=cohort,
            host=host_loader(),
            source=source_loader(),
        ),
        PROTOCOL_DIGEST_FIELD,
    )
    protocol = validate_protocol(protocol)
    _write_json(output, protocol, exclusive=True)
    return protocol


def _require_gitignored_operational_path(output: Path) -> None:
    """Keep the frozen runtime protocol from dirtying its training checkout."""

    repo_root = Path(__file__).resolve().parents[1]
    resolved = output.resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        return
    relative = resolved.relative_to(repo_root.resolve()).as_posix()
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative],
            cwd=repo_root,
            check=False,
        )
    except OSError as error:
        raise RuntimeError("Could not check the protocol output path") from error
    if completed.returncode == 0:
        return
    if completed.returncode == 1:
        raise ValueError(
            "An operational protocol inside the repository must be git-ignored; "
            "write it under checkpoints/ so training retains a clean source identity"
        )
    raise RuntimeError("git check-ignore failed for the protocol output path")


def _checkpoint_identity(
    checkpoint: Path,
    *,
    arm: Mapping[str, Any],
    training_seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from wandering_light.rollout_policy import (
        CONFIG_FILE,
        TRAINING_MANIFEST_FILE,
        assert_weights_only_checkpoint,
    )

    assert_weights_only_checkpoint(checkpoint)
    config = _read_json(checkpoint / CONFIG_FILE, label="policy config")
    manifest = _read_json(
        checkpoint / TRAINING_MANIFEST_FILE,
        label="training manifest",
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("objective") != "witness-next-action-cross-entropy"
    ):
        raise ValueError("Checkpoint training did not complete the frozen objective")
    raw_source_code = manifest.get("source_code")
    source_code = dict(raw_source_code) if isinstance(raw_source_code, Mapping) else {}
    protocol_source = protocol.get("source", {})
    expected_runner = next(
        (
            row
            for row in protocol_source.get("files", ())
            if row.get("path") == "experiments/train_rollout_policy.py"
        ),
        None,
    )
    actual_runner = source_code.get("runner")
    actual_runner = dict(actual_runner) if isinstance(actual_runner, Mapping) else {}
    expected_runner_identity = (
        {
            "path": expected_runner.get("path"),
            "sha256": expected_runner.get("sha256"),
        }
        if isinstance(expected_runner, Mapping)
        else {}
    )
    source_comparisons = (
        ("dirty", False, source_code.get("dirty")),
        (
            "head_commit",
            protocol_source.get("head_commit"),
            source_code.get("head_commit"),
        ),
        (
            "runner.path",
            expected_runner_identity.get("path"),
            actual_runner.get("path"),
        ),
        (
            "runner.sha256",
            expected_runner_identity.get("sha256"),
            actual_runner.get("sha256"),
        ),
    )
    source_protocol_comparison = {
        "matches": all(
            expected == actual for _, expected, actual in source_comparisons
        ),
        "mismatches": [
            {"field": field, "expected": expected, "actual": actual}
            for field, expected, actual in source_comparisons
            if expected != actual
        ],
    }
    architecture = config.get("architecture", {})
    expected_dimensions = arm["architecture"]
    actual_dimensions = {key: architecture.get(key) for key in expected_dimensions}
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            f"Checkpoint architecture does not match arm {arm['arm_id']}: "
            f"{actual_dimensions}"
        )
    runtime_model = manifest.get("runtime", {}).get("model", {})
    if runtime_model.get("parameters") != arm["parameters"]:
        raise ValueError("Checkpoint parameter count differs from its frozen arm")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Checkpoint has no training-runtime identity")
    architecture = runtime_model.get("architecture_config")
    if (
        runtime.get("resolved_device") not in {TRAINING_DEVICE, f"{TRAINING_DEVICE}:0"}
        or runtime.get("precision") != "fp32"
        or runtime.get("floating_parameter_dtypes") != ["torch.float32"]
        or not isinstance(architecture, Mapping)
        or any(
            architecture.get(name) != expected
            for name, expected in arm["architecture"].items()
        )
        or not isinstance(runtime.get("optimizer"), Mapping)
        or not isinstance(runtime.get("accelerator"), Mapping)
        or not isinstance(runtime.get("software"), Mapping)
    ):
        raise ValueError(
            "Checkpoint training runtime differs from the frozen fp32 GPU run"
        )
    hyperparameters = manifest.get("hyperparameters", {})
    if hyperparameters.get("seed") != training_seed:
        raise ValueError("Checkpoint seed differs from the requested paired seed")
    arm_config = _arm_config_by_id(str(arm["arm_id"]))
    expected_hyperparameters = {
        "epochs": TRAINING_EPOCHS,
        "learning_rate": arm_config.learning_rate,
        "device": TRAINING_DEVICE,
        "train_batch_size": arm_config.train_batch_size,
        "gradient_accumulation_steps": arm_config.gradient_accumulation_steps,
        "effective_train_batch_size": EFFECTIVE_TRAIN_BATCH_SIZE,
        "validation_batch_size": arm_config.validation_batch_size,
        "selection": CHECKPOINT_SELECTION,
        "weights_only": True,
    }
    for field, expected in expected_hyperparameters.items():
        if hyperparameters.get(field) != expected:
            raise ValueError(
                f"Checkpoint training hyperparameter {field!r} differs from "
                f"the protocol: {hyperparameters.get(field)!r} != {expected!r}"
            )
    corpus = protocol["corpus"]
    source = manifest.get("source_corpus", {})
    if source.get("manifest_digest") != corpus["manifest_digest"]:
        raise ValueError("Checkpoint was trained on another corpus")
    if (
        source.get("basis_set_id"),
        source.get("basis_set_digest"),
    ) != (corpus["basis_set_id"], corpus["basis_set_digest"]):
        raise ValueError("Checkpoint source-corpus basis differs from the protocol")
    source_verification = source.get("verification", {})
    if not isinstance(source_verification, Mapping):
        raise ValueError("Checkpoint has no source-corpus verification evidence")
    _require_selective_corpus_verification(
        source_verification,
        manifest_digest=corpus["manifest_digest"],
        selected_splits=("discovery", "validation"),
        manifest_splits=("discovery", "test", "validation"),
        required_references=corpus.get("reference_corpora_excluded", ()),
        label="Checkpoint source corpus",
    )
    task_roles = source.get("task_roles", {})
    if (
        task_roles.get("discovery", {}).get("role") != "training-only"
        or task_roles.get("validation", {}).get("role") != "model-selection-only"
    ):
        raise ValueError("Checkpoint corpus split roles differ from the protocol")
    config_basis = config.get("basis_set", {})
    if (
        config_basis.get("basis_set_id"),
        config_basis.get("basis_set_digest"),
    ) != (corpus["basis_set_id"], corpus["basis_set_digest"]):
        raise ValueError("Checkpoint basis differs from the frozen corpus basis")
    if manifest.get("prefix_examples", {}).get("test") != 0:
        raise ValueError("Checkpoint materialized held-out prefix examples")
    trainer_state = manifest.get("trainer_state", {})
    best_epoch = trainer_state.get("best_epoch")
    best_loss = trainer_state.get("best_validation_cross_entropy")
    if (
        trainer_state.get("epochs_completed") != TRAINING_EPOCHS
        or type(best_epoch) is not int
        or not 1 <= best_epoch <= TRAINING_EPOCHS
        or isinstance(best_loss, bool)
        or not isinstance(best_loss, int | float)
        or not math.isfinite(float(best_loss))
    ):
        raise ValueError("Checkpoint has invalid minimum-validation-CE selection state")
    files = sorted(path for path in checkpoint.iterdir() if path.is_file())
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
        "artifacts": artifacts,
        "artifacts_sha256": _sha256_bytes(_canonical_json(artifacts).encode("utf-8")),
        "parameters": arm["parameters"],
        "training_seed": training_seed,
        "source_code": source_code,
        "source_protocol_comparison": source_protocol_comparison,
        "training_runtime": dict(runtime),
        "training_environment": {
            key: runtime.get(key)
            for key in (
                "resolved_device",
                "precision",
                "floating_parameter_dtypes",
                "optimizer",
                "accelerator",
                "software",
            )
        },
    }


def _load_scorer(checkpoint: Path, device: str) -> Any:
    from wandering_light.rollout_policy import RolloutPolicy

    return RolloutPolicy.load(checkpoint, device=device)


def _predictor_factory(
    scorer: Any, beam_width: int, max_depth: int, execution_budget: int
) -> Any:
    from wandering_light.solver import RolloutPredictor

    return RolloutPredictor(
        scorer,
        beam_width=beam_width,
        max_depth=max_depth,
        budget=execution_budget,
    )


def _observer_factory() -> Any:
    from wandering_light.compute_optimal import ScoredVisitObserver

    return ScoredVisitObserver()


def _synchronize(device: str) -> None:
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested without CUDA")
        torch.cuda.synchronize(resolved)


def _result_signature(result: Any) -> dict[str, Any]:
    return {
        "termination": result.termination,
        "executions": result.executions,
        "solution_function_names": [function.name for function in result.function_defs],
        "trace": [
            {
                "depth": row.depth,
                "retained_paths": [list(path) for path in row.retained_paths],
                "unique_successors": row.unique_successors,
            }
            for row in getattr(result, "trace", ())
        ],
    }


def _verify_result(
    record: Any,
    result: Any,
    functions: FunctionDefSet,
) -> tuple[bool, list[str]]:
    from wandering_light.executor import Executor
    from wandering_light.trajectory import TrajectorySpec

    if result.termination not in {
        "solved",
        "budget",
        "max_depth",
        "frontier_exhausted",
    }:
        raise ValueError(f"Rollout returned unknown termination {result.termination!r}")
    if (
        type(result.executions) is not int
        or result.executions < 0
        or result.executions > EXECUTION_BUDGET
    ):
        raise ValueError(
            f"Rollout returned invalid execution cost {result.executions!r}"
        )
    names = [function.name for function in result.function_defs]
    solved = result.termination == "solved"
    if solved:
        execution = Executor(functions).execute_trajectory(
            TrajectorySpec(record.input_value, result.function_defs)
        )
        if (
            not execution.success
            or execution.trajectory is None
            or execution.trajectory.output != record.output_value
        ):
            raise ValueError(f"Rollout reported a false solve for {record.task_id}")
    elif names:
        raise ValueError("Failed rollout returned a non-empty solution")
    return solved, names


def _visit_payload(visit: Any) -> dict[str, Any]:
    if hasattr(visit, "to_dict"):
        value = visit.to_dict()
    elif isinstance(visit, Mapping):
        value = dict(visit)
    else:
        raise TypeError(f"Scored visit is not serializable: {type(visit)!r}")
    if not isinstance(value, dict):
        raise TypeError("ScoredVisit.to_dict() must return a dictionary")
    if "state" not in value and "state_serialized" in value:
        value["state"] = value.pop("state_serialized")
    if "compatible_actions" not in value and "compatible_action_names" in value:
        value["compatible_actions"] = value.pop("compatible_action_names")
    required = {"depth", "state", "path", "scores", "compatible_actions"}
    if missing := required - value.keys():
        raise ValueError(f"Scored visit is missing fields: {sorted(missing)}")
    compatible = list(value["compatible_actions"])
    scores = value["scores"]
    if not isinstance(scores, Mapping):
        raise ValueError("Scored visit scores must be a mapping")
    # The full score row is losslessly captured by the observer, but only the
    # production-compatible logits contribute to H and are worth persisting.
    value["scores"] = {name: float(scores[name]) for name in compatible}
    value["compatible_actions"] = compatible
    value["path"] = list(value["path"])
    normalizer = value.get("log_normalizer")
    if isinstance(normalizer, int | float) and math.isfinite(float(normalizer)):
        value["log_normalizer"] = float(normalizer)
        value["mask_empty"] = False
    else:
        value["log_normalizer"] = None
        value["mask_empty"] = True
    return value


def _time_task(
    *,
    record: Any,
    predictor: Any,
    functions: FunctionDefSet,
    arm_id: str,
    training_seed: int,
    device: str,
    synchronizer: Callable[[str], None],
    clock: Callable[[], float],
    result_verifier: Callable[[Any, Any, FunctionDefSet], tuple[bool, list[str]]],
) -> tuple[dict[str, Any], Any]:
    """Run and verify one uninstrumented search inside the timed phase."""

    synchronizer(device)
    started = clock()
    timed = predictor.search_with_cost(
        record.input_value,
        record.output_value,
        functions,
    )
    synchronizer(device)
    seconds = clock() - started
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Timed rollout produced invalid duration {seconds!r}")
    solved, names = result_verifier(record, timed, functions)
    witness = tuple(record.witness_function_names)
    row = {
        "task_id": record.task_id,
        "split": record.split,
        "certified_distance": record.metadata["certified_distance"],
        "root_digest": record.metadata["root_digest"],
        "arm_id": arm_id,
        "training_seed": training_seed,
        "device": device,
        "success": solved,
        "seconds": seconds,
        "executions": timed.executions,
        "termination": timed.termination,
        "solution_length": len(names) if solved else None,
        "solution_function_names": names if solved else [],
        "witness_exact_match": solved and tuple(names) == witness,
        "result_signature": _result_signature(timed),
        "observer_replay_exact": False,
        "scored_visits": None,
        "sampled_visits": None,
        "scored_visits_by_rollout_depth": None,
    }
    return row, timed


def _visit_rank(task_id: str, visit_index: int, visit: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "sampling_seed": VISIT_SAMPLING_SEED,
                "task_id": task_id,
                "rollout_depth": visit["depth"],
                "visit_index": visit_index,
                "state": visit["state"],
                "path": visit["path"],
            }
        ).encode("utf-8")
    )


def _sample_visits(
    task_id: str,
    visits: Sequence[Any],
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, int]]:
    by_depth: dict[int, list[tuple[str, int, dict[str, Any]]]] = {}
    for visit_index, raw_visit in enumerate(visits):
        visit = _visit_payload(raw_visit)
        depth = visit["depth"]
        if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
            raise ValueError(f"Scored visit has invalid rollout depth {depth!r}")
        by_depth.setdefault(depth, []).append(
            (_visit_rank(task_id, visit_index, visit), visit_index, visit)
        )
    sampled = []
    for depth_rows in by_depth.values():
        sampled.extend(
            (visit_index, visit)
            for _, visit_index, visit in sorted(depth_rows)[:VISITS_PER_TASK_DEPTH]
        )
    sampled.sort(key=lambda item: item[0])
    counts = {
        str(depth): len(depth_rows) for depth, depth_rows in sorted(by_depth.items())
    }
    return sampled, counts


def _oracle_value(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field)


def _result_dict(value: Any, *, label: str) -> dict[str, Any]:
    result = value.to_dict() if hasattr(value, "to_dict") else value
    if not isinstance(result, Mapping):
        raise TypeError(f"{label} must be a mapping or expose to_dict()")
    return dict(result)


def _annotate_visit(
    *,
    record: Any,
    visit: dict[str, Any],
    functions: FunctionDefSet,
    basis_set_digest: str,
    oracle_limits: Any,
    oracle_cache: dict[str, dict[str, Any]],
    state_parser: Callable[[str], Any],
    oracle_certifier: Callable[..., Any],
    bits_scorer: Callable[..., Any],
) -> tuple[dict[str, Any], bool]:
    state = state_parser(visit["state"])
    oracle_identity = {
        "state_search_key": repr(state.search_key()),
        "target_canonical_key": repr(record.output_value.canonical_key()),
        "basis_set_digest": basis_set_digest,
        "limits": {
            "max_depth": ORACLE_MAX_DEPTH,
            "max_states": ORACLE_MAX_STATES,
            "max_executions": ORACLE_MAX_EXECUTIONS,
        },
    }
    cache_key = _sha256_bytes(_canonical_json(oracle_identity).encode("utf-8"))
    cache_hit = cache_key in oracle_cache
    if cache_hit:
        cache_entry = oracle_cache[cache_key]
        if cache_entry.get("query") != oracle_identity:
            raise ValueError(f"Oracle cache query collision: {cache_key}")
        oracle = cache_entry["result"]
    else:
        oracle = _result_dict(
            oracle_certifier(
                state,
                record.output_value,
                functions,
                limits=oracle_limits,
            ),
            label="oracle result",
        )
        oracle_cache[cache_key] = {
            "query": oracle_identity,
            "result": oracle,
        }
    status = _oracle_value(oracle, "status")
    complete = _oracle_value(oracle, "optimal_actions_complete")
    if status == "already_solved":
        raise RuntimeError("Production rollout scored a state that was already solved")
    bits = None
    if status == "certified" and complete is True:
        bits = bits_scorer(
            visit["scores"],
            visit["compatible_actions"],
            _oracle_value(oracle, "optimal_action_names"),
            epsilon=OPTIMAL_BITS_EPSILON,
        )
    return {
        **visit,
        "oracle_query_sha256": cache_key,
        "oracle": dict(oracle),
        "masked_optimal_bits": (
            _result_dict(bits, label="masked optimal bits")
            if bits is not None
            else None
        ),
    }, cache_hit


def _replay_task(
    *,
    record: Any,
    timed_row: Mapping[str, Any],
    predictor: Any,
    functions: FunctionDefSet,
    arm_id: str,
    training_seed: int,
    device: str,
    basis_set_digest: str,
    observer_factory: Callable[[], Any],
    result_verifier: Callable[[Any, Any, FunctionDefSet], tuple[bool, list[str]]],
    oracle_limits: Any,
    oracle_cache: dict[str, dict[str, Any]],
    state_parser: Callable[[str], Any],
    oracle_certifier: Callable[..., Any],
    bits_scorer: Callable[..., Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    observer = observer_factory()
    replay = predictor.search_with_cost(
        record.input_value,
        record.output_value,
        functions,
        observer=observer,
    )
    if _result_signature(replay) != timed_row.get("result_signature"):
        raise RuntimeError(
            f"Observer replay changed rollout result for task {record.task_id}"
        )
    replay_solved, replay_names = result_verifier(record, replay, functions)
    expected = (bool(timed_row["success"]), list(timed_row["solution_function_names"]))
    if (replay_solved, replay_names) != expected:
        raise RuntimeError(
            f"Observer replay changed executor verification for task {record.task_id}"
        )

    sampled, counts_by_depth = _sample_visits(record.task_id, observer.visits)
    sampled_by_depth = Counter(str(visit["depth"]) for _, visit in sampled)
    witness = tuple(record.witness_function_names)
    rows = []
    cache_hits = 0
    for sample_index, (visit_index, visit) in enumerate(sampled):
        annotated, cache_hit = _annotate_visit(
            record=record,
            visit=visit,
            functions=functions,
            basis_set_digest=basis_set_digest,
            oracle_limits=oracle_limits,
            oracle_cache=oracle_cache,
            state_parser=state_parser,
            oracle_certifier=oracle_certifier,
            bits_scorer=bits_scorer,
        )
        cache_hits += int(cache_hit)
        path = tuple(annotated["path"])
        on_witness = path == witness[: len(path)]
        rows.append(
            {
                "task_id": record.task_id,
                "certified_distance": record.metadata["certified_distance"],
                "root_digest": record.metadata["root_digest"],
                "arm_id": arm_id,
                "training_seed": training_seed,
                "device": device,
                "visit_index": visit_index,
                "sample_index": sample_index,
                "sampling_stratum": {
                    "task_id": record.task_id,
                    "rollout_depth": annotated["depth"],
                    "total_visits": counts_by_depth[str(annotated["depth"])],
                    "sampled_visits": sampled_by_depth[str(annotated["depth"])],
                },
                "sampling_weight": (
                    counts_by_depth[str(annotated["depth"])]
                    / sampled_by_depth[str(annotated["depth"])]
                ),
                "target_serialized": record.output_value.to_string(),
                "recorded_witness_prefix": on_witness,
                "recorded_witness_next_action": (
                    witness[len(path)]
                    if on_witness and len(path) < len(witness)
                    else None
                ),
                "visit": annotated,
            }
        )
    row = dict(timed_row)
    row.update(
        {
            "observer_replay_exact": True,
            "scored_visits": len(observer.visits),
            "sampled_visits": len(rows),
            "scored_visits_by_rollout_depth": counts_by_depth,
        }
    )
    return row, rows, cache_hits


def _auc_to_dict(value: Any) -> dict[str, Any]:
    result = value.to_dict() if hasattr(value, "to_dict") else value
    if not isinstance(result, Mapping):
        raise TypeError("AUC result must be a mapping or expose to_dict()")
    result = dict(result)
    for key in ("area", "normalized_area"):
        number = result.get(key)
        if not isinstance(number, int | float) or not math.isfinite(float(number)):
            raise ValueError(f"AUC result has no finite {key}")
    return result


def _distance_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from wandering_light.compute_optimal import execution_auc, wall_clock_auc

    output = {}
    for distance in CERTIFIED_DISTANCES:
        selected = [row for row in rows if row["certified_distance"] == distance]
        if len(selected) != TASKS_PER_DISTANCE:
            raise ValueError(
                f"Distance {distance} has {len(selected)} tasks; "
                f"expected {TASKS_PER_DISTANCE}"
            )
        hit_seconds = [row["seconds"] if row["success"] else None for row in selected]
        hit_executions = [
            row["executions"] if row["success"] else None for row in selected
        ]
        solved = sum(bool(row["success"]) for row in selected)
        output[str(distance)] = {
            "tasks": len(selected),
            "solved": solved,
            "solve_rate": solved / len(selected),
            "wall_clock_auc": _auc_to_dict(
                wall_clock_auc(
                    hit_seconds,
                    min_seconds=WALL_SECONDS_MIN,
                    max_seconds=WALL_SECONDS_MAX,
                )
            ),
            "execution_auc": _auc_to_dict(
                execution_auc(
                    hit_executions,
                    min_executions=EXECUTIONS_MIN,
                    max_executions=EXECUTION_BUDGET,
                )
            ),
            "mean_terminal_seconds": statistics.fmean(
                float(row["seconds"]) for row in selected
            ),
            "mean_terminal_executions": statistics.fmean(
                int(row["executions"]) for row in selected
            ),
        }
    return output


_BITS_FIELDS = ("bits", "raw_bits", "uniform_bits", "bits_saved")
_REPLAY_TELEMETRY_FIELDS = frozenset(
    {
        "observer_replay_exact",
        "scored_visits",
        "sampled_visits",
        "scored_visits_by_rollout_depth",
    }
)


def _timed_row_core(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if key not in _REPLAY_TELEMETRY_FIELDS
    }


def _mean_bits_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    weights = [float(row.get("sampling_weight", 1.0)) for row in rows]
    denominator = sum(weights)
    return {
        field: sum(
            weight * float(row["visit"]["masked_optimal_bits"][field])
            for row, weight in zip(rows, weights, strict=True)
        )
        / denominator
        for field in _BITS_FIELDS
    }


def _task_balanced_bits(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float] | None, int]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    if not by_task:
        return None, 0
    task_means = [_mean_bits_fields(task_rows) for task_rows in by_task.values()]
    return (
        {
            field: statistics.fmean(
                float(task_mean[field])
                for task_mean in task_means
                if task_mean is not None
            )
            for field in _BITS_FIELDS
        },
        len(by_task),
    )


def _h_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_visits: int,
) -> dict[str, Any]:
    certified = [
        row for row in rows if row["visit"].get("masked_optimal_bits") is not None
    ]
    sampled = len(rows)
    coverage = len(certified) / sampled if sampled else 0.0
    sampled_weight = sum(float(row.get("sampling_weight", 1.0)) for row in rows)
    certified_weight = sum(float(row.get("sampling_weight", 1.0)) for row in certified)
    population_coverage = certified_weight / sampled_weight if sampled_weight else 0.0
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    task_coverages = []
    for task_rows in by_task.values():
        task_weight = sum(float(row.get("sampling_weight", 1.0)) for row in task_rows)
        task_certified_weight = sum(
            float(row.get("sampling_weight", 1.0))
            for row in task_rows
            if row["visit"].get("masked_optimal_bits") is not None
        )
        task_coverages.append(task_certified_weight / task_weight)
    task_coverage = statistics.fmean(task_coverages) if task_coverages else 0.0
    state_headline = bool(certified) and population_coverage >= MIN_H_CERTIFIED_COVERAGE
    task_headline = bool(certified) and task_coverage >= MIN_H_CERTIFIED_COVERAGE
    state_conditional = _mean_bits_fields(certified)
    task_conditional, tasks_with_certified = _task_balanced_bits(certified)
    queries: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        visit = row["visit"]
        queries.setdefault(visit["oracle_query_sha256"], visit["oracle"])
    oracle_status_counts = Counter(
        str(row["visit"]["oracle"].get("status")) for row in rows
    )
    unique_status_counts = Counter(
        str(oracle.get("status")) for oracle in queries.values()
    )
    unresolved_reasons = Counter(
        str(row["visit"]["oracle"].get("stop_reason"))
        for row in rows
        if row["visit"]["oracle"].get("status") == "unresolved"
    )
    effort_fields = ("reached_states", "attempted_executions", "failed_executions")
    return {
        "total_visits": total_visits,
        "sampled_visits": sampled,
        "sample_fraction": sampled / total_visits if total_visits else 0.0,
        "certified_visits": len(certified),
        "certified_coverage": coverage,
        "population_weighted_certified_coverage": population_coverage,
        "task_balanced_certified_coverage": task_coverage,
        "headline_min_certified_coverage": MIN_H_CERTIFIED_COVERAGE,
        "headline_available": state_headline and task_headline,
        "tasks_with_sampled_visits": len({str(row["task_id"]) for row in rows}),
        "tasks_with_certified_visits": tasks_with_certified,
        "unique_state_target_queries": len(queries),
        "oracle_status_counts": dict(sorted(oracle_status_counts.items())),
        "oracle_unique_status_counts": dict(sorted(unique_status_counts.items())),
        "unresolved_reason_counts": dict(sorted(unresolved_reasons.items())),
        "oracle_effort_unique_queries": {
            field: sum(int(oracle.get(field, 0)) for oracle in queries.values())
            for field in effort_fields
        },
        "state_weighted": {
            "headline_available": state_headline,
            "certified_coverage": population_coverage,
            "mean_bits": state_conditional["bits"] if state_headline else None,
            "certified_only_mean_bits": (
                state_conditional["bits"] if state_conditional is not None else None
            ),
            "conditional_on_certified_labels": state_conditional,
        },
        "task_balanced": {
            "headline_available": task_headline,
            "certified_coverage": task_coverage,
            "mean_bits": task_conditional["bits"] if task_headline else None,
            "certified_only_mean_bits": (
                task_conditional["bits"] if task_conditional is not None else None
            ),
            "conditional_on_certified_labels": task_conditional,
        },
    }


def _h_metrics(
    visit_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_distance = {}
    for distance in CERTIFIED_DISTANCES:
        selected_visits = [
            row for row in visit_rows if row["certified_distance"] == distance
        ]
        selected_tasks = [
            row for row in task_rows if row["certified_distance"] == distance
        ]
        total = sum(int(row["scored_visits"]) for row in selected_tasks)
        group = _h_group(selected_visits, total_visits=total)
        by_depth = {}
        for depth in range(1, MAX_DEPTH + 1):
            depth_rows = [
                row for row in selected_visits if row["visit"]["depth"] == depth
            ]
            depth_total = sum(
                int(row["scored_visits_by_rollout_depth"].get(str(depth), 0))
                for row in selected_tasks
            )
            by_depth[str(depth)] = _h_group(depth_rows, total_visits=depth_total)
        group["by_rollout_depth"] = by_depth
        by_distance[str(distance)] = group
    return {
        "epsilon": OPTIMAL_BITS_EPSILON,
        "by_certified_distance": by_distance,
    }


def _validate_task_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell: Mapping[str, Any],
    task_order: Mapping[str, Any],
    cohort: Mapping[str, Any],
    require_replay: bool = True,
) -> None:
    expected_ids = list(task_order["task_ids"])
    actual_ids = [str(row.get("task_id")) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("Task rows differ from the frozen cohort or its order")
    cohort_tasks = {str(task["task_id"]): task for task in cohort.get("tasks", ())}
    if set(cohort_tasks) != set(expected_ids):
        raise ValueError("Frozen cohort metadata differs from its task IDs")
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("Evaluation row is not validation-only")
        if any(row.get(key) != value for key, value in cell.items()):
            raise ValueError("Evaluation row cell identity differs from its summary")
        expected_task = cohort_tasks[str(row["task_id"])]
        if row.get("certified_distance") != expected_task.get(
            "certified_distance"
        ) or row.get("root_digest") != expected_task.get("root_digest"):
            raise ValueError("Evaluation row metadata differs from the frozen task")
        if row.get("success") is not (row.get("termination") == "solved"):
            raise ValueError("Evaluation row success and termination disagree")
        if (
            not isinstance(row.get("seconds"), int | float)
            or isinstance(row.get("seconds"), bool)
            or not math.isfinite(float(row["seconds"]))
            or row["seconds"] <= 0
        ):
            raise ValueError("Evaluation row has invalid wall-clock cost")
        executions = row.get("executions")
        if type(executions) is not int or not 0 <= executions <= EXECUTION_BUDGET:
            raise ValueError("Evaluation row has invalid execution cost")
        signature = row.get("result_signature")
        if (
            not isinstance(signature, Mapping)
            or signature.get("termination") != row.get("termination")
            or signature.get("executions") != executions
            or signature.get("solution_function_names")
            != row.get("solution_function_names")
            or not isinstance(signature.get("trace"), list)
        ):
            raise ValueError("Evaluation row has an invalid replay signature")
        visits = row.get("scored_visits")
        sampled = row.get("sampled_visits")
        by_depth = row.get("scored_visits_by_rollout_depth")
        if not require_replay:
            if (
                row.get("observer_replay_exact") is not False
                or visits is not None
                or sampled is not None
                or by_depth is not None
            ):
                raise ValueError("Timing row contains observer replay data")
            continue
        if row.get("observer_replay_exact") is not True:
            raise ValueError("Evaluation row has no exact observer replay")
        if (
            type(visits) is not int
            or visits < 0
            or type(sampled) is not int
            or not 0 <= sampled <= visits
            or not isinstance(by_depth, Mapping)
            or sum(by_depth.values()) != visits
            or sampled > VISITS_PER_TASK_DEPTH * len(by_depth)
        ):
            raise ValueError("Evaluation row has invalid visit sampling counts")
    histogram = Counter(str(row["certified_distance"]) for row in rows)
    if histogram != Counter(
        {str(distance): TASKS_PER_DISTANCE for distance in CERTIFIED_DISTANCES}
    ):
        raise ValueError("Evaluation rows are not balanced at 12 tasks per distance")


def _publish_timing_cell(
    output_dir: Path,
    *,
    protocol: Mapping[str, Any],
    cell: Mapping[str, Any],
    cell_ordinal: int,
    arm: Mapping[str, Any],
    host: Mapping[str, Any],
    corpus: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    timing_window: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_task_rows(
        rows,
        cell=cell,
        task_order=protocol["evaluation_task_order"],
        cohort=protocol["cohort"],
        require_replay=False,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        rows_path = staging / TIMED_ROWS_FILE
        _write_jsonl_gz([dict(row) for row in rows], rows_path)
        summary = _seal(
            {
                "schema_version": SCHEMA_VERSION,
                "study_id": STUDY_ID,
                "status": TIMING_STATUS,
                "protocol_sha256": protocol[PROTOCOL_DIGEST_FIELD],
                "cell": dict(cell),
                "cell_ordinal": cell_ordinal,
                "arm": dict(arm),
                "evaluation_host": dict(host),
                "corpus": dict(corpus),
                "cohort_sha256": protocol["cohort"]["cohort_sha256"],
                "evaluation_task_order_sha256": protocol["evaluation_task_order"][
                    "task_ids_sha256"
                ],
                "search": protocol["search"],
                "checkpoint": dict(checkpoint),
                "warmup": {
                    "scorer_calls": 2,
                    "batch_sizes": [1, BEAM_WIDTH],
                    "outside_task_timing": True,
                    "synchronized_after_each": True,
                },
                "timing_window": dict(timing_window),
                "tasks": len(rows),
                "metrics": _distance_metrics(rows),
                "artifact": _artifact_identity(rows_path),
            },
            TIMING_DIGEST_FIELD,
        )
        _write_json(staging / TIMING_SUMMARY_FILE, summary)
        staging.replace(output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _load_timing(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _verify_seal(
        _read_json(path, label="timing summary"),
        TIMING_DIGEST_FIELD,
        label="timing summary",
    )
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("study_id") != STUDY_ID
        or summary.get("status") != TIMING_STATUS
    ):
        raise ValueError(f"Unsupported timing summary: {path}")
    identity = summary.get("artifact")
    if not isinstance(identity, Mapping):
        raise ValueError("Timing summary has no task-row artifact")
    artifact = path.parent / str(identity.get("path"))
    if (
        not artifact.is_file()
        or artifact.stat().st_size != identity.get("bytes")
        or _sha256_file(artifact) != identity.get("sha256")
    ):
        raise ValueError("Timing task rows differ from their sealed identity")
    rows = _validated_jsonl_rows(
        _read_jsonl_gz(artifact),
        label="Timing task-row artifact",
    )
    return summary, rows


def _validate_timing_cell(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Path,
    protocol: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    timing_matrix: Mapping[str, Any] | None,
) -> None:
    if summary.get("protocol_sha256") != protocol[PROTOCOL_DIGEST_FIELD]:
        raise ValueError("Timing cell used another protocol")
    if summary.get("cell") != dict(cell):
        raise ValueError("Timing summary has another cell identity")
    if summary.get("cell_ordinal") != _cell_ordinal(protocol, cell):
        raise ValueError("Timing summary has the wrong frozen ordinal")
    if (
        summary.get("arm") != dict(arm)
        or summary.get("evaluation_host") != protocol["evaluation_host"]
        or summary.get("corpus") != protocol["corpus"]
        or summary.get("cohort_sha256") != protocol["cohort"]["cohort_sha256"]
        or summary.get("evaluation_task_order_sha256")
        != protocol["evaluation_task_order"]["task_ids_sha256"]
        or summary.get("search") != protocol["search"]
    ):
        raise ValueError("Timing summary differs from the frozen protocol")
    timed_checkpoint = summary.get("checkpoint")
    if (
        not isinstance(timed_checkpoint, Mapping)
        or timed_checkpoint.get("artifacts_sha256")
        != checkpoint.get("artifacts_sha256")
        or not isinstance(timed_checkpoint.get("training_environment"), Mapping)
    ):
        raise ValueError("Timing summary used another checkpoint")
    _validate_task_rows(
        rows,
        cell=cell,
        task_order=protocol["evaluation_task_order"],
        cohort=protocol["cohort"],
        require_replay=False,
    )
    if summary.get("tasks") != len(rows) or summary.get("metrics") != _distance_metrics(
        rows
    ):
        raise ValueError("Timing summary metrics differ from its task rows")
    window = summary.get("timing_window")
    if not isinstance(window, Mapping):
        raise ValueError("Timing summary has no timing window")
    try:
        started = datetime.fromisoformat(str(window["started_at_utc"]))
        completed = datetime.fromisoformat(str(window["completed_at_utc"]))
    except (KeyError, ValueError) as error:
        raise ValueError("Timing summary has an invalid timing window") from error
    if (
        started.tzinfo is None
        or completed.tzinfo is None
        or completed < started
        or not isinstance(window.get("environment_before"), Mapping)
        or not isinstance(window.get("environment_after"), Mapping)
    ):
        raise ValueError("Timing summary has an invalid timing window")
    affinity_before = window["environment_before"].get("cpu_affinity")
    affinity_after = window["environment_after"].get("cpu_affinity")
    if (
        not isinstance(affinity_before, list)
        or not affinity_before
        or affinity_before != affinity_after
    ):
        raise ValueError("Timing cell changed or omitted its pinned CPU affinity")
    if timing_matrix is not None:
        matching = [
            entry
            for entry in timing_matrix["timings"]
            if entry.get("cell") == dict(cell)
        ]
        if (
            len(matching) != 1
            or matching[0].get("timing_sha256") != summary.get(TIMING_DIGEST_FIELD)
            or Path(str(matching[0].get("summary_path"))).resolve() != path.resolve()
        ):
            raise ValueError("Timing cell is absent from the sealed timing matrix")


def load_timing_matrix(
    path: Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    matrix = _verify_seal(
        _read_json(path, label="timing matrix"),
        TIMING_MATRIX_DIGEST_FIELD,
        label="timing matrix",
    )
    if (
        matrix.get("schema_version") != SCHEMA_VERSION
        or matrix.get("study_id") != STUDY_ID
        or matrix.get("status") != TIMING_MATRIX_STATUS
        or not isinstance(matrix.get("timings"), list)
        or len(matrix["timings"]) != EVALUATION_CELL_COUNT
    ):
        raise ValueError("Unsupported or incomplete timing matrix")
    if protocol is not None and (
        matrix.get("protocol_sha256") != protocol[PROTOCOL_DIGEST_FIELD]
        or matrix.get("evaluation_cell_order_sha256")
        != protocol["evaluation_cell_order"]["cells_sha256"]
    ):
        raise ValueError("Timing matrix belongs to another protocol")
    return matrix


def seal_timing_matrix(
    protocol_path: Path,
    timing_paths: Sequence[Path],
    *,
    output: Path,
) -> dict[str, Any]:
    with _exclusive_protocol_lock(protocol_path):
        protocol = load_protocol(protocol_path)
        if output.exists():
            raise FileExistsError(f"Timing matrix already exists: {output}")
        expected_cells = {
            (arm_id, seed, device)
            for arm_id in ARM_IDS
            for seed in TRAINING_SEEDS
            for device in DEVICES
        }
        loaded: dict[tuple[str, int, str], tuple[dict[str, Any], Path]] = {}
        for path in timing_paths:
            summary, rows = _load_timing(path)
            raw_cell = summary.get("cell")
            if not isinstance(raw_cell, Mapping):
                raise ValueError("Timing summary has no cell")
            cell_key = (
                str(raw_cell.get("arm_id")),
                int(raw_cell.get("training_seed")),
                str(raw_cell.get("device")),
            )
            if cell_key not in expected_cells or cell_key in loaded:
                raise ValueError(f"Unexpected or duplicate timing cell {cell_key}")
            cell = {
                "arm_id": cell_key[0],
                "training_seed": cell_key[1],
                "device": cell_key[2],
            }
            arm = _arm_by_id(cell_key[0])
            checkpoint = summary.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"Timing cell {cell_key} has no checkpoint identity")
            _validate_timing_cell(
                summary,
                rows,
                path=path,
                protocol=protocol,
                cell=cell,
                arm=arm,
                checkpoint=checkpoint,
                timing_matrix=None,
            )
            loaded[cell_key] = (summary, path)
        missing = sorted(expected_cells - set(loaded))
        if missing:
            raise ValueError(f"Timing matrix is incomplete; missing={missing}")

        ordered = sorted(loaded.values(), key=lambda row: row[0]["cell_ordinal"])
        previous_completed: datetime | None = None
        cpu_affinities = []
        timings = []
        for summary, path in ordered:
            window = summary["timing_window"]
            started = datetime.fromisoformat(window["started_at_utc"])
            completed = datetime.fromisoformat(window["completed_at_utc"])
            if previous_completed is not None and started < previous_completed:
                raise ValueError("Timing cells do not follow the frozen cell order")
            previous_completed = completed
            checkpoint = summary["checkpoint"]
            cpu_affinities.append(window["environment_before"]["cpu_affinity"])
            timings.append(
                {
                    "cell": summary["cell"],
                    "cell_ordinal": summary["cell_ordinal"],
                    "summary_path": str(path.resolve()),
                    "timing_sha256": summary[TIMING_DIGEST_FIELD],
                    "task_rows_sha256": summary["artifact"]["sha256"],
                    "checkpoint_artifacts_sha256": checkpoint.get("artifacts_sha256"),
                    "timing_window": window,
                }
            )
        training_environments = []
        for arm_id in ARM_IDS:
            for seed in TRAINING_SEEDS:
                cpu = loaded[(arm_id, seed, "cpu")][0]["checkpoint"]
                cuda = loaded[(arm_id, seed, "cuda")][0]["checkpoint"]
                if cpu.get("artifacts_sha256") != cuda.get("artifacts_sha256"):
                    raise ValueError(
                        f"CPU/CUDA timing used different checkpoints for {(arm_id, seed)}"
                    )
                training_environments.append(
                    {
                        "arm_id": arm_id,
                        "training_seed": seed,
                        "environment": _canonical_copy(cpu.get("training_environment")),
                    }
                )
        if len({_canonical_json(value) for value in cpu_affinities}) != 1:
            raise ValueError("Timing matrix mixes CPU affinities")

        environment_identities = {
            _canonical_json(row["environment"]) for row in training_environments
        }

        sealed_at = datetime.now(UTC)
        matrix = _seal(
            {
                "schema_version": SCHEMA_VERSION,
                "study_id": STUDY_ID,
                "status": TIMING_MATRIX_STATUS,
                "protocol_sha256": protocol[PROTOCOL_DIGEST_FIELD],
                "evaluation_cell_order_sha256": protocol["evaluation_cell_order"][
                    "cells_sha256"
                ],
                "sealed_at_utc": sealed_at.isoformat(),
                "latest_timing_completed_at_utc": (
                    previous_completed.isoformat()
                    if previous_completed is not None
                    else None
                ),
                "training_environments": {
                    "all_identical": len(environment_identities) == 1,
                    "by_checkpoint": training_environments,
                },
                "timings": timings,
            },
            TIMING_MATRIX_DIGEST_FIELD,
        )
        _write_json(output, matrix, exclusive=True)
        return matrix


@_serialized_evaluation
def evaluate_cell(
    config: EvaluationConfig,
    *,
    protocol_loader: Callable[[Path], dict[str, Any]] = load_protocol,
    validation_loader: Callable[
        [Path], tuple[dict[str, Any], FunctionDefSet, list[Any]]
    ] = _load_validation_corpus,
    host_loader: Callable[[], Mapping[str, Any]] = evaluation_host,
    checkpoint_validator: Callable[..., dict[str, Any]] = _checkpoint_identity,
    scorer_loader: Callable[[Path, str], Any] = _load_scorer,
    predictor_builder: Callable[[Any, int, int, int], Any] = _predictor_factory,
    observer_builder: Callable[[], Any] = _observer_factory,
    synchronizer: Callable[[str], None] = _synchronize,
    clock: Callable[[], float] = time.perf_counter,
    result_verifier: Callable[
        [Any, Any, FunctionDefSet], tuple[bool, list[str]]
    ] = _verify_result,
    state_parser: Callable[[str], Any] | None = None,
    oracle_certifier: Callable[..., Any] | None = None,
    bits_scorer: Callable[..., Any] | None = None,
    oracle_limits: Any | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one architecture/seed/device cell."""

    protocol = protocol_loader(config.protocol)
    arm = _arm_by_id(config.arm_id)
    if config.training_seed not in TRAINING_SEEDS:
        raise ValueError("Training seed is not part of the frozen paired sweep")
    if config.device not in DEVICES:
        raise ValueError("Device is not part of the frozen CPU/CUDA sweep")
    cell = {
        "arm_id": arm["arm_id"],
        "training_seed": config.training_seed,
        "device": config.device,
    }
    cell_ordinal = _cell_ordinal(protocol, cell)
    annotating = config.timing_summary is not None or config.timing_matrix is not None
    if annotating and (config.timing_summary is None or config.timing_matrix is None):
        raise ValueError("Annotation requires both a timing summary and timing matrix")
    if config.output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {config.output_dir}")
    host = dict(host_loader())
    if host != protocol["evaluation_host"]:
        raise ValueError(
            "Evaluation host differs from the frozen wall-clock host: "
            f"current={host}, frozen={protocol['evaluation_host']}"
        )
    manifest, functions, all_records = validation_loader(config.corpus_dir)
    corpus = protocol["corpus"]
    if (
        manifest.get("manifest_digest"),
        manifest.get("basis_set_id"),
        manifest.get("basis_set_digest"),
    ) != (
        corpus["manifest_digest"],
        corpus["basis_set_id"],
        corpus["basis_set_digest"],
    ):
        raise ValueError("Evaluation corpus differs from the frozen protocol")
    records = _validate_cohort_records(
        all_records,
        protocol["cohort"],
        protocol["evaluation_task_order"],
    )
    if len(functions.functions) != FROZEN_ACTION_COUNT:
        raise ValueError("Evaluation basis action count differs from the frozen sweep")
    checkpoint = checkpoint_validator(
        config.checkpoint,
        arm=arm,
        training_seed=config.training_seed,
        protocol=protocol,
    )
    scorer = scorer_loader(config.checkpoint, config.device)

    if annotating:
        if state_parser is None:
            from wandering_light.typed_list import TypedList

            state_parser = TypedList.from_str
        if oracle_certifier is None or bits_scorer is None or oracle_limits is None:
            from wandering_light.compute_optimal import (
                OracleLimits,
                PreparedOptimalActionOracle,
                masked_optimal_set_bits,
            )

            oracle_certifier = oracle_certifier or PreparedOptimalActionOracle(
                functions
            )
            bits_scorer = bits_scorer or masked_optimal_set_bits
            oracle_limits = oracle_limits or OracleLimits(
                max_depth=ORACLE_MAX_DEPTH,
                max_states=ORACLE_MAX_STATES,
                max_executions=ORACLE_MAX_EXECUTIONS,
            )

    timing_summary: dict[str, Any] | None = None
    timing_matrix: dict[str, Any] | None = None
    timed_rows: list[dict[str, Any]] = []
    oracle_cache_path = config.oracle_cache or config.protocol.with_name(
        ORACLE_CACHE_FILE
    )
    oracle_cache: dict[str, dict[str, Any]] = {}
    annotation_started_at: datetime | None = None
    if annotating:
        assert config.timing_summary is not None
        assert config.timing_matrix is not None
        timing_matrix = load_timing_matrix(
            config.timing_matrix,
            protocol=protocol,
        )
        timing_summary, timed_rows = _load_timing(config.timing_summary)
        _validate_timing_cell(
            timing_summary,
            timed_rows,
            path=config.timing_summary,
            protocol=protocol,
            cell=cell,
            arm=arm,
            checkpoint=checkpoint,
            timing_matrix=timing_matrix,
        )
        # Fail before model warmup or replay if the shared cache is foreign/corrupt.
        oracle_cache = _load_oracle_cache(oracle_cache_path, protocol=protocol)
        annotation_started_at = datetime.now(UTC)
        try:
            matrix_sealed_at = datetime.fromisoformat(timing_matrix["sealed_at_utc"])
            latest_timing = datetime.fromisoformat(
                timing_matrix["latest_timing_completed_at_utc"]
            )
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Timing matrix has an invalid global barrier") from error
        if (
            matrix_sealed_at.tzinfo is None
            or latest_timing.tzinfo is None
            or matrix_sealed_at < latest_timing
            or annotation_started_at < matrix_sealed_at
        ):
            raise ValueError("Annotation began before the sealed global timing barrier")

    # Warm both shapes used by production rollout, outside every task interval.
    first = records[0]
    for batch_size in (1, BEAM_WIDTH):
        scorer.score_actions_batch(
            [(first.input_value, first.output_value)] * batch_size,
            functions,
        )
        synchronizer(config.device)
    predictor = predictor_builder(
        scorer,
        BEAM_WIDTH,
        MAX_DEPTH,
        EXECUTION_BUDGET,
    )

    if not annotating:
        timing_started_at = datetime.now(UTC)
        timing_environment_before = _timing_environment()
        for index, record in enumerate(records, start=1):
            row, _ = _time_task(
                record=record,
                predictor=predictor,
                functions=functions,
                arm_id=str(arm["arm_id"]),
                training_seed=config.training_seed,
                device=config.device,
                synchronizer=synchronizer,
                clock=clock,
                result_verifier=result_verifier,
            )
            timed_rows.append(row)
            if config.progress:
                print(
                    f"[timed {arm['arm_id']} seed={config.training_seed} "
                    f"{config.device}] {index}/{len(records)} "
                    f"d={row['certified_distance']} "
                    f"{row['termination']} {row['seconds']:.4f}s",
                    flush=True,
                )
        timing_environment_after = _timing_environment()
        timing_completed_at = datetime.now(UTC)
        return _publish_timing_cell(
            config.output_dir,
            protocol=protocol,
            cell=cell,
            cell_ordinal=cell_ordinal,
            arm=arm,
            host=host,
            corpus=corpus,
            checkpoint=checkpoint,
            rows=timed_rows,
            timing_window={
                "started_at_utc": timing_started_at.isoformat(),
                "completed_at_utc": timing_completed_at.isoformat(),
                "environment_before": timing_environment_before,
                "environment_after": timing_environment_after,
            },
        )

    assert timing_summary is not None
    assert timing_matrix is not None
    assert annotation_started_at is not None
    assert state_parser is not None
    assert oracle_certifier is not None
    assert bits_scorer is not None
    assert oracle_limits is not None
    oracle_cache_entries_before = len(oracle_cache)
    oracle_cache_hits = 0
    cell_oracle_queries: set[str] = set()
    task_rows = []
    visit_rows = []
    for index, (record, timed_row) in enumerate(
        zip(records, timed_rows, strict=True),
        start=1,
    ):
        row, visits, cache_hits = _replay_task(
            record=record,
            timed_row=timed_row,
            predictor=predictor,
            functions=functions,
            arm_id=str(arm["arm_id"]),
            training_seed=config.training_seed,
            device=config.device,
            basis_set_digest=str(corpus["basis_set_digest"]),
            observer_factory=observer_builder,
            result_verifier=result_verifier,
            oracle_limits=oracle_limits,
            oracle_cache=oracle_cache,
            state_parser=state_parser,
            oracle_certifier=oracle_certifier,
            bits_scorer=bits_scorer,
        )
        task_rows.append(row)
        visit_rows.extend(visits)
        cell_oracle_queries.update(
            str(visit["visit"]["oracle_query_sha256"]) for visit in visits
        )
        oracle_cache_hits += cache_hits
        if index % ORACLE_CACHE_SAVE_EVERY_TASKS == 0:
            _save_oracle_cache(
                oracle_cache_path,
                protocol=protocol,
                entries=oracle_cache,
            )
        if config.progress:
            print(
                f"[replay {arm['arm_id']} seed={config.training_seed} "
                f"{config.device}] {index}/{len(records)} "
                f"visits={row['scored_visits']} sampled={row['sampled_visits']}",
                flush=True,
            )
    if len(records) % ORACLE_CACHE_SAVE_EVERY_TASKS:
        _save_oracle_cache(
            oracle_cache_path,
            protocol=protocol,
            entries=oracle_cache,
        )
    annotation_completed_at = datetime.now(UTC)
    _validate_task_rows(
        task_rows,
        cell=cell,
        task_order=protocol["evaluation_task_order"],
        cohort=protocol["cohort"],
    )
    metrics = _distance_metrics(task_rows)
    h_metrics = _h_metrics(visit_rows, task_rows)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.staging-",
            dir=config.output_dir.parent,
        )
    )
    try:
        task_path = staging / TASK_ROWS_FILE
        visit_path = staging / VISIT_ROWS_FILE
        _write_jsonl_gz(task_rows, task_path)
        _write_jsonl_gz(visit_rows, visit_path)
        summary = _seal(
            {
                "schema_version": SCHEMA_VERSION,
                "study_id": STUDY_ID,
                "status": EVALUATION_STATUS,
                "protocol_sha256": protocol[PROTOCOL_DIGEST_FIELD],
                "timing_sha256": timing_summary[TIMING_DIGEST_FIELD],
                "timing_matrix_sha256": timing_matrix[TIMING_MATRIX_DIGEST_FIELD],
                "timing_summary_path": str(config.timing_summary.resolve()),
                "timing_matrix_path": str(config.timing_matrix.resolve()),
                "cell": cell,
                "cell_ordinal": cell_ordinal,
                "arm": dict(arm),
                "evaluation_host": host,
                "corpus": corpus,
                "cohort_sha256": protocol["cohort"]["cohort_sha256"],
                "evaluation_task_order_sha256": protocol["evaluation_task_order"][
                    "task_ids_sha256"
                ],
                "search": protocol["search"],
                "checkpoint": checkpoint,
                "warmup": {
                    "scorer_calls": 2,
                    "batch_sizes": [1, BEAM_WIDTH],
                    "outside_task_timing": True,
                    "synchronized_after_each": True,
                },
                "timing_window": timing_summary["timing_window"],
                "annotation_window": {
                    "started_at_utc": annotation_started_at.isoformat(),
                    "completed_at_utc": annotation_completed_at.isoformat(),
                },
                "tasks": len(task_rows),
                "scored_visits": sum(row["scored_visits"] for row in task_rows),
                "sampled_visits": len(visit_rows),
                "visit_sampling": {
                    "total_visits": sum(row["scored_visits"] for row in task_rows),
                    "sampled_visits": len(visit_rows),
                    "maximum_per_task_depth": VISITS_PER_TASK_DEPTH,
                    "sampling_seed": VISIT_SAMPLING_SEED,
                },
                "oracle_cache": {
                    "path": str(oracle_cache_path.resolve()),
                    "unique_queries_in_cell": len(cell_oracle_queries),
                    "cache_hits": oracle_cache_hits,
                    "entries_before": oracle_cache_entries_before,
                    "entries_after": len(oracle_cache),
                    "key": [
                        "TypedList.search_key(state)",
                        "TypedList.canonical_key(target)",
                        "basis_set_digest",
                    ],
                },
                "metrics": metrics,
                "h_metrics": h_metrics,
                "artifacts": {
                    "tasks": _artifact_identity(task_path),
                    "visits": _artifact_identity(visit_path),
                },
            },
            EVALUATION_DIGEST_FIELD,
        )
        _write_json(staging / SUMMARY_FILE, summary)
        staging.replace(config.output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _load_evaluation(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    value = _verify_seal(
        _read_json(path, label="evaluation summary"),
        EVALUATION_DIGEST_FIELD,
        label="evaluation summary",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("status") != EVALUATION_STATUS
    ):
        raise ValueError(f"Unsupported evaluation summary: {path}")
    artifacts = value.get("artifacts", {})
    for name in ("tasks", "visits"):
        identity = artifacts.get(name)
        if not isinstance(identity, Mapping):
            raise ValueError(f"Evaluation has no {name} artifact identity")
        artifact = path.parent / str(identity.get("path"))
        if (
            not artifact.is_file()
            or artifact.stat().st_size != identity.get("bytes")
            or _sha256_file(artifact) != identity.get("sha256")
        ):
            raise ValueError(f"Evaluation {name} artifact differs from its identity")
    task_rows = _validated_jsonl_rows(
        _read_jsonl_gz(path.parent / artifacts["tasks"]["path"]),
        label="Evaluation task-row artifact",
    )
    visit_rows = _validated_jsonl_rows(
        _read_jsonl_gz(path.parent / artifacts["visits"]["path"]),
        label="Evaluation visit-row artifact",
    )
    return value, task_rows, visit_rows


def _validate_visit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_rows: Sequence[Mapping[str, Any]],
    cell: Mapping[str, Any],
) -> None:
    expected_tasks = {str(row["task_id"]): row for row in task_rows}
    sampled_by_task: Counter[str] = Counter()
    sampled_by_stratum: Counter[tuple[str, str]] = Counter()
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id not in expected_tasks:
            raise ValueError(
                "Sampled visit belongs to a task outside the frozen cohort"
            )
        if any(row.get(key) != value for key, value in cell.items()):
            raise ValueError("Sampled visit cell identity differs from its summary")
        parent_task = expected_tasks[task_id]
        if row.get("certified_distance") != parent_task.get(
            "certified_distance"
        ) or row.get("root_digest") != parent_task.get("root_digest"):
            raise ValueError("Sampled visit metadata differs from its parent task")
        visit = row.get("visit")
        if not isinstance(visit, Mapping):
            raise ValueError("Sampled visit payload is missing")
        if not isinstance(visit.get("oracle"), Mapping):
            raise ValueError("Sampled visit has no explicit oracle result")
        status = visit["oracle"].get("status")
        bits = visit.get("masked_optimal_bits")
        labeled = (
            status == "certified"
            and visit["oracle"].get("optimal_actions_complete") is True
        )
        if labeled != isinstance(bits, Mapping):
            raise ValueError("Sampled visit certification and bits disagree")
        depth = str(visit.get("depth"))
        stratum = row.get("sampling_stratum")
        if not isinstance(stratum, Mapping):
            raise ValueError("Sampled visit has no sampling-stratum identity")
        expected_total = expected_tasks[task_id]["scored_visits_by_rollout_depth"].get(
            depth
        )
        if type(expected_total) is not int or expected_total <= 0:
            raise ValueError("Sampled visit has invalid stratum weight")
        expected_sampled_count = min(VISITS_PER_TASK_DEPTH, expected_total)
        weight = row.get("sampling_weight")
        if (
            stratum.get("task_id") != task_id
            or str(stratum.get("rollout_depth")) != depth
            or stratum.get("total_visits") != expected_total
            or stratum.get("sampled_visits") != expected_sampled_count
            or not isinstance(weight, int | float)
            or float(weight) != expected_total / expected_sampled_count
        ):
            raise ValueError("Sampled visit has invalid stratum weight")
        sampled_by_task[task_id] += 1
        sampled_by_stratum[(task_id, depth)] += 1
    expected_sampled = Counter(
        {
            str(row["task_id"]): int(row["sampled_visits"])
            for row in task_rows
            if row["sampled_visits"]
        }
    )
    if sampled_by_task != expected_sampled:
        raise ValueError("Sampled visits differ from task-level sampling counts")
    for (task_id, depth), count in sampled_by_stratum.items():
        expected = min(
            VISITS_PER_TASK_DEPTH,
            expected_tasks[task_id]["scored_visits_by_rollout_depth"][depth],
        )
        if count != expected:
            raise ValueError("Sampled visits differ from their frozen stratum size")


def _auc_summary(seed_rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    areas = [float(row[name]["area"]) for row in seed_rows]
    normalized = [float(row[name]["normalized_area"]) for row in seed_rows]
    return {
        "mean": statistics.fmean(areas),
        "sample_sd": statistics.stdev(areas),
        "normalized_mean": statistics.fmean(normalized),
        "normalized_sample_sd": statistics.stdev(normalized),
    }


def _paired_auc_difference(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    left_seeds = [int(row["training_seed"]) for row in left]
    right_seeds = [int(row["training_seed"]) for row in right]
    if left_seeds != right_seeds or left_seeds != list(TRAINING_SEEDS):
        raise ValueError("AUC comparison does not preserve the paired training seeds")
    by_seed = [
        {
            "training_seed": seed,
            "area_difference": float(left_row[name]["area"])
            - float(right_row[name]["area"]),
            "normalized_area_difference": float(left_row[name]["normalized_area"])
            - float(right_row[name]["normalized_area"]),
        }
        for seed, left_row, right_row in zip(
            left_seeds,
            left,
            right,
            strict=True,
        )
    ]
    areas = [float(row["area_difference"]) for row in by_seed]
    normalized = [float(row["normalized_area_difference"]) for row in by_seed]
    return {
        "by_training_seed": by_seed,
        "mean_area_difference": statistics.fmean(areas),
        "sample_sd_area_difference": statistics.stdev(areas),
        "mean_normalized_area_difference": statistics.fmean(normalized),
        "sample_sd_normalized_area_difference": statistics.stdev(normalized),
    }


def _spread(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
    }


def _aggregate_h(seed_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "certified_coverage": _spread(
            [float(group["certified_coverage"]) for group in seed_groups]
        ),
        "population_weighted_certified_coverage": _spread(
            [
                float(group["population_weighted_certified_coverage"])
                for group in seed_groups
            ]
        ),
        "task_balanced_certified_coverage": _spread(
            [float(group["task_balanced_certified_coverage"]) for group in seed_groups]
        ),
        "sampled_visits": [int(group["sampled_visits"]) for group in seed_groups],
        "certified_visits": [int(group["certified_visits"]) for group in seed_groups],
    }
    for weighting in ("state_weighted", "task_balanced"):
        seed_values = [group[weighting].get("mean_bits") for group in seed_groups]
        available = all(value is not None for value in seed_values)
        conditional = [
            group[weighting].get("certified_only_mean_bits") for group in seed_groups
        ]
        result[weighting] = {
            "headline_available": available,
            "seed_mean_bits": seed_values,
            "mean_bits": (
                statistics.fmean(float(value) for value in seed_values)
                if available
                else None
            ),
            "sample_sd_bits": (
                statistics.stdev(float(value) for value in seed_values)
                if available
                else None
            ),
            "certified_only_seed_mean_bits": conditional,
        }
    depths = sorted(
        set.intersection(
            *(set(group.get("by_rollout_depth", {})) for group in seed_groups)
        ),
        key=int,
    )
    result["by_rollout_depth"] = {
        depth: _aggregate_h([group["by_rollout_depth"][depth] for group in seed_groups])
        for depth in depths
    }
    return result


def aggregate_evaluations(
    protocol_path: Path,
    evaluation_paths: Sequence[Path],
    *,
    output: Path,
) -> dict[str, Any]:
    """Require and aggregate the full architecture/seed/device product."""

    protocol = load_protocol(protocol_path)
    if output.exists():
        raise FileExistsError(f"Aggregate already exists: {output}")
    expected_cells = {
        (arm_id, seed, device)
        for arm_id in ARM_IDS
        for seed in TRAINING_SEEDS
        for device in DEVICES
    }
    loaded: dict[
        tuple[str, int, str],
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    timing_matrix_references: list[tuple[str, str]] = []
    for path in evaluation_paths:
        summary, rows, visit_rows = _load_evaluation(path)
        cell_mapping = summary.get("cell", {})
        cell = (
            str(cell_mapping.get("arm_id")),
            cell_mapping.get("training_seed"),
            str(cell_mapping.get("device")),
        )
        if cell not in expected_cells:
            raise ValueError(f"Unexpected evaluation cell {cell}")
        if cell in loaded:
            raise ValueError(f"Duplicate evaluation cell {cell}")
        if summary.get("protocol_sha256") != protocol[PROTOCOL_DIGEST_FIELD]:
            raise ValueError(f"Evaluation {cell} used another protocol")
        timing_matrix_path = summary.get("timing_matrix_path")
        timing_matrix_digest = summary.get("timing_matrix_sha256")
        if not isinstance(timing_matrix_path, str) or not isinstance(
            timing_matrix_digest, str
        ):
            raise ValueError(f"Evaluation {cell} has no timing-matrix identity")
        timing_matrix_references.append(
            (str(Path(timing_matrix_path).resolve()), timing_matrix_digest)
        )
        if summary.get("evaluation_host") != protocol["evaluation_host"]:
            raise ValueError(f"Evaluation {cell} used another host")
        if summary.get("cohort_sha256") != protocol["cohort"]["cohort_sha256"]:
            raise ValueError(f"Evaluation {cell} used another cohort")
        if (
            summary.get("evaluation_task_order_sha256")
            != protocol["evaluation_task_order"]["task_ids_sha256"]
        ):
            raise ValueError(f"Evaluation {cell} used another task order")
        if summary.get("corpus") != protocol["corpus"]:
            raise ValueError(f"Evaluation {cell} used another corpus identity")
        if summary.get("search") != protocol["search"]:
            raise ValueError(f"Evaluation {cell} used another search protocol")
        if summary.get("cell_ordinal") != _cell_ordinal(
            protocol,
            {
                "arm_id": cell[0],
                "training_seed": cell[1],
                "device": cell[2],
            },
        ):
            raise ValueError(f"Evaluation {cell} has the wrong frozen ordinal")
        timing_window = summary.get("timing_window")
        if not isinstance(timing_window, Mapping):
            raise ValueError(f"Evaluation {cell} has no timing window")
        try:
            started_at = datetime.fromisoformat(str(timing_window["started_at_utc"]))
            completed_at = datetime.fromisoformat(
                str(timing_window["completed_at_utc"])
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Evaluation {cell} has an invalid timing window"
            ) from error
        if (
            started_at.tzinfo is None
            or completed_at.tzinfo is None
            or completed_at < started_at
            or not isinstance(timing_window.get("environment_before"), Mapping)
            or not isinstance(timing_window.get("environment_after"), Mapping)
        ):
            raise ValueError(f"Evaluation {cell} has an invalid timing window")
        expected_arm = _arm_by_id(cell[0])
        if summary.get("arm") != expected_arm:
            raise ValueError(f"Evaluation {cell} used another architecture")
        _validate_task_rows(
            rows,
            cell={
                "arm_id": cell[0],
                "training_seed": cell[1],
                "device": cell[2],
            },
            task_order=protocol["evaluation_task_order"],
            cohort=protocol["cohort"],
        )
        _validate_visit_rows(
            visit_rows,
            task_rows=rows,
            cell={
                "arm_id": cell[0],
                "training_seed": cell[1],
                "device": cell[2],
            },
        )
        recomputed = _distance_metrics(rows)
        if recomputed != summary.get("metrics"):
            raise ValueError(f"Evaluation {cell} metrics differ from raw rows")
        recomputed_h = _h_metrics(visit_rows, rows)
        if recomputed_h != summary.get("h_metrics"):
            raise ValueError(f"Evaluation {cell} H metrics differ from raw visits")
        sampling = summary.get("visit_sampling", {})
        if sampling.get("total_visits") != sum(
            int(row["scored_visits"]) for row in rows
        ) or sampling.get("sampled_visits") != len(visit_rows):
            raise ValueError(
                f"Evaluation {cell} visit totals differ from raw artifacts"
            )
        loaded[cell] = (summary, rows, visit_rows)
    missing = sorted(expected_cells - set(loaded))
    if missing:
        raise ValueError(f"Evaluation matrix is incomplete; missing={missing}")
    if len(loaded) != len(expected_cells):
        raise ValueError("Evaluation matrix is not the exact Cartesian product")
    if len(set(timing_matrix_references)) != 1:
        raise ValueError("Evaluation cells used different timing matrices")
    timing_matrix_path, timing_matrix_digest = timing_matrix_references[0]
    timing_matrix = load_timing_matrix(Path(timing_matrix_path), protocol=protocol)
    if timing_matrix[TIMING_MATRIX_DIGEST_FIELD] != timing_matrix_digest:
        raise ValueError("Evaluation timing-matrix digest differs from its artifact")
    matrix_by_cell = {
        (
            str(entry["cell"]["arm_id"]),
            int(entry["cell"]["training_seed"]),
            str(entry["cell"]["device"]),
        ): entry
        for entry in timing_matrix["timings"]
    }
    for cell, (summary, final_rows, _) in loaded.items():
        entry = matrix_by_cell.get(cell)
        if (
            entry is None
            or summary.get("timing_sha256") != entry.get("timing_sha256")
            or Path(str(summary.get("timing_summary_path"))).resolve()
            != Path(str(entry.get("summary_path"))).resolve()
        ):
            raise ValueError(f"Evaluation {cell} used a timing outside the matrix")
        timing_summary, timed_rows = _load_timing(Path(str(entry["summary_path"])))
        if timing_summary[TIMING_DIGEST_FIELD] != entry["timing_sha256"]:
            raise ValueError(f"Timing matrix has the wrong digest for {cell}")
        if [_timed_row_core(row) for row in final_rows] != [
            _timed_row_core(row) for row in timed_rows
        ]:
            raise ValueError(f"Evaluation {cell} changed its sealed timing rows")
    previous_completed_at: datetime | None = None
    for summary, _, _ in sorted(
        loaded.values(),
        key=lambda loaded_row: int(loaded_row[0]["cell_ordinal"]),
    ):
        timing_window = summary["timing_window"]
        started_at = datetime.fromisoformat(timing_window["started_at_utc"])
        completed_at = datetime.fromisoformat(timing_window["completed_at_utc"])
        if previous_completed_at is not None and started_at < previous_completed_at:
            raise ValueError(
                "Evaluation timing windows do not follow the frozen cell order"
            )
        previous_completed_at = completed_at
    barrier = datetime.fromisoformat(timing_matrix["sealed_at_utc"])
    annotation_windows = []
    for summary, _, _ in loaded.values():
        window = summary.get("annotation_window")
        if not isinstance(window, Mapping):
            raise ValueError("Evaluation has no annotation window")
        try:
            started = datetime.fromisoformat(str(window["started_at_utc"]))
            completed = datetime.fromisoformat(str(window["completed_at_utc"]))
        except (KeyError, ValueError) as error:
            raise ValueError("Evaluation has an invalid annotation window") from error
        if (
            started.tzinfo is None
            or completed.tzinfo is None
            or started < barrier
            or completed < started
        ):
            raise ValueError("Evaluation annotation crossed the timing barrier")
        annotation_windows.append((started, completed))
    previous_annotation_completed: datetime | None = None
    for started, completed in sorted(annotation_windows):
        if (
            previous_annotation_completed is not None
            and started < previous_annotation_completed
        ):
            raise ValueError("Evaluation annotation windows overlap")
        previous_annotation_completed = completed

    checkpoint_inputs = []
    training_sources = []
    training_environments = []
    for arm_id in ARM_IDS:
        for seed in TRAINING_SEEDS:
            cpu_checkpoint = loaded[(arm_id, seed, "cpu")][0]["checkpoint"]
            cuda_checkpoint = loaded[(arm_id, seed, "cuda")][0]["checkpoint"]
            if cpu_checkpoint.get("artifacts_sha256") != cuda_checkpoint.get(
                "artifacts_sha256"
            ):
                raise ValueError(
                    f"CPU/CUDA cells used different checkpoints for {(arm_id, seed)}"
                )
            source_code = cpu_checkpoint.get("source_code")
            training_sources.append(
                {
                    "arm_id": arm_id,
                    "training_seed": seed,
                    "source_code": _canonical_copy(source_code),
                    "protocol_comparison": _canonical_copy(
                        cpu_checkpoint.get("source_protocol_comparison")
                    ),
                }
            )
            environment = cpu_checkpoint.get("training_environment")
            training_environments.append(
                {
                    "arm_id": arm_id,
                    "training_seed": seed,
                    "environment": _canonical_copy(environment),
                }
            )
            checkpoint_inputs.append(
                {
                    "arm_id": arm_id,
                    "training_seed": seed,
                    "parameters": cpu_checkpoint.get("parameters"),
                    "artifacts_sha256": cpu_checkpoint.get("artifacts_sha256"),
                    "training_runtime": cpu_checkpoint.get("training_runtime"),
                    "source_protocol_comparison": cpu_checkpoint.get(
                        "source_protocol_comparison"
                    ),
                }
            )
    source_identities = {
        _canonical_json(row["source_code"]) for row in training_sources
    }
    environment_identities = {
        _canonical_json(row["environment"]) for row in training_environments
    }

    evaluation_inputs = [
        {
            "cell": {
                "arm_id": cell[0],
                "training_seed": cell[1],
                "device": cell[2],
            },
            "evaluation_sha256": summary[EVALUATION_DIGEST_FIELD],
            "timing_sha256": summary[TIMING_DIGEST_FIELD],
            "timing_matrix_sha256": summary[TIMING_MATRIX_DIGEST_FIELD],
            "task_rows_sha256": summary["artifacts"]["tasks"]["sha256"],
            "visit_rows_sha256": summary["artifacts"]["visits"]["sha256"],
        }
        for cell, (summary, _, _) in sorted(loaded.items())
    ]

    by_arm: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        by_device = {}
        for device in DEVICES:
            by_distance = {}
            for distance in CERTIFIED_DISTANCES:
                seeds = []
                for seed in TRAINING_SEEDS:
                    evaluation = loaded[(arm_id, seed, device)][0]
                    metrics = evaluation["metrics"][str(distance)]
                    seeds.append(
                        {
                            "training_seed": seed,
                            "tasks": metrics["tasks"],
                            "solved": metrics["solved"],
                            "solve_rate": metrics["solve_rate"],
                            "wall_clock_auc": metrics["wall_clock_auc"],
                            "execution_auc": metrics["execution_auc"],
                            "mean_terminal_seconds": metrics["mean_terminal_seconds"],
                            "mean_terminal_executions": metrics[
                                "mean_terminal_executions"
                            ],
                            "h": evaluation["h_metrics"]["by_certified_distance"][
                                str(distance)
                            ],
                        }
                    )
                by_distance[str(distance)] = {
                    "seeds": seeds,
                    "wall_clock_auc": _auc_summary(seeds, "wall_clock_auc"),
                    "execution_auc": _auc_summary(seeds, "execution_auc"),
                    "solve_rate": {
                        "mean": statistics.fmean(row["solve_rate"] for row in seeds),
                        "sample_sd": statistics.stdev(
                            row["solve_rate"] for row in seeds
                        ),
                    },
                    "h": _aggregate_h([row["h"] for row in seeds]),
                }
            by_device[device] = {"by_certified_distance": by_distance}
        by_arm[arm_id] = {
            "arm": _arm_by_id(arm_id),
            "by_device": by_device,
        }
    paired_differences = {}
    for distance in CERTIFIED_DISTANCES:
        by_device = {}
        for device in DEVICES:
            comparisons = {}
            for left_index, left_arm in enumerate(ARM_IDS):
                for right_arm in ARM_IDS[left_index + 1 :]:
                    left_rows = by_arm[left_arm]["by_device"][device][
                        "by_certified_distance"
                    ][str(distance)]["seeds"]
                    right_rows = by_arm[right_arm]["by_device"][device][
                        "by_certified_distance"
                    ][str(distance)]["seeds"]
                    comparisons[f"{left_arm}-minus-{right_arm}"] = {
                        "left_arm_id": left_arm,
                        "right_arm_id": right_arm,
                        "wall_clock_auc": _paired_auc_difference(
                            left_rows,
                            right_rows,
                            "wall_clock_auc",
                        ),
                        "execution_auc": _paired_auc_difference(
                            left_rows,
                            right_rows,
                            "execution_auc",
                        ),
                    }
            by_device[device] = comparisons
        paired_differences[str(distance)] = by_device
    paired_device_differences = {}
    for distance in CERTIFIED_DISTANCES:
        by_arm_device = {}
        for arm_id in ARM_IDS:
            cpu_rows = by_arm[arm_id]["by_device"]["cpu"]["by_certified_distance"][
                str(distance)
            ]["seeds"]
            cuda_rows = by_arm[arm_id]["by_device"]["cuda"]["by_certified_distance"][
                str(distance)
            ]["seeds"]
            by_arm_device[arm_id] = {
                "direction": "cpu-minus-cuda",
                "wall_clock_auc": _paired_auc_difference(
                    cpu_rows,
                    cuda_rows,
                    "wall_clock_auc",
                ),
                "execution_auc": _paired_auc_difference(
                    cpu_rows,
                    cuda_rows,
                    "execution_auc",
                ),
            }
        paired_device_differences[str(distance)] = by_arm_device

    device_parity_rows = []
    for arm_id in ARM_IDS:
        for seed in TRAINING_SEEDS:
            cpu_tasks = loaded[(arm_id, seed, "cpu")][1]
            cuda_tasks = loaded[(arm_id, seed, "cuda")][1]
            mismatches = []
            for cpu_row, cuda_row in zip(cpu_tasks, cuda_tasks, strict=True):
                if cpu_row["task_id"] != cuda_row["task_id"]:
                    raise ValueError("CPU/CUDA task rows use different frozen order")
                fields = (
                    "success",
                    "termination",
                    "solution_function_names",
                    "executions",
                )
                differences = {
                    field: {"cpu": cpu_row[field], "cuda": cuda_row[field]}
                    for field in fields
                    if cpu_row[field] != cuda_row[field]
                }
                if differences:
                    mismatches.append(
                        {
                            "task_id": cpu_row["task_id"],
                            "certified_distance": cpu_row["certified_distance"],
                            "differences": differences,
                        }
                    )
            device_parity_rows.append(
                {
                    "arm_id": arm_id,
                    "training_seed": seed,
                    "tasks": len(cpu_tasks),
                    "mismatched_tasks": len(mismatches),
                    "mismatches": mismatches,
                }
            )
    device_parity = {
        "fields": [
            "success",
            "termination",
            "solution_function_names",
            "executions",
        ],
        "all_exact": all(not row["mismatched_tasks"] for row in device_parity_rows),
        "by_arm_seed": device_parity_rows,
    }
    selection = {}
    for distance in CERTIFIED_DISTANCES:
        by_device = {}
        for device in DEVICES:
            wall_scores = {
                arm_id: by_arm[arm_id]["by_device"][device]["by_certified_distance"][
                    str(distance)
                ]["wall_clock_auc"]["normalized_mean"]
                for arm_id in ARM_IDS
            }
            execution_scores = {
                arm_id: by_arm[arm_id]["by_device"][device]["by_certified_distance"][
                    str(distance)
                ]["execution_auc"]["normalized_mean"]
                for arm_id in ARM_IDS
            }
            best_wall = max(
                ARM_IDS,
                key=lambda arm_id: (wall_scores[arm_id], -ARM_IDS.index(arm_id)),
            )
            best_execution = max(
                ARM_IDS,
                key=lambda arm_id: (
                    execution_scores[arm_id],
                    -ARM_IDS.index(arm_id),
                ),
            )
            by_device[device] = {
                "best_wall_clock_arm": {
                    "arm_id": best_wall,
                    "seed_mean_normalized_auc": wall_scores[best_wall],
                },
                "best_execution_arm": {
                    "arm_id": best_execution,
                    "seed_mean_normalized_auc": execution_scores[best_execution],
                },
            }
        global_candidates = [
            (
                arm_id,
                device,
                by_arm[arm_id]["by_device"][device]["by_certified_distance"][
                    str(distance)
                ]["wall_clock_auc"]["normalized_mean"],
            )
            for arm_id in ARM_IDS
            for device in DEVICES
        ]
        global_best = max(
            global_candidates,
            key=lambda row: (
                row[2],
                -ARM_IDS.index(row[0]),
                -DEVICES.index(row[1]),
            ),
        )
        selection[str(distance)] = {
            "by_device": by_device,
            "global_best_wall_clock": {
                "arm_id": global_best[0],
                "device": global_best[1],
                "seed_mean_normalized_auc": global_best[2],
            },
        }
    result = _seal(
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "status": AGGREGATE_STATUS,
            "protocol_sha256": protocol[PROTOCOL_DIGEST_FIELD],
            "timing_matrix_sha256": timing_matrix_digest,
            "evaluation_host": protocol["evaluation_host"],
            "cohort_sha256": protocol["cohort"]["cohort_sha256"],
            "evaluation_task_order": protocol["evaluation_task_order"],
            "source": protocol["source"],
            "cells": len(loaded),
            "expected_cells": len(expected_cells),
            "training_seeds": list(TRAINING_SEEDS),
            "devices": list(DEVICES),
            "inputs": {
                "timing_matrix": {
                    "path": timing_matrix_path,
                    "timing_matrix_sha256": timing_matrix_digest,
                    "timings": timing_matrix["timings"],
                },
                "training_sources": {
                    "all_identical": len(source_identities) == 1,
                    "by_checkpoint": training_sources,
                },
                "training_environments": {
                    "all_identical": len(environment_identities) == 1,
                    "by_checkpoint": training_environments,
                },
                "checkpoints": checkpoint_inputs,
                "evaluations": evaluation_inputs,
            },
            "by_arm": by_arm,
            "paired_arm_differences_by_certified_distance": paired_differences,
            "paired_device_differences_by_certified_distance": (
                paired_device_differences
            ),
            "cpu_cuda_task_parity": device_parity,
            "selection_by_certified_distance": selection,
        },
        AGGREGATE_DIGEST_FIELD,
    )
    _write_json(output, result, exclusive=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    freeze.add_argument("--output", type=Path, required=True)

    def add_cell_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--arm", choices=ARM_IDS, required=True)
        command.add_argument("--seed", choices=TRAINING_SEEDS, type=int, required=True)
        command.add_argument("--device", choices=DEVICES, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
        command.add_argument("--quiet", action="store_true")

    timing = commands.add_parser("time")
    add_cell_arguments(timing)

    seal_timings = commands.add_parser("seal-timings")
    seal_timings.add_argument("--protocol", type=Path, required=True)
    seal_timings.add_argument(
        "--timing",
        action="append",
        type=Path,
        required=True,
        help=(
            "timing_summary.json; repeat exactly "
            f"{EVALUATION_CELL_COUNT} times in any argument order"
        ),
    )
    seal_timings.add_argument("--output", type=Path, required=True)

    annotate = commands.add_parser("annotate")
    add_cell_arguments(annotate)
    annotate.add_argument("--timing-summary", type=Path, required=True)
    annotate.add_argument("--timing-matrix", type=Path, required=True)
    annotate.add_argument(
        "--oracle-cache",
        type=Path,
        help="shared cache; defaults to oracle_labels.json beside the protocol",
    )

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument(
        "--evaluation",
        action="append",
        type=Path,
        required=True,
        help=(f"evaluation summary.json; repeat exactly {EVALUATION_CELL_COUNT} times"),
    )
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        result = freeze_protocol(arguments.corpus_dir, output=arguments.output)
    elif arguments.command in {"time", "annotate"}:
        result = evaluate_cell(
            EvaluationConfig(
                protocol=arguments.protocol,
                checkpoint=arguments.checkpoint,
                arm_id=arguments.arm,
                training_seed=arguments.seed,
                device=arguments.device,
                output_dir=arguments.output_dir,
                corpus_dir=arguments.corpus_dir,
                oracle_cache=(
                    arguments.oracle_cache if arguments.command == "annotate" else None
                ),
                timing_summary=(
                    arguments.timing_summary
                    if arguments.command == "annotate"
                    else None
                ),
                timing_matrix=(
                    arguments.timing_matrix if arguments.command == "annotate" else None
                ),
                progress=not arguments.quiet,
            )
        )
    elif arguments.command == "seal-timings":
        result = seal_timing_matrix(
            arguments.protocol,
            arguments.timing,
            output=arguments.output,
        )
    else:
        result = aggregate_evaluations(
            arguments.protocol,
            arguments.evaluation,
            output=arguments.output,
        )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
