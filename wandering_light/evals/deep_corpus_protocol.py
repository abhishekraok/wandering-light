"""Frozen evaluation protocol primitives for the deep-corpus-v2 scratch study.

The module intentionally has no model, CUDA, Hub, or W&B dependency.  It owns
the immutable identities and state transitions that must exist before training
or held-out scoring code is allowed to run:

* deterministic, balanced validation and test cohorts;
* content-addressed generation, tokenizer, evaluator, and BFS settings;
* validation-selected identities for an ordered subset of the predeclared arms;
* candidate-level parse/execution funnels over nested sampling prefixes; and
* a once-only test-opening ledger whose interrupted run can be resumed without
  creating a second test opening.

All hashes use canonical JSON and include a ``sha256:`` prefix.  Builders return
plain JSON-compatible dictionaries so the resulting locks can be stored on the
Hub or as W&B artifacts without importing this package to inspect them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord


PROTOCOL_SCHEMA_VERSION = 1
STUDY_ID = "deep-corpus-v2-scratch-scaling"
CERTIFIED_DISTANCES = tuple(range(1, 9))
TASKS_PER_DISTANCE = 12
COHORT_SIZE = len(CERTIFIED_DISTANCES) * TASKS_PER_DISTANCE
SAMPLE_PREFIXES = (1, 8, 64)

ARM_IDS = (
    "scratch-10m",
    "opt-125m-scratch",
    "smollm2-360m-scratch",
)
ALLOWED_ARM_IDS = frozenset(ARM_IDS)

PARSE_FAILURE_CATEGORIES = (
    "empty",
    "unknown_function",
    "malformed_separator",
    "truncation",
    "other",
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_COHORT_SELECTION = "sha256-rank-v1"
_PROTOCOL_STATUS = "frozen-before-training"
_SELECTION_STATUS = "validation-selection-locked"
_LEDGER_STATUSES = frozenset({"prepared", "in_progress", "interrupted", "complete"})
_ARM_LEDGER_STATUSES = frozenset({"pending", "complete"})

SOURCE_AMENDMENT_FAILURE_EVIDENCE = {
    "exception": {
        "type": "ValueError",
        "unused_model_kwargs": ["token_type_ids"],
    },
    "failure_boundary": "before-first-model-forward",
    "generated_candidates": 0,
    "model_forward_calls": 0,
    "result_artifact_absent": True,
    "scored_tasks": 0,
    "split": "validation",
    "stage": "candidate-generation-before-artifact-write",
    "test_access_count": 0,
}

CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE = {
    "split": "validation",
    "stage": "validation-summary-build",
    "failure_boundary": (
        "after-candidate-generation-and-bfs-before-atomic-result-publication"
    ),
    "exception": {
        "type": "ValueError",
        "reason": "non-exhaustive-bfs-budget-termination",
    },
    "candidate_generation_complete": True,
    "candidate_tasks": 96,
    "generated_candidates": 6_144,
    "bfs_complete": True,
    "bfs_rows": 96,
    "bfs_budget_terminations": [
        {
            "task_id": (
                "5ce61393fdc9d8461d00f7094a38ce145eb4ca2ad27079c07b7eda5954b7e46a"
            ),
            "certified_distance": 8,
            "termination": "budget",
            "executions": 20_000_000,
        }
    ],
    "result_artifact_absent": True,
    "selection_lock_artifact_absent": True,
    "wandb_receipt_artifact_absent": True,
    "test_access_count": 0,
    "test_opening_ledger_absent": True,
}


def canonical_json(value: Any) -> str:
    """Serialize one JSON value canonically, rejecting NaN and non-JSON data."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON data: {error}") from error


def canonical_copy(value: Any) -> Any:
    """Return a detached JSON-normalized copy of ``value``."""

    return json.loads(canonical_json(value))


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_active_arm_ids(active_arm_ids: Sequence[str]) -> tuple[str, ...]:
    """Return a nonempty, canonical-order subset of the frozen arm IDs."""

    if isinstance(active_arm_ids, (str, bytes)):
        raise ValueError("active arm IDs must be a sequence, not a string")
    normalized = tuple(active_arm_ids)
    if not normalized:
        raise ValueError("at least one active arm is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("active arm IDs must be distinct")
    if any(arm_id not in ALLOWED_ARM_IDS for arm_id in normalized):
        raise ValueError("active arm IDs contain an unexpected arm")
    canonical = tuple(arm_id for arm_id in ARM_IDS if arm_id in normalized)
    if normalized != canonical:
        raise ValueError("active arm IDs must follow the frozen arm order")
    return normalized


def _deferred_arm_ids(active_arm_ids: Sequence[str]) -> tuple[str, ...]:
    active = validate_active_arm_ids(active_arm_ids)
    return tuple(arm_id for arm_id in ARM_IDS if arm_id not in active)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256:<hex> digest")
    return value


def _seal(body: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    normalized = canonical_copy(dict(body))
    normalized[digest_field] = sha256_json(normalized)
    return normalized


def _verify_seal(
    value: Mapping[str, Any],
    digest_field: str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized = canonical_copy(dict(value))
    recorded = _require_sha256(normalized.pop(digest_field, None), label=digest_field)
    actual = sha256_json(normalized)
    if recorded != actual:
        raise ValueError(f"{label} content changed after it was frozen")
    if expected_sha256 is not None and recorded != _require_sha256(
        expected_sha256, label=f"expected {label} digest"
    ):
        raise ValueError(f"{label} does not match its expected digest")
    normalized[digest_field] = recorded
    return normalized


def file_identity(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Identify a regular file by portable root-relative path, size, and hash."""

    root_path = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root_path)
    except ValueError as error:
        raise ValueError(
            f"identity file is outside root {root_path}: {resolved}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f"identity file is missing: {resolved}")
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _build_file_set_identity(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    metadata: Mapping[str, Any] | None,
    digest_field: str,
) -> dict[str, Any]:
    identities = sorted(
        (file_identity(path, root=root) for path in paths),
        key=lambda row: row["path"],
    )
    if not identities:
        raise ValueError("at least one identity file is required")
    names = [row["path"] for row in identities]
    if len(set(names)) != len(names):
        raise ValueError("identity file paths must be unique")
    return _seal(
        {
            "files": identities,
            "metadata": canonical_copy(dict(metadata or {})),
        },
        digest_field,
    )


def _validate_file_set_identity(
    identity: Mapping[str, Any],
    *,
    digest_field: str,
    label: str,
    root: str | Path | None,
) -> dict[str, Any]:
    normalized = _verify_seal(identity, digest_field, label=label)
    files = normalized.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} must contain at least one file")
    paths: list[str] = []
    for row in files:
        if not isinstance(row, dict):
            raise ValueError(f"{label} file identity is malformed")
        path = row.get("path")
        byte_count = row.get("bytes")
        digest = row.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or Path(path).parts[0] == ".."
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError(f"{label} file identity is malformed")
        _require_sha256(digest, label=f"{label} file digest")
        paths.append(path)
        if root is not None:
            actual = Path(root).resolve() / path
            if not actual.is_file():
                raise ValueError(f"{label} file is missing: {path}")
            if actual.stat().st_size != byte_count or sha256_file(actual) != digest:
                raise ValueError(f"{label} file changed after freezing: {path}")
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError(f"{label} files must be unique and sorted")
    if not isinstance(normalized.get("metadata"), dict):
        raise ValueError(f"{label} metadata must be a mapping")
    return normalized


def build_tokenizer_identity(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _build_file_set_identity(
        paths,
        root=root,
        metadata=metadata,
        digest_field="tokenizer_sha256",
    )


def validate_tokenizer_identity(
    identity: Mapping[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    return _validate_file_set_identity(
        identity,
        digest_field="tokenizer_sha256",
        label="tokenizer identity",
        root=root,
    )


def build_evaluator_source_identity(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _build_file_set_identity(
        paths,
        root=root,
        metadata=metadata,
        digest_field="source_sha256",
    )


def validate_evaluator_source_identity(
    identity: Mapping[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    return _validate_file_set_identity(
        identity,
        digest_field="source_sha256",
        label="evaluator source identity",
        root=root,
    )


def _normalize_changed_source_allowlist(paths: Sequence[str]) -> list[str]:
    if isinstance(paths, (str, bytes)) or not paths:
        raise ValueError("changed-source allowlist must be a nonempty sequence")
    normalized: list[str] = []
    for value in paths:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("changed-source allowlist contains an unsafe path")
        path = Path(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("changed-source allowlist contains an unsafe path")
        normalized.append(value)
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("changed-source allowlist must be unique and sorted")
    return normalized


def _evaluator_files_by_path(identity: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["path"]: row for row in identity["files"]}


def _validate_evaluator_source_transition(
    predecessor_sources: Mapping[str, Any],
    successor_sources: Mapping[str, Any],
    *,
    changed_source_allowlist: Sequence[str],
) -> None:
    predecessor_metadata = predecessor_sources.get("metadata", {})
    successor_metadata = successor_sources.get("metadata", {})
    predecessor_commit = predecessor_metadata.get("git_commit")
    successor_commit = successor_metadata.get("git_commit")
    if (
        not isinstance(predecessor_commit, str)
        or _GIT_COMMIT_RE.fullmatch(predecessor_commit) is None
        or not isinstance(successor_commit, str)
        or _GIT_COMMIT_RE.fullmatch(successor_commit) is None
        or predecessor_commit == successor_commit
    ):
        raise ValueError(
            "protocol source amendment requires distinct predecessor/successor commits"
        )
    if {
        key: value for key, value in predecessor_metadata.items() if key != "git_commit"
    } != {
        key: value for key, value in successor_metadata.items() if key != "git_commit"
    }:
        raise ValueError("protocol source amendment changed evaluator-source metadata")

    predecessor_files = _evaluator_files_by_path(predecessor_sources)
    successor_files = _evaluator_files_by_path(successor_sources)
    if set(predecessor_files) != set(successor_files):
        raise ValueError("protocol source amendment changed the evaluator file set")
    changed_evaluator_files = {
        path
        for path in predecessor_files
        if predecessor_files[path] != successor_files[path]
    }
    if not changed_evaluator_files:
        raise ValueError("protocol source amendment did not change evaluator sources")
    if not changed_evaluator_files.issubset(changed_source_allowlist):
        raise ValueError(
            "changed evaluator source is absent from the amendment allowlist"
        )


def validate_source_amendment(
    amendment: Mapping[str, Any],
    *,
    successor_evaluator_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a post-training, pre-test evaluator-source amendment."""

    normalized = _verify_seal(
        amendment,
        "source_amendment_sha256",
        label="protocol source amendment",
    )
    schema_version = normalized.get("schema_version")
    common_fields = {
        "schema_version",
        "predecessor_protocol_lock_sha256",
        "training_evaluator_sources",
        "reason",
        "changed_source_allowlist",
        "failure_evidence",
        "source_amendment_sha256",
    }
    if schema_version == 1:
        expected_fields = common_fields
    elif schema_version == 2:
        expected_fields = {
            *common_fields,
            "predecessor_evaluator_sources",
            "predecessor_source_amendment",
        }
    else:
        raise ValueError("unsupported protocol source-amendment schema")
    if set(normalized) != expected_fields:
        raise ValueError("protocol source amendment has unexpected fields")
    _require_sha256(
        normalized.get("predecessor_protocol_lock_sha256"),
        label="predecessor protocol lock digest",
    )
    reason = normalized.get("reason")
    if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
        raise ValueError(
            "protocol source-amendment reason must be nonempty and trimmed"
        )
    allowlist = _normalize_changed_source_allowlist(
        normalized.get("changed_source_allowlist", [])
    )
    training_sources = validate_evaluator_source_identity(
        normalized.get("training_evaluator_sources", {})
    )
    successor_sources = validate_evaluator_source_identity(successor_evaluator_sources)
    if schema_version == 1:
        if normalized.get("failure_evidence") != SOURCE_AMENDMENT_FAILURE_EVIDENCE:
            raise ValueError(
                "protocol source amendment requires zero-output pre-forward failure "
                "evidence"
            )
        predecessor_sources = training_sources
    else:
        if (
            normalized.get("failure_evidence")
            != CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE
        ):
            raise ValueError(
                "chained protocol source amendment requires exact completed-work "
                "pre-test failure evidence"
            )
        predecessor_sources = validate_evaluator_source_identity(
            normalized.get("predecessor_evaluator_sources", {})
        )
        predecessor_amendment = validate_source_amendment(
            normalized.get("predecessor_source_amendment", {}),
            successor_evaluator_sources=predecessor_sources,
        )
        if predecessor_amendment["schema_version"] != 1:
            raise ValueError(
                "v2 protocol source amendment requires the pinned v1 predecessor"
            )
        if predecessor_amendment["training_evaluator_sources"] != training_sources:
            raise ValueError(
                "chained protocol source amendment changed training-source lineage"
            )

    _validate_evaluator_source_transition(
        predecessor_sources,
        successor_sources,
        changed_source_allowlist=allowlist,
    )
    return normalized


def build_source_amendment(
    *,
    predecessor_protocol_lock_sha256: str,
    training_evaluator_sources: Mapping[str, Any],
    successor_evaluator_sources: Mapping[str, Any],
    reason: str,
    changed_source_allowlist: Sequence[str],
    failure_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    amendment = _seal(
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "predecessor_protocol_lock_sha256": predecessor_protocol_lock_sha256,
            "training_evaluator_sources": canonical_copy(
                dict(training_evaluator_sources)
            ),
            "reason": reason,
            "changed_source_allowlist": list(changed_source_allowlist),
            "failure_evidence": canonical_copy(dict(failure_evidence)),
        },
        "source_amendment_sha256",
    )
    return validate_source_amendment(
        amendment,
        successor_evaluator_sources=successor_evaluator_sources,
    )


def build_chained_source_amendment(
    *,
    predecessor_protocol_lock_sha256: str,
    predecessor_evaluator_sources: Mapping[str, Any],
    predecessor_source_amendment: Mapping[str, Any],
    successor_evaluator_sources: Mapping[str, Any],
    reason: str,
    changed_source_allowlist: Sequence[str],
    failure_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a v2 source-only amendment around an already amended predecessor."""

    normalized_predecessor_sources = validate_evaluator_source_identity(
        predecessor_evaluator_sources
    )
    normalized_predecessor_amendment = validate_source_amendment(
        predecessor_source_amendment,
        successor_evaluator_sources=normalized_predecessor_sources,
    )
    amendment = _seal(
        {
            "schema_version": 2,
            "predecessor_protocol_lock_sha256": predecessor_protocol_lock_sha256,
            "training_evaluator_sources": canonical_copy(
                normalized_predecessor_amendment["training_evaluator_sources"]
            ),
            "predecessor_evaluator_sources": canonical_copy(
                normalized_predecessor_sources
            ),
            "predecessor_source_amendment": canonical_copy(
                normalized_predecessor_amendment
            ),
            "reason": reason,
            "changed_source_allowlist": list(changed_source_allowlist),
            "failure_evidence": canonical_copy(dict(failure_evidence)),
        },
        "source_amendment_sha256",
    )
    return validate_source_amendment(
        amendment,
        successor_evaluator_sources=successor_evaluator_sources,
    )


def protocol_training_evaluator_sources(
    protocol_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the sources that produced training and feasibility artifacts."""

    successor_sources = validate_evaluator_source_identity(
        protocol_lock.get("evaluator_sources", {})
    )
    amendment = protocol_lock.get("source_amendment")
    if amendment is None:
        return successor_sources
    normalized = validate_source_amendment(
        amendment,
        successor_evaluator_sources=successor_sources,
    )
    return canonical_copy(normalized["training_evaluator_sources"])


def build_generation_config(
    *,
    sampling_seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    max_sequence_length: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reserved = {
        "sampling_seed",
        "sample_prefixes",
        "samples_per_task",
        "temperature",
        "top_p",
        "max_new_tokens",
        "batch_size",
        "max_sequence_length",
        "do_sample",
    }
    extra_values = dict(extra or {})
    overlap = reserved.intersection(extra_values)
    if overlap:
        raise ValueError(f"generation extras override frozen fields: {sorted(overlap)}")
    value = _seal(
        {
            "sampling_seed": sampling_seed,
            "sample_prefixes": list(SAMPLE_PREFIXES),
            "samples_per_task": SAMPLE_PREFIXES[-1],
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "do_sample": True,
            **extra_values,
        },
        "generation_sha256",
    )
    return validate_generation_config(value)


def validate_generation_config(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _verify_seal(
        value, "generation_sha256", label="generation configuration"
    )
    seed = normalized.get("sampling_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sampling_seed must be a non-negative integer")
    if tuple(normalized.get("sample_prefixes", ())) != SAMPLE_PREFIXES:
        raise ValueError("generation must use nested sample prefixes 1, 8, and 64")
    if normalized.get("samples_per_task") != SAMPLE_PREFIXES[-1]:
        raise ValueError("samples_per_task must equal the largest sample prefix")
    temperature = normalized.get("temperature")
    top_p = normalized.get("top_p")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, int | float)
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
    ):
        raise ValueError("top_p must be finite and in (0, 1]")
    for field in ("max_new_tokens", "batch_size", "max_sequence_length"):
        item = normalized.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if normalized["max_new_tokens"] >= normalized["max_sequence_length"]:
        raise ValueError("max_new_tokens must be smaller than max_sequence_length")
    if normalized.get("do_sample") is not True:
        raise ValueError("the frozen generation protocol must sample")
    return normalized


def build_bfs_settings(
    *,
    budget: int,
    max_depth: int = CERTIFIED_DISTANCES[-1],
    cost_definition: str = (
        "one execution for every type-compatible basis function attempted, "
        "including attempts whose function raises"
    ),
    state_identity: str = "TypedList.search_key",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reserved = {"budget", "max_depth", "cost_definition", "state_identity"}
    extra_values = dict(extra or {})
    overlap = reserved.intersection(extra_values)
    if overlap:
        raise ValueError(f"BFS extras override frozen fields: {sorted(overlap)}")
    value = _seal(
        {
            "budget": budget,
            "max_depth": max_depth,
            "cost_definition": cost_definition,
            "state_identity": state_identity,
            **extra_values,
        },
        "bfs_sha256",
    )
    return validate_bfs_settings(value)


def validate_bfs_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _verify_seal(value, "bfs_sha256", label="BFS settings")
    budget = normalized.get("budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("BFS budget must be a positive integer")
    if normalized.get("max_depth") != CERTIFIED_DISTANCES[-1]:
        raise ValueError("matched BFS must search through certified distance 8")
    for field in ("cost_definition", "state_identity"):
        if not isinstance(normalized.get(field), str) or not normalized[field].strip():
            raise ValueError(f"BFS {field} must be a non-empty string")
    return normalized


def _record_value(record: BasisTaskRecord | Mapping[str, Any], field: str) -> Any:
    if isinstance(record, Mapping):
        if field not in record:
            raise ValueError(f"corpus record has no {field!r}")
        return record[field]
    try:
        return getattr(record, field)
    except AttributeError as error:
        raise ValueError(f"corpus record has no {field!r}") from error


def _record_metadata(record: BasisTaskRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    value = _record_value(record, "metadata")
    if not isinstance(value, Mapping):
        raise ValueError("corpus record metadata must be a mapping")
    return value


def _record_cohort_entry(
    record: BasisTaskRecord | Mapping[str, Any], *, split: str, seed: int
) -> dict[str, Any]:
    task_id = _record_value(record, "task_id")
    record_split = _record_value(record, "split")
    metadata = _record_metadata(record)
    distance = metadata.get("certified_distance")
    root_digest = metadata.get("root_digest")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("corpus task_id must be a non-empty string")
    if record_split != split:
        raise ValueError(f"task {task_id} is not a member of split {split!r}")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, int)
        or distance not in CERTIFIED_DISTANCES
    ):
        raise ValueError(f"task {task_id} has invalid certified distance {distance!r}")
    if not isinstance(root_digest, str) or not root_digest:
        raise ValueError(f"task {task_id} has no root digest")
    rank = sha256_json(
        {
            "selection": _COHORT_SELECTION,
            "seed": seed,
            "split": split,
            "certified_distance": distance,
            "task_id": task_id,
        }
    )
    return {
        "task_id": task_id,
        "split": split,
        "certified_distance": distance,
        "root_digest": root_digest,
        "selection_rank": rank,
    }


def freeze_cohort(
    records: Sequence[BasisTaskRecord | Mapping[str, Any]],
    *,
    split: str,
    selection_seed: int,
) -> dict[str, Any]:
    """Choose exactly 12 tasks per distance by a stable content-hash ranking."""

    if split not in {"validation", "test"}:
        raise ValueError("frozen cohorts are defined only for validation and test")
    if (
        isinstance(selection_seed, bool)
        or not isinstance(selection_seed, int)
        or selection_seed < 0
    ):
        raise ValueError("selection_seed must be a non-negative integer")

    entries = [
        _record_cohort_entry(record, split=split, seed=selection_seed)
        for record in records
        if _record_value(record, "split") == split
    ]
    task_ids = [entry["task_id"] for entry in entries]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"source split {split!r} contains duplicate task IDs")

    by_distance: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_distance[entry["certified_distance"]].append(entry)
    if set(by_distance) != set(CERTIFIED_DISTANCES):
        raise ValueError(
            f"source split {split!r} must cover certified distances 1 through 8"
        )

    selected: list[dict[str, Any]] = []
    for distance in CERTIFIED_DISTANCES:
        pool = sorted(
            by_distance[distance],
            key=lambda row: (row["selection_rank"], row["task_id"]),
        )
        if len(pool) < TASKS_PER_DISTANCE:
            raise ValueError(
                f"split {split!r} distance {distance} has {len(pool)} tasks; "
                f"{TASKS_PER_DISTANCE} are required"
            )
        selected.extend(pool[:TASKS_PER_DISTANCE])

    selected_ids = [row["task_id"] for row in selected]
    body = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "selection": _COHORT_SELECTION,
        "selection_seed": selection_seed,
        "split": split,
        "tasks": selected,
        "task_ids": selected_ids,
        "task_ids_sha256": sha256_json(selected_ids),
        "size": len(selected),
        "distance_histogram": {
            str(distance): sum(
                row["certified_distance"] == distance for row in selected
            )
            for distance in CERTIFIED_DISTANCES
        },
    }
    return _seal(body, "cohort_sha256")


def validate_frozen_cohort(
    cohort: Mapping[str, Any],
    *,
    records: Sequence[BasisTaskRecord | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = _verify_seal(cohort, "cohort_sha256", label="frozen cohort")
    if normalized.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported frozen cohort schema")
    if normalized.get("selection") != _COHORT_SELECTION:
        raise ValueError("unexpected cohort selection algorithm")
    split = normalized.get("split")
    seed = normalized.get("selection_seed")
    if split not in {"validation", "test"}:
        raise ValueError("frozen cohort has an invalid split")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("frozen cohort has an invalid selection seed")
    tasks = normalized.get("tasks")
    task_ids = normalized.get("task_ids")
    if not isinstance(tasks, list) or not isinstance(task_ids, list):
        raise ValueError("frozen cohort must contain tasks and task IDs")
    if len(tasks) != COHORT_SIZE or normalized.get("size") != COHORT_SIZE:
        raise ValueError(f"frozen cohort must contain exactly {COHORT_SIZE} tasks")
    if task_ids != [row.get("task_id") for row in tasks]:
        raise ValueError("frozen cohort task IDs do not match its task entries")
    if len(set(task_ids)) != COHORT_SIZE:
        raise ValueError("frozen cohort task IDs must be unique")
    if normalized.get("task_ids_sha256") != sha256_json(task_ids):
        raise ValueError("frozen cohort task-ID digest is invalid")

    histogram = dict.fromkeys((str(distance) for distance in CERTIFIED_DISTANCES), 0)
    previous_key: tuple[int, str, str] | None = None
    for row in tasks:
        if not isinstance(row, dict):
            raise ValueError("frozen cohort task entry is malformed")
        if row.get("split") != split:
            raise ValueError("frozen cohort contains a task from another split")
        distance = row.get("certified_distance")
        if (
            isinstance(distance, bool)
            or not isinstance(distance, int)
            or distance not in CERTIFIED_DISTANCES
        ):
            raise ValueError("frozen cohort contains an invalid distance")
        expected_rank = sha256_json(
            {
                "selection": _COHORT_SELECTION,
                "seed": seed,
                "split": split,
                "certified_distance": distance,
                "task_id": row.get("task_id"),
            }
        )
        if row.get("selection_rank") != expected_rank:
            raise ValueError("frozen cohort selection rank is invalid")
        if not isinstance(row.get("root_digest"), str) or not row["root_digest"]:
            raise ValueError("frozen cohort task has no root digest")
        key = (distance, expected_rank, row["task_id"])
        if previous_key is not None and key < previous_key:
            raise ValueError("frozen cohort tasks are not in deterministic order")
        previous_key = key
        histogram[str(distance)] += 1
    if histogram != dict(normalized.get("distance_histogram", {})) or any(
        count != TASKS_PER_DISTANCE for count in histogram.values()
    ):
        raise ValueError("frozen cohort is not balanced at 12 tasks per distance")

    if records is not None:
        rebuilt = freeze_cohort(records, split=split, selection_seed=seed)
        if rebuilt != normalized:
            raise ValueError(
                "frozen cohort does not match deterministic source membership"
            )
    return normalized


def build_frozen_cohorts(
    records: Sequence[BasisTaskRecord | Mapping[str, Any]], *, selection_seed: int
) -> dict[str, Any]:
    records = list(records)
    relevant_ids: list[str] = []
    for record in records:
        if _record_value(record, "split") in {"validation", "test"}:
            relevant_ids.append(_record_value(record, "task_id"))
    if len(set(relevant_ids)) != len(relevant_ids):
        raise ValueError(
            "validation and test source task IDs must be globally disjoint"
        )
    validation = freeze_cohort(
        records, split="validation", selection_seed=selection_seed
    )
    test = freeze_cohort(records, split="test", selection_seed=selection_seed)
    if set(validation["task_ids"]).intersection(test["task_ids"]):
        raise ValueError("frozen validation and test cohorts overlap")
    return _seal(
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "selection_seed": selection_seed,
            "validation": validation,
            "test": test,
        },
        "cohorts_sha256",
    )


def validate_frozen_cohorts(
    cohorts: Mapping[str, Any],
    *,
    records: Sequence[BasisTaskRecord | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = _verify_seal(cohorts, "cohorts_sha256", label="frozen cohorts")
    if normalized.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported frozen-cohort collection schema")
    if set(normalized) != {
        "schema_version",
        "selection_seed",
        "validation",
        "test",
        "cohorts_sha256",
    }:
        raise ValueError("frozen-cohort collection has unexpected fields")
    source = list(records) if records is not None else None
    validation = validate_frozen_cohort(normalized["validation"], records=source)
    test = validate_frozen_cohort(normalized["test"], records=source)
    if (
        validation["selection_seed"] != normalized["selection_seed"]
        or test["selection_seed"] != normalized["selection_seed"]
    ):
        raise ValueError("frozen cohorts do not share their declared selection seed")
    if set(validation["task_ids"]).intersection(test["task_ids"]):
        raise ValueError("frozen validation and test cohorts overlap")
    return normalized


def build_protocol_lock(
    *,
    corpus_identity: Mapping[str, Any],
    cohorts: Mapping[str, Any],
    generation: Mapping[str, Any],
    bfs: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    evaluator_sources: Mapping[str, Any],
    source_amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(corpus_identity, Mapping) or not corpus_identity:
        raise ValueError("corpus_identity must be a non-empty mapping")
    normalized_cohorts = validate_frozen_cohorts(cohorts)
    normalized_generation = validate_generation_config(generation)
    normalized_bfs = validate_bfs_settings(bfs)
    normalized_tokenizer = validate_tokenizer_identity(tokenizer)
    normalized_sources = validate_evaluator_source_identity(evaluator_sources)
    body = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": _PROTOCOL_STATUS,
        "allowed_arm_ids": list(ARM_IDS),
        "corpus": canonical_copy(dict(corpus_identity)),
        "cohorts": normalized_cohorts,
        "generation": normalized_generation,
        "bfs": normalized_bfs,
        "tokenizer": normalized_tokenizer,
        "evaluator_sources": normalized_sources,
    }
    if source_amendment is not None:
        body["source_amendment"] = validate_source_amendment(
            source_amendment,
            successor_evaluator_sources=normalized_sources,
        )
    return _seal(body, "protocol_lock_sha256")


def validate_protocol_lock(
    protocol_lock: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
    records: Sequence[BasisTaskRecord | Mapping[str, Any]] | None = None,
    tokenizer_root: str | Path | None = None,
    evaluator_root: str | Path | None = None,
) -> dict[str, Any]:
    normalized = _verify_seal(
        protocol_lock,
        "protocol_lock_sha256",
        label="protocol lock",
        expected_sha256=expected_sha256,
    )
    if normalized.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported protocol-lock schema")
    if normalized.get("study_id") != STUDY_ID or normalized.get("status") != (
        _PROTOCOL_STATUS
    ):
        raise ValueError("protocol lock is not the frozen issue-54 study")
    if tuple(normalized.get("allowed_arm_ids", ())) != ARM_IDS:
        raise ValueError("protocol lock does not contain the three predeclared arms")
    expected_fields = {
        "schema_version",
        "study_id",
        "status",
        "allowed_arm_ids",
        "corpus",
        "cohorts",
        "generation",
        "bfs",
        "tokenizer",
        "evaluator_sources",
        "protocol_lock_sha256",
    }
    if "source_amendment" in normalized:
        expected_fields.add("source_amendment")
    if set(normalized) != expected_fields:
        raise ValueError("protocol lock has unexpected fields")
    if not isinstance(normalized.get("corpus"), dict) or not normalized["corpus"]:
        raise ValueError("protocol lock has no corpus identity")
    validate_frozen_cohorts(normalized["cohorts"], records=records)
    validate_generation_config(normalized["generation"])
    validate_bfs_settings(normalized["bfs"])
    validate_tokenizer_identity(normalized["tokenizer"], root=tokenizer_root)
    successor_sources = validate_evaluator_source_identity(
        normalized["evaluator_sources"], root=evaluator_root
    )
    if "source_amendment" in normalized:
        amendment = validate_source_amendment(
            normalized["source_amendment"],
            successor_evaluator_sources=successor_sources,
        )
        training_sources = amendment["training_evaluator_sources"]
        reconstructed_predecessor = {
            key: canonical_copy(value)
            for key, value in normalized.items()
            if key not in {"protocol_lock_sha256", "source_amendment"}
        }
        if amendment["schema_version"] == 1:
            reconstructed_predecessor["evaluator_sources"] = canonical_copy(
                training_sources
            )
        else:
            reconstructed_predecessor["evaluator_sources"] = canonical_copy(
                amendment["predecessor_evaluator_sources"]
            )
            reconstructed_predecessor["source_amendment"] = canonical_copy(
                amendment["predecessor_source_amendment"]
            )
        if sha256_json(reconstructed_predecessor) != amendment.get(
            "predecessor_protocol_lock_sha256"
        ):
            raise ValueError(
                "source amendment does not reconstruct its predecessor protocol lock"
            )
        generator_source = normalized["corpus"].get("generator_source")
        if not isinstance(generator_source, Mapping):
            raise ValueError(
                "amended protocol corpus has no preserved generator-source binding"
            )
        if generator_source.get("protocol_git_commit") != training_sources[
            "metadata"
        ].get("git_commit") or generator_source.get(
            "evaluator_source_sha256"
        ) != training_sources.get("source_sha256"):
            raise ValueError(
                "amended protocol corpus generator is not bound to training sources"
            )
    return normalized


@dataclass(frozen=True)
class ParseFailure:
    category: str
    unknown_function_names: tuple[str, ...] = ()
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.category not in PARSE_FAILURE_CATEGORIES:
            raise ValueError(f"unknown parse-failure category: {self.category!r}")
        if any(
            not isinstance(name, str) or not name
            for name in self.unknown_function_names
        ):
            raise ValueError("unknown function names must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "unknown_function_names": list(self.unknown_function_names),
            "detail": self.detail,
        }


def _parse_failure_from_mapping(value: Mapping[str, Any]) -> ParseFailure:
    names = value.get("unknown_function_names", ())
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError("unknown_function_names must be a sequence")
    detail = value.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("parse-failure detail must be a string or null")
    return ParseFailure(
        category=value.get("category"),
        unknown_function_names=tuple(sorted(set(names))),
        detail=detail,
    )


def _unknown_names_from_error(error: str) -> tuple[str, ...]:
    if "unknown function" not in error.lower():
        return ()
    opening = error.find("[")
    if opening >= 0:
        try:
            value = ast.literal_eval(error[opening:])
        except (SyntaxError, ValueError):
            value = None
        if isinstance(value, list) and all(
            isinstance(name, str) and name for name in value
        ):
            return tuple(sorted(set(value)))
    return ()


def classify_parse_failure(candidate: Mapping[str, Any]) -> ParseFailure | None:
    """Return one stable failure category for an unparseable candidate row."""

    parse_valid = candidate.get("parse_valid")
    if not isinstance(parse_valid, bool):
        raise ValueError("candidate parse_valid must be a boolean")
    explicit = candidate.get("parse_failure")
    if parse_valid:
        if explicit is not None:
            raise ValueError("a parse-valid candidate cannot have a parse failure")
        return None
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise ValueError("parse_failure must be a mapping or null")
        return _parse_failure_from_mapping(explicit)

    response = candidate.get("raw_response", "")
    if not isinstance(response, str):
        return ParseFailure("other", detail="response is not a string")
    error = candidate.get("parse_error")
    error_text = error if isinstance(error, str) else ""
    stripped = response.strip()
    if not stripped:
        return ParseFailure("empty", detail=error_text or "empty response")

    finish_reason = str(candidate.get("finish_reason", "")).lower()
    truncated = candidate.get("truncated") is True or finish_reason in {
        "length",
        "max_length",
        "max_new_tokens",
    }
    if truncated or "truncat" in error_text.lower():
        return ParseFailure(
            "truncation", detail=error_text or "generation length limit"
        )

    supplied_names = candidate.get("unknown_function_names", ())
    if isinstance(supplied_names, str):
        raise ValueError("unknown_function_names must be a sequence")
    if not isinstance(supplied_names, Sequence):
        raise ValueError("unknown_function_names must be a sequence")
    names = tuple(sorted(set(supplied_names))) or _unknown_names_from_error(error_text)
    if names or "unknown function" in error_text.lower():
        return ParseFailure(
            "unknown_function",
            unknown_function_names=names,
            detail=error_text or None,
        )

    lowered_error = error_text.lower()
    malformed = (
        "separator" in lowered_error
        or "empty function name" in lowered_error
        or "comma-separated" in lowered_error
        or stripped.startswith(",")
        or stripped.endswith(",")
        or ",," in stripped
        or ";" in stripped
        or "|" in stripped
        or "\n" in stripped
    )
    if malformed:
        return ParseFailure("malformed_separator", detail=error_text or None)
    return ParseFailure("other", detail=error_text or None)


def _candidate_bool(
    row: Mapping[str, Any],
    field: str,
    *,
    aliases: Sequence[str] = (),
    default: bool = False,
) -> bool:
    for name in (field, *aliases):
        if name in row:
            value = row[name]
            if not isinstance(value, bool):
                raise ValueError(f"candidate {name} must be a boolean")
            return value
    return default


def _normalize_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    arm_id = row.get("arm_id")
    task_id = row.get("task_id")
    sample_index = row.get("sample_index")
    if arm_id not in ALLOWED_ARM_IDS:
        raise ValueError(f"unexpected candidate arm ID: {arm_id!r}")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("candidate task_id must be a non-empty string")
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise ValueError("candidate sample_index must be a non-negative integer")
    parse_valid = _candidate_bool(row, "parse_valid")
    executable = _candidate_bool(row, "executor_success", aliases=("executable",))
    type_mismatch = _candidate_bool(row, "type_mismatch")
    if "type_mismatch" not in row:
        failure_kind = row.get("execution_failure_category")
        error = row.get("execution_error")
        type_mismatch = failure_kind == "type_mismatch" or (
            isinstance(error, str) and error.startswith("Type mismatch")
        )
    execution_correct = _candidate_bool(row, "execution_correct")
    exact_match = _candidate_bool(row, "exact_match")
    if not parse_valid and (
        executable or type_mismatch or execution_correct or exact_match
    ):
        raise ValueError("unparseable candidates cannot execute or match")
    if executable and type_mismatch:
        raise ValueError("an executable candidate cannot be a type mismatch")
    if execution_correct and not executable:
        raise ValueError("execution-correct candidates must be executable")
    if exact_match and not execution_correct:
        raise ValueError("an exact witness match must be execution-correct")
    failure = classify_parse_failure(row)
    return {
        "arm_id": arm_id,
        "task_id": task_id,
        "sample_index": sample_index,
        "parse_valid": parse_valid,
        "executable": executable,
        "type_mismatch": type_mismatch,
        "execution_correct": execution_correct,
        "exact_match": exact_match,
        "parse_failure": failure,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _funnel_for_prefix(
    task_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sample_count: int,
    top_n: int,
) -> dict[str, Any]:
    prefix = [row for rows in task_rows.values() for row in rows[:sample_count]]
    candidates = len(prefix)
    parseable = sum(row["parse_valid"] for row in prefix)
    unparseable = candidates - parseable
    executable = sum(row["executable"] for row in prefix)
    type_mismatch = sum(row["type_mismatch"] for row in prefix)
    execution_correct = sum(row["execution_correct"] for row in prefix)
    exact_match = sum(row["exact_match"] for row in prefix)
    execution_failure_other = parseable - executable - type_mismatch
    if execution_failure_other < 0:
        raise ValueError("candidate funnel does not reconcile")

    failure_counts = dict.fromkeys(PARSE_FAILURE_CATEGORIES, 0)
    reason_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    unknown_name_counts: Counter[str] = Counter()
    for row in prefix:
        failure = row["parse_failure"]
        if failure is None:
            continue
        failure_counts[failure.category] += 1
        names = tuple(failure.unknown_function_names)
        reason_counts[(failure.category, names)] += 1
        unknown_name_counts.update(names)
    if sum(failure_counts.values()) != unparseable:
        raise ValueError("parse-failure categories do not reconcile")

    top_reasons = []
    for (category, names), count in sorted(
        reason_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )[:top_n]:
        top_reasons.append(
            {
                "category": category,
                "unknown_function_names": list(names),
                "count": count,
                "candidate_rate": _rate(count, candidates),
                "fraction_of_unparseable": _rate(count, unparseable),
            }
        )

    tasks = len(task_rows)
    tasks_solved = sum(
        any(row["execution_correct"] for row in rows[:sample_count])
        for rows in task_rows.values()
    )
    tasks_exact = sum(
        any(row["exact_match"] for row in rows[:sample_count])
        for rows in task_rows.values()
    )
    return {
        "tasks": tasks,
        "sample_count": sample_count,
        "candidates": candidates,
        "parseable": parseable,
        "parseable_rate": _rate(parseable, candidates),
        "unparseable": unparseable,
        "unparseable_rate": _rate(unparseable, candidates),
        "executable": executable,
        "executable_rate": _rate(executable, candidates),
        "executable_given_parseable_rate": _rate(executable, parseable),
        "type_mismatch": type_mismatch,
        "type_mismatch_rate": _rate(type_mismatch, candidates),
        "type_mismatch_given_parseable_rate": _rate(type_mismatch, parseable),
        "execution_failure_other": execution_failure_other,
        "execution_failure_other_rate": _rate(execution_failure_other, candidates),
        "execution_correct": execution_correct,
        "execution_correct_rate": _rate(execution_correct, candidates),
        "correct_given_parseable_rate": _rate(execution_correct, parseable),
        "correct_given_executable_rate": _rate(execution_correct, executable),
        "exact_match": exact_match,
        "exact_match_rate": _rate(exact_match, candidates),
        "tasks_solved": tasks_solved,
        "solve_rate": _rate(tasks_solved, tasks),
        "tasks_exact_match": tasks_exact,
        "exact_solve_rate": _rate(tasks_exact, tasks),
        "parse_failure_counts": failure_counts,
        "parse_failure_rates": {
            category: _rate(count, candidates)
            for category, count in failure_counts.items()
        },
        "top_parse_failure_reasons": top_reasons,
        "top_unknown_function_names": [
            {"name": name, "count": count}
            for name, count in sorted(
                unknown_name_counts.items(), key=lambda item: (-item[1], item[0])
            )[:top_n]
        ],
    }


def aggregate_candidate_funnels(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str] | None = None,
    active_arm_ids: Sequence[str] | None = None,
    sample_prefixes: Sequence[int] = SAMPLE_PREFIXES,
    top_n: int = 5,
) -> dict[str, Any]:
    """Aggregate complete ordered candidate prefixes for the active model arms."""

    if tuple(sample_prefixes) != SAMPLE_PREFIXES:
        raise ValueError("candidate funnels require prefixes 1, 8, and 64")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    grouped: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw_row in candidate_rows:
        row = _normalize_candidate(raw_row)
        grouped[row["arm_id"]][row["task_id"]].append(row)
    inferred = tuple(arm_id for arm_id in ARM_IDS if arm_id in grouped)
    active = validate_active_arm_ids(
        inferred if active_arm_ids is None else active_arm_ids
    )
    if set(grouped) != set(active):
        raise ValueError("candidate rows do not cover exactly the active arms")

    expected_ids = set(expected_task_ids) if expected_task_ids is not None else None
    if expected_task_ids is not None and len(expected_ids) != len(expected_task_ids):
        raise ValueError("expected task IDs must be unique")
    reference_tasks: set[str] | None = None
    for arm_id in active:
        arm_tasks = set(grouped[arm_id])
        if not arm_tasks:
            raise ValueError(f"candidate arm {arm_id!r} has no tasks")
        if reference_tasks is None:
            reference_tasks = arm_tasks
        elif arm_tasks != reference_tasks:
            raise ValueError("candidate arms do not share one task cohort")
        if expected_ids is not None and arm_tasks != expected_ids:
            raise ValueError(
                f"candidate arm {arm_id!r} does not match the frozen cohort"
            )
        for task_id, rows in grouped[arm_id].items():
            indices = [row["sample_index"] for row in rows]
            if indices != list(range(SAMPLE_PREFIXES[-1])):
                raise ValueError(
                    f"candidate samples for {arm_id}/{task_id} are not one ordered "
                    "complete prefix from 0 through 63"
                )

    arms = {
        arm_id: {
            "tasks": len(grouped[arm_id]),
            "by_k": {
                str(sample_count): _funnel_for_prefix(
                    grouped[arm_id], sample_count=sample_count, top_n=top_n
                )
                for sample_count in SAMPLE_PREFIXES
            },
        }
        for arm_id in active
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "sample_prefixes": list(SAMPLE_PREFIXES),
        "arm_ids": list(active),
        "arms": arms,
    }


def build_selection_lock(
    *,
    arm_id: str,
    protocol_lock_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    validation_selection: Mapping[str, Any],
) -> dict[str, Any]:
    if arm_id not in ALLOWED_ARM_IDS:
        raise ValueError(f"unexpected arm ID: {arm_id!r}")
    _require_sha256(protocol_lock_sha256, label="protocol lock digest")
    if not isinstance(checkpoint_identity, Mapping) or not checkpoint_identity:
        raise ValueError("checkpoint_identity must be a non-empty mapping")
    if not isinstance(validation_selection, Mapping) or not validation_selection:
        raise ValueError("validation_selection must be a non-empty mapping")
    return _seal(
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "status": _SELECTION_STATUS,
            "arm_id": arm_id,
            "protocol_lock_sha256": protocol_lock_sha256,
            "checkpoint": canonical_copy(dict(checkpoint_identity)),
            "validation_selection": canonical_copy(dict(validation_selection)),
        },
        "selection_lock_sha256",
    )


def validate_selection_lock(
    selection_lock: Mapping[str, Any], *, protocol_lock_sha256: str
) -> dict[str, Any]:
    normalized = _verify_seal(
        selection_lock, "selection_lock_sha256", label="selection lock"
    )
    if normalized.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported selection-lock schema")
    if normalized.get("status") != _SELECTION_STATUS:
        raise ValueError("selection lock is not final")
    if normalized.get("arm_id") not in ALLOWED_ARM_IDS:
        raise ValueError("selection lock has an unexpected arm")
    if normalized.get("protocol_lock_sha256") != _require_sha256(
        protocol_lock_sha256, label="protocol lock digest"
    ):
        raise ValueError("selection lock belongs to a different protocol")
    if (
        not isinstance(normalized.get("checkpoint"), dict)
        or not normalized["checkpoint"]
    ):
        raise ValueError("selection lock has no checkpoint identity")
    if (
        not isinstance(normalized.get("validation_selection"), dict)
        or not normalized["validation_selection"]
    ):
        raise ValueError("selection lock has no validation selection evidence")
    return normalized


def validate_selection_locks(
    selection_locks: Sequence[Mapping[str, Any]],
    *,
    protocol_lock_sha256: str,
    active_arm_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not selection_locks:
        raise ValueError("at least one validation selection lock is required")
    if len(selection_locks) > len(ARM_IDS):
        raise ValueError("too many validation selection locks")
    by_arm: dict[str, dict[str, Any]] = {}
    checkpoint_digests: set[str] = set()
    for value in selection_locks:
        normalized = validate_selection_lock(
            value, protocol_lock_sha256=protocol_lock_sha256
        )
        arm_id = normalized["arm_id"]
        if arm_id in by_arm:
            raise ValueError(f"duplicate selection lock for arm {arm_id!r}")
        by_arm[arm_id] = normalized
        checkpoint_digest = sha256_json(normalized["checkpoint"])
        if checkpoint_digest in checkpoint_digests:
            raise ValueError("selection locks must identify distinct checkpoints")
        checkpoint_digests.add(checkpoint_digest)
    inferred = tuple(arm_id for arm_id in ARM_IDS if arm_id in by_arm)
    active = validate_active_arm_ids(
        inferred if active_arm_ids is None else active_arm_ids
    )
    if set(by_arm) != set(active):
        raise ValueError("selection locks must cover exactly the active arms")
    return {arm_id: by_arm[arm_id] for arm_id in active}


def _seal_ledger(body: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(body, "ledger_sha256")


def new_test_opening_ledger(
    *,
    opening_id: str,
    protocol_lock_sha256: str,
    selection_locks: Sequence[Mapping[str, Any]],
    active_arm_ids: Sequence[str] | None = None,
    feasibility_artifact_digest: str | None = None,
) -> dict[str, Any]:
    if not isinstance(opening_id, str) or not opening_id.strip():
        raise ValueError("opening_id must be a non-empty string")
    by_arm = validate_selection_locks(
        selection_locks,
        protocol_lock_sha256=protocol_lock_sha256,
        active_arm_ids=active_arm_ids,
    )
    active = tuple(by_arm)
    deferred = _deferred_arm_ids(active)
    if feasibility_artifact_digest is not None:
        _require_sha256(
            feasibility_artifact_digest, label="feasibility artifact digest"
        )
    body = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "opening_id": opening_id,
        "protocol_lock_sha256": protocol_lock_sha256,
        "active_arm_ids": list(active),
        "deferred_arm_ids": list(deferred),
        "feasibility_artifact_digest": feasibility_artifact_digest,
        "selection_lock_sha256_by_arm": {
            arm_id: by_arm[arm_id]["selection_lock_sha256"] for arm_id in active
        },
        "status": "prepared",
        "test_access_count": 0,
        "revision": 0,
        "arms": {
            arm_id: {"status": "pending", "result_identity": None} for arm_id in active
        },
        "shared_bfs_identity": None,
        "history": [{"revision": 0, "event": "prepared"}],
    }
    return _seal_ledger(body)


def validate_test_opening_ledger(
    ledger: Mapping[str, Any],
    *,
    protocol_lock_sha256: str | None = None,
    selection_locks: Sequence[Mapping[str, Any]] | None = None,
    active_arm_ids: Sequence[str] | None = None,
    deferred_arm_ids: Sequence[str] | None = None,
    feasibility_artifact_digest: str | None = None,
) -> dict[str, Any]:
    normalized = _verify_seal(ledger, "ledger_sha256", label="test-opening ledger")
    if (
        normalized.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or normalized.get("study_id") != STUDY_ID
    ):
        raise ValueError("test-opening ledger has the wrong schema or study")
    if (
        not isinstance(normalized.get("opening_id"), str)
        or not normalized["opening_id"]
    ):
        raise ValueError("test-opening ledger has no opening ID")
    protocol_digest = _require_sha256(
        normalized.get("protocol_lock_sha256"), label="ledger protocol digest"
    )
    if protocol_lock_sha256 is not None and protocol_digest != _require_sha256(
        protocol_lock_sha256, label="protocol lock digest"
    ):
        raise ValueError("test-opening ledger belongs to another protocol")
    status = normalized.get("status")
    if status not in _LEDGER_STATUSES:
        raise ValueError("test-opening ledger has an invalid status")
    access_count = normalized.get("test_access_count")
    if isinstance(access_count, bool) or access_count not in {0, 1}:
        raise ValueError("test-opening ledger has an invalid access count")
    if (status == "prepared") != (access_count == 0):
        raise ValueError("test-opening ledger access count contradicts its status")

    active = validate_active_arm_ids(normalized.get("active_arm_ids", ()))
    deferred = _deferred_arm_ids(active)
    if normalized.get("deferred_arm_ids") != list(deferred):
        raise ValueError("test-opening ledger has the wrong deferred arms")
    if active_arm_ids is not None and active != validate_active_arm_ids(active_arm_ids):
        raise ValueError("test-opening ledger active arms changed")
    if deferred_arm_ids is not None and tuple(deferred_arm_ids) != deferred:
        raise ValueError("test-opening ledger deferred arms changed")
    recorded_feasibility = normalized.get("feasibility_artifact_digest")
    if recorded_feasibility is not None:
        _require_sha256(
            recorded_feasibility, label="ledger feasibility artifact digest"
        )
    if (
        feasibility_artifact_digest is not None
        and recorded_feasibility
        != _require_sha256(
            feasibility_artifact_digest, label="feasibility artifact digest"
        )
    ):
        raise ValueError("test-opening ledger feasibility evidence changed")

    lock_digests = normalized.get("selection_lock_sha256_by_arm")
    if not isinstance(lock_digests, dict) or set(lock_digests) != set(active):
        raise ValueError("test-opening ledger does not identify the active selections")
    for digest in lock_digests.values():
        _require_sha256(digest, label="selection lock digest")
    if selection_locks is not None:
        by_arm = validate_selection_locks(
            selection_locks,
            protocol_lock_sha256=protocol_digest,
            active_arm_ids=active,
        )
        expected = {
            arm_id: by_arm[arm_id]["selection_lock_sha256"] for arm_id in active
        }
        if lock_digests != expected:
            raise ValueError("test-opening ledger selections changed")

    arms = normalized.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(active):
        raise ValueError("test-opening ledger does not contain exactly the active arms")
    complete_arms = 0
    for arm_id, arm in arms.items():
        if not isinstance(arm, dict) or arm.get("status") not in _ARM_LEDGER_STATUSES:
            raise ValueError(f"test-opening ledger arm {arm_id!r} is malformed")
        result = arm.get("result_identity")
        if arm["status"] == "complete":
            if not isinstance(result, dict) or not result:
                raise ValueError(f"completed arm {arm_id!r} has no result identity")
            complete_arms += 1
        elif result is not None:
            raise ValueError(f"pending arm {arm_id!r} already has a result")
    if status == "complete":
        if complete_arms != len(active):
            raise ValueError("completed test opening has incomplete arms")
        if (
            not isinstance(normalized.get("shared_bfs_identity"), dict)
            or not normalized["shared_bfs_identity"]
        ):
            raise ValueError("completed test opening has no shared BFS identity")
    elif normalized.get("shared_bfs_identity") is not None:
        raise ValueError("incomplete test opening cannot have a final BFS identity")

    revision = normalized.get("revision")
    history = normalized.get("history")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(history, list)
        or len(history) != revision + 1
        or [row.get("revision") for row in history] != list(range(revision + 1))
    ):
        raise ValueError("test-opening ledger revision history is malformed")
    return normalized


def _ledger_transition(
    ledger: Mapping[str, Any], *, event: str, updates: Mapping[str, Any]
) -> dict[str, Any]:
    current = validate_test_opening_ledger(ledger)
    body = canonical_copy(current)
    body.pop("ledger_sha256")
    body.update(canonical_copy(dict(updates)))
    body["revision"] = current["revision"] + 1
    body["history"].append({"revision": body["revision"], "event": event})
    return validate_test_opening_ledger(_seal_ledger(body))


def begin_test_opening(ledger: Mapping[str, Any]) -> dict[str, Any]:
    current = validate_test_opening_ledger(ledger)
    if current["status"] != "prepared" or current["test_access_count"] != 0:
        raise ValueError("held-out test has already been opened")
    return _ledger_transition(
        current,
        event="test-opened",
        updates={"status": "in_progress", "test_access_count": 1},
    )


def interrupt_test_opening(ledger: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    current = validate_test_opening_ledger(ledger)
    if current["status"] != "in_progress":
        raise ValueError("only an in-progress test opening can be interrupted")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an interruption reason is required")
    return _ledger_transition(
        current,
        event="interrupted:" + reason.strip(),
        updates={"status": "interrupted"},
    )


def resume_test_opening(ledger: Mapping[str, Any]) -> dict[str, Any]:
    current = validate_test_opening_ledger(ledger)
    if current["status"] == "in_progress":
        return current
    if current["status"] != "interrupted":
        raise ValueError("only the same interrupted test opening can be resumed")
    return _ledger_transition(
        current, event="resumed", updates={"status": "in_progress"}
    )


def record_test_arm_result(
    ledger: Mapping[str, Any],
    *,
    arm_id: str,
    result_identity: Mapping[str, Any],
) -> dict[str, Any]:
    current = validate_test_opening_ledger(ledger)
    if current["status"] != "in_progress":
        raise ValueError("test arm results require an in-progress opening")
    if arm_id not in current["active_arm_ids"]:
        raise ValueError(f"arm {arm_id!r} is not active for this test opening")
    if not isinstance(result_identity, Mapping) or not result_identity:
        raise ValueError("result_identity must be a non-empty mapping")
    normalized_result = canonical_copy(dict(result_identity))
    arm = current["arms"][arm_id]
    if arm["status"] == "complete":
        if arm["result_identity"] == normalized_result:
            return current
        raise ValueError(f"test arm {arm_id!r} was already scored differently")
    arms = canonical_copy(current["arms"])
    arms[arm_id] = {"status": "complete", "result_identity": normalized_result}
    return _ledger_transition(
        current, event=f"arm-complete:{arm_id}", updates={"arms": arms}
    )


def complete_test_opening(
    ledger: Mapping[str, Any], *, shared_bfs_identity: Mapping[str, Any]
) -> dict[str, Any]:
    current = validate_test_opening_ledger(ledger)
    normalized_bfs = canonical_copy(dict(shared_bfs_identity))
    if not normalized_bfs:
        raise ValueError("shared_bfs_identity must be a non-empty mapping")
    if current["status"] == "complete":
        if current["shared_bfs_identity"] == normalized_bfs:
            return current
        raise ValueError("completed test opening cannot change its BFS identity")
    if current["status"] != "in_progress":
        raise ValueError("only an in-progress test opening can complete")
    if any(arm["status"] != "complete" for arm in current["arms"].values()):
        raise ValueError(
            "all active test arms must complete before sealing the opening"
        )
    return _ledger_transition(
        current,
        event="test-opening-complete",
        updates={"status": "complete", "shared_bfs_identity": normalized_bfs},
    )


def create_test_opening_ledger_file(
    path: str | Path, ledger: Mapping[str, Any]
) -> Path:
    """Create a ledger with exclusive-create semantics; never overwrite one."""

    normalized = validate_test_opening_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as output_file:
        json.dump(normalized, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    return output


def load_test_opening_ledger(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load test-opening ledger: {error}") from error
    return validate_test_opening_ledger(value)


def replace_test_opening_ledger_file(
    path: str | Path,
    ledger: Mapping[str, Any],
    *,
    expected_previous_sha256: str,
) -> Path:
    """Atomically persist one validated next revision with optimistic locking."""

    output = Path(path)
    previous = load_test_opening_ledger(output)
    expected = _require_sha256(
        expected_previous_sha256, label="expected previous ledger digest"
    )
    if previous["ledger_sha256"] != expected:
        raise ValueError("test-opening ledger changed concurrently")
    normalized = validate_test_opening_ledger(ledger)
    immutable_fields = (
        "study_id",
        "opening_id",
        "protocol_lock_sha256",
        "active_arm_ids",
        "deferred_arm_ids",
        "feasibility_artifact_digest",
        "selection_lock_sha256_by_arm",
    )
    if any(normalized[field] != previous[field] for field in immutable_fields):
        raise ValueError("test-opening ledger identity cannot change")
    if normalized["ledger_sha256"] == previous["ledger_sha256"]:
        return output
    if normalized["revision"] != previous["revision"] + 1:
        raise ValueError("test-opening ledger must advance by exactly one revision")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(normalized, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output


__all__ = [
    "ALLOWED_ARM_IDS",
    "ARM_IDS",
    "CERTIFIED_DISTANCES",
    "CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE",
    "COHORT_SIZE",
    "PARSE_FAILURE_CATEGORIES",
    "PROTOCOL_SCHEMA_VERSION",
    "SAMPLE_PREFIXES",
    "SOURCE_AMENDMENT_FAILURE_EVIDENCE",
    "STUDY_ID",
    "TASKS_PER_DISTANCE",
    "ParseFailure",
    "aggregate_candidate_funnels",
    "begin_test_opening",
    "build_bfs_settings",
    "build_chained_source_amendment",
    "build_evaluator_source_identity",
    "build_frozen_cohorts",
    "build_generation_config",
    "build_protocol_lock",
    "build_selection_lock",
    "build_source_amendment",
    "build_tokenizer_identity",
    "canonical_copy",
    "canonical_json",
    "classify_parse_failure",
    "complete_test_opening",
    "create_test_opening_ledger_file",
    "file_identity",
    "freeze_cohort",
    "interrupt_test_opening",
    "load_test_opening_ledger",
    "new_test_opening_ledger",
    "protocol_training_evaluator_sources",
    "record_test_arm_result",
    "replace_test_opening_ledger_file",
    "resume_test_opening",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_active_arm_ids",
    "validate_bfs_settings",
    "validate_evaluator_source_identity",
    "validate_frozen_cohort",
    "validate_frozen_cohorts",
    "validate_generation_config",
    "validate_protocol_lock",
    "validate_selection_lock",
    "validate_selection_locks",
    "validate_source_amendment",
    "validate_test_opening_ledger",
    "validate_tokenizer_identity",
]
