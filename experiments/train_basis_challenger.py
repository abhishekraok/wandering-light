"""Basis-bound supervised continuation training for an induction challenger.

This runner deliberately separates data validation from training.  ``--dry-run``
loads the immutable basis, verifies every selected corpus file and witness, and
builds a small prompt/completion preview without importing a model, requiring a
GPU, or creating a W&B run.

``--verified-results`` replaces random-walk witnesses with successful programs
from a complete pilot evaluation.  It validates and hashes the sibling aggregate,
joins every result to the frozen corpus, re-executes every success, and can rewrite
programs into a direct child basis using its immutable deprecation/addition
provenance.  Both matched arms therefore train on the same exact task IDs.

An actual run is intentionally stricter: it requires CUDA and an explicitly named
W&B run in the personal entity, performs completion-only SFT from an exact local
checkpoint (content hashes) or Hugging Face commit (40-hex revision), creates no
intermediate checkpoints, and saves a weights-only final artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from wandering_light.basis_set import BasisSet, load_basis_set
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.llm_utils import generate_train_prompt
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord


SCHEMA_VERSION = 1
PERSONAL_WANDB_ENTITY = "abhishekraok-na"
DEFAULT_WANDB_PROJECT = "wandering-light-basis"
DEFAULT_LOCAL_CHECKPOINT = Path(
    "checkpoints/saved/rl/induction_opt_125m_sft_434k_rl_6k_with_lp"
)
DEFAULT_HF_CHECKPOINT = "abhishekraok/induction-basicfns-opt125m-sft434k-rl-6k-with-lp"
DEFAULT_HF_REVISION = "0c4ea07bfa618321b8dc5ce956ce5a86560d99a7"
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_CHECKPOINT_NAMES = frozenset(
    {
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
    }
)


@dataclass(frozen=True)
class BaseModelSpec:
    """Exact warm-start identity, independent of how Transformers loads it."""

    requested: str
    resolved: str
    revision: str | None
    canonical_hf_repo: str | None
    canonical_hf_revision: str | None
    local_files: Mapping[str, Mapping[str, int | str]]
    local_tree_digest: str | None
    wandb_run_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "revision": self.revision,
            "canonical_hf_repo": self.canonical_hf_repo,
            "canonical_hf_revision": self.canonical_hf_revision,
            "local_files": dict(self.local_files),
            "local_tree_digest": self.local_tree_digest,
            "wandb_run_url": self.wandb_run_url,
        }


@dataclass(frozen=True)
class ChallengerConfig:
    """Configuration shared by dry-run validation and an actual continuation."""

    corpus_dir: Path
    output_dir: Path
    basis_set_id: str = "default"
    splits: tuple[str, ...] = ("discovery",)
    base_model: str = str(DEFAULT_LOCAL_CHECKPOINT)
    base_model_revision: str | None = DEFAULT_HF_REVISION
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = 256
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    logging_steps: int = 100
    seed: int = 42
    max_train_records: int | None = None
    precision: str = "bf16"
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_run_name: str | None = None
    dry_run: bool = False
    dry_run_samples: int = 8
    verified_results: Path | None = None


@dataclass(frozen=True)
class PreparedCorpus:
    """Strictly verified prompt/completion rows and their source identity."""

    rows: tuple[Mapping[str, Any], ...]
    total_validated_records: int
    selected_records: int
    source: Mapping[str, Any]


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


def validate_personal_wandb_entity(entity: str) -> str:
    """Hard-stop before a run can be created in a work organization."""
    if entity.strip().lower() != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            "Basis challenger W&B writes are restricted to the personal entity "
            f"{PERSONAL_WANDB_ENTITY!r}; got {entity!r}"
        )
    return PERSONAL_WANDB_ENTITY


def _validate_config(config: ChallengerConfig) -> None:
    validate_personal_wandb_entity(config.wandb_entity)
    if not config.splits or any(not split for split in config.splits):
        raise ValueError("At least one non-empty corpus split is required")
    if len(set(config.splits)) != len(config.splits):
        raise ValueError("Corpus splits must not be repeated")
    if config.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.max_length <= 0 or config.logging_steps <= 0:
        raise ValueError("max_length and logging_steps must be positive")
    if not 0 <= config.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between zero and one")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.max_train_records is not None and config.max_train_records <= 0:
        raise ValueError("max_train_records must be positive when supplied")
    if config.dry_run_samples <= 0:
        raise ValueError("dry_run_samples must be positive")
    if config.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be one of: bf16, fp16, fp32")
    if not config.dry_run and not config.wandb_run_name:
        raise ValueError(
            "An actual training run requires --wandb-run-name so the final "
            "artifact has personal W&B lineage"
        )


def resolve_base_model(model: str, revision: str | None) -> BaseModelSpec:
    """Resolve through the pilot's exact checkpoint-lineage verifier.

    Sharing this resolver is important: the well-known local path is only labeled
    as the canonical personal HF checkpoint after its weights size, trainer step,
    and ``wandb_run.url`` have been checked.  The complete durable tree digest is
    then retained in the training lineage instead of hashing a different file set.
    """
    from experiments.basis_library_pilot import resolve_model_spec

    spec = resolve_model_spec(model, revision)
    files = dict(spec.local_files)
    if files:
        if "config.json" not in files:
            raise ValueError(f"Local base model has no config.json: {spec.resolved}")
        if not any(
            name.endswith((".safetensors", ".bin"))
            and ("model" in name or "pytorch" in name)
            for name in files
        ):
            raise ValueError(f"Local base model has no model weights: {spec.resolved}")
    return BaseModelSpec(
        requested=spec.requested,
        resolved=spec.resolved,
        revision=spec.revision,
        canonical_hf_repo=spec.canonical_hf_repo,
        canonical_hf_revision=spec.canonical_hf_revision,
        local_files=files,
        local_tree_digest=spec.local_tree_digest,
        wandb_run_url=spec.wandb_run_url,
    )


def _basis_manifest(basis: BasisSet) -> dict[str, Any]:
    return basis.to_manifest()


def _source_corpus_identity(
    corpus_dir: Path,
    manifest: Mapping[str, Any],
    splits: Sequence[str],
) -> dict[str, Any]:
    manifest_path = corpus_dir / "manifest.json"
    split_files: dict[str, Any] = {}
    for split in splits:
        metadata = manifest["splits"][split]
        path = corpus_dir / metadata["path"]
        split_files[split] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "records": metadata["size"],
        }
    return {
        "root": str(corpus_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "requested_splits": list(splits),
        "split_files": split_files,
    }


def _validate_and_convert_record(
    record: BasisTaskRecord,
    *,
    executor: Executor,
    functions_by_id: Mapping[str, Any],
    prompt_context: FunctionDefSet,
    build_row: bool,
) -> dict[str, Any] | None:
    if not record.witness_function_ids:
        raise ValueError(f"Task {record.task_id} has an empty training witness")
    witness = []
    for step, (function_id, function_name) in enumerate(
        zip(
            record.witness_function_ids,
            record.witness_function_names,
            strict=True,
        )
    ):
        function = functions_by_id.get(function_id)
        if function is None:
            raise ValueError(
                f"Task {record.task_id} witness step {step} references unknown "
                f"function ID {function_id!r}"
            )
        if function.name != function_name:
            raise ValueError(
                f"Task {record.task_id} witness step {step} maps {function_id!r} "
                f"to {function.name!r}, not recorded name {function_name!r}"
            )
        witness.append(function)

    solution = FunctionDefList(witness)
    specification = TrajectorySpec(
        input_list=record.input_value,
        function_defs=solution,
    )
    execution = executor.execute_trajectory(specification)
    if (
        not execution.success
        or execution.trajectory is None
        or execution.trajectory.output != record.output_value
    ):
        detail = execution.error_msg if not execution.success else "output mismatch"
        raise ValueError(
            f"Task {record.task_id} witness does not reproduce its frozen output: "
            f"{detail}"
        )

    if not build_row:
        return None

    example = generate_train_prompt(
        input_list=record.input_value,
        output_list=record.output_value,
        # The existing helper eagerly formats this palette even when it is omitted
        # from the prompt.  An empty context preserves the exact no-palette format
        # without doing O(number of basis functions) discarded work per record.
        available_functions=prompt_context,
        solution=solution,
        include_available_functions=False,
    )
    expected_completion = ",".join(record.witness_function_names)
    if example.output_text != expected_completion:
        raise ValueError(
            f"Task {record.task_id} completion differs from its witness names"
        )
    return {
        "prompt": example.input_text,
        "completion": example.output_text,
        "task_id": record.task_id,
        "split": record.split,
        "witness_length": record.witness_length,
        "witness_function_ids": list(record.witness_function_ids),
    }


def _build_verified_row(
    record: BasisTaskRecord,
    solution: FunctionDefList,
    *,
    prompt_context: FunctionDefSet,
) -> dict[str, Any]:
    if not solution:
        raise ValueError(f"Task {record.task_id} has an empty verified program")
    example = generate_train_prompt(
        input_list=record.input_value,
        output_list=record.output_value,
        available_functions=prompt_context,
        solution=solution,
        include_available_functions=False,
    )
    names = [function.name for function in solution]
    if example.output_text != ",".join(names):
        raise ValueError(
            f"Task {record.task_id} completion differs from its verified program"
        )
    return {
        "prompt": example.input_text,
        "completion": example.output_text,
        "task_id": record.task_id,
        "split": record.split,
        "witness_length": record.witness_length,
        "solution_function_ids": [
            function.metadata["basis_function_id"] for function in solution
        ],
    }


def _verified_result_paths(results_path: Path) -> tuple[Path, Path]:
    path = results_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Verified results file does not exist: {path}")
    if path.name != "results.jsonl.gz":
        raise ValueError(
            "--verified-results must name the pilot's results.jsonl.gz artifact"
        )
    aggregate_path = path.with_name("aggregate.json")
    if not aggregate_path.is_file():
        raise FileNotFoundError(
            f"Verified results have no sibling aggregate.json: {aggregate_path}"
        )
    return path, aggregate_path


def _load_aggregate(path: Path) -> dict[str, Any]:
    try:
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read verified aggregate {path}: {error}"
        ) from error
    if not isinstance(aggregate, dict):
        raise ValueError("Verified aggregate must be a JSON object")
    return aggregate


def _require_result_basis(aggregate: Mapping[str, Any], source_basis: BasisSet) -> None:
    expected = {
        "task_source_basis_set_id": source_basis.basis_set_id,
        "task_source_basis_set_digest": source_basis.digest,
        "evaluation_basis_set_id": source_basis.basis_set_id,
        "evaluation_basis_set_digest": source_basis.digest,
    }
    for key, value in expected.items():
        if aggregate.get(key) != value:
            raise ValueError(
                f"Verified aggregate {key} mismatch: "
                f"{aggregate.get(key)!r} != {value!r}"
            )
    model = aggregate.get("model")
    if not isinstance(model, dict):
        raise ValueError("Verified aggregate has no exact model identity")
    if (
        model.get("canonical_hf_repo") != DEFAULT_HF_CHECKPOINT
        or model.get("canonical_hf_revision") != DEFAULT_HF_REVISION
    ):
        raise ValueError(
            "Verified results were not produced by the canonical default solver "
            "checkpoint"
        )


def _validate_aggregate_outcomes(
    aggregate: Mapping[str, Any],
    *,
    result_rows: Sequence[Mapping[str, Any]],
) -> None:
    from experiments.basis_library_pilot import (
        _dimension_summary,
        _outcome_summary,
    )

    expected = {
        "overall": _outcome_summary(result_rows),
        "by_split": _dimension_summary(result_rows, "split"),
        "by_input_type": _dimension_summary(result_rows, "input_type"),
        "by_witness_length": _dimension_summary(result_rows, "witness_length"),
    }
    for key, value in expected.items():
        if aggregate.get(key) != value:
            raise ValueError(f"Verified aggregate {key} does not match its result rows")


def _validated_result_solutions(
    *,
    results_path: Path,
    aggregate: Mapping[str, Any],
    records: Sequence[BasisTaskRecord],
    source_basis: BasisSet,
) -> tuple[dict[str, tuple[Any, ...]], list[dict[str, Any]]]:
    """Validate the complete result artifact and re-execute every success."""
    records_by_id = {record.task_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Loaded source corpus contains duplicate task IDs")
    runtime_functions = source_basis.as_function_set()
    functions_by_id = {
        function.metadata["basis_function_id"]: function
        for function in runtime_functions
    }
    executor = Executor(runtime_functions)
    solutions: dict[str, tuple[Any, ...]] = {}
    result_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with gzip.open(results_path, "rt", encoding="utf-8") as result_file:
            for line_number, line in enumerate(result_file, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid verified result at line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Verified result line {line_number} is not an object"
                    )
                task_id = row.get("task_id")
                if not isinstance(task_id, str) or task_id not in records_by_id:
                    raise ValueError(
                        f"Verified result line {line_number} has unknown task ID "
                        f"{task_id!r}"
                    )
                if task_id in seen:
                    raise ValueError(f"Duplicate verified result for task {task_id}")
                seen.add(task_id)
                record = records_by_id[task_id]
                if (
                    row.get("split") != record.split
                    or row.get("input_type") != record.metadata["input_type"]
                    or row.get("witness_length") != record.witness_length
                ):
                    raise ValueError(
                        f"Verified result metadata mismatch for task {task_id}"
                    )
                success = row.get("success")
                if not isinstance(success, bool):
                    raise ValueError(
                        f"Verified result success is not boolean for task {task_id}"
                    )
                function_ids = row.get("solution_function_ids")
                function_names = row.get("solution_function_names")
                if not isinstance(function_ids, list) or not isinstance(
                    function_names, list
                ):
                    raise ValueError(
                        f"Verified result has invalid function arrays for task {task_id}"
                    )
                if len(function_ids) != len(function_names):
                    raise ValueError(
                        f"Verified result function arrays differ for task {task_id}"
                    )
                if not success:
                    if function_ids or row.get("solution_length") is not None:
                        raise ValueError(
                            f"Failed result carries a solution for task {task_id}"
                        )
                    result_rows.append(row)
                    continue
                if not function_ids or row.get("solution_length") != len(function_ids):
                    raise ValueError(
                        f"Successful result has invalid solution length for task {task_id}"
                    )
                solution = []
                for step, (function_id, function_name) in enumerate(
                    zip(function_ids, function_names, strict=True)
                ):
                    function = functions_by_id.get(function_id)
                    if function is None or function.name != function_name:
                        raise ValueError(
                            f"Verified result task {task_id} step {step} has "
                            "a function ID/name outside the source basis"
                        )
                    solution.append(function)
                execution = executor.execute_trajectory(
                    TrajectorySpec(record.input_value, FunctionDefList(solution))
                )
                if (
                    not execution.success
                    or execution.trajectory is None
                    or execution.trajectory.output != record.output_value
                ):
                    raise ValueError(
                        f"Verified result program does not reproduce task {task_id}"
                    )
                solutions[task_id] = tuple(solution)
                result_rows.append(row)
    except OSError as error:
        raise ValueError(
            f"Could not read verified results {results_path}: {error}"
        ) from error

    missing = set(records_by_id) - seen
    if missing:
        raise ValueError(f"Verified results omit {len(missing)} evaluated corpus tasks")
    _validate_aggregate_outcomes(aggregate, result_rows=result_rows)
    return solutions, result_rows


@dataclass(frozen=True)
class _RewriteRules:
    source_basis_set_id: str
    target_basis_set_id: str
    target_by_name: Mapping[str, Any]
    deprecations: Mapping[str, str | None]
    deprecation_ids: Mapping[str, str]
    additions: tuple[tuple[tuple[str, ...], str], ...]


def _derive_rewrite_rules(
    source_basis: BasisSet,
    target_basis: BasisSet,
) -> _RewriteRules:
    target_runtime = target_basis.as_function_set()
    target_by_name = {function.name: function for function in target_runtime}
    if source_basis.basis_set_id == target_basis.basis_set_id:
        if source_basis.digest != target_basis.digest:
            raise ValueError("Equal basis-set IDs have different digests")
        return _RewriteRules(
            source_basis_set_id=source_basis.basis_set_id,
            target_basis_set_id=target_basis.basis_set_id,
            target_by_name=target_by_name,
            deprecations={},
            deprecation_ids={},
            additions=(),
        )

    if target_basis.parent_basis_set_id != source_basis.basis_set_id:
        raise ValueError(
            "Verified-result rewriting supports only the exact source basis or "
            "one direct immutable child"
        )
    if target_basis.provenance.get("parent_basis_set_digest") != source_basis.digest:
        raise ValueError("Target provenance does not pin the source basis digest")

    source_by_name = {function.name: function for function in source_basis.functions}
    target_manifest_by_name = {
        function.name: function for function in target_basis.functions
    }
    shared_names = set(source_by_name).intersection(target_manifest_by_name)
    changed_shared = sorted(
        name
        for name in shared_names
        if source_by_name[name].function_id != target_manifest_by_name[name].function_id
    )
    if changed_shared:
        raise ValueError(
            "Child basis changes definitions in place instead of using new names: "
            f"{changed_shared}"
        )

    raw_deprecations = target_basis.provenance.get("deprecations")
    raw_additions = target_basis.provenance.get("additions")
    if not isinstance(raw_deprecations, Sequence) or isinstance(
        raw_deprecations, str | bytes
    ):
        raise ValueError("Child target provenance has no deprecations sequence")
    if not isinstance(raw_additions, Sequence) or isinstance(
        raw_additions, str | bytes
    ):
        raise ValueError("Child target provenance has no additions sequence")

    deprecations: dict[str, str | None] = {}
    deprecation_ids: dict[str, str] = {}
    for index, item in enumerate(raw_deprecations):
        if not isinstance(item, Mapping):
            raise ValueError(f"Deprecation {index} is not an object")
        name = item.get("function_name")
        function_id = item.get("function_id")
        replacement = item.get("replacement")
        source_function = source_by_name.get(name)
        if (
            not isinstance(name, str)
            or source_function is None
            or function_id != source_function.function_id
        ):
            raise ValueError(
                f"Deprecation {index} does not identify an exact source function"
            )
        if name in deprecations:
            raise ValueError(f"Duplicate deprecation rule for {name!r}")
        if name in target_by_name:
            raise ValueError(f"Deprecated function {name!r} remains in child basis")
        if replacement == "zero-step identity":
            target_name = None
        elif isinstance(replacement, str) and replacement in target_by_name:
            target_name = replacement
        else:
            raise ValueError(
                f"Deprecation {name!r} has invalid replacement {replacement!r}"
            )
        deprecations[name] = target_name
        deprecation_ids[name] = function_id

    removed_names = set(source_by_name) - set(target_manifest_by_name)
    if removed_names != set(deprecations):
        raise ValueError(
            "Child basis omissions do not exactly match provenance deprecations: "
            f"omitted={sorted(removed_names)}, "
            f"declared={sorted(deprecations)}"
        )

    additions: list[tuple[tuple[str, ...], str]] = []
    seen_added_names: set[str] = set()
    seen_sequences: set[tuple[str, ...]] = set()
    for index, item in enumerate(raw_additions):
        if not isinstance(item, Mapping):
            raise ValueError(f"Addition {index} is not an object")
        name = item.get("function_name")
        source_sequence = item.get("source_sequence")
        if (
            not isinstance(name, str)
            or name not in target_by_name
            or name in source_by_name
        ):
            raise ValueError(
                f"Addition {index} does not identify a new target function"
            )
        if (
            not isinstance(source_sequence, Sequence)
            or isinstance(source_sequence, str | bytes)
            or len(source_sequence) < 2
            or any(
                not isinstance(source_name, str) or source_name not in source_by_name
                for source_name in source_sequence
            )
        ):
            raise ValueError(f"Addition {name!r} has an invalid source_sequence")
        sequence = tuple(source_sequence)
        if name in seen_added_names or sequence in seen_sequences:
            raise ValueError("Child provenance contains duplicate addition rules")
        seen_added_names.add(name)
        seen_sequences.add(sequence)
        additions.append((sequence, name))

    added_names = set(target_manifest_by_name) - set(source_by_name)
    if added_names != seen_added_names:
        raise ValueError(
            "Child basis additions do not exactly match provenance: "
            f"added={sorted(added_names)}, declared={sorted(seen_added_names)}"
        )
    additions.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return _RewriteRules(
        source_basis_set_id=source_basis.basis_set_id,
        target_basis_set_id=target_basis.basis_set_id,
        target_by_name=target_by_name,
        deprecations=deprecations,
        deprecation_ids=deprecation_ids,
        additions=tuple(additions),
    )


def _rewrite_verified_solution(
    source_solution: Sequence[Any],
    rules: _RewriteRules,
) -> tuple[FunctionDefList, Counter[str]]:
    names = [function.name for function in source_solution]
    events: Counter[str] = Counter()
    rewritten_names: list[str] = []
    index = 0
    while index < len(names):
        collapsed = False
        for source_sequence, target_name in rules.additions:
            width = len(source_sequence)
            if tuple(names[index : index + width]) == source_sequence:
                rewritten_names.append(target_name)
                events[f"collapse:{','.join(source_sequence)}->{target_name}"] += 1
                index += width
                collapsed = True
                break
        if collapsed:
            continue
        source_name = names[index]
        if source_name in rules.deprecations:
            replacement = rules.deprecations[source_name]
            if replacement is None:
                events[f"drop:{source_name}"] += 1
            else:
                rewritten_names.append(replacement)
                events[f"replace:{source_name}->{replacement}"] += 1
        else:
            rewritten_names.append(source_name)
        index += 1

    try:
        rewritten = [rules.target_by_name[name] for name in rewritten_names]
    except KeyError as error:
        raise ValueError(
            f"Rewritten verified program references missing target {error.args[0]!r}"
        ) from error
    if not rewritten:
        raise ValueError("Provenance rewrite produced an empty target program")
    return FunctionDefList(rewritten), events


def _rewrite_summary(
    *,
    rules: _RewriteRules,
    all_events: Counter[str],
    source_steps: int,
    target_steps: int,
    records: int,
    changed_records: int,
) -> dict[str, Any]:
    return {
        "mode": (
            "retain-source-solutions"
            if rules.source_basis_set_id == rules.target_basis_set_id
            else "direct-child-provenance-rewrite"
        ),
        "source_basis_set_id": rules.source_basis_set_id,
        "target_basis_set_id": rules.target_basis_set_id,
        "records": records,
        "changed_records": changed_records,
        "source_steps": source_steps,
        "target_steps": target_steps,
        "net_step_change": target_steps - source_steps,
        "events": dict(sorted(all_events.items())),
    }


def _prepare_witness_training_corpus(
    config: ChallengerConfig,
    *,
    preview_only: bool | None = None,
) -> tuple[BasisSet, PreparedCorpus]:
    """Validate the complete requested corpus, then build selected SFT rows."""
    from experiments.basis_library_pilot import load_corpus

    _validate_config(config)
    basis = load_basis_set(config.basis_set_id)
    manifest, records = load_corpus(config.corpus_dir, splits=config.splits)
    if manifest["basis_set_id"] != basis.basis_set_id:
        raise ValueError(
            "Corpus basis-set ID does not match the requested immutable basis: "
            f"{manifest['basis_set_id']!r} != {basis.basis_set_id!r}"
        )
    if manifest["basis_set_digest"] != basis.digest:
        raise ValueError(
            "Corpus basis-set digest does not match the requested immutable basis"
        )

    available_functions = basis.as_function_set()
    functions_by_id = {
        function.metadata["basis_function_id"]: function
        for function in available_functions
    }
    executor = Executor(available_functions)
    selected_limit = config.max_train_records
    use_preview = config.dry_run if preview_only is None else preview_only
    if use_preview:
        selected_limit = min(
            config.dry_run_samples,
            selected_limit if selected_limit is not None else len(records),
        )
    if selected_limit is None:
        selected_limit = len(records)

    # Validate every witness, even when only a prefix is selected for training or
    # preview.  Retain prompt strings only for the selected records.
    prompt_context = FunctionDefSet()
    selected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        converted = _validate_and_convert_record(
            record,
            executor=executor,
            functions_by_id=functions_by_id,
            prompt_context=prompt_context,
            build_row=index < selected_limit,
        )
        if converted is not None:
            selected.append(converted)
    if not records:
        raise ValueError("The requested corpus splits contain no training records")
    selected_task_ids = [row["task_id"] for row in selected]
    source = _source_corpus_identity(config.corpus_dir, manifest, config.splits)
    source.update(
        {
            "total_validated_records": len(records),
            "selected_records": len(selected),
            "selection": "source-order-prefix",
            "selected_task_ids_digest": _sha256_bytes(
                _canonical_json(selected_task_ids).encode("utf-8")
            ),
        }
    )
    return basis, PreparedCorpus(
        rows=tuple(selected),
        total_validated_records=len(records),
        selected_records=len(selected),
        source=source,
    )


def _prepare_verified_training_corpus(
    config: ChallengerConfig,
    *,
    preview_only: bool | None = None,
) -> tuple[BasisSet, PreparedCorpus]:
    from experiments.basis_library_pilot import load_corpus

    if config.verified_results is None:
        raise ValueError("Verified-results preparation requires a result artifact")
    results_path, aggregate_path = _verified_result_paths(config.verified_results)
    aggregate = _load_aggregate(aggregate_path)
    evaluated_splits = aggregate.get("evaluated_splits")
    if (
        not isinstance(evaluated_splits, list)
        or not evaluated_splits
        or any(not isinstance(split, str) or not split for split in evaluated_splits)
        or len(set(evaluated_splits)) != len(evaluated_splits)
    ):
        raise ValueError("Verified aggregate has invalid evaluated_splits")
    missing_splits = set(config.splits) - set(evaluated_splits)
    if missing_splits:
        raise ValueError(
            f"Requested training splits were not evaluated: {sorted(missing_splits)}"
        )

    manifest, evaluated_records = load_corpus(
        config.corpus_dir, splits=tuple(evaluated_splits)
    )
    if aggregate.get("corpus_manifest_digest") != manifest["manifest_digest"]:
        raise ValueError(
            "Verified aggregate corpus manifest digest does not match the loaded "
            "source corpus"
        )
    source_basis = load_basis_set(manifest["basis_set_id"])
    if source_basis.digest != manifest["basis_set_digest"]:
        raise ValueError("Loaded source corpus has a different basis digest")
    _require_result_basis(aggregate, source_basis)
    solutions, result_rows = _validated_result_solutions(
        results_path=results_path,
        aggregate=aggregate,
        records=evaluated_records,
        source_basis=source_basis,
    )

    records_by_split: dict[str, list[BasisTaskRecord]] = {
        split: [] for split in evaluated_splits
    }
    for record in evaluated_records:
        records_by_split[record.split].append(record)
    requested_records = [
        record for split in config.splits for record in records_by_split[split]
    ]
    successful_records = [
        record for record in requested_records if record.task_id in solutions
    ]
    if not successful_records:
        raise ValueError("Requested splits contain no verified solver successes")

    target_basis = load_basis_set(config.basis_set_id)
    rules = _derive_rewrite_rules(source_basis, target_basis)
    target_executor = Executor(list(rules.target_by_name.values()))
    rewritten_by_task: dict[str, FunctionDefList] = {}
    events_by_task: dict[str, Counter[str]] = {}
    source_steps_by_task: dict[str, int] = {}
    target_steps_by_task: dict[str, int] = {}
    changed_by_task: dict[str, bool] = {}
    for record in successful_records:
        source_solution = solutions[record.task_id]
        rewritten, events = _rewrite_verified_solution(source_solution, rules)
        execution = target_executor.execute_trajectory(
            TrajectorySpec(record.input_value, rewritten)
        )
        if (
            not execution.success
            or execution.trajectory is None
            or execution.trajectory.output != record.output_value
        ):
            raise ValueError(
                "Rewritten target-basis program does not reproduce verified task "
                f"{record.task_id}"
            )
        source_names = [function.name for function in source_solution]
        target_names = [function.name for function in rewritten]
        rewritten_by_task[record.task_id] = rewritten
        events_by_task[record.task_id] = events
        source_steps_by_task[record.task_id] = len(source_solution)
        target_steps_by_task[record.task_id] = len(rewritten)
        changed_by_task[record.task_id] = source_names != target_names

    planned_records = successful_records
    if config.max_train_records is not None:
        planned_records = planned_records[: config.max_train_records]
    if not planned_records:
        raise ValueError("Verified-success selection produced no training records")
    common_task_ids = [record.task_id for record in planned_records]
    use_preview = config.dry_run if preview_only is None else preview_only
    materialized_records = planned_records
    if use_preview:
        materialized_records = materialized_records[: config.dry_run_samples]
    prompt_context = FunctionDefSet()
    selected = [
        _build_verified_row(
            record,
            rewritten_by_task[record.task_id],
            prompt_context=prompt_context,
        )
        for record in materialized_records
    ]

    def summarize(records: Sequence[BasisTaskRecord]) -> dict[str, Any]:
        all_events: Counter[str] = Counter()
        for record in records:
            all_events.update(events_by_task[record.task_id])
        return _rewrite_summary(
            rules=rules,
            all_events=all_events,
            source_steps=sum(
                source_steps_by_task[record.task_id] for record in records
            ),
            target_steps=sum(
                target_steps_by_task[record.task_id] for record in records
            ),
            records=len(records),
            changed_records=sum(changed_by_task[record.task_id] for record in records),
        )

    source = _source_corpus_identity(config.corpus_dir, manifest, config.splits)
    common_digest = _sha256_bytes(_canonical_json(common_task_ids).encode("utf-8"))
    source.update(
        {
            "training_data_mode": "verified-default-solver-successes",
            "total_validated_records": len(requested_records),
            "evaluated_result_records": len(result_rows),
            "requested_verified_successes": len(successful_records),
            "planned_training_records": len(planned_records),
            "materialized_records": len(selected),
            "selection": "source-order-verified-success-prefix",
            "common_task_ids_digest": common_digest,
            "selected_task_ids_digest": common_digest,
            "verified_results": {
                "path": str(results_path),
                "bytes": results_path.stat().st_size,
                "sha256": _sha256_file(results_path),
                "aggregate_path": str(aggregate_path),
                "aggregate_bytes": aggregate_path.stat().st_size,
                "aggregate_sha256": _sha256_file(aggregate_path),
                "evaluated_splits": list(evaluated_splits),
                "solver_model": dict(aggregate["model"]),
            },
            "rewrite_counts": {
                "all_requested_successes": summarize(successful_records),
                "common_training_tasks": summarize(planned_records),
            },
        }
    )
    return target_basis, PreparedCorpus(
        rows=tuple(selected),
        total_validated_records=len(requested_records),
        selected_records=len(selected),
        source=source,
    )


def prepare_training_corpus(
    config: ChallengerConfig,
    *,
    preview_only: bool | None = None,
) -> tuple[BasisSet, PreparedCorpus]:
    """Validate the complete source and build completion-only SFT rows."""
    _validate_config(config)
    if config.verified_results is not None:
        return _prepare_verified_training_corpus(config, preview_only=preview_only)
    return _prepare_witness_training_corpus(config, preview_only=preview_only)


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def run_git_bytes(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    try:
        commit = run_git("rev-parse", "HEAD")
        if not _GIT_COMMIT_RE.fullmatch(commit):
            raise ValueError(f"Invalid git commit returned by git: {commit!r}")
        status = run_git("status", "--porcelain", "--untracked-files=all")
        tracked_diff = run_git("diff", "--binary", "HEAD")
        raw_untracked = run_git_bytes(
            "ls-files", "--others", "--exclude-standard", "-z"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not record the source git identity") from error
    untracked_files: list[dict[str, Any]] = []
    for raw_path in sorted(filter(None, raw_untracked.split(b"\0"))):
        try:
            relative_path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("Untracked git path is not valid UTF-8") from error
        path = repo_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Training source identity only supports regular untracked files: "
                f"{relative_path}"
            )
        untracked_files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff.encode("utf-8")),
        "untracked_files": untracked_files,
        "untracked_files_digest": _sha256_bytes(
            _canonical_json(untracked_files).encode("utf-8")
        ),
        "runner": {
            "path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }


def _require_unchanged_source(
    repo_root: Path, expected: Mapping[str, Any]
) -> None:
    if _git_identity(repo_root) != dict(expected):
        raise RuntimeError(
            "Repository source changed after the training process started; "
            "discard this run and relaunch from a stable tree"
        )


def _hyperparameters(config: ChallengerConfig) -> dict[str, Any]:
    return {
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "logging_steps": config.logging_steps,
        "seed": config.seed,
        "data_seed": config.seed,
        "precision": config.precision,
        "completion_only_loss": True,
        "save_strategy": "no",
        "full_determinism": True,
        "max_train_records": config.max_train_records,
    }


def build_training_manifest(
    *,
    config: ChallengerConfig,
    basis: BasisSet,
    corpus: PreparedCorpus,
    base_model: BaseModelSpec,
    repo_root: Path,
    status: str,
    wandb_url: str | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    source_code: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete machine-readable lineage record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "basis_set": basis.identity_dict(),
        "source_corpus": dict(corpus.source),
        "base_model": base_model.to_dict(),
        "hyperparameters": _hyperparameters(config),
        "source_code": dict(source_code or _git_identity(repo_root)),
        "wandb": {
            "entity": PERSONAL_WANDB_ENTITY,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "url": wandb_url,
        },
        "trainer_state": dict(trainer_state or {}),
    }


def build_sft_config(config: ChallengerConfig, *, report_to_wandb: bool):
    """Construct deterministic, completion-only, no-checkpoint TRL arguments."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(config.output_dir),
        overwrite_output_dir=False,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="no",
        completion_only_loss=True,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=True,
        dataloader_num_workers=0,
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        report_to=["wandb"] if report_to_wandb else [],
        run_name=config.wandb_run_name if report_to_wandb else None,
        push_to_hub=False,
    )


def _validate_wandb_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme not in {"http", "https"}
        or len(parts) < 4
        or parts[0].lower() != PERSONAL_WANDB_ENTITY
        or parts[-2] != "runs"
    ):
        raise ValueError(
            f"W&B returned a run URL outside the required personal entity: {url!r}"
        )
    return url


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_wandb_link(path: Path, url: str) -> None:
    path.write_text(f"[InternetShortcut]\nURL={url}\n", encoding="utf-8")


def assert_weights_only_artifact(
    output_dir: Path, *, require_complete: bool = False
) -> None:
    """Reject optimizer, scheduler, RNG, scaler, or intermediate state."""
    forbidden: list[str] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _FORBIDDEN_CHECKPOINT_NAMES or path.name.startswith(
            "rng_state_"
        ):
            forbidden.append(path.relative_to(output_dir).as_posix())
    checkpoints = [
        path.relative_to(output_dir).as_posix()
        for path in output_dir.glob("checkpoint-*")
    ]
    if forbidden or checkpoints:
        raise RuntimeError(
            "Final artifact is not weights-only: "
            f"forbidden_files={sorted(forbidden)}, checkpoints={sorted(checkpoints)}"
        )
    if require_complete:
        required = {
            "config.json",
            "tokenizer_config.json",
            "trainer_state.json",
            "training_args.bin",
            "training_args.json",
            "wandb_run.url",
            "basis_set.json",
            "training_manifest.json",
        }
        missing = sorted(name for name in required if not (output_dir / name).is_file())
        has_weights = (
            any(output_dir.glob("*.safetensors"))
            or (output_dir / "pytorch_model.bin").is_file()
        )
        if missing or not has_weights:
            raise RuntimeError(
                "Final artifact is incomplete: "
                f"missing={missing}, has_model_weights={has_weights}"
            )


def _save_final_artifact(
    *,
    trainer: Any,
    training_args: Any,
    basis: BasisSet,
    manifest: Mapping[str, Any],
    output_dir: Path,
    wandb_url: str,
) -> None:
    trainer.save_model(str(output_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    _write_json(output_dir / "training_args.json", training_args.to_dict())
    _write_json(output_dir / "basis_set.json", _basis_manifest(basis))
    _write_json(output_dir / "training_manifest.json", manifest)
    _write_wandb_link(output_dir / "wandb_run.url", wandb_url)
    assert_weights_only_artifact(output_dir, require_complete=True)


def _train(
    *,
    config: ChallengerConfig,
    basis: BasisSet,
    corpus: PreparedCorpus,
    base_model: BaseModelSpec,
    repo_root: Path,
    source_code: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform one GPU-only SFT run; all heavy imports are below dry-run."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTTrainer

    import wandb

    if not torch.cuda.is_available():
        raise RuntimeError("Basis challenger training requires a CUDA GPU")
    if config.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new artifact path: "
            f"{config.output_dir}"
        )

    _require_unchanged_source(repo_root, source_code)
    validate_personal_wandb_entity(config.wandb_entity)
    set_seed(config.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[config.precision]

    run = wandb.init(
        entity=PERSONAL_WANDB_ENTITY,
        project=config.wandb_project,
        name=config.wandb_run_name,
        tags=["basis-challenger", "sft", basis.basis_set_id],
        config={
            "basis_set_id": basis.basis_set_id,
            "basis_set_digest": basis.digest,
            "source_corpus": dict(corpus.source),
            "base_model": base_model.to_dict(),
            **_hyperparameters(config),
        },
    )
    try:
        wandb_url = _validate_wandb_url(str(run.url))
        tokenizer = AutoTokenizer.from_pretrained(
            base_model.resolved,
            revision=base_model.revision,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model.resolved,
            revision=base_model.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        training_args = build_sft_config(config, report_to_wandb=True)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list([dict(row) for row in corpus.rows]),
            processing_class=tokenizer,
        )
        train_result = trainer.train()
        state = {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "train_metrics": dict(train_result.metrics),
        }
        _require_unchanged_source(repo_root, source_code)
        manifest = build_training_manifest(
            config=config,
            basis=basis,
            corpus=corpus,
            base_model=base_model,
            repo_root=repo_root,
            status="complete",
            wandb_url=wandb_url,
            trainer_state=state,
            source_code=source_code,
        )
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _save_final_artifact(
            trainer=trainer,
            training_args=training_args,
            basis=basis,
            manifest=manifest,
            output_dir=config.output_dir,
            wandb_url=wandb_url,
        )
        run.summary.update(
            {
                "artifact_path": str(config.output_dir.resolve()),
                "basis_set_id": basis.basis_set_id,
                "basis_set_digest": basis.digest,
                "selected_records": corpus.selected_records,
            }
        )
        return manifest
    finally:
        run.finish()


def run_challenger(config: ChallengerConfig) -> dict[str, Any]:
    """Validate lineage and either return a preview or train the challenger."""
    _validate_config(config)
    repo_root = Path(__file__).resolve().parents[1]
    source_code = _git_identity(repo_root)
    basis, corpus = prepare_training_corpus(config)
    base_model = resolve_base_model(config.base_model, config.base_model_revision)
    _require_unchanged_source(repo_root, source_code)
    if config.verified_results is not None:
        result_model = corpus.source["verified_results"]["solver_model"]
        if (
            result_model.get("canonical_hf_repo") != base_model.canonical_hf_repo
            or result_model.get("canonical_hf_revision")
            != base_model.canonical_hf_revision
        ):
            raise ValueError(
                "Training base model does not match the verified solver checkpoint"
            )
        result_tree_digest = result_model.get("local_tree_digest")
        if (
            base_model.local_tree_digest is not None
            and result_tree_digest != base_model.local_tree_digest
        ):
            raise ValueError(
                "Training base local tree differs from the verified solver checkpoint"
            )
    if config.dry_run:
        manifest = build_training_manifest(
            config=config,
            basis=basis,
            corpus=corpus,
            base_model=base_model,
            repo_root=repo_root,
            status="dry-run",
            source_code=source_code,
        )
        manifest["preview"] = {
            "examples": [dict(row) for row in corpus.rows],
            "note": "No model, CUDA runtime, trainer, or W&B run was loaded.",
        }
        return manifest
    return _train(
        config=config,
        basis=basis,
        corpus=corpus,
        base_model=base_model,
        repo_root=repo_root,
        source_code=source_code,
    )


def _parse_args(argv: Sequence[str] | None = None) -> ChallengerConfig:
    parser = argparse.ArgumentParser(
        description="Completion-only SFT continuation for an immutable basis set"
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis-set-id", default="default")
    parser.add_argument("--splits", nargs="+", default=["discovery"])
    parser.add_argument("--base-model", default=str(DEFAULT_LOCAL_CHECKPOINT))
    parser.add_argument("--base-model-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--wandb-entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=8)
    parser.add_argument(
        "--verified-results",
        type=Path,
        help=(
            "Train on re-executed successful programs from a pilot "
            "results.jsonl.gz and its sibling aggregate.json"
        ),
    )
    arguments = parser.parse_args(argv)
    return ChallengerConfig(
        corpus_dir=arguments.corpus_dir,
        output_dir=arguments.output_dir,
        basis_set_id=arguments.basis_set_id,
        splits=tuple(arguments.splits),
        base_model=arguments.base_model,
        base_model_revision=arguments.base_model_revision,
        num_train_epochs=arguments.num_train_epochs,
        learning_rate=arguments.learning_rate,
        per_device_train_batch_size=arguments.per_device_train_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        max_length=arguments.max_length,
        warmup_ratio=arguments.warmup_ratio,
        weight_decay=arguments.weight_decay,
        logging_steps=arguments.logging_steps,
        seed=arguments.seed,
        max_train_records=arguments.max_train_records,
        precision=arguments.precision,
        wandb_entity=arguments.wandb_entity,
        wandb_project=arguments.wandb_project,
        wandb_run_name=arguments.wandb_run_name,
        dry_run=arguments.dry_run,
        dry_run_samples=arguments.dry_run_samples,
        verified_results=arguments.verified_results,
    )


def main(argv: Sequence[str] | None = None) -> None:
    manifest = run_challenger(_parse_args(argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
