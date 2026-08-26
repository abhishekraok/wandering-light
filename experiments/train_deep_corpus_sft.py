"""Strict completion-only SFT on the certified deep corpus.

The corpus split is part of the experiment, not a convenience to be recreated:
``discovery`` is the only training split, ``validation`` is the only split made
available to the trainer for evaluation, and ``test`` remains held out for the
separate execution-based evaluator.  Before either a dry run or a real run this
module verifies the immutable manifest and basis digests, every split file, every
recorded witness, and the root-level split boundary.

``--dry-run`` deliberately stops before importing the model stack or W&B.  A real
run requires CUDA, an immutable pretrained Hugging Face repository/revision pair,
and an explicitly named W&B run in the owner's personal entity.  It creates no
intermediate checkpoints and writes one weights-only final artifact with complete
source, corpus, task-selection, basis, model, trainer, and W&B lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from wandering_light.basis_set import (
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.llm_utils import generate_train_prompt
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.basis_set import BasisSet
    from wandering_light.function_def import FunctionDef


SCHEMA_VERSION = 1
PERSONAL_WANDB_ENTITY = "abhishekraok-na"
DEFAULT_WANDB_PROJECT = "wandering-light-deep-corpus-v1"
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v1")
# Fine-tune an immutable pretrained sub-0.5B base. Prior task-relative checkpoints
# may have been trained against a different basis and are intentionally rejected.
DEFAULT_BASE_MODEL_REPO = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_BASE_MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
PRETRAINED_MODEL_LOADER = "transformers.AutoModelForCausalLM.from_pretrained"
APPROVED_BASE_MODEL_REVISIONS = {
    "facebook/opt-125m": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    DEFAULT_BASE_MODEL_REPO: DEFAULT_BASE_MODEL_REVISION,
}
DEFAULT_MAX_LENGTH = 2048
MAX_TOKEN_LENGTH_EXCLUSION_FRACTION = 0.001
TRAIN_SPLIT = "discovery"
VALIDATION_SPLIT = "validation"
HELD_OUT_SPLIT = "test"
REQUIRED_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, HELD_OUT_SPLIT)

_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_WANDB_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_CHECKPOINT_NAMES = frozenset(
    {"optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth"}
)


@dataclass(frozen=True)
class BaseModelSpec:
    """One immutable pretrained Hugging Face base-model identity."""

    repo_id: str
    revision: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": "huggingface",
            "repo_id": self.repo_id,
            "revision": self.revision,
            "weight_initialization": "pretrained",
            "model_loader": PRETRAINED_MODEL_LOADER,
        }


@dataclass(frozen=True)
class DeepCorpusSFTConfig:
    """Configuration shared by validation-only and actual SFT runs."""

    output_dir: Path
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO
    base_model_revision: str = DEFAULT_BASE_MODEL_REVISION
    min_train_distance: int = 1
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = DEFAULT_MAX_LENGTH
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    logging_steps: int = 100
    seed: int = 42
    max_train_records: int | None = None
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_run_name: str | None = None
    dry_run: bool = False
    dry_run_samples: int = 8


@dataclass(frozen=True)
class PreparedDeepCorpus:
    """Verified trainer rows and content-addressed task provenance."""

    train_rows: tuple[Mapping[str, Any], ...]
    validation_rows: tuple[Mapping[str, Any], ...]
    total_verified_records: int
    planned_train_records: int
    planned_validation_records: int
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
    """Prevent an overnight run from writing into a work organization."""
    if entity.strip().lower() != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            "Deep-corpus W&B writes are restricted to the personal entity "
            f"{PERSONAL_WANDB_ENTITY!r}; got {entity!r}"
        )
    return PERSONAL_WANDB_ENTITY


def resolve_base_model(repo_id: str, revision: str) -> BaseModelSpec:
    """Validate an approved foundation-model identity without using the network."""
    if not isinstance(repo_id, str) or not _HF_REPO_RE.fullmatch(repo_id):
        raise ValueError(
            "The base model must be an exact Hugging Face owner/repository ID; "
            f"got {repo_id!r}"
        )
    if not isinstance(revision, str) or not _HF_COMMIT_RE.fullmatch(revision):
        raise ValueError(
            "The base model revision must be an immutable lowercase 40-hex commit "
            f"SHA; got {revision!r}"
        )
    if APPROVED_BASE_MODEL_REVISIONS.get(repo_id) != revision:
        raise ValueError(
            "Base model is not an approved foundation-model revision for issue #50; "
            "task-relative checkpoints are forbidden"
        )
    return BaseModelSpec(repo_id=repo_id, revision=revision)


def ensure_corpus(corpus_dir: Path) -> Path:
    """Fetch missing split payloads from the manifest's immutable Hub revision."""

    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deep-corpus manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Deep-corpus manifest has no split metadata")
    missing = [
        corpus_dir / metadata["path"]
        for metadata in splits.values()
        if isinstance(metadata, Mapping)
        and not (corpus_dir / metadata["path"]).is_file()
    ]
    if missing:
        fetch_corpus(manifest_path, destination=corpus_dir)
    still_missing = [path for path in missing if not path.is_file()]
    if still_missing:
        raise FileNotFoundError(
            "Deep-corpus payload fetch did not materialize required files: "
            f"{[str(path) for path in still_missing]}"
        )
    return corpus_dir


def _validate_config(config: DeepCorpusSFTConfig) -> None:
    validate_personal_wandb_entity(config.wandb_entity)
    if config.min_train_distance not in {1, 4}:
        raise ValueError("min_train_distance must be exactly 1 or 4")
    if config.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if config.per_device_eval_batch_size <= 0:
        raise ValueError("per_device_eval_batch_size must be positive")
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
    if not config.wandb_project or not _WANDB_PROJECT_RE.fullmatch(
        config.wandb_project
    ):
        raise ValueError("wandb_project must be a non-empty W&B project slug")
    if not config.dry_run and (
        not isinstance(config.wandb_run_name, str) or not config.wandb_run_name.strip()
    ):
        raise ValueError(
            "An actual training run requires --wandb-run-name for recoverable "
            "personal W&B lineage"
        )


def _functions_by_id(
    available_functions: FunctionDefSet,
) -> dict[str, FunctionDef]:
    functions: dict[str, FunctionDef] = {}
    for function in available_functions:
        metadata = function.metadata or {}
        function_id = metadata.get("basis_function_id")
        if not isinstance(function_id, str) or not function_id:
            raise ValueError(f"Basis function {function.name!r} has no stable ID")
        if function_id in functions:
            raise ValueError(f"Duplicate basis function ID {function_id!r}")
        functions[function_id] = function
    return functions


def _training_row_for_record(
    record: BasisTaskRecord,
    *,
    functions_by_id: Mapping[str, FunctionDef],
    executor: Executor,
) -> dict[str, Any]:
    metadata = record.metadata
    distance = metadata.get("certified_distance")
    if type(distance) is not int or distance <= 0:
        raise ValueError(f"Task {record.task_id} has no valid certified distance")
    if record.witness_length != distance:
        raise ValueError(
            f"Task {record.task_id} witness length does not match its certified "
            "distance"
        )
    root_digest = metadata.get("root_digest")
    if not isinstance(root_digest, str) or not _SHA256_RE.fullmatch(root_digest):
        raise ValueError(f"Task {record.task_id} has no valid root digest")

    witness: list[FunctionDef] = []
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
    execution = executor.execute_trajectory(
        TrajectorySpec(input_list=record.input_value, function_defs=solution)
    )
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

    example = generate_train_prompt(
        input_list=record.input_value,
        output_list=record.output_value,
        # The palette is intentionally absent from this task format.  Passing an
        # empty set also avoids eagerly formatting a palette that is discarded.
        available_functions=FunctionDefSet(),
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
        "certified_distance": distance,
        "root_digest": root_digest,
        "root_index": metadata.get("root_index"),
        "witness_function_ids": list(record.witness_function_ids),
        "witness_function_names": list(record.witness_function_names),
        "generator": record.generator,
        "seed": record.seed,
        "source_index": record.source_index,
    }


def training_row_for_record(
    record: BasisTaskRecord,
    available_functions: FunctionDefSet,
) -> dict[str, Any]:
    """Return the canonical prompt/completion row for one certified task.

    This public, side-effect-free helper is shared with evaluation so fine-tuning
    and generation cannot silently drift to different prompt formats.  It also
    validates the exact witness ID/name mapping and executes the witness before
    returning a label.
    """
    # Executor increments per-definition demand counters.  Run against deep
    # copies so this prompt helper remains pure from its caller's perspective.
    runtime_functions = FunctionDefSet(
        [function.model_copy(deep=True) for function in available_functions]
    )
    functions_by_id = _functions_by_id(runtime_functions)
    return _training_row_for_record(
        record,
        functions_by_id=functions_by_id,
        executor=Executor(runtime_functions),
    )


def filter_rows_for_complete_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    max_excluded_fraction: float = MAX_TOKEN_LENGTH_EXCLUSION_FRACTION,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    """Exclude only examples that cannot retain their complete SFT label.

    TRL appends EOS and then right-truncates prompt-completion examples.  A row
    longer than the model context would therefore receive a partial or empty
    completion mask without this explicit gate.  The tiny excluded tail is
    content-addressed in the final manifest, and a material exclusion rate is a
    hard error rather than a silent change to the corpus.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not 0 <= max_excluded_fraction <= 1:
        raise ValueError("max_excluded_fraction must be between zero and one")
    eos_token = tokenizer.eos_token
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("Completion-only SFT requires a tokenizer EOS token")

    retained: list[Mapping[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    observed_lengths: list[int] = []
    batch_size = 512
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            str(row["prompt"])
            + str(row["completion"])
            + ("" if str(row["completion"]).endswith(eos_token) else eos_token)
            for row in batch
        ]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        for row, token_ids in zip(batch, encoded, strict=True):
            length = len(token_ids)
            observed_lengths.append(length)
            if length <= max_length:
                retained.append(row)
                continue
            excluded.append(
                {
                    "task_id": row["task_id"],
                    "split": row["split"],
                    "certified_distance": row["certified_distance"],
                    "tokens_with_eos": length,
                }
            )

    excluded_fraction = len(excluded) / len(rows) if rows else 0.0
    if not retained:
        raise ValueError("Token-length filtering retained no examples")
    if excluded_fraction > max_excluded_fraction:
        raise ValueError(
            "Token-length filtering would materially change the selected corpus: "
            f"excluded={len(excluded)}/{len(rows)} ({excluded_fraction:.3%}), "
            f"allowed={max_excluded_fraction:.3%}; increase --max-length"
        )
    retained_task_ids = [str(row["task_id"]) for row in retained]
    return tuple(retained), {
        "policy": (
            "tokenize prompt+completion+EOS exactly as TRL and exclude rows whose "
            "complete label exceeds max_length"
        ),
        "max_length": max_length,
        "selected_records": len(rows),
        "retained_records": len(retained),
        "excluded_records": len(excluded),
        "excluded_fraction": excluded_fraction,
        "max_observed_tokens": max(observed_lengths, default=0),
        "retained_task_ids_digest": _sha256_bytes(
            _canonical_json(retained_task_ids).encode("utf-8")
        ),
        "excluded": excluded,
        "complete_labels_only": True,
    }


def _validate_deep_record_metadata(
    record: BasisTaskRecord,
    *,
    manifest: Mapping[str, Any],
) -> tuple[int, str, int]:
    from experiments import generate_deep_corpus as deep

    if record.generator != manifest.get("generator"):
        raise ValueError(
            f"Task {record.task_id} generator differs from the corpus manifest"
        )
    if record.seed != manifest.get("seed"):
        raise ValueError(f"Task {record.task_id} seed differs from the manifest")
    metadata = record.metadata
    distance = metadata.get("certified_distance")
    if type(distance) is not int or distance <= 0:
        raise ValueError(f"Task {record.task_id} has an invalid certified distance")
    if distance != record.witness_length:
        raise ValueError(f"Task {record.task_id} witness length mismatch")
    if metadata.get("certification") not in {
        deep.CERTIFICATION_COMPLETE,
        deep.CERTIFICATION_FRONTIER,
    }:
        raise ValueError(f"Task {record.task_id} has invalid certification metadata")
    root_index = metadata.get("root_index")
    if type(root_index) is not int or root_index < 0:
        raise ValueError(f"Task {record.task_id} has an invalid root index")
    root_digest = metadata.get("root_digest")
    expected_root_digest = _sha256_bytes(record.input.encode("utf-8"))
    if root_digest != expected_root_digest:
        raise ValueError(
            f"Task {record.task_id} root digest does not identify its input root"
        )
    return root_index, root_digest, distance


def _verify_root_partition(
    manifest: Mapping[str, Any],
    records: Sequence[BasisTaskRecord],
) -> None:
    """Check the manifest's root-level split assignment without re-splitting."""
    from experiments import generate_deep_corpus as deep

    if manifest.get("schema_version") != deep.PIPELINE_SCHEMA_VERSION:
        raise ValueError("Corpus manifest has an unsupported schema version")
    if manifest.get("generator") != deep.GENERATOR_NAME:
        raise ValueError("Corpus manifest is not deep-corpus-v1 generation output")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(REQUIRED_SPLITS):
        raise ValueError(
            "Deep-corpus training requires exactly discovery, validation, and test"
        )
    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, str) or not split_policy.startswith("by root:"):
        raise ValueError("Corpus manifest does not declare a by-root split policy")
    if manifest.get("global_task_count") != len(records):
        raise ValueError("Corpus global task count does not match loaded records")

    task_ids: set[str] = set()
    roots_to_splits: defaultdict[int, set[str]] = defaultdict(set)
    digests_to_splits: defaultdict[str, set[str]] = defaultdict(set)
    root_to_digest: dict[int, str] = {}
    digest_to_root: dict[str, int] = {}
    roots_by_split: defaultdict[str, set[int]] = defaultdict(set)
    digests_by_split: defaultdict[str, set[str]] = defaultdict(set)
    source_indices: defaultdict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.task_id in task_ids:
            raise ValueError(f"Duplicate corpus task ID {record.task_id}")
        task_ids.add(record.task_id)
        if record.split not in REQUIRED_SPLITS:
            raise ValueError(f"Task {record.task_id} has an unknown split")
        root_index, root_digest, _ = _validate_deep_record_metadata(
            record, manifest=manifest
        )
        previous_digest = root_to_digest.setdefault(root_index, root_digest)
        previous_root = digest_to_root.setdefault(root_digest, root_index)
        if previous_digest != root_digest or previous_root != root_index:
            raise ValueError("Corpus root index and root digest are not one-to-one")
        roots_to_splits[root_index].add(record.split)
        digests_to_splits[root_digest].add(record.split)
        roots_by_split[record.split].add(root_index)
        digests_by_split[record.split].add(root_digest)
        source_indices[record.split].append(record.source_index)

    leaking_indices = sorted(
        root for root, assigned in roots_to_splits.items() if len(assigned) != 1
    )
    leaking_digests = sorted(
        digest for digest, assigned in digests_to_splits.items() if len(assigned) != 1
    )
    if leaking_indices or leaking_digests:
        raise ValueError(
            "Corpus roots leak across splits: "
            f"root_indices={leaking_indices}, root_digests={leaking_digests}"
        )

    manifest_split_roots = manifest.get("split_roots")
    if not isinstance(manifest_split_roots, Mapping):
        raise ValueError("Corpus manifest has no exact split_roots mapping")
    for split in REQUIRED_SPLITS:
        metadata = splits[split]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Corpus split metadata is invalid for {split!r}")
        if source_indices[split] != list(range(len(source_indices[split]))):
            raise ValueError(f"Corpus source indices are not ordered for {split!r}")
        expected_roots = manifest_split_roots.get(split)
        if not isinstance(expected_roots, list) or any(
            type(root) is not int for root in expected_roots
        ):
            raise ValueError(f"Manifest split_roots is invalid for {split!r}")
        if sorted(expected_roots) != sorted(roots_by_split[split]):
            raise ValueError(f"Manifest split_roots mismatch for {split!r}")
        if metadata.get("roots") != len(roots_by_split[split]):
            raise ValueError(f"Manifest root count mismatch for {split!r}")
        if metadata.get("root_digests") != len(digests_by_split[split]):
            raise ValueError(f"Manifest root digest count mismatch for {split!r}")


def _task_identity(
    records: Sequence[BasisTaskRecord],
    *,
    role: str,
    selection: str,
) -> dict[str, Any]:
    task_ids = [record.task_id for record in records]
    root_digests = sorted({record.metadata["root_digest"] for record in records})
    return {
        "role": role,
        "split": records[0].split if records else None,
        "selection": selection,
        "records": len(records),
        "roots": len(root_digests),
        "by_certified_distance": {
            str(distance): count
            for distance, count in sorted(
                Counter(
                    record.metadata["certified_distance"] for record in records
                ).items()
            )
        },
        "ordered_task_ids_digest": _sha256_bytes(
            _canonical_json(task_ids).encode("utf-8")
        ),
        "task_id_set_digest": _sha256_bytes(
            _canonical_json(sorted(task_ids)).encode("utf-8")
        ),
        "root_digests_digest": _sha256_bytes(
            _canonical_json(root_digests).encode("utf-8")
        ),
        "generators": dict(sorted(Counter(r.generator for r in records).items())),
        "seeds": sorted({record.seed for record in records}),
    }


def _source_corpus_identity(
    corpus_dir: Path,
    manifest: Mapping[str, Any],
    *,
    verification: Mapping[str, Any],
    train_records: Sequence[BasisTaskRecord],
    validation_records: Sequence[BasisTaskRecord],
    test_records: Sequence[BasisTaskRecord],
    min_train_distance: int,
    max_train_records: int | None,
) -> dict[str, Any]:
    root = corpus_dir.resolve()
    manifest_path = root / "manifest.json"
    split_files: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        metadata = manifest["splits"][split]
        path = root / metadata["path"]
        split_files[split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "records": metadata["size"],
        }
    hub = manifest.get("hub")
    if hub is not None:
        if not isinstance(hub, Mapping):
            raise ValueError("Corpus hub provenance must be a JSON object")
        revision = hub.get("revision")
        if not isinstance(revision, str) or not _HF_COMMIT_RE.fullmatch(revision):
            raise ValueError("Corpus Hub revision is not an immutable 40-hex SHA")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "manifest_schema_version": manifest["schema_version"],
        "generator": manifest["generator"],
        "generator_seed": manifest["seed"],
        "generator_config": dict(manifest["config"]),
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "distance_semantics": manifest["distance_semantics"],
        "split_policy": manifest["split_policy"],
        "hub": dict(hub) if isinstance(hub, Mapping) else None,
        "split_files": split_files,
        "verification": {
            "records": verification["records"],
            "distance_histogram": dict(verification["distance_histogram"]),
            "all_witnesses_reexecuted": not verification["witness_failures"],
            "root_split_leaks": list(verification["roots_leaking_across_splits"]),
            "ok": verification["ok"],
        },
        "task_roles": {
            TRAIN_SPLIT: _task_identity(
                train_records,
                role="training-only",
                selection=(
                    f"file-order records with certified_distance >= "
                    f"{min_train_distance}; max_train_records={max_train_records}"
                ),
            ),
            VALIDATION_SPLIT: _task_identity(
                validation_records,
                role="trainer-evaluation-and-tuning-only",
                selection="all records in original file order",
            ),
            HELD_OUT_SPLIT: _task_identity(
                test_records,
                role="held-out-evaluation-only; never passed to trainer",
                selection="all records retained outside training",
            ),
        },
    }


def prepare_training_corpus(
    config: DeepCorpusSFTConfig,
    *,
    preview_only: bool | None = None,
) -> tuple[BasisSet, PreparedDeepCorpus]:
    """Verify the complete corpus, then materialize discovery/validation rows."""
    from experiments import generate_deep_corpus as deep

    _validate_config(config)
    ensure_corpus(config.corpus_dir)
    manifest, records = deep.load_corpus(config.corpus_dir, splits=REQUIRED_SPLITS)
    basis = load_basis_set(manifest["basis_set_id"])
    require_reproducible_basis_runtime(basis)
    if manifest["basis_set_id"] != basis.basis_set_id:
        raise ValueError("Corpus manifest basis-set ID does not match the loaded basis")
    if manifest["basis_set_digest"] != basis.digest:
        raise ValueError(
            "Corpus manifest basis-set digest does not match the loaded basis"
        )

    # This is intentionally the complete verifier: it reloads all three files,
    # re-executes every witness, and checks root digests do not cross splits.
    verification = deep.verify_corpus(
        config.corpus_dir,
        recertify=0,
        seed=manifest["seed"],
    )
    if not verification["ok"]:
        raise ValueError(
            "Deep corpus failed witness/root-split verification: "
            f"witness_failures={verification['witness_failures'][:8]}, "
            "root_leaks="
            f"{verification['roots_leaking_across_splits'][:8]}"
        )
    _verify_root_partition(manifest, records)

    by_split: dict[str, list[BasisTaskRecord]] = {
        split: [] for split in REQUIRED_SPLITS
    }
    for record in records:
        by_split[record.split].append(record)
    train_records = [
        record
        for record in by_split[TRAIN_SPLIT]
        if record.metadata["certified_distance"] >= config.min_train_distance
    ]
    if config.max_train_records is not None:
        train_records = train_records[: config.max_train_records]
    validation_records = by_split[VALIDATION_SPLIT]
    test_records = by_split[HELD_OUT_SPLIT]
    if not train_records:
        raise ValueError("The training-distance filter selected no discovery records")
    if not validation_records:
        raise ValueError("The deep corpus has no validation records")
    if not test_records:
        raise ValueError("The deep corpus has no held-out test records")

    source = _source_corpus_identity(
        config.corpus_dir,
        manifest,
        verification=verification,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        min_train_distance=config.min_train_distance,
        max_train_records=config.max_train_records,
    )
    use_preview = config.dry_run if preview_only is None else preview_only
    materialized_train = train_records
    materialized_validation = validation_records
    if use_preview:
        materialized_train = materialized_train[: config.dry_run_samples]
        materialized_validation = materialized_validation[: config.dry_run_samples]

    available_functions = basis.as_function_set()
    functions_by_id = _functions_by_id(available_functions)
    executor = Executor(available_functions)
    train_rows = tuple(
        _training_row_for_record(
            record,
            functions_by_id=functions_by_id,
            executor=executor,
        )
        for record in materialized_train
    )
    validation_rows = tuple(
        _training_row_for_record(
            record,
            functions_by_id=functions_by_id,
            executor=executor,
        )
        for record in materialized_validation
    )
    return basis, PreparedDeepCorpus(
        train_rows=train_rows,
        validation_rows=validation_rows,
        total_verified_records=len(records),
        planned_train_records=len(train_records),
        planned_validation_records=len(validation_records),
        source=source,
    )


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_identity(
    repo_root: Path,
    *,
    ignored_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Content-address both committed and current uncommitted training source."""
    ignored = tuple(path.resolve() for path in ignored_paths)

    def run_git(*arguments: str, text: bool = True) -> str | bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
        )
        return completed.stdout

    try:
        commit = str(run_git("rev-parse", "HEAD")).strip()
        if not _GIT_COMMIT_RE.fullmatch(commit):
            raise ValueError(f"Invalid git commit returned by git: {commit!r}")
        tracked_diff = str(run_git("diff", "--binary", "HEAD"))
        raw_untracked = bytes(
            run_git("ls-files", "--others", "--exclude-standard", "-z", text=False)
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not record the source git identity") from error

    untracked_files: list[dict[str, Any]] = []
    for raw_path in sorted(filter(None, raw_untracked.split(b"\0"))):
        try:
            relative_path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("Untracked git path is not valid UTF-8") from error
        path = (repo_root / relative_path).resolve()
        if any(_is_under(path, ignored_path) for ignored_path in ignored):
            continue
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
        "dirty": bool(tracked_diff or untracked_files),
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
    repo_root: Path,
    expected: Mapping[str, Any],
    *,
    ignored_paths: Sequence[Path],
) -> None:
    if _git_identity(repo_root, ignored_paths=ignored_paths) != dict(expected):
        raise RuntimeError(
            "Repository source changed after training validation started; discard "
            "this run and relaunch from a stable tree"
        )


def _hyperparameters(config: DeepCorpusSFTConfig) -> dict[str, Any]:
    return {
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "held_out_split": HELD_OUT_SPLIT,
        "min_train_distance": config.min_train_distance,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
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
        "gradient_checkpointing": config.gradient_checkpointing,
        "attention_implementation": "sdpa",
        "completion_only_loss": True,
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "save_only_model": True,
        "full_determinism": True,
        "max_train_records": config.max_train_records,
        "resumption_supported": False,
    }


def build_training_manifest(
    *,
    config: DeepCorpusSFTConfig,
    basis: BasisSet,
    corpus: PreparedDeepCorpus,
    base_model: BaseModelSpec,
    repo_root: Path,
    status: str,
    source_code: Mapping[str, Any],
    started_at_utc: str,
    wandb_url: str | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    execution_environment: Mapping[str, Any] | None = None,
    tokenization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable lineage record saved with final weights."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at_utc,
        "basis_set": basis.identity_dict(),
        "source_corpus": dict(corpus.source),
        "base_model": base_model.to_dict(),
        "hyperparameters": _hyperparameters(config),
        "source_code": dict(source_code),
        "wandb": {
            "entity": PERSONAL_WANDB_ENTITY,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "url": wandb_url,
        },
        "trainer_state": dict(trainer_state or {}),
        "execution_environment": dict(execution_environment or {}),
        "tokenization": dict(tokenization or {}),
        "repo_root": str(repo_root),
    }


def build_sft_config(config: DeepCorpusSFTConfig, *, report_to_wandb: bool):
    """Construct deterministic completion-only TRL arguments with no saves."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(config.output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
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
        eval_strategy="epoch",
        save_strategy="no",
        save_only_model=True,
        save_safetensors=True,
        completion_only_loss=True,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=True,
        dataloader_num_workers=0,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        report_to=["wandb"] if report_to_wandb else [],
        run_name=config.wandb_run_name if report_to_wandb else None,
        push_to_hub=False,
        packing=False,
        include_num_input_tokens_seen=True,
    )


def validate_wandb_url(url: str, *, project: str) -> str:
    """Require the canonical hosted URL for this personal entity/project."""
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_wandb_link(path: Path, url: str) -> None:
    path.write_text(f"[InternetShortcut]\nURL={url}\n", encoding="utf-8")


def assert_weights_only_artifact(
    output_dir: Path,
    *,
    require_complete: bool = False,
) -> None:
    """Reject resumable state and intermediate checkpoints anywhere in output."""
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
        for path in output_dir.rglob("checkpoint-*")
    ]
    if forbidden or checkpoints:
        raise RuntimeError(
            "Final artifact is not weights-only: "
            f"forbidden_files={sorted(forbidden)}, "
            f"checkpoints={sorted(checkpoints)}"
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
            "corpus_manifest.json",
            "training_manifest.json",
        }
        missing = sorted(name for name in required if not (output_dir / name).is_file())
        has_weights = any(output_dir.glob("*.safetensors")) or any(
            path.name.startswith("pytorch_model") and path.suffix == ".bin"
            for path in output_dir.iterdir()
            if path.is_file()
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
    corpus_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    output_dir: Path,
    wandb_url: str,
) -> None:
    trainer.save_model(str(output_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    _write_json(output_dir / "training_args.json", training_args.to_dict())
    _write_json(output_dir / "basis_set.json", basis.to_manifest())
    _write_json(output_dir / "corpus_manifest.json", corpus_manifest)
    _write_json(output_dir / "training_manifest.json", training_manifest)
    _write_wandb_link(output_dir / "wandb_run.url", wandb_url)
    assert_weights_only_artifact(output_dir, require_complete=True)


def _cuda_environment(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": f"cuda:{index}",
        "accelerator": {
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        },
        "software": {
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }


def _train(
    *,
    config: DeepCorpusSFTConfig,
    basis: BasisSet,
    corpus: PreparedDeepCorpus,
    base_model: BaseModelSpec,
    repo_root: Path,
    source_code: Mapping[str, Any],
    ignored_source_paths: Sequence[Path],
    started_at_utc: str,
) -> dict[str, Any]:
    """Perform one CUDA-only run; all heavyweight imports remain below dry-run."""
    import torch
    import wandb
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Deep-corpus SFT requires an actual CUDA GPU")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but the CUDA device lacks bf16 support")
    if config.output_dir.exists():
        raise FileExistsError(
            "Output directory already exists; choose a new final artifact path: "
            f"{config.output_dir}"
        )
    if len(corpus.train_rows) != corpus.planned_train_records:
        raise RuntimeError("Actual training did not materialize every selected task")
    if len(corpus.validation_rows) != corpus.planned_validation_records:
        raise RuntimeError("Actual training did not materialize all validation tasks")

    _require_unchanged_source(
        repo_root,
        source_code,
        ignored_paths=ignored_source_paths,
    )
    set_seed(config.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[config.precision]
    execution_environment = _cuda_environment(torch)

    model_config = AutoConfig.from_pretrained(
        base_model.repo_id,
        revision=base_model.revision,
    )
    model_context = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(model_context, int) or model_context <= 0:
        raise ValueError("Base model does not declare a positive context length")
    if config.max_length > model_context:
        raise ValueError(
            f"max_length={config.max_length} exceeds the base model context "
            f"length {model_context}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model.repo_id,
        revision=base_model.revision,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Base tokenizer has neither a pad token nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_rows, train_tokenization = filter_rows_for_complete_labels(
        corpus.train_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    validation_rows, validation_tokenization = filter_rows_for_complete_labels(
        corpus.validation_rows,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    tokenization = {
        "model_max_position_embeddings": model_context,
        "train": train_tokenization,
        "validation": validation_tokenization,
    }

    wandb_work_dir = ignored_source_paths[1]
    wandb_work_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=PERSONAL_WANDB_ENTITY,
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode="online",
        dir=str(wandb_work_dir),
        tags=[
            "deep-corpus-v1",
            "completion-sft",
            f"min-distance-{config.min_train_distance}",
            basis.basis_set_id,
        ],
        config={
            "basis_set": basis.identity_dict(),
            "source_corpus": dict(corpus.source),
            "base_model": base_model.to_dict(),
            "tokenization": tokenization,
            **_hyperparameters(config),
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create an online run")
    try:
        wandb_url = validate_wandb_url(str(run.url), project=config.wandb_project)
        model = AutoModelForCausalLM.from_pretrained(
            base_model.repo_id,
            revision=base_model.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        training_args = build_sft_config(config, report_to_wandb=True)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list([dict(row) for row in train_rows]),
            eval_dataset=Dataset.from_list([dict(row) for row in validation_rows]),
            processing_class=tokenizer,
        )
        train_result = trainer.train()
        final_validation_metrics = trainer.evaluate()
        state = {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "train_metrics": dict(train_result.metrics),
            "final_validation_metrics": dict(final_validation_metrics),
            "tokenization": tokenization,
        }
        _require_unchanged_source(
            repo_root,
            source_code,
            ignored_paths=ignored_source_paths,
        )
        training_manifest = build_training_manifest(
            config=config,
            basis=basis,
            corpus=corpus,
            base_model=base_model,
            repo_root=repo_root,
            status="complete",
            source_code=source_code,
            started_at_utc=started_at_utc,
            wandb_url=wandb_url,
            trainer_state=state,
            execution_environment=execution_environment,
            tokenization=tokenization,
        )
        training_manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        corpus_manifest = json.loads(
            (config.corpus_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _save_final_artifact(
            trainer=trainer,
            training_args=training_args,
            basis=basis,
            corpus_manifest=corpus_manifest,
            training_manifest=training_manifest,
            output_dir=config.output_dir,
            wandb_url=wandb_url,
        )
        run.summary.update(
            {
                "artifact_path": str(config.output_dir.resolve()),
                "basis_set_id": basis.basis_set_id,
                "basis_set_digest": basis.digest,
                "corpus_manifest_digest": corpus.source["manifest_digest"],
                "selected_training_records": corpus.planned_train_records,
                "training_records": len(train_rows),
                "selected_validation_records": corpus.planned_validation_records,
                "validation_records": len(validation_rows),
                "token_length_exclusions": (
                    train_tokenization["excluded_records"]
                    + validation_tokenization["excluded_records"]
                ),
                "min_train_distance": config.min_train_distance,
            }
        )
        return training_manifest
    finally:
        run.finish()


def run_training(config: DeepCorpusSFTConfig) -> dict[str, Any]:
    """Verify lineage and either return a dry-run preview or launch CUDA SFT."""
    _validate_config(config)
    started_at_utc = datetime.now(UTC).isoformat()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = config.output_dir.resolve()
    wandb_work_dir = output_dir.parent / f".{output_dir.name}.wandb-work"
    ignored_source_paths = (output_dir, wandb_work_dir)
    source_code = _git_identity(repo_root, ignored_paths=ignored_source_paths)
    basis, corpus = prepare_training_corpus(config)
    base_model = resolve_base_model(
        config.base_model_repo,
        config.base_model_revision,
    )
    _require_unchanged_source(
        repo_root,
        source_code,
        ignored_paths=ignored_source_paths,
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
            started_at_utc=started_at_utc,
        )
        manifest["preview"] = {
            "train_examples": [dict(row) for row in corpus.train_rows],
            "validation_examples": [dict(row) for row in corpus.validation_rows],
            "note": (
                "All corpus files, witnesses, and root splits were verified. No "
                "model, CUDA runtime, trainer, output artifact, or W&B run was "
                "loaded or created."
            ),
        }
        return manifest
    return _train(
        config=config,
        basis=basis,
        corpus=corpus,
        base_model=base_model,
        repo_root=repo_root,
        source_code=source_code,
        ignored_source_paths=ignored_source_paths,
        started_at_utc=started_at_utc,
    )


# Descriptive alias for callers that prefer the experiment name over the generic
# runner verb.
run_deep_corpus_sft = run_training


def _parse_args(argv: Sequence[str] | None = None) -> DeepCorpusSFTConfig:
    parser = argparse.ArgumentParser(
        description=(
            "CUDA completion-only SFT on verified deep-corpus-v1 discovery tasks"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--base-model-repo",
        "--base-model",
        dest="base_model_repo",
        default=DEFAULT_BASE_MODEL_REPO,
    )
    parser.add_argument(
        "--base-model-revision",
        default=DEFAULT_BASE_MODEL_REVISION,
    )
    parser.add_argument("--min-train-distance", type=int, choices=(1, 4), default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--wandb-entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=8)
    arguments = parser.parse_args(argv)
    return DeepCorpusSFTConfig(
        output_dir=arguments.output_dir,
        corpus_dir=arguments.corpus_dir,
        base_model_repo=arguments.base_model_repo,
        base_model_revision=arguments.base_model_revision,
        min_train_distance=arguments.min_train_distance,
        num_train_epochs=arguments.num_train_epochs,
        learning_rate=arguments.learning_rate,
        per_device_train_batch_size=arguments.per_device_train_batch_size,
        per_device_eval_batch_size=arguments.per_device_eval_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        max_length=arguments.max_length,
        warmup_ratio=arguments.warmup_ratio,
        weight_decay=arguments.weight_decay,
        logging_steps=arguments.logging_steps,
        seed=arguments.seed,
        max_train_records=arguments.max_train_records,
        precision=arguments.precision,
        gradient_checkpointing=arguments.gradient_checkpointing,
        wandb_entity=arguments.wandb_entity,
        wandb_project=arguments.wandb_project,
        wandb_run_name=arguments.wandb_run_name,
        dry_run=arguments.dry_run,
        dry_run_samples=arguments.dry_run_samples,
    )


def main(argv: Sequence[str] | None = None) -> None:
    manifest = run_training(_parse_args(argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
