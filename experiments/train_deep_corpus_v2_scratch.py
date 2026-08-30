"""Train issue #54's three causal LMs from pristine random initialization.

The runner consumes the one frozen deep-corpus-v2 tokenizer, uses discovery as
the only training distribution, evaluates loss on validation, and never exposes
the held-out test split to a model.  Pilot runs are fixed to one quarter epoch
over the predeclared learning-rate grid.  Final runs require a validation-only
selection lock and recreate the exact pristine initial state before training.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments import generate_deep_corpus as deep
from experiments.train_deep_corpus_sft import (
    _git_identity,
    _require_unchanged_source,
    _write_json,
    _write_wandb_link,
    assert_weights_only_artifact,
    training_row_for_record,
    validate_wandb_url,
)
from wandering_light.basis_set import load_basis_set, require_reproducible_basis_runtime
from wandering_light.training.deep_corpus_scratch import (
    ARM_SPECS,
    DATA_ORDER_SEED,
    DEFAULT_WANDB_PROJECT,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    FEASIBILITY_HARD_LIMIT_HOURS,
    INITIALIZATION_SEED,
    LR_GRID,
    MAX_FINAL_EPOCHS,
    PERSONAL_WANDB_ENTITY,
    PILOT_EPOCHS,
    TOKENIZER_MAX_LENGTH,
    ResolvedScratchConfig,
    assert_config_created_model,
    build_scratch_study_identity,
    classify_validation_trend,
    configure_deterministic_cuda_environment,
    digest_json,
    exact_model_parameter_count,
    initialize_model_from_config,
    resolve_scratch_config,
    select_pilot_learning_rate,
    validate_final_restart,
)
from wandering_light.training.deep_corpus_tokenizer import (
    DEFAULT_REFERENCE_CORPUS_DIR,
    DETERMINISTIC_PYTHON_HASH_SEED,
    FIXED_SPECIAL_TOKENS,
    TOKENIZER_FILENAMES,
    build_tokenization_audit,
    canonical_corpus_rows,
    require_deterministic_python_hash_seed,
    validate_frozen_tokenizer,
)
from wandering_light.training.deep_corpus_tokenizer import (
    MANIFEST_FILENAME as TOKENIZER_MANIFEST_FILENAME,
)

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.basis_set import BasisSet


SCHEMA_VERSION = 1
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v2")
DEFAULT_TOKENIZER_DIR = Path("wandering_light/training/tokenizers/deep_corpus_v2")
TRAIN_SPLIT = "discovery"
VALIDATION_SPLIT = "validation"
HELD_OUT_SPLIT = "test"
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
# The 360M arm's exact longest batch exceeds one RTX 4090's physical VRAM
# without checkpointing. Non-reentrant checkpointing avoids needless full-layer
# recomputation while keeping the frozen batch and all model/data semantics.
GRADIENT_CHECKPOINTING_ARMS = frozenset({"smollm2-360m-scratch"})
GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}
TRAINING_MANIFEST_FILENAME = "training_manifest.json"
_PENDING_DURABILITY_STATUS = "pending-durable-verification"
_CANONICAL_TRANSIENT_ROOTS = (Path("/tmp"), Path("/var/tmp"))
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_NORMAL_RUNTIME_SCOPE = (
    "exact elapsed time measured at the final durable completion transition, "
    "after W&B summary and finish"
)
_RECOVERY_RUNTIME_SCOPE = (
    "right-censored lower bound combining the known original invocation prefix "
    "and measured recovery"
)
_RECOVERY_OMITTED_INTERVAL_SCOPE = (
    "after the pending manifest's original runtime measurement through recovery start"
)
_RECOVERY_OMITTED_INTERVAL_REASON = (
    "the producer tail and inter-invocation gap were not measured"
)
_RECOVERY_WANDB_REASON = "recovery performs no W&B write or finalization"
_DURABLE_ARTIFACT_REQUIRED_FILES = frozenset(
    {
        "basis_set.json",
        "config.json",
        "corpus_manifest.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        TOKENIZER_MANIFEST_FILENAME,
        "trainer_state.json",
        "training_args.bin",
        "training_args.json",
        "wandb_run.url",
    }
)


@dataclass(frozen=True)
class ScratchRunConfig:
    output_dir: Path
    corpus_dir: Path
    tokenizer_dir: Path
    arm_id: str
    phase: str
    learning_rate: float
    num_train_epochs: float
    durable_output_root: Path | None = None
    reference_corpus_dirs: tuple[Path, ...] = (DEFAULT_REFERENCE_CORPUS_DIR,)
    feasibility_manifest: Path | None = None
    wandb_run_name: str | None = None
    selection_lock: Path | None = None
    dry_run: bool = False
    initialization_seed: int = INITIALIZATION_SEED
    data_order_seed: int = DATA_ORDER_SEED
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT


@dataclass
class _CheckpointCompletionLock:
    checkpoint_dir: Path
    path: Path
    descriptor: int
    held: bool = True


@dataclass(frozen=True)
class PreparedScratchData:
    basis: BasisSet
    corpus_manifest: Mapping[str, Any]
    tokenizer_manifest: Mapping[str, Any]
    train_rows: tuple[Mapping[str, Any], ...]
    validation_rows: tuple[Mapping[str, Any], ...]
    train_audit: Mapping[str, Any]
    validation_audit: Mapping[str, Any]
    source_identity: Mapping[str, Any]


@dataclass(frozen=True)
class WallClockDeadline:
    """One invocation's share of the arm-level 16-hour hard envelope."""

    started_monotonic: float
    allocation_seconds: float
    clock: Any = time.monotonic

    def elapsed_seconds(self) -> float:
        return float(self.clock()) - self.started_monotonic

    def check(self, stage: str) -> None:
        elapsed = self.elapsed_seconds()
        if elapsed >= self.allocation_seconds:
            raise RuntimeError(
                "Scratch training exceeded its arm-level hard-stop allocation: "
                f"stage={stage}, elapsed_seconds={elapsed:.3f}, "
                f"allocation_seconds={self.allocation_seconds:.3f}"
            )


def hard_stop_allocation(arm_gate: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    """Allocate the 16-hour arm envelope across three pilots and one final."""

    if phase not in {"pilot", "final"}:
        raise ValueError("Hard-stop phase must be pilot or final")
    if arm_gate.get("launch_allowed") is not True or arm_gate.get("route") != "direct":
        raise ValueError("Hard-stop allocation requires a direct feasible arm")
    baseline = arm_gate.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Hard-stop allocation has no conservative baseline")
    pilot_run_hours = baseline.get("projected_pilot_run_hours")
    final_hours = baseline.get("projected_final_hours")
    total_hours = baseline.get("projected_pilot_plus_final_hours")
    values = (pilot_run_hours, final_hours, total_hours)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in values
    ):
        raise ValueError("Hard-stop allocation projection is incomplete")
    pilot_run_hours = float(pilot_run_hours)
    final_hours = float(final_hours)
    total_hours = float(total_hours)
    reconstructed = len(LR_GRID) * pilot_run_hours + final_hours
    if not math.isclose(reconstructed, total_hours, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Hard-stop allocation does not cover three pilots and final")
    if total_hours > 14:
        raise ValueError("Direct hard-stop allocation requires a <=14-hour projection")
    scale = FEASIBILITY_HARD_LIMIT_HOURS / total_hours
    pilot_allocation = pilot_run_hours * scale
    final_allocation = final_hours * scale
    allocated_total = len(LR_GRID) * pilot_allocation + final_allocation
    if not math.isclose(
        allocated_total,
        FEASIBILITY_HARD_LIMIT_HOURS,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError("Per-run allocations do not sum to the arm hard limit")
    current_hours = pilot_allocation if phase == "pilot" else final_allocation
    return {
        "scope": (
            "one invocation; three pilot allocations plus one final allocation "
            "sum to the arm-level 16-hour hard envelope"
        ),
        "retry_accounting": (
            "retries are separate invocations and are not claimed as cumulatively "
            "accounted by this callback"
        ),
        "pilot_runs": len(LR_GRID),
        "pilot_run_allocation_hours": pilot_allocation,
        "final_run_allocation_hours": final_allocation,
        "allocated_arm_total_hours": allocated_total,
        "current_phase": phase,
        "current_run_allocation_hours": current_hours,
        "current_run_allocation_seconds": current_hours * 3600,
    }


def make_wall_clock_hard_stop_callback(deadline: WallClockDeadline) -> Any:
    """Create a Transformers callback that raises before an artifact can publish."""

    from transformers import TrainerCallback

    class WallClockHardStopCallback(TrainerCallback):
        def _check(self, stage: str) -> None:
            deadline.check(stage)

        def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            self._check("train_begin")
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            self._check("step_end")
            return control

        def on_evaluate(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            self._check("evaluate")
            return control

        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            self._check("save")
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            self._check("train_end")
            return control

    return WallClockHardStopCallback()


def _durable_output_contract(
    destination: Path,
    declared_root: Path,
    *,
    configured_temp_root: Path | None = None,
    source_worktree: Path | None = None,
) -> dict[str, Any]:
    """Bind an actual checkpoint to an explicit non-transient output root."""

    destination = destination.resolve()
    declared_root = declared_root.resolve()
    configured_temp_root = Path(
        tempfile.gettempdir() if configured_temp_root is None else configured_temp_root
    ).resolve()
    transient_roots = tuple(
        dict.fromkeys(
            root.resolve()
            for root in (*_CANONICAL_TRANSIENT_ROOTS, configured_temp_root)
        )
    )
    source_worktree = (
        Path(__file__).resolve().parents[1]
        if source_worktree is None
        else source_worktree.resolve()
    )
    if not declared_root.is_dir():
        raise ValueError(
            f"Durable output root must already exist as a directory: {declared_root}"
        )
    if any(
        declared_root == transient_root or declared_root.is_relative_to(transient_root)
        for transient_root in transient_roots
    ):
        raise ValueError(
            "Durable output root must not be inside a canonical or configured "
            "transient tree"
        )
    if declared_root == source_worktree or declared_root.is_relative_to(
        source_worktree
    ):
        raise ValueError("Durable output root must be outside the source worktree")
    if destination == declared_root or not destination.is_relative_to(declared_root):
        raise ValueError("Training output must be a child of the durable output root")
    if any(
        destination == transient_root or destination.is_relative_to(transient_root)
        for transient_root in transient_roots
    ):
        raise ValueError(
            "Training output must not be inside a canonical or configured "
            "transient tree"
        )
    return {
        "schema_version": 1,
        "declared_output_root": str(declared_root),
        "artifact_relative_path": destination.relative_to(declared_root).as_posix(),
    }


def _validate_run_config(config: ScratchRunConfig) -> None:
    if config.arm_id not in ARM_SPECS:
        raise ValueError(f"Unknown scratch arm: {config.arm_id!r}")
    if config.phase not in {"pilot", "final"}:
        raise ValueError("phase must be 'pilot' or 'final'")
    if config.wandb_entity != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            f"W&B writes are restricted to personal entity {PERSONAL_WANDB_ENTITY!r}"
        )
    if config.wandb_project != DEFAULT_WANDB_PROJECT:
        raise ValueError(f"W&B project is frozen at {DEFAULT_WANDB_PROJECT!r}")
    if config.initialization_seed != INITIALIZATION_SEED:
        raise ValueError(f"initialization seed is frozen at {INITIALIZATION_SEED}")
    if config.data_order_seed != DATA_ORDER_SEED:
        raise ValueError(f"data-order seed is frozen at {DATA_ORDER_SEED}")
    if config.learning_rate not in LR_GRID:
        raise ValueError(f"learning rate must be one of {LR_GRID}")
    if config.phase == "pilot":
        if config.num_train_epochs != PILOT_EPOCHS:
            raise ValueError("Every pilot must run exactly one quarter epoch")
        if config.selection_lock is not None:
            raise ValueError("A pilot must not consume a final selection lock")
    else:
        if config.num_train_epochs != MAX_FINAL_EPOCHS:
            raise ValueError(f"Final runs must use exactly {MAX_FINAL_EPOCHS:g} epochs")
        if config.selection_lock is None:
            raise ValueError("Final training requires a validation-only LR lock")
    if not config.dry_run and (
        not isinstance(config.wandb_run_name, str) or not config.wandb_run_name.strip()
    ):
        raise ValueError("An actual run requires a recoverable W&B run name")
    if not config.dry_run and config.feasibility_manifest is None:
        raise ValueError("An actual run requires a passing feasibility manifest")
    if not config.dry_run:
        if config.durable_output_root is None:
            raise ValueError("An actual run requires an explicit durable output root")
        _durable_output_contract(config.output_dir, config.durable_output_root)


def _task_ids_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return digest_json([str(row["task_id"]) for row in rows])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _recorded_source_identity(source_code: Mapping[str, Any]) -> dict[str, Any]:
    """Replace the imported helper's runner pointer with this actual runner."""

    runner_path = Path(__file__).resolve()
    repo_root = runner_path.parents[1]
    helper = source_code.get("runner")
    return {
        **dict(source_code),
        "runner": {
            "path": runner_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(runner_path),
        },
        "source_identity_helper": helper,
    }


def _retained_rows(
    rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    retained_ids = [str(row["task_id"]) for row in audit["retained"]]
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Canonical training rows contain duplicate task IDs")
    try:
        retained = tuple(by_id[task_id] for task_id in retained_ids)
    except KeyError as error:
        raise ValueError("Tokenization audit references an unknown task") from error
    if _task_ids_digest(retained) != audit["retained_task_ids_digest"]:
        raise ValueError(
            "Retained row order differs from the frozen tokenization audit"
        )
    return retained


def _source_identity(
    *,
    corpus_dir: Path,
    corpus_manifest: Mapping[str, Any],
    tokenizer_manifest: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[BasisTaskRecord]],
    train_audit: Mapping[str, Any],
    validation_audit: Mapping[str, Any],
    verification: Mapping[str, Any],
    reference_identities: Sequence[deep.CorpusIdentities],
) -> dict[str, Any]:
    return {
        "corpus_dir": str(corpus_dir.resolve()),
        "corpus_manifest_digest": corpus_manifest["manifest_digest"],
        "corpus_hub": dict(corpus_manifest.get("hub", {})),
        "basis_set_id": corpus_manifest["basis_set_id"],
        "basis_set_digest": corpus_manifest["basis_set_digest"],
        "tokenizer_artifact_digest": tokenizer_manifest["artifact_digest"],
        "tokenizer_source_manifest_digest": tokenizer_manifest["source"][
            "manifest_digest"
        ],
        "verification": {
            "ok": verification["ok"],
            "records": verification["records"],
            "distance_histogram": dict(verification["distance_histogram"]),
            "root_digest_set_sha256": verification["root_digest_set_sha256"],
            "task_id_set_sha256": verification["task_id_set_sha256"],
            "all_witnesses_reexecuted": not verification["witness_failures"],
            "target_contract_failures": list(verification["target_contract_failures"]),
            "reference_root_digest_overlaps": list(
                verification["reference_root_digest_overlaps"]
            ),
            "reference_root_key_overlaps": list(
                verification["reference_root_key_overlaps"]
            ),
            "reference_task_id_overlaps": list(
                verification["reference_task_id_overlaps"]
            ),
        },
        "reference_corpora": [
            identity.manifest_reference() for identity in reference_identities
        ],
        "task_roles": {
            TRAIN_SPLIT: {
                "split": TRAIN_SPLIT,
                "role": "training-only",
                "selected_records": len(records_by_split[TRAIN_SPLIT]),
                "retained_records": train_audit["retained_records"],
                "selected_task_ids_digest": train_audit["selected_task_ids_digest"],
                "retained_task_ids_digest": train_audit["retained_task_ids_digest"],
            },
            VALIDATION_SPLIT: {
                "split": VALIDATION_SPLIT,
                "role": "loss-evaluation-and-tuning-only",
                "selected_records": len(records_by_split[VALIDATION_SPLIT]),
                "retained_records": validation_audit["retained_records"],
                "selected_task_ids_digest": validation_audit[
                    "selected_task_ids_digest"
                ],
                "retained_task_ids_digest": validation_audit[
                    "retained_task_ids_digest"
                ],
            },
            HELD_OUT_SPLIT: {
                "split": HELD_OUT_SPLIT,
                "role": "held-out-evaluation-only; never model-scored here",
                "records": len(records_by_split[HELD_OUT_SPLIT]),
                "task_ids_digest": digest_json(
                    [record.task_id for record in records_by_split[HELD_OUT_SPLIT]]
                ),
            },
        },
    }


def prepare_scratch_data(config: ScratchRunConfig) -> PreparedScratchData:
    """Verify all provenance, then expose discovery/validation model rows only."""

    require_deterministic_python_hash_seed()
    _validate_run_config(config)
    corpus_manifest, records = deep.load_corpus(config.corpus_dir)
    reference_identities = tuple(
        deep.collect_corpus_identities(reference_dir)
        for reference_dir in config.reference_corpus_dirs
    )
    verification = deep.verify_corpus(
        config.corpus_dir,
        recertify=0,
        seed=corpus_manifest["seed"],
        reference_identities=reference_identities,
    )
    if not verification["ok"]:
        raise ValueError("deep-corpus-v2 failed complete witness/leakage verification")
    basis = load_basis_set(corpus_manifest["basis_set_id"])
    require_reproducible_basis_runtime(basis)
    if basis.digest != corpus_manifest["basis_set_digest"]:
        raise ValueError("Corpus basis digest differs from the loaded basis")
    tokenizer_manifest = validate_frozen_tokenizer(
        config.tokenizer_dir,
        expected_source_manifest_digest=corpus_manifest["manifest_digest"],
    )
    if tokenizer_manifest["basis_set"] != basis.identity_dict():
        raise ValueError("Shared tokenizer basis differs from the corpus basis")

    records_by_split: dict[str, list[BasisTaskRecord]] = {
        TRAIN_SPLIT: [],
        VALIDATION_SPLIT: [],
        HELD_OUT_SPLIT: [],
    }
    for record in records:
        try:
            records_by_split[record.split].append(record)
        except KeyError as error:
            raise ValueError(f"Unexpected corpus split: {record.split!r}") from error
    if any(not records_by_split[split] for split in records_by_split):
        raise ValueError(
            "Corpus must contain non-empty discovery, validation, and test"
        )

    available = basis.as_function_set()
    train_rows = canonical_corpus_rows(
        records_by_split[TRAIN_SPLIT],
        available,
        allowed_splits=(TRAIN_SPLIT,),
        row_builder=training_row_for_record,
    )
    validation_rows = canonical_corpus_rows(
        records_by_split[VALIDATION_SPLIT],
        available,
        allowed_splits=(VALIDATION_SPLIT,),
        row_builder=training_row_for_record,
    )
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.tokenizer_dir,
        local_files_only=True,
    )
    train_audit = build_tokenization_audit(
        tokenizer,
        train_rows,
        max_length=TOKENIZER_MAX_LENGTH,
    )
    validation_audit = build_tokenization_audit(
        tokenizer,
        validation_rows,
        max_length=TOKENIZER_MAX_LENGTH,
    )
    frozen_train = tokenizer_manifest["selection"]
    for field in (
        "selected_records",
        "retained_records",
        "excluded_records",
        "selected_task_ids_digest",
        "retained_task_ids_digest",
        "excluded_task_ids_digest",
        "token_id_digests",
    ):
        if train_audit[field] != frozen_train[field]:
            raise ValueError(
                f"Discovery tokenization drifted from frozen artifact: {field}"
            )

    retained_train = _retained_rows(train_rows, train_audit)
    retained_validation = _retained_rows(validation_rows, validation_audit)
    source = _source_identity(
        corpus_dir=config.corpus_dir,
        corpus_manifest=corpus_manifest,
        tokenizer_manifest=tokenizer_manifest,
        records_by_split=records_by_split,
        train_audit=train_audit,
        validation_audit=validation_audit,
        verification=verification,
        reference_identities=reference_identities,
    )
    return PreparedScratchData(
        basis=basis,
        corpus_manifest=corpus_manifest,
        tokenizer_manifest=tokenizer_manifest,
        train_rows=retained_train,
        validation_rows=retained_validation,
        train_audit=train_audit,
        validation_audit=validation_audit,
        source_identity=source,
    )


def resolve_run_architecture(
    config: ScratchRunConfig,
    data: PreparedScratchData,
) -> ResolvedScratchConfig:
    tokenizer = data.tokenizer_manifest["tokenizer"]
    specials = {row["role"]: row["id"] for row in tokenizer["special_tokens"]}
    return resolve_scratch_config(
        config.arm_id,
        vocab_size=int(tokenizer["total_vocab_size"]),
        bos_token_id=int(specials["bos_token"]),
        eos_token_id=int(specials["eos_token"]),
        pad_token_id=int(specials["pad_token"]),
    )


def build_sft_config(
    config: ScratchRunConfig,
    *,
    report_to_wandb: bool,
    precision_bf16: bool = True,
):
    from trl import SFTConfig

    checkpointing = config.arm_id in GRADIENT_CHECKPOINTING_ARMS
    return SFTConfig(
        output_dir=str(config.output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_length=TOKENIZER_MAX_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_safetensors=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=True,
        seed=config.initialization_seed,
        data_seed=config.data_order_seed,
        full_determinism=True,
        dataloader_num_workers=0,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs=(
            dict(GRADIENT_CHECKPOINTING_KWARGS) if checkpointing else None
        ),
        bf16=precision_bf16,
        fp16=False,
        report_to=["wandb"] if report_to_wandb else [],
        run_name=config.wandb_run_name if report_to_wandb else None,
        push_to_hub=False,
        packing=False,
        include_num_input_tokens_seen=True,
    )


def _loss_curve(log_history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    preceding_train: Mapping[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    for row in log_history:
        if "loss" in row:
            preceding_train = row
        if "eval_loss" not in row:
            continue
        train_loss = (
            float(preceding_train["loss"]) if preceding_train is not None else None
        )
        validation_loss = float(row["eval_loss"])
        curve.append(
            {
                "step": int(row.get("step", 0)),
                "epoch": float(row.get("epoch", 0)),
                "preceding_train_window_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_minus_train_gap": (
                    validation_loss - train_loss if train_loss is not None else None
                ),
                "eval_runtime": row.get("eval_runtime"),
            }
        )
    return curve


def _training_protocol(config: ScratchRunConfig) -> dict[str, Any]:
    checkpointing = config.arm_id in GRADIENT_CHECKPOINTING_ARMS
    return {
        "phase": config.phase,
        "primary_training_distribution": "all certified distances 1-8",
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "held_out_split": HELD_OUT_SPLIT,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "learning_rate_grid": list(LR_GRID),
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
        "max_length": TOKENIZER_MAX_LENGTH,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        "pythonhashseed": DETERMINISTIC_PYTHON_HASH_SEED,
        "gradient_checkpointing": checkpointing,
        "gradient_checkpointing_kwargs": (
            dict(GRADIENT_CHECKPOINTING_KWARGS) if checkpointing else {}
        ),
        "completion_only_loss": True,
        "eval_strategy": "epoch",
        "validation_selection": "minimum eval_loss",
        "best_checkpoint_retained": True,
        "initialization_seed": config.initialization_seed,
        "data_order_seed": config.data_order_seed,
        "resumption_supported": False,
    }


def _study_identity_for_run(
    config: ScratchRunConfig,
    data: PreparedScratchData,
    resolved: ResolvedScratchConfig,
) -> dict[str, Any]:
    protocol = _training_protocol(config)
    return build_scratch_study_identity(
        architecture=resolved.identity_dict(),
        corpus_manifest_digest=data.corpus_manifest["manifest_digest"],
        tokenizer_artifact_digest=data.tokenizer_manifest["artifact_digest"],
        retained_training_task_ids_digest=data.train_audit["retained_task_ids_digest"],
        per_device_batch_size=protocol["per_device_train_batch_size"],
        gradient_accumulation_steps=protocol["gradient_accumulation_steps"],
        max_length=protocol["max_length"],
        optimizer=protocol["optimizer"],
        precision=protocol["precision"],
        gradient_checkpointing=protocol["gradient_checkpointing"],
        gradient_checkpointing_kwargs=protocol["gradient_checkpointing_kwargs"],
        cublas_workspace_config=protocol["cublas_workspace_config"],
        pythonhashseed=protocol["pythonhashseed"],
        initialization_seed=config.initialization_seed,
        data_order_seed=config.data_order_seed,
    )


def _workload_identity(
    config: ScratchRunConfig,
    data: PreparedScratchData,
) -> dict[str, Any]:
    retained = data.train_audit["statistics"]["retained"]["overall"]
    combined_per_pass = int(retained["combined"]["total"])
    completion_per_pass = int(retained["completion_with_eos"]["total"])
    return {
        "unique_training_examples": len(data.train_rows),
        "unique_validation_examples": len(data.validation_rows),
        "requested_data_passes": config.num_train_epochs,
        "planned_example_presentations": len(data.train_rows) * config.num_train_epochs,
        "combined_tokens_per_full_training_pass": combined_per_pass,
        "completion_loss_tokens_per_full_training_pass": completion_per_pass,
        "planned_combined_training_tokens": combined_per_pass * config.num_train_epochs,
        "planned_completion_loss_tokens": completion_per_pass * config.num_train_epochs,
    }


def _build_manifest(
    *,
    config: ScratchRunConfig,
    data: PreparedScratchData,
    resolved: ResolvedScratchConfig,
    initialization: Mapping[str, Any],
    source_code: Mapping[str, Any],
    status: str,
    started_at_utc: str,
    wandb_url: str | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    execution_environment: Mapping[str, Any] | None = None,
    selection_lock: Mapping[str, Any] | None = None,
    feasibility_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    curve = list((trainer_state or {}).get("loss_curve", []))
    trend = (
        classify_validation_trend(
            [{"eval_loss": row["validation_loss"]} for row in curve]
        )
        if curve
        else None
    )
    best_validation_loss = (
        min(float(row["validation_loss"]) for row in curve) if curve else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "phase": config.phase,
        "arm_id": config.arm_id,
        "started_at_utc": started_at_utc,
        "architecture": resolved.identity_dict(),
        "creation": dict(initialization),
        "initial_state_digest": initialization.get("initial_state_digest"),
        "initialization_seed": config.initialization_seed,
        "data_order_seed": config.data_order_seed,
        "effective_config_digest": resolved.effective_config_digest,
        "exact_trainable_parameters": resolved.exact_trainable_parameters,
        "source_data": dict(data.source_identity),
        "tokenizer_digest": data.tokenizer_manifest["artifact_digest"],
        "retained_training_task_ids_digest": data.train_audit[
            "retained_task_ids_digest"
        ],
        "tokenization": {
            "discovery": dict(data.train_audit),
            "validation": dict(data.validation_audit),
        },
        "training_protocol": _training_protocol(config),
        "workload": _workload_identity(config, data),
        # Duplicated at the top level because LR-selection locks are intentionally
        # small and validate these fields without parsing a nested protocol.
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "best_validation_loss": best_validation_loss,
        "selection_lock": dict(selection_lock or {}),
        "feasibility_gate": dict(feasibility_gate or {}),
        "source_code": _recorded_source_identity(source_code),
        "trainer_state": dict(trainer_state or {}),
        "validation_trend": trend,
        "execution_environment": dict(execution_environment or {}),
        "durability_contract": (
            _durable_output_contract(config.output_dir, config.durable_output_root)
            if config.durable_output_root is not None
            else None
        ),
        "wandb": {
            "entity": PERSONAL_WANDB_ENTITY,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "url": wandb_url,
        },
    }


def _cuda_environment(
    torch: Any, *, gpu_memory_telemetry: Mapping[str, Any]
) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    measurement_scope = gpu_memory_telemetry.get("measurement_scope")
    if measurement_scope == "process":
        memory_measurement_note = (
            "peak is nvidia-smi process-attributed physical usage; torch "
            "allocated/reserved are allocator telemetry; physical_vram_bytes "
            "is device capacity"
        )
    elif measurement_scope == "physical_gpu_total":
        memory_measurement_note = (
            "nvidia-smi exposed no compute-process rows; peak is conservative "
            "total usage of the identified physical GPU, including baseline; "
            "torch allocated/reserved are allocator telemetry"
        )
    else:
        raise ValueError("GPU-memory telemetry has an unknown measurement scope")
    return {
        "device": f"cuda:{index}",
        "name": str(properties.name),
        "physical_vram_bytes": int(properties.total_memory),
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        "peak_observed_gpu_memory_bytes": gpu_memory_telemetry[
            "peak_observed_used_bytes"
        ],
        "gpu_memory_telemetry": dict(gpu_memory_telemetry),
        "memory_measurement_note": memory_measurement_note,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _current_gpu_probe_identity(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = str(properties.uuid) if getattr(properties, "uuid", None) else None
    if not uuid:
        raise RuntimeError(
            "Physical GPU UUID is required for process-level peak-memory sampling"
        )
    identity = {
        "logical_device": f"cuda:{index}",
        "visible_cuda_devices": int(torch.cuda.device_count()),
        "name": str(properties.name),
        "uuid": uuid,
        "physical_vram_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }
    return {**identity, "identity_digest": digest_json(identity)}


def _validate_probe_source_identity(
    manifest: Mapping[str, Any],
    *,
    current_source: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Require the launch-authorizing probe to come from this clean commit."""

    probe_source = manifest.get("source_code")
    if not isinstance(probe_source, Mapping):
        raise ValueError("Feasibility artifact has no producer-source identity")
    if (
        probe_source.get("dirty") is not False
        or current_source.get("dirty") is not False
    ):
        raise ValueError("Feasibility and training producers must both be clean")
    commit = current_source.get("commit")
    if not isinstance(commit, str) or probe_source.get("commit") != commit:
        raise ValueError("Feasibility probe was produced by another Git commit")
    expected_path = "experiments/probe_deep_corpus_v2_scratch.py"
    expected_file = repo_root / expected_path
    expected_runner = {
        "path": expected_path,
        "sha256": _sha256_file(expected_file),
    }
    if probe_source.get("probe_runner") != expected_runner:
        raise ValueError("Feasibility probe runner differs from the current source")
    return dict(probe_source)


def _validate_feasibility_for_run(
    *,
    config: ScratchRunConfig,
    data: PreparedScratchData,
    resolved: ResolvedScratchConfig,
    torch: Any,
    source_code: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.probe_deep_corpus_v2_scratch import load_probe_manifest

    if config.feasibility_manifest is None:
        raise ValueError("A full run has no feasibility artifact")
    manifest = load_probe_manifest(
        config.feasibility_manifest,
        require_launch_allowed=True,
        arm_id=config.arm_id,
    )
    probe_source = _validate_probe_source_identity(
        manifest,
        current_source=source_code,
        repo_root=Path(__file__).resolve().parents[1],
    )
    source = manifest["source_data"]
    expected_source = {
        "corpus_manifest_digest": data.corpus_manifest["manifest_digest"],
        "tokenizer_artifact_digest": data.tokenizer_manifest["artifact_digest"],
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ValueError(f"Feasibility artifact changed study identity: {field}")
    if (
        source.get("task_roles", {})
        .get(TRAIN_SPLIT, {})
        .get("retained_task_ids_digest")
        != data.train_audit["retained_task_ids_digest"]
    ):
        raise ValueError(
            "Feasibility artifact changed study identity: retained training tasks"
        )
    expected_references = list(data.source_identity.get("reference_corpora", ()))
    if not expected_references:
        raise ValueError("Prepared training data has no reference-corpus identity")
    if source.get("reference_corpora") != expected_references or source.get(
        "reference_corpora_digest"
    ) != digest_json(expected_references):
        raise ValueError(
            "Feasibility artifact changed study identity: reference corpora"
        )
    arm_results = {row["arm_id"]: row for row in manifest.get("arm_results", [])}
    result = arm_results.get(config.arm_id)
    if not isinstance(result, dict):
        raise ValueError("Feasibility artifact has no result for this arm")
    if result.get("architecture") != resolved.identity_dict():
        raise ValueError("Feasibility artifact used another scratch architecture")
    expected_study_digest = digest_json(_study_identity_for_run(config, data, resolved))
    if result.get("study_identity_digest") != expected_study_digest:
        raise ValueError("Feasibility artifact changed the frozen training protocol")
    actual_gpu = _current_gpu_probe_identity(torch)
    if result.get("gpu", {}).get("identity_digest") != actual_gpu["identity_digest"]:
        raise ValueError("Full run is not using the physical GPU measured by the probe")
    arm_gate = manifest["launch_gate"]["per_arm"][config.arm_id]
    if arm_gate.get("route") != "direct":
        raise RuntimeError(
            "Validated-optimization feasibility requires a cumulative 16-hour "
            "hard-stop orchestrator; this runner currently accepts direct <=14-hour "
            "authorizations only"
        )
    allocation = hard_stop_allocation(arm_gate, phase=config.phase)
    return {
        "artifact_path": str(config.feasibility_manifest.resolve()),
        "artifact_digest": manifest["artifact_digest"],
        "arm_id": config.arm_id,
        "arm_gate": dict(arm_gate),
        "probe_gpu": dict(result["gpu"]),
        "run_gpu": actual_gpu,
        "study_identity_digest": result["study_identity_digest"],
        "probe_source": probe_source,
        "hard_stop_allocation": allocation,
    }


def _selection_lock(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read selection lock {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Selection lock must be a JSON object")
    return value


def _remove_intermediate_checkpoints(output_dir: Path) -> None:
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)


def _create_training_staging_dir(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )


def _existing_training_staging_dirs(destination: Path) -> tuple[Path, ...]:
    """Return prior scoped staging dirs so a crash does not dirty-block recovery."""

    destination = destination.resolve()
    return tuple(
        sorted(
            (
                path.resolve()
                for path in destination.parent.glob(f".{destination.name}.staging-*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: path.name,
        )
    )


def _completion_lock_path(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    return checkpoint_dir.parent / f".{checkpoint_dir.name}.completion.lock"


def _acquire_checkpoint_completion_lock(
    checkpoint_dir: Path,
) -> _CheckpointCompletionLock:
    """Acquire the producer/recovery exclusion lock without waiting."""

    checkpoint_dir = checkpoint_dir.resolve()
    path = _completion_lock_path(checkpoint_dir)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(
            f"Checkpoint completion is already in progress: {checkpoint_dir}"
        ) from error
    return _CheckpointCompletionLock(
        checkpoint_dir=checkpoint_dir,
        path=path,
        descriptor=descriptor,
    )


def _release_checkpoint_completion_lock(lock: _CheckpointCompletionLock) -> None:
    if not lock.held:
        return
    lock.held = False
    # Closing the descriptor releases flock even if the process exits abruptly.
    os.close(lock.descriptor)


def _atomic_rename_noreplace(
    source: Path,
    destination: Path,
    *,
    libc: Any | None = None,
) -> None:
    """Atomically publish a directory without ever replacing an existing path."""

    if sys.platform != "linux":
        raise RuntimeError(
            "Atomic no-replace checkpoint publication requires Linux renameat2"
        )
    library = ctypes.CDLL(None, use_errno=True) if libc is None else libc
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "Atomic no-replace checkpoint publication is unavailable: renameat2"
        )
    if libc is None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "Checkpoint destination already exists",
            str(destination),
        )
    unsupported = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    if error_number in unsupported:
        raise RuntimeError(
            "Atomic no-replace checkpoint publication is unsupported by this "
            "kernel or filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _publish_training_staging(staging: Path, destination: Path) -> None:
    staging = staging.resolve()
    destination = destination.resolve()
    if staging.parent != destination.parent or not staging.name.startswith(
        f".{destination.name}.staging-"
    ):
        raise ValueError("Training staging path is not scoped to its destination")
    if not staging.is_dir():
        raise FileNotFoundError(f"Training staging directory is missing: {staging}")
    _atomic_rename_noreplace(staging, destination)


def _cleanup_training_staging(staging: Path, destination: Path) -> None:
    staging = staging.resolve()
    destination = destination.resolve()
    if staging.parent != destination.parent or not staging.name.startswith(
        f".{destination.name}.staging-"
    ):
        raise ValueError("Refusing to clean an unscoped training staging path")
    if staging.exists():
        shutil.rmtree(staging)


def _published_training_args(
    training_args: Any, *, staging: Path, destination: Path
) -> dict[str, Any]:
    """Rewrite ephemeral staging paths to their atomically published location."""

    staging_text = str(staging.resolve())
    destination_text = str(destination.resolve())

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, tuple):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and (
            value == staging_text or value.startswith(staging_text + os.sep)
        ):
            return destination_text + value[len(staging_text) :]
        return value

    published = rewrite(training_args.to_dict())
    if not isinstance(published, dict):
        raise TypeError("Training arguments did not serialize to a mapping")
    if staging_text in json.dumps(published, sort_keys=True):
        raise RuntimeError(
            "Published training arguments retain an ephemeral staging path"
        )
    return published


def _seal_training_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    return {**payload, "manifest_digest": digest_json(payload)}


def _validate_training_manifest_seal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(manifest)
    recorded = normalized.pop("manifest_digest", None)
    if recorded != digest_json(normalized):
        raise ValueError("Training manifest digest mismatch")
    return {**normalized, "manifest_digest": recorded}


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _checkpoint_content_identity(checkpoint_dir: Path) -> dict[str, Any]:
    """Hash every published payload file except the self-sealed manifest."""

    checkpoint_dir = checkpoint_dir.resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Checkpoint payload contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint_dir).as_posix()
        if relative == TRAINING_MANIFEST_FILENAME:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError("Checkpoint payload contains no identity-bearing files")
    payload = {
        "schema_version": 1,
        "scope": f"all regular files except {TRAINING_MANIFEST_FILENAME}",
        "files": files,
    }
    return {**payload, "content_digest": digest_json(payload)}


def _validate_checkpoint_content_identity(
    checkpoint_dir: Path,
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    actual = _checkpoint_content_identity(checkpoint_dir)
    if dict(recorded) != actual:
        raise RuntimeError("Published checkpoint content identity mismatch")
    return actual


def _validate_durable_checkpoint_files(checkpoint_dir: Path) -> None:
    assert_weights_only_artifact(checkpoint_dir, require_complete=True)
    missing = sorted(
        name
        for name in _DURABLE_ARTIFACT_REQUIRED_FILES
        if not (checkpoint_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(
            f"Published checkpoint is missing durable metadata: {missing}"
        )


def _validate_wandb_link(checkpoint_dir: Path, *, expected_url: str) -> None:
    validate_wandb_url(expected_url, project=DEFAULT_WANDB_PROJECT)
    path = checkpoint_dir / "wandb_run.url"
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            f"Could not read W&B checkpoint link {path}: {error}"
        ) from error
    if value != f"[InternetShortcut]\nURL={expected_url}\n":
        raise ValueError("Published checkpoint W&B link changed")


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _checkpoint_expected_identities(
    checkpoint_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read architecture and tokenizer identities only from durable payload data."""

    arm_id = manifest.get("arm_id")
    architecture = manifest.get("architecture")
    if not isinstance(arm_id, str) or arm_id not in ARM_SPECS:
        raise RuntimeError("Checkpoint manifest has an unknown scratch arm")
    if not isinstance(architecture, Mapping):
        raise RuntimeError("Checkpoint manifest has no architecture identity")
    spec = ARM_SPECS[arm_id]
    exact_parameters = _positive_int(
        architecture.get("exact_trainable_parameters"),
        label="Architecture exact_trainable_parameters",
    )
    vocab_size = _positive_int(
        architecture.get("vocab_size"),
        label="Architecture vocab_size",
    )
    budget_value = _positive_int(
        architecture.get("budget_dimension_value"),
        label="Architecture budget_dimension_value",
    )
    expected_architecture = {
        "family": spec.family,
        "exact_trainable_parameters": exact_parameters,
        "vocab_size": vocab_size,
        "budget_dimension": spec.budget_dimension,
        "budget_dimension_value": budget_value,
    }
    if (
        architecture.get("arm_id") != arm_id
        or architecture.get("family") != expected_architecture["family"]
        or architecture.get("budget_dimension")
        != expected_architecture["budget_dimension"]
        or manifest.get("exact_trainable_parameters") != exact_parameters
    ):
        raise RuntimeError("Checkpoint architecture identity is inconsistent")

    tokenizer_manifest = _read_json_object(
        checkpoint_dir / TOKENIZER_MANIFEST_FILENAME,
        label="tokenizer manifest",
    )
    tokenizer_digest = tokenizer_manifest.get("artifact_digest")
    digest_payload = dict(tokenizer_manifest)
    digest_payload.pop("artifact_digest", None)
    digest_payload.pop("hub", None)
    if (
        not isinstance(tokenizer_digest, str)
        or tokenizer_digest != digest_json(digest_payload)
        or manifest.get("tokenizer_digest") != tokenizer_digest
    ):
        raise RuntimeError("Checkpoint tokenizer artifact digest is inconsistent")
    tokenizer_identity = tokenizer_manifest.get("tokenizer")
    if not isinstance(tokenizer_identity, Mapping):
        raise RuntimeError("Checkpoint tokenizer manifest has no tokenizer identity")
    if tokenizer_identity.get("total_vocab_size") != vocab_size:
        raise RuntimeError("Checkpoint tokenizer vocabulary identity is inconsistent")
    files = tokenizer_identity.get("files")
    if not isinstance(files, Mapping) or set(files) != set(TOKENIZER_FILENAMES):
        raise RuntimeError("Checkpoint tokenizer file identity is incomplete")
    for name in TOKENIZER_FILENAMES:
        identity = files[name]
        path = checkpoint_dir / name
        if (
            not isinstance(identity, Mapping)
            or type(identity.get("bytes")) is not int
            or identity["bytes"] < 0
            or not isinstance(identity.get("sha256"), str)
            or path.stat().st_size != identity["bytes"]
            or _sha256_file(path) != identity["sha256"]
        ):
            raise RuntimeError(f"Checkpoint tokenizer file identity changed: {name}")

    special_rows = tokenizer_identity.get("special_tokens")
    if not isinstance(special_rows, list):
        raise RuntimeError("Checkpoint tokenizer special-token identity is missing")
    expected_contents = dict(FIXED_SPECIAL_TOKENS)
    special_tokens: dict[str, dict[str, Any]] = {}
    for row in special_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("Checkpoint tokenizer special-token row is malformed")
        role = row.get("role")
        content = row.get("content")
        token_id = row.get("id")
        if (
            not isinstance(role, str)
            or role in special_tokens
            or expected_contents.get(role) != content
            or type(token_id) is not int
            or token_id < 0
            or token_id >= vocab_size
        ):
            raise RuntimeError("Checkpoint tokenizer special-token identity changed")
        special_tokens[role] = {"content": content, "id": token_id}
    if set(special_tokens) != set(expected_contents):
        raise RuntimeError("Checkpoint tokenizer special-token identity is incomplete")
    return expected_architecture, special_tokens


def _reload_published_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_architecture: Mapping[str, Any],
    expected_special_tokens: Mapping[str, Mapping[str, Any]],
    auto_model_class: Any | None = None,
    tokenizer_class: Any | None = None,
) -> dict[str, Any]:
    """Reload the local checkpoint and verify its model/tokenizer identities."""

    if auto_model_class is None or tokenizer_class is None:
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

        auto_model_class = auto_model_class or AutoModelForCausalLM
        tokenizer_class = tokenizer_class or PreTrainedTokenizerFast

    model = None
    tokenizer = None
    try:
        model = auto_model_class.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True,
        )
        tokenizer = tokenizer_class.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True,
        )
        model_config = model.config
        expected_config = {
            "model_type": expected_architecture["family"],
            "vocab_size": expected_architecture["vocab_size"],
            expected_architecture["budget_dimension"]: expected_architecture[
                "budget_dimension_value"
            ],
        }
        for field, expected in expected_config.items():
            if getattr(model_config, field, None) != expected:
                raise RuntimeError(
                    f"Reloaded checkpoint changed model config field {field!r}"
                )
        parameter_count = exact_model_parameter_count(model)
        if parameter_count != expected_architecture["exact_trainable_parameters"]:
            raise RuntimeError(
                "Reloaded checkpoint changed the trainable parameter count: "
                f"actual={parameter_count}, "
                f"expected={expected_architecture['exact_trainable_parameters']}"
            )
        tokenizer_size = len(tokenizer)
        if tokenizer_size != expected_architecture["vocab_size"]:
            raise RuntimeError(
                "Reloaded checkpoint changed tokenizer vocabulary size: "
                f"actual={tokenizer_size}, "
                f"expected={expected_architecture['vocab_size']}"
            )
        config_special_roles = frozenset({"bos_token", "eos_token", "pad_token"})
        actual_specials: dict[str, int] = {}
        actual_special_contents: dict[str, str] = {}
        actual_special_identities: dict[str, dict[str, Any]] = {}
        for role, identity in expected_special_tokens.items():
            attribute = f"{role}_id"
            expected_id = identity["id"]
            expected_content = identity["content"]
            actual_id = getattr(tokenizer, attribute, None)
            actual_content = getattr(tokenizer, role, None)
            if (
                actual_id != expected_id
                or actual_content != expected_content
                or (
                    role in config_special_roles
                    and getattr(model_config, attribute, None) != expected_id
                )
            ):
                raise RuntimeError(f"Reloaded checkpoint changed the {role} identity")
            actual_special_identities[role] = {
                "content": actual_content,
                "id": actual_id,
            }
            if role in config_special_roles:
                actual_specials[role] = actual_id
                actual_special_contents[role] = actual_content
        return {
            "local_files_only": True,
            "model": {
                "family": expected_architecture["family"],
                "exact_trainable_parameters": parameter_count,
                "vocab_size": int(model_config.vocab_size),
                "budget_dimension": expected_architecture["budget_dimension"],
                "budget_dimension_value": int(
                    getattr(model_config, expected_architecture["budget_dimension"])
                ),
            },
            "tokenizer": {
                "total_vocab_size": tokenizer_size,
                "special_token_ids": actual_specials,
                "special_token_contents": actual_special_contents,
                "special_token_identities": actual_special_identities,
            },
        }
    finally:
        del tokenizer
        del model


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_completed_training_manifest(
    checkpoint_dir: Path,
    manifest: Mapping[str, Any],
    *,
    completion_lock: _CheckpointCompletionLock,
) -> None:
    """Compare-and-replace pending state while holding the checkpoint lock."""

    checkpoint_dir = checkpoint_dir.resolve()
    if not completion_lock.held or completion_lock.checkpoint_dir != checkpoint_dir:
        raise RuntimeError("Completed manifest write requires its held checkpoint lock")
    complete = _validate_training_manifest_seal(manifest)
    if complete.get("status") != "complete":
        raise RuntimeError("Final manifest write requires complete status")
    verification = complete.get("durability_verification")
    expected_pending_digest = (
        verification.get("pending_manifest_digest")
        if isinstance(verification, Mapping)
        else None
    )
    pending = _validate_training_manifest_seal(
        _read_json_object(
            checkpoint_dir / TRAINING_MANIFEST_FILENAME,
            label="pending training manifest",
        )
    )
    if (
        pending.get("status") != _PENDING_DURABILITY_STATUS
        or pending.get("manifest_digest") != expected_pending_digest
    ):
        raise RuntimeError(
            "Checkpoint completion state changed; refusing to overwrite its manifest"
        )
    _write_json_atomic(checkpoint_dir / TRAINING_MANIFEST_FILENAME, complete)


def _runtime_seconds(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise RuntimeError(f"{label} must be finite non-negative seconds")
    return float(value)


def _validate_completion_runtime_provenance(
    manifest: Mapping[str, Any],
    *,
    completion_path: str,
) -> None:
    environment = manifest.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise RuntimeError("Completed checkpoint has no runtime provenance")
    elapsed = _runtime_seconds(
        environment.get("full_invocation_elapsed_seconds"),
        label="full_invocation_elapsed_seconds",
    )
    censoring = environment.get("full_invocation_runtime_censoring")
    if completion_path == "training":
        if (
            environment.get("full_invocation_runtime_scope") != _NORMAL_RUNTIME_SCOPE
            or censoring is not None
        ):
            raise RuntimeError("Normal completion runtime must be exact")
        return
    if (
        environment.get("full_invocation_runtime_scope") != _RECOVERY_RUNTIME_SCOPE
        or not isinstance(censoring, Mapping)
        or censoring.get("status") != "right-censored-lower-bound"
        or censoring.get("omitted_interval_seconds") is not None
        or censoring.get("omitted_interval_scope") != _RECOVERY_OMITTED_INTERVAL_SCOPE
        or censoring.get("omitted_interval_reason") != _RECOVERY_OMITTED_INTERVAL_REASON
    ):
        raise RuntimeError("Recovery runtime censoring provenance is incomplete")
    original = _runtime_seconds(
        censoring.get("known_original_lower_bound_seconds"),
        label="known_original_lower_bound_seconds",
    )
    recovery = _runtime_seconds(
        censoring.get("recovery_elapsed_seconds"),
        label="recovery_elapsed_seconds",
    )
    combined = _runtime_seconds(
        censoring.get("combined_lower_bound_seconds"),
        label="combined_lower_bound_seconds",
    )
    if not (
        math.isclose(combined, original + recovery, rel_tol=0, abs_tol=1e-9)
        and math.isclose(elapsed, combined, rel_tol=0, abs_tol=1e-9)
    ):
        raise RuntimeError("Recovery runtime lower bound is inconsistent")


def _validate_completed_training_manifest(
    checkpoint_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completion candidate without requiring it to be on disk."""

    checkpoint_dir = checkpoint_dir.resolve()
    _validate_durable_checkpoint_files(checkpoint_dir)
    manifest = _validate_training_manifest_seal(manifest)
    if manifest.get("status") != "complete":
        raise RuntimeError("Published checkpoint has not passed durable verification")
    durability_contract = manifest.get("durability_contract")
    if not isinstance(durability_contract, Mapping):
        raise RuntimeError("Completed checkpoint has no durable output-root contract")
    recorded_root = durability_contract.get("declared_output_root")
    if not isinstance(recorded_root, str) or durability_contract != (
        _durable_output_contract(checkpoint_dir, Path(recorded_root))
    ):
        raise RuntimeError("Completed checkpoint changed its durable output root")
    identity = manifest.get("checkpoint_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Completed checkpoint has no content identity")
    _validate_checkpoint_content_identity(checkpoint_dir, identity)
    wandb = manifest.get("wandb")
    if (
        not isinstance(wandb, Mapping)
        or wandb.get("entity") != PERSONAL_WANDB_ENTITY
        or wandb.get("project") != DEFAULT_WANDB_PROJECT
        or not isinstance(wandb.get("url"), str)
    ):
        raise RuntimeError("Completed checkpoint has no W&B run identity")
    _validate_wandb_link(checkpoint_dir, expected_url=wandb["url"])
    verification = manifest.get("durability_verification")
    if not isinstance(verification, Mapping):
        raise RuntimeError("Completed checkpoint has no durability receipt")
    reload_receipt = verification.get("reload")
    if not isinstance(reload_receipt, Mapping):
        raise RuntimeError("Completed checkpoint has no reload receipt")
    model_receipt = reload_receipt.get("model")
    tokenizer_receipt = reload_receipt.get("tokenizer")
    if not isinstance(model_receipt, Mapping) or not isinstance(
        tokenizer_receipt, Mapping
    ):
        raise RuntimeError("Completed checkpoint reload receipt is incomplete")
    expected_architecture, expected_special_tokens = _checkpoint_expected_identities(
        checkpoint_dir, manifest
    )
    model_special_roles = {"bos_token", "eos_token", "pad_token"}
    expected_special_ids = {
        role: identity["id"]
        for role, identity in expected_special_tokens.items()
        if role in model_special_roles
    }
    expected_special_contents = {
        role: identity["content"]
        for role, identity in expected_special_tokens.items()
        if role in model_special_roles
    }
    completion_path = verification.get("completion_path")
    completed_at = manifest.get("completed_at_utc")
    wandb_finalization = verification.get("wandb_finalization")
    if completion_path == "training":
        expected_wandb_finalization = {
            "summary_write_performed": True,
            "finish_performed": True,
            "completed_before_manifest_transition": True,
        }
    else:
        expected_wandb_finalization = {
            "summary_write_performed": False,
            "finish_performed": False,
            "reason": _RECOVERY_WANDB_REASON,
        }
    if (
        verification.get("status") != "verified"
        or verification.get("method") != "post-atomic-publish-local-reload"
        or completion_path not in {"training", "recovery"}
        or not isinstance(verification.get("pending_manifest_digest"), str)
        or not isinstance(completed_at, str)
        or not completed_at.strip()
        or wandb_finalization != expected_wandb_finalization
        or verification.get("published_outside_transient_staging") is not True
        or verification.get("declared_output_root") != recorded_root
        or verification.get("artifact_relative_path")
        != durability_contract.get("artifact_relative_path")
        or verification.get("content_digest") != identity.get("content_digest")
        or reload_receipt.get("local_files_only") is not True
        or dict(model_receipt) != expected_architecture
        or tokenizer_receipt.get("total_vocab_size")
        != expected_architecture["vocab_size"]
        or tokenizer_receipt.get("special_token_ids") != expected_special_ids
        or tokenizer_receipt.get("special_token_contents") != expected_special_contents
        or tokenizer_receipt.get("special_token_identities") != expected_special_tokens
    ):
        raise RuntimeError("Completed checkpoint durability receipt is inconsistent")
    _validate_completion_runtime_provenance(
        manifest,
        completion_path=completion_path,
    )
    return manifest


def _validate_completed_training_artifact(checkpoint_dir: Path) -> dict[str, Any]:
    """Accept only a sealed, content-verified, post-reload checkpoint."""

    checkpoint_dir = checkpoint_dir.resolve()
    return _validate_completed_training_manifest(
        checkpoint_dir,
        _read_json_object(
            checkpoint_dir / TRAINING_MANIFEST_FILENAME,
            label="training manifest",
        ),
    )


def _prepare_training_artifact_completion(
    checkpoint_dir: Path,
    *,
    durable_output_root: Path,
    completion_path: str,
    transient_staging: Path | None = None,
    pending_manifest: Mapping[str, Any] | None = None,
    completion_check: Callable[[], None] | None = None,
    auto_model_class: Any | None = None,
    tokenizer_class: Any | None = None,
) -> dict[str, Any]:
    """Verify a pending artifact without assigning final time or runtime."""

    checkpoint_dir = checkpoint_dir.resolve()
    if completion_path not in {"training", "recovery"}:
        raise ValueError("Unknown durable completion path")
    if transient_staging is not None:
        transient_staging = transient_staging.resolve()
        if checkpoint_dir == transient_staging or transient_staging.exists():
            raise RuntimeError(
                "Durable verification must run after staging publication"
            )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Published checkpoint is missing: {checkpoint_dir}")
    durability_contract = _durable_output_contract(
        checkpoint_dir,
        durable_output_root,
    )
    recorded = _validate_training_manifest_seal(
        _read_json_object(
            checkpoint_dir / TRAINING_MANIFEST_FILENAME,
            label="pending training manifest",
        )
    )
    if pending_manifest is not None and recorded != dict(pending_manifest):
        raise RuntimeError("Published checkpoint manifest changed during publication")
    if recorded.get("status") != _PENDING_DURABILITY_STATUS:
        raise RuntimeError("Checkpoint was marked complete before durable verification")
    if recorded.get("durability_contract") != durability_contract:
        raise RuntimeError(
            "Published checkpoint changed its durable output-root contract"
        )
    _validate_durable_checkpoint_files(checkpoint_dir)
    identity = recorded.get("checkpoint_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Pending checkpoint has no content identity")
    _validate_checkpoint_content_identity(checkpoint_dir, identity)
    wandb = recorded.get("wandb")
    if (
        not isinstance(wandb, Mapping)
        or wandb.get("entity") != PERSONAL_WANDB_ENTITY
        or wandb.get("project") != DEFAULT_WANDB_PROJECT
        or not isinstance(wandb.get("url"), str)
    ):
        raise RuntimeError("Pending checkpoint has no W&B run identity")
    _validate_wandb_link(checkpoint_dir, expected_url=wandb["url"])
    expected_architecture, expected_special_tokens = _checkpoint_expected_identities(
        checkpoint_dir, recorded
    )
    reload_receipt = _reload_published_checkpoint(
        checkpoint_dir,
        expected_architecture=expected_architecture,
        expected_special_tokens=expected_special_tokens,
        auto_model_class=auto_model_class,
        tokenizer_class=tokenizer_class,
    )
    if completion_check is not None:
        completion_check()

    prepared = dict(recorded)
    prepared.pop("manifest_digest", None)
    prepared["durability_verification"] = {
        "status": "verified",
        "method": "post-atomic-publish-local-reload",
        "completion_path": completion_path,
        "pending_manifest_digest": recorded["manifest_digest"],
        "published_outside_transient_staging": True,
        "declared_output_root": durability_contract["declared_output_root"],
        "artifact_relative_path": durability_contract["artifact_relative_path"],
        "content_digest": identity["content_digest"],
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "reload": reload_receipt,
        "wandb_finalization": (
            {
                "summary_write_performed": True,
                "finish_performed": True,
                "completed_before_manifest_transition": True,
            }
            if completion_path == "training"
            else {
                "summary_write_performed": False,
                "finish_performed": False,
                "reason": _RECOVERY_WANDB_REASON,
            }
        ),
    }
    return prepared


def _finalize_training_artifact_completion(
    checkpoint_dir: Path,
    prepared: Mapping[str, Any],
    *,
    completed_at_utc: str,
    runtime_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign final-transition provenance and seal the completed manifest."""

    if prepared.get("status") != _PENDING_DURABILITY_STATUS:
        raise RuntimeError("Completion finalization requires pending prepared state")
    complete = dict(prepared)
    complete.pop("manifest_digest", None)
    complete["status"] = "complete"
    complete["completed_at_utc"] = completed_at_utc
    environment = dict(complete.get("execution_environment", {}))
    environment.pop("full_invocation_runtime_censoring", None)
    complete["execution_environment"] = {
        **environment,
        **dict(runtime_provenance),
    }
    complete = _seal_training_manifest(complete)
    return _validate_completed_training_manifest(checkpoint_dir, complete)


def _normal_runtime_provenance(elapsed_seconds: float) -> dict[str, Any]:
    return {
        "full_invocation_elapsed_seconds": _runtime_seconds(
            elapsed_seconds,
            label="full_invocation_elapsed_seconds",
        ),
        "full_invocation_runtime_scope": _NORMAL_RUNTIME_SCOPE,
    }


def _recovery_runtime_provenance(
    prepared: Mapping[str, Any],
    *,
    recovery_elapsed_seconds: float,
) -> dict[str, Any]:
    environment = prepared.get("execution_environment")
    wall_clock = (
        environment.get("wall_clock_hard_stop")
        if isinstance(environment, Mapping)
        else None
    )
    if not isinstance(wall_clock, Mapping):
        raise RuntimeError("Pending checkpoint has no original runtime lower bound")
    original = _runtime_seconds(
        wall_clock.get("elapsed_before_artifact_write_seconds"),
        label="elapsed_before_artifact_write_seconds",
    )
    recovery = _runtime_seconds(
        recovery_elapsed_seconds,
        label="recovery_elapsed_seconds",
    )
    combined = original + recovery
    return {
        "full_invocation_elapsed_seconds": combined,
        "full_invocation_runtime_scope": _RECOVERY_RUNTIME_SCOPE,
        "full_invocation_runtime_censoring": {
            "status": "right-censored-lower-bound",
            "known_original_lower_bound_seconds": original,
            "recovery_elapsed_seconds": recovery,
            "combined_lower_bound_seconds": combined,
            "omitted_interval_seconds": None,
            "omitted_interval_scope": _RECOVERY_OMITTED_INTERVAL_SCOPE,
            "omitted_interval_reason": _RECOVERY_OMITTED_INTERVAL_REASON,
        },
    }


def recover_pending_training_artifact(
    checkpoint_dir: Path,
    durable_output_root: Path,
    *,
    auto_model_class: Any | None = None,
    tokenizer_class: Any | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    utc_now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Complete a sealed pending checkpoint without training or remote writes."""

    monotonic_clock = time.monotonic if monotonic_clock is None else monotonic_clock
    utc_now = (lambda: datetime.now(UTC)) if utc_now is None else utc_now
    recovery_started = monotonic_clock()
    checkpoint_dir = checkpoint_dir.resolve()
    completion_lock = _acquire_checkpoint_completion_lock(checkpoint_dir)
    try:
        prepared = _prepare_training_artifact_completion(
            checkpoint_dir,
            durable_output_root=durable_output_root,
            completion_path="recovery",
            auto_model_class=auto_model_class,
            tokenizer_class=tokenizer_class,
        )
        recovery_elapsed = monotonic_clock() - recovery_started
        completed_at = utc_now()
        if completed_at.tzinfo is None:
            raise RuntimeError("Recovery completion timestamp must be timezone-aware")
        complete = _finalize_training_artifact_completion(
            checkpoint_dir,
            prepared,
            completed_at_utc=completed_at.astimezone(UTC).isoformat(),
            runtime_provenance=_recovery_runtime_provenance(
                prepared,
                recovery_elapsed_seconds=recovery_elapsed,
            ),
        )
        # The compare-and-replace status transition is the final artifact mutation.
        _write_completed_training_manifest(
            checkpoint_dir,
            complete,
            completion_lock=completion_lock,
        )
        return complete
    finally:
        _release_checkpoint_completion_lock(completion_lock)


def _run_cuda_training(
    *,
    config: ScratchRunConfig,
    data: PreparedScratchData,
    resolved: ResolvedScratchConfig,
    source_code: Mapping[str, Any],
    ignored_source_paths: Sequence[Path],
    started_at_utc: str,
    run_started_monotonic: float,
) -> dict[str, Any]:
    configure_deterministic_cuda_environment()
    import torch
    import wandb
    from datasets import Dataset
    from transformers import PreTrainedTokenizerFast
    from trl import SFTTrainer

    from experiments.probe_deep_corpus_v2_scratch import (
        NvidiaSmiGpuMemorySampler,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Scratch training requires an actual CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Scratch training requires bf16 support")
    destination = config.output_dir.resolve()
    staging = _create_training_staging_dir(destination)
    staged_config = replace(config, output_dir=staging)
    active_ignored_paths = (*ignored_source_paths, staging)
    run = None
    sampler = None
    staging_published = False
    completion_lock = None
    try:
        if source_code.get("dirty"):
            raise RuntimeError("Actual scratch training requires a clean source tree")
        _require_unchanged_source(
            Path(__file__).resolve().parents[1],
            source_code,
            ignored_paths=active_ignored_paths,
        )
        feasibility_gate = _validate_feasibility_for_run(
            config=config,
            data=data,
            resolved=resolved,
            torch=torch,
            source_code=source_code,
        )
        allocation = feasibility_gate["hard_stop_allocation"]
        deadline = WallClockDeadline(
            started_monotonic=run_started_monotonic,
            allocation_seconds=float(allocation["current_run_allocation_seconds"]),
        )
        deadline.check("before_model_setup")

        actual_gpu = feasibility_gate["run_gpu"]
        sampler = NvidiaSmiGpuMemorySampler(gpu_uuid=actual_gpu["uuid"])
        sampler.start()
        torch.cuda.reset_peak_memory_stats()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.tokenizer_dir,
            local_files_only=True,
        )
        model, initialization = initialize_model_from_config(
            resolved,
            initialization_seed=config.initialization_seed,
            torch_dtype=torch.bfloat16,
        )
        sampler.sample_now()
        deadline.check("after_model_setup")
        lock = _selection_lock(config.selection_lock)
        if config.phase == "final":
            assert lock is not None
            validate_final_restart(
                selection_lock=lock,
                arm_id=config.arm_id,
                learning_rate=config.learning_rate,
                initialization_provenance=initialization,
                num_train_epochs=config.num_train_epochs,
            )

        wandb_work_dir = ignored_source_paths[1]
        wandb_work_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            entity=PERSONAL_WANDB_ENTITY,
            project=config.wandb_project,
            name=config.wandb_run_name,
            mode="online",
            dir=str(wandb_work_dir),
            tags=["deep-corpus-v2", "scratch", config.phase, config.arm_id],
            config={
                "architecture": resolved.identity_dict(),
                "creation": initialization,
                "source_data": dict(data.source_identity),
                "tokenizer_digest": data.tokenizer_manifest["artifact_digest"],
                **_training_protocol(config),
            },
        )
        if run is None:
            raise RuntimeError("W&B did not create an online run")
        deadline.check("after_wandb_setup")
        wandb_url = validate_wandb_url(str(run.url), project=config.wandb_project)
        training_args = build_sft_config(staged_config, report_to_wandb=True)
        hard_stop_callback = make_wall_clock_hard_stop_callback(deadline)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list([dict(row) for row in data.train_rows]),
            eval_dataset=Dataset.from_list([dict(row) for row in data.validation_rows]),
            processing_class=tokenizer,
            callbacks=[hard_stop_callback],
        )
        if trainer.model is not model:
            raise RuntimeError(
                "SFTTrainer did not retain the config-created model object"
            )
        assert_config_created_model(trainer.model, resolved)
        deadline.check("before_train")
        train_result = trainer.train()
        deadline.check("after_train")
        final_validation = trainer.evaluate()
        deadline.check("after_final_validation")
        sampler.sample_now()
        loss_curve = _loss_curve(trainer.state.log_history)
        state = {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "num_input_tokens_seen": getattr(
                trainer.state, "num_input_tokens_seen", None
            ),
            "total_flos": trainer.state.total_flos,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": (
                Path(trainer.state.best_model_checkpoint).name
                if trainer.state.best_model_checkpoint
                else None
            ),
            "train_metrics": dict(train_result.metrics),
            "selected_validation_metrics": dict(final_validation),
            "loss_curve": loss_curve,
        }
        _require_unchanged_source(
            Path(__file__).resolve().parents[1],
            source_code,
            ignored_paths=active_ignored_paths,
        )
        gpu_memory_telemetry = sampler.stop()
        sampler = None
        execution_environment = _cuda_environment(
            torch,
            gpu_memory_telemetry=gpu_memory_telemetry,
        )
        execution_environment["wall_clock_hard_stop"] = {
            **allocation,
            "elapsed_before_artifact_write_seconds": deadline.elapsed_seconds(),
            "enforcement": (
                "checked at setup boundaries, every optimizer step, epoch "
                "evaluation/save callbacks, final evaluation, and publication"
            ),
        }
        manifest = _build_manifest(
            config=config,
            data=data,
            resolved=resolved,
            initialization=initialization,
            source_code=source_code,
            status=_PENDING_DURABILITY_STATUS,
            started_at_utc=started_at_utc,
            wandb_url=wandb_url,
            trainer_state=state,
            execution_environment=execution_environment,
            selection_lock=lock,
            feasibility_gate=feasibility_gate,
        )
        deadline.check("before_final_artifact_write")
        trainer.save_model(str(staging))
        trainer.state.save_to_json(str(staging / "trainer_state.json"))
        _write_json(
            staging / "training_args.json",
            _published_training_args(
                training_args,
                staging=staging,
                destination=destination,
            ),
        )
        _write_json(staging / "basis_set.json", data.basis.to_manifest())
        _write_json(staging / "corpus_manifest.json", data.corpus_manifest)
        shutil.copyfile(
            config.tokenizer_dir / TOKENIZER_MANIFEST_FILENAME,
            staging / TOKENIZER_MANIFEST_FILENAME,
        )
        _write_wandb_link(staging / "wandb_run.url", wandb_url)
        _remove_intermediate_checkpoints(staging)
        manifest["checkpoint_identity"] = _checkpoint_content_identity(staging)
        manifest = _seal_training_manifest(manifest)
        _write_json(staging / TRAINING_MANIFEST_FILENAME, manifest)
        assert_weights_only_artifact(staging, require_complete=True)
        deadline.check("after_final_artifact_write")
        _require_unchanged_source(
            Path(__file__).resolve().parents[1],
            source_code,
            ignored_paths=active_ignored_paths,
        )
        deadline.check("before_atomic_publish")
        completion_lock = _acquire_checkpoint_completion_lock(destination)
        _publish_training_staging(staging, destination)
        staging_published = True
        assert config.durable_output_root is not None
        manifest = _prepare_training_artifact_completion(
            destination,
            transient_staging=staging,
            durable_output_root=config.durable_output_root,
            completion_path="training",
            pending_manifest=manifest,
            completion_check=lambda: deadline.check("after_durable_checkpoint_reload"),
        )
        _require_unchanged_source(
            Path(__file__).resolve().parents[1],
            source_code,
            ignored_paths=active_ignored_paths,
        )
        run.summary.update(
            {
                "artifact_path": str(destination),
                "artifact_content_digest": manifest["checkpoint_identity"][
                    "content_digest"
                ],
                "durability_verified": True,
                "arm_id": config.arm_id,
                "phase": config.phase,
                "exact_trainable_parameters": resolved.exact_trainable_parameters,
                "initial_state_digest": initialization["initial_state_digest"],
                "corpus_manifest_digest": data.corpus_manifest["manifest_digest"],
                "tokenizer_digest": data.tokenizer_manifest["artifact_digest"],
                "training_examples": len(data.train_rows),
                "validation_examples": len(data.validation_rows),
                "best_validation_loss": manifest["best_validation_loss"],
                "optimizer_steps": state["global_step"],
                "num_input_tokens_seen": state["num_input_tokens_seen"],
                "planned_combined_training_tokens": manifest["workload"][
                    "planned_combined_training_tokens"
                ],
                "peak_torch_allocated_bytes": execution_environment[
                    "peak_torch_allocated_bytes"
                ],
                "peak_torch_reserved_bytes": execution_environment[
                    "peak_torch_reserved_bytes"
                ],
                "peak_observed_gpu_memory_bytes": execution_environment[
                    "peak_observed_gpu_memory_bytes"
                ],
            }
        )
        run.finish()
        run = None
        deadline.check("before_durable_completion_transition")
        manifest = _finalize_training_artifact_completion(
            destination,
            manifest,
            completed_at_utc=datetime.now(UTC).isoformat(),
            runtime_provenance=_normal_runtime_provenance(deadline.elapsed_seconds()),
        )
        # Every failure-capable validation and external mutation is complete.
        # The on-disk status transition is the final artifact mutation.
        assert completion_lock is not None
        _write_completed_training_manifest(
            destination,
            manifest,
            completion_lock=completion_lock,
        )
        return manifest
    finally:
        active_error = sys.exc_info()[1]
        try:
            if sampler is not None:
                try:
                    sampler.stop()
                except RuntimeError:
                    if active_error is None:
                        raise
        finally:
            try:
                if run is not None:
                    run.finish()
            finally:
                try:
                    if not staging_published:
                        _cleanup_training_staging(staging, destination)
                finally:
                    if completion_lock is not None:
                        _release_checkpoint_completion_lock(completion_lock)


def run_training(config: ScratchRunConfig) -> dict[str, Any]:
    _validate_run_config(config)
    run_started_monotonic = time.monotonic()
    started_at_utc = datetime.now(UTC).isoformat()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = config.output_dir.resolve()
    if not config.dry_run and output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    wandb_work_dir = output_dir.parent / f".{output_dir.name}.wandb-work"
    ignored_source_paths = (
        output_dir,
        wandb_work_dir,
        *_existing_training_staging_dirs(output_dir),
    )
    source_code = _git_identity(repo_root, ignored_paths=ignored_source_paths)
    data = prepare_scratch_data(config)
    resolved = resolve_run_architecture(config, data)
    if config.dry_run:
        preview = _build_manifest(
            config=config,
            data=data,
            resolved=resolved,
            initialization={
                "creation_method": "AutoModelForCausalLM.from_config",
                "pretrained_weights_loaded": False,
                "initialization_seed": config.initialization_seed,
                "initial_state_digest": None,
            },
            source_code=source_code,
            status="dry-run",
            started_at_utc=started_at_utc,
            selection_lock=_selection_lock(config.selection_lock),
        )
        preview["planned_steps"] = math.ceil(
            len(data.train_rows)
            * config.num_train_epochs
            / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        )
        return preview
    return _run_cuda_training(
        config=config,
        data=data,
        resolved=resolved,
        source_code=source_code,
        ignored_source_paths=ignored_source_paths,
        started_at_utc=started_at_utc,
        run_started_monotonic=run_started_monotonic,
    )


def write_learning_rate_lock(
    arm_id: str,
    pilot_manifest_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Selection lock already exists: {output_path}")
    manifests = []
    for path in pilot_manifest_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not read pilot manifest {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"Pilot manifest is not a JSON object: {path}")
        manifests.append(value)
    lock = select_pilot_learning_rate(arm_id, manifests)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, lock)
    return lock


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--durable-output-root",
        type=Path,
        help=(
            "Existing non-temporary root that must contain --output-dir; "
            "required for actual runs"
        ),
    )
    train.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    train.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    train.add_argument(
        "--reference-corpus-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Pinned corpus whose roots/tasks V2 excludes; repeat for every "
            "target-contract reference (defaults to deep-corpus-v1)"
        ),
    )
    train.add_argument("--arm-id", choices=tuple(ARM_SPECS), required=True)
    train.add_argument("--phase", choices=("pilot", "final"), required=True)
    train.add_argument("--learning-rate", type=float, choices=LR_GRID, required=True)
    train.add_argument("--num-train-epochs", type=float, required=True)
    train.add_argument("--wandb-run-name")
    train.add_argument("--selection-lock", type=Path)
    train.add_argument("--feasibility-manifest", type=Path)
    train.add_argument("--dry-run", action="store_true")

    select = subparsers.add_parser("select")
    select.add_argument("--arm-id", choices=tuple(ARM_SPECS), required=True)
    select.add_argument("--pilot-manifest", type=Path, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)

    recover = subparsers.add_parser(
        "recover",
        help="Locally verify and complete a sealed pending durable checkpoint",
    )
    recover.add_argument("--checkpoint-dir", type=Path, required=True)
    recover.add_argument("--durable-output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    if arguments.mode == "select":
        result = write_learning_rate_lock(
            arguments.arm_id,
            arguments.pilot_manifest,
            arguments.output,
        )
    elif arguments.mode == "recover":
        result = recover_pending_training_artifact(
            arguments.checkpoint_dir,
            arguments.durable_output_root,
        )
    else:
        result = run_training(
            ScratchRunConfig(
                output_dir=arguments.output_dir,
                durable_output_root=arguments.durable_output_root,
                corpus_dir=arguments.corpus_dir,
                tokenizer_dir=arguments.tokenizer_dir,
                reference_corpus_dirs=tuple(
                    arguments.reference_corpus_dir or (DEFAULT_REFERENCE_CORPUS_DIR,)
                ),
                arm_id=arguments.arm_id,
                phase=arguments.phase,
                learning_rate=arguments.learning_rate,
                num_train_epochs=arguments.num_train_epochs,
                wandb_run_name=arguments.wandb_run_name,
                selection_lock=arguments.selection_lock,
                feasibility_manifest=arguments.feasibility_manifest,
                dry_run=arguments.dry_run,
            )
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
