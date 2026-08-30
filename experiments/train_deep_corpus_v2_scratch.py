"""Train issue #54's three causal LMs from pristine random initialization.

The runner consumes the one frozen deep-corpus-v2 tokenizer, uses discovery as
the only training distribution, evaluates loss on validation, and never exposes
the held-out test split to a model.  Pilot runs are fixed to one quarter epoch
over the predeclared learning-rate grid.  Final runs require a validation-only
selection lock and recreate the exact pristine initial state before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
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
    initialize_model_from_config,
    resolve_scratch_config,
    select_pilot_learning_rate,
    validate_final_restart,
)
from wandering_light.training.deep_corpus_tokenizer import (
    DEFAULT_REFERENCE_CORPUS_DIR,
    DETERMINISTIC_PYTHON_HASH_SEED,
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


@dataclass(frozen=True)
class ScratchRunConfig:
    output_dir: Path
    corpus_dir: Path
    tokenizer_dir: Path
    arm_id: str
    phase: str
    learning_rate: float
    num_train_epochs: float
    reference_corpus_dirs: tuple[Path, ...] = (DEFAULT_REFERENCE_CORPUS_DIR,)
    feasibility_manifest: Path | None = None
    wandb_run_name: str | None = None
    selection_lock: Path | None = None
    dry_run: bool = False
    initialization_seed: int = INITIALIZATION_SEED
    data_order_seed: int = DATA_ORDER_SEED
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT


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
        if not 0 < config.num_train_epochs <= MAX_FINAL_EPOCHS:
            raise ValueError(f"Final runs may use at most {MAX_FINAL_EPOCHS:g} epochs")
        if config.selection_lock is None:
            raise ValueError("Final training requires a validation-only LR lock")
    if not config.dry_run and (
        not isinstance(config.wandb_run_name, str) or not config.wandb_run_name.strip()
    ):
        raise ValueError("An actual run requires a recoverable W&B run name")
    if not config.dry_run and config.feasibility_manifest is None:
        raise ValueError("An actual run requires a passing feasibility manifest")


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


def _publish_training_staging(staging: Path, destination: Path) -> None:
    staging = staging.resolve()
    destination = destination.resolve()
    if staging.parent != destination.parent or not staging.name.startswith(
        f".{destination.name}.staging-"
    ):
        raise ValueError("Training staging path is not scoped to its destination")
    if not staging.is_dir():
        raise FileNotFoundError(f"Training staging directory is missing: {staging}")
    if destination.exists():
        raise FileExistsError(f"Output appeared before atomic publish: {destination}")
    staging.rename(destination)


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
            status="complete",
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
        manifest["execution_environment"]["full_invocation_elapsed_seconds"] = (
            deadline.elapsed_seconds()
        )
        manifest["execution_environment"]["full_invocation_runtime_scope"] = (
            "run_training entry through model/tokenizer/config/trainer artifact writes; "
            "excludes only manifest serialization, W&B summary finalization, and atomic rename"
        )
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(staging / "training_manifest.json", manifest)
        assert_weights_only_artifact(staging, require_complete=True)
        deadline.check("after_final_artifact_write")
        _require_unchanged_source(
            Path(__file__).resolve().parents[1],
            source_code,
            ignored_paths=active_ignored_paths,
        )
        run.summary.update(
            {
                "artifact_path": str(destination),
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
        deadline.check("before_atomic_publish")
        _publish_training_staging(staging, destination)
        assert_weights_only_artifact(destination, require_complete=True)
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
                _cleanup_training_staging(staging, destination)


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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    if arguments.mode == "select":
        result = write_learning_rate_lock(
            arguments.arm_id,
            arguments.pilot_manifest,
            arguments.output,
        )
    else:
        result = run_training(
            ScratchRunConfig(
                output_dir=arguments.output_dir,
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
