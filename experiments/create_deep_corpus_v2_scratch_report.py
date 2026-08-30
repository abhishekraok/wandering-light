"""Build issue #54's deterministic local JSON and Markdown report.

This command performs no network access.  It revalidates the frozen protocol,
all training/evaluation identities, the once-only test ledger, and the W&B
receipts that were written by the separate online publisher.  It then writes a
fresh output directory atomically.

Run it while every validation/test result and checkpoint remains at the live
absolute path sealed by evaluation.  Relocation is intentionally rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiments.evaluate_deep_corpus_v2_scratch import (
    BFS_FILENAME,
    DEFAULT_TOKENIZER_DIR,
    SUMMARY_DIGEST_FIELD,
    SUMMARY_FILENAME,
    TEST_LEDGER_FILENAME,
    TEST_SUMMARY_FILENAME,
    _training_outcomes,
    _validate_checkpoint_feasibility_gate,
    _validate_raw_artifacts,
    load_protocol_lock,
    load_validation_results,
    validate_evaluation_summary,
    validate_wandb_receipt,
)
from experiments.freeze_deep_corpus_v2_protocol import (
    BFS_EXECUTION_SAFETY_BUDGET,
    GENERATION_BATCH_SIZE,
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    PROTOCOL_SELECTION_SEED,
    REPO_ROOT,
)
from experiments.probe_deep_corpus_v2_scratch import load_probe_manifest
from experiments.train_deep_corpus_sft import validate_wandb_url
from wandering_light.evals.deep_corpus_protocol import (
    ARM_IDS,
    SAMPLE_PREFIXES,
    canonical_copy,
    sha256_file,
    sha256_json,
    validate_test_opening_ledger,
)
from wandering_light.training.deep_corpus_scratch import (
    DEFAULT_WANDB_PROJECT,
    LR_GRID,
    PERSONAL_WANDB_ENTITY,
)
from wandering_light.training.deep_corpus_tokenizer import (
    DETAIL_FILENAME,
    validate_frozen_tokenizer,
)

SCHEMA_VERSION = 1
REPORT_JSON_FILENAME = "report.json"
REPORT_MARKDOWN_FILENAME = "report.md"
REPORT_DIGEST_FIELD = "report_sha256"
DISTANCES = tuple(range(1, 9))
TARGET_DISTANCES = tuple(range(5, 9))
ARM_LABELS = {
    "scratch-10m": "10M",
    "opt-125m-scratch": "OPT 125M",
    "smollm2-360m-scratch": "SmolLM2 360M",
}
FROZEN_WORKLOAD_INVARIANTS = (
    "training data",
    "epochs",
    "context length",
    "parameter count",
)


@dataclass(frozen=True)
class ReportInputs:
    protocol_lock: Path
    corpus_manifest: Path
    tokenizer_dir: Path
    feasibility_manifest: Path
    validation_results: tuple[Path, ...]
    pilot_manifests: tuple[Path, ...]
    test_dir: Path
    output_dir: Path
    repo_root: Path = REPO_ROOT


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{label} is outside its valid range")
    return parsed


def _validate_issue_54_protocol(protocol: Mapping[str, Any]) -> None:
    if tuple(protocol.get("allowed_arm_ids", ())) != ARM_IDS:
        raise ValueError("Protocol does not contain exactly issue #54's three arms")
    generation = protocol.get("generation", {})
    expected_generation = {
        "sampling_seed": PROTOCOL_SELECTION_SEED,
        "sample_prefixes": list(SAMPLE_PREFIXES),
        "samples_per_task": SAMPLE_PREFIXES[-1],
        "temperature": GENERATION_TEMPERATURE,
        "top_p": GENERATION_TOP_P,
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "batch_size": GENERATION_BATCH_SIZE,
        "max_sequence_length": 2048,
        "do_sample": True,
    }
    if {key: generation.get(key) for key in expected_generation} != expected_generation:
        raise ValueError("Protocol generation settings differ from issue #54")
    bfs = protocol.get("bfs", {})
    if (bfs.get("budget"), bfs.get("max_depth")) != (
        BFS_EXECUTION_SAFETY_BUDGET,
        8,
    ):
        raise ValueError("Protocol BFS settings differ from issue #54")


def _study_arm_scope(
    feasibility: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the canonical active/deferred partition sealed by feasibility."""

    gate = feasibility.get("launch_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Feasibility manifest has no launch gate")
    per_arm = gate.get("per_arm")
    if not isinstance(per_arm, Mapping) or set(per_arm) != set(ARM_IDS):
        raise ValueError("Feasibility launch gate must cover all three arms")
    active = tuple(
        arm_id
        for arm_id in ARM_IDS
        if isinstance(per_arm[arm_id], Mapping)
        and per_arm[arm_id].get("launch_allowed") is True
    )
    deferred = tuple(arm_id for arm_id in ARM_IDS if arm_id not in active)
    if not active:
        raise ValueError("Feasibility manifest authorizes no reportable arm")
    if gate.get("authorized_arm_ids") != list(active):
        raise ValueError("Feasibility authorized-arm list differs from per-arm gates")
    if gate.get("blocked_arm_ids") != list(deferred):
        raise ValueError("Feasibility blocked-arm list differs from per-arm gates")
    return active, deferred


def _deferred_arm_report(arm_id: str, *, arm_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve why an arm was deferred and the non-destructive ways it can fit."""

    if arm_gate.get("launch_allowed") is not False:
        raise ValueError(f"Deferred arm {arm_id} is unexpectedly launch-authorized")
    if arm_gate.get("route") != "blocked":
        raise ValueError(f"Deferred arm {arm_id} does not use the blocked route")
    baseline = arm_gate.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"Deferred arm {arm_id} has no runtime projection")
    required = (
        "projected_pilot_run_hours",
        "projected_pilot_hours",
        "projected_final_hours",
        "projected_pilot_plus_final_hours",
    )
    projection = canonical_copy(dict(baseline))
    for field in required:
        projection[field] = _finite_number(
            baseline.get(field), label=f"{arm_id} {field}", minimum=0
        )
    pilot_runs = baseline.get("pilot_runs", len(LR_GRID))
    if type(pilot_runs) is not int or pilot_runs != len(LR_GRID):
        raise ValueError(f"Deferred arm {arm_id} projection has the wrong pilot count")
    projection["pilot_runs"] = pilot_runs
    if not math.isclose(
        projection["projected_pilot_hours"],
        projection["projected_pilot_run_hours"] * pilot_runs,
    ) or not math.isclose(
        projection["projected_pilot_plus_final_hours"],
        projection["projected_pilot_hours"] + projection["projected_final_hours"],
    ):
        raise ValueError(f"Deferred arm {arm_id} runtime projection is inconsistent")
    projection_includes = baseline.get("projection_includes")
    if (
        not isinstance(projection_includes, list)
        or not all(isinstance(item, str) for item in projection_includes)
        or not any("quarter-epoch pilot" in item for item in projection_includes)
        or not any("three final epoch" in item for item in projection_includes)
        or not any("15%" in item for item in projection_includes)
    ):
        raise ValueError(f"Deferred arm {arm_id} projection context is incomplete")
    return {
        "status": "deferred",
        "route": arm_gate.get("route"),
        "reason": arm_gate.get("reason"),
        "projection": projection,
        "projection_context": {
            "learning_rate_pilots": pilot_runs,
            "full_final_epochs": 3,
            "margin_percent": 15,
            "includes": canonical_copy(projection_includes),
        },
        "fit_options": [
            {
                "option": "fewer_learning_rate_pilots",
                "effect": (
                    "reduce pilot overhead under a separately frozen LR-selection "
                    "protocol"
                ),
                "requires_protocol_change": True,
            },
            {
                "option": "own_night",
                "effect": "run under its own separately authorized runtime envelope",
                "requires_protocol_change": False,
            },
        ],
        "must_not_shrink": list(FROZEN_WORKLOAD_INVARIANTS),
    }


def _validate_feasibility_producer(
    feasibility: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> None:
    source = feasibility.get("source_code")
    evaluator_sources = protocol.get("evaluator_sources")
    if not isinstance(source, Mapping) or not isinstance(evaluator_sources, Mapping):
        raise ValueError("Feasibility/protocol producer identity is missing")
    metadata = evaluator_sources.get("metadata")
    files = evaluator_sources.get("files")
    if not isinstance(metadata, Mapping) or not isinstance(files, list):
        raise ValueError("Frozen evaluator source identity is incomplete")
    if source.get("dirty") is not False or source.get("commit") != metadata.get(
        "git_commit"
    ):
        raise ValueError("Feasibility probe was produced outside the frozen commit")
    path = "experiments/probe_deep_corpus_v2_scratch.py"
    frozen = next(
        (row for row in files if isinstance(row, Mapping) and row.get("path") == path),
        None,
    )
    runner = source.get("probe_runner")
    if (
        not isinstance(frozen, Mapping)
        or not isinstance(runner, Mapping)
        or runner.get("path") != path
        or runner.get("sha256") != frozen.get("sha256")
    ):
        raise ValueError(
            "Feasibility probe runner differs from frozen evaluator source"
        )


def _validate_combined_test_summary(
    summary: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    active_arm_ids: Sequence[str] = ARM_IDS,
    deferred_arm_ids: Sequence[str] = (),
    feasibility_artifact_digest: str | None = None,
) -> dict[str, Any]:
    active = tuple(active_arm_ids)
    deferred = tuple(deferred_arm_ids)
    value = canonical_copy(dict(summary))
    recorded = value.pop(SUMMARY_DIGEST_FIELD, None)
    if recorded != sha256_json(value):
        raise ValueError("Combined test summary changed after it was sealed")
    value[SUMMARY_DIGEST_FIELD] = recorded
    if (value.get("schema_version"), value.get("status"), value.get("split")) != (
        SCHEMA_VERSION,
        "complete",
        "test",
    ):
        raise ValueError("Combined test summary is not complete")
    if value.get("protocol_lock_sha256") != protocol["protocol_lock_sha256"]:
        raise ValueError("Combined test summary belongs to another protocol")
    if value.get("test_cohort_sha256") != protocol["cohorts"]["test"]["cohort_sha256"]:
        raise ValueError("Combined test summary belongs to another test cohort")
    if value.get("test_access_count") != 1:
        raise ValueError("Combined test summary does not prove exactly one opening")
    has_explicit_scope = "active_arm_ids" in value or "deferred_arm_ids" in value
    if deferred or has_explicit_scope:
        if value.get("active_arm_ids") != list(active) or value.get(
            "deferred_arm_ids"
        ) != list(deferred):
            raise ValueError("Combined test summary changed its active/deferred scope")
        if value.get("feasibility_artifact_digest") != feasibility_artifact_digest:
            raise ValueError("Combined test summary used another feasibility artifact")
    elif active != ARM_IDS:
        raise ValueError("Partial combined test summary has no explicit arm scope")
    for field in ("arm_summary_sha256_by_arm", "arms"):
        if not isinstance(value.get(field), dict) or set(value[field]) != set(active):
            raise ValueError(f"Combined test summary has invalid {field}")
    funnels = value.get("candidate_funnels", {})
    if (
        not isinstance(funnels, dict)
        or funnels.get("arm_ids") != list(active)
        or tuple(funnels.get("sample_prefixes", ())) != SAMPLE_PREFIXES
        or not isinstance(funnels.get("arms"), dict)
        or set(funnels["arms"]) != set(active)
    ):
        raise ValueError("Combined test summary has incomplete candidate funnels")
    return value


def _load_test_arms(
    test_dir: Path,
    *,
    protocol: Mapping[str, Any],
    combined: Mapping[str, Any],
    validation: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[str, Any],
    active_arm_ids: Sequence[str] = ARM_IDS,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for arm_id in active_arm_ids:
        root = test_dir / "arms" / arm_id
        summary = validate_evaluation_summary(
            _read_json(root / SUMMARY_FILENAME, label="test arm summary"),
            protocol_lock_sha256=protocol["protocol_lock_sha256"],
            split="test",
            arm_id=arm_id,
        )
        _validate_raw_artifacts(root, summary)
        receipt = validate_wandb_receipt(root, summary=summary)
        if (
            summary[SUMMARY_DIGEST_FIELD]
            != combined["arm_summary_sha256_by_arm"][arm_id]
        ):
            raise ValueError("Combined test summary changed an arm result")
        identity = ledger["arms"][arm_id]["result_identity"]
        expected_identity = {
            "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
            "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
            "candidates_sha256": summary["artifacts"]["candidates"]["sha256"],
        }
        if identity != expected_identity:
            raise ValueError("Test ledger arm identity differs from its sealed result")
        selected = validation[arm_id]["checkpoint"]
        if summary["checkpoint"] != selected:
            raise ValueError("Test arm did not use its validation-locked checkpoint")
        expected_combined_arm = {
            "model": summary["model"],
            "bfs": summary["bfs"],
            "matched_compute": summary["matched_compute"],
            "training": summary["training"],
            "generation_runtime": summary["generation"]["runtime"],
        }
        if combined["arms"][arm_id] != expected_combined_arm:
            raise ValueError("Combined test summary changed an arm result")
        if combined["candidate_funnels"]["arms"][arm_id] != summary["candidate_funnel"]:
            raise ValueError("Combined test summary changed an arm candidate funnel")
        results[arm_id] = {
            "summary": summary,
            "wandb_receipt": receipt,
            "directory": root,
        }
    return results


def _study_result_by_arm(
    feasibility: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = feasibility.get("arm_results")
    if not isinstance(rows, list):
        raise ValueError("Feasibility manifest has no arm results")
    by_arm = {row.get("arm_id"): row for row in rows if isinstance(row, Mapping)}
    if set(by_arm) != set(ARM_IDS) or len(rows) != len(ARM_IDS):
        raise ValueError("Feasibility manifest must contain exactly three arm results")
    return {arm_id: by_arm[arm_id] for arm_id in ARM_IDS}


def _validate_feasibility_link(
    manifest: Mapping[str, Any],
    *,
    arm_id: str,
    feasibility: Mapping[str, Any],
    study_result: Mapping[str, Any],
) -> None:
    gate = _validate_checkpoint_feasibility_gate(manifest, arm_id=arm_id)
    if gate.get("artifact_digest") != feasibility.get("artifact_digest"):
        raise ValueError("Training run used another feasibility artifact")
    if gate.get("study_identity_digest") != study_result.get("study_identity_digest"):
        raise ValueError("Training run used another feasibility study identity")
    expected_gate = feasibility.get("launch_gate", {}).get("per_arm", {}).get(arm_id)
    if canonical_copy(gate.get("arm_gate")) != canonical_copy(expected_gate):
        raise ValueError("Training run's feasibility authorization changed")


def _loss_curve(manifest: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    trainer = manifest.get("trainer_state", {})
    curve = trainer.get("loss_curve") if isinstance(trainer, Mapping) else None
    if not isinstance(curve, list) or not curve:
        raise ValueError(f"{label} has no observed train/validation loss curve")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(curve):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} loss curve row is malformed")
        train_loss = _finite_number(
            row.get("preceding_train_window_loss"),
            label=f"{label} train loss {index}",
            minimum=0,
        )
        validation_loss = _finite_number(
            row.get("validation_loss"),
            label=f"{label} validation loss {index}",
            minimum=0,
        )
        normalized.append(
            {
                **canonical_copy(dict(row)),
                "train_validation_gap": validation_loss - train_loss,
            }
        )
    return normalized


def _training_run_outcome(
    manifest: Mapping[str, Any], *, arm_id: str, phase: str
) -> dict[str, Any]:
    if (
        manifest.get("status") != "complete"
        or manifest.get("phase") != phase
        or manifest.get("arm_id") != arm_id
    ):
        raise ValueError(f"{arm_id} {phase} training manifest is not complete")
    creation = manifest.get("creation", {})
    if (
        not isinstance(creation, Mapping)
        or creation.get("creation_method") != "AutoModelForCausalLM.from_config"
        or creation.get("pretrained_weights_loaded") is not False
    ):
        raise ValueError(f"{arm_id} {phase} lacks random-initialization provenance")
    trainer = manifest.get("trainer_state", {})
    workload = manifest.get("workload", {})
    environment = manifest.get("execution_environment", {})
    protocol = manifest.get("training_protocol", {})
    metrics = trainer.get("train_metrics", {}) if isinstance(trainer, Mapping) else {}
    if not all(
        isinstance(value, Mapping)
        for value in (trainer, workload, environment, protocol, metrics)
    ):
        raise ValueError(f"{arm_id} {phase} has malformed training outcomes")
    for field in ("cublas_workspace_config", "pythonhashseed"):
        if environment.get(field) != protocol.get(field):
            raise ValueError(
                f"{arm_id} {phase} deterministic runtime differs from its protocol: "
                f"{field}"
            )
    steps = trainer.get("global_step")
    tokens = trainer.get("num_input_tokens_seen")
    if type(steps) is not int or steps <= 0 or type(tokens) is not int or tokens <= 0:
        raise ValueError(f"{arm_id} {phase} lacks exact observed steps/tokens")
    trainer_runtime = _finite_number(
        metrics.get("train_runtime"),
        label=f"{arm_id} {phase} trainer runtime",
        minimum=0,
    )
    runtime = _finite_number(
        environment.get("full_invocation_elapsed_seconds"),
        label=f"{arm_id} {phase} full invocation runtime",
        minimum=0,
    )
    runtime_scope = environment.get("full_invocation_runtime_scope")
    if not isinstance(runtime_scope, str) or not runtime_scope:
        raise ValueError(f"{arm_id} {phase} lacks a full-runtime scope")
    allocated = environment.get("peak_torch_allocated_bytes")
    reserved = environment.get("peak_torch_reserved_bytes")
    observed_peak = environment.get("peak_observed_gpu_memory_bytes")
    gpu_memory_telemetry = environment.get("gpu_memory_telemetry")
    memory_measurement_note = environment.get("memory_measurement_note")
    physical = environment.get("physical_vram_bytes")
    measurement_scope = (
        gpu_memory_telemetry.get("measurement_scope")
        if isinstance(gpu_memory_telemetry, Mapping)
        else None
    )
    telemetry_peak = (
        gpu_memory_telemetry.get("peak_observed_used_bytes")
        if isinstance(gpu_memory_telemetry, Mapping)
        else None
    )
    if (
        type(allocated) is not int
        or type(reserved) is not int
        or type(observed_peak) is not int
        or not isinstance(gpu_memory_telemetry, Mapping)
        or not gpu_memory_telemetry
        or measurement_scope not in {"process", "physical_gpu_total"}
        or telemetry_peak != observed_peak
        or type(gpu_memory_telemetry.get("samples")) is not int
        or gpu_memory_telemetry["samples"] <= 0
        or not isinstance(memory_measurement_note, str)
        or not memory_measurement_note
        or type(physical) is not int
        or not 0 < allocated <= reserved <= physical
        or not 0 < observed_peak <= physical
    ):
        raise ValueError(f"{arm_id} {phase} has invalid VRAM telemetry")
    if measurement_scope == "process" and (
        gpu_memory_telemetry.get("peak_process_used_bytes") != observed_peak
    ):
        raise ValueError(f"{arm_id} {phase} has invalid process VRAM telemetry")
    if measurement_scope == "physical_gpu_total" and (
        type(gpu_memory_telemetry.get("baseline_observed_used_bytes")) is not int
        or not 0
        <= gpu_memory_telemetry["baseline_observed_used_bytes"]
        <= observed_peak
        or gpu_memory_telemetry.get("peak_physical_gpu_used_bytes") != observed_peak
        or not isinstance(gpu_memory_telemetry.get("limitation"), str)
        or not gpu_memory_telemetry["limitation"]
    ):
        raise ValueError(
            f"{arm_id} {phase} has invalid physical-GPU-total VRAM telemetry"
        )
    wandb = manifest.get("wandb", {})
    if (
        not isinstance(wandb, Mapping)
        or wandb.get("entity") != PERSONAL_WANDB_ENTITY
        or wandb.get("project") != DEFAULT_WANDB_PROJECT
    ):
        raise ValueError(f"{arm_id} {phase} training W&B identity is invalid")
    wandb_url = validate_wandb_url(str(wandb.get("url")), project=DEFAULT_WANDB_PROJECT)
    curve = _loss_curve(manifest, label=f"{arm_id} {phase}")
    return {
        "phase": phase,
        "learning_rate": manifest.get("learning_rate"),
        "epochs": manifest.get("num_train_epochs"),
        "global_steps": steps,
        "num_input_tokens_seen": tokens,
        "unique_training_examples": workload.get("unique_training_examples"),
        "planned_example_presentations": workload.get("planned_example_presentations"),
        "planned_combined_training_tokens": workload.get(
            "planned_combined_training_tokens"
        ),
        "planned_completion_loss_tokens": workload.get(
            "planned_completion_loss_tokens"
        ),
        "runtime_seconds": runtime,
        "runtime_scope": runtime_scope,
        "trainer_train_runtime_seconds": trainer_runtime,
        "peak_torch_allocated_bytes": allocated,
        "peak_torch_reserved_bytes": reserved,
        "peak_observed_gpu_memory_bytes": observed_peak,
        "gpu_memory_telemetry": canonical_copy(dict(gpu_memory_telemetry)),
        "physical_vram_bytes": physical,
        "vram_measurement_note": memory_measurement_note,
        "best_validation_loss": manifest.get("best_validation_loss"),
        "validation_trend": canonical_copy(manifest.get("validation_trend")),
        "loss_curve": curve,
        "wandb_url": wandb_url,
        "manifest_digest": sha256_json(manifest),
    }


def _shared_training_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest.get("source_data", {})
    roles = source.get("task_roles", {}) if isinstance(source, Mapping) else {}
    discovery = roles.get("discovery", {}) if isinstance(roles, Mapping) else {}
    protocol = manifest.get("training_protocol", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    normalized_protocol = {
        key: value
        for key, value in protocol.items()
        if key not in {"phase", "num_train_epochs", "learning_rate"}
    }
    return {
        "architecture": canonical_copy(manifest.get("architecture")),
        "initial_state_digest": manifest.get("initial_state_digest"),
        "initialization_seed": manifest.get("initialization_seed"),
        "data_order_seed": manifest.get("data_order_seed"),
        "tokenizer_digest": manifest.get("tokenizer_digest"),
        "corpus_manifest_digest": source.get("corpus_manifest_digest"),
        "retained_training_task_ids_digest": discovery.get("retained_task_ids_digest"),
        "effective_config_digest": manifest.get("effective_config_digest"),
        "exact_trainable_parameters": manifest.get("exact_trainable_parameters"),
        "training_protocol": canonical_copy(normalized_protocol),
    }


def _load_pilot_outcomes(
    paths: Sequence[Path],
    *,
    validation: Mapping[str, Mapping[str, Any]],
    feasibility: Mapping[str, Any],
    feasibility_results: Mapping[str, Mapping[str, Any]],
    active_arm_ids: Sequence[str] = ARM_IDS,
) -> dict[str, list[dict[str, Any]]]:
    active = tuple(active_arm_ids)
    expected_count = len(active) * len(LR_GRID)
    if len(paths) != expected_count:
        count_label = "nine" if expected_count == 9 else str(expected_count)
        raise ValueError(
            f"Exactly {count_label} pilot training manifests are required for "
            f"the active arms"
        )
    manifests: dict[tuple[str, float], dict[str, Any]] = {}
    for path in paths:
        manifest = _read_json(path, label="pilot training manifest")
        arm_id = manifest.get("arm_id")
        rate = manifest.get("learning_rate")
        if (
            arm_id not in active
            or isinstance(rate, bool)
            or not isinstance(rate, (int, float))
        ):
            raise ValueError("Pilot manifest has an invalid arm or learning rate")
        key = (arm_id, float(rate))
        if key in manifests:
            raise ValueError("Pilot manifests contain a duplicate arm/learning rate")
        manifests[key] = manifest
    expected = {(arm_id, float(rate)) for arm_id in active for rate in LR_GRID}
    if set(manifests) != expected:
        raise ValueError("Pilots must cover every active arm and predeclared LR")

    outcomes: dict[str, list[dict[str, Any]]] = {}
    for arm_id in active:
        final = validation[arm_id]["training_manifest"]
        final_identity = _shared_training_identity(final)
        lock = final.get("selection_lock", {})
        pilot_lock_rows = lock.get("pilots") if isinstance(lock, Mapping) else None
        if not isinstance(pilot_lock_rows, list) or len(pilot_lock_rows) != len(
            LR_GRID
        ):
            raise ValueError(f"{arm_id} final checkpoint has no three-pilot lock")
        locked = {float(row["learning_rate"]): row for row in pilot_lock_rows}
        arm_outcomes: list[dict[str, Any]] = []
        for rate in LR_GRID:
            manifest = manifests[(arm_id, float(rate))]
            if sha256_json(manifest) != locked[float(rate)].get("manifest_digest"):
                raise ValueError("Pilot manifest digest differs from final LR lock")
            if float(manifest.get("best_validation_loss", math.inf)) != float(
                locked[float(rate)].get("best_validation_loss", -math.inf)
            ):
                raise ValueError("Pilot validation loss differs from final LR lock")
            if _shared_training_identity(manifest) != final_identity:
                raise ValueError("Pilot and final training identities differ")
            _validate_feasibility_link(
                manifest,
                arm_id=arm_id,
                feasibility=feasibility,
                study_result=feasibility_results[arm_id],
            )
            arm_outcomes.append(
                _training_run_outcome(manifest, arm_id=arm_id, phase="pilot")
            )
        selected_rate = float(lock.get("selected_learning_rate", -1))
        selected_manifest = manifests.get((arm_id, selected_rate))
        if selected_manifest is None or sha256_json(selected_manifest) != lock.get(
            "selected_pilot_manifest_digest"
        ):
            raise ValueError("Selected pilot identity differs from final LR lock")
        outcomes[arm_id] = arm_outcomes
    return outcomes


def _distance_aggregate(
    model: Mapping[str, Any], distances: Sequence[int]
) -> dict[str, Any]:
    rows = model.get("by_certified_distance_at_64", {})
    try:
        selected = [rows[str(distance)] for distance in distances]
        tasks = sum(int(row["tasks"]) for row in selected)
        solved = sum(int(row["solved"]) for row in selected)
        exact = sum(int(row["exact_matches"]) for row in selected)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Evaluation has incomplete per-distance @64 metrics"
        ) from error
    if tasks <= 0:
        raise ValueError("Evaluation distance aggregate has no tasks")
    return {
        "distances": list(distances),
        "tasks": tasks,
        "solved": solved,
        "solve_rate": solved / tasks,
        "exact_matches": exact,
        "exact_match_rate": exact / tasks,
    }


def _evaluation_view(
    *,
    model: Mapping[str, Any],
    candidate_funnel: Mapping[str, Any],
    bfs: Mapping[str, Any],
    matched_compute: Mapping[str, Any],
    generation_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if set(model.get("overall", {})) != {str(value) for value in SAMPLE_PREFIXES}:
        raise ValueError("Evaluation is missing solve@1/8/64")
    if set(model.get("by_certified_distance_at_64", {})) != {
        str(value) for value in DISTANCES
    }:
        raise ValueError("Evaluation is missing a distance-1..8 @64 row")
    if set(candidate_funnel.get("by_k", {})) != {
        str(value) for value in SAMPLE_PREFIXES
    }:
        raise ValueError("Evaluation is missing candidate funnels at 1/8/64")
    return {
        "solve_at_k": canonical_copy(model["overall"]),
        "by_certified_distance_at_64": canonical_copy(
            model["by_certified_distance_at_64"]
        ),
        "distance_5_8_at_64": _distance_aggregate(model, TARGET_DISTANCES),
        "distance_7_8_at_64": _distance_aggregate(model, (7, 8)),
        "candidate_funnel": canonical_copy(candidate_funnel),
        "bfs": canonical_copy(bfs),
        "matched_compute": canonical_copy(matched_compute),
        "generation_runtime": canonical_copy(generation_runtime),
    }


def _tokenizer_report(
    manifest: Mapping[str, Any], *, tokenizer_dir: Path
) -> dict[str, Any]:
    tokenizer = manifest.get("tokenizer", {})
    selection = manifest.get("selection", {})
    statistics = (
        selection.get("statistics", {}) if isinstance(selection, Mapping) else {}
    )
    retained = statistics.get("retained", {}) if isinstance(statistics, Mapping) else {}
    by_distance = (
        retained.get("by_certified_distance", {})
        if isinstance(retained, Mapping)
        else {}
    )
    excluded = selection.get("excluded") if isinstance(selection, Mapping) else None
    if excluded is None and isinstance(selection, Mapping):
        detail_identity = selection.get("detail_file")
        if not isinstance(detail_identity, Mapping):
            raise ValueError("Tokenizer artifact has no detached selection detail")
        detail_path = tokenizer_dir.resolve() / DETAIL_FILENAME
        if (
            detail_identity.get("path") != DETAIL_FILENAME
            or not detail_path.is_file()
            or detail_path.stat().st_size != detail_identity.get("bytes")
            or sha256_file(detail_path) != detail_identity.get("sha256")
        ):
            raise ValueError("Tokenizer selection detail changed after it was sealed")
        detail = _read_json(detail_path, label="tokenizer selection detail")
        detail_selection = detail.get("selection")
        if not isinstance(detail_selection, Mapping):
            raise ValueError("Tokenizer selection detail is incomplete")
        excluded = detail_selection.get("excluded")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("max_length") != 2048
        or set(by_distance) != {str(value) for value in DISTANCES}
        or not isinstance(excluded, list)
        or selection.get("excluded_records") != len(excluded)
    ):
        raise ValueError("Tokenizer artifact lacks complete length/exclusion reporting")
    return {
        "artifact_digest": manifest["artifact_digest"],
        "total_vocab_size": tokenizer.get("total_vocab_size"),
        "base_vocab_size": tokenizer.get("base_vocab_size"),
        "max_length": tokenizer.get("max_length"),
        "selected_records": selection.get("selected_records"),
        "retained_records": selection.get("retained_records"),
        "excluded_records": selection.get("excluded_records"),
        "retained_token_statistics_by_distance": canonical_copy(by_distance),
        "excluded_rows": canonical_copy(excluded),
        "token_id_digests": canonical_copy(selection.get("token_id_digests")),
    }


def _decision_rules(
    validation: Mapping[str, Mapping[str, Any]],
    test: Mapping[str, Mapping[str, Any]],
    *,
    active_arm_ids: Sequence[str] = ARM_IDS,
    deferred_arm_ids: Sequence[str] = (),
) -> dict[str, Any]:
    active = tuple(active_arm_ids)
    deferred = tuple(deferred_arm_ids)
    validation_target = {
        arm_id: validation[arm_id]["distance_5_8_at_64"] for arm_id in active
    }
    test_target = {arm_id: test[arm_id]["distance_5_8_at_64"] for arm_id in active}
    larger_helps: dict[str, bool] = {}
    for smaller, larger in pairwise(ARM_IDS):
        if smaller in validation_target and larger in validation_target:
            larger_helps[f"{larger}_vs_{smaller}"] = (
                validation_target[larger]["solve_rate"]
                > validation_target[smaller]["solve_rate"]
            )
    all_zero = all(
        int(test[arm_id]["by_certified_distance_at_64"][str(distance)]["solved"]) == 0
        for arm_id in active
        for distance in TARGET_DISTANCES
    )
    kill = {}
    for arm_id in active:
        try:
            matched = test[arm_id]["matched_compute"]["7-8"]["matched"]["64"]
            cheaper = matched["model_cheaper_than_bfs_at_equal_or_higher_solve_count"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Evaluation lacks the frozen distance-7/8 kill metric"
            ) from error
        if type(cheaper) is not bool:
            raise ValueError("Distance-7/8 cheaper-than-BFS flag is invalid")
        kill[arm_id] = {
            "triggered": not cheaper,
            "model_cheaper_than_bfs_at_equal_or_higher_solve_count": cheaper,
            "rule": (
                "trigger when the model is not cheaper than exhaustive search at "
                "equal or higher solve count on distance 7-8"
            ),
        }
    capacity_arm = "smollm2-360m-scratch"
    active_smaller_arms = tuple(
        arm_id for arm_id in ARM_IDS[:-1] if arm_id in test_target
    )
    capacity_mismatch = None
    if capacity_arm in test_target and active_smaller_arms:
        capacity_mismatch = test_target[capacity_arm]["solve_rate"] < max(
            test_target[arm_id]["solve_rate"] for arm_id in active_smaller_arms
        )
    if all_zero:
        if deferred:
            active_labels = ", ".join(ARM_LABELS[arm_id] for arm_id in active)
            deferred_labels = ", ".join(ARM_LABELS[arm_id] for arm_id in deferred)
            model_noun = "model" if len(active) == 1 else "models"
            arm_noun, verb = (
                ("arm", "provides") if len(deferred) == 1 else ("arms", "provide")
            )
            long_distance_statement = (
                f"The active {model_noun} ({active_labels}) remained 0/12 at every "
                f"distance from 5 through 8; the deferred {arm_noun} "
                f"({deferred_labels}) {verb} no evidence for or against the "
                "long-distance regime."
            )
        else:
            long_distance_statement = (
                "All three models remained 0/12 at every distance from 5 through 8; "
                "10x data plus scratch scaling did not solve the long-distance regime."
            )
    else:
        long_distance_statement = (
            "At least one predeclared model solved a held-out distance-5..8 task; "
            "see the per-distance table without tuning on test."
        )
    return {
        "active_arm_ids": list(active),
        "deferred_arm_ids": list(deferred),
        "validation_distance_5_8": validation_target,
        "larger_model_helps_on_validation_distance_5_8": larger_helps,
        "all_models_zero_at_each_test_distance_5_8": all_zero,
        "long_distance_statement": long_distance_statement,
        "distance_7_8_kill_criterion": kill,
        "capacity_data_mismatch_360m_loses_on_test_distance_5_8": capacity_mismatch,
        "capacity_caveat": (
            "The arms also differ in transformer family; this bounded comparison "
            "cannot isolate parameter count as a pure causal scaling variable."
        ),
        "seed_caveat": (
            "This is a one-seed bounded feasibility comparison, not a scaling law; "
            "validation-positive models require replication under additional seeds."
        ),
    }


def _pct(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def build_markdown(report: Mapping[str, Any]) -> str:
    validation = report["evaluations"]["validation"]
    test = report["evaluations"]["test"]
    training = report["training"]
    tokenizer = report["tokenizer"]
    decisions = report["decision_rules"]
    scope = report.get("study_scope", {})
    active_arm_ids = tuple(scope.get("active_arm_ids", ARM_IDS))
    deferred_arm_ids = tuple(scope.get("deferred_arm_ids", ()))

    provenance_rows = []
    for arm_id in active_arm_ids:
        identity = training[arm_id]["identity"]
        final = training[arm_id]["final"]
        provenance_rows.append(
            [
                arm_id,
                identity["exact_trainable_parameters"],
                identity["initialization_seed"],
                identity["data_order_seed"],
                identity["initial_state_digest"],
                identity["effective_config_digest"],
                final["wandb_url"],
            ]
        )
    distance_rows = []
    for distance in DISTANCES:
        model_rows = [
            test[arm_id]["by_certified_distance_at_64"][str(distance)]
            for arm_id in active_arm_ids
        ]
        bfs = test[active_arm_ids[0]]["bfs"]["by_certified_distance"][str(distance)]
        distance_rows.append(
            [
                distance,
                *(f"{row['solved']}/{row['tasks']}" for row in model_rows),
                f"{bfs['solved']}/{bfs['tasks']}",
            ]
        )
    target_distance_rows = [
        row for row in distance_rows if int(row[0]) in TARGET_DISTANCES
    ]
    headline_rows = []
    for label, distances, field in (
        ("distance 5-8", TARGET_DISTANCES, "distance_5_8_at_64"),
        ("distance 7-8 (headline)", (7, 8), "distance_7_8_at_64"),
    ):
        aggregates = [test[arm_id][field] for arm_id in active_arm_ids]
        bfs_rows_for_group = [
            test[active_arm_ids[0]]["bfs"]["by_certified_distance"][str(distance)]
            for distance in distances
        ]
        bfs_solved = sum(int(row["solved"]) for row in bfs_rows_for_group)
        bfs_tasks = sum(int(row["tasks"]) for row in bfs_rows_for_group)
        headline_rows.append(
            [
                label,
                *(
                    f"{row['solved']}/{row['tasks']} ({_pct(row['solve_rate'])})"
                    for row in aggregates
                ),
                f"{bfs_solved}/{bfs_tasks} ({_pct(bfs_solved / bfs_tasks)})",
            ]
        )
    token_rows = []
    for distance in DISTANCES:
        row = tokenizer["retained_token_statistics_by_distance"][str(distance)]
        token_rows.append(
            [
                distance,
                row["prompt"]["mean"],
                row["prompt"]["max"],
                row["completion_with_eos"]["mean"],
                row["completion_with_eos"]["max"],
                row["combined"]["max"],
            ]
        )
    training_rows = []
    loss_rows = []
    for arm_id in active_arm_ids:
        for run in [*training[arm_id]["pilots"], training[arm_id]["final"]]:
            label = (
                f"pilot lr={run['learning_rate']:g}"
                if run["phase"] == "pilot"
                else "final"
            )
            training_rows.append(
                [
                    arm_id,
                    label,
                    run["global_steps"],
                    run["num_input_tokens_seen"],
                    f"{run['runtime_seconds']:.1f}",
                    f"{run['trainer_train_runtime_seconds']:.1f}",
                    run["peak_torch_allocated_bytes"],
                    run["peak_torch_reserved_bytes"],
                    run["peak_observed_gpu_memory_bytes"],
                    run["physical_vram_bytes"],
                    run["best_validation_loss"],
                ]
            )
            for curve in run["loss_curve"]:
                loss_rows.append(
                    [
                        arm_id,
                        label,
                        curve.get("step"),
                        curve.get("preceding_train_window_loss"),
                        curve.get("validation_loss"),
                        f"{curve['train_validation_gap']:.6f}",
                    ]
                )
    solve_rows = []
    for split, evaluations in (("validation", validation), ("test", test)):
        for arm_id in active_arm_ids:
            for sample_count in SAMPLE_PREFIXES:
                row = evaluations[arm_id]["solve_at_k"][str(sample_count)]
                solve_rows.append(
                    [
                        split,
                        arm_id,
                        sample_count,
                        f"{row['solved']}/{row['tasks']}",
                        _pct(row["solve_rate"]),
                        f"{row['exact_matches']}/{row['tasks']}",
                        _pct(row["exact_match_rate"]),
                        f"{row['mean_basis_executions']:.1f}",
                    ]
                )
    bfs_rows = []
    matched_rows = []
    generation_rows = []
    for split, evaluations in (("validation", validation), ("test", test)):
        shared_bfs = evaluations[active_arm_ids[0]]["bfs"]
        overall_bfs = shared_bfs["overall"]
        bfs_rows.append(
            [
                split,
                "all",
                overall_bfs["solved"],
                overall_bfs["tasks"],
                overall_bfs["total_basis_executions"],
                overall_bfs["mean_basis_executions"],
            ]
        )
        for distance in DISTANCES:
            row = shared_bfs["by_certified_distance"][str(distance)]
            bfs_rows.append(
                [
                    split,
                    distance,
                    row["solved"],
                    row["tasks"],
                    row["total_executions"],
                    row["mean_executions"],
                ]
            )
        for arm_id in active_arm_ids:
            runtime = evaluations[arm_id]["generation_runtime"]
            generation_rows.append(
                [
                    split,
                    arm_id,
                    runtime.get("seconds"),
                    runtime.get("samples"),
                    runtime.get("samples_per_second"),
                    runtime.get("peak_torch_allocated_bytes"),
                    runtime.get("peak_torch_reserved_bytes"),
                    runtime.get("peak_observed_gpu_memory_bytes", "n/a"),
                ]
            )
            groups = evaluations[arm_id]["matched_compute"]
            for group_name in sorted(groups):
                matched_by_k = groups[group_name].get("matched", {})
                for sample_count in SAMPLE_PREFIXES:
                    row = matched_by_k.get(str(sample_count))
                    if row is None:
                        continue
                    ratio = row.get("bfs_to_model_total_execution_ratio")
                    matched_rows.append(
                        [
                            split,
                            arm_id,
                            group_name,
                            sample_count,
                            f"{row['solved']}/{row['tasks']}",
                            row["mean_executions_spent"],
                            row[
                                "bfs_uniform_budget_at_or_below_model_mean_execution_spend"
                            ],
                            row["bfs_solved_at_model_mean_execution_spend"],
                            row["bfs_uniform_budget_for_at_least_model_solve_count"],
                            "n/a" if ratio is None else f"{float(ratio):.3f}x",
                            row[
                                "model_cheaper_than_bfs_at_equal_or_higher_solve_count"
                            ],
                        ]
                    )
    funnel_rows = []
    for split, evaluations in (("validation", validation), ("test", test)):
        for arm_id in active_arm_ids:
            for sample_count in SAMPLE_PREFIXES:
                row = evaluations[arm_id]["candidate_funnel"]["by_k"][str(sample_count)]
                funnel_rows.append(
                    [
                        split,
                        arm_id,
                        sample_count,
                        _pct(row["unparseable_rate"]),
                        _pct(row["executable_given_parseable_rate"]),
                        _pct(row["type_mismatch_rate"]),
                        _pct(row["correct_given_parseable_rate"]),
                        ", ".join(
                            (
                                f"{item['category']}"
                                + (
                                    ":" + ",".join(item["unknown_function_names"])
                                    if item["unknown_function_names"]
                                    else ""
                                )
                                + f" ({item['count']})"
                            )
                            for item in row.get("top_parse_failure_reasons", [])
                        )
                        or "none",
                    ]
                )
    feasibility_rows = []
    for arm_id in ARM_IDS:
        gate = report["feasibility"]["launch_gate"]["per_arm"][arm_id]
        projection = gate["baseline"]
        feasibility_rows.append(
            [
                arm_id,
                "deferred" if arm_id in deferred_arm_ids else gate["route"],
                projection["projected_pilot_hours"],
                projection["projected_final_hours"],
                projection["projected_pilot_plus_final_hours"],
                gate["runtime_hard_stop_required"],
            ]
        )
    excluded_rows = [
        [
            row.get("task_id"),
            row.get("certified_distance"),
            row.get("prompt_tokens"),
            row.get("completion_tokens_with_eos"),
            row.get("combined_tokens"),
            row.get("reason"),
        ]
        for row in tokenizer["excluded_rows"]
    ]
    kill_lines = "\n".join(
        f"- `{arm_id}`: **{'triggered' if value['triggered'] else 'not triggered'}**"
        for arm_id, value in decisions["distance_7_8_kill_criterion"].items()
    )
    primary_headers = [
        "distance",
        *(f"{ARM_LABELS[arm_id]} @64" for arm_id in active_arm_ids),
        "BFS",
    ]
    deferred_sections = []
    for arm_id in deferred_arm_ids:
        deferred = report["deferred_arms"][arm_id]
        projection = deferred["projection"]
        deferred_sections.append(
            "### "
            + ARM_LABELS[arm_id]
            + " — deferred\n\n"
            + f"Projected total: **{projection['projected_pilot_plus_final_hours']:.2f} "
            "hours**, including three learning-rate pilots, the full three-epoch "
            "run, and the fixed 15% margin. It can fit by using fewer learning-rate "
            "pilots under a separately frozen LR-selection protocol, or by assigning "
            "it its own night. Do not shrink training data, epochs, context length, "
            "or parameter count to force it into the current night."
        )
    deferred_markdown = (
        "## Deferred arms\n\n" + "\n\n".join(deferred_sections)
        if deferred_sections
        else ""
    )
    capacity_mismatch = decisions[
        "capacity_data_mismatch_360m_loses_on_test_distance_5_8"
    ]
    if capacity_mismatch is None and "smollm2-360m-scratch" in deferred_arm_ids:
        capacity_statement = (
            "The 360M arm was deferred, so this report makes no held-out 360M "
            "capacity/data comparison."
        )
    elif capacity_mismatch is None:
        capacity_statement = (
            "No smaller active arm was available for a held-out 360M "
            "capacity/data comparison."
        )
    elif capacity_mismatch:
        capacity_statement = (
            "The 360M arm lost to a smaller arm on held-out distance 5-8: report "
            "this as a capacity/data mismatch and do not tune on test."
        )
    else:
        capacity_statement = "No predeclared 360M capacity/data mismatch was triggered."
    return (
        "\n\n".join(
            [
                "# Deep corpus v2 scratch scaling report (issue #54)",
                decisions["long_distance_statement"],
                "## Primary held-out comparison: certified distance 5-8\n\n"
                + "Distance 7-8 is the headline slice.\n\n"
                + _table(primary_headers, target_distance_rows)
                + "\n\n"
                + _table(primary_headers, headline_rows),
                "## Complete solve@64 by certified distance 1-8\n\n"
                + _table(primary_headers, distance_rows),
                "## Initialization and final checkpoint provenance\n\n"
                + _table(
                    [
                        "arm",
                        "exact params",
                        "init seed",
                        "order seed",
                        "initial state",
                        "config digest",
                        "training run",
                    ],
                    provenance_rows,
                ),
                "All embeddings and model weights started randomly via `from_config`; no "
                "pretrained weights or previous task checkpoint were loaded.",
                "## Corpus and tokenizer\n\n"
                f"Corpus manifest: `{report['corpus']['manifest_digest']}`. Tokenizer: "
                f"`{tokenizer['artifact_digest']}`, vocabulary {tokenizer['total_vocab_size']}, "
                f"maximum length {tokenizer['max_length']}. Retained "
                f"{tokenizer['retained_records']}/{tokenizer['selected_records']} discovery rows.",
                _table(
                    [
                        "distance",
                        "prompt mean",
                        "prompt max",
                        "completion+EOS mean",
                        "completion+EOS max",
                        "combined max",
                    ],
                    token_rows,
                ),
                "### Every excluded discovery row\n\n"
                + (
                    _table(
                        [
                            "task",
                            "distance",
                            "prompt",
                            "completion+EOS",
                            "combined",
                            "reason",
                        ],
                        excluded_rows,
                    )
                    if excluded_rows
                    else "None."
                ),
                "## Feasibility projections\n\n"
                + _table(
                    ["arm", "route", "pilot h", "final h", "total h", "hard stop"],
                    feasibility_rows,
                ),
                deferred_markdown,
                "## Observed pilot and final training\n\n"
                + _table(
                    [
                        "arm",
                        "run",
                        "steps",
                        "tokens",
                        "runtime s",
                        "trainer runtime s",
                        "torch allocated",
                        "torch reserved",
                        "observed peak",
                        "physical capacity",
                        "best val loss",
                    ],
                    training_rows,
                ),
                "Allocated/reserved memory is PyTorch allocator telemetry; observed "
                "peak carries an explicit process or physical-GPU-total scope; "
                "physical capacity is reported separately. Runtime covers each invocation through its "
                "model/tokenizer/config/trainer artifact writes; trainer runtime is "
                "shown separately.",
                "### Train/validation loss and gap at every evaluation\n\n"
                + _table(
                    ["arm", "run", "step", "train loss", "validation loss", "gap"],
                    loss_rows,
                ),
                "## Validation and once-opened test solve@k\n\n"
                + _table(
                    [
                        "split",
                        "arm",
                        "k",
                        "execution correct",
                        "solve rate",
                        "exact witness",
                        "exact rate",
                        "mean basis executions",
                    ],
                    solve_rows,
                ),
                "## Exhaustive BFS measured cost\n\n"
                + _table(
                    [
                        "split",
                        "distance",
                        "solved",
                        "tasks",
                        "total executions",
                        "mean executions",
                    ],
                    bfs_rows,
                ),
                "## Matched basis-execution comparison\n\n"
                + _table(
                    [
                        "split",
                        "arm",
                        "distance group",
                        "k",
                        "model solved",
                        "model mean spend",
                        "BFS budget <= model mean",
                        "BFS solved at model mean",
                        "BFS budget for >= model solves",
                        "BFS/model execution ratio",
                        "model cheaper",
                    ],
                    matched_rows,
                ),
                "## Generation runtime and memory diagnostics\n\n"
                + _table(
                    [
                        "split",
                        "arm",
                        "seconds",
                        "samples",
                        "samples/s",
                        "torch allocated",
                        "torch reserved",
                        "observed peak",
                    ],
                    generation_rows,
                ),
                "## Candidate parse/execution funnels\n\n"
                + _table(
                    [
                        "split",
                        "arm",
                        "k",
                        "unparseable",
                        "executable | parse",
                        "type mismatch",
                        "execution-correct | parseable",
                        "top reasons",
                    ],
                    funnel_rows,
                ),
                "Malformed candidates consume zero basis executions; neural generation cost "
                "is not included in the matched BFS basis-execution unit. Because these "
                "random-init arms use basis names as atomic tokens, a non-negligible "
                "unparseable rate is treated as a tokenizer or prompt-format bug before "
                "drawing a capacity or data-scale conclusion.",
                "## Predeclared decisions\n\n"
                + decisions["long_distance_statement"]
                + "\n\n"
                + kill_lines
                + "\n\n"
                + capacity_statement,
                decisions["capacity_caveat"],
                decisions["seed_caveat"],
                "The 20M BFS budget is a per-task safety ceiling, not measured cost; all "
                "cost claims above use observed row executions and the frozen matched curve.",
            ]
        )
        + "\n"
    )


def _seal_report(body: Mapping[str, Any]) -> dict[str, Any]:
    value = canonical_copy(dict(body))
    value[REPORT_DIGEST_FIELD] = sha256_json(value)
    return value


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    value = canonical_copy(dict(report))
    digest = value.pop(REPORT_DIGEST_FIELD, None)
    if digest != sha256_json(value):
        raise ValueError("Report changed after it was sealed")
    value[REPORT_DIGEST_FIELD] = digest
    if (value.get("schema_version"), value.get("status"), value.get("study_id")) != (
        SCHEMA_VERSION,
        "complete",
        "deep-corpus-v2-scratch-issue-54",
    ):
        raise ValueError("Report has the wrong schema or study")
    return value


def build_report(inputs: ReportInputs) -> tuple[dict[str, Any], str]:
    protocol = load_protocol_lock(
        inputs.protocol_lock,
        tokenizer_dir=inputs.tokenizer_dir,
        repo_root=inputs.repo_root,
    )
    _validate_issue_54_protocol(protocol)
    corpus = _read_json(inputs.corpus_manifest, label="corpus manifest")
    if canonical_copy(corpus) != canonical_copy(protocol["corpus"]["manifest"]):
        raise ValueError("Explicit corpus manifest differs from the frozen protocol")
    tokenizer_manifest = validate_frozen_tokenizer(
        inputs.tokenizer_dir,
        expected_source_manifest_digest=corpus["manifest_digest"],
    )
    if (
        tokenizer_manifest["artifact_digest"]
        != protocol["tokenizer"]["metadata"]["artifact_digest"]
    ):
        raise ValueError("Tokenizer artifact differs from the frozen protocol")
    feasibility = load_probe_manifest(inputs.feasibility_manifest)
    _validate_feasibility_producer(feasibility, protocol=protocol)
    feasibility_results = _study_result_by_arm(feasibility)
    active_arm_ids, deferred_arm_ids = _study_arm_scope(feasibility)
    if feasibility.get("source_data", {}).get("corpus_manifest_digest") != corpus.get(
        "manifest_digest"
    ) or feasibility.get("source_data", {}).get(
        "tokenizer_artifact_digest"
    ) != tokenizer_manifest.get("artifact_digest"):
        raise ValueError("Feasibility study used another corpus/tokenizer")

    validation, selection_locks = load_validation_results(
        inputs.validation_results, protocol_lock=protocol
    )
    if tuple(validation) != active_arm_ids or len(selection_locks) != len(
        active_arm_ids
    ):
        raise ValueError(
            "Validation results do not match the feasibility-authorized arms"
        )
    validation_bfs = canonical_copy(validation[active_arm_ids[0]]["summary"]["bfs"])
    if any(
        canonical_copy(validation[arm_id]["summary"]["bfs"]) != validation_bfs
        for arm_id in active_arm_ids[1:]
    ):
        raise ValueError("Validation arms do not share one deterministic BFS result")
    final_outcomes: dict[str, dict[str, Any]] = {}
    for arm_id in active_arm_ids:
        manifest = validation[arm_id]["training_manifest"]
        _validate_feasibility_link(
            manifest,
            arm_id=arm_id,
            feasibility=feasibility,
            study_result=feasibility_results[arm_id],
        )
        final_outcomes[arm_id] = _training_run_outcome(
            manifest, arm_id=arm_id, phase="final"
        )
        if _training_outcomes(manifest) != validation[arm_id]["summary"]["training"]:
            raise ValueError("Validation summary changed its final training outcomes")
    pilots = _load_pilot_outcomes(
        inputs.pilot_manifests,
        validation=validation,
        feasibility=feasibility,
        feasibility_results=feasibility_results,
        active_arm_ids=active_arm_ids,
    )

    test_dir = inputs.test_dir.resolve()
    combined = _validate_combined_test_summary(
        _read_json(test_dir / TEST_SUMMARY_FILENAME, label="combined test summary"),
        protocol=protocol,
        active_arm_ids=active_arm_ids,
        deferred_arm_ids=deferred_arm_ids,
        feasibility_artifact_digest=feasibility["artifact_digest"],
    )
    ledger_scope: dict[str, Any] = {}
    if deferred_arm_ids or combined.get("feasibility_artifact_digest") is not None:
        ledger_scope = {
            "active_arm_ids": active_arm_ids,
            "deferred_arm_ids": deferred_arm_ids,
            "feasibility_artifact_digest": feasibility["artifact_digest"],
        }
    ledger = validate_test_opening_ledger(
        _read_json(test_dir / TEST_LEDGER_FILENAME, label="test ledger"),
        protocol_lock_sha256=protocol["protocol_lock_sha256"],
        selection_locks=selection_locks,
        **ledger_scope,
    )
    if ledger["status"] != "complete" or ledger["test_access_count"] != 1:
        raise ValueError("Held-out test ledger is not complete and once-opened")
    if combined["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Combined test summary belongs to another ledger revision")
    if combined["shared_bfs"] != ledger["shared_bfs_identity"]:
        raise ValueError("Combined test summary changed the shared BFS identity")
    bfs_path = test_dir / BFS_FILENAME
    if (
        not bfs_path.is_file()
        or bfs_path.stat().st_size != combined["shared_bfs"].get("bytes")
        or sha256_file(bfs_path) != combined["shared_bfs"].get("sha256")
    ):
        raise ValueError("Shared held-out BFS artifact changed")
    test_arms = _load_test_arms(
        test_dir,
        protocol=protocol,
        combined=combined,
        validation=validation,
        ledger=ledger,
        active_arm_ids=active_arm_ids,
    )

    validation_views: dict[str, dict[str, Any]] = {}
    test_views: dict[str, dict[str, Any]] = {}
    for arm_id in active_arm_ids:
        validation_summary = validation[arm_id]["summary"]
        validation_views[arm_id] = {
            **_evaluation_view(
                model=validation_summary["model"],
                candidate_funnel=validation_summary["candidate_funnel"],
                bfs=validation_summary["bfs"],
                matched_compute=validation_summary["matched_compute"],
                generation_runtime=validation_summary["generation"]["runtime"],
            ),
            "evaluation_summary_sha256": validation_summary[SUMMARY_DIGEST_FIELD],
            "wandb_receipt": canonical_copy(validation[arm_id]["wandb_receipt"]),
        }
        test_summary = test_arms[arm_id]["summary"]
        test_views[arm_id] = {
            **_evaluation_view(
                model=test_summary["model"],
                candidate_funnel=combined["candidate_funnels"]["arms"][arm_id],
                bfs=test_summary["bfs"],
                matched_compute=test_summary["matched_compute"],
                generation_runtime=test_summary["generation"]["runtime"],
            ),
            "evaluation_summary_sha256": test_summary[SUMMARY_DIGEST_FIELD],
            "wandb_receipt": canonical_copy(test_arms[arm_id]["wandb_receipt"]),
        }

    training = {
        arm_id: {
            "identity": _shared_training_identity(
                validation[arm_id]["training_manifest"]
            ),
            "pilot_selection_lock": canonical_copy(
                validation[arm_id]["training_manifest"]["selection_lock"]
            ),
            "pilots": pilots[arm_id],
            "final": final_outcomes[arm_id],
        }
        for arm_id in active_arm_ids
    }
    deferred_arms = {
        arm_id: _deferred_arm_report(
            arm_id,
            arm_gate=feasibility["launch_gate"]["per_arm"][arm_id],
        )
        for arm_id in deferred_arm_ids
    }
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "study_id": "deep-corpus-v2-scratch-issue-54",
        "study_scope": {
            "active_arm_ids": list(active_arm_ids),
            "deferred_arm_ids": list(deferred_arm_ids),
            "status": "partial-with-deferred-arms"
            if deferred_arm_ids
            else "all-arms-complete",
            "feasibility_artifact_digest": feasibility["artifact_digest"],
        },
        "protocol": {
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "validation_cohort_sha256": protocol["cohorts"]["validation"][
                "cohort_sha256"
            ],
            "test_cohort_sha256": protocol["cohorts"]["test"]["cohort_sha256"],
            "generation": canonical_copy(protocol["generation"]),
            "bfs": canonical_copy(protocol["bfs"]),
            "evaluator_source_sha256": protocol["evaluator_sources"]["source_sha256"],
        },
        "corpus": {
            "manifest_digest": corpus["manifest_digest"],
            "basis_set_id": corpus["basis_set_id"],
            "basis_set_digest": corpus["basis_set_digest"],
            "hub": canonical_copy(corpus.get("hub")),
            "statistics": canonical_copy(corpus.get("statistics")),
            "target_contract": canonical_copy(corpus.get("target_contract")),
        },
        "tokenizer": _tokenizer_report(
            tokenizer_manifest, tokenizer_dir=inputs.tokenizer_dir
        ),
        "feasibility": {
            "artifact_digest": feasibility["artifact_digest"],
            "launch_gate": canonical_copy(feasibility["launch_gate"]),
            "arm_results": canonical_copy(feasibility["arm_results"]),
        },
        "training": training,
        "deferred_arms": deferred_arms,
        "test_opening": {
            "ledger_sha256": ledger["ledger_sha256"],
            "opening_id": ledger["opening_id"],
            "test_access_count": ledger["test_access_count"],
            "shared_bfs": canonical_copy(ledger["shared_bfs_identity"]),
            "combined_summary_sha256": combined[SUMMARY_DIGEST_FIELD],
            "active_arm_ids": list(active_arm_ids),
            "deferred_arm_ids": list(deferred_arm_ids),
            "feasibility_artifact_digest": feasibility["artifact_digest"],
        },
        "evaluations": {
            "validation": validation_views,
            "test": test_views,
        },
        "decision_rules": _decision_rules(
            validation_views,
            test_views,
            active_arm_ids=active_arm_ids,
            deferred_arm_ids=deferred_arm_ids,
        ),
        "report_builder_sha256": sha256_file(Path(__file__)),
    }
    preliminary = _seal_report(body)
    markdown = build_markdown(preliminary)
    body["report_markdown_sha256"] = (
        "sha256:" + hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    )
    report = validate_report(_seal_report(body))
    return report, markdown


def _write_file(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_report(
    output_dir: str | Path, report: Mapping[str, Any], markdown: str
) -> Path:
    normalized = validate_report(report)
    expected_markdown = normalized.get("report_markdown_sha256")
    actual_markdown = "sha256:" + hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    if expected_markdown != actual_markdown:
        raise ValueError("Markdown differs from the sealed report")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Report output must be a fresh directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-", dir=destination.parent
        )
    )
    try:
        _write_file(
            temporary / REPORT_JSON_FILENAME,
            json.dumps(normalized, allow_nan=False, indent=2, sort_keys=True) + "\n",
        )
        _write_file(temporary / REPORT_MARKDOWN_FILENAME, markdown)
        loaded = _read_json(temporary / REPORT_JSON_FILENAME, label="written report")
        if (
            loaded != normalized
            or (temporary / REPORT_MARKDOWN_FILENAME).read_text(encoding="utf-8")
            != markdown
        ):
            raise RuntimeError("Report artifacts changed during serialization")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def create_report(inputs: ReportInputs) -> dict[str, Any]:
    report, markdown = build_report(inputs)
    write_report(inputs.output_dir, report, markdown)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> ReportInputs:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Run before relocating checkpoints or evaluation directories; all sealed "
            "absolute paths and file digests are revalidated live."
        ),
    )
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--feasibility-manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-result", type=Path, action="append", required=True
    )
    parser.add_argument("--pilot-manifest", type=Path, action="append", required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    return ReportInputs(
        protocol_lock=args.protocol_lock,
        corpus_manifest=args.corpus_manifest,
        tokenizer_dir=args.tokenizer_dir,
        feasibility_manifest=args.feasibility_manifest,
        validation_results=tuple(args.validation_result),
        pilot_manifests=tuple(args.pilot_manifest),
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    inputs = _parse_args(argv)
    report = create_report(inputs)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": report[REPORT_DIGEST_FIELD],
                "output_dir": str(inputs.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
