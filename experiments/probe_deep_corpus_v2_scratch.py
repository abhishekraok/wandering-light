"""Prove issue #54's scratch study fits one GPU and its runtime envelope.

The default invocation is a provenance-verifying dry run.  ``--execute`` is an
explicit, expensive operation: it proves the eight longest retained examples
fit as one bf16 forward/backward batch for every arm, then measures exactly 500
optimizer steps with the frozen trainer protocol.  No model checkpoint is kept.

The resulting content-addressed manifest is the launch gate consumed by the
full training runner.  Each arm is decided independently: a direct launch
requires a projection of at most 14 hours.  A slower arm may proceed only when
its baseline remains within 16 hours and a separately content-addressed,
protocol-compatible optimization plan has been validated at at most 14 hours
with a 16-hour hard-stop monitor.  The sum across arms is diagnostic only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.training.deep_corpus_scratch import (
    ARM_SPECS,
    DATA_ORDER_SEED,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    FEASIBILITY_HARD_LIMIT_HOURS,
    FEASIBILITY_SOFT_LIMIT_HOURS,
    FEASIBILITY_STEPS,
    INITIALIZATION_SEED,
    LR_GRID,
    MAX_FINAL_EPOCHS,
    PILOT_EPOCHS,
    TOKENIZER_MAX_LENGTH,
    assert_config_created_model,
    build_scratch_study_identity,
    configure_deterministic_cuda_environment,
    digest_json,
    initialize_model_from_config,
)
from wandering_light.training.deep_corpus_tokenizer import (
    DEFAULT_REFERENCE_CORPUS_DIR,
    DETERMINISTIC_PYTHON_HASH_SEED,
    encode_training_row,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "wandering-light-deep-corpus-v2-scratch-feasibility"
OPTIMIZATION_ARTIFACT_TYPE = (
    "wandering-light-deep-corpus-v2-scratch-feasibility-optimization"
)
MANIFEST_FILENAME = "feasibility_manifest.json"
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v2")
DEFAULT_TOKENIZER_DIR = Path("wandering_light/training/tokenizers/deep_corpus_v2")
PROBE_BATCH_SIZE = 8
PROBE_GRADIENT_ACCUMULATION_STEPS = 1
PROBE_LEARNING_RATE = LR_GRID[1]
PROBE_PRECISION = "bf16"
EXPECTED_ARM_IDS = tuple(ARM_SPECS)
PROJECTION_SERVICE_OVERHEAD_FACTOR = 1.15
PILOT_VALIDATIONS_PER_RUN = 2
PILOT_CHECKPOINT_WRITES_PER_RUN = 2
FINAL_VALIDATIONS = math.ceil(MAX_FINAL_EPOCHS) + 1
FINAL_CHECKPOINT_WRITES = math.ceil(MAX_FINAL_EPOCHS) + 1
PROCESS_MEMORY_POLL_INTERVAL_SECONDS = 0.1
NVIDIA_SMI_COMMAND_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class FeasibilityProbeConfig:
    """Inputs for one all-arm feasibility artifact."""

    output_dir: Path
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR
    reference_corpus_dirs: tuple[Path, ...] = (DEFAULT_REFERENCE_CORPUS_DIR,)
    execute: bool = False
    optimization_plan: Path | None = None


@dataclass(frozen=True)
class LongestRetainedBatch:
    """The stable top-eight retained rows by complete token length."""

    rows: tuple[Mapping[str, Any], ...]
    task_ids: tuple[str, ...]
    combined_token_lengths: tuple[int, ...]
    task_ids_digest: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "selection": "largest combined_tokens; stable source order breaks ties",
            "batch_size": len(self.rows),
            "task_ids": list(self.task_ids),
            "task_ids_digest": self.task_ids_digest,
            "combined_token_lengths": list(self.combined_token_lengths),
            "padded_sequence_length": max(self.combined_token_lengths),
        }


def parse_nvidia_smi_process_memory(
    output: str,
    *,
    pid: int,
    gpu_uuid: str,
) -> int | None:
    """Return this process's memory on one physical GPU from CSV output."""

    if type(pid) is not int or pid <= 0:
        raise ValueError("GPU-memory sampling requires a positive process ID")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("GPU-memory sampling requires a physical GPU UUID")
    matches: list[int] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 3:
            raise ValueError("nvidia-smi compute-apps output is malformed")
        raw_pid, raw_uuid, raw_mebibytes = columns
        try:
            row_pid = int(raw_pid)
        except ValueError as error:
            raise ValueError("nvidia-smi returned a non-integer process ID") from error
        if row_pid != pid or not _gpu_uuids_match(raw_uuid, gpu_uuid):
            continue
        try:
            mebibytes = int(raw_mebibytes)
        except ValueError as error:
            raise ValueError(
                "nvidia-smi did not report numeric process GPU memory"
            ) from error
        if mebibytes < 0:
            raise ValueError("nvidia-smi reported negative process GPU memory")
        matches.append(mebibytes * 1024 * 1024)
    if len(matches) > 1:
        raise ValueError("nvidia-smi returned duplicate rows for this process/GPU")
    return matches[0] if matches else None


def _canonical_gpu_uuid(value: str) -> str:
    """Normalize the CUDA/nvidia-smi spelling of one physical GPU UUID."""

    normalized = value.strip()
    if normalized.casefold().startswith("gpu-"):
        normalized = normalized[4:]
    if not normalized:
        raise ValueError("GPU UUID is empty after normalization")
    return normalized.casefold()


def _gpu_uuids_match(left: str, right: str) -> bool:
    return _canonical_gpu_uuid(left) == _canonical_gpu_uuid(right)


def parse_nvidia_smi_physical_gpu_memory(
    output: str,
    *,
    gpu_uuid: str,
) -> int | None:
    """Return total used memory for one physical GPU from CSV output."""

    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("GPU-memory sampling requires a physical GPU UUID")
    matches: list[int] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 2:
            raise ValueError("nvidia-smi GPU memory output is malformed")
        raw_uuid, raw_mebibytes = columns
        if not _gpu_uuids_match(raw_uuid, gpu_uuid):
            continue
        try:
            mebibytes = int(raw_mebibytes)
        except ValueError as error:
            raise ValueError(
                "nvidia-smi did not report numeric physical GPU memory"
            ) from error
        if mebibytes < 0:
            raise ValueError("nvidia-smi reported negative physical GPU memory")
        matches.append(mebibytes * 1024 * 1024)
    if len(matches) > 1:
        raise ValueError("nvidia-smi returned duplicate rows for this physical GPU")
    return matches[0] if matches else None


class NvidiaSmiGpuMemorySampler:
    """Poll physical VRAM, preferring process attribution when NVIDIA exposes it."""

    def __init__(
        self,
        *,
        gpu_uuid: str,
        pid: int | None = None,
        poll_interval_seconds: float = PROCESS_MEMORY_POLL_INTERVAL_SECONDS,
        command_runner: Any = subprocess.run,
    ) -> None:
        if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
            raise ValueError("Process GPU-memory sampler requires a GPU UUID")
        if poll_interval_seconds <= 0:
            raise ValueError("GPU-memory poll interval must be positive")
        self.gpu_uuid = gpu_uuid
        self.pid = os.getpid() if pid is None else pid
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._command_runner = command_runner
        self._process_samples: list[int] = []
        self._physical_gpu_samples: list[int] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query_process(self) -> int | None:
        completed = self._command_runner(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_COMMAND_TIMEOUT_SECONDS,
        )
        return parse_nvidia_smi_process_memory(
            completed.stdout,
            pid=self.pid,
            gpu_uuid=self.gpu_uuid,
        )

    def _query_physical_gpu(self) -> int | None:
        completed = self._command_runner(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_COMMAND_TIMEOUT_SECONDS,
        )
        return parse_nvidia_smi_physical_gpu_memory(
            completed.stdout,
            gpu_uuid=self.gpu_uuid,
        )

    def sample_now(self) -> int | None:
        try:
            process_value = self._query_process()
            if process_value is not None:
                self._process_samples.append(process_value)
                return process_value
            physical_gpu_value = self._query_physical_gpu()
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self._error = error
            return None
        if physical_gpu_value is not None:
            self._physical_gpu_samples.append(physical_gpu_value)
        return physical_gpu_value

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            self.sample_now()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU-memory sampler was already started")
        self.sample_now()
        self._thread = threading.Thread(
            target=self._poll,
            name="nvidia-smi-gpu-memory",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("GPU-memory sampler was not started")
        self._stop.set()
        self._thread.join(timeout=2 * NVIDIA_SMI_COMMAND_TIMEOUT_SECONDS + 1.0)
        if self._thread.is_alive():
            raise RuntimeError(
                "GPU-memory polling thread did not stop after the nvidia-smi "
                "command timeouts"
            )
        self._thread = None
        self.sample_now()
        if self._error is not None:
            raise RuntimeError(f"GPU memory sampling failed: {self._error}") from (
                self._error
            )
        if self._process_samples:
            return {
                "source": "nvidia-smi compute-apps used_gpu_memory polling",
                "measurement_scope": "process",
                "gpu_uuid": self.gpu_uuid,
                "namespace_pid": self.pid,
                "poll_interval_seconds": self.poll_interval_seconds,
                "samples": len(self._process_samples),
                "peak_observed_used_bytes": max(self._process_samples),
                "peak_process_used_bytes": max(self._process_samples),
            }
        if not self._physical_gpu_samples:
            raise RuntimeError(
                "nvidia-smi exposed neither this process nor physical GPU memory "
                "on the measured GPU"
            )
        return {
            "source": "nvidia-smi physical-GPU memory.used polling",
            "measurement_scope": "physical_gpu_total",
            "gpu_uuid": self.gpu_uuid,
            "namespace_pid": self.pid,
            "poll_interval_seconds": self.poll_interval_seconds,
            "samples": len(self._physical_gpu_samples),
            "baseline_observed_used_bytes": self._physical_gpu_samples[0],
            "peak_observed_used_bytes": max(self._physical_gpu_samples),
            "peak_physical_gpu_used_bytes": max(self._physical_gpu_samples),
            "limitation": (
                "nvidia-smi exposed no compute-process row; the observed peak is "
                "the conservative total usage of the identified physical GPU"
            ),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_config(config: FeasibilityProbeConfig) -> None:
    if config.execute and config.output_dir.exists():
        raise FileExistsError(
            f"Feasibility output already exists: {config.output_dir.resolve()}"
        )
    if config.optimization_plan is not None and not config.optimization_plan.is_file():
        raise FileNotFoundError(
            f"Validated optimization plan is missing: {config.optimization_plan}"
        )
    if not config.reference_corpus_dirs:
        raise ValueError("At least one pinned reference corpus is required")
    if len(set(config.reference_corpus_dirs)) != len(config.reference_corpus_dirs):
        raise ValueError("Reference corpus directories must be unique")


def projected_step_plan(
    retained_training_records: int,
    *,
    batch_size: int = PROBE_BATCH_SIZE,
) -> dict[str, Any]:
    """Return the conservative frozen pilot-plus-final optimizer-step plan."""

    if type(retained_training_records) is not int or retained_training_records <= 0:
        raise ValueError("retained_training_records must be a positive integer")
    if type(batch_size) is not int or batch_size != PROBE_BATCH_SIZE:
        raise ValueError(f"Feasibility batch size is frozen at {PROBE_BATCH_SIZE}")
    steps_per_epoch = math.ceil(retained_training_records / batch_size)
    pilot_steps_per_run = math.ceil(steps_per_epoch * PILOT_EPOCHS)
    pilot_runs_per_arm = len(LR_GRID)
    pilot_steps_per_arm = pilot_steps_per_run * pilot_runs_per_arm
    final_steps_per_arm = math.ceil(steps_per_epoch * MAX_FINAL_EPOCHS)
    return {
        "retained_training_records": retained_training_records,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": PROBE_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": batch_size * PROBE_GRADIENT_ACCUMULATION_STEPS,
        "steps_per_epoch": steps_per_epoch,
        "pilot_epochs_per_run": PILOT_EPOCHS,
        "pilot_runs_per_arm": pilot_runs_per_arm,
        "pilot_steps_per_run": pilot_steps_per_run,
        "pilot_steps_per_arm": pilot_steps_per_arm,
        "final_epoch_cap": MAX_FINAL_EPOCHS,
        "final_steps_per_arm": final_steps_per_arm,
        "pilot_plus_final_steps_per_arm": (pilot_steps_per_arm + final_steps_per_arm),
        "arms": len(EXPECTED_ARM_IDS),
    }


def select_longest_retained_batch(
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    batch_size: int = PROBE_BATCH_SIZE,
) -> LongestRetainedBatch:
    """Select the exact stable top-eight rows declared retained by the audit."""

    if type(batch_size) is not int or batch_size != PROBE_BATCH_SIZE:
        raise ValueError(f"Longest-batch proof requires batch {PROBE_BATCH_SIZE}")
    retained = audit.get("retained")
    if not isinstance(retained, list) or len(retained) < batch_size:
        raise ValueError(
            f"Tokenization audit must retain at least {batch_size} training rows"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id or task_id in by_id:
            raise ValueError("Canonical rows have missing or duplicate task IDs")
        by_id[task_id] = row

    ranked: list[tuple[int, int, str]] = []
    for source_index, item in enumerate(retained):
        if not isinstance(item, dict):
            raise ValueError("Tokenization retained list must contain objects")
        task_id = item.get("task_id")
        length = item.get("combined_tokens")
        if not isinstance(task_id, str) or task_id not in by_id:
            raise ValueError("Tokenization audit references an unknown retained task")
        if type(length) is not int or not 0 < length <= TOKENIZER_MAX_LENGTH:
            raise ValueError(f"Retained task {task_id} has an invalid token length")
        ranked.append((-length, source_index, task_id))
    selected = sorted(ranked)[:batch_size]
    task_ids = tuple(task_id for _, _, task_id in selected)
    lengths = tuple(-negative_length for negative_length, _, _ in selected)
    return LongestRetainedBatch(
        rows=tuple(by_id[task_id] for task_id in task_ids),
        task_ids=task_ids,
        combined_token_lengths=lengths,
        task_ids_digest=digest_json(list(task_ids)),
    )


def _validate_arm_measurement(result: Mapping[str, Any]) -> None:
    arm_id = result.get("arm_id")
    if arm_id not in ARM_SPECS:
        raise ValueError(f"Unknown feasibility arm result: {arm_id!r}")
    if result.get("status") != "passed":
        raise ValueError(f"Feasibility arm {arm_id} did not pass")
    longest = result.get("longest_batch")
    if (
        not isinstance(longest, dict)
        or longest.get("passed") is not True
        or longest.get("batch_size") != PROBE_BATCH_SIZE
        or longest.get("precision") != PROBE_PRECISION
    ):
        raise ValueError(f"Arm {arm_id} lacks the exact bf16 batch-8 proof")
    throughput = result.get("throughput")
    if not isinstance(throughput, dict):
        raise ValueError(f"Arm {arm_id} has no throughput measurement")
    steps = throughput.get("optimizer_steps")
    elapsed = throughput.get("elapsed_seconds")
    if steps != FEASIBILITY_STEPS:
        raise ValueError(
            f"Arm {arm_id} must measure exactly {FEASIBILITY_STEPS} optimizer steps"
        )
    if (
        not isinstance(elapsed, int | float)
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError(f"Arm {arm_id} has invalid elapsed seconds")
    protocol_overhead = throughput.get("protocol_overhead")
    if not isinstance(protocol_overhead, dict):
        raise ValueError(f"Arm {arm_id} did not measure protocol overhead")
    required_overhead = {
        "passed": True,
        "full_validation_evaluations_measured": 1,
        "weights_only_checkpoint_writes_measured": 1,
        "setup_wandb_and_best_reload_factor": PROJECTION_SERVICE_OVERHEAD_FACTOR,
    }
    if any(
        protocol_overhead.get(field) != expected
        for field, expected in required_overhead.items()
    ):
        raise ValueError(f"Arm {arm_id} protocol-overhead contract drifted")
    for field in ("validation_elapsed_seconds", "checkpoint_elapsed_seconds"):
        value = protocol_overhead.get(field)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Arm {arm_id} has invalid measured {field}")
    gpu = result.get("gpu")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("identity_digest"), str):
        raise ValueError(f"Arm {arm_id} has no physical GPU identity")
    physical = gpu.get("physical_vram_bytes")
    if type(physical) is not int or physical <= 0:
        raise ValueError(f"Arm {arm_id} has invalid physical VRAM")
    gpu_memory = result.get("gpu_memory_telemetry")
    if not isinstance(gpu_memory, dict):
        raise ValueError(f"Arm {arm_id} has no physical peak GPU memory")
    measurement_scope = gpu_memory.get("measurement_scope")
    if (
        gpu_memory.get("gpu_uuid") != gpu.get("uuid")
        or measurement_scope not in {"process", "physical_gpu_total"}
        or type(gpu_memory.get("namespace_pid")) is not int
        or type(gpu_memory.get("samples")) is not int
        or gpu_memory["samples"] <= 0
        or type(gpu_memory.get("peak_observed_used_bytes")) is not int
        or not 0 < gpu_memory["peak_observed_used_bytes"] <= physical
    ):
        raise ValueError(f"Arm {arm_id} has invalid physical peak GPU memory")
    if measurement_scope == "process" and (
        gpu_memory.get("peak_process_used_bytes")
        != gpu_memory["peak_observed_used_bytes"]
    ):
        raise ValueError(f"Arm {arm_id} has invalid process peak GPU memory")
    if measurement_scope == "physical_gpu_total":
        baseline = gpu_memory.get("baseline_observed_used_bytes")
        peak = gpu_memory.get("peak_physical_gpu_used_bytes")
        if (
            type(baseline) is not int
            or type(peak) is not int
            or not 0 <= baseline <= peak <= physical
            or peak != gpu_memory["peak_observed_used_bytes"]
            or not isinstance(gpu_memory.get("limitation"), str)
            or not gpu_memory["limitation"]
        ):
            raise ValueError(f"Arm {arm_id} has invalid physical-GPU-total peak memory")
    for section_name in ("longest_batch", "throughput"):
        section = result[section_name]
        allocated = section.get("peak_torch_allocated_bytes")
        reserved = section.get("peak_torch_reserved_bytes")
        if (
            type(allocated) is not int
            or type(reserved) is not int
            or not 0 < allocated <= reserved <= physical
        ):
            raise ValueError(f"Arm {arm_id} has invalid {section_name} peak memory")
    if not isinstance(result.get("study_identity_digest"), str):
        raise ValueError(f"Arm {arm_id} has no frozen study identity digest")


def project_arm_runtime(
    result: Mapping[str, Any],
    step_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Project all pilots and final including measured protocol overhead."""

    _validate_arm_measurement(result)
    throughput = result["throughput"]
    elapsed = float(throughput["elapsed_seconds"])
    seconds_per_step = elapsed / FEASIBILITY_STEPS
    overhead = throughput["protocol_overhead"]
    validation_seconds = float(overhead["validation_elapsed_seconds"])
    checkpoint_seconds = float(overhead["checkpoint_elapsed_seconds"])
    pilot_steps_per_run = int(step_plan["pilot_steps_per_run"])
    pilot_runs = int(step_plan["pilot_runs_per_arm"])
    pilot_training_seconds_per_run = seconds_per_step * pilot_steps_per_run
    pilot_seconds_per_run_before_margin = (
        pilot_training_seconds_per_run
        + PILOT_VALIDATIONS_PER_RUN * validation_seconds
        + PILOT_CHECKPOINT_WRITES_PER_RUN * checkpoint_seconds
    )
    pilot_seconds_per_run = (
        pilot_seconds_per_run_before_margin * PROJECTION_SERVICE_OVERHEAD_FACTOR
    )
    pilot_steps = int(step_plan["pilot_steps_per_arm"])
    pilot_seconds = pilot_seconds_per_run * pilot_runs
    final_steps = int(step_plan["final_steps_per_arm"])
    final_training_seconds = seconds_per_step * final_steps
    final_seconds_before_margin = (
        final_training_seconds
        + FINAL_VALIDATIONS * validation_seconds
        + FINAL_CHECKPOINT_WRITES * checkpoint_seconds
    )
    final_seconds = final_seconds_before_margin * PROJECTION_SERVICE_OVERHEAD_FACTOR
    total_seconds = pilot_seconds + final_seconds
    return {
        "arm_id": result["arm_id"],
        "measured_optimizer_steps": FEASIBILITY_STEPS,
        "measured_elapsed_seconds": elapsed,
        "optimizer_steps_per_second": FEASIBILITY_STEPS / elapsed,
        "seconds_per_optimizer_step": seconds_per_step,
        "measured_full_validation_seconds": validation_seconds,
        "measured_weights_only_checkpoint_seconds": checkpoint_seconds,
        "setup_wandb_and_best_reload_factor": PROJECTION_SERVICE_OVERHEAD_FACTOR,
        "pilot_runs": pilot_runs,
        "projected_pilot_steps_per_run": pilot_steps_per_run,
        "projected_pilot_validations_per_run": PILOT_VALIDATIONS_PER_RUN,
        "projected_pilot_checkpoint_writes_per_run": (PILOT_CHECKPOINT_WRITES_PER_RUN),
        "projected_pilot_run_hours": pilot_seconds_per_run / 3600,
        "projected_pilot_steps": pilot_steps,
        "projected_pilot_hours": pilot_seconds / 3600,
        "projected_final_validations": FINAL_VALIDATIONS,
        "projected_final_checkpoint_writes": FINAL_CHECKPOINT_WRITES,
        "projected_final_steps": final_steps,
        "projected_final_hours": final_seconds / 3600,
        "projected_pilot_plus_final_hours": total_seconds / 3600,
        "projection_includes": [
            "all three quarter-epoch pilot training runs",
            "one epoch-end plus one explicit validation per pilot",
            "one epoch-end plus one final artifact write per pilot",
            "three final epoch validations plus one explicit final validation",
            "three final epoch checkpoint writes plus one final artifact write",
            (
                "15% setup/W&B/best-checkpoint-reload/final-artifact margin on "
                "every projected run"
            ),
        ],
    }


def _results_by_arm(
    arm_results: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for result in arm_results:
        arm_id = result.get("arm_id")
        if not isinstance(arm_id, str) or arm_id in results:
            raise ValueError("Feasibility results have missing or duplicate arm IDs")
        results[arm_id] = result
    missing = sorted(set(EXPECTED_ARM_IDS) - set(results))
    extra = sorted(set(results) - set(EXPECTED_ARM_IDS))
    if missing or extra:
        raise ValueError(
            f"Feasibility requires exactly all arms; missing={missing}, extra={extra}"
        )
    return results


def _validate_content_digest(value: Mapping[str, Any]) -> str:
    stored = value.get("artifact_digest")
    payload = dict(value)
    payload.pop("artifact_digest", None)
    expected = digest_json(payload)
    if stored != expected:
        raise ValueError(f"Content digest mismatch: {stored!r} != {expected!r}")
    return expected


def _validated_optimization_projections(
    plan: Mapping[str, Any],
    *,
    baseline_by_arm: Mapping[str, Mapping[str, Any]],
    step_plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _validate_content_digest(plan)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Optimization plan has an unsupported schema version")
    if plan.get("artifact_type") != OPTIMIZATION_ARTIFACT_TYPE:
        raise ValueError("Optimization plan has the wrong artifact type")
    if plan.get("status") != "validated":
        raise ValueError("Optimization plan status is not validated")
    validation = plan.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Optimization plan has no validation contract")
    required_validation = {
        "preserves_frozen_study_protocol": True,
        "runtime_monitoring_validated": True,
        "hard_stop_hours": FEASIBILITY_HARD_LIMIT_HOURS,
    }
    for field, expected in required_validation.items():
        if validation.get(field) != expected:
            raise ValueError(
                f"Optimization validation field {field!r} must be {expected!r}"
            )
    measurements = plan.get("arm_measurements")
    if not isinstance(measurements, list):
        raise ValueError("Optimization plan must contain arm measurements")
    if not measurements:
        raise ValueError("Optimization plan contains no arm measurements")
    optimized_by_arm: dict[str, Mapping[str, Any]] = {}
    for measurement in measurements:
        arm_id = measurement.get("arm_id")
        if arm_id not in ARM_SPECS or arm_id in optimized_by_arm:
            raise ValueError(
                "Optimization measurements have unknown or duplicate arm IDs"
            )
        optimized_by_arm[arm_id] = measurement
    projections: dict[str, dict[str, Any]] = {}
    for arm_id, optimized in optimized_by_arm.items():
        baseline = baseline_by_arm[arm_id]
        if optimized.get("study_identity_digest") != baseline.get(
            "study_identity_digest"
        ):
            raise ValueError(f"Optimization changed {arm_id}'s frozen study identity")
        if optimized.get("gpu", {}).get("identity_digest") != baseline.get(
            "gpu", {}
        ).get("identity_digest"):
            raise ValueError(f"Optimization measured {arm_id} on another GPU")
        projections[arm_id] = {
            "artifact_digest": plan["artifact_digest"],
            "projection": project_arm_runtime(optimized, step_plan),
        }
    return projections


def evaluate_launch_gate(
    arm_results: Sequence[Mapping[str, Any]],
    *,
    retained_training_records: int,
    optimization_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the 14-hour soft and 16-hour hard launch rule per arm."""

    step_plan = projected_step_plan(retained_training_records)
    try:
        baseline_by_arm = _results_by_arm(arm_results)
    except ValueError as error:
        return {
            "launch_allowed": False,
            "all_arms_authorized": False,
            "authorized_arm_ids": [],
            "blocked_arm_ids": list(EXPECTED_ARM_IDS),
            "reason": str(error),
            "soft_limit_hours": FEASIBILITY_SOFT_LIMIT_HOURS,
            "hard_limit_hours": FEASIBILITY_HARD_LIMIT_HOURS,
            "runtime_hard_stop_required": True,
            "step_plan": step_plan,
            "per_arm": {
                arm_id: {
                    "launch_allowed": False,
                    "route": "blocked",
                    "reason": str(error),
                    "baseline": None,
                    "optimization": None,
                }
                for arm_id in EXPECTED_ARM_IDS
            },
            "diagnostic_totals": None,
        }

    optimization_by_arm: dict[str, dict[str, Any]] = {}
    optimization_error: str | None = None
    if optimization_plan is not None:
        try:
            optimization_by_arm = _validated_optimization_projections(
                optimization_plan,
                baseline_by_arm=baseline_by_arm,
                step_plan=step_plan,
            )
        except ValueError as error:
            optimization_error = str(error)

    per_arm: dict[str, dict[str, Any]] = {}
    baseline_hours: list[float] = []
    effective_hours: list[float] = []
    for arm_id in EXPECTED_ARM_IDS:
        try:
            baseline = project_arm_runtime(baseline_by_arm[arm_id], step_plan)
        except ValueError as error:
            per_arm[arm_id] = {
                "launch_allowed": False,
                "route": "blocked",
                "reason": str(error),
                "baseline": None,
                "optimization": None,
            }
            continue
        projected = baseline["projected_pilot_plus_final_hours"]
        baseline_hours.append(projected)
        common = {
            "soft_limit_hours": FEASIBILITY_SOFT_LIMIT_HOURS,
            "hard_limit_hours": FEASIBILITY_HARD_LIMIT_HOURS,
            "runtime_hard_stop_required": True,
            "baseline": baseline,
        }
        if projected <= FEASIBILITY_SOFT_LIMIT_HOURS:
            effective_hours.append(projected)
            per_arm[arm_id] = {
                "launch_allowed": True,
                "route": "direct",
                "reason": None,
                **common,
                "optimization": None,
            }
            continue
        if projected > FEASIBILITY_HARD_LIMIT_HOURS:
            per_arm[arm_id] = {
                "launch_allowed": False,
                "route": "blocked",
                "reason": "baseline projection exceeds the 16-hour hard limit",
                **common,
                "optimization": None,
            }
            continue
        optimized = optimization_by_arm.get(arm_id)
        if optimized is None:
            reason = (
                f"invalid optimization plan: {optimization_error}"
                if optimization_error is not None
                else (
                    "baseline projection exceeds 14 hours and this arm has no "
                    "validated optimization measurement"
                )
            )
            per_arm[arm_id] = {
                "launch_allowed": False,
                "route": "blocked",
                "reason": reason,
                **common,
                "optimization": None,
            }
            continue
        optimized_hours = optimized["projection"]["projected_pilot_plus_final_hours"]
        if optimized_hours > FEASIBILITY_SOFT_LIMIT_HOURS:
            per_arm[arm_id] = {
                "launch_allowed": False,
                "route": "blocked",
                "reason": "validated optimization still projects above 14 hours",
                **common,
                "optimization": optimized,
            }
            continue
        effective_hours.append(optimized_hours)
        per_arm[arm_id] = {
            "launch_allowed": True,
            "route": "validated-optimization-plan",
            "reason": None,
            **common,
            "optimization": optimized,
        }

    authorized = [
        arm_id for arm_id in EXPECTED_ARM_IDS if per_arm[arm_id]["launch_allowed"]
    ]
    blocked = [arm_id for arm_id in EXPECTED_ARM_IDS if arm_id not in authorized]
    return {
        "launch_allowed": not blocked,
        "all_arms_authorized": not blocked,
        "authorized_arm_ids": authorized,
        "blocked_arm_ids": blocked,
        "reason": None if not blocked else "one or more arms are not feasible",
        "soft_limit_hours": FEASIBILITY_SOFT_LIMIT_HOURS,
        "hard_limit_hours": FEASIBILITY_HARD_LIMIT_HOURS,
        "runtime_hard_stop_required": True,
        "step_plan": step_plan,
        "per_arm": per_arm,
        "diagnostic_totals": {
            "decision_scope": "per-arm; totals never reject an arm",
            "baseline_projected_all_arms_hours": (
                sum(baseline_hours)
                if len(baseline_hours) == len(EXPECTED_ARM_IDS)
                else None
            ),
            "authorized_effective_projected_hours": sum(effective_hours),
        },
    }


def _not_run_gate(retained_training_records: int) -> dict[str, Any]:
    return {
        "launch_allowed": False,
        "all_arms_authorized": False,
        "authorized_arm_ids": [],
        "blocked_arm_ids": list(EXPECTED_ARM_IDS),
        "reason": "the GPU proofs and exact 500-step measurements have not run",
        "soft_limit_hours": FEASIBILITY_SOFT_LIMIT_HOURS,
        "hard_limit_hours": FEASIBILITY_HARD_LIMIT_HOURS,
        "runtime_hard_stop_required": True,
        "step_plan": projected_step_plan(retained_training_records),
        "per_arm": {
            arm_id: {
                "launch_allowed": False,
                "route": "dry-run",
                "reason": "GPU proof and throughput measurement not run",
                "baseline": None,
                "optimization": None,
            }
            for arm_id in EXPECTED_ARM_IDS
        },
        "diagnostic_totals": None,
    }


def build_probe_manifest(
    *,
    status: str,
    started_at_utc: str,
    source_code: Mapping[str, Any],
    source_data: Mapping[str, Any],
    retained_training_records: int,
    arm_results: Sequence[Mapping[str, Any]],
    optimization_plan: Mapping[str, Any] | None = None,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a canonical, self-digesting feasibility artifact."""

    if status not in {"dry-run", "complete"}:
        raise ValueError("Probe manifest status must be dry-run or complete")
    gate = (
        _not_run_gate(retained_training_records)
        if status == "dry-run"
        else evaluate_launch_gate(
            arm_results,
            retained_training_records=retained_training_records,
            optimization_plan=optimization_plan,
        )
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "source_code": dict(source_code),
        "source_data": dict(source_data),
        "retained_training_records": retained_training_records,
        "probe_protocol": {
            "arms": list(EXPECTED_ARM_IDS),
            "optimizer_steps_per_arm": FEASIBILITY_STEPS,
            "per_device_batch_size": PROBE_BATCH_SIZE,
            "gradient_accumulation_steps": PROBE_GRADIENT_ACCUMULATION_STEPS,
            "precision": PROBE_PRECISION,
            "max_length": TOKENIZER_MAX_LENGTH,
            "learning_rate": PROBE_LEARNING_RATE,
            "initialization_seed": INITIALIZATION_SEED,
            "data_order_seed": DATA_ORDER_SEED,
            "longest_batch_proof": (
                "the stable eight largest retained combined-token sequences"
            ),
            "projection_scope": (
                "three quarter-epoch LR pilots plus one final at the three-epoch cap "
                "for every arm"
            ),
            "projection_overhead": {
                "measured_full_validation_evaluations": 1,
                "measured_weights_only_checkpoint_writes": 1,
                "pilot_runs": len(LR_GRID),
                "pilot_validations_per_run": PILOT_VALIDATIONS_PER_RUN,
                "pilot_checkpoint_writes_per_run": (PILOT_CHECKPOINT_WRITES_PER_RUN),
                "final_validations": FINAL_VALIDATIONS,
                "final_checkpoint_writes": FINAL_CHECKPOINT_WRITES,
                "setup_wandb_and_best_reload_factor": (
                    PROJECTION_SERVICE_OVERHEAD_FACTOR
                ),
                "factor_scope": (
                    "setup, online W&B, best-checkpoint reload, and final artifact "
                    "serialization not present in the bounded local measurement"
                ),
            },
            "throughput_deviations": [
                "W&B disabled",
                "scheduler horizon fixed to the 500-step probe",
            ],
        },
        "arm_results": [dict(result) for result in arm_results],
        "optimization_plan": (
            dict(optimization_plan) if optimization_plan is not None else None
        ),
        "launch_gate": gate,
    }
    payload["artifact_digest"] = digest_json(payload)
    return payload


def validate_probe_manifest(
    manifest: Mapping[str, Any],
    *,
    require_launch_allowed: bool = False,
    arm_id: str | None = None,
) -> dict[str, Any]:
    """Validate an artifact before a full-run launcher consumes it."""

    _validate_content_digest(manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Feasibility manifest has an unsupported schema version")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Feasibility manifest has the wrong artifact type")
    if manifest.get("status") not in {"dry-run", "complete"}:
        raise ValueError("Feasibility manifest has an invalid status")
    protocol = manifest.get("probe_protocol")
    if not isinstance(protocol, dict) or (
        protocol.get("optimizer_steps_per_arm") != FEASIBILITY_STEPS
        or protocol.get("per_device_batch_size") != PROBE_BATCH_SIZE
        or protocol.get("gradient_accumulation_steps")
        != PROBE_GRADIENT_ACCUMULATION_STEPS
        or protocol.get("precision") != PROBE_PRECISION
        or protocol.get("max_length") != TOKENIZER_MAX_LENGTH
        or protocol.get("arms") != list(EXPECTED_ARM_IDS)
    ):
        raise ValueError("Feasibility manifest probe protocol drifted")
    expected_projection_overhead = {
        "measured_full_validation_evaluations": 1,
        "measured_weights_only_checkpoint_writes": 1,
        "pilot_runs": len(LR_GRID),
        "pilot_validations_per_run": PILOT_VALIDATIONS_PER_RUN,
        "pilot_checkpoint_writes_per_run": PILOT_CHECKPOINT_WRITES_PER_RUN,
        "final_validations": FINAL_VALIDATIONS,
        "final_checkpoint_writes": FINAL_CHECKPOINT_WRITES,
        "setup_wandb_and_best_reload_factor": PROJECTION_SERVICE_OVERHEAD_FACTOR,
        "factor_scope": (
            "setup, online W&B, best-checkpoint reload, and final artifact "
            "serialization not present in the bounded local measurement"
        ),
    }
    if protocol.get("projection_overhead") != expected_projection_overhead:
        raise ValueError("Feasibility projection overhead contract drifted")
    source_data = manifest.get("source_data")
    if not isinstance(source_data, dict):
        raise ValueError("Feasibility manifest has no source-data identity")
    references = source_data.get("reference_corpora")
    if not isinstance(references, list) or not references:
        raise ValueError("Feasibility source identity has no reference corpora")
    if source_data.get("reference_corpora_digest") != digest_json(references):
        raise ValueError("Feasibility reference-corpus identity digest changed")
    gate = manifest.get("launch_gate")
    if not isinstance(gate, dict):
        raise ValueError("Feasibility manifest has no launch gate")
    retained_records = manifest.get("retained_training_records")
    if type(retained_records) is not int or retained_records <= 0:
        raise ValueError("Feasibility manifest has no retained training count")
    status = manifest["status"]
    if status == "complete":
        arm_results = manifest.get("arm_results")
        if not isinstance(arm_results, list):
            raise ValueError("Complete feasibility manifest has no arm results")
        optimization_plan = manifest.get("optimization_plan")
        if optimization_plan is not None and not isinstance(optimization_plan, dict):
            raise ValueError("Embedded optimization plan must be a JSON object")
        expected_gate = evaluate_launch_gate(
            arm_results,
            retained_training_records=retained_records,
            optimization_plan=optimization_plan,
        )
    else:
        expected_gate = _not_run_gate(retained_records)
    if gate != expected_gate:
        raise ValueError("Feasibility launch gate does not match its measurements")
    if arm_id is not None and arm_id not in ARM_SPECS:
        raise ValueError(f"Unknown feasibility arm requested: {arm_id!r}")
    if require_launch_allowed:
        if status != "complete":
            raise ValueError("Feasibility artifact does not authorize full runs")
        if arm_id is None:
            authorized = gate.get("all_arms_authorized") is True
        else:
            authorized = (
                gate.get("per_arm", {}).get(arm_id, {}).get("launch_allowed") is True
            )
        if not authorized:
            target = "all arms" if arm_id is None else f"arm {arm_id!r}"
            raise ValueError(f"Feasibility artifact does not authorize {target}")
    return dict(manifest)


def load_probe_manifest(
    path: str | Path,
    *,
    require_launch_allowed: bool = False,
    arm_id: str | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read feasibility manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("Feasibility manifest must be a JSON object")
    return validate_probe_manifest(
        value,
        require_launch_allowed=require_launch_allowed,
        arm_id=arm_id,
    )


def _load_optimization_plan(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read optimization plan {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Optimization plan must be a JSON object")
    _validate_content_digest(value)
    return value


def _gpu_identity(torch: Any) -> dict[str, Any]:
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
    return {
        **identity,
        "identity_digest": digest_json(identity),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _configure_gradient_checkpointing(
    model: Any, *, enabled: bool, kwargs: Mapping[str, Any]
) -> None:
    if enabled:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(kwargs)
        )


def _assert_bf16_model(model: Any, torch: Any) -> None:
    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            f"Scratch probe model is not entirely bf16: {sorted(map(str, floating_dtypes))}"
        )


def _longest_batch_proof(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    selected: LongestRetainedBatch,
    gradient_checkpointing: bool,
    gradient_checkpointing_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    _configure_gradient_checkpointing(
        model,
        enabled=gradient_checkpointing,
        kwargs=gradient_checkpointing_kwargs,
    )
    model.to("cuda")
    model.train()
    encoded_rows = [encode_training_row(tokenizer, row) for row in selected.rows]
    observed_lengths = tuple(len(row["combined_ids"]) for row in encoded_rows)
    if observed_lengths != selected.combined_token_lengths:
        raise ValueError(
            "Longest-batch token lengths differ from the verified tokenization audit"
        )
    maximum = max(observed_lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Shared tokenizer has no PAD token ID")
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for encoded in encoded_rows:
        ids = list(encoded["combined_ids"])
        completion_mask = list(encoded["completion_mask"])
        padding = maximum - len(ids)
        input_ids.append([*ids, *([pad_id] * padding)])
        attention_masks.append([*([1] * len(ids)), *([0] * padding)])
        labels.append(
            [
                token_id if selected_token else -100
                for token_id, selected_token in zip(ids, completion_mask, strict=True)
            ]
            + [-100] * padding
        )
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda"),
        "attention_mask": torch.tensor(
            attention_masks,
            dtype=torch.long,
            device="cuda",
        ),
        "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
    }
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Longest-batch forward produced no finite loss")
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        **selected.identity_dict(),
        "passed": True,
        "precision": PROBE_PRECISION,
        "forward_backward_seconds": elapsed,
        "loss": float(loss.detach().float().cpu()),
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    model.zero_grad(set_to_none=True)
    del batch, outputs, loss
    return result


def _throughput_probe(
    *,
    torch: Any,
    runner: Any,
    resolved: Any,
    data: Any,
    tokenizer: Any,
    work_dir: Path,
    expected_initialization_digest: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    from datasets import Dataset
    from trl import SFTTrainer

    from experiments.train_deep_corpus_sft import assert_weights_only_artifact

    model, initialization = initialize_model_from_config(
        resolved,
        initialization_seed=INITIALIZATION_SEED,
        torch_dtype=torch.bfloat16,
    )
    _assert_bf16_model(model, torch)
    if initialization["initial_state_digest"] != expected_initialization_digest:
        raise RuntimeError("Throughput model did not restart from the proven state")
    probe_config = runner.ScratchRunConfig(
        output_dir=work_dir,
        corpus_dir=Path("."),
        tokenizer_dir=Path("."),
        arm_id=resolved.spec.arm_id,
        phase="pilot",
        learning_rate=PROBE_LEARNING_RATE,
        num_train_epochs=PILOT_EPOCHS,
        dry_run=True,
    )
    training_args = runner.build_sft_config(
        probe_config,
        report_to_wandb=False,
        precision_bf16=True,
    )
    training_args.max_steps = FEASIBILITY_STEPS
    training_args.eval_strategy = "no"
    training_args.save_strategy = "no"
    training_args.load_best_model_at_end = False
    training_args.disable_tqdm = True
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list([dict(row) for row in data.train_rows]),
        eval_dataset=Dataset.from_list([dict(row) for row in data.validation_rows]),
        processing_class=tokenizer,
    )
    if trainer.model is not model:
        raise RuntimeError("Probe trainer replaced the config-created model")
    assert_config_created_model(trainer.model, resolved)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if trainer.state.global_step != FEASIBILITY_STEPS:
        raise RuntimeError(
            f"Probe stopped at {trainer.state.global_step}, not {FEASIBILITY_STEPS}"
        )
    torch.cuda.synchronize()
    evaluation_started = time.perf_counter()
    evaluation_metrics = trainer.evaluate()
    torch.cuda.synchronize()
    evaluation_elapsed = time.perf_counter() - evaluation_started
    measured_checkpoint = work_dir / "measured-weights-only-checkpoint"
    if measured_checkpoint.exists():
        raise FileExistsError(
            f"Measured checkpoint path already exists: {measured_checkpoint}"
        )
    checkpoint_started = time.perf_counter()
    trainer.save_model(str(measured_checkpoint))
    checkpoint_elapsed = time.perf_counter() - checkpoint_started
    assert_weights_only_artifact(measured_checkpoint)
    result = {
        "passed": True,
        "optimizer_steps": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "optimizer_steps_per_second": FEASIBILITY_STEPS / elapsed,
        "examples_per_second_at_batch_8": (
            FEASIBILITY_STEPS * PROBE_BATCH_SIZE / elapsed
        ),
        "num_input_tokens_seen": int(
            getattr(trainer.state, "num_input_tokens_seen", 0)
        ),
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "trainer_metrics": dict(train_result.metrics),
        "protocol_overhead": {
            "passed": True,
            "full_validation_evaluations_measured": 1,
            "validation_examples": len(data.validation_rows),
            "validation_elapsed_seconds": evaluation_elapsed,
            "validation_metrics": dict(evaluation_metrics),
            "weights_only_checkpoint_writes_measured": 1,
            "checkpoint_elapsed_seconds": checkpoint_elapsed,
            "checkpoint_bytes": sum(
                path.stat().st_size
                for path in measured_checkpoint.rglob("*")
                if path.is_file()
            ),
            "setup_wandb_and_best_reload_factor": (PROJECTION_SERVICE_OVERHEAD_FACTOR),
            "factor_scope": (
                "setup, online W&B, best-checkpoint reload, and final artifact "
                "serialization"
            ),
        },
    }
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, initialization


def _arm_study_identity(
    *,
    resolved: Any,
    data: Any,
    runner: Any,
) -> tuple[dict[str, Any], str]:
    protocol = build_scratch_study_identity(
        architecture=resolved.identity_dict(),
        corpus_manifest_digest=data.corpus_manifest["manifest_digest"],
        tokenizer_artifact_digest=data.tokenizer_manifest["artifact_digest"],
        retained_training_task_ids_digest=data.train_audit[
            "retained_task_ids_digest"
        ],
        per_device_batch_size=runner.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=runner.GRADIENT_ACCUMULATION_STEPS,
        max_length=TOKENIZER_MAX_LENGTH,
        optimizer="adamw_torch",
        precision=PROBE_PRECISION,
        gradient_checkpointing=(
            resolved.spec.arm_id in runner.GRADIENT_CHECKPOINTING_ARMS
        ),
        gradient_checkpointing_kwargs=(
            runner.GRADIENT_CHECKPOINTING_KWARGS
            if resolved.spec.arm_id in runner.GRADIENT_CHECKPOINTING_ARMS
            else {}
        ),
        cublas_workspace_config=DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        pythonhashseed=DETERMINISTIC_PYTHON_HASH_SEED,
        initialization_seed=INITIALIZATION_SEED,
        data_order_seed=DATA_ORDER_SEED,
    )
    return protocol, digest_json(protocol)


def _execute_arm(
    *,
    torch: Any,
    runner: Any,
    resolved: Any,
    data: Any,
    tokenizer: Any,
    gpu: Mapping[str, Any],
    work_root: Path,
) -> dict[str, Any]:
    arm_id = resolved.spec.arm_id
    selected = select_longest_retained_batch(data.train_rows, data.train_audit)
    protocol, study_digest = _arm_study_identity(
        resolved=resolved,
        data=data,
        runner=runner,
    )
    result: dict[str, Any] = {
        "arm_id": arm_id,
        "status": "failed",
        "architecture": resolved.identity_dict(),
        "study_protocol": protocol,
        "study_identity_digest": study_digest,
        "gpu": dict(gpu),
        "longest_batch": {
            **selected.identity_dict(),
            "passed": False,
            "precision": PROBE_PRECISION,
        },
        "throughput": None,
    }
    sampler = NvidiaSmiGpuMemorySampler(gpu_uuid=str(gpu.get("uuid") or ""))
    sampler.start()
    model = None
    try:
        model, initialization = initialize_model_from_config(
            resolved,
            initialization_seed=INITIALIZATION_SEED,
            torch_dtype=torch.bfloat16,
        )
        _assert_bf16_model(model, torch)
        result["initialization"] = dict(initialization)
        proof = _longest_batch_proof(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            selected=selected,
            gradient_checkpointing=(arm_id in runner.GRADIENT_CHECKPOINTING_ARMS),
            gradient_checkpointing_kwargs=(
                runner.GRADIENT_CHECKPOINTING_KWARGS
                if arm_id in runner.GRADIENT_CHECKPOINTING_ARMS
                else {}
            ),
        )
        result["longest_batch"] = proof
        sampler.sample_now()
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        throughput, throughput_initialization = _throughput_probe(
            torch=torch,
            runner=runner,
            resolved=resolved,
            data=data,
            tokenizer=tokenizer,
            work_dir=work_root / arm_id,
            expected_initialization_digest=initialization["initial_state_digest"],
        )
        result["throughput_initialization"] = dict(throughput_initialization)
        result["throughput"] = throughput
        sampler.sample_now()
        result["status"] = "passed"
        result["peak_torch_allocated_bytes"] = max(
            proof["peak_torch_allocated_bytes"],
            throughput["peak_torch_allocated_bytes"],
        )
        result["peak_torch_reserved_bytes"] = max(
            proof["peak_torch_reserved_bytes"],
            throughput["peak_torch_reserved_bytes"],
        )
        return result
    except torch.cuda.OutOfMemoryError as error:
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "stage": (
                "throughput"
                if result["longest_batch"].get("passed")
                else "longest_batch"
            ),
        }
        return result
    finally:
        active_error = sys.exc_info()[1]
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        try:
            result["gpu_memory_telemetry"] = sampler.stop()
        except RuntimeError as sampling_error:
            if active_error is None:
                raise
            result["gpu_memory_telemetry_failure"] = {
                "type": type(sampling_error).__name__,
                "message": str(sampling_error),
            }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_probe(config: FeasibilityProbeConfig) -> dict[str, Any]:
    """Verify inputs, optionally execute all GPU probes, and freeze one artifact."""

    _validate_config(config)
    from experiments import train_deep_corpus_v2_scratch as runner
    from experiments.train_deep_corpus_sft import (
        _git_identity,
        _require_unchanged_source,
    )

    if runner.PER_DEVICE_BATCH_SIZE != PROBE_BATCH_SIZE:
        raise ValueError("Training runner batch size drifted from the feasibility gate")
    if runner.GRADIENT_ACCUMULATION_STEPS != PROBE_GRADIENT_ACCUMULATION_STEPS:
        raise ValueError(
            "Training runner accumulation drifted from the feasibility gate"
        )
    started_at = datetime.now(UTC).isoformat()
    repo_root = Path(__file__).resolve().parents[1]
    destination = config.output_dir.resolve()
    source_code = _git_identity(repo_root, ignored_paths=(destination,))
    if config.execute and source_code["dirty"]:
        raise RuntimeError("Actual feasibility probes require a clean source tree")

    base_config = runner.ScratchRunConfig(
        output_dir=destination,
        corpus_dir=config.corpus_dir,
        tokenizer_dir=config.tokenizer_dir,
        reference_corpus_dirs=config.reference_corpus_dirs,
        arm_id=EXPECTED_ARM_IDS[0],
        phase="pilot",
        learning_rate=PROBE_LEARNING_RATE,
        num_train_epochs=PILOT_EPOCHS,
        dry_run=True,
    )
    data = runner.prepare_scratch_data(base_config)
    minimum_probe_examples = FEASIBILITY_STEPS * PROBE_BATCH_SIZE
    if len(data.train_rows) < minimum_probe_examples:
        raise ValueError(
            "The exact 500-step batch-8 probe requires at least "
            f"{minimum_probe_examples} retained discovery rows"
        )
    resolved_by_arm = {}
    for arm_id in EXPECTED_ARM_IDS:
        arm_config = runner.ScratchRunConfig(
            **{
                **base_config.__dict__,
                "arm_id": arm_id,
            }
        )
        resolved_by_arm[arm_id] = runner.resolve_run_architecture(arm_config, data)
    selected = select_longest_retained_batch(data.train_rows, data.train_audit)
    references = list(data.source_identity.get("reference_corpora", ()))
    if not references:
        raise ValueError("Prepared scratch data has no reference-corpus identity")
    source_data = {
        **dict(data.source_identity),
        "reference_corpora_digest": digest_json(references),
        "longest_retained_batch": selected.identity_dict(),
    }
    if not config.execute:
        planned_results = []
        for arm_id in EXPECTED_ARM_IDS:
            resolved = resolved_by_arm[arm_id]
            protocol, study_digest = _arm_study_identity(
                resolved=resolved,
                data=data,
                runner=runner,
            )
            planned_results.append(
                {
                    "arm_id": arm_id,
                    "status": "planned",
                    "architecture": resolved.identity_dict(),
                    "study_protocol": protocol,
                    "study_identity_digest": study_digest,
                    "longest_batch": {
                        **selected.identity_dict(),
                        "passed": None,
                        "precision": PROBE_PRECISION,
                    },
                    "throughput": {
                        "optimizer_steps": FEASIBILITY_STEPS,
                        "status": "not-run",
                    },
                    "gpu": None,
                }
            )
        return build_probe_manifest(
            status="dry-run",
            started_at_utc=started_at,
            source_code={
                **source_code,
                "probe_runner": {
                    "path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
            },
            source_data=source_data,
            retained_training_records=len(data.train_rows),
            arm_results=planned_results,
        )

    configure_deterministic_cuda_environment()
    import torch
    from transformers import PreTrainedTokenizerFast

    if not torch.cuda.is_available():
        raise RuntimeError("Actual feasibility probes require a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Feasibility probes require exactly one visible CUDA GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("Feasibility probes must not run under distributed training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Actual feasibility probes require bf16 support")
    optimization_plan = _load_optimization_plan(config.optimization_plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-",
            dir=destination.parent,
        )
    )
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.tokenizer_dir,
            local_files_only=True,
        )
        gpu = _gpu_identity(torch)
        work_root = temporary / "work"
        work_root.mkdir()
        arm_results = [
            _execute_arm(
                torch=torch,
                runner=runner,
                resolved=resolved_by_arm[arm_id],
                data=data,
                tokenizer=tokenizer,
                gpu=gpu,
                work_root=work_root,
            )
            for arm_id in EXPECTED_ARM_IDS
        ]
        shutil.rmtree(work_root)
        _require_unchanged_source(
            repo_root,
            source_code,
            ignored_paths=(destination, temporary),
        )
        manifest = build_probe_manifest(
            status="complete",
            started_at_utc=started_at,
            completed_at_utc=datetime.now(UTC).isoformat(),
            source_code={
                **source_code,
                "probe_runner": {
                    "path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
            },
            source_data=source_data,
            retained_training_records=len(data.train_rows),
            arm_results=arm_results,
            optimization_plan=optimization_plan,
        )
        validate_probe_manifest(manifest)
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        if load_probe_manifest(temporary / MANIFEST_FILENAME) != manifest:
            raise RuntimeError("Serialized feasibility manifest changed on reload")
        if destination.exists():
            raise FileExistsError(
                f"Feasibility output appeared during probing: {destination}"
            )
        temporary.rename(destination)
        return load_probe_manifest(destination / MANIFEST_FILENAME)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument(
        "--reference-corpus-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Pinned corpus whose roots/tasks V2 excludes; repeat for every "
            "target-contract reference (defaults to deep-corpus-v1)"
        ),
    )
    parser.add_argument("--optimization-plan", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"run every longest-batch proof and exact {FEASIBILITY_STEPS}-step probe",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    result = run_probe(
        FeasibilityProbeConfig(
            output_dir=arguments.output_dir,
            corpus_dir=arguments.corpus_dir,
            tokenizer_dir=arguments.tokenizer_dir,
            reference_corpus_dirs=tuple(
                arguments.reference_corpus_dir or (DEFAULT_REFERENCE_CORPUS_DIR,)
            ),
            execute=arguments.execute,
            optimization_plan=arguments.optimization_plan,
        )
    )
    printable = {
        "status": result["status"],
        "artifact_digest": result["artifact_digest"],
        "output_dir": (
            str(arguments.output_dir.resolve()) if arguments.execute else None
        ),
        "launch_gate": result["launch_gate"],
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
