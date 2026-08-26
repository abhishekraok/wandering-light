"""Pure contract tests for issue #54's feasibility gate."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments import probe_deep_corpus_v2_scratch as probe
from wandering_light.training.deep_corpus_scratch import (
    FEASIBILITY_HARD_LIMIT_HOURS,
    FEASIBILITY_SOFT_LIMIT_HOURS,
    FEASIBILITY_STEPS,
    digest_json,
)


def _arm_result(arm_id: str, *, seconds_per_step: float = 1.0):
    physical = 24_000_000_000
    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    return {
        "arm_id": arm_id,
        "status": "passed",
        "study_identity_digest": digest_json([arm_id, "study"]),
        "gpu": {
            "identity_digest": digest_json("one-gpu"),
            "uuid": gpu_uuid,
            "physical_vram_bytes": physical,
        },
        "process_gpu_memory": {
            "source": "nvidia-smi compute-apps used_gpu_memory polling",
            "gpu_uuid": gpu_uuid,
            "pid": 123,
            "samples": 10,
            "peak_process_used_bytes": 18_000_000_000,
        },
        "longest_batch": {
            "passed": True,
            "batch_size": 8,
            "precision": "bf16",
            "peak_torch_allocated_bytes": 10_000_000_000,
            "peak_torch_reserved_bytes": 12_000_000_000,
        },
        "throughput": {
            "passed": True,
            "optimizer_steps": FEASIBILITY_STEPS,
            "elapsed_seconds": FEASIBILITY_STEPS * seconds_per_step,
            "peak_torch_allocated_bytes": 14_000_000_000,
            "peak_torch_reserved_bytes": 16_000_000_000,
            "protocol_overhead": {
                "passed": True,
                "full_validation_evaluations_measured": 1,
                "validation_elapsed_seconds": 1.0,
                "weights_only_checkpoint_writes_measured": 1,
                "checkpoint_elapsed_seconds": 1.0,
                "setup_wandb_and_best_reload_factor": (
                    probe.PROJECTION_SERVICE_OVERHEAD_FACTOR
                ),
            },
        },
    }


def _all_results(*, seconds_per_step: float = 1.0):
    return [
        _arm_result(arm_id, seconds_per_step=seconds_per_step)
        for arm_id in probe.EXPECTED_ARM_IDS
    ]


def _source_data():
    references = [
        {
            "manifest_digest": "sha256:" + "3" * 64,
            "root_digest_set_sha256": "sha256:" + "4" * 64,
            "task_id_set_sha256": "sha256:" + "5" * 64,
        }
    ]
    return {
        "corpus_manifest_digest": "sha256:" + "1" * 64,
        "tokenizer_artifact_digest": "sha256:" + "2" * 64,
        "reference_corpora": references,
        "reference_corpora_digest": digest_json(references),
    }


def _optimization_plan(results, *, seconds_per_step: float = 1.0, **overrides):
    measurements = []
    by_arm = {result["arm_id"]: result for result in results}
    for arm_id in probe.EXPECTED_ARM_IDS:
        optimized = _arm_result(arm_id, seconds_per_step=seconds_per_step)
        optimized["study_identity_digest"] = by_arm[arm_id]["study_identity_digest"]
        optimized["gpu"] = dict(by_arm[arm_id]["gpu"])
        measurements.append(optimized)
    value = {
        "schema_version": probe.SCHEMA_VERSION,
        "artifact_type": probe.OPTIMIZATION_ARTIFACT_TYPE,
        "status": "validated",
        "validation": {
            "preserves_frozen_study_protocol": True,
            "runtime_monitoring_validated": True,
            "hard_stop_hours": FEASIBILITY_HARD_LIMIT_HOURS,
        },
        "arm_measurements": measurements,
    }
    value.update(overrides)
    value["artifact_digest"] = digest_json(value)
    return value


def test_step_projection_covers_every_lr_pilot_and_final_epoch_cap():
    plan = probe.projected_step_plan(8_001)

    assert plan["steps_per_epoch"] == 1_001
    assert plan["pilot_runs_per_arm"] == 3
    assert plan["pilot_steps_per_run"] == 251
    assert plan["pilot_steps_per_arm"] == 753
    assert plan["final_steps_per_arm"] == 3_003
    assert plan["pilot_plus_final_steps_per_arm"] == 3_756
    with pytest.raises(ValueError, match="frozen at 8"):
        probe.projected_step_plan(8_001, batch_size=4)

    projection = probe.project_arm_runtime(_arm_result("scratch-10m"), plan)
    assert projection["pilot_runs"] == 3
    assert projection["projected_pilot_validations_per_run"] == 2
    assert projection["projected_pilot_checkpoint_writes_per_run"] == 2
    assert projection["projected_final_validations"] == 4
    assert projection["projected_final_checkpoint_writes"] == 4
    assert projection["setup_wandb_and_best_reload_factor"] == 1.15


def test_longest_batch_is_exact_stable_and_content_addressed():
    rows = tuple(
        {
            "task_id": f"task-{index}",
            "split": "discovery",
            "certified_distance": index % 8 + 1,
            "prompt": f"prompt-{index}",
            "completion": "inc",
        }
        for index in range(12)
    )
    lengths = [10, 90, 30, 80, 70, 60, 50, 40, 100, 90, 20, 5]
    audit = {
        "retained": [
            {
                "task_id": row["task_id"],
                "combined_tokens": length,
            }
            for row, length in zip(rows, lengths, strict=True)
        ]
    }

    selected = probe.select_longest_retained_batch(rows, audit)

    assert selected.task_ids == (
        "task-8",
        "task-1",
        "task-9",
        "task-3",
        "task-4",
        "task-5",
        "task-6",
        "task-7",
    )
    assert selected.combined_token_lengths == (100, 90, 90, 80, 70, 60, 50, 40)
    assert selected.task_ids_digest == digest_json(list(selected.task_ids))


def test_direct_gate_is_per_arm_and_aggregate_is_diagnostic_only():
    # Each arm is about 5.2 hours, while the diagnostic three-arm sum is above
    # 14 hours. All arms remain directly authorized because #54 gates per arm.
    accepted = probe.evaluate_launch_gate(
        _all_results(seconds_per_step=5.0),
        retained_training_records=8_000,
    )
    assert accepted["launch_allowed"]
    assert accepted["all_arms_authorized"]
    assert all(gate["route"] == "direct" for gate in accepted["per_arm"].values())
    assert accepted["diagnostic_totals"]["baseline_projected_all_arms_hours"] > 14

    incomplete = probe.evaluate_launch_gate(
        _all_results()[:-1],
        retained_training_records=8_000,
    )
    assert not incomplete["launch_allowed"]
    assert "exactly all arms" in incomplete["reason"]

    failed = _all_results()
    failed[0]["longest_batch"]["passed"] = False
    rejected = probe.evaluate_launch_gate(
        failed,
        retained_training_records=8_000,
    )
    assert not rejected["launch_allowed"]
    assert "batch-8 proof" in rejected["per_arm"][probe.EXPECTED_ARM_IDS[0]]["reason"]


def test_optimization_route_is_fail_closed_between_14_and_16_hours():
    # 8,000 records gives 3,750 projected steps per arm. At 14.4 seconds/step,
    # each arm is 15 hours: optimization is permitted but required per arm.
    baseline = _all_results(seconds_per_step=13.0)
    without_plan = probe.evaluate_launch_gate(
        baseline,
        retained_training_records=8_000,
    )
    assert not without_plan["launch_allowed"]
    assert all(
        14 < gate["baseline"]["projected_pilot_plus_final_hours"] < 16
        for gate in without_plan["per_arm"].values()
    )

    plan = _optimization_plan(baseline, seconds_per_step=10.0)
    accepted = probe.evaluate_launch_gate(
        baseline,
        retained_training_records=8_000,
        optimization_plan=plan,
    )
    assert accepted["launch_allowed"]
    assert all(
        gate["route"] == "validated-optimization-plan"
        and gate["optimization"]["projection"]["projected_pilot_plus_final_hours"] <= 14
        for gate in accepted["per_arm"].values()
    )
    assert accepted["runtime_hard_stop_required"]

    tampered = dict(plan)
    tampered["validation"] = {
        **plan["validation"],
        "runtime_monitoring_validated": False,
    }
    tampered["artifact_digest"] = digest_json(
        {key: value for key, value in tampered.items() if key != "artifact_digest"}
    )
    rejected = probe.evaluate_launch_gate(
        baseline,
        retained_training_records=8_000,
        optimization_plan=tampered,
    )
    assert not rejected["launch_allowed"]
    assert all(
        "runtime_monitoring_validated" in gate["reason"]
        for gate in rejected["per_arm"].values()
    )


def test_optimization_plan_may_validate_only_the_slow_arm():
    baseline = _all_results(seconds_per_step=1.0)
    slow_arm = probe.EXPECTED_ARM_IDS[0]
    baseline[0]["throughput"]["elapsed_seconds"] = FEASIBILITY_STEPS * 13.0
    plan = _optimization_plan(baseline, seconds_per_step=10.0)
    plan["arm_measurements"] = [plan["arm_measurements"][0]]
    plan.pop("artifact_digest")
    plan["artifact_digest"] = digest_json(plan)

    gate = probe.evaluate_launch_gate(
        baseline,
        retained_training_records=8_000,
        optimization_plan=plan,
    )

    assert gate["all_arms_authorized"]
    assert gate["per_arm"][slow_arm]["route"] == "validated-optimization-plan"
    assert all(
        gate["per_arm"][arm_id]["route"] == "direct"
        for arm_id in probe.EXPECTED_ARM_IDS[1:]
    )


def test_sixteen_hour_hard_limit_cannot_be_overridden_by_optimization():
    baseline = _all_results(seconds_per_step=16.0)
    plan = _optimization_plan(baseline, seconds_per_step=1.0)

    gate = probe.evaluate_launch_gate(
        baseline,
        retained_training_records=8_000,
        optimization_plan=plan,
    )

    assert not gate["launch_allowed"]
    assert all(
        arm_gate["baseline"]["projected_pilot_plus_final_hours"] > 16
        and "16-hour hard limit" in arm_gate["reason"]
        for arm_gate in gate["per_arm"].values()
    )


def test_manifest_is_canonical_tamper_evident_and_dry_run_never_authorizes():
    manifest = probe.build_probe_manifest(
        status="complete",
        started_at_utc="2026-08-26T00:00:00+00:00",
        completed_at_utc="2026-08-26T00:10:00+00:00",
        source_code={"commit": "a" * 40, "dirty": False},
        source_data=_source_data(),
        retained_training_records=8_000,
        arm_results=_all_results(),
    )

    assert (
        probe.validate_probe_manifest(
            manifest,
            require_launch_allowed=True,
        )
        == manifest
    )
    payload = dict(manifest)
    stored = payload.pop("artifact_digest")
    assert stored == digest_json(payload)

    tampered = json.loads(json.dumps(manifest))
    tampered["arm_results"][0]["throughput"]["elapsed_seconds"] += 1
    with pytest.raises(ValueError, match="Content digest mismatch"):
        probe.validate_probe_manifest(tampered)

    # Recomputing the outer digest is not enough to forge an authorization: the
    # consumer independently derives the launch decision from the measurements.
    stale_gate = json.loads(json.dumps(manifest))
    stale_gate["arm_results"][0]["throughput"]["elapsed_seconds"] *= 100
    stale_gate.pop("artifact_digest")
    stale_gate["artifact_digest"] = digest_json(stale_gate)
    with pytest.raises(ValueError, match="does not match its measurements"):
        probe.validate_probe_manifest(stale_gate)

    dry_run = probe.build_probe_manifest(
        status="dry-run",
        started_at_utc="2026-08-26T00:00:00+00:00",
        source_code={},
        source_data=_source_data(),
        retained_training_records=8_000,
        arm_results=[],
    )
    with pytest.raises(ValueError, match="does not authorize"):
        probe.validate_probe_manifest(dry_run, require_launch_allowed=True)
    assert dry_run["launch_gate"]["soft_limit_hours"] == (FEASIBILITY_SOFT_LIMIT_HOURS)


def test_manifest_consumer_authorizes_a_named_arm_independently(tmp_path):
    results = _all_results(seconds_per_step=1.0)
    blocked_arm = probe.EXPECTED_ARM_IDS[-1]
    results[-1]["throughput"]["elapsed_seconds"] = FEASIBILITY_STEPS * 16.0
    manifest = probe.build_probe_manifest(
        status="complete",
        started_at_utc="2026-08-26T00:00:00+00:00",
        completed_at_utc="2026-08-26T00:10:00+00:00",
        source_code={"commit": "a" * 40, "dirty": False},
        source_data=_source_data(),
        retained_training_records=8_000,
        arm_results=results,
    )
    path = tmp_path / probe.MANIFEST_FILENAME
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert not manifest["launch_gate"]["all_arms_authorized"]
    assert (
        probe.load_probe_manifest(
            path,
            require_launch_allowed=True,
            arm_id=probe.EXPECTED_ARM_IDS[0],
        )
        == manifest
    )
    with pytest.raises(ValueError, match=blocked_arm):
        probe.load_probe_manifest(
            path,
            require_launch_allowed=True,
            arm_id=blocked_arm,
        )
    with pytest.raises(ValueError, match="all arms"):
        probe.load_probe_manifest(path, require_launch_allowed=True)


def test_execute_requires_fresh_output_but_dry_run_does_not(tmp_path):
    output = tmp_path / "probe"
    output.mkdir()

    probe._validate_config(probe.FeasibilityProbeConfig(output_dir=output))
    with pytest.raises(FileExistsError, match="already exists"):
        probe._validate_config(
            probe.FeasibilityProbeConfig(output_dir=output, execute=True)
        )


def test_reference_corpora_are_repeatable_and_content_bound(tmp_path):
    arguments = probe._parse_args(
        [
            "--output-dir",
            str(tmp_path / "probe"),
            "--reference-corpus-dir",
            str(tmp_path / "v1"),
            "--reference-corpus-dir",
            str(tmp_path / "older"),
        ]
    )
    assert arguments.reference_corpus_dir == [tmp_path / "v1", tmp_path / "older"]

    manifest = probe.build_probe_manifest(
        status="complete",
        started_at_utc="2026-08-26T00:00:00+00:00",
        completed_at_utc="2026-08-26T00:10:00+00:00",
        source_code={"commit": "a" * 40, "dirty": False},
        source_data=_source_data(),
        retained_training_records=8_000,
        arm_results=_all_results(),
    )
    changed = json.loads(json.dumps(manifest))
    changed["source_data"]["reference_corpora"][0]["manifest_digest"] = (
        "sha256:" + "9" * 64
    )
    changed.pop("artifact_digest")
    changed["artifact_digest"] = digest_json(changed)
    with pytest.raises(ValueError, match="reference-corpus identity digest"):
        probe.validate_probe_manifest(changed)


def test_process_gpu_memory_sampler_is_uuid_pid_scoped_and_fail_closed():
    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    outputs = iter(
        [
            f"999, {gpu_uuid}, 100\n123, {gpu_uuid}, 1200\n",
            f"123, {gpu_uuid}, 1400\n",
            f"123, {gpu_uuid}, 1300\n",
        ]
    )

    def command_runner(*_args, **_kwargs):
        return SimpleNamespace(stdout=next(outputs))

    sampler = probe.NvidiaSmiProcessMemorySampler(
        gpu_uuid=gpu_uuid,
        pid=123,
        poll_interval_seconds=60,
        command_runner=command_runner,
    )
    sampler.start()
    sampler.sample_now()
    measurement = sampler.stop()
    assert measurement["peak_process_used_bytes"] == 1400 * 1024 * 1024
    assert measurement["samples"] == 3

    assert (
        probe.parse_nvidia_smi_process_memory(
            f"999, {gpu_uuid}, 100\n",
            pid=123,
            gpu_uuid=gpu_uuid,
        )
        is None
    )
    missing = probe.NvidiaSmiProcessMemorySampler(
        gpu_uuid=gpu_uuid,
        pid=123,
        poll_interval_seconds=60,
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            stdout=f"999, {gpu_uuid}, 100\n"
        ),
    )
    missing.start()
    with pytest.raises(RuntimeError, match="never observed"):
        missing.stop()
