"""Focused runner tests for the deep-corpus-v2 scratch study."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import probe_deep_corpus_v2_scratch as probe
from experiments import train_deep_corpus_v2_scratch as runner
from wandering_light.training import deep_corpus_scratch as scratch


def _config(tmp_path, **overrides):
    values = {
        "output_dir": tmp_path / "artifact",
        "corpus_dir": tmp_path / "corpus",
        "tokenizer_dir": tmp_path / "tokenizer",
        "arm_id": "scratch-10m",
        "phase": "pilot",
        "learning_rate": 1e-4,
        "num_train_epochs": 0.25,
        "dry_run": True,
    }
    values.update(overrides)
    return runner.ScratchRunConfig(**values)


def test_run_contract_rejects_distribution_and_seed_drift(tmp_path):
    runner._validate_run_config(_config(tmp_path))
    with pytest.raises(ValueError, match="quarter epoch"):
        runner._validate_run_config(_config(tmp_path, num_train_epochs=1))
    with pytest.raises(ValueError, match="learning rate"):
        runner._validate_run_config(_config(tmp_path, learning_rate=2e-5))
    with pytest.raises(ValueError, match="initialization seed"):
        runner._validate_run_config(_config(tmp_path, initialization_seed=9))
    with pytest.raises(ValueError, match="data-order seed"):
        runner._validate_run_config(_config(tmp_path, data_order_seed=9))
    with pytest.raises(ValueError, match="feasibility manifest"):
        runner._validate_run_config(
            _config(tmp_path, dry_run=False, wandb_run_name="actual-run")
        )


def test_data_preparation_requires_frozen_python_hash_seed(tmp_path, monkeypatch):
    def reject_seed():
        raise RuntimeError("fixed Python hash seed required")

    monkeypatch.setattr(
        runner,
        "require_deterministic_python_hash_seed",
        reject_seed,
    )
    with pytest.raises(RuntimeError, match="fixed Python hash seed required"):
        runner.prepare_scratch_data(_config(tmp_path))


def test_final_requires_selection_lock_and_at_most_three_epochs(tmp_path):
    with pytest.raises(ValueError, match="requires a validation-only LR lock"):
        runner._validate_run_config(
            _config(tmp_path, phase="final", num_train_epochs=3)
        )
    lock = tmp_path / "selection.json"
    runner._validate_run_config(
        _config(
            tmp_path,
            phase="final",
            num_train_epochs=3,
            selection_lock=lock,
        )
    )
    with pytest.raises(ValueError, match="at most"):
        runner._validate_run_config(
            _config(
                tmp_path,
                phase="final",
                num_train_epochs=3.1,
                selection_lock=lock,
            )
        )


def test_sft_protocol_discloses_required_360m_checkpointing(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert len(runner.GRADIENT_CHECKPOINTING_ARMS) == 1
    assert "smollm2-360m-scratch" in runner.GRADIENT_CHECKPOINTING_ARMS
    small = runner.build_sft_config(
        _config(tmp_path), report_to_wandb=False, precision_bf16=False
    )
    large = runner.build_sft_config(
        _config(tmp_path, arm_id="smollm2-360m-scratch"),
        report_to_wandb=False,
        precision_bf16=False,
    )
    assert small.per_device_train_batch_size == large.per_device_train_batch_size == 8
    assert small.gradient_accumulation_steps == large.gradient_accumulation_steps == 1
    assert small.max_length == large.max_length == 2_048
    assert small.completion_only_loss and large.completion_only_loss
    assert small.save_only_model and large.save_only_model
    assert small.load_best_model_at_end and large.load_best_model_at_end
    assert not small.gradient_checkpointing
    assert large.gradient_checkpointing
    assert large.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert runner._training_protocol(_config(tmp_path))["cublas_workspace_config"] == (
        scratch.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    )
    assert runner._training_protocol(_config(tmp_path))["pythonhashseed"] == "0"
    small_protocol = runner._training_protocol(_config(tmp_path))
    large_protocol = runner._training_protocol(
        _config(tmp_path, arm_id="smollm2-360m-scratch")
    )
    assert small_protocol["gradient_checkpointing_kwargs"] == {}
    assert large_protocol["gradient_checkpointing"]
    assert large_protocol["gradient_checkpointing_kwargs"] == {
        "use_reentrant": False
    }


def _pilot(rate, loss):
    return {
        "status": "complete",
        "phase": "pilot",
        "arm_id": "scratch-10m",
        "num_train_epochs": 0.25,
        "learning_rate": rate,
        "best_validation_loss": loss,
        "initial_state_digest": "sha256:" + "1" * 64,
        "initialization_seed": scratch.INITIALIZATION_SEED,
        "data_order_seed": scratch.DATA_ORDER_SEED,
        "tokenizer_digest": "sha256:" + "2" * 64,
        "retained_training_task_ids_digest": "sha256:" + "3" * 64,
        "effective_config_digest": "sha256:" + "4" * 64,
        "exact_trainable_parameters": 10_000_001,
    }


def test_selection_writer_uses_exact_three_pilots_and_refuses_overwrite(tmp_path):
    paths = []
    for index, (rate, loss) in enumerate(
        zip(scratch.LR_GRID, (0.9, 0.7, 0.8), strict=True)
    ):
        path = tmp_path / f"pilot-{index}.json"
        path.write_text(json.dumps(_pilot(rate, loss)), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "selection.json"
    lock = runner.write_learning_rate_lock("scratch-10m", paths, output)
    assert lock["selected_learning_rate"] == 3e-4
    with pytest.raises(FileExistsError):
        runner.write_learning_rate_lock("scratch-10m", paths, output)


def test_loss_curve_reports_local_gap_and_trend_inputs():
    curve = runner._loss_curve(
        [
            {"loss": 1.0, "step": 10, "epoch": 0.9},
            {"eval_loss": 1.1, "step": 11, "epoch": 1.0},
            {"loss": 0.8, "step": 20, "epoch": 1.9},
            {"eval_loss": 0.9, "step": 21, "epoch": 2.0},
        ]
    )
    assert curve[0]["validation_minus_train_gap"] == pytest.approx(0.1)
    assert curve[1]["validation_minus_train_gap"] == pytest.approx(0.1)


def test_manifest_records_config_only_initialization_and_all_exclusions(tmp_path):
    config = _config(tmp_path)
    spec = scratch.ARM_SPECS["scratch-10m"]
    resolved = scratch.ResolvedScratchConfig(
        spec=spec,
        config=SimpleNamespace(vocab_size=4_192, ffn_dim=2_240),
        upstream_config_digest="sha256:" + "a" * 64,
        effective_config_digest="sha256:" + "b" * 64,
        exact_trainable_parameters=10_079_872,
        parameter_error=79_872,
    )
    data = runner.PreparedScratchData(
        basis=SimpleNamespace(),
        corpus_manifest={"manifest_digest": "sha256:" + "c" * 64},
        tokenizer_manifest={"artifact_digest": "sha256:" + "d" * 64},
        train_rows=(),
        validation_rows=(),
        train_audit={
            "retained_task_ids_digest": "sha256:" + "e" * 64,
            "excluded": [{"task_id": "train-long"}],
            "statistics": {
                "retained": {
                    "overall": {
                        "combined": {"total": 100},
                        "completion_with_eos": {"total": 20},
                    }
                }
            },
        },
        validation_audit={"excluded": [{"task_id": "validation-long"}]},
        source_identity={},
    )
    manifest = runner._build_manifest(
        config=config,
        data=data,
        resolved=resolved,
        initialization={
            "creation_method": "AutoModelForCausalLM.from_config",
            "pretrained_weights_loaded": False,
            "initial_state_digest": "sha256:" + "f" * 64,
            "initialization_seed": scratch.INITIALIZATION_SEED,
        },
        source_code={"dirty": False},
        status="complete",
        started_at_utc="2026-08-26T00:00:00+00:00",
        trainer_state={
            "loss_curve": [
                {
                    "validation_loss": 1.0,
                    "validation_minus_train_gap": 0.1,
                }
            ]
        },
    )
    assert manifest["creation"]["pretrained_weights_loaded"] is False
    assert manifest["source_code"]["runner"]["path"] == (
        "experiments/train_deep_corpus_v2_scratch.py"
    )
    assert manifest["exact_trainable_parameters"] == 10_079_872
    assert manifest["workload"]["planned_combined_training_tokens"] == 25
    assert manifest["tokenization"]["discovery"]["excluded"][0]["task_id"] == (
        "train-long"
    )
    assert manifest["tokenization"]["validation"]["excluded"][0]["task_id"] == (
        "validation-long"
    )


def _direct_arm_gate():
    return {
        "launch_allowed": True,
        "route": "direct",
        "baseline": {
            "projected_pilot_run_hours": 1.0,
            "projected_pilot_hours": 3.0,
            "projected_final_hours": 5.0,
            "projected_pilot_plus_final_hours": 8.0,
        },
    }


def test_hard_stop_allocations_cover_one_arm_and_callback_raises():
    pilot = runner.hard_stop_allocation(_direct_arm_gate(), phase="pilot")
    final = runner.hard_stop_allocation(_direct_arm_gate(), phase="final")
    assert 3 * pilot["pilot_run_allocation_hours"] + final[
        "final_run_allocation_hours"
    ] == pytest.approx(16)
    assert pilot["current_run_allocation_hours"] == pytest.approx(2)
    assert final["current_run_allocation_hours"] == pytest.approx(10)
    assert "retries" in pilot["retry_accounting"]

    now = {"value": 100.0}
    deadline = runner.WallClockDeadline(
        started_monotonic=100.0,
        allocation_seconds=10.0,
        clock=lambda: now["value"],
    )
    callback = runner.make_wall_clock_hard_stop_callback(deadline)
    control = SimpleNamespace()
    assert callback.on_step_end(None, None, control) is control
    now["value"] = 110.0
    with pytest.raises(RuntimeError, match="hard-stop allocation"):
        callback.on_evaluate(None, None, control)

    invalid = _direct_arm_gate()
    invalid["baseline"] = {
        **invalid["baseline"],
        "projected_pilot_plus_final_hours": 14.1,
    }
    with pytest.raises(ValueError, match="cover three pilots"):
        runner.hard_stop_allocation(invalid, phase="pilot")


def test_probe_source_must_match_the_clean_training_commit():
    repo_root = Path(runner.__file__).resolve().parents[1]
    current = {"commit": "a" * 40, "dirty": False}
    manifest = {
        "source_code": {
            **current,
            "probe_runner": {
                "path": "experiments/probe_deep_corpus_v2_scratch.py",
                "sha256": runner._sha256_file(Path(probe.__file__)),
            },
        }
    }
    assert (
        runner._validate_probe_source_identity(
            manifest, current_source=current, repo_root=repo_root
        )["commit"]
        == current["commit"]
    )

    manifest["source_code"]["commit"] = "b" * 40
    with pytest.raises(ValueError, match="another Git commit"):
        runner._validate_probe_source_identity(
            manifest, current_source=current, repo_root=repo_root
        )
    manifest["source_code"]["commit"] = current["commit"]
    manifest["source_code"]["probe_runner"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="runner differs"):
        runner._validate_probe_source_identity(
            manifest, current_source=current, repo_root=repo_root
        )


def test_atomic_training_staging_publishes_only_complete_path(tmp_path):
    destination = tmp_path / "artifact"
    stale = tmp_path / ".artifact.staging-stale-prior-process"
    stale.mkdir()
    staging = runner._create_training_staging_dir(destination)
    assert staging != stale
    assert stale.resolve() in runner._existing_training_staging_dirs(destination)
    assert not destination.exists()
    (staging / "marker").write_text("complete", encoding="utf-8")

    runner._publish_training_staging(staging, destination)

    assert (destination / "marker").read_text(encoding="utf-8") == "complete"
    assert stale.is_dir()
    runner._cleanup_training_staging(staging, destination)
    with pytest.raises(FileExistsError, match="already exists"):
        runner._create_training_staging_dir(destination)


def test_failed_training_staging_is_cleaned_without_touching_stale(tmp_path):
    destination = tmp_path / "artifact"
    stale = tmp_path / ".artifact.staging-stale-prior-process"
    stale.mkdir()
    staging = runner._create_training_staging_dir(destination)
    (staging / "partial").write_text("partial", encoding="utf-8")

    runner._cleanup_training_staging(staging, destination)

    assert not staging.exists()
    assert not destination.exists()
    assert stale.exists()


def test_published_training_args_contain_no_deleted_staging_path(tmp_path):
    staging = (tmp_path / ".artifact.staging-123").resolve()
    destination = (tmp_path / "artifact").resolve()
    arguments = SimpleNamespace(
        to_dict=lambda: {
            "output_dir": str(staging),
            "logging_dir": str(staging / "runs"),
            "nested": [str(staging / "checkpoint")],
        }
    )

    published = runner._published_training_args(
        arguments, staging=staging, destination=destination
    )

    assert published["output_dir"] == str(destination)
    assert published["logging_dir"] == str(destination / "runs")
    assert str(staging) not in json.dumps(published)


def test_execution_environment_distinguishes_observed_peak_from_allocator():
    properties = SimpleNamespace(
        name="RTX 4090",
        total_memory=24_000_000_000,
        major=8,
        minor=9,
    )
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 999)),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda _index: properties,
            max_memory_allocated=lambda _index: 10_000,
            max_memory_reserved=lambda _index: 12_000,
        ),
    )
    telemetry = {
        "measurement_scope": "process",
        "peak_observed_used_bytes": 15_000,
        "peak_process_used_bytes": 15_000,
        "source": "nvidia-smi",
        "gpu_uuid": "GPU-test",
        "namespace_pid": 123,
        "samples": 4,
    }

    environment = runner._cuda_environment(
        fake_torch,
        gpu_memory_telemetry=telemetry,
    )

    assert environment["peak_observed_gpu_memory_bytes"] == 15_000
    assert environment["gpu_memory_telemetry"] == telemetry
    assert "process-attributed" in environment["memory_measurement_note"]
    assert environment["peak_torch_allocated_bytes"] == 10_000
    assert environment["physical_vram_bytes"] == 24_000_000_000

    physical_telemetry = {
        "measurement_scope": "physical_gpu_total",
        "peak_observed_used_bytes": 16_000,
        "peak_physical_gpu_used_bytes": 16_000,
        "baseline_observed_used_bytes": 2_000,
        "source": "nvidia-smi",
        "gpu_uuid": "GPU-test",
        "namespace_pid": 123,
        "samples": 4,
        "limitation": "compute-process rows unavailable in this namespace",
    }
    physical_environment = runner._cuda_environment(
        fake_torch,
        gpu_memory_telemetry=physical_telemetry,
    )
    assert physical_environment["peak_observed_gpu_memory_bytes"] == 16_000
    assert "conservative total usage" in physical_environment["memory_measurement_note"]
    assert "peak_process_gpu_memory_bytes" not in physical_environment


def test_training_and_probe_use_the_same_physical_gpu_identity_digest():
    properties = SimpleNamespace(
        name="RTX 4090",
        uuid="GPU-11111111-2222-3333-4444-555555555555",
        total_memory=24_000_000_000,
        major=8,
        minor=9,
    )
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 999)),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            device_count=lambda: 1,
            get_device_properties=lambda _index: properties,
        ),
    )

    assert (
        runner._current_gpu_probe_identity(fake_torch)["identity_digest"]
        == probe._gpu_identity(fake_torch)["identity_digest"]
    )
