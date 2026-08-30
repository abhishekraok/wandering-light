"""Focused runner tests for the deep-corpus-v2 scratch study."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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


def test_actual_run_requires_non_temporary_durable_output_root(tmp_path):
    actual = {
        "dry_run": False,
        "wandb_run_name": "actual-run",
        "feasibility_manifest": tmp_path / "gate.json",
    }
    with pytest.raises(ValueError, match="explicit durable output root"):
        runner._validate_run_config(_config(tmp_path, **actual))
    with pytest.raises(ValueError, match="canonical or configured transient tree"):
        runner._validate_run_config(
            _config(tmp_path, durable_output_root=tmp_path, **actual)
        )


def test_durable_output_root_requires_strict_containment(tmp_path, monkeypatch):
    os_temp = tmp_path / "simulated-os-temp"
    source_worktree = tmp_path / "simulated-worktree"
    durable_root = tmp_path / "durable"
    for path in (os_temp, source_worktree, durable_root):
        path.mkdir()
    monkeypatch.setattr(
        runner,
        "_CANONICAL_TRANSIENT_ROOTS",
        (tmp_path / "canonical-tmp", tmp_path / "canonical-var-tmp"),
    )

    contract = runner._durable_output_contract(
        durable_root / "arm-final",
        durable_root,
        configured_temp_root=os_temp,
        source_worktree=source_worktree,
    )
    assert contract["declared_output_root"] == str(durable_root.resolve())
    assert contract["artifact_relative_path"] == "arm-final"
    with pytest.raises(ValueError, match="child of the durable output root"):
        runner._durable_output_contract(
            tmp_path / "outside",
            durable_root,
            configured_temp_root=os_temp,
            source_worktree=source_worktree,
        )
    with pytest.raises(ValueError, match="canonical or configured transient tree"):
        runner._durable_output_contract(
            os_temp / "run",
            os_temp,
            configured_temp_root=os_temp,
            source_worktree=source_worktree,
        )
    worktree_artifacts = source_worktree / "artifacts"
    worktree_artifacts.mkdir()
    with pytest.raises(ValueError, match="outside the source worktree"):
        runner._durable_output_contract(
            worktree_artifacts / "run",
            worktree_artifacts,
            configured_temp_root=os_temp,
            source_worktree=source_worktree,
        )


@pytest.mark.parametrize("transient_root", [Path("/tmp"), Path("/var/tmp")])
def test_canonical_transient_roots_are_rejected_independent_of_configured_temp(
    tmp_path, transient_root
):
    configured_temp = tmp_path / "configured-temp-elsewhere"
    configured_temp.mkdir()

    with pytest.raises(ValueError, match="canonical or configured transient tree"):
        runner._durable_output_contract(
            transient_root / "wandering-light-test-artifact",
            transient_root,
            configured_temp_root=configured_temp,
            source_worktree=tmp_path / "unrelated-worktree",
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


def test_final_requires_selection_lock_and_exactly_three_epochs(tmp_path):
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
    for epochs in (2.9, 3.1):
        with pytest.raises(ValueError, match="exactly 3 epochs"):
            runner._validate_run_config(
                _config(
                    tmp_path,
                    phase="final",
                    num_train_epochs=epochs,
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
    assert large_protocol["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


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


def test_atomic_training_publish_never_replaces_an_existing_destination(tmp_path):
    destination = tmp_path / "artifact"
    staging = runner._create_training_staging_dir(destination)
    (staging / "marker").write_text("new", encoding="utf-8")
    destination.mkdir()
    (destination / "marker").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_training_staging(staging, destination)

    assert (destination / "marker").read_text(encoding="utf-8") == "existing"
    assert (staging / "marker").read_text(encoding="utf-8") == "new"


def test_atomic_training_publish_fails_closed_without_renameat2(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(RuntimeError, match="renameat2"):
        runner._atomic_rename_noreplace(
            source,
            tmp_path / "destination",
            libc=SimpleNamespace(),
        )


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


def _durable_checkpoint_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_CANONICAL_TRANSIENT_ROOTS",
        (tmp_path / "canonical-tmp", tmp_path / "canonical-var-tmp"),
    )
    monkeypatch.setattr(
        runner.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "simulated-os-temp"),
    )
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    destination = durable_root / "artifact"
    staging = runner._create_training_staging_dir(destination)
    for name in runner._DURABLE_ARTIFACT_REQUIRED_FILES - {
        "wandb_run.url",
        runner.TOKENIZER_MANIFEST_FILENAME,
    }:
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test payload for {name}".encode())
    tokenizer_payload = {
        "schema_version": 1,
        "artifact_type": "wandering-light-deep-corpus-v2-tokenizer",
        "tokenizer": {
            "total_vocab_size": 4,
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": runner._sha256_file(staging / name),
                }
                for name in runner.TOKENIZER_FILENAMES
            },
            "special_tokens": [
                {"role": role, "content": content, "id": token_id}
                for token_id, (role, content) in enumerate(runner.FIXED_SPECIAL_TOKENS)
            ],
        },
    }
    tokenizer_payload["artifact_digest"] = scratch.digest_json(tokenizer_payload)
    (staging / runner.TOKENIZER_MANIFEST_FILENAME).write_text(
        json.dumps(tokenizer_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "model.safetensors").write_bytes(b"weights")
    wandb_url = (
        "https://wandb.ai/abhishekraok-na/"
        "wandering-light-deep-corpus-v2-scratch/runs/durable-test"
    )
    (staging / "wandb_run.url").write_text(
        f"[InternetShortcut]\nURL={wandb_url}\n",
        encoding="utf-8",
    )
    spec = scratch.ARM_SPECS["scratch-10m"]
    resolved = scratch.ResolvedScratchConfig(
        spec=spec,
        config=SimpleNamespace(
            model_type="opt",
            vocab_size=4,
            ffn_dim=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        ),
        upstream_config_digest="sha256:" + "a" * 64,
        effective_config_digest="sha256:" + "b" * 64,
        exact_trainable_parameters=11,
        parameter_error=1,
    )
    manifest = {
        "schema_version": 1,
        "status": runner._PENDING_DURABILITY_STATUS,
        "phase": "final",
        "arm_id": "scratch-10m",
        "architecture": resolved.identity_dict(),
        "exact_trainable_parameters": 11,
        "tokenizer_digest": tokenizer_payload["artifact_digest"],
        "execution_environment": {
            "wall_clock_hard_stop": {
                "elapsed_before_artifact_write_seconds": 12.5,
            }
        },
        "durability_contract": runner._durable_output_contract(
            destination,
            durable_root,
        ),
        "wandb": {
            "entity": scratch.PERSONAL_WANDB_ENTITY,
            "project": scratch.DEFAULT_WANDB_PROJECT,
            "url": wandb_url,
        },
        "checkpoint_identity": runner._checkpoint_content_identity(staging),
    }
    manifest = runner._seal_training_manifest(manifest)
    (staging / runner.TRAINING_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner._publish_training_staging(staging, destination)
    return destination, staging, durable_root, manifest, wandb_url


def _durable_reload_fakes():
    calls = []

    class Parameter:
        requires_grad = True

        @staticmethod
        def numel():
            return 11

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return SimpleNamespace(
                config=SimpleNamespace(
                    model_type="opt",
                    vocab_size=4,
                    ffn_dim=64,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                ),
                parameters=lambda: [Parameter()],
            )

    class LoadedTokenizer:
        pad_token = "<|pad|>"
        bos_token = "<|bos|>"
        eos_token = "<|eos|>"
        unk_token = "<|unk|>"
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3

        def __len__(self):
            return 4

    class TokenizerFactory:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return LoadedTokenizer()

    return ModelFactory, TokenizerFactory, calls


def test_completed_run_requires_post_publish_content_and_reload_gate(
    tmp_path, monkeypatch
):
    destination, staging, durable_root, pending, _wandb_url = (
        _durable_checkpoint_fixture(tmp_path, monkeypatch)
    )
    model_factory, tokenizer_factory, calls = _durable_reload_fakes()

    prepared = runner._prepare_training_artifact_completion(
        destination,
        transient_staging=staging,
        durable_output_root=durable_root,
        completion_path="training",
        pending_manifest=pending,
        auto_model_class=model_factory,
        tokenizer_class=tokenizer_factory,
    )
    assert "completed_at_utc" not in prepared
    assert "full_invocation_elapsed_seconds" not in prepared["execution_environment"]
    complete = runner._finalize_training_artifact_completion(
        destination,
        prepared,
        completed_at_utc="2026-08-30T12:00:00+00:00",
        runtime_provenance=runner._normal_runtime_provenance(20.0),
    )

    assert complete["status"] == "complete"
    assert complete["durability_verification"]["status"] == "verified"
    assert complete["completed_at_utc"] == "2026-08-30T12:00:00+00:00"
    assert complete["execution_environment"]["full_invocation_elapsed_seconds"] == 20
    assert (
        complete["execution_environment"]["full_invocation_runtime_scope"]
        == runner._NORMAL_RUNTIME_SCOPE
    )
    assert "full_invocation_runtime_censoring" not in complete["execution_environment"]
    assert complete["durability_verification"]["wandb_finalization"] == {
        "summary_write_performed": True,
        "finish_performed": True,
        "completed_before_manifest_transition": True,
    }
    assert (
        complete["durability_verification"]["content_digest"]
        == complete["checkpoint_identity"]["content_digest"]
    )
    assert [kind for kind, _path, _kwargs in calls] == ["model", "tokenizer"]
    assert all(kwargs == {"local_files_only": True} for _kind, _path, kwargs in calls)
    assert not staging.exists()
    recorded_pending = json.loads(
        (destination / runner.TRAINING_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert recorded_pending["status"] == runner._PENDING_DURABILITY_STATUS
    completion_lock = runner._acquire_checkpoint_completion_lock(destination)
    try:
        runner._write_completed_training_manifest(
            destination,
            complete,
            completion_lock=completion_lock,
        )
    finally:
        runner._release_checkpoint_completion_lock(completion_lock)
    assert runner._validate_completed_training_artifact(destination) == complete

    (destination / "model.safetensors").write_bytes(b"tampered weights")
    with pytest.raises(RuntimeError, match="content identity mismatch"):
        runner._validate_completed_training_artifact(destination)


def test_reload_failure_leaves_published_checkpoint_incomplete(tmp_path, monkeypatch):
    destination, staging, durable_root, pending, _wandb_url = (
        _durable_checkpoint_fixture(tmp_path, monkeypatch)
    )

    class RejectReload:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            raise RuntimeError("reload failed")

    with pytest.raises(RuntimeError, match="reload failed"):
        runner._prepare_training_artifact_completion(
            destination,
            transient_staging=staging,
            durable_output_root=durable_root,
            completion_path="training",
            pending_manifest=pending,
            auto_model_class=RejectReload,
            tokenizer_class=RejectReload,
        )

    recorded = json.loads(
        (destination / runner.TRAINING_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert recorded["status"] == runner._PENDING_DURABILITY_STATUS
    with pytest.raises(RuntimeError, match="has not passed durable verification"):
        runner._validate_completed_training_artifact(destination)


def test_recovery_locally_completes_a_sealed_pending_checkpoint(tmp_path, monkeypatch):
    destination, _staging, durable_root, _pending, _wandb_url = (
        _durable_checkpoint_fixture(tmp_path, monkeypatch)
    )
    model_factory, tokenizer_factory, calls = _durable_reload_fakes()
    clock_values = iter((100.0, 104.25))

    complete = runner.recover_pending_training_artifact(
        destination,
        durable_root,
        auto_model_class=model_factory,
        tokenizer_class=tokenizer_factory,
        monotonic_clock=lambda: next(clock_values),
        utc_now=lambda: datetime(2026, 8, 30, 13, 0, tzinfo=UTC),
    )

    assert complete["status"] == "complete"
    assert complete["durability_verification"]["completion_path"] == "recovery"
    assert complete["durability_verification"]["wandb_finalization"] == {
        "summary_write_performed": False,
        "finish_performed": False,
        "reason": runner._RECOVERY_WANDB_REASON,
    }
    environment = complete["execution_environment"]
    assert environment["full_invocation_elapsed_seconds"] == 16.75
    assert environment["full_invocation_runtime_scope"] == (
        runner._RECOVERY_RUNTIME_SCOPE
    )
    assert environment["full_invocation_runtime_censoring"] == {
        "status": "right-censored-lower-bound",
        "known_original_lower_bound_seconds": 12.5,
        "recovery_elapsed_seconds": 4.25,
        "combined_lower_bound_seconds": 16.75,
        "omitted_interval_seconds": None,
        "omitted_interval_scope": runner._RECOVERY_OMITTED_INTERVAL_SCOPE,
        "omitted_interval_reason": runner._RECOVERY_OMITTED_INTERVAL_REASON,
    }
    assert all(kwargs == {"local_files_only": True} for _kind, _path, kwargs in calls)
    assert runner._validate_completed_training_artifact(destination) == complete


def test_recovery_refuses_a_live_producer_completion_lock(tmp_path, monkeypatch):
    destination, _staging, durable_root, _pending, _wandb_url = (
        _durable_checkpoint_fixture(tmp_path, monkeypatch)
    )
    model_factory, tokenizer_factory, calls = _durable_reload_fakes()
    producer_lock = runner._acquire_checkpoint_completion_lock(destination)
    try:
        with pytest.raises(RuntimeError, match="completion is already in progress"):
            runner.recover_pending_training_artifact(
                destination,
                durable_root,
                auto_model_class=model_factory,
                tokenizer_class=tokenizer_factory,
            )
    finally:
        runner._release_checkpoint_completion_lock(producer_lock)

    assert calls == []
    recorded = json.loads(
        (destination / runner.TRAINING_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert recorded["status"] == runner._PENDING_DURABILITY_STATUS


def test_second_recovery_cannot_overwrite_a_completed_manifest(tmp_path, monkeypatch):
    destination, _staging, durable_root, _pending, _wandb_url = (
        _durable_checkpoint_fixture(tmp_path, monkeypatch)
    )
    model_factory, tokenizer_factory, _calls = _durable_reload_fakes()
    first_clock = iter((10.0, 11.0))
    runner.recover_pending_training_artifact(
        destination,
        durable_root,
        auto_model_class=model_factory,
        tokenizer_class=tokenizer_factory,
        monotonic_clock=lambda: next(first_clock),
        utc_now=lambda: datetime(2026, 8, 30, 13, 0, tzinfo=UTC),
    )
    manifest_path = destination / runner.TRAINING_MANIFEST_FILENAME
    first_bytes = manifest_path.read_bytes()

    with pytest.raises(RuntimeError, match="marked complete"):
        runner.recover_pending_training_artifact(
            destination,
            durable_root,
            auto_model_class=model_factory,
            tokenizer_class=tokenizer_factory,
            monotonic_clock=lambda: 20.0,
            utc_now=lambda: datetime(2026, 8, 30, 14, 0, tzinfo=UTC),
        )

    assert manifest_path.read_bytes() == first_bytes


@pytest.mark.parametrize(
    "invalid_field",
    [
        "local_files_only",
        "budget_dimension",
        "budget_dimension_value",
        "special_token_ids",
        "special_token_contents",
    ],
)
def test_completed_acceptance_validates_full_reload_receipt(
    tmp_path, monkeypatch, invalid_field
):
    destination, staging, durable_root, pending, _wandb_url = (
        _durable_checkpoint_fixture(tmp_path, monkeypatch)
    )
    model_factory, tokenizer_factory, _calls = _durable_reload_fakes()
    prepared = runner._prepare_training_artifact_completion(
        destination,
        transient_staging=staging,
        durable_output_root=durable_root,
        completion_path="training",
        pending_manifest=pending,
        auto_model_class=model_factory,
        tokenizer_class=tokenizer_factory,
    )
    complete = runner._finalize_training_artifact_completion(
        destination,
        prepared,
        completed_at_utc="2026-08-30T12:00:00+00:00",
        runtime_provenance=runner._normal_runtime_provenance(20.0),
    )
    tampered = json.loads(json.dumps(complete))
    reload_receipt = tampered["durability_verification"]["reload"]
    if invalid_field == "local_files_only":
        reload_receipt[invalid_field] = False
    elif invalid_field in {"budget_dimension", "budget_dimension_value"}:
        reload_receipt["model"][invalid_field] = "wrong"
    elif invalid_field == "special_token_ids":
        reload_receipt["tokenizer"][invalid_field]["bos_token"] = 3
    else:
        reload_receipt["tokenizer"][invalid_field]["bos_token"] = "wrong"
    tampered = runner._seal_training_manifest(tampered)
    completion_lock = runner._acquire_checkpoint_completion_lock(destination)
    try:
        runner._write_completed_training_manifest(
            destination,
            tampered,
            completion_lock=completion_lock,
        )
    finally:
        runner._release_checkpoint_completion_lock(completion_lock)

    with pytest.raises(RuntimeError, match="receipt is inconsistent"):
        runner._validate_completed_training_artifact(destination)


def test_recover_cli_requires_checkpoint_and_durable_root(tmp_path):
    arguments = runner._parse_args(
        [
            "recover",
            "--checkpoint-dir",
            str(tmp_path / "artifact"),
            "--durable-output-root",
            str(tmp_path / "durable"),
        ]
    )

    assert arguments.mode == "recover"
    assert arguments.checkpoint_dir == tmp_path / "artifact"
    assert arguments.durable_output_root == tmp_path / "durable"


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
