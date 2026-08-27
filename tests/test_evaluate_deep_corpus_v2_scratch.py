from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments import evaluate_deep_corpus_v2_scratch as evaluate
from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.evals.deep_corpus_protocol import (
    ARM_IDS,
    SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    begin_test_opening,
    build_selection_lock,
    build_source_amendment,
    complete_test_opening,
    create_test_opening_ledger_file,
    load_test_opening_ledger,
    new_test_opening_ledger,
    record_test_arm_result,
)
from wandering_light.training.deep_corpus_scratch import ARM_SPECS, digest_json
from wandering_light.typed_list import TypedList


def test_candidate_generation_requires_frozen_python_hash_seed(tmp_path, monkeypatch):
    def reject_seed():
        raise RuntimeError("fixed Python hash seed required")

    monkeypatch.setattr(evaluate, "require_deterministic_python_hash_seed", reject_seed)
    with pytest.raises(RuntimeError, match="fixed Python hash seed required"):
        evaluate.generate_scratch_candidates(
            (),
            arm_id=ARM_IDS[0],
            checkpoint=tmp_path / "checkpoint",
            tokenizer_dir=tmp_path / "tokenizer",
            available_functions=None,
            generation={},
            progress=False,
        )


def test_candidate_generation_disables_token_type_ids(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    seen: dict[str, object] = {}

    class FakeBatch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "right"

        def __call__(self, _prompts, **kwargs):
            seen.update(kwargs)
            batch = FakeBatch(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 1]]),
            )
            if kwargs.get("return_token_type_ids") is not False:
                batch["token_type_ids"] = torch.tensor([[0, 0]])
            return batch

        def batch_decode(self, _tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ["inc"]

    tokenizer = FakeTokenizer()

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=32)
        device = "cpu"

        def to(self, _device):
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            assert "token_type_ids" not in kwargs
            input_ids = kwargs["input_ids"]
            return torch.cat((input_ids, torch.tensor([[2]])), dim=1)

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeModelFactory,
        PreTrainedTokenizerFast=FakeTokenizerFactory,
        set_seed=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        evaluate, "require_deterministic_python_hash_seed", lambda: None
    )
    monkeypatch.setattr(
        evaluate, "configure_deterministic_cuda_environment", lambda: ":16:8"
    )
    monkeypatch.setattr(evaluate, "Executor", lambda _functions: object())
    monkeypatch.setattr(
        evaluate,
        "training_row_for_record",
        lambda *_args, **_kwargs: {"prompt": "prompt"},
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_candidate",
        lambda *_args, **_kwargs: {
            "execution_correct": True,
            "execution_error": None,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake")

    record = SimpleNamespace(metadata={"certified_distance": 1})
    rows, _runtime = evaluate.generate_scratch_candidates(
        [record],
        arm_id=ARM_IDS[0],
        checkpoint=tmp_path / "checkpoint",
        tokenizer_dir=tmp_path / "tokenizer",
        available_functions=object(),
        generation={
            "sampling_seed": 1,
            "max_sequence_length": 16,
            "max_new_tokens": 1,
            "samples_per_task": 1,
            "batch_size": 1,
            "temperature": 0.8,
            "top_p": 0.95,
        },
        progress=False,
    )

    assert seen["return_token_type_ids"] is False
    assert len(rows) == 1


PROTOCOL_DIGEST = "sha256:" + "1" * 64
TOKENIZER_DIGEST = "sha256:" + "2" * 64
TRAIN_RUNNER_DIGEST = "sha256:" + "b" * 64
SOURCE_HELPER_DIGEST = "sha256:" + "c" * 64
ARCHITECTURE_SOURCE_DIGEST = "sha256:" + "d" * 64
PROBE_RUNNER_DIGEST = "sha256:" + "e" * 64
EVALUATOR_RUNNER_DIGEST = "sha256:" + "0" * 64


def _record(distance: int, *, split: str = "test") -> BasisTaskRecord:
    value = (10_000 if split == "validation" else 20_000) + distance
    return BasisTaskRecord.create(
        split=split,
        input_value=TypedList([value], int),
        output_value=TypedList([value + distance], int),
        witness_function_ids=[f"bf:inc:{index}" for index in range(distance)],
        witness_function_names=["inc"] * distance,
        basis_set_id="wl-core-v1",
        basis_set_digest="sha256:" + "3" * 64,
        generator="unit-test",
        seed=1,
        source_index=distance,
        metadata={
            "certified_distance": distance,
            "root_digest": f"sha256:{value:064x}",
            "root_index": distance,
        },
    )


@pytest.fixture
def records() -> list[BasisTaskRecord]:
    return [_record(distance) for distance in range(1, 9)]


@pytest.fixture
def protocol() -> dict:
    evaluator_source_body = {
        "metadata": {"git_commit": "a" * 40},
        "files": [
            {
                "path": "experiments/evaluate_deep_corpus_v2_scratch.py",
                "bytes": 1,
                "sha256": EVALUATOR_RUNNER_DIGEST,
            },
            {
                "path": "experiments/probe_deep_corpus_v2_scratch.py",
                "bytes": 1,
                "sha256": PROBE_RUNNER_DIGEST,
            },
            {
                "path": "experiments/train_deep_corpus_sft.py",
                "bytes": 1,
                "sha256": SOURCE_HELPER_DIGEST,
            },
            {
                "path": "experiments/train_deep_corpus_v2_scratch.py",
                "bytes": 1,
                "sha256": TRAIN_RUNNER_DIGEST,
            },
            {
                "path": "wandering_light/training/deep_corpus_scratch.py",
                "bytes": 1,
                "sha256": ARCHITECTURE_SOURCE_DIGEST,
            },
        ],
    }
    evaluator_sources = {
        **evaluator_source_body,
        "source_sha256": evaluate.sha256_json(evaluator_source_body),
    }
    return {
        "protocol_lock_sha256": PROTOCOL_DIGEST,
        "corpus": {
            "manifest": {
                "manifest_digest": "sha256:" + "4" * 64,
                "basis_set_id": "wl-core-v1",
                "basis_set_digest": "sha256:" + "3" * 64,
            }
        },
        "cohorts": {
            "validation": {"cohort_sha256": "sha256:" + "5" * 64},
            "test": {"cohort_sha256": "sha256:" + "6" * 64},
        },
        "generation": {
            "sampling_seed": 20260826,
            "sample_prefixes": [1, 8, 64],
            "samples_per_task": 64,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 64,
            "batch_size": 32,
            "max_sequence_length": 2048,
            "do_sample": True,
            "pythonhashseed": "0",
            "generation_sha256": "sha256:" + "7" * 64,
        },
        "bfs": {
            "budget": 20_000_000,
            "max_depth": 8,
            "cost_definition": "one basis execution",
            "state_identity": "TypedList.search_key",
            "bfs_sha256": "sha256:" + "8" * 64,
        },
        "tokenizer": {
            "metadata": {
                "artifact_digest": TOKENIZER_DIGEST,
                "total_vocab_size": 600,
            }
        },
        "evaluator_sources": evaluator_sources,
    }


def _amended_protocol(protocol: dict) -> dict:
    amended = deepcopy(protocol)
    training_sources = deepcopy(protocol["evaluator_sources"])
    successor_body = deepcopy(training_sources)
    successor_body.pop("source_sha256")
    successor_body["metadata"]["git_commit"] = "b" * 40
    successor_body["files"][0]["sha256"] = "sha256:" + "9" * 64
    successor_sources = {
        **successor_body,
        "source_sha256": evaluate.sha256_json(successor_body),
    }
    amended["evaluator_sources"] = successor_sources
    amended["source_amendment"] = build_source_amendment(
        predecessor_protocol_lock_sha256=protocol["protocol_lock_sha256"],
        training_evaluator_sources=training_sources,
        successor_evaluator_sources=successor_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=[
            "experiments/evaluate_deep_corpus_v2_scratch.py"
        ],
        failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    )
    return amended


def _training_manifest(arm_id: str) -> dict:
    spec = ARM_SPECS[arm_id]
    parameter_error = ARM_IDS.index(arm_id)
    exact_parameters = spec.target_parameters + parameter_error
    initial_digest = "sha256:" + arm_id[0] * 64
    effective_digest = "sha256:" + "9" * 64
    retained_digest = "sha256:" + "a" * 64
    architecture = {
        "arm_id": arm_id,
        "label": spec.label,
        "family": spec.family,
        "config_source": {
            "repo_id": spec.config_repo,
            "revision": spec.config_revision,
            "digest": "sha256:" + "c" * 64,
        },
        "effective_config_digest": effective_digest,
        "target_trainable_parameters": spec.target_parameters,
        "exact_trainable_parameters": exact_parameters,
        "parameter_error": parameter_error,
        "parameter_error_fraction": parameter_error / spec.target_parameters,
        "budget_dimension": spec.budget_dimension,
        "budget_dimension_value": spec.budget_minimum,
        "vocab_size": 600,
    }
    study = {
        "architecture": architecture,
        "corpus_manifest_digest": "sha256:" + "4" * 64,
        "tokenizer_artifact_digest": TOKENIZER_DIGEST,
        "retained_training_task_ids_digest": retained_digest,
        "per_device_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_length": 2048,
        "optimizer": "adamw_torch",
        "precision": "bf16",
        "gradient_checkpointing": arm_id == "smollm2-360m-scratch",
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False}
            if arm_id == "smollm2-360m-scratch"
            else {}
        ),
        "cublas_workspace_config": ":16:8",
        "pythonhashseed": "0",
        "initialization_seed": 54_001,
        "data_order_seed": 54_002,
    }
    selection_body = {
        "status": "locked-before-final-training",
        "arm_id": arm_id,
        "selected_learning_rate": 0.0003,
        "initial_state_digest": initial_digest,
        "initialization_seed": 54_001,
        "effective_config_digest": effective_digest,
        "exact_trainable_parameters": exact_parameters,
    }
    return {
        "status": "complete",
        "phase": "final",
        "arm_id": arm_id,
        "architecture": architecture,
        "creation": {
            "creation_method": "AutoModelForCausalLM.from_config",
            "pretrained_weights_loaded": False,
            "initial_state_digest": initial_digest,
            "initialization_seed": 54_001,
            "effective_config_digest": effective_digest,
            "exact_trainable_parameters": exact_parameters,
        },
        "initialization_seed": 54_001,
        "data_order_seed": 54_002,
        "initial_state_digest": initial_digest,
        "effective_config_digest": effective_digest,
        "exact_trainable_parameters": exact_parameters,
        "tokenizer_digest": TOKENIZER_DIGEST,
        "source_data": {
            "corpus_manifest_digest": "sha256:" + "4" * 64,
            "task_roles": {
                "discovery": {
                    "retained_records": 100,
                    "retained_task_ids_digest": retained_digest,
                },
                "test": {"role": "held-out-evaluation-only; never model-scored here"},
            },
        },
        "learning_rate": 0.0003,
        "num_train_epochs": 3,
        "training_protocol": {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "max_length": 2048,
            "optimizer": "adamw_torch",
            "precision": "bf16",
            "gradient_checkpointing": arm_id == "smollm2-360m-scratch",
            "gradient_checkpointing_kwargs": (
                {"use_reentrant": False}
                if arm_id == "smollm2-360m-scratch"
                else {}
            ),
            "cublas_workspace_config": ":16:8",
            "pythonhashseed": "0",
        },
        "workload": {
            "unique_training_examples": 100,
            "planned_example_presentations": 300,
            "planned_combined_training_tokens": 120_000,
            "planned_completion_loss_tokens": 60_000,
        },
        "selection_lock": {
            **selection_body,
            "lock_digest": digest_json(selection_body),
        },
        "source_code": {
            "commit": "a" * 40,
            "dirty": False,
            "untracked_files": [],
            "runner": {
                "path": "experiments/train_deep_corpus_v2_scratch.py",
                "sha256": TRAIN_RUNNER_DIGEST,
            },
            "source_identity_helper": {
                "path": "experiments/train_deep_corpus_sft.py",
                "sha256": SOURCE_HELPER_DIGEST,
            },
        },
        "feasibility_gate": {
            "artifact_digest": "sha256:" + "f" * 64,
            "artifact_path": "/sealed/feasibility_manifest.json",
            "arm_id": arm_id,
            "arm_gate": {
                "launch_allowed": True,
                "route": "direct",
                "runtime_hard_stop_required": True,
                "baseline": {"projected_pilot_plus_final_hours": 13.5},
            },
            "probe_gpu": {"identity_digest": "sha256:" + "1" * 64},
            "run_gpu": {"identity_digest": "sha256:" + "1" * 64},
            "probe_source": {"commit": "a" * 40},
            "hard_stop_allocation": {"deadline_monotonic": 123.0},
            "study_identity_digest": digest_json(study),
        },
        "best_validation_loss": 0.5,
        "validation_trend": {"classification": "compute-limited"},
        "trainer_state": {
            "global_step": 123,
            "num_input_tokens_seen": 123_456,
            "train_metrics": {
                "train_runtime": 45.0,
            },
            "loss_curve": [
                {
                    "step": 123,
                    "validation_loss": 0.5,
                    "preceding_train_window_loss": 0.4,
                }
            ],
        },
        "execution_environment": {
            "cublas_workspace_config": ":16:8",
            "pythonhashseed": "0",
            "physical_vram_bytes": 24_000_000_000,
            "peak_torch_allocated_bytes": 12_000_000_000,
            "peak_torch_reserved_bytes": 13_000_000_000,
            "peak_observed_gpu_memory_bytes": 14_000_000_000,
            "gpu_memory_telemetry": {
                "measurement_scope": "process",
                "source": "nvidia-smi compute-apps used_gpu_memory polling",
                "gpu_uuid": "GPU-unit-test",
            },
            "memory_measurement_note": "unit-test process-scoped measurement",
        },
        "wandb": {"url": "https://wandb.ai/example"},
    }


def _checkpoint_identity(arm_id: str) -> dict:
    index = ARM_IDS.index(arm_id) + 1
    return {
        "path": f"/checkpoints/{arm_id}",
        "arm_id": arm_id,
        "checkpoint_sha256": "sha256:" + str(index) * 64,
        "exact_trainable_parameters": 10_000_000 + index,
    }


def _write_checkpoint(checkpoint, manifest):
    checkpoint.mkdir()
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
        "training_args.json",
        "wandb_run.url",
        "basis_set.json",
        "corpus_manifest.json",
    ):
        (checkpoint / name).write_bytes(b"metadata")


def _candidate_rows(records, arm_id):
    rows = []
    for record in records:
        distance = record.metadata["certified_distance"]
        for sample_index in range(64):
            unparseable = sample_index == 0
            correct = sample_index == 1
            row = {
                "arm_id": arm_id,
                "task_id": record.task_id,
                "split": record.split,
                "certified_distance": distance,
                "root_digest": record.metadata["root_digest"],
                "sample_index": sample_index,
                "raw_response": "" if unparseable else "inc",
                "parse_valid": not unparseable,
                "parse_error": "empty response" if unparseable else None,
                "parsed_function_names": [] if unparseable else ["inc"],
                "program_length": 0 if unparseable else distance if correct else 1,
                "charged_executions": (
                    0 if unparseable else distance if correct else 1
                ),
                "exact_match": correct,
                "executor_success": correct,
                "execution_correct": correct,
                "failed_at_step": None if correct or unparseable else 0,
                "execution_error": (
                    None
                    if correct or unparseable
                    else "Type mismatch at function 'inc'"
                ),
                "actual_output": "correct" if correct else None,
                "type_mismatch": not unparseable and not correct,
                "truncated": False,
                "finish_reason": "eos",
            }
            rows.append(row)
    return rows


def _bfs_rows(records):
    return [
        {
            "task_id": record.task_id,
            "split": record.split,
            "certified_distance": record.metadata["certified_distance"],
            "root_digest": record.metadata["root_digest"],
            "success": True,
            "executions": record.metadata["certified_distance"] * 10,
            "solution_length": record.metadata["certified_distance"],
            "solution_function_names": ["inc"] * record.metadata["certified_distance"],
            "termination": "solved",
            "seconds": 0.01,
        }
        for record in records
    ]


def test_summary_seal_rejects_mutation(protocol):
    summary = evaluate._seal_summary(
        {
            "schema_version": 1,
            "status": "complete",
            "split": "validation",
            "arm_id": ARM_IDS[0],
            "protocol_lock_sha256": PROTOCOL_DIGEST,
            "checkpoint": {"digest": "one"},
            "generation": {
                "runtime": {
                    "cublas_workspace_config": ":16:8",
                    "pythonhashseed": "0",
                }
            },
            "artifacts": {},
        }
    )
    assert (
        evaluate.validate_evaluation_summary(
            summary,
            protocol_lock_sha256=protocol["protocol_lock_sha256"],
            split="validation",
        )
        == summary
    )
    changed = deepcopy(summary)
    changed["checkpoint"]["digest"] = "two"
    with pytest.raises(ValueError, match="changed after it was sealed"):
        evaluate.validate_evaluation_summary(
            changed, protocol_lock_sha256=protocol["protocol_lock_sha256"]
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cublas_workspace_config", ":4096:8"),
        ("cublas_workspace_config", None),
        ("pythonhashseed", "1"),
        ("pythonhashseed", None),
    ],
)
def test_resealed_summary_rejects_wrong_or_missing_generation_runtime(
    protocol, field, replacement
):
    body = {
        "schema_version": 1,
        "status": "complete",
        "split": "validation",
        "arm_id": ARM_IDS[0],
        "protocol_lock_sha256": PROTOCOL_DIGEST,
        "checkpoint": {"digest": "one"},
        "generation": {
            "runtime": {
                "cublas_workspace_config": ":16:8",
                "pythonhashseed": "0",
            }
        },
        "artifacts": {},
    }
    if replacement is None:
        body["generation"]["runtime"].pop(field)
    else:
        body["generation"]["runtime"][field] = replacement
    summary = evaluate._seal_summary(body)

    with pytest.raises(ValueError, match="deterministic protocol"):
        evaluate.validate_evaluation_summary(
            summary, protocol_lock_sha256=protocol["protocol_lock_sha256"]
        )


def test_checkpoint_identity_requires_final_config_only_artifact(tmp_path, protocol):
    arm_id = ARM_IDS[0]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = _training_manifest(arm_id)
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
        "training_args.json",
        "wandb_run.url",
        "basis_set.json",
        "corpus_manifest.json",
    ):
        (checkpoint / name).write_bytes(b"metadata")

    identity, loaded = evaluate.checkpoint_identity(
        checkpoint, arm_id=arm_id, protocol_lock=protocol
    )
    assert loaded == manifest
    assert identity["arm_id"] == arm_id
    assert identity["checkpoint_sha256"].startswith("sha256:")
    assert identity["initial_state_digest"] == manifest["initial_state_digest"]

    manifest["creation"]["pretrained_weights_loaded"] = True
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="scratch provenance"):
        evaluate.checkpoint_identity(checkpoint, arm_id=arm_id, protocol_lock=protocol)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["source_code"].update(commit="c" * 40),
            "source commit differs",
        ),
        (
            lambda manifest: manifest["source_code"]["runner"].update(
                sha256="sha256:" + "0" * 64
            ),
            "runner differs",
        ),
        (
            lambda manifest: manifest["source_code"]["source_identity_helper"].update(
                sha256="sha256:" + "0" * 64
            ),
            "source-identity helper differs",
        ),
        (
            lambda manifest: manifest["architecture"].update(family="llama"),
            "frozen ARM_SPECS",
        ),
        (
            lambda manifest: manifest["architecture"]["config_source"].update(
                revision="main"
            ),
            "upstream architecture config",
        ),
        (
            lambda manifest: manifest["architecture"].update(
                budget_dimension_value=ARM_SPECS[ARM_IDS[0]].budget_minimum + 1
            ),
            "resolved architecture identity",
        ),
    ],
)
def test_checkpoint_identity_rejects_source_or_architecture_mutation(
    tmp_path, protocol, mutation, message
):
    arm_id = ARM_IDS[0]
    manifest = _training_manifest(arm_id)
    mutation(manifest)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, manifest)

    with pytest.raises(ValueError, match=message):
        evaluate.checkpoint_identity(checkpoint, arm_id=arm_id, protocol_lock=protocol)


def test_amended_protocol_validates_old_checkpoint_against_training_sources(
    tmp_path, protocol
):
    arm_id = ARM_IDS[0]
    checkpoint = tmp_path / "checkpoint"
    manifest = _training_manifest(arm_id)
    _write_checkpoint(checkpoint, manifest)

    identity, loaded = evaluate.checkpoint_identity(
        checkpoint,
        arm_id=arm_id,
        protocol_lock=_amended_protocol(protocol),
    )

    assert identity["arm_id"] == arm_id
    assert loaded == manifest


def test_checkpoint_identity_requires_frozen_arm_spec_source(tmp_path, protocol):
    arm_id = ARM_IDS[0]
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, _training_manifest(arm_id))
    changed_protocol = deepcopy(protocol)
    changed_protocol["evaluator_sources"]["files"] = changed_protocol[
        "evaluator_sources"
    ]["files"][:1]
    changed_source_body = deepcopy(changed_protocol["evaluator_sources"])
    changed_source_body.pop("source_sha256")
    changed_protocol["evaluator_sources"]["source_sha256"] = evaluate.sha256_json(
        changed_source_body
    )

    with pytest.raises(ValueError, match="do not identify"):
        evaluate.checkpoint_identity(
            checkpoint, arm_id=arm_id, protocol_lock=changed_protocol
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gate: gate.pop("artifact_digest"), "artifact digest"),
        (lambda gate: gate.update(arm_id=ARM_IDS[1]), "another arm"),
        (
            lambda gate: gate["arm_gate"].update(launch_allowed=False),
            "direct feasibility authorization",
        ),
        (
            lambda gate: gate["arm_gate"].update(route="validated-optimization-plan"),
            "direct feasibility authorization",
        ),
        (
            lambda gate: gate["arm_gate"]["baseline"].update(
                projected_pilot_plus_final_hours=14.01
            ),
            "exceeds 14 hours",
        ),
        (
            lambda gate: gate.update(study_identity_digest="sha256:" + "0" * 64),
            "study identity does not match",
        ),
    ],
)
def test_checkpoint_identity_rejects_inadequate_feasibility_gate(
    tmp_path, protocol, mutation, message
):
    arm_id = ARM_IDS[0]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = _training_manifest(arm_id)
    mutation(manifest["feasibility_gate"])
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
        "training_args.json",
        "wandb_run.url",
        "basis_set.json",
        "corpus_manifest.json",
    ):
        (checkpoint / name).write_bytes(b"metadata")

    with pytest.raises(ValueError, match=message):
        evaluate.checkpoint_identity(checkpoint, arm_id=arm_id, protocol_lock=protocol)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cublas_workspace_config", ":4096:8"),
        ("pythonhashseed", "1"),
        ("gradient_checkpointing", True),
        ("gradient_checkpointing_kwargs", {"use_reentrant": True}),
    ],
)
def test_checkpoint_identity_binds_deterministic_runtime(
    tmp_path, protocol, field, replacement
):
    arm_id = ARM_IDS[0]
    manifest = _training_manifest(arm_id)
    manifest["training_protocol"][field] = replacement
    if field in manifest["execution_environment"]:
        manifest["execution_environment"][field] = replacement
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, manifest)

    with pytest.raises(ValueError, match="study identity does not match"):
        evaluate.checkpoint_identity(
            checkpoint, arm_id=arm_id, protocol_lock=protocol
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cublas_workspace_config", ":4096:8"),
        ("pythonhashseed", "1"),
    ],
)
def test_checkpoint_identity_cross_checks_observed_deterministic_runtime(
    tmp_path, protocol, field, replacement
):
    arm_id = ARM_IDS[0]
    manifest = _training_manifest(arm_id)
    manifest["execution_environment"][field] = replacement
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, manifest)

    with pytest.raises(ValueError, match="deterministic runtime differs"):
        evaluate.checkpoint_identity(
            checkpoint, arm_id=arm_id, protocol_lock=protocol
        )


def test_360m_checkpoint_identity_binds_non_reentrant_checkpointing(
    tmp_path, protocol
):
    arm_id = "smollm2-360m-scratch"
    manifest = _training_manifest(arm_id)
    manifest["training_protocol"]["gradient_checkpointing_kwargs"] = {
        "use_reentrant": True
    }
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, manifest)

    with pytest.raises(ValueError, match="study identity does not match"):
        evaluate.checkpoint_identity(
            checkpoint, arm_id=arm_id, protocol_lock=protocol
        )


def test_training_outcomes_fall_back_to_planned_token_workload():
    manifest = _training_manifest(ARM_IDS[0])
    manifest["trainer_state"].pop("num_input_tokens_seen")

    outcomes = evaluate._training_outcomes(manifest)

    assert outcomes["actual_input_tokens_seen"] is None
    assert outcomes["input_tokens_seen"] == 120_000
    assert outcomes["input_tokens_seen_source"] == (
        "workload.planned_combined_training_tokens"
    )
    assert outcomes["planned_example_presentations"] == 300
    assert outcomes["planned_completion_loss_tokens"] == 60_000


def test_arm_summary_reports_prefix_distance_funnel_cost_and_training(
    protocol, records
):
    arm_id = ARM_IDS[0]
    rows = _candidate_rows(records, arm_id)
    bfs = _bfs_rows(records)
    summary = evaluate.build_arm_summary(
        split="test",
        arm_id=arm_id,
        protocol_lock=protocol,
        checkpoint=_checkpoint_identity(arm_id),
        training_manifest=_training_manifest(arm_id),
        records=records,
        candidate_rows=rows,
        generation_runtime={
            "seconds": 1.0,
            "cublas_workspace_config": ":16:8",
            "pythonhashseed": "0",
            "peak_torch_allocated_bytes": 1_000,
            "peak_torch_reserved_bytes": 1_100,
        },
        bfs_rows=bfs,
        artifacts={"candidates": {}, "bfs": {}},
    )

    assert summary["model"]["overall"]["1"]["solve_rate"] == 0
    assert summary["model"]["overall"]["8"]["solve_rate"] == 1
    assert summary["model"]["overall"]["64"]["exact_match_rate"] == 1
    assert set(summary["model"]["by_certified_distance_at_64"]) == {
        str(distance) for distance in range(1, 9)
    }
    funnel = summary["candidate_funnel"]["by_k"]["64"]
    assert funnel["unparseable"] == len(records)
    assert funnel["parse_failure_counts"]["empty"] == len(records)
    assert funnel["correct_given_parseable_rate"] > 0
    assert summary["bfs"]["overall"]["mean_basis_executions"] == 45
    assert summary["training"]["global_steps"] == 123
    assert summary["training"]["input_tokens_seen"] == 123_456
    assert summary["training"]["input_tokens_seen_source"] == (
        "trainer_state.num_input_tokens_seen"
    )
    assert summary["training"]["peak_torch_allocated_bytes"] == 12_000_000_000
    assert summary["training"]["peak_torch_reserved_bytes"] == 13_000_000_000
    assert summary["training"]["peak_observed_gpu_memory_bytes"] == 14_000_000_000
    assert summary["training"]["gpu_memory_telemetry"]["gpu_uuid"] == ("GPU-unit-test")
    assert summary["training"]["physical_vram_bytes"] == 24_000_000_000


def test_wandb_publisher_keeps_sealed_result_immutable_and_writes_receipt(
    tmp_path, protocol, records, monkeypatch
):
    arm_id = ARM_IDS[0]
    result_dir = tmp_path / "result"
    rows = _candidate_rows(records, arm_id)
    bfs = _bfs_rows(records)
    summary, selection = evaluate._atomic_arm_result(
        result_dir,
        split="test",
        arm_id=arm_id,
        protocol_lock=protocol,
        checkpoint=_checkpoint_identity(arm_id),
        training_manifest=_training_manifest(arm_id),
        records=records,
        candidate_rows=rows,
        generation_runtime={
            "seconds": 1.0,
            "cublas_workspace_config": ":16:8",
            "pythonhashseed": "0",
            "peak_torch_allocated_bytes": 1_000,
            "peak_torch_reserved_bytes": 1_100,
        },
        bfs_rows=bfs,
        write_selection=False,
    )
    assert selection is None
    original_summary_bytes = (result_dir / evaluate.SUMMARY_FILENAME).read_bytes()
    monkeypatch.setattr(evaluate, "load_protocol_lock", lambda *_, **__: protocol)

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.files = []

        def add_file(self, path, *, name):
            self.files.append((path, name))

    class FakeRun:
        def __init__(self, run_id):
            self.id = run_id
            self.url = (
                f"https://wandb.ai/{evaluate.PERSONAL_WANDB_ENTITY}/"
                f"{evaluate.DEFAULT_WANDB_PROJECT}/runs/{run_id}"
            )
            self.summary = {}
            self.logged = []
            self.artifacts = []
            self.finished = False

        def log(self, value):
            self.logged.append(value)

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self):
            self.finished = True

    class FakeWandb:
        def __init__(self):
            self.init_kwargs = None
            self.run = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            self.run = FakeRun(kwargs["id"])
            return self.run

        Artifact = FakeArtifact

    fake = FakeWandb()
    receipt = evaluate.publish_evaluation_to_wandb(
        protocol_lock_path=tmp_path / "protocol.json",
        tokenizer_dir=tmp_path / "tokenizer",
        result_dir=result_dir,
        wandb_module=fake,
    )

    assert (
        result_dir / evaluate.SUMMARY_FILENAME
    ).read_bytes() == original_summary_bytes
    assert (
        receipt["evaluation_summary_sha256"] == summary[evaluate.SUMMARY_DIGEST_FIELD]
    )
    assert receipt["entity"] == "abhishekraok-na"
    assert receipt["project"] == "wandering-light-deep-corpus-v2-scratch"
    assert receipt["run_name"] == f"issue-54-v2-test-{arm_id}"
    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["resume"] == "allow"
    assert fake.run.finished is True
    assert len(fake.run.artifacts) == 1
    assert evaluate.validate_wandb_receipt(result_dir, summary=summary) == receipt
    with pytest.raises(FileExistsError, match="receipt already exists"):
        evaluate.publish_evaluation_to_wandb(
            protocol_lock_path=tmp_path / "protocol.json",
            tokenizer_dir=tmp_path / "tokenizer",
            result_dir=result_dir,
            wandb_module=fake,
        )

    changed = deepcopy(receipt)
    changed["url"] = "https://wandb.ai/another/project/runs/run"
    (result_dir / evaluate.WANDB_RECEIPT_FILENAME).write_text(
        json.dumps(changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after it was sealed"):
        evaluate.validate_wandb_receipt(result_dir, summary=summary)


def _selection_locks(protocol, arm_ids=ARM_IDS):
    locks = []
    validation = {}
    for arm_id in arm_ids:
        checkpoint = _checkpoint_identity(arm_id)
        summary_digest = "sha256:" + str(ARM_IDS.index(arm_id) + 4) * 64
        lock = build_selection_lock(
            arm_id=arm_id,
            protocol_lock_sha256=protocol["protocol_lock_sha256"],
            checkpoint_identity=checkpoint,
            validation_selection={"validation_summary_sha256": summary_digest},
        )
        locks.append(lock)
        validation[arm_id] = {
            "selection_lock": lock,
            "summary": {
                evaluate.SUMMARY_DIGEST_FIELD: summary_digest,
            },
            "wandb_receipt": {
                evaluate.WANDB_RECEIPT_DIGEST_FIELD: (
                    "sha256:" + str(ARM_IDS.index(arm_id) + 7) * 64
                )
            },
            "checkpoint": checkpoint,
            "training_manifest": _training_manifest(arm_id),
        }
    return validation, locks


def _feasibility_manifest(active_arm_ids=ARM_IDS):
    active = tuple(active_arm_ids)
    deferred = tuple(arm_id for arm_id in ARM_IDS if arm_id not in active)
    artifact_digest = "sha256:" + "f" * 64
    per_arm = {}
    arm_results = []
    for arm_id in ARM_IDS:
        training_gate = _training_manifest(arm_id)["feasibility_gate"]
        arm_results.append(
            {
                "arm_id": arm_id,
                "study_identity_digest": training_gate["study_identity_digest"],
            }
        )
        per_arm[arm_id] = (
            training_gate["arm_gate"]
            if arm_id in active
            else {
                "launch_allowed": False,
                "route": "blocked",
                "reason": "projected runtime exceeds the limit",
            }
        )
    return {
        "status": "complete",
        "artifact_digest": artifact_digest,
        "source_code": {
            "commit": "a" * 40,
            "dirty": False,
            "untracked_files": [],
            "runner": {
                "path": "experiments/train_deep_corpus_sft.py",
                "sha256": SOURCE_HELPER_DIGEST,
            },
            "probe_runner": {
                "path": "experiments/probe_deep_corpus_v2_scratch.py",
                "sha256": PROBE_RUNNER_DIGEST,
            },
        },
        "arm_results": arm_results,
        "launch_gate": {
            "authorized_arm_ids": list(active),
            "blocked_arm_ids": list(deferred),
            "per_arm": per_arm,
        },
    }


def _write_readiness(test_dir, protocol, validation, feasibility=None):
    feasibility = feasibility or _feasibility_manifest(tuple(validation))
    identity = evaluate._validate_test_feasibility(
        feasibility,
        protocol_lock=protocol,
        validation_results=validation,
    )
    body = evaluate._test_readiness_body(
        protocol_lock=protocol,
        validation_results=validation,
        feasibility_identity=identity,
    )
    value = {**body, "readiness_sha256": evaluate.sha256_json(body)}
    (test_dir / evaluate.TEST_READINESS_FILENAME).write_text(
        json.dumps(value), encoding="utf-8"
    )


def test_prepare_test_opening_creates_prepared_zero_access_ledger(
    tmp_path, protocol, monkeypatch
):
    validation, locks = _selection_locks(protocol)
    monkeypatch.setattr(evaluate, "load_protocol_lock", lambda *_, **__: protocol)
    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (validation, locks),
    )
    feasibility = _feasibility_manifest()
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: feasibility
    )
    test_dir = tmp_path / "held-out"

    result = evaluate.prepare_test_opening(
        protocol_lock_path=tmp_path / "protocol.json",
        tokenizer_dir=tmp_path / "tokenizer",
        validation_result_dirs=[tmp_path / arm for arm in ARM_IDS],
        feasibility_manifest_path=tmp_path / "feasibility.json",
        test_dir=test_dir,
        opening_id="issue-54-final-test",
    )

    ledger = load_test_opening_ledger(test_dir / evaluate.TEST_LEDGER_FILENAME)
    assert result["status"] == "prepared"
    assert ledger["status"] == "prepared"
    assert ledger["test_access_count"] == 0
    assert (test_dir / evaluate.TEST_READINESS_FILENAME).is_file()
    assert (
        evaluate.validate_test_readiness(
            test_dir,
            protocol_lock=protocol,
            validation_results=validation,
            feasibility_manifest=feasibility,
        )["status"]
        == "locked-before-held-out-test"
    )
    with pytest.raises(FileExistsError):
        evaluate.prepare_test_opening(
            protocol_lock_path=tmp_path / "protocol.json",
            tokenizer_dir=tmp_path / "tokenizer",
            validation_result_dirs=[tmp_path / arm for arm in ARM_IDS],
            feasibility_manifest_path=tmp_path / "feasibility.json",
            test_dir=test_dir,
            opening_id="another-opening",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: source.update(commit="b" * 40),
        lambda source: source["probe_runner"].update(
            sha256="sha256:" + "7" * 64
        ),
        lambda source: source["runner"].update(sha256="sha256:" + "7" * 64),
        lambda source: source.update(untracked_files=[{"path": "untracked.py"}]),
    ],
)
def test_feasibility_producer_is_rejected_before_test_ledger_creation(
    tmp_path, protocol, monkeypatch, mutation
):
    validation, locks = _selection_locks(protocol)
    feasibility = _feasibility_manifest()
    mutation(feasibility["source_code"])
    monkeypatch.setattr(evaluate, "load_protocol_lock", lambda *_, **__: protocol)
    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (validation, locks),
    )
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: feasibility
    )
    test_dir = tmp_path / "held-out"

    with pytest.raises(ValueError, match="Feasibility"):
        evaluate.prepare_test_opening(
            protocol_lock_path=tmp_path / "protocol.json",
            tokenizer_dir=tmp_path / "tokenizer",
            validation_result_dirs=[tmp_path / arm for arm in ARM_IDS],
            feasibility_manifest_path=tmp_path / "feasibility.json",
            test_dir=test_dir,
            opening_id="issue-54-final-test",
        )

    assert not test_dir.exists()


def test_amended_protocol_validates_feasibility_against_training_sources(protocol):
    feasibility = _feasibility_manifest()
    amended = _amended_protocol(protocol)

    evaluate.validate_feasibility_producer(feasibility, protocol_lock=amended)
    feasibility["source_code"]["commit"] = "b" * 40
    with pytest.raises(ValueError, match="training commit"):
        evaluate.validate_feasibility_producer(feasibility, protocol_lock=amended)


def test_test_opening_is_one_shared_atomic_access_and_recovers_without_rescoring(
    tmp_path, protocol, records, monkeypatch
):
    validation, locks = _selection_locks(protocol)
    feasibility = _feasibility_manifest()
    test_dir = tmp_path / "held-out"
    test_dir.mkdir()
    ledger = new_test_opening_ledger(
        opening_id="issue-54-final-test",
        protocol_lock_sha256=protocol["protocol_lock_sha256"],
        selection_locks=locks,
        feasibility_artifact_digest=feasibility["artifact_digest"],
    )
    create_test_opening_ledger_file(test_dir / evaluate.TEST_LEDGER_FILENAME, ledger)
    _write_readiness(test_dir, protocol, validation, feasibility)
    monkeypatch.setattr(evaluate, "load_protocol_lock", lambda *_, **__: protocol)
    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (validation, locks),
    )
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: feasibility
    )

    def load_records(*_args, **_kwargs):
        durable = load_test_opening_ledger(test_dir / evaluate.TEST_LEDGER_FILENAME)
        assert durable["test_access_count"] == 1
        assert durable["status"] in {"in_progress", "complete"}
        basis = SimpleNamespace(as_function_set=object)
        return basis, records, {"ok": True}

    monkeypatch.setattr(evaluate, "load_frozen_cohort_records", load_records)
    calls = Counter()
    fail_once = {ARM_IDS[1]: True}

    def candidates(records, *, arm_id, **_kwargs):
        calls[arm_id] += 1
        if fail_once.get(arm_id):
            fail_once[arm_id] = False
            raise RuntimeError("simulated crash")
        return _candidate_rows(records, arm_id), {
            "seconds": 1.0,
            "cublas_workspace_config": ":16:8",
            "pythonhashseed": "0",
            "peak_torch_allocated_bytes": 100,
            "peak_torch_reserved_bytes": 110,
        }

    def bfs(records, **_kwargs):
        return _bfs_rows(records)

    common = {
        "protocol_lock_path": tmp_path / "protocol.json",
        "corpus_dir": tmp_path / "corpus",
        "tokenizer_dir": tmp_path / "tokenizer",
        "validation_result_dirs": [tmp_path / arm for arm in ARM_IDS],
        "feasibility_manifest_path": tmp_path / "feasibility.json",
        "test_dir": test_dir,
        "reference_corpus_dirs": [tmp_path / "v1"],
        "candidate_provider": candidates,
        "bfs_provider": bfs,
    }
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate.run_test_opening(**common)
    interrupted = load_test_opening_ledger(test_dir / evaluate.TEST_LEDGER_FILENAME)
    assert interrupted["status"] == "interrupted"
    assert interrupted["test_access_count"] == 1
    assert interrupted["arms"][ARM_IDS[0]]["status"] == "complete"
    assert interrupted["arms"][ARM_IDS[1]]["status"] == "pending"

    real_advance = evaluate._advance_ledger
    kill_before_ledger_completion = {"armed": True}

    def advance_then_kill(path, current, next_value):
        if (
            kill_before_ledger_completion["armed"]
            and next_value["status"] == "complete"
        ):
            kill_before_ledger_completion["armed"] = False
            raise SystemExit("simulated process kill after summary publication")
        return real_advance(path, current, next_value)

    monkeypatch.setattr(evaluate, "_advance_ledger", advance_then_kill)
    with pytest.raises(SystemExit, match="after summary publication"):
        evaluate.run_test_opening(**common)
    after_kill = load_test_opening_ledger(test_dir / evaluate.TEST_LEDGER_FILENAME)
    assert after_kill["status"] == "in_progress"
    assert (test_dir / evaluate.TEST_SUMMARY_FILENAME).is_file()

    monkeypatch.setattr(evaluate, "_advance_ledger", real_advance)

    def no_rescore(*_args, **_kwargs):
        raise AssertionError("completed test opening must only reload artifacts")

    common["candidate_provider"] = no_rescore
    common["bfs_provider"] = no_rescore
    result = evaluate.run_test_opening(**common)
    assert result["status"] == "complete"
    assert result["ledger"]["test_access_count"] == 1
    assert calls[ARM_IDS[0]] == 1
    assert calls[ARM_IDS[1]] == 2
    assert calls[ARM_IDS[2]] == 1
    assert result["summary"]["candidate_funnels"]["arm_ids"] == list(ARM_IDS)
    repeated = evaluate.run_test_opening(**common)
    assert repeated["ledger"]["test_access_count"] == 1
    assert repeated["summary"] == result["summary"]

    summary_path = test_dir / evaluate.TEST_SUMMARY_FILENAME
    mutated = json.loads(summary_path.read_text(encoding="utf-8"))
    mutated["arms"][ARM_IDS[0]]["generation_runtime"]["seconds"] = 999
    summary_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after sealing"):
        evaluate.run_test_opening(**common)


def test_two_active_arms_open_once_and_deferred_arm_cannot_be_added(
    tmp_path, protocol, records, monkeypatch
):
    active = ARM_IDS[:2]
    deferred = ARM_IDS[2:]
    validation, locks = _selection_locks(protocol, active)
    feasibility = _feasibility_manifest(active)
    monkeypatch.setattr(evaluate, "load_protocol_lock", lambda *_, **__: protocol)
    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (validation, locks),
    )
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: feasibility
    )
    test_dir = tmp_path / "held-out-two-active"
    evaluate.prepare_test_opening(
        protocol_lock_path=tmp_path / "protocol.json",
        tokenizer_dir=tmp_path / "tokenizer",
        validation_result_dirs=[tmp_path / arm_id for arm_id in active],
        feasibility_manifest_path=tmp_path / "feasibility.json",
        test_dir=test_dir,
        opening_id="issue-54-two-active",
    )
    prepared = load_test_opening_ledger(test_dir / evaluate.TEST_LEDGER_FILENAME)
    assert prepared["active_arm_ids"] == list(active)
    assert prepared["deferred_arm_ids"] == list(deferred)
    assert prepared["feasibility_artifact_digest"] == feasibility["artifact_digest"]
    assert "artifact_path" in validation[active[0]]["training_manifest"][
        "feasibility_gate"
    ]
    tampered_feasibility = deepcopy(feasibility)
    tampered_feasibility["launch_gate"]["per_arm"][deferred[0]][
        "launch_allowed"
    ] = True
    with pytest.raises(ValueError, match="not feasibility-blocked"):
        evaluate._validate_test_feasibility(
            tampered_feasibility,
            protocol_lock=protocol,
            validation_results=validation,
        )

    basis = SimpleNamespace(as_function_set=object)
    monkeypatch.setattr(
        evaluate,
        "load_frozen_cohort_records",
        lambda **_kwargs: (basis, records, {"ok": True}),
    )
    calls = Counter()

    def candidates(records, *, arm_id, **_kwargs):
        calls[arm_id] += 1
        return _candidate_rows(records, arm_id), {
            "seconds": 1.0,
            "cublas_workspace_config": ":16:8",
            "pythonhashseed": "0",
            "peak_torch_allocated_bytes": 100,
            "peak_torch_reserved_bytes": 110,
        }

    common = {
        "protocol_lock_path": tmp_path / "protocol.json",
        "corpus_dir": tmp_path / "corpus",
        "tokenizer_dir": tmp_path / "tokenizer",
        "validation_result_dirs": [tmp_path / arm_id for arm_id in active],
        "feasibility_manifest_path": tmp_path / "feasibility.json",
        "test_dir": test_dir,
        "reference_corpus_dirs": [tmp_path / "v1"],
        "candidate_provider": candidates,
        "bfs_provider": lambda records, **_kwargs: _bfs_rows(records),
    }
    result = evaluate.run_test_opening(**common)
    assert result["ledger"]["test_access_count"] == 1
    assert result["summary"]["active_arm_ids"] == list(active)
    assert result["summary"]["deferred_arm_ids"] == list(deferred)
    assert result["summary"]["candidate_funnels"]["arm_ids"] == list(active)
    assert set(result["summary"]["arms"]) == set(active)
    assert calls == Counter(dict.fromkeys(active, 1))
    assert not (test_dir / "arms" / deferred[0]).exists()

    all_validation, all_locks = _selection_locks(protocol)
    all_feasibility = _feasibility_manifest()
    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (all_validation, all_locks),
    )
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: all_feasibility
    )
    with pytest.raises(ValueError, match="frozen opening"):
        evaluate.run_test_opening(
            **{
                **common,
                "validation_result_dirs": [tmp_path / arm_id for arm_id in ARM_IDS],
            }
        )

    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (validation, locks),
    )
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: feasibility
    )
    (test_dir / "arms" / deferred[0]).mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be added"):
        evaluate.run_test_opening(**common)


def test_complete_ledger_with_missing_arm_refuses_rescore(
    tmp_path, protocol, records, monkeypatch
):
    validation, locks = _selection_locks(protocol)
    feasibility = _feasibility_manifest()
    test_dir = tmp_path / "held-out"
    test_dir.mkdir()
    ledger = new_test_opening_ledger(
        opening_id="opening",
        protocol_lock_sha256=PROTOCOL_DIGEST,
        selection_locks=locks,
        feasibility_artifact_digest=feasibility["artifact_digest"],
    )
    ledger = begin_test_opening(ledger)
    for arm_id in ARM_IDS:
        ledger = record_test_arm_result(
            ledger,
            arm_id=arm_id,
            result_identity={"summary": f"sha256:{ARM_IDS.index(arm_id) + 1}"},
        )
    ledger = complete_test_opening(
        ledger,
        shared_bfs_identity={"sha256": "sha256:" + "b" * 64},
    )
    create_test_opening_ledger_file(test_dir / evaluate.TEST_LEDGER_FILENAME, ledger)
    _write_readiness(test_dir, protocol, validation, feasibility)
    monkeypatch.setattr(evaluate, "load_protocol_lock", lambda *_, **__: protocol)
    monkeypatch.setattr(
        evaluate,
        "load_validation_results",
        lambda *_, **__: (validation, locks),
    )
    monkeypatch.setattr(
        evaluate, "load_probe_manifest", lambda *_args, **_kwargs: feasibility
    )
    basis = SimpleNamespace(as_function_set=object)
    monkeypatch.setattr(
        evaluate,
        "load_frozen_cohort_records",
        lambda **_kwargs: (basis, records, {"ok": True}),
    )

    with pytest.raises(ValueError, match="lost its shared BFS artifact"):
        evaluate.run_test_opening(
            protocol_lock_path=tmp_path / "protocol.json",
            corpus_dir=tmp_path / "corpus",
            tokenizer_dir=tmp_path / "tokenizer",
            validation_result_dirs=[tmp_path / arm for arm in ARM_IDS],
            feasibility_manifest_path=tmp_path / "feasibility.json",
            test_dir=test_dir,
            reference_corpus_dirs=[tmp_path / "v1"],
            candidate_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("must not rescore")
            ),
            bfs_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("must not rerun BFS")
            ),
        )
