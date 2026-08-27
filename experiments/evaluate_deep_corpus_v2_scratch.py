"""Evaluate issue #54's scratch arms under one frozen protocol.

Validation is intentionally arm-local: each completed final checkpoint may be
scored and locked independently.  Held-out test scoring is intentionally not.
The test command accepts a nonempty ordered subset of validation results whose
complement the feasibility gate blocked, atomically opens one shared ledger
before loading the test cohort, evaluates or recovers every active arm, runs one
shared BFS baseline, and seals the opening.  Deferred arms can never be added to
that opening.

No command in this module trains a model or chooses a checkpoint.  The only
checkpoint identity accepted for test is the one already sealed by that arm's
validation selection lock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments import generate_deep_corpus as deep
from experiments.evaluate_deep_corpus_sft import (
    aggregate_model_candidates,
    evaluate_bfs_records,
    evaluate_candidate,
    matched_compute_comparison,
    read_jsonl_gz,
    validate_bfs_rows,
    write_jsonl_gz,
)
from experiments.freeze_deep_corpus_v2_protocol import REPO_ROOT
from experiments.prepare_deep_corpus_v2 import (
    V1_CORPUS_DIR as DEFAULT_REFERENCE_CORPUS_DIR,
)
from experiments.probe_deep_corpus_v2_scratch import load_probe_manifest
from experiments.train_deep_corpus_sft import (
    assert_weights_only_artifact,
    training_row_for_record,
    validate_wandb_url,
)
from wandering_light.basis_set import load_basis_set
from wandering_light.evals.deep_corpus_protocol import (
    ARM_IDS,
    SAMPLE_PREFIXES,
    aggregate_candidate_funnels,
    begin_test_opening,
    build_selection_lock,
    canonical_copy,
    complete_test_opening,
    create_test_opening_ledger_file,
    interrupt_test_opening,
    load_test_opening_ledger,
    new_test_opening_ledger,
    record_test_arm_result,
    replace_test_opening_ledger_file,
    resume_test_opening,
    sha256_file,
    sha256_json,
    validate_active_arm_ids,
    validate_frozen_cohort,
    validate_protocol_lock,
    validate_selection_locks,
    validate_test_opening_ledger,
)
from wandering_light.executor import Executor
from wandering_light.training.deep_corpus_scratch import (
    ARM_SPECS,
    DEFAULT_WANDB_PROJECT,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    PARAMETER_TOLERANCE_FRACTION,
    PERSONAL_WANDB_ENTITY,
    build_scratch_study_identity,
    configure_deterministic_cuda_environment,
    validate_final_restart,
)
from wandering_light.training.deep_corpus_tokenizer import (
    DETERMINISTIC_PYTHON_HASH_SEED,
    require_deterministic_python_hash_seed,
)

SCHEMA_VERSION = 1
SUMMARY_DIGEST_FIELD = "evaluation_summary_sha256"
SUMMARY_FILENAME = "summary.json"
CANDIDATES_FILENAME = "model_candidates.jsonl.gz"
BFS_FILENAME = "bfs_results.jsonl.gz"
SELECTION_LOCK_FILENAME = "selection_lock.json"
TEST_LEDGER_FILENAME = "test_opening_ledger.json"
TEST_READINESS_FILENAME = "test_readiness.json"
TEST_SUMMARY_FILENAME = "test_summary.json"
WANDB_RECEIPT_FILENAME = "wandb_evaluation_receipt.json"
WANDB_RECEIPT_DIGEST_FIELD = "wandb_receipt_sha256"
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v2")
DEFAULT_TOKENIZER_DIR = Path("wandering_light/training/tokenizers/deep_corpus_v2")


CandidateProvider = Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]
BFSProvider = Callable[..., list[dict[str, Any]]]


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json(
    path: Path, value: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _write_json_atomic_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete JSON file without overwriting an existing receipt."""

    temporary = path.with_name(f".{path.name}.publishing-{os.getpid()}")
    try:
        _write_json(temporary, value, exclusive=True)
        # A same-directory hard link is both atomic and fail-if-present.  Unlike
        # os.replace, two publishers can never overwrite one another's receipt.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _seal_summary(body: Mapping[str, Any]) -> dict[str, Any]:
    value = canonical_copy(dict(body))
    value[SUMMARY_DIGEST_FIELD] = sha256_json(value)
    return value


def _validate_generation_runtime(runtime: Any) -> None:
    if not isinstance(runtime, Mapping) or (
        runtime.get("cublas_workspace_config")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("pythonhashseed") != DETERMINISTIC_PYTHON_HASH_SEED
    ):
        raise ValueError("Generation runtime differs from the deterministic protocol")


def validate_evaluation_summary(
    summary: Mapping[str, Any],
    *,
    protocol_lock_sha256: str,
    split: str | None = None,
    arm_id: str | None = None,
) -> dict[str, Any]:
    value = canonical_copy(dict(summary))
    recorded = value.pop(SUMMARY_DIGEST_FIELD, None)
    if recorded != sha256_json(value):
        raise ValueError("Evaluation summary changed after it was sealed")
    value[SUMMARY_DIGEST_FIELD] = recorded
    if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != (
        "complete"
    ):
        raise ValueError("Evaluation summary is not complete")
    if value.get("protocol_lock_sha256") != protocol_lock_sha256:
        raise ValueError("Evaluation summary belongs to another protocol")
    if value.get("split") not in {"validation", "test"}:
        raise ValueError("Evaluation summary has an invalid split")
    if split is not None and value["split"] != split:
        raise ValueError(f"Expected a {split} evaluation summary")
    if value.get("arm_id") not in ARM_IDS:
        raise ValueError("Evaluation summary has an invalid arm")
    if arm_id is not None and value["arm_id"] != arm_id:
        raise ValueError(f"Evaluation summary does not belong to {arm_id}")
    if not isinstance(value.get("checkpoint"), dict) or not value["checkpoint"]:
        raise ValueError("Evaluation summary has no checkpoint identity")
    if not isinstance(value.get("artifacts"), dict):
        raise ValueError("Evaluation summary has no raw artifact identities")
    generation = value.get("generation")
    runtime = generation.get("runtime") if isinstance(generation, Mapping) else None
    _validate_generation_runtime(runtime)
    return value


def _protocol_manifest(protocol_lock: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = protocol_lock.get("corpus", {}).get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Protocol lock has no embedded corpus manifest")
    return manifest


def load_protocol_lock(
    path: str | Path,
    *,
    tokenizer_dir: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    value = _read_json(path, label="protocol lock")
    validated = validate_protocol_lock(
        value,
        tokenizer_root=tokenizer_dir,
        evaluator_root=repo_root,
    )
    if (
        validated.get("generation", {}).get("pythonhashseed")
        != DETERMINISTIC_PYTHON_HASH_SEED
    ):
        raise ValueError("Protocol lock has no deterministic Python hash seed")
    return validated


def _verify_reference_corpora(
    reference_corpus_dirs: Sequence[str | Path],
) -> tuple[deep.CorpusIdentities, ...]:
    if not reference_corpus_dirs:
        raise ValueError("At least one frozen reference corpus is required")
    return tuple(
        deep.collect_corpus_identities(Path(path).resolve())
        for path in reference_corpus_dirs
    )


def load_frozen_cohort_records(
    *,
    corpus_dir: str | Path,
    protocol_lock: Mapping[str, Any],
    split: str,
    reference_corpus_dirs: Sequence[str | Path] = (DEFAULT_REFERENCE_CORPUS_DIR,),
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Reverify the corpus and return records in the lock's exact task order."""

    if split not in {"validation", "test"}:
        raise ValueError("Only validation and test cohorts can be evaluated")
    root = Path(corpus_dir).resolve()
    expected_manifest = _protocol_manifest(protocol_lock)
    manifest, split_records = deep.load_corpus(root, splits=(split,))
    if canonical_copy(manifest) != canonical_copy(dict(expected_manifest)):
        raise ValueError("Evaluation corpus differs from the frozen protocol")
    references = _verify_reference_corpora(reference_corpus_dirs)
    verification = deep.verify_corpus(
        root,
        recertify=0,
        seed=manifest["seed"],
        reference_identities=references,
    )
    if verification.get("ok") is not True:
        raise ValueError(
            "Evaluation corpus failed complete witness/leakage verification"
        )
    cohort = validate_frozen_cohort(
        protocol_lock["cohorts"][split], records=split_records
    )
    by_id = {record.task_id: record for record in split_records}
    if len(by_id) != len(split_records):
        raise ValueError(f"Corpus {split} split contains duplicate task IDs")
    records = [by_id[task_id] for task_id in cohort["task_ids"]]
    basis = load_basis_set(manifest["basis_set_id"])
    if basis.digest != manifest["basis_set_digest"]:
        raise ValueError("Evaluation basis differs from the corpus manifest")
    return basis, records, verification


def _checkpoint_files(checkpoint: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(checkpoint.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _training_outcomes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    trainer = manifest.get("trainer_state", {})
    metrics = trainer.get("train_metrics", {}) if isinstance(trainer, Mapping) else {}
    workload = manifest.get("workload", {})
    if not isinstance(workload, Mapping):
        workload = {}
    source = manifest.get("source_data", {})
    roles = source.get("task_roles", {}) if isinstance(source, Mapping) else {}
    discovery = roles.get("discovery", {}) if isinstance(roles, Mapping) else {}
    environment = manifest.get("execution_environment", {})
    if not isinstance(environment, Mapping):
        environment = {}
    actual_input_tokens = (
        trainer.get("num_input_tokens_seen") if isinstance(trainer, Mapping) else None
    )
    planned_input_tokens = workload.get("planned_combined_training_tokens")
    trainer_runtime = (
        metrics.get("train_runtime") if isinstance(metrics, Mapping) else None
    )
    full_runtime = environment.get("full_invocation_elapsed_seconds")
    return {
        "exact_trainable_parameters": manifest.get("exact_trainable_parameters"),
        "initialization_seed": manifest.get("initialization_seed"),
        "initial_state_digest": manifest.get("initial_state_digest"),
        "creation_method": manifest.get("creation", {}).get("creation_method"),
        "pretrained_weights_loaded": manifest.get("creation", {}).get(
            "pretrained_weights_loaded"
        ),
        "learning_rate": manifest.get("learning_rate"),
        "epochs": manifest.get("num_train_epochs"),
        "global_steps": trainer.get("global_step")
        if isinstance(trainer, Mapping)
        else None,
        "training_examples": workload.get(
            "unique_training_examples", discovery.get("retained_records")
        ),
        "planned_example_presentations": workload.get("planned_example_presentations"),
        "planned_combined_training_tokens": planned_input_tokens,
        "planned_completion_loss_tokens": workload.get(
            "planned_completion_loss_tokens"
        ),
        "actual_input_tokens_seen": actual_input_tokens,
        "input_tokens_seen": (
            actual_input_tokens
            if actual_input_tokens is not None
            else planned_input_tokens
        ),
        "input_tokens_seen_source": (
            "trainer_state.num_input_tokens_seen"
            if actual_input_tokens is not None
            else "workload.planned_combined_training_tokens"
        ),
        "runtime_seconds": full_runtime,
        "runtime_scope": environment.get("full_invocation_runtime_scope"),
        "trainer_train_runtime_seconds": trainer_runtime,
        "peak_torch_allocated_bytes": environment.get("peak_torch_allocated_bytes"),
        "peak_torch_reserved_bytes": environment.get("peak_torch_reserved_bytes"),
        "peak_observed_gpu_memory_bytes": environment.get(
            "peak_observed_gpu_memory_bytes"
        ),
        "gpu_memory_telemetry": canonical_copy(environment.get("gpu_memory_telemetry")),
        "physical_vram_bytes": environment.get("physical_vram_bytes"),
        "vram_measurement_note": environment.get("memory_measurement_note"),
        "best_validation_loss": manifest.get("best_validation_loss"),
        "loss_trend": canonical_copy(manifest.get("validation_trend")),
        "loss_curve": canonical_copy(
            trainer.get("loss_curve", []) if isinstance(trainer, Mapping) else []
        ),
        "wandb": canonical_copy(manifest.get("wandb", {})),
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a canonical SHA-256 digest")
    return value


def _validate_checkpoint_feasibility_gate(
    manifest: Mapping[str, Any], *, arm_id: str
) -> dict[str, Any]:
    gate = manifest.get("feasibility_gate")
    if not isinstance(gate, Mapping) or not gate:
        raise ValueError("Final checkpoint has no feasibility gate")
    if gate.get("arm_id") != arm_id:
        raise ValueError("Final checkpoint feasibility gate belongs to another arm")
    _require_sha256(gate.get("artifact_digest"), label="Feasibility artifact digest")
    study_digest = _require_sha256(
        gate.get("study_identity_digest"), label="Feasibility study identity"
    )
    arm_gate = gate.get("arm_gate")
    if not isinstance(arm_gate, Mapping):
        raise ValueError("Final checkpoint has no per-arm feasibility authorization")
    if arm_gate.get("launch_allowed") is not True or arm_gate.get("route") != "direct":
        raise ValueError("Final checkpoint lacks a direct feasibility authorization")
    baseline = arm_gate.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Direct feasibility authorization has no baseline projection")
    projected_hours = baseline.get("projected_pilot_plus_final_hours")
    if (
        isinstance(projected_hours, bool)
        or not isinstance(projected_hours, (int, float))
        or not 0 < float(projected_hours) <= 14
    ):
        raise ValueError("Direct feasibility authorization exceeds 14 hours")
    if arm_gate.get("runtime_hard_stop_required") is not True:
        raise ValueError(
            "Direct feasibility authorization lacks its hard-stop requirement"
        )

    expected_study = _checkpoint_study_identity(manifest)
    if sha256_json(expected_study) != study_digest:
        raise ValueError("Final checkpoint feasibility study identity does not match")
    return canonical_copy(dict(gate))


def _checkpoint_study_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the feasibility identity from independent checkpoint provenance."""

    source = manifest.get("source_data", {})
    roles = source.get("task_roles", {}) if isinstance(source, Mapping) else {}
    discovery = roles.get("discovery", {}) if isinstance(roles, Mapping) else {}
    protocol = manifest.get("training_protocol", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    environment = manifest.get("execution_environment", {})
    if not isinstance(environment, Mapping):
        environment = {}
    values = {
        "architecture": manifest.get("architecture"),
        "corpus_manifest_digest": source.get("corpus_manifest_digest"),
        "tokenizer_artifact_digest": manifest.get("tokenizer_digest"),
        "retained_training_task_ids_digest": discovery.get("retained_task_ids_digest"),
        "per_device_batch_size": protocol.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": protocol.get("gradient_accumulation_steps"),
        "max_length": protocol.get("max_length"),
        "optimizer": protocol.get("optimizer"),
        "precision": protocol.get("precision"),
        "gradient_checkpointing": protocol.get("gradient_checkpointing"),
        "gradient_checkpointing_kwargs": protocol.get(
            "gradient_checkpointing_kwargs"
        ),
        "cublas_workspace_config": protocol.get("cublas_workspace_config"),
        "pythonhashseed": protocol.get("pythonhashseed"),
        "initialization_seed": manifest.get("initialization_seed"),
        "data_order_seed": manifest.get("data_order_seed"),
    }
    if not isinstance(values["architecture"], Mapping) or any(
        value is None for value in values.values()
    ):
        raise ValueError("Final checkpoint cannot reconstruct its feasibility study")
    for field in ("cublas_workspace_config", "pythonhashseed"):
        if environment.get(field) != protocol.get(field):
            raise ValueError(
                "Final checkpoint deterministic runtime differs from training "
                f"protocol: {field}"
            )
    return build_scratch_study_identity(**values)


def _frozen_source_file(
    protocol_lock: Mapping[str, Any], *, path: str
) -> dict[str, Any]:
    files = protocol_lock.get("evaluator_sources", {}).get("files", [])
    matches = [
        row for row in files if isinstance(row, Mapping) and row.get("path") == path
    ]
    if len(matches) != 1:
        raise ValueError(f"Frozen evaluator sources do not identify {path}")
    return canonical_copy(dict(matches[0]))


def _validate_checkpoint_source_and_architecture(
    manifest: Mapping[str, Any],
    *,
    arm_id: str,
    protocol_lock: Mapping[str, Any],
) -> None:
    # ARM_SPECS is meaningful as a frozen contract only when its defining
    # module is itself covered by the validated protocol source identity.
    _frozen_source_file(
        protocol_lock, path="wandering_light/training/deep_corpus_scratch.py"
    )
    spec = ARM_SPECS[arm_id]
    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Final checkpoint has no resolved architecture identity")
    expected_architecture = {
        "arm_id": arm_id,
        "label": spec.label,
        "family": spec.family,
        "target_trainable_parameters": spec.target_parameters,
        "budget_dimension": spec.budget_dimension,
    }
    if {key: architecture.get(key) for key in expected_architecture} != (
        expected_architecture
    ):
        raise ValueError("Final checkpoint architecture differs from frozen ARM_SPECS")
    config_source = architecture.get("config_source")
    if not isinstance(config_source, Mapping) or (
        config_source.get("repo_id"),
        config_source.get("revision"),
    ) != (spec.config_repo, spec.config_revision):
        raise ValueError("Final checkpoint used another upstream architecture config")
    _require_sha256(
        config_source.get("digest"), label="Upstream architecture config digest"
    )
    exact = manifest.get("exact_trainable_parameters")
    parameter_error = architecture.get("parameter_error")
    budget_value = architecture.get("budget_dimension_value")
    if (
        architecture.get("exact_trainable_parameters") != exact
        or architecture.get("effective_config_digest")
        != manifest.get("effective_config_digest")
        or type(parameter_error) is not int
        or parameter_error != abs(int(exact) - spec.target_parameters)
        or parameter_error / spec.target_parameters > PARAMETER_TOLERANCE_FRACTION
        or not math.isclose(
            float(architecture.get("parameter_error_fraction", math.inf)),
            parameter_error / spec.target_parameters,
            rel_tol=0,
            abs_tol=0,
        )
        or type(budget_value) is not int
        or not spec.budget_minimum <= budget_value <= spec.budget_maximum
        or budget_value % spec.budget_multiple
        or architecture.get("vocab_size")
        != protocol_lock["tokenizer"]["metadata"]["total_vocab_size"]
    ):
        raise ValueError(
            "Final checkpoint resolved architecture identity is inconsistent"
        )
    creation = manifest.get("creation", {})
    for field in (
        "effective_config_digest",
        "exact_trainable_parameters",
        "initial_state_digest",
        "initialization_seed",
    ):
        if creation.get(field) != manifest.get(field):
            raise ValueError("Final checkpoint creation provenance changed after init")

    source = manifest.get("source_code")
    if not isinstance(source, Mapping) or source.get("dirty") is not False:
        raise ValueError("Final checkpoint was not produced from a clean source tree")
    frozen_sources = protocol_lock.get("evaluator_sources", {})
    frozen_commit = frozen_sources.get("metadata", {}).get("git_commit")
    if source.get("commit") != frozen_commit:
        raise ValueError("Final checkpoint source commit differs from the protocol")
    if source.get("untracked_files") != []:
        raise ValueError("Final checkpoint source identity contains untracked files")
    runner_path = "experiments/train_deep_corpus_v2_scratch.py"
    frozen_runner = _frozen_source_file(protocol_lock, path=runner_path)
    if source.get("runner") != {
        "path": runner_path,
        "sha256": frozen_runner["sha256"],
    }:
        raise ValueError("Final checkpoint runner differs from the frozen source")


def checkpoint_identity(
    checkpoint_dir: str | Path,
    *,
    arm_id: str,
    protocol_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = Path(checkpoint_dir).resolve()
    assert_weights_only_artifact(checkpoint, require_complete=True)
    manifest_path = checkpoint / "training_manifest.json"
    manifest = _read_json(manifest_path, label="final training manifest")
    if (manifest.get("status"), manifest.get("phase"), manifest.get("arm_id")) != (
        "complete",
        "final",
        arm_id,
    ):
        raise ValueError("Checkpoint is not the completed final artifact for its arm")
    corpus_manifest = _protocol_manifest(protocol_lock)
    source = manifest.get("source_data", {})
    if source.get("corpus_manifest_digest") != corpus_manifest.get("manifest_digest"):
        raise ValueError("Final checkpoint was trained from another corpus")
    tokenizer_digest = protocol_lock["tokenizer"]["metadata"]["artifact_digest"]
    if manifest.get("tokenizer_digest") != tokenizer_digest:
        raise ValueError("Final checkpoint used another tokenizer artifact")
    creation = manifest.get("creation", {})
    if creation.get("creation_method") != "AutoModelForCausalLM.from_config" or (
        creation.get("pretrained_weights_loaded") is not False
    ):
        raise ValueError("Final checkpoint lacks config-only scratch provenance")
    if type(manifest.get("exact_trainable_parameters")) is not int or (
        manifest["exact_trainable_parameters"] <= 0
    ):
        raise ValueError("Final checkpoint has no exact parameter count")
    initial_digest = manifest.get("initial_state_digest")
    if not isinstance(initial_digest, str) or not initial_digest.startswith("sha256:"):
        raise ValueError("Final checkpoint has no initial-state digest")
    _validate_checkpoint_source_and_architecture(
        manifest,
        arm_id=arm_id,
        protocol_lock=protocol_lock,
    )
    roles = source.get("task_roles", {})
    if roles.get("test", {}).get("role") != (
        "held-out-evaluation-only; never model-scored here"
    ):
        raise ValueError("Final checkpoint does not prove held-out test isolation")
    feasibility_gate = _validate_checkpoint_feasibility_gate(manifest, arm_id=arm_id)
    validate_final_restart(
        selection_lock=manifest.get("selection_lock", {}),
        arm_id=arm_id,
        learning_rate=float(manifest.get("learning_rate", -1)),
        initialization_provenance=creation,
        num_train_epochs=float(manifest.get("num_train_epochs", -1)),
    )
    weights = [
        path
        for path in checkpoint.iterdir()
        if path.is_file()
        and (
            path.suffix == ".safetensors"
            or (path.suffix == ".bin" and path.name.startswith("pytorch_model"))
        )
    ]
    if not weights:
        raise ValueError("Final checkpoint has no model weights")
    files = _checkpoint_files(checkpoint)
    identity = {
        "path": str(checkpoint),
        "arm_id": arm_id,
        "training_manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "checkpoint_sha256": sha256_json(files),
        "exact_trainable_parameters": manifest["exact_trainable_parameters"],
        "initial_state_digest": initial_digest,
        "effective_config_digest": manifest.get("effective_config_digest"),
        "tokenizer_digest": manifest["tokenizer_digest"],
        "corpus_manifest_digest": source["corpus_manifest_digest"],
        "feasibility_artifact_digest": feasibility_gate["artifact_digest"],
        "feasibility_study_identity_digest": feasibility_gate["study_identity_digest"],
    }
    return identity, manifest


def generate_scratch_candidates(
    records: Sequence[Any],
    *,
    arm_id: str,
    checkpoint: Path,
    tokenizer_dir: Path,
    available_functions: Any,
    generation: Mapping[str, Any],
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one ordered 64-sample draw using the frozen shared tokenizer."""

    require_deterministic_python_hash_seed()
    cublas_workspace_config = configure_deterministic_cuda_environment()
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Scratch evaluation requires a CUDA GPU")
    set_seed(int(generation["sampling_seed"]), deterministic=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_dir, local_files_only=True
    )
    tokenizer.padding_side = "left"
    executor = Executor(available_functions)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    if int(getattr(model.config, "max_position_embeddings", 0)) < int(
        generation["max_sequence_length"]
    ):
        del model
        torch.cuda.empty_cache()
        raise ValueError("Checkpoint context window is smaller than the protocol lock")
    torch.cuda.reset_peak_memory_stats()
    maximum_sequence = int(generation["max_sequence_length"])
    max_new_tokens = int(generation["max_new_tokens"])
    prompt_max_length = maximum_sequence - max_new_tokens
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for record_index, record in enumerate(records, start=1):
            prompt = training_row_for_record(record, available_functions)["prompt"]
            responses: list[tuple[str, bool]] = []
            while len(responses) < int(generation["samples_per_task"]):
                count = min(
                    int(generation["batch_size"]),
                    int(generation["samples_per_task"]) - len(responses),
                )
                inputs = tokenizer(
                    [prompt] * count,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=prompt_max_length,
                ).to(model.device)
                width = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=float(generation["temperature"]),
                        top_p=float(generation["top_p"]),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                generated = outputs[:, width:]
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for token_ids, response in zip(generated, decoded, strict=True):
                    token_list = token_ids.tolist()
                    truncated = (
                        len(token_list) == max_new_tokens
                        and tokenizer.eos_token_id not in token_list
                    )
                    responses.append((response, truncated))
            for sample_index, (response, truncated) in enumerate(responses):
                row = evaluate_candidate(
                    record,
                    sample_index=sample_index,
                    response=response,
                    available_functions=available_functions,
                    executor=executor,
                )
                row["arm_id"] = arm_id
                row["truncated"] = truncated
                row["finish_reason"] = "max_new_tokens" if truncated else "eos"
                row["type_mismatch"] = bool(
                    isinstance(row.get("execution_error"), str)
                    and row["execution_error"].startswith("Type mismatch")
                )
                rows.append(row)
            if progress:
                solved = sum(
                    bool(row["execution_correct"])
                    for row in rows[-int(generation["samples_per_task"]) :]
                )
                print(
                    f"[{arm_id}] {record_index}/{len(records)} "
                    f"d={record.metadata['certified_distance']} "
                    f"correct_samples={solved}/{generation['samples_per_task']}",
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        return rows, {
            "seconds": elapsed,
            "samples": len(rows),
            "samples_per_second": len(rows) / elapsed if elapsed else None,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cublas_workspace_config": cublas_workspace_config,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    finally:
        del model
        torch.cuda.empty_cache()


def _arm_funnel(
    rows: Sequence[Mapping[str, Any]], *, arm_id: str, task_ids: Sequence[str]
) -> dict[str, Any]:
    return aggregate_candidate_funnels(
        rows,
        expected_task_ids=task_ids,
        active_arm_ids=(arm_id,),
    )["arms"][arm_id]


def _overall_model_metrics(
    per_task: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sample_count in SAMPLE_PREFIXES:
        rows = [metrics[sample_count] for metrics in per_task.values()]
        costs = [int(row["executions_spent"]) for row in rows]
        hit_costs = [
            int(row["execution_hit_cost"])
            for row in rows
            if row["execution_hit_cost"] is not None
        ]
        tasks = len(rows)
        solved = sum(bool(row["execution_correct"]) for row in rows)
        exact = sum(bool(row["exact_match"]) for row in rows)
        result[str(sample_count)] = {
            "tasks": tasks,
            "solved": solved,
            "solve_rate": solved / tasks if tasks else 0.0,
            "exact_matches": exact,
            "exact_match_rate": exact / tasks if tasks else 0.0,
            "total_basis_executions": sum(costs),
            "mean_basis_executions": sum(costs) / tasks if tasks else 0.0,
            "mean_hit_basis_executions": (
                sum(hit_costs) / len(hit_costs) if hit_costs else 0.0
            ),
        }
    return result


def _overall_bfs_metrics(bfs_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks = len(bfs_rows)
    executions = [int(row["executions"]) for row in bfs_rows]
    solved = sum(bool(row["success"]) for row in bfs_rows)
    return {
        "tasks": tasks,
        "solved": solved,
        "solve_rate": solved / tasks if tasks else 0.0,
        "total_basis_executions": sum(executions),
        "mean_basis_executions": sum(executions) / tasks if tasks else 0.0,
    }


def build_arm_summary(
    *,
    split: str,
    arm_id: str,
    protocol_lock: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    records: Sequence[Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    generation_runtime: Mapping[str, Any],
    bfs_rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_generation_runtime(generation_runtime)
    task_ids = [record.task_id for record in records]
    model_by_distance, per_task = aggregate_model_candidates(
        candidate_rows,
        expected_records=records,
        sample_counts=SAMPLE_PREFIXES,
    )
    bfs_by_distance = validate_bfs_rows(bfs_rows, records)
    matched = matched_compute_comparison(
        per_task=per_task,
        bfs_rows=bfs_rows,
        records=records,
        sample_counts=SAMPLE_PREFIXES,
    )
    funnel = _arm_funnel(candidate_rows, arm_id=arm_id, task_ids=task_ids)
    overall = _overall_model_metrics(per_task)
    by_distance_at_64 = {
        distance: metrics[str(SAMPLE_PREFIXES[-1])]
        for distance, metrics in model_by_distance["by_certified_distance"].items()
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "arm_id": arm_id,
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "cohort_sha256": protocol_lock["cohorts"][split]["cohort_sha256"],
        "checkpoint": canonical_copy(dict(checkpoint)),
        "generation": {
            **canonical_copy(dict(protocol_lock["generation"])),
            "runtime": canonical_copy(dict(generation_runtime)),
        },
        "bfs_configuration": canonical_copy(dict(protocol_lock["bfs"])),
        "model": {
            "overall": overall,
            "by_certified_distance": model_by_distance["by_certified_distance"],
            "by_certified_distance_at_64": by_distance_at_64,
            "distance_7_8": model_by_distance["distance_7_8"],
        },
        "candidate_funnel": funnel,
        "bfs": {
            "overall": _overall_bfs_metrics(bfs_rows),
            **bfs_by_distance,
        },
        "matched_compute": matched,
        "training": _training_outcomes(training_manifest),
        "artifacts": canonical_copy(dict(artifacts)),
    }
    return _seal_summary(body)


def _artifact_identity(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_arm_result(
    output_dir: Path,
    *,
    split: str,
    arm_id: str,
    protocol_lock: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    records: Sequence[Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    generation_runtime: Mapping[str, Any],
    bfs_rows: Sequence[Mapping[str, Any]],
    write_selection: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-", dir=destination.parent
        )
    )
    try:
        candidates_path = write_jsonl_gz(
            candidate_rows, temporary / CANDIDATES_FILENAME
        )
        bfs_path = write_jsonl_gz(bfs_rows, temporary / BFS_FILENAME)
        artifact_identities = {
            "candidates": _artifact_identity(candidates_path, root=temporary),
            "bfs": _artifact_identity(bfs_path, root=temporary),
        }
        summary = build_arm_summary(
            split=split,
            arm_id=arm_id,
            protocol_lock=protocol_lock,
            checkpoint=checkpoint,
            training_manifest=training_manifest,
            records=records,
            candidate_rows=candidate_rows,
            generation_runtime=generation_runtime,
            bfs_rows=bfs_rows,
            artifacts=artifact_identities,
        )
        _write_json(temporary / SUMMARY_FILENAME, summary, exclusive=True)
        selection_lock = None
        if write_selection:
            selection_lock = build_selection_lock(
                arm_id=arm_id,
                protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
                checkpoint_identity=checkpoint,
                validation_selection={
                    "split": "validation",
                    "validation_cohort_sha256": protocol_lock["cohorts"]["validation"][
                        "cohort_sha256"
                    ],
                    "validation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
                    "selection_rule": (
                        "final checkpoint fixed by validation-only LR and best-loss "
                        "selection before held-out test"
                    ),
                },
            )
            _write_json(
                temporary / SELECTION_LOCK_FILENAME,
                selection_lock,
                exclusive=True,
            )
        if destination.exists():
            raise FileExistsError(f"Evaluation output appeared: {destination}")
        temporary.rename(destination)
        return summary, selection_lock
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_validation_evaluation(
    *,
    protocol_lock_path: str | Path,
    corpus_dir: str | Path,
    tokenizer_dir: str | Path,
    arm_id: str,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    reference_corpus_dirs: Sequence[str | Path] = (DEFAULT_REFERENCE_CORPUS_DIR,),
    repo_root: str | Path = REPO_ROOT,
    dry_run: bool = False,
    candidate_provider: CandidateProvider = generate_scratch_candidates,
    bfs_provider: BFSProvider = evaluate_bfs_records,
) -> dict[str, Any]:
    protocol = load_protocol_lock(
        protocol_lock_path, tokenizer_dir=tokenizer_dir, repo_root=repo_root
    )
    if arm_id not in ARM_IDS:
        raise ValueError(f"Unexpected scratch arm: {arm_id!r}")
    basis, records, verification = load_frozen_cohort_records(
        corpus_dir=corpus_dir,
        protocol_lock=protocol,
        split="validation",
        reference_corpus_dirs=reference_corpus_dirs,
    )
    checkpoint, training_manifest = checkpoint_identity(
        checkpoint_dir, arm_id=arm_id, protocol_lock=protocol
    )
    plan = {
        "status": "dry-run",
        "split": "validation",
        "arm_id": arm_id,
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "cohort_sha256": protocol["cohorts"]["validation"]["cohort_sha256"],
        "tasks": len(records),
        "checkpoint": checkpoint,
        "verification": verification,
    }
    if dry_run:
        return plan
    candidates, runtime = candidate_provider(
        records,
        arm_id=arm_id,
        checkpoint=Path(checkpoint_dir).resolve(),
        tokenizer_dir=Path(tokenizer_dir).resolve(),
        available_functions=basis.as_function_set(),
        generation=protocol["generation"],
        progress=True,
    )
    bfs_rows = bfs_provider(
        records,
        available_functions=basis.as_function_set(),
        budget=protocol["bfs"]["budget"],
        max_depth=protocol["bfs"]["max_depth"],
        progress=True,
    )
    summary, selection = _atomic_arm_result(
        Path(output_dir),
        split="validation",
        arm_id=arm_id,
        protocol_lock=protocol,
        checkpoint=checkpoint,
        training_manifest=training_manifest,
        records=records,
        candidate_rows=candidates,
        generation_runtime=runtime,
        bfs_rows=bfs_rows,
        write_selection=True,
    )
    assert selection is not None
    return {
        "status": "complete",
        "summary": summary,
        "selection_lock": selection,
    }


def _validate_raw_artifacts(result_dir: Path, summary: Mapping[str, Any]) -> None:
    for key in ("candidates", "bfs"):
        identity = summary["artifacts"].get(key)
        if not isinstance(identity, Mapping):
            raise ValueError(f"Evaluation summary has no {key} artifact")
        relative = Path(str(identity.get("path", "")))
        path = (result_dir / relative).resolve()
        try:
            path.relative_to(result_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Evaluation {key} artifact path escapes its result"
            ) from error
        if (
            not path.is_file()
            or path.stat().st_size != identity.get("bytes")
            or sha256_file(path) != identity.get("sha256")
        ):
            raise ValueError(f"Validation result {key} artifact changed")


def _wandb_run_name(summary: Mapping[str, Any]) -> str:
    return f"issue-54-v2-{summary['split']}-{summary['arm_id']}"


def _wandb_run_id(summary: Mapping[str, Any]) -> str:
    digest = str(summary[SUMMARY_DIGEST_FIELD]).removeprefix("sha256:")
    return f"i54{digest[:20]}"


def validate_wandb_receipt(
    result_dir: str | Path,
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the separate, immutable W&B publication receipt for one arm."""

    root = Path(result_dir).resolve()
    value = _read_json(root / WANDB_RECEIPT_FILENAME, label="W&B evaluation receipt")
    recorded = value.pop(WANDB_RECEIPT_DIGEST_FIELD, None)
    if recorded != sha256_json(value):
        raise ValueError("W&B evaluation receipt changed after it was sealed")
    value[WANDB_RECEIPT_DIGEST_FIELD] = recorded
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "published",
        "protocol_lock_sha256": summary["protocol_lock_sha256"],
        "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
        "split": summary["split"],
        "arm_id": summary["arm_id"],
        "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
        "entity": PERSONAL_WANDB_ENTITY,
        "project": DEFAULT_WANDB_PROJECT,
        "run_name": _wandb_run_name(summary),
        "run_id": _wandb_run_id(summary),
    }
    if {key: value.get(key) for key in expected} != expected:
        raise ValueError("W&B evaluation receipt does not match its sealed result")
    url = validate_wandb_url(str(value.get("url")), project=DEFAULT_WANDB_PROJECT)
    if url.rstrip("/").split("/")[-1] != expected["run_id"]:
        raise ValueError("W&B evaluation receipt URL has another run ID")
    if not isinstance(value.get("artifact_name"), str) or not value["artifact_name"]:
        raise ValueError("W&B evaluation receipt has no artifact name")
    if (
        not isinstance(value.get("published_at_utc"), str)
        or not value["published_at_utc"]
    ):
        raise ValueError("W&B evaluation receipt has no publication time")
    allowed = {
        *expected,
        "url",
        "artifact_name",
        "published_at_utc",
        WANDB_RECEIPT_DIGEST_FIELD,
    }
    if set(value) != allowed:
        raise ValueError("W&B evaluation receipt has unexpected fields")
    return value


def _log_wandb_evaluation_metrics(run: Any, summary: Mapping[str, Any]) -> None:
    for sample_count in SAMPLE_PREFIXES:
        key = str(sample_count)
        model = summary["model"]["overall"][key]
        funnel = summary["candidate_funnel"]["by_k"][key]
        run.log(
            {
                "solve_at_k/k": sample_count,
                "solve_at_k/execution_rate": model["solve_rate"],
                "solve_at_k/exact_match_rate": model["exact_match_rate"],
                "solve_at_k/mean_basis_executions": model["mean_basis_executions"],
                "funnel/unparseable_rate": funnel["unparseable_rate"],
                "funnel/executable_rate": funnel["executable_rate"],
                "funnel/correct_given_parseable_rate": funnel[
                    "correct_given_parseable_rate"
                ],
            }
        )
    for distance, model in summary["model"]["by_certified_distance_at_64"].items():
        run.log(
            {
                "distance_at_64/certified_distance": int(distance),
                "distance_at_64/execution_rate": model["solve_rate"],
                "distance_at_64/exact_match_rate": model["exact_match_rate"],
            }
        )


def publish_evaluation_to_wandb(
    *,
    protocol_lock_path: str | Path,
    tokenizer_dir: str | Path,
    result_dir: str | Path,
    repo_root: str | Path = REPO_ROOT,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Publish one sealed arm result and atomically add a separate receipt.

    The summary and raw artifacts remain byte-for-byte unchanged.  A deterministic
    run ID makes a retry after a process crash resume the same hosted run instead
    of creating a second evaluation.
    """

    protocol = load_protocol_lock(
        protocol_lock_path, tokenizer_dir=tokenizer_dir, repo_root=repo_root
    )
    root = Path(result_dir).resolve()
    receipt_path = root / WANDB_RECEIPT_FILENAME
    if receipt_path.exists():
        raise FileExistsError(f"W&B receipt already exists: {receipt_path}")
    summary = validate_evaluation_summary(
        _read_json(root / SUMMARY_FILENAME, label="evaluation summary"),
        protocol_lock_sha256=protocol["protocol_lock_sha256"],
    )
    _validate_raw_artifacts(root, summary)
    if wandb_module is None:
        import wandb as wandb_module

    run_name = _wandb_run_name(summary)
    run_id = _wandb_run_id(summary)
    artifact_name = f"deep-corpus-v2-{summary['split']}-{summary['arm_id']}-{run_id}"
    run = wandb_module.init(
        entity=PERSONAL_WANDB_ENTITY,
        project=DEFAULT_WANDB_PROJECT,
        name=run_name,
        id=run_id,
        resume="allow",
        mode="online",
        tags=["deep-corpus-v2", "issue-54", "scratch", summary["split"]],
        config={
            "schema_version": SCHEMA_VERSION,
            "protocol_lock_sha256": summary["protocol_lock_sha256"],
            "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
            "split": summary["split"],
            "arm_id": summary["arm_id"],
            "checkpoint": summary["checkpoint"],
            "generation": {
                key: value
                for key, value in summary["generation"].items()
                if key != "runtime"
            },
            "bfs": summary["bfs_configuration"],
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create an online evaluation run")
    try:
        url = validate_wandb_url(str(run.url), project=DEFAULT_WANDB_PROJECT)
        if url.rstrip("/").split("/")[-1] != run_id:
            raise ValueError("W&B returned a URL for another evaluation run ID")
        _log_wandb_evaluation_metrics(run, summary)
        run.summary.update(
            {
                "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
                "protocol_lock_sha256": summary["protocol_lock_sha256"],
                "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
                "solve_at_1": summary["model"]["overall"]["1"]["solve_rate"],
                "solve_at_8": summary["model"]["overall"]["8"]["solve_rate"],
                "solve_at_64": summary["model"]["overall"]["64"]["solve_rate"],
                "exact_match_at_64": summary["model"]["overall"]["64"][
                    "exact_match_rate"
                ],
                "unparseable_at_64": summary["candidate_funnel"]["by_k"]["64"][
                    "unparseable_rate"
                ],
                "correct_given_parseable_at_64": summary["candidate_funnel"]["by_k"][
                    "64"
                ]["correct_given_parseable_rate"],
            }
        )
        artifact = wandb_module.Artifact(
            name=artifact_name,
            type="evaluation",
            metadata={
                "protocol_lock_sha256": summary["protocol_lock_sha256"],
                "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
                "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
            },
        )
        for name in (
            SUMMARY_FILENAME,
            CANDIDATES_FILENAME,
            BFS_FILENAME,
            SELECTION_LOCK_FILENAME,
        ):
            path = root / name
            if path.is_file():
                artifact.add_file(str(path), name=name)
        run.log_artifact(artifact)
    finally:
        run.finish()

    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "published",
        "protocol_lock_sha256": summary["protocol_lock_sha256"],
        "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
        "split": summary["split"],
        "arm_id": summary["arm_id"],
        "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
        "entity": PERSONAL_WANDB_ENTITY,
        "project": DEFAULT_WANDB_PROJECT,
        "run_name": run_name,
        "run_id": run_id,
        "url": url,
        "artifact_name": artifact_name,
        "published_at_utc": datetime.now(UTC).isoformat(),
    }
    receipt = {**body, WANDB_RECEIPT_DIGEST_FIELD: sha256_json(body)}
    _write_json_atomic_exclusive(receipt_path, receipt)
    return validate_wandb_receipt(root, summary=summary)


def load_validation_results(
    result_dirs: Sequence[str | Path],
    *,
    protocol_lock: Mapping[str, Any],
    require_wandb_receipts: bool = True,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not 1 <= len(result_dirs) <= len(ARM_IDS):
        raise ValueError("One to three validation result directories are required")
    by_arm: dict[str, dict[str, Any]] = {}
    locks: list[dict[str, Any]] = []
    for raw_dir in result_dirs:
        result_dir = Path(raw_dir).resolve()
        summary = validate_evaluation_summary(
            _read_json(result_dir / SUMMARY_FILENAME, label="validation summary"),
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            split="validation",
        )
        arm_id = summary["arm_id"]
        if arm_id in by_arm:
            raise ValueError(f"Duplicate validation result for {arm_id}")
        _validate_raw_artifacts(result_dir, summary)
        wandb_receipt = (
            validate_wandb_receipt(result_dir, summary=summary)
            if require_wandb_receipts
            else None
        )
        selection = _read_json(
            result_dir / SELECTION_LOCK_FILENAME, label="selection lock"
        )
        if selection.get("arm_id") != arm_id:
            raise ValueError("Selection lock and validation summary arms differ")
        validation_selection = selection.get("validation_selection", {})
        if validation_selection.get("validation_summary_sha256") != summary.get(
            SUMMARY_DIGEST_FIELD
        ):
            raise ValueError("Selection lock does not seal its validation result")
        live_checkpoint, training_manifest = checkpoint_identity(
            summary["checkpoint"]["path"],
            arm_id=arm_id,
            protocol_lock=protocol_lock,
        )
        if live_checkpoint != summary["checkpoint"] or live_checkpoint != selection.get(
            "checkpoint"
        ):
            raise ValueError("Final checkpoint changed after validation selection")
        by_arm[arm_id] = {
            "directory": result_dir,
            "summary": summary,
            "selection_lock": selection,
            "checkpoint": live_checkpoint,
            "training_manifest": training_manifest,
            "wandb_receipt": wandb_receipt,
        }
        locks.append(selection)
    canonical_locks = validate_selection_locks(
        locks, protocol_lock_sha256=protocol_lock["protocol_lock_sha256"]
    )
    active = tuple(canonical_locks)
    if set(by_arm) != set(active):
        raise ValueError("Validation results do not match their active selection locks")
    return {arm_id: by_arm[arm_id] for arm_id in active}, [
        canonical_locks[arm_id] for arm_id in active
    ]


def _validate_test_feasibility(
    feasibility_manifest: Mapping[str, Any],
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind an exact active/deferred split to one validated feasibility artifact."""

    active = validate_active_arm_ids(tuple(validation_results))
    deferred = tuple(arm_id for arm_id in ARM_IDS if arm_id not in active)
    if feasibility_manifest.get("status") != "complete":
        raise ValueError("Held-out test requires a complete feasibility manifest")
    artifact_digest = _require_sha256(
        feasibility_manifest.get("artifact_digest"),
        label="Feasibility artifact digest",
    )
    gate = feasibility_manifest.get("launch_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Feasibility manifest has no launch gate")
    if gate.get("authorized_arm_ids") != list(active) or gate.get(
        "blocked_arm_ids"
    ) != list(deferred):
        raise ValueError(
            "Feasibility authorization does not match the active/deferred arms"
        )
    per_arm = gate.get("per_arm")
    if not isinstance(per_arm, Mapping) or set(per_arm) != set(ARM_IDS):
        raise ValueError("Feasibility gate does not cover every frozen arm")
    if any(not isinstance(per_arm[arm_id], Mapping) for arm_id in ARM_IDS):
        raise ValueError("Feasibility per-arm gates must be mappings")
    for arm_id in active:
        if per_arm[arm_id].get("launch_allowed") is not True:
            raise ValueError(f"Active arm {arm_id!r} is not feasibility-authorized")
    for arm_id in deferred:
        if per_arm[arm_id].get("launch_allowed") is not False or per_arm[
            arm_id
        ].get("route") != "blocked":
            raise ValueError(f"Deferred arm {arm_id!r} is not feasibility-blocked")

    arm_results = feasibility_manifest.get("arm_results")
    if not isinstance(arm_results, list):
        raise ValueError("Feasibility manifest has no arm results")
    results_by_arm = {
        row.get("arm_id"): row for row in arm_results if isinstance(row, Mapping)
    }
    if set(results_by_arm) != set(ARM_IDS) or len(arm_results) != len(ARM_IDS):
        raise ValueError("Feasibility results do not cover every frozen arm exactly")
    for arm_id in active:
        training_manifest = validation_results[arm_id].get("training_manifest")
        if not isinstance(training_manifest, Mapping):
            raise ValueError(f"Active arm {arm_id!r} has no training manifest")
        checkpoint_gate = _validate_checkpoint_feasibility_gate(
            training_manifest, arm_id=arm_id
        )
        if (
            checkpoint_gate["artifact_digest"] != artifact_digest
            or checkpoint_gate["arm_gate"]
            != canonical_copy(dict(per_arm[arm_id]))
            or checkpoint_gate["study_identity_digest"]
            != results_by_arm[arm_id].get("study_identity_digest")
        ):
            raise ValueError(
                f"Active arm {arm_id!r} checkpoint used another feasibility gate"
            )
    return {
        "artifact_digest": artifact_digest,
        "active_arm_ids": list(active),
        "deferred_arm_ids": list(deferred),
    }


def _test_readiness_body(
    *,
    protocol_lock: Mapping[str, Any],
    validation_results: Mapping[str, Mapping[str, Any]],
    feasibility_identity: Mapping[str, Any],
) -> dict[str, Any]:
    active = validate_active_arm_ids(feasibility_identity["active_arm_ids"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "locked-before-held-out-test",
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "test_cohort_sha256": protocol_lock["cohorts"]["test"]["cohort_sha256"],
        "active_arm_ids": list(active),
        "deferred_arm_ids": list(feasibility_identity["deferred_arm_ids"]),
        "feasibility_artifact_digest": feasibility_identity["artifact_digest"],
        "selection_lock_sha256_by_arm": {
            arm_id: validation_results[arm_id]["selection_lock"][
                "selection_lock_sha256"
            ]
            for arm_id in active
        },
        "validation_summary_sha256_by_arm": {
            arm_id: validation_results[arm_id]["summary"][SUMMARY_DIGEST_FIELD]
            for arm_id in active
        },
        "validation_wandb_receipt_sha256_by_arm": {
            arm_id: validation_results[arm_id]["wandb_receipt"][
                WANDB_RECEIPT_DIGEST_FIELD
            ]
            for arm_id in active
        },
        "final_checkpoint_sha256_by_arm": {
            arm_id: validation_results[arm_id]["checkpoint"]["checkpoint_sha256"]
            for arm_id in active
        },
    }


def validate_test_readiness(
    test_dir: str | Path,
    *,
    protocol_lock: Mapping[str, Any],
    validation_results: Mapping[str, Mapping[str, Any]],
    feasibility_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = _read_json(
        Path(test_dir) / TEST_READINESS_FILENAME, label="test readiness lock"
    )
    recorded = value.pop("readiness_sha256", None)
    if recorded != sha256_json(value):
        raise ValueError("Test readiness lock changed after it was sealed")
    value["readiness_sha256"] = recorded
    feasibility_identity = _validate_test_feasibility(
        feasibility_manifest, validation_results=validation_results
    )
    expected = _test_readiness_body(
        protocol_lock=protocol_lock,
        validation_results=validation_results,
        feasibility_identity=feasibility_identity,
    )
    if {key: value.get(key) for key in expected} != expected:
        raise ValueError("Test readiness lock no longer matches the frozen opening")
    if set(value) != {*expected, "readiness_sha256"}:
        raise ValueError("Test readiness lock has unexpected fields")
    return value


def prepare_test_opening(
    *,
    protocol_lock_path: str | Path,
    tokenizer_dir: str | Path,
    validation_result_dirs: Sequence[str | Path],
    feasibility_manifest_path: str | Path,
    test_dir: str | Path,
    opening_id: str,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    protocol = load_protocol_lock(
        protocol_lock_path, tokenizer_dir=tokenizer_dir, repo_root=repo_root
    )
    results, locks = load_validation_results(
        validation_result_dirs, protocol_lock=protocol
    )
    feasibility_manifest = load_probe_manifest(feasibility_manifest_path)
    feasibility_identity = _validate_test_feasibility(
        feasibility_manifest, validation_results=results
    )
    active = tuple(feasibility_identity["active_arm_ids"])
    ledger = new_test_opening_ledger(
        opening_id=opening_id,
        protocol_lock_sha256=protocol["protocol_lock_sha256"],
        selection_locks=locks,
        active_arm_ids=active,
        feasibility_artifact_digest=feasibility_identity["artifact_digest"],
    )
    destination = Path(test_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Test opening directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.preparing-", dir=destination.parent
        )
    )
    try:
        create_test_opening_ledger_file(temporary / TEST_LEDGER_FILENAME, ledger)
        readiness_body = _test_readiness_body(
            protocol_lock=protocol,
            validation_results=results,
            feasibility_identity=feasibility_identity,
        )
        readiness = {
            **readiness_body,
            "readiness_sha256": sha256_json(readiness_body),
        }
        _write_json(temporary / TEST_READINESS_FILENAME, readiness, exclusive=True)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "prepared",
        "test_dir": str(destination),
        "ledger_sha256": ledger["ledger_sha256"],
        "readiness_sha256": readiness["readiness_sha256"],
    }


def _advance_ledger(
    path: Path, previous: Mapping[str, Any], next_value: Mapping[str, Any]
):
    replace_test_opening_ledger_file(
        path,
        next_value,
        expected_previous_sha256=previous["ledger_sha256"],
    )
    return load_test_opening_ledger(path)


def _arm_result_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
        "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
        "candidates_sha256": summary["artifacts"]["candidates"]["sha256"],
    }


def _load_existing_test_arm(
    arm_dir: Path,
    *,
    arm_id: str,
    protocol: Mapping[str, Any],
    selected_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = validate_evaluation_summary(
        _read_json(arm_dir / SUMMARY_FILENAME, label="test arm summary"),
        protocol_lock_sha256=protocol["protocol_lock_sha256"],
        split="test",
        arm_id=arm_id,
    )
    if summary["checkpoint"] != selected_checkpoint:
        raise ValueError("Test arm used a checkpoint outside its validation lock")
    _validate_raw_artifacts(arm_dir, summary)
    return summary, read_jsonl_gz(arm_dir / CANDIDATES_FILENAME)


def _atomic_shared_bfs(
    path: Path,
    *,
    records: Sequence[Any],
    available_functions: Any,
    bfs: Mapping[str, Any],
    bfs_provider: BFSProvider,
) -> list[dict[str, Any]]:
    if path.exists():
        rows = read_jsonl_gz(path)
        validate_bfs_rows(rows, records)
        return rows
    temporary = path.with_name(f".{path.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"BFS temporary artifact already exists: {temporary}")
    try:
        rows = bfs_provider(
            records,
            available_functions=available_functions,
            budget=bfs["budget"],
            max_depth=bfs["max_depth"],
            progress=True,
        )
        validate_bfs_rows(rows, records)
        write_jsonl_gz(rows, temporary)
        os.replace(temporary, path)
        return rows
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_combined_test_summary(
    test_dir: Path,
    *,
    protocol: Mapping[str, Any],
    arm_summaries: Mapping[str, Mapping[str, Any]],
    all_candidates: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    bfs_path: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    active = validate_active_arm_ids(ledger["active_arm_ids"])
    deferred = tuple(ledger["deferred_arm_ids"])
    if set(arm_summaries) != set(active):
        raise ValueError("Combined test summary does not cover exactly the active arms")
    funnels = aggregate_candidate_funnels(
        all_candidates,
        expected_task_ids=task_ids,
        active_arm_ids=active,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "split": "test",
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "test_cohort_sha256": protocol["cohorts"]["test"]["cohort_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "test_access_count": ledger["test_access_count"],
        "active_arm_ids": list(active),
        "deferred_arm_ids": list(deferred),
        "feasibility_artifact_digest": ledger["feasibility_artifact_digest"],
        "arm_summary_sha256_by_arm": {
            arm_id: arm_summaries[arm_id][SUMMARY_DIGEST_FIELD] for arm_id in active
        },
        "candidate_funnels": funnels,
        "shared_bfs": {
            "path": BFS_FILENAME,
            "bytes": bfs_path.stat().st_size,
            "sha256": sha256_file(bfs_path),
        },
        "arms": {
            arm_id: {
                "model": arm_summaries[arm_id]["model"],
                "bfs": arm_summaries[arm_id]["bfs"],
                "matched_compute": arm_summaries[arm_id]["matched_compute"],
                "training": arm_summaries[arm_id]["training"],
                "generation_runtime": arm_summaries[arm_id]["generation"]["runtime"],
            }
            for arm_id in active
        },
    }
    summary = _seal_summary(body)
    path = test_dir / TEST_SUMMARY_FILENAME
    if path.exists():
        existing = _read_json(path, label="combined test summary")
        recorded = existing.get(SUMMARY_DIGEST_FIELD)
        unsealed = {
            key: value for key, value in existing.items() if key != SUMMARY_DIGEST_FIELD
        }
        if recorded != sha256_json(unsealed):
            raise ValueError("Combined held-out test summary changed after sealing")
        if existing != summary:
            raise ValueError("Combined held-out test summary already differs")
        return existing
    try:
        _write_json_atomic_exclusive(path, summary)
    except FileExistsError:
        # A concurrent recovery may have won the immutable publish race.  It is
        # acceptable only when it produced the byte-for-byte canonical result.
        existing = _read_json(path, label="combined test summary")
        if existing != summary:
            raise ValueError("Combined held-out test summary already differs") from None
        return existing
    published = _read_json(path, label="combined test summary")
    if published != summary:
        raise RuntimeError("Atomic combined test summary publication changed data")
    return published


def run_test_opening(
    *,
    protocol_lock_path: str | Path,
    corpus_dir: str | Path,
    tokenizer_dir: str | Path,
    validation_result_dirs: Sequence[str | Path],
    feasibility_manifest_path: str | Path,
    test_dir: str | Path,
    reference_corpus_dirs: Sequence[str | Path] = (DEFAULT_REFERENCE_CORPUS_DIR,),
    repo_root: str | Path = REPO_ROOT,
    candidate_provider: CandidateProvider = generate_scratch_candidates,
    bfs_provider: BFSProvider = evaluate_bfs_records,
) -> dict[str, Any]:
    """Open the held-out cohort once and finish or recover all active arms."""

    protocol = load_protocol_lock(
        protocol_lock_path, tokenizer_dir=tokenizer_dir, repo_root=repo_root
    )
    validation, locks = load_validation_results(
        validation_result_dirs, protocol_lock=protocol
    )
    feasibility_manifest = load_probe_manifest(feasibility_manifest_path)
    root = Path(test_dir).resolve()
    readiness = validate_test_readiness(
        root,
        protocol_lock=protocol,
        validation_results=validation,
        feasibility_manifest=feasibility_manifest,
    )
    active = validate_active_arm_ids(readiness["active_arm_ids"])
    deferred = tuple(readiness["deferred_arm_ids"])
    ledger_path = root / TEST_LEDGER_FILENAME
    ledger = load_test_opening_ledger(ledger_path)
    validate_test_opening_ledger(
        ledger,
        protocol_lock_sha256=protocol["protocol_lock_sha256"],
        selection_locks=locks,
        active_arm_ids=active,
        deferred_arm_ids=deferred,
        feasibility_artifact_digest=readiness["feasibility_artifact_digest"],
    )
    for arm_id in deferred:
        if (root / "arms" / arm_id).exists():
            raise ValueError(f"Deferred arm {arm_id!r} cannot be added to this opening")
    if ledger["status"] == "prepared":
        ledger = _advance_ledger(ledger_path, ledger, begin_test_opening(ledger))
    elif ledger["status"] == "interrupted":
        ledger = _advance_ledger(ledger_path, ledger, resume_test_opening(ledger))
    elif ledger["status"] not in {"in_progress", "complete"}:
        raise ValueError("Held-out test ledger is not resumable")

    try:
        # The test records are first loaded only after the durable access-count
        # transition above.  Any failure before it therefore remains unopened.
        basis, records, verification = load_frozen_cohort_records(
            corpus_dir=corpus_dir,
            protocol_lock=protocol,
            split="test",
            reference_corpus_dirs=reference_corpus_dirs,
        )
        functions = basis.as_function_set()
        bfs_path = root / BFS_FILENAME
        if ledger["status"] == "complete" and not bfs_path.is_file():
            raise ValueError("Completed test opening lost its shared BFS artifact")
        bfs_rows = _atomic_shared_bfs(
            bfs_path,
            records=records,
            available_functions=functions,
            bfs=protocol["bfs"],
            bfs_provider=bfs_provider,
        )
        summaries: dict[str, dict[str, Any]] = {}
        all_candidates: list[dict[str, Any]] = []
        for arm_id in active:
            arm_dir = root / "arms" / arm_id
            if ledger["arms"][arm_id]["status"] == "complete" and not (
                arm_dir.is_dir()
            ):
                raise ValueError(
                    f"Recorded test result for {arm_id} is missing; refusing to rescore"
                )
            if arm_dir.exists():
                summary, candidates = _load_existing_test_arm(
                    arm_dir,
                    arm_id=arm_id,
                    protocol=protocol,
                    selected_checkpoint=validation[arm_id]["checkpoint"],
                )
            else:
                candidates, runtime = candidate_provider(
                    records,
                    arm_id=arm_id,
                    checkpoint=Path(validation[arm_id]["checkpoint"]["path"]),
                    tokenizer_dir=Path(tokenizer_dir).resolve(),
                    available_functions=functions,
                    generation=protocol["generation"],
                    progress=True,
                )
                summary, selection = _atomic_arm_result(
                    arm_dir,
                    split="test",
                    arm_id=arm_id,
                    protocol_lock=protocol,
                    checkpoint=validation[arm_id]["checkpoint"],
                    training_manifest=validation[arm_id]["training_manifest"],
                    records=records,
                    candidate_rows=candidates,
                    generation_runtime=runtime,
                    bfs_rows=bfs_rows,
                    write_selection=False,
                )
                if selection is not None:
                    raise AssertionError("Held-out test must not select checkpoints")
            summaries[arm_id] = summary
            all_candidates.extend(candidates)
            result_identity = _arm_result_identity(summary)
            current = load_test_opening_ledger(ledger_path)
            if current["status"] == "complete":
                if current["arms"][arm_id]["result_identity"] != result_identity:
                    raise ValueError(
                        f"Completed test ledger identity differs for {arm_id}"
                    )
                ledger = current
            else:
                updated = record_test_arm_result(
                    current, arm_id=arm_id, result_identity=result_identity
                )
                if updated["ledger_sha256"] != current["ledger_sha256"]:
                    ledger = _advance_ledger(ledger_path, current, updated)
                else:
                    ledger = current

        bfs_identity = {
            "path": BFS_FILENAME,
            "bytes": bfs_path.stat().st_size,
            "sha256": sha256_file(bfs_path),
        }
        current = load_test_opening_ledger(ledger_path)
        completed = complete_test_opening(current, shared_bfs_identity=bfs_identity)
        # Publish the sealed deterministic result against the prospective
        # completed ledger before committing that last state transition.  A
        # process killed in between can reconstruct the same summary from the
        # already durable arm/BFS artifacts and finish without rescoring.
        combined = _write_combined_test_summary(
            root,
            protocol=protocol,
            arm_summaries=summaries,
            all_candidates=all_candidates,
            task_ids=[record.task_id for record in records],
            bfs_path=bfs_path,
            ledger=completed,
        )
        if completed["ledger_sha256"] != current["ledger_sha256"]:
            ledger = _advance_ledger(ledger_path, current, completed)
        else:
            ledger = current
        return {
            "status": "complete",
            "ledger": ledger,
            "summary": combined,
            "verification": verification,
        }
    except Exception:
        current = load_test_opening_ledger(ledger_path)
        # Once a complete summary exists, do not mutate the ledger history: the
        # summary binds the prospective completion digest, and the next run can
        # finish that exact transition.  Atomic publication means the file is
        # either absent or complete.
        if (
            current["status"] == "in_progress"
            and not (root / TEST_SUMMARY_FILENAME).is_file()
        ):
            interrupted = interrupt_test_opening(
                current, reason="evaluation process raised before completion"
            )
            _advance_ledger(ledger_path, current, interrupted)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--protocol-lock", type=Path, required=True)
    validation.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    validation.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    validation.add_argument("--arm-id", choices=ARM_IDS, required=True)
    validation.add_argument("--checkpoint", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    validation.add_argument(
        "--reference-corpus-dir",
        type=Path,
        action="append",
        default=[],
    )
    validation.add_argument("--dry-run", action="store_true")

    prepare = subparsers.add_parser("prepare-test")
    prepare.add_argument("--protocol-lock", type=Path, required=True)
    prepare.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    prepare.add_argument(
        "--validation-result", type=Path, action="append", required=True
    )
    prepare.add_argument("--feasibility-manifest", type=Path, required=True)
    prepare.add_argument("--test-dir", type=Path, required=True)
    prepare.add_argument("--opening-id", required=True)
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)

    test = subparsers.add_parser("test")
    test.add_argument("--protocol-lock", type=Path, required=True)
    test.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    test.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    test.add_argument("--validation-result", type=Path, action="append", required=True)
    test.add_argument("--feasibility-manifest", type=Path, required=True)
    test.add_argument("--test-dir", type=Path, required=True)
    test.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    test.add_argument(
        "--reference-corpus-dir",
        type=Path,
        action="append",
        default=[],
    )

    publish = subparsers.add_parser(
        "publish-wandb",
        help="publish one sealed validation/test arm result without mutating it",
    )
    publish.add_argument("--protocol-lock", type=Path, required=True)
    publish.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    publish.add_argument("--result-dir", type=Path, required=True)
    publish.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validation":
        result = run_validation_evaluation(
            protocol_lock_path=args.protocol_lock,
            corpus_dir=args.corpus_dir,
            tokenizer_dir=args.tokenizer_dir,
            arm_id=args.arm_id,
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            reference_corpus_dirs=tuple(
                args.reference_corpus_dir or (DEFAULT_REFERENCE_CORPUS_DIR,)
            ),
            repo_root=args.repo_root,
            dry_run=args.dry_run,
        )
    elif args.command == "prepare-test":
        result = prepare_test_opening(
            protocol_lock_path=args.protocol_lock,
            tokenizer_dir=args.tokenizer_dir,
            validation_result_dirs=tuple(args.validation_result),
            feasibility_manifest_path=args.feasibility_manifest,
            test_dir=args.test_dir,
            opening_id=args.opening_id,
            repo_root=args.repo_root,
        )
    elif args.command == "test":
        result = run_test_opening(
            protocol_lock_path=args.protocol_lock,
            corpus_dir=args.corpus_dir,
            tokenizer_dir=args.tokenizer_dir,
            validation_result_dirs=tuple(args.validation_result),
            feasibility_manifest_path=args.feasibility_manifest,
            test_dir=args.test_dir,
            reference_corpus_dirs=tuple(
                args.reference_corpus_dir or (DEFAULT_REFERENCE_CORPUS_DIR,)
            ),
            repo_root=args.repo_root,
        )
    else:
        receipt = publish_evaluation_to_wandb(
            protocol_lock_path=args.protocol_lock,
            tokenizer_dir=args.tokenizer_dir,
            result_dir=args.result_dir,
            repo_root=args.repo_root,
        )
        result = {
            "status": receipt["status"],
            "protocol_lock_sha256": receipt["protocol_lock_sha256"],
            "wandb_receipt": receipt,
        }
    printable = {
        "status": result["status"],
        "protocol_lock_sha256": (
            result.get("summary", {}).get("protocol_lock_sha256")
            or result.get("protocol_lock_sha256")
        ),
        "summary_sha256": result.get("summary", {}).get(SUMMARY_DIGEST_FIELD),
        "ledger_sha256": result.get("ledger", {}).get("ledger_sha256")
        or result.get("ledger_sha256"),
        "wandb_url": result.get("wandb_receipt", {}).get("url"),
        "wandb_receipt_sha256": result.get("wandb_receipt", {}).get(
            WANDB_RECEIPT_DIGEST_FIELD
        ),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
