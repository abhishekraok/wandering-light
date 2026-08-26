from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments import evaluate_deep_corpus_sft as evaluation
from experiments.evaluate_deep_corpus_sft import (
    DEFAULT_WANDB_PROJECT,
    EvaluationConfig,
    _checkpoint_identity,
    _log_wandb_run,
    _sha256_file,
    _validate_config,
    aggregate_model_candidates,
    configure_deterministic_cuda_environment,
    configure_unbounded_int_serialization,
    consume_test_selection_lock,
    evaluate_candidate,
    matched_compute_comparison,
    parse_program,
    select_balanced_cohort,
    validate_bfs_rows,
    validate_locked_test_protocol,
    validate_test_selection_lock,
)
from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.basis_set import load_basis_set
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.typed_list import TypedList


@pytest.fixture
def arithmetic_basis() -> FunctionDefSet:
    return FunctionDefSet(
        [
            FunctionDef(
                name="inc",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x + 1",
            ),
            FunctionDef(
                name="double",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x * 2",
            ),
            FunctionDef(
                name="str_len",
                input_type="builtins.str",
                output_type="builtins.int",
                code="return len(x)",
            ),
            FunctionDef(
                name="inverse",
                input_type="builtins.int",
                output_type="builtins.float",
                code="return 1 / x",
            ),
        ]
    )


def _record(
    *,
    distance: int = 2,
    split: str = "test",
    input_value: TypedList | None = None,
    output_value: TypedList | None = None,
    source_index: int = 0,
) -> BasisTaskRecord:
    return BasisTaskRecord.create(
        split=split,
        input_value=input_value or TypedList([1], int),
        output_value=output_value or TypedList([3], int),
        witness_function_ids=["bf:inc:test", "bf:inc:test"],
        witness_function_names=["inc", "inc"],
        basis_set_id="test-basis-v1",
        basis_set_digest="sha256:" + "1" * 64,
        generator="unit-test",
        seed=7,
        source_index=source_index,
        metadata={
            "certified_distance": distance,
            "root_digest": f"sha256:{source_index:064x}",
        },
    )


def _write_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    basis = load_basis_set("default")
    wandb_url = (
        f"https://wandb.ai/abhishekraok-na/{DEFAULT_WANDB_PROJECT}/runs/training-run"
    )
    training_manifest = {
        "status": "complete",
        "base_model": {
            "source": "huggingface",
            "repo_id": "HuggingFaceTB/SmolLM2-360M",
            "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weight_initialization": "pretrained",
            "model_loader": "transformers.AutoModelForCausalLM.from_pretrained",
        },
        "basis_set": basis.identity_dict(),
        "source_corpus": {
            "manifest_digest": "sha256:" + "a" * 64,
            "task_roles": {
                split: {
                    "split": split,
                    "role": role,
                    "records": 1,
                }
                for split, role in {
                    "discovery": "training-only",
                    "validation": "trainer-evaluation-and-tuning-only",
                    "test": "held-out-evaluation-only; never passed to trainer",
                }.items()
            },
        },
        "hyperparameters": {
            "train_split": "discovery",
            "validation_split": "validation",
            "held_out_split": "test",
        },
        "tokenization": {
            role: {
                "complete_labels_only": True,
                "selected_records": 1,
                "retained_records": 1,
                "excluded_records": 0,
            }
            for role in ("train", "validation")
        },
        "wandb": {
            "entity": "abhishekraok-na",
            "project": DEFAULT_WANDB_PROJECT,
            "url": wandb_url,
        },
    }
    json_files = {
        "config.json": {},
        "tokenizer_config.json": {},
        "trainer_state.json": {},
        "training_args.json": {},
        "basis_set.json": basis.to_manifest(),
        "corpus_manifest.json": {},
        "training_manifest.json": training_manifest,
    }
    for name, value in json_files.items():
        (checkpoint / name).write_text(json.dumps(value), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model-weights")
    (checkpoint / "training_args.bin").write_bytes(b"trainer-metadata")
    (checkpoint / "wandb_run.url").write_text(wandb_url, encoding="utf-8")
    return checkpoint, basis, training_manifest


def test_deterministic_cuda_environment_uses_supported_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert configure_deterministic_cuda_environment() == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert configure_deterministic_cuda_environment() == ":16:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unsupported")
    assert configure_deterministic_cuda_environment() == ":4096:8"


def test_evaluation_defaults_to_validation_and_test_requires_a_lock(tmp_path):
    config = EvaluationConfig(
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        dry_run=True,
    )
    assert config.split == "validation"
    _validate_config(config)

    with pytest.raises(ValueError, match="selection-lock"):
        _validate_config(
            EvaluationConfig(
                checkpoint=config.checkpoint,
                output_dir=config.output_dir,
                split="test",
                dry_run=False,
                wandb_run_name="test",
            )
        )
    lock_path = tmp_path / "selection-lock.json"
    with pytest.raises(ValueError, match="matched BFS"):
        _validate_config(
            EvaluationConfig(
                checkpoint=config.checkpoint,
                output_dir=config.output_dir,
                split="test",
                selection_lock=lock_path,
                run_bfs=False,
                wandb_run_name="test",
            )
        )


def test_huge_integer_output_is_serialized_without_external_env(arithmetic_basis):
    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("Python runtime has no integer conversion guard")
    previous = sys.get_int_max_str_digits()
    huge_basis = FunctionDefSet(
        [
            FunctionDef(
                name="huge",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return 10 ** 5000",
            )
        ]
    )
    try:
        sys.set_int_max_str_digits(4300)
        row = evaluate_candidate(
            _record(distance=1, output_value=TypedList([0], int)),
            sample_index=0,
            response="huge",
            available_functions=huge_basis,
        )
        assert configure_unbounded_int_serialization() == 0
        assert row["executor_success"] is True
        assert len(row["actual_output"]) > 4_300
    finally:
        sys.set_int_max_str_digits(previous)


def _write_selection_lock(tmp_path, checkpoint):
    signature = [
        {"name": path.name, "sha256": _sha256_file(path)}
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    ]
    generation = {
        "samples_per_task": 64,
        "sample_counts": [1, 8, 64],
        "seed": 20260816,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 64,
        "batch_size": 32,
        "cost_definition": evaluation.MODEL_COST_DEFINITION,
    }
    payload = {
        "status": "locked-before-held-out-test-scoring",
        "locked_at_utc": "2026-08-25T06:00:00+00:00",
        "discipline": {
            "selection_data": "validation summaries only",
            "held_out_test_integrity_read_allowed": True,
            "held_out_test_model_scored": False,
            "held_out_test_outcomes_used_for_selection": False,
        },
        "frozen_evaluator": {
            "path": str(Path(evaluation.__file__).resolve()),
            "sha256": _sha256_file(Path(evaluation.__file__)),
        },
        "validation_identity": {
            "corpus_manifest_digest": "sha256:" + "a" * 64,
            "basis_set_digest": "sha256:" + "b" * 64,
            "cohort_task_ids_digest": "sha256:" + "c" * 64,
            "cohort_protocol": {
                "seed": 20260816,
                "selection": "distance-stratified-within-frozen-root-split",
                "tasks": 96,
                "by_certified_distance": {
                    str(distance): 12 for distance in range(1, 9)
                },
            },
            "generation_protocol": generation,
        },
        "champion": {
            "checkpoint_signature": signature,
            "checkpoint_training_manifest_sha256": _sha256_file(
                checkpoint / "training_manifest.json"
            ),
        },
    }
    path = tmp_path / "selection-lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_test_lock_freezes_protocol_and_is_single_use(tmp_path):
    checkpoint, _, _ = _write_checkpoint(tmp_path)
    lock_path = _write_selection_lock(tmp_path, checkpoint)
    access = validate_test_selection_lock(lock_path, checkpoint=checkpoint)
    config = EvaluationConfig(
        checkpoint=checkpoint,
        output_dir=tmp_path / "test-output",
        split="test",
        selection_lock=lock_path,
        wandb_run_name="test",
    )
    cohort = {
        "seed": config.seed,
        "selection": "distance-stratified-within-frozen-root-split",
        "tasks": 96,
        "by_certified_distance": {str(distance): 12 for distance in range(1, 9)},
    }
    corpus = {
        "manifest_digest": "sha256:" + "a" * 64,
        "basis_set_digest": "sha256:" + "b" * 64,
    }
    validate_locked_test_protocol(
        access,
        config=config,
        corpus_identity=corpus,
        cohort_identity=cohort,
    )
    with pytest.raises(ValueError, match="locked protocol"):
        validate_locked_test_protocol(
            access,
            config=EvaluationConfig(**{**config.__dict__, "temperature": 0.7}),
            corpus_identity=corpus,
            cohort_identity=cohort,
        )

    access["test_scoring_started_at_utc"] = "2026-08-25T06:01:00+00:00"
    receipt = consume_test_selection_lock(
        lock_path,
        test_access=access,
        output_dir=config.output_dir,
    )
    assert receipt["selection_lock_sha256"] == access["selection_lock_sha256"]
    with pytest.raises(RuntimeError, match="already been consumed"):
        consume_test_selection_lock(
            lock_path,
            test_access=access,
            output_dir=config.output_dir,
        )


def test_strict_program_parser_distinguishes_malformed_from_empty_program(
    arithmetic_basis,
):
    valid = parse_program(" inc, double ", arithmetic_basis)
    assert valid.valid
    assert valid.names == ("inc", "double")
    assert [function.name for function in valid.functions] == ["inc", "double"]

    assert not parse_program("", arithmetic_basis).valid
    assert not parse_program("inc,", arithmetic_basis).valid
    unknown = parse_program("inc,not_in_basis", arithmetic_basis)
    assert not unknown.valid
    assert unknown.names == ("inc", "not_in_basis")


def test_checkpoint_identity_accepts_runner_schema_and_only_hashes_model_weights(
    tmp_path,
):
    checkpoint, basis, _ = _write_checkpoint(tmp_path)

    identity = _checkpoint_identity(
        checkpoint,
        "sha256:" + "a" * 64,
        basis,
        project=DEFAULT_WANDB_PROJECT,
    )

    assert [weight["name"] for weight in identity["weights"]] == ["model.safetensors"]


def test_checkpoint_identity_rejects_split_leakage_and_invalid_initialization(tmp_path):
    checkpoint, basis, manifest = _write_checkpoint(tmp_path)
    manifest["source_corpus"]["task_roles"]["test"]["role"] = "training-only"
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="task-role provenance"):
        _checkpoint_identity(
            checkpoint,
            "sha256:" + "a" * 64,
            basis,
            project=DEFAULT_WANDB_PROJECT,
        )

    manifest["source_corpus"]["task_roles"]["test"]["role"] = (
        "held-out-evaluation-only; never passed to trainer"
    )
    manifest["base_model"]["repo_id"] = "abhishekraok/previous-checkpoint"
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid pretrained base"):
        _checkpoint_identity(
            checkpoint,
            "sha256:" + "a" * 64,
            basis,
            project=DEFAULT_WANDB_PROJECT,
        )

    manifest["base_model"]["repo_id"] = "HuggingFaceTB/SmolLM2-360M"
    manifest["base_model"]["weight_initialization"] = "random"
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pretrained weights"):
        _checkpoint_identity(
            checkpoint,
            "sha256:" + "a" * 64,
            basis,
            project=DEFAULT_WANDB_PROJECT,
        )


def test_alternative_program_is_execution_correct_but_not_exact_match(
    arithmetic_basis,
):
    row = evaluate_candidate(
        _record(),
        sample_index=0,
        response="double,inc",
        available_functions=arithmetic_basis,
    )

    assert row["parse_valid"]
    assert row["executor_success"]
    assert row["execution_correct"]
    assert not row["exact_match"]
    assert row["charged_executions"] == 2
    assert row["actual_output"] == TypedList([3], int).to_string()


def test_invalid_and_failing_program_cost_semantics_are_explicit(arithmetic_basis):
    record = _record()
    malformed = evaluate_candidate(
        record,
        sample_index=0,
        response="unknown",
        available_functions=arithmetic_basis,
    )
    type_failure = evaluate_candidate(
        record,
        sample_index=1,
        response="str_len,inc",
        available_functions=arithmetic_basis,
    )
    runtime_failure = evaluate_candidate(
        _record(
            input_value=TypedList([0], int),
            output_value=TypedList([1], int),
            source_index=1,
        ),
        sample_index=2,
        response="inverse,inc",
        available_functions=arithmetic_basis,
    )

    assert not malformed["parse_valid"]
    assert malformed["charged_executions"] == 0
    assert type_failure["parse_valid"]
    assert not type_failure["executor_success"]
    assert type_failure["failed_at_step"] == 0
    assert type_failure["charged_executions"] == 2
    assert runtime_failure["parse_valid"]
    assert not runtime_failure["executor_success"]
    assert runtime_failure["failed_at_step"] == 0
    assert runtime_failure["charged_executions"] == 2


def _ordered_candidate_rows(record, arithmetic_basis):
    responses = ["inc", "double,inc", "inc,inc", *([""] * 61)]
    return [
        evaluate_candidate(
            record,
            sample_index=index,
            response=response,
            available_functions=arithmetic_basis,
        )
        for index, response in enumerate(responses)
    ]


def test_solve_prefixes_share_one_draw_and_stop_cost_at_first_execution_hit(
    arithmetic_basis,
):
    record = _record(distance=8)
    rows = _ordered_candidate_rows(record, arithmetic_basis)

    summary, per_task = aggregate_model_candidates(
        rows,
        expected_records=[record],
        sample_counts=(1, 8, 64),
    )

    solve_1 = per_task[record.task_id][1]
    solve_8 = per_task[record.task_id][8]
    solve_64 = per_task[record.task_id][64]
    assert not solve_1["execution_correct"]
    assert solve_1["executions_spent"] == 1
    assert solve_8["execution_correct"]
    assert solve_8["executions_spent"] == 3
    assert solve_8["execution_hit_sample"] == 2
    assert solve_8["exact_match"]
    assert solve_64 == {**solve_8, "samples_available": 64}
    assert summary["by_certified_distance"]["8"]["64"]["solve_rate"] == 1.0
    assert summary["by_certified_distance"]["8"]["64"]["exact_match_rate"] == 1.0


def test_aggregation_rejects_missing_or_non_prefix_samples(arithmetic_basis):
    record = _record(distance=8)
    rows = _ordered_candidate_rows(record, arithmetic_basis)

    with pytest.raises(ValueError, match="complete ordered prefix"):
        aggregate_model_candidates(
            rows[:-1], expected_records=[record], sample_counts=(1, 8, 64)
        )
    altered = [dict(row) for row in rows]
    altered[0]["task_id"] = "not-the-cohort-task"
    with pytest.raises(ValueError, match="unexpected task"):
        aggregate_model_candidates(
            altered, expected_records=[record], sample_counts=(1, 8, 64)
        )


def test_matched_compute_uses_the_same_task_and_reports_execution_ratio(
    arithmetic_basis,
):
    record = _record(distance=8)
    rows = _ordered_candidate_rows(record, arithmetic_basis)
    _, per_task = aggregate_model_candidates(
        rows, expected_records=[record], sample_counts=(1, 8, 64)
    )
    bfs_rows = [
        {
            "task_id": record.task_id,
            "split": "test",
            "certified_distance": 8,
            "success": True,
            "executions": 100,
            "termination": "solved",
            "seconds": 1.0,
        }
    ]

    matched = matched_compute_comparison(
        per_task=per_task,
        bfs_rows=bfs_rows,
        records=[record],
        sample_counts=(1, 8, 64),
    )

    primary = matched["7-8"]["matched"]["64"]
    assert primary["solved"] == 1
    assert primary["total_executions_spent"] == 3
    assert primary["bfs_uniform_budget_for_at_least_model_solve_count"] == 100
    assert primary["bfs_total_executions_for_at_least_model_solve_count"] == 100
    assert primary["bfs_to_model_total_execution_ratio"] == pytest.approx(100 / 3)
    assert primary["model_cheaper_than_bfs_at_equal_or_higher_solve_count"]


def test_equal_execution_comparison_uses_bfs_actual_capped_spend():
    records = [
        _record(
            distance=8,
            input_value=TypedList([source_index + 1], int),
            output_value=TypedList([source_index + 3], int),
            source_index=source_index,
        )
        for source_index in range(2)
    ]
    solved_at_sixty = {
        "execution_correct": True,
        "exact_match": False,
        "executions_spent": 60,
        "execution_hit_cost": 60,
        "execution_hit_sample": 1,
        "samples_available": 64,
    }
    per_task = {
        record.task_id: {
            samples: {**solved_at_sixty, "samples_available": samples}
            for samples in (1, 8, 64)
        }
        for record in records
    }
    bfs_rows = [
        {
            "task_id": record.task_id,
            "split": "test",
            "certified_distance": 8,
            "success": True,
            "executions": executions,
            "termination": "solved",
            "seconds": 0.1,
        }
        for record, executions in zip(records, (1, 100), strict=True)
    ]

    matched = matched_compute_comparison(
        per_task=per_task,
        bfs_rows=bfs_rows,
        records=records,
        sample_counts=(1, 8, 64),
    )["8"]["matched"]["64"]

    # A per-task budget of 100 spends only (1 + 100) / 2 = 50.5 on BFS,
    # so it is still below the model's mean spend of 60 and solves both tasks.
    assert matched["bfs_uniform_budget_at_or_below_model_mean_execution_spend"] == 100
    assert matched["bfs_mean_executions_at_model_mean_execution_spend"] == 50.5
    assert matched["bfs_solve_rate_at_model_mean_execution_spend"] == 1.0


def test_bfs_validation_requires_exact_task_set_and_complete_solutions():
    record = _record(distance=8)
    successful = {
        "task_id": record.task_id,
        "split": "test",
        "certified_distance": 8,
        "success": True,
        "executions": 10,
        "termination": "solved",
        "seconds": 0.1,
    }
    assert (
        validate_bfs_rows([successful], [record])["by_certified_distance"]["8"][
            "solve_rate"
        ]
        == 1.0
    )

    with pytest.raises(ValueError, match="do not match"):
        validate_bfs_rows([{**successful, "task_id": "other"}], [record])
    with pytest.raises(ValueError, match="not an exhaustive baseline"):
        validate_bfs_rows(
            [{**successful, "success": False, "termination": "budget"}], [record]
        )


def test_balanced_cohort_is_deterministic_and_never_changes_split_membership():
    records = []
    for distance in range(1, 9):
        for offset in range(3):
            source_index = distance * 10 + offset
            records.append(
                _record(
                    distance=distance,
                    split="test",
                    input_value=TypedList([source_index], int),
                    output_value=TypedList([source_index + 2], int),
                    source_index=source_index,
                )
            )

    first = select_balanced_cohort(records, tasks_per_distance=2, seed=19)
    second = select_balanced_cohort(records, tasks_per_distance=2, seed=19)
    assert [record.task_id for record in first] == [record.task_id for record in second]
    assert all(record.split == "test" for record in first)
    assert {
        distance: sum(
            record.metadata["certified_distance"] == distance for record in first
        )
        for distance in range(1, 9)
    } == dict.fromkeys(range(1, 9), 2)


def test_validation_wandb_logging_supports_model_only_summary(tmp_path, monkeypatch):
    import wandb

    metric = {
        "solve_rate": 0.25,
        "exact_match_rate": 0.125,
        "mean_executions_spent": 3.0,
    }
    model = {
        "sample_counts": [1, 8, 64],
        "by_certified_distance": {
            str(distance): {str(samples): dict(metric) for samples in (1, 8, 64)}
            for distance in range(1, 9)
        },
        "distance_7_8": {str(samples): dict(metric) for samples in (1, 8, 64)},
    }
    summary = {
        "corpus": {
            "manifest_digest": "sha256:" + "1" * 64,
            "basis_set_digest": "sha256:" + "2" * 64,
        },
        "cohort": {"task_ids_digest": "sha256:" + "3" * 64},
        "checkpoint": {"path": "/checkpoint"},
        "model": model,
        "matched_compute": None,
        "primary": model["distance_7_8"]["64"],
        "verdict": {"kill_criterion_triggered": None},
    }

    class DummyArtifact:
        def add_file(self, *args, **kwargs):
            pass

    class DummyRun:
        url = (
            "https://wandb.ai/abhishekraok-na/"
            f"{DEFAULT_WANDB_PROJECT}/runs/validation-run"
        )
        id = "validation-run"

        def __init__(self):
            self.logs = []
            self.summary = {}

        def log(self, value):
            self.logs.append(value)

        def log_artifact(self, artifact):
            self.artifact = artifact

        def finish(self):
            self.finished = True

    run = DummyRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(wandb, "Table", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb, "Artifact", lambda **kwargs: DummyArtifact())
    config = EvaluationConfig(
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path,
        split="validation",
        wandb_run_name="validation",
    )

    url = _log_wandb_run(
        config=config,
        summary=summary,
        candidate_rows=[],
        bfs_rows=[],
        output_dir=tmp_path,
    )

    assert url == run.url
    assert run.finished is True
    assert not any(key.startswith("eval/bfs") for logged in run.logs for key in logged)
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["wandb"]["url"] == run.url
