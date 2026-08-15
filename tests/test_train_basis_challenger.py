import json

import pytest

from experiments import basis_library_pilot as pilot
from experiments import train_basis_challenger as challenger
from wandering_light.basis_dataset import BasisTaskRecord, write_basis_task_records
from wandering_light.basis_set import load_basis_set
from wandering_light.typed_list import TypedList


def _write_corpus(tmp_path, *, witness_name="inc", output_values=None):
    basis = load_basis_set("default")
    function = next(item for item in basis.functions if item.name == "inc")
    output_values = output_values or [2, 3]
    record = BasisTaskRecord.create(
        split="discovery",
        input_value=TypedList([1, 2]),
        output_value=TypedList(output_values),
        witness_function_ids=[function.function_id],
        witness_function_names=[witness_name],
        basis_set_id=basis.basis_set_id,
        basis_set_digest=basis.digest,
        generator="test",
        seed=17,
        source_index=0,
        metadata={
            "input_type": "builtins.int",
            "requested_witness_length": 1,
        },
    )
    corpus_dir = tmp_path / "corpus"
    split_path = corpus_dir / "discovery.jsonl.gz"
    write_basis_task_records([record], split_path)
    witness_coverage = [
        {
            "function_id": item.function_id,
            "function_name": item.name,
            "total_occurrences": int(item.function_id == function.function_id),
            "task_coverage": int(item.function_id == function.function_id),
        }
        for item in basis.functions
    ]
    manifest = {
        "schema_version": 1,
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "global_task_count": 1,
        "balance": {
            "dimensions": ["input_type", "requested_witness_length"],
            "input_types": ["builtins.int"],
            "witness_lengths": [1],
            "maximum_cell_count_difference": 0,
        },
        "splits": {
            "discovery": {
                "path": split_path.name,
                "sha256": pilot._sha256_file(split_path),
                "size": 1,
                "by_input_type": {"builtins.int": 1},
                "by_witness_length": {"1": 1},
                "witness_function_coverage": {
                    "functions_exposed": 1,
                    "functions_available": len(basis.functions),
                    "total_witness_steps": 1,
                    "functions": witness_coverage,
                },
            }
        },
    }
    manifest["manifest_digest"] = pilot._sha256_bytes(
        pilot._canonical_json(manifest).encode()
    )
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return corpus_dir


def _write_dummy_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    return model


def _write_verified_results(corpus_dir, tmp_path, *, solution_names=("inc",)):
    manifest, records = pilot.load_corpus(corpus_dir, splits=("discovery",))
    record = records[0]
    basis = load_basis_set(manifest["basis_set_id"])
    by_name = {function.name: function for function in basis.functions}
    solution = [by_name[name] for name in solution_names]
    row = {
        "task_id": record.task_id,
        "split": record.split,
        "input_type": record.metadata["input_type"],
        "witness_length": record.witness_length,
        "success": True,
        "solution_length": len(solution),
        "solution_function_ids": [function.function_id for function in solution],
        "solution_function_names": [function.name for function in solution],
        "batch_index": 0,
        "error": None,
    }
    result_dir = tmp_path / "evaluation"
    results_path = result_dir / "results.jsonl.gz"
    pilot._write_jsonl_gzip([row], results_path)
    aggregate = {
        "schema_version": 1,
        "task_source_basis_set_id": basis.basis_set_id,
        "task_source_basis_set_digest": basis.digest,
        "evaluation_basis_set_id": basis.basis_set_id,
        "evaluation_basis_set_digest": basis.digest,
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "corpus_manifest_digest": manifest["manifest_digest"],
        "evaluated_splits": ["discovery"],
        "model": {
            "canonical_hf_repo": challenger.DEFAULT_HF_CHECKPOINT,
            "canonical_hf_revision": challenger.DEFAULT_HF_REVISION,
            "local_tree_digest": "sha256:" + "1" * 64,
        },
        "overall": {
            "tasks": 1,
            "successes": 1,
            "failures": 0,
            "solve_rate": 1.0,
            "mean_solution_length_success": len(solution),
        },
        "by_split": {
            "discovery": {
                "tasks": 1,
                "successes": 1,
                "failures": 0,
                "solve_rate": 1.0,
                "mean_solution_length_success": len(solution),
            }
        },
        "by_input_type": {
            "builtins.int": {
                "tasks": 1,
                "successes": 1,
                "failures": 0,
                "solve_rate": 1.0,
                "mean_solution_length_success": len(solution),
            }
        },
        "by_witness_length": {
            "1": {
                "tasks": 1,
                "successes": 1,
                "failures": 0,
                "solve_rate": 1.0,
                "mean_solution_length_success": len(solution),
            }
        },
    }
    aggregate_path = result_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results_path, aggregate_path, row


def _config(tmp_path, corpus_dir, model, **overrides):
    values = {
        "corpus_dir": corpus_dir,
        "output_dir": tmp_path / "artifact",
        "base_model": str(model),
        "dry_run": True,
        "dry_run_samples": 1,
    }
    values.update(overrides)
    return challenger.ChallengerConfig(**values)


def test_personal_wandb_entity_is_a_hard_gate():
    assert (
        challenger.validate_personal_wandb_entity("abhishekraok-na")
        == "abhishekraok-na"
    )
    with pytest.raises(ValueError, match="personal entity"):
        challenger.validate_personal_wandb_entity("ai2-llm")
    with pytest.raises(ValueError, match="personal entity"):
        challenger.validate_personal_wandb_entity("allenai")


def test_strict_corpus_builds_existing_completion_only_prompt(tmp_path):
    corpus_dir = _write_corpus(tmp_path)
    model = _write_dummy_model(tmp_path)
    basis, prepared = challenger.prepare_training_corpus(
        _config(tmp_path, corpus_dir, model)
    )

    assert basis.basis_set_id == "wl-core-v1"
    assert prepared.total_validated_records == prepared.selected_records == 1
    row = prepared.rows[0]
    assert "Available functions" not in row["prompt"]
    assert row["prompt"] == "Input: [1, 2]\nTarget Output: [2, 3]\n\nAnswer:"
    assert row["completion"] == "inc"
    assert prepared.source["split_files"]["discovery"]["sha256"].startswith("sha256:")


def test_strict_corpus_rejects_mismatched_witness_identity(tmp_path):
    corpus_dir = _write_corpus(tmp_path, witness_name="dec")
    model = _write_dummy_model(tmp_path)
    with pytest.raises(ValueError, match="name mismatch"):
        challenger.prepare_training_corpus(_config(tmp_path, corpus_dir, model))


def test_strict_corpus_rejects_witness_that_does_not_reproduce_output(tmp_path):
    corpus_dir = _write_corpus(tmp_path, output_values=[3, 4])
    model = _write_dummy_model(tmp_path)
    with pytest.raises(ValueError, match="does not reproduce"):
        challenger.prepare_training_corpus(_config(tmp_path, corpus_dir, model))


def test_exact_base_model_identity_for_local_and_hf(tmp_path):
    model = _write_dummy_model(tmp_path)
    local = challenger.resolve_base_model(str(model), None)
    assert local.revision is None
    assert local.local_files["model.safetensors"]["sha256"].startswith("sha256:")

    remote = challenger.resolve_base_model("abhishekraok/model", "a" * 40)
    assert remote.revision == "a" * 40
    with pytest.raises(ValueError, match="40-hex"):
        challenger.resolve_base_model("abhishekraok/model", "main")


def test_dry_run_never_calls_training_or_creates_output(tmp_path, monkeypatch):
    corpus_dir = _write_corpus(tmp_path)
    model = _write_dummy_model(tmp_path)
    config = _config(tmp_path, corpus_dir, model)

    def fail_training(**kwargs):
        raise AssertionError(f"training was called: {kwargs}")

    monkeypatch.setattr(challenger, "_train", fail_training)
    manifest = challenger.run_challenger(config)

    assert manifest["status"] == "dry-run"
    assert manifest["preview"]["examples"][0]["completion"] == "inc"
    assert manifest["basis_set"]["basis_set_digest"].startswith("sha256:")
    assert manifest["source_corpus"]["manifest_file_sha256"].startswith("sha256:")
    assert manifest["base_model"]["local_files"]["model.safetensors"][
        "sha256"
    ].startswith("sha256:")
    assert not config.output_dir.exists()


def test_sft_config_disables_checkpoints_and_uses_completion_loss(tmp_path):
    config = challenger.ChallengerConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "artifact",
        precision="fp32",
        wandb_run_name="test",
    )
    training_args = challenger.build_sft_config(config, report_to_wandb=True)
    assert str(training_args.save_strategy) in {"no", "SaveStrategy.NO"}
    assert training_args.completion_only_loss is True
    assert training_args.full_determinism is True
    assert training_args.seed == training_args.data_seed == 42
    assert training_args.report_to == ["wandb"]


def test_weights_only_guard_rejects_disposable_training_state(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"weights")
    challenger.assert_weights_only_artifact(artifact)

    (artifact / "optimizer.pt").write_bytes(b"state")
    with pytest.raises(RuntimeError, match=r"optimizer\.pt"):
        challenger.assert_weights_only_artifact(artifact)


def test_verified_results_reexecute_and_record_exact_common_tasks(tmp_path):
    corpus_dir = _write_corpus(tmp_path)
    results_path, aggregate_path, _row = _write_verified_results(corpus_dir, tmp_path)
    model = _write_dummy_model(tmp_path)
    basis, prepared = challenger.prepare_training_corpus(
        _config(
            tmp_path,
            corpus_dir,
            model,
            verified_results=results_path,
        )
    )

    assert basis.basis_set_id == "wl-core-v1"
    assert prepared.rows[0]["completion"] == "inc"
    assert prepared.source["training_data_mode"] == (
        "verified-default-solver-successes"
    )
    assert "common_task_ids" not in prepared.source
    verified = prepared.source["verified_results"]
    assert verified["sha256"] == challenger._sha256_file(results_path)
    assert verified["aggregate_sha256"] == challenger._sha256_file(aggregate_path)
    assert prepared.source["rewrite_counts"]["common_training_tasks"]["mode"] == (
        "retain-source-solutions"
    )


def test_verified_results_rewrite_direct_child_from_provenance(tmp_path):
    corpus_dir = _write_corpus(tmp_path)
    results_path, _, _ = _write_verified_results(
        corpus_dir,
        tmp_path,
        solution_names=("inc", "identity_int"),
    )
    model = _write_dummy_model(tmp_path)
    basis, prepared = challenger.prepare_training_corpus(
        _config(
            tmp_path,
            corpus_dir,
            model,
            basis_set_id="pilot-compressed",
            verified_results=results_path,
        )
    )

    assert basis.basis_set_id == "wl-pilot-compressed-v1"
    assert prepared.rows[0]["completion"] == "inc"
    rewrite = prepared.source["rewrite_counts"]["common_training_tasks"]
    assert rewrite["events"] == {"drop:identity_int": 1}
    assert rewrite["source_steps"] == 2
    assert rewrite["target_steps"] == 1
    assert rewrite["changed_records"] == 1


def test_verified_results_reject_function_identity_and_aggregate_mismatch(tmp_path):
    corpus_dir = _write_corpus(tmp_path)
    results_path, aggregate_path, row = _write_verified_results(corpus_dir, tmp_path)
    model = _write_dummy_model(tmp_path)
    row["solution_function_names"] = ["dec"]
    pilot._write_jsonl_gzip([row], results_path)
    with pytest.raises(ValueError, match="outside the source basis"):
        challenger.prepare_training_corpus(
            _config(
                tmp_path,
                corpus_dir,
                model,
                verified_results=results_path,
            )
        )

    results_path, aggregate_path, _ = _write_verified_results(corpus_dir, tmp_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["corpus_manifest_digest"] = "sha256:" + "0" * 64
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(ValueError, match="corpus manifest digest"):
        challenger.prepare_training_corpus(
            _config(
                tmp_path,
                corpus_dir,
                model,
                verified_results=results_path,
            )
        )


def test_provenance_rewrite_rejects_empty_target_program():
    source = load_basis_set("wl-core-v1")
    target = load_basis_set("pilot-compressed")
    rules = challenger._derive_rewrite_rules(source, target)
    source_runtime = source.as_function_set()
    identity = source_runtime.name_to_function["identity_int"]

    with pytest.raises(ValueError, match="empty target program"):
        challenger._rewrite_verified_solution([identity], rules)
