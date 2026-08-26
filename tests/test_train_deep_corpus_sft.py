import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import generate_deep_corpus as deep
from experiments import train_deep_corpus_sft as sft
from wandering_light.basis_dataset import BasisTaskRecord, write_basis_task_records
from wandering_light.basis_set import load_basis_set
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList


def _record(
    *,
    split,
    input_values,
    witness_names,
    source_index,
    root_index,
    seed=17,
    recorded_names=None,
):
    basis = load_basis_set("default")
    available = basis.as_function_set()
    functions = [available.name_to_function[name] for name in witness_names]
    input_value = TypedList(input_values)
    execution = Executor(available).execute_trajectory(
        TrajectorySpec(input_value, FunctionDefList(functions))
    )
    assert execution.success and execution.trajectory is not None
    return BasisTaskRecord.create(
        split=split,
        input_value=input_value,
        output_value=execution.trajectory.output,
        witness_function_ids=[
            function.metadata["basis_function_id"] for function in functions
        ],
        witness_function_names=recorded_names or witness_names,
        basis_set_id=basis.basis_set_id,
        basis_set_digest=basis.digest,
        generator=deep.GENERATOR_NAME,
        seed=seed,
        source_index=source_index,
        metadata={
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "certified_distance": len(functions),
            "certification": deep.CERTIFICATION_COMPLETE,
            "expansion_certified_depth": len(functions),
            "optimal_first_action_ids": [functions[0].metadata["basis_function_id"]],
            "optimal_first_action_names": [functions[0].name],
            "optimal_first_actions_complete": True,
            "optimal_last_action_ids": [functions[-1].metadata["basis_function_id"]],
            "optimal_last_actions_complete": True,
            "root_index": root_index,
            "root_digest": sft._sha256_bytes(input_value.to_string().encode("utf-8")),
            "distance_shell_size": 1,
        },
    )


def _write_corpus(
    tmp_path,
    *,
    bad_test_witness=False,
    leaking_root=False,
    wrong_basis_digest=False,
):
    basis = load_basis_set("default")
    discovery = [
        _record(
            split="discovery",
            input_values=[1, 2],
            witness_names=["inc"],
            source_index=0,
            root_index=0,
        ),
        _record(
            split="discovery",
            input_values=[10, 20],
            witness_names=["inc"] * 4,
            source_index=1,
            root_index=1,
        ),
    ]
    validation_input = [3, 4]
    validation = [
        _record(
            split="validation",
            input_values=validation_input,
            witness_names=["inc"],
            source_index=0,
            root_index=2,
        )
    ]
    test_input = validation_input if leaking_root else [5, 6]
    test = [
        _record(
            split="test",
            input_values=test_input,
            witness_names=["inc"] * (2 if leaking_root else 1),
            recorded_names=["dec"] if bad_test_witness else None,
            source_index=0,
            root_index=2 if leaking_root else 3,
        )
    ]
    records_by_split = {
        "discovery": discovery,
        "validation": validation,
        "test": test,
    }
    corpus_dir = tmp_path / "corpus"
    split_metadata = {}
    split_roots = {}
    for split in deep.SPLIT_ORDER:
        records = records_by_split[split]
        path = corpus_dir / f"{split}.jsonl.gz"
        write_basis_task_records(records, path)
        roots = sorted({record.metadata["root_index"] for record in records})
        root_digests = {record.metadata["root_digest"] for record in records}
        split_metadata[split] = {
            "path": path.name,
            "sha256": deep._sha256_file(path),
            "size": len(records),
            "roots": len(roots),
            "root_digests": len(root_digests),
            "seed": 17,
        }
        split_roots[split] = roots
    manifest = {
        "schema_version": deep.PIPELINE_SCHEMA_VERSION,
        "generator": deep.GENERATOR_NAME,
        "distance_semantics": (
            "shortest number of basis applications from input to any output equal "
            "to the recorded one under TypedList.__eq__"
        ),
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": (
            "sha256:" + "0" * 64 if wrong_basis_digest else basis.digest
        ),
        "seed": 17,
        "config": {"min_distance": 1},
        "split_policy": ("by root: every task from one root shares that root's split"),
        "global_task_count": sum(len(records) for records in records_by_split.values()),
        "splits": split_metadata,
        "split_roots": split_roots,
    }
    manifest["manifest_digest"] = deep._sha256_bytes(
        deep._canonical_json(manifest).encode("utf-8")
    )
    manifest["hub"] = {
        "repo_id": "abhishekraok/wandering-light-deep-corpus-v1",
        "repo_type": "dataset",
        "revision": "f" * 40,
    }
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return corpus_dir


def _config(tmp_path, corpus_dir, **overrides):
    values = {
        "output_dir": tmp_path / "artifact",
        "corpus_dir": corpus_dir,
        "dry_run": True,
        "dry_run_samples": 2,
    }
    values.update(overrides)
    return sft.DeepCorpusSFTConfig(**values)


def test_import_does_not_load_model_or_wandb_stack():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import experiments.train_deep_corpus_sft; "
            "print(*(name in sys.modules for name in "
            "('torch', 'transformers', 'trl', 'wandb')))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False False False False"


def test_pretrained_base_model_is_exactly_pinned_and_explicit():
    assert sft.DEFAULT_WANDB_PROJECT == "wandering-light-deep-corpus-v1"
    assert sft.DEFAULT_BASE_MODEL_REPO == "HuggingFaceTB/SmolLM2-360M"
    assert sft.DEFAULT_BASE_MODEL_REVISION == "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
    assert sft.DEFAULT_MAX_LENGTH == 2048
    spec = sft.resolve_base_model(
        sft.DEFAULT_BASE_MODEL_REPO,
        sft.DEFAULT_BASE_MODEL_REVISION,
    )
    assert spec.to_dict() == {
        "source": "huggingface",
        "repo_id": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "weight_initialization": "pretrained",
        "model_loader": "transformers.AutoModelForCausalLM.from_pretrained",
    }
    assert (
        sft.resolve_base_model(
            "facebook/opt-125m", "27dcfa74d334bc871f3234de431e71c6eeba5dd6"
        ).repo_id
        == "facebook/opt-125m"
    )
    with pytest.raises(ValueError, match="40-hex"):
        sft.resolve_base_model("facebook/opt-125m", "main")
    with pytest.raises(ValueError, match="owner/repository"):
        sft.resolve_base_model("/tmp/mutable-checkpoint", "a" * 40)
    with pytest.raises(ValueError, match="approved foundation-model"):
        sft.resolve_base_model("abhishekraok/previous-checkpoint", "a" * 40)


def test_missing_corpus_payload_is_fetched_from_pinned_manifest(tmp_path, monkeypatch):
    corpus_dir = _write_corpus(tmp_path)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = corpus_dir / manifest["splits"]["test"]["path"]
    expected_bytes = payload.read_bytes()
    payload.unlink()
    calls = []

    def fake_fetch(path, *, destination):
        calls.append((path, destination))
        payload.write_bytes(expected_bytes)

    monkeypatch.setattr(sft, "fetch_corpus", fake_fetch)

    assert sft.ensure_corpus(corpus_dir) == corpus_dir
    assert calls == [(manifest_path, corpus_dir)]
    assert payload.read_bytes() == expected_bytes


def test_token_length_filter_preserves_complete_labels_and_records_exclusions():
    class CharacterTokenizer:
        eos_token = "!"

        def __call__(self, texts, **kwargs):
            assert kwargs == {"add_special_tokens": True, "truncation": False}
            return {"input_ids": [list(range(len(text))) for text in texts]}

    rows = (
        {
            "prompt": "a",
            "completion": "b",
            "task_id": "kept",
            "split": "discovery",
            "certified_distance": 1,
        },
        {
            "prompt": "abcdef",
            "completion": "x",
            "task_id": "excluded",
            "split": "discovery",
            "certified_distance": 8,
        },
    )

    retained, audit = sft.filter_rows_for_complete_labels(
        rows,
        tokenizer=CharacterTokenizer(),
        max_length=6,
        max_excluded_fraction=0.5,
    )

    assert [row["task_id"] for row in retained] == ["kept"]
    assert audit["complete_labels_only"] is True
    assert audit["selected_records"] == 2
    assert audit["retained_records"] == 1
    assert audit["excluded"] == [
        {
            "task_id": "excluded",
            "split": "discovery",
            "certified_distance": 8,
            "tokens_with_eos": 8,
        }
    ]

    with pytest.raises(ValueError, match="materially change"):
        sft.filter_rows_for_complete_labels(
            rows,
            tokenizer=CharacterTokenizer(),
            max_length=6,
        )


def test_public_training_row_is_exact_and_carries_task_provenance(tmp_path):
    corpus_dir = _write_corpus(tmp_path)
    _, records = deep.load_corpus(corpus_dir, splits=("discovery",))
    basis = load_basis_set("default")
    available = basis.as_function_set()
    usage_before = [function.usage_count for function in available]

    row = sft.training_row_for_record(records[0], available)

    assert row["prompt"] == "Input: [1, 2]\nTarget Output: [2, 3]\n\nAnswer:"
    assert row["completion"] == "inc"
    assert row["split"] == "discovery"
    assert row["certified_distance"] == 1
    assert row["root_digest"] == records[0].metadata["root_digest"]
    assert row["task_id"] == records[0].task_id
    assert row["witness_function_names"] == ["inc"]
    assert [function.usage_count for function in available] == usage_before


def test_preparation_verifies_all_splits_but_only_materializes_train_and_validation(
    tmp_path,
):
    corpus_dir = _write_corpus(tmp_path)
    basis, prepared = sft.prepare_training_corpus(
        _config(tmp_path, corpus_dir, min_train_distance=4),
        preview_only=False,
    )

    assert basis.basis_set_id == "wl-core-v1"
    assert prepared.total_verified_records == 4
    assert prepared.planned_train_records == 1
    assert prepared.planned_validation_records == 1
    assert {row["split"] for row in prepared.train_rows} == {"discovery"}
    assert {row["certified_distance"] for row in prepared.train_rows} == {4}
    assert {row["split"] for row in prepared.validation_rows} == {"validation"}
    assert all(row["split"] != "test" for row in prepared.train_rows)
    assert all(row["split"] != "test" for row in prepared.validation_rows)
    roles = prepared.source["task_roles"]
    assert roles["discovery"]["role"] == "training-only"
    assert roles["validation"]["role"].startswith("trainer-evaluation")
    assert roles["test"]["role"].startswith("held-out-evaluation-only")
    assert prepared.source["verification"]["all_witnesses_reexecuted"] is True
    assert prepared.source["manifest_file_sha256"].startswith("sha256:")


def test_held_out_witness_is_verified_even_though_test_is_not_materialized(tmp_path):
    corpus_dir = _write_corpus(tmp_path, bad_test_witness=True)
    with pytest.raises(ValueError, match="witness/root-split verification"):
        sft.prepare_training_corpus(_config(tmp_path, corpus_dir))


def test_root_leak_is_a_hard_gate(tmp_path):
    corpus_dir = _write_corpus(tmp_path, leaking_root=True)
    with pytest.raises(ValueError, match="witness/root-split verification"):
        sft.prepare_training_corpus(_config(tmp_path, corpus_dir))


def test_manifest_basis_digest_must_match_loaded_basis(tmp_path):
    corpus_dir = _write_corpus(tmp_path, wrong_basis_digest=True)
    with pytest.raises(ValueError, match="basis-set digest"):
        sft.prepare_training_corpus(_config(tmp_path, corpus_dir))


def test_config_restricts_distance_entity_and_actual_run_name(tmp_path):
    corpus_dir = tmp_path / "corpus"
    with pytest.raises(ValueError, match="exactly 1 or 4"):
        sft._validate_config(_config(tmp_path, corpus_dir, min_train_distance=2))
    with pytest.raises(ValueError, match="personal entity"):
        sft._validate_config(_config(tmp_path, corpus_dir, wandb_entity="ai2-llm"))
    with pytest.raises(ValueError, match="wandb-run-name"):
        sft._validate_config(
            _config(tmp_path, corpus_dir, dry_run=False, wandb_run_name=None)
        )


def test_dry_run_never_calls_training_or_creates_output(tmp_path, monkeypatch):
    corpus_dir = _write_corpus(tmp_path)
    config = _config(tmp_path, corpus_dir, dry_run_samples=1)

    def fail_training(**kwargs):
        raise AssertionError(f"training was called: {kwargs}")

    monkeypatch.setattr(sft, "_train", fail_training)
    manifest = sft.run_training(config)

    assert manifest["status"] == "dry-run"
    assert manifest["base_model"]["repo_id"] == "HuggingFaceTB/SmolLM2-360M"
    assert manifest["preview"]["train_examples"][0]["completion"] == "inc"
    assert manifest["preview"]["validation_examples"][0]["split"] == "validation"
    assert manifest["wandb"]["entity"] == "abhishekraok-na"
    assert not config.output_dir.exists()


def test_sft_config_is_completion_only_deterministic_and_checkpoint_free(tmp_path):
    config = sft.DeepCorpusSFTConfig(
        output_dir=tmp_path / "artifact",
        precision="fp32",
        wandb_run_name="test",
    )
    training_args = sft.build_sft_config(config, report_to_wandb=True)

    assert str(training_args.save_strategy) in {"no", "SaveStrategy.NO"}
    assert str(training_args.eval_strategy) in {"epoch", "IntervalStrategy.EPOCH"}
    assert training_args.save_only_model is True
    assert training_args.completion_only_loss is True
    assert training_args.full_determinism is True
    assert training_args.seed == training_args.data_seed == 42
    assert training_args.report_to == ["wandb"]


def test_wandb_url_must_be_personal_and_in_the_configured_project():
    url = "https://wandb.ai/abhishekraok-na/deep-project/runs/abc123"
    assert sft.validate_wandb_url(url, project="deep-project") == url
    with pytest.raises(ValueError, match="personal entity/project"):
        sft.validate_wandb_url(
            "https://wandb.ai/ai2-llm/deep-project/runs/abc123",
            project="deep-project",
        )
    with pytest.raises(ValueError, match="personal entity/project"):
        sft.validate_wandb_url(
            "https://example.com/abhishekraok-na/deep-project/runs/abc123",
            project="deep-project",
        )


def test_weights_only_guard_rejects_disposable_training_state(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"weights")
    sft.assert_weights_only_artifact(artifact)

    (artifact / "optimizer.pt").write_bytes(b"state")
    with pytest.raises(RuntimeError, match=r"optimizer\.pt"):
        sft.assert_weights_only_artifact(artifact)
