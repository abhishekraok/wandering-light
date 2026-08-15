import gzip
import json
import random
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import basis_library_pilot as pilot
from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.basis_set import load_basis_set
from wandering_light.function_usage import FunctionUsageTracker


def _small_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    manifest, _ = pilot.generate_corpus(
        basis_set_id="default",
        split_sizes={"discovery": 60, "validation": 0, "test": 0},
        split_seeds={"discovery": 1729, "validation": 2718, "test": 3141},
        output_dir=corpus_dir,
        max_attempts_per_record=500,
    )
    return corpus_dir, manifest


def test_import_does_not_load_torch_or_initialize_cuda():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import experiments.basis_library_pilot; "
                "print('torch' in sys.modules)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def _reseal_manifest(corpus_dir, manifest):
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    manifest["manifest_digest"] = pilot._sha256_bytes(
        pilot._canonical_json(payload).encode()
    )
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_split_rows(corpus_dir, split):
    with gzip.open(corpus_dir / f"{split}.jsonl.gz", "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _replace_split_rows(corpus_dir, manifest, split, rows):
    path = corpus_dir / f"{split}.jsonl.gz"
    pilot._write_jsonl_gzip(rows, path)
    manifest["splits"][split]["sha256"] = pilot._sha256_file(path)


def _refresh_split_statistics(manifest, split, rows):
    records = [BasisTaskRecord.from_dict(row) for row in rows]
    metadata = manifest["splits"][split]
    metadata["size"] = len(records)
    metadata["by_input_type"] = dict(
        sorted(Counter(row.metadata["input_type"] for row in records).items())
    )
    metadata["by_witness_length"] = {
        str(length): count
        for length, count in sorted(
            Counter(row.witness_length for row in records).items()
        )
    }
    occurrences = Counter()
    coverage = Counter()
    for record in records:
        occurrences.update(record.witness_function_ids)
        coverage.update(set(record.witness_function_ids))
    basis = load_basis_set(manifest["basis_set_id"])
    functions = [
        {
            "function_id": function.function_id,
            "function_name": function.name,
            "total_occurrences": occurrences[function.function_id],
            "task_coverage": coverage[function.function_id],
        }
        for function in basis.functions
    ]
    metadata["witness_function_coverage"] = {
        "functions_exposed": sum(row["total_occurrences"] > 0 for row in functions),
        "functions_available": len(functions),
        "total_witness_steps": sum(occurrences.values()),
        "functions": functions,
    }


def test_wandb_entity_is_hard_limited_to_personal_account():
    assert pilot.validate_personal_wandb_entity("abhishekraok-na") == "abhishekraok-na"
    with pytest.raises(ValueError, match="personal W&B entity"):
        pilot.validate_personal_wandb_entity("ai2-llm")
    with pytest.raises(ValueError, match="personal W&B entity"):
        pilot.validate_personal_wandb_entity("AI2")


def test_checkpoint_basis_requires_fixed_pythonhashseed(monkeypatch):
    basis = load_basis_set("checkpoint-rl-6k-with-lp")
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        pilot._require_reproducible_runtime(basis)


def test_corpus_is_balanced_filtered_deduplicated_and_reproducible(tmp_path):
    first_dir, first_manifest = _small_corpus(tmp_path / "first")
    second_dir, second_manifest = _small_corpus(tmp_path / "second")

    assert first_manifest["manifest_digest"] == second_manifest["manifest_digest"]
    for filename in ("discovery.jsonl.gz", "validation.jsonl.gz", "test.jsonl.gz"):
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()

    manifest, records = pilot.load_corpus(first_dir)
    assert manifest["global_task_count"] == len(records) == 60
    assert manifest["generator"] == "balanced-stratified-random-walk-v2"
    assert manifest["value_generator"]["name"] == "builtin-behavior-strata-v2"
    exposure = manifest["splits"]["discovery"]["witness_function_coverage"]
    assert exposure["functions_available"] == len(exposure["functions"]) == 118
    assert exposure["total_witness_steps"] == 180
    assert len({record.task_id for record in records}) == 60
    assert Counter(record.witness_length for record in records) == {
        1: 12,
        2: 12,
        3: 12,
        4: 12,
        5: 12,
    }
    assert set(
        Counter(record.metadata["input_type"] for record in records).values()
    ) == {5}
    for record in records:
        assert record.input_value != record.output_value
        assert pilot._has_multiple_output_values(record.output_value)

    discovery_path = first_dir / "discovery.jsonl.gz"
    discovery_path.write_bytes(discovery_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Corpus file digest mismatch"):
        pilot.load_corpus(first_dir)


def test_load_corpus_rejects_non_basename_split_paths(tmp_path):
    corpus_dir, original_manifest = _small_corpus(tmp_path)
    for unsafe_path in (
        "../discovery.jsonl.gz",
        "/tmp/discovery.jsonl.gz",
        r"..\discovery.jsonl.gz",
        r"C:\corpus\discovery.jsonl.gz",
    ):
        manifest = deepcopy(original_manifest)
        manifest["splits"]["discovery"]["path"] = unsafe_path
        _reseal_manifest(corpus_dir, manifest)
        with pytest.raises(ValueError, match="safe basename"):
            pilot.load_corpus(corpus_dir, splits=("validation",))


def test_load_corpus_checks_global_count_only_for_full_load(tmp_path):
    corpus_dir, manifest = _small_corpus(tmp_path)
    manifest["global_task_count"] += 1
    _reseal_manifest(corpus_dir, manifest)

    _, discovery = pilot.load_corpus(corpus_dir, splits=("discovery",))
    assert len(discovery) == 60
    with pytest.raises(ValueError, match="global record count mismatch"):
        pilot.load_corpus(corpus_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("input_type", "input_type metadata mismatch"),
        ("requested_length", "witness length mismatch"),
        ("serialized_length", "serialized witness_length mismatch"),
        ("unknown_function", "unknown basis-function ID"),
        ("wrong_function_name", "name mismatch"),
    ],
)
def test_load_corpus_validates_record_provenance(tmp_path, mutation, message):
    corpus_dir, manifest = _small_corpus(tmp_path)
    rows = _read_split_rows(corpus_dir, "discovery")
    row = rows[0]
    if mutation == "input_type":
        row["metadata"]["input_type"] = "builtins.object"
    elif mutation == "requested_length":
        row["metadata"]["requested_witness_length"] += 1
    elif mutation == "serialized_length":
        row["witness_length"] += 1
    elif mutation == "unknown_function":
        row["witness_function_ids"][0] = "bf:not-in-source-basis:0000000000000000"
    else:
        row["witness_function_names"][0] = "not_the_registered_name"
    _replace_split_rows(corpus_dir, manifest, "discovery", rows)
    _reseal_manifest(corpus_dir, manifest)

    with pytest.raises(ValueError, match=message):
        pilot.load_corpus(corpus_dir, splits=("discovery",))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("by_input_type", "by-input-type count mismatch"),
        ("by_witness_length", "witness-length count mismatch"),
        ("witness_function_coverage", "witness-function coverage mismatch"),
    ],
)
def test_load_corpus_recomputes_split_statistics(tmp_path, field, message):
    corpus_dir, manifest = _small_corpus(tmp_path)
    if field == "witness_function_coverage":
        manifest["splits"]["discovery"][field]["total_witness_steps"] += 1
    else:
        first_key = next(iter(manifest["splits"]["discovery"][field]))
        manifest["splits"]["discovery"][field][first_key] += 1
    _reseal_manifest(corpus_dir, manifest)

    with pytest.raises(ValueError, match=message):
        pilot.load_corpus(corpus_dir, splits=("discovery",))


def test_load_corpus_validates_full_corpus_balance_claim(tmp_path):
    corpus_dir, manifest = _small_corpus(tmp_path)
    rows = _read_split_rows(corpus_dir, "discovery")
    row = next(row for row in rows if row["witness_length"] == 1)
    row["witness_function_ids"].append(row["witness_function_ids"][0])
    row["witness_function_names"].append(row["witness_function_names"][0])
    row["witness_length"] = 2
    row["metadata"]["requested_witness_length"] = 2
    _replace_split_rows(corpus_dir, manifest, "discovery", rows)
    _refresh_split_statistics(manifest, "discovery", rows)
    _reseal_manifest(corpus_dir, manifest)

    # The selected split remains useful for targeted consumers, whose requested
    # subset does not claim to establish whole-corpus balance.
    _, selected = pilot.load_corpus(corpus_dir, splits=("discovery",))
    assert len(selected) == 60
    with pytest.raises(ValueError, match="Corpus balance mismatch"):
        pilot.load_corpus(corpus_dir)


def test_model_spec_resolves_local_hashes_and_exact_hf_fallback(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"opt"}', encoding="utf-8")
    (checkpoint / "wandb_run.url").write_text(
        "https://wandb.ai/abhishekraok-na/project/runs/personal\n",
        encoding="utf-8",
    )
    local = pilot.resolve_model_spec(str(checkpoint), pilot.DEFAULT_HF_REVISION)
    assert local.resolved == str(checkpoint.resolve())
    assert local.revision is None
    assert local.local_files["config.json"]["sha256"].startswith("sha256:")
    assert local.local_tree_digest.startswith("sha256:")
    assert local.wandb_run_url.endswith("/personal")

    missing_default = tmp_path / "missing-default"
    monkeypatch.setattr(pilot, "DEFAULT_LOCAL_CHECKPOINT", missing_default)
    remote = pilot.resolve_model_spec(str(missing_default), pilot.DEFAULT_HF_REVISION)
    assert remote.resolved == pilot.DEFAULT_HF_CHECKPOINT
    assert remote.revision == pilot.DEFAULT_HF_REVISION
    with pytest.raises(ValueError, match="40-hex commit SHA"):
        pilot.resolve_model_spec("abhishekraok/a-model", None)
    with pytest.raises(ValueError, match="40-hex commit SHA"):
        pilot.resolve_model_spec("abhishekraok/a-model", "main")
    uppercase_revision = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    immutable = pilot.resolve_model_spec("abhishekraok/a-model", uppercase_revision)
    assert immutable.revision == uppercase_revision.lower()


def test_known_default_local_checkpoint_requires_verified_lineage(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "known-default"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 12}), encoding="utf-8"
    )
    (checkpoint / "wandb_run.url").write_text(
        "[InternetShortcut]\nURL=https://wandb.ai/abhishekraok-na/p/runs/r\n",
        encoding="utf-8",
    )
    nested = checkpoint / "tokenizer"
    nested.mkdir()
    (nested / "added_tokens.json").write_text("{}", encoding="utf-8")
    (checkpoint / "generation_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "download.lock").write_text("transient", encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"disposable")
    cache = checkpoint / ".cache"
    cache.mkdir()
    (cache / "ignored.json").write_text("ignored", encoding="utf-8")

    monkeypatch.setattr(pilot, "DEFAULT_LOCAL_CHECKPOINT", checkpoint)
    monkeypatch.setattr(pilot, "DEFAULT_CHECKPOINT_MODEL_BYTES", len(b"weights"))
    monkeypatch.setattr(pilot, "DEFAULT_CHECKPOINT_GLOBAL_STEP", 12)
    monkeypatch.setattr(
        pilot,
        "DEFAULT_CHECKPOINT_WANDB_URL",
        "https://wandb.ai/abhishekraok-na/p/runs/r",
    )
    verified = pilot.resolve_model_spec(str(checkpoint), pilot.DEFAULT_HF_REVISION)
    assert verified.canonical_hf_repo == pilot.DEFAULT_HF_CHECKPOINT
    assert verified.canonical_hf_revision == pilot.DEFAULT_HF_REVISION
    assert "tokenizer/added_tokens.json" in verified.local_files
    assert "generation_config.json" in verified.local_files
    assert "download.lock" not in verified.local_files
    assert "optimizer.pt" not in verified.local_files
    assert ".cache/ignored.json" not in verified.local_files

    original_digest = verified.local_tree_digest
    (nested / "added_tokens.json").write_text('{"new":1}', encoding="utf-8")
    changed = pilot.resolve_model_spec(str(checkpoint), pilot.DEFAULT_HF_REVISION)
    assert changed.local_tree_digest != original_digest

    (checkpoint / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="canonical lineage verification"):
        pilot.resolve_model_spec(str(checkpoint), pilot.DEFAULT_HF_REVISION)


def test_rich_value_strata_cover_current_predicate_branches():
    rng = random.Random(42)
    strings = pilot._rich_value_pool(str, rng)
    assert any(value.startswith("a") for value in strings)
    assert any(value.endswith("z") for value in strings)
    assert any(value.isspace() for value in strings)
    assert any(value.isdigit() for value in strings)
    assert any(value and value.lower() == value.lower()[::-1] for value in strings)

    tuples = pilot._rich_value_pool(tuple, rng)
    assert any(None in value for value in tuples)
    dictionaries = pilot._rich_value_pool(dict, rng)
    assert any(
        len(value.values()) != len(set(value.values())) for value in dictionaries
    )
    byte_strings = pilot._rich_value_pool(bytes, rng)
    assert b"" in byte_strings
    assert any(value and all(byte < 128 for byte in value) for value in byte_strings)
    assert any(any(byte >= 128 for byte in value) for value in byte_strings)
    ranges = pilot._rich_value_pool(range, rng)
    assert any(value.step < 0 for value in ranges)
    assert any(value.start < 0 for value in ranges)


def test_aggregate_reports_occurrences_coverage_and_subsequences():
    basis = load_basis_set("default")
    functions = basis.as_function_set().functions[:2]
    tracker = FunctionUsageTracker(basis.basis_set_id, basis.digest)
    tracker.record_solution(
        [functions[0], functions[1], functions[0]],
        basis_set_id=basis.basis_set_id,
        basis_digest=basis.digest,
    )
    identities = [pilot._function_identity(function) for function in functions]
    rows = [
        {
            "split": "test",
            "input_type": "builtins.int",
            "witness_length": 3,
            "success": True,
            "solution_length": 3,
        },
        {
            "split": "test",
            "input_type": "builtins.int",
            "witness_length": 2,
            "success": False,
            "solution_length": None,
        },
    ]
    aggregate = pilot.aggregate_results(
        rows=rows,
        basis=basis,
        tracker=tracker,
        solved_sequences=[[identities[0], identities[1], identities[0]]],
        batch_latencies_seconds=[0.2],
        batch_sizes=[2],
        top_subsequences=10,
    )

    by_id = {
        row["function_id"]: row for row in aggregate["function_usage"]["functions"]
    }
    assert by_id[identities[0][0]]["total_occurrences"] == 2
    assert by_id[identities[0][0]]["task_coverage"] == 1
    assert aggregate["overall"]["solve_rate"] == 0.5
    assert aggregate["latency"]["tasks_per_second"] == 10
    assert {
        tuple(row["function_ids"])
        for row in aggregate["frequent_contiguous_subsequences"]
    } >= {
        (identities[0][0], identities[1][0]),
        (identities[1][0], identities[0][0]),
        (identities[0][0], identities[1][0], identities[0][0]),
    }


def test_capture_execution_environment_records_cpu_and_effective_batches():
    environment = pilot.capture_execution_environment(
        requested_device="cpu",
        requested_batch_size=13,
        budget=1,
        observed_batch_sizes=[13, 13, 8],
    )

    assert environment["schema_version"] == 1
    assert environment["requested_device"] == "cpu"
    assert environment["resolved_device"] == "cpu"
    assert environment["hardware"]["accelerator"]["kind"] == "cpu"
    assert environment["hardware_fingerprint"].startswith("sha256:")
    assert environment["batch_protocol"] == {
        "requested_batch_size": 13,
        "solver_inference_batch_size": 13,
        "candidates_per_task": 1,
        "observed_batch_count": 3,
        "observed_batch_size_histogram": {"8": 1, "13": 2},
        "effective_inference_batch_count": 3,
        "effective_inference_batch_size_histogram": {"8": 1, "13": 2},
    }

    budgeted = pilot.capture_execution_environment(
        requested_device="cpu",
        requested_batch_size=13,
        budget=2,
        observed_batch_sizes=[13, 8],
    )
    assert budgeted["batch_protocol"]["candidates_per_task"] == 2
    assert budgeted["batch_protocol"]["effective_inference_batch_count"] == 4
    assert budgeted["batch_protocol"]["effective_inference_batch_size_histogram"] == {
        "3": 1,
        "13": 3,
    }


def test_evaluation_writes_compact_rows_from_frozen_outputs(tmp_path, monkeypatch):
    corpus_dir, _ = _small_corpus(tmp_path)
    checked_basis_ids = []
    require_reproducible_runtime = pilot._require_reproducible_runtime

    def check_runtime(basis):
        checked_basis_ids.append(basis.basis_set_id)
        require_reproducible_runtime(basis)

    monkeypatch.setattr(pilot, "_require_reproducible_runtime", check_runtime)

    class AlwaysFailSolver:
        def solve_batch(self, problems, available_functions):
            assert problems
            assert len(available_functions) == 116
            return [
                SimpleNamespace(
                    success=False,
                    trajectory=None,
                    error_msg="x" * 1_000,
                )
                for _ in problems
            ]

    monkeypatch.setattr(
        pilot,
        "_create_checkpoint_solver",
        lambda **kwargs: AlwaysFailSolver(),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    aggregate, paths = pilot.evaluate_corpus(
        corpus_dir=corpus_dir,
        output_dir=tmp_path / "evaluation",
        splits=("discovery",),
        evaluation_basis_set_id="pilot-compressed",
        model=str(checkpoint),
        model_revision=None,
        batch_size=13,
        budget=1,
        deterministic_decoding=True,
        temperature=0.1,
        max_new_tokens=32,
        seed=42,
        device="cpu",
    )

    assert aggregate["overall"] == {
        "tasks": 60,
        "successes": 0,
        "failures": 60,
        "solve_rate": 0.0,
        "mean_solution_length_success": None,
    }
    assert aggregate["task_source_basis_set_id"] == "wl-core-v1"
    assert aggregate["evaluation_basis_set_id"] == "wl-pilot-compressed-v1"
    assert aggregate["execution_environment"]["resolved_device"] == "cpu"
    assert aggregate["execution_environment"]["batch_protocol"] == {
        "requested_batch_size": 13,
        "solver_inference_batch_size": 13,
        "candidates_per_task": 1,
        "observed_batch_count": 5,
        "observed_batch_size_histogram": {"8": 1, "13": 4},
        "effective_inference_batch_count": 5,
        "effective_inference_batch_size_histogram": {"8": 1, "13": 4},
    }
    assert checked_basis_ids == ["wl-pilot-compressed-v1"]
    assert (
        aggregate["task_source_basis_set_digest"]
        != aggregate["evaluation_basis_set_digest"]
    )
    assert all(path.is_file() for path in paths)
    with gzip.open(paths[0], "rt", encoding="utf-8") as result_file:
        first_row = json.loads(next(result_file))
    assert "input" not in first_row
    assert "output" not in first_row
    assert len(first_row["error"]) == 240
