"""Protocol and orchestration tests for the compute-optimal size sweep."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import compute_optimal_policy_size as study
from wandering_light.compute_optimal import OptimalActionOracleResult
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.typed_list import TypedList

EXPECTED_PARAMETERS = {
    "90k": 90_374,
    "1m": 724_022,
    "10m": 6_504_310,
}
EXPECTED_LEARNING_RATES = {
    "90k": 2e-3,
    "1m": 2e-3,
    "10m": 5e-4,
}


def _training_runtime(
    parameters: int,
    architecture: dict[str, int],
    learning_rate: float = 2e-3,
) -> dict[str, Any]:
    return {
        "resolved_device": "cuda:0",
        "precision": "fp32",
        "floating_parameter_dtypes": ["torch.float32"],
        "model": {
            "parameters": parameters,
            "architecture_config": architecture,
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "defaults": {"lr": learning_rate},
        },
        "accelerator": {"index": 0, "name": "fake-gpu"},
        "software": {"torch_version": "unit-test"},
    }


@dataclass(frozen=True)
class FakeRecord:
    task_id: str
    split: str
    input_value: TypedList
    output_value: TypedList
    witness_function_names: tuple[str, ...]
    metadata: dict[str, Any]


def _records() -> list[FakeRecord]:
    records = []
    for distance in study.CERTIFIED_DISTANCES:
        for offset in range(study.TASKS_PER_DISTANCE):
            value = distance * 100 + offset
            records.append(
                FakeRecord(
                    task_id=f"validation-d{distance}-{offset:02d}",
                    split="validation",
                    input_value=TypedList([value]),
                    output_value=TypedList([value + distance]),
                    witness_function_names=("a",) * distance,
                    metadata={
                        "certified_distance": distance,
                        "root_digest": f"sha256:{value:064x}",
                    },
                )
            )
    return records


def _functions() -> SimpleNamespace:
    names = ("a", "b", *(f"action-{index}" for index in range(116)))
    return SimpleNamespace(
        functions=tuple(SimpleNamespace(name=name) for name in names)
    )


def _manifest() -> dict[str, Any]:
    return {
        "manifest_digest": "sha256:" + "1" * 64,
        "basis_set_id": "unit-test-basis",
        "basis_set_digest": "sha256:" + "2" * 64,
        "splits": {
            "validation": {
                "path": "validation.jsonl.gz",
                "sha256": "sha256:" + "3" * 64,
                "size": study.TASKS_PER_DISTANCE * len(study.CERTIFIED_DISTANCES),
                "by_certified_distance": {
                    str(distance): study.TASKS_PER_DISTANCE
                    for distance in study.CERTIFIED_DISTANCES
                },
            }
        },
    }


def _selective_verification(
    manifest_digest: str,
    selected_splits: tuple[str, ...],
    *,
    manifest_splits: tuple[str, ...] = ("discovery", "test", "validation"),
    required_references: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "manifest_digest": manifest_digest,
        "scope": {
            "kind": "selected-splits",
            "selected_splits": list(selected_splits),
            "manifest_splits": sorted(manifest_splits),
        },
        "checks": {
            "manifest_identity": {
                "status": "passed",
                "manifest_digest": manifest_digest,
            },
            **{
                check: {"status": "passed", "failures": []}
                for check in study._SELECTIVE_CORPUS_PASSES
                if check not in {"manifest_identity", "reference_overlap"}
            },
            "reference_overlap": {
                "status": "passed",
                "failures": [],
                "required_references": [
                    dict(reference) for reference in required_references
                ],
                "verified_references": [
                    dict(reference) for reference in required_references
                ],
            },
            **{
                check: {"status": "skipped", "reason": reason}
                for check, reason in study._SELECTIVE_CORPUS_SKIP_REASONS.items()
            },
        },
        "verified_splits": list(selected_splits),
        "full_corpus_verified": None,
        "target_contract_failures": None,
        "ok": True,
    }


@pytest.fixture
def corpus() -> SimpleNamespace:
    records = _records()
    functions = _functions()
    manifest = _manifest()
    return SimpleNamespace(
        records=records,
        functions=functions,
        manifest=manifest,
        loader=lambda _path: (manifest, functions, records),
    )


@pytest.fixture
def frozen_protocol(tmp_path: Path, corpus: SimpleNamespace) -> SimpleNamespace:
    path = tmp_path / "protocol.json"
    host = {"node": "unit-test-host", "cpu": "fake", "cuda": "fake"}
    protocol = study.freeze_protocol(
        Path("unused-corpus"),
        output=path,
        validation_loader=corpus.loader,
        host_loader=lambda: host,
        clean_source_checker=lambda: None,
    )
    return SimpleNamespace(
        path=path,
        protocol=protocol,
        host=host,
        **vars(corpus),
    )


def test_freeze_pins_validation_arms_search_metric_and_oracle(
    frozen_protocol: SimpleNamespace,
) -> None:
    protocol = frozen_protocol.protocol

    assert protocol["status"] == study.PROTOCOL_STATUS
    assert protocol["discipline"] == {
        "evaluation_split": "validation",
        "held_out_model_scoring": False,
        "cohort_shared_across_all_cells": True,
        "paired_training_seeds": True,
        "evaluation_cells_sequential": True,
        "global_phase_barrier": [
            "all-12-cell-timing",
            "sealed-timing-matrix",
            "observer-replay-and-oracle-annotation",
        ],
        "evaluation_lock": {
            "file": study.EVALUATION_LOCK_FILE,
            "mode": "exclusive-nonblocking-whole-cell",
        },
    }
    assert protocol["cohort"]["split"] == "validation"
    assert protocol["cohort"]["size"] == 96
    assert protocol["cohort"]["distance_histogram"] == {
        str(distance): 12 for distance in range(1, 9)
    }
    task_order = protocol["evaluation_task_order"]
    assert task_order["method"] == "sha256-rank"
    assert task_order["seed"] == study.EVALUATION_TASK_ORDER_SEED
    assert set(task_order["task_ids"]) == set(protocol["cohort"]["task_ids"])
    assert task_order["task_ids"] != protocol["cohort"]["task_ids"]
    cell_order = protocol["evaluation_cell_order"]
    assert study.EVALUATION_CELL_COUNT == 12
    assert len(cell_order["cells"]) == 12
    assert [cell["ordinal"] for cell in cell_order["cells"]] == list(range(12))
    for pair_index in range(6):
        pair = cell_order["cells"][2 * pair_index : 2 * pair_index + 2]
        assert pair[0]["arm_id"] == pair[1]["arm_id"]
        assert pair[0]["training_seed"] == pair[1]["training_seed"]
        assert {pair[0]["device"], pair[1]["device"]} == {"cpu", "cuda"}
        assert pair[0]["device"] == ("cpu" if pair_index % 2 == 0 else "cuda")

    assert [arm["arm_id"] for arm in protocol["arms"]] == list(EXPECTED_PARAMETERS)
    assert {
        arm["arm_id"]: arm["parameters"] for arm in protocol["arms"]
    } == EXPECTED_PARAMETERS
    assert protocol["training_seeds"] == [0, 1]
    assert protocol["devices"] == ["cpu", "cuda"]
    assert protocol["corpus"]["reference_corpora_excluded"] == []
    assert protocol["training"] == {
        "source_corpus": "deep-corpus-v2",
        "materialized_splits": ["discovery", "validation"],
        "held_out_prefix_examples": 0,
        "epochs": 12,
        "per_arm_learning_rates": EXPECTED_LEARNING_RATES,
        "device": "cuda",
        "effective_train_batch_size": 256,
        "per_arm_batching": {
            "90k": {
                "train_batch_size": 256,
                "gradient_accumulation_steps": 1,
                "validation_batch_size": 512,
            },
            "1m": {
                "train_batch_size": 256,
                "gradient_accumulation_steps": 1,
                "validation_batch_size": 512,
            },
            "10m": {
                "train_batch_size": 256,
                "gradient_accumulation_steps": 1,
                "validation_batch_size": 256,
            },
        },
        "checkpoint_selection": "minimum validation cross-entropy",
        "weights_only": True,
    }
    assert protocol["search"]["beam_width"] == 64
    assert protocol["search"]["max_depth"] == 8
    assert protocol["search"]["execution_budget"] == 100_000

    wall = protocol["metrics"]["wall_clock"]
    assert wall["log2_min"] == -7.0
    assert wall["log2_max"] == 4.0
    assert wall["min_seconds"] == 2**-7
    assert wall["max_seconds"] == 2**4
    assert protocol["metrics"]["executions"]["min_executions"] == 1
    assert protocol["metrics"]["executions"]["max_executions"] == 100_000

    in_situ = protocol["metrics"]["in_situ_bits"]
    assert in_situ["epsilon"] == 0.01
    assert in_situ["sample"]["maximum_per_task_depth"] == 2
    assert in_situ["oracle"] == {
        "max_depth": 6,
        "max_states": 20_000,
        "max_executions": 100_000,
        "state_identity": "TypedList.search_key",
        "target_identity": "TypedList.canonical_key",
    }
    assert in_situ["headline_min_certified_coverage"] == 0.8
    assert protocol["evaluation_host"] == frozen_protocol.host
    source_paths = {row["path"] for row in protocol["source"]["files"]}
    assert "wandering_light/basis_dataset.py" in source_paths
    assert "experiments/basis_library_pilot.py" in source_paths
    assert study.load_protocol(frozen_protocol.path) == protocol


def test_operational_protocol_inside_repo_must_be_gitignored() -> None:
    with pytest.raises(ValueError, match="git-ignored"):
        study._require_gitignored_operational_path(Path("protocol.json"))

    study._require_gitignored_operational_path(
        Path("checkpoints/compute-optimal-policy-size/protocol.json")
    )


def test_freeze_rejects_dirty_source_before_loading_corpus(
    tmp_path: Path, corpus: SimpleNamespace
) -> None:
    loaded = False

    def loader(_path: Path):
        nonlocal loaded
        loaded = True
        return corpus.manifest, corpus.functions, corpus.records

    def reject_dirty() -> None:
        raise RuntimeError("dirty checkout")

    with pytest.raises(RuntimeError, match="dirty checkout"):
        study.freeze_protocol(
            Path("unused-corpus"),
            output=tmp_path / "protocol.json",
            validation_loader=loader,
            host_loader=lambda: {"node": "unused"},
            clean_source_checker=reject_dirty,
        )

    assert loaded is False
    assert not (tmp_path / "protocol.json").exists()


def test_evaluation_lock_rejects_overlapping_cells(
    frozen_protocol: SimpleNamespace,
) -> None:
    with (
        study._exclusive_protocol_lock(frozen_protocol.path),
        pytest.raises(RuntimeError, match="Another compute-optimal evaluation"),
        study._exclusive_protocol_lock(frozen_protocol.path),
    ):
        pass


def test_artifact_primitives_share_canonical_normalization(tmp_path: Path) -> None:
    from wandering_light.evals.deep_corpus_protocol import (
        canonical_copy,
        canonical_json,
        file_identity,
        sha256_bytes,
        sha256_file,
        sha256_json,
    )

    digest_field = "artifact_sha256"
    body = {
        "tuple_value": (1, 2),
        "integer_keys": {2: "two", 10: "ten"},
        digest_field: "sha256:stale",
    }
    normalized = canonical_copy(
        {key: value for key, value in body.items() if key != digest_field}
    )
    expected = {**normalized, digest_field: sha256_json(normalized)}

    sealed = study._seal(body, digest_field)

    assert sealed == expected
    assert study._seal(sealed, digest_field) == sealed
    assert study._verify_seal(sealed, digest_field, label="test artifact") == sealed
    assert study._canonical_json(normalized) == canonical_json(normalized)
    assert study._sha256_bytes(b"shared") == sha256_bytes(b"shared")

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"shared")
    assert study._sha256_file(artifact) == sha256_file(artifact)
    assert study._artifact_identity(artifact) == file_identity(artifact, root=tmp_path)


def test_arm_lookup_is_cached_immutable_trainer_configuration() -> None:
    from experiments.train_rollout_policy import (
        POLICY_SIZE_BATCHING,
        POLICY_SIZE_LEARNING_RATES,
    )

    configs = study._frozen_arm_configs()

    assert tuple(config.arm_id for config in configs) == study.ARM_IDS
    assert {
        config.arm_id: (
            config.train_batch_size,
            config.gradient_accumulation_steps,
            config.validation_batch_size,
        )
        for config in configs
    } == {arm_id: POLICY_SIZE_BATCHING[arm_id] for arm_id in study.ARM_IDS}
    assert {
        config.arm_id: config.learning_rate for config in configs
    } == {arm_id: POLICY_SIZE_LEARNING_RATES[arm_id] for arm_id in study.ARM_IDS}
    assert study._arm_config_by_id("10m") is study._arm_config_by_id("10m")

    materialized = study._frozen_arms()
    materialized[0]["architecture"]["hidden"] = -1
    assert study._arm_by_id("90k")["architecture"]["hidden"] > 0


def test_corpus_loader_requests_validation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import generate_deep_corpus
    from wandering_light import basis_set, corpus_hub

    calls = []
    manifest = _manifest()
    reference = {"manifest_digest": "sha256:" + "4" * 64}
    manifest["reference_corpora_excluded"] = [reference]
    manifest["target_contract"] = {
        "reference_manifest_digests": [reference["manifest_digest"]]
    }
    manifest["splits"].update(
        {
            "discovery": {"path": "discovery.jsonl.gz"},
            "test": {"path": "test.jsonl.gz"},
        }
    )
    records = _records()
    palettes = []

    def load_corpus(path: Path, *, splits: tuple[str, ...]):
        calls.append(("load", path, splits))
        return manifest, records

    def verify_loaded_corpus(
        loaded_manifest: dict[str, Any],
        loaded_records: list[FakeRecord],
        *,
        splits: tuple[str, ...],
        recertify: int,
        reference_identities: tuple[Any, ...],
    ) -> dict[str, Any]:
        assert loaded_manifest is manifest
        assert loaded_records is records
        assert reference_identities == ("reference-identities",)
        calls.append(("verify", splits, recertify))
        return _selective_verification(
            manifest["manifest_digest"],
            splits,
            manifest_splits=tuple(manifest["splits"]),
            required_references=(reference,),
        )

    def load_required_reference_identities(
        loaded_manifest: dict[str, Any],
        corpus_dirs: tuple[Path, ...],
        *,
        fetch_if_missing: bool = True,
    ) -> tuple[str, ...]:
        assert loaded_manifest is manifest
        assert corpus_dirs == study.DEFAULT_REFERENCE_CORPUS_DIRS
        assert fetch_if_missing is True
        calls.append(("references", corpus_dirs))
        return ("reference-identities",)

    def fetch_corpus(
        manifest_path: Path,
        *,
        destination: Path,
        splits: tuple[str, ...],
        include_details: bool,
    ) -> None:
        calls.append(("fetch", manifest_path, destination, splits, include_details))

    def as_function_set():
        palette = FunctionDefSet(
            [
                FunctionDef(
                    name="a",
                    input_type="builtins.int",
                    output_type="builtins.int",
                    code="return x + 1",
                )
            ]
        )
        palettes.append(palette)
        return palette

    fake_basis = SimpleNamespace(
        digest=manifest["basis_set_digest"],
        as_function_set=as_function_set,
    )
    monkeypatch.setattr(generate_deep_corpus, "load_corpus", load_corpus)
    monkeypatch.setattr(
        generate_deep_corpus,
        "verify_loaded_corpus",
        verify_loaded_corpus,
    )
    monkeypatch.setattr(
        generate_deep_corpus,
        "load_required_reference_identities",
        load_required_reference_identities,
    )
    monkeypatch.setattr(basis_set, "load_basis_set", lambda _basis_id: fake_basis)
    monkeypatch.setattr(corpus_hub, "fetch_corpus", fetch_corpus)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    study._write_json(corpus_dir / "manifest.json", manifest)
    study._load_validation_corpus_cached.cache_clear()

    first = study._load_validation_corpus(corpus_dir)
    first[1].functions[0].increment_usage()
    second = study._load_validation_corpus(corpus_dir / ".." / "corpus")

    assert first[0] == second[0] == manifest
    assert first[2] == second[2] == records
    assert first[0] is not second[0]
    assert first[1] is palettes[0]
    assert second[1] is palettes[1]
    assert first[1] is not second[1]
    assert first[1].functions[0].usage_count == 1
    assert second[1].functions[0].usage_count == 0
    assert calls == [
        (
            "fetch",
            corpus_dir / "manifest.json",
            corpus_dir,
            ("validation",),
            False,
        ),
        ("load", corpus_dir, ("validation",)),
        ("references", study.DEFAULT_REFERENCE_CORPUS_DIRS),
        ("verify", ("validation",), 0),
    ]
    study._load_validation_corpus_cached.cache_clear()


@pytest.mark.parametrize(
    ("check", "evidence"),
    [
        ("witness_replay", {"status": "passed"}),
        (
            "record_provenance",
            {"status": "passed", "failures": ["contradictory failure"]},
        ),
        (
            "manifest_identity",
            {"status": "passed", "manifest_digest": "sha256:different"},
        ),
        ("target_contract", {"status": "skipped", "reason": ""}),
        (
            "distance_recertification",
            {"status": "skipped", "reason": "recertify=1"},
        ),
        (
            "distance_recertification",
            {
                "status": "skipped",
                "reason": "recertify=0",
                "failures": ["contradictory failure"],
            },
        ),
        (
            "reference_overlap",
            {
                "status": "passed",
                "failures": [],
                "required_references": [],
                "verified_references": [{"manifest_digest": "sha256:" + "9" * 64}],
            },
        ),
    ],
)
def test_selective_verification_gate_rejects_contradictory_check_evidence(
    check: str,
    evidence: dict[str, Any],
) -> None:
    digest = "sha256:" + "1" * 64
    verification = _selective_verification(
        digest,
        ("validation",),
    )
    verification["checks"][check] = evidence

    with pytest.raises(ValueError, match="exact selected-split verification evidence"):
        study._require_selective_corpus_verification(
            verification,
            manifest_digest=digest,
            selected_splits=("validation",),
            manifest_splits=("discovery", "test", "validation"),
            required_references=(),
            label="Validation corpus",
        )


def test_selective_verification_gate_requires_exact_pinned_reference_evidence() -> None:
    digest = "sha256:" + "1" * 64
    reference = {"manifest_digest": "sha256:" + "2" * 64, "tasks": 31_350}
    verification = _selective_verification(
        digest,
        ("validation",),
        required_references=(reference,),
    )
    arguments = {
        "manifest_digest": digest,
        "selected_splits": ("validation",),
        "manifest_splits": ("discovery", "test", "validation"),
        "required_references": (reference,),
        "label": "Validation corpus",
    }
    study._require_selective_corpus_verification(verification, **arguments)

    verification["checks"]["reference_overlap"]["verified_references"] = []
    with pytest.raises(ValueError, match="exact selected-split verification evidence"):
        study._require_selective_corpus_verification(verification, **arguments)


def test_runner_jsonl_helpers_are_deterministic_and_require_object_rows(
    tmp_path: Path,
) -> None:
    rows = [{"z": 1, "a": [2, 3]}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    study._write_jsonl_gz(rows, first)
    study._write_jsonl_gz(rows, second)

    assert first.read_bytes() == second.read_bytes()
    assert (
        study._validated_jsonl_rows(
            study._read_jsonl_gz(first),
            label="test artifact",
        )
        == rows
    )

    malformed = tmp_path / "malformed.jsonl.gz"
    study._write_jsonl_gz([{"valid": True}, 1], malformed)
    with pytest.raises(ValueError, match="non-object row"):
        study._validated_jsonl_rows(
            study._read_jsonl_gz(malformed),
            label="test artifact",
        )


@pytest.mark.parametrize(
    (
        "arm_id",
        "train_batch_size",
        "gradient_accumulation_steps",
        "validation_batch_size",
        "learning_rate",
    ),
    [
        ("90k", 256, 1, 512, 2e-3),
        ("1m", 256, 1, 512, 2e-3),
        ("10m", 256, 1, 256, 5e-4),
    ],
)
def test_checkpoint_identity_requires_the_frozen_training_recipe(
    tmp_path: Path,
    frozen_protocol: SimpleNamespace,
    arm_id: str,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    validation_batch_size: int,
    learning_rate: float,
) -> None:
    from wandering_light.rollout_policy import (
        CONFIG_FILE,
        MODEL_FILE,
        TRAINER_STATE_FILE,
        TRAINING_MANIFEST_FILE,
        WANDB_LINK_FILE,
    )

    checkpoint = tmp_path / f"checkpoint-{arm_id}"
    checkpoint.mkdir()
    arm = next(
        arm for arm in frozen_protocol.protocol["arms"] if arm["arm_id"] == arm_id
    )
    corpus = frozen_protocol.protocol["corpus"]
    study._write_json(
        checkpoint / CONFIG_FILE,
        {
            "architecture": arm["architecture"],
            "basis_set": {
                "basis_set_id": corpus["basis_set_id"],
                "basis_set_digest": corpus["basis_set_digest"],
            },
        },
    )
    training_manifest = {
        "status": "complete",
        "objective": "witness-next-action-cross-entropy",
        "source_code": {
            "head_commit": frozen_protocol.protocol["source"]["head_commit"],
            "dirty": False,
            "runner": {
                "path": "experiments/train_rollout_policy.py",
                "sha256": next(
                    row["sha256"]
                    for row in frozen_protocol.protocol["source"]["files"]
                    if row["path"] == "experiments/train_rollout_policy.py"
                ),
            },
        },
        "runtime": _training_runtime(
            arm["parameters"], arm["architecture"], learning_rate
        ),
        "hyperparameters": {
            "seed": 1,
            "epochs": 12,
            "learning_rate": learning_rate,
            "device": "cuda",
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_train_batch_size": 256,
            "validation_batch_size": validation_batch_size,
            "selection": "minimum validation cross-entropy",
            "weights_only": True,
        },
        "source_corpus": {
            "name": "deep-corpus-v2",
            "manifest_digest": corpus["manifest_digest"],
            "basis_set_id": corpus["basis_set_id"],
            "basis_set_digest": corpus["basis_set_digest"],
            "verification": _selective_verification(
                corpus["manifest_digest"],
                ("discovery", "validation"),
                required_references=tuple(corpus["reference_corpora_excluded"]),
            ),
            "task_roles": {
                "discovery": {"role": "training-only"},
                "validation": {"role": "model-selection-only"},
            },
        },
        "prefix_examples": {"test": 0},
        "trainer_state": {
            "epochs_completed": 12,
            "best_epoch": 7,
            "best_validation_cross_entropy": 0.25,
        },
    }
    study._write_json(checkpoint / TRAINING_MANIFEST_FILE, training_manifest)
    study._write_json(checkpoint / TRAINER_STATE_FILE, {"best_epoch": 7})
    (checkpoint / MODEL_FILE).write_bytes(b"weights-only-fixture")
    (checkpoint / WANDB_LINK_FILE).write_text(
        "https://wandb.ai/unit/project/runs/run\n", encoding="utf-8"
    )

    identity = study._checkpoint_identity(
        checkpoint,
        arm=arm,
        training_seed=1,
        protocol=frozen_protocol.protocol,
    )
    assert identity["parameters"] == EXPECTED_PARAMETERS[arm_id]
    assert identity["training_seed"] == 1
    assert identity["source_protocol_comparison"] == {
        "matches": True,
        "mismatches": [],
    }
    assert {artifact["name"] for artifact in identity["artifacts"]} == {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINING_MANIFEST_FILE,
        TRAINER_STATE_FILE,
        WANDB_LINK_FILE,
    }

    training_manifest["source_code"]["head_commit"] = "older-source"
    training_manifest["source_code"]["runner"]["sha256"] = "sha256:older-runner"
    study._write_json(checkpoint / TRAINING_MANIFEST_FILE, training_manifest)
    mismatched_identity = study._checkpoint_identity(
        checkpoint,
        arm=arm,
        training_seed=1,
        protocol=frozen_protocol.protocol,
    )
    assert mismatched_identity["source_protocol_comparison"]["matches"] is False
    assert {
        row["field"]
        for row in mismatched_identity["source_protocol_comparison"]["mismatches"]
    } == {"head_commit", "runner.sha256"}

    training_manifest["hyperparameters"]["train_batch_size"] = (
        128 if train_batch_size == 256 else 256
    )
    study._write_json(checkpoint / TRAINING_MANIFEST_FILE, training_manifest)
    with pytest.raises(ValueError, match="train_batch_size"):
        study._checkpoint_identity(
            checkpoint,
            arm=arm,
            training_seed=1,
            protocol=frozen_protocol.protocol,
        )

    training_manifest["hyperparameters"]["train_batch_size"] = train_batch_size
    training_manifest["hyperparameters"]["validation_batch_size"] = (
        128 if validation_batch_size == 512 else 512
    )
    study._write_json(checkpoint / TRAINING_MANIFEST_FILE, training_manifest)
    with pytest.raises(ValueError, match="validation_batch_size"):
        study._checkpoint_identity(
            checkpoint,
            arm=arm,
            training_seed=1,
            protocol=frozen_protocol.protocol,
        )

    training_manifest["hyperparameters"]["validation_batch_size"] = (
        validation_batch_size
    )
    training_manifest["source_corpus"]["verification"]["full_corpus_verified"] = False
    study._write_json(checkpoint / TRAINING_MANIFEST_FILE, training_manifest)
    with pytest.raises(ValueError, match="selected-split verification evidence"):
        study._checkpoint_identity(
            checkpoint,
            arm=arm,
            training_seed=1,
            protocol=frozen_protocol.protocol,
        )


@pytest.mark.parametrize(
    ("depth_counts", "recorded_total"),
    [({}, None), ({"1": 0}, 0)],
)
def test_visit_validation_rejects_missing_or_empty_stratum_without_dividing(
    depth_counts: dict[str, int], recorded_total: int | None
) -> None:
    cell = {"arm_id": "90k", "training_seed": 0, "device": "cpu"}
    task_rows = [
        {
            "task_id": "validation-task",
            "certified_distance": 1,
            "root_digest": "sha256:root",
            "sampled_visits": 0,
            "scored_visits_by_rollout_depth": depth_counts,
        }
    ]
    visit_rows = [
        {
            **cell,
            "task_id": "validation-task",
            "certified_distance": 1,
            "root_digest": "sha256:root",
            "sampling_stratum": {
                "task_id": "validation-task",
                "rollout_depth": 1,
                "total_visits": recorded_total,
                "sampled_visits": 0,
            },
            "sampling_weight": 1.0,
            "visit": {
                "depth": 1,
                "oracle": {
                    "status": "unresolved",
                    "optimal_actions_complete": False,
                },
                "masked_optimal_bits": None,
            },
        }
    ]

    with pytest.raises(ValueError, match="invalid stratum weight"):
        study._validate_visit_rows(visit_rows, task_rows=task_rows, cell=cell)


@dataclass(frozen=True)
class FakeRolloutResult:
    termination: str
    executions: int
    function_defs: tuple[SimpleNamespace, ...]
    trace: tuple[Any, ...] = ()


class FakeScorer:
    def __init__(self, events: list[tuple[Any, ...]], executions: int) -> None:
        self.events = events
        self.executions = executions
        self.batch_sizes: list[int] = []

    def score_actions_batch(self, problems, _functions):
        batch_size = len(problems)
        self.batch_sizes.append(batch_size)
        self.events.append(("warm", batch_size))
        return [{"a": 0.0, "b": 0.0} for _ in problems]


class FakeObserver:
    def __init__(self) -> None:
        self.visits: list[dict[str, Any]] = []


class FakePredictor:
    def __init__(
        self,
        scorer: FakeScorer,
        events: list[tuple[Any, ...]],
    ) -> None:
        self.scorer = scorer
        self.events = events

    def search_with_cost(self, source, target, _functions, observer=None):
        kind = "timed" if observer is None else "replay"
        base = source.items[0]
        self.events.append((kind, base))
        distance = target.items[0] - base
        if observer is not None:
            for visit_index in range(3):
                state = TypedList([base * 10 + visit_index])
                observer.visits.append(
                    {
                        "depth": 1,
                        "parent_index": visit_index,
                        "state": state.to_string(),
                        "path": [],
                        "path_log_probability": -float(visit_index),
                        "scores": {"a": 0.0, "b": 0.0},
                        "compatible_actions": ["a", "b"],
                        "log_normalizer": math.log(2.0),
                    }
                )
        return FakeRolloutResult(
            termination="solved",
            executions=self.scorer.executions,
            function_defs=tuple(SimpleNamespace(name="a") for _ in range(distance)),
        )


class FakeClock:
    def __init__(self, duration: float, events: list[tuple[Any, ...]]) -> None:
        self.duration = duration
        self.events = events
        self.calls = 0

    def __call__(self) -> float:
        pair, side = divmod(self.calls, 2)
        self.calls += 1
        self.events.append(("clock", pair, side))
        return pair * 100.0 + side * self.duration


class FakeOracle:
    def __init__(self) -> None:
        self.calls: list[tuple[TypedList, TypedList]] = []

    def __call__(self, source, target, _functions, *, limits):
        assert limits.max_depth == 6
        assert limits.max_states == 20_000
        assert limits.max_executions == 100_000
        self.calls.append((source, target))
        unresolved = (source.items[0] // 10) % 100 == 0
        if unresolved:
            return OptimalActionOracleResult(
                status="unresolved",
                stop_reason="max_states",
                distance=None,
                observed_target_distance=None,
                optimal_action_names=(),
                optimal_actions_complete=False,
                certified_depth=1,
                reached_states=20_000,
                attempted_executions=10,
                failed_executions=0,
            )
        return OptimalActionOracleResult(
            status="certified",
            stop_reason=None,
            distance=1,
            observed_target_distance=1,
            optimal_action_names=("a",),
            optimal_actions_complete=True,
            certified_depth=1,
            reached_states=2,
            attempted_executions=2,
            failed_executions=0,
        )


def _fake_result_verifier(record, result, _functions):
    names = [function.name for function in result.function_defs]
    assert names == list(record.witness_function_names)
    return result.termination == "solved", names


def _fake_checkpoint(checkpoint: Path, **kwargs: Any) -> dict[str, Any]:
    protocol = kwargs["protocol"]
    arm_id = kwargs["arm"]["arm_id"]
    learning_rate = protocol["training"]["per_arm_learning_rates"][arm_id]
    source_head = (
        "older-checkpoint-source"
        if arm_id == "90k"
        else protocol["source"]["head_commit"]
    )
    runner_sha256 = next(
        row["sha256"]
        for row in protocol["source"]["files"]
        if row["path"] == "experiments/train_rollout_policy.py"
    )
    return {
        "path": str(checkpoint),
        "parameters": kwargs["arm"]["parameters"],
        "training_seed": kwargs["training_seed"],
        "artifacts_sha256": f"sha256:{arm_id}-{kwargs['training_seed']}",
        "source_code": {
            "head_commit": source_head,
            "dirty": False,
            "runner": {
                "path": "experiments/train_rollout_policy.py",
                "sha256": runner_sha256,
            },
        },
        "source_protocol_comparison": {
            "matches": arm_id != "90k",
            "mismatches": (
                []
                if arm_id != "90k"
                else [
                    {
                        "field": "head_commit",
                        "expected": protocol["source"]["head_commit"],
                        "actual": source_head,
                    }
                ]
            ),
        },
        "training_runtime": _training_runtime(
            kwargs["arm"]["parameters"],
            kwargs["arm"]["architecture"],
            learning_rate,
        ),
        "training_environment": {
            key: value
            for key, value in _training_runtime(
                kwargs["arm"]["parameters"],
                kwargs["arm"]["architecture"],
                learning_rate,
            ).items()
            if key != "model"
        },
    }


def _run_fake_cell_phase(
    frozen: SimpleNamespace,
    *,
    output_dir: Path,
    arm_id: str,
    seed: int,
    device: str,
    duration: float,
    executions: int,
    timing_summary: Path | None = None,
    timing_matrix: Path | None = None,
    inject_oracle: bool = True,
) -> tuple[dict[str, Any], SimpleNamespace]:
    events: list[tuple[Any, ...]] = []
    verification_calls: list[tuple[str, int]] = []
    scorer = FakeScorer(events, executions)
    oracle = FakeOracle()
    clock = FakeClock(duration, events)

    def build_predictor(scorer_arg, beam_width, max_depth, budget):
        assert scorer_arg is scorer
        assert (beam_width, max_depth, budget) == (64, 8, 100_000)
        return FakePredictor(scorer, events)

    def synchronize(resolved_device: str) -> None:
        assert resolved_device == device
        events.append(("sync", resolved_device))

    def verify(record, result, functions):
        verification_calls.append((record.task_id, result.executions))
        return _fake_result_verifier(record, result, functions)

    summary = study.evaluate_cell(
        study.EvaluationConfig(
            protocol=frozen.path,
            checkpoint=Path(f"checkpoint-{arm_id}-{seed}"),
            arm_id=arm_id,
            training_seed=seed,
            device=device,
            output_dir=output_dir,
            corpus_dir=Path("unused-corpus"),
            timing_summary=timing_summary,
            timing_matrix=timing_matrix,
            progress=False,
        ),
        validation_loader=frozen.loader,
        host_loader=lambda: frozen.host,
        checkpoint_validator=_fake_checkpoint,
        scorer_loader=lambda _checkpoint, _device: scorer,
        predictor_builder=build_predictor,
        observer_builder=FakeObserver,
        synchronizer=synchronize,
        clock=clock,
        result_verifier=verify,
        oracle_certifier=oracle if inject_oracle else None,
    )
    return summary, SimpleNamespace(
        events=events,
        scorer=scorer,
        oracle=oracle,
        clock=clock,
        verification_calls=verification_calls,
    )


def _write_fake_timing_matrix(
    frozen: SimpleNamespace,
    *,
    timing_summary_path: Path,
    output: Path,
) -> Path:
    summary = study._read_json(timing_summary_path, label="timing summary")
    completed = summary["timing_window"]["completed_at_utc"]
    timings = []
    for cell in frozen.protocol["evaluation_cell_order"]["cells"]:
        identity = {key: cell[key] for key in ("arm_id", "training_seed", "device")}
        if identity == summary["cell"]:
            path = str(timing_summary_path.resolve())
            digest = summary[study.TIMING_DIGEST_FIELD]
        else:
            path = str(
                (output.parent / f"placeholder-{cell['ordinal']}.json").resolve()
            )
            digest = f"sha256:placeholder-{cell['ordinal']}"
        timings.append(
            {
                "cell": identity,
                "cell_ordinal": cell["ordinal"],
                "summary_path": path,
                "timing_sha256": digest,
            }
        )
    matrix = study._seal(
        {
            "schema_version": study.SCHEMA_VERSION,
            "study_id": study.STUDY_ID,
            "status": study.TIMING_MATRIX_STATUS,
            "protocol_sha256": frozen.protocol[study.PROTOCOL_DIGEST_FIELD],
            "evaluation_cell_order_sha256": frozen.protocol["evaluation_cell_order"][
                "cells_sha256"
            ],
            "sealed_at_utc": datetime.now(UTC).isoformat(),
            "latest_timing_completed_at_utc": completed,
            "timings": timings,
        },
        study.TIMING_MATRIX_DIGEST_FIELD,
    )
    study._write_json(output, matrix)
    return output


def _evaluate_fake_cell(
    frozen: SimpleNamespace,
    *,
    output_dir: Path,
    arm_id: str,
    seed: int,
    device: str,
    duration: float,
    executions: int,
) -> tuple[dict[str, Any], SimpleNamespace]:
    timing_dir = output_dir.with_name(f"{output_dir.name}-timing")
    _, timing_fakes = _run_fake_cell_phase(
        frozen,
        output_dir=timing_dir,
        arm_id=arm_id,
        seed=seed,
        device=device,
        duration=duration,
        executions=executions,
    )
    timing_summary = timing_dir / study.TIMING_SUMMARY_FILE
    timing_matrix = _write_fake_timing_matrix(
        frozen,
        timing_summary_path=timing_summary,
        output=output_dir.with_name(f"{output_dir.name}-timing-matrix.json"),
    )
    summary, annotation_fakes = _run_fake_cell_phase(
        frozen,
        output_dir=output_dir,
        arm_id=arm_id,
        seed=seed,
        device=device,
        duration=duration,
        executions=executions,
        timing_summary=timing_summary,
        timing_matrix=timing_matrix,
    )
    return summary, SimpleNamespace(
        timing=timing_fakes,
        annotation=annotation_fakes,
        events=[*timing_fakes.events, *annotation_fakes.events],
        scorer=annotation_fakes.scorer,
        oracle=annotation_fakes.oracle,
        clock=timing_fakes.clock,
        verification_calls=[
            *timing_fakes.verification_calls,
            *annotation_fakes.verification_calls,
        ],
    )


def test_default_oracle_is_prepared_once_only_for_annotation(
    tmp_path: Path,
    frozen_protocol: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wandering_light import compute_optimal

    prepared: list[tuple[FakeOracle, Any]] = []

    class CountingPreparedOracle(FakeOracle):
        def __init__(self, functions: Any) -> None:
            super().__init__()
            prepared.append((self, functions))

    monkeypatch.setattr(
        compute_optimal,
        "PreparedOptimalActionOracle",
        CountingPreparedOracle,
    )
    timing_dir = tmp_path / "default-oracle-timing"
    _run_fake_cell_phase(
        frozen_protocol,
        output_dir=timing_dir,
        arm_id="90k",
        seed=0,
        device="cpu",
        duration=0.125,
        executions=64,
        inject_oracle=False,
    )
    assert prepared == []

    timing_summary = timing_dir / study.TIMING_SUMMARY_FILE
    timing_matrix = _write_fake_timing_matrix(
        frozen_protocol,
        timing_summary_path=timing_summary,
        output=tmp_path / "default-oracle-timing-matrix.json",
    )
    _run_fake_cell_phase(
        frozen_protocol,
        output_dir=tmp_path / "default-oracle-annotation",
        arm_id="90k",
        seed=0,
        device="cpu",
        duration=0.125,
        executions=64,
        timing_summary=timing_summary,
        timing_matrix=timing_matrix,
        inject_oracle=False,
    )

    assert len(prepared) == 1
    oracle, functions = prepared[0]
    assert functions is frozen_protocol.functions
    assert len(oracle.calls) == 192


def test_evaluate_isolates_timing_and_records_oracle_coverage(
    tmp_path: Path, frozen_protocol: SimpleNamespace
) -> None:
    output_dir = tmp_path / "evaluation"
    summary, fakes = _evaluate_fake_cell(
        frozen_protocol,
        output_dir=output_dir,
        arm_id="90k",
        seed=0,
        device="cuda",
        duration=0.125,
        executions=64,
    )

    assert fakes.scorer.batch_sizes == [1, 64]
    assert fakes.timing.oracle.calls == []
    assert all(event[0] != "replay" for event in fakes.timing.events)
    assert all(event[0] != "timed" for event in fakes.annotation.events)
    assert summary["timing_sha256"].startswith("sha256:")
    assert summary["timing_matrix_sha256"].startswith("sha256:")
    assert summary["warmup"] == {
        "scorer_calls": 2,
        "batch_sizes": [1, 64],
        "outside_task_timing": True,
        "synchronized_after_each": True,
    }
    search_events = [event for event in fakes.events if event[0] in {"timed", "replay"}]
    assert [event[0] for event in search_events] == ["timed"] * 96 + ["replay"] * 96
    assert [event[1] for event in search_events[:96]] == [
        event[1] for event in search_events[96:]
    ]
    assert sum(event[0] == "sync" for event in fakes.events) == 4 + 2 * 96
    assert fakes.clock.calls == 2 * 96
    assert len(fakes.verification_calls) == 2 * 96

    task_rows = study._read_jsonl_gz(output_dir / study.TASK_ROWS_FILE)
    visit_rows = study._read_jsonl_gz(output_dir / study.VISIT_ROWS_FILE)
    assert len(task_rows) == 96
    assert len(visit_rows) == 192
    assert all(row["split"] == "validation" for row in task_rows)
    assert all(row["seconds"] == pytest.approx(0.125) for row in task_rows)
    assert all(row["observer_replay_exact"] for row in task_rows)
    assert {(row["scored_visits"], row["sampled_visits"]) for row in task_rows} == {
        (3, 2)
    }
    assert all("oracle" in row["visit"] for row in visit_rows)
    assert all("masked_optimal_bits" in row["visit"] for row in visit_rows)
    assert sum(row["visit"]["masked_optimal_bits"] is None for row in visit_rows) == 16
    assert len(fakes.oracle.calls) == 192
    cell = summary["cell"]
    bad_tasks = [dict(row) for row in task_rows]
    bad_tasks[0]["certified_distance"] = (
        1 if bad_tasks[0]["certified_distance"] != 1 else 2
    )
    with pytest.raises(ValueError, match="frozen task"):
        study._validate_task_rows(
            bad_tasks,
            cell=cell,
            task_order=frozen_protocol.protocol["evaluation_task_order"],
            cohort=frozen_protocol.protocol["cohort"],
        )
    bad_visits = [dict(row) for row in visit_rows]
    bad_visits[0]["root_digest"] = "sha256:wrong-root"
    with pytest.raises(ValueError, match="parent task"):
        study._validate_visit_rows(
            bad_visits,
            task_rows=task_rows,
            cell=cell,
        )
    assert summary["oracle_cache"] == {
        "path": str((frozen_protocol.path.parent / study.ORACLE_CACHE_FILE).resolve()),
        "unique_queries_in_cell": 192,
        "cache_hits": 0,
        "entries_before": 0,
        "entries_after": 192,
        "key": [
            "TypedList.search_key(state)",
            "TypedList.canonical_key(target)",
            "basis_set_digest",
        ],
    }
    assert summary["visit_sampling"] == {
        "total_visits": 288,
        "sampled_visits": 192,
        "maximum_per_task_depth": 2,
        "sampling_seed": study.VISIT_SAMPLING_SEED,
    }

    for distance in study.CERTIFIED_DISTANCES:
        h = summary["h_metrics"]["by_certified_distance"][str(distance)]
        assert h["total_visits"] == 36
        assert h["sampled_visits"] == 24
        assert h["certified_visits"] == 22
        assert h["certified_coverage"] == pytest.approx(11 / 12)
        assert h["headline_available"] is True
        assert h["state_weighted"]["mean_bits"] is not None
        assert h["task_balanced"]["mean_bits"] is not None
        assert h["by_rollout_depth"]["1"]["certified_visits"] == 22

    cached_summary, cached_fakes = _evaluate_fake_cell(
        frozen_protocol,
        output_dir=tmp_path / "evaluation-from-cache",
        arm_id="1m",
        seed=0,
        device="cpu",
        duration=0.25,
        executions=32,
    )
    assert cached_fakes.oracle.calls == []
    assert cached_summary["oracle_cache"]["cache_hits"] == 192
    assert cached_summary["oracle_cache"]["entries_before"] == 192
    assert cached_summary["oracle_cache"]["entries_after"] == 192


def test_h_headline_uses_population_weighted_label_coverage() -> None:
    bits = {
        "bits": 1.0,
        "raw_bits": 1.0,
        "uniform_bits": 2.0,
        "bits_saved": 1.0,
    }
    rows = [
        {
            "task_id": f"certified-{index}",
            "sampling_weight": 1.0,
            "visit": {
                "oracle_query_sha256": f"certified-{index}",
                "oracle": {"status": "certified", "stop_reason": None},
                "masked_optimal_bits": bits,
            },
        }
        for index in range(9)
    ]
    rows.append(
        {
            "task_id": "large-unresolved-stratum",
            "sampling_weight": 100.0,
            "visit": {
                "oracle_query_sha256": "unresolved",
                "oracle": {
                    "status": "unresolved",
                    "stop_reason": "max_states",
                },
                "masked_optimal_bits": None,
            },
        }
    )

    result = study._h_group(rows, total_visits=109)

    assert result["certified_coverage"] == pytest.approx(0.9)
    assert result["population_weighted_certified_coverage"] == pytest.approx(9 / 109)
    assert result["task_balanced_certified_coverage"] == pytest.approx(0.9)
    assert result["headline_available"] is False
    assert result["state_weighted"]["mean_bits"] is None
    assert result["state_weighted"]["headline_available"] is False
    assert result["task_balanced"]["headline_available"] is True
    assert result["task_balanced"]["mean_bits"] == pytest.approx(1.0)
    assert result["state_weighted"]["certified_only_mean_bits"] == pytest.approx(1.0)


def test_h_task_balanced_headline_uses_task_balanced_coverage() -> None:
    bits = {
        "bits": 1.0,
        "raw_bits": 1.0,
        "uniform_bits": 2.0,
        "bits_saved": 1.0,
    }
    rows = [
        {
            "task_id": "long-certified-task",
            "sampling_weight": 100.0,
            "visit": {
                "oracle_query_sha256": "certified",
                "oracle": {"status": "certified", "stop_reason": None},
                "masked_optimal_bits": bits,
            },
        }
    ]
    rows.extend(
        {
            "task_id": f"unresolved-{index}",
            "sampling_weight": 1.0,
            "visit": {
                "oracle_query_sha256": f"unresolved-{index}",
                "oracle": {"status": "unresolved", "stop_reason": "max_states"},
                "masked_optimal_bits": None,
            },
        }
        for index in range(9)
    )

    result = study._h_group(rows, total_visits=109)

    assert result["population_weighted_certified_coverage"] == pytest.approx(100 / 109)
    assert result["task_balanced_certified_coverage"] == pytest.approx(0.1)
    assert result["state_weighted"]["headline_available"] is True
    assert result["state_weighted"]["mean_bits"] == pytest.approx(1.0)
    assert result["task_balanced"]["headline_available"] is False
    assert result["task_balanced"]["mean_bits"] is None


def test_aggregate_requires_full_matrix_and_selects_best_arms(
    tmp_path: Path, frozen_protocol: SimpleNamespace
) -> None:
    execution_costs = {"90k": 64, "1m": 32, "10m": 16}
    cells = frozen_protocol.protocol["evaluation_cell_order"]["cells"]
    timing_paths: dict[tuple[str, int, str], Path] = {}
    for cell in cells:
        arm_id = cell["arm_id"]
        seed = cell["training_seed"]
        device = cell["device"]
        arm_index = list(EXPECTED_PARAMETERS).index(arm_id)
        timing_dir = tmp_path / f"timing-{arm_id}-seed{seed}-{device}"
        _run_fake_cell_phase(
            frozen_protocol,
            output_dir=timing_dir,
            arm_id=arm_id,
            seed=seed,
            device=device,
            duration=2 ** (-6 + arm_index + seed / 4),
            executions=execution_costs[arm_id] * (seed + 1),
        )
        timing_paths[(arm_id, seed, device)] = timing_dir / study.TIMING_SUMMARY_FILE
    timing_matrix_path = tmp_path / "timing-matrix.json"
    timing_matrix = study.seal_timing_matrix(
        frozen_protocol.path,
        list(timing_paths.values()),
        output=timing_matrix_path,
    )
    assert timing_matrix["training_environments"]["all_identical"] is False

    summaries = []
    for cell in cells:
        arm_id = cell["arm_id"]
        seed = cell["training_seed"]
        device = cell["device"]
        arm_index = list(EXPECTED_PARAMETERS).index(arm_id)
        output_dir = tmp_path / f"{arm_id}-seed{seed}-{device}"
        _run_fake_cell_phase(
            frozen_protocol,
            output_dir=output_dir,
            arm_id=arm_id,
            seed=seed,
            device=device,
            duration=2 ** (-6 + arm_index + seed / 4),
            executions=execution_costs[arm_id] * (seed + 1),
            timing_summary=timing_paths[(arm_id, seed, device)],
            timing_matrix=timing_matrix_path,
        )
        summaries.append(output_dir / study.SUMMARY_FILE)

    with pytest.raises(ValueError, match="incomplete"):
        study.aggregate_evaluations(
            frozen_protocol.path,
            summaries[:-1],
            output=tmp_path / "missing.json",
        )

    bad_dir = tmp_path / "wrong-host"
    shutil.copytree(summaries[0].parent, bad_dir)
    bad_summary_path = bad_dir / study.SUMMARY_FILE
    bad_summary = study._read_json(bad_summary_path, label="evaluation")
    bad_summary["evaluation_host"] = {"node": "another-host"}
    study._write_json(
        bad_summary_path,
        study._seal(bad_summary, study.EVALUATION_DIGEST_FIELD),
    )
    with pytest.raises(ValueError, match="another host"):
        study.aggregate_evaluations(
            frozen_protocol.path,
            [bad_summary_path, *summaries[1:]],
            output=tmp_path / "wrong-host.json",
        )

    wrong_matrix_dir = tmp_path / "wrong-matrix"
    shutil.copytree(summaries[0].parent, wrong_matrix_dir)
    wrong_matrix_path = wrong_matrix_dir / study.SUMMARY_FILE
    wrong_matrix = study._read_json(wrong_matrix_path, label="evaluation")
    wrong_matrix["timing_matrix_sha256"] = "sha256:different-matrix"
    study._write_json(
        wrong_matrix_path,
        study._seal(wrong_matrix, study.EVALUATION_DIGEST_FIELD),
    )
    with pytest.raises(ValueError, match="different timing matrices"):
        study.aggregate_evaluations(
            frozen_protocol.path,
            [wrong_matrix_path, *summaries[1:]],
            output=tmp_path / "wrong-matrix.json",
        )

    wrong_checkpoint_dir = tmp_path / "wrong-checkpoint"
    shutil.copytree(summaries[0].parent, wrong_checkpoint_dir)
    wrong_checkpoint_path = wrong_checkpoint_dir / study.SUMMARY_FILE
    wrong_checkpoint = study._read_json(wrong_checkpoint_path, label="evaluation")
    wrong_checkpoint["checkpoint"]["artifacts_sha256"] = "sha256:different"
    study._write_json(
        wrong_checkpoint_path,
        study._seal(wrong_checkpoint, study.EVALUATION_DIGEST_FIELD),
    )
    with pytest.raises(ValueError, match="different checkpoints"):
        study.aggregate_evaluations(
            frozen_protocol.path,
            [wrong_checkpoint_path, *summaries[1:]],
            output=tmp_path / "wrong-checkpoint.json",
        )

    changed_timing_dir = tmp_path / "changed-timing-row"
    shutil.copytree(summaries[0].parent, changed_timing_dir)
    changed_timing_path = changed_timing_dir / study.SUMMARY_FILE
    changed_timing = study._read_json(changed_timing_path, label="evaluation")
    task_path = changed_timing_dir / study.TASK_ROWS_FILE
    task_rows = study._read_jsonl_gz(task_path)
    task_rows[0]["seconds"] *= 2
    study._write_jsonl_gz(task_rows, task_path)
    changed_timing["metrics"] = study._distance_metrics(task_rows)
    changed_timing["artifacts"]["tasks"] = study._artifact_identity(task_path)
    study._write_json(
        changed_timing_path,
        study._seal(changed_timing, study.EVALUATION_DIGEST_FIELD),
    )
    with pytest.raises(ValueError, match="changed its sealed timing rows"):
        study.aggregate_evaluations(
            frozen_protocol.path,
            [changed_timing_path, *summaries[1:]],
            output=tmp_path / "changed-timing-row.json",
        )

    aggregate = study.aggregate_evaluations(
        frozen_protocol.path,
        summaries,
        output=tmp_path / "aggregate.json",
    )

    assert aggregate["cells"] == aggregate["expected_cells"] == 12
    assert len(aggregate["inputs"]["checkpoints"]) == 6
    assert len(aggregate["inputs"]["evaluations"]) == 12
    assert aggregate["inputs"]["timing_matrix"]["path"] == str(
        timing_matrix_path.resolve()
    )
    assert aggregate["inputs"]["training_sources"]["all_identical"] is False
    assert aggregate["inputs"]["training_environments"]["all_identical"] is False
    assert aggregate["source"] == frozen_protocol.protocol["source"]
    assert set(aggregate["by_arm"]) == set(EXPECTED_PARAMETERS)
    paired = aggregate["paired_arm_differences_by_certified_distance"]["1"]["cpu"]
    assert set(paired) == {
        "90k-minus-1m",
        "90k-minus-10m",
        "1m-minus-10m",
    }
    assert (
        paired["90k-minus-1m"]["wall_clock_auc"]["mean_normalized_area_difference"] > 0
    )
    assert (
        paired["90k-minus-1m"]["execution_auc"]["mean_normalized_area_difference"] < 0
    )
    assert [
        row["training_seed"]
        for row in paired["90k-minus-1m"]["wall_clock_auc"]["by_training_seed"]
    ] == [0, 1]
    device_delta = aggregate["paired_device_differences_by_certified_distance"]["1"][
        "90k"
    ]
    assert device_delta["direction"] == "cpu-minus-cuda"
    assert device_delta["wall_clock_auc"][
        "mean_normalized_area_difference"
    ] == pytest.approx(0.0)
    assert aggregate["cpu_cuda_task_parity"]["all_exact"] is True
    for distance in study.CERTIFIED_DISTANCES:
        for device in study.DEVICES:
            selected = aggregate["selection_by_certified_distance"][str(distance)][
                "by_device"
            ][device]
            assert selected["best_wall_clock_arm"]["arm_id"] == "90k"
            assert selected["best_execution_arm"]["arm_id"] == "10m"

    group = aggregate["by_arm"]["90k"]["by_device"]["cpu"]["by_certified_distance"]["1"]
    assert [row["training_seed"] for row in group["seeds"]] == [0, 1]
    assert all("h" in row for row in group["seeds"])
    assert group["wall_clock_auc"]["normalized_sample_sd"] > 0
    assert group["execution_auc"]["normalized_sample_sd"] > 0
    assert group["h"]["state_weighted"]["headline_available"] is True
    assert group["h"]["state_weighted"]["mean_bits"] is not None
    assert group["h"]["state_weighted"]["sample_sd_bits"] is not None
