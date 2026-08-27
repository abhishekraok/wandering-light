from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments import freeze_deep_corpus_v2_protocol as freeze
from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.basis_set import load_basis_set
from wandering_light.evals.deep_corpus_protocol import (
    COHORT_SIZE,
    sha256_file,
    validate_protocol_lock,
)
from wandering_light.typed_list import TypedList

BASIS = load_basis_set(freeze.BASIS_SET_ID)
COMMIT = "a" * 40


def _record(*, split: str, distance: int, offset: int) -> BasisTaskRecord:
    split_base = 10_000 if split == "validation" else 20_000
    unique = split_base + distance * 100 + offset
    input_value = TypedList([unique], int)
    return BasisTaskRecord.create(
        split=split,
        input_value=input_value,
        output_value=TypedList([unique + distance], int),
        witness_function_ids=[f"bf:inc:{step}" for step in range(distance)],
        witness_function_names=["inc"] * distance,
        basis_set_id=freeze.BASIS_SET_ID,
        basis_set_digest=freeze.BASIS_SET_DIGEST,
        generator="unit-test",
        seed=freeze.ISSUE_54_SEED,
        source_index=unique,
        metadata={
            "certified_distance": distance,
            "root_digest": f"sha256:{unique:064x}",
            "root_index": unique,
        },
    )


@pytest.fixture
def records() -> list[BasisTaskRecord]:
    return [
        _record(split=split, distance=distance, offset=offset)
        for split in ("validation", "test")
        for distance in range(1, 9)
        for offset in range(13)
    ]


def _manifest(record_count: int) -> dict:
    split_size = record_count // 2
    return {
        "manifest_digest": "sha256:" + "1" * 64,
        "basis_set_id": freeze.BASIS_SET_ID,
        "basis_set_digest": freeze.BASIS_SET_DIGEST,
        "generator": "unit-test",
        "seed": freeze.ISSUE_54_SEED,
        "global_task_count": record_count,
        "target_contract": {
            "profile": freeze.PROFILE_NAME,
            "generator_source": {
                "git_commit": COMMIT,
                "tree_state": "clean",
                "runner": "experiments/prepare_deep_corpus_v2.py",
                "generator": "experiments/generate_deep_corpus.py",
            },
        },
        "hub": {
            "repo_id": freeze.EXPECTED_CORPUS_REPO_ID,
            "repo_type": "dataset",
            "revision": "b" * 40,
        },
        "splits": {
            "discovery": {
                "path": "discovery.jsonl.gz",
                "size": 1,
                "roots": 1,
                "by_certified_distance": {str(index): 1 for index in range(1, 9)},
            },
            "validation": {
                "path": "validation.jsonl.gz",
                "size": split_size,
                "roots": split_size,
                "by_certified_distance": {str(index): 13 for index in range(1, 9)},
            },
            "test": {
                "path": "test.jsonl.gz",
                "size": split_size,
                "roots": split_size,
                "by_certified_distance": {str(index): 13 for index in range(1, 9)},
            },
        },
    }


def _source_identity(manifest: dict) -> dict:
    return {
        "corpus": "deep-corpus-v2",
        "manifest_digest": manifest["manifest_digest"],
        "manifest_file_sha256": "sha256:" + "2" * 64,
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "generator": manifest["generator"],
        "generator_seed": manifest["seed"],
        "hub": deepcopy(manifest["hub"]),
        "verification": {
            "records": manifest["global_task_count"],
            "distance_histogram": {str(index): 26 for index in range(1, 9)},
            "all_witnesses_reexecuted": True,
            "root_split_leaks": [],
            "ok": True,
        },
        "discovery": {
            "split": "discovery",
            "records": 1,
            "roots": 1,
            "by_certified_distance": {str(index): 1 for index in range(1, 9)},
            "split_file": {
                "name": "discovery.jsonl.gz",
                "bytes": 1,
                "sha256": "sha256:" + "3" * 64,
            },
            "ordered_task_ids_digest": "sha256:" + "4" * 64,
            "task_id_set_digest": "sha256:" + "4" * 64,
        },
    }


def _write_corpus(tmp_path, manifest: dict):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split in ("discovery", "validation", "test"):
        (corpus / manifest["splits"][split]["path"]).write_bytes(f"{split}\n".encode())
    return corpus


def _tokenizer_manifest(source: dict) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "wandering-light-deep-corpus-v2-shared-tokenizer",
        "artifact_digest": "sha256:" + "5" * 64,
        "basis_set": BASIS.identity_dict(),
        "source": deepcopy(source),
        "tokenizer": {
            "max_length": freeze.FIXED_MAX_LENGTH,
            "base_vocab_size": 512,
            "total_vocab_size": 600,
        },
        "training_text": {"pythonhashseed": "0"},
    }


def _write_tokenizer(tmp_path, manifest: dict):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in freeze.TOKENIZER_FILENAMES:
        (tokenizer / name).write_text(f"{name}\n", encoding="utf-8")
    (tokenizer / freeze.TOKENIZER_MANIFEST_FILENAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tokenizer


@pytest.fixture
def protocol_inputs(tmp_path, records, monkeypatch):
    manifest = _manifest(len(records))
    source = _source_identity(manifest)
    corpus = _write_corpus(tmp_path, manifest)
    tokenizer_manifest = _tokenizer_manifest(source)
    tokenizer = _write_tokenizer(tmp_path, tokenizer_manifest)
    repo = tmp_path / "repo"
    repo.mkdir()
    evaluator = repo / "evaluate.py"
    evaluator.write_text("def evaluate():\n    return True\n", encoding="utf-8")
    experiments = repo / "experiments"
    experiments.mkdir()
    prepare = experiments / "prepare_deep_corpus_v2.py"
    prepare.write_text("def prepare():\n    return True\n", encoding="utf-8")
    generator = experiments / "generate_deep_corpus.py"
    generator.write_text("def generate():\n    return True\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "_git_commit_is_ancestor", lambda *_: True)
    monkeypatch.setattr(
        freeze,
        "_git_file_identity",
        lambda _root, _commit, relative: freeze.file_identity(
            repo / relative, root=repo
        ),
    )
    return {
        "repo": repo,
        "evaluator": evaluator,
        "prepare": prepare,
        "generator": generator,
        "corpus": corpus,
        "manifest": manifest,
        "source": source,
        "tokenizer": tokenizer,
        "tokenizer_manifest": tokenizer_manifest,
        "records": records,
    }


@pytest.fixture
def artifacts(protocol_inputs):
    values = protocol_inputs
    return freeze.build_protocol_artifacts(
        repo_root=values["repo"],
        git_commit=COMMIT,
        corpus_dir=values["corpus"],
        corpus_manifest=values["manifest"],
        records=values["records"],
        verified_source_identity=values["source"],
        tokenizer_dir=values["tokenizer"],
        tokenizer_manifest=values["tokenizer_manifest"],
        source_paths=[
            values["evaluator"],
            values["prepare"],
            values["generator"],
        ],
        source_roots=("evaluate.py",),
    )


def test_default_evaluator_roots_predeclare_freezer_and_issue54_runner():
    assert "experiments/freeze_deep_corpus_v2_protocol.py" in (
        freeze.EVALUATOR_SOURCE_ROOTS
    )
    assert "experiments/evaluate_deep_corpus_v2_scratch.py" in (
        freeze.EVALUATOR_SOURCE_ROOTS
    )
    assert "experiments/create_deep_corpus_v2_scratch_report.py" in (
        freeze.EVALUATOR_SOURCE_ROOTS
    )
    assert "experiments/train_deep_corpus_sft.py" in freeze.REQUIRED_EVALUATOR_FILES
    for path in (
        "wandering_light/executor.py",
        "wandering_light/trajectory.py",
        "wandering_light/typed_list.py",
        "wandering_light/basis_set.py",
    ):
        assert path in freeze.REQUIRED_EVALUATOR_FILES


def test_evaluator_source_paths_follows_transitive_local_imports(tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "wandering_light").mkdir()
    root = tmp_path / "experiments" / "root.py"
    helper = tmp_path / "wandering_light" / "helper.py"
    leaf = tmp_path / "wandering_light" / "leaf.py"
    root.write_text("from wandering_light.helper import call\n", encoding="utf-8")
    helper.write_text("from wandering_light import leaf\n", encoding="utf-8")
    leaf.write_text("VALUE = 1\n", encoding="utf-8")

    paths = freeze.evaluator_source_paths(
        tmp_path,
        source_roots=("experiments/root.py",),
        static_files=(),
    )

    assert {path.relative_to(tmp_path).as_posix() for path in paths} == {
        "experiments/root.py",
        "wandering_light/helper.py",
        "wandering_light/leaf.py",
    }


def test_default_evaluator_source_closure_is_complete_in_worktree():
    paths = freeze.evaluator_source_paths(freeze.REPO_ROOT)
    relative = {path.relative_to(freeze.REPO_ROOT).as_posix() for path in paths}

    assert relative >= freeze.REQUIRED_EVALUATOR_FILES
    assert "experiments/evaluate_deep_corpus_v2_scratch.py" in relative
    assert "experiments/train_deep_corpus_v2_scratch.py" in relative
    assert "wandering_light/training/deep_corpus_scratch.py" in relative


def test_evaluator_source_paths_rejects_a_missing_declared_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="dependency root is missing"):
        freeze.evaluator_source_paths(
            tmp_path,
            source_roots=("experiments/future_runner.py",),
            static_files=(),
        )


def test_dirty_tree_gate_allows_only_explicit_output_and_payload_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Protocol Test"],
        check=True,
    )
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "initial"], check=True
    )
    output_file = repo / "artifacts" / "partial.json"
    output_file.parent.mkdir()
    output_file.write_text("{}", encoding="utf-8")
    payload = repo / "corpus" / "validation.jsonl.gz"
    payload.parent.mkdir()
    payload.write_bytes(b"payload")

    commit = freeze.assert_clean_source_tree(
        repo,
        ignored_paths=(output_file.parent, payload),
    )
    assert len(commit) == 40

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"dirty source tree.*source.py"):
        freeze.assert_clean_source_tree(
            repo,
            ignored_paths=(output_file.parent, payload),
        )


def test_manifest_gate_requires_exact_profile_basis_seed_and_pinned_revision(records):
    manifest = _manifest(len(records))
    freeze._validate_issue_54_manifest(manifest)

    mutations = (
        ("basis_set_digest", "sha256:" + "9" * 64),
        ("seed", 1),
    )
    for field, value in mutations:
        changed = deepcopy(manifest)
        changed[field] = value
        with pytest.raises(ValueError):
            freeze._validate_issue_54_manifest(changed)
    changed = deepcopy(manifest)
    changed["target_contract"]["profile"] = "some-other-profile"
    with pytest.raises(ValueError, match="profile"):
        freeze._validate_issue_54_manifest(changed)
    changed = deepcopy(manifest)
    changed["hub"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable Hub revision"):
        freeze._validate_issue_54_manifest(changed)
    changed = deepcopy(manifest)
    changed["target_contract"]["generator_source"]["tree_state"] = "dirty"
    with pytest.raises(ValueError, match="generator-source provenance"):
        freeze._validate_issue_54_manifest(changed)
    with pytest.raises(ValueError, match="protocol source commit is malformed"):
        freeze._validate_issue_54_manifest(manifest, expected_git_commit="main")


def test_tokenizer_source_must_equal_fully_reconstructed_source(protocol_inputs):
    values = protocol_inputs
    freeze._validate_tokenizer_source(
        values["tokenizer_manifest"],
        corpus_manifest=values["manifest"],
        verified_source_identity=values["source"],
        basis_identity=BASIS.identity_dict(),
    )

    changed = deepcopy(values["tokenizer_manifest"])
    changed["source"]["discovery"]["ordered_task_ids_digest"] = "sha256:" + "9" * 64
    with pytest.raises(ValueError, match="currently verified"):
        freeze._validate_tokenizer_source(
            changed,
            corpus_manifest=values["manifest"],
            verified_source_identity=values["source"],
            basis_identity=BASIS.identity_dict(),
        )


def test_artifacts_freeze_exact_cohorts_generation_bfs_and_file_identities(
    artifacts, protocol_inputs
):
    lock = artifacts[freeze.PROTOCOL_LOCK_FILENAME]
    assert set(artifacts) == set(freeze.ARTIFACT_FILENAMES)
    assert lock["cohorts"]["validation"]["size"] == COHORT_SIZE
    assert lock["cohorts"]["test"]["size"] == COHORT_SIZE
    assert not set(lock["cohorts"]["validation"]["task_ids"]).intersection(
        lock["cohorts"]["test"]["task_ids"]
    )
    assert lock["generation"] == {
        "batch_size": 32,
        "do_sample": True,
        "generation_sha256": lock["generation"]["generation_sha256"],
        "max_new_tokens": 64,
        "max_sequence_length": 2048,
        "prompt_rendering": "PR53 Python repr under fixed hash seed",
        "pythonhashseed": "0",
        "sample_prefixes": [1, 8, 64],
        "samples_per_task": 64,
        "sampling_seed": 20260826,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    assert lock["bfs"]["budget"] == 20_000_000
    assert lock["bfs"]["max_depth"] == 8
    assert lock["bfs"]["budget_role"] == "per-task execution safety ceiling"
    assert len(lock["corpus"]["files"]) == 4
    assert {row["path"] for row in lock["tokenizer"]["files"]} == {
        freeze.TOKENIZER_MANIFEST_FILENAME,
        *freeze.TOKENIZER_FILENAMES,
    }
    assert lock["evaluator_sources"]["files"] == sorted(
        [
            {
                "path": path.relative_to(protocol_inputs["repo"]).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                protocol_inputs["evaluator"],
                protocol_inputs["prepare"],
                protocol_inputs["generator"],
            )
        ],
        key=lambda row: row["path"],
    )
    generator_source = lock["corpus"]["generator_source"]
    assert generator_source["git_commit"] == COMMIT
    assert generator_source["protocol_git_commit"] == COMMIT
    assert generator_source["generator_is_protocol_ancestor"] is True
    assert generator_source["tree_state"] == "clean"
    assert (
        generator_source["evaluator_source_sha256"]
        == lock["evaluator_sources"]["source_sha256"]
    )
    assert {row["path"]: row["sha256"] for row in generator_source["files"]} == {
        "experiments/prepare_deep_corpus_v2.py": sha256_file(
            protocol_inputs["prepare"]
        ),
        "experiments/generate_deep_corpus.py": sha256_file(
            protocol_inputs["generator"]
        ),
    }
    assert artifacts[freeze.VALIDATION_COHORT_FILENAME] == lock["cohorts"]["validation"]
    assert artifacts[freeze.TEST_COHORT_FILENAME] == lock["cohorts"]["test"]


def test_source_amendment_copies_study_inputs_and_preserves_generator_binding(
    artifacts, protocol_inputs
):
    predecessor = artifacts[freeze.PROTOCOL_LOCK_FILENAME]
    training_sources = predecessor["evaluator_sources"]
    protocol_inputs["evaluator"].write_text("CHANGED = True\n", encoding="utf-8")
    successor_sources = freeze.build_evaluator_source_identity(
        [
            protocol_inputs["evaluator"],
            protocol_inputs["prepare"],
            protocol_inputs["generator"],
        ],
        root=protocol_inputs["repo"],
        metadata={
            **training_sources["metadata"],
            "git_commit": "b" * 40,
        },
    )

    amended = freeze.build_amended_protocol_artifacts(
        artifacts,
        successor_evaluator_sources=successor_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=["evaluate.py"],
    )
    successor = amended[freeze.PROTOCOL_LOCK_FILENAME]

    assert (
        successor["source_amendment"]["predecessor_protocol_lock_sha256"]
        == predecessor["protocol_lock_sha256"]
    )
    assert (
        successor["source_amendment"]["training_evaluator_sources"] == training_sources
    )
    assert (
        successor["corpus"]["generator_source"]
        == predecessor["corpus"]["generator_source"]
    )
    for field in set(predecessor) - {
        "evaluator_sources",
        "protocol_lock_sha256",
    }:
        assert successor[field] == predecessor[field]
    assert (
        amended[freeze.VALIDATION_COHORT_FILENAME]
        == artifacts[freeze.VALIDATION_COHORT_FILENAME]
    )
    assert (
        amended[freeze.TEST_COHORT_FILENAME] == artifacts[freeze.TEST_COHORT_FILENAME]
    )
    with pytest.raises(ValueError, match="cannot itself be amended"):
        freeze.build_amended_protocol_artifacts(
            amended,
            successor_evaluator_sources=successor_sources,
            reason="Attempt to chain an amendment.",
            changed_source_allowlist=["evaluate.py"],
        )

    tampered_corpus = deepcopy(successor["corpus"])
    binding = tampered_corpus["generator_source"]
    binding["protocol_git_commit"] = "b" * 40
    binding_body = deepcopy(binding)
    binding_body.pop("generator_source_sha256")
    binding["generator_source_sha256"] = freeze.sha256_json(binding_body)
    tampered_predecessor = freeze.build_protocol_lock(
        corpus_identity=tampered_corpus,
        cohorts=successor["cohorts"],
        generation=successor["generation"],
        bfs=successor["bfs"],
        tokenizer=successor["tokenizer"],
        evaluator_sources=training_sources,
    )
    tampered_amendment = freeze.build_source_amendment(
        predecessor_protocol_lock_sha256=tampered_predecessor["protocol_lock_sha256"],
        training_evaluator_sources=training_sources,
        successor_evaluator_sources=successor_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=["evaluate.py"],
        failure_evidence=freeze.SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    )
    rebound = freeze.build_protocol_lock(
        corpus_identity=tampered_corpus,
        cohorts=successor["cohorts"],
        generation=successor["generation"],
        bfs=successor["bfs"],
        tokenizer=successor["tokenizer"],
        evaluator_sources=successor_sources,
        source_amendment=tampered_amendment,
    )
    with pytest.raises(ValueError, match="not bound to training sources"):
        validate_protocol_lock(rebound)


def test_historical_evaluator_sources_are_validated_from_git_objects(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Protocol Test"],
        check=True,
    )
    source = repo / "experiments" / "evaluate.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "predecessor"],
        check=True,
    )
    predecessor_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor_sources = freeze.build_evaluator_source_identity(
        [source],
        root=repo,
        metadata={
            "git_commit": predecessor_commit,
            "roots": ["experiments/evaluate.py"],
        },
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "successor"], check=True)
    successor_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert (
        freeze._validate_evaluator_sources_from_git(predecessor_sources, repo_root=repo)
        == predecessor_sources
    )
    assert freeze._git_commit_is_ancestor(repo, predecessor_commit, successor_commit)
    assert freeze._git_changed_paths(
        repo, predecessor=predecessor_commit, successor=successor_commit
    ) == ("experiments/evaluate.py",)

    falsely_backdated = freeze.build_evaluator_source_identity(
        [source],
        root=repo,
        metadata={
            "git_commit": predecessor_commit,
            "roots": ["experiments/evaluate.py"],
        },
    )
    with pytest.raises(ValueError, match="recorded Git object"):
        freeze._validate_evaluator_sources_from_git(falsely_backdated, repo_root=repo)

    added = repo / "experiments" / "new.py"
    added.write_text("VALUE = 3\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "add"], check=True)
    added_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="only modified tracked files"):
        freeze._git_changed_paths(
            repo, predecessor=successor_commit, successor=added_commit
        )


def test_source_amendment_freezer_writes_atomic_successor(
    tmp_path, artifacts, protocol_inputs, monkeypatch
):
    predecessor_dir = tmp_path / "predecessor"
    freeze.write_protocol_artifacts(
        predecessor_dir,
        artifacts,
        repo_root=protocol_inputs["repo"],
        tokenizer_dir=protocol_inputs["tokenizer"],
        records=protocol_inputs["records"],
    )
    protocol_inputs["evaluator"].write_text("CHANGED = True\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "assert_clean_source_tree", lambda *_, **__: "b" * 40)
    monkeypatch.setattr(
        freeze,
        "_validate_evaluator_sources_from_git",
        lambda identity, **_kwargs: freeze.validate_evaluator_source_identity(identity),
    )
    monkeypatch.setattr(
        freeze,
        "_git_changed_paths",
        lambda *_args, **_kwargs: ("evaluate.py",),
    )
    output = tmp_path / "successor"

    result = freeze.freeze_protocol_source_amendment(
        predecessor_protocol_dir=predecessor_dir,
        tokenizer_dir=protocol_inputs["tokenizer"],
        output_dir=output,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=("evaluate.py",),
        failed_result_artifact=tmp_path / "failed-validation-result",
        test_opening_ledger=tmp_path / "held-out" / "test_opening_ledger.json",
        scored_tasks=0,
        generated_candidates=0,
        repo_root=protocol_inputs["repo"],
        evaluator_source_roots=("evaluate.py",),
        evaluator_static_files=(
            "experiments/prepare_deep_corpus_v2.py",
            "experiments/generate_deep_corpus.py",
        ),
    )

    successor = json.loads(
        (output / freeze.PROTOCOL_LOCK_FILENAME).read_text(encoding="utf-8")
    )
    predecessor = artifacts[freeze.PROTOCOL_LOCK_FILENAME]
    assert result["status"] == "written"
    assert result["failure_evidence"] == freeze.SOURCE_AMENDMENT_FAILURE_EVIDENCE
    assert (
        successor["source_amendment"]["predecessor_protocol_lock_sha256"]
        == predecessor["protocol_lock_sha256"]
    )
    assert (
        successor["source_amendment"]["training_evaluator_sources"]
        == predecessor["evaluator_sources"]
    )
    assert {path.name for path in output.iterdir()} == set(freeze.ARTIFACT_FILENAMES)

    predecessor_alias = tmp_path / "predecessor-alias"
    predecessor_alias.symlink_to(predecessor_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="separate from predecessor/tokenizer"):
        freeze.freeze_protocol_source_amendment(
            predecessor_protocol_dir=predecessor_dir,
            tokenizer_dir=protocol_inputs["tokenizer"],
            output_dir=predecessor_alias / "nested-successor",
            reason="Suppress tokenizer token-type IDs before retrying validation.",
            changed_source_allowlist=("evaluate.py",),
            failed_result_artifact=tmp_path / "another-failed-result",
            test_opening_ledger=(
                tmp_path / "another-held-out" / "test_opening_ledger.json"
            ),
            scored_tasks=0,
            generated_candidates=0,
            repo_root=protocol_inputs["repo"],
            evaluator_source_roots=("evaluate.py",),
            evaluator_static_files=(
                "experiments/prepare_deep_corpus_v2.py",
                "experiments/generate_deep_corpus.py",
            ),
        )


def test_chained_source_amendment_freezer_pins_lineage_and_absence_evidence(
    tmp_path, artifacts, protocol_inputs, monkeypatch
):
    training_lock = artifacts[freeze.PROTOCOL_LOCK_FILENAME]
    training_sources = training_lock["evaluator_sources"]
    protocol_inputs["evaluator"].write_text("FIRST = True\n", encoding="utf-8")
    first_sources = freeze.build_evaluator_source_identity(
        [
            protocol_inputs["evaluator"],
            protocol_inputs["prepare"],
            protocol_inputs["generator"],
        ],
        root=protocol_inputs["repo"],
        metadata={**training_sources["metadata"], "git_commit": "b" * 40},
    )
    first_artifacts = freeze.build_amended_protocol_artifacts(
        artifacts,
        successor_evaluator_sources=first_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=("evaluate.py",),
    )
    predecessor_dir = tmp_path / "amended-predecessor"
    freeze.write_protocol_artifacts(
        predecessor_dir,
        first_artifacts,
        repo_root=protocol_inputs["repo"],
        tokenizer_dir=protocol_inputs["tokenizer"],
        records=None,
    )

    protocol_inputs["evaluator"].write_text("SECOND = True\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "assert_clean_source_tree", lambda *_, **__: "c" * 40)
    monkeypatch.setattr(
        freeze,
        "_validate_evaluator_sources_from_git",
        lambda identity, **_kwargs: freeze.validate_evaluator_source_identity(identity),
    )
    compared_commits = {}

    def changed_paths(_root, *, predecessor, successor):
        compared_commits.update(predecessor=predecessor, successor=successor)
        return ("evaluate.py",)

    monkeypatch.setattr(freeze, "_git_changed_paths", changed_paths)
    result_dir = tmp_path / "validation" / "scratch-10m"
    evidence_paths = {
        "failed_result_artifact": result_dir,
        "failed_selection_lock_artifact": result_dir / "selection_lock.json",
        "failed_wandb_receipt_artifact": (result_dir / "wandb_evaluation_receipt.json"),
        "test_opening_ledger": tmp_path / "test" / "test_opening_ledger.json",
    }
    common = {
        "predecessor_protocol_dir": predecessor_dir,
        "tokenizer_dir": protocol_inputs["tokenizer"],
        "reason": "Reuse completed validation BFS without changing its semantics.",
        "changed_source_allowlist": ("evaluate.py",),
        **evidence_paths,
        "candidate_tasks": 96,
        "generated_candidates": 6_144,
        "bfs_rows": 96,
        "bfs_budget_termination_task_id": (
            "5ce61393fdc9d8461d00f7094a38ce145eb4ca2ad27079c07b7eda5954b7e46a"
        ),
        "bfs_budget_termination_distance": 8,
        "bfs_budget_termination_executions": 20_000_000,
        "test_access_count": 0,
        "repo_root": protocol_inputs["repo"],
        "evaluator_source_roots": ("evaluate.py",),
        "evaluator_static_files": (
            "experiments/prepare_deep_corpus_v2.py",
            "experiments/generate_deep_corpus.py",
        ),
    }
    output = tmp_path / "chained-successor"
    result = freeze.freeze_protocol_chained_source_amendment(
        output_dir=output,
        **common,
    )

    successor = json.loads(
        (output / freeze.PROTOCOL_LOCK_FILENAME).read_text(encoding="utf-8")
    )
    first = first_artifacts[freeze.PROTOCOL_LOCK_FILENAME]
    assert result["status"] == "written"
    assert compared_commits == {"predecessor": "b" * 40, "successor": "c" * 40}
    assert result["failure_evidence"] == (
        freeze.CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE
    )
    assert result["checked_absent_paths"] == {
        label: str(path) for label, path in evidence_paths.items()
    }
    assert successor["source_amendment"]["schema_version"] == 2
    assert (
        successor["source_amendment"]["predecessor_protocol_lock_sha256"]
        == first["protocol_lock_sha256"]
    )
    assert (
        successor["source_amendment"]["predecessor_evaluator_sources"]
        == first["evaluator_sources"]
    )
    assert (
        successor["source_amendment"]["predecessor_source_amendment"]
        == first["source_amendment"]
    )
    for field in ("corpus", "cohorts", "generation", "bfs", "tokenizer"):
        assert successor[field] == first[field]
    assert {path.name for path in output.iterdir()} == set(freeze.ARTIFACT_FILENAMES)

    with pytest.raises(ValueError, match="BFS rows=96"):
        freeze.freeze_protocol_chained_source_amendment(
            output_dir=tmp_path / "wrong-count-successor",
            **{**common, "bfs_rows": 95},
        )
    evidence_paths["failed_selection_lock_artifact"].parent.mkdir(
        parents=True, exist_ok=True
    )
    evidence_paths["failed_selection_lock_artifact"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refuses existing validation/test"):
        freeze.freeze_protocol_chained_source_amendment(
            output_dir=tmp_path / "existing-output-successor",
            **common,
        )


def test_protocol_revalidation_detects_evaluator_source_mutation(
    artifacts, protocol_inputs
):
    lock = artifacts[freeze.PROTOCOL_LOCK_FILENAME]
    protocol_inputs["evaluator"].write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source identity file changed"):
        validate_protocol_lock(
            lock,
            records=protocol_inputs["records"],
            tokenizer_root=protocol_inputs["tokenizer"],
            evaluator_root=protocol_inputs["repo"],
        )


def test_generator_source_binding_rejects_unfrozen_or_mutated_sources(
    protocol_inputs, monkeypatch
):
    values = protocol_inputs
    source_paths = [values["evaluator"], values["prepare"], values["generator"]]
    evaluator_sources = freeze.build_evaluator_source_identity(
        source_paths,
        root=values["repo"],
        metadata={"git_commit": COMMIT},
    )

    binding = freeze._generator_source_binding(
        values["manifest"],
        repo_root=values["repo"],
        git_commit=COMMIT,
        evaluator_sources=evaluator_sources,
    )
    assert binding["generator_source_sha256"].startswith("sha256:")

    monkeypatch.setattr(freeze, "_git_commit_is_ancestor", lambda *_: False)
    with pytest.raises(ValueError, match="not an ancestor"):
        freeze._generator_source_binding(
            values["manifest"],
            repo_root=values["repo"],
            git_commit=COMMIT,
            evaluator_sources=evaluator_sources,
        )
    monkeypatch.setattr(freeze, "_git_commit_is_ancestor", lambda *_: True)

    original_git_identity = freeze._git_file_identity
    monkeypatch.setattr(
        freeze,
        "_git_file_identity",
        lambda root, commit, relative: (
            {"path": relative, "bytes": 1, "sha256": "sha256:" + "0" * 64}
            if relative.endswith("generate_deep_corpus.py")
            else original_git_identity(root, commit, relative)
        ),
    )
    with pytest.raises(ValueError, match="changed after the recorded"):
        freeze._generator_source_binding(
            values["manifest"],
            repo_root=values["repo"],
            git_commit=COMMIT,
            evaluator_sources=evaluator_sources,
        )
    monkeypatch.setattr(freeze, "_git_file_identity", original_git_identity)

    incomplete_sources = freeze.build_evaluator_source_identity(
        [values["evaluator"], values["prepare"]],
        root=values["repo"],
        metadata={"git_commit": COMMIT},
    )
    with pytest.raises(ValueError, match="not covered by the frozen source"):
        freeze._generator_source_binding(
            values["manifest"],
            repo_root=values["repo"],
            git_commit=COMMIT,
            evaluator_sources=incomplete_sources,
        )

    values["generator"].write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not covered by the frozen source"):
        freeze._generator_source_binding(
            values["manifest"],
            repo_root=values["repo"],
            git_commit=COMMIT,
            evaluator_sources=evaluator_sources,
        )


def test_atomic_writer_publishes_exact_three_files_and_requires_fresh_destination(
    tmp_path, artifacts, protocol_inputs
):
    output = tmp_path / "locks" / "issue-54"
    result = freeze.write_protocol_artifacts(
        output,
        artifacts,
        repo_root=protocol_inputs["repo"],
        tokenizer_dir=protocol_inputs["tokenizer"],
        records=protocol_inputs["records"],
    )

    assert result == output.resolve()
    assert {path.name for path in output.iterdir()} == set(freeze.ARTIFACT_FILENAMES)
    loaded = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in output.iterdir()
    }
    assert loaded == artifacts
    with pytest.raises(FileExistsError, match="fresh path"):
        freeze.write_protocol_artifacts(
            output,
            artifacts,
            repo_root=protocol_inputs["repo"],
            tokenizer_dir=protocol_inputs["tokenizer"],
            records=protocol_inputs["records"],
        )


def test_atomic_writer_refuses_dangling_failure_evidence_path(
    tmp_path, artifacts, protocol_inputs
):
    dangling = tmp_path / "missing-result"
    dangling.symlink_to(tmp_path / "does-not-exist")
    output = tmp_path / "amended-lock"

    with pytest.raises(RuntimeError, match="output appeared"):
        freeze.write_protocol_artifacts(
            output,
            artifacts,
            repo_root=protocol_inputs["repo"],
            tokenizer_dir=protocol_inputs["tokenizer"],
            records=protocol_inputs["records"],
            must_remain_absent=(dangling,),
        )

    assert not output.exists()


def test_atomic_writer_refuses_dangling_and_racing_output_symlinks(
    tmp_path, artifacts, protocol_inputs, monkeypatch
):
    dangling = tmp_path / "dangling-output"
    dangling.symlink_to(tmp_path / "missing-output-target")
    with pytest.raises(FileExistsError, match="fresh path"):
        freeze.write_protocol_artifacts(
            dangling,
            artifacts,
            repo_root=protocol_inputs["repo"],
            tokenizer_dir=protocol_inputs["tokenizer"],
            records=protocol_inputs["records"],
        )

    racing = tmp_path / "racing-output"
    real_validate = freeze.validate_protocol_lock

    def validate_then_race(*args, **kwargs):
        value = real_validate(*args, **kwargs)
        racing.symlink_to(tmp_path / "missing-race-target")
        return value

    monkeypatch.setattr(freeze, "validate_protocol_lock", validate_then_race)
    with pytest.raises(FileExistsError, match="appeared while it was being built"):
        freeze.write_protocol_artifacts(
            racing,
            artifacts,
            repo_root=protocol_inputs["repo"],
            tokenizer_dir=protocol_inputs["tokenizer"],
            records=protocol_inputs["records"],
        )
    assert racing.is_symlink()


def test_dry_run_fully_builds_but_does_not_create_output(
    tmp_path, protocol_inputs, monkeypatch
):
    values = protocol_inputs
    verified = SimpleNamespace(
        manifest=values["manifest"],
        basis=BASIS,
        source_identity=values["source"],
    )
    monkeypatch.setattr(freeze, "assert_clean_source_tree", lambda *_, **__: COMMIT)
    monkeypatch.setattr(freeze, "load_verified_v2_discovery", lambda *_, **__: verified)
    monkeypatch.setattr(
        freeze.deep,
        "load_corpus",
        lambda *_args, **_kwargs: (values["manifest"], values["records"]),
    )
    monkeypatch.setattr(
        freeze,
        "validate_frozen_tokenizer",
        lambda *_args, **_kwargs: values["tokenizer_manifest"],
    )
    output = tmp_path / "must-not-exist" / "locks"

    result = freeze.freeze_protocol(
        corpus_dir=values["corpus"],
        tokenizer_dir=values["tokenizer"],
        output_dir=output,
        repo_root=values["repo"],
        dry_run=True,
        evaluator_source_roots=("evaluate.py",),
        evaluator_static_files=(),
        extra_evaluator_sources=(
            "experiments/prepare_deep_corpus_v2.py",
            "experiments/generate_deep_corpus.py",
        ),
    )

    assert result["status"] == "dry-run"
    assert result["validation_tasks"] == result["test_tasks"] == COHORT_SIZE
    assert result["generation_seed"] == 20260826
    assert result["bfs_execution_safety_budget"] == 20_000_000
    assert result["wrote_files"] == []
    assert not output.exists()
    assert not output.parent.exists()


def test_corpus_payload_paths_rejects_nested_or_malformed_split_paths(tmp_path):
    manifest = _manifest(208)
    corpus = _write_corpus(tmp_path, manifest)
    assert [path.name for path in freeze.corpus_payload_paths(corpus)] == [
        "discovery.jsonl.gz",
        "validation.jsonl.gz",
        "test.jsonl.gz",
    ]

    manifest["splits"]["test"]["path"] = "../secret.jsonl.gz"
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="safe payload filename"):
        freeze.corpus_payload_paths(corpus)
