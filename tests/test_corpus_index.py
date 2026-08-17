import gzip
import hashlib
import json
from pathlib import Path

import pytest

from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.basis_set import load_basis_set
from wandering_light.evals.corpus_index import (
    CorpusFilters,
    CorpusIndexError,
    corpus_source,
    discover_corpus_sources,
    ensure_corpus_index,
)
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _manifest_digest(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    payload.pop("hub", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _legacy_row(index: int, functions: list[str]) -> dict:
    return {
        "schema_version": 1,
        "split": "eval",
        "source_index": index,
        "input": TypedList([index], item_type=int).to_string(),
        "output": TypedList([index + len(functions)], item_type=int).to_string(),
        "original_functions": functions,
        "relabeled_functions": functions,
        "original_length": len(functions),
        "relabeled_length": len(functions),
        "lower_bound": len(functions),
        "upper_bound": len(functions),
        "certified": True,
        "method": "bfs",
    }


def _basis_record(
    *,
    source_index: int,
    function_names: list[str],
    optimal_first: list[str],
) -> BasisTaskRecord:
    basis = load_basis_set("default")
    functions = basis.as_function_set()
    selected = [functions.name_to_function[name] for name in function_names]
    input_value = TypedList([source_index], item_type=int)
    execution = Executor(functions).execute_trajectory(
        TrajectorySpec(input_value, FunctionDefList(selected))
    )
    assert execution.success
    ids_by_name = {
        function.name: function.metadata["basis_function_id"] for function in functions
    }
    return BasisTaskRecord.create(
        split="discovery",
        input_value=input_value,
        output_value=execution.trajectory.output,
        witness_function_ids=[ids_by_name[name] for name in function_names],
        witness_function_names=function_names,
        basis_set_id=basis.basis_set_id,
        basis_set_digest=basis.digest,
        generator="test-generator",
        seed=7,
        source_index=source_index,
        metadata={
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "certified_distance": len(function_names),
            "certification": "complete-bfs-expansion",
            "root_index": 12,
            "optimal_first_action_ids": [ids_by_name[name] for name in optimal_first],
            # Deliberately reverse names: deep-corpus action IDs and names are
            # independently sorted, so an index must resolve IDs through basis.
            "optimal_first_action_names": list(reversed(optimal_first)),
            "optimal_last_action_ids": [ids_by_name[function_names[-1]]],
        },
    )


def _write_manifest_corpus(path: Path) -> tuple[Path, list[BasisTaskRecord]]:
    basis = load_basis_set("default")
    records = [
        _basis_record(
            source_index=1,
            function_names=["inc"],
            optimal_first=["inc"],
        ),
        _basis_record(
            source_index=2,
            function_names=["inc", "double"],
            optimal_first=["inc", "double"],
        ),
    ]
    split_path = _write_rows(
        path / "discovery.jsonl.gz", [record.to_dict() for record in records]
    )
    manifest = {
        "schema_version": 1,
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "splits": {
            "discovery": {
                "path": split_path.name,
                "sha256": _sha256(split_path),
                "size": len(records),
            }
        },
        "hub": {"repo_id": "example/corpus", "revision": "abc123"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, records


def test_indexes_legacy_rows_and_pages_without_materializing(tmp_path):
    data_path = _write_rows(
        tmp_path / "legacy.jsonl.gz",
        [_legacy_row(0, ["inc"]), _legacy_row(1, ["inc", "double"])],
    )

    index = ensure_corpus_index(corpus_source(data_path), cache_dir=tmp_path / "cache")

    assert index.stats().records == 2
    assert index.values("distance") == (1, 2)
    assert index.count_records(CorpusFilters(min_distance=2)) == 1
    page = index.records(limit=1)
    assert len(page) == 1
    assert page[0].distance == 2
    detail = index.get_record(page[0].row_id)
    assert detail.schema_kind == "shortest-path-v1"
    assert detail.witness_function_names == ("inc", "double")
    assert detail.basis_set_id is None


def test_manifest_corpus_filters_by_stable_function_id_and_role(tmp_path):
    corpus_dir, _ = _write_manifest_corpus(tmp_path / "deep")
    source = corpus_source(corpus_dir)
    index = ensure_corpus_index(source, cache_dir=tmp_path / "cache")
    basis = load_basis_set("default")
    ids = {function.name: function.function_id for function in basis.functions}

    assert source.hub_repo_id == "example/corpus"
    assert index.stats().records == 2
    assert index.stats().roots == 1
    assert (
        index.count_records(
            CorpusFilters(
                function_keys=(ids["inc"], ids["double"]),
                function_match="all",
            )
        )
        == 1
    )
    assert (
        index.count_records(
            CorpusFilters(
                function_keys=(ids["double"],),
                function_roles=("optimal_first",),
            )
        )
        == 1
    )
    counts = index.function_counts(roles=("optimal_first",))
    assert {(item.function_key, item.function_name) for item in counts} >= {
        (ids["inc"], "inc"),
        (ids["double"], "double"),
    }
    totals = index.function_totals(roles=("witness", "optimal_first"))
    totals_by_key = {item.function_key: item.records for item in totals}
    assert totals_by_key[ids["inc"]] == 2
    assert totals_by_key[ids["double"]] == 1


def test_discovers_manifest_as_one_source_and_excludes_its_split_files(tmp_path):
    corpus_dir, _ = _write_manifest_corpus(tmp_path / "deep")
    standalone = _write_rows(tmp_path / "old.jsonl.gz", [_legacy_row(0, ["inc"])])

    sources = discover_corpus_sources([tmp_path])

    assert {source.path for source in sources} == {
        corpus_dir.resolve(),
        standalone.resolve(),
    }
    assert corpus_source(corpus_dir / "discovery.jsonl.gz").path == corpus_dir.resolve()


def test_rebuilds_index_when_standalone_source_changes(tmp_path):
    path = _write_rows(tmp_path / "legacy.jsonl.gz", [_legacy_row(0, ["inc"])])
    source = corpus_source(path)
    first = ensure_corpus_index(source, cache_dir=tmp_path / "cache")
    assert first.stats().records == 1

    _write_rows(
        path,
        [_legacy_row(0, ["inc"]), _legacy_row(1, ["inc", "double"])],
    )
    second = ensure_corpus_index(source, cache_dir=tmp_path / "cache")

    assert second.database_path == first.database_path
    assert second.stats().records == 2


def test_rejects_manifest_path_traversal(tmp_path):
    corpus_dir = tmp_path / "unsafe"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"splits": {"test": {"path": "../escape.jsonl.gz"}}}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusIndexError, match="unsafe corpus filename"):
        corpus_source(corpus_dir)


def test_discovery_does_not_downgrade_rejected_manifest_splits(tmp_path):
    corpus_dir = tmp_path / "unsafe"
    corpus_dir.mkdir()
    _write_rows(corpus_dir / "discovery.jsonl.gz", [_legacy_row(0, ["inc"])])
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"splits": {"test": {"path": "../escape.jsonl.gz"}}}),
        encoding="utf-8",
    )

    errors = []
    assert (
        discover_corpus_sources(
            [tmp_path], on_error=lambda path, error: errors.append((path, error))
        )
        == ()
    )
    assert len(errors) == 1
    assert "unsafe corpus filename" in str(errors[0][1])


def test_rejects_manifest_split_symlink_outside_corpus(tmp_path):
    outside = _write_rows(tmp_path / "outside.jsonl.gz", [_legacy_row(0, ["inc"])])
    corpus_dir = tmp_path / "deep"
    corpus_dir.mkdir()
    (corpus_dir / "discovery.jsonl.gz").symlink_to(outside)
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"splits": {"discovery": {"path": "discovery.jsonl.gz"}}}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusIndexError, match="escapes"):
        corpus_source(corpus_dir)
    assert [source.path for source in discover_corpus_sources([tmp_path])] == [
        outside.resolve()
    ]


def test_manifest_digest_excludes_hub_but_detects_body_tampering(tmp_path):
    corpus_dir, _ = _write_manifest_corpus(tmp_path / "deep")
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_digest"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert corpus_source(corpus_dir).hub_repo_id == "example/corpus"

    manifest["generator"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusIndexError, match="manifest digest mismatch"):
        corpus_source(corpus_dir)


def test_rejects_global_count_inconsistent_with_split_sizes(tmp_path):
    corpus_dir, _ = _write_manifest_corpus(tmp_path / "deep")
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["global_task_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CorpusIndexError, match="global_task_count"):
        corpus_source(corpus_dir)


def test_rejects_split_digest_mismatch(tmp_path):
    corpus_dir, _ = _write_manifest_corpus(tmp_path / "deep")
    split_path = corpus_dir / "discovery.jsonl.gz"
    split_path.write_bytes(split_path.read_bytes() + b"tampered")

    with pytest.raises(CorpusIndexError, match="digest mismatch"):
        ensure_corpus_index(corpus_source(corpus_dir), cache_dir=tmp_path / "cache")


def test_rejects_manifest_basis_digest_mismatch(tmp_path):
    corpus_dir, _ = _write_manifest_corpus(tmp_path / "deep")
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["basis_set_digest"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CorpusIndexError, match="installed"):
        ensure_corpus_index(corpus_source(corpus_dir), cache_dir=tmp_path / "cache")


def test_rejects_tampered_basis_task_identity(tmp_path):
    corpus_dir, records = _write_manifest_corpus(tmp_path / "deep")
    split_path = corpus_dir / "discovery.jsonl.gz"
    rows = [record.to_dict() for record in records]
    rows[0]["task_id"] = "0" * 64
    _write_rows(split_path, rows)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["discovery"]["sha256"] = _sha256(split_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CorpusIndexError, match="task_id mismatch"):
        ensure_corpus_index(corpus_source(corpus_dir), cache_dir=tmp_path / "cache")


def test_rejects_manifest_split_label_and_record_count_mismatches(tmp_path):
    corpus_dir, records = _write_manifest_corpus(tmp_path / "deep")
    split_path = corpus_dir / "discovery.jsonl.gz"
    rows = [record.to_dict() for record in records]
    rows[0]["split"] = "test"
    _write_rows(split_path, rows)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["discovery"]["sha256"] = _sha256(split_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CorpusIndexError, match="does not match manifest split"):
        ensure_corpus_index(corpus_source(corpus_dir), cache_dir=tmp_path / "cache")

    rows[0]["split"] = "discovery"
    _write_rows(split_path, rows)
    manifest["splits"]["discovery"]["sha256"] = _sha256(split_path)
    manifest["splits"]["discovery"]["size"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusIndexError, match="record count mismatch"):
        ensure_corpus_index(corpus_source(corpus_dir), cache_dir=tmp_path / "cache")


def test_schema_two_rows_cannot_fall_back_to_legacy_validation(tmp_path):
    row = _legacy_row(0, ["inc"])
    row["schema_version"] = 2
    path = _write_rows(tmp_path / "corrupt-v2.jsonl.gz", [row])

    with pytest.raises(CorpusIndexError, match="invalid corpus record"):
        ensure_corpus_index(corpus_source(path), cache_dir=tmp_path / "cache")


def test_task_prefix_lookup_is_bounded(tmp_path):
    corpus_dir, records = _write_manifest_corpus(tmp_path / "deep")
    index = ensure_corpus_index(corpus_source(corpus_dir), cache_dir=tmp_path / "cache")

    matches = index.find_task(records[0].task_id[:12])

    assert [match.task_id for match in matches] == [records[0].task_id]
    assert index.find_task("%") == ()
