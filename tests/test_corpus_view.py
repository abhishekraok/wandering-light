"""Corpus loading, summarising and filtering, without a browser session."""

import gzip
import hashlib
import json

import pytest

from wandering_light.basis_dataset import BasisTaskRecord, write_basis_task_records
from wandering_light.evals import corpus_view
from wandering_light.typed_list import TypedList

BASIS_ID = "wl-core-v1"
BASIS_DIGEST = "sha256:" + "0" * 64


def _record(
    *,
    split: str,
    items: list[int],
    output: list[int],
    witness: list[str],
    distance: int,
    optimal_first: list[str] | None = None,
    optimal_last_ids: list[str] | None = None,
    certification: str = "complete-bfs-expansion",
    source_index: int = 0,
) -> BasisTaskRecord:
    return BasisTaskRecord.create(
        split=split,
        input_value=TypedList(items, item_type=int),
        output_value=TypedList(output, item_type=int),
        witness_function_ids=[f"bf:{name}:0" for name in witness],
        witness_function_names=witness,
        basis_set_id=BASIS_ID,
        basis_set_digest=BASIS_DIGEST,
        generator="test",
        seed=1,
        source_index=source_index,
        metadata={
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "certified_distance": distance,
            "certification": certification,
            "expansion_certified_depth": 6,
            "optimal_first_action_ids": [],
            "optimal_first_action_names": optimal_first or witness[:1],
            "optimal_first_actions_complete": True,
            "optimal_last_action_ids": optimal_last_ids or [],
            "optimal_last_actions_complete": True,
            "root_index": 0,
            "root_digest": "sha256:root",
            "distance_shell_size": 7,
        },
    )


@pytest.fixture
def corpus(tmp_path):
    """A two-split corpus directory with a manifest that matches its payload."""
    records = {
        "discovery": [
            _record(
                split="discovery", items=[1], output=[2], witness=["inc"], distance=1
            ),
            _record(
                split="discovery",
                items=[2],
                output=[8],
                witness=["double", "double"],
                distance=2,
                source_index=1,
            ),
        ],
        "test": [
            _record(
                split="test",
                items=[3],
                output=[9],
                witness=["square"],
                distance=1,
                certification="frontier-extension",
            ),
        ],
    }
    root = tmp_path / "corpora" / "tiny_v1"
    root.mkdir(parents=True)
    splits = {}
    for split, split_records in records.items():
        path = write_basis_task_records(split_records, root / f"{split}.jsonl.gz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        splits[split] = {
            "path": path.name,
            "sha256": f"sha256:{digest}",
            "size": len(split_records),
            "roots": 1,
            "by_certified_distance": {
                str(distance): sum(
                    1
                    for record in split_records
                    if record.metadata["certified_distance"] == distance
                )
                for distance in sorted(
                    {r.metadata["certified_distance"] for r in split_records}
                )
            },
            "by_certification": {"complete-bfs-expansion": len(split_records)},
            "optimal_first_actions": {"mean": 1.0},
        }
    manifest = {
        "basis_set_id": BASIS_ID,
        "basis_set_digest": BASIS_DIGEST,
        "generator": "test-generator",
        "global_task_count": 3,
        "split_roots": {"discovery": [0], "test": [1]},
        "splits": splits,
        "expansion": {"reached_states": 42, "wall_seconds": 120},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus_view.CorpusRef(name=root.name, directory=root), manifest


def test_discover_corpora_finds_manifest_directories(corpus, tmp_path):
    ref, _ = corpus
    found = corpus_view.discover_corpora(tmp_path / "corpora")
    assert [r.name for r in found] == [ref.name]
    assert found[0].manifest_path == ref.manifest_path


def test_discover_corpora_on_missing_root_is_empty(tmp_path):
    assert corpus_view.discover_corpora(tmp_path / "nope") == []


def test_missing_splits_reports_absent_payload(corpus):
    ref, manifest = corpus
    assert corpus_view.missing_splits(ref, manifest) == []
    (ref.directory / "test.jsonl.gz").unlink()
    assert corpus_view.missing_splits(ref, manifest) == ["test"]


def test_load_records_enforces_manifest_provenance(corpus):
    ref, manifest = corpus
    records = corpus_view.load_records(ref, manifest, "discovery")
    assert [r.witness_function_names for r in records] == [
        ("inc",),
        ("double", "double"),
    ]

    wrong = {**manifest, "basis_set_id": "wl-pilot-compressed-v1"}
    with pytest.raises(ValueError, match="basis set ID mismatch"):
        corpus_view.load_records(ref, wrong, "discovery")


def test_load_records_limit_reads_a_prefix(corpus):
    ref, manifest = corpus
    assert len(corpus_view.load_records(ref, manifest, "discovery", limit=1)) == 1


def test_corpus_distance_profile_sums_selected_splits(corpus):
    _ref, manifest = corpus
    everything = corpus_view.corpus_distance_profile(manifest, name="tiny")
    assert everything.counts == {1: 2, 2: 1}
    assert everything.total == 3
    assert everything.share()[1] == pytest.approx(2 / 3)

    only_test = corpus_view.corpus_distance_profile(
        manifest, name="tiny", splits=["test"]
    )
    assert only_test.counts == {1: 1}


def test_corpus_headline_reads_manifest_fields(corpus):
    _ref, manifest = corpus
    headline = corpus_view.corpus_headline(manifest)
    assert headline["tasks"] == 3
    assert headline["roots"] == 2
    assert headline["basis_set_id"] == BASIS_ID
    assert headline["reached_states"] == 42


def test_relabel_profiles_separate_certified_from_nominal(tmp_path):
    records = [
        {"certified": True, "relabeled_length": 1, "original_length": 3},
        {"certified": True, "relabeled_length": 2, "original_length": 4},
        {"certified": False, "relabeled_length": 4, "original_length": 4},
        {"certified": True, "relabeled_length": 0, "original_length": 2},
    ]
    path = tmp_path / "relabel.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    certified, nominal = corpus_view.relabel_distance_profiles(
        corpus_view.read_relabel_records(path), name="legacy"
    )
    # Uncertified evidence and the identity relabel are both excluded.
    assert certified.counts == {1: 1, 2: 1}
    assert nominal.counts == {2: 1, 3: 1, 4: 2}
    assert certified.name == "legacy (certified)"


def test_profile_rows_carry_share_per_dataset():
    profile = corpus_view.DistanceProfile(name="d", counts={1: 1, 2: 3})
    rows = corpus_view.profile_rows([profile])
    assert [row["distance"] for row in rows] == [1, 2]
    assert rows[1]["share"] == pytest.approx(0.75)


def test_filter_records_combines_predicates(corpus):
    ref, manifest = corpus
    records = corpus_view.load_records(ref, manifest, "discovery") + (
        corpus_view.load_records(ref, manifest, "test")
    )
    assert len(corpus_view.filter_records(records, distances=[1])) == 2
    assert len(corpus_view.filter_records(records, function_name="double")) == 1
    assert corpus_view.filter_records(
        records, certifications=["frontier-extension"], distances=[1]
    )[0].witness_function_names == ("square",)
    assert corpus_view.filter_records(records, function_name="nope") == []


def test_function_stats_separates_uses_from_tasks(corpus):
    ref, manifest = corpus
    records = corpus_view.load_records(ref, manifest, "discovery")
    stats = corpus_view.function_stats(records)
    # `double` is applied twice inside one witness.
    assert stats["double"].witness_uses == 2
    assert stats["double"].witness_tasks == 1
    assert stats["double"].optimal_first == 1
    assert stats["inc"].by_distance == {1: 1}
    assert stats["double"].row()["mean_distance"] == pytest.approx(2.0)


def test_function_stats_counts_optimal_last_only_with_a_name_map(tmp_path):
    record = _record(
        split="test",
        items=[1],
        output=[2],
        witness=["inc"],
        distance=1,
        optimal_last_ids=["bf:inc:0"],
    )
    assert corpus_view.function_stats([record])["inc"].optimal_last == 0
    resolved = corpus_view.function_stats([record], id_to_name={"bf:inc:0": "inc"})
    assert resolved["inc"].optimal_last == 1


def test_record_row_flattens_metadata(corpus):
    ref, manifest = corpus
    row = corpus_view.record_row(corpus_view.load_records(ref, manifest, "test")[0])
    assert row["split"] == "test"
    assert row["distance"] == 1
    assert row["witness"] == "square"
    assert row["certification"] == "frontier-extension"
