"""Merging journal evidence does not require executing candidate code."""

import gzip
import json
from pathlib import Path

from experiments.pbe_scale.merge import merge
from experiments.pbe_scale.validate import canonical, digest


def _record(identifier, signature, status="accepted", length=5, **candidate_updates):
    candidate = {
        "id": identifier,
        "family": "aggregation",
        "template_id": "summary",
        "program": "sum",
        **candidate_updates,
    }
    record = {
        "candidate": candidate,
        "candidate_digest": digest(candidate),
        "status": status,
        "stage": "verified" if status == "accepted" else "reference",
        "signature": signature,
    }
    if status != "accepted":
        return record | {"reason": "allowlist missing helper"}
    task = {
        "behaviour": identifier,
        "split": "train",
        "skeleton": "fold",
        "length": length,
    }
    return record | {"task": task, "verification_count": 80, "scalar_exhaustive": False}


def _source(path, records, seed=42, legacy=False, **config_updates):
    (path / "records").mkdir(parents=True)
    (path / "provenance").mkdir()
    origin = {
        "config": {
            "seed": seed,
            "dev_share": 0.0,
            "shared_probes": 64,
            **config_updates,
        },
        "source_data_digests": {str(path / "input.jsonl"): "data-sha"},
        "dirty": True,
    }
    key = digest(origin)
    if not legacy:
        (path / "provenance" / f"{key}.json").write_text(canonical(origin))
    for index, original in enumerate(records):
        record = dict(original)
        record["provenance" if legacy else "provenance_id"] = origin if legacy else key
        (path / "records" / f"{index:08d}.json").write_text(canonical(record))
    (path / "summary.json").write_text(canonical({"provenance": origin}))


def _rows(path, name):
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


def test_cross_directory_duplicates_and_id_collision(tmp_path: Path):
    a, b, output = (tmp_path / name for name in ("a", "b", "out"))
    _source(a, [_record("first", "same-behavior", length=4)])
    _source(
        b,
        [
            _record("second", "same-behavior"),
            _record("first", "different-behavior", program="len", length=6),
        ],
    )
    summary = merge([b, a], output)
    assert (summary["accepted"], summary["rejected"]) == (1, 2)
    rejected = _rows(output, "rejected.jsonl")
    assert rejected[0]["duplicate_of"] == "first"
    assert rejected[0]["stage"] == "dedup"
    assert rejected[1]["stage"] == "schema"
    assert "different content" in rejected[1]["reason"]
    assert all("task" not in row for row in rejected)
    assert summary["merge"]["replay_performed"] is False
    with gzip.open(output / "tasks_train.jsonl.gz", "rt") as handle:
        assert [json.loads(line)["behaviour"] for line in handle] == ["first"]


def test_accepted_attempt_preferred_and_provenance_compact(tmp_path: Path):
    a, b, output = (tmp_path / name for name in ("a", "b", "out"))
    _source(a, [_record("same", "fingerprint", status="rejected")], legacy=True)
    _source(b, [_record("same", "fingerprint")])
    summary = merge([a, b], output)
    assert summary["accepted"] == summary["candidates"] == 1
    assert summary["merge"]["source_records"] == 2
    row = _rows(output, "accepted.jsonl")[0]
    assert "provenance" not in row
    assert len(row["merge_attempts"]) == 2
    assert [attempt["status"] for attempt in row["merge_attempts"]] == [
        "rejected",
        "accepted",
    ]
    assert all(len(attempt["sha256"]) == 64 for attempt in row["merge_attempts"])
    assert (output / "provenance" / f"{row['provenance_id']}.json").exists()
    assert str(a / "input.jsonl") in summary["provenance"]["source_data_digests"]
    assert str(b / "input.jsonl") in summary["provenance"]["source_data_digests"]
    assert "provenance" not in row["merge_attempts"][0]


def test_different_seed_fingerprints_are_not_deduplicated(tmp_path: Path):
    a, b, output = (tmp_path / name for name in ("a", "b", "out"))
    _source(a, [_record("first", "same-digest")], seed=42)
    _source(b, [_record("second", "same-digest")], seed=43)
    summary = merge([a, b], output)
    assert summary["accepted"] == 2
    assert summary["merge"]["fingerprint_namespaces"] == 2
    assert "Cross-namespace duplicates may remain" in summary["resume_caveats"][0]


def test_shortest_verified_representative_supersedes_earlier_long_witness(
    tmp_path: Path,
):
    a, b, output = (tmp_path / name for name in ("a", "b", "out"))
    _source(
        a,
        [_record("long-count", "length-function", length=178)],
        output="old",
        dev_share=0.1,
    )
    _source(
        b,
        [_record("short-count", "length-function", length=7)],
        output="new",
        dev_share=0.0,
    )
    summary = merge([a, b], output)
    assert summary["accepted"] == 1
    assert summary["merge"]["fingerprint_namespaces"] == 1
    assert _rows(output, "accepted.jsonl")[0]["candidate"]["id"] == "short-count"
    rejected = _rows(output, "rejected.jsonl")[0]
    assert rejected["candidate"]["id"] == "long-count"
    assert rejected["duplicate_of"] == "short-count"
    assert rejected["verified_witness_length"] == 178
    assert "shorter known verified witness" in rejected["reason"]


def test_equal_lengths_choose_smallest_candidate_digest(tmp_path: Path):
    a, b, output = (tmp_path / name for name in ("a", "b", "out"))
    first, second = _record("a", "same"), _record("b", "same")
    _source(a, [first])
    _source(b, [second])
    merge([a, b], output)
    winner = min([first, second], key=lambda record: record["candidate_digest"])
    assert (
        _rows(output, "accepted.jsonl")[0]["candidate_digest"]
        == winner["candidate_digest"]
    )
