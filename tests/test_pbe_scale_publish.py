"""Release metadata preserves source text and joins to existing task bytes."""

import copy
import gzip
import json

import pytest

from experiments.pbe_scale.publish import extract_metadata


def fixture(tmp_path):
    rows = []
    for name in ("kept", "excluded"):
        rows.append(
            {
                "candidate": {
                    "id": name,
                    "family": "fixture",
                    "template_id": "custom",
                    "input_kind": "int",
                    "reference": "def f(x):\n    raise RuntimeError(x)\n",
                    "program": "custom",
                    "macros": {"custom": "dup add"},
                    "parameters": {"value": 3},
                },
                "candidate_digest": f"digest-{name}",
                "status": "accepted",
                "signature": f"signature-{name}",
                "provenance_id": "origin",
                "verification_count": 21,
                "task": {
                    "behaviour": name,
                    "split": "train",
                    "program": ["dup", "add"],
                },
                "merge_attempts": [{"path": "/private/journal"}],
                "source": "/private/source",
            }
        )
    accepted = tmp_path / "accepted.jsonl"
    excluded = tmp_path / "exclusions.json"
    tasks = tmp_path / "tasks.jsonl.gz"
    excluded.write_text(
        json.dumps(
            {
                "excluded_hashes": ["digest-excluded"],
                "excluded_ids": ["excluded"],
            }
        )
    )
    with gzip.open(tasks, "wt") as stream:
        stream.write(json.dumps(rows[0]["task"]) + "\n")
    accepted.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows, accepted, excluded, tasks


def test_metadata_preserves_reference_custom_macros_and_exclusions(tmp_path):
    rows, accepted, excluded, tasks = fixture(tmp_path)
    first, second = tmp_path / "first.gz", tmp_path / "second.gz"
    summary = extract_metadata(accepted, excluded, tasks, first)
    assert summary == extract_metadata(accepted, excluded, tasks, second)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt") as stream:
        metadata = [json.loads(line) for line in stream]
    assert metadata == [
        {
            key: value
            for key, value in rows[0].items()
            if key not in {"task", "merge_attempts", "source"}
        }
    ]
    assert summary["metadata_records"] == 1
    assert summary["excluded_ids"] == ["excluded"]


@pytest.mark.parametrize(
    "defect", ["duplicate", "missing", "association", "task-content"]
)
def test_metadata_rejects_invalid_joins(tmp_path, defect):
    rows, accepted, excluded, tasks = fixture(tmp_path)
    rows = copy.deepcopy(rows)
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "missing":
        rows.pop(0)
    elif defect == "association":
        rows[0]["task"]["behaviour"] = "another-behavior"
    else:
        rows[0]["task"]["program"] = ["drop"]
    accepted.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "metadata.gz"
    with pytest.raises(ValueError):
        extract_metadata(accepted, excluded, tasks, output)
    assert not output.exists()
