"""Compare synthetic solve records; never run a solver or load an eval corpus."""

import gzip
import json

import pytest

from experiments.pbe_scale import eval_compare as compare


def records(tmp_path):
    payload = tmp_path / "tasks.jsonl.gz"
    with gzip.open(payload, "wt") as stream:
        stream.writelines(
            json.dumps({"behaviour": name}) + "\n" for name in ("a", "a", "b")
        )
    compare.write(
        tmp_path / "input_manifest.json",
        {
            "sha256": compare.sha(payload),
            "expected_sha256": compare.sha(payload),
            "tasks": 3,
            "behaviors": 2,
        },
    )
    for label, solved in (
        ("old", [True, False, False]),
        ("mixed", [False, True, True]),
    ):
        checkpoint = tmp_path / label
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(label.encode())
        compare.write(checkpoint / "policy_config.json", {"encoding": "machine"})
        compare.write(checkpoint / "trainer_state.json", {"seed": 0, "epoch": 3})
        wandb_key = "wandb_url" if label == "old" else "wandb_run.url"
        compare.write(
            checkpoint / "training_manifest.json",
            {wandb_key: f"https://wandb.test/{label}"},
        )
        compare.write(
            tmp_path / f"{label}.json",
            {
                "system": dict.fromkeys(compare.KNOBS, "fixed"),
                "checkpoint": str(checkpoint),
                "tasks_file": str(payload),
                "host": {"threads": 8},
                "commit": "test",
                "summary": {"seconds": 10, "executions": 123},
                "rows": [
                    {"index": i, "behaviour": "a" if i < 2 else "b", "solved": flag}
                    for i, flag in enumerate(solved)
                ],
            },
        )
    return tmp_path / "old.json", tmp_path / "mixed.json"


def test_paired_counts_and_plot(tmp_path):
    old, mixed = records(tmp_path)
    result = compare.compare(old, mixed, tmp_path)
    assert (
        result["task_gain"],
        result["task_loss"],
        result["task_delta"],
        result["behavior_delta"],
    ) == (2, 1, 1, 1)
    assert result["sources"]["old"]["wandb_run.url"] == "https://wandb.test/old"
    assert result["sources"]["mixed"]["wandb_run.url"] == "https://wandb.test/mixed"
    assert (tmp_path / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert (
        json.loads((tmp_path / "comparison.json").read_text())["status"] == "complete"
    )


def test_incomplete_is_not_a_zero(tmp_path):
    old, mixed = records(tmp_path)
    record = compare.read(mixed)
    record["rows"].pop()
    mixed.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incomplete"):
        compare.compare(old, mixed, tmp_path)
    assert not (tmp_path / "comparison.json").exists()
    mixed.unlink()
    with pytest.raises(FileNotFoundError):
        compare.compare(old, mixed, tmp_path)


@pytest.mark.parametrize(
    "change", ["duplicate", "wrong-index", "wrong-behaviour", "ungraded"]
)
def test_malformed_or_unpaired_rows_are_not_complete(tmp_path, change):
    old, mixed = records(tmp_path)
    record = compare.read(mixed)
    if change == "duplicate":
        record["rows"][1] = record["rows"][0]
    elif change == "wrong-index":
        record["rows"][0]["index"] = 100
    elif change == "wrong-behaviour":
        record["rows"][0]["behaviour"] = "other"
    else:
        record["rows"][0]["solved"] = "false"
    # Even two identically malformed runs must not pass the frozen-input check.
    for path in (old, mixed):
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"incomplete|boolean"):
        compare.compare(old, mixed, tmp_path)
    assert not (tmp_path / "comparison.json").exists()


def test_search_mismatch_is_a_caveat(tmp_path, monkeypatch):
    old, mixed = records(tmp_path)
    record = compare.read(mixed)
    record["system"]["max_depth"] = "different"
    mixed.write_text(json.dumps(record))
    monkeypatch.setattr(compare, "plot", lambda *args: None)
    result = compare.compare(old, mixed, tmp_path)
    assert any("Search settings differ" in caveat for caveat in result["caveats"])
    assert result["task_delta"] == 1


def test_checkpoint_and_digest_mismatches_are_advisory(tmp_path, monkeypatch):
    old, mixed = records(tmp_path)
    state = tmp_path / "mixed/trainer_state.json"
    state.write_text(json.dumps({"epoch": 2, "seed": 1}))
    frozen = compare.read(tmp_path / "input_manifest.json")
    frozen["expected_sha256"] = "different"
    (tmp_path / "input_manifest.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(compare, "plot", lambda *args: None)
    result = compare.compare(old, mixed, tmp_path)
    assert result["status"] == "complete"
    assert any("planned seed 0, epoch 3" in caveat for caveat in result["caveats"])
    assert any("payload digest differs" in caveat for caveat in result["caveats"])


def test_plot_failure_does_not_publish_complete_json(tmp_path, monkeypatch):
    old, mixed = records(tmp_path)

    def fail_plot(*args):
        raise OSError("cannot write figure")

    monkeypatch.setattr(compare, "plot", fail_plot)
    with pytest.raises(OSError, match="cannot write figure"):
        compare.compare(old, mixed, tmp_path)
    assert not (tmp_path / "comparison.json").exists()


def test_freeze_copies_only_manifest_eval_payload(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "eval.jsonl.gz"
    with gzip.open(source, "wt") as stream:
        stream.writelines(
            json.dumps({"behaviour": name}) + "\n" for name in ("a", "a", "b")
        )
    compare.write(
        manifest,
        {
            "splits": {
                "eval": {
                    "path": source.name,
                    "records": 3,
                    "behaviours": 2,
                    "sha256": "sha256:" + compare.sha(source),
                }
            },
            "hub": {},
        },
    )
    monkeypatch.setattr(compare, "V4", manifest)
    result = compare.freeze(tmp_path / "frozen")
    assert result["sha256"] == compare.sha(source)
    assert (result["tasks"], result["behaviors"]) == (3, 2)
    assert (tmp_path / "frozen/tasks.jsonl.gz").read_bytes() == source.read_bytes()
