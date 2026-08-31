"""The static stack-PBE overview is complete, pinned, and reproducible."""

import gzip
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments.create_stack_pbe_overview import render_report
from experiments.stack_pbe_scope import analyze_tasks, analyze_verified_corpus, main
from wandering_light.stack_pbe import Task, load

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "test_data" / "stack_pbe_fixture.jsonl.gz"
V4 = ROOT / "wandering_light" / "evals" / "data" / "stack_pbe_v4"


@pytest.fixture(scope="module")
def fixture_tasks() -> list[Task]:
    return load(FIXTURE)


@pytest.fixture(scope="module")
def fixture_analysis(fixture_tasks) -> dict:
    return analyze_tasks(fixture_tasks, corpus="fixture")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_verified_fixture(root: Path, tasks: list[Task]) -> Path:
    root.mkdir()
    split_metadata = {}
    for split in sorted({task.split for task in tasks}):
        rows = [task for task in tasks if task.split == split]
        path = root / f"{split}.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for task in rows:
                handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
        split_metadata[split] = {
            "path": path.name,
            "records": len(rows),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "behaviours": len({task.behaviour for task in rows}),
            "skeletons": len({task.skeleton for task in rows}),
        }

    shown = {value for task in tasks for value in task.inputs}
    hidden = {value for task in tasks for value in task.held_out_inputs}
    manifest = {
        "basis": "mark-28",
        "catalogue": {
            "behaviours": len({task.behaviour for task in tasks}),
            "skeletons": len({task.skeleton for task in tasks}),
            "digest": "sha256:fixture-catalogue",
        },
        "corpus": "stack_pbe_fixture",
        "counts": {
            "tasks": len(tasks),
            "shown_examples": sum(task.examples for task in tasks),
            "held_out_examples": sum(task.held_out for task in tasks),
            "distinct_input_stacks": len(shown | hidden),
        },
        "hub": {
            "repo_id": "acct/stack-pbe-fixture",
            "revision": "0123456789abcdef",
        },
        "split_policy": "skeleton",
        "splits": split_metadata,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_fixture_reproduces_v4_representative_metrics(fixture_analysis):
    analysis = fixture_analysis
    assert (
        analysis["tasks"],
        analysis["behaviours"],
        analysis["skeletons"],
    ) == (440, 220, 215)
    assert (
        analysis["min_length"],
        analysis["median_length"],
        analysis["max_length"],
    ) == (1, 43, 253)
    assert analysis["splits"]["train"] | {"behaviour_names": []} == {
        "tasks": 350,
        "behaviours": 175,
        "skeletons": 171,
        "behaviour_names": [],
    }
    assert analysis["splits"]["eval"] | {"behaviour_names": []} == {
        "tasks": 90,
        "behaviours": 45,
        "skeletons": 44,
        "behaviour_names": [],
    }
    assert analysis["behaviour_overlap"] == []
    assert analysis["skeleton_overlap"] == []

    overall = analysis["scaffolds"]["overall"]
    assert (overall["eligible"], overall["excluded_short"]) == (192, 28)
    assert overall["median"] == pytest.approx(0.7977272727272727)
    eval_to_train = analysis["scaffolds"]["eval_to_train"]
    assert (eval_to_train["eligible"], eval_to_train["excluded_short"]) == (36, 9)
    assert eval_to_train["median"] == pytest.approx(0.7679211469534051)
    assert analysis["inseparable_groups"]["size_histogram"] == {1: 220}


def test_input_shapes_are_parsed_and_exhaustive(fixture_analysis):
    shapes = {row["key"]: row for row in fixture_analysis["input_shapes"]}
    assert {key: row["tasks"] for key, row in shapes.items()} == {
        "list": 246,
        "list+list": 62,
        "list+scalar": 42,
        "list+scalar+scalar": 42,
        "scalar+scalar+scalar": 22,
        "scalar+scalar": 16,
        "scalar": 4,
        "scalar+scalar+scalar+scalar": 4,
        "list+list+list": 2,
    }
    assert sum(row["tasks"] for row in shapes.values()) == fixture_analysis["tasks"]
    assert (
        sum(row["behaviours"] for row in shapes.values())
        == fixture_analysis["behaviours"]
    )


def test_analysis_rejects_behaviour_metadata_that_drifts_between_rows(fixture_tasks):
    first, second = fixture_tasks[:2]
    assert first.behaviour == second.behaviour
    corrupted = replace(second, program=("sub",))

    with pytest.raises(ValueError, match="canonical behaviour metadata"):
        analyze_tasks([first, corrupted])


def test_verified_analysis_rejects_manifest_count_drift(tmp_path, fixture_tasks):
    corpus = _write_verified_fixture(tmp_path / "corpus", fixture_tasks)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["eval"]["records"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"splits\.eval\.records"):
        analyze_verified_corpus(corpus)


def test_renderer_is_deterministic_and_self_contained(tmp_path, fixture_tasks):
    corpus = _write_verified_fixture(tmp_path / "corpus", fixture_tasks)
    analysis = analyze_verified_corpus(corpus)

    first = render_report(analysis)
    second = render_report(analysis)

    assert first == second
    assert first.startswith("<!doctype html>")
    assert "<style>" in first
    assert first.count("<svg") >= 4
    assert "<script>" in first
    assert "<script src" not in first
    assert "<link" not in first
    assert "fetch(" not in first
    assert "8,800" not in first  # fixture task total, not a hard-coded live figure
    for text in (
        "Witness length",
        "Program structure vs. length",
        "Skeleton-disjoint split",
        "Input-shape mix",
        "Shared scaffolding",
        "Behaviours",
        "Provenance",
        analysis["provenance"]["manifest_sha256"],
    ):
        assert text in first


def test_text_cli_still_accepts_a_payload(capsys):
    assert main(["--corpus", str(FIXTURE)]) == 0
    output = capsys.readouterr().out
    assert "440 tasks over 220 behaviours" in output
    assert "median share" in output


def test_full_v4_snapshot_exact_metrics_when_payload_is_present():
    payloads = list(V4.glob("*.jsonl.gz"))
    if len(payloads) != 2:
        pytest.skip("v4 payload lives on the Hub")

    analysis = analyze_verified_corpus(V4)
    assert {
        "tasks": analysis["tasks"],
        "behaviours": analysis["behaviours"],
        "skeletons": analysis["skeletons"],
        "shown_examples": analysis["shown_examples"],
        "held_out_examples": analysis["held_out_examples"],
        "distinct_input_stacks": analysis["distinct_input_stacks"],
        "min_length": analysis["min_length"],
        "median_length": analysis["median_length"],
        "max_length": analysis["max_length"],
    } == {
        "tasks": 8800,
        "behaviours": 220,
        "skeletons": 215,
        "shown_examples": 39589,
        "held_out_examples": 70400,
        "distinct_input_stacks": 83488,
        "min_length": 1,
        "median_length": 43,
        "max_length": 253,
    }
    assert {row["key"]: row["tasks"] for row in analysis["input_shapes"]} == {
        "list": 4920,
        "list+list": 1240,
        "list+scalar": 840,
        "list+scalar+scalar": 840,
        "scalar+scalar+scalar": 440,
        "scalar+scalar": 320,
        "scalar": 80,
        "scalar+scalar+scalar+scalar": 80,
        "list+list+list": 40,
    }
    assert analysis["provenance"]["manifest_sha256"] == (
        "sha256:26fd110c3596a39e55b201944633f29bca8b6768a225f18a51bfffd80372518b"
    )
