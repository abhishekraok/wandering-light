"""stack_pbe_world_v1 builder (#120): rows are labelled by execution and round-trip."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wandering_light.stack_pbe import Task, load

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    path = ROOT / "experiments" / "world_behaviours" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(tmp_path: Path, program: str, macro_program: str) -> Path:
    behaviour = {
        "id": "mbpp-0",
        "source": "mbpp",
        "description": "Write a function to double a number. Then something else.",
        "tests": [{"args": [n], "out": 2 * n} for n in range(8)],
        "shown": [0, 1, 2],
        "held_out": [3, 4, 5, 6, 7],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "behaviours.jsonl").write_text(json.dumps(behaviour) + "\n")
    report = {
        "results": [
            {
                "id": "mbpp-0",
                "solved": True,
                "program": program,
                "macro_program": macro_program,
            }
        ]
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path


def test_build_writes_verified_rows_and_round_trips(tmp_path: Path) -> None:
    build_corpus = _module("build_corpus")
    probe = _probe(tmp_path / "probe", "dup add", "dup add")
    tasks = build_corpus.build(probe)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.program == ("dup", "add")
    assert task.macro_program == "dup add"
    assert task.witness_source == "translated"
    assert task.split == "train"
    assert task.inputs == ("[ 0 ]", "[ 1 ]", "[ 2 ]")
    assert task.outputs == ("[ 0 ]", "[ 2 ]", "[ 4 ]")
    assert task.held_out == 5
    assert task.summary == "Write a function to double a number."
    assert task.tags == ("mbpp",)

    corpus_dir = tmp_path / "corpus"
    build_corpus.write(tasks, corpus_dir, probe)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["split_policy"] == "train-only"
    loaded = load(corpus_dir)
    assert loaded == tasks


def test_repeated_inputs_are_not_shown_and_held_out_at_once(tmp_path: Path) -> None:
    build_corpus = _module("build_corpus")
    probe = _probe(tmp_path / "probe", "dup add", "dup add")
    behaviour = json.loads((probe / "behaviours.jsonl").read_text())
    behaviour["tests"][4] = behaviour["tests"][
        1
    ]  # a held-out input repeats a shown one
    (probe / "behaviours.jsonl").write_text(json.dumps(behaviour) + "\n")
    (task,) = build_corpus.build(probe)
    assert task.examples == 3
    assert task.held_out == 4
    assert not set(task.inputs) & set(task.held_out_inputs)


def test_build_refuses_a_witness_that_disagrees_with_the_tests(tmp_path: Path) -> None:
    build_corpus = _module("build_corpus")
    probe = _probe(tmp_path / "probe", "dup mul", "dup mul")
    with pytest.raises(Exception, match="disagrees"):
        build_corpus.build(probe)


def test_macro_program_defaults_for_older_payloads() -> None:
    task = Task(
        behaviour="b",
        summary="s",
        split="train",
        task_hash_split="train",
        inputs=("[ 1 ]",),
        outputs=("[ 2 ]",),
        held_out_inputs=(),
        held_out_outputs=(),
        program=("1", "add"),
        length=2,
        examples=1,
        held_out=0,
        fuel=2,
        tags=(),
    )
    assert task.macro_program == ""
