"""The search audit records what it ran on and keeps the programs it found (#142)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from experiments import stack_pbe_search_audit as audit
from wandering_light.stack_pbe import load, write_corpus

FIXTURE = Path(__file__).resolve().parent / "test_data" / "stack_pbe_fixture.jsonl.gz"


def test_audit_one_keeps_every_program_that_survives_the_held_out_examples():
    task = next(t for t in load(FIXTURE) if t.behaviour == "add_pair")
    row = audit.audit_one(task, depth=task.length, basis=task.basis, index=7)
    assert row["fits_held_out"] == len(row["fits_held_out_programs"]) >= 1
    assert " ".join(task.program) in row["fits_held_out_programs"]
    assert (row["task_index"], row["split"]) == (7, task.split)
    assert row["task_id"] == audit.task_id(task) and len(row["task_id"]) == 12
    assert row["states"] > 0


def test_task_ids_follow_the_shown_examples_and_are_distinct():
    tasks = load(FIXTURE)
    ids = [audit.task_id(t) for t in tasks]
    assert len(set(ids)) == len(ids)
    assert audit.task_id(tasks[0]) == ids[0]


def test_indexing_does_not_change_which_tasks_a_seed_samples():
    """The seed-17 sample every report quotes must stay the same tasks."""
    tasks = load(FIXTURE)
    plain = list(tasks)
    random.Random(17).shuffle(plain)
    indexed = list(enumerate(tasks))
    random.Random(17).shuffle(indexed)
    assert [t for _, t in indexed][:40] == plain[:40]
    assert all(tasks[i] == t for i, t in indexed[:40])


def test_the_written_audit_records_its_provenance(tmp_path):
    tasks = [t for t in load(FIXTURE) if t.length == 1][:3]
    corpus = tmp_path / "corpus"
    write_corpus(tasks, corpus, config={"fixture": True})
    out = tmp_path / "audit.json"
    argv = ["--corpus", str(corpus), "--depth", "1", "--tasks", "2", "--seed", "3"]
    assert audit.main([*argv, "--out", str(out)]) == 0

    written = json.loads(out.read_text())
    manifest = json.loads((corpus / "manifest.json").read_text())
    assert (written["seed"], written["tasks"], written["depth"]) == (3, 2, 1)
    assert (written["basis"], written["guard"]) == (manifest["basis"], "off")
    assert written["corpus"] == str(corpus)
    assert written["corpus_name"] == manifest["corpus"]
    assert written["hub"] is None
    assert written["manifest_catalogue"] == manifest["catalogue"]
    assert written["catalogue_digest"] == manifest["catalogue"]["digest"]
    assert written["catalogue_source_digest"] == manifest["catalogue"]["source_digest"]
    assert len(written["rows"]) == 2
    for row in written["rows"]:
        assert {"task_index", "task_id", "split", "states"} <= set(row)
        assert row["fits_held_out"] == len(row["fits_held_out_programs"])
        assert tasks[row["task_index"]].split == row["split"]
