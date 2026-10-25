"""Task augmentation reshuffles existing pairs without program execution."""

import gzip
import json

import pytest

from experiments.pbe_scale.augment import augment, views_for_task
from wandering_light.stack_pbe import assign_task_hash_split


def task(count=208, rare=False):
    return {
        "behaviour": f"behavior-{count}",
        "summary": "verified fixture",
        "split": "train",
        "task_hash_split": "train",
        "inputs": [f"input-{i}" for i in range(6)],
        "outputs": ["0"] * 6 if rare else [str(i % 3) for i in range(6)],
        "held_out_inputs": [f"input-{i}" for i in range(6, count)],
        "held_out_outputs": [
            str(int(i == count - 1)) if rare else str(i % 3) for i in range(6, count)
        ],
        "program": ["sum"],
        "length": 1,
        "examples": 6,
        "held_out": count - 6,
        "fuel": 123,
        "tags": ["fixture"],
        "skeleton": "fold",
    }


@pytest.mark.parametrize("count", [21, 208, 441])
def test_view_disjointness_diversity_and_preserved_witness(count):
    original = task(count, rare=True)
    rows = list(views_for_task(original, 16, 0))
    assert rows == list(views_for_task(original, 16, 0))
    assert len({frozenset(row["inputs"]) for row in rows}) == 16
    hidden = set(rows[0]["held_out_inputs"])
    assert len(hidden) == (5 if count == 21 else 32)
    assert not hidden & {value for row in rows for value in row["inputs"]}
    pairs = dict(
        zip(
            original["inputs"] + original["held_out_inputs"],
            original["outputs"] + original["held_out_outputs"],
            strict=True,
        )
    )
    for row in rows:
        assert len(set(row["inputs"])) == 6
        assert len(set(row["outputs"])) == 2
        assert row["held_out_inputs"] == rows[0]["held_out_inputs"]
        assert row["outputs"] == [pairs[value] for value in row["inputs"]]
        assert row["program"] == original["program"]
        assert row["behaviour"] == original["behaviour"]
        assert row["fuel"] == 0 and "fuel-unmeasured" in row["tags"]
        assert row["task_hash_split"] == assign_task_hash_split(
            tuple(row["inputs"]), tuple(row["outputs"])
        )


def test_stream_export_is_reproducible_and_counts_views_separately(tmp_path):
    source = tmp_path / "tasks.gz"
    with gzip.open(source, "wt") as stream:
        stream.write(json.dumps(task()) + "\n")
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    first = augment(source, a)
    second = augment(source, b)
    assert a.read_bytes() == b.read_bytes()
    assert first["distinct_behaviors"] == first["distinct_programs"] == 1
    assert first["task_views"] == first["full_prefix_rows"] == 16
    assert first["output_sha256"] == second["output_sha256"]
    assert first["execution_performed"] is False
    assert (tmp_path / "a.gz.manifest.json").exists()


def test_malformed_and_eval_tasks_fail():
    for change in [
        {"split": "eval"},
        {"outputs": []},
        {"length": 2},
        {"unexpected_id": "stale"},
    ]:
        with pytest.raises((ValueError, TypeError)):
            list(views_for_task(task() | change, 16, 0))


def test_compensating_shown_hidden_output_count_shift_is_rejected():
    original = task()
    shifted = original | {
        "outputs": original["outputs"][:-1],
        "held_out_outputs": [original["outputs"][-1], *original["held_out_outputs"]],
    }
    assert len(shifted["outputs"] + shifted["held_out_outputs"]) == 208
    with pytest.raises(ValueError, match="inconsistent example counts"):
        list(views_for_task(shifted, 16, 0))
