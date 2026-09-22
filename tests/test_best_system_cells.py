"""The report stages of the #158 cells driver and the #159 size driver, on synthetic cells."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.best_system_cells as cells
import experiments.size_at_fixed_epoch as size
import experiments.step3_verify_solves as verify
from wandering_light.solve import BEST


def row(behaviour, length, solved, program, source="search"):
    return {
        "behaviour": behaviour,
        "length": length,
        "bucket": "1-5" if length <= 5 else "6-10",
        "solved": solved,
        "shown_fit": solved,
        "program": program,
        "source": source,
        "seconds": 0.5,
        "executions": 100,
    }


ROWS = [
    row("sum", 5, True, ["0", "[", "add", "]", "fold"]),
    row("sum", 5, True, ["0", "[", "add", "]", "fold"], source="library"),
    row("add_pair", 1, True, ["add"]),
    row("last", 8, False, None),
]


@pytest.fixture
def fake_check(monkeypatch):
    """`sum`'s program is a catalogue witness; `add` is called a shortcut without draws."""

    def check(program, behaviour, draws, seed):
        # `add` really is `add_pair`'s catalogue witness, so a fake that keyed on
        # the catalogue would verify both rows and test nothing; key on the
        # program instead, and let `sum` be the verified one.
        if behaviour.name == "sum":
            return {"verdict": "equals_catalogue_witness"}
        return {"verdict": "shortcut"}

    monkeypatch.setattr(verify, "check", check)
    return check


def test_cells_report_counts_verified_behaviours_and_caches_verdicts(
    tmp_path, fake_check
):
    run_dir = tmp_path / "run"
    (run_dir / "timing").mkdir(parents=True)
    import dataclasses

    system = dataclasses.asdict(dataclasses.replace(BEST, beam_width=128))
    (run_dir / "timing" / "b128.json").write_text(
        json.dumps(
            {
                "cell": "b128",
                "system": system,
                "summary": {
                    "from_library": 1,
                    "memorised": 0,
                    "seconds": 2.0,
                    "executions": 400,
                },
                "host": {"hostname": "test"},
                "commit": "abc",
                "rows": ROWS,
            }
        )
    )
    cells.report(run_dir, tmp_path / "out", lambda line: None)
    table = json.loads((tmp_path / "out" / "cells.json").read_text())
    assert len(table) == 1 and table[0]["cell"] == "b128"
    assert (table[0]["tasks"], table[0]["graded_behaviours"]) == (3, 2)
    assert table[0]["verified_behaviours"] == 1 and table[0]["verified"] == ["sum"]
    assert table[0]["shortcut_only"] == ["add_pair"]
    assert table[0]["buckets"] == {
        "1-5": 3,
        "6-10": 0,
        "11-20": 0,
        "21-40": 0,
        "41+": 0,
    }
    cached = json.loads((run_dir / "verified.json").read_text())
    assert set(cached) == {"sum :: 0 [ add ] fold", "add_pair :: add"}
    text = (tmp_path / "out" / "tables.md").read_text()
    # tasks 3, of which 2 came from a search and 1 from the library.
    assert table[0]["search_solved"] == 2
    assert "| b128 | 128 | 16384 | on | 3 | 2 | 2 | **1** |" in text
    # The totals come from the task file: 4 tasks over 3 behaviours here.
    assert "| tasks /4 |" in text and "**verified /3**" in text
    assert table[0]["unverifiable"] == []


def test_cells_report_grades_world_tasks_without_calling_them_shortcuts(
    tmp_path, fake_check
):
    """A world behaviour has no sampler: graded, unverifiable, never a shortcut."""
    import dataclasses

    run_dir = tmp_path / "run"
    (run_dir / "timing").mkdir(parents=True)
    rows = [
        row("mbpp-436", 9, True, ["[", "0", "lt?", "]", "filter"]),
        row("codearc-695", 40, False, None),
        row("sum", 5, True, ["0", "[", "add", "]", "fold"]),
    ]
    (run_dir / "timing" / "b64.json").write_text(
        json.dumps(
            {
                "cell": "b64",
                "system": dataclasses.asdict(BEST),
                "summary": {
                    "from_library": 0,
                    "memorised": 0,
                    "seconds": 1.0,
                    "executions": 10,
                },
                "host": {"hostname": "test"},
                "commit": "abc",
                "rows": rows,
            }
        )
    )
    cells.report(run_dir, tmp_path / "out", lambda line: None)
    [cell] = json.loads((tmp_path / "out" / "cells.json").read_text())
    assert cell["graded_behaviours"] == 2
    assert cell["verified"] == ["sum"]
    assert cell["unverifiable"] == ["mbpp-436"]
    assert cell["shortcut_only"] == []
    cached = json.loads((run_dir / "verified.json").read_text())
    assert cached["mbpp-436 :: [ 0 lt? ] filter"] == {"verdict": cells.UNVERIFIABLE}
    assert "| tasks /3 |" in (tmp_path / "out" / "tables.md").read_text()


def test_size_report_grids_verified_behaviours_by_size_and_epoch(tmp_path, fake_check):
    run_dir = tmp_path / "run"
    (run_dir / "timing").mkdir(parents=True)
    for arm, seed, epoch, rows in (
        ("1m", 0, 3, ROWS),
        ("1m", 1, 3, ROWS[:1] + ROWS[3:]),
        ("10m", 0, 3, ROWS),
    ):
        (run_dir / "timing" / f"{arm}-s{seed}-cpu-e{epoch}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "seed": seed,
                    "policy_epoch": epoch,
                    "device": "cpu",
                    "rows": rows,
                }
            )
        )
    size.report(run_dir, tmp_path / "out", lambda line: None)
    table = json.loads((tmp_path / "out" / "cells.json").read_text())
    assert {(t["size"], t["seed"]) for t in table} == {("1m", 0), ("1m", 1), ("10m", 0)}
    by_key = {(t["size"], t["seed"]): t for t in table}
    assert by_key["1m", 0]["verified_behaviours"] == 1
    assert by_key["1m", 0]["verified_by_bucket"]["1-5"] == 1
    text = (tmp_path / "out" / "tables.md").read_text()
    assert "| epoch | 1m | 10m |" in text
    assert "| 3 | 1.00 (1 / 1) | 1.00 (1) |" in text
