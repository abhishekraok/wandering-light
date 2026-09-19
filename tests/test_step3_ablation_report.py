"""The ablation reader: named-behaviour split and paired differences (keep-below-ablation, #136)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from experiments import step3_ablation_report as report

if TYPE_CHECKING:
    from pathlib import Path


def _row(behaviour: str, program: str, solved: bool) -> dict:
    return {"behaviour": behaviour, "program": program.split(), "solved": solved}


def test_cell_counts_split_out_the_named_behaviour() -> None:
    cell = {
        "rows": [
            _row("keep_below", "[ lt? ] swap cons filter", True),
            _row("keep_below", "[ lt? ] swap cons filter", True),
            _row("reverse", "reverse", True),
            _row("reverse", "bogus", True),
            _row("min_pair", "min", False),
        ]
    }
    witness = {("keep_below", "[ lt? ] swap cons filter")}
    passing = witness | {("reverse", "reverse")}
    assert report.cell_counts(cell, passing, "keep_below", witness) == {
        "verified": 3,
        "named": 2,
        "other": 1,
        "witness": 2,
        "named_tasks": 2,
    }
    assert report.cell_counts(cell, passing, "keep_below")["witness"] == 0


def test_paired_differences_skip_unsearched_cells() -> None:
    grid = {
        "mixed": {0: {1: {"verified": 10}, 2: {"verified": 12}}},
        "ablated": {0: {1: {"verified": 7}}},
    }
    assert report.paired(grid, "ablated", "mixed", "verified", [0], [1, 2]) == {
        1: [-3],
        2: [None],
    }


def _snapshot(run_dir: Path, seed: int, epoch: int, ce: float, rows: int) -> None:
    d = run_dir / "policies" / f"1m-s{seed}-e{epoch}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text(
        json.dumps({"validation_cross_entropy": ce, "train_rows": rows})
    )


def _cell(run_dir: Path, seed: int, epoch: int, named: int, other: int) -> None:
    rows = [_row("keep_below", "[ lt? ] swap cons filter", i < named) for i in range(3)]
    rows += [_row("reverse", "reverse", i < other) for i in range(3)]
    (run_dir / "timing").mkdir(exist_ok=True)
    (run_dir / "timing" / f"1m-s{seed}-e{epoch}.json").write_text(
        json.dumps(
            {
                "policy_epoch": epoch,
                "eval_corpus": "stack_pbe_v4",
                "beam_width": 64,
                "max_depth": 48,
                "rows": rows,
            }
        )
    )


def test_end_to_end_tables(tmp_path: Path) -> None:
    mixed, ablated = tmp_path / "mixed", tmp_path / "ablated"
    for epoch, ce in ((1, 0.9), (2, 0.85)):
        _snapshot(mixed, 0, epoch, ce, 400)
        _snapshot(ablated, 0, epoch, ce + 0.02, 300)
        _cell(mixed, 0, epoch, named=3, other=1)
        _cell(ablated, 0, epoch, named=0, other=2)
    verified = tmp_path / "verified.json"
    verified.write_text(
        json.dumps(
            {
                "programs": [
                    {
                        "behaviour": "keep_below",
                        "program": "[ lt? ] swap cons filter",
                        "verdict": "equals_catalogue_witness",
                    },
                    {
                        "behaviour": "reverse",
                        "program": "reverse",
                        "verdict": "genuine",
                        "draws": {},
                    },
                ]
            }
        )
    )
    out = tmp_path / "out"
    assert (
        report.main(
            [
                "--arm-dir",
                f"mixed={mixed}",
                "--arm-dir",
                f"ablated={ablated}",
                "--verified",
                str(verified),
                "--seeds",
                "0",
                "--epochs",
                "1",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads((out / "ablation.json").read_text())
    assert result["train_rows"] == {"mixed": 400, "ablated": 300}
    assert result["named_total"] == {"mixed": 6, "ablated": 0}
    # The witness-equal solves stay verified and are counted apart (#142).
    assert result["grid"]["mixed"]["0"]["1"]["verified"] == 4
    assert result["grid"]["mixed"]["0"]["1"]["witness"] == 3
    assert result["grid"]["ablated"]["0"]["1"]["witness"] == 0
    assert result["paired"]["ablated"]["verified"] == {"1": [-2], "2": [-2]}
    assert result["paired"]["ablated"]["other"] == {"1": [1], "2": [1]}
    assert result["best_ce"]["ablated"]["0"] == {"ce": 0.87, "epoch": 2}
    text = (out / "ablation.md").read_text()
    assert "| 2 | 4/3 | 2/0 |" in text


def test_an_unchecked_solved_program_is_refused_unless_allowed(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed"
    _snapshot(mixed, 0, 1, 0.9, 400)
    _cell(mixed, 0, 1, named=1, other=1)
    verified = tmp_path / "verified.json"
    verified.write_text(
        json.dumps(
            {
                "programs": [
                    {"behaviour": "reverse", "program": "reverse", "verdict": "genuine"}
                ]
            }
        )
    )
    args = [
        "--arm-dir",
        f"mixed={mixed}",
        "--verified",
        str(verified),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--out",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        report.main(args)
    assert report.main([*args, "--allow-unchecked"]) == 0
    result = json.loads((tmp_path / "out" / "ablation.json").read_text())
    assert result["grid"]["mixed"]["0"]["1"]["verified"] == 1
