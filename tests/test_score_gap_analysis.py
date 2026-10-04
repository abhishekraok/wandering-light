"""Synthetic phase-control gates, not real selection, inference or data attempt."""

import hashlib
import json
from dataclasses import replace

import pytest

from experiments.score_gap_analysis import analyze
from experiments.score_gap_profiles import Cell, Snapshot
from experiments.score_gap_storage import (
    AttemptReceipts,
    DiagnosticProvenanceError,
    PayloadBudget,
)


def panel(gap=0.0):
    groups = tuple(f"manual-{i}" for i in range(48))
    layers = (Snapshot((0.0,) * 39 + (-gap,), 40), Snapshot((0.0,) + (-100.0,) * 63, 1))
    cells = tuple(
        Cell(group, "selection" if i % 2 == 0 else "confirmation", seed, "fit", layers)
        for i, group in enumerate(groups)
        for seed in range(3)
    )
    return groups, cells


def run(tmp_path, cells, groups):
    receipts = AttemptReceipts(tmp_path / "synthetic-control")
    receipts.reserve({"synthetic": True})
    output = tmp_path / "analysis.jsonl"
    report = analyze(
        cells, groups, receipts, PayloadBudget(deadline=10.0, clock=lambda: 0.0), output
    )
    payload = output.read_bytes()
    assert report["bytes"] == len(payload)
    assert report["sha256"] == hashlib.sha256(payload).hexdigest()
    rows = [json.loads(line) for line in payload.splitlines()]
    lock = json.loads((receipts.root / "selection-lock.json").read_text())["metadata"]
    prefix = payload[: lock["selection_prefix_bytes"]]
    assert hashlib.sha256(prefix).hexdigest() == lock["selection_prefix_sha256"]
    return report, rows, lock


def test_pass_smallest_grid_exact_slots_lock_then_confirmation(tmp_path):
    groups, cells = panel(gap=0.5)
    report, rows, lock = run(tmp_path, cells, groups)
    assert report["terminal"] == "PASS_PROFILE_FEASIBILITY"
    assert report["tau"] == lock["tau"] == 0.5
    assert len([row for row in rows if row["kind"] == "selection_profile"]) == 24
    confirmation = [row for row in rows if row["kind"] == "confirmation_profile"]
    assert len(confirmation) == 3 and {row["tau"] for row in confirmation} == {0.5}
    assert next(
        i for i, row in enumerate(rows) if row["kind"] == "selection_lock"
    ) < next(i for i, row in enumerate(rows) if row["kind"] == "confirmation_profile")
    assert confirmation[0]["band_slots_equal_group"] == {
        "numerator": 41,
        "denominator": 104,
    }
    assert confirmation[0]["prefix_slots_equal_group"] == {
        "numerator": 8,
        "denominator": 13,
    }
    assert confirmation[0]["gain"] == {"numerator": 1, "denominator": 1}
    assert len([row for row in rows if row["kind"] == "prefix_curve"]) == 384
    assert confirmation[0]["per_depth"][0]["present_cells"] == 24


def test_confirmation_support_precedes_selection_headroom(tmp_path):
    groups, cells = panel()
    perfect = (Snapshot((0.0,) * 40, 1), Snapshot((0.0,) * 64, 1))
    changed = tuple(
        replace(cell, layers=perfect)
        if cell.half == "selection"
        else replace(cell, termination="budget", layers=())
        for cell in cells
    )
    report, rows, lock = run(tmp_path, changed, groups)
    assert report["terminal"] == "INSUFFICIENT_LINEAGE_SUPPORT"
    assert lock["tau"] is None
    assert not any(row["kind"] == "confirmation_profile" for row in rows)
    assert len([row for row in rows if row["kind"] == "selection_profile"]) == 24
    population = next(
        row
        for row in rows
        if row["kind"] == "population" and row["half"] == "confirmation"
    )
    assert population["groups"] == population["no_scored_layers"] == 24


def test_selection_headroom_null_lock_not_route_kill(tmp_path):
    groups, cells = panel()
    perfect = (Snapshot((0.0,) * 40, 1), Snapshot((0.0,) * 64, 1))
    report, rows, lock = run(
        tmp_path, tuple(replace(cell, layers=perfect) for cell in cells), groups
    )
    assert report["terminal"] == "NO_HEADROOM_OVER_PREFIX32"
    assert lock["tau"] is None
    assert not any(row["kind"] == "confirmation_profile" for row in rows)


def test_confirmation_failure_never_rechooses_wider_grid(tmp_path):
    groups, cells = panel(gap=0.5)
    layers = (Snapshot((0.0,) * 39 + (-0.75,), 40), cells[0].layers[1])
    changed = tuple(
        replace(cell, layers=layers) if cell.half == "confirmation" else cell
        for cell in cells
    )
    report, rows, lock = run(tmp_path, changed, groups)
    assert report["terminal"] == "NO_CONFIRMATION_PROFILE"
    assert report["tau"] == lock["tau"] == 0.5
    assert {row["tau"] for row in rows if row["kind"] == "confirmation_profile"} == {
        0.5
    }
    assert all(
        not check["protection"] for check in report["confirmation_checks"].values()
    )


def test_wrong_order_and_missing_cells_rejected_before_lock(tmp_path):
    groups, cells = panel()
    with pytest.raises(DiagnosticProvenanceError, match="order"):
        run(tmp_path, tuple(reversed(cells)), groups)
    assert not (tmp_path / "synthetic-control/selection-lock.json").exists()
    assert not (tmp_path / "analysis.jsonl").exists()
