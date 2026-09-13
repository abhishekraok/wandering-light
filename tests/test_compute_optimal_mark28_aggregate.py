"""aggregate() reads one eval corpus and one search budget (Codex review on #135).

world-plus-v4-policy (#134) writes v4 cells and world cells into one run
directory under different filenames. aggregate() keys its tables by device, arm
and seed, so without scoping one corpus's cell silently overwrites the other's.
The committed timing cells of that run are the fixture.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments import compute_optimal_mark28 as driver

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "reports" / "world-plus-v4-policy-20260912" / "v4only" / "timing"
V4 = ROOT / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
WORLD = ROOT / "wandering_light/evals/data/stack_pbe_world_v3/manifest.json"


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    timing = tmp_path / "timing"
    timing.mkdir()
    for path in CELLS.glob("*.json"):
        shutil.copy(path, timing / path.name)
    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [V4, WORLD])
    monkeypatch.setattr(driver, "CORPUS_OF", {})
    monkeypatch.setattr(driver, "EVAL_CORPUS", None)
    monkeypatch.setattr(driver, "BEAM_WIDTH", driver.DEFAULT_BEAM_WIDTH)
    monkeypatch.setattr(driver, "MAX_DEPTH", driver.DEFAULT_MAX_DEPTH)
    return tmp_path


def test_the_primary_corpus_is_the_default_scope(run_dir: Path) -> None:
    cells = driver.aggregate_cells(run_dir)
    assert sorted(c["seed"] for c in cells) == [0, 1, 2]
    assert {len(c["rows"]) for c in cells} == {300}
    assert {c["eval_corpus"] for c in cells} == {"stack_pbe_v4"}


def test_a_second_corpus_is_read_on_its_own(run_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "EVAL_CORPUS", "stack_pbe_world_v3")
    cells = driver.aggregate_cells(run_dir)
    assert sorted(c["seed"] for c in cells) == [0, 1, 2]
    assert {len(c["rows"]) for c in cells} == {55}


def test_another_budget_is_not_mixed_in(run_dir: Path, monkeypatch) -> None:
    deep = json.loads((run_dir / "timing" / "1m-s0-cuda.json").read_text())
    deep["max_depth"] = 192
    (run_dir / "timing" / "1m-s0-cuda-d192.json").write_text(json.dumps(deep))
    cells = driver.aggregate_cells(run_dir)
    assert all(c["max_depth"] == 48 for c in cells)
    assert len(cells) == 3
    monkeypatch.setattr(driver, "MAX_DEPTH", 192)
    assert [c["max_depth"] for c in driver.aggregate_cells(run_dir)] == [192]


def test_two_cells_for_one_key_are_refused(run_dir: Path) -> None:
    twin = json.loads((run_dir / "timing" / "1m-s0-cuda.json").read_text())
    (run_dir / "timing" / "1m-s0-cuda-copy.json").write_text(json.dumps(twin))
    with pytest.raises(ValueError, match="second cell"):
        driver.aggregate_cells(run_dir)


def test_aggregate_writes_world_coverage_for_the_world_scope(
    run_dir: Path, monkeypatch
) -> None:
    """The reproduction Codex gave: world scope must not report v4 coverage."""
    monkeypatch.setattr(driver, "EVAL_CORPUS", "stack_pbe_world_v3")
    driver.aggregate(run_dir, lambda line: None)
    written = json.loads((run_dir / "aggregate.json").read_text())
    tasks = {
        seed_cell["tasks"]
        for per_arm in written["cost_to_clear"]["cuda"].values()
        for seed_cell in per_arm.values()
    }
    assert tasks == {55}
