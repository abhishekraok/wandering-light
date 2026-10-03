"""Runner safeguards tested without solver/gate execution or model/data calls."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.score_band_gate import atomic_json, catalog_ids, json_value, run_gate
from wandering_light.stack_machine import Bool, Int


def test_catalog_preserves_fixed_unique_ids_and_adverse_cases():
    ids = catalog_ids()
    assert len(ids) == len(set(ids)) == 455
    assert sum(x.startswith("disabled/") for x in ids) == 324
    assert {"A-safe", "B-only-solution-cut", "C-later-first-fit"}.issubset(ids)
    assert "unit/last-row-band" in ids and "stop/last-ranked-width/None" in ids
    assert sum(x.startswith("observation/") for x in ids) == 29


def test_strict_json_retains_nonfinite_scope_and_typed_identity(tmp_path):
    path = tmp_path / "final.json"
    atomic_json(
        path, [Int(1), Bool(True), np.nan, np.inf, -np.inf, np.bool_(True)], 1000
    )
    value = json.loads(path.read_text())
    assert value == [
        {"type": "Int", "value": 1},
        {"type": "Bool", "value": True},
        "NaN",
        "+inf",
        "-inf",
        True,
    ]
    assert not path.with_suffix(".json.tmp").exists()


def test_atomic_output_limit_has_no_final_or_temporary(tmp_path):
    path = tmp_path / "final.json"
    with pytest.raises(ValueError, match="byte bound"):
        atomic_json(path, {"large": "x" * 1000}, 100)
    assert not path.exists() and not path.with_suffix(".json.tmp").exists()


def test_missing_code_receipt_prevents_attempt_reservation(tmp_path):
    out = tmp_path / "gate"
    with pytest.raises(FileNotFoundError):
        run_gate(out, tmp_path / "absent.json")
    assert not out.exists()


def test_numpy_scalar_conversion_is_not_stringified():
    assert json_value((np.float32(1.5), np.int64(2))) == [1.5, 2]


def test_expired_deadline_prevents_final_publication(tmp_path):
    path = tmp_path / "final.json"
    with pytest.raises(TimeoutError, match="before atomic"):
        atomic_json(path, {"complete": True}, 1000, deadline=0)
    assert not path.exists()
