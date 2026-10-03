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


@pytest.mark.parametrize(
    "case_id",
    [
        "A-safe",
        "B-only-solution-cut",
        "C-later-first-fit",
        "score/executable-nan/late",
        "score/all-inf",
    ],
)
def test_runner_records_fixed_case_and_legal_mask_bytes(case_id):
    from experiments.score_band_fixtures import semantic_cases
    from experiments.score_band_gate import run_search_case

    case = next(c for c in semantic_cases() if c.id == case_id)
    result, observation = run_search_case(case, "semantic")
    payload = json.dumps(json_value((result, observation)), allow_nan=False)
    assert result["check"] == observation["check"] == "PASS"
    assert result["id"] == case_id and observation["id"] == f"observation/{case_id}"
    assert len(payload) < 1_000_000


def test_mask_digest_is_structurally_distinct_from_plain_string():
    from experiments.score_band_gate import digest

    assert digest(b"\x00\x01") != digest("0001")
    assert json_value(b"\x00\x01") == {"type": "bytes", "hex": "0001"}
