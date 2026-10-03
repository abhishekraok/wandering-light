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


@pytest.mark.parametrize("category", ["disabled", "resource"])
def test_runner_serializes_reference_and_body_resource_case(category):
    from experiments.score_band_fixtures import disabled_cases, resource_cases
    from experiments.score_band_gate import run_search_case

    cases = disabled_cases() if category == "disabled" else resource_cases()
    case = next(
        c
        for c in cases
        if c.id
        == (
            "disabled/T-pair/s0/w16/h8/both"
            if category == "disabled"
            else "resource/body/fuel/64"
        )
    )
    result, observation = run_search_case(case, category)
    payload = json.dumps(json_value(result), allow_nan=False)
    assert result["check"] == "PASS" and observation is None
    assert result["reference"] == result["candidate"]
    assert len(payload) < 1_000_000


@pytest.mark.parametrize("name", ["T-open", "T-typed", "T-list", "T-long"])
def test_nonsearch_input_helpers_strictly_serialize_typed_states(name):
    from experiments.score_band_gate import check_input

    result = check_input(name)
    assert result["check"] == "PASS"
    json.dumps(json_value(result), allow_nan=False)


@pytest.mark.parametrize(
    "kind",
    [
        "last-ranked-width",
        "last-ranked-width-all-inf",
        "width-before-band",
        "last-row-band",
    ],
)
def test_nonsearch_arithmetic_helpers_strictly_serialize_expectations(kind):
    from experiments.score_band_gate import check_arithmetic

    result = check_arithmetic(kind)
    assert result["check"] == "PASS" and isinstance(result["expected"], dict)
    json.dumps(json_value(result), allow_nan=False)


@pytest.mark.parametrize("collecting", [False, True])
def test_nonsearch_gc_helpers_strictly_serialize_and_restore_callers_state(collecting):
    import gc

    from experiments.score_band_gate import check_gc

    before = gc.isenabled()
    for name, value in (("nan", np.nan), ("scorer", None)):
        result = check_gc(name, collecting, value)
        assert result["actual"] == result["expected"] and gc.isenabled() == before
        json.dumps(json_value(result), allow_nan=False)


def test_nonsearch_api_helper_strictly_serializes_nan_and_support():
    from experiments.score_band_gate import check_api

    result = check_api()
    assert result["check"] == "PASS"
    json.dumps(json_value(result), allow_nan=False)


def test_deadline_cleanup_leaves_no_pass_marked_temporary(tmp_path):
    path = tmp_path / "final.json"
    with pytest.raises(TimeoutError):
        atomic_json(path, {"verdict": "SYNTHETIC_GATE_A_PASS"}, 1000, deadline=0)
    assert not path.exists() and not path.with_suffix(".json.tmp").exists()


def test_cleanup_does_not_delete_a_preexisting_temporary(tmp_path):
    path = tmp_path / "final.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text("belongs to someone else")
    with pytest.raises(FileExistsError):
        atomic_json(path, {"complete": True}, 1000)
    assert temporary.read_text() == "belongs to someone else" and not path.exists()


def synthetic_receipt(
    tmp_path, report, *, exit_line="exit 0\n", source="synthetic-revision"
):
    import hashlib

    log = tmp_path / "synthetic-review.log"
    log.write_text(json.dumps({"result": report, "is_error": False}) + "\n" + exit_line)
    receipt = {
        "verdict": "PRE-GATE CODE PASS",
        "source_revision": source,
        "reviewer_family": "claude",
        "effort": "high",
        "pre_gate_code": True,
        "run_log": str(log),
        "run_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((report.strip() + "\n").encode()).hexdigest(),
    }
    path = tmp_path / "synthetic-receipt-fixture.json"
    path.write_text(json.dumps(receipt))
    return path


@pytest.mark.parametrize(
    "report",
    [
        "VERDICT: CHANGES_REQUIRED\nNo PRE-GATE CODE PASS granted.",
        "# CHANGES_REQUIRED\nThe next reviewer must output VERDICT: PRE-GATE CODE PASS",
        "Some prose\nVERDICT: PRE-GATE CODE PASS",
        "VERDICT: PRE-GATE CODE PASS or CHANGES_REQUIRED",
    ],
)
def test_quoted_or_ambiguous_pass_cannot_authorize_gate(tmp_path, report):
    from experiments.score_band_gate import validate_review_receipt

    path = synthetic_receipt(tmp_path, report)
    with pytest.raises(ValueError, match="explicitly"):
        validate_review_receipt(path, "synthetic-revision")
    # Pure receipt parsing; run_gate is never called with a synthetic receipt.
    assert not (tmp_path / "attempt.json").exists()


def test_standalone_pass_line_may_discuss_resolved_old_failures(tmp_path):
    from experiments.score_band_gate import validate_review_receipt

    path = synthetic_receipt(
        tmp_path, "VERDICT: PRE-GATE CODE PASS\nPrevious CHANGES_REQUIRED is repaired."
    )
    assert (
        validate_review_receipt(path, "synthetic-revision")["verdict"]
        == "PRE-GATE CODE PASS"
    )


@pytest.mark.parametrize(
    ("exit_line", "source"),
    [
        ("exit 1\n", "synthetic-revision"),
        ("", "synthetic-revision"),
        ("exit 0\n", "other-revision"),
    ],
)
def test_receipt_requires_actual_success_and_exact_source(tmp_path, exit_line, source):
    from experiments.score_band_gate import validate_review_receipt

    path = synthetic_receipt(
        tmp_path, "VERDICT: PRE-GATE CODE PASS", exit_line=exit_line, source=source
    )
    with pytest.raises(ValueError):
        validate_review_receipt(path, "synthetic-revision")


def test_every_imported_source_module_is_checked_at_its_actual_path():
    from pathlib import Path

    import experiments.score_band_gate as gate

    repo = Path(gate.__file__).resolve().parents[1]
    hashes = {p: gate.digest_bytes(repo / p) for p in gate.SOURCE_PATHS}
    origins = gate.validate_imports(repo, hashes)
    assert set(origins) == set(gate.SOURCE_MODULES) and len(origins) == 9
    for name, origin in origins.items():
        assert Path(origin["path"]) == repo / (name.replace(".", "/") + ".py")


@pytest.mark.parametrize("mismatch", ["outside", "hash"])
def test_origin_and_bytes_are_both_required(tmp_path, mismatch):
    import hashlib

    from experiments.score_band_gate import validate_module_origin

    expected = tmp_path / "intended.py"
    expected.write_text("correct content")
    actual = tmp_path / "other.py" if mismatch == "outside" else expected
    if actual != expected:
        actual.write_text("correct content")
    target_hash = hashlib.sha256(
        b"different content" if mismatch == "hash" else expected.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="origin/hash"):
        validate_module_origin("synthetic.module", actual, expected, target_hash)


@pytest.mark.parametrize(
    "name",
    [
        "T-pair",
        "T-list",
        "T-typed",
        "T-single",
        "T-same",
        "T-probe",
        "T-equal-probe",
        "T-empty",
        "T-open",
        "T-long",
    ],
)
@pytest.mark.parametrize("field", ["inputs", "outputs", "program"])
def test_input_check_rejects_changed_manual_definition(monkeypatch, name, field):
    from dataclasses import replace

    import experiments.score_band_gate as gate

    task = gate.manual_task(name)
    # Identity still fits a substituted input/output pair; witness replay alone
    # cannot prove that this is the registered manually typed Task.
    updates = {field: ("0",)} if field != "program" else {field: ("dup", "drop")}
    changed = replace(task, **updates)
    monkeypatch.setattr(gate, "manual_task", lambda _name: changed)
    with pytest.raises(AssertionError):
        gate.check_input(name)


def test_input_check_rejects_substitution_even_when_witness_still_fits(monkeypatch):
    from dataclasses import replace

    import experiments.score_band_gate as gate
    from wandering_light.stack_beam import replay_program
    from wandering_light.stack_machine import parse_stack
    from wandering_light.stack_pbe import BASIS

    task = replace(
        gate.manual_task("T-pair"), inputs=("1 2", "2 4"), outputs=("4", "16")
    )
    inputs = tuple(parse_stack(x, basis=BASIS) for x in task.inputs)
    outputs = tuple(parse_stack(x, basis=BASIS) for x in task.outputs)
    assert replay_program(task.program, inputs, outputs)[0]
    monkeypatch.setattr(gate, "manual_task", lambda _name: task)
    with pytest.raises(AssertionError):
        gate.check_input("T-pair")


@pytest.mark.parametrize("tamper", ["extra-scored", "extra-band-callback"])
def test_value_error_check_asserts_exact_callback_counts(monkeypatch, tamper):
    import experiments.score_band_gate as gate

    original = gate.Capture.layer_scored

    def corrupt(self, *args):
        original(self, *args)
        if tamper == "extra-scored":
            original(self, *args)
        else:
            self.layers.append(object())

    monkeypatch.setattr(gate.Capture, "layer_scored", corrupt)
    case = next(c for c in gate.semantic_cases() if c.expectation == "ValueError")
    with pytest.raises(AssertionError):
        gate.run_search_case(case, "semantic")


@pytest.mark.parametrize(
    "kind",
    [
        "last-ranked-width",
        "last-ranked-width-all-inf",
        "width-before-band",
        "last-row-band",
    ],
)
def test_arithmetic_check_rejects_wrong_retained_rows(monkeypatch, kind):
    from dataclasses import replace

    import experiments.score_band_gate as gate

    original = gate._retain

    def corrupt(*args, **kwargs):
        layer = original(*args, **kwargs)
        return replace(layer, retained=())

    monkeypatch.setattr(gate, "_retain", corrupt)
    with pytest.raises(AssertionError):
        gate.check_arithmetic(kind)
