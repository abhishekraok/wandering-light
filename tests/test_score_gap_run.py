"""Fixed driver on fresh dummy files/mock CPU policies only; no real attempt."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import score_gap_provenance as provenance
from experiments import score_gap_run as run
from experiments.score_gap_inputs import hash_file
from experiments.score_gap_storage import (
    DiagnosticProvenanceError,
    PayloadBudget,
    canonical_json,
)
from tests.test_score_gap_inputs import fixture
from wandering_light.stack_policy import OP_NAMES, node_encoder


def ledger():
    return PayloadBudget(deadline=10, clock=lambda: 0)


def review_fixture(tmp_path, source, *, verdict="COMPLETE CODE PASS", exit_code=0):
    root = tmp_path / "synthetic-review"
    root.mkdir()
    report = f"VERDICT: {verdict}\nSynthetic fixture, not a real review.\n"
    report_path = root / "report.md"
    report_path.write_text(report)
    log_path = root / "run.log"
    log_path.write_text(
        json.dumps({"is_error": False, "result": report}) + f"\nexit {exit_code}\n"
    )
    receipt = {
        "schema": "SG-CODE-REVIEW1",
        "issue": 180,
        "verdict": "COMPLETE CODE PASS",
        "source_revision": source["revision"],
        "source_files": source["files"],
        "reviewer_family": "claude",
        "effort": "high",
        "complete_code": True,
        "run_log": {"path": str(log_path), **hash_file(log_path, ledger())},
        "report": {"path": str(report_path), **hash_file(report_path, ledger())},
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt_path, receipt


class MockScorer:
    def __init__(self, *, nan=False):
        self.device = torch.device("cpu")
        self.encoding = "difference"
        self.model = torch.nn.Linear(1, 1).eval()
        self.model.actions = OP_NAMES
        self.nan = nan

    def encoder(self, target):
        return node_encoder(target, "difference")

    def log_probs(self, blocks, legal):
        if not self.nan:
            pytest.fail("manual one-op fits should not reach policy inference")
        return np.full(legal.shape, np.nan, dtype=np.float32)


def driver_fixture(tmp_path, monkeypatch, *, size=60, nonfit=False):
    spec, _, rows = fixture(tmp_path, size=size, compressed=False)
    train = Path(spec["inputs"]["train"]["path"])
    for row in rows:
        row.update(inputs=["2 3"], outputs=["999" if nonfit else "6"])
    train.write_bytes(b"".join(canonical_json(row) for row in rows))
    spec["inputs"]["train"]["sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    repo = train.parent
    spec_path = repo / "spec.json"
    spec_path.write_bytes(canonical_json(spec))
    source = {
        "schema": "SG-SOURCE1",
        "repo": str(repo),
        "revision": "a" * 40,
        "files": {"spec.json": hash_file(spec_path, ledger())},
        "modules": {},
        "functions": {},
    }
    receipt_path, _ = review_fixture(tmp_path, source)
    # Tests never manufacture actual code eligibility: mock the code-only
    # source capture/recheck, then exercise real reservation/input/search/
    # recording/replay/profile/publication on the synthetic paths below.
    monkeypatch.setattr(provenance, "capture_source", lambda *args: source)
    monkeypatch.setattr(provenance, "recheck_source", lambda *args: None)
    monkeypatch.setattr(provenance, "check_runtime", lambda *args: None)
    monkeypatch.setattr(
        provenance,
        "configure_cpu",
        lambda: {"schema": "SG-RUNTIME1", "synthetic_only": True},
    )
    out = tmp_path / "synthetic-attempt"
    return spec_path, repo, out, receipt_path


def test_complete_144_imported_firstfits_before_allseal_replay_and_profile(
    tmp_path, monkeypatch
):
    spec_path, repo, out, review = driver_fixture(tmp_path, monkeypatch)
    loads = []

    def loader(path):
        assert (out / "inputs/input-seal.json").exists()
        assert not (out / "analysis.jsonl").exists()
        loads.append(path.name)
        return MockScorer()

    original = run.load_cell
    replayed = []

    def read(*args):
        assert len(list((out / "cells").glob("*/cell-seal.json"))) == 144
        assert not (out / "analysis.jsonl").exists()
        replayed.append(args[2])
        return original(*args)

    monkeypatch.setattr(run, "load_cell", read)
    assert run.run_diagnostic(spec_path, repo, out, review, loader=loader) == 0
    assert loads == ["dummy-s0", "dummy-s1", "dummy-s2"]
    assert len(replayed) == 144
    result = json.loads((out / "run.json").read_bytes())
    assert result["reading"] == "INSUFFICIENT_LINEAGE_SUPPORT"
    assert result["profile"]["tau"] is None
    assert result["profile"]["confirmation_checks"] == {}
    assert all(ref["result"]["program"] == ["mul"] for ref in result["cells"])
    assert all(ref["result"]["depth"] == 1 for ref in result["cells"])
    assert [ref["identity"]["seed"] for ref in result["cells"]] == [0, 1, 2] * 48
    assert result["limits"] == run.LIMITS and result["search"] == run.SEARCH
    assert result["progress"]["cells_replayed"] == 144
    assert result["independent_verification"] == "REQUIRED_AFTER_RUN_NOT_YET_PERFORMED"
    terminal = json.loads((out / "terminal.json").read_bytes())
    assert terminal["kind"] == "final"
    assert (
        terminal["metadata"]["payload_bytes"]
        == result["ledger"]["bytes_before_manifest"] + (out / "run.json").stat().st_size
    )
    assert terminal["metadata"]["attempts"] == result["ledger"]["attempts"]
    rows = [
        json.loads(line) for line in (out / "analysis.jsonl").read_text().splitlines()
    ]
    assert len([row for row in rows if row["kind"] == "selection_profile"]) == 24
    assert not any(row["kind"] == "confirmation_profile" for row in rows)
    assert len([row for row in rows if row["kind"] == "prefix_curve"]) == 384


def test_insufficient_panel_no_loader_search_or_profile_and_no_retry(
    tmp_path, monkeypatch
):
    spec_path, repo, out, review = driver_fixture(tmp_path, monkeypatch, size=2)

    def loader(*args):
        pytest.fail("insufficient selection loaded a model")

    assert run.run_diagnostic(spec_path, repo, out, review, loader=loader) == 0
    result = json.loads((out / "run.json").read_bytes())
    assert result["reading"] == "INSUFFICIENT_SELECTION"
    assert result["cells"] == [] and result["profile"] is None
    assert not (out / "analysis.jsonl").exists()
    assert not (out / "cells").exists()
    previous = (out / "terminal.json").read_bytes()
    monkeypatch.setattr(
        run, "seal_inputs", lambda *args: pytest.fail("retry read input")
    )
    with pytest.raises(FileExistsError):
        run.run_diagnostic(spec_path, repo, out, review, loader=loader)
    assert (out / "terminal.json").read_bytes() == previous


@pytest.mark.parametrize(
    "failure",
    ["shared_model", "wrong_encoding", "model_load", "attempt_cap", "bytes", "nan"],
)
def test_runwide_failures_are_spent_and_never_profile_denominator_cells(
    tmp_path, monkeypatch, failure
):
    spec_path, repo, out, review = driver_fixture(
        tmp_path, monkeypatch, nonfit=failure == "nan"
    )
    shared = MockScorer()

    def loader(path):
        if failure == "model_load":
            raise RuntimeError("synthetic loader failure")
        if failure == "shared_model":
            return shared
        scorer = MockScorer(nan=failure == "nan")
        if failure == "wrong_encoding":
            scorer.encoding = "machine"
        return scorer

    if failure in ("attempt_cap", "bytes"):

        def constrained(**kwargs):
            return PayloadBudget(
                **kwargs,
                **(
                    {"max_attempts": 0}
                    if failure == "attempt_cap"
                    else {"max_bytes": 1}
                ),
            )

        monkeypatch.setattr(run, "PayloadBudget", constrained)
    assert run.run_diagnostic(spec_path, repo, out, review, loader=loader) == 1
    terminal = json.loads((out / "terminal.json").read_bytes())
    assert terminal["kind"] == "failure"
    assert terminal["metadata"]["complete"] is False
    assert not (out / "run.json").exists()
    assert not (out / "analysis.jsonl").exists()
    assert not (out / "selection-lock.json").exists()
    assert len(list((out / "cells").glob("*/cell-seal.json"))) == 0
    if failure in ("attempt_cap", "bytes"):
        assert terminal["metadata"]["exception"] == "DiagnosticResourceError"
    if failure in ("shared_model", "wrong_encoding", "nan"):
        assert terminal["metadata"]["exception"] == "DiagnosticProvenanceError"


def test_spec_pin_changed_after_review_fails_before_payload_or_model(
    tmp_path, monkeypatch
):
    spec_path, repo, out, review = driver_fixture(tmp_path, monkeypatch)
    spec_path.write_bytes(spec_path.read_bytes() + b"\n")
    monkeypatch.setattr(
        run, "seal_inputs", lambda *args: pytest.fail("changed spec opened payload")
    )
    assert (
        run.run_diagnostic(
            spec_path,
            repo,
            out,
            review,
            loader=lambda *args: pytest.fail("loaded model"),
        )
        == 1
    )
    terminal = json.loads((out / "terminal.json").read_bytes())
    assert terminal["metadata"]["progress"]["models_loaded"] == 0
    assert terminal["metadata"]["exception"] == "DiagnosticProvenanceError"


@pytest.mark.parametrize(
    "mutation", ["design_only", "source", "quoted_pass", "exit", "report_hash"]
)
def test_actual_review_guard_refuses_before_reservation(tmp_path, mutation):
    source = {
        "repo": str(tmp_path / "frozen"),
        "revision": "b" * 40,
        "files": {"fake.py": {"bytes": 1, "sha256": "a" * 64}},
    }
    review, receipt = review_fixture(
        tmp_path,
        source,
        verdict="CHANGES_REQUIRED\nQuoted VERDICT: COMPLETE CODE PASS"
        if mutation == "quoted_pass"
        else "COMPLETE CODE PASS",
        exit_code=1 if mutation == "exit" else 0,
    )
    if mutation == "design_only":
        receipt["complete_code"] = False
    elif mutation == "source":
        receipt["source_revision"] = "c" * 40
    elif mutation == "report_hash":
        receipt["report"]["sha256"] = "0" * 64
    review.write_bytes(canonical_json(receipt))
    with pytest.raises(DiagnosticProvenanceError):
        provenance.require_review(review, source, ledger())


def test_module_origin_rejects_byte_equal_external_path(tmp_path):
    root = tmp_path / "frozen"
    inside = root / "wandering_light/stack_machine.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("# synthetic source bytes\n")
    outside = tmp_path / "same.py"
    outside.write_bytes(inside.read_bytes())
    with pytest.raises(DiagnosticProvenanceError, match="outside intended"):
        provenance.module_origins(
            root,
            {"wandering_light.stack_machine": SimpleNamespace(__file__=str(outside))},
            ledger(),
        )
    origins = provenance.module_origins(
        root,
        {"wandering_light.stack_machine": SimpleNamespace(__file__=str(inside))},
        ledger(),
    )
    assert (
        origins["wandering_light.stack_machine"]["sha256"]
        == hashlib.sha256(inside.read_bytes()).hexdigest()
    )


def test_missing_complete_verifier_is_a_pre_reservation_source_refusal(
    tmp_path, monkeypatch
):
    def git(repo, *args):
        if args[0] == "rev-parse":
            return "a" * 40
        if args[0] == "status":
            return ""
        return "\n".join(
            name
            for name in provenance.REQUIRED_FILES
            if not name.endswith("score_gap_verify.py")
        )

    monkeypatch.setattr(provenance, "_git", git)
    monkeypatch.setattr(
        provenance,
        "module_origins",
        lambda *args: pytest.fail("incomplete source reached hashes"),
    )
    with pytest.raises(DiagnosticProvenanceError, match="complete tracked"):
        provenance.capture_source(tmp_path, ledger())


@pytest.mark.parametrize(
    "mutation", ["none", "memory", "runtime", "cgroup", "inactive"]
)
def test_literal_external_resource_enforcement_contract(mutation):
    unit = "wl-score-gap-feasibility-20261003.service"
    cgroup = "/synthetic.slice/" + unit
    properties = {
        "MemoryMax": str(4 * 1024**3),
        "RuntimeMaxUSec": "1h 30min",
        "ControlGroup": cgroup,
        "ActiveState": "active",
    }
    if mutation == "memory":
        properties["MemoryMax"] = "max"
    elif mutation == "runtime":
        properties["RuntimeMaxUSec"] = "45min"
    elif mutation == "cgroup":
        properties["ControlGroup"] = "/other.service"
    elif mutation == "inactive":
        properties["ActiveState"] = "inactive"
    if mutation == "none":
        assert (
            provenance.validate_job_limits(unit, cgroup, str(4 * 1024**3), properties)[
                "runtime_max_seconds"
            ]
            == 5400
        )
    else:
        with pytest.raises(DiagnosticProvenanceError):
            provenance.validate_job_limits(unit, cgroup, str(4 * 1024**3), properties)


def test_exact_closed_global_schemas_reject_bool_counter_and_extra_fields(
    tmp_path, monkeypatch
):
    spec, repo, out, review = driver_fixture(tmp_path, monkeypatch, size=2)
    assert (
        run.run_diagnostic(
            spec, repo, out, review, loader=lambda *args: pytest.fail("model loaded")
        )
        == 0
    )
    original = json.loads((out / "run.json").read_bytes())
    for mutation in ("counter", "extra", "seeds", "ledger", "input_path"):
        row = json.loads(json.dumps(original))
        if mutation == "counter":
            row["progress"]["models_loaded"] = False
        elif mutation == "extra":
            row["hidden"] = ["99"]
        elif mutation == "seeds":
            row["seeds"][0] = False
        elif mutation == "ledger":
            row["ledger"]["attempts"] = False
        else:
            row["input"]["directory"] = "../inputs"
        with pytest.raises(DiagnosticProvenanceError):
            run.validate_run_shape(row)


def test_resealed_global_controls_cannot_change_analysis_terminal_or_cell_identity(
    tmp_path, monkeypatch
):
    spec, repo, out, review = driver_fixture(tmp_path, monkeypatch)
    assert (
        run.run_diagnostic(spec, repo, out, review, loader=lambda *args: MockScorer())
        == 0
    )
    original = json.loads((out / "run.json").read_bytes())

    def validate(row):
        path = out / "run.json"
        path.write_bytes(canonical_json(row))
        pin = hash_file(path, ledger())
        saved_budget = ledger()
        saved_budget.bytes_written = (
            row["ledger"]["bytes_before_manifest"] + pin["bytes"]
        )
        saved_budget.attempts_written = row["ledger"]["attempts"]
        run.validate_publication(out, row, pin, saved_budget)

    validate(original)
    for mutation in ("analysis_terminal", "identity_bool", "refusal_sum"):
        row = json.loads(json.dumps(original))
        if mutation == "analysis_terminal":
            row["reading"] = row["profile"]["terminal"] = "PASS_PROFILE_FEASIBILITY"
        elif mutation == "identity_bool":
            row["cells"][0]["identity"]["seed"] = False
        else:
            row["cells"][0]["refusal_counts"]["ordinary"] += 1
        with pytest.raises(DiagnosticProvenanceError):
            validate(row)


def test_default_loader_is_uncalibrated_imported_cpu_policy_wrapper(monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.actions = OP_NAMES
    model.encoding = "difference"
    calls = []

    def load(checkpoint, *, device):
        calls.append((checkpoint, device))
        return model

    monkeypatch.setattr(run, "load_policy", load)
    scorer = run.default_loader(Path("synthetic-not-read"))
    assert calls == [(Path("synthetic-not-read"), "cpu")]
    assert type(scorer) is run.PolicyScorer
    assert scorer.encoding == "difference" and str(scorer.device) == "cpu"
    assert scorer.model.training is False
