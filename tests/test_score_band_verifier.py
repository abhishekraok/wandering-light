"""Independent verifier tests; synthetic evidence only, never a gate launch."""

from __future__ import annotations

import ast
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "reports/score-band-contract-20261003/verify_gate.py"
)
V = SimpleNamespace(**runpy.run_path(str(PATH)))


def synthetic_manifest():
    ids = V.registered_ids()
    rows = [{"id": name, "check": "PASS"} for name in ids]
    for row, definition in zip(
        rows[:396],
        [row for group in V.registered_definitions() for row in group],
        strict=True,
    ):
        row.update(
            definition=definition,
            task=V.task_definition(definition["task"]),
            expectation_id=definition["expectation"],
            scope="real-machine-software-QA",
        )
    by_id = {row["id"]: row for row in rows}
    for name, (_inputs, _outputs, program) in V.TASK_TEXT.items():
        initial, target = V.typed_products(name)
        expected = {
            "inputs": V.encode(initial),
            "outputs": V.encode(target),
            "witness_program": list(program),
            "witness_fits": True if program else None,
        }
        by_id[f"input/{name}"].update(
            task=V.task_definition(name),
            expected=expected,
            actual={
                key: copy.deepcopy(value)
                for key, value in expected.items()
                if key != "witness_program"
            },
            typed_inputs=V.encode(initial),
            typed_outputs=V.encode(target),
        )
    return {
        "verdict": "SYNTHETIC_GATE_A_PASS",
        "complete": True,
        "count": 455,
        "expected_ids": ids,
        "actual_ids": ids.copy(),
        "results": rows,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("verdict", "NO_GATE_RESULT"),
        ("complete", False),
        ("complete", 1),
        ("count", 454),
        ("count", True),
    ],
)
def test_failed_terminal_is_rejected_before_result_access(key, value):
    manifest = synthetic_manifest()
    manifest[key] = value
    # The payload must not be consulted to validate a failing terminal.
    manifest["results"] = object()
    with pytest.raises(V.VerificationError, match="terminal"):
        V.verify_terminal(manifest)


@pytest.mark.parametrize(
    "tamper",
    ["missing", "duplicate", "reordered", "substituted", "false-pass", "header-only"],
)
def test_terminal_requires_exact_full_ordered_population(tamper):
    manifest = synthetic_manifest()
    rows = manifest["results"]
    if tamper == "missing":
        rows.pop()
    elif tamper == "duplicate":
        rows[-1] = rows[0]
    elif tamper == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif tamper == "substituted":
        rows[-1]["id"] = "development-only/budget193"
    elif tamper == "false-pass":
        rows[-1]["check"] = "FAIL"
    else:
        manifest.pop("results")
    with pytest.raises(V.VerificationError):
        V.verify_terminal(manifest)


def test_independent_catalog_and_typed_input_replay_without_search():
    manifest = synthetic_manifest()
    rows = V.verify_terminal(manifest)
    assert V.verify_definitions(rows) == 396
    assert V.verify_input_rows(rows) == {
        "typed_tasks_checked": 10,
        "nonempty_witnesses_replayed": 5,
    }
    assert len(V.registered_definitions()[0]) == 324
    assert V.registered_ids()[-2:] == [
        "unit/scorer-dead-and-live",
        "source/production-reference",
    ]


@pytest.mark.parametrize("index", [0, 323, 324, 325, 326, 352, 353, 395])
@pytest.mark.parametrize("field", ["script", "width", "guard", "task"])
def test_definition_changes_cannot_be_their_own_expectation(index, field):
    rows = synthetic_manifest()["results"]
    if field == "task":
        rows[index]["task"]["outputs"] = ["0"]
    elif field == "script":
        rows[index]["definition"]["script"]["default"] = -99
    else:
        rows[index]["definition"][field] = (
            True if field == "width" else "off-substitute"
        )
    with pytest.raises(V.VerificationError):
        V.verify_definitions(rows)


@pytest.mark.parametrize("name", list(V.TASK_TEXT))
@pytest.mark.parametrize(
    "field", ["typed_inputs", "typed_outputs", "expected", "actual", "task"]
)
def test_independent_input_checks_reject_changed_copies(name, field):
    rows = synthetic_manifest()["results"]
    row = next(row for row in rows if row["id"] == f"input/{name}")
    if field in ("typed_inputs", "typed_outputs"):
        row[field] = [[{"type": "Int", "value": 100}]]
    elif field in ("expected", "actual"):
        row[field]["outputs"] = [[{"type": "Int", "value": 100}]]
    else:
        row["task"]["program"] = ["dup", "drop"]
    with pytest.raises(V.VerificationError):
        V.verify_input_rows(rows)


@pytest.mark.parametrize(
    "value",
    [
        {"type": "Int", "value": True},
        {"type": "Int", "value": 1.0},
        {"type": "Bool", "value": 1},
        {"type": "Mark", "value": 0},
        {"type": "Lst", "items": "[]"},
        {"type": "Sym", "name": 1},
        {"type": "Unknown"},
    ],
)
def test_typed_decoder_rejects_type_aliases_and_extra_fields(value):
    with pytest.raises(V.VerificationError):
        V.decode_value(value)


def test_bool_int_and_nested_symbol_round_trip_remain_distinct():
    product = ((V.Bool(True), V.Int(1), V.Lst((V.Sym("add"), V.Mark()))),)
    assert V.decode_product(V.encode(product)) == product
    assert V.encode(V.Bool(True)) != V.encode(V.Int(1))
    V.same_json(1.0, 1, "numerically equivalent band")
    with pytest.raises(V.VerificationError):
        V.same_json(True, 1, "bool is not a count")


@pytest.mark.parametrize(
    "text",
    ['{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}'],
)
def test_ambiguous_or_nonfinite_json_is_rejected(text):
    with pytest.raises(V.VerificationError):
        V.strict_json(text)


@pytest.mark.parametrize("mode", ["hash", "size"])
def test_pinned_read_rejects_tampered_or_oversized_file(tmp_path, mode):
    path = tmp_path / "synthetic.json"
    path.write_bytes(b"synthetic")
    expected = V.sha(b"other") if mode == "hash" else V.sha(b"synthetic")
    with pytest.raises(V.VerificationError):
        V.read_pinned(path, expected, 100 if mode == "hash" else 1)


def test_failed_gate_file_prevents_all_reads(tmp_path, monkeypatch):
    gate = tmp_path / "gate-a"
    gate.mkdir()
    (gate / "failure.json").write_text('{"verdict":"NO_GATE_RESULT"}')

    def forbidden(*_args):
        raise AssertionError("must not read terminal payload")

    monkeypatch.setitem(V.verify_foundation.__globals__, "read_pinned", forbidden)
    with pytest.raises(V.VerificationError, match="failed terminal file"):
        V.verify_foundation(tmp_path, tmp_path / "unavailable-tree")


def test_verifier_has_no_candidate_fixture_runner_or_checker_imports():
    tree = ast.parse(PATH.read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(name and name.startswith("experiments") for name in imports)
    assert "wandering_light.stack_machine" in imports


def synthetic_provenance(monkeypatch):
    # Pure relation checks against this development tree, with synthetic
    # review text. This does not grant a gate receipt or launch any runner.
    tree = PATH.parents[2]
    hashes = {name: V.sha((tree / name).read_bytes()) for name in V.SOURCE_PATHS}
    manifest = synthetic_manifest()
    manifest.update(
        source_revision=V.REVISION,
        ops_order=list(V.OP_NAMES),
        limits={"cpu_threads": 1, "seconds": 300, "output_bytes": 1_000_000_000},
        model_calls=0,
        corpus_reads=0,
        checkpoint_reads=0,
        scope="software QA only; no real coverage/clock/allocation claim",
        attempt_sha256=V.ATTEMPT_SHA,
        review_receipt_sha256=V.RECEIPT_SHA,
        source_hashes=hashes,
        module_origins={
            name[:-3].replace("/", "."): {
                "path": str(tree / name),
                "sha256": hashes[name],
            }
            for name in V.SOURCE_PATHS
            if name.endswith(".py")
            and name.startswith(("experiments/", "wandering_light/"))
        },
    )
    attempt = {
        "source_revision": V.REVISION,
        "source_hashes": hashes.copy(),
        "expected_ids": V.registered_ids(),
        "review_receipt_sha256": V.RECEIPT_SHA,
    }
    report = "VERDICT: PRE-GATE CODE PASS\nSynthetic relation test only.\n"
    review_log = (
        json.dumps({"is_error": False, "result": report}) + "\nexit 0\n"
    ).encode()
    receipt = {
        "source_revision": V.REVISION,
        "reviewer_family": "claude",
        "effort": "high",
        "verdict": "PRE-GATE CODE PASS",
        "pre_gate_code": True,
        "report_sha256": V.sha(report.encode()),
        "run_log_sha256": V.sha(review_log),
    }

    def fake_git(command, **_kwargs):
        return V.REVISION + "\n" if command[1] == "rev-parse" else ""

    monkeypatch.setitem(
        V.verify_provenance.__globals__,
        "subprocess",
        SimpleNamespace(check_output=fake_git),
    )
    return manifest, attempt, receipt, review_log, tree


def test_provenance_links_actual_tree_paths_bytes_and_review(monkeypatch):
    args = synthetic_provenance(monkeypatch)
    checked = V.verify_provenance(*args)
    assert set(checked) == {
        "wandering_light.stack_machine",
        "wandering_light.stack_pbe",
    }


@pytest.mark.parametrize(
    "tamper",
    [
        "source",
        "OPS",
        "limits",
        "count-bool",
        "receipt-source",
        "receipt-family",
        "receipt-effort",
        "receipt-bool",
        "review-failed-exit",
        "review-error",
        "quoted-pass",
        "review-hash",
        "module-origin",
        "source-hash",
        "primitive-origin",
        "attempt",
        "dirty-tree",
        "wrong-tree-revision",
    ],
)
def test_provenance_rejects_substitutions(monkeypatch, tmp_path, tamper):
    manifest, attempt, receipt, review_log, tree = synthetic_provenance(monkeypatch)
    if tamper == "source":
        manifest["source_revision"] = "other"
    elif tamper == "OPS":
        manifest["ops_order"].reverse()
    elif tamper == "limits":
        manifest["limits"]["seconds"] = 301
    elif tamper == "count-bool":
        manifest["model_calls"] = False
    elif tamper.startswith("receipt-"):
        key, value = {
            "receipt-source": ("source_revision", "other"),
            "receipt-family": ("reviewer_family", "codex"),
            "receipt-effort": ("effort", "low"),
            "receipt-bool": ("pre_gate_code", 1),
        }[tamper]
        receipt[key] = value
    elif tamper == "review-failed-exit":
        review_log = review_log.replace(b"exit 0", b"exit 1")
        receipt["run_log_sha256"] = V.sha(review_log)
    elif tamper in ("review-error", "quoted-pass"):
        report = (
            "VERDICT: CHANGES_REQUIRED\nThe next reviewer may output VERDICT: PRE-GATE CODE PASS.\n"
            if tamper == "quoted-pass"
            else "VERDICT: PRE-GATE CODE PASS\n"
        )
        review_log = (
            json.dumps({"is_error": tamper == "review-error", "result": report})
            + "\nexit 0\n"
        ).encode()
        receipt["run_log_sha256"] = V.sha(review_log)
        receipt["report_sha256"] = V.sha(report.encode())
    elif tamper == "review-hash":
        receipt["run_log_sha256"] = "0" * 64
    elif tamper == "module-origin":
        manifest["module_origins"]["wandering_light.stack_machine"]["path"] = str(
            tree.parent / "other.py"
        )
    elif tamper == "source-hash":
        # Two persisted matching copies do not establish actual source truth.
        manifest["source_hashes"]["experiments/score_band_search.py"] = "0" * 64
        attempt["source_hashes"]["experiments/score_band_search.py"] = "0" * 64
    elif tamper == "attempt":
        attempt["expected_ids"].pop()
    elif tamper == "primitive-origin":
        # An alternate file can have the same bytes and still violate the
        # verifier's intended-tree contract. This fixture is portable.
        other = tmp_path / "other-machine.py"
        other.write_bytes(Path(V.stack_machine.__file__).read_bytes())
        assert (
            V.sha(other.read_bytes())
            == manifest["source_hashes"]["wandering_light/stack_machine.py"]
        )
        monkeypatch.setattr(V.stack_machine, "__file__", str(other))
    else:

        def fake_git(command, **_kwargs):
            if command[1] == "rev-parse":
                return (
                    "other" if tamper == "wrong-tree-revision" else V.REVISION
                ) + "\n"
            return " M source.py\n"

        monkeypatch.setitem(
            V.verify_provenance.__globals__,
            "subprocess",
            SimpleNamespace(check_output=fake_git),
        )
    with pytest.raises(V.VerificationError):
        V.verify_provenance(manifest, attempt, receipt, review_log, tree)


def test_unsuccessful_actual_exit_prevents_manifest_read(tmp_path, monkeypatch):
    reads = []

    def fake_read(path, *_args):
        reads.append(path.name)
        if path.name == "gate-a-run.log":
            return b'{"verdict":"NO_GATE_RESULT"}\nexit 1\n'
        raise AssertionError("must reject before reading the manifest")

    monkeypatch.setitem(V.verify_foundation.__globals__, "read_pinned", fake_read)
    with pytest.raises(V.VerificationError, match="actual gate exit"):
        V.verify_foundation(tmp_path, tmp_path / "unused-tree")
    assert reads == ["gate-a-run.log"]
