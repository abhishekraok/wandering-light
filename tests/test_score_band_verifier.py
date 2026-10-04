"""Independent verifier tests; synthetic evidence only, never a gate launch."""

from __future__ import annotations

import ast
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
        "wandering_light.stack_beam",
        "wandering_light.stack_policy",
        "wandering_light.quotation_guard",
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


def find_case(name):
    return next(
        row
        for group in V.registered_definitions()
        for row in group
        if row["id"] == name
    )


@pytest.mark.parametrize(
    ("name", "program", "termination", "expansions", "executions", "fuel"),
    [
        ("A-safe", ("mul", "dup", "add"), "fit", 3, 36, 36),
        ("B-only-solution-cut", None, "max_depth", 3, 43, 43),
        ("C-later-first-fit", ("mul", "dup", "add"), "fit", 3, 36, 36),
        ("fit/before-size", ("add",), "fit", 1, 10, 10),
        ("resource/past-deadline", None, "wall_clock_cap", 1, 0, 0),
    ],
)
def test_oracle_manual_outcomes_and_actual_expansion_charges(
    name, program, termination, expansions, executions, fuel
):
    replay = V.oracle_search(find_case(name))
    result = replay["result"]
    assert (
        result["program"],
        result["termination"],
        result["expansions"],
        result["executions"],
        result["fuel"],
    ) == (program, termination, expansions, executions, fuel)
    assert replay["events"][-1][0] == "terminated"


@pytest.mark.parametrize(
    "name",
    [
        "A-safe",
        "B-only-solution-cut",
        "C-later-first-fit",
        "stop/last-ranked-width/None",
        "resource/body/fuel/64",
    ],
)
def test_independent_disabled_replay_matches_imported_production(name):
    case = dict(find_case(name), band=None)
    oracle = V.oracle_search(case)
    production = V.imported_reference(case)
    assert V.encode(oracle["result"]) == V.encode(production["result"])
    assert V.encode(oracle["events"]) == V.encode(production["events"])


@pytest.mark.parametrize("budget", [193, 194, 339, 340, 395, 396])
def test_oracle_parent_entry_is_charged_even_without_an_op0_attempt(budget):
    case = dict(find_case("resource/body/executions/None"), budget=budget)
    oracle = V.oracle_search(case)
    reference = V.imported_reference(case)
    assert V.encode(oracle["result"]) == V.encode(reference["result"])
    assert V.encode(oracle["events"]) == V.encode(reference["events"])
    assert oracle["result"]["termination"] == "budget"


@pytest.mark.parametrize(
    "name",
    [
        "score/executable-nan/late",
        "score/executable-posinf/size",
        "score/executable-nan/seen",
    ],
)
def test_oracle_invalid_executable_scores_stop_before_retention(name):
    replay = V.oracle_search(find_case(name))
    assert replay["exception"] == "ValueError" and replay["result"] is None
    assert not replay["layers"]
    assert replay["events"][-1][0] == "scored"


def synthetic_search_row(name):
    case = find_case(name)
    oracle = V.oracle_search(case)
    reference = V.oracle_search(dict(case, band=None))
    layer_fields = ("ranked_count", "ranked_sha256", "old_seen_sha256")
    expected_layers = [
        {key: value for key, value in layer.items() if key not in layer_fields}
        for layer in oracle["layers"]
    ]
    row = V.encode(
        {
            "id": name,
            "candidate_trace": V.summary_of_replay(oracle),
            "reference_trace": V.summary_of_replay(reference, reference=True),
            "candidate": oracle["result"],
            "reference": reference["result"],
            "band_pruned": oracle["band_pruned"],
            "width_pruned": oracle["result"]["pruned"],
            "scorer_calls": oracle["scorer_calls"],
            "expected": {
                "reference": reference["result"],
                "candidate": oracle["result"],
                "candidate_layers": expected_layers,
                "band_pruned": oracle["band_pruned"],
                "reference_trace_sha256": V.trace_digest(reference["events"]),
            },
        }
    )
    return row, case


@pytest.mark.parametrize(
    "tamper",
    [
        "event-hash",
        "attempt-count",
        "final-layer",
        "layer-order",
        "ranked-hash",
        "old-seen-hash",
        "parent",
        "retained",
        "anchor",
        "stop",
        "unread-verdict",
        "scorer-calls",
        "charged-expansion",
        "expected-and-actual",
        "reference-event",
    ],
)
def test_search_result_verifier_rejects_corrupt_saved_outputs(tamper):
    row, case = synthetic_search_row("A-safe")
    trace = row["candidate_trace"]
    if tamper == "event-hash":
        trace["event_sha256"] = "0" * 64
    elif tamper == "attempt-count":
        trace["attempts"] -= 1
    elif tamper == "final-layer":
        trace["layers"].pop()
    elif tamper == "layer-order":
        trace["layers"].reverse()
    elif tamper in ("ranked-hash", "old-seen-hash"):
        trace["layers"][-1][
            "ranked_sha256" if tamper == "ranked-hash" else "old_seen_sha256"
        ] = "0" * 64
    elif tamper == "parent":
        trace["layers"][-1]["parents"] = [[99, 99]]
    elif tamper == "retained":
        trace["layers"][-1]["retained"] = []
    elif tamper == "anchor":
        trace["layers"][-1]["anchor"] = 1
    elif tamper == "stop":
        trace["layers"][-1]["stop"] = "exhausted"
    elif tamper == "unread-verdict":
        trace["layers"][-1]["verdicts_read_only"]["(0, 27)"] = "duplicate"
    elif tamper == "scorer-calls":
        row["scorer_calls"] += 1
    elif tamper == "charged-expansion":
        row["candidate"]["expansions"] -= 1
    elif tamper == "expected-and-actual":
        row["candidate"]["executions"] += 1
        row["expected"]["candidate"]["executions"] += 1
    else:
        row["reference_trace"]["event_sha256"] = "0" * 64
    with pytest.raises(V.VerificationError):
        V.verify_search_row(row, case, "semantic")


def test_legal_mask_and_score_digests_are_not_counts(monkeypatch):
    row, case = synthetic_search_row("A-safe")
    original = V.oracle_search

    def corrupt(definition):
        replay = original(definition)
        at = next(i for i, event in enumerate(replay["events"]) if event[0] == "scored")
        event = replay["events"][at]
        mask = bytes([1 - event[3][0]]) + event[3][1:]
        replay["events"][at] = (*event[:3], mask, event[4])
        return replay

    monkeypatch.setitem(V.verify_search_row.__globals__, "oracle_search", corrupt)
    with pytest.raises(V.VerificationError, match="candidate trace"):
        V.verify_search_row(row, case, "semantic")


def test_typed_seen_reentry_and_unexpanded_final_retention():
    reentry = V.oracle_search(find_case("seen/reentry"))
    assert reentry["layers"][1]["retained"][0][1] == ("swap", "mul")
    final = V.oracle_search(find_case("boundary/included"))
    assert len(final["layers"]) == 1
    assert not any(event[0] == "started" and event[1] == 1 for event in final["events"])
    assert final["layers"][0]["parents"] == (
        (0, V.OP_NAMES.index("mul")),
        (0, V.OP_NAMES.index("drop")),
    )


def test_numpy_support_values_and_mask_digest_have_explicit_types():
    assert V.encode(
        [np.float64(-np.inf), np.float64(np.nan), np.int64(1), np.bool_(True)]
    ) == ["-inf", "NaN", 1, True]
    assert V.trace_digest(b"\x00\x01") != V.trace_digest("0001")


def synthetic_controls():
    rows = [{"id": name, "check": "PASS"} for name in V.registered_ids()]
    by_id = {row["id"]: row for row in rows}
    replayed = {}
    for case in V.registered_definitions()[1]:
        replay = V.oracle_search(case)
        replayed[case["id"]] = replay, None
        value = (
            {"exception": "ValueError"}
            if replay["exception"]
            else {"result": replay["result"], "band_pruned": replay["band_pruned"]}
        )
        row = by_id[f"observation/{case['id']}"]
        row.update(expected=value, actual=copy.deepcopy(value))
        if not replay["exception"]:
            row["band_pruned"] = replay["band_pruned"]
    for parameter, tau in (
        ("negative", -1),
        ("nan", "NaN"),
        ("posinf", "+inf"),
        ("neginf", "-inf"),
        ("bool", True),
        ("string", "1"),
    ):
        for enabled in (False, True):
            value = {"exception": "ValueError", "gc_enabled": enabled}
            by_id[f"parameter/{parameter}/gc{int(enabled)}"].update(
                expected=value, actual=copy.deepcopy(value), collecting=enabled, tau=tau
            )
    for enabled in (False, True):
        value = {"exception": "RuntimeError", "gc_enabled": enabled}
        by_id[f"exception/scorer/gc{int(enabled)}"].update(
            expected=value, actual=copy.deepcopy(value), collecting=enabled, tau=None
        )
    root = ((V.Int(9),),)
    products = [((V.Int(0),),), ((V.Int(1),),)]
    for kind, width, scores in (
        ("last-ranked-width", 2, [0.0, 0.0]),
        ("last-ranked-width-all-inf", 2, [-np.inf, -np.inf]),
        ("width-before-band", 1, [0.0, 0.0]),
        ("last-row-band", 4, [0.0, -2.0]),
    ):
        ranked = [
            (0, i, product, (V.OP_NAMES[i],), scores[i])
            for i, product in enumerate(products)
        ]
        retained, _seen, layer = V.retention_from_definition(
            ranked,
            {V.product_key(root): root},
            V.definition("unit", width=width, band=1),
            0,
        )
        actual = {
            key: value
            for key, value in layer.items()
            if key
            not in (
                "verdicts_read_only",
                "ranked_count",
                "ranked_sha256",
                "old_seen_sha256",
            )
        }
        actual.update(
            type="BandLayer",
            verdicts=layer["verdicts_read_only"],
            ranked=tuple(ranked),
            old_seen=frozenset((root,)),
        )
        by_id[f"unit/{kind}"].update(
            scope="arithmetic-only/no-solver-outcome",
            input={"scores": scores, "width": width, "tau": 1, "typed_rows": products},
            expected={
                "stop": layer["stop"],
                "width_pruned": layer["width_pruned"],
                "band_pruned": layer["band_pruned"],
                "retained": retained,
                "read_verdict_count": len(layer["verdicts_read_only"]),
            },
            actual=actual,
        )
    values = [[np.nan] * 28, [0.0] + [-np.inf] * 27]
    warnings = [
        "divide by zero encountered in log",
        "invalid value encountered in subtract",
    ]
    by_id["unit/scorer-dead-and-live"].update(
        scope="scorer-API-only/no-model/no-solver-outcome",
        mask=[[False] * 28, [True] + [False] * 27],
        actual=values,
        warnings=warnings,
        expected={"values": copy.deepcopy(values), "warnings": warnings.copy()},
    )
    by_id["source/production-reference"].update(
        expected=V.REFERENCE_SHA, actual=V.REFERENCE_SHA
    )
    return V.encode(rows), replayed


def test_all_remaining_controls_have_an_independent_registered_expectation():
    rows, replayed = synthetic_controls()
    checked = V.verify_control_rows(rows, replayed)
    assert len(checked) == 49
    assert len(set(checked)) == 49


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("observation/A-safe", "actual"),
        ("observation/score/executable-nan/late", "actual"),
        ("parameter/bool/gc0", "collecting"),
        ("parameter/nan/gc1", "tau"),
        ("exception/scorer/gc0", "actual"),
        ("unit/last-ranked-width", "actual"),
        ("unit/last-ranked-width-all-inf", "actual"),
        ("unit/width-before-band", "expected"),
        ("unit/last-row-band", "input"),
        ("unit/scorer-dead-and-live", "mask"),
        ("source/production-reference", "actual"),
    ],
)
def test_controls_reject_corrupt_values_and_scope(name, field):
    rows, replayed = synthetic_controls()
    row = next(row for row in rows if row["id"] == name)
    row[field] = "substituted"
    with pytest.raises(V.VerificationError):
        V.verify_control_rows(rows, replayed)


def test_matching_saved_expected_and_actual_cannot_replace_control_truth():
    rows, replayed = synthetic_controls()
    row = next(row for row in rows if row["id"] == "parameter/bool/gc0")
    row["expected"]["gc_enabled"] = True
    row["actual"]["gc_enabled"] = True
    row["collecting"] = True
    with pytest.raises(V.VerificationError):
        V.verify_control_rows(rows, replayed)


@pytest.mark.parametrize(
    "name",
    [
        "anchor/seen",
        "anchor/close",
        "anchor/type",
        "anchor/size",
        "seen/reentry",
        "identity/encoding",
        "boundary/included",
        "boundary/top-ties-w1",
        "stop/width-before-unread-guard",
        "resource/body/fuel/64",
    ],
)
def test_registered_semantic_predicates_are_checked_without_candidate(name):
    case = find_case(name)
    replay = V.oracle_search(case)
    reference = V.oracle_search(dict(case, band=None))
    V.verify_registered_predicates(case, replay, reference)
