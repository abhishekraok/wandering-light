"""Independent completed-output verifier for the one preserved synthetic gate.

Rebuilds literal definitions, machine attempts, scores and definition-form
retention; compares imported ordinary production search independently. Never
imports experiments or calls candidate search, runner, fixture constructors or
development checkers. Never writes gate files or authorizes evidence landing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wandering_light import (
    quotation_guard,
    stack_beam,
    stack_machine,
    stack_pbe,
    stack_policy,
)
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    parse_stack,
    step_raw,
    value_key,
)

REVISION = "826a8bbe3c0059032ca6c190a766cbc09226efc4"
FINAL_SHA = "a9d9fffbda6401673593506dfda0d13f5510e5b0a76c79f72bbcd95ef29da70a"
ATTEMPT_SHA = "b30933347e9cebba24879a88fb7a515cb594cbe55fb1cf11638bf10bda856102"
LOG_SHA = "a21ac9931f687940a3e542381f7412e6c32c6c6d85d698994446fd869b96c952"
RECEIPT_SHA = "d6c437c743d0414c2afd3e33270eb292e4cf4c88d99200aba1df2a2782f847ee"
REFERENCE_SHA = "dcf4558a6cfa11d8b8dde9aaa02ff21c9837ec16279a692f9b7e723c94ec616b"
SOURCE_PATHS = (
    "experiments/score_band_search.py",
    "experiments/score_band_fixtures.py",
    "experiments/score_band_checks.py",
    "experiments/score_band_gate.py",
    "tests/test_score_band_search.py",
    "tests/test_score_band_gate.py",
    "reports/score-band-contract-20261003/DESIGN.md",
    "reports/score-band-contract-20261003/AMENDMENT_A.md",
    "reports/score-band-contract-20261003/AMENDMENT_B.md",
    "reports/score-band-contract-20261003/FIXTURES.md",
    "wandering_light/stack_beam.py",
    "wandering_light/stack_machine.py",
    "wandering_light/stack_pbe.py",
    "wandering_light/stack_policy.py",
    "wandering_light/quotation_guard.py",
)
OP_NAMES = (
    "0",
    "1",
    "nil",
    "true",
    "false",
    "dup",
    "swap",
    "drop",
    "over",
    "cons",
    "uncons",
    "null?",
    "add",
    "sub",
    "mul",
    "div",
    "eq?",
    "lt?",
    "dip",
    "quote",
    "cat",
    "i",
    "map",
    "filter",
    "fold",
    "ifte",
    "[",
    "]",
)
TASK_TEXT = {
    "T-pair": (("2 3", "4 5"), ("12", "40"), ("mul", "dup", "add")),
    "T-list": (
        ("(1 2)", "(3 4)"),
        ("(2 1)", "(4 3)"),
        ("nil", "[", "cons", "]", "fold"),
    ),
    "T-typed": (
        ("(true 1) false", "(false 2) true"),
        ("(1 true) false", "(2 false) true"),
        ("swap", "nil", "[", "cons", "]", "fold", "swap"),
    ),
    "T-single": (("2 3",), ("12",), ("mul", "dup", "add")),
    "T-same": (("2 2",), ("4",), ("add",)),
    "T-probe": (("2 3",), ("9999",), ()),
    "T-equal-probe": (("2 2",), ("9999",), ()),
    "T-empty": (("[]",), ("9999",), ()),
    "T-open": (("MARK 0",), ("9999",), ()),
    "T-long": ((" ".join("0" for _ in range(163)),), ("9999",), ()),
}


class VerificationError(ValueError):
    """Evidence did not meet the fixed verifier contract."""


def require(condition, label):
    if not condition:
        raise VerificationError(label)


def same_json(actual, expected, label):
    """Compare numeric 1/1.0 equally, but never equate bool with int."""
    if isinstance(expected, dict):
        require(type(actual) is dict and actual.keys() == expected.keys(), label)
        for key in expected:
            same_json(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is list and len(actual) == len(expected), label)
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            same_json(left, right, f"{label}/{index}")
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and actual == expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise VerificationError(f"unnamed nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, expected, limit):
    require(path.stat().st_size <= limit, f"byte limit: {path.name}")
    data = path.read_bytes()
    require(sha(data) == expected, f"hash: {path.name}")
    return data


def script(rules=(), *, default=-100.0, illegal="-inf", seed=None, raises=False):
    return {
        "rules": [list(row) for row in rules],
        "default": default,
        "illegal": illegal,
        "seed": seed,
        "raises": raises,
    }


def priorities(root, later=()):
    return script([(0, 0, name, value) for name, value in root] + list(later))


def definition(name, task="T-probe", **changes):
    value = {
        "id": name,
        "task": task,
        "script": script(),
        "width": 4,
        "horizon": 1,
        "guard": "off",
        "cap": None,
        "budget": None,
        "unit": "executions",
        "deadline": None,
        "band": 0.0,
        "expectation": "retention-oracle",
    }
    value.update(changes)
    return value


def registered_definitions():
    """Separate literal implementation of the pre-run FIXTURES.md catalog."""
    disabled = [
        definition(
            f"disabled/{task}/s{seed}/w{width}/h{horizon}/{guard}",
            task,
            script=script(seed=seed),
            width=width,
            horizon=horizon,
            guard=guard,
            band=None,
            expectation="reference-equivalence",
        )
        for task in ("T-pair", "T-list", "T-typed")
        for seed in (0, 1, 2)
        for width in (1, 4, 16)
        for horizon in (1, 4, 8)
        for guard in ("off", "body", "close", "both")
    ]
    semantic = [
        definition(
            "A-safe",
            "T-single",
            horizon=3,
            band=1,
            script=priorities(
                [("mul", 0), ("drop", -2), ("quote", -3), ("dup", -4)],
                [(1, 0, "dup", 0)],
            ),
            expectation="safe-6-vs-3",
        ),
        definition(
            "B-only-solution-cut",
            "T-single",
            horizon=3,
            band=1,
            script=priorities(
                [("drop", 0), ("mul", -2), ("quote", -3), ("dup", -4)],
                [(1, 0, "dup", -10), (1, 1, "dup", 0)],
            ),
            expectation="reference-fit-finite-failure",
        ),
        definition(
            "C-later-first-fit",
            "T-single",
            horizon=3,
            band=1,
            script=priorities(
                [("mul", 0), ("dup", -2), ("drop", -3), ("quote", -4)],
                [(1, 0, "dup", -10), (1, 1, "add", 0)],
            ),
            expectation="changed-first-fit",
        ),
        definition(
            "fit/enumeration",
            "T-same",
            script=script(raises=True),
            expectation="add-no-scorer",
        ),
        definition(
            "fit/before-size",
            "T-same",
            script=script(raises=True),
            cap=0,
            expectation="add-no-scorer",
        ),
        definition(
            "anchor/seen",
            "T-equal-probe",
            script=priorities([("swap", 0), ("add", -2), ("mul", -3)]),
        ),
        definition(
            "anchor/close",
            "T-empty",
            script=priorities([("[", 0), ("0", -2)]),
            guard="close",
        ),
        definition(
            "anchor/type",
            "T-open",
            script=priorities([("null?", 0), ("]", -2)]),
            guard="body",
        ),
        definition("anchor/size", script=priorities([("dup", 0), ("mul", -2)]), cap=2),
        definition(
            "duplicate/same-layer",
            "T-equal-probe",
            script=priorities([("add", 0), ("mul", 0)]),
        ),
        definition(
            "seen/reentry",
            script=priorities(
                [("swap", 0), ("mul", -2), ("drop", -3), ("quote", -4)],
                [(1, 0, "mul", 0)],
            ),
            horizon=2,
            band=1,
        ),
        definition(
            "identity/encoding",
            "T-long",
            script=priorities([("dup", 0), ("drop", 0)]),
            cap=200,
        ),
        definition(
            "boundary/included",
            script=priorities([("mul", 0), ("drop", -1), ("quote", -2)]),
            band=1,
        ),
        definition(
            "boundary/top-ties",
            script=priorities([("mul", 0), ("drop", 0), ("quote", -1)]),
        ),
        definition(
            "boundary/top-ties-w1",
            script=priorities([("mul", 0), ("drop", 0), ("quote", -1)]),
            width=1,
        ),
        definition(
            "ordering/cumulative",
            "T-single",
            script=priorities(
                [("mul", 0), ("dup", -2), ("drop", -3), ("quote", -4)],
                [(1, 0, "dup", -10), (1, 1, "add", 0)],
            ),
            horizon=3,
            band=1,
        ),
        definition(
            "band/inactive",
            script=script(default=0),
            horizon=2,
            band=2,
            expectation="reference-equivalence",
        ),
        definition(
            "score/all-inf", script=script(default="-inf"), expectation="all-inf"
        ),
        definition(
            "score/illegal-nan",
            script=script(default=0, illegal="NaN"),
            horizon=2,
            expectation="reference-equivalence",
        ),
        definition(
            "empty/eligible", "T-empty", cap=0, expectation="frontier-exhausted"
        ),
        definition(
            "stop/width-before-unread-guard",
            "T-empty",
            script=script(default=0),
            width=5,
            guard="close",
            cap=1,
            band=2,
            expectation="width",
        ),
    ]
    for band in (None, 1):
        semantic.append(
            definition(
                f"stop/last-ranked-width/{band}",
                "T-empty",
                script=script(default=0),
                width=6,
                band=band,
                expectation="exhausted-at-width",
            )
        )
    for name, value in (("nan", "NaN"), ("posinf", "+inf")):
        for scope, task, op, cap in (
            ("late", "T-probe", "quote", None),
            ("size", "T-probe", "dup", 2),
            ("seen", "T-equal-probe", "swap", None),
        ):
            semantic.append(
                definition(
                    f"score/executable-{name}/{scope}",
                    task,
                    script=priorities([(op, value)]),
                    cap=cap,
                    expectation="ValueError",
                )
            )
    resource = []
    for unit in ("executions", "fuel", "executions+fuel"):
        for cap in (None, 64):
            for budget in (0, 1, 7):
                for band in (None, 1):
                    resource.append(
                        definition(
                            f"resource/{unit}/{cap}/{budget}/{band}",
                            "T-pair",
                            horizon=3,
                            guard="both",
                            cap=cap,
                            budget=budget,
                            unit=unit,
                            band=band,
                            expectation="budget-before-score",
                        )
                    )
            resource.append(
                definition(
                    f"resource/body/{unit}/{cap}",
                    "T-list",
                    script=script(
                        [
                            (index, 0, op, 0)
                            for index, op in enumerate(
                                ("nil", "[", "cons", "]", "fold")
                            )
                        ]
                    ),
                    horizon=6,
                    cap=cap,
                    budget=100,
                    unit=unit,
                    band=None,
                    expectation="reference-equivalence",
                )
            )
    resource.append(
        definition(
            "resource/past-deadline",
            "T-same",
            script=script(raises=True),
            deadline=0,
            expectation="past-deadline",
        )
    )
    require(
        (len(disabled), len(semantic), len(resource)) == (324, 29, 43),
        "definition counts",
    )
    return disabled, semantic, resource


def registered_ids():
    disabled, semantic, resource = registered_definitions()
    ids = [row["id"] for row in disabled + semantic + resource]
    ids.extend(f"observation/{row['id']}" for row in semantic)
    ids.extend(f"input/{name}" for name in TASK_TEXT)
    ids.extend(
        f"parameter/{name}/gc{state}"
        for name in ("negative", "nan", "posinf", "neginf", "bool", "string")
        for state in (0, 1)
    )
    ids.extend(f"exception/scorer/gc{state}" for state in (0, 1))
    ids.extend(
        f"unit/{name}"
        for name in (
            "last-ranked-width",
            "last-ranked-width-all-inf",
            "width-before-band",
            "last-row-band",
        )
    )
    ids.extend(("unit/scorer-dead-and-live", "source/production-reference"))
    require(len(ids) == len(set(ids)) == 455, "independent ID catalog")
    return ids


def verify_terminal(manifest):
    """Never access results of failed/incomplete/malformed terminal headers."""
    require(type(manifest) is dict, "terminal object")
    require(manifest.get("verdict") == "SYNTHETIC_GATE_A_PASS", "terminal verdict")
    require(manifest.get("complete") is True, "terminal completeness")
    require(
        type(manifest.get("count")) is int and manifest["count"] == 455,
        "terminal count",
    )
    ids = registered_ids()
    same_json(manifest.get("expected_ids"), ids, "expected IDs")
    same_json(manifest.get("actual_ids"), ids, "actual IDs")
    rows = manifest.get("results")
    require(type(rows) is list and len(rows) == 455, "complete rows")
    require(all(type(row) is dict for row in rows), "row objects")
    same_json([row.get("id") for row in rows], ids, "ordered row IDs")
    require(all(row.get("check") == "PASS" for row in rows), "row check")
    return rows


def task_definition(name):
    inputs, outputs, program = TASK_TEXT[name]
    return {
        "behaviour": name,
        "summary": "Manually typed score-band QA fixture",
        "split": "train",
        "task_hash_split": "train",
        "inputs": list(inputs),
        "outputs": list(outputs),
        "held_out_inputs": [],
        "held_out_outputs": [],
        "program": list(program),
        "length": len(program),
        "examples": len(inputs),
        "held_out": 0,
        "fuel": 0,
        "tags": ["synthetic"],
        "skeleton": "",
        "alternative_programs": [],
        "basis": "mark-28",
        "witness_source": "synthetic-manual" if program else "synthetic-no-witness",
        "macro_program": "",
        "schema_version": 2,
    }


def verify_definitions(rows):
    definitions = [row for group in registered_definitions() for row in group]
    require(len(rows) == 455, "definition population")
    for row, expected in zip(rows[:396], definitions, strict=True):
        name = expected["id"]
        same_json(row.get("definition"), expected, f"{name}/definition")
        same_json(row.get("task"), task_definition(expected["task"]), f"{name}/task")
        same_json(
            row.get("expectation_id"), expected["expectation"], f"{name}/expectation"
        )
        same_json(row.get("scope"), "real-machine-software-QA", f"{name}/scope")
    return len(definitions)


def typed_products(name):
    probe = ((Int(9999),),)
    table = {
        "T-pair": (((Int(2), Int(3)), (Int(4), Int(5))), ((Int(12),), (Int(40),))),
        "T-list": (
            ((Lst((Int(1), Int(2))),), (Lst((Int(3), Int(4))),)),
            ((Lst((Int(2), Int(1))),), (Lst((Int(4), Int(3))),)),
        ),
        "T-typed": (
            (
                (Lst((Bool(True), Int(1))), Bool(False)),
                (Lst((Bool(False), Int(2))), Bool(True)),
            ),
            (
                (Lst((Int(1), Bool(True))), Bool(False)),
                (Lst((Int(2), Bool(False))), Bool(True)),
            ),
        ),
        "T-single": (((Int(2), Int(3)),), ((Int(12),),)),
        "T-same": (((Int(2), Int(2)),), ((Int(4),),)),
        "T-probe": (((Int(2), Int(3)),), probe),
        "T-equal-probe": (((Int(2), Int(2)),), probe),
        "T-empty": (((),), probe),
        "T-open": (((Mark(), Int(0)),), probe),
        "T-long": (((Int(0),) * 163,), probe),
    }
    return table[name]


def encode(value):
    """Independent typed JSON encoding, also usable for later event digests."""
    if isinstance(value, (Int, Bool)):
        return {"type": type(value).__name__, "value": value.value}
    if isinstance(value, Sym):
        return {"type": "Sym", "name": value.name}
    if isinstance(value, Mark):
        return {"type": "Mark"}
    if isinstance(value, Lst):
        return {"type": "Lst", "items": encode(value.items)}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (encode(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, np.generic):
        return encode(value.item())
    if type(value) is float and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+inf" if value > 0 else "-inf"
    require(
        value is None or type(value) in (str, bool, int, float),
        "unrecognized encoder type",
    )
    return value


def decode_value(value):
    require(type(value) is dict, "typed value object")
    kind = value.get("type")
    if kind == "Int":
        require(
            value.keys() == {"type", "value"} and type(value["value"]) is int,
            "typed Int",
        )
        return Int(value["value"])
    if kind == "Bool":
        require(
            value.keys() == {"type", "value"} and type(value["value"]) is bool,
            "typed Bool",
        )
        return Bool(value["value"])
    if kind == "Sym":
        require(
            value.keys() == {"type", "name"} and type(value["name"]) is str, "typed Sym"
        )
        return Sym(value["name"])
    if kind == "Mark":
        require(value.keys() == {"type"}, "typed Mark")
        return Mark()
    if kind == "Lst":
        require(
            value.keys() == {"type", "items"} and type(value["items"]) is list,
            "typed Lst",
        )
        return Lst(tuple(decode_value(item) for item in value["items"]))
    raise VerificationError("unknown typed value")


def decode_product(value):
    require(
        type(value) is list and all(type(stack) is list for stack in value),
        "typed product",
    )
    return tuple(tuple(decode_value(item) for item in stack) for stack in value)


def replay_witness(program, initial):
    """Replay each op/example directly, with the production per-call fuel."""
    op_table = {op.name: op for op in stack_pbe.OPS}
    product = initial
    for name in program:
        require(name in op_table, "witness op")
        successor = []
        for stack in product:
            outcome = step_raw(op_table[name], stack, DEFAULT_FUEL_LIMIT, 0)
            require(not isinstance(outcome, str), f"witness refusal: {name}")
            successor.append(outcome[0])
        product = tuple(successor)
    return product


def verify_input_rows(rows):
    by_id = {row["id"]: row for row in rows}
    witnesses = 0
    for name, (texts, target_texts, program) in TASK_TEXT.items():
        row = by_id[f"input/{name}"]
        initial, target = typed_products(name)
        require(
            tuple(parse_stack(text, basis="mark-28") for text in texts) == initial,
            f"{name}/input parser",
        )
        require(
            tuple(parse_stack(text, basis="mark-28") for text in target_texts)
            == target,
            f"{name}/output parser",
        )
        same_json(row.get("task"), task_definition(name), f"{name}/task")
        expected = {
            "inputs": encode(initial),
            "outputs": encode(target),
            "witness_program": list(program),
            "witness_fits": True if program else None,
        }
        actual = {
            key: value for key, value in expected.items() if key != "witness_program"
        }
        same_json(row.get("expected"), expected, f"{name}/expected")
        same_json(row.get("actual"), actual, f"{name}/actual")
        for key, product in (("typed_inputs", initial), ("typed_outputs", target)):
            require(decode_product(row.get(key)) == product, f"{name}/{key}")
        # Decode both persisted copies too, so e.g. Int.value=1.0 is rejected.
        for copy in ("expected", "actual"):
            require(
                decode_product(row[copy]["inputs"]) == initial, f"{name}/{copy}/inputs"
            )
            require(
                decode_product(row[copy]["outputs"]) == target, f"{name}/{copy}/outputs"
            )
        if program:
            require(replay_witness(program, initial) == target, f"{name}/witness fit")
            witnesses += 1
    return {
        "typed_tasks_checked": len(TASK_TEXT),
        "nonempty_witnesses_replayed": witnesses,
    }


def verify_provenance(manifest, attempt, receipt, review_log, frozen_tree):
    require(manifest.get("source_revision") == REVISION, "source revision")
    same_json(manifest.get("ops_order"), list(OP_NAMES), "OPS order")
    same_json(
        manifest.get("limits"),
        {"cpu_threads": 1, "seconds": 300, "output_bytes": 1_000_000_000},
        "limits",
    )
    for key in ("model_calls", "corpus_reads", "checkpoint_reads"):
        require(type(manifest.get(key)) is int and manifest[key] == 0, key)
    require(
        manifest.get("scope")
        == "software QA only; no real coverage/clock/allocation claim",
        "scope",
    )
    require(manifest.get("attempt_sha256") == ATTEMPT_SHA, "attempt link")
    require(manifest.get("review_receipt_sha256") == RECEIPT_SHA, "receipt link")
    for key, value in {
        "source_revision": REVISION,
        "reviewer_family": "claude",
        "effort": "high",
        "verdict": "PRE-GATE CODE PASS",
        "pre_gate_code": True,
    }.items():
        same_json(receipt.get(key), value, f"review/{key}")
    lines = review_log.decode().splitlines()
    require(lines and lines[-1] == "exit 0", "actual successful review exit")
    response = strict_json("\n".join(lines[:-1]))
    require(
        response.get("is_error") is False and type(response.get("result")) is str,
        "actual review result",
    )
    report = response["result"].strip() + "\n"
    require(
        report.splitlines()[0] == "VERDICT: PRE-GATE CODE PASS",
        "explicit actual review verdict",
    )
    require(sha(report.encode()) == receipt.get("report_sha256"), "actual report hash")
    require(sha(review_log) == receipt.get("run_log_sha256"), "actual review log hash")
    frozen_tree = frozen_tree.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=frozen_tree, text=True
    ).strip()
    require(revision == REVISION, "frozen tree revision")
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=frozen_tree, text=True
        ).strip(),
        "frozen tree clean",
    )
    hashes = {name: sha((frozen_tree / name).read_bytes()) for name in SOURCE_PATHS}
    require(
        hashes["wandering_light/stack_beam.py"] == REFERENCE_SHA,
        "production reference hash",
    )
    same_json(manifest.get("source_hashes"), hashes, "actual frozen source hashes")
    same_json(
        attempt,
        {
            "source_revision": REVISION,
            "source_hashes": hashes,
            "expected_ids": registered_ids(),
            "review_receipt_sha256": RECEIPT_SHA,
        },
        "actual attempt",
    )
    origins = {
        name[:-3].replace("/", "."): {
            "path": str(frozen_tree / name),
            "sha256": hashes[name],
        }
        for name in SOURCE_PATHS
        if name.endswith(".py")
        and name.startswith(("experiments/", "wandering_light/"))
    }
    same_json(manifest.get("module_origins"), origins, "actual original module origins")
    require(tuple(op.name for op in stack_pbe.OPS) == OP_NAMES, "verifier OPS order")
    # This verifier runs from separate development source. Its established
    # low-level primitives must be byte-identical to the frozen production ones.
    used = {}
    verifier_tree = Path(__file__).resolve().parents[2]
    for module in (stack_machine, stack_pbe, stack_beam, stack_policy, quotation_guard):
        name = module.__name__.replace(".", "/") + ".py"
        path = Path(module.__file__).resolve()
        require(path == verifier_tree / name, f"verifier primitive origin: {name}")
        require(
            sha(path.read_bytes()) == hashes[name], f"verifier primitive hash: {name}"
        )
        used[module.__name__] = {"path": str(path), "sha256": hashes[name]}
    return used


def trace_digest(value):
    return sha(json.dumps(encode(value), sort_keys=True, allow_nan=False).encode())


def product_key(product):
    return tuple(tuple(value_key(value) for value in stack) for stack in product)


def total_atoms(product):
    values = [value for stack in product for value in stack]
    count = 0
    while values:
        value = values.pop()
        count += 1
        if isinstance(value, Lst):
            values.extend(value.items)
    return count


def numeric_score(value):
    named = {"NaN": math.nan, "+inf": math.inf, "-inf": -math.inf}
    return named[value] if isinstance(value, str) else float(value)


def scripted_values(spec, call, legal):
    """Independent fixed call/row/actual-op addressing; no model or encoding."""
    require(not spec["raises"], "unexpected scorer call before a fit")
    values = np.full(legal.shape, numeric_score(spec["default"]), dtype=np.float64)
    if spec["seed"] is not None:
        seed = spec["seed"]
        for row in range(legal.shape[0]):
            for index in range(len(OP_NAMES)):
                values[row, index] = -(
                    (7 * index + 11 * row + 13 * call + 17 * seed) % 29
                )
    for at_call, row, name, value in spec["rules"]:
        if at_call == call and row < legal.shape[0]:
            values[row, OP_NAMES.index(name)] = numeric_score(value)
    return np.where(legal, values, numeric_score(spec["illegal"]))


def attempt_product(product, op):
    """Rebuild one attempt, including first-refusing-example partial charges."""
    moved, fuel = [], 0
    for example, stack in enumerate(product):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None, f"example {example}: {outcome}", fuel, example
        moved.append(outcome[0])
        fuel += outcome[1]
    return tuple(moved), None, fuel, len(product)


def retention_from_definition(ranked, old_seen, case, depth):
    """Ordinary C followed by its band prefix, separate from candidate scan.

    Each row includes its already accumulated score. Classify the ordinary
    width-limited list first; only the prefix before its first band exclusion
    contributes to the real retained seen set. Emit no unread-tail verdicts.
    """
    ordinary, classifications = [], {}
    ordinary_seen = set(old_seen)
    remaining = case["horizon"] - depth - 1
    for rank, row in enumerate(ranked):
        node, index, product, _program, _score = row
        guard_reason = (
            quotation_guard.blocked(product[0], remaining, case["guard"])
            if case["guard"] != "off"
            else None
        )
        key = product_key(product)
        reason = (
            f"guard:{guard_reason}"
            if guard_reason is not None
            else "guard:size"
            if case["cap"] is not None and total_atoms(product) > case["cap"]
            else "duplicate"
            if key in ordinary_seen
            else "eligible"
        )
        classifications[rank] = reason
        if reason == "eligible":
            ordinary_seen.add(key)
            ordinary.append((rank, row))
            if len(ordinary) == case["width"]:
                break
    anchor = ordinary[0][1][4] if ordinary else None
    tau = None if case["band"] is None else float(case["band"])
    cutoff = anchor - tau if anchor is not None and tau is not None else None
    prefix = []
    excluded = None
    for rank, row in ordinary:
        if cutoff is not None and row[4] < cutoff:
            excluded = (rank, row)
            break
        prefix.append((rank, row))
    if excluded is not None:
        stop, last_read = "score_band", excluded[0]
        rejected = excluded[1][:2]
    elif len(prefix) == case["width"] and prefix[-1][0] + 1 < len(ranked):
        stop, last_read, rejected = "width", prefix[-1][0], None
    else:
        stop, last_read, rejected = "exhausted", len(ranked) - 1, None
    kept_ranks = {rank for rank, _row in prefix}
    verdicts = {}
    for rank in range(last_read + 1):
        node, index = ranked[rank][:2]
        verdicts[node, index] = (
            "retained"
            if rank in kept_ranks
            else "score_band"
            if excluded is not None and rank == excluded[0]
            else classifications[rank]
        )
        require(verdicts[node, index] != "eligible", "unclassified read verdict")
    kept_rows = [row for _rank, row in prefix]
    retained = tuple(
        (product, program, score) for _n, _i, product, program, score in kept_rows
    )
    parents = tuple((node, index) for node, index, _p, _pr, _s in kept_rows)
    new_seen = dict(old_seen)
    for _node, _index, product, _program, _score in kept_rows:
        new_seen[product_key(product)] = product
    layer = {
        "depth": depth,
        "retained": retained,
        "parents": parents,
        "verdicts_read_only": verdicts,
        "anchor": anchor,
        "tau": tau,
        "stop": stop,
        "first_band_rejected": rejected,
        "all_inf_fallback": tau is not None and anchor == -math.inf,
        "width_pruned": stop == "width",
        "band_pruned": stop == "score_band",
        "ranked_count": len(ranked),
        "ranked_sha256": trace_digest(tuple(ranked)),
        "old_seen_sha256": trace_digest(frozenset(old_seen.values())),
    }
    return retained, new_seen, layer


def oracle_search(case):
    """Definition-form machine replay; never calls any search implementation."""
    initial, target = typed_products(case["task"])
    beam = ((initial, (), 0.0),)
    seen = {product_key(initial): initial}
    events, layers = [], []
    executions = fuel = expansions = calls = 0
    width_pruned = band_pruned = False
    require(case["deadline"] in (None, 0), "only fixed past-deadline replay supported")

    def finish(program, termination, depth):
        events.append(("terminated", termination, depth, executions, fuel, expansions))
        return {
            "result": {
                "program": program,
                "termination": termination,
                "depth": depth,
                "executions": executions,
                "fuel": fuel,
                "expansions": expansions,
                "pruned": width_pruned,
            },
            "events": events,
            "layers": layers,
            "band_pruned": band_pruned,
            "scorer_calls": calls,
            "exception": None,
        }

    for depth in range(case["horizon"]):
        events.append(("started", depth, tuple(beam)))
        successors = []
        legal = np.zeros((len(beam), len(OP_NAMES)), dtype=bool)
        for node, (product, program, parent_score) in enumerate(beam):
            # Entering a parent costs one expansion before its first cap check.
            expansions += 1
            for index, op in enumerate(stack_pbe.OPS):
                spent = {
                    "executions": executions,
                    "fuel": fuel,
                    "executions+fuel": executions + fuel,
                }[case["unit"]]
                if case["budget"] is not None and spent >= case["budget"]:
                    return finish(None, "budget", depth + 1)
                if case["deadline"] == 0:
                    return finish(None, "wall_clock_cap", depth + 1)
                moved, reason, charged, stepped = attempt_product(product, op)
                events.append(
                    ("attempt", depth, node, index, moved, reason, charged, stepped)
                )
                executions += stepped
                fuel += charged
                if moved is None:
                    continue
                legal[node, index] = True
                extended = (*program, op.name)
                if product_key(moved) == product_key(target):
                    return finish(extended, "fit", depth + 1)
                successors.append((node, index, moved, extended, parent_score))
        if not successors:
            return finish(None, "frontier_exhausted", depth)
        values = scripted_values(case["script"], calls, legal)
        calls += 1
        observed = tuple(
            "nan" if np.isnan(value) else float(value) for value in values[legal]
        )
        events.append(("scored", depth, legal.shape, legal.tobytes(), observed))
        scored = [
            (node, index, product, program, parent_score + float(values[node, index]))
            for node, index, product, program, parent_score in successors
        ]
        # Validate every executable nonfit score before ranking or eligibility.
        if case["band"] is not None and any(
            math.isnan(row[4]) or row[4] == math.inf for row in scored
        ):
            return {
                "result": None,
                "events": events,
                "layers": layers,
                "band_pruned": band_pruned,
                "scorer_calls": calls,
                "exception": "ValueError",
                "executions": executions,
                "fuel": fuel,
                "expansions": expansions,
            }
        ranked = sorted(scored, key=lambda row: row[4], reverse=True)
        beam, seen, layer = retention_from_definition(ranked, seen, case, depth)
        layers.append(layer)
        events.append(
            ("retained", depth, beam, layer["verdicts_read_only"], layer["parents"])
        )
        width_pruned = width_pruned or layer["width_pruned"]
        band_pruned = band_pruned or layer["band_pruned"]
        if not beam:
            return finish(None, "frontier_exhausted", depth + 1)
    return finish(None, "max_depth", case["horizon"])


def summary_of_replay(replay, *, reference=False):
    return encode(
        {
            "events": len(replay["events"]),
            "event_sha256": trace_digest(replay["events"]),
            "attempts": sum(event[0] == "attempt" for event in replay["events"]),
            "layers": [] if reference else replay["layers"],
        }
    )


class ReferenceCapture(stack_beam.BeamObserver):
    """Own ordinary production observer; no experiment observer dependency."""

    def __init__(self):
        self.events = []

    def layer_started(self, depth, beam):
        self.events.append(("started", depth, tuple(beam)))

    def expansion_attempted(self, depth, node, index, successor, reason, fuel, stepped):
        self.events.append(
            ("attempt", depth, node, index, successor, reason, fuel, stepped)
        )

    def layer_scored(self, depth, values, legal):
        self.events.append(
            (
                "scored",
                depth,
                legal.shape,
                legal.tobytes(),
                tuple(
                    "nan" if np.isnan(value) else float(value)
                    for value in values[legal]
                ),
            )
        )

    def layer_retained(self, depth, beam, verdicts, parents):
        self.events.append(
            ("retained", depth, tuple(beam), dict(verdicts), tuple(parents))
        )

    def search_terminated(self, termination, depth, executions, fuel, expansions):
        self.events.append(
            ("terminated", termination, depth, executions, fuel, expansions)
        )


class ReferenceScorer:
    def __init__(self, spec):
        self.spec, self.calls = spec, 0

    def log_probs(self, blocks, legal):
        require(
            len(blocks) == legal.shape[0] and legal.shape[1] == len(OP_NAMES),
            "reference score shape",
        )
        values = scripted_values(self.spec, self.calls, legal)
        self.calls += 1
        return values


def imported_reference(case):
    task = task_definition(case["task"])
    for key in (
        "inputs",
        "outputs",
        "held_out_inputs",
        "held_out_outputs",
        "program",
        "tags",
        "alternative_programs",
    ):
        task[key] = tuple(task[key])
    capture = ReferenceCapture()
    scorer = ReferenceScorer(case["script"])
    result = stack_beam.beam_solve(
        stack_pbe.Task(**task),
        scorer,
        observer=capture,
        beam_width=case["width"],
        max_depth=case["horizon"],
        guard=case["guard"],
        max_state_atoms=case["cap"],
        budget=case["budget"],
        budget_unit=case["unit"],
        deadline=case["deadline"],
    )
    return {
        "result": {
            key: getattr(result, key)
            for key in (
                "program",
                "termination",
                "depth",
                "executions",
                "fuel",
                "expansions",
                "pruned",
            )
        },
        "events": capture.events,
        "layers": [],
        "band_pruned": False,
        "scorer_calls": scorer.calls,
        "exception": None,
    }


def verify_search_row(row, case, category):
    name = case["id"]
    replay = oracle_search(case)
    same_json(
        row.get("candidate_trace"), summary_of_replay(replay), f"{name}/candidate trace"
    )
    if case["expectation"] == "ValueError":
        require(replay["exception"] == "ValueError", f"{name}/invalid score exception")
        expected = {
            "exception": "ValueError",
            "scored_layers": 1,
            "retained_layers": 0,
            "counters_from_replay": {
                "executions": replay["executions"],
                "fuel": replay["fuel"],
                "expansions": None,
            },
        }
        actual = {
            key: value
            for key, value in expected.items()
            if key != "counters_from_replay"
        }
        same_json(row.get("expected"), expected, f"{name}/expected exception")
        same_json(row.get("actual"), actual, f"{name}/actual exception")
        require(
            sum(event[0] == "scored" for event in replay["events"]) == 1
            and not replay["layers"],
            f"{name}/exception callbacks",
        )
        return replay, None
    require(replay["exception"] is None, f"{name}/successful oracle")
    reference_case = dict(case, band=None)
    reference_oracle = oracle_search(reference_case)
    reference = imported_reference(reference_case)
    same_json(
        encode(reference["result"]),
        encode(reference_oracle["result"]),
        f"{name}/production versus independent result",
    )
    same_json(
        encode(reference["events"]),
        encode(reference_oracle["events"]),
        f"{name}/production versus independent events",
    )
    same_json(
        row.get("reference"),
        encode(reference_oracle["result"]),
        f"{name}/reference result",
    )
    same_json(
        row.get("reference_trace"),
        summary_of_replay(reference_oracle, reference=True),
        f"{name}/reference trace",
    )
    same_json(
        row.get("candidate"), encode(replay["result"]), f"{name}/candidate result"
    )
    same_json(row.get("band_pruned"), replay["band_pruned"], f"{name}/band pruning")
    same_json(
        row.get("width_pruned"), replay["result"]["pruned"], f"{name}/width pruning"
    )
    same_json(row.get("scorer_calls"), replay["scorer_calls"], f"{name}/scorer calls")
    expected_layers = [
        {
            key: value
            for key, value in layer.items()
            if key not in ("ranked_count", "ranked_sha256", "old_seen_sha256")
        }
        for layer in replay["layers"]
    ]
    expected = {
        "reference": reference_oracle["result"],
        "candidate": replay["result"],
        "candidate_layers": expected_layers,
        "band_pruned": replay["band_pruned"],
        "reference_trace_sha256": trace_digest(reference_oracle["events"]),
    }
    same_json(row.get("expected"), encode(expected), f"{name}/persisted expectations")
    if category != "semantic" or case["expectation"] in (
        "reference-equivalence",
        "all-inf",
        "exhausted-at-width",
        "add-no-scorer",
    ):
        same_json(
            encode(replay["result"]),
            encode(reference_oracle["result"]),
            f"{name}/declared equivalence",
        )
        same_json(
            encode(replay["events"]),
            encode(reference_oracle["events"]),
            f"{name}/declared event equivalence",
        )
        require(not replay["band_pruned"], f"{name}/equivalence band")
    for result in (replay["result"], reference_oracle["result"]):
        if result["program"] is not None:
            initial, target = typed_products(case["task"])
            require(
                replay_witness(result["program"], initial) == target,
                f"{name}/returned shown fit",
            )
    return replay, reference_oracle


def verify_search_rows(rows):
    disabled, semantic, resource = registered_definitions()
    require(len(rows) == 455, "search population")
    by_id = {row["id"]: row for row in rows}
    replayed = {}
    for category, cases in (
        ("disabled", disabled),
        ("semantic", semantic),
        ("resource", resource),
    ):
        for case in cases:
            replayed[case["id"]] = verify_search_row(by_id[case["id"]], case, category)
            verify_registered_predicates(case, *replayed[case["id"]])
    # Manual mandatory useful/adverse witnesses and paired charge scope.
    a, a_ref = replayed["A-safe"]
    b, b_ref = replayed["B-only-solution-cut"]
    c, c_ref = replayed["C-later-first-fit"]
    require(
        a["result"]["program"] == a_ref["result"]["program"] == ("mul", "dup", "add"),
        "A unchanged first fit",
    )
    require(
        (a_ref["result"]["expansions"], a["result"]["expansions"]) == (6, 3),
        "A paired expansion construction",
    )
    require(
        b_ref["result"]["program"] == ("mul", "dup", "add")
        and b["result"]["program"] is None
        and b["result"]["termination"] == "max_depth",
        "B solution loss",
    )
    require(
        c_ref["result"]["program"] == ("dup", "add", "mul")
        and c["result"]["program"] == ("mul", "dup", "add"),
        "C changed first fit",
    )
    return replayed


def verify_registered_predicates(case, replay, reference):
    """Independently check the named semantic/resource construction claims."""
    name = case["id"]
    if case["expectation"] == "ValueError":
        return
    result = replay["result"]
    layers = replay["layers"]
    first = layers[0] if layers else None
    names = [OP_NAMES[index] for _node, index in first["parents"]] if first else []
    if name.startswith("fit/"):
        require(
            result["program"] == ("add",) and replay["scorer_calls"] == 0,
            f"{name}/fit before score and cap",
        )
    anchors = {
        "anchor/seen": ("add", -2, "swap", "duplicate"),
        "anchor/close": ("0", -2, "[", "guard:close"),
        "anchor/type": ("]", -2, "null?", "guard:type"),
        "anchor/size": ("mul", -2, "dup", "guard:size"),
    }
    if name in anchors:
        kept, anchor, rejected, verdict = anchors[name]
        require(
            names == [kept]
            and first["anchor"] == anchor
            and first["verdicts_read_only"][0, OP_NAMES.index(rejected)] == verdict,
            f"{name}/eligible anchor",
        )
    elif name == "duplicate/same-layer":
        require(
            names == ["add"]
            and first["verdicts_read_only"][0, OP_NAMES.index("mul")] == "duplicate",
            f"{name}/typed duplicate",
        )
    elif name == "seen/reentry":
        require(
            layers[1]["retained"][0][1] == ("swap", "mul"),
            "rejected state later reentry",
        )
        event = next(
            event
            for event in reference["events"]
            if event[0] == "retained" and event[1] == 1
        )
        require(
            event[3][0, OP_NAMES.index("mul")] == "duplicate",
            "reference remembered state",
        )
    elif name == "identity/encoding":
        require(names == ["dup", "drop"], "full typed identity retention")
        left, right = [node[0] for node in first["retained"]]
        require(product_key(left) != product_key(right), "full states distinct")
        _initial, target = typed_products(case["task"])
        require(
            np.array_equal(
                stack_policy.encode_node(left, target),
                stack_policy.encode_node(right, target),
            ),
            "capped encodings equal",
        )
    elif name == "boundary/included":
        require(
            names == ["mul", "drop"] and first["stop"] == "score_band",
            "inclusive boundary",
        )
    elif name in ("boundary/top-ties", "boundary/top-ties-w1"):
        require(
            names == (["drop"] if case["width"] == 1 else ["drop", "mul"]),
            "stable actual OPS tie order",
        )
    elif name == "score/all-inf":
        require(
            first["all_inf_fallback"] and not replay["band_pruned"], "all-inf fallback"
        )
    elif name == "empty/eligible":
        require(
            result["termination"] == "frontier_exhausted"
            and first["anchor"] is None
            and not result["pruned"]
            and not replay["band_pruned"],
            "empty eligible frontier",
        )
    elif name == "stop/width-before-unread-guard":
        require(
            first["stop"] == "width"
            and (0, OP_NAMES.index("[")) not in first["verdicts_read_only"],
            "unread guard remains unknown",
        )
    elif name.startswith("stop/last-ranked-width/"):
        require(
            len(first["retained"]) == 6
            and first["stop"] == "exhausted"
            and not result["pruned"]
            and not replay["band_pruned"],
            "last ranked width exhausted",
        )
    elif name == "ordering/cumulative":
        require(
            reference["result"]["program"] == ("dup", "add", "mul")
            and result["program"] == ("mul", "dup", "add"),
            "cumulative ordering changes later fit",
        )
    if case["expectation"] == "budget-before-score":
        charged = (
            0
            if case["budget"] == 0
            else 2
            if case["budget"] == 1
            else 4
            if case["unit"] == "executions+fuel"
            else 8
        )
        require(
            result["termination"] == "budget"
            and result["executions"] == result["fuel"] == charged
            and result["expansions"] == 1
            and replay["scorer_calls"] == 0,
            f"{name}/before-score budget overshoot",
        )
    elif case["expectation"] == "past-deadline":
        require(
            result["termination"] == "wall_clock_cap"
            and result["executions"] == result["fuel"] == 0
            and result["expansions"] == 1
            and replay["scorer_calls"] == 0,
            "past-deadline preattempt charge",
        )
    elif name.startswith("resource/body/"):
        event = next(
            event
            for event in replay["events"]
            if event[0] == "attempt"
            and event[1] == 0
            and event[3] == OP_NAMES.index("i")
        )
        require(event[6] > event[7] > 0, f"{name}/root-i inner body fuel")


def verify_control_rows(rows, replayed):
    by_id = {row["id"]: row for row in rows}
    checked = []

    def check(name, expected):
        same_json(by_id[name], encode({"id": name, "check": "PASS", **expected}), name)
        checked.append(name)

    for case in registered_definitions()[1]:
        replay, _reference = replayed[case["id"]]
        name = f"observation/{case['id']}"
        if replay["exception"] is not None:
            value = {"exception": "ValueError"}
            check(name, {"expected": value, "actual": value})
        else:
            value = {"result": replay["result"], "band_pruned": replay["band_pruned"]}
            check(
                name,
                {
                    "expected": value,
                    "actual": value,
                    "band_pruned": replay["band_pruned"],
                },
            )
    invalid = {
        "negative": -1,
        "nan": "NaN",
        "posinf": "+inf",
        "neginf": "-inf",
        "bool": True,
        "string": "1",
    }
    for parameter, value in invalid.items():
        for collecting in (False, True):
            name = f"parameter/{parameter}/gc{int(collecting)}"
            outcome = {"exception": "ValueError", "gc_enabled": collecting}
            check(
                name,
                {
                    "expected": outcome,
                    "actual": outcome,
                    "collecting": collecting,
                    "tau": value,
                },
            )
    for collecting in (False, True):
        name = f"exception/scorer/gc{int(collecting)}"
        outcome = {"exception": "RuntimeError", "gc_enabled": collecting}
        check(
            name,
            {
                "expected": outcome,
                "actual": outcome,
                "collecting": collecting,
                "tau": None,
            },
        )
    root = ((Int(9),),)
    products = [((Int(0),),), ((Int(1),),)]
    for kind in (
        "last-ranked-width",
        "last-ranked-width-all-inf",
        "width-before-band",
        "last-row-band",
    ):
        scores = (
            [0.0, -2.0]
            if kind == "last-row-band"
            else [-math.inf, -math.inf]
            if kind == "last-ranked-width-all-inf"
            else [0.0, 0.0]
        )
        width = (
            1 if kind == "width-before-band" else 4 if kind == "last-row-band" else 2
        )
        ranked = [
            (0, index, product, (OP_NAMES[index],), scores[index])
            for index, product in enumerate(products)
        ]
        retained, _seen, layer = retention_from_definition(
            ranked,
            {product_key(root): root},
            definition("arithmetic-only", width=width, band=1),
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
        expected = {
            "stop": layer["stop"],
            "width_pruned": layer["width_pruned"],
            "band_pruned": layer["band_pruned"],
            "retained": retained,
            "read_verdict_count": len(layer["verdicts_read_only"]),
        }
        check(
            f"unit/{kind}",
            {
                "scope": "arithmetic-only/no-solver-outcome",
                "input": {
                    "scores": scores,
                    "width": width,
                    "tau": 1,
                    "typed_rows": products,
                },
                "expected": expected,
                "actual": actual,
            },
        )
    mask = np.zeros((2, len(OP_NAMES)), dtype=bool)
    mask[1, 0] = True
    expected_values = [
        [math.nan] * len(OP_NAMES),
        [0.0] + [-math.inf] * (len(OP_NAMES) - 1),
    ]
    expected_warnings = [
        "divide by zero encountered in log",
        "invalid value encountered in subtract",
    ]
    stub = SimpleNamespace(logits=lambda _blocks: np.zeros((2, len(OP_NAMES))))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        values = stack_policy.PolicyScorer.log_probs(stub, [None, None], mask)
    same_json(
        encode(values.tolist()),
        encode(expected_values),
        "independent production scorer API values",
    )
    require(
        sorted({str(item.message) for item in observed}) == expected_warnings,
        "independent production scorer API warnings",
    )
    check(
        "unit/scorer-dead-and-live",
        {
            "scope": "scorer-API-only/no-model/no-solver-outcome",
            "mask": mask.tolist(),
            "actual": expected_values,
            "warnings": expected_warnings,
            "expected": {"values": expected_values, "warnings": expected_warnings},
        },
    )
    check(
        "source/production-reference",
        {"expected": REFERENCE_SHA, "actual": REFERENCE_SHA},
    )
    require(len(checked) == len(set(checked)) == 49, "noninput control population")
    return checked


def load_verified_evidence(run_root, frozen_tree):
    """Fixed original evidence only; no CLI bypass for hashes or source."""
    gate = run_root / "gate-a"
    require(not (gate / "failure.json").exists(), "failed terminal file")
    # Refuse unsuccessful exits before even reading the manifest/results.
    log = read_pinned(run_root / "gate-a-run.log", LOG_SHA, 1_000_000)
    lines = log.decode().splitlines()
    require(lines and lines[-1] == "exit 0", "actual gate exit")
    same_json(
        strict_json("\n".join(lines[:-1])),
        {"verdict": "SYNTHETIC_GATE_A_PASS", "count": 455},
        "actual gate log",
    )
    manifest = strict_json(read_pinned(gate / "final.json", FINAL_SHA, 999_000_000))
    rows = verify_terminal(manifest)
    attempt = strict_json(read_pinned(gate / "attempt.json", ATTEMPT_SHA, 1_000_000))
    review_dir = run_root / "amended-pre-gate-review"
    receipt = strict_json(
        read_pinned(review_dir / "receipt.json", RECEIPT_SHA, 1_000_000)
    )
    require(
        Path(receipt["run_log"]).resolve() == (review_dir / "run.log").resolve(),
        "review log path",
    )
    review_log = read_pinned(
        review_dir / "run.log", receipt["run_log_sha256"], 10_000_000
    )
    origins = verify_provenance(manifest, attempt, receipt, review_log, frozen_tree)
    return rows, origins


def verify_foundation(run_root, frozen_tree):
    rows, origins = load_verified_evidence(run_root, frozen_tree)
    definitions = verify_definitions(rows)
    inputs = verify_input_rows(rows)
    return {
        "phase": "TERMINAL_PROVENANCE_DEFINITIONS_INPUTS_VERIFIED",
        "independent_gate_verdict": "PENDING_SEARCH_REPLAY_AND_CONTROLS",
        "original_source_revision": REVISION,
        "original_final_sha256": FINAL_SHA,
        "ordered_ids_checked": len(rows),
        "search_definitions_checked": definitions,
        **inputs,
        "verifier_primitive_origins": origins,
    }


def verify_complete(run_root, frozen_tree):
    verifier_tree = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=verifier_tree, text=True
    ).strip()
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=verifier_tree, text=True
        ).strip(),
        "verifier source clean before replay",
    )
    own_sha = sha(Path(__file__).read_bytes())
    rows, origins = load_verified_evidence(run_root, frozen_tree)
    definitions = verify_definitions(rows)
    inputs = verify_input_rows(rows)
    replayed = verify_search_rows(rows)
    controls = verify_control_rows(rows, replayed)
    checked = set(replayed) | {f"input/{name}" for name in TASK_TEXT} | set(controls)
    require(checked == set(registered_ids()), "all fixed IDs independently checked")
    require(
        sha((run_root / "gate-a/final.json").read_bytes()) == FINAL_SHA,
        "original final unchanged after replay",
    )
    require(
        sha(Path(__file__).read_bytes()) == own_sha
        and subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=verifier_tree, text=True
        ).strip()
        == revision,
        "verifier source unchanged after replay",
    )
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=verifier_tree, text=True
        ).strip(),
        "verifier source clean after replay",
    )
    for module in (stack_machine, stack_pbe, stack_beam, stack_policy, quotation_guard):
        origin = origins[module.__name__]
        require(
            Path(module.__file__).resolve() == Path(origin["path"])
            and sha(Path(origin["path"]).read_bytes()) == origin["sha256"],
            "verifier primitive unchanged after replay",
        )
    paired = {}
    for name in ("A-safe", "B-only-solution-cut", "C-later-first-fit"):
        own, reference = replayed[name]
        paired[name] = {"reference": reference["result"], "finite": own["result"]}
    return encode(
        {
            "independent_gate_verdict": "INDEPENDENT_VERIFICATION_PASS_FINAL_REVIEW_PENDING",
            "scope": "constructed software QA; no real allocation/coverage/clock claim",
            "original_source_revision": REVISION,
            "original_final_sha256": FINAL_SHA,
            "verifier_source_revision": revision,
            "verifier_sha256": own_sha,
            "ordered_ids_checked": len(rows),
            "search_definitions_checked": definitions,
            "search_rows_reconstructed": len(replayed),
            "noninput_controls_checked": len(controls),
            **inputs,
            "ordinary_production_reference_comparisons": sum(
                reference is not None for _own, reference in replayed.values()
            ),
            "mandatory_pairs": paired,
            "verifier_primitive_origins": origins,
            "gc_scope": "fixed saved GC/exception assertions checked; candidate is never re-executed",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-tree", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("foundation", "complete"), default="complete"
    )
    args = parser.parse_args()
    verify = verify_complete if args.phase == "complete" else verify_foundation
    print(json.dumps(verify(args.run_root, args.frozen_tree), sort_keys=True))


if __name__ == "__main__":
    main()
