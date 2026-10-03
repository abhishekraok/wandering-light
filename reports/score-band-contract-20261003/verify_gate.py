"""Independent completed-output verifier foundation for synthetic Gate A.

This revision verifies terminals, provenance, literal definitions and typed
inputs. Search replay and remaining controls are PENDING: it cannot issue an
independent Gate A PASS. Never imports experiments or calls candidate search,
runner, fixture constructors or development checkers. Never writes gate files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from wandering_light import stack_machine, stack_pbe
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    parse_stack,
    step_raw,
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
    for module in (stack_machine, stack_pbe):
        name = module.__name__.replace(".", "/") + ".py"
        path = Path(module.__file__).resolve()
        require(
            sha(path.read_bytes()) == hashes[name], f"verifier primitive hash: {name}"
        )
        used[module.__name__] = {"path": str(path), "sha256": hashes[name]}
    return used


def verify_foundation(run_root, frozen_tree):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-tree", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(verify_foundation(args.run_root, args.frozen_tree), sort_keys=True)
    )


if __name__ == "__main__":
    main()
