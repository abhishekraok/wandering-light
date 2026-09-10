"""Verify mark-28 programs against world-authored behaviours (#118).

A program may be written with macros: ``macros.txt`` holds one ``name: tokens``
definition per line, and any token naming a macro expands recursively before
the program runs. The expanded program is what the executor sees and what the
corpus would store; the macro form is what a person writes.

Usage:
    # one program, verbose
    uv run python experiments/world_behaviours/verify.py behaviours.jsonl \
        --id mbpp-12 --program "0 [ add ] fold"
    # every program in a file: {"id": ..., "program": ...} per line
    uv run python experiments/world_behaviours/verify.py behaviours.jsonl \
        --programs programs.jsonl [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wandering_light.stack_machine import Config, format_stack, value_key
from wandering_light.stack_pbe import OP_BY_NAME, run_program, to_stack, to_value

HERE = Path(__file__).parent
MACRO_FILE = HERE / "macros.txt"


def load_macros(path: Path = MACRO_FILE) -> dict[str, list[str]]:
    """Load ``path`` and every sibling ``macros*.txt``; names must not collide."""
    macros: dict[str, list[str]] = {}
    files = sorted(path.parent.glob("macros*.txt")) if path.parent.exists() else []
    if path.exists() and path not in files:
        files.append(path)
    for file in files:
        for raw in file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name, body = line.split(":", 1)
            name = name.strip()
            if name in OP_BY_NAME:
                raise ValueError(f"{file.name}: macro {name!r} shadows a basis op")
            if name in macros and macros[name] != body.split():
                raise ValueError(
                    f"{file.name}: macro {name!r} defined twice with different bodies"
                )
            macros[name] = body.split()
    return macros


def expand(
    program: list[str], macros: dict[str, list[str]], depth: int = 0
) -> list[str]:
    if depth > 32:
        raise ValueError("macro recursion too deep")
    out: list[str] = []
    for tok in program:
        if tok in macros:
            out.extend(expand(macros[tok], macros, depth + 1))
        else:
            out.append(tok)
    return out


def check(program: list[str], behaviour: dict) -> dict:
    """Run ``program`` on every test; return pass counts and the first failures."""
    unknown = [t for t in program if t not in OP_BY_NAME]
    if unknown:
        return {
            "parsed": False,
            "unknown": unknown,
            "passed": 0,
            "total": len(behaviour["tests"]),
        }
    passed, failures = 0, []
    for index, test in enumerate(behaviour["tests"]):
        stack = to_stack(tuple(test["args"]))
        outcome = run_program(program, stack)
        want = (to_value(test["out"]),)
        if isinstance(outcome, str):
            failures.append(
                {
                    "test": index,
                    "args": test["args"],
                    "want": test["out"],
                    "refused": outcome,
                }
            )
            continue
        got, _fuel = outcome
        if tuple(map(value_key, got)) == tuple(map(value_key, want)):
            passed += 1
        else:
            failures.append(
                {
                    "test": index,
                    "args": test["args"],
                    "want": test["out"],
                    "got": format_stack(Config(got, 0)),
                }
            )
    return {
        "parsed": True,
        "passed": passed,
        "total": len(behaviour["tests"]),
        "failures": failures[:4],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("behaviours", type=Path)
    parser.add_argument("--id")
    parser.add_argument(
        "--show", action="store_true", help="print the behaviour named by --id"
    )
    parser.add_argument("--program")
    parser.add_argument("--programs", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--macros", type=Path, default=MACRO_FILE)
    args = parser.parse_args(argv)

    behaviours = {json.loads(l)["id"]: json.loads(l) for l in args.behaviours.open()}
    macros = load_macros(args.macros)

    if args.show:
        b = behaviours[args.id]
        print(
            f"# {b['id']}  ({b['source']}, {len(b['tests'])} tests, shown {len(b['shown'])})"
        )
        print(b["description"].strip())
        print("\n--- reference solution ---")
        print(b["solution"].strip()[-1500:])
        print("\n--- tests: args -> out  (stack is args bottom to top) ---")
        for t in b["tests"]:
            print(f"  {t['args']} -> {t['out']}")
        return 0

    if args.program is not None:
        if args.id not in behaviours:
            print(f"unknown id {args.id!r}; known: {sorted(behaviours)[:5]}...")
            return 2
        expanded = expand(args.program.split(), macros)
        result = check(expanded, behaviours[args.id])
        result["expanded"] = " ".join(expanded)
        result["length"] = len(expanded)
        print(json.dumps(result, indent=1))
        return 0 if result.get("passed") == result["total"] else 1

    rows = [json.loads(l) for l in args.programs.open()] if args.programs else []
    report = []
    for row in rows:
        b = behaviours[row["id"]]
        expanded = expand(row["program"].split(), macros)
        result = check(expanded, b)
        report.append(
            {
                "id": row["id"],
                "source": b["source"],
                "macro_program": row["program"],
                "program": " ".join(expanded),
                "macro_length": len(row["program"].split()),
                "length": len(expanded),
                "solved": result.get("passed") == result["total"],
                **result,
            }
        )
    solved = sum(r["solved"] for r in report)
    print(f"{solved}/{len(report)} verified on every test")
    for r in report:
        mark = "ok " if r["solved"] else "FAIL"
        print(
            f"  {mark} {r['id']:<40} {r['passed']}/{r['total']}  len {r['length']} (macro {r['macro_length']})"
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0 if solved == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
