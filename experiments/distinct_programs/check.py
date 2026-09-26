"""Author and acceptance tool for distinct-programs (#174).

Two authoring passes share this tool. A behaviour from a source that ships no
reference (BIG-bench ``list_functions``, DreamCoder, LambdaBeam) first gets a
Python reference written against its given tests; `extend` then adds examples
drawn to break the properties those tests share, exactly as world-authoring-500
(#127) extended MBPP. Every behaviour then gets a staged `mark-28` witness,
verified on every test and measured with the staged-witness-reauthoring (#155)
rule quantities.

    uv run python experiments/distinct_programs/check.py B.jsonl --show lf-c001
    uv run python experiments/distinct_programs/check.py B.jsonl --reference lf-c001 ref.py
    uv run python experiments/distinct_programs/check.py B.jsonl --try lf-c001 "<program>"
    uv run python experiments/distinct_programs/check.py B.jsonl --file programs_a.jsonl
    uv run python experiments/distinct_programs/check.py B.jsonl --references refs_a.jsonl --out B2.jsonl

``--reference`` checks one reference file (a function named ``f``) against the
given tests. ``--references`` takes ``{"id", "solution"}`` lines, keeps those
that reproduce every given test, extends them and writes the behaviours out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.staged_witnesses.check import measure, within_rules
from experiments.world_behaviours.extract import SEED, extend, run_solution
from experiments.world_behaviours.verify import check as run_tests
from wandering_light.stack_pbe import to_stack

ENTRY = "f"


def load(path: Path) -> dict[str, dict]:
    return {
        (b := json.loads(line))["id"]: b
        for line in path.read_text().splitlines()
        if line
    }


def reference_failures(behaviour: dict, solution: str) -> list[dict]:
    """The given tests the reference does not reproduce (empty means it passes)."""
    given = [t for t in behaviour["tests"] if not t.get("generated")]
    results = run_solution(solution, ENTRY, [t["args"] for t in given])
    return [
        {"args": t["args"], "want": t["out"], "got": r[1] if r[0] == "ok" else r}
        for t, r in zip(given, results, strict=True)
        if r[0] != "ok" or r[1] != t["out"]
    ]


def assess(behaviour: dict, text: str) -> dict[str, Any]:
    """Verify on every test, then measure the staged-rule quantities.

    Verdicts: ``staged`` (rules hold, two or more executions with a closed
    state after each), ``plain`` (rules hold, one cycle), ``rule_break``
    (verified, but a body, write run or park exceeds the #155 caps).
    """
    program = text.split()
    tested = run_tests(program, behaviour)
    result: dict[str, Any] = {
        "id": behaviour["id"],
        "program": text,
        "verified": tested["parsed"] and tested["passed"] == tested["total"],
        "passed": f"{tested['passed']}/{tested['total']}",
    }
    if not result["verified"]:
        result["failures"] = tested.get("failures") or tested.get("unknown")
        return result
    metrics = measure(tuple(program), to_stack(tuple(behaviour["tests"][0]["args"])))
    broken = within_rules(metrics)
    result["metrics"] = metrics
    result["broken_rules"] = broken
    if broken:
        result["verdict"] = "rule_break"
    elif metrics["post_execution_states"] >= 2:
        result["verdict"] = "staged"
    else:
        result["verdict"] = "plain"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("behaviours", type=Path)
    parser.add_argument("--show")
    parser.add_argument("--reference", nargs=2, metavar=("ID", "FILE"))
    parser.add_argument("--references", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--try", dest="try_", nargs=2, metavar=("ID", "PROGRAM"))
    parser.add_argument("--file", type=Path)
    args = parser.parse_args(argv)
    behaviours = load(args.behaviours)

    if args.show:
        b = behaviours[args.show]
        print(f"{b['id']}  ({b['source']})  {b.get('name', '')}")
        print(b.get("description", ""))
        if b.get("solution"):
            print("\n--- reference ---\n" + b["solution"])
        print(f"\n--- tests ({len(b['tests'])}) ---")
        for t in b["tests"]:
            tag = "  generated" if t.get("generated") else ""
            print(f"{json.dumps(t['args'])} -> {json.dumps(t['out'])}{tag}")
        return 0
    if args.reference:
        ident, file = args.reference
        bad = reference_failures(behaviours[ident], Path(file).read_text())
        print(json.dumps({"id": ident, "passes": not bad, "failures": bad[:4]}))
        return 0 if not bad else 1
    if args.references:
        kept, rejected = [], []
        for line in args.references.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            b = dict(behaviours[row["id"]])
            if not row.get("solution") or reference_failures(b, row["solution"]):
                rejected.append(row["id"])
                continue
            b.update(solution=row["solution"], entry=ENTRY, needs_reference=False)
            extended = extend(SEED, b)
            (kept if extended else rejected).append(extended or row["id"])
        if args.out:
            args.out.write_text("".join(json.dumps(b) + "\n" for b in kept))
        print(json.dumps({"kept": len(kept), "rejected": rejected}))
        return 0
    if args.try_:
        ident, text = args.try_
        print(json.dumps(assess(behaviours[ident], text), indent=1))
        return 0
    if args.file:
        rows = [json.loads(x) for x in args.file.read_text().splitlines() if x.strip()]
        bad = 0
        for row in rows:
            if not row.get("program"):
                print(json.dumps({"id": row["id"], "status": row.get("status")}))
                continue
            r = assess(behaviours[row["id"]], row["program"])
            bad += not r["verified"]
            print(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in ("id", "verified", "passed", "verdict", "broken_rules")
                    }
                )
            )
        print(f"{len(rows) - bad} of {len(rows)} rows verify or are declared unsolved")
        return 1 if bad else 0
    parser.error("give one of --show, --reference, --references, --try, --file")
    return 2


if __name__ == "__main__":
    sys.exit(main())
