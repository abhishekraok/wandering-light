"""Author and acceptance tool for short-body-authoring (#167): capped-body programs for
the #101-cohort behaviours `BEST` does not solve.

**Analysis only.** These programs answer whether a short-body program exists for
the unsolved eval behaviours and what it would cost a search; they are never
training data, exactly as the authored eval witnesses are not.

    PYTHONPATH=. uv run python experiments/short_body/check.py --list
    PYTHONPATH=. uv run python experiments/short_body/check.py --show BEHAVIOUR
    PYTHONPATH=. uv run python experiments/short_body/check.py --try BEHAVIOUR "PROGRAM" [--cap 8]
    PYTHONPATH=. uv run python experiments/short_body/check.py --file FILE.jsonl [--cap 8]

A program is verified exactly as staged-witness-reauthoring (#155) verifies:
against the behaviour's Python reference on 500 fresh samples, and on every v4
task's shown and held-out pairs. It is within the cap if no quotation it
executes holds more than the cap (4 atoms, or 8 in the fallback round), nested
quotations counted, as #155's `measure` counts them. Beside #155's metrics it
reports the longest body in ops and the segments a segment-reset search would
see: the top level cut after each top-level combinator, as segment-composite
(#160) cuts it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.solver_rescoring import body_lengths  # noqa: E402
from experiments.staged_transfer import step_kinds  # noqa: E402
from experiments.staged_witnesses import check as staged  # noqa: E402

CAPS = (4, 8)
SEPARATION = REPO / "reports" / "solver-rescoring-20260923" / "separation.json"


def unsolved() -> list[str]:
    """The cohort behaviours `BEST`'s `b64` cell neither verifies nor grades (#165)."""
    rows = json.loads(SEPARATION.read_text())["rows"]
    return [r["behaviour"] for r in rows if r["status"] == "unsolved"]


def segment_lengths(task) -> list[int]:
    """Ops per segment: cut after each top-level combinator, as the composite cuts."""
    lengths, current = [], 0
    for kind in step_kinds(task):
        current += 1
        if kind == "run":
            lengths.append(current)
            current = 0
    if current or not lengths:
        lengths.append(current)
    return lengths


def shape(task) -> dict[str, int]:
    bodies, segments = body_lengths(task.program), segment_lengths(task)
    return {
        "ops": len(task.program),
        "quotations": len(bodies),
        "longest_body": max(bodies, default=0),
        "segments": len(segments),
        "longest_segment": max(segments),
    }


def eval_skeletons(tasks: dict[str, list]) -> set:
    """Skeletons of the eval behaviours: a train program must not share one."""
    return {
        staged.skeleton(group[0].program)
        for group in tasks.values()
        if group[0].split != "train"
    }


def needs_capping(tasks: dict[str, list], cap: int, segment_cap: int) -> list[str]:
    """Train behaviours whose witness breaks the body cap or the segment cap."""
    out = []
    for name, group in sorted(tasks.items()):
        witness = group[0]
        if witness.split != "train":
            continue
        root = staged.parse_stack(witness.inputs[0], basis=staged.BASIS)
        body = staged.measure(witness.program, root)["largest_quotation"]
        if body > cap or shape(witness)["longest_segment"] > segment_cap:
            out.append(name)
    return out


def assess(
    name: str,
    text: str,
    tasks: dict[str, list],
    cap: int = 4,
    segment_cap: int | None = None,
    skeletons: set | None = None,
) -> dict[str, Any]:
    out = staged.assess(name, text, tasks, skeletons or set())
    result: dict[str, Any] = {
        "behaviour": name,
        "program": text,
        "verified": out["verified"],
    }
    if not out["verified"]:
        result["failure"] = out["failure"]
        return result
    metrics = out["staged"]
    witness = tasks[name][0]
    mine = dataclasses.replace(witness, program=tuple(text.split()))
    result.update(
        cap=cap,
        within_cap=metrics["largest_quotation"] <= cap,
        metrics={**metrics, **shape(mine)},
        witness_metrics={**out["fused"], **shape(witness)},
    )
    if segment_cap is not None:
        result["segment_cap"] = segment_cap
        result["within_segment_cap"] = (
            result["metrics"]["longest_segment"] <= segment_cap
        )
    if skeletons:
        # Training data must never reproduce an eval behaviour's skeleton.
        result["eval_collision"] = out["verdict"] == "collides with an eval skeleton"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="the 25 behaviours")
    parser.add_argument("--show", metavar="BEHAVIOUR")
    parser.add_argument(
        "--try", dest="attempt", nargs=2, metavar=("BEHAVIOUR", "PROGRAM")
    )
    parser.add_argument("--file", type=Path, help="jsonl of {behaviour, program}")
    parser.add_argument("--cap", type=int, choices=CAPS, default=4)
    parser.add_argument(
        "--segment-cap", type=int, default=None, help="longest closed-state segment"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train-side authoring: reject eval-skeleton collisions",
    )
    parser.add_argument(
        "--list-train",
        action="store_true",
        help="train behaviours breaking --cap or --segment-cap",
    )
    args = parser.parse_args(argv)
    tasks = staged.v4_tasks()
    skeletons = eval_skeletons(tasks) if args.train else None
    if args.list_train:
        for name in needs_capping(tasks, args.cap, args.segment_cap or 10**9):
            print(f"{name}\t{tasks[name][0].length}")
    if args.list:
        for name in unsolved():
            print(f"{name}\t{tasks[name][0].length}")
    if args.show:
        staged.show(args.show, tasks)
        print(f"\nCap: no executed quotation over {args.cap} atoms, nested counted.")
    if args.attempt:
        print(
            json.dumps(
                assess(*args.attempt, tasks, args.cap, args.segment_cap, skeletons),
                indent=1,
            )
        )
    if args.file:
        for line in args.file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                verdict = assess(
                    row["behaviour"],
                    row["program"],
                    tasks,
                    args.cap,
                    args.segment_cap,
                    skeletons,
                )
                print(json.dumps({**row, **verdict}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
