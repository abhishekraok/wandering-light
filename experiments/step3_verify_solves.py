"""Check that every beam solve computes its behaviour, not just its held-out examples (#134).

A solve is graded on a task's held-out examples, and those come from the same
sampler as the shown ones, so a program that fits a regularity of the sampler
passes them without computing the behaviour. Two such solves in the #101
campaign were only found by hand on 2026-09-09 (`bounds_cover_list_flag`'s 44-op
program reads only the list head; `all_positive`'s 28-op program returns the
minimum). This checks every distinct solved program instead:

- a program equal to one of the behaviour's catalogue witnesses is genuine by
  construction, since `Behaviour` witnesses are verified against the reference;
- any other program is run against the Python reference on fresh draws from the
  behaviour's own sampler and on perturbed draws (wider values, lengths 0-10,
  shuffled), and is genuine only if it agrees on every draw the reference
  accepts. The first disagreement is kept as a counterexample.

Every timing cell is then rescored: graded solves against verified solves, per
witness-length bucket.

Usage:
    uv run python experiments/step3_verify_solves.py \
        --run-dir checkpoints/step3-20260912/v4only \
        --run-dir checkpoints/step3-20260912/mixed \
        --out reports/world-plus-v4-policy-20260912/verified_solves.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from wandering_light.stack_machine import value_key
from wandering_light.stack_pbe import CATALOGUE, run_program, to_stack, to_value

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41+")
WIDE = tuple(range(-20, 21))


def perturb(rng: random.Random, args: tuple) -> tuple:
    """Same argument shapes, off the sampler's distribution."""
    out = []
    for arg in args:
        if isinstance(arg, list):
            values = [rng.choice(WIDE) for _ in range(rng.randint(0, 10))]
            rng.shuffle(values)
            out.append(values)
        elif isinstance(arg, bool):
            out.append(rng.choice((True, False)))
        elif isinstance(arg, int):
            out.append(rng.choice(WIDE))
        else:
            out.append(arg)
    return tuple(out)


def agrees(program: tuple[str, ...], behaviour, args: tuple) -> bool | None:
    """True/False on a draw the reference accepts; None if it refuses the input."""
    try:
        expected = behaviour.reference(*args)
        want = value_key(to_value(expected))
    except Exception:
        return None
    outcome = run_program(program, to_stack(args))
    if isinstance(outcome, str):
        return False
    stack, _fuel = outcome
    return len(stack) == 1 and value_key(stack[0]) == want


def check(program: tuple[str, ...], behaviour, draws: int, seed: int) -> dict[str, Any]:
    if program in behaviour.programs:
        return {"verdict": "genuine", "why": "equals a catalogue witness"}
    rng = random.Random(seed)
    tallies = {}
    for mode in ("sampler", "perturbed"):
        tested = failed = 0
        counterexample = None
        for _ in range(draws):
            args = behaviour.sample(rng)
            if mode == "perturbed":
                args = perturb(rng, args)
            verdict = agrees(program, behaviour, args)
            if verdict is None:
                continue
            tested += 1
            if not verdict:
                failed += 1
                if counterexample is None:
                    counterexample = {
                        "args": [list(a) if isinstance(a, list) else a for a in args],
                        "reference": behaviour.reference(*args),
                    }
        tallies[mode] = {
            "tested": tested,
            "failed": failed,
            "counterexample": counterexample,
        }
    genuine = all(t["failed"] == 0 and t["tested"] > 0 for t in tallies.values())
    return {"verdict": "genuine" if genuine else "shortcut", "draws": tallies}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=134)
    parser.add_argument("--eval-corpus", default="stack_pbe_v4")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    by_name = {b.name: b for b in CATALOGUE}
    cells = []
    for run_dir in args.run_dir:
        for path in sorted((run_dir / "timing").glob("*.json")):
            cell = json.loads(path.read_text())
            if cell.get("eval_corpus", args.eval_corpus) != args.eval_corpus:
                continue
            if cell.get("max_depth") != 48 or cell.get("beam_width") != 64:
                continue
            cells.append((run_dir.name, cell))

    distinct: dict[tuple[str, str], dict[str, Any]] = {}
    for _, cell in cells:
        for row in cell["rows"]:
            if row["solved"]:
                distinct.setdefault((row["behaviour"], " ".join(row["program"])), {})
    for (name, program), slot in sorted(distinct.items()):
        slot.update(check(tuple(program.split()), by_name[name], args.draws, args.seed))
        tag = slot["verdict"]
        extra = ""
        if "draws" in slot:
            s, p = slot["draws"]["sampler"], slot["draws"]["perturbed"]
            extra = f" sampler {s['failed']}/{s['tested']} failed, perturbed {p['failed']}/{p['tested']} failed"
            if p["counterexample"] or s["counterexample"]:
                extra += f" e.g. {(s['counterexample'] or p['counterexample'])}"
        print(f"{tag:8s} {name:28s} {len(program.split()):3d} ops{extra}")

    table = []
    for arm, cell in cells:
        graded = dict.fromkeys(BUCKETS, 0)
        verified = dict.fromkeys(BUCKETS, 0)
        for row in cell["rows"]:
            if not row["solved"]:
                continue
            graded[row["bucket"]] += 1
            key = (row["behaviour"], " ".join(row["program"]))
            if distinct[key]["verdict"] == "genuine":
                verified[row["bucket"]] += 1
        table.append(
            {
                "arm": arm,
                "seed": cell["seed"],
                "graded": graded,
                "verified": verified,
                "graded_total": sum(graded.values()),
                "verified_total": sum(verified.values()),
                "tasks": len(cell["rows"]),
            }
        )
        print(
            f"{arm:8s} s{cell['seed']}: graded {sum(graded.values())}, verified "
            f"{sum(verified.values())} of {len(cell['rows'])} | "
            + " ".join(f"{b} {verified[b]}/{graded[b]}" for b in BUCKETS)
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "draws_per_mode": args.draws,
                    "seed": args.seed,
                    "programs": [
                        {"behaviour": b, "program": p, **v}
                        for (b, p), v in sorted(distinct.items())
                    ],
                    "cells": table,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
