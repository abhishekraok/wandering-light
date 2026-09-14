"""Check that every beam solve computes its behaviour, not just its held-out examples (#134).

A solve is graded on a task's held-out examples, and those come from the same
sampler as the shown ones, so a program that fits a regularity of the sampler
passes them without computing the behaviour. Two such solves in the #101
campaign were only found by hand on 2026-09-09 (`bounds_cover_list_flag`'s 44-op
program reads only the list head; `all_positive`'s 28-op program returns the
minimum). This checks every distinct solved program instead:

- a program equal to one of the behaviour's catalogue witnesses gets the verdict
  ``equals_catalogue_witness`` and no draws. It is exactly as trusted as the
  corpus's own verification, which checks witnesses against the reference on
  draws from the behaviour's sampler only, so it is counted apart from the
  draw-tested solves rather than among them (#142);
- any other program is run against the Python reference in three modes: fresh
  draws from the behaviour's own sampler, perturbed draws (wider values, lengths
  0-10) and dense draws (integers and list values from -2..2, lengths 0-10,
  booleans unchanged), which force the duplicate and tie patterns that rare-case
  shortcuts get wrong and wide perturbations never produce. ``--no-dense`` drops
  the third. A draw counts only where the canonical witness itself computes the
  reference, so inputs outside the corpus's domain are skipped, not scored, and
  each mode records how many it skipped.

Verdict: ``shortcut`` if any mode finds a disagreement; ``genuine`` if every mode
tested at least ``--min-tested`` in-domain draws without one; otherwise
``inconclusive``. Fewer draws than the floor can only make a verdict
``inconclusive``, never ``genuine``. A pass is a finite test, not a proof: a
program that fails on a fraction p of inputs passes n draws with probability
(1 - p)^n.

Every timing cell at the searched budget (``--beam-width``, ``--max-depth``) is
then rescored: graded, verified, witness-equal and inconclusive solves per
witness-length bucket. ``verified`` is the draw-tested ``genuine`` verdicts only.

Usage:
    uv run python experiments/step3_verify_solves.py \\
        --run-dir checkpoints/step3-20260912/v4only \\
        --run-dir checkpoints/step3-20260912/mixed \\
        --out reports/world-plus-v4-policy-20260912/verified_solves_epochs.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from wandering_light.stack_machine import value_key
from wandering_light.stack_pbe import (
    CATALOGUE,
    catalogue_digest,
    catalogue_source_digest,
    run_program,
    to_stack,
    to_value,
)

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41+")
WIDE = tuple(range(-20, 21))
DENSE = tuple(range(-2, 3))
BEAM_WIDTH = 64
MAX_DEPTH = 48
DRAWS = 10_000
MIN_TESTED = 500
# Which rescoring column a verdict lands in; `shortcut` lands in none.
COLUMN_OF = {
    "genuine": "verified",
    "inconclusive": "inconclusive",
    "equals_catalogue_witness": "witness",
}

Key = tuple[str, str]


def partition(programs: list[dict[str, Any]]) -> tuple[set[Key], set[Key]]:
    """Keys of the draw-tested ``genuine`` and the witness-equal programs in a file.

    The reports define a verified solve as either (`RESULTS_DETAILED.md`), so
    their readers take the union and count the witness share beside it; this
    script's own table keeps the two apart. A file written before #142 graded
    witness-equal programs ``genuine`` with a ``why`` and no ``draws``, and
    reads the same way.
    """
    genuine: set[Key] = set()
    witness: set[Key] = set()
    for p in programs:
        key = (p["behaviour"], p["program"])
        if p["verdict"] == "equals_catalogue_witness" or (
            p["verdict"] == "genuine" and "draws" not in p
        ):
            witness.add(key)
        elif p["verdict"] == "genuine":
            genuine.add(key)
    return genuine, witness


def perturb(rng: random.Random, args: tuple) -> tuple:
    """Same argument shapes, off the sampler's distribution."""
    out = []
    for arg in args:
        if isinstance(arg, list):
            values = [rng.choice(WIDE) for _ in range(rng.randint(0, 10))]
            # A no-op on iid values, kept so the perturbed stream (and every
            # recorded perturbed tally) reproduces exactly.
            rng.shuffle(values)
            out.append(values)
        elif isinstance(arg, bool):
            out.append(rng.choice((True, False)))
        elif isinstance(arg, int):
            out.append(rng.choice(WIDE))
        else:
            out.append(arg)
    return tuple(out)


def densify(rng: random.Random, args: tuple) -> tuple:
    """Same shapes, every integer and list value from -2..2: duplicates and ties."""
    out = []
    for arg in args:
        if isinstance(arg, list):
            out.append([rng.choice(DENSE) for _ in range(rng.randint(0, 10))])
        elif isinstance(arg, bool):
            out.append(arg)
        elif isinstance(arg, int):
            out.append(rng.choice(DENSE))
        else:
            out.append(arg)
    return tuple(out)


def _computes(program: tuple[str, ...], args: tuple, want: tuple) -> bool:
    outcome = run_program(program, to_stack(args))
    if isinstance(outcome, str):
        return False
    stack, _fuel = outcome
    return len(stack) == 1 and value_key(stack[0]) == want


def agrees(program: tuple[str, ...], behaviour, args: tuple) -> bool | None:
    """True/False on a draw inside the behaviour's operating domain, else None.

    The domain is where the corpus's own verified witness computes the
    reference. A perturbed draw can leave it (an empty list for `drop_last`,
    where the catalogue witness's `uncons` refuses too), and a found program
    that fails there is not a shortcut, just as partial as the label it was
    trained on.
    """
    try:
        want = value_key(to_value(behaviour.reference(*args)))
    except Exception:
        return None
    if not _computes(behaviour.program, args, want):
        return None
    return _computes(program, args, want)


def verdict_of(tallies: dict[str, dict[str, Any]], floor: int) -> str:
    if any(t["failed"] for t in tallies.values()):
        return "shortcut"
    if all(t["tested"] >= floor for t in tallies.values()):
        return "genuine"
    return "inconclusive"


def check(
    program: tuple[str, ...],
    behaviour,
    draws: int,
    seed: int,
    dense: bool = True,
    min_tested: int = MIN_TESTED,
) -> dict[str, Any]:
    if program in behaviour.programs:
        return {
            "verdict": "equals_catalogue_witness",
            "why": "equals a catalogue witness",
        }
    # sampler and perturbed share one stream, as when these verdicts were first
    # recorded; dense has its own, so adding it moves neither.
    shared = random.Random(seed)
    streams = {"sampler": shared, "perturbed": shared}
    if dense:
        streams["dense"] = random.Random(f"{seed}:dense")
    tallies: dict[str, dict[str, Any]] = {}
    for mode, rng in streams.items():
        tested = failed = 0
        counterexample = None
        for _ in range(draws):
            args = behaviour.sample(rng)
            if mode == "perturbed":
                args = perturb(rng, args)
            elif mode == "dense":
                args = densify(rng, args)
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
            # Out of the corpus's domain: neither for nor against the program.
            "skipped": draws - tested,
            "counterexample": counterexample,
        }
    # The floor is the floor. Fewer draws than it cannot reach `genuine`.
    return {"verdict": verdict_of(tallies, min_tested), "draws": tallies}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=134)
    parser.add_argument(
        "--dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dense draws (-2..2) catch the tie and duplicate shortcuts; on by default",
    )
    parser.add_argument("--min-tested", type=int, default=MIN_TESTED)
    parser.add_argument("--eval-corpus", default="stack_pbe_v4")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=BEAM_WIDTH,
        help="rescore only timing cells searched at this beam width",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help="rescore only timing cells searched to this depth",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budget = (args.beam_width, args.max_depth)

    by_name = {b.name: b for b in CATALOGUE}
    cells = []
    skipped_cells = []
    for run_dir in args.run_dir:
        for path in sorted((run_dir / "timing").glob("*.json")):
            cell = json.loads(path.read_text())
            if cell.get("eval_corpus", args.eval_corpus) != args.eval_corpus:
                continue
            if (cell.get("beam_width"), cell.get("max_depth")) != budget:
                skipped_cells.append(str(path))
                continue
            cells.append((run_dir, cell))
    for path in skipped_cells:
        print(f"skipping {path}: not searched at beam {budget[0]} x depth {budget[1]}")

    distinct: dict[tuple[str, str], dict[str, Any]] = {}
    for _, cell in cells:
        for row in cell["rows"]:
            if row["solved"]:
                distinct.setdefault((row["behaviour"], " ".join(row["program"])), {})
    for (name, program), slot in sorted(distinct.items()):
        slot.update(
            check(
                tuple(program.split()),
                by_name[name],
                args.draws,
                args.seed,
                args.dense,
                args.min_tested,
            )
        )
        extra = ""
        if "draws" in slot:
            extra = " " + ", ".join(
                f"{mode} {t['failed']}/{t['tested']} failed, {t['skipped']} skipped"
                for mode, t in slot["draws"].items()
            )
            example = next(
                (
                    t["counterexample"]
                    for t in slot["draws"].values()
                    if t["counterexample"]
                ),
                None,
            )
            if example:
                extra += f" e.g. {example}"
        print(f"{slot['verdict']:24s} {name:28s} {len(program.split()):3d} ops{extra}")

    table = []
    for run_dir, cell in cells:
        counts = {
            k: dict.fromkeys(BUCKETS, 0)
            for k in ("graded", "verified", "witness", "inconclusive")
        }
        for row in cell["rows"]:
            if not row["solved"]:
                continue
            counts["graded"][row["bucket"]] += 1
            verdict = distinct[(row["behaviour"], " ".join(row["program"]))]["verdict"]
            column = COLUMN_OF.get(verdict)
            if column:
                counts[column][row["bucket"]] += 1
        epoch = cell.get("policy_epoch")
        totals = {f"{k}_total": sum(v.values()) for k, v in counts.items()}
        table.append(
            {
                "run_dir": str(run_dir),
                "arm": run_dir.name,
                "seed": cell["seed"],
                "policy_epoch": epoch,
                **counts,
                **totals,
                "tasks": len(cell["rows"]),
            }
        )
        label = f"{run_dir.name} s{cell['seed']}" + (
            "" if epoch is None else f"-e{epoch}"
        )
        print(
            f"{label:14s}: graded {totals['graded_total']}, verified "
            f"{totals['verified_total']}, witness {totals['witness_total']}, "
            f"inconclusive {totals['inconclusive_total']} of {len(cell['rows'])} | "
            + " ".join(
                f"{b} {counts['verified'][b]}/{counts['graded'][b]}" for b in BUCKETS
            )
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "draws_per_mode": args.draws,
                    "seed": args.seed,
                    "modes": [
                        "sampler",
                        "perturbed",
                        *(["dense"] if args.dense else []),
                    ],
                    "min_tested": args.min_tested,
                    "beam_width": args.beam_width,
                    "max_depth": args.max_depth,
                    "eval_corpus": args.eval_corpus,
                    "catalogue_digest": catalogue_digest(),
                    "catalogue_source_digest": catalogue_source_digest(),
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
