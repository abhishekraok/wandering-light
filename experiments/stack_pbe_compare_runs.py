"""Pair two saved `stack_pbe_baseline.py` runs task by task and regrade them.

No inference. The two JSONs already hold one candidate per task; this script
joins them to the corpus tasks they were scored on, re-runs the grader per
example (the saved files carry only a verdict per task), and records what the
machine did on every example -- the stack after each op, or the op that
refused and why. That is what a comparison needs and a summary cannot show:
`uncons` on an empty list is a different failure from a program that runs and
returns the wrong list.

Alignment is checked, not assumed. A run written before `task_index` was
recorded is joined by re-drawing the sample the way the baseline script draws
it (split, seed, `--tasks`), and every field the run *does* carry --
behaviour sequence, `task_index`, `witness` -- must agree with the corpus or
the script stops. Metadata a run lacks is reported as missing, never filled in.

    uv run python experiments/stack_pbe_compare_runs.py \\
        --a reports/stack-pbe-v4-baseline-20260901/luna_v4_sample40.json --label-a luna \\
        --b reports/stack-pbe-v4-codex-exec-20260910/astra_high_v4_sample40.json --label-b astra \\
        --out reports/<dir>/paired.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    format_value,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, Task, load
from wandering_light.stack_pbe_eval import grade

RUN_FIELDS = (
    "corpus",
    "model",
    "provider",
    "effort",
    "guard",
    "partial",
    "tasks_planned",
)
CANDIDATE_FIELDS = ("task_index", "witness", "generation", "errors")
VERDICTS = ("solved", "memorised", "failed")


def draw_sample(tasks: list[Task], split: str, seed: int, count: int) -> list[Task]:
    """The selection `stack_pbe_baseline.py` makes, so an index-less run can be joined."""
    chosen = [t for t in tasks if t.split == split] if split != "all" else list(tasks)
    rng = random.Random(seed)
    if count:
        rng.shuffle(chosen)
        chosen = chosen[:count]
    return chosen


def verdict_of(score) -> str:
    return "solved" if score.solved else "memorised" if score.memorised else "failed"


def trace(program: tuple[str, ...], given: str) -> dict:
    """Stack after every op, mirroring `stack_pbe.run_program` step for step."""
    stack = parse_stack(given, basis=BASIS)
    steps = [[format_value(v) for v in stack]]
    fuel = 0
    for position, name in enumerate(program):
        op = OP_BY_NAME.get(name)
        if op is None:
            return {
                "steps": steps,
                "refused_at": position,
                "reason": f"unknown op {name!r}",
            }
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - fuel, 0)
        if isinstance(outcome, str):
            return {"steps": steps, "refused_at": position, "reason": outcome}
        stack, spent = outcome
        fuel += spent
        steps.append([format_value(v) for v in stack])
    return {"steps": steps, "refused_at": None, "reason": None, "fuel": fuel}


def examples(program: tuple[str, ...], ins, outs) -> list[dict]:
    rows = []
    for given, wanted in zip(ins, outs, strict=True):
        t = trace(program, given)
        if t["refused_at"] is not None:
            rows.append(
                {
                    "input": given,
                    "expected": wanted,
                    "result": "illegal",
                    "got": None,
                    "trace": t,
                }
            )
            continue
        got = " ".join(t["steps"][-1])
        want = " ".join(format_value(v) for v in parse_stack(wanted, basis=BASIS))
        rows.append(
            {
                "input": given,
                "expected": wanted,
                "result": "pass" if got == want else "wrong",
                "got": got,
                "trace": t,
            }
        )
    return rows


def align(run: dict, sample: list[Task], index_of: dict[int, int]) -> dict:
    """Every field the run carries must agree with the corpus; absent ones are reported."""
    cands = run["candidates"]
    if len(cands) > len(sample):
        raise SystemExit(
            f"run has {len(cands)} candidates but the sample has {len(sample)}"
        )
    paired = sample[: len(cands)]
    checks = {
        "behaviour_sequence": [c["behaviour"] for c in cands]
        == [t.behaviour for t in paired],
    }
    for field, expected in (
        ("task_index", [index_of[id(t)] for t in paired]),
        ("witness", [" ".join(t.program) for t in paired]),
    ):
        if all(field in c for c in cands):
            checks[field] = [c[field] for c in cands] == expected
        else:
            checks[field] = "missing"
    if any(v is False for v in checks.values()):
        raise SystemExit(f"alignment failed: {checks}")
    return checks


def describe(run: dict) -> dict:
    cands = run["candidates"]
    return {
        "run_fields": {f: run.get(f, "missing") for f in RUN_FIELDS},
        "candidate_fields": {f: sum(f in c for c in cands) for f in CANDIDATE_FIELDS},
        "candidates": len(cands),
        "saved_summary": {
            k: run.get(k) for k in ("solved", "memorised", "shown_fit", "illegal")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v4")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--tasks", type=int, default=None, help="default: the run's own count"
    )
    parser.add_argument(
        "--out", default=None, help="paired JSON with per-example verdicts and traces"
    )
    args = parser.parse_args(argv)

    runs = {
        args.label_a: json.loads(Path(args.a).read_text()),
        args.label_b: json.loads(Path(args.b).read_text()),
    }
    paths = {args.label_a: args.a, args.label_b: args.b}
    all_tasks = load(args.corpus)
    index_of = {id(t): i for i, t in enumerate(all_tasks)}
    count = args.tasks or max(len(r["candidates"]) for r in runs.values())
    sample = draw_sample(all_tasks, args.split, args.seed, count)

    alignment = {label: align(run, sample, index_of) for label, run in runs.items()}
    metadata = {label: describe(run) for label, run in runs.items()}

    rows = []
    regrade_mismatch = Counter()
    for position, task in enumerate(sample):
        row = {
            "position": position,
            "task_index": index_of[id(task)],
            "behaviour": task.behaviour,
            "summary": task.summary,
            "witness": " ".join(task.program),
            "witness_length": task.length,
            "tags": list(task.tags),
            "shown": list(zip(task.inputs, task.outputs, strict=True)),
            "held_out": list(
                zip(task.held_out_inputs, task.held_out_outputs, strict=True)
            ),
            "runs": {},
        }
        for label, run in runs.items():
            if position >= len(run["candidates"]):
                row["runs"][label] = None
                continue
            saved = run["candidates"][position]
            score = grade(saved["candidate"], task)
            program = tuple(saved["candidate"].split())
            entry = {
                "candidate": saved["candidate"],
                "saved_verdict": saved["verdict"],
                "verdict": verdict_of(score),
                "parsed": score.parsed,
                "legal": score.legal,
                "shown_fit": score.shown,
                "held_out_fit": score.held_out,
                "shown": examples(program, task.inputs, task.outputs)
                if score.parsed
                else [],
                "held_out": examples(
                    program, task.held_out_inputs, task.held_out_outputs
                )
                if score.parsed
                else [],
                **{f: saved[f] for f in ("generation", "errors") if f in saved},
            }
            if entry["verdict"] != saved["verdict"]:
                regrade_mismatch[label] += 1
            row["runs"][label] = entry
        rows.append(row)

    a, b = args.label_a, args.label_b
    table = Counter(
        (r["runs"][a]["verdict"], r["runs"][b]["verdict"])
        for r in rows
        if r["runs"][a] and r["runs"][b]
    )
    differing = [
        r
        for r in rows
        if r["runs"][a]
        and r["runs"][b]
        and r["runs"][a]["verdict"] != r["runs"][b]["verdict"]
    ]

    print(
        f"{a}: {paths[a]}\n{b}: {paths[b]}\ncorpus: {args.corpus}, split {args.split}, seed {args.seed}, {len(sample)} tasks\n"
    )
    for label in runs:
        print(
            f"{label}: alignment {alignment[label]}; regrade disagrees with saved verdict on {regrade_mismatch[label]}"
        )
        missing = [
            f for f, v in metadata[label]["run_fields"].items() if v == "missing"
        ]
        absent = [f for f, n in metadata[label]["candidate_fields"].items() if n == 0]
        print(
            f"   missing run fields: {missing or 'none'}; candidate fields absent: {absent or 'none'}"
        )
    print(
        f"\n{'':>10}"
        + "".join(f"{v:>11}" for v in VERDICTS)
        + f"   (rows {a}, cols {b})"
    )
    for p in VERDICTS:
        print(f"{p:>10}" + "".join(f"{table[(p, q)]:>11}" for q in VERDICTS))
    print(f"\ndiffer on {len(differing)} of {len(rows)} tasks:")
    for r in differing:
        ra, rb = r["runs"][a], r["runs"][b]
        print(
            f"  #{r['position']:>2} idx {r['task_index']:>4} {r['behaviour']:<32} wit {r['witness_length']:>3}  {a}: {ra['verdict']:<9} {b}: {rb['verdict']}"
        )

    if args.out:
        payload = {
            "corpus": args.corpus,
            "split": args.split,
            "seed": args.seed,
            "runs": {
                label: {
                    "path": paths[label],
                    **metadata[label],
                    "alignment": alignment[label],
                    "regrade_mismatches": regrade_mismatch[label],
                }
                for label in runs
            },
            "contingency": {
                f"{p}|{q}": table[(p, q)] for p in VERDICTS for q in VERDICTS
            },
            "tasks": rows,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
