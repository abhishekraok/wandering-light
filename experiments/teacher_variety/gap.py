"""Train-eval gap for teacher-variety (#175): is a solved eval behaviour one
that some training program already computes?

For each of the cohort's 43 eval behaviours, every training program (the v4 fit
witnesses, fused and staged, and the Sol and Opus witnesses of the #174
behaviours) is run on that behaviour's cohort examples (all of them, shown and
held-out, deduplicated). **Agreement** is the fraction it gets right; **nearest** is the
best agreement over programs, reported for v4 programs alone and with the new
ones, beside the shortest cohort witness. A behaviour's **solve count** is how many of the 12 #174/#175 cells
(4 arms x 3 seeds) verified it.

    uv run python experiments/teacher_variety/gap.py \
        --run-dir checkpoints/teacher-variety-20260927 \
        --verdicts reports/teacher-variety-20260927/verified_solves.json \
        --out reports/teacher-variety-20260927/gap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

import experiments.compute_optimal_mark28 as co
import experiments.staged_arms as sa
from wandering_light.stack_pbe_eval import _matches
from wandering_light.stack_policy import validation_skeleton

PROGRAMS: list[tuple[str, tuple[str, ...]]] = []


def train_programs(tasks) -> list[tuple[str, tuple[str, ...]]]:
    fit = [t for t in tasks if t.split == "train" and not validation_skeleton(t)]
    side = sa.treated(False)
    v4 = {" ".join(t.program) for t in fit}
    v4 |= {side[t.behaviour]["program"] for t in fit if t.behaviour in side}
    new = set()
    for path in (sa.NEW_TASKS_FILE, sa.OPUS_TASKS_FILE):
        new |= {" ".join(t.program) for t in sa.new_tasks(path)}
    out = [("v4", tuple(p.split())) for p in sorted(v4)]
    return out + [("new", tuple(p.split())) for p in sorted(new - v4)]


def examples(tasks) -> dict[str, list[tuple[str, str]]]:
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for _, t in co.cohort(tasks):
        for ins, outs in (
            (t.inputs, t.outputs),
            (t.held_out_inputs, t.held_out_outputs),
        ):
            for given, wanted in zip(ins, outs, strict=True):
                pairs[t.behaviour].setdefault(given, wanted)
    return {b: sorted(p.items()) for b, p in pairs.items()}


def _init(programs):
    PROGRAMS[:] = programs


def nearest(item: tuple[str, list[tuple[str, str]]]) -> dict:
    behaviour, pairs = item
    best = {"v4": (0.0, ""), "new": (0.0, "")}
    for source, program in PROGRAMS:
        right = sum(bool(_matches(program, g, w)) for g, w in pairs)
        score = right / len(pairs)
        if score > best[source][0]:
            best[source] = (score, " ".join(program))
    return {
        "behaviour": behaviour,
        "pairs": len(pairs),
        "nearest_v4": best["v4"][0],
        "nearest_v4_program": best["v4"][1],
        "nearest_new": best["new"][0],
        "nearest_new_program": best["new"][1],
        "nearest_all": max(best["v4"][0], best["new"][0]),
    }


def solve_counts(run_dir: Path, verdicts_path: Path) -> dict[str, int]:
    verdicts = {
        (r["behaviour"], r["program"]): r["verdict"]
        for r in json.loads(verdicts_path.read_text())["programs"]
    }
    counts: dict[str, int] = defaultdict(int)
    for path in sorted((run_dir / "timing").glob("*.json")):
        cell = json.loads(path.read_text())
        solved = {
            r["behaviour"]
            for r in cell["rows"]
            if r["solved"]
            and verdicts.get((r["behaviour"], " ".join(r["program"] or ())))
            in sa.VERIFIED
        }
        for b in solved:
            counts[b] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    tasks = co.corpus()
    programs = train_programs(tasks)
    pairs = examples(tasks)
    counts = solve_counts(args.run_dir, args.verdicts)
    print(f"{len(programs)} train programs, {len(pairs)} eval behaviours", flush=True)
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(programs,)
    ) as pool:
        rows = list(pool.map(nearest, sorted(pairs.items())))
    length: dict[str, int] = {}
    for _, t in co.cohort(tasks):
        length[t.behaviour] = min(length.get(t.behaviour, t.length), t.length)
    for row in rows:
        row["solve_count"] = counts.get(row["behaviour"], 0)
        row["witness_length"] = length[row["behaviour"]]
    rows.sort(key=lambda r: (-r["nearest_all"], r["behaviour"]))
    args.out.write_text(json.dumps(rows, indent=1))
    print("| behaviour | nearest v4 | nearest new | witness ops | solved cells /12 |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['behaviour']} | {r['nearest_v4']:.2f} | {r['nearest_new']:.2f} "
            f"| {r['witness_length']} | {r['solve_count']} |"
        )
    for name, group in (
        ("solved (>=6 of 12)", [r for r in rows if r["solve_count"] >= 6]),
        ("never solved", [r for r in rows if r["solve_count"] == 0]),
    ):
        ops = sorted(r["witness_length"] for r in group)
        print(
            f"{name}: n={len(group)} nearest v4 {mean(r['nearest_v4'] for r in group):.2f} "
            f"nearest all {mean(r['nearest_all'] for r in group):.2f} "
            f"witness ops median {median(ops)} range {ops[0]}-{ops[-1]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
