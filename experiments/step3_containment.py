"""Verified solves split by whether world fit programs contain the eval witness (world-plus-v4-policy, #134).

A v4 cohort behaviour is *contained* when one of its eval witnesses occurs as a
contiguous run of whole ops inside at least one world-v3 fit program, and none of
its witnesses occurs inside any v4 fit program. Fit programs are rebuilt the way
the driver builds fit rows: train split, minus the drop list, minus validation
skeletons. Solves are read at matched epochs and at each seed's peak epoch, so
the split is not confounded with the epoch best-validation selection picked.

Usage:
    uv run python experiments/step3_containment.py \
        --baseline checkpoints/step3-20260913-epochs/v4only \
        --mixed checkpoints/step3-20260913-epochs/mixed \
        --baseline-best checkpoints/step3-20260912/v4only \
        --mixed-best checkpoints/step3-20260912/mixed \
        --verified reports/world-plus-v4-policy-20260912/verified_solves_epochs.json \
        --drop reports/world-plus-v4-policy-20260912/drop_from_world_train.txt \
        --out reports/world-plus-v4-policy-20260912
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import compute_optimal_mark28 as driver  # noqa: E402
from experiments.step3_epochs_report import find_cell  # noqa: E402
from experiments.step3_verify_solves import partition  # noqa: E402
from wandering_light.stack_policy import validation_skeleton  # noqa: E402

V4 = REPO / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
W3 = REPO / "wandering_light/evals/data/stack_pbe_world_v3/manifest.json"


def span_in(span: str, program: str) -> bool:
    """Whether ``span`` occurs in ``program`` as whole space-separated ops."""
    return f" {span} " in f" {program} "


def containment(
    witnesses: dict[str, list[str]],
    v4_fit: dict[str, str],
    world_fit: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Per behaviour: witness lengths and how many fit programs contain a witness."""
    return {
        b: {
            "witness_ops": sorted({len(w.split()) for w in ws}),
            "world_fit": sum(
                any(span_in(w, p) for w in ws) for p in world_fit.values()
            ),
            "v4_fit": sum(any(span_in(w, p) for w in ws) for p in v4_fit.values()),
        }
        for b, ws in sorted(witnesses.items())
    }


def contained(table: dict[str, dict[str, Any]]) -> set[str]:
    return {b for b, c in table.items() if c["world_fit"] and not c["v4_fit"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--baseline-best", type=Path)
    parser.add_argument("--mixed-best", type=Path)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--drop", type=Path, required=True)
    parser.add_argument("--arm", default="1m")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    driver.CORPUS_MANIFESTS = [V4, W3]
    tasks = driver.corpus()
    corpus_of = driver.CORPUS_OF
    drop = {
        line.strip()
        for line in args.drop.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    def fit(corpus: str) -> dict[str, str]:
        return {
            t.behaviour: " ".join(t.program)
            for t in tasks
            if t.split == "train"
            and corpus_of[t.behaviour] == corpus
            and t.behaviour not in drop
            and not validation_skeleton(t)
        }

    v4_fit, world_fit = fit("stack_pbe_v4"), fit("stack_pbe_world_v3")

    verified = json.loads(args.verified.read_text())
    # Draw-tested or a catalogue witness, as the recorded reports define a
    # verified solve. `keep_below`'s solves are all the latter (#142).
    draw_tested, witness = partition(verified["programs"])
    genuine = draw_tested | witness
    arms = {
        "v4-only": (args.baseline, args.baseline_best),
        "v4+world": (args.mixed, args.mixed_best),
    }
    grid: dict[tuple[str, int, int], collections.Counter[str]] = {}
    cohort_tasks: set[int] = set()
    for name, (run_dir, best_dir) in arms.items():
        for seed in args.seeds:
            for epoch in args.epochs:
                cell, _ = find_cell(run_dir, best_dir, args.arm, seed, epoch)
                if cell is None:
                    continue
                solves: collections.Counter[str] = collections.Counter()
                for r in cell["rows"]:
                    cohort_tasks.add(r["task_index"])
                    key = (r["behaviour"], " ".join(r["program"] or ()))
                    if r["solved"] and key in genuine:
                        solves[r["behaviour"]] += 1
                grid[(name, seed, epoch)] = solves

    v4_tasks = [t for t in tasks if corpus_of[t.behaviour] == "stack_pbe_v4"]
    witnesses: dict[str, list[str]] = collections.defaultdict(list)
    for index in sorted(cohort_tasks):
        task = v4_tasks[index]
        program = " ".join(task.program)
        if program not in witnesses[task.behaviour]:
            witnesses[task.behaviour].append(program)
    table = containment(witnesses, v4_fit, world_fit)
    inside = contained(table)
    cohort = set(table)

    def split(cells: list[collections.Counter[str]]) -> tuple[int, int]:
        total = sum(cells, collections.Counter())
        return (
            sum(total[b] for b in inside),
            sum(total[b] for b in cohort - inside),
        )

    lines = [
        "## Verified solves by witness containment (v4 cohort, beam 64 x depth 48)",
        "",
        "Contained: a behaviour's eval witness occurs, as whole ops, inside at least "
        "one world-v3 fit program and inside no v4 fit program (fit = train split, "
        "minus the drop list, minus validation skeletons). "
        f"{len(inside)} of the cohort's {len(cohort)} behaviours are contained; "
        f"fit programs: v4 {len(v4_fit)}, world {len(world_fit)}.",
        "",
        "| contained behaviour | witness ops | world fit programs containing it |",
        "|---|---|---|",
        *(
            f"| `{b}` | {', '.join(map(str, table[b]['witness_ops']))} | "
            f"{table[b]['world_fit']} |"
            for b in sorted(inside)
        ),
        "",
        "Solves summed over seeds; the last column is the other behaviours' "
        "paired difference per seed.",
        "",
        "| epoch | contained v4-only | contained v4+world | difference | other v4-only "
        "| other v4+world | difference | other, per seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_epoch: dict[str, Any] = {}

    def row(label: str, picks: dict[tuple[str, int], int]) -> None:
        a = [grid[("v4-only", s, picks[("v4-only", s)])] for s in args.seeds]
        b = [grid[("v4+world", s, picks[("v4+world", s)])] for s in args.seeds]
        (ca, oa), (cb, ob) = split(a), split(b)
        per_seed = [split([y])[1] - split([x])[1] for x, y in zip(a, b, strict=True)]
        by_epoch[label] = {
            "contained": [ca, cb],
            "other": [oa, ob],
            "other_per_seed": per_seed,
            "epochs": {f"{n} s{s}": e for (n, s), e in picks.items()},
        }
        lines.append(
            f"| {label} | {ca} | {cb} | {cb - ca:+d} | {oa} | {ob} | {ob - oa:+d} | "
            + ", ".join(f"{d:+d}" for d in per_seed)
            + f" (mean {statistics.mean(per_seed):+.1f}) |"
        )

    for epoch in args.epochs:
        picks = {(n, s): epoch for n in arms for s in args.seeds}
        if all((n, s, e) in grid for (n, s), e in picks.items()):
            row(str(epoch), picks)
    peaks = {}
    for n in arms:
        for s in args.seeds:
            searched = [e for e in args.epochs if (n, s, e) in grid]
            if searched:
                peaks[(n, s)] = max(
                    searched, key=lambda e: (sum(grid[(n, s, e)].values()), -e)
                )
    if len(peaks) == len(arms) * len(args.seeds):
        row("each seed's peak", peaks)
    lines.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "containment.md").write_text("\n".join(lines) + "\n")
    (args.out / "containment.json").write_text(
        json.dumps(
            {
                "definition": "witness inside >=1 world fit program and no v4 fit program",
                "fit_programs": {"v4": len(v4_fit), "world": len(world_fit)},
                "behaviours": table,
                "contained": sorted(inside),
                "solves": by_epoch,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
