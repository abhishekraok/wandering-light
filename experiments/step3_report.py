"""Pair the two step-3 arms into one report (#134).

`aggregate` in the #101 driver pairs over seeds within one corpus and its arm
axis is model size, so the corpus axis does not fit it. This reads both run
directories instead and emits the paired table: validation cross-entropy per
seed on the pinned v4 holdout, per-bucket solve rate on the shared 300-task v4
cohort, the cost-to-clear triple, and whatever `step3_ce.py` recorded.

Nothing here recomputes a metric the artifacts already hold; a number that is
not in a trainer_state, a timing cell or ce.json does not appear.

Usage:
    uv run python experiments/step3_report.py \
        --baseline checkpoints/step3-20260912/v4only \
        --mixed checkpoints/step3-20260912/mixed \
        --ce reports/world-plus-v4-policy-20260912/ce.json \
        --out reports/world-plus-v4-policy-20260912
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41+")


def trainer_states(run_dir: Path, arm: str) -> dict[int, dict[str, Any]]:
    out = {}
    for path in sorted((run_dir / "policies").glob(f"{arm}-s*/trainer_state.json")):
        state = json.loads(path.read_text())
        out[state["seed"]] = state
    return out


PRIMARY_BUDGET = (64, 48)  # (beam width, max depth) of every #101 cell


def cells(
    run_dir: Path,
    arm: str,
    corpus: str | None,
    budget: tuple[int, int] = PRIMARY_BUDGET,
    policy_epoch: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Timing cells for one arm, one eval corpus and one search budget, by seed.

    Keyed by seed alone, so the budget filter is what stops a structural cell
    (same corpus and seed, deeper search) from silently replacing the primary
    cell it is meant to be read beside.
    """
    out = {}
    for path in sorted((run_dir / "timing").glob(f"{arm}-s*.json")):
        cell = json.loads(path.read_text())
        if corpus is not None and cell.get("eval_corpus") != corpus:
            continue
        if (cell.get("beam_width"), cell.get("max_depth")) != budget:
            continue
        if cell.get("policy_epoch") != policy_epoch:
            continue
        if cell["seed"] in out:
            raise ValueError(f"{run_dir}: two {corpus} cells for seed {cell['seed']}")
        out[cell["seed"]] = cell
    return out


def spread(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def scored_rows(cell: dict[str, Any], excluded: frozenset[str]) -> list[dict[str, Any]]:
    """The rows a result is read on: every row except excluded behaviours."""
    return [r for r in cell["rows"] if r["behaviour"] not in excluded]


def bucket_rates(
    cell: dict[str, Any], excluded: frozenset[str] = frozenset()
) -> dict[str, tuple[int, int]]:
    rows_all = scored_rows(cell, excluded)
    out: dict[str, tuple[int, int]] = {}
    for bucket in BUCKETS:
        rows = [r for r in rows_all if r["bucket"] == bucket]
        out[bucket] = (sum(r["solved"] for r in rows), len(rows))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--arm", default="1m")
    parser.add_argument("--ce", type=Path)
    parser.add_argument(
        "--verified",
        type=Path,
        help="verified_solves.json from step3_verify_solves.py; leads the v4 solve table",
    )
    parser.add_argument(
        "--world-exclude",
        type=Path,
        help="world eval behaviours already seen in training; scored separately",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    arms = {"v4-only": args.baseline, "v4+world": args.mixed}
    world_excluded: frozenset[str] = frozenset()
    if args.world_exclude:
        world_excluded = frozenset(
            line.strip()
            for line in args.world_exclude.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    states = {name: trainer_states(path, args.arm) for name, path in arms.items()}
    seeds = sorted(set(states["v4-only"]) & set(states["v4+world"]))
    report: dict[str, Any] = {"arm": args.arm, "seeds": seeds, "validation_ce": {}}

    lines = [
        "## Validation cross-entropy on the pinned v4 holdout "
        "(26,040 rows, the same rows in both arms)",
        "",
        "| seed | v4-only | best epoch | v4+world | best epoch | delta |",
        "|---|---|---|---|---|---|",
    ]
    deltas = []
    for seed in seeds:
        a, b = states["v4-only"][seed], states["v4+world"][seed]
        delta = b["best_validation_cross_entropy"] - a["best_validation_cross_entropy"]
        deltas.append(delta)
        report["validation_ce"][seed] = {
            "v4_only": a["best_validation_cross_entropy"],
            "v4_world": b["best_validation_cross_entropy"],
            "delta": delta,
            "best_epoch": {"v4_only": a["best_epoch"], "v4_world": b["best_epoch"]},
            "rows": {
                "v4_only": (a.get("train_rows"), a.get("validation_rows")),
                "v4_world": (b.get("train_rows"), b.get("validation_rows")),
            },
        }
        lines.append(
            f"| {seed} | {a['best_validation_cross_entropy']:.4f} | {a['best_epoch']} "
            f"| {b['best_validation_cross_entropy']:.4f} | {b['best_epoch']} "
            f"| {delta:+.4f} |"
        )
    mean_a = [states["v4-only"][s]["best_validation_cross_entropy"] for s in seeds]
    mean_b = [states["v4+world"][s]["best_validation_cross_entropy"] for s in seeds]
    lines += [
        f"| **mean ± sd** | {spread(mean_a)} | | {spread(mean_b)} | | "
        f"{spread(deltas)} |",
        "",
        "A delta is only worth reading against the seed spread in the same row.",
        "",
    ]
    report["validation_ce_summary"] = {
        "v4_only": mean_a,
        "v4_world": mean_b,
        "delta_mean": statistics.mean(deltas) if deltas else None,
        "delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
    }

    if args.verified and args.verified.exists():
        verified = json.loads(args.verified.read_text())
        report["verified_solves"] = verified
        arm_name = {args.baseline.name: "v4-only", args.mixed.name: "v4+world"}
        by_arm: dict[str, dict[int, dict[str, Any]]] = {}
        wanted = {
            str(args.baseline.resolve()): "v4-only",
            str(args.mixed.resolve()): "v4+world",
        }
        for cell in verified["cells"]:
            # One verification file can cover several run dirs and epoch
            # snapshots; the paired table is the best checkpoints of these two.
            if cell.get("policy_epoch") is not None:
                continue
            if "run_dir" in cell:
                name = wanted.get(str(Path(cell["run_dir"]).resolve()))
                if name is None:
                    continue
            else:
                name = arm_name.get(cell["arm"], cell["arm"])
            by_arm.setdefault(name, {})[cell["seed"]] = cell
        dense_note = (
            f" and {verified['draws_per_mode']:,} dense draws"
            if "dense" in verified.get("modes", [])
            else ""
        )
        columns = [
            (name, seed) for name in arms for seed in sorted(by_arm.get(name, {}))
        ]
        lines += [
            "## Verified solves, v4 eval cohort (300 tasks, beam 64 x depth 48)",
            "",
            "A graded solve passes the task's held-out examples. A verified solve also "
            "computes the behaviour: its program is a catalogue witness, or it agrees "
            f"with the Python reference on {verified['draws_per_mode']:,} sampler draws "
            f"and {verified['draws_per_mode']:,} perturbed draws{dense_note}, each counted only where the catalogue witness computes the reference. Cells read verified / graded.",
            "",
            "| bucket | " + " | ".join(f"{n} s{s}" for n, s in columns) + " |",
            "|---" * (1 + len(columns)) + "|",
        ]
        for bucket in BUCKETS:
            row = [bucket]
            for name, seed in columns:
                cell = by_arm[name][seed]
                row.append(f"{cell['verified'][bucket]}/{cell['graded'][bucket]}")
            lines.append("| " + " | ".join(row) + " |")
        total = ["**all**"]
        for name, seed in columns:
            cell = by_arm[name][seed]
            total.append(f"{cell['verified_total']}/{cell['graded_total']}")
        lines += ["| " + " | ".join(total) + " |", ""]
        # The file may also judge programs only an epoch snapshot solved; list
        # the ones these best-checkpoint cells solved.
        best_solved = {
            (row["behaviour"], " ".join(row["program"]))
            for path in arms.values()
            for cell in cells(path, args.arm, "stack_pbe_v4").values()
            for row in cell["rows"]
            if row["solved"]
        }
        in_table = [
            p
            for p in verified["programs"]
            if (p["behaviour"], p["program"]) in best_solved
        ]
        shortcuts = [p for p in in_table if p["verdict"] == "shortcut"]
        undecided = [p for p in in_table if p["verdict"] == "inconclusive"]
        if undecided:
            lines += [
                "Programs the verification could not decide (too few in-domain draws "
                "in a mode), counted as neither verified nor shortcut:",
                "",
                *(
                    f"- `{p['behaviour']}`, {len(p['program'].split())} ops"
                    for p in undecided
                ),
                "",
            ]
        if shortcuts:
            lines += [
                "Programs that pass held-out grading without computing the behaviour:",
                "",
            ]
            for item in shortcuts:
                draws = item.get("draws", {})
                modes = [m for m in ("sampler", "perturbed", "dense") if m in draws]
                example = next(
                    (
                        draws[m]["counterexample"]
                        for m in modes
                        if draws[m].get("counterexample")
                    ),
                    None,
                )
                fails = ", ".join(
                    f"{draws[m]['failed']}/{draws[m]['tested']} {m}" for m in modes
                )
                lines.append(
                    f"- `{item['behaviour']}`, {len(item['program'].split())} ops: "
                    f"fails {fails} draws; e.g. {example}"
                )
            lines.append("")

    for corpus, title in (
        ("stack_pbe_v4", "v4 eval cohort (300 tasks, beam 64 x depth 48)"),
        ("stack_pbe_world_v3", "world eval (55 tasks, same search)"),
    ):
        per_arm = {name: cells(path, args.arm, corpus) for name, path in arms.items()}
        if not any(per_arm.values()):
            continue
        excluded = world_excluded if corpus == "stack_pbe_world_v3" else frozenset()
        lines += [f"## Solve rate, {title}", ""]
        if excluded:
            lines += [
                f"Scored on the {len(next(iter(next(iter(per_arm.values())).values()))['rows']) - len(excluded)} "
                f"clean behaviours; the {len(excluded)} whose program or skeleton "
                "already appears in a training set are counted in the last row only.",
                "",
            ]
        header = "| bucket | " + " | ".join(
            f"{name} s{seed}" for name in arms for seed in sorted(per_arm[name])
        )
        lines += [
            header + " |",
            "|---" * (1 + sum(len(v) for v in per_arm.values())) + "|",
        ]
        rates = {
            name: {seed: bucket_rates(cell, excluded) for seed, cell in seeded.items()}
            for name, seeded in per_arm.items()
        }
        for bucket in BUCKETS:
            row = [bucket]
            for name in arms:
                for seed in sorted(per_arm[name]):
                    hit, total = rates[name][seed][bucket]
                    row.append(f"{hit}/{total}" if total else "-")
            lines.append("| " + " | ".join(row) + " |")
        overall = ["**all scored**"]
        for name in arms:
            for seed in sorted(per_arm[name]):
                rows = scored_rows(per_arm[name][seed], excluded)
                overall.append(f"{sum(r['solved'] for r in rows)}/{len(rows)}")
        lines.append("| " + " | ".join(overall) + " |")
        if excluded:
            dirty = ["excluded (seen in training)"]
            for name in arms:
                for seed in sorted(per_arm[name]):
                    rows = [
                        r
                        for r in per_arm[name][seed]["rows"]
                        if r["behaviour"] in excluded
                    ]
                    dirty.append(f"{sum(r['solved'] for r in rows)}/{len(rows)}")
            lines.append("| " + " | ".join(dirty) + " |")
        lines.append("")
        report.setdefault("solve", {})[corpus] = {
            name: {
                seed: {
                    "buckets": {b: list(rates[name][seed][b]) for b in BUCKETS},
                    "solved": sum(r["solved"] for r in scored_rows(cell, excluded)),
                    "shown_fit": sum(
                        r["shown_fit"] for r in scored_rows(cell, excluded)
                    ),
                    "tasks": len(scored_rows(cell, excluded)),
                    "coverage": sum(r["solved"] for r in scored_rows(cell, excluded))
                    / max(1, len(scored_rows(cell, excluded))),
                    "cost_covered_seconds": sum(
                        r["seconds"] for r in scored_rows(cell, excluded) if r["solved"]
                    ),
                    "total_seconds": sum(
                        r["seconds"] for r in scored_rows(cell, excluded)
                    ),
                    "excluded_behaviours": sorted(excluded),
                    "excluded_solved": sum(
                        r["solved"] for r in cell["rows"] if r["behaviour"] in excluded
                    ),
                    "beam_width": cell["beam_width"],
                    "max_depth": cell["max_depth"],
                }
                for seed, cell in per_arm[name].items()
            }
            for name in arms
        }

    structural = {
        name: cells(path, args.arm, "stack_pbe_world_v3", budget=(64, 192))
        for name, path in arms.items()
    }
    if any(structural.values()):
        primary = {
            name: cells(path, args.arm, "stack_pbe_world_v3")
            for name, path in arms.items()
        }
        lines += [
            "## Structural check: world eval at depth 192 (beam 64)",
            "",
            "Same weights and beam as the primary world cell, four times the depth. "
            "Clean behaviours only.",
            "",
            "| arm | seed | clean within depth 48 | solved at depth 48 | clean within depth 192 | solved at depth 192 | wall at 192 (s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in arms:
            for seed, deep in sorted(structural[name].items()):
                shallow = primary[name].get(seed)
                deep_rows = scored_rows(deep, world_excluded)
                row = [name, str(seed)]
                if shallow:
                    rows48 = scored_rows(shallow, world_excluded)
                    row += [
                        str(sum(r["length"] <= 48 for r in rows48)),
                        f"{sum(r['solved'] for r in rows48)}/{len(rows48)}",
                    ]
                else:
                    row += ["-", "-"]
                row += [
                    str(sum(r["length"] <= 192 for r in deep_rows)),
                    f"{sum(r['solved'] for r in deep_rows)}/{len(deep_rows)}",
                    f"{deep['cell_seconds']:.0f}",
                ]
                lines.append("| " + " | ".join(row) + " |")
                report.setdefault("structural", {}).setdefault(name, {})[seed] = {
                    "solved_depth_192": sum(r["solved"] for r in deep_rows),
                    "tasks": len(deep_rows),
                    "within_depth_192": sum(r["length"] <= 192 for r in deep_rows),
                    "solved_behaviours": sorted(
                        {r["behaviour"] for r in deep_rows if r["solved"]}
                    ),
                    "cell_seconds": deep["cell_seconds"],
                }
        lines.append("")

    if args.ce and args.ce.exists():
        ce = json.loads(args.ce.read_text())
        report["ce"] = ce
        names = sorted(next(iter(ce["checkpoints"].values()))["cross_entropy"])
        lines += [
            "## Cross-entropy on held-out row sets (experiments/step3_ce.py)",
            "",
            "| checkpoint | " + " | ".join(names) + " |",
            "|---" * (1 + len(names)) + "|",
        ]
        for checkpoint, scored in sorted(ce["checkpoints"].items()):
            short = "/".join(Path(checkpoint).parts[-3:])
            row = [short] + [
                (
                    f"{scored['cross_entropy'][n]:.4f}"
                    if scored["cross_entropy"][n] is not None
                    else "empty"
                )
                for n in names
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines += [
            "",
            "Row-set sizes: "
            + ", ".join(
                f"{n} {ce['row_sets'][n]['tasks']} tasks / {ce['row_sets'][n]['rows']} rows"
                for n in names
            ),
            "",
        ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired.md").write_text("\n".join(lines) + "\n")
    (args.out / "paired.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
