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


def cells(run_dir: Path, arm: str, corpus: str | None) -> dict[int, dict[str, Any]]:
    """Timing cells for one arm and one eval corpus, keyed by seed."""
    out = {}
    for path in sorted((run_dir / "timing").glob(f"{arm}-s*.json")):
        cell = json.loads(path.read_text())
        name = cell.get("eval_corpus")
        if corpus is not None and name != corpus:
            continue
        out[cell["seed"]] = cell
    return out


def spread(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def bucket_rates(cell: dict[str, Any]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for bucket in BUCKETS:
        rows = [r for r in cell["rows"] if r["bucket"] == bucket]
        out[bucket] = (sum(r["solved"] for r in rows), len(rows))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--arm", default="1m")
    parser.add_argument("--ce", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    arms = {"v4-only": args.baseline, "v4+world": args.mixed}
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

    for corpus, title in (
        ("stack_pbe_v4", "v4 eval cohort (300 tasks, beam 64 x depth 48)"),
        ("stack_pbe_world_v3", "world eval (55 tasks, same search)"),
    ):
        per_arm = {name: cells(path, args.arm, corpus) for name, path in arms.items()}
        if not any(per_arm.values()):
            continue
        lines += [f"## Solve rate, {title}", ""]
        header = "| bucket | " + " | ".join(
            f"{name} s{seed}" for name in arms for seed in sorted(per_arm[name])
        )
        lines += [
            header + " |",
            "|---" * (1 + sum(len(v) for v in per_arm.values())) + "|",
        ]
        rates = {
            name: {seed: bucket_rates(cell) for seed, cell in seeded.items()}
            for name, seeded in per_arm.items()
        }
        for bucket in BUCKETS:
            row = [bucket]
            for name in arms:
                for seed in sorted(per_arm[name]):
                    hit, total = rates[name][seed][bucket]
                    row.append(f"{hit}/{total}" if total else "-")
            lines.append("| " + " | ".join(row) + " |")
        overall = ["**all**"]
        for name in arms:
            for seed in sorted(per_arm[name]):
                cell = per_arm[name][seed]
                hit = sum(r["solved"] for r in cell["rows"])
                overall.append(f"{hit}/{len(cell['rows'])}")
        lines += ["| " + " | ".join(overall) + " |", ""]
        report.setdefault("solve", {})[corpus] = {
            name: {
                seed: {
                    "buckets": {b: list(rates[name][seed][b]) for b in BUCKETS},
                    "solved": sum(r["solved"] for r in cell["rows"]),
                    "shown_fit": sum(r["shown_fit"] for r in cell["rows"]),
                    "tasks": len(cell["rows"]),
                    "coverage": sum(r["solved"] for r in cell["rows"])
                    / len(cell["rows"]),
                    "cost_covered_seconds": sum(
                        r["seconds"] for r in cell["rows"] if r["solved"]
                    ),
                    "total_seconds": sum(r["seconds"] for r in cell["rows"]),
                    "beam_width": cell["beam_width"],
                    "max_depth": cell["max_depth"],
                }
                for seed, cell in per_arm[name].items()
            }
            for name in arms
        }

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
            row = [short] + [f"{scored['cross_entropy'][n]:.4f}" for n in names]
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
