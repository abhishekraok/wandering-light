"""Solves at matched epochs for the two step-3 arms (world-plus-v4-policy, #134).

The best-validation checkpoint confounds a data effect with training length:
every v4-only run stops at epoch 1 or 2 on validation CE, and solve counts track
the selected epoch. This reads an epoch-resolved study, where both arms saved
weights at every epoch and were searched at chosen epochs, and lays solves out
by epoch so the arms can be compared at the same epoch and at similar step
counts. A (seed, epoch) the study did not search is taken from the original
best-checkpoint run when that run's best epoch is that epoch, since training is
deterministic and the study checks the weights are identical.

Usage:
    uv run python experiments/step3_epochs_report.py \
        --baseline checkpoints/step3-20260913-epochs/v4only \
        --mixed checkpoints/step3-20260913-epochs/mixed \
        --baseline-best checkpoints/step3-20260912/v4only \
        --mixed-best checkpoints/step3-20260912/mixed \
        --verified reports/world-plus-v4-policy-20260912/verified_solves_epochs.json \
        --out reports/world-plus-v4-policy-20260912
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41+")
BUDGET = (64, 48)
BATCH = 256


def snapshot_state(
    run_dir: Path, arm: str, seed: int, epoch: int
) -> dict[str, Any] | None:
    path = run_dir / "policies" / f"{arm}-s{seed}-e{epoch}" / "trainer_state.json"
    return json.loads(path.read_text()) if path.exists() else None


def find_cell(
    run_dir: Path, best_dir: Path | None, arm: str, seed: int, epoch: int
) -> tuple[dict[str, Any] | None, str]:
    """The v4-cohort cell for (seed, epoch), and where it came from."""
    for path in sorted((run_dir / "timing").glob(f"{arm}-s{seed}-*.json")):
        cell = json.loads(path.read_text())
        if (
            cell.get("policy_epoch") == epoch
            and cell.get("eval_corpus", "stack_pbe_v4") == "stack_pbe_v4"
            and (cell.get("beam_width"), cell.get("max_depth")) == BUDGET
        ):
            return cell, "study"
    if best_dir is not None:
        state_path = best_dir / "policies" / f"{arm}-s{seed}" / "trainer_state.json"
        cell_path = best_dir / "timing" / f"{arm}-s{seed}-cuda.json"
        if (
            state_path.exists()
            and cell_path.exists()
            and json.loads(state_path.read_text())["best_epoch"] == epoch
        ):
            return json.loads(cell_path.read_text()), "best"
    return None, "-"


def counts(
    cell: dict[str, Any], genuine: set[tuple[str, str]] | None
) -> dict[str, Any]:
    solved = [r for r in cell["rows"] if r["solved"]]
    ok = (
        solved
        if genuine is None
        else [r for r in solved if (r["behaviour"], " ".join(r["program"])) in genuine]
    )
    return {
        "graded": len(solved),
        "verified": len(ok),
        "buckets": {b: sum(r["bucket"] == b for r in ok) for b in BUCKETS},
        "tasks": len(cell["rows"]),
        "unverified_programs": sorted(
            {(r["behaviour"], " ".join(r["program"])) for r in solved}
            - (genuine or set())
        )
        if genuine is not None
        else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--baseline-best", type=Path)
    parser.add_argument("--mixed-best", type=Path)
    parser.add_argument("--verified", type=Path)
    parser.add_argument(
        "--allow-unchecked",
        action="store_true",
        help="count solved programs the verification file never checked as unverified",
    )
    parser.add_argument("--arm", default="1m")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    genuine = None
    checked: set[tuple[str, str]] = set()
    inconclusive: list[tuple[str, str]] = []
    if args.verified and args.verified.exists():
        verified = json.loads(args.verified.read_text())
        genuine = {
            (p["behaviour"], p["program"])
            for p in verified["programs"]
            if p["verdict"] == "genuine"
        }
        checked = {(p["behaviour"], p["program"]) for p in verified["programs"]}
        inconclusive = sorted(
            (p["behaviour"], p["program"])
            for p in verified["programs"]
            if p["verdict"] == "inconclusive"
        )
    arms = {
        "v4-only": (args.baseline, args.baseline_best),
        "v4+world": (args.mixed, args.mixed_best),
    }
    grid: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    for name, (run_dir, best_dir) in arms.items():
        for seed in args.seeds:
            for epoch in args.epochs:
                state = snapshot_state(run_dir, args.arm, seed, epoch)
                cell, source = find_cell(run_dir, best_dir, args.arm, seed, epoch)
                entry: dict[str, Any] = {"source": source}
                if state:
                    entry["validation_ce"] = state["validation_cross_entropy"]
                    entry["train_ce"] = state["train_cross_entropy"]
                    entry["steps"] = epoch * -(-state["train_rows"] // BATCH)
                if cell:
                    entry.update(counts(cell, genuine))
                grid.setdefault(name, {}).setdefault(seed, {})[epoch] = entry

    if genuine is not None:
        # A program the verification file never saw is not a shortcut, it is
        # unchecked; counting it as unverified would undercount silently.
        unchecked = sorted(
            {
                (r["behaviour"], " ".join(r["program"]))
                for name, (run_dir, best_dir) in arms.items()
                for seed in args.seeds
                for epoch in args.epochs
                for cell in [find_cell(run_dir, best_dir, args.arm, seed, epoch)[0]]
                if cell
                for r in cell["rows"]
                if r["solved"]
            }
            - checked
        )
        if unchecked and not args.allow_unchecked:
            parser.error(
                f"{len(unchecked)} solved programs are not in {args.verified}; run "
                f"step3_verify_solves.py over these run dirs first: {unchecked[:5]}"
            )

    label = "verified / graded" if genuine is not None else "graded"
    lines = [
        f"## Solves by epoch on the 300-task v4 cohort ({label}; beam 64 x depth 48)",
        "",
        "Each cell: solves, then validation CE on the pinned v4 holdout at that epoch. "
        "`*` marks a cell reused from the best-checkpoint run.",
        "",
        "| epoch | " + " | ".join(f"{n} s{s}" for n in arms for s in args.seeds) + " |",
        "|---" * (1 + len(arms) * len(args.seeds)) + "|",
    ]
    for epoch in args.epochs:
        row = [str(epoch)]
        for name in arms:
            for seed in args.seeds:
                e = grid[name][seed][epoch]
                if "graded" in e:
                    solves = (
                        f"{e['verified']}/{e['graded']}"
                        if genuine is not None
                        else str(e["graded"])
                    )
                    mark = "*" if e["source"] == "best" else ""
                    ce = f" ({e['validation_ce']:.3f})" if "validation_ce" in e else ""
                    row.append(f"{solves}{mark}{ce}")
                elif "validation_ce" in e:
                    row.append(f"- ({e['validation_ce']:.3f})")
                else:
                    row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        f"## Matched-epoch paired difference in "
        f"{'verified' if genuine is not None else 'graded'} solves (v4+world minus v4-only)",
        "",
    ]
    lines += ["| epoch | s0 | s1 | s2 | mean |", "|---|---|---|---|---|"]
    key = "verified" if genuine is not None else "graded"
    summary: dict[int, Any] = {}
    for epoch in args.epochs:
        deltas = []
        cells_row = []
        for seed in args.seeds:
            a, b = grid["v4-only"][seed][epoch], grid["v4+world"][seed][epoch]
            if key in a and key in b:
                deltas.append(b[key] - a[key])
                cells_row.append(f"{b[key] - a[key]:+d}")
            else:
                cells_row.append("-")
        mean = (
            f"{statistics.mean(deltas):+.1f}" if len(deltas) == len(args.seeds) else "-"
        )
        summary[epoch] = {"deltas": deltas, "complete": len(deltas) == len(args.seeds)}
        lines.append(f"| {epoch} | " + " | ".join(cells_row) + f" | {mean} |")
    lines += [
        "",
        "Steps per epoch: v4-only "
        + ", ".join(
            str(grid["v4-only"][s][1].get("steps", "-")) for s in args.seeds[:1]
        )
        + ", v4+world "
        + ", ".join(
            str(grid["v4+world"][s][1].get("steps", "-")) for s in args.seeds[:1]
        )
        + " (batch 256). Arm B's epoch e is about arm A's epoch 1.37e in steps.",
        "",
    ]
    if inconclusive:
        lines += [
            "Programs the verification could not decide (too few in-domain draws in a "
            "mode), counted as neither verified nor shortcut:",
            "",
            *(f"- `{b}`, {len(p.split())} ops" for b, p in inconclusive),
            "",
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "epochs.md").write_text("\n".join(lines) + "\n")
    (args.out / "epochs.json").write_text(
        json.dumps(
            {"grid": grid, "matched_epoch": summary},
            indent=2,
            sort_keys=True,
            default=list,
        )
        + "\n"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
