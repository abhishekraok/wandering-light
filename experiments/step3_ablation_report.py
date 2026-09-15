"""Validation CE and verified solves across the ablation arms (keep-below-ablation, #136).

#134 found the v4+world search gain sits in one behaviour, `keep_below`, whose
witness occurs whole in 156 world fit programs. This lays the #136 arms next to
#134's at matched epochs: holdout CE, verified cohort solves, solves of the
named behaviour, and the paired difference against a reference arm with that
behaviour taken out. Cells come from the same epoch-study layout as
``step3_epochs_report`` (``policies/<arm>-s<seed>-e<epoch>`` snapshots and
``timing/`` cells with ``policy_epoch``), with the best-checkpoint fallback.

Usage:
    uv run python experiments/step3_ablation_report.py \
        --arm-dir v4only=checkpoints/step3-20260913-epochs/v4only \
        --arm-dir mixed=checkpoints/step3-20260913-epochs/mixed \
        --arm-dir ablated=checkpoints/step3-20260913-ablation/ablated \
        --arm-dir random=checkpoints/step3-20260913-ablation/random \
        --best v4only=checkpoints/step3-20260912/v4only \
        --best mixed=checkpoints/step3-20260912/mixed \
        --verified reports/keep-below-ablation-20260913/verified_solves_ablation.json \
        --out reports/keep-below-ablation-20260913
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.step3_epochs_report import find_cell, snapshot_state  # noqa: E402
from experiments.step3_verify_solves import partition  # noqa: E402

Grid = dict[str, dict[int, dict[int, dict[str, Any]]]]


def cell_counts(
    cell: dict[str, Any],
    passing: set[tuple[str, str]],
    behaviour: str,
    witness: set[tuple[str, str]] = frozenset(),
) -> dict[str, int]:
    """Verified solves in a cell, how many are ``behaviour``, how many a witness."""
    ok = [
        r
        for r in cell["rows"]
        if r["solved"] and (r["behaviour"], " ".join(r["program"])) in passing
    ]
    named = sum(r["behaviour"] == behaviour for r in ok)
    return {
        "verified": len(ok),
        "named": named,
        "other": len(ok) - named,
        "witness": sum((r["behaviour"], " ".join(r["program"])) in witness for r in ok),
        "named_tasks": sum(r["behaviour"] == behaviour for r in cell["rows"]),
    }


def paired(
    grid: Grid, arm: str, reference: str, key: str, seeds: list[int], epochs: list[int]
) -> dict[int, list[int | None]]:
    """Per epoch, ``arm`` minus ``reference`` on ``key`` per seed (None if unsearched)."""
    out = {}
    for epoch in epochs:
        deltas: list[int | None] = []
        for seed in seeds:
            a = grid[reference].get(seed, {}).get(epoch, {})
            b = grid[arm].get(seed, {}).get(epoch, {})
            deltas.append(b[key] - a[key] if key in a and key in b else None)
        out[epoch] = deltas
    return out


def fmt_mean(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.4f}" if values else "-"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def _col(v: int | None) -> str:
    return "-" if v is None else f"{v:+d}"


def _mean(vs: list[int | None]) -> str:
    return f"{statistics.mean(vs):+.1f}" if all(v is not None for v in vs) else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm-dir", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--best", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument(
        "--allow-unchecked",
        action="store_true",
        help="count solved programs the verification file never checked as unverified",
    )
    parser.add_argument("--behaviour", default="keep_below")
    parser.add_argument("--reference", default="mixed")
    parser.add_argument("--arm", default="1m")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    arms = {k: Path(v) for k, v in (s.split("=", 1) for s in args.arm_dir)}
    best = {k: Path(v) for k, v in (s.split("=", 1) for s in args.best)}
    verified = json.loads(args.verified.read_text())
    # A verified solve is draw-tested or a catalogue witness, as the recorded
    # reports define it; `witness` counts the latter apart (#142).
    genuine, witness = partition(verified["programs"])
    passing = genuine | witness
    checked = {(p["behaviour"], p["program"]) for p in verified["programs"]}

    grid: Grid = {}
    unchecked: set[tuple[str, str]] = set()
    train_rows: dict[str, int] = {}
    for name, run_dir in arms.items():
        for seed in args.seeds:
            for epoch in args.epochs:
                entry: dict[str, Any] = {}
                state = snapshot_state(run_dir, args.arm, seed, epoch)
                if state:
                    entry["validation_ce"] = state["validation_cross_entropy"]
                    train_rows.setdefault(name, state["train_rows"])
                cell, source = find_cell(run_dir, best.get(name), args.arm, seed, epoch)
                if cell:
                    entry.update(cell_counts(cell, passing, args.behaviour, witness))
                    entry["source"] = source
                    unchecked |= {
                        (r["behaviour"], " ".join(r["program"]))
                        for r in cell["rows"]
                        if r["solved"]
                    } - checked
                if entry:
                    grid.setdefault(name, {}).setdefault(seed, {})[epoch] = entry
    # A solved program the verification file never saw is unchecked, not
    # unverified; counting it as unverified would undercount silently (Codex
    # review on #140).
    if unchecked and not args.allow_unchecked:
        parser.error(
            f"{len(unchecked)} solved programs are not in {args.verified}; run "
            f"step3_verify_solves.py over these run dirs first: {sorted(unchecked)[:5]}"
        )

    names = [n for n in arms if n in grid]
    seeds_of = {n: [s for s in args.seeds if s in grid[n]] for n in names}
    header = "| epoch | " + " | ".join(f"{n} s{s}" for n in names for s in seeds_of[n])
    rule = "|---|" + "---|" * sum(len(seeds_of[n]) for n in names)
    lines = [
        "## Fit rows per arm (batch 256; validation is the pinned v4 holdout)",
        "",
        "| arm | fit rows | seeds |",
        "|---|---|---|",
        *(
            f"| {n} | {train_rows.get(n, '-')} | {', '.join(map(str, seeds_of[n]))} |"
            for n in names
        ),
        "",
        "## Validation cross-entropy by epoch",
        "",
        header + " |",
        rule,
    ]
    for epoch in args.epochs:
        row = []
        for n in names:
            for s in seeds_of[n]:
                e = grid[n][s].get(epoch, {})
                row.append(f"{e['validation_ce']:.4f}" if "validation_ce" in e else "-")
        lines.append(f"| {epoch} | " + " | ".join(row) + " |")
    best_ce: dict[str, dict[int, dict[str, Any]]] = {}
    for n in names:
        for s in seeds_of[n]:
            cands = [
                (e["validation_ce"], ep)
                for ep, e in grid[n][s].items()
                if "validation_ce" in e
            ]
            if cands:
                ce, ep = min(cands)
                best_ce.setdefault(n, {})[s] = {"ce": ce, "epoch": ep}
    lines += [
        "",
        "Best epoch per seed (minimum validation CE), and the paired difference "
        "against the reference arm on the same seed.",
        "",
        f"| arm | best CE per seed | mean ± sd | minus {args.reference} |",
        "|---|---|---|---|",
    ]
    ce_diff: dict[str, list[float]] = {}
    ref_best = best_ce.get(args.reference, {})
    for n in names:
        if n not in best_ce:
            continue
        cells = [f"{v['ce']:.4f} (e{v['epoch']})" for v in best_ce[n].values()]
        diffs = [
            best_ce[n][s]["ce"] - ref_best[s]["ce"]
            for s in best_ce[n]
            if s in ref_best and n != args.reference
        ]
        ce_diff[n] = diffs
        lines.append(
            f"| {n} | {', '.join(cells)} | "
            f"{fmt_mean([v['ce'] for v in best_ce[n].values()])} | "
            + (", ".join(f"{d:+.4f}" for d in diffs) if diffs else "-")
            + " |"
        )

    lines += [
        "",
        "## Verified solves by epoch (v4 cohort, beam 64 x depth 48; "
        f"cells read all / `{args.behaviour}`)",
        "",
        header + " |",
        rule,
    ]
    for epoch in args.epochs:
        row = []
        for n in names:
            for s in seeds_of[n]:
                e = grid[n][s].get(epoch, {})
                mark = "*" if e.get("source") == "best" else ""
                row.append(
                    f"{e['verified']}/{e['named']}{mark}" if "verified" in e else "-"
                )
        lines.append(f"| {epoch} | " + " | ".join(row) + " |")
    lines += ["", "`*`: cell taken from the original best-checkpoint run."]

    diffs_out: dict[str, Any] = {}
    for n in names:
        if n == args.reference:
            continue
        seeds = [s for s in seeds_of[n] if s in seeds_of.get(args.reference, [])]
        if not seeds:
            continue
        lines += [
            "",
            f"## Matched-epoch paired difference, {n} minus {args.reference}",
            "",
            f"Columns: all verified solves, then with `{args.behaviour}` removed.",
            "",
            "| epoch | "
            + " | ".join(f"s{s}" for s in seeds)
            + " | mean | "
            + " | ".join(f"other s{s}" for s in seeds)
            + " | other mean |",
            "|---|" + "---|" * (2 * len(seeds) + 2),
        ]
        all_d = paired(grid, n, args.reference, "verified", seeds, args.epochs)
        other_d = paired(grid, n, args.reference, "other", seeds, args.epochs)
        diffs_out[n] = {"verified": all_d, "other": other_d}
        for epoch in args.epochs:
            a, o = all_d[epoch], other_d[epoch]
            lines.append(
                f"| {epoch} | "
                + " | ".join(_col(v) for v in a)
                + f" | {_mean(a)} | "
                + " | ".join(_col(v) for v in o)
                + f" | {_mean(o)} |"
            )

    named_total = {
        n: sum(e.get("named", 0) for s in seeds_of[n] for e in grid[n][s].values())
        for n in names
    }
    lines += [
        "",
        f"`{args.behaviour}` verified solves summed over searched cells: "
        + ", ".join(f"{n} {named_total[n]}" for n in names)
        + ".",
        "",
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.md").write_text("\n".join(lines) + "\n")
    (args.out / "ablation.json").write_text(
        json.dumps(
            {
                "behaviour": args.behaviour,
                "reference": args.reference,
                "train_rows": train_rows,
                "grid": {
                    n: {str(s): {str(e): v for e, v in d.items()} for s, d in g.items()}
                    for n, g in grid.items()
                },
                "best_ce": {
                    n: {str(s): v for s, v in d.items()} for n, d in best_ce.items()
                },
                "best_ce_minus_reference": ce_diff,
                "paired": {
                    n: {k: {str(e): v for e, v in d.items()} for k, d in kd.items()}
                    for n, kd in diffs_out.items()
                },
                "named_total": named_total,
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
