"""Does ``L x deficit < ln W`` predict which behaviours a beam solves?

The budget argument: the beam keeps the ``W`` highest cumulative
log-probabilities, so a path that falls ``d`` nats behind the best available one
has roughly ``e^d`` paths ahead of it, and a width-``W`` beam tolerates about
``ln W`` nats -- 4.159 at ``W = 64``. A witness of length ``L`` whose policy
deficit averages ``delta`` per step accumulates ``L * delta``. The prediction is
that behaviours with ``L * delta`` below the tolerance get solved and the rest do
not, whatever else is true about them.

That argument already explains the cohort's shape without being tested: the
budget ``ln W / L`` falls 0.832 / 0.693 / 0.260 / 0.107 / 0.053 nats per step
across the witness-length buckets and behaviour coverage falls 100% / 67% / 67% /
33% / 11% with it. But that is the budget against the *median length*, not
against the policy's *measured* deficit, and the two could disagree.

**This is analysis, not a method.** It replays cohort witnesses, so everything
here has seen the eval set. Nothing it produces may select a checkpoint, an epoch
or a hyper-parameter; `search_proxy.py`'s train-side sets exist for that. What it
is for is telling `highway-construction (#96)`, `mine-v4-library (#148)` and
`witness-trajectory-shape (#150)` the factor by which a witness must shorten
before the budget clears the measured deficit -- a target, per behaviour, rather
than a guess.

    uv run python experiments/deficit_law.py --proxy <cohort proxy json> \
        --cell <a cache-off cohort cell> --out reports/.../deficit_law.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TOLERANCE = math.log(64)


def confusion(behaviours: dict[str, dict[str, float]], solved: set[str]) -> dict[str, Any]:
    """The 2x2 of (budget says solvable) against (a beam actually solved it)."""
    tp = fp = tn = fn = 0
    for name, stats in behaviours.items():
        predicted = stats["accumulated"] < TOLERANCE
        actual = name in solved
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1  # solved despite the budget saying no -- the interesting cell
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "predicted_solvable_and_solved": tp,
        "predicted_solvable_not_solved": fp,
        "predicted_unsolvable_but_solved": fn,
        "predicted_unsolvable_and_not_solved": tn,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "behaviours": total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    proxy = json.loads(args.proxy.read_text())
    if "cohort_by_behaviour" not in proxy:
        raise SystemExit(f"{args.proxy} has no cohort rows; score it with --sets cohort")
    behaviours = proxy["cohort_by_behaviour"]
    cell = json.loads(args.cell.read_text())
    solved = {r["behaviour"] for r in cell["rows"] if r["solved"]}

    rows = sorted(behaviours.items(), key=lambda kv: kv[1]["accumulated"])
    print(f"tolerance ln(64) = {TOLERANCE:.3f} nats\n")
    print(f"{'behaviour':34s} {'L':>5s} {'deficit':>8s} {'L*deficit':>10s} {'budget':>7s} {'solved':>7s}")
    for name, stats in rows:
        print(
            f"{name:34s} {stats['length']:>5.0f} {stats['deficit']:>8.3f} "
            f"{stats['accumulated']:>10.2f} {stats['budget_ratio']:>7.2f} "
            f"{'yes' if name in solved else 'no':>7s}"
        )

    table = confusion(behaviours, solved)
    print(f"\nprediction: solved iff L*deficit < {TOLERANCE:.3f}")
    for key, value in table.items():
        print(f"  {key:38s} {value if isinstance(value, int) else f'{value:.3f}'}")

    need = {
        name: stats["accumulated"] / TOLERANCE
        for name, stats in behaviours.items()
        if name not in solved and stats["accumulated"] >= TOLERANCE
    }
    if need:
        ordered = sorted(need.items(), key=lambda kv: kv[1])
        print(f"\nshortening factor a library must deliver, {len(need)} unsolved behaviours:")
        print(f"  easiest  {ordered[0][0]}: {ordered[0][1]:.1f}x")
        print(f"  median   {ordered[len(ordered) // 2][1]:.1f}x")
        print(f"  hardest  {ordered[-1][0]}: {ordered[-1][1]:.1f}x")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "tolerance": TOLERANCE,
                    "cell": str(args.cell),
                    "proxy": str(args.proxy),
                    "confusion": table,
                    "by_behaviour": {
                        name: {**stats, "solved": name in solved}
                        for name, stats in behaviours.items()
                    },
                    "shortening_factor_needed": need,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
