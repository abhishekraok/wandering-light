"""Verified and novel behaviours per arm and seed for distinct-programs (#174).

    uv run python experiments/distinct_programs/plot.py \
        --containment checkpoints/distinct-programs-20260926/containment.json \
        --out reports/distinct-programs-20260926/arms.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ARMS = [
    ("staged", "staged\n(#155 baseline)"),
    ("staged_new", "+ new programs\n(exposure-matched)"),
    ("staged_new_fixed", "+ new programs\n(compute-matched)"),
]
PANELS = [
    ("verified", "Verified eval behaviours (of 43)", None),
    ("novel", "…of which solved by a program not in training", None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--containment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.containment.read_text())
    base = {k: sum(c[k] for c in data["staged"]) / 3 for k in ("verified", "novel")}
    bars = {"verified": base["verified"] + 2, "novel": base["novel"] + 3}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, title, _) in zip(axes, PANELS, strict=True):
        for x, (arm, _) in enumerate(ARMS):
            ys = [c[key] for c in data[arm]]
            ax.scatter(
                [x + d for d in (-0.08, 0, 0.08)], ys, s=40, color="#3b6ea5", zorder=3
            )
            m = sum(ys) / len(ys)
            ax.hlines(m, x - 0.25, x + 0.25, color="#222", lw=2)
            ax.annotate(f"{m:.1f}", (x + 0.27, m), va="center", fontsize=9)
        ax.axhline(bars[key], color="#c0392b", ls="--", lw=1)
        ax.text(
            2.45,
            bars[key],
            "success bar",
            color="#c0392b",
            fontsize=8,
            ha="right",
            va="bottom",
        )
        ax.set_xticks(range(len(ARMS)), [label for _, label in ARMS], fontsize=9)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 20 if key == "verified" else 10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "395 new distinct programs (#174): no gain in eval behaviours (dots = seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
