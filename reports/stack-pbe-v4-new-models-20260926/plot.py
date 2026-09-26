"""Accuracy against API-list cost per task, effort levels joined per model (#173).

Reads summary.json (written by analyse.py). New models solid, earlier rows dashed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ORDER = (  # model, colour (fixed categorical slots), new in #173
    ("claude-opus-5-5", "#2a78d6", True),
    ("gpt-6-sol", "#eb6834", True),
    ("gpt-6-luna", "#1baf7a", True),
    ("claude-opus-5", "#4a3aa7", False),
    ("gpt-6-astra", "#e87ba4", False),
    ("gpt-5.6-terra", "#eda100", False),
    ("gpt-5.6-luna", "#008300", False),
)
EFFORT = {"low": "L", "medium": "M", "high": "H", "xhigh": "X"}

rows = json.loads((HERE / "summary.json").read_text())
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
ax.axhspan(22, 26, color="#e8e7e1", zorder=0)
ax.text(
    0.00105, 25.6, "top cluster, ±3-task noise", color="#6b6a64", fontsize=8, va="top"
)
for model, colour, new in ORDER:
    pts = sorted(
        (r["cost_per_task"], r["solved"], r["effort"])
        for r in rows
        if r["model"] == model
    )
    xs, ys, _ = zip(*pts, strict=True)
    marker = "o" if model.startswith("claude") else "s"
    ax.plot(
        xs,
        ys,
        color=colour,
        lw=2,
        ls="-" if new else (0, (4, 2)),
        marker=marker,
        ms=8,
        mec="white",
        mew=1.5,
        label=model + (" (new)" if new else ""),
        zorder=3,
    )
    above = model == "gpt-6-astra"  # sits on Opus 5.5's medium/high points
    for x, y, e in pts:
        ax.annotate(
            EFFORT[e],
            (x, y),
            xytext=(0, 7 if above else -13),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#6b6a64",
        )
    anchor, offset, ha = (
        ((xs[0], ys[0]), (-8, 3), "right")
        if above
        else ((xs[-1], ys[-1]), (8, 3), "left")
    )
    ax.annotate(
        model,
        anchor,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        fontsize=8,
        color="#2b2b29",
        weight="bold" if new else "normal",
    )
ax.set_xscale("log")
ax.set_xlim(0.001, 0.4)
ax.set_ylim(8, 27)
ax.set_xlabel(
    "$ per task at API list price (recorded output tokens, 1.3k-token prompt)"
)
ax.set_ylabel("solved, of 40 (held-out graded)")
ax.set_title(
    "stack_pbe_v4 seed-17 sample: accuracy against cost, effort joined per model\n"
    "L/M/H/X = effort low/medium/high/xhigh",
    fontsize=10,
    loc="left",
)
ax.grid(True, which="major", color="#e8e7e1", lw=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(HERE / "accuracy_vs_cost.png")
