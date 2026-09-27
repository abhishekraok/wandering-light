"""Accuracy against API-list cost per task, effort levels joined per model.

Reads summary.json (written by analyse.py). Latest generation of each line solid, earlier dashed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ORDER = (  # model, colour, latest generation of its line (solid) or earlier (dashed)
    ("claude-fable-5-1", "#7a3db8", True),
    ("claude-fable-5", "#b58ae0", False),
    ("claude-opus-5-5", "#2a78d6", True),
    ("claude-opus-5", "#8fb8ea", False),
    ("gpt-6-astra", "#c2185b", True),
    ("gpt-6-sol", "#eb6834", True),
    ("gpt-6-luna", "#1baf7a", True),
    ("gpt-5.6-sol", "#f2a57f", False),
    ("gpt-5.6-terra", "#eda100", False),
    ("gpt-5.6-luna", "#7fd3b2", False),
)
EFFORT = {"low": "L", "medium": "M", "high": "H", "xhigh": "X", "max": "Mx"}

rows = json.loads((HERE / "summary.json").read_text())
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
ax.axhspan(22, 27.3, color="#e8e7e1", zorder=0)
ax.text(1e-5, 26.9, "top cluster, ±3-task noise", color="#6b6a64", fontsize=8, va="top")
for model, colour, new in ORDER:
    pts = sorted(
        (r["cost_per_task"], r["solved"], r["effort"])
        for r in rows
        if r["model"] == model
    )
    if not pts:
        continue
    pts.sort(key=lambda p: list(EFFORT).index(p[2]))  # joined in effort order
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
        label=model,
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
        ((xs[0], ys[0]), (4, -24), "center")
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
    )
# Local solvers (local/local_rows.json): $ from wall clock at cloud rental rates
# (c7i vCPU-hour for CPU, $0.40/h for an RTX 4090), not API list prices.
local = {
    r["label"]: r for r in json.loads((HERE / "local" / "local_rows.json").read_text())
}
bfs = [local[f"BFS {k} nodes"] for k in ("1k", "10k", "100k")]
ax.plot(
    [r["cost_per_task"] for r in bfs],
    [r["solved"] for r in bfs],
    color="#8a8983",
    lw=1.5,
    marker="^",
    ms=7,
    mec="white",
    label="BFS (CPU)",
    zorder=3,
)
for r, k in zip(bfs, ("1k", "10k", "100k"), strict=True):
    ax.annotate(
        k,
        (r["cost_per_task"], r["solved"]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color="#6b6a64",
    )
for label, name, colour, marker in (
    ("staged-s0-e3 (solve.BEST)", "1m policy, staged (best system)", "#111111", "*"),
    ("fused-s0-e3", "1m policy, fused witnesses", "#555555", "*"),
    ("qwen3:30b-a3b (ollama)", "qwen3 30b-a3b (4090)", "#8d5b3a", "D"),
    ("gpt-oss:20b high (ollama)", "gpt-oss 20b high (4090)", "#b0784f", "D"),
):
    r = local[label]
    ax.plot(
        r["cost_per_task"],
        r["solved"],
        color=colour,
        marker=marker,
        ms=13 if marker == "*" else 7,
        mec="white",
        ls="none",
        label=name,
        zorder=4,
    )
    ax.annotate(
        name,
        (r["cost_per_task"], r["solved"]),
        xytext=(8, -3),
        textcoords="offset points",
        fontsize=7.5,
        color="#2b2b29",
    )
ax.set_xscale("log")
ax.set_xlim(8e-6, 3)
ax.set_ylim(-1.5, 28)
ax.set_xlabel(
    "$ per task: API list price for hosted models (1.3k-token prompt + recorded output);\n"
    "wall clock x cloud rental rate for local solvers"
)
ax.set_ylabel("solved, of 40 (held-out graded)")
ax.set_title(
    "stack_pbe_v4 seed-17 sample: accuracy against cost, effort joined per model\n"
    "L/M/H/X/Mx = effort low/medium/high/xhigh/max",
    fontsize=10,
    loc="left",
)
ax.grid(True, which="major", color="#e8e7e1", lw=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(HERE / "accuracy_vs_cost.png")
