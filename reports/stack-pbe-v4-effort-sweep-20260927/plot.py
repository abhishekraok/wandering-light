"""Accuracy against API-list cost per task, effort levels joined per model.

Reads summary.json (written by analyse.py). Latest generation of each line coloured and labelled, earlier ones grey.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ORDER = (  # model, colour, latest generation (coloured, labelled) or earlier (grey)
    ("claude-sonnet-5-5", "#006d77", True),
    ("claude-fable-5-1", "#7a3db8", True),
    ("claude-fable-5", "#b58ae0", False),
    ("claude-opus-5-5", "#2a78d6", True),
    ("claude-opus-5", "#8fb8ea", False),
    ("gpt-6-astra", "#c2185b", True),
    ("gpt-6.1-sol", "#b83a0b", True),
    ("gpt-6-sol", "#eb6834", False),
    ("gpt-6-luna", "#1baf7a", True),
    ("gpt-5.6-sol", "#f2a57f", False),
    ("gpt-5.6-terra", "#eda100", False),
    ("gpt-5.6-luna", "#7fd3b2", False),
)
EFFORT = {"low": "L", "medium": "M", "high": "H", "xhigh": "X", "max": "Mx"}

rows = json.loads((HERE / "summary.json").read_text())
refusals = {(r["model"], r["effort"]): r.get("refusals", 0) for r in rows}
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
ax.axhspan(22, 27.3, color="#e8e7e1", zorder=0)
ax.text(1e-5, 26.9, "top cluster, ±3-task noise", color="#6b6a64", fontsize=8, va="top")
LABEL = {  # current generation only: anchor effort, offset (points), alignment
    "claude-sonnet-5-5": ("xhigh", (0, 9), "center"),
    "claude-fable-5-1": ("xhigh", (8, 3), "left"),
    "claude-opus-5-5": ("low", (-8, -3), "right"),
    "gpt-6-astra": ("low", (0, 20), "center"),  # above Opus 5.5's medium/high
    "gpt-6.1-sol": ("low", (-8, -3), "right"),
    "gpt-6-luna": ("max", (8, 3), "left"),
}
for model, colour, new in ORDER:
    pts = [
        (r["cost_per_task"], r["solved"], r["effort"])
        for r in rows
        if r["model"] == model and r["cost_per_task"] is not None
    ]
    if not pts:
        continue
    pts.sort(key=lambda p: list(EFFORT).index(p[2]))  # joined in effort order
    xs, ys, _ = zip(*pts, strict=True)
    if not new:  # earlier generation: unlabelled grey context
        ax.plot(xs, ys, color="#cfcec8", lw=1, marker="o", ms=3, zorder=2)
        continue
    ax.plot(
        xs,
        ys,
        color=colour,
        lw=2,
        marker="o" if model.startswith("claude") else "s",
        ms=8,
        mec="white",
        mew=1.5,
        zorder=3,
    )
    above = model == "gpt-6-astra"
    for x, y, e in pts:
        ax.annotate(
            EFFORT[e] + ("*" if refusals[(model, e)] else ""),
            (x, y),
            xytext=(0, 7 if above else -13),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#6b6a64",
        )
    effort, offset, ha = LABEL[model]
    x, y, _ = next(p for p in pts if p[2] == effort)
    ax.annotate(
        model,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        fontsize=8.5,
        fontweight="bold",
        color=colour,
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
    zorder=3,
)
ax.annotate(
    "BFS (CPU)",
    (bfs[-1]["cost_per_task"], bfs[-1]["solved"]),
    xytext=(8, -3),
    textcoords="offset points",
    fontsize=7.5,
    color="#2b2b29",
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
ax.set_ylim(-1.5, max(28, max(r["solved"] for r in rows) + 1))
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
notes = [
    "grey: earlier generations (gpt-6-sol, gpt-5.6 sol/terra/luna,\n"
    "claude-opus-5, claude-fable-5), unlabelled; all rows in table.md"
]
for r in rows:
    if r["model"] != "claude-sonnet-5-5":
        continue
    if r["cost_per_task"] is None:
        notes.append(
            f"Sonnet {r['effort']}: {r['solved']}/40\n"
            f"not plotted: incomplete\npricing metadata for\n{40 - r['usage_calls']}/40 calls"
        )
    elif r.get("refusals"):
        notes.append(
            f"*Sonnet {r['effort']}: {r['refusals']} refusals\n"
            "(unsolved; output tokens priced)."
        )
ax.text(
    0.99,
    0.03,
    "\n\n".join(notes),
    transform=ax.transAxes,
    fontsize=8,
    color="#6b6a64",
    ha="right",
    va="bottom",
)
fig.tight_layout()
fig.savefig(HERE / "accuracy_vs_cost.png")
