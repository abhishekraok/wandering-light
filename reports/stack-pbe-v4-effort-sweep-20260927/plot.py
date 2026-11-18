"""Accuracy against API-list cost per task, effort levels joined per model.

Reads summary.json (written by analyse.py). Current-generation frontier points
are coloured; all other hosted points are grey. The step line includes every
hosted generation, while local systems retain distinct styling.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ORDER = (  # model, colour, current generation (eligible for colour)
    ("claude-sonnet-5-5", "#006d77", True),
    ("claude-haiku-5-5", "#d4a017", True),
    ("claude-fable-5-2", "#7a3db8", True),
    ("claude-fable-5-1", "#7a3db8", False),
    ("claude-fable-5", "#b58ae0", False),
    ("claude-opus-5-5", "#2a78d6", True),
    ("claude-opus-5", "#8fb8ea", False),
    ("gpt-6.2-astra", "#c2185b", True),
    ("gpt-6-astra", "#c2185b", False),
    ("gpt-6.2-sol", "#b83a0b", True),
    ("gpt-6.1-sol", "#eb6834", False),
    ("gpt-6.1-terra", "#8b6508", True),
    ("gpt-6-sol", "#eb6834", False),
    ("gpt-6-luna", "#1baf7a", True),
    ("gpt-5.6-sol", "#f2a57f", False),
    ("gpt-5.6-terra", "#eda100", False),
    ("gpt-5.6-luna", "#7fd3b2", False),
)
EFFORT = {"low": "L", "medium": "M", "high": "H", "xhigh": "X", "max": "Mx"}

saved_rows = json.loads((HERE / "summary.json").read_text())
rows = [
    r
    for r in saved_rows
    if r["cost_per_task"] is not None
    and (
        r["model"] not in ("gpt-6.1-terra", "gpt-6.2-sol", "gpt-6.2-astra")
        or r.get("model_confirmed_calls") == 40
    )
]
# Keep exact ties: equally good/costly models receive the same treatment.
frontier = [
    r
    for r in rows
    if not any(
        s["cost_per_task"] <= r["cost_per_task"]
        and s["solved"] >= r["solved"]
        and (s["cost_per_task"] < r["cost_per_task"] or s["solved"] > r["solved"])
        for s in rows
    )
]
frontier_keys = {(r["model"], r["effort"]) for r in frontier}
step_points = sorted({(r["cost_per_task"], r["solved"]) for r in frontier})
best_score = max(r["solved"] for r in rows)
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
ax.axhspan(best_score - 3, best_score, color="#e8e7e1", zorder=0)
ax.text(
    1e-5,
    best_score - 0.2,
    "within noise (±3) of the best score",
    color="#6b6a64",
    fontsize=8,
    va="top",
)
ax.step(
    [p[0] for p in step_points],
    [p[1] for p in step_points],
    where="post",
    color="#454545",
    lw=0.9,
    ls="--",
    zorder=2.5,
)
for model, colour, new in ORDER:
    pts = [
        (r["cost_per_task"], r["solved"], r["effort"])
        for r in rows
        if r["model"] == model
    ]
    if not pts:
        continue
    pts.sort(key=lambda p: list(EFFORT).index(p[2]))  # joined in effort order
    xs, ys, _ = zip(*pts, strict=True)
    ax.plot(xs, ys, color="#cfcec8", lw=0.7, marker="o", ms=3, zorder=2)
    highlighted = [p for p in pts if new and (model, p[2]) in frontier_keys]
    if not highlighted:
        continue
    ax.scatter(
        [p[0] for p in highlighted],
        [p[1] for p in highlighted],
        color=colour,
        marker="o" if model.startswith("claude") else "s",
        s=64,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    for x, y, e in highlighted:
        ax.annotate(
            EFFORT[e],
            (x, y),
            xytext=(5, -13) if model == "gpt-6.2-astra" else (-5, -13),
            textcoords="offset points",
            ha="left" if model == "gpt-6.2-astra" else "right",
            fontsize=7,
            color="#6b6a64",
        )
    # Anchor to a frontier point, including future newly dominant models.
    x, y, _ = highlighted[-1]
    ax.annotate(
        model,
        (x, y),
        xytext=(-10, 8) if model == "gpt-6.2-sol" else (10, 4),
        textcoords="offset points",
        ha="right" if model == "gpt-6.2-sol" else "left",
        fontsize=8.5,
        fontweight="bold",
        color=colour,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1},
        zorder=5,
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
    color="#3b6b87",
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
    color="#3b6b87",
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
    ("qwen3:30b-a3b (ollama)", "qwen3 30b-a3b (4090)", "#aaaaaa", "D"),
    ("gpt-oss:20b high (ollama)", "gpt-oss 20b high (4090)", "#aaaaaa", "D"),
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
ax.set_ylim(-1.5, max(28, best_score + 1))
ax.set_xlabel(
    "$ per task: API list price for hosted models (1.3k-token prompt + recorded output);\n"
    "wall clock x cloud rental rate for local solvers"
)
ax.set_ylabel("solved, of 40 (held-out graded)")
ax.set_title(
    "stack_pbe_v4 seed-17 sample: accuracy against cost, hosted cost frontier dashed\n"
    "L/M/H/X/Mx = effort low/medium/high/xhigh/max",
    fontsize=10,
    loc="left",
)
ax.grid(True, which="major", color="#e8e7e1", lw=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
notes = [
    "colour: current-generation hosted frontier points + own systems\n"
    "grey: other hosted points (including older frontier endpoints); rows in table.md\n"
    "dashed steps: all hosted generations; local systems excluded",
    "Terra 6.1 L/M use authorized identity reruns; originals retained.",
]
for r in saved_rows:
    if r["model"] != "claude-sonnet-5-5":
        continue
    if r["cost_per_task"] is None:
        notes.append(
            f"Sonnet {r['effort']}: {r['solved']}/40\n"
            f"not plotted: incomplete\npricing metadata for\n{40 - r['usage_calls']}/40 calls"
        )
    elif r.get("refusals"):
        notes.append(
            f"Sonnet {r['effort']}: {r['refusals']} refusals\n"
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
