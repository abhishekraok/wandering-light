"""Summarise new-models-sample40 (#173) against the earlier rows on the same sample.

Per arm: solved / memorised / shown-only / illegal, output tokens per task,
the model that served (Claude arms), tool calls, solves by witness-length
bucket, paired agreement with Opus 5 xhigh and astra high, and the cost the
same tokens would have at API list prices (prompt taken as the 1.3k tokens
the task needs; Codex's ~6k tool-schema residual is left out). Prints Markdown.
"""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
REPORTS = HERE.parent
BUCKETS = ((1, 10), (11, 20), (21, 40), (41, 1000))
PROMPT_TOKENS = 1300
PRICE = {  # $ per 1M input, output (list prices, 2026-09-26)
    "gpt-6-astra": (10, 50),
    "gpt-6-sol": (2, 10),
    "gpt-6-luna": (0.1, 0.5),
    "gpt-5.6-terra": (2, 12),
    "gpt-5.6-luna": (0.2, 1.2),
    "claude-opus-5": (5, 25),
    "claude-opus-5-5": (4, 20),
}
ARMS = (
    ("stack-pbe-v4-codex-exec-20260910/astra_low", "gpt-6-astra", "low"),
    ("stack-pbe-v4-codex-exec-20260910/astra_medium", "gpt-6-astra", "medium"),
    ("stack-pbe-v4-codex-exec-20260910/astra_high", "gpt-6-astra", "high"),
    ("stack-pbe-v4-codex-exec-20260910/terra_high", "gpt-5.6-terra", "high"),
    ("stack-pbe-v4-codex-exec-20260910/luna_high", "gpt-5.6-luna", "high"),
    ("stack-pbe-v4-claude-exec-20260913/opus_low", "claude-opus-5", "low"),
    ("stack-pbe-v4-claude-exec-20260913/opus_high", "claude-opus-5", "high"),
    ("stack-pbe-v4-claude-exec-20260913/opus_xhigh", "claude-opus-5", "xhigh"),
    ("stack-pbe-v4-new-models-20260926/sol6_high", "gpt-6-sol", "high"),
    ("stack-pbe-v4-new-models-20260926/luna6_high", "gpt-6-luna", "high"),
    ("stack-pbe-v4-new-models-20260926/opus55_low", "claude-opus-5-5", "low"),
    ("stack-pbe-v4-new-models-20260926/opus55_medium", "claude-opus-5-5", "medium"),
    ("stack-pbe-v4-new-models-20260926/opus55_high", "claude-opus-5-5", "high"),
    ("stack-pbe-v4-new-models-20260926/opus55_xhigh", "claude-opus-5-5", "xhigh"),
)


def load(stem: str) -> dict:
    return json.loads((REPORTS / f"{stem}_v4_sample40.json").read_text())


def main() -> None:
    api = json.loads(
        (REPORTS / "stack-pbe-v4-baseline-20260901/luna_v4_sample40.json").read_text()
    )
    length = [len(c["witness"].split()) for c in api["candidates"]]
    runs = {(m, e): load(stem) for stem, m, e in ARMS}
    solved = {
        k: [c["verdict"] == "solved" for c in d["candidates"]] for k, d in runs.items()
    }
    ref = {
        "opus5 xhigh": solved[("claude-opus-5", "xhigh")],
        "astra high": solved[("gpt-6-astra", "high")],
    }
    print(
        "| model | effort | solved | memorised | illegal | shown-only | out tok med / mean | "
        "1-10 | 11-20 | 21-40 | 41+ | agree opus5-xh | agree astra-h | $/task | $/1800 | served | tool calls |"
    )
    print("|" + "---|" * 17)
    rows = []
    for (m, e), d in runs.items():
        cands = d["candidates"]
        gens = [c["generation"][-1] for c in cands]
        out = [g["usage"]["output_tokens"] for g in gens]
        pi, po = PRICE[m]
        cost = statistics.mean(PROMPT_TOKENS * pi / 1e6 + o * po / 1e6 for o in out)
        buckets = []
        for lo, hi in BUCKETS:
            idx = [i for i, n in enumerate(length) if lo <= n <= hi]
            buckets.append(f"{sum(solved[(m, e)][i] for i in idx)}/{len(idx)}")
        agree = [
            sum(a == b for a, b in zip(solved[(m, e)], r, strict=True))
            for r in ref.values()
        ]
        served = sorted({x for g in gens for x in g.get("model", [])}) or ["-"]
        tools = sum(
            g.get("tool_calls", 0) + bool(g.get("text_tool_call")) for g in gens
        )
        print(
            f"| `{m}` | {e} | **{d['solved']}** | {d['memorised']} | {d['illegal']} | "
            f"{d['shown_rate']:.0%} | {statistics.median(out):,.0f} / {statistics.mean(out):,.0f} | "
            + " | ".join(buckets)
            + f" | {agree[0]}/40 | {agree[1]}/40 | {cost:.4f} | {cost * 1800:,.1f} | {', '.join(served)} | {tools} |"
        )
        rows.append(
            {
                "model": m,
                "effort": e,
                "solved": d["solved"],
                "cost_per_task": cost,
                "median_out": statistics.median(out),
                "mean_out": statistics.mean(out),
            }
        )
    (HERE / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
