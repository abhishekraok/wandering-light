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
PRICE = {  # $ per 1M input, output; Sonnet 5.5 verified 2026-09-28, 6.1 Sol 2026-09-29
    "gpt-6.1-sol": (2, 10),
    "gpt-6-astra": (10, 50),
    "gpt-6-sol": (2, 10),
    "gpt-6-luna": (0.1, 0.5),
    "gpt-5.6-sol": (4, 20),
    "gpt-5.6-terra": (2, 12),
    "gpt-5.6-luna": (0.2, 1.2),
    "claude-opus-5": (5, 25),
    "claude-opus-5-5": (4, 20),
    "claude-sonnet-5-5": (2, 10),
    "claude-fable-5": (10, 50),
    "claude-fable-5-1": (15, 75),  # inferred from claude -p costUSD
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
    ("stack-pbe-v4-new-models-20260926/sol6_low", "gpt-6-sol", "low"),
    ("stack-pbe-v4-new-models-20260926/sol6_medium", "gpt-6-sol", "medium"),
    ("stack-pbe-v4-new-models-20260926/sol6_high", "gpt-6-sol", "high"),
    ("stack-pbe-v4-new-models-20260926/sol6_xhigh", "gpt-6-sol", "xhigh"),
    ("stack-pbe-v4-new-models-20260926/sol6_max", "gpt-6-sol", "max"),
    ("stack-pbe-v4-new-models-20260926/luna6_low", "gpt-6-luna", "low"),
    ("stack-pbe-v4-new-models-20260926/luna6_medium", "gpt-6-luna", "medium"),
    ("stack-pbe-v4-new-models-20260926/luna6_high", "gpt-6-luna", "high"),
    ("stack-pbe-v4-new-models-20260926/luna6_xhigh", "gpt-6-luna", "xhigh"),
    ("stack-pbe-v4-new-models-20260926/luna6_max", "gpt-6-luna", "max"),
    ("stack-pbe-v4-new-models-20260926/opus55_low", "claude-opus-5-5", "low"),
    ("stack-pbe-v4-new-models-20260926/opus55_medium", "claude-opus-5-5", "medium"),
    ("stack-pbe-v4-new-models-20260926/opus55_high", "claude-opus-5-5", "high"),
    ("stack-pbe-v4-new-models-20260926/opus55_xhigh", "claude-opus-5-5", "xhigh"),
    ("stack-pbe-v4-effort-sweep-20260927/sol56_low", "gpt-5.6-sol", "low"),
    ("stack-pbe-v4-effort-sweep-20260927/sol56_medium", "gpt-5.6-sol", "medium"),
    ("stack-pbe-v4-effort-sweep-20260927/sol56_high", "gpt-5.6-sol", "high"),
    ("stack-pbe-v4-effort-sweep-20260927/sol56_xhigh", "gpt-5.6-sol", "xhigh"),
    ("stack-pbe-v4-effort-sweep-20260927/sol56_max", "gpt-5.6-sol", "max"),
    ("stack-pbe-v4-effort-sweep-20260927/terra56_low", "gpt-5.6-terra", "low"),
    ("stack-pbe-v4-effort-sweep-20260927/terra56_medium", "gpt-5.6-terra", "medium"),
    ("stack-pbe-v4-effort-sweep-20260927/terra56_xhigh", "gpt-5.6-terra", "xhigh"),
    ("stack-pbe-v4-effort-sweep-20260927/terra56_max", "gpt-5.6-terra", "max"),
    ("stack-pbe-v4-effort-sweep-20260927/luna56_low", "gpt-5.6-luna", "low"),
    ("stack-pbe-v4-effort-sweep-20260927/luna56_medium", "gpt-5.6-luna", "medium"),
    ("stack-pbe-v4-effort-sweep-20260927/luna56_xhigh", "gpt-5.6-luna", "xhigh"),
    ("stack-pbe-v4-effort-sweep-20260927/luna56_max", "gpt-5.6-luna", "max"),
    ("stack-pbe-v4-effort-sweep-20260927/fable5_low", "claude-fable-5", "low"),
    ("stack-pbe-v4-effort-sweep-20260927/fable5_medium", "claude-fable-5", "medium"),
    ("stack-pbe-v4-effort-sweep-20260927/fable5_high", "claude-fable-5", "high"),
    ("stack-pbe-v4-effort-sweep-20260927/fable5_xhigh", "claude-fable-5", "xhigh"),
    ("stack-pbe-v4-effort-sweep-20260927/fable5_max", "claude-fable-5", "max"),
    ("stack-pbe-v4-effort-sweep-20260927/fable51_low", "claude-fable-5-1", "low"),
    ("stack-pbe-v4-effort-sweep-20260927/fable51_medium", "claude-fable-5-1", "medium"),
    ("stack-pbe-v4-effort-sweep-20260927/fable51_high", "claude-fable-5-1", "high"),
    ("stack-pbe-v4-effort-sweep-20260927/fable51_xhigh", "claude-fable-5-1", "xhigh"),
    ("stack-pbe-v4-effort-sweep-20260927/fable51_max", "claude-fable-5-1", "max"),
    ("stack-pbe-v4-sonnet55-20260928/sonnet55_low", "claude-sonnet-5-5", "low"),
    ("stack-pbe-v4-sonnet55-20260928/sonnet55_medium", "claude-sonnet-5-5", "medium"),
    ("stack-pbe-v4-sonnet55-20260928/sonnet55_high", "claude-sonnet-5-5", "high"),
    ("stack-pbe-v4-sonnet55-20260928/sonnet55_xhigh", "claude-sonnet-5-5", "xhigh"),
    ("stack-pbe-v4-sonnet55-20260928/sonnet55_max", "claude-sonnet-5-5", "max"),
    ("stack-pbe-v4-sol61-20260929/sol61_low", "gpt-6.1-sol", "low"),
    ("stack-pbe-v4-sol61-20260929/sol61_medium", "gpt-6.1-sol", "medium"),
    ("stack-pbe-v4-sol61-20260929/sol61_high", "gpt-6.1-sol", "high"),
    ("stack-pbe-v4-sol61-20260929/sol61_xhigh", "gpt-6.1-sol", "xhigh"),
    ("stack-pbe-v4-sol61-20260929/sol61_max", "gpt-6.1-sol", "max"),
)


def load(stem: str) -> dict:
    return json.loads((REPORTS / f"{stem}_v4_sample40.json").read_text())


def main() -> None:
    api = json.loads(
        (REPORTS / "stack-pbe-v4-baseline-20260901/luna_v4_sample40.json").read_text()
    )
    length = [len(c["witness"].split()) for c in api["candidates"]]
    runs = {
        (m, e): load(stem)
        for stem, m, e in ARMS
        if (REPORTS / f"{stem}_v4_sample40.json").exists()
    }
    runs = {
        key: run
        for key, run in runs.items()
        if not run.get("partial") and len(run["candidates"]) == 40
    }
    solved = {
        k: [c["verdict"] == "solved" for c in d["candidates"]] for k, d in runs.items()
    }
    ref = {
        "opus5 xhigh": solved[("claude-opus-5", "xhigh")],
        "astra high": solved[("gpt-6-astra", "high")],
    }
    print(
        "| model | effort | solved | memorised | illegal | shown-only | out tok med / mean | "
        "1-10 | 11-20 | 21-40 | 41+ | agree opus5-xh | agree astra-h | $/task | $/1800 | served | tool/turn proxy |"
    )
    print("|" + "---|" * 17)
    rows = []
    for (m, e), d in runs.items():
        cands = d["candidates"]
        gens = [c["generation"][-1] for c in cands]
        # a call killed at the 900 s deadline reports no usage; it is left out
        # of the token and cost figures, not counted as zero
        out = [g["usage"]["output_tokens"] for g in gens if "usage" in g]
        if m == "claude-sonnet-5-5":
            out = [
                g["usage"]["output_tokens"]
                for g in gens
                if g.get("model") == [m]
                and g.get("usage", {}).get("output_tokens", 0) > 0
            ]
        pi, po = PRICE[m]
        cost = (
            None
            if m == "claude-sonnet-5-5" and len(out) != 40
            else statistics.mean(PROMPT_TOKENS * pi / 1e6 + o * po / 1e6 for o in out)
        )
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
        token_text = (
            f"{statistics.median(out):,.0f} / {statistics.mean(out):,.0f}"
            if out
            else "unavailable"
        )
        if m == "claude-sonnet-5-5" and len(out) != 40:
            token_text += f" ({len(out)}/40 calls)"
        cost_text = (
            f"{cost:.4f} | {cost * 1800:,.1f}"
            if cost is not None
            else "unavailable | unavailable"
        )
        print(
            f"| `{m}` | {e} | **{d['solved']}** | {d['memorised']} | {d['illegal']} | "
            f"{d['shown_rate']:.0%} | {token_text} | "
            + " | ".join(buckets)
            + f" | {agree[0]}/40 | {agree[1]}/40 | {cost_text} | {', '.join(served)} | {tools} |"
        )
        rows.append(
            {
                "model": m,
                "effort": e,
                "solved": d["solved"],
                "cost_per_task": cost,
                "median_out": statistics.median(out) if out else None,
                "mean_out": statistics.mean(out) if out else None,
            }
        )
        if m == "claude-sonnet-5-5":
            rows[-1]["usage_calls"] = len(out)
            rows[-1]["refusals"] = sum(g.get("stop_reason") == "refusal" for g in gens)
    (HERE / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
