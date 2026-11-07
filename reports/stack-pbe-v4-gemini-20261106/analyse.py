"""Join the 4 shards per Gemini arm; price each call; write summary.json and table.md.

Same cost convention as ../stack-pbe-v4-effort-sweep-20260927/analyse.py: list
price x (1.3k-token prompt + recorded output tokens, thinking included).
Usage: analyse.py [OUT]  (default: the main checkout's checkpoints dir).
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

OUT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else Path.home() / "repos/wandering-light/checkpoints/gemini-sweep-20261106"
)
PROMPT_TOKENS = 1300
PRICE = {  # $ per 1M input, output (thinking billed as output); aistudio pricing page, 2026-11-06
    "gemini-4-argon": (2, 10),  # through 2026-11-30; $4/$20 from 2026-12-01
    "gemini-3.8-flash": (0.75, 3.75),  # through 2026-12-31; $1.50/$7.50 from 2027-01-01
}
ARMS = [
    (f"{short}_{effort}", model, effort)
    for model, short in (("gemini-4-argon", "argon4"), ("gemini-3.8-flash", "flash38"))
    for effort in ("low", "medium", "high")
]


def main() -> None:
    rows, lines = [], []
    lines.append(
        "| model | effort | solved | memorised | tasks | errors | retried (first error -> failed again) | out tok med / mean | "
        "think tok mean | prompt tok med | $/task | finish | served | s/call med |"
    )
    lines.append("|" + "---|" * 14)
    for tag, model, effort in ARMS:
        shards = [OUT / f"{tag}_v4_sample40_shard{k}.json" for k in range(4)]
        if not all(p.exists() for p in shards):
            print(f"{tag}: {sum(p.exists() for p in shards)}/4 shards", file=sys.stderr)
            continue
        runs = [json.loads(p.read_text()) for p in shards]
        cands = [c for r in runs for c in r["candidates"]]
        partial = any(r["partial"] for r in runs)
        gens = [g for c in cands for g in c.get("generation", [])]
        out = [g["usage"]["output_tokens"] for g in gens]
        think = [g["usage"]["thinking_tokens"] for g in gens]
        prompt = [g["usage"]["input_tokens"] for g in gens]
        pi, po = PRICE[model]
        cost = statistics.mean(PROMPT_TOKENS * pi / 1e6 + o * po / 1e6 for o in out)
        errors = Counter(e for c in cands for e in c.get("errors", []))
        retried = [c["retry"] for c in cands if "retry" in c]
        first = Counter(e for r in retried for e in r["first_errors"])
        again = Counter(r["error"] for r in retried if "error" in r)
        solved = sum(r["solved"] for r in runs)
        memorised = sum(r["memorised"] for r in runs)
        served = sorted({m for g in gens for m in g["model"]})
        finish = Counter(g["stop_reason"] for g in gens)
        secs = statistics.median(g["seconds"] for g in gens)
        lines.append(
            f"| `{model}` | {effort} | **{solved}** | {memorised} | {len(cands)}"
            f"{' PARTIAL' if partial else ''} | {dict(errors) or 0} | "
            f"{len(retried)} ({dict(first) or '-'} -> {dict(again) or 0}) | "
            f"{statistics.median(out):,.0f} / {statistics.mean(out):,.0f} | "
            f"{statistics.mean(think):,.0f} | {statistics.median(prompt):,.0f} | "
            f"{cost:.4f} | {dict(finish)} | {', '.join(served)} ({len(gens)}/{len(cands)}) | {secs:.0f} |"
        )
        rows.append(
            {
                "model": model,
                "effort": effort,
                "solved": solved,
                "tasks": len(cands),
                "partial": partial,
                "cost_per_task": cost,
                "usage_calls": len(gens),
                "errors": dict(errors),
                "retried": len(retried),
                "failed_again": dict(again),
                "median_out": statistics.median(out),
                "mean_out": statistics.mean(out),
            }
        )
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
