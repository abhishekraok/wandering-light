"""Re-grade the saved candidates and audit protocol/cost before plotting them."""

import argparse
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
from pathlib import Path

from wandering_light.stack_pbe import load
from wandering_light.stack_pbe_eval import grade, summarise

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODEL = "claude-sonnet-5-5"
PRICE_URL = "https://www.anthropic.com/claude-sonnet-5-5"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    tasks = load(ROOT / "wandering_light/evals/data/stack_pbe_v4")
    selected = [i for i, t in enumerate(tasks) if t.split == "eval"]
    random.Random(17).shuffle(selected)
    selected = selected[:40]
    rows, failures = [], []
    print(
        "| effort | solved /40 | memorised | illegal | generation failures | shown-only | usage calls /40 | median / mean recorded output tokens | $/task | call seconds |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for effort in EFFORTS:
        name = f"sonnet55_{effort}_v4_sample40.json"
        d = json.loads((HERE / name).read_text())
        assert d["tasks"] == d["tasks_planned"] == 40 and not d["partial"]
        assert (d["model"], d["effort"], d["provider"]) == (MODEL, effort, "claude")
        assert [c["task_index"] for c in d["candidates"]] == selected
        scores, generations = [], []
        for c in d["candidates"]:
            assert len(c["generation"]) == 1
            g = c["generation"][0]
            assert g.get("model", []) in ([], [MODEL]), g
            assert not g["text_tool_call"], g
            if g["tool_calls"]:
                # This adapter derives the field from num_turns - 1 plus
                # permission denials. It is not an execution trace.
                assert c.get("errors") and g["stop_reason"] == "refusal", g
            if c.get("errors"):
                assert g["errors"] and c["candidate"] == "", c
                assert c["verdict"] == "failed", c
                failures.append(
                    {
                        "effort": effort,
                        "task_index": c["task_index"],
                        "errors": g["errors"],
                        "stop_reason": g.get("stop_reason"),
                        "reported_tool_call_proxy": g["tool_calls"],
                        "seconds": g["seconds"],
                    }
                )
            else:
                assert g["model"] == [MODEL], g
                assert not g["errors"] and g["stop_reason"] == "end_turn", g
                assert g["usage"]["output_tokens"] > 0, g
            score = grade(c["candidate"], tasks[c["task_index"]])
            verdict = (
                "solved"
                if score.solved
                else "memorised"
                if score.memorised
                else "failed"
            )
            assert verdict == c["verdict"], c
            scores.append(score)
            generations.append(g)
        checked = summarise(scores)
        for key in ("solved", "memorised", "illegal", "shown_fit"):
            assert getattr(checked, key) == d[key]
        outputs = [
            g["usage"]["output_tokens"]
            for g in generations
            if g.get("model") == [MODEL]
            and g.get("usage", {}).get("output_tokens", 0) > 0
        ]
        seconds = sum(g["seconds"] for g in generations)
        # Same estimate as the existing plot: fixed 1.3k-token task prompt,
        # all recorded output (including thinking), no CLI residual or cache.
        cost = (
            (1300 * 2 + statistics.mean(outputs) * 10) / 1e6
            if len(outputs) == 40
            else None
        )
        row = {
            "model": MODEL,
            "effort": effort,
            "solved": d["solved"],
            "memorised": d["memorised"],
            "illegal": d["illegal"],
            "generation_failures": sum(bool(c.get("errors")) for c in d["candidates"]),
            "reported_tool_call_proxy": sum(g["tool_calls"] for g in generations),
            "usage_calls": len(outputs),
            "model_confirmed_calls": sum(
                g.get("model") == [MODEL] for g in generations
            ),
            "shown_rate": d["shown_rate"],
            "cost_per_task": cost,
            "median_out": statistics.median(outputs) if outputs else None,
            "mean_out": statistics.mean(outputs) if outputs else None,
            "seconds": round(seconds, 1),
            "artifact": name,
        }
        rows.append(row)
        token_text = (
            f"{row['median_out']:,.0f} / {row['mean_out']:,.0f}"
            if outputs
            else "unavailable"
        )
        cost_text = f"{cost:.5f}" if cost is not None else "unavailable"
        print(
            f"| {effort} | {d['solved']} | {d['memorised']} | {d['illegal']} | {row['generation_failures']} | {d['shown_rate']:.1%} | "
            f"{len(outputs)} | {token_text} | {cost_text} | {seconds:.0f} |"
        )
    provenance = {
        "benchmark_base_commit": "52131dd",  # recorded original run base
        "analysis_python": platform.python_version(),
        "analysis_claude": (
            subprocess.check_output(["claude", "--version"], text=True).strip()
            if shutil.which("claude")
            else None
        ),
        "pricing_source": PRICE_URL,
        "input_usd_per_million": 2,
        "output_usd_per_million": 10,
        "plot_prompt_tokens": 1300,
        "task_indices": selected,
        "analysis_source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "experiments/stack_pbe_baseline.py",
                "wandering_light/solver.py",
                "wandering_light/stack_machine.py",
                "wandering_light/stack_pbe_eval.py",
                "wandering_light/evals/data/stack_pbe_v4/manifest.json",
                "uv.lock",
            )
        },
        "attempted_calls": len(rows) * 40,
        "successful_calls": len(rows) * 40 - len(failures),
        "generation_failures": failures,
        "usage_available_calls": sum(r["usage_calls"] for r in rows),
        "model_confirmed_calls": sum(r["model_confirmed_calls"] for r in rows),
        "regraded_candidates": len(rows) * 40,
        "reported_tool_call_proxy": sum(r["reported_tool_call_proxy"] for r in rows),
        "tools_disabled_in_command": True,
        "text_tool_call_attempts": 0,
        "served_models": [MODEL],
        "rows": rows,
    }
    (HERE / "validation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (HERE / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
