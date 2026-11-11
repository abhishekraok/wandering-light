"""Independently re-grade saved candidates and audit their original Claude output."""

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from experiments.stack_pbe_baseline import parse_response
from wandering_light.solver import CLAUDE_EXEC_SYSTEM_PROMPT, ClaudeExecTokenGenerator
from wandering_light.stack_pbe import load
from wandering_light.stack_pbe_eval import grade, summarise

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "claude-fable-5-2"
ARTIFACT = "fable52_low_v4_sample40.json"
PRICE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICES = {"input": 8, "output": 40, "write_5m": 10, "write_1h": 16, "read": 0.20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE)
    args = parser.parse_args()
    directory = args.data_dir
    tasks = load(ROOT / "wandering_light/evals/data/stack_pbe_v4")
    selected = [i for i, t in enumerate(tasks) if t.split == "eval"]
    random.Random(17).shuffle(selected)
    selected = selected[:40]
    data_path = directory / ARTIFACT
    data = json.loads(data_path.read_text())
    assert data["tasks"] == data["tasks_planned"] == 40 and not data["partial"]
    assert (data["backend"], data["provider"], data["model"], data["effort"]) == (
        "llm",
        "claude",
        MODEL,
        "low",
    )
    assert data["split"] == "eval"
    assert [c["task_index"] for c in data["candidates"]] == selected
    expected_command = [
        "claude",
        "-p",
        "--model",
        MODEL,
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--system-prompt",
        CLAUDE_EXEC_SYSTEM_PROMPT,
        "--effort",
        "low",
    ]
    preflight = json.loads((directory / "preflight.json").read_text())
    assert preflight["command"] == expected_command
    preflight_text, preflight_meta = ClaudeExecTokenGenerator.parse_result(
        preflight["stdout"]
    )
    assert preflight["passed"] and preflight["returncode"] == 0
    assert preflight["requested_model"] == MODEL and preflight["effort"] == "low"
    assert preflight_meta["model"] == [MODEL] and not preflight_meta["errors"]
    assert preflight_text.strip() == "OK" and preflight_meta["tool_calls"] == 0

    audit = json.loads((directory / "cli-audit.json").read_text())
    assert audit["artifact"] == ARTIFACT and len(audit["calls"]) == 40
    scores, generations, calls, failures, actual_costs, cli_costs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, (candidate, call) in enumerate(
        zip(data["candidates"], audit["calls"], strict=True), 1
    ):
        assert call["call"] == f"call-{index:03d}"
        assert call["command"] == expected_command
        assert (
            hashlib.sha256(call["stdout"].encode()).hexdigest() == call["stdout_sha256"]
        )
        assert len(candidate["generation"]) == 1
        generation = candidate["generation"][0]
        raw = json.loads(call["stdout"])
        message, metadata = ClaudeExecTokenGenerator.parse_result(call["stdout"])
        for key, value in metadata.items():
            assert generation[key] == value, (index, key)
        served = sorted(raw.get("modelUsage") or {})
        assert generation.get("model", []) == served
        assert served in ([], [MODEL]), (index, served)
        if candidate.get("errors"):
            assert metadata["errors"] and candidate["candidate"] == ""
            failures.append(
                {
                    "call": index,
                    "errors": candidate["errors"],
                    "cli_errors": metadata["errors"],
                }
            )
        else:
            assert served == [MODEL] and not metadata["errors"]
            assert raw["stop_reason"] == "end_turn"
        # This is a turn/permission proxy, not an execution trace. Preserve it.
        assert not metadata["text_tool_call"]
        assert not raw.get("permission_denials")
        assert not any((raw.get("usage", {}).get("server_tool_use") or {}).values())
        if metadata["tool_calls"]:
            assert metadata["tool_calls"] == 1
            assert metadata["usage"]["output_tokens"] > 128_000
        task = tasks[candidate["task_index"]]
        score = grade(candidate["candidate"], task)
        if not candidate.get("errors"):
            # Replay the original parser on CLI text, separately from saved candidate.
            assert grade(parse_response(message), task) == score, index
        verdict = (
            "solved" if score.solved else "memorised" if score.memorised else "failed"
        )
        assert candidate["verdict"] == verdict, index
        scores.append(score)
        generations.append(generation)
        usage = raw.get("usage") or {}
        writes = usage.get("cache_creation") or {}
        write_5m = writes.get("ephemeral_5m_input_tokens", 0)
        write_1h = writes.get("ephemeral_1h_input_tokens", 0)
        assert write_5m + write_1h == usage.get("cache_creation_input_tokens", 0)
        actual_costs.append(
            (
                usage.get("input_tokens", 0) * PRICES["input"]
                + usage.get("output_tokens", 0) * PRICES["output"]
                + write_5m * PRICES["write_5m"]
                + write_1h * PRICES["write_1h"]
                + usage.get("cache_read_input_tokens", 0) * PRICES["read"]
            )
            / 1e6
        )
        cli_costs.append(raw.get("total_cost_usd", 0))
        calls.append(
            {
                "call": index,
                "task_index": candidate["task_index"],
                "served_models": served,
                "identity_confirmed": served == [MODEL],
                "verdict": verdict,
                "score": asdict(score),
                "generation_errors": candidate.get("errors", []),
                "cli_errors": metadata["errors"],
                "reported_tool_call_proxy": metadata["tool_calls"],
                "num_turns": raw.get("num_turns"),
                "stop_reason": raw.get("stop_reason"),
                "output_tokens": usage.get("output_tokens"),
                "stdout_sha256": call["stdout_sha256"],
                "stderr_sha256": call["stderr_sha256"],
            }
        )
    checked = summarise(scores)
    for key in ("solved", "memorised", "illegal", "shown_fit"):
        assert getattr(checked, key) == data[key], key
    outputs = [
        g["usage"]["output_tokens"]
        for g in generations
        if g.get("model") == [MODEL] and g.get("usage", {}).get("output_tokens", 0) > 0
    ]
    confirmed = sum(c["identity_confirmed"] for c in calls)
    cost = (
        (1300 * PRICES["input"] + statistics.mean(outputs) * PRICES["output"]) / 1e6
        if len(outputs) == confirmed == 40
        else None
    )
    seconds = sum(g["seconds"] for g in generations)
    row = {
        "model": MODEL,
        "effort": "low",
        "solved": checked.solved,
        "memorised": checked.memorised,
        "illegal": checked.illegal,
        "generation_failures": len(failures),
        "usage_calls": len(outputs),
        "model_confirmed_calls": confirmed,
        "reported_tool_call_proxy": sum(g["tool_calls"] for g in generations),
        "shown_rate": data["shown_rate"],
        "cost_per_task": cost,
        "median_out": statistics.median(outputs) if outputs else None,
        "mean_out": statistics.mean(outputs) if outputs else None,
        "seconds": round(seconds, 1),
        "artifact": ARTIFACT,
    }
    provenance_path = data_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    caveats = []
    digest_matches = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest() == provenance.get("artifact_sha256")
    if not digest_matches:
        caveats.append("Artifact digest differs from inference-time sidecar")
    for name, digest in provenance["inference_source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            caveats.append(f"Current source differs from inference-time source: {name}")
    validation = {
        "model": MODEL,
        "effort": "low",
        "artifact": ARTIFACT,
        "task_indices": selected,
        "attempted_calls": 40,
        "successful_calls": 40 - len(failures),
        "generation_failures": failures,
        "usage_available_calls": len(outputs),
        "model_confirmed_calls": confirmed,
        "regraded_candidates": 40,
        "original_responses_reparsed": 40 - len(failures),
        "tools_disabled_in_command": True,
        "text_tool_call_attempts": 0,
        "reported_tool_call_proxy": row["reported_tool_call_proxy"],
        "preflight_excluded_from_scores_and_cost": True,
        "inference_provenance": provenance,
        "artifact_digest_matches": digest_matches,
        "provenance_caveats": caveats,
        "pricing_source": PRICE_URL,
        "pricing_checked_date": "2026-11-10",
        "prices_usd_per_million": PRICES,
        "plot_prompt_tokens": 1300,
        "recorded_token_list_equivalent_usd": sum(actual_costs),
        "cli_reported_total_cost_usd": sum(cli_costs),
        "calls": calls,
        "rows": [row],
    }
    (directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps([row], indent=2) + "\n")
    print(
        "| effort | solved /40 | memorised | illegal | generation failures | shown-only | usage calls /40 | median / mean recorded output tokens | $/task | call seconds |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    token_text = (
        f"{row['median_out']:,.0f} / {row['mean_out']:,.0f}"
        if outputs
        else "unavailable"
    )
    cost_text = f"{cost:.5f}" if cost is not None else "unavailable"
    print(
        f"| low | {checked.solved} | {checked.memorised} | {checked.illegal} | {len(failures)} | {data['shown_rate']:.1%} | {len(outputs)} | {token_text} | {cost_text} | {seconds:.0f} |"
    )


if __name__ == "__main__":
    main()
