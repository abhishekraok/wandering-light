"""Independently regrade completed fixed-sample arms and audit raw CLI evidence."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from wandering_light.solver import CodexExecTokenGenerator
from wandering_light.stack_pbe import load
from wandering_light.stack_pbe_eval import grade, summarise

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "gpt-6.1-terra"
EFFORTS = ("low", "medium", "high", "xhigh")


def inference_provenance(path):
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        return None, ["Inference provenance was not recorded for this arm."]
    provenance = json.loads(sidecar.read_text())
    caveats = []
    if (
        provenance.get("artifact_sha256")
        != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        caveats.append("Inference provenance artifact digest is missing or differs.")
    if provenance.get("source_dirty") is not False:
        caveats.append("Inference checkout was dirty or its dirty flag is unknown.")
    for name, digest in provenance.get("inference_source_sha256", {}).items():
        source = ROOT / name
        if (
            not source.exists()
            or hashlib.sha256(source.read_bytes()).hexdigest() != digest
        ):
            caveats.append(f"Analysis source differs from inference: {name}")
    return provenance, caveats


def server_models(call):
    """Read server response fields, never the local command's requested model."""
    models = set()
    for event in call["received_model_events"]:
        response = event.get("response") or {}
        for container in (event, response):
            headers = container.get("headers") or {}
            for key, value in headers.items():
                if key.lower() in ("openai-model", "x-openai-model"):
                    model_value = (
                        value[0] if isinstance(value, list) and value else value
                    )
                    if isinstance(model_value, str):
                        models.add(model_value)
        if isinstance(response.get("model"), str):
            models.add(response["model"])
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.data_dir
    evidence = json.loads((out / "cli-audit.json").read_text())
    tasks = load(ROOT / "wandering_light/evals/data/stack_pbe_v4")
    selected = [i for i, t in enumerate(tasks) if t.split == "eval"]
    random.Random(17).shuffle(selected)
    selected = selected[:40]
    rows, audits = [], []
    print(
        "| effort | solved /40 | memorised | illegal | usage calls | mean output tokens | $/task | server identity confirmed /40 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for effort in EFFORTS:
        path = out / f"terra61_{effort}_v4_sample40.json"
        if not path.exists():
            continue
        provenance, caveats = inference_provenance(path)
        data = json.loads(path.read_text())
        assert data["tasks"] == data["tasks_planned"] == 40 and not data["partial"]
        assert (data["model"], data["provider"], data["effort"]) == (
            MODEL,
            "codex",
            effort,
        )
        assert [c["task_index"] for c in data["candidates"]] == selected
        scores, outputs, calls = [], [], []
        for number, candidate in enumerate(data["candidates"], 1):
            assert len(candidate["generation"]) == 1
            generation = candidate["generation"][0]
            score = grade(candidate["candidate"], tasks[candidate["task_index"]])
            verdict = (
                "solved"
                if score.solved
                else "memorised"
                if score.memorised
                else "failed"
            )
            assert verdict == candidate["verdict"]
            scores.append(score)
            usage = generation.get("usage", {})
            if "output_tokens" in usage:
                outputs.append(usage["output_tokens"])
            call = evidence[path.stem][number - 1]
            assert call["call"] == f"call-{number:03d}"
            command = call["command"]
            assert command[command.index("-m") + 1] == MODEL
            assert f'model_reasoning_effort="{effort}"' in command
            stdout = call["stdout"]
            _, raw_meta = CodexExecTokenGenerator.parse_events(stdout.splitlines())
            for key in ("usage", "tool_calls", "errors"):
                assert raw_meta.get(key) == generation.get(key)
            models = server_models(call)
            calls.append(
                {
                    "task_index": candidate["task_index"],
                    "server_models": models,
                    "identity_confirmed": models == [MODEL],
                    "tool_calls": generation.get("tool_calls", 0),
                    "cli_error_events": generation.get("errors", []),
                    "generation_exceptions": candidate.get("errors", []),
                }
            )
        checked = summarise(scores)
        for key in ("solved", "memorised", "illegal", "shown_fit"):
            assert getattr(checked, key) == data[key]
        cost = (
            (1300 + statistics.mean(outputs) * 5) / 1e6 if len(outputs) == 40 else None
        )
        confirmed = sum(c["identity_confirmed"] for c in calls)
        row = {
            "model": MODEL,
            "effort": effort,
            "solved": data["solved"],
            "memorised": data["memorised"],
            "illegal": data["illegal"],
            "usage_calls": len(outputs),
            "mean_out": statistics.mean(outputs) if outputs else None,
            "cost_per_task": cost,
            "model_confirmed_calls": confirmed,
        }
        rows.append(row)
        audits.append(
            {
                "effort": effort,
                "artifact": path.name,
                "regraded_candidates": 40,
                "saved_grades_agree": True,
                "inference_provenance": provenance,
                "provenance_caveats": caveats,
                "calls": calls,
            }
        )
        cost_text = f"{cost:.6f}" if cost is not None else "unavailable"
        mean_text = f"{row['mean_out']:.1f}" if outputs else "unavailable"
        print(
            f"| {effort} | {data['solved']} | {data['memorised']} | {data['illegal']} | "
            f"{len(outputs)} | {mean_text} | {cost_text} | {confirmed} |"
        )
    validation = {
        "requested_model": MODEL,
        "task_indices": selected,
        "cli_audit_sha256": hashlib.sha256(
            (out / "cli-audit.json").read_bytes()
        ).hexdigest(),
        "expected_efforts": list(EFFORTS),
        "completed_efforts": [r["effort"] for r in rows],
        "plot_prompt_tokens": 1300,
        "input_usd_per_million": 1,
        "output_usd_per_million": 5,
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
        "pricing_checked": "2026-10-27",
        "regraded_candidates": len(rows) * 40,
        "model_confirmed_calls": sum(r["model_confirmed_calls"] for r in rows),
        "all_server_identities_confirmed": bool(rows)
        and all(r["model_confirmed_calls"] == 40 for r in rows),
        "analysis_source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "experiments/stack_pbe_baseline.py",
                "wandering_light/solver.py",
                "wandering_light/stack_pbe_eval.py",
                "wandering_light/stack_machine.py",
                "wandering_light/evals/data/stack_pbe_v4/manifest.json",
            )
        },
        "arms": audits,
    }
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
