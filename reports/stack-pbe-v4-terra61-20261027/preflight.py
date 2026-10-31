"""One lowest-effort serving-model check, excluded from benchmark scores."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from wandering_light.solver import CodexExecTokenGenerator

MODEL = "gpt-6.1-terra"
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite existing preflight evidence")
    raw = args.out.parent / "raw" / args.out.stem
    raw.mkdir(parents=True, exist_ok=False)
    os.environ["RUST_LOG"] = os.environ.get("RUST_LOG") or (
        "codex_api::sse::responses=trace,tungstenite::protocol=trace"
    )
    generator = CodexExecTokenGenerator(MODEL, effort="low")
    command = generator.command()
    prompt = "Reply exactly OK."
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        env=generator.env,
        timeout=900,
        check=False,
    )
    (raw / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (raw / "stdout.jsonl").write_text(result.stdout)
    (raw / "stderr.txt").write_text(result.stderr)
    models, events = set(), []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", result.stderr).splitlines():
        marker = next(
            (m for m in ("SSE event: ", "Received message ") if m in line), None
        )
        if marker is None:
            continue
        try:
            event = json.loads(line.split(marker, 1)[1])
        except ValueError:
            continue
        response = event.get("response") or {}
        selected = {k: response[k] for k in ("id", "model") if k in response}
        if isinstance(response.get("model"), str):
            models.add(response["model"])
        for container in (event, response):
            for key, value in (container.get("headers") or {}).items():
                if key.lower() not in ("openai-model", "x-openai-model"):
                    continue
                model_value = value[0] if isinstance(value, list) and value else value
                if isinstance(model_value, str):
                    models.add(model_value)
                    selected.setdefault("model_headers", {})[key] = model_value
        if selected:
            events.append({"type": event.get("type"), "response": selected})
    _, meta = generator.parse_events(result.stdout.splitlines())
    passed = (
        result.returncode == 0
        and models == {MODEL}
        and meta.get("tool_calls") == 0
        and bool(meta.get("usage"))
    )
    audit = {
        "purpose": "Separate lowest-effort preflight; excluded from evaluation",
        "requested_model": MODEL,
        "effort": "low",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cli_version": subprocess.check_output(
            ["codex", "--version"], text=True
        ).strip(),
        "identity_trace_logging": os.environ.get("RUST_LOG"),
        "command": command,
        "prompt": prompt,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "received_model_events": events,
        "served_models": sorted(models),
        "parsed_metadata": meta,
        "passed": passed,
    }
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"passed": passed, "served_models": sorted(models)}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
