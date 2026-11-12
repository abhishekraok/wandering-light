"""One lowest-effort serving-model check, excluded from benchmark scores."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from wandering_light.solver import ClaudeExecTokenGenerator

MODEL = "claude-fable-5-2"
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite existing preflight evidence")
    raw = args.out.parent / "raw" / args.out.stem
    raw.mkdir(parents=True, exist_ok=False)
    generator = ClaudeExecTokenGenerator(MODEL, effort="low")
    command = generator.command()
    prompt = "Reply exactly OK."
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=generator.workdir,
        env=generator.env,
        timeout=generator.timeout,
        check=False,
    )
    (raw / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (raw / "stdout.json").write_text(result.stdout)
    (raw / "stderr.txt").write_text(result.stderr)
    message, meta = generator.parse_result(result.stdout)
    served = meta.get("model", [])
    passed = (
        result.returncode == 0
        and served == [MODEL]
        and not meta.get("errors")
        and meta.get("tool_calls") == 0
        and not meta.get("text_tool_call")
        and meta.get("usage", {}).get("output_tokens", 0) > 0
        and meta.get("stop_reason") == "end_turn"
        and message.strip() == "OK"
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
            ["claude", "--version"], text=True
        ).strip(),
        "command": command,
        "prompt": prompt,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "served_models": served,
        "parsed_metadata": meta,
        "passed": passed,
    }
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"passed": passed, "served_models": served}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
