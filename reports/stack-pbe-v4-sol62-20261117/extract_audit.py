"""Retain CLI stdout and server model fields without copying full transport traces."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for arm in sorted(args.data_dir.glob("sol62_*_v4_sample40.json")):
        calls = []
        for command in sorted(
            (args.data_dir / "raw" / arm.stem).glob("*.command.json")
        ):
            stem = command.name.removesuffix(".command.json")
            stderr = command.parent / (stem + ".stderr.txt")
            payload = (
                stderr.read_bytes()
                if stderr.exists()
                else gzip.decompress(stderr.with_suffix(".txt.gz").read_bytes())
            )
            events = []
            for line in re.sub(r"\x1b\[[0-9;]*m", "", payload.decode()).splitlines():
                marker = next(
                    (m for m in ("SSE event: ", "Received message ") if m in line), None
                )
                if marker is None:
                    continue
                try:
                    event = json.loads(line.split(marker, 1)[1])
                except ValueError:
                    continue
                evidence = {"type": event.get("type")}
                for key in ("headers", "response"):
                    container = event.get(key) or {}
                    if key == "response":
                        selected = {
                            k: container[k] for k in ("id", "model") if k in container
                        }
                        headers = container.get("headers") or {}
                    else:
                        selected, headers = {}, container
                    model_headers = {
                        k: v
                        for k, v in headers.items()
                        if k.lower() in ("openai-model", "x-openai-model")
                    }
                    if model_headers:
                        selected.update(
                            {"headers": model_headers}
                            if key == "response"
                            else model_headers
                        )
                    if selected:
                        evidence[key] = selected
                if "response" in evidence or "headers" in evidence:
                    events.append(evidence)
            calls.append(
                {
                    "call": stem,
                    "command": json.loads(command.read_text()),
                    "stdout": (command.parent / (stem + ".stdout.jsonl")).read_text(),
                    "stderr_sha256": hashlib.sha256(payload).hexdigest(),
                    "received_model_events": events,
                }
            )
        arms[arm.stem] = calls
    args.out.write_text(json.dumps(arms, indent=2) + "\n")


if __name__ == "__main__":
    main()
