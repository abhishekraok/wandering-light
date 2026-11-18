"""Retain CLI stdout, served-model fields and completion metadata without copying full transport traces."""

import argparse
import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path


def tool_results(lines):
    """Decode outgoing websocket frames, retaining only tool-result status."""
    decoder = zlib.decompressobj(wbits=-15)
    statuses = []
    for number, line in enumerate(lines):
        if "Sending frame:" not in line:
            continue
        frame = lines[number + 1 : number + 11]
        if "opcode: TEXT" not in frame:
            continue
        payload = next(
            (
                match[1]
                for entry in frame
                if (match := re.fullmatch(r"payload: 0x([0-9a-f]+)", entry))
            ),
            None,
        )
        if payload is None:
            continue
        try:
            data = (
                decoder.decompress(bytes.fromhex(payload) + b"\0\0\xff\xff")
                if "reserved: true false false" in frame
                else bytes.fromhex(payload)
            )
            request = json.loads(data)
        except (ValueError, zlib.error):
            continue
        for item in request.get("input") or []:
            if item.get("type") in ("custom_tool_call_output", "function_call_output"):
                statuses.append(
                    {
                        "type": item["type"],
                        "code_mode_host_disabled": item.get("output")
                        == "code-mode host is disabled",
                    }
                )
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for arm in sorted(args.data_dir.glob("astra62_*_v4_sample40.json")):
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
            lines = re.sub(r"\x1b\[[0-9;]*m", "", payload.decode()).splitlines()
            for line in lines:
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
                        if event.get("type") == "response.completed":
                            usage = container.get("usage") or {}
                            selected["usage"] = {
                                k: usage[k]
                                for k in ("input_tokens", "output_tokens")
                                if k in usage
                            }
                            selected["output_types"] = [
                                item.get("type")
                                for item in container.get("output") or []
                            ]
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
                    "tool_results": tool_results(lines)
                    if any(
                        kind in ("custom_tool_call", "function_call")
                        for event in events
                        for kind in event.get("response", {}).get("output_types", [])
                    )
                    else [],
                }
            )
        arms[arm.stem] = calls
    args.out.write_text(json.dumps(arms, indent=2) + "\n")


if __name__ == "__main__":
    main()
