"""Freeze shards and supervise validators under one external 12 GiB scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def prepare(paths: list[Path], output_dir: Path, workers: int = 8) -> dict:
    if workers < 1 or not paths:
        raise ValueError("positive workers and at least one input are required")
    output_dir.mkdir(parents=True, exist_ok=False)
    shards = [output_dir / f"input-{i:02d}.jsonl" for i in range(workers)]
    handles = [path.open("wb") for path in shards]
    seen, sources, counts = set(), {}, [0] * workers
    try:
        for path in sorted({path.resolve() for path in paths}):
            data = path.read_bytes()
            sources[str(path)] = hashlib.sha256(data).hexdigest()
            for line in data.splitlines():
                if not line.strip():
                    continue
                candidate = json.loads(line)
                canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
                key = hashlib.sha256(canonical.encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                index = int(key, 16) % workers
                handles[index].write(line + b"\n")
                counts[index] += 1
    finally:
        for handle in handles:
            handle.close()
    manifest = {
        "sources": sources,
        "unique_candidates": len(seen),
        "workers": workers,
        "shards": [
            {
                "path": str(path.resolve()),
                "candidates": count,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, count in zip(shards, counts, strict=True)
        ],
    }
    _save(output_dir / "inputs.json", manifest)
    return manifest


def _progress(root: Path, workers: int) -> dict:
    totals = dict.fromkeys(("candidates", "accepted", "rejected"), 0)
    for index in range(workers):
        path = root / f"shard-{index}" / "summary.json"
        if path.exists():
            report = json.loads(path.read_text())
            for key in totals:
                totals[key] += report[key]
    return totals


def supervise(
    manifest: dict,
    output_dir: Path,
    validation_root: Path,
    seed: int = 20261024,
    dev_share: float = 0.0,
    poll_seconds: float = 10,
) -> dict:
    processes, logs, children = [], [], []
    started = time.monotonic()
    baseline = _progress(validation_root, manifest["workers"])
    previous_handler = signal.signal(
        signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum)
    )
    try:
        for index, shard in enumerate(manifest["shards"]):
            command = [sys.executable, "-m", "experiments.pbe_scale.validate"]
            command.extend(("--input", shard["path"]))
            command.extend(("--output-dir", str(validation_root / f"shard-{index}")))
            command.extend(("--seed", str(seed), "--dev-share", str(dev_share)))
            log_path = output_dir / f"worker-{index:02d}.log"
            log = log_path.open("wb")
            logs.append(log)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            children.append(
                {"pid": process.pid, "command": command, "log": str(log_path)}
            )
            _save(output_dir / "processes.json", {"children": children})
        while True:
            codes = [process.poll() for process in processes]
            progress = _progress(validation_root, manifest["workers"])
            progress.update(
                running=codes.count(None), seconds=round(time.monotonic() - started, 1)
            )
            print(json.dumps(progress), flush=True)
            if any(code not in (None, 0) for code in codes) or all(
                code is not None for code in codes
            ):
                break
            time.sleep(poll_seconds)
    finally:
        for process in processes:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.kill()
                process.wait()
        for log in logs:
            log.close()
        codes = [process.returncode for process in processes]
        result = {
            "status": "complete"
            if len(codes) == manifest["workers"] and all(code == 0 for code in codes)
            else "incomplete",
            "returncodes": codes,
            "before": baseline,
            "after": _progress(validation_root, manifest["workers"]),
            "children": children,
        }
        _save(output_dir / "result.json", result)
        signal.signal(signal.SIGTERM, previous_handler)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dev-share", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = prepare(args.input, args.output_dir, args.workers)
    if args.prepare_only:
        return 0
    result = supervise(
        manifest, args.output_dir, args.validation_root, args.seed, args.dev_share
    )
    print(
        json.dumps({key: result[key] for key in ("status", "returncodes", "after")}),
        flush=True,
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
