"""Local finisher: run once under one MemoryMax=12G / MemorySwapMax=0 scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(PACIFIC).isoformat()


def save(path: Path, state: dict) -> None:
    state["updated_at"] = now()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def wait_ready(run: Path, status_path: Path, state: dict, poll: float) -> None:
    while True:
        initial = read(run / "v3-round-000/result.json")
        campaign = read(run / "campaign_status.json")
        if initial.get("status") == "incomplete":
            raise RuntimeError("initial v3 validation is incomplete")
        if initial.get("status") == "complete" and campaign.get("status") == "stopped":
            return
        state.update(
            status="waiting",
            waiting_for={
                "initial_validation": initial.get("status"),
                "author_campaign": campaign.get("status"),
            },
        )
        save(status_path, state)
        time.sleep(poll)


def plan(run: Path) -> list[tuple[str, list[str]]]:
    authors = sorted((run / "candidates").glob("batch-*.jsonl"))
    if not authors:
        raise RuntimeError("no completed author candidate files")
    common = "--arm 1m --seed 0 --epochs 3 --learning-rate 0.000125 --batch-size 256 --device cuda --wandb-mode online"
    shards = " ".join(f"@validated-v3/shard-{i}" for i in range(8))
    specs = (
        (
            "final-authors",
            "verify_batch",
            "--input @authors --output-dir @v3-round-001 --validation-root @validated-v3 --workers 8 --seed 20261024 --dev-share 0",
        ),
        (
            "final-anchors",
            "validate",
            "--input @anchor-candidates.jsonl --output-dir @anchor-validation-v3-final --seed 20261024 --dev-share 0",
        ),
        ("merge", "merge", shards + " @anchor-validation-v3-final --output @corpus-v3"),
        (
            "audit",
            "audit_eval",
            "--accepted @corpus-v3/accepted.jsonl --output-dir @audit-v4-eval --filtered-tasks @audit-v4-eval/tasks_train_filtered.jsonl.gz --workers 8 --seed 188",
        ),
        (
            "augment",
            "augment",
            "--tasks @audit-v4-eval/tasks_train_filtered.jsonl.gz --output @tasks_train_16views.jsonl.gz --views 16 --seed 0",
        ),
        (
            "report",
            "report",
            "--corpus @corpus-v3 --campaign @ --output-dir @final-report --audit @audit-v4-eval/exclusions.json --view-manifest @tasks_train_16views.jsonl.gz.manifest.json",
        ),
        (
            "prepare-new",
            "train",
            "prepare --input @audit-v4-eval/tasks_train_filtered.jsonl.gz --output-dir @new-fit-packed32 --prefixes-per-task 32 --seed 0",
        ),
        (
            "train-old",
            "train",
            "train --data @baseline-fit-packed32 --output-dir @policies/old-staged-1m-s0 "
            + common,
        ),
        (
            "train-mixed",
            "train",
            "train --data @new-fit-packed32 --mix-data @baseline-fit-packed32 --mix-share 0.2 --output-dir @policies/mixed-1m-s0 "
            + common,
        ),
    )
    result = []
    for name, module, template in specs:
        command = [sys.executable, "-m", f"experiments.pbe_scale.{module}"]
        # Expand whole argv items; paths are never parsed as shell text.
        for word in template.split():
            if word == "@authors":
                command.extend(map(str, authors))
            else:
                command.append(str(run / word[1:]) if word.startswith("@") else word)
        result.append((name, command))
    return result


def terminate(process) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_pipeline(run: Path, poll: float = 15) -> int:
    run = run.resolve()
    status_path = run / "pipeline_status.json"
    if status_path.exists():
        raise FileExistsError(
            "finisher is single-run; inspect existing pipeline_status.json"
        )
    state = {
        "status": "waiting",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "stages": [],
        "config": {"run_dir": str(run), "poll_seconds": poll},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    process = None
    previous = signal.signal(
        signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum)
    )
    save(status_path, state)
    try:
        wait_ready(run, status_path, state, poll)
        state.pop("waiting_for", None)
        for name, command in plan(run):
            record = {
                "name": name,
                "command": command,
                "status": "starting",
                "started_at": now(),
                "log": str(run / "logs" / f"finish-{name}.log"),
            }
            state.update(status="running", current_stage=name)
            state["stages"].append(record)
            save(status_path, state)
            print(f"{now()} starting {name}", flush=True)
            Path(record["log"]).parent.mkdir(exist_ok=True)
            with Path(record["log"]).open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                record.update(
                    status="running", pid=process.pid, process_group=process.pid
                )
                save(status_path, state)
                while process.poll() is None:
                    time.sleep(poll)
                    save(status_path, state)
                record.update(
                    returncode=process.returncode,
                    finished_at=now(),
                    status="complete" if process.returncode == 0 else "incomplete",
                )
                save(status_path, state)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"{name} exited {process.returncode}; see {record['log']}"
                    )
                process = None
        state.update(status="complete", finished_at=now())
        return 0
    except BaseException as error:
        state.update(
            status="incomplete",
            error=f"{type(error).__name__}: {error}",
            finished_at=now(),
        )
        return 1
    finally:
        if process is not None and state["stages"]:
            terminate(process)
            state["stages"][-1].update(
                status="incomplete", returncode=process.returncode
            )
        save(status_path, state)
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll seconds must be positive")
    raise SystemExit(run_pipeline(args.run_dir, args.poll_seconds))
