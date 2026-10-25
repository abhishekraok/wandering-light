"""Supervise plan-billed PBE authors; never execute their generated programs.

Example: python -m experiments.pbe_scale.campaign --output-dir <main>/checkpoints/
pbe-scale-20261024 --deadline 2026-10-25T00:00:00-07:00 --concurrency 8
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

MODEL = "gpt-6.1-sol"
EFFORT = "low"
PACIFIC = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parents[2]
THEMES = (
    "signal and time-series summaries, signed changes and local peaks",
    "event patterns, threshold crossings, alternating signs and gaps",
    "data cleaning, clipping, deduplication and robust order statistics",
    "small combinatorial counts, pairs, triples and bounded selections",
    "integer arithmetic and decimal digit transformations",
    "runs, streaks, change points and run-length summaries",
    "pair similarity, intersections and distance summaries of two lists",
    "interval counts, threshold bands and cumulative exceedances",
    "histograms and frequency-weighted summaries",
    "position-weighted statistics, prefixes and index selection",
    "branching list transformations with meaningful edge-case behavior",
    "nested aggregation and neighborhood comparisons",
)
REQUIRED = (
    "id",
    "family",
    "template_id",
    "description",
    "input_kind",
    "program",
    "reference",
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["programs"],
    "properties": {
        "programs": {
            "type": "array",
            "minItems": 20,
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(REQUIRED),
                "properties": {
                    **{key: {"type": "string"} for key in REQUIRED},
                    "input_kind": {
                        "type": "string",
                        "enum": ["list", "list_int", "two_lists", "int", "two_ints"],
                    },
                },
            },
        }
    },
}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def timestamp(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def quota_from_event(event: dict, source: str) -> dict | None:
    limits = (event.get("payload") or {}).get("rate_limits") or {}
    for key in ("primary", "secondary"):
        window = limits.get(key) or {}
        if window.get("window_minutes") != 10080:
            continue
        try:
            used = float(window["used_percent"])
            observed = timestamp(event["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(used) or not 0 <= used <= 100:
            continue
        return {
            "remaining_percent": 100 - used,
            "observed_at": observed,
            "resets_at": window.get("resets_at"),
            "source": source,
        }
    return None


def latest_quota(sessions: Path, now: float | None = None) -> dict | None:
    """Read weekly receipts from recent rollout tails without querying an API."""
    now = time.time() if now is None else now
    files = []
    for path in sessions.glob("**/*.jsonl"):
        try:
            modified = path.stat().st_mtime
            if modified >= now - 7 * 86400:
                files.append((modified, path))
        except OSError:
            continue
    latest = None
    for _, path in sorted(files, reverse=True)[:128]:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 512 * 1024))
                lines = stream.read().decode(errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if '"rate_limits"' not in line:
                continue
            try:
                reading = quota_from_event(json.loads(line), str(path))
            except (ValueError, AttributeError, TypeError):
                continue
            if reading and (
                latest is None or reading["observed_at"] > latest["observed_at"]
            ):
                latest = reading
    return latest


def quota_problem(reading: dict | None, now: float, max_age: float) -> str | None:
    if reading is None:
        return "weekly quota receipt missing"
    remaining = reading["remaining_percent"]
    if remaining < 5 or remaining > 95:
        return f"weekly remaining {remaining:g}% outside authorized [5%, 95%]"
    age = now - reading["observed_at"]
    if age > max_age or age < -30:
        return f"weekly quota receipt stale or future-dated (age {age:.0f}s)"
    return None


def make_prompt(index: int, rows: int, notes_path: Path | None = None) -> str:
    reference = (ROOT / "experiments/world_behaviours/REFERENCE.md").read_text()
    macros = (ROOT / "experiments/world_behaviours/macros.txt").read_text()
    theme = THEMES[index % len(THEMES)]
    notes = notes_path.read_text() if notes_path and notes_path.exists() else ""
    return f"""Author exactly {rows} novel PBE training program rows. Return only the schema JSON.
Do not call tools, read files, execute code, edit files, or delegate. Everything needed is below.
Do not use Claude, other agents, API keys, metered APIs, subprocesses, or any code execution.
Batch: {index:06d}; variation seed: {104729 * (index + 1)}; assigned theme: {theme}.
Use IDs scale-{index:06d}-01, etc. Choose meaningful, diverse behaviors, not constant tweaks.
Prefer useful programs of 50-200 expanded primitive ops, with composition and real computation;
never pad with no-ops or make constant-only variants. Use only the canonical macros below.
Each row has id, family (semantic family), template_id (algorithm structure, not unique row ID),
description (complete semantics including edge cases), input_kind, program (macro string),
reference (independent pure Python source defining def f(...), NOT a stack interpreter).
Use meaningful shared template_id groups such as filtered-weighted-sum or adjacent-pair-count;
do not use row IDs or arbitrary unique names as template IDs. Python source has no type annotations.
input_kind means: list=f(xs); list_int=f(xs,n); two_lists=f(xs,ys);
int=f(n); two_ints=f(a,b). The entire domain is lists of length 0..12 containing integers
-10..10, including empty lists, and scalar integers -10..10, including zero and negatives.
Make every row total on its whole domain; guard empty minima/maxima, divisors and indexing.
Outputs are integers or flat integer lists, never bools. Reference has no imports, I/O,
external state, recursion, eval, exec or effects; use ordinary pure builtins, loops and arithmetic.
The program must leave exactly one value. Independently implement the description in Python;
do not translate stack ops mechanically. Do not use or optimize against any held-out world tasks.
The reference document below explains machine semantics. Its old author-workflow commands,
file editing, assigned behavior IDs and delegation instructions do NOT apply to this batch.

<machine-reference>\n{reference}\n</machine-reference>
<canonical-macros>\n{macros}\n</canonical-macros>
<campaign-guidance>\n{notes}\n</campaign-guidance>
"""


def worker_command(directory: Path, schema: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "--ignore-user-config",
        "--json",
        "-s",
        "read-only",
        "-m",
        MODEL,
        "-c",
        f"model_reasoning_effort={EFFORT}",
        "-c",
        "project_doc_max_bytes=0",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(directory / "output.json"),
        "--skip-git-repo-check",
        "-C",
        str(directory),
        "-",
    ]


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        env.pop(key, None)
    return env


def collect_output(directory: Path, destination: Path, record: dict) -> int:
    payload = json.loads((directory / "output.json").read_text())
    rows = payload["programs"]
    if not isinstance(rows, list) or not 20 <= len(rows) <= 40:
        raise ValueError("response must contain 20-40 program rows")
    usage = None
    for line in (directory / "events.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    provenance = {
        "source_batch": directory.name,
        "model": MODEL,
        "effort": EFFORT,
        "prompt_sha256": record["prompt_sha256"],
        "token_usage": usage,
    }
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) for k in REQUIRED
        ):
            raise ValueError("response row missing required string fields")
        if row["id"] in identifiers:
            raise ValueError("duplicate row id within batch")
        identifiers.add(row["id"])
        row["_provenance"] = provenance
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(destination)
    record["token_usage"] = usage
    return len(rows)


def stop_workers(active: dict, records: dict, reason: str) -> None:
    """Signal only process groups created and still owned by this supervisor."""
    for process, _handles in active.values():
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for batch, (process, handles) in active.items():
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # A worker can exit between wait timing out and escalation.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        finally:
            for handle in handles:
                handle.close()
            records[batch].update(
                status="stopped", reason=reason, returncode=process.returncode
            )
    active.clear()


def run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep the handle open for the full run; competing supervisors fail immediately.
    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, output)


def run_locked(args: argparse.Namespace, output: Path) -> int:
    authors, candidates = output / "authors", output / "candidates"
    authors.mkdir(exist_ok=True)
    candidates.mkdir(exist_ok=True)
    schema = output / "author-schema.json"
    atomic_json(schema, SCHEMA)
    status_file = output / "campaign_status.json"
    previous = {}
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        if previous.get("status") == "running":
            raise RuntimeError(
                "previous campaign was not stopped cleanly; inspect its recorded PIDs first"
            )
    records: dict[str, dict] = dict(previous.get("batches", {}))
    active: dict[str, tuple] = {}
    state: dict[str, Any] = {
        "status": "running",
        "supervisor_pid": os.getpid(),
        "model": MODEL,
        "effort": EFFORT,
        "deadline": args.deadline.isoformat(),
        "batches": records,
    }
    next_index = 1 + max(
        (int(p.name.split("-")[1]) for p in authors.glob("batch-*-a*")), default=-1
    )
    launched = 0
    consecutive_failures = 0
    retries: list[tuple[int, int]] = []
    stale_since = None
    next_poll = 0.0
    stop_reason = "completed max batches"
    stopped = False

    def persist() -> None:
        state["updated_at"] = datetime.now(PACIFIC).isoformat()
        state["active_workers"] = len(active)
        state["candidate_rows"] = sum(
            r.get("candidate_rows", 0) for r in records.values()
        )
        atomic_json(status_file, state)

    def log_quota(reading: dict | None, problem: str | None) -> None:
        with (output / "quota.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "at": datetime.now(PACIFIC).isoformat(),
                        "quota": reading,
                        "problem": problem,
                    }
                )
                + "\n"
            )
        state["quota"] = reading
        state["quota_problem"] = problem

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stopped, stop_reason
        stopped = True
        stop_reason = f"supervisor received signal {signum}"

    old_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    persist()
    try:
        if args.dry_run:
            help_result = subprocess.run(
                ["codex", "exec", "--help"], capture_output=True, text=True, check=True
            )
            for flag in (
                "--ignore-user-config",
                "--output-schema",
                "--json",
                "--output-last-message",
            ):
                if flag not in help_result.stdout:
                    raise RuntimeError(f"installed codex lacks {flag}")
            preview = authors / "dry-run"
            preview.mkdir(exist_ok=True)
            (preview / "prompt.txt").write_text(
                make_prompt(next_index, args.rows, output / "author-notes.txt")
            )
            reading = latest_quota(args.sessions_dir)
            log_quota(reading, quota_problem(reading, time.time(), args.quota_max_age))
            state["preview_command"] = worker_command(preview, schema)
            stop_reason = "dry run; no model requests made; schema written for review"
            return 0
        while not stopped:
            now = time.time()
            if now >= args.deadline.timestamp():
                stop_reason = "authorized deadline reached"
                break
            if now >= next_poll:
                reading = latest_quota(args.sessions_dir, now)
                problem = quota_problem(reading, now, args.quota_max_age)
                log_quota(reading, problem)
                next_poll = now + args.quota_poll
                if reading and (
                    reading["remaining_percent"] < 5
                    or reading["remaining_percent"] > 95
                ):
                    stop_reason = problem or "weekly quota stop"
                    break
                stale_since = (stale_since or now) if problem else None
                if stale_since is not None and now - stale_since >= args.quota_grace:
                    stop_reason = f"{problem}; fail-closed grace exhausted"
                    break
            for batch, (process, handles) in list(active.items()):
                if process.poll() is None:
                    continue
                for handle in handles:
                    handle.close()
                record = records[batch]
                record.update(
                    returncode=process.returncode,
                    finished_at=datetime.now(PACIFIC).isoformat(),
                )
                del active[batch]
                try:
                    if process.returncode:
                        raise ValueError(f"worker exited {process.returncode}")
                    record["candidate_rows"] = collect_output(
                        authors / batch, candidates / f"{batch}.jsonl", record
                    )
                    record["status"] = "complete"
                    consecutive_failures = 0
                except (ValueError, OSError, KeyError, TypeError) as error:
                    record.update(status="failed", error=str(error))
                    consecutive_failures += 1
                    if record["attempt"] < args.max_retries:
                        retries.append((record["index"], record["attempt"] + 1))
                persist()
            if consecutive_failures >= args.max_consecutive_failures:
                stop_reason = (
                    f"stopped after {consecutive_failures} consecutive worker failures"
                )
                break
            if not active and launched >= args.max_batches and not retries:
                break
            while len(active) < args.concurrency and (
                launched < args.max_batches or retries
            ):
                # Scan again at each launch, including within an initial concurrency burst.
                now = time.time()
                reading = latest_quota(args.sessions_dir, now)
                problem = quota_problem(reading, now, args.quota_max_age)
                log_quota(reading, problem)
                if reading and (
                    reading["remaining_percent"] < 5
                    or reading["remaining_percent"] > 95
                ):
                    stopped, stop_reason = True, problem or "weekly quota stop"
                if problem or now >= args.deadline.timestamp() or stopped:
                    break
                if retries:
                    index, attempt = retries.pop(0)
                else:
                    index, attempt = next_index, 0
                    next_index += 1
                    launched += 1
                batch = f"batch-{index:06d}-a{attempt}"
                directory = authors / batch
                directory.mkdir(exist_ok=False)
                prompt = make_prompt(index, args.rows, output / "author-notes.txt")
                (directory / "prompt.txt").write_text(prompt)
                handles = (
                    (directory / "prompt.txt").open("r"),
                    (directory / "events.jsonl").open("w"),
                    (directory / "stderr.log").open("w"),
                )
                record = {
                    "index": index,
                    "attempt": attempt,
                    "status": "starting",
                    "started_at": datetime.now(PACIFIC).isoformat(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "theme": THEMES[index % len(THEMES)],
                }
                records[batch] = record
                persist()
                try:
                    process = subprocess.Popen(
                        worker_command(directory, schema),
                        stdin=handles[0],
                        stdout=handles[1],
                        stderr=handles[2],
                        env=worker_environment(),
                        start_new_session=True,
                    )
                except BaseException:
                    for handle in handles:
                        handle.close()
                    record["status"] = "launch_failed"
                    raise
                active[batch] = (process, handles)
                record.update(
                    status="running", pid=process.pid, process_group=process.pid
                )
                persist()
                print(
                    f"{datetime.now(PACIFIC).isoformat()} launched {batch} pid={process.pid}; weekly remaining={reading['remaining_percent']:g}%",
                    flush=True,
                )
            persist()
            time.sleep(1)
    except BaseException as error:
        stop_reason = f"supervisor error: {type(error).__name__}: {error}"
        raise
    finally:
        stop_workers(active, records, stop_reason)
        state.update(status="stopped", stop_reason=stop_reason)
        persist()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        print(
            f"Campaign stopped: {stop_reason}; candidate rows={state['candidate_rows']}",
            flush=True,
        )
    return 0


def deadline_argument(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    return value.replace(tzinfo=PACIFIC) if value.tzinfo is None else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--deadline",
        type=deadline_argument,
        required=True,
        help="ISO timestamp; naive times mean America/Los_Angeles",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=10000)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-consecutive-failures", type=int, default=8)
    parser.add_argument("--quota-poll", type=float, default=15)
    parser.add_argument("--quota-max-age", type=float, default=180)
    parser.add_argument("--quota-grace", type=float, default=45)
    parser.add_argument(
        "--sessions-dir", type=Path, default=Path.home() / ".codex/sessions"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_batches < 1
        or not 20 <= args.rows <= 40
        or args.max_retries < 0
        or args.max_consecutive_failures < 1
    ):
        parser.error(
            "positive concurrency/batches/failure limit, nonnegative retries, and 20-40 rows required"
        )
    if args.quota_poll <= 0 or args.quota_max_age <= 0 or args.quota_grace < 0:
        parser.error("quota poll/max age must be positive; grace must be nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
