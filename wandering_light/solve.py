"""Run the current best system on a task file.

    uv run python -m wandering_light.solve TASKS

``TASKS`` is a `stack_pbe` payload (``.jsonl`` or ``.jsonl.gz``) or a corpus
directory, read with `stack_pbe.load`. The system is `BEST`: a stack policy
pinned to a Hub commit, fetched once into the cache (`stack_policy_hub`), guiding
`beam_solve` on this machine's CPU.

The protocol is cost to clear (`docs/metrics.md`): every task is attempted in
file order, each under its own wall clock, and the clock covers everything the
solver does for that task -- parsing the shown examples, replaying the library,
and the search, policy forward passes included. The library is the programs the
run has already found: each is replayed on the new task's *shown* examples,
shortest first, before any search, and the first that fits is the answer. Only
the shown examples are read while the clock can run. Grading against the
held-out examples happens after the last task's clock stops, because grading
runs programs through the executor and would warm caches a later search reads.

Output: a header naming the system, the checkpoint and the host; one progress
line per task on stderr; a table with each task's verdict and seconds; and a
summary of coverage and total wall clock over every task attempted.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light import stack_pbe
from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import Mark, parse_stack
from wandering_light.stack_pbe import BASIS
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy_hub import fetch_stack_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TextIO

    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class System:
    """Everything that decides what the solver does, pinned."""

    name: str
    repo_id: str
    revision: str  # a 40-hex Hub commit id
    beam_width: int
    max_depth: int
    guard: str
    library: bool  # replay programs solved earlier in the run before searching
    measured: str  # where the number this system is best on was measured

    def describe(self) -> str:
        library = "on" if self.library else "off"
        return (
            f"{self.name} = {self.repo_id}@{self.revision[:12]}; "
            f"beam {self.beam_width} x depth {self.max_depth}, "
            f"guard {self.guard}, library {library}"
        )


BEST = System(
    "staged-s0-e3",
    "abhishekraok/wandering-light-stack-policy",
    "e5a0b1391c5268c3ae04a49e6862f19a30779bb2",
    64,
    48,
    "both",
    True,
    "The staged arm of staged-witness-reauthoring (#155), seed 0, epoch 3: 143 of "
    "300 cohort tasks graded solved, 16 of 43 behaviours verified, at beam 64 x "
    "depth 48, guard both, on CUDA with every task searched from scratch "
    "(docs/RESULTS_DETAILED.md); the library is solved-program-cache (#151), "
    "measured there on other policies and not with this one.",
)


@dataclass
class Row:
    """One task attempted: what the solver returned, what it cost, how it graded."""

    index: int
    task: Task
    program: tuple[str, ...] | None
    source: str  # search | library
    termination: str  # fit | max_depth | frontier_exhausted | budget | refused
    seconds: float
    executions: int
    fuel: int
    expansions: int
    depth: int
    solved: bool = False
    shown_fit: bool = False

    @property
    def verdict(self) -> str:
        if self.solved:
            return "solved"
        if self.shown_fit:
            return "memorised"
        return "unsolved"

    def record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "index": self.index,
            "behaviour": self.task.behaviour,
            "length": self.task.length,
            "examples": self.task.examples,
            "program": list(self.program) if self.program is not None else None,
        }
        for field in fields(self):
            if field.name not in record and field.name != "task":
                record[field.name] = getattr(self, field.name)
        return record


@dataclass(frozen=True)
class Summary:
    tasks: int
    solved: int
    memorised: int
    from_library: int
    seconds: float
    executions: int

    @property
    def coverage(self) -> float:
        return self.solved / self.tasks if self.tasks else 0.0

    def line(self) -> str:
        return (
            f"solved {self.solved}/{self.tasks} (coverage {self.coverage:.3f}); "
            f"wall clock {self.seconds:.1f} s over every task attempted; "
            f"executions {self.executions:,}; "
            f"{self.from_library} solved from the library; "
            f"{self.memorised} memorised"
        )


def summarise(rows: Sequence[Row]) -> Summary:
    return Summary(
        tasks=len(rows),
        solved=sum(row.solved for row in rows),
        memorised=sum(row.verdict == "memorised" for row in rows),
        from_library=sum(row.solved and row.source == "library" for row in rows),
        seconds=sum(row.seconds for row in rows),
        executions=sum(row.executions for row in rows),
    )


def _attempt(
    index: int,
    task: Task,
    scorer: PolicyScorer,
    system: System,
    library: list[tuple[str, ...]],
) -> Row:
    """One task under one clock: replay the library, then search if nothing fit."""
    started = time.perf_counter()
    executions = fuel = 0
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if system.library:
        for program in library:
            fits, spent, spent_fuel = replay_program(program, inputs, outputs)
            executions += spent
            fuel += spent_fuel
            if fits:
                return Row(
                    index,
                    task,
                    program,
                    "library",
                    "fit",
                    time.perf_counter() - started,
                    executions,
                    fuel,
                    0,
                    0,
                )
    if system.guard in ("close", "both") and any(
        isinstance(value, Mark) for stack in outputs for value in stack
    ):
        # `beam_solve` raises here: the close guard needs mark-free targets.
        # One such task must not cost the run every other task's verdict.
        return Row(
            index,
            task,
            None,
            "search",
            "refused",
            time.perf_counter() - started,
            executions,
            fuel,
            0,
            0,
        )
    result = beam_solve(
        task,
        scorer,
        beam_width=system.beam_width,
        max_depth=system.max_depth,
        guard=system.guard,
    )
    found = result.program
    if system.library and found is not None and found not in library:
        library.append(found)
        library.sort(key=len)  # stable: equal lengths keep the order found
    return Row(
        index,
        task,
        result.program,
        "search",
        result.termination,
        time.perf_counter() - started,
        executions + result.executions,
        fuel + result.fuel,
        result.expansions,
        result.depth,
    )


def solve_tasks(
    tasks: Sequence[Task],
    scorer: PolicyScorer,
    system: System,
    *,
    progress: Callable[[Row], None] | None = None,
) -> list[Row]:
    """Attempt every task in order, then grade every row on its held-out examples."""
    # The tasks and the model live for the whole run; freezing moves them out of
    # the generations the collector scans (see `beam_solve`).
    gc.freeze()
    library: list[tuple[str, ...]] = []
    rows = []
    for index, task in enumerate(tasks):
        row = _attempt(index, task, scorer, system, library)
        rows.append(row)
        if progress is not None:
            progress(row)
    # Every clock has stopped; nothing below reaches the library or a search.
    for row in rows:
        score = grade(row.program, row.task) if row.program else None
        row.solved = bool(score and score.solved)
        row.shown_fit = bool(score and score.shown)
    return rows


def host(device: str) -> dict[str, Any]:
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": device,
    }


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def table(rows: Sequence[Row]) -> list[str]:
    width = max([9, *(len(row.task.behaviour) for row in rows)])
    lines = [
        f"{'#':>5}  {'behaviour':<{width}}  {'verdict':<9}  {'seconds':>8}  source"
    ]
    lines.extend(
        f"{row.index:>5}  {row.task.behaviour:<{width}}  {row.verdict:<9}  "
        f"{row.seconds:>8.3f}  {row.source}"
        for row in rows
    )
    return lines


def solve_file(
    path: str | Path,
    *,
    system: System = BEST,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    json_path: str | Path | None = None,
    out: TextIO | None = None,
    log: TextIO | None = None,
) -> Summary:
    """Load the policy and the tasks, solve, print the table and the summary."""
    from wandering_light.stack_policy import PolicyScorer, load_policy

    out = out if out is not None else sys.stdout
    log = log if log is not None else sys.stderr
    local = checkpoint is not None
    policy_dir = (
        Path(checkpoint)
        if local
        else fetch_stack_policy(system.repo_id, system.revision)
    )
    scorer = PolicyScorer(load_policy(policy_dir, device=device), device)
    tasks = stack_pbe.load(path)
    machine = host(device)

    def say(line: str) -> None:
        print(line, file=out, flush=True)

    say(f"system  {system.describe()}")
    say(f"policy  {policy_dir}" + ("  (local, not BEST)" if local else ""))
    say(
        f"host    {machine['hostname']}; {machine['platform']}; "
        f"torch {machine['torch']}, {machine['threads']} threads, {device}"
    )
    say(f"tasks   {len(tasks)} from {path}")

    def progress(row: Row) -> None:
        print(
            f"[{row.index + 1}/{len(tasks)}] {row.task.behaviour}: "
            f"{row.source} {row.termination} in {row.seconds:.3f} s",
            file=log,
            flush=True,
        )

    rows = solve_tasks(tasks, scorer, system, progress=progress)
    summary = summarise(rows)
    for line in table(rows):
        say(line)
    say(summary.line())

    if json_path is not None:
        report = {
            "system": asdict(system),
            "checkpoint": str(policy_dir),
            "checkpoint_is_best": not local,
            "host": machine,
            "commit": _commit(),
            "tasks_file": str(path),
            "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "summary": {**asdict(summary), "coverage": summary.coverage},
            "rows": [row.record() for row in rows],
        }
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.solve",
        description="Run the current best system on a stack_pbe task file.",
    )
    parser.add_argument(
        "tasks", type=Path, help="a .jsonl or .jsonl.gz payload, or a corpus directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a local stack-policy checkpoint instead of BEST's pinned one",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json", type=Path, default=None, help="also write every row here"
    )
    args = parser.parse_args(argv)
    solve_file(
        args.tasks,
        system=BEST,
        checkpoint=args.checkpoint,
        device=args.device,
        json_path=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
