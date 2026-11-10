"""Run the current best system on a task file.

    uv run python -m wandering_light.solve TASKS

``TASKS`` is anything `stack_pbe.load` reads: one payload (``.jsonl`` or
``.jsonl.gz``) or a corpus directory. Every task is attempted, in file order,
by `BEST`: a published policy checkpoint, the beam it was measured with, and a
library of the programs this run has already found, replayed before each search.

What is timed and what is not is the point of the module:

- One clock per task covers the library replay, the search, and adding a found
  program to the library. `docs/metrics.md` allows "a persistent library across
  tasks, and library construction charged in the same clock"; this is that.
- A library program is accepted on the task's **shown** examples, the evidence
  the search is given, and nothing else.
- Grading on the held-out examples happens once, after the last clock has
  stopped. `grade` runs programs through the executor, whose `run_body` cache is
  process-global, so grading between searches would warm later ones with
  examples the solver is not supposed to have seen.

The totals are over every task attempted, solved or not: coverage is the
fraction solved on held-out examples, and wall-clock is the sum of the per-task
clocks, including the time spent failing.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import torch

from wandering_light import stack_pbe
from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import PolicyScorer, load_policy
from wandering_light.stack_policy_hub import fetch_stack_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from wandering_light.stack_pbe import Task


@dataclass(frozen=True)
class System:
    """A solver pinned down far enough to be run again: weights and search."""

    name: str
    repo_id: str
    revision: str  # a Hub commit id, so the weights cannot change under the name
    beam_width: int
    max_depth: int
    guard: str
    library: bool
    measured: str  # where this system's standing comes from

    def describe(self) -> str:
        return (
            f"{self.name} = {self.repo_id}@{self.revision[:12]}; "
            f"beam {self.beam_width} x depth {self.max_depth}, guard {self.guard}, "
            f"library {'on' if self.library else 'off'}"
        )


BEST = System(
    "staged-s0-e3",
    "abhishekraok/wandering-light-stack-policy",
    "e5a0b1391c5268c3ae04a49e6862f19a30779bb2",
    64,
    48,
    "both",
    True,
    "The staged arm, seed 0, epoch 3, of staged-witness-reauthoring (#155), "
    "reports/staged-arms-20260920: 143 of 300 tasks graded solved and 16 of 43 "
    "behaviours verified on the #101 stack_pbe_v4 eval cohort at this beam, depth "
    "and guard, on CUDA, with every task searched from scratch (library off).",
)


@dataclass
class Row:
    """One task's attempt. ``solved`` and ``shown_fit`` are filled in by grading."""

    index: int  # position in the task file, from 0
    task: Task
    program: tuple[str, ...] | None
    source: str  # search | library
    termination: str
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
        return "memorised" if self.shown_fit else "unsolved"

    def record(self) -> dict[str, Any]:
        """The row as JSON: the task by name and size, not its examples."""
        record: dict[str, Any] = {
            "index": self.index,
            "behaviour": self.task.behaviour,
            "length": self.task.length,
            "examples": self.task.examples,
            "program": None if self.program is None else list(self.program),
        }
        for field in dataclasses.fields(self):
            if field.name not in ("task", *record):
                record[field.name] = getattr(self, field.name)
        return record


@dataclass(frozen=True)
class Summary:
    tasks: int
    solved: int
    memorised: int
    from_library: int  # solved by a replayed program, not by a search
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


def summarise(rows: Iterable[Row]) -> Summary:
    rows = list(rows)
    return Summary(
        tasks=len(rows),
        solved=sum(row.solved for row in rows),
        memorised=sum(row.verdict == "memorised" for row in rows),
        from_library=sum(row.solved and row.source == "library" for row in rows),
        seconds=sum(row.seconds for row in rows),
        executions=sum(row.executions for row in rows),
    )


def solve_tasks(
    tasks: Sequence[Task],
    scorer: PolicyScorer,
    system: System,
    *,
    progress: Callable[[Row], None] | None = None,
) -> list[Row]:
    """Attempt every task in order; grade them all once the last clock stops.

    ``progress`` is called with each row as its clock stops, outside the clock
    and before any grading, so a row it sees has no verdict yet.
    """
    library: list[tuple[str, ...]] = []  # programs found so far, shortest first
    rows: list[Row] = []
    for index, task in enumerate(tasks):
        started = time.perf_counter()
        program: tuple[str, ...] | None = None
        source, termination = "search", ""
        executions = fuel = expansions = depth = 0
        if library:
            inputs = tuple(
                parse_stack(text, basis=stack_pbe.BASIS) for text in task.inputs
            )
            outputs = tuple(
                parse_stack(text, basis=stack_pbe.BASIS) for text in task.outputs
            )
            for candidate in library:
                fits, spent, burnt = replay_program(candidate, inputs, outputs)
                executions += spent
                fuel += burnt
                if fits:
                    program, source, termination = candidate, "library", "fit"
                    break
        if program is None:
            result = beam_solve(
                task,
                scorer,
                beam_width=system.beam_width,
                max_depth=system.max_depth,
                guard=system.guard,
            )
            program, termination = result.program, result.termination
            executions += result.executions
            fuel += result.fuel
            expansions, depth = result.expansions, result.depth
            if system.library and program is not None and program not in library:
                library.append(program)
                library.sort(key=len)
        seconds = time.perf_counter() - started
        row = Row(
            index,
            task,
            program,
            source,
            termination,
            seconds,
            executions,
            fuel,
            expansions,
            depth,
        )
        rows.append(row)
        if progress is not None:
            progress(row)
    # Every clock has stopped. Held-out examples are read from here on only.
    for row in rows:
        if row.program is not None:
            score = grade(row.program, row.task)
            row.solved, row.shown_fit = score.solved, score.shown
    return rows


def _host(device: str) -> dict[str, Any]:
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
    }


def _commit() -> str | None:
    """The checkout this module was run from, when it is one."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


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
    """Run ``system`` on the tasks at ``path`` and report.

    ``checkpoint`` swaps in a local policy directory for the published one. The
    search is still ``system``'s, but the run is no longer ``system``, and the
    output says so.
    """
    out = sys.stdout if out is None else out
    log = sys.stderr if log is None else log
    tasks = stack_pbe.load(path)
    if checkpoint is None:
        policy_dir = fetch_stack_policy(system.repo_id, system.revision)
    else:
        policy_dir = Path(checkpoint)
    host = _host(device)
    local = "" if checkpoint is None else "  (local, not BEST)"
    out.write(f"system  {system.describe()}\n")
    out.write(f"policy  {policy_dir}{local}\n")
    out.write(
        f"host    {host['hostname']}; {host['platform']}; device {device}"
        + (f" ({host['cuda_device']})" if host["cuda_device"] else "")
        + f"; torch {host['torch']}, {host['torch_threads']} threads\n"
    )
    out.write(f"tasks   {len(tasks)} from {path}\n")
    out.flush()

    scorer = PolicyScorer(load_policy(policy_dir, device=device), device)

    def progress(row: Row) -> None:
        log.write(
            f"[{row.index + 1}/{len(tasks)}] {row.task.behaviour}: "
            f"{row.termination} in {row.seconds:.3f} s, {row.source}\n"
        )
        log.flush()

    # The tasks and the model live for the whole run and never become garbage;
    # freezing takes them out of what the collector scans between searches
    # (`beam_solve` turns it off during one), as the measured cells did.
    gc.freeze()
    try:
        rows = solve_tasks(tasks, scorer, system, progress=progress)
    finally:
        gc.unfreeze()
    summary = summarise(rows)

    width = max([9, *(len(row.task.behaviour) for row in rows)])
    out.write(
        f"{'#':>5}  {'behaviour':<{width}}  {'verdict':<9}  {'seconds':>8}  source\n"
    )
    for row in rows:
        out.write(
            f"{row.index:>5}  {row.task.behaviour:<{width}}  {row.verdict:<9}  "
            f"{row.seconds:>8.3f}  {row.source}\n"
        )
    out.write(summary.line() + "\n")
    out.flush()

    if json_path is not None:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "system": dataclasses.asdict(system),
            "checkpoint": str(policy_dir),
            "checkpoint_is_best": checkpoint is None,
            "host": host,
            "commit": _commit(),
            "tasks_file": str(path),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "summary": {**dataclasses.asdict(summary), "coverage": summary.coverage},
            "rows": [row.record() for row in rows],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.solve",
        description=f"Run the current best system on a task file: {BEST.describe()}.",
        epilog=f"Measured: {BEST.measured}",
    )
    parser.add_argument(
        "tasks",
        type=Path,
        help="a stack_pbe payload (.jsonl or .jsonl.gz) or a corpus directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a local stack-policy directory to use instead of the published one",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the run here as JSON"
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
