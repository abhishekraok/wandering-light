"""Run the current best system on a task file.

    uv run python -m wandering_light.solve TASKS [--checkpoint DIR] [--json OUT]

``TASKS`` is a stack_pbe payload (``.jsonl`` or ``.jsonl.gz``) or a corpus
directory, read with ``stack_pbe.load``. Every task is attempted in file
order. Each task's clock covers replaying the programs solved earlier in the
run on its shown examples and, if none fits, a policy-guided beam search; a
program the search finds joins that library. Held-out examples are read only
after the last clock stops, by ``stack_pbe_eval.grade``, so no search or
replay is warmed by them.

What "best" means is the ``BEST`` constant: a policy checkpoint pinned to a
Hub commit, plus the search settings it was measured with. Passing
``--checkpoint`` runs a local policy under the same settings, and the output
says it is not ``BEST``.
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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, load
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy_hub import fetch_stack_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class System:
    """A solver configuration: which policy, and how the search uses it."""

    name: str
    repo_id: str
    revision: str
    beam_width: int
    max_depth: int
    guard: str
    library: bool
    measured: str

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
    "The staged arm of staged-witness-reauthoring (#155), seed 0, epoch 3, "
    "measured at beam 64 x depth 48 on the 300-task / 43-behaviour stack_pbe_v4 "
    "eval cohort (docs/STATUS.md, reports/staged-arms-20260920).",
)


@dataclass
class Row:
    """One task's attempt; ``solved`` and ``shown_fit`` are filled in by grading."""

    index: int
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
        for name, value in asdict(self).items():
            if name not in ("index", "task", "program"):
                record[name] = value
        record["verdict"] = self.verdict
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
        solved=sum(r.solved for r in rows),
        memorised=sum(r.verdict == "memorised" for r in rows),
        from_library=sum(r.solved and r.source == "library" for r in rows),
        seconds=sum(r.seconds for r in rows),
        executions=sum(r.executions for r in rows),
    )


def solve_tasks(
    tasks: Sequence[Task],
    scorer: PolicyScorer,
    system: System,
    *,
    progress: Callable[[Row], None] | None = None,
) -> list[Row]:
    """Attempt every task in order, then grade them all."""
    library: list[tuple[str, ...]] = []
    rows: list[Row] = []
    # The tasks and the model live for the whole run; see solve_with_cache.
    gc.freeze()
    for index, task in enumerate(tasks):
        started = time.perf_counter()
        program: tuple[str, ...] | None = None
        source, termination = "search", ""
        executions = fuel = expansions = depth = 0
        if system.library and library:
            inputs = tuple(parse_stack(t, basis=BASIS) for t in task.inputs)
            outputs = tuple(parse_stack(t, basis=BASIS) for t in task.outputs)
            for candidate in library:
                fits, spent, spent_fuel = replay_program(candidate, inputs, outputs)
                executions += spent
                fuel += spent_fuel
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
        row = Row(
            index,
            task,
            program,
            source,
            termination,
            time.perf_counter() - started,
            executions,
            fuel,
            expansions,
            depth,
        )
        rows.append(row)
        if progress is not None:
            progress(row)
    # Grading reads the held-out examples and warms the executor's caches, so
    # it waits until every clock has stopped.
    for row in rows:
        if row.program:
            score = grade(row.program, row.task)
            row.solved, row.shown_fit = score.solved, score.shown
    return rows


def _host(device: str) -> dict[str, Any]:
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


def _table(rows: Sequence[Row]) -> list[str]:
    width = max([9, *(len(r.task.behaviour) for r in rows)])
    lines = [
        f"{'#':>5}  {'behaviour':<{width}}  {'verdict':<9}  {'seconds':>8}  source"
    ]
    lines += [
        f"{r.index:>5}  {r.task.behaviour:<{width}}  {r.verdict:<9}  "
        f"{r.seconds:>8.3f}  {r.source}"
        for r in rows
    ]
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
    from wandering_light.stack_policy import PolicyScorer, load_policy

    out = out if out is not None else sys.stdout
    log = log if log is not None else sys.stderr
    local = checkpoint is not None
    directory = (
        Path(checkpoint)
        if local
        else fetch_stack_policy(system.repo_id, system.revision)
    )
    scorer = PolicyScorer(load_policy(directory, device=device), device)
    tasks = load(path)
    host = _host(device)

    def say(line: str) -> None:
        print(line, file=out, flush=True)

    say(f"system  {system.describe()}")
    say(f"policy  {directory}" + ("  (local, not BEST)" if local else ""))
    say(
        f"host    {host['hostname']}; {host['platform']}; torch {host['torch']}, "
        f"{host['threads']} threads, {device}"
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
    for line in _table(rows):
        say(line)
    say(summary.line())
    if json_path is not None:
        report = {
            "system": asdict(system),
            "checkpoint": str(directory),
            "checkpoint_is_best": not local,
            "host": host,
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
        description=f"Run the best system ({BEST.describe()}) on a task file.",
    )
    parser.add_argument(
        "tasks", help="a stack_pbe payload (.jsonl or .jsonl.gz) or corpus directory"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="a local stack-policy checkpoint instead of BEST's pinned one",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None, help="also write a JSON report here")
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
