"""Solve a stack PBE task file in order and report held-out coverage and search cost."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, load
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import PolicyScorer, load_policy
from wandering_light.stack_policy_hub import DEFAULT_REPO_ID, fetch_stack_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import TextIO

    from wandering_light.stack_pbe import Task


@dataclass(frozen=True)
class System:
    name: str
    repo_id: str
    revision: str
    beam_width: int
    max_depth: int
    guard: str
    library: bool
    measured: str

    def describe(self) -> str:
        return (
            f"{self.name} = {self.repo_id}@{self.revision[:12]}; "
            f"beam {self.beam_width} x depth {self.max_depth}, guard {self.guard}, "
            f"library {'on' if self.library else 'off'}"
        )


BEST = System(
    "staged-s0-e3",
    DEFAULT_REPO_ID,
    "e5a0b1391c5268c3ae04a49e6862f19a30779bb2",
    64,
    48,
    "both",
    True,
    "The 300-task stack_pbe_v4 eval cohort was used to measure the staged policy "
    "(reports/staged-arms-20260920) and library replay with an earlier policy "
    "(reports/solved-program-cache-20260919).",
)


@dataclass
class Row:
    index: int
    task: Task
    program: tuple[str, ...] | None
    source: str
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

    def record(self) -> dict:
        return {
            "index": self.index,
            "behaviour": self.task.behaviour,
            "length": self.task.length,
            "examples": self.task.examples,
            "program": list(self.program) if self.program is not None else None,
            "source": self.source,
            "termination": self.termination,
            "seconds": self.seconds,
            "executions": self.executions,
            "fuel": self.fuel,
            "expansions": self.expansions,
            "depth": self.depth,
            "solved": self.solved,
            "shown_fit": self.shown_fit,
            "verdict": self.verdict,
        }


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
            f"executions {self.executions:,}; {self.from_library} solved from the library; "
            f"{self.memorised} memorised"
        )


def summarise(rows: Sequence[Row]) -> Summary:
    return Summary(
        tasks=len(rows),
        solved=sum(row.solved for row in rows),
        memorised=sum(row.verdict == "memorised" for row in rows),
        from_library=sum(row.source == "library" and row.solved for row in rows),
        seconds=sum((row.seconds for row in rows), 0.0),
        executions=sum(row.executions for row in rows),
    )


def solve_tasks(
    tasks: Iterable[Task],
    scorer: PolicyScorer,
    system: System,
    *,
    progress: Callable[[Row], None] | None = None,
) -> list[Row]:
    """Report progress after each timer stops and return rows with held-out grades."""
    library: list[tuple[str, ...]] = []
    rows = []
    for index, task in enumerate(tasks):
        started = time.perf_counter()
        program = None
        executions = fuel = expansions = depth = 0
        if library:
            inputs = tuple(parse_stack(s, basis=BASIS) for s in task.inputs)
            outputs = tuple(parse_stack(s, basis=BASIS) for s in task.outputs)
            for candidate in library:
                fits, stepped, spent = replay_program(candidate, inputs, outputs)
                executions += stepped
                fuel += spent
                if fits:
                    program = candidate
                    break

        if program is None:
            result = beam_solve(
                task,
                scorer,
                beam_width=system.beam_width,
                max_depth=system.max_depth,
                guard=system.guard,
            )
            source, termination, program = "search", result.termination, result.program
            executions += result.executions
            fuel += result.fuel
            expansions, depth = result.expansions, result.depth
            if system.library and program is not None and program not in library:
                library.append(program)
                library.sort(key=len)
        else:
            source, termination = "library", "fit"

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

    # Grading populates the executor's shared cache. Seal every timer first.
    for row in rows:
        if row.program is not None:
            score = grade(row.program, row.task)
            row.solved, row.shown_fit = score.solved, score.shown
    return rows


def _commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


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
    if out is None:
        out = sys.stdout
    if log is None:
        log = sys.stderr
    tasks = load(path)
    directory = (
        fetch_stack_policy(system.repo_id, system.revision)
        if checkpoint is None
        else Path(checkpoint)
    )
    scorer = PolicyScorer(load_policy(directory, device=device), device)
    host = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    print(f"system  {system.describe()}", file=out)
    print(
        f"policy  {directory}"
        + ("  (local, not BEST)" if checkpoint is not None else ""),
        file=out,
    )
    print(f"host    {host['hostname']}  {host['platform']}  {device}", file=out)
    print(f"tasks   {len(tasks)} from {path}", file=out, flush=True)

    def progress(row: Row) -> None:
        print(
            f"{row.index:>5}  {row.task.behaviour}  {row.source} {row.termination}  {row.seconds:.3f} s",
            file=log,
            flush=True,
        )

    rows = solve_tasks(tasks, scorer, system, progress=progress)
    w = max(len("behaviour"), max((len(r.task.behaviour) for r in rows), default=0))
    print(
        f"{'#':>5}  {'behaviour':<{w}}  {'verdict':<9}  {'seconds':>8}  source",
        file=out,
    )
    for row in rows:
        print(
            f"{row.index:>5}  {row.task.behaviour:<{w}}  {row.verdict:<9}  "
            f"{row.seconds:>8.3f}  {row.source}",
            file=out,
        )
    summary = summarise(rows)
    print(summary.line(), file=out)
    if json_path is not None:
        report = {
            "system": asdict(system),
            "checkpoint": str(directory),
            "checkpoint_is_best": checkpoint is None,
            "host": host,
            "commit": _commit(),
            "tasks_file": str(path),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "summary": {**asdict(summary), "coverage": summary.coverage},
            "rows": [row.record() for row in rows],
        }
        dest = Path(json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks", type=Path, help="stack_pbe .jsonl, .jsonl.gz, or corpus directory"
    )
    parser.add_argument("--checkpoint", type=Path, help="use a local stack policy")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument(
        "--json", type=Path, help="write task results and run provenance"
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
