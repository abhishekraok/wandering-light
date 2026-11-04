"""One command runs the current best system on a task file.

    uv run python -m wandering_light.solve TASKS

``TASKS`` is a `stack_pbe` payload, one task per line as
`wandering_light.stack_pbe` writes it (``.jsonl`` or ``.jsonl.gz``), or a
corpus directory read through its manifest. Every task is attempted in file
order. One line per task says whether the program found passes the task's
held-out examples and the wall-clock charged for it; the last line is the
pair `docs/metrics.md` defines, coverage and total wall-clock over every task
attempted, solved or not. A library of the programs solved so far in the run
is replayed before each search and built inside the clock, which the metric
allows.

`BEST` is the whole definition of "the current best system": one checkpoint
pinned to an immutable Hub revision, and the search it runs under. Nothing
else decides this, so changing it is one edit here, made when a cohort
measurement on best-system-command (#158) says so. The checkpoint is fetched
once into `stack_policy_hub.default_cache_dir` and read from there after.

The clock covers library replay, search and policy inference, per task, on
the CPU unless a device is named. Grading runs after every task has been
timed, as solved-program-cache (#151) does, so the executor's memo table is
never warmed by held-out examples that a later search could read.
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
from typing import TYPE_CHECKING, Any

import torch

from wandering_light.stack_beam import (
    beam_solve,
    replay_program,
    segment_beam_solve,
    spent_in,
    widening_beam_solve,
)
from wandering_light.stack_machine import parse_stack, run_body
from wandering_light.stack_pbe import BASIS, load
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import PolicyScorer, load_policy
from wandering_light.stack_policy_hub import fetch_stack_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TextIO

    from wandering_light.stack_pbe import Task

REPO = Path(__file__).resolve().parents[1]
SEARCHES = ("flat", "segment", "widening")


@dataclass(frozen=True)
class System:
    """A checkpoint and the search it runs under. `BEST` is the one that counts."""

    name: str
    repo_id: str
    revision: str
    beam_width: int
    max_depth: int
    guard: str
    library: bool
    measured: str
    # beam-state-size-unbounded (#157): a successor with more atoms than this is
    # refused. None leaves the beam unbounded.
    max_state_atoms: int | None = None
    # short-segment-campaign (#168): "flat" is `beam_solve`, beam_width x
    # max_depth; "segment" is `segment_beam_solve`, beam_width per root, `roots`
    # roots and up to max_segments segments of at most segment_depth ops, with
    # max_depth unused; "widening" is `widening_beam_solve`, beam_width the
    # first width and max_depth each restart's depth, and needs a budget.
    search: str = "flat"
    roots: int = 16
    segment_depth: int = 12
    max_segments: int = 12

    def __post_init__(self) -> None:
        if self.search not in SEARCHES:
            raise ValueError(
                f"unknown search {self.search!r}; choose one of {SEARCHES}"
            )

    def describe(self) -> str:
        text = (
            f"{self.name} = {self.repo_id}@{self.revision[:12]}; beam {self.beam_width}"
            f" x depth {self.max_depth}, guard {self.guard}, library "
            f"{'on' if self.library else 'off'}, state cap "
            f"{self.max_state_atoms if self.max_state_atoms is not None else 'none'}"
        )
        if self.search == "widening":
            text += ", search widening: the width doubles per restart within a budget"
        elif self.search == "segment":
            text += (
                f", search {self.search}: {self.roots} roots, {self.max_segments} "
                f"segments of up to {self.segment_depth} ops, the beam per root"
            )
        return text


BEST = System(
    name="staged-s0-e3",
    repo_id="abhishekraok/wandering-light-stack-policy",
    revision="e5a0b1391c5268c3ae04a49e6862f19a30779bb2",
    beam_width=64,
    max_depth=48,
    guard="both",
    library=True,
    measured=(
        "best-system-command (#158), this exact configuration on the #101 300-task "
        "v4 cohort, UltraDesk CPU: 180 tasks graded, 18 of 43 behaviours graded, "
        "16 verified, 154 s. Library off it is 143 tasks and the same 16, which is "
        "staged-witness-reauthoring (#155)'s own cell for this checkpoint"
    ),
    # 1 << 14: the largest state along any v4 witness on any of its inputs is
    # 124 atoms, along any world-v3 witness 2,642 (measured on #158); the #157
    # runaway sat at a median 19,980 and a peak of 1,998,856.
    max_state_atoms=16_384,
)


@dataclass
class Row:
    index: int
    task: Task
    program: tuple[str, ...] | None
    source: str  # "search", or "library" when a program solved earlier fit
    termination: str
    seconds: float
    executions: int
    fuel: int
    expansions: int
    depth: int
    solved: bool = False
    shown_fit: bool = False
    # `widening_beam_solve`'s per-restart dict, for a widening search that ran.
    widening: dict | None = None

    @property
    def verdict(self) -> str:
        if self.solved:
            return "solved"
        if self.shown_fit:
            return "memorised"
        return "unsolved"

    def record(self) -> dict[str, Any]:
        record = {
            "index": self.index,
            "behaviour": self.task.behaviour,
            "length": self.task.length,
            "examples": self.task.examples,
            "program": list(self.program) if self.program else None,
            "source": self.source,
            "termination": self.termination,
            "seconds": self.seconds,
            "executions": self.executions,
            "fuel": self.fuel,
            "expansions": self.expansions,
            "depth": self.depth,
            "solved": self.solved,
            "shown_fit": self.shown_fit,
        }
        if self.widening is not None:
            record["widening"] = self.widening
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
            f"executions {self.executions:,}; {self.from_library} solved from the "
            f"library; {self.memorised} memorised"
        )


def summarise(rows: Sequence[Row]) -> Summary:
    return Summary(
        tasks=len(rows),
        solved=sum(r.solved for r in rows),
        memorised=sum(r.shown_fit and not r.solved for r in rows),
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
    budgets: Sequence[int | None] | None = None,
    seconds_cap: float | None = None,
    budget_unit: str = "executions",
) -> list[Row]:
    """Attempt every task in order under ``system``; grade after the last clock stops.

    ``budgets[i]``, when given, caps task i's primitive executions, library
    replay included: the search gets what the replay left. ``seconds_cap``
    caps each task's wall-clock, replay included; a search it stops ends
    ``wall_clock_cap``. Both None is the uncapped run `BEST` measures.

    Budgets count ``budget_unit`` (`stack_beam.BUDGET_UNITS`), replay included;
    the segment search budgets executions only, and a widening search needs a
    budget for every task. All three are refused before any task runs.
    """
    if budgets is not None and len(budgets) != len(tasks):
        raise ValueError(f"{len(budgets)} budgets for {len(tasks)} tasks")
    spent_in(budget_unit, 0, 0)
    if system.search == "segment" and budget_unit != "executions":
        raise ValueError("the segment search budgets executions only")
    if system.search == "widening" and (
        budgets is None or any(b is None for b in budgets)
    ):
        raise ValueError("a widening search needs a budget for every task")
    library: list[tuple[str, ...]] = []
    rows: list[Row] = []
    # `run_body` memoises quotation bodies process-globally and for the life of
    # the process. Grading a task runs its held-out examples through that memo,
    # so a second run in the same process would start with work it never paid
    # for and its clock would depend on what ran before it. Clearing here makes
    # every run cold, which is the only clock that means anything across cells.
    run_body.cache_clear()
    # The corpus and the model live for the whole run; freezing them keeps the
    # collector `beam_solve` turns off from scanning them between tasks.
    gc.freeze()
    for index, task in enumerate(tasks):
        started = time.perf_counter()
        deadline = started + seconds_cap if seconds_cap is not None else None
        program: tuple[str, ...] | None = None
        source, termination = "search", "library_miss"
        executions = fuel = expansions = depth = 0
        widening: dict | None = None
        if system.library:
            inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
            outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
            for candidate in library:
                fits, spent, spent_fuel = replay_program(candidate, inputs, outputs)
                executions += spent
                fuel += spent_fuel
                if fits:
                    program, source, termination = candidate, "library", "fit"
                    break
        if program is None:
            budget = budgets[index] if budgets is not None else None
            if budget is not None:
                budget = max(0, budget - spent_in(budget_unit, executions, fuel))
            if system.search == "segment":
                result = segment_beam_solve(
                    task,
                    scorer,
                    beam_width=system.beam_width,
                    roots=system.roots,
                    segment_depth=system.segment_depth,
                    max_segments=system.max_segments,
                    guard=system.guard,
                    budget=budget,
                    max_state_atoms=system.max_state_atoms,
                    deadline=deadline,
                )
            elif system.search == "widening":
                result, widening = widening_beam_solve(
                    task,
                    scorer,
                    budget=budget,
                    start_width=system.beam_width,
                    max_depth=system.max_depth,
                    guard=system.guard,
                    max_state_atoms=system.max_state_atoms,
                    deadline=deadline,
                    budget_unit=budget_unit,
                )
            else:
                result = beam_solve(
                    task,
                    scorer,
                    beam_width=system.beam_width,
                    max_depth=system.max_depth,
                    guard=system.guard,
                    budget=budget,
                    max_state_atoms=system.max_state_atoms,
                    deadline=deadline,
                    budget_unit=budget_unit,
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
            widening=widening,
        )
        rows.append(row)
        if progress is not None:
            progress(row)
    for row in rows:
        score = grade(row.program, row.task) if row.program else None
        row.solved = bool(score and score.solved)
        row.shown_fit = bool(score and score.shown)
    return rows


def host(device: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cuda_device": torch.cuda.get_device_name(0)
        if device != "cpu" and torch.cuda.is_available()
        else None,
    }


def commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_table(rows: Sequence[Row], out: TextIO) -> None:
    width = max((len(r.task.behaviour) for r in rows), default=9)
    width = max(width, len("behaviour"))
    out.write(
        f"{'#':>5}  {'behaviour':<{width}}  {'verdict':<9}  {'seconds':>8}  source\n"
    )
    for row in rows:
        out.write(
            f"{row.index:>5}  {row.task.behaviour:<{width}}  {row.verdict:<9}  "
            f"{row.seconds:>8.3f}  {row.source}\n"
        )


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
    """Run ``system`` over the file's tasks; print the table to ``out``, progress to ``log``."""
    out = sys.stdout if out is None else out
    log = sys.stderr if log is None else log
    path = Path(path)
    tasks = load(path)
    policy_dir = (
        Path(checkpoint)
        if checkpoint is not None
        else fetch_stack_policy(system.repo_id, system.revision)
    )
    scorer = PolicyScorer(load_policy(policy_dir, device=device), device)
    fingerprint = host(device)
    head = commit()
    out.write(f"system  {system.describe()}\n")
    out.write(f"policy  {policy_dir}{'  (local, not BEST)' if checkpoint else ''}\n")
    out.write(
        f"host    {fingerprint['hostname']} {device}, torch {fingerprint['torch']}, "
        f"{fingerprint['threads']} threads, commit {head or 'unknown'}\n"
    )
    out.write(f"tasks   {len(tasks)} from {path}\n")
    out.flush()

    def progress(row: Row) -> None:
        log.write(
            f"{row.index + 1}/{len(tasks)} {row.task.behaviour} {row.termination} "
            f"{row.seconds:.2f}s\n"
        )
        log.flush()

    rows = solve_tasks(tasks, scorer, system, progress=progress)
    summary = summarise(rows)
    write_table(rows, out)
    out.write(summary.line() + "\n")
    out.flush()
    if json_path is not None:
        record = {
            "system": asdict(system),
            "checkpoint": str(policy_dir),
            "checkpoint_is_best": checkpoint is None,
            "host": fingerprint,
            "commit": head,
            "tasks_file": str(path),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "summary": asdict(summary),
            "rows": [row.record() for row in rows],
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.solve",
        description="Run the current best system on a task file.",
    )
    parser.add_argument("tasks", type=Path, help="a stack_pbe payload or corpus dir")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a local checkpoint in place of BEST's; the output says so",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json", type=Path, default=None, help="also write every row and the host here"
    )
    args = parser.parse_args(argv)
    solve_file(
        args.tasks, checkpoint=args.checkpoint, device=args.device, json_path=args.json
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
