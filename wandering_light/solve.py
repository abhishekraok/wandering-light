"""Run the current best system on a task file.

    uv run python -m wandering_light.solve TASKS

``TASKS`` is a `stack_pbe` payload (``.jsonl`` or ``.jsonl.gz``) or a corpus
directory, read with `stack_pbe.load`. Every task is attempted in file order,
and the system is exactly one thing, `BEST`: a published checkpoint pinned to a
Hub commit, a beam width, a depth, a guard, and whether programs solved earlier
in the run are replayed before each search. The checkpoint is fetched once
into `stack_policy_hub.default_cache_dir` and read from there after that.

The protocol is `docs/metrics.md`'s cost to clear. One clock per task covers
the library replay and the search; grading on the held-out examples happens
after the last clock stops, so nothing the solver does is warmed by an example
it was not shown. The verdict per task is ``solved`` (right on data it was not
given), ``memorised`` (fits the shown examples only) or ``unsolved``; coverage
is solved over tasks attempted, and the wall-clock is summed over every task,
solved or not, because an unsolved task is paid for too.
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

from wandering_light import stack_pbe
from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy_hub import fetch_stack_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class System:
    """Everything that decides what the solver does, in one record."""

    name: str
    repo_id: str
    revision: str  # a Hub commit id
    beam_width: int
    max_depth: int
    guard: str  # off | close | type | both
    library: bool  # replay programs solved earlier in the run before searching
    measured: str  # where the numbers behind choosing it were taken

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
    "The staged arm's seed 0 at epoch 3 of staged-witness-reauthoring (#155), "
    "measured on the 300-task, 43-behaviour stack_pbe_v4 eval cohort at "
    "beam 64 x depth 48 (reports/staged-arms-20260920); the library follows "
    "solved-program-cache (#151).",
)


@dataclass(slots=True)
class Row:
    """One task's attempt. Filled in during the run, graded after it."""

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
        """The JSON-able row: the task by its labels, everything else as is."""
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


@dataclass(frozen=True, slots=True)
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
    """Attempt every task in order under ``system``; grade after the last clock.

    With ``system.library`` the programs found so far, shortest first, are
    replayed on the task's shown examples inside its clock, charged like the
    beam, and the first that fits is the answer. Otherwise the beam searches.
    A program the beam finds joins the library (no duplicates, sorted by
    length). ``progress`` sees each row as its clock stops, before grading.
    """
    library: list[tuple[str, ...]] = []
    rows: list[Row] = []
    # The corpus and the model live for the whole run; freezing moves them out
    # of the generations the collector scans, which `beam_solve` turns off.
    gc.freeze()
    for index, task in enumerate(tasks):
        started = time.perf_counter()
        program: tuple[str, ...] | None = None
        source, termination = "search", "library_miss"
        executions = fuel = expansions = depth = 0
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
    # Grading is sealed off from the clocks: it runs the executor, whose
    # process-global LRU would otherwise warm later searches.
    for row in rows:
        if row.program is None:
            continue
        score = grade(row.program, row.task)
        row.solved = score.solved
        row.shown_fit = score.shown
    return rows


def host() -> dict[str, Any]:
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def _host_line(info: dict[str, Any]) -> str:
    device = info["cuda_device"] or "no cuda"
    return (
        f"{info['hostname']}; {info['platform']}; torch {info['torch']}, "
        f"{info['threads']} threads; {device}"
    )


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


def solve_file(
    path: str | Path,
    *,
    system: System = BEST,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    json_path: str | Path | None = None,
    out=None,
    log=None,
) -> Summary:
    """Run ``system`` on every task in ``path``; print the verdicts; return the summary.

    ``checkpoint`` overrides the system's published policy with a local
    directory, which the header and the JSON both say, so a number from a
    local checkpoint is never mistaken for one from `BEST`.
    """
    from wandering_light.stack_policy import PolicyScorer, load_policy

    out = sys.stdout if out is None else out
    log = sys.stderr if log is None else log
    tasks_path = Path(path)
    tasks = stack_pbe.load(tasks_path)
    if checkpoint is not None:
        policy_dir = Path(checkpoint)
        policy_note = "  (local, not BEST)"
    else:
        policy_dir = fetch_stack_policy(system.repo_id, system.revision)
        policy_note = ""
    scorer = PolicyScorer(load_policy(policy_dir, device=device), device)
    info = host()

    print(f"system  {system.describe()}", file=out)
    print(f"policy  {policy_dir}{policy_note}", file=out)
    print(f"host    {_host_line(info)}", file=out)
    print(f"tasks   {len(tasks)} from {tasks_path}", file=out, flush=True)

    def progress(row: Row) -> None:
        print(
            f"[{row.index + 1}/{len(tasks)}] {row.task.behaviour}: "
            f"{row.termination} by {row.source} in {row.seconds:.3f} s, "
            f"{row.executions:,} executions",
            file=log,
            flush=True,
        )

    rows = solve_tasks(tasks, scorer, system, progress=progress)
    summary = summarise(rows)

    w = max([9, *(len(r.task.behaviour) for r in rows)])
    print(
        f"{'#':>5}  {'behaviour':<{w}}  {'verdict':<9}  {'seconds':>8}  source",
        file=out,
    )
    for r in rows:
        print(
            f"{r.index:>5}  {r.task.behaviour:<{w}}  {r.verdict:<9}  "
            f"{r.seconds:>8.3f}  {r.source}",
            file=out,
        )
    print(summary.line(), file=out, flush=True)

    if json_path is not None:
        record = {
            "system": asdict(system),
            "checkpoint": str(policy_dir),
            "checkpoint_is_best": checkpoint is None,
            "host": info,
            "commit": _commit(),
            "tasks_file": str(tasks_path),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "summary": {**asdict(summary), "coverage": summary.coverage},
            "rows": [r.record() for r in rows],
        }
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.solve",
        description="Run the current best system on a task file.",
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
        help="a local stack-policy checkpoint instead of BEST's published one",
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
