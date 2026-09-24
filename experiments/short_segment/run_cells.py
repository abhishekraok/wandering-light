"""Run short-segment-campaign (#168) search cells on at most ``--jobs`` GPU processes.

Each cell is one `search.py run` subprocess, its output in ``<out-dir>/<run>/``
and its log beside it. A cell whose output exists is skipped, and so is one
whose checkpoint has not finished training (no ``training/<arm>-s<seed>.json``
or an incomplete epoch-3 snapshot): run the queue again when it lands. A
``flat-matched`` cell starts only after its checkpoint's ``segment`` cell has
written its output, since that is where its per-task budgets come from.

    uv run python experiments/short_segment/run_cells.py --dry-run
    uv run python experiments/short_segment/run_cells.py --jobs 3
    uv run python experiments/short_segment/run_cells.py --only staged-arms-20260920:staged:0

At most 3 jobs: a fourth lane of the 1m policy fills the 4090 and stalls
silently (#80's campaign on UltraDesk).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.hindsight_subtargets as hs  # noqa: E402
from experiments.short_segment import search  # noqa: E402
from wandering_light.stack_policy_hub import is_checkpoint  # noqa: E402

SEEDS = (0, 1, 2)
# (run dir under the main checkout's checkpoints/, arms). BEST, staged s0 of the
# first, goes first so the equivalence check is the first number in.
PLAN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("staged-arms-20260920", ("staged", "fused", "refused", "both")),
    ("short-segment-difference-20260923", ("fused", "staged", "capped")),
    ("short-segment-machine-20260923", ("capped",)),
)


@dataclass(frozen=True)
class Cell:
    run_dir: Path
    arm: str
    seed: int
    search: str

    def output(self, out_dir: Path, roots: int, width: int) -> Path:
        return search.cell_path(
            out_dir,
            self.run_dir,
            self.arm,
            self.seed,
            self.search,
            roots=roots,
            width=width,
        )

    def trained(self) -> bool:
        policy = hs.policy_dir(self.run_dir, self.arm, self.seed, search.EPOCH)
        done = self.run_dir / "training" / f"{self.arm}-s{self.seed}.json"
        return done.exists() and is_checkpoint(policy)

    def label(self) -> str:
        return f"{self.run_dir.name}:{self.arm}:s{self.seed}:{self.search}"


def plan(checkpoints: Path, searches, only: set[str] | None) -> list[Cell]:
    """Every cell, flat and segment of every checkpoint before any flat-matched one."""
    triples = [
        (checkpoints / run, arm, seed)
        for run, arms in PLAN
        for arm in arms
        for seed in SEEDS
        if not only or f"{run}:{arm}:{seed}" in only
    ]
    first = [
        Cell(*triple, s)
        for triple in triples
        for s in ("flat", "segment")
        if s in searches
    ]
    matched = [Cell(*triple, "flat-matched") for triple in triples]
    return first + (matched if "flat-matched" in searches else [])


def command(cell: Cell, args) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("search.py")),
        "run",
        "--out-dir",
        str(args.out_dir),
        "--run-dir",
        str(cell.run_dir),
        "--arm",
        cell.arm,
        "--seed",
        str(cell.seed),
        "--search",
        cell.search,
        "--roots",
        str(args.roots),
        "--width",
        str(args.width),
        "--device",
        args.device,
        "--seconds-cap",
        str(args.seconds_cap),
        "--threads",
        str(args.threads),
    ]


def run_queue(cells: list[Cell], args, log=print) -> int:
    """Start cells as lanes free up; return how many failed."""
    out = {cell: cell.output(args.out_dir, args.roots, args.width) for cell in cells}
    pending = []
    for cell in cells:
        if out[cell].exists():
            log(f"skip {cell.label()}: {out[cell].name} exists")
        elif not cell.trained():
            log(f"skip {cell.label()}: checkpoint not trained yet")
        else:
            pending.append(cell)
    if args.dry_run:
        for cell in pending:
            log("would run " + " ".join(command(cell, args)))
        return 0
    running: dict[Cell, subprocess.Popen] = {}
    failed: set[Cell] = set()
    while pending or running:
        for cell, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[cell]
            if code == 0 and out[cell].exists():
                log(f"done {cell.label()}")
            else:
                failed.add(cell)
                log(f"FAILED {cell.label()} (exit {code}); see its .log")
        for cell in list(pending):
            if len(running) >= args.jobs:
                break
            if cell.search == "flat-matched":
                segment = Cell(cell.run_dir, cell.arm, cell.seed, "segment")
                if segment in running or segment in pending:
                    continue  # its budgets are not written yet
                if not segment.output(args.out_dir, args.roots, args.width).exists():
                    pending.remove(cell)
                    failed.add(cell)
                    log(f"FAILED {cell.label()}: its segment cell has no output")
                    continue
            pending.remove(cell)
            out[cell].parent.mkdir(parents=True, exist_ok=True)
            handle = open(out[cell].with_suffix(".log"), "a")  # noqa: SIM115
            running[cell] = subprocess.Popen(
                command(cell, args), stdout=handle, stderr=subprocess.STDOUT, cwd=REPO
            )
            handle.close()  # the child holds its own descriptor
            log(f"start {cell.label()} (pid {running[cell].pid})")
        if running:
            time.sleep(args.poll)
    log(f"queue finished: {len(failed)} failed")
    return len(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=search.main_checkout() / "checkpoints",
        help="where the PLAN's run dirs live",
    )
    parser.add_argument("--out-dir", type=Path, default=search.DEFAULT_OUT)
    parser.add_argument(
        "--searches", default=",".join(search.SEARCHES), help="comma-separated"
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated run:arm:seed triples, e.g. staged-arms-20260920:staged:0",
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roots", type=int, default=search.ROOTS)
    parser.add_argument("--width", type=int, default=search.MATCHED_WIDTH)
    parser.add_argument("--seconds-cap", type=float, default=search.TASK_SECONDS_CAP)
    parser.add_argument("--threads", type=int, default=4, help="torch threads per job")
    parser.add_argument(
        "--poll", type=float, default=10.0, help="seconds between polls"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    searches = [s for s in args.searches.split(",") if s]
    unknown = set(searches) - set(search.SEARCHES)
    if unknown:
        parser.error(f"unknown searches {sorted(unknown)}")
    only = set(args.only.split(",")) if args.only else None
    cells = plan(args.checkpoints, searches, only)

    def log(line: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)

    return 1 if run_queue(cells, args, log) else 0


if __name__ == "__main__":
    sys.exit(main())
