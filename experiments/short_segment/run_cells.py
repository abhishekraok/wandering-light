"""Run the registered short-segment-campaign (#168) search cells on at most ``--jobs`` CPU lanes.

The six registered checkpoints, three seeds each, under flat-48, segment and
flat-widening. Each cell is one `search.py run` subprocess, its output in
``<out-dir>/<run>/`` and its log beside it. A cell whose output exists is
skipped, and so is one whose checkpoint has not finished training (no
``training/<arm>-s<seed>.json`` or an incomplete epoch-3 snapshot): run the
queue again when it lands. A ``flat-widening`` cell starts only after its
checkpoint's ``segment`` and ``flat`` cells have written their outputs, since
its per-task budgets are the larger of their spends; while it waits, later
cells take the lane.

    uv run python experiments/short_segment/run_cells.py --dry-run
    uv run python experiments/short_segment/run_cells.py
    uv run python experiments/short_segment/run_cells.py --only short-segment-difference-20260923:capped:0

Two lanes of 8 threads by default: the standing rule on UltraDesk is at most two
CPU processes of 8 threads, since four once cut throughput fourfold.
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
from wandering_light.stack_beam import BUDGET_UNITS  # noqa: E402
from wandering_light.stack_policy_hub import is_checkpoint  # noqa: E402

SEEDS = (0, 1, 2)
DIFFERENCE = "short-segment-difference-20260923"
MACHINE = "short-segment-machine-20260923"
STAGED_ARMS = "staged-arms-20260920"
# The six registered cells, (run dir under the main checkout's checkpoints/,
# arm), in queue order: capped/difference, capped/machine, then staged and fused
# in each encoding. Machine: staged-arms fused and staged, and the machine run's
# capped; difference: the difference run's fused, staged and capped.
REGISTERED: tuple[tuple[str, str], ...] = (
    (DIFFERENCE, "capped"),
    (MACHINE, "capped"),
    (STAGED_ARMS, "staged"),
    (DIFFERENCE, "staged"),
    (STAGED_ARMS, "fused"),
    (DIFFERENCE, "fused"),
)


@dataclass(frozen=True)
class Cell:
    run_dir: Path
    arm: str
    seed: int
    search: str

    def output(self, args) -> Path:
        return search.cell_path(
            args.out_dir,
            self.run_dir,
            self.arm,
            self.seed,
            self.search,
            roots=args.roots,
            segment_depth=args.segment_depth,
            budget_unit=args.budget_unit,
        )

    def trained(self) -> bool:
        policy = hs.policy_dir(self.run_dir, self.arm, self.seed, search.EPOCH)
        done = self.run_dir / "training" / f"{self.arm}-s{self.seed}.json"
        return done.exists() and is_checkpoint(policy)

    def label(self) -> str:
        return f"{self.run_dir.name}:{self.arm}:s{self.seed}:{self.search}"


def plan(checkpoints: Path, searches, only: set[str] | None) -> list[Cell]:
    """Seed 0 of every registered cell, in `REGISTERED` order, then seeds 1-2.

    `BEST`'s flat cell (staged-arms staged s0) is moved up to follow
    capped/difference s0, so the equivalence check is among the first numbers.
    """
    order = [s for s in search.SEARCHES if s in searches]
    cells = [
        Cell(checkpoints / run, arm, seed, s)
        for seeds in ((0,), SEEDS[1:])
        for run, arm in REGISTERED
        for seed in seeds
        for s in order
        if not only or f"{run}:{arm}:{seed}" in only
    ]
    best = Cell(checkpoints / search.BEST_CELL[0], *search.BEST_CELL[1:], "flat")
    if best in cells:
        cells.remove(best)
        first = sum(
            c.run_dir.name == DIFFERENCE and c.arm == "capped" and c.seed == 0
            for c in cells
        )
        cells.insert(first, best)
    return cells


def command(cell: Cell, args) -> list[str]:
    return (
        [
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
            "--budget-unit",
            args.budget_unit,
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--lanes",
            str(args.jobs),
        ]
        + (
            ["--segment-depth", str(args.segment_depth)]
            if args.segment_depth is not None
            else []
        )
        + (
            ["--seconds-cap", str(args.seconds_cap)]
            if args.seconds_cap is not None
            else []
        )
    )


def run_queue(cells: list[Cell], args, log=print) -> int:
    """Start cells as lanes free up; return how many failed."""
    out = {cell: cell.output(args) for cell in cells}
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
            if cell.search == "flat-widening":
                sources = [
                    Cell(cell.run_dir, cell.arm, cell.seed, s)
                    for s in ("segment", "flat")
                ]
                if any(s in running or s in pending for s in sources):
                    continue  # its budgets are not written yet
                missing = [s.search for s in sources if not s.output(args).exists()]
                if missing:
                    pending.remove(cell)
                    failed.add(cell)
                    log(f"FAILED {cell.label()}: no {' or '.join(missing)} output")
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
        help="where the registered run dirs live",
    )
    parser.add_argument("--out-dir", type=Path, default=search.DEFAULT_OUT)
    parser.add_argument(
        "--searches", default=",".join(search.SEARCHES), help="comma-separated"
    )
    parser.add_argument(
        "--only",
        default=None,
        help=f"comma-separated run:arm:seed triples, e.g. {DIFFERENCE}:capped:0",
    )
    parser.add_argument("--jobs", type=int, default=2, help="lanes side by side")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--roots", type=int, default=search.ROOTS)
    parser.add_argument(
        "--segment-depth",
        type=int,
        default=None,
        help="override every arm's registered segment depth",
    )
    parser.add_argument(
        "--budget-unit", choices=BUDGET_UNITS, default=search.BUDGET_UNIT
    )
    parser.add_argument(
        "--seconds-cap",
        type=float,
        default=None,
        help="override the per-search default (120 s, 300 s for flat-widening)",
    )
    parser.add_argument("--threads", type=int, default=8, help="torch threads per lane")
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
