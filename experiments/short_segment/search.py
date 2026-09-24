"""short-segment-campaign (#168): one checkpoint on the #101 cohort under one search.

The cohort is compute-optimal-mark-28 (#101)'s 300 v4 eval tasks, 43
behaviours. A cell is one epoch-3 checkpoint of a `staged_arms` run under one
of three searches, each `solve.solve_tasks` (library on, guard both, the #157
state cap, grading after the last task so no held-out example is read during
search):

    flat          `solve.BEST`'s search, width 64 x depth 48, on a local checkpoint
    segment       the segment-reset beam: width 64 per root, ``--roots`` K roots,
                  12 segments of at most the arm's segment depth in ops
                  (``SEGMENT_DEPTHS``: capped 8, staged 16, fused 48, the 95th
                  percentile of each arm's own train segment lengths, in both
                  encodings; ``--segment-depth`` overrides it)
    flat-matched  the flat beam widened to ``--width`` (1024), each task capped at
                  the primitive executions the same checkpoint's ``segment`` cell
                  (same K) spent on it, library replay included, so a gain of
                  ``segment`` over it is the reset's and not the compute's.
                  Two ways it can fail to spend that: at width 1024 a task the
                  segment beam fits cheaply runs out before the wide beam is
                  deep enough, and a task it cannot fit ends ``max_depth``
                  under budget. The cell's ``budget_use`` counts both.

Every task also has a wall-clock cap (``--seconds-cap``, 120 s); a search it
stops ends ``wall_clock_cap`` and the row keeps what it spent. One JSON per
(run dir, arm, seed, search, K, segment depth or width) under ``--out-dir``, skipped when it
exists, so a queue resumes::

    uv run python experiments/short_segment/search.py run \\
        --run-dir checkpoints/staged-arms-20260920 --arm staged --seed 0 --search flat
    uv run python experiments/short_segment/search.py report --out-dir checkpoints/short-segment-search

`report` draw-tests every distinct solved program once with best-system-command
(#158)'s rule (`step3_verify_solves.check`, 10,000 draws per mode, seed 134,
cached in ``<out-dir>/verified.json``), the rule staged-witness-reauthoring
(#155) used too. A behaviour is verified when some solve of it passes.
`run_cells.py` runs a list of cells on a few GPU processes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.best_system_cells as bsc  # noqa: E402
import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
from wandering_light import solve  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    CONFIG_FILE,
    MACHINE,
    PolicyScorer,
    load_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

SEARCHES = ("flat", "segment", "flat-matched")
EPOCH = 3  # staged_arms.EPOCHS: the snapshot every arm is scored at (#154)
ROOTS = 16
MATCHED_WIDTH = 1024
# The advisor's choice (#168): the 95th percentile of each arm's own train
# segment lengths, for both encodings. An arm not listed needs --segment-depth.
SEGMENT_DEPTHS = {"capped": 8, "staged": 16, "fused": 48}
MAX_SEGMENTS = 12
TASK_SECONDS_CAP = 120.0
# `BEST`'s checkpoint is this run's staged s0 epoch 3 (same safetensors sha256,
# 2113f001...); under flat it must give best-system-command (#158)'s 16.
BEST_CELL = ("staged-arms-20260920", "staged", 0)
BEST_VERIFIED = 16


def main_checkout() -> Path:
    """The repository's main working tree, where run dirs live (not a worktree)."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return REPO


DEFAULT_OUT = main_checkout() / "checkpoints" / "short-segment-search"


def segment_depth_for(arm: str, override: int | None = None) -> int:
    """``override`` when given, else the arm's registered segment depth."""
    if override is not None:
        return override
    if arm not in SEGMENT_DEPTHS:
        raise ValueError(
            f"no registered segment depth for arm {arm!r} ({SEGMENT_DEPTHS}); "
            "pass --segment-depth"
        )
    return SEGMENT_DEPTHS[arm]


def system_for(
    search: str,
    *,
    roots: int = ROOTS,
    width: int = MATCHED_WIDTH,
    segment_depth: int = 12,
):
    """The `solve.System` of a search: `BEST` with the one or two settings changed."""
    common = {
        "name": f"short-segment {search}",
        "repo_id": "local",
        "revision": "checkpoint",
        "measured": "short-segment-campaign (#168)",
    }
    if search == "flat":
        return dataclasses.replace(solve.BEST, **common)
    if search == "segment":
        return dataclasses.replace(
            solve.BEST,
            **common,
            search="segment",
            roots=roots,
            segment_depth=segment_depth,
            max_segments=MAX_SEGMENTS,
        )
    if search == "flat-matched":
        return dataclasses.replace(solve.BEST, **common, beam_width=width)
    raise ValueError(f"unknown search {search!r}; choose one of {SEARCHES}")


def search_tag(
    search: str,
    *,
    roots: int = ROOTS,
    width: int = MATCHED_WIDTH,
    segment_depth: int = 12,
) -> str:
    """The cell's name for its search; a matched cell names the segment cell it matches."""
    if search == "flat":
        return f"flat-w{solve.BEST.beam_width}"
    if search == "segment":
        return f"segment-k{roots}-d{segment_depth}"
    if search == "flat-matched":
        return f"flat-matched-w{width}-k{roots}-d{segment_depth}"
    raise ValueError(f"unknown search {search!r}; choose one of {SEARCHES}")


def cell_path(
    out_dir: Path,
    run_dir: Path,
    arm: str,
    seed: int,
    search: str,
    *,
    roots: int = ROOTS,
    width: int = MATCHED_WIDTH,
    segment_depth: int | None = None,
    positions: Sequence[int] | None = None,
) -> Path:
    """``segment_depth`` None is the arm's registered depth."""
    tag = search_tag(
        search,
        roots=roots,
        width=width,
        segment_depth=segment_depth_for(arm, segment_depth),
    )
    subset = f"-subset{len(positions)}" if positions is not None else ""
    return out_dir / run_dir.name / f"{arm}-s{seed}-e{EPOCH}-{tag}{subset}.json"


def cohort_tasks(positions: Sequence[int] | None = None) -> list[tuple[int, Task]]:
    """``(task_index, task)`` of the #101 cohort, in cohort order, or the given positions."""
    cohort = co.cohort(co.corpus())
    return cohort if positions is None else [cohort[p] for p in positions]


def matched_budgets(segment_cell: Path, task_indices: Sequence[int]) -> list[int]:
    """Per task, the executions the segment cell spent on it, in ``task_indices`` order."""
    if not segment_cell.exists():
        raise FileNotFoundError(
            f"flat-matched needs its segment cell first: {segment_cell} does not exist"
        )
    spent = {
        row["task_index"]: row["executions"]
        for row in json.loads(segment_cell.read_text())["rows"]
    }
    missing = [index for index in task_indices if index not in spent]
    if missing:
        raise ValueError(f"{segment_cell} has no row for tasks {missing[:5]}")
    return [spent[index] for index in task_indices]


def checkpoint_encoding(policy: Path) -> str:
    return json.loads((policy / CONFIG_FILE).read_text()).get("encoding", MACHINE)


def budget_use(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Did a matched cell spend its budgets? None for a cell without them."""
    budgeted = [r for r in rows if r.get("budget") is not None]
    if not budgeted:
        return None
    ended: dict[str, int] = defaultdict(int)
    for r in budgeted:
        ended[r["termination"]] += 1
    spent = sum(r["executions"] for r in budgeted)
    total = sum(r["budget"] for r in budgeted)
    return {
        "tasks": len(budgeted),
        "ended_by": dict(sorted(ended.items())),
        "executions": spent,
        "budget": total,
        "spent_fraction": spent / total if total else None,
    }


def run_cell(
    out_dir: Path,
    run_dir: Path,
    arm: str,
    seed: int,
    search: str,
    *,
    roots: int = ROOTS,
    width: int = MATCHED_WIDTH,
    segment_depth: int | None = None,
    device: str = "cpu",
    seconds_cap: float | None = TASK_SECONDS_CAP,
    threads: int = co.CPU_THREADS,
    positions: Sequence[int] | None = None,
    log=print,
) -> Path:
    out = cell_path(
        out_dir,
        run_dir,
        arm,
        seed,
        search,
        roots=roots,
        width=width,
        segment_depth=segment_depth,
        positions=positions,
    )
    if out.exists():
        log(f"skip {out}: exists")
        return out
    depth = segment_depth_for(arm, segment_depth)
    system = system_for(search, roots=roots, width=width, segment_depth=depth)
    policy = hs.policy_dir(run_dir, arm, seed, EPOCH)
    cohort = cohort_tasks(positions)
    indices = [index for index, _ in cohort]
    budgets = budget_source = None
    if search == "flat-matched":
        budget_source = cell_path(
            out_dir,
            run_dir,
            arm,
            seed,
            "segment",
            roots=roots,
            segment_depth=depth,
            positions=positions,
        )
        budgets = matched_budgets(budget_source, indices)
    torch.set_num_threads(threads)
    scorer = PolicyScorer(load_policy(policy, device=device), device)
    log(f"run {out.name}: {system.describe()}; policy {policy} ({scorer.encoding})")

    def progress(row: solve.Row) -> None:
        log(
            f"{row.index + 1}/{len(cohort)} {row.task.behaviour} {row.termination} "
            f"{row.executions:,} exec {row.seconds:.2f}s"
        )

    started = time.perf_counter()
    rows = solve.solve_tasks(
        [task for _, task in cohort],
        scorer,
        system,
        progress=progress,
        budgets=budgets,
        seconds_cap=seconds_cap,
    )
    cell_seconds = time.perf_counter() - started
    records = []
    for task_index, row in zip(indices, rows, strict=True):
        record = row.record()
        record.update(
            {
                "task_index": task_index,
                "bucket": co.bucket_of(row.task.length),
                "budget": budgets[row.index] if budgets is not None else None,
                "capped": row.termination == "wall_clock_cap",
            }
        )
        records.append(record)
    summary = solve.summarise(rows)
    cell = {
        "run_dir": str(run_dir),
        "run": run_dir.name,
        "arm": arm,
        "seed": seed,
        "policy_epoch": EPOCH,
        "checkpoint": str(policy),
        "encoding": checkpoint_encoding(policy),
        "scorer_encoding": scorer.encoding,
        "search": search,
        "search_tag": search_tag(search, roots=roots, width=width, segment_depth=depth),
        "roots": roots if search != "flat" else None,
        # The segment beam's, or for a matched cell the one of the segment cell
        # its budgets come from; None for flat.
        "segment_depth": depth if search != "flat" else None,
        "beam_width": system.beam_width,
        "max_depth": system.max_depth,
        "guard": system.guard,
        "system": dataclasses.asdict(system),
        "task_seconds_cap": seconds_cap,
        "budget_source": str(budget_source) if budget_source else None,
        "cohort_positions": list(positions) if positions is not None else None,
        "device": device,
        "threads": threads,
        "host": solve.host(device),
        "eval_corpus": co.primary_corpus(),
        "provenance": co.provenance(),
        "cell_seconds": cell_seconds,
        "summary": dataclasses.asdict(summary),
        "capped": sum(r["capped"] for r in records),
        "budget_use": budget_use(records),
        "rows": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".json.partial")
    partial.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    os.replace(partial, out)  # a crash never leaves a file that reads as done
    log(
        f"RESULT {run_dir.name} {out.name}: {summary.line()}; "
        f"{cell['capped']} capped; {cell_seconds:.0f}s"
    )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def load_cells(out_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(out_dir.glob("*/*.json"))]


def cell_facts(cell: dict[str, Any], verdicts: dict[str, Any]) -> dict[str, Any]:
    rows = cell["rows"]
    solved = [r for r in rows if r["solved"]]

    def verdict(r):
        return verdicts[f"{r['behaviour']} :: {' '.join(r['program'])}"]["verdict"]

    graded = {r["behaviour"] for r in solved}
    verified = {r["behaviour"] for r in solved if verdict(r) in bsc.VERIFIED}
    executions = [r["executions"] for r in rows]
    return {
        "seed": cell["seed"],
        "tasks": len(rows),
        "graded_tasks": len(solved),
        "graded": sorted(graded),
        "verified": sorted(verified),
        # Shortcuts, two kinds: a program that fits the shown examples and fails
        # the held-out ones, and a graded solve the draw test rejects.
        "heldout_fail_tasks": sum(r["shown_fit"] and not r["solved"] for r in rows),
        "unverified_solve_tasks": sum(verdict(r) not in bsc.VERIFIED for r in solved),
        "unverified_behaviours": sorted(graded - verified),
        "executions": sum(executions),
        "median_executions": statistics.median(executions) if executions else 0,
        "seconds": sum(r["seconds"] for r in rows),
        "capped": sum(r["termination"] == "wall_clock_cap" for r in rows),
        "budget_use": budget_use(rows),
    }


def report(out_dir: Path, out: Path, log=print) -> Path:
    cells = load_cells(out_dir)
    verdicts = bsc.verified_programs(out_dir, cells, log)
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        subset = cell["cohort_positions"] is not None
        key = (
            subset,
            cell["encoding"],
            cell["arm"],
            cell["run"],
            SEARCHES.index(cell["search"]),
            cell["search_tag"],
        )
        groups[key].append(cell_facts(cell, verdicts))
    table = []
    for key in sorted(groups):
        subset, encoding, arm, run, _, tag = key
        facts = sorted(groups[key], key=lambda f: f["seed"])
        verified = [len(f["verified"]) for f in facts]
        table.append(
            {
                "run": run,
                "arm": arm,
                "encoding": encoding,
                "search": tag,
                "subset": subset,
                "is_best": (run, arm) == BEST_CELL[:2]
                and any(f["seed"] == BEST_CELL[2] for f in facts),
                "seeds": [f["seed"] for f in facts],
                "verified_per_seed": verified,
                "verified_mean": statistics.mean(verified),
                "union": sorted(set().union(*(f["verified"] for f in facts))),
                "cells": facts,
            }
        )

    def mean_of(facts, field):
        return statistics.mean(f[field] for f in facts)

    lines = [
        "| arm | encoding | search | run | tasks | verified /43 per seed | mean | "
        "executions (sum) | median exec / task | clock (s) | capped | "
        "held-out fails (tasks) | unverified solves (tasks) | budget spent |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in table:
        facts = t["cells"]
        per_seed = " / ".join(f"s{f['seed']}:{len(f['verified'])}" for f in facts)
        use = [f["budget_use"] for f in facts if f["budget_use"]]
        spent = (
            " / ".join(
                f"{u['ended_by'].get('budget', 0)}/{u['tasks']} ended by budget, "
                f"{u['spent_fraction'] or 0:.2f} spent"
                for u in use
            )
            if use
            else ""
        )
        label = f"**{t['arm']}** (BEST s0)" if t["is_best"] else t["arm"]
        lines.append(
            f"| {label} | {t['encoding']} | {t['search']}{' (subset)' if t['subset'] else ''} "
            f"| {t['run']} | {facts[0]['tasks']} | {per_seed} | {t['verified_mean']:.2f} | "
            f"{mean_of(facts, 'executions'):,.0f} | {mean_of(facts, 'median_executions'):,.0f} | "
            f"{mean_of(facts, 'seconds'):.0f} | {mean_of(facts, 'capped'):.1f} | "
            f"{mean_of(facts, 'heldout_fail_tasks'):.1f} | "
            f"{mean_of(facts, 'unverified_solve_tasks'):.1f} | {spent} |"
        )
    best = [
        f
        for t in table
        if t["is_best"] and t["search"] == search_tag("flat") and not t["subset"]
        for f in t["cells"]
        if f["seed"] == BEST_CELL[2]
    ]
    lines.append("")
    if best:
        got = len(best[0]["verified"])
        lines.append(
            f"Equivalence check: `BEST` (staged s0 e3) under flat verifies {got}; "
            f"best-system-command (#158) measured {BEST_VERIFIED}. "
            + ("Reproduced." if got == BEST_VERIFIED else "**Not reproduced.**")
        )
    else:
        lines.append("Equivalence check: `BEST`'s flat cell has not run yet.")
    lines += [
        "",
        "Executions, clock, capped and shortcut columns are means over seeds. The "
        "clock is the sum of per-task seconds (library replay, search and policy "
        "inference). Budget spent is flat-matched only, per seed.",
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(table, indent=2) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return out / "report.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--search", choices=SEARCHES)
    parser.add_argument("--roots", type=int, default=ROOTS, help="K, segment roots")
    parser.add_argument(
        "--width", type=int, default=MATCHED_WIDTH, help="flat-matched beam width"
    )
    parser.add_argument(
        "--segment-depth",
        type=int,
        default=None,
        help=f"ops per segment; default the arm's, {SEGMENT_DEPTHS}",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seconds-cap", type=float, default=TASK_SECONDS_CAP)
    parser.add_argument("--threads", type=int, default=co.CPU_THREADS)
    parser.add_argument(
        "--positions",
        default=None,
        help="comma-separated cohort positions (0-299), for a smoke test",
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory")
    args = parser.parse_args(argv)

    if args.stage == "report":
        report(args.out_dir, args.out or args.out_dir)
        return 0
    if args.run_dir is None or args.arm is None or args.seed is None or not args.search:
        parser.error("run needs --run-dir, --arm, --seed and --search")
    positions = [int(p) for p in args.positions.split(",")] if args.positions else None
    run_cell(
        args.out_dir,
        args.run_dir,
        args.arm,
        args.seed,
        args.search,
        roots=args.roots,
        width=args.width,
        segment_depth=args.segment_depth,
        device=args.device,
        seconds_cap=args.seconds_cap,
        threads=args.threads,
        positions=positions,
        log=lambda line: print(line, flush=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
