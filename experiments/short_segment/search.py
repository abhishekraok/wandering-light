"""short-segment-campaign (#168): one checkpoint on the #101 cohort under one search.

The cohort is compute-optimal-mark-28 (#101)'s 300 v4 eval tasks, 43
behaviours. A cell is one epoch-3 checkpoint of a `staged_arms` run under one
of three searches, each `solve.solve_tasks` (library on, guard both, the #157
state cap, grading after the last task so no held-out example is read during
search):

    flat           `solve.BEST`'s search, width 64 x depth 48, on a local
                   checkpoint: the baseline column, flat-48
    segment        the segment-reset beam: width 64 per root, K = ``--roots``
                   (16) roots, 12 segments of at most the arm's segment depth in
                   ops (``SEGMENT_DEPTHS``: capped 8, staged 16, fused 48, the
                   95th percentile of each arm's own train segment lengths, in
                   both encodings; ``--segment-depth`` overrides it)
    flat-widening  the matched control: `stack_beam.widening_beam_solve` at the
                   segment cell's token horizon (12 x its segment depth: 96,
                   192, 576), width 64 then 128, 256, ... from scratch each time,
                   all restarts on one counter capped per task at what the same
                   checkpoint's segment cell spent on that task, library replay
                   included, so a gain of segment over it is the reset's and not
                   the compute's. ``--budget-unit`` is the counter (below).

Every task also has a wall-clock cap (``--seconds-cap``, 120 s); a search it
stops ends ``wall_clock_cap`` and the row keeps what it spent. One JSON per
(run dir, arm, seed, search, K, segment depth, budget unit) under
``--out-dir``, skipped when it exists, so a queue resumes::

    uv run python experiments/short_segment/search.py run \\
        --run-dir checkpoints/staged-arms-20260920 --arm staged --seed 0 --search flat
    uv run python experiments/short_segment/search.py report --out-dir checkpoints/short-segment-search

**Counters.** ``executions`` is one per op applied to one shown example stack,
the unit best-system-command (#158) and the registration call primitive
executions; a combinator counts once however much its body runs. ``fuel`` is
the machine's cost of those same applications: 1 for the op plus every step of
any quotation body it runs. Both are charged at the same place in both beams,
before each op attempt, and library replay charges the same units. The
widening budget counts ``executions+fuel`` by default (``fuel`` alone, or
``executions``, by flag).

`report` draw-tests every distinct solved program once with best-system-command
(#158)'s rule (`step3_verify_solves.check`, 10,000 draws per mode, seed 134,
cached in ``<out-dir>/verified.json``), the rule staged-witness-reauthoring
(#155) used too. A behaviour is verified when some solve of it passes.
`run_cells.py` runs the registered cells on at most two CPU lanes.
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
from wandering_light.stack_beam import BUDGET_UNITS, spent_in  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    CONFIG_FILE,
    MACHINE,
    PolicyScorer,
    load_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

SEARCHES = ("flat", "segment", "flat-widening")
EPOCH = 3  # staged_arms.EPOCHS: the snapshot every arm is scored at (#154)
ROOTS = 16
# The advisor's choice (#168): the 95th percentile of each arm's own train
# segment lengths, for both encodings. An arm not listed needs --segment-depth.
SEGMENT_DEPTHS = {"capped": 8, "staged": 16, "fused": 48}
MAX_SEGMENTS = 12
BUDGET_UNIT = "executions+fuel"
UNIT_TAGS = {"executions": "exec", "fuel": "fuel", "executions+fuel": "execfuel"}
TASK_SECONDS_CAP = 120.0
MERGE_RULE = (
    "equal closed states (structurally equal products) merge across roots, keeping "
    "the highest cumulative log-probability and its program (a tie keeps the first "
    "merged); the top K unexpanded ones by that score are the next roots"
)
# Multiples of flat-48's own per-task cost for the paired solved-within-budget curves.
CURVE_MULTIPLES = (1, 4, 16, 45)
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


def system_for(search: str, *, roots: int = ROOTS, segment_depth: int = 12):
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
    if search == "flat-widening":
        return dataclasses.replace(
            solve.BEST,
            **common,
            search="widening",
            max_depth=MAX_SEGMENTS * segment_depth,
        )
    raise ValueError(f"unknown search {search!r}; choose one of {SEARCHES}")


def search_tag(
    search: str,
    *,
    roots: int = ROOTS,
    segment_depth: int = 12,
    budget_unit: str = BUDGET_UNIT,
) -> str:
    """The cell's name for its search; a widening cell names the segment cell it matches."""
    if search == "flat":
        return f"flat-w{solve.BEST.beam_width}"
    if search == "segment":
        return f"segment-k{roots}-d{segment_depth}"
    if search == "flat-widening":
        return (
            f"flat-widening-k{roots}-d{segment_depth}-h{MAX_SEGMENTS * segment_depth}"
            f"-{UNIT_TAGS[budget_unit]}"
        )
    raise ValueError(f"unknown search {search!r}; choose one of {SEARCHES}")


def cell_path(
    out_dir: Path,
    run_dir: Path,
    arm: str,
    seed: int,
    search: str,
    *,
    roots: int = ROOTS,
    segment_depth: int | None = None,
    budget_unit: str = BUDGET_UNIT,
    positions: Sequence[int] | None = None,
) -> Path:
    """``segment_depth`` None is the arm's registered depth."""
    tag = search_tag(
        search,
        roots=roots,
        segment_depth=segment_depth_for(arm, segment_depth),
        budget_unit=budget_unit,
    )
    subset = f"-subset{len(positions)}" if positions is not None else ""
    return out_dir / run_dir.name / f"{arm}-s{seed}-e{EPOCH}-{tag}{subset}.json"


def cohort_tasks(positions: Sequence[int] | None = None) -> list[tuple[int, Task]]:
    """``(task_index, task)`` of the #101 cohort, in cohort order, or the given positions."""
    cohort = co.cohort(co.corpus())
    return cohort if positions is None else [cohort[p] for p in positions]


def matched_budgets(
    segment_cell: Path, task_indices: Sequence[int], unit: str = "executions"
) -> list[int]:
    """Per task, what the segment cell spent on it in ``unit``, in ``task_indices`` order."""
    if not segment_cell.exists():
        raise FileNotFoundError(
            f"flat-widening needs its segment cell first: {segment_cell} does not exist"
        )
    spent = {
        row["task_index"]: spent_in(unit, row["executions"], row["fuel"])
        for row in json.loads(segment_cell.read_text())["rows"]
    }
    missing = [index for index in task_indices if index not in spent]
    if missing:
        raise ValueError(f"{segment_cell} has no row for tasks {missing[:5]}")
    return [spent[index] for index in task_indices]


def checkpoint_encoding(policy: Path) -> str:
    return json.loads((policy / CONFIG_FILE).read_text()).get("encoding", MACHINE)


def row_record(
    row: solve.Row, task_index: int, budget: int | None, unit: str | None
) -> dict[str, Any]:
    record = row.record()
    fit = row.program is not None
    record.update(
        {
            "task_index": task_index,
            "bucket": co.bucket_of(row.task.length),
            "budget": budget,
            "budget_unit": unit if budget is not None else None,
            "capped": row.termination == "wall_clock_cap",
            # Every search returns at its first shown fit, so the counters of a
            # row with a program are the counters at that fit.
            "executions_at_fit": row.executions if fit else None,
            "fuel_at_fit": row.fuel if fit else None,
        }
    )
    if budget is not None:
        spent = spent_in(unit, row.executions, row.fuel)
        record["budget_spent"] = spent
        record["budget_share"] = spent / budget if budget else None
        record["overshoot"] = max(0, spent - budget)
    if row.widening is not None:
        record["fit_width"] = row.widening["fit_width"]
        record["largest_width_completed"] = row.widening["largest_width_completed"]
    return record


def budget_use(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Did a widening cell spend its budgets? None for a cell without them."""
    budgeted = [r for r in rows if r.get("budget") is not None]
    if not budgeted:
        return None
    ended: dict[str, int] = defaultdict(int)
    for r in budgeted:
        ended[r["termination"]] += 1
    spent = sum(r["budget_spent"] for r in budgeted)
    total = sum(r["budget"] for r in budgeted)
    # A task the library fit never searched, so it has no widths.
    widest = [r.get("largest_width_completed") or 0 for r in budgeted]
    return {
        "unit": budgeted[0]["budget_unit"],
        "tasks": len(budgeted),
        "ended_by": dict(sorted(ended.items())),
        "spent": spent,
        "budget": total,
        "spent_fraction": spent / total if total else None,
        "overshoot": sum(r["overshoot"] for r in budgeted),
        "overshoot_tasks": sum(r["overshoot"] > 0 for r in budgeted),
        "saturated": sum(
            bool((r.get("widening") or {}).get("saturated")) for r in rows
        ),
        "median_largest_width_completed": statistics.median(widest),
        "max_largest_width_completed": max(widest),
    }


def run_cell(
    out_dir: Path,
    run_dir: Path,
    arm: str,
    seed: int,
    search: str,
    *,
    roots: int = ROOTS,
    segment_depth: int | None = None,
    budget_unit: str = BUDGET_UNIT,
    device: str = "cpu",
    seconds_cap: float | None = TASK_SECONDS_CAP,
    threads: int = co.CPU_THREADS,
    lanes: int | None = None,
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
        segment_depth=segment_depth,
        budget_unit=budget_unit,
        positions=positions,
    )
    if out.exists():
        log(f"skip {out}: exists")
        return out
    depth = segment_depth_for(arm, segment_depth)
    system = system_for(search, roots=roots, segment_depth=depth)
    policy = hs.policy_dir(run_dir, arm, seed, EPOCH)
    cohort = cohort_tasks(positions)
    indices = [index for index, _ in cohort]
    budgets = budget_source = None
    unit = budget_unit if search == "flat-widening" else None
    if search == "flat-widening":
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
        budgets = matched_budgets(budget_source, indices, budget_unit)
    torch.set_num_threads(threads)
    scorer = PolicyScorer(load_policy(policy, device=device), device)
    log(f"run {out.name}: {system.describe()}; policy {policy} ({scorer.encoding})")

    def progress(row: solve.Row) -> None:
        log(
            f"{row.index + 1}/{len(cohort)} {row.task.behaviour} {row.termination} "
            f"{row.executions:,} exec {row.fuel:,} fuel {row.seconds:.2f}s"
        )

    started = time.perf_counter()
    rows = solve.solve_tasks(
        [task for _, task in cohort],
        scorer,
        system,
        progress=progress,
        budgets=budgets,
        seconds_cap=seconds_cap,
        budget_unit=unit or "executions",
    )
    cell_seconds = time.perf_counter() - started
    records = [
        row_record(
            row, task_index, budgets[row.index] if budgets is not None else None, unit
        )
        for task_index, row in zip(indices, rows, strict=True)
    ]
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
        "search_tag": search_tag(
            search, roots=roots, segment_depth=depth, budget_unit=budget_unit
        ),
        # Frozen per cell: K, the depth (the segment beam's, or for a widening
        # cell that of the segment cell its budgets come from), the merge rule.
        "roots": roots if search != "flat" else None,
        "segment_depth": depth if search != "flat" else None,
        "max_segments": MAX_SEGMENTS if search != "flat" else None,
        "merge_rule": MERGE_RULE if search == "segment" else None,
        "budget_unit": unit,
        "beam_width": system.beam_width,
        "max_depth": system.max_depth,
        "guard": system.guard,
        "system": dataclasses.asdict(system),
        "task_seconds_cap": seconds_cap,
        "budget_source": str(budget_source) if budget_source else None,
        "cohort_positions": list(positions) if positions is not None else None,
        "device": device,
        "threads": threads,
        "lanes": lanes,
        "host": solve.host(device),
        "eval_corpus": co.primary_corpus(),
        "provenance": co.provenance(),
        "cell_seconds": cell_seconds,
        # Throughput of this process, one lane of `lanes` running side by side.
        "tasks_per_hour": len(rows) / cell_seconds * 3600 if cell_seconds else None,
        "summary": dataclasses.asdict(summary),
        "fuel": sum(r.fuel for r in rows),
        "capped": sum(r["capped"] for r in records),
        "budget_use": budget_use(records),
        "rows": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".json.partial")
    partial.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    os.replace(partial, out)  # a crash never leaves a file that reads as done
    log(
        f"RESULT {run_dir.name} {out.name}: {summary.line()}; fuel {cell['fuel']:,}; "
        f"{cell['capped']} capped; {cell_seconds:.0f}s, "
        f"{cell['tasks_per_hour']:.0f} tasks/h"
    )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

NOTES = f"""\
## What the searches share

- **Counters.** `executions` is one per op applied to one shown example stack:
  flat's (and widening's) local counter in `_beam_solve` and `_Tally.executions`
  in the segment beam are incremented by the same `step_all_quiet` count at the
  same place, before each op attempt, and library replay charges the same unit.
  A combinator counts once whatever its body runs. `fuel` is those same
  applications' machine cost, 1 for the op plus every quotation-body step, so
  `fuel - executions` is the body steps. "Primitive executions", the registered
  column, is `executions`; the widening budget counts the unit in its cell
  (`budget_unit`), from the segment row's counters. A restart that passes its
  budget mid-attempt is charged in full; `overshoot` reports it.
- **Verification point.** Both beams test every executable successor against
  the shown outputs, `moved == outputs`, as soon as it is made: flat in
  `_beam_solve` before ranking, the guard or the visited set; the segment beam
  in `_run_segment` before the open/closed split, ranking and pruning, and
  `_segment_beam_solve` passes `stop_at_fit=True`, so it stops at the first
  shown fit, as flat and widening do. Held-out examples are read only when
  grading, after the last task.
- **Merge rule.** {MERGE_RULE}. K = {ROOTS} and each arm's segment depth
  ({SEGMENT_DEPTHS}, {MAX_SEGMENTS} segments) are recorded in every cell.
- **Horizon.** flat-48 is `BEST`'s search at depth 48. flat-widening runs at the
  segment cell's token horizon, {MAX_SEGMENTS} x its depth.
- **Bearing, not a result.** The 6-task CPU smoke of 2026-09-24 ran `BEST` at
  the old segment depth 12: flat-48 solved 5/6 with 613k executions, the segment
  beam 4/6 with 27.5M (45x), missing capped_sum, which flat-48 fits at 117k.
"""


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
        "fuel": sum(r.get("fuel", 0) for r in rows),
        "seconds": sum(r["seconds"] for r in rows),
        "tasks_per_hour": cell.get("tasks_per_hour"),
        "capped": sum(r["termination"] == "wall_clock_cap" for r in rows),
        "budget_use": cell.get("budget_use") if cell["search"] != "flat" else None,
    }


def verified_rows(cell: dict[str, Any], verdicts: dict[str, Any]) -> list[dict]:
    return [
        r
        for r in cell["rows"]
        if r["solved"]
        and verdicts[f"{r['behaviour']} :: {' '.join(r['program'])}"]["verdict"]
        in bsc.VERIFIED
    ]


def paired(
    cells: list[dict[str, Any]], verdicts: dict[str, Any], unit: str
) -> list[dict[str, Any]]:
    """Per checkpoint seed: each search's gains and losses against flat-48, and curves.

    A curve point at multiple m counts the tasks a search solved (graded) and the
    behaviours it verified with a fit costing at most m times flat-48's own cost
    on that task, both counted in ``unit``.
    """
    by_checkpoint: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for cell in cells:
        if cell["cohort_positions"] is not None:
            continue
        key = (cell["run"], cell["arm"], cell["encoding"], cell["seed"])
        by_checkpoint[key][cell["search_tag"]] = cell
    out = []
    for (run, arm, encoding, seed), searches in sorted(by_checkpoint.items()):
        flat = searches.get(search_tag("flat"))
        if flat is None:
            continue
        cost = {
            r["task_index"]: spent_in(unit, r["executions"], r["fuel"])
            for r in flat["rows"]
        }
        base = {r["behaviour"] for r in verified_rows(flat, verdicts)}
        for tag, cell in sorted(searches.items()):
            good = verified_rows(cell, verdicts)
            mine = {r["behaviour"] for r in good}
            curve = {}
            for multiple in CURVE_MULTIPLES:

                def within(r, multiple=multiple, cost=cost):
                    fit = spent_in(unit, r["executions"], r["fuel"])
                    return r["program"] is not None and fit <= multiple * cost.get(
                        r["task_index"], 0
                    )

                curve[multiple] = {
                    "tasks": sum(r["solved"] and within(r) for r in cell["rows"]),
                    "verified_behaviours": len(
                        {r["behaviour"] for r in good if within(r)}
                    ),
                }
            out.append(
                {
                    "run": run,
                    "arm": arm,
                    "encoding": encoding,
                    "seed": seed,
                    "search": tag,
                    "verified": len(mine),
                    "gains": sorted(mine - base),
                    "losses": sorted(base - mine),
                    "curve": curve,
                }
            )
    return out


def report(out_dir: Path, out: Path, log=print, unit: str = BUDGET_UNIT) -> Path:
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
        values = [f[field] for f in facts if f[field] is not None]
        return statistics.mean(values) if values else 0

    lines = [
        "| arm | encoding | search | run | tasks | verified /43 per seed | mean | "
        "primitive executions (sum) | fuel (sum) | median exec / task | clock (s) | "
        "tasks/h/lane | capped | held-out fails (tasks) | unverified solves (tasks) | "
        "budget spent |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in table:
        facts = t["cells"]
        per_seed = " / ".join(f"s{f['seed']}:{len(f['verified'])}" for f in facts)
        use = [f["budget_use"] for f in facts if f["budget_use"]]
        spent = " / ".join(
            f"{u['ended_by'].get('budget', 0)}/{u['tasks']} ended by budget, "
            f"{u['spent_fraction'] or 0:.2f} of {u['unit']} spent, overshoot "
            f"{u['overshoot']:,} on {u['overshoot_tasks']}, widest {u['max_largest_width_completed']}"
            for u in use
        )
        label = f"**{t['arm']}** (BEST s0)" if t["is_best"] else t["arm"]
        lines.append(
            f"| {label} | {t['encoding']} | {t['search']}{' (subset)' if t['subset'] else ''} "
            f"| {t['run']} | {facts[0]['tasks']} | {per_seed} | {t['verified_mean']:.2f} | "
            f"{mean_of(facts, 'executions'):,.0f} | {mean_of(facts, 'fuel'):,.0f} | "
            f"{mean_of(facts, 'median_executions'):,.0f} | "
            f"{mean_of(facts, 'seconds'):.0f} | {mean_of(facts, 'tasks_per_hour'):.0f} | "
            f"{mean_of(facts, 'capped'):.1f} | "
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
        "Executions, fuel, clock, capped and shortcut columns are means over seeds. "
        "The clock is the sum of per-task seconds (library replay, search and policy "
        "inference). Budget spent is flat-widening only, per seed.",
    ]
    pairs = paired(cells, verdicts, unit)
    if pairs:
        header = " | ".join(f"{m}x" for m in CURVE_MULTIPLES)
        lines += [
            "",
            "## Per checkpoint against flat-48",
            "",
            f"Solved within m x flat-48's own per-task cost, in {unit}: graded tasks "
            "/ verified behaviours.",
            "",
            f"| arm | encoding | run | seed | search | verified | {header} | "
            "gains over flat-48 | losses |",
            "|---|---|---|---:|---|---:|" + "---:|" * len(CURVE_MULTIPLES) + "---|---|",
        ]
        for p in pairs:
            points = " | ".join(
                f"{p['curve'][m]['tasks']} / {p['curve'][m]['verified_behaviours']}"
                for m in CURVE_MULTIPLES
            )
            lines.append(
                f"| {p['arm']} | {p['encoding']} | {p['run']} | {p['seed']} | "
                f"{p['search']} | {p['verified']} | {points} | "
                f"{', '.join(p['gains']) or '-'} | {', '.join(p['losses']) or '-'} |"
            )
    lines += ["", NOTES]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps({"table": table, "paired": pairs}, indent=2) + "\n"
    )
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
        "--segment-depth",
        type=int,
        default=None,
        help=f"ops per segment; default the arm's, {SEGMENT_DEPTHS}",
    )
    parser.add_argument(
        "--budget-unit",
        choices=BUDGET_UNITS,
        default=BUDGET_UNIT,
        help="what the widening budget counts; the report's curves too",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds-cap", type=float, default=TASK_SECONDS_CAP)
    parser.add_argument("--threads", type=int, default=co.CPU_THREADS)
    parser.add_argument(
        "--lanes", type=int, default=None, help="processes side by side, recorded"
    )
    parser.add_argument(
        "--positions",
        default=None,
        help="comma-separated cohort positions (0-299), for a smoke test",
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory")
    args = parser.parse_args(argv)

    if args.stage == "report":
        report(args.out_dir, args.out or args.out_dir, unit=args.budget_unit)
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
        segment_depth=args.segment_depth,
        budget_unit=args.budget_unit,
        device=args.device,
        seconds_cap=args.seconds_cap,
        threads=args.threads,
        lanes=args.lanes,
        positions=positions,
        log=lambda line: print(line, flush=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
