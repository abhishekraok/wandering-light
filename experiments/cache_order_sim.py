"""What the solved-program cache is worth, over any number of task orders, exactly.

The cache does not change what a search returns; it only decides whether a task
is searched at all. So one recorded **cache-off** cell plus a fits matrix -- does
program p reproduce task t's shown examples, and does it survive held-out
grading there -- determines the cache-on outcome for *every* order without
searching again. Live cells then only have to price the replay's wall-clock.

That matters because the cohort's own order is privileged: `cohort()` returns
tasks bucket-major, and witness-length bucket is not visible to a solver that
sees only shown examples, so it is not an order the system could choose. The
headline here is the mean over random orders; bucket-major is reported beside it
as the upper bound it is.

Three things this reports that a solve count hides:

- **behaviours**, not only tasks. The cohort is 43 behaviours behind 300 tasks.
  A cache converts per-instance search variance into within-behaviour coverage
  and cannot solve a behaviour no search solved, so the behaviour count is the
  number that says whether anything new was learned.
- **regressions**: tasks the cache-off run verified and the cache-on run lost,
  because a shorter cached program fit the shown examples first and failed
  held-out. Shortest-first replay prefers exactly the shown-pair coincidences.
- **the split of cache hits** into ones that verify and ones that do not.

    uv run python experiments/cache_order_sim.py --cell <cache-off cell.json> \
        --out reports/.../order_sim.json --orders 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments.compute_optimal_mark28 as co  # noqa: E402
from experiments.solve_with_cache import replay_program  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS  # noqa: E402
from wandering_light.stack_pbe_eval import grade  # noqa: E402

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task

RULES = ("shortest-first", "longest-first", "insertion-order")


def fits_matrix(
    cohort: list[tuple[int, Task]], programs: list[tuple[str, ...]]
) -> dict[int, dict[int, tuple[bool, bool, int]]]:
    """``task position -> program index -> (fits shown, survives held-out, executions)``."""
    matrix: dict[int, dict[int, tuple[bool, bool, int]]] = {}
    for position, (_index, task) in enumerate(cohort):
        inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
        outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
        row: dict[int, tuple[bool, bool, int]] = {}
        for p, program in enumerate(programs):
            fits, spent = replay_program(program, inputs, outputs)
            solved = bool(grade(program, task).solved) if fits else False
            row[p] = (fits, solved, spent)
        matrix[position] = row
    return matrix


def simulate(
    order: list[int],
    cohort: list[tuple[int, Task]],
    searched: dict[int, dict[str, Any]],
    program_index: dict[tuple[str, ...], int],
    programs: list[tuple[str, ...]],
    matrix: dict[int, dict[int, tuple[bool, bool, int]]],
    rule: str,
) -> dict[str, Any]:
    """One pass over ``order`` under the cache rule; exact, no search."""
    cache: list[int] = []  # program indices, in replay order
    solved = 0
    behaviours: set[str] = set()
    hits = verified_hits = 0
    replay_executions = 0
    search_executions = 0
    searches = 0
    per_task: dict[int, bool] = {}
    for position in order:
        _index, task = cohort[position]
        row = matrix[position]
        served: int | None = None
        for p in cache:
            fits, ok, spent = row[p]
            replay_executions += spent
            if fits:
                served = p
                break
        if served is not None:
            hits += 1
            ok = row[served][1]
            verified_hits += ok
            per_task[position] = ok
        else:
            cell = searched[position]
            searches += 1
            search_executions += cell.get("executions", 0)
            per_task[position] = bool(cell["solved"])
            if cell["program"] is not None:
                p = program_index[tuple(cell["program"])]
                if p not in cache:
                    cache.append(p)
                    if rule == "shortest-first":
                        cache.sort(key=lambda q: len(programs[q]))
                    elif rule == "longest-first":
                        cache.sort(key=lambda q: -len(programs[q]))
        if per_task[position]:
            solved += 1
            behaviours.add(task.behaviour)
    return {
        "solved": solved,
        "behaviours": len(behaviours),
        "cache_hits": hits,
        "verified_cache_hits": verified_hits,
        "searches": searches,
        "replay_executions": replay_executions,
        "search_executions": search_executions,
        "per_task": per_task,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True, help="a cache-off cell")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orders", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cell = json.loads(args.cell.read_text())
    # A cell written before this driver existed has no ``cache`` field and ran
    # without one, so it is a cache-off cell and can seed the simulation.
    if cell.get("cache", "off") != "off":
        raise SystemExit(f"{args.cell} is not a cache-off cell")
    tasks = co.corpus()
    cohort = co.cohort(tasks)
    if cell.get("order_seed") is not None:
        raise SystemExit("simulate from a cohort-order cell so positions line up")
    by_index = {r["task_index"]: r for r in cell["rows"]}
    searched = {pos: by_index[index] for pos, (index, _t) in enumerate(cohort)}

    programs = sorted(
        {tuple(r["program"]) for r in cell["rows"] if r["program"] is not None}, key=len
    )
    program_index = {p: i for i, p in enumerate(programs)}
    matrix = fits_matrix(cohort, programs)

    baseline_solved = sum(r["solved"] for r in cell["rows"])
    baseline_behaviours = len({r["behaviour"] for r in cell["rows"] if r["solved"]})
    baseline_per_task = {pos: bool(searched[pos]["solved"]) for pos in range(len(cohort))}

    report: dict[str, Any] = {
        "cell": str(args.cell),
        "cohort_tasks": len(cohort),
        "cohort_behaviours": len({t.behaviour for _i, t in cohort}),
        "distinct_programs_returned": len(programs),
        "baseline": {"solved": baseline_solved, "behaviours": baseline_behaviours},
        "orders": args.orders,
        "arms": {},
    }

    rng = random.Random(args.seed)
    order_list = [list(range(len(cohort)))] + [
        rng.sample(range(len(cohort)), len(cohort)) for _ in range(args.orders)
    ]
    for rule in RULES:
        runs = [
            simulate(order, cohort, searched, program_index, programs, matrix, rule)
            for order in order_list
        ]
        bucket_major, random_runs = runs[0], runs[1:]
        regressions = [
            sum(1 for pos, ok in r["per_task"].items() if baseline_per_task[pos] and not ok)
            for r in random_runs
        ]
        report["arms"][rule] = {
            "bucket_major": {k: v for k, v in bucket_major.items() if k != "per_task"},
            "bucket_major_regressions": sum(
                1 for pos, ok in bucket_major["per_task"].items()
                if baseline_per_task[pos] and not ok
            ),
            "random_order": {
                "solved_mean": mean(r["solved"] for r in random_runs),
                "solved_sd": pstdev([r["solved"] for r in random_runs]),
                "solved_min": min(r["solved"] for r in random_runs),
                "solved_max": max(r["solved"] for r in random_runs),
                "behaviours_mean": mean(r["behaviours"] for r in random_runs),
                "behaviours_min": min(r["behaviours"] for r in random_runs),
                "behaviours_max": max(r["behaviours"] for r in random_runs),
                "cache_hits_mean": mean(r["cache_hits"] for r in random_runs),
                "verified_cache_hits_mean": mean(r["verified_cache_hits"] for r in random_runs),
                "searches_mean": mean(r["searches"] for r in random_runs),
                "replay_executions_mean": mean(r["replay_executions"] for r in random_runs),
                "search_executions_mean": mean(r["search_executions"] for r in random_runs),
                "regressions_mean": mean(regressions),
                "regressions_max": max(regressions),
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    base_exec = sum(r.get("executions", 0) for r in cell["rows"])
    print(
        f"cohort {report['cohort_tasks']} tasks / {report['cohort_behaviours']} behaviours; "
        f"cache-off {baseline_solved} tasks / {baseline_behaviours} behaviours, "
        f"{base_exec:,} executions"
    )
    for rule, arm in report["arms"].items():
        r = arm["random_order"]
        print(
            f"  {rule:16s} random order: {r['solved_mean']:.1f} ± {r['solved_sd']:.1f} tasks "
            f"[{r['solved_min']}-{r['solved_max']}], {r['behaviours_mean']:.1f} behaviours "
            f"[{r['behaviours_min']}-{r['behaviours_max']}], {r['searches_mean']:.0f} searches, "
            f"{r['cache_hits_mean']:.0f} hits ({r['verified_cache_hits_mean']:.0f} verified), "
            f"regressions {r['regressions_mean']:.1f} (max {r['regressions_max']}), "
            f"executions {r['search_executions_mean'] + r['replay_executions_mean']:,.0f}"
        )
        b = arm["bucket_major"]
        print(
            f"  {'':16s} bucket-major (privileged): {b['solved']} tasks, {b['behaviours']} "
            f"behaviours, {b['searches']} searches, regressions {arm['bucket_major_regressions']}"
        )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
