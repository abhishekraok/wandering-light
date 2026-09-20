"""Cost to clear with a solved-program cache carried across the task set.

`docs/metrics.md` prices reuse: cost to clear allows "a persistent library
across tasks, and library construction charged in the same clock", and the order
over `V` is the system's. No `mark-28` cell has ever used that. `time_cell` in
`compute_optimal_mark28.py` runs `beam_solve` from scratch on every one of the
300 cohort tasks, and the cohort is **43 distinct behaviours** -- `negate_each`
alone is 21 of the 300 tasks -- so the search rediscovers the same program up to
twenty times and pays full price for each.

This driver keeps every program a search returns and tries the cache, shortest
first, before searching. A cached program is accepted on the new task's **shown**
examples only, exactly the evidence the solver is given; held-out grading still
decides whether the task counts as solved, and a cached shortcut that fits the
shown examples and fails held-out costs the task, as it should. Nothing here
reads a held-out example, a behaviour name or a reference implementation.

Both the replay and the search are charged to the same clock, so the reported
pair is a cost to clear in #95's sense rather than a sum over independent tasks.

`--cache off` reproduces the #101 protocol in the same harness, so the A/B is one
script and one host.

    uv run python experiments/solve_with_cache.py --policy <checkpoint> \
        --out checkpoints/cache/cell.json --cache on
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from wandering_light.stack_beam import beam_solve, step_all_quiet  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS, OP_BY_NAME  # noqa: E402
from wandering_light.stack_pbe_eval import grade  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task


def replay_program(
    program: Sequence[str], inputs: tuple[tuple, ...], outputs: tuple[tuple, ...]
) -> tuple[bool, int, int]:
    """Does ``program`` take every shown input to its shown output, and at what cost?

    Charged like the beam: one execution per op run on one example stack, and an
    op the machine refuses ends the attempt. Returns ``(fits, executions, fuel)``.
    """
    executions = 0
    fuel = 0
    product = inputs
    for name in program:
        moved, spent, stepped = step_all_quiet(product, OP_BY_NAME[name])
        executions += stepped
        fuel += spent
        if moved is None:
            return False, executions, fuel
        product = moved
    return product == outputs, executions, fuel


def run_cell(
    tasks: list[Task],
    scorer: PolicyScorer,
    *,
    use_cache: bool,
    disjoint: bool,
    beam_width: int,
    max_depth: int,
    order_seed: int | None,
    limit: int = 0,
) -> dict[str, Any]:
    cohort = co.cohort(tasks)
    if limit:  # smoke tests only; a limited cell is not a cost to clear
        cohort = [cohort[b * 60 + i] for b in range(5) for i in range(limit)]
    if order_seed is not None:
        cohort = list(cohort)
        random.Random(order_seed).shuffle(cohort)
    cache: list[tuple[str, ...]] = []
    provenance: dict[tuple[str, ...], str] = {}  # program -> behaviour it was found on
    rows = []
    # The corpus, the model and the cohort are alive for the whole cell and
    # never become garbage; freezing moves them out of the generations the
    # collector scans, so what `beam_solve` turns off is all it has to.
    gc.freeze()
    started = time.perf_counter()
    for index, task in cohort:
        task_started = time.perf_counter()
        program: tuple[str, ...] | None = None
        source = "search"
        executions = fuel = expansions = depth = 0
        termination = "cache_miss"
        replay_executions = replay_fuel = 0
        if use_cache:
            # Parsed here and not above so the cache-off arm is not charged for a
            # parse it never uses; ``beam_solve`` parses its own inputs either way.
            inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
            outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
            for candidate in cache:
                if disjoint and provenance.get(candidate) == task.behaviour:
                    continue
                fits, spent, spent_fuel = replay_program(candidate, inputs, outputs)
                replay_executions += spent
                replay_fuel += spent_fuel
                if fits:
                    program, source, termination = candidate, "cache", "fit"
                    break
        if program is None:
            result = beam_solve(
                task, scorer, beam_width=beam_width, max_depth=max_depth, guard=co.GUARD
            )
            program = result.program
            termination = result.termination
            executions, fuel = result.executions, result.fuel
            expansions, depth = result.expansions, result.depth
            if use_cache and program is not None and program not in cache:
                cache.append(program)
                provenance[program] = task.behaviour
                cache.sort(key=len)
        seconds = time.perf_counter() - task_started
        rows.append(
            {
                "task": task,
                "task_index": index,
                "behaviour": task.behaviour,
                "length": task.length,
                "bucket": co.bucket_of(task.length),
                "examples": task.examples,
                "program": list(program) if program else None,
                "_program": program,
                "source": source,
                "termination": termination,
                "seconds": seconds,
                "executions": executions + replay_executions,
                "replay_executions": replay_executions,
                "fuel": fuel + replay_fuel,
                "search_fuel": fuel,
                "expansions": expansions,
                "depth": depth,
                "cache_size": len(cache),
            }
        )
    cell_seconds = time.perf_counter() - started
    # Grading is the harness's job and is sealed off from the clock: `grade`
    # runs the program through the executor, which populates `run_body`'s
    # process-global LRU, so grading inside the loop would warm state that later
    # searches read. Held-out results never reach the cache in either case.
    for row in rows:
        found, task = row.pop("_program"), row.pop("task")
        score = grade(found, task) if found else None
        row["solved"] = bool(score and score.solved)
        row["shown_fit"] = bool(score and score.shown)
    return {
        "cell_seconds": cell_seconds,
        "timed_seconds": sum(r["seconds"] for r in rows),
        "cache": ("disjoint" if disjoint else "on") if use_cache else "off",
        "order_seed": order_seed,
        "beam_width": beam_width,
        "max_depth": max_depth,
        "guard": co.GUARD,
        "cache_programs": [list(p) for p in cache],
        "rows": rows,
    }


def summarise(cell: dict[str, Any]) -> str:
    rows = cell["rows"]
    solved = sum(r["solved"] for r in rows)
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r["solved"])
    buckets = " ".join(
        f"{b}:{sum(by_bucket[b])}/{len(by_bucket[b])}"
        for b in ("1-5", "6-10", "11-20", "21-40", "41+")
        if b in by_bucket
    )
    from_cache = sum(1 for r in rows if r["source"] == "cache")
    # The cohort is 43 behaviours behind 300 tasks, so a cache that converts
    # per-instance search variance into within-behaviour coverage moves tasks
    # without moving behaviours. Report both or the number reads as new solves.
    behaviours = len({r["behaviour"] for r in rows if r["solved"]})
    total_behaviours = len({r["behaviour"] for r in rows})
    return (
        f"cache {cell['cache']} beam {cell['beam_width']} depth {cell['max_depth']}: "
        f"{solved}/{len(rows)} tasks, {behaviours}/{total_behaviours} behaviours, "
        f"{cell['cell_seconds']:.0f}s, {sum(r['executions'] for r in rows):,} executions, "
        f"{from_cache} served from a cache of {len(cell['cache_programs'])}  [{buckets}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache",
        choices=("on", "off", "disjoint"),
        default="on",
        help=(
            "disjoint is a DIAGNOSTIC control, not a solver: it replays only programs "
            "found on a different behaviour, which needs behaviour labels the solver "
            "does not have. It separates cross-behaviour reuse from the within-"
            "behaviour recurrence the cohort's 43-behaviour construction supplies."
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--beam", type=int, default=co.BEAM_WIDTH)
    parser.add_argument("--depth", type=int, default=co.MAX_DEPTH)
    parser.add_argument(
        "--order-seed",
        type=int,
        default=None,
        help="shuffle the task order; omit for cohort order",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="tasks per bucket; smoke tests only, not a cost to clear",
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"skip {args.out} (exists)")
        return 0
    if args.device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    tasks = co.corpus()
    scorer = PolicyScorer(load_policy(args.policy, device=args.device), args.device)
    cell = run_cell(
        tasks,
        scorer,
        use_cache=args.cache in ("on", "disjoint"),
        disjoint=args.cache == "disjoint",
        beam_width=args.beam,
        max_depth=args.depth,
        order_seed=args.order_seed,
        limit=args.limit,
    )
    cell |= {
        "limit": args.limit,
        "policy": str(args.policy),
        "device": args.device,
        "cpu_threads": co.CPU_THREADS,
        "eval_corpus": co.primary_corpus(),
        "host": co.host(),
        "provenance": co.provenance(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    print(f"RESULT {args.out.name} {summarise(cell)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
