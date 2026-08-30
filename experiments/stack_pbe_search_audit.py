"""Is a task pinned down by its examples, and is its witness the shortest one?

Two questions the corpus cannot answer about itself.

1. **Under-determination.** A PBE task is only a task if the examples force the
   behaviour. The failure mode to rule out is a short program that reproduces
   the shown outputs without computing anything -- pushing a constant, or
   projecting some part of the input. Enumerating *every* program up to depth
   `d` and re-checking each fit against the task's held-out examples measures
   that directly: a program that fits what was shown and fails what was hidden
   is a spurious fit, and it is exactly what the held-out set exists to catch.

2. **Witness length.** Witnesses here are authored, not certified. An
   exhaustive sweep to depth `d` that finds nothing shorter certifies the
   witness shortest when `d >= len(witness)`, and otherwise establishes the
   weaker "no program of `d` ops or fewer solves this".

Search is a *measurement* of the corpus, not the thing that produced it -- the
inversion the whole generator rests on. Cost grows with the basis: roughly 20x
per depth, ~40 s per task at depth 5 on one core.

    uv run python experiments/stack_pbe_search_audit.py --depth 4 --tasks 40
"""

from __future__ import annotations

import argparse
import json
import random
import time

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Value,
    ops_for_basis,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, Task, load

Behaviour = tuple[tuple[Value, ...], ...]


def key_of(behaviour: Behaviour) -> tuple:
    return tuple(tuple(value_key(v) for v in stack) for stack in behaviour)


def step_all(behaviour: Behaviour, op) -> Behaviour | None:
    """One op across every example at once, or None if any example refuses."""
    moved = []
    for stack in behaviour:
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None
        moved.append(outcome[0])
    return tuple(moved)


def matching_programs(
    inputs: Behaviour, target: tuple, depth: int, basis: str
) -> tuple[list[tuple[str, ...]], int]:
    """Every program of at most `depth` ops whose behaviour is `target`."""
    ops = ops_for_basis(basis)
    seen = {key_of(inputs)}
    frontier = {key_of(inputs): (inputs, ())}
    matches: list[tuple[str, ...]] = []
    for _ in range(depth):
        nxt: dict = {}
        for _key, (behaviour, program) in frontier.items():
            for op in ops:
                moved = step_all(behaviour, op)
                if moved is None:
                    continue
                extended = (*program, op.name)
                moved_key = key_of(moved)
                if moved_key == target:
                    matches.append(extended)
                if moved_key in seen:
                    continue
                seen.add(moved_key)
                nxt[moved_key] = (moved, extended)
        frontier = nxt
    return matches, len(seen)


def stacks(texts: tuple[str, ...]) -> Behaviour:
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def audit_one(task: Task, depth: int, basis: str) -> dict:
    """Search one task, then re-check every fit on the held-out examples."""
    shown_in, shown_out = stacks(task.inputs), stacks(task.outputs)
    hidden_in, hidden_out = stacks(task.held_out_inputs), stacks(task.held_out_outputs)
    matches, states = matching_programs(shown_in, key_of(shown_out), depth, basis)

    generalises, spurious = [], []
    hidden_target = key_of(hidden_out)
    for program in matches:
        behaviour: Behaviour | None = hidden_in
        for name in program:
            op = next(o for o in ops_for_basis(basis) if o.name == name)
            behaviour = step_all(behaviour, op)
            if behaviour is None:
                break
        if behaviour is not None and key_of(behaviour) == hidden_target:
            generalises.append(program)
        else:
            spurious.append(program)
    return {
        "behaviour": task.behaviour,
        "examples": task.examples,
        "witness_length": task.length,
        "states": states,
        "fits_shown": len(matches),
        "fits_held_out": len(generalises),
        "spurious": len(spurious),
        "shortest_fit": min((len(p) for p in matches), default=None),
        "example_spurious": " ".join(spurious[0]) if spurious else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v4")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    tasks = load(args.corpus)
    random.Random(args.seed).shuffle(tasks)
    sample = tasks[: args.tasks]

    started = time.time()
    print(f"depth <= {args.depth}, {len(sample)} tasks, basis {args.basis}\n")
    header = f"{'behaviour':<18}{'k':>2}{'wit':>5}{'fits':>6}{'holds':>6}{'bogus':>6}{'states':>10}"
    print(header)
    rows = []
    for task in sample:
        row = audit_one(task, args.depth, args.basis)
        rows.append(row)
        print(
            f"{row['behaviour']:<18}{row['examples']:>2}{row['witness_length']:>5}"
            f"{row['fits_shown']:>6}{row['fits_held_out']:>6}{row['spurious']:>6}"
            f"{row['states']:>10,}"
        )

    solved = [r for r in rows if r["fits_shown"]]
    bogus = [r for r in rows if r["spurious"]]
    beat = [
        r
        for r in rows
        if r["shortest_fit"] is not None and r["shortest_fit"] < r["witness_length"]
    ]
    certified = [r for r in rows if args.depth >= r["witness_length"]]
    print(
        f"\n{len(solved)}/{len(rows)} tasks admit a program of <= {args.depth} ops "
        f"fitting the shown examples"
    )
    print(
        f"{len(bogus)}/{len(rows)} admit one that fits the shown examples and "
        f"fails the held-out ones"
    )
    print(
        f"{len(beat)}/{len(rows)} have a fit shorter than the authored witness; "
        f"{len(certified)} witnesses are within the swept depth and so certified shortest"
    )
    print(f"{time.time() - started:.0f}s")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {"depth": args.depth, "basis": args.basis, "rows": rows},
                handle,
                indent=2,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
