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

``--guard`` reinstates at search level the check that capture bypasses inside an
open quotation. It changes how much is walked, never what is found: see
`wandering_light.quotation_guard` and `reports/mark-28-guard-20260901/`.

``--out`` records, beside every row, what the sweep ran on: seed, corpus path
and name, the manifest's Hub revision and catalogue digests, and the live
catalogue's digests, so a row can be told apart from one on a regenerated
corpus (#142). Each row names its task (index in the corpus, a content id,
split) and lists every program that survived the held-out examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from wandering_light.quotation_guard import MODES, blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Value,
    ops_for_basis,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import (
    BASIS,
    Task,
    catalogue_digest,
    catalogue_source_digest,
    load,
)

Behaviour = tuple[tuple[Value, ...], ...]


def key_of(behaviour: Behaviour) -> tuple:
    return tuple(tuple(value_key(v) for v in stack) for stack in behaviour)


def task_id(task: Task) -> str:
    """Twelve hex digits over behaviour, shown inputs and outputs.

    Stable across corpus order and shuffling, unlike an index; the shown
    examples are what makes two tasks the same task.
    """
    payload = json.dumps([task.behaviour, list(task.inputs), list(task.outputs)])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


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
    inputs: Behaviour, target: tuple, depth: int, basis: str, guard: str = "off"
) -> tuple[list[tuple[str, ...]], int]:
    """Every program of at most `depth` ops whose behaviour is `target`.

    `guard` drops successors sitting inside an open quotation that can never be
    completed into a running one -- see `wandering_light.quotation_guard`. A
    guarded state is never entered in `seen`, so `states` reports the size of the
    space actually walked. Candidates are matched against the target *before*
    the guard is consulted, so no fit can be lost to it.
    """
    ops = ops_for_basis(basis)
    seen = {key_of(inputs)}
    frontier = {key_of(inputs): (inputs, ())}
    matches: list[tuple[str, ...]] = []
    for step in range(depth):
        remaining = depth - step - 1
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
                if blocked(moved[0], remaining, guard) is not None:
                    continue
                seen.add(moved_key)
                nxt[moved_key] = (moved, extended)
        frontier = nxt
    return matches, len(seen)


def stacks(texts: tuple[str, ...]) -> Behaviour:
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def audit_one(
    task: Task, depth: int, basis: str, guard: str = "off", index: int | None = None
) -> dict:
    """Search one task, then re-check every fit on the held-out examples."""
    shown_in, shown_out = stacks(task.inputs), stacks(task.outputs)
    hidden_in, hidden_out = stacks(task.held_out_inputs), stacks(task.held_out_outputs)
    matches, states = matching_programs(
        shown_in, key_of(shown_out), depth, basis, guard
    )

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
        "task_index": index,
        "task_id": task_id(task),
        "split": task.split,
        "behaviour": task.behaviour,
        "examples": task.examples,
        "witness_length": task.length,
        "states": states,
        "fits_shown": len(matches),
        "fits_held_out": len(generalises),
        "fits_held_out_programs": [" ".join(p) for p in generalises],
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
    parser.add_argument("--guard", default="off", choices=MODES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    manifest = (
        json.loads((corpus / "manifest.json").read_text()) if corpus.is_dir() else {}
    )
    # Shuffling (index, task) pairs draws the same permutation as shuffling the
    # tasks did, so a seed still names the sample it always named.
    indexed = list(enumerate(load(args.corpus)))
    random.Random(args.seed).shuffle(indexed)
    sample = indexed[: args.tasks]

    started = time.time()
    print(
        f"depth <= {args.depth}, {len(sample)} tasks, basis {args.basis}, "
        f"guard {args.guard}\n"
    )
    header = f"{'behaviour':<18}{'k':>2}{'wit':>5}{'fits':>6}{'holds':>6}{'bogus':>6}{'states':>10}"
    print(header)
    rows = []
    for index, task in sample:
        row = audit_one(task, args.depth, args.basis, args.guard, index=index)
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
                {
                    "depth": args.depth,
                    "basis": args.basis,
                    "guard": args.guard,
                    "seed": args.seed,
                    "tasks": args.tasks,
                    "corpus": args.corpus,
                    "corpus_name": manifest.get("corpus"),
                    "hub": manifest.get("hub"),
                    "manifest_catalogue": manifest.get("catalogue"),
                    "catalogue_digest": catalogue_digest(),
                    "catalogue_source_digest": catalogue_source_digest(),
                    "rows": rows,
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
