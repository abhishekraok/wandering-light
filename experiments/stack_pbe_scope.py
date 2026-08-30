"""What does the behaviour-first corpus contain, measured against the old one?

`stack_corpus_scope.py` measured the harvested corpus and is what
`stack-corpus-no-higher-order (#59)` rests on: 17 distinct input sets across
1,857 tasks, 36.6% of tasks handed a quotation (76.8% of which never ran it),
0% of programs building one, 11.8% running code at all, 5.7% using a
combinator, and 55.3% using no arithmetic, predicate or combinator whatsoever.

This is the same audit against `wandering_light/stack_pbe.py`'s output, so the
two are directly comparable. It reads a corpus and asserts nothing; the
comparison table is printed for a report to quote.

    uv run python experiments/stack_pbe_scope.py --corpus wandering_light/evals/data/stack_pbe_v3.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from wandering_light.stack_machine import Lst, Sym, Value, parse_stack
from wandering_light.stack_pbe import (
    BASIS,
    CATALOGUE,
    independent_groups,
    load,
    scaffold_sharing,
)

COMBINATORS = {"map", "filter", "fold", "ifte"}
RUNS_CODE = COMBINATORS | {"i", "dip"}
ARITHMETIC = {"add", "sub", "mul", "div"}
PREDICATES = {"eq?", "lt?", "null?"}
BRACKETS = {"[", "]"}

# The published figures for `stack_pbe_v2.jsonl.gz`, from #59. Kept here so the
# comparison cannot drift from what STATUS records.
V2 = {
    "tasks": 1857,
    "distinct_input_stacks": 17,
    "quotation_in_input": 0.366,
    "builds_quotation": 0.0,
    "runs_code": 0.118,
    "uses_combinator": 0.057,
    "no_computation": 0.553,
}


def has_quotation(value: Value) -> bool:
    """A value that is or contains an op name -- i.e. a program, not data."""
    if isinstance(value, Sym):
        return True
    if isinstance(value, Lst):
        return any(has_quotation(item) for item in value.items)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if corpus.is_dir() and not any(corpus.glob("*.jsonl.gz")):
        # The manifest is tracked and the payload is not; say so, rather than
        # failing with a missing-file error that looks like a broken checkout.
        print(
            f"{corpus} holds a manifest but no payload -- the corpus lives on the "
            f"Hub, not in git. Fetch it with:\n\n"
            f'  uv run python -c "from wandering_light.corpus_hub import '
            f"fetch_corpus; fetch_corpus('{corpus}/manifest.json')\"\n\n"
            f"or point --corpus at a directory you generated yourself.",
            file=sys.stderr,
        )
        return 1

    tasks = load(args.corpus)
    n = len(tasks)

    shown = {i for task in tasks for i in task.inputs}
    hidden = {i for task in tasks for i in task.held_out_inputs}
    input_sets = {tuple(sorted(task.inputs)) for task in tasks}
    seeded = sum(
        any(
            has_quotation(v)
            for text in task.inputs
            for v in parse_stack(text, basis=BASIS)
        )
        for task in tasks
    )
    programs = [set(task.program) for task in tasks]
    stats = {
        "corpus": args.corpus,
        "tasks": n,
        "behaviours": len({task.behaviour for task in tasks}),
        "skeletons": len({task.skeleton for task in tasks if task.skeleton}),
        "distinct_input_stacks": len(shown | hidden),
        "distinct_shown_input_stacks": len(shown),
        "distinct_input_sets": len(input_sets),
        "shown_examples": sum(task.examples for task in tasks),
        "held_out_examples": sum(task.held_out for task in tasks),
        "quotation_in_input": seeded / n,
        "builds_quotation": sum(bool(p & BRACKETS) for p in programs) / n,
        "runs_code": sum(bool(p & RUNS_CODE) for p in programs) / n,
        "uses_combinator": sum(bool(p & COMBINATORS) for p in programs) / n,
        "no_computation": sum(
            not (p & (ARITHMETIC | PREDICATES | RUNS_CODE)) for p in programs
        )
        / n,
        "builds_closure": sum("closure" in task.tags for task in tasks) / n,
        "median_length": statistics.median(task.length for task in tasks),
        "max_length": max(task.length for task in tasks),
        "median_fuel": statistics.median(task.fuel for task in tasks),
    }

    print(
        f"{args.corpus}: {n:,} tasks over {stats['behaviours']} behaviours "
        f"and {stats['skeletons']} distinct skeletons\n"
    )
    rows = [
        (
            "distinct input stacks",
            stats["distinct_input_stacks"],
            V2["distinct_input_stacks"],
        ),
        (
            "distinct input sets (one per task)",
            stats["distinct_input_sets"],
            V2["distinct_input_stacks"],
        ),
    ]
    print(f"{'':<38}{'v3':>12}{'v2 (#59)':>12}")
    for label, got, was in rows:
        print(f"{label:<38}{got:>12,}{was:>12,}")
    for key in (
        "quotation_in_input",
        "builds_quotation",
        "runs_code",
        "uses_combinator",
        "no_computation",
    ):
        label = key.replace("_", " ")
        print(f"{label:<38}{stats[key]:>11.1%}{V2[key]:>12.1%}")
    print(
        f"{'builds a closure over task data':<38}{stats['builds_closure']:>11.1%}{'--':>12}"
    )

    print(
        f"\nwitness length: median {stats['median_length']}, max {stats['max_length']}; "
        f"median fuel {stats['median_fuel']}"
    )
    print(
        f"examples: {stats['shown_examples']:,} shown + "
        f"{stats['held_out_examples']:,} held out"
    )
    lengths = Counter(task.length for task in tasks)
    print(f"length histogram: {dict(sorted(lengths.items()))}")

    splits = Counter(task.split for task in tasks)
    by_split: dict[str, set[str]] = {}
    shapes: dict[str, set[str]] = {}
    for task in tasks:
        by_split.setdefault(task.split, set()).add(task.behaviour)
        shapes.setdefault(task.split, set()).add(task.skeleton)
    print("\nsplit is on the skeleton, so no program shape spans both halves:")
    for name in sorted(by_split):
        print(
            f"  {name:<6}{splits[name]:>6,} tasks  {len(shapes[name]):>3} skeletons  "
            f"{len(by_split[name]):>3} behaviours  {', '.join(sorted(by_split[name]))}"
        )
    print(
        f"  behaviour overlap: "
        f"{sorted(by_split.get('train', set()) & by_split.get('eval', set()))}"
    )
    print(
        f"  skeleton overlap:  "
        f"{sorted(shapes.get('train', set()) & shapes.get('eval', set()))}"
    )

    witnesses = Counter(1 + len(task.alternative_programs) for task in tasks)
    print(f"\nwitnesses per task: {dict(sorted(witnesses.items()))}")

    # Skeleton count alone is gameable: a cross product of predicates against
    # maps yields one distinct skeleton per pair while carrying only as many
    # ideas as it has factors, and the compositions chain into a group that
    # cannot be split at all. Print the group profile next to the count.
    groups = independent_groups(CATALOGUE)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    largest = sizes[0] if sizes else 0
    print(
        f"\ninseparable groups: {len(groups)} across {len(CATALOGUE)} behaviours; "
        f"largest holds {largest} ({largest / max(len(CATALOGUE), 1):.1%})"
    )
    print(f"  group sizes: {sizes[:8]}{' ...' if len(sizes) > 8 else ''}")
    shares = scaffold_sharing(CATALOGUE)
    if shares:
        middle = statistics.median(share for share, _ in shares)
        print(
            f"\nshared scaffolding: a witness's median share of 8-op windows that "
            f"another witness also contains is {middle:.1%}"
        )
        print(
            "  This is the basis, not the catalogue -- it bounds what a held-out "
            "score means rather than invalidating one."
        )

    if largest > len(CATALOGUE) * 0.2:
        print(
            "  WARNING: the largest group exceeds a fifth of the catalogue, so "
            "no split can hold it out without moving that whole share."
        )

    missing = {b.name for b in CATALOGUE} - {task.behaviour for task in tasks}
    if missing:
        print(f"\nbehaviours with no task: {sorted(missing)}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(stats, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
