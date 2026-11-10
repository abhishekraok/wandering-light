"""Certified-shortest-path corpus generation for the stack machine.

A task here is a *programming-by-example* task: several (input, output) pairs
that one program must satisfy simultaneously. That is the shape the substrate
actually wants, and it falls out of one observation:

    the multi-example guard is the conjunction of the per-example guards.

A program is legal on a task iff every op in it is legal on every example at
every step. So search over k examples at once by carrying a k-tuple of stacks
as a single state. That tuple is the whole state -- ops read nothing else --
so it is Markov, and deduplicating on it is exact.

Certification is then free. Breadth-first search is complete at every depth it
finishes, so the depth at which a behaviour first appears *is* the length of
its shortest program. No separate re-certification pass, no unreachability
argument: #37's forward certification, but with the behaviour tuple as the
state.

Shortest is not cheapest, though. The looping combinators let one outer op run
its quotation many times, so a short program can do far more work than a
longer one with the same behaviour. `expand_cost` labels each behaviour with
its cheapest program under Levin's Kt instead, `|program| + log2(fuel)` with
fuel totalled over the task's examples, by uniform-cost search.

    uv run python -m wandering_light.stack_machine_corpus --out corpus.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

from wandering_light.stack_machine import (
    CORE_OPS,
    DEFAULT_FUEL_LIMIT,
    Bool,
    Config,
    Int,
    Lst,
    Value,
    format_stack,
    parse_stack,
    step_raw,
    value_key,
)

SCHEMA_VERSION = 3

# The basis this corpus is certified against. A "certified shortest" label is
# only meaningful relative to one, since adding ops can only shorten programs.
#
# This is the 26 executable ops: the first-order 18 plus dip, quote, cat, i and
# the four combinators. It deliberately excludes the 26 symbol pushes. Symbols
# are legal in every state, which triples the branching factor and costs two
# whole depths of certification -- measured at 32x growth per depth against
# 5.9x here. Quotations still reach the combinators, by arriving as part of a
# task's input rather than being built op by op.
OPS = CORE_OPS
BASIS = "executable-26"

TRAIN_FRACTION = 80  # out of 100

# A behaviour is the tuple of stacks a program leaves, one per example.
Behaviour = tuple[tuple[Value, ...], ...]


@dataclass(frozen=True)
class Task:
    """One PBE task with a certified-shortest witness program."""

    split: str
    family: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    program: tuple[str, ...]
    length: int
    witnesses: int  # distinct programs of this length with this behaviour
    certified: bool
    examples: int
    basis: str = BASIS
    schema_version: int = SCHEMA_VERSION


def assign_split(inputs: tuple[str, ...], outputs: tuple[str, ...]) -> str:
    """Deterministic train/eval split keyed on the task itself.

    Hashing the task rather than shuffling means the same task always lands in
    the same split, so regenerating the corpus cannot quietly move a task
    across the boundary. SHA-256 rather than `hash`, so it does not depend on
    the interpreter.
    """
    payload = json.dumps([list(inputs), list(outputs)], sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return "train" if int(digest[:8], 16) % 100 < TRAIN_FRACTION else "eval"


def encode_task(task: Task) -> dict[str, str]:
    """The text form a sequence model would be trained on.

    Prompt is the examples; completion is the program. Kept here rather than in
    a training script so the corpus and its encoding cannot drift apart.
    """
    pairs = " ; ".join(
        f"{given} -> {wanted}"
        for given, wanted in zip(task.inputs, task.outputs, strict=True)
    )
    return {"prompt": f"{pairs} =>", "completion": " " + " ".join(task.program)}


def behaviour_key(behaviour: Behaviour) -> tuple:
    """Structural identity for a behaviour, built from `value_key`."""
    return tuple(tuple(value_key(v) for v in stack) for stack in behaviour)


def step_all_costed(
    behaviour: Behaviour, example_fuels: tuple[int, ...], op
) -> tuple[Behaviour, tuple[int, ...]] | None:
    """Apply `op` to every example, charging each its own fuel.

    `example_fuels` is what each example has spent so far; each runs under its
    own remaining budget, `DEFAULT_FUEL_LIMIT - fuel`. Returns the moved
    behaviour and the updated fuels, or None if any example refuses.

    Uses `step_raw` rather than guard-then-apply: an op that runs code cannot
    know its legality without running, so asking twice runs the body twice.
    """
    if len(behaviour) != len(example_fuels):
        raise ValueError(
            f"{len(behaviour)} examples but {len(example_fuels)} fuel counts"
        )
    moved = []
    fuels = []
    for stack, fuel in zip(behaviour, example_fuels, strict=True):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - fuel, 0)
        if isinstance(outcome, str):
            return None
        moved.append(outcome[0])
        fuels.append(fuel + outcome[1])
    return tuple(moved), tuple(fuels)


def step_all(behaviour: Behaviour, op) -> Behaviour | None:
    """Apply `op` to every example, or None if any example's guard refuses.

    Each example gets the full budget: this is one step taken in isolation.
    """
    outcome = step_all_costed(behaviour, (0,) * len(behaviour), op)
    return None if outcome is None else outcome[0]


def expand(
    inputs: Behaviour, max_depth: int, max_states: int = 2_000_000
) -> tuple[dict, bool]:
    """BFS over behaviours. Returns {key: (depth, program, witnesses)}, complete.

    `witnesses` counts distinct op-sequences of that same shortest length which
    realise the behaviour -- how under-determined the task is by its examples.
    """
    start = behaviour_key(inputs)
    found = {start: (0, (), 1)}
    frontier = {start: (inputs, 1)}
    complete = True

    for depth in range(1, max_depth + 1):
        nxt: dict = {}
        counts: dict = defaultdict(int)
        for key, (behaviour, paths) in frontier.items():
            program = found[key][1]
            for op in OPS:
                moved = step_all(behaviour, op)
                if moved is None:
                    continue
                mkey = behaviour_key(moved)
                counts[mkey] += paths
                if mkey in found or mkey in nxt:
                    continue
                nxt[mkey] = (moved, (*program, op.name))
        if not nxt:
            break
        for mkey, (_behaviour, program) in nxt.items():
            found[mkey] = (depth, program, counts[mkey])
        frontier = {k: (v[0], counts[k]) for k, v in nxt.items()}
        if len(found) > max_states:
            complete = False
            break
    return found, complete


# --------------------------------------------------------------------------
# Cost-ordered search under Levin's Kt
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostedPath:
    """A program together with the fuel it spent on each example.

    Kt is `|program| + log2(fuel)`, with fuel summed over the task's examples.
    `score` is 2**Kt computed exactly as an integer, `fuel << length`, so
    comparisons never touch floating point.
    """

    program: tuple[str, ...]
    example_fuels: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.program)

    @property
    def fuel(self) -> int:
        return sum(self.example_fuels)

    @property
    def score(self) -> int:
        if not self.program:
            return 0
        return self.fuel << self.length

    @property
    def kt(self) -> float:
        score = self.score
        # 0.0 for the empty program. A non-empty program can only score 0 on a
        # task with no examples; that is reported as 0.0 too rather than -inf.
        return math.log2(score) if score else 0.0


def _dominates(
    length: int, fuels: tuple[int, ...], program: tuple[str, ...], other
) -> bool:
    """True when label (length, fuels, program) makes `other` redundant.

    No greater length and no greater fuel on any example means every extension
    of `other` is matched by the same extension of this label, which is legal
    (more budget left, more depth left) and no more expensive. Exact ties are
    broken on the program so that the survivor is the tie-break winner.
    """
    other_length, other_fuels, other_program = other[0], other[1], other[2]
    if length > other_length:
        return False
    if any(a > b for a, b in zip(fuels, other_fuels, strict=True)):
        return False
    if length == other_length and fuels == other_fuels:
        return program <= other_program
    return True


def expand_cost(
    inputs: Behaviour,
    max_depth: int,
    max_states: int | None = 2_000_000,
    *,
    max_score: int | None = None,
    ops=OPS,
) -> tuple[dict, bool]:
    """Uniform-cost search over behaviours, cheapest Kt first.

    Returns `({behaviour_key: CostedPath}, complete)`, where each path is the
    cheapest program reaching that behaviour with at most `max_depth` ops,
    ties broken by `(score, length, program)`.

    Kt is monotone along a path but not additive: two arrivals at the same
    behaviour cannot be ranked by their score alone, because a longer arrival
    with less fuel can be cheaper now while the shorter one wins after an
    expensive suffix (the suffix's fuel is multiplied by 2**length). So every
    behaviour keeps a Pareto frontier of labels over (length, fuel per
    example), and only labels dominated on all of them are pruned.

    Labels are popped in strictly increasing `(score, length, program)` order,
    since every extension increases the length and never lowers the score. The
    first label popped for a behaviour is therefore its cheapest path, and a
    run cut short by `max_states` still returns exact labels: the
    `max_states` cheapest behaviours.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if max_states is not None and max_states < 1:
        raise ValueError(f"max_states must be None or >= 1, got {max_states}")
    if max_score is not None and max_score < 0:
        raise ValueError(f"max_score must be None or >= 0, got {max_score}")

    start = behaviour_key(inputs)
    zero = (0,) * len(inputs)
    behaviours: dict = {start: inputs}
    # Per behaviour, the live non-dominated labels: [length, fuels, program, alive]
    frontiers: dict = {start: [[0, zero, (), True]]}
    root_label = frontiers[start][0]
    queue: list = [(0, 0, (), start, root_label)]
    found: dict = {}

    while queue:
        _score, length, program, key, label = heapq.heappop(queue)
        if not label[3]:
            continue  # dominated after it was queued
        fuels = label[1]
        if key not in found:
            if max_states is not None and len(found) >= max_states:
                return found, False
            found[key] = CostedPath(program, fuels)
        if length >= max_depth:
            continue
        behaviour = behaviours[key]
        for op in ops:
            outcome = step_all_costed(behaviour, fuels, op)
            if outcome is None:
                continue
            moved, moved_fuels = outcome
            moved_length = length + 1
            moved_program = (*program, op.name)
            moved_score = sum(moved_fuels) << moved_length
            if max_score is not None and moved_score > max_score:
                continue
            mkey = behaviour_key(moved)
            frontier = frontiers.get(mkey)
            if frontier is None:
                behaviours[mkey] = moved
                frontier = frontiers[mkey] = []
            elif any(
                _dominates(
                    other[0],
                    other[1],
                    other[2],
                    (moved_length, moved_fuels, moved_program),
                )
                for other in frontier
            ):
                continue
            survivors = []
            for other in frontier:
                if _dominates(moved_length, moved_fuels, moved_program, other):
                    other[3] = False
                else:
                    survivors.append(other)
            new_label = [moved_length, moved_fuels, moved_program, True]
            survivors.append(new_label)
            frontiers[mkey] = survivors
            heapq.heappush(
                queue, (moved_score, moved_length, moved_program, mkey, new_label)
            )
    return found, True


def replay_costed(
    inputs: Behaviour, program: tuple[str, ...]
) -> tuple[Behaviour, CostedPath]:
    """Run a program on every example, charging fuel cumulatively per example.

    Raises ValueError if any step is illegal (or names no op in the basis).
    """
    by_name = {op.name: op for op in OPS}
    behaviour = inputs
    fuels = (0,) * len(inputs)
    for name in program:
        op = by_name.get(name)
        if op is None:
            raise ValueError(f"{name!r} is not in the basis")
        outcome = step_all_costed(behaviour, fuels, op)
        if outcome is None:
            raise ValueError(f"{name} is illegal partway through {program}")
        behaviour, fuels = outcome
    return behaviour, CostedPath(tuple(program), fuels)


def is_repetition(program: tuple[str, ...]) -> bool:
    """Reject one op repeated, like `quote quote` or `dup dup`.

    These are valid programs and useless tasks: they say nothing the length-one
    version does not already say, and `quote` in particular manufactures a new
    behaviour from any state, so it would otherwise pad the corpus.
    """
    return len(program) > 1 and len(set(program)) == 1


def is_useful(inputs: Behaviour, outputs: Behaviour) -> bool:
    """Reject tasks that would teach nothing.

    Three filters, and the first is the one that matters. Harvesting every
    reachable behaviour yields mostly rubbish -- programs that pile values on
    the stack without consuming anything, so the "output" is scratch space with
    an answer somewhere inside it. Requiring a single value left on the stack
    forces the program to be a function of its input rather than an
    accumulation on top of it.

    The other two: identity teaches nothing, and a constant output means the
    program ignored its input, so the examples constrain nothing.
    """
    if any(len(stack) != 1 for stack in outputs):
        return False
    if behaviour_key(inputs) == behaviour_key(outputs):
        return False
    first = behaviour_key((outputs[0],))
    return not all(behaviour_key((s,)) == first for s in outputs)


# --------------------------------------------------------------------------
# Task families: sets of inputs a single program should handle together
# --------------------------------------------------------------------------


def bits(*values: int) -> tuple[Value, ...]:
    return (Lst(tuple(Int(v) for v in values)),)


FAMILIES: dict[str, list[Behaviour]] = {
    # one three-element list of bits per example
    "bitlist3": [
        (bits(1, 0, 1), bits(0, 1, 1), bits(1, 1, 0)),
        (bits(0, 0, 1), bits(1, 0, 0), bits(0, 1, 0)),
    ],
    # two integers on the stack
    "int_pair": [
        ((Int(3), Int(5)), (Int(2), Int(7)), (Int(9), Int(4))),
        ((Int(6), Int(2)), (Int(1), Int(8)), (Int(5), Int(5))),
    ],
    # a list under an integer
    "list_int": [
        ((Lst((Int(1), Int(0))), Int(4)), (Lst((Int(0), Int(1))), Int(7))),
        ((Lst((Int(1), Int(1))), Int(2)), (Lst((Int(0), Int(0))), Int(9))),
    ],
    # two lists
    "list_pair": [
        (
            (Lst((Int(1),)), Lst((Int(0), Int(1)))),
            (Lst((Int(0),)), Lst((Int(1), Int(1)))),
        ),
    ],
    # a boolean and an integer, to reach the predicates
    "bool_int": [
        ((Bool(True), Int(3)), (Bool(False), Int(6))),
    ],
    # Language modelling, fixed context width. Every context of k bits is an
    # example, so a task is "predict the next bit from the previous k" and one
    # program must be right on all 2**k of them. With the context length fixed
    # this needs no loop, which is why it fits the basis as it stands.
    "lm_order2": [
        tuple((Int(a), Int(b)) for a in (0, 1) for b in (0, 1)),
    ],
    "lm_order3": [
        tuple((Int(a), Int(b), Int(c)) for a in (0, 1) for b in (0, 1) for c in (0, 1)),
    ],
    # From here the input carries a quotation. This is what makes map, fold,
    # filter, i and dip reachable without paying for 26 symbol pushes: the
    # program does not build the function, the task hands it over.
    "with_map": [
        (
            parse_stack("(1 0 1) (1 add)"),
            parse_stack("(0 1 1) (1 add)"),
            parse_stack("(1 1 0) (1 add)"),
        ),
        (
            parse_stack("(2 5) (dup mul)"),
            parse_stack("(3 4) (dup mul)"),
        ),
    ],
    "with_fold": [
        (
            parse_stack("(1 2 3) 0 (add)"),
            parse_stack("(4 5) 0 (add)"),
            parse_stack("(7) 0 (add)"),
        ),
        (
            parse_stack("(1 2 3) 1 (mul)"),
            parse_stack("(2 2 2) 1 (mul)"),
        ),
    ],
    "with_filter": [
        (
            parse_stack("(1 0 1 0) (0 eq?)"),
            parse_stack("(0 0 1) (0 eq?)"),
            parse_stack("(1 1) (0 eq?)"),
        ),
    ],
    "with_apply": [
        (
            parse_stack("3 5 (add)"),
            parse_stack("2 7 (add)"),
            parse_stack("9 4 (add)"),
        ),
        (
            parse_stack("6 (dup mul)"),
            parse_stack("4 (dup mul)"),
        ),
    ],
}


def generate(max_depth: int, per_depth: int, seed: int) -> tuple[list[Task], dict]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    stats: dict = {}
    for family, input_sets in FAMILIES.items():
        by_depth: dict[int, list[Task]] = defaultdict(list)
        searched = 0
        for inputs in input_sets:
            found, complete = expand(inputs, max_depth)
            searched += len(found)
            for _key, (depth, program, witnesses) in found.items():
                if depth == 0:
                    continue
                outputs = replay(inputs, program)
                if not is_useful(inputs, outputs) or is_repetition(program):
                    continue
                encoded_inputs = tuple(format_stack(Config(s)) for s in inputs)
                encoded_outputs = tuple(format_stack(Config(s)) for s in outputs)
                by_depth[depth].append(
                    Task(
                        split=assign_split(encoded_inputs, encoded_outputs),
                        family=family,
                        inputs=encoded_inputs,
                        outputs=encoded_outputs,
                        program=program,
                        length=depth,
                        witnesses=witnesses,
                        certified=complete,
                        examples=len(inputs),
                    )
                )
        picked = []
        for depth in sorted(by_depth):
            pool = by_depth[depth]
            rng.shuffle(pool)
            picked.extend(pool[:per_depth])
        tasks.extend(picked)
        stats[family] = {
            "behaviours_searched": searched,
            "usable": sum(len(v) for v in by_depth.values()),
            "kept": len(picked),
            "by_depth": {d: len(by_depth[d]) for d in sorted(by_depth)},
        }
    return tasks, stats


def replay(inputs: Behaviour, program: tuple[str, ...]) -> Behaviour:
    """Run a program on every example. Raises if any step is illegal."""
    by_name = {op.name: op for op in OPS}
    behaviour = inputs
    for name in program:
        moved = step_all(behaviour, by_name[name])
        if moved is None:
            raise ValueError(f"{name} is illegal partway through {program}")
        behaviour = moved
    return behaviour


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.stack_machine_corpus"
    )
    parser.add_argument("--out", default="stack_corpus_v1.jsonl.gz")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--per-depth", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    started = time.time()
    tasks, stats = generate(args.max_depth, args.per_depth, args.seed)

    with gzip.open(args.out, "wt") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")

    by_depth: dict[int, int] = defaultdict(int)
    for task in tasks:
        by_depth[task.length] += 1
    print(
        f"wrote {len(tasks)} tasks to {args.out} in {time.time() - started:.1f}s",
        file=sys.stderr,
    )
    print(f"  by length: {dict(sorted(by_depth.items()))}", file=sys.stderr)
    for family, s in stats.items():
        print(
            f"  {family:<10} searched {s['behaviours_searched']:>9,} behaviours, "
            f"{s['usable']:>7,} usable, kept {s['kept']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
