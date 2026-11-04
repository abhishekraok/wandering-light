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

Because every op costs exactly one fuel, shortest is also fastest here, so the
label is unambiguous. That stops being true the moment a looping combinator
enters the basis: one outer `map` can run its quotation many times, so the
fewest-op program and the cheapest program come apart.

`expand_cost` is the search for the cheapest one. It orders paths by Levin's
Kt, `|program| + log2(fuel)`, where fuel is the total work over all of a task's
examples. Kt is monotone along a path but not additive: how much a suffix
costs depends on the length and fuel of the prefix it lands on. So the search
keeps a Pareto frontier of arrivals per behaviour rather than one best label,
and only prunes an arrival that another one beats on length and on every
example's fuel at once.

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
from typing import TYPE_CHECKING

from wandering_light.stack_machine import (
    CORE_OPS,
    DEFAULT_FUEL_LIMIT,
    Bool,
    Config,
    Int,
    Lst,
    Op,
    Value,
    format_stack,
    parse_stack,
    step_raw,
    value_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

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


@dataclass(frozen=True, slots=True)
class CostedPath:
    """A program together with the fuel it spent on each example.

    `score` is `fuel << length`, an exact integer stand-in for `2**Kt` with
    Levin's `Kt = length + log2(fuel)`. Comparing scores compares Kt without
    ever touching a float, which keeps the search order total and exact.
    """

    program: tuple[str, ...]
    example_fuels: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.program)

    @property
    def fuel(self) -> int:
        """Total work over every example."""
        return sum(self.example_fuels)

    @property
    def score(self) -> int:
        """`2**Kt` as an exact integer; 0 for the empty program."""
        return self.fuel << self.length

    @property
    def kt(self) -> float:
        """Levin's Kt, `length + log2(fuel)`; 0.0 for the empty program."""
        score = self.score
        return math.log2(score) if score else 0.0

    def dominates(self, other: CostedPath) -> bool:
        """True when `other` can never beat this path after any suffix.

        A suffix costs the same from either arrival -- the stacks are the same
        and an op's cost does not depend on its budget until the budget runs
        out -- so no greater length and no greater fuel on every example means
        no greater Kt after every suffix as well.
        """
        return self.length <= other.length and all(
            mine <= theirs
            for mine, theirs in zip(
                self.example_fuels, other.example_fuels, strict=True
            )
        )


def step_all_costed(
    behaviour: Behaviour, example_fuels: tuple[int, ...], op: Op
) -> tuple[Behaviour, tuple[int, ...]] | None:
    """Apply `op` to every example, tracking the fuel each one has spent.

    Each example runs under its own remaining budget, `DEFAULT_FUEL_LIMIT`
    less what it has already spent. Returns the moved behaviour and the new
    per-example fuels, or None if any example refuses -- for want of fuel or
    for any other reason.

    Uses `step_raw` rather than guard-then-apply: an op that runs code cannot
    know its legality without running, so asking twice runs the body twice.
    """
    if len(behaviour) != len(example_fuels):
        raise ValueError(
            f"{len(behaviour)} example(s) but {len(example_fuels)} fuel(s)"
        )
    out = []
    fuels = []
    for stack, spent in zip(behaviour, example_fuels, strict=True):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - spent, 0)
        if isinstance(outcome, str):
            return None
        out.append(outcome[0])
        fuels.append(spent + outcome[1])
    return tuple(out), tuple(fuels)


def step_all(behaviour: Behaviour, op: Op) -> Behaviour | None:
    """Apply `op` to every example, or None if any example's guard refuses.

    Every example starts with the full budget, as the BFS does not carry fuel.
    """
    moved = step_all_costed(behaviour, (0,) * len(behaviour), op)
    return None if moved is None else moved[0]


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


def expand_cost(
    inputs: Behaviour,
    max_depth: int,
    max_states: int | None = 2_000_000,
    *,
    max_score: int | None = None,
    ops: Iterable[Op] = OPS,
) -> tuple[dict, bool]:
    """Uniform-cost search over behaviours by Kt. Returns {key: CostedPath}, complete.

    Paths come off the queue in order of `score`, ties broken by `(length,
    program)`, and a path shorter than `max_depth` is extended by every op of
    `ops`. Candidates whose score exceeds `max_score` are dropped. The first
    arrival at a behaviour is therefore its cheapest program under Kt, and
    the root maps to the empty path.

    Kt is monotone but not additive. A longer, low-fuel arrival can be the
    cheaper one now while a shorter, high-fuel one wins after an expensive
    common suffix, because that suffix doubles with every op of length but
    adds to fuel. So every behaviour keeps a Pareto frontier of arrivals, and
    an arrival is pruned only when a kept one has no greater length and no
    greater fuel on every example; then nothing built on it could ever win.

    `complete` is True once the queue is exhausted. The search stops with
    `(found, False)` the moment a new behaviour would take it past
    `max_states` distinct behaviours, so what it has found is still certified.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if max_states is not None and max_states < 1:
        raise ValueError(f"max_states must be None or >= 1, got {max_states}")
    if max_score is not None and max_score < 0:
        raise ValueError(f"max_score must be None or >= 0, got {max_score}")
    ops = tuple(ops)

    found: dict = {}
    # Arrivals already expanded, per behaviour: the Pareto frontier.
    frontier: dict = defaultdict(list)
    root = CostedPath((), (0,) * len(inputs))
    # (score, length, program, tiebreak, path, behaviour). The counter only
    # keeps the heap from ever comparing paths or behaviours.
    counter = 0
    queue = [(root.score, root.length, root.program, counter, root, inputs)]
    complete = True

    while queue:
        _score, _length, _program, _tiebreak, path, behaviour = heapq.heappop(queue)
        key = behaviour_key(behaviour)
        kept = frontier[key]
        if any(other.dominates(path) for other in kept):
            continue
        if key not in found:
            if max_states is not None and len(found) >= max_states:
                complete = False
                break
            found[key] = path
        kept.append(path)
        if path.length >= max_depth:
            continue
        for op in ops:
            moved = step_all_costed(behaviour, path.example_fuels, op)
            if moved is None:
                continue
            next_behaviour, fuels = moved
            candidate = CostedPath((*path.program, op.name), fuels)
            if max_score is not None and candidate.score > max_score:
                continue
            next_key = behaviour_key(next_behaviour)
            if any(other.dominates(candidate) for other in frontier[next_key]):
                continue
            counter += 1
            heapq.heappush(
                queue,
                (
                    candidate.score,
                    candidate.length,
                    candidate.program,
                    counter,
                    candidate,
                    next_behaviour,
                ),
            )
    return found, complete


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


def replay_costed(
    inputs: Behaviour, program: tuple[str, ...]
) -> tuple[Behaviour, CostedPath]:
    """Run a program on every example, tracking fuel. Raises on an illegal step.

    An unknown op name counts as an illegal step too, so the caller sees one
    `ValueError` for any program the machine would refuse.
    """
    by_name = {op.name: op for op in OPS}
    behaviour = inputs
    fuels = (0,) * len(inputs)
    for name in program:
        op = by_name.get(name)
        if op is None:
            raise ValueError(f"unknown op {name!r} in {program}")
        moved = step_all_costed(behaviour, fuels, op)
        if moved is None:
            raise ValueError(f"{name} is illegal partway through {program}")
        behaviour, fuels = moved
    return behaviour, CostedPath(tuple(program), fuels)


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
