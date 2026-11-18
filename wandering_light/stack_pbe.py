"""Behaviour-first PBE task generation for the stack machine.

The corpus this replaces was *harvested*: enumerate every program up to depth
``d`` from a handful of seed stacks, keep the behaviourally-distinct ones, call
each survivor a task. Three defects followed from that one decision and are
recorded in ``docs/STATUS.md``: seventeen distinct input sets across 1,857
tasks, quotations handed to the solver inside the task input, and a
certified-shortest label that made building a quotation strictly dominated.
The deeper problem is that "this task is useless" was not falsifiable, because
usefulness was *defined* by reachability.

This module inverts that. A person writes the behaviour; the machine is asked
to meet it.

* ``Behaviour`` pairs a plain-Python ``reference`` -- ``sum``, ``max``, keep
  the evens -- with a ``witness`` program over the stack machine. The two are
  written independently and every task checks that they agree, so a bug has to
  occur twice, in two languages, to survive.
* ``sample`` draws the input *data* fresh for every example of every task.
  Inputs are numbers and lists of numbers. **Nothing in a task input is ever a
  quotation**, which is what made the old corpus's higher-order share
  meaningless.
* A behaviour with no witness is informative: it marks the edge of what the
  basis expresses, rather than being silently absent because no enumeration
  happened to reach it.

The basis is ``mark-28``, and that is forced, not chosen. Once quotations stop
arriving in the input, a program can only obtain one by building it, and no op
in ``executable-26`` puts an op *name* on the stack. Every quotation reachable
there is therefore pure data, which ``map`` can only apply as the identity and
which ``filter`` and ``fold`` reject outright. See
``experiments/stack_pbe_basis_floor.py`` for the measurement.

    uv run python -m wandering_light.stack_pbe --out corpus.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import math
import random
import sys
import textwrap
import time
import warnings
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from wandering_light import corpus_hub
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    MARK_OPS,
    Config,
    Int,
    Lst,
    MachineError,
    Op,
    Value,
    format_stack,
    step_raw,
    value_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SCHEMA_VERSION = 2

# Forced by the design, not preferred: see the module docstring.
BASIS = "mark-28"
OPS: tuple[Op, ...] = MARK_OPS
OP_BY_NAME: dict[str, Op] = {op.name: op for op in OPS}

TRAIN_FRACTION = 80  # out of 100

# A task's arguments in plain Python: ints and lists of ints, bottom of the
# stack first. The stack machine form is derived; this is what a person reads.
Arg = int | list[int]
Args = tuple[Arg, ...]
Result = int | list[int]


# ---------------------------------------------------------------------------
# Plain Python <-> machine values
# ---------------------------------------------------------------------------


def to_value(arg: Arg) -> Value:
    if isinstance(arg, bool):  # bool subclasses int; the machine keeps them apart
        raise MachineError("task data is ints and lists of ints, not bools")
    if isinstance(arg, int):
        return Int(arg)
    if isinstance(arg, list):
        return Lst(tuple(to_value(item) for item in arg))
    raise MachineError(f"cannot encode {arg!r} as a machine value")


def to_stack(args: Args) -> tuple[Value, ...]:
    return tuple(to_value(arg) for arg in args)


def run_program(
    program: Sequence[str], stack: tuple[Value, ...]
) -> tuple[tuple[Value, ...], int] | str:
    """Run a named program over a stack, returning ``(stack, fuel)`` or a reason.

    Uses ``step_raw`` rather than ``run`` so a refusal comes back as a string
    the way the machine's own guards report one, instead of an exception.
    """
    fuel = 0
    for name in program:
        op = OP_BY_NAME.get(name)
        if op is None:
            return f"unknown op {name!r}"
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - fuel, 0)
        if isinstance(outcome, str):
            return f"{name}: {outcome}"
        stack, spent = outcome
        fuel += spent
    return stack, fuel


# ---------------------------------------------------------------------------
# Behaviours
# ---------------------------------------------------------------------------


# Ops that carry no structure: the literals a program can push, and the
# arithmetic and predicates that are interchangeable with one another. Blanking
# them turns a witness into its *shape*, which is what `skeleton` is for.
INTERCHANGEABLE = frozenset(
    {"0", "1", "add", "sub", "mul", "div", "eq?", "lt?", "null?"}
)


def skeleton(program: Sequence[str]) -> str:
    """A witness with its literals and interchangeable ops blanked out.

    `keep_zeros` is `[ 0 eq? ] filter` and `keep_negative` is `[ 0 lt? ]
    filter`. Two different functions, one shape. Counting behaviours therefore
    overstates how much a corpus contains, and splitting on behaviours leaks:
    a solver that learns the shape in train gets its siblings in eval for free.

    This is the unit the catalogue is sized and split by. Behaviour count is
    reported alongside it, never instead of it.
    """
    return " ".join("_" if op in INTERCHANGEABLE else op for op in program)


def independent_groups(
    catalogue: Sequence[Behaviour] = (),
) -> dict[str, list[str]]:
    """Behaviours grouped so that no group can be split without leaking.

    Two behaviours belong together when they share a component idea -- when one
    is `a_then_b` and the other is `a_then_c`, learning the first hands over
    half of the second. A split may move a whole group but never divide one.

    This exists because the skeleton count alone is gameable. Cross-producting
    twelve predicates against twelve maps yields 144 genuinely distinct
    skeletons from 24 ideas, every one of which passes `skeleton`, and the
    result cannot be split at all: the compositions chain into a single group
    holding half the catalogue. Report the group profile beside the skeleton
    count, or the count will be believed.
    """
    catalogue = catalogue or CATALOGUE
    parent = {b.name: b.name for b in catalogue}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    shared: dict[str, list[str]] = {}
    for behaviour in catalogue:
        components = (
            behaviour.name.split("_then_")
            if "_then_" in behaviour.name
            else [behaviour.name]
        )
        for component in components:
            shared.setdefault(component, []).append(behaviour.name)
    for names in shared.values():
        for name in names[1:]:
            root_a, root_b = find(names[0]), find(name)
            if root_a != root_b:
                parent[root_a] = root_b

    groups: dict[str, list[str]] = {}
    for behaviour in catalogue:
        groups.setdefault(find(behaviour.name), []).append(behaviour.name)
    return groups


def scaffold_sharing(
    catalogue: Sequence[Behaviour] = (), window: int = 8
) -> list[tuple[float, str]]:
    """How much of each witness is op-sequence another witness also contains.

    The group audit asks whether two behaviours share an *idea*. This asks the
    weaker and more awkward question of whether they share *code*, and the
    answer on this substrate is: nearly all of it. Twenty-eight ops, no local
    variables, and every non-trivial program a fold around a hand-rolled
    accumulator means the same twenty-op preamble appears everywhere.

    That is a property of the basis rather than of any catalogue -- the forty
    behaviours written before this measurement existed score 66.7%, the later
    ones 79.3%. It is recorded because it bounds what a held-out score means:
    a solver may assemble much of an unseen witness from fragments of seen
    ones, so success is evidence of composition, not of invention.
    """
    catalogue = catalogue or CATALOGUE
    seen: dict[tuple[str, ...], set[str]] = {}
    for behaviour in catalogue:
        program = behaviour.program
        for index in range(len(program) - window + 1):
            seen.setdefault(program[index : index + window], set()).add(behaviour.name)

    shares = []
    for behaviour in catalogue:
        program = behaviour.program
        windows = [
            program[index : index + window]
            for index in range(len(program) - window + 1)
        ]
        if not windows:
            continue
        elsewhere = sum(1 for gram in windows if len(seen[gram]) > 1)
        shares.append((elsewhere / len(windows), behaviour.name))
    return sorted(shares, reverse=True)


@dataclass(frozen=True)
class Behaviour:
    """One hand-written problem: what it computes, and programs that do.

    ``reference`` and ``witnesses`` are independent statements of the same
    function. ``verify`` checks them against each other on fresh data, which is
    the only reason to trust any of them.

    ``witnesses`` is a tuple because a behaviour usually has more than one
    correct program -- `sum` is both `0 [ add ] fold` and `uncons [ add ]
    fold`. Recording only one makes a corpus look like it has a single right
    answer per task, which is false, and invites scoring a solver on string
    match. Every witness is verified; the first is the canonical label.
    """

    name: str
    summary: str
    witnesses: tuple[str, ...]
    sample: Callable[[random.Random], Args]
    reference: Callable[..., Result]
    tags: frozenset[str] = field(default_factory=frozenset)
    # Optional per-task guard on the drawn example set, for behaviours whose
    # examples can be jointly uninformative in a way the generic guards miss.
    accept: Callable[[list[Args], list[Result]], bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.witnesses, str):  # a bare string is a silent bug
            raise MachineError(f"{self.name}: witnesses must be a tuple of programs")
        if not self.witnesses:
            raise MachineError(f"{self.name}: needs at least one witness")

    @property
    def witness(self) -> str:
        """The canonical program. Shortest first is a convention, not a claim."""
        return self.witnesses[0]

    @property
    def program(self) -> tuple[str, ...]:
        return tuple(self.witness.split())

    @property
    def programs(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(w.split()) for w in self.witnesses)

    @property
    def length(self) -> int:
        return len(self.program)

    @property
    def skeleton(self) -> str:
        return skeleton(self.program)


def _draw(rng: random.Random, pool: Sequence[int], lo: int, hi: int) -> list[int]:
    return [rng.choice(pool) for _ in range(rng.randint(lo, hi))]


SMALL = tuple(range(-9, 10))
POSITIVE = tuple(range(1, 10))
SPARSE = (0, 0, 0, 0, *range(-4, 5))  # zero-rich, for the zero predicates
MIXED = (*range(-6, 0), *range(1, 7), 0)


def _list_of(pool: Sequence[int], lo: int = 2, hi: int = 6):
    return lambda rng: (_draw(rng, pool, lo, hi),)


def _list_and_int(pool: Sequence[int], other: Sequence[int], lo: int = 2, hi: int = 6):
    return lambda rng: (_draw(rng, pool, lo, hi), rng.choice(other))


def _two_lists(pool: Sequence[int], lo: int = 1, hi: int = 4):
    return lambda rng: (_draw(rng, pool, lo, hi), _draw(rng, pool, lo, hi))


def _flat_pairs(pool: Sequence[int] = SMALL, lo: int = 1, hi: int = 5):
    def sample(rng: random.Random) -> Args:
        pairs = rng.randint(lo, hi)
        return (_draw(rng, pool, 2 * pairs, 2 * pairs),)

    return sample


def _pairs(xs: list[int]) -> list[tuple[int, int]]:
    return list(zip(xs[::2], xs[1::2], strict=True))


def _two_unequal_lists(rng: random.Random) -> Args:
    left_size = rng.randint(1, 7)
    right_size = rng.randint(1, 7)
    if left_size == right_size:
        right_size = right_size % 7 + 1
    return (
        _draw(rng, SMALL, left_size, left_size),
        _draw(rng, SMALL, right_size, right_size),
    )


def _two_lists_with_shared_prefix(rng: random.Random) -> Args:
    prefix = _draw(rng, SMALL, 0, 4)
    left_tail = _draw(rng, SMALL, 0, 3)
    right_tail = _draw(rng, SMALL, 0, 3)
    if not left_tail and not right_tail:
        left_tail = [rng.choice(SMALL)]
    if left_tail and right_tail and left_tail[0] == right_tail[0]:
        choices = tuple(value for value in SMALL if value != left_tail[0])
        right_tail[0] = rng.choice(choices)
    return [*prefix, *left_tail], [*prefix, *right_tail]


def _list_and_count(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 2, 8)
    return xs, rng.randint(0, len(xs) + 2)


def _list_and_valid_index(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 1, 7)
    return xs, rng.randrange(len(xs))


def _list_and_rotation(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 2, 8)
    return xs, rng.randint(0, len(xs))


def _three_lists(rng: random.Random) -> Args:
    return (
        _draw(rng, SMALL, 1, 6),
        _draw(rng, SMALL, 1, 6),
        _draw(rng, SMALL, 1, 6),
    )


def _partly_matching_lists(rng: random.Random) -> Args:
    size = rng.randint(2, 6)
    left = _draw(rng, SMALL, size, size)
    right = _draw(rng, SMALL, size, size)
    right[0] = left[0]
    replacement = next(value for value in SMALL if value != left[1])
    right[1] = replacement
    return left, right


def _sorted_lists(rng: random.Random) -> Args:
    return (
        sorted(_draw(rng, SMALL, 1, 5)),
        sorted(_draw(rng, SMALL, 1, 5)),
    )


def _observable_stable_pairs(rng: random.Random) -> Args:
    size = rng.randint(3, 5)
    low = rng.randint(-4, 3)
    high = rng.randint(low + 1, 4)
    first_payload = rng.randint(-9, 8)
    second_payload = rng.randint(first_payload + 1, 9)
    pairs = [
        (high, first_payload),
        (low, rng.choice(SMALL)),
        (high, second_payload),
        *[(rng.randint(low, high), rng.choice(SMALL)) for _ in range(size - 3)],
    ]
    return ([value for pair in pairs for value in pair],)


def _stable_sort_pairs(xs: list[int]) -> list[int]:
    pairs = [xs[index : index + 2] for index in range(0, len(xs), 2)]
    return [value for pair in sorted(pairs, key=lambda pair: pair[0]) for value in pair]


def _evaluate_polynomial(coefficients: list[int], x: int) -> int:
    total = 0
    for coefficient in coefficients:
        total = total * x + coefficient
    return total


def _maybe_sorted(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 2, 7)
    if rng.choice((False, True)):
        xs.sort()
    elif xs == sorted(xs):
        xs.reverse()
    return (xs,)


def _interval_with_member(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 2, 7)
    lo = rng.randint(-9, 7)
    hi = rng.randint(lo + 2, 9)
    xs[0] = rng.randint(lo + 1, hi - 1)
    return xs, lo, hi


def _target_radius(rng: random.Random) -> Args:
    return _draw(rng, SMALL, 2, 7), rng.choice(SMALL), rng.randint(1, 6)


def _triangle(rng: random.Random) -> Args:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    if rng.choice((False, True)):
        c = rng.randint(abs(a - b) + 1, a + b - 1)
    else:
        c = a + b + rng.randint(0, 3)
    return a, b, c


def _pair(pool: Sequence[int]):
    return lambda rng: (rng.choice(pool), rng.choice(pool))


def _scalar_of(pool: Sequence[int]):
    return lambda rng: (rng.choice(pool),)


def _triple_of(pool: Sequence[int]):
    return lambda rng: (rng.choice(pool), rng.choice(pool), rng.choice(pool))


def _three_positive_scalars(rng: random.Random) -> Args:
    return rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 12)


def _two_intervals(rng: random.Random) -> Args:
    first = sorted((rng.choice(SMALL), rng.choice(SMALL)))
    second = sorted((rng.choice(SMALL), rng.choice(SMALL)))
    return first[0], first[1], second[0], second[1]


def _pair_nonzero_divisor(rng: random.Random) -> Args:
    return rng.choice(SMALL), rng.choice(tuple(value for value in MIXED if value != 0))


def _value_interval(rng: random.Random) -> Args:
    value, left, right = (
        rng.choice(SMALL),
        rng.choice(SMALL),
        rng.choice(SMALL),
    )
    return value, min(left, right), max(left, right)


def _list_old_new(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 2, 6)
    old = rng.choice(xs)
    replacements = tuple(value for value in SMALL if value != old)
    return xs, old, rng.choice(replacements)


def _list_interval(rng: random.Random) -> Args:
    xs = _draw(rng, SMALL, 2, 6)
    lo = rng.randint(-9, 8)
    hi = rng.randint(lo + 1, 9)
    return xs, lo, hi


def _list_bounds_with_two_hits(rng: random.Random) -> Args:
    lo = rng.randint(-9, 7)
    hi = rng.randint(lo + 1, 9)
    xs = _draw(rng, SMALL, 3, 8)
    xs[0] = rng.randint(lo, hi)
    xs[1] = rng.randint(lo, hi)
    return xs, lo, hi


def _list_existing_value(rng: random.Random) -> Args:
    xs = _draw(rng, SPARSE, 2, 7)
    return xs, rng.choice(xs)


def _maybe_constant_list(rng: random.Random) -> Args:
    size = rng.randint(2, 6)
    if rng.choice((False, True)):
        return ([rng.choice(SMALL)] * size,)
    return (_draw(rng, SMALL, size, size),)


def _equal_length_lists(rng: random.Random) -> Args:
    size = rng.randint(1, 5)
    return _draw(rng, SMALL, size, size), _draw(rng, SMALL, size, size)


def _list_and_nonzero_replacement(rng: random.Random) -> Args:
    xs = _draw(rng, SPARSE, 2, 7)
    if 0 not in xs:
        xs[0] = 0
    return xs, rng.choice(tuple(value for value in MIXED if value != 0))


def _maybe_palindrome(rng: random.Random) -> Args:
    size = rng.randint(2, 6)
    if rng.choice((False, True)):
        left = _draw(rng, SMALL, (size + 1) // 2, (size + 1) // 2)
        return ([*left, *reversed(left[: size // 2])],)
    return (_draw(rng, SMALL, size, size),)


def _shrinks(args_list: list[Args], results: list[Result]) -> bool:
    """For the filters: some example must drop an item, and some must keep one.

    Without this a drawn set can be all-pass or all-reject, which makes the
    examples consistent with the identity or with `nil`, and so says nothing
    about the predicate.
    """
    kept = [len(r) for r in results if isinstance(r, list)]
    sizes = [len(a[0]) for a in args_list if isinstance(a[0], list)]
    return any(k > 0 for k in kept) and any(
        k < n for k, n in zip(kept, sizes, strict=True)
    )


def _shrinks_flat(args_list: list[Args], results: list[Result]) -> bool:
    sizes = [len(args[0]) for args in args_list if isinstance(args[0], list)]
    kept = [len(result) for result in results if isinstance(result, list)]
    return any(kept) and any(
        kept_size < size for kept_size, size in zip(kept, sizes, strict=True)
    )


def _has_matches_and_mismatches(args_list: list[Args], results: list[Result]) -> bool:
    distances = [result for result in results if isinstance(result, int)]
    sizes = [len(args[0]) for args in args_list if isinstance(args[0], list)]
    return any(distance > 0 for distance in distances) and any(
        distance < size for distance, size in zip(distances, sizes, strict=True)
    )


def _brackets_target(args_list: list[Args], _results: list[Result]) -> bool:
    return any(
        isinstance(args[0], list)
        and isinstance(args[1], int)
        and min(args[0]) < args[1] < max(args[0])
        for args in args_list
    )


def _even(value: int) -> bool:
    # The witness computes `x - 2 * (x // 2)`, which is Python's floor
    # semantics, so the reference must floor too rather than use `abs`.
    return value - 2 * (value // 2) == 0


def _longest_strict_increase(xs: list[int]) -> int:
    best = current = 1
    for left, right in pairwise(xs):
        current = current + 1 if left < right else 1
        best = max(best, current)
    return best


def _record_extreme_count(xs: list[int]) -> int:
    lo = hi = xs[0]
    count = 1
    for x in xs[1:]:
        if x < lo:
            lo = x
            count += 1
        elif x > hi:
            hi = x
            count += 1
    return count


def _largest_zero_block_sum(xs: list[int]) -> int:
    best = current = 0
    for x in xs:
        if x == 0:
            current = 0
        else:
            current += x
            best = max(best, current)
    return best


def _running_ranges(xs: list[int]) -> list[int]:
    lo = hi = xs[0]
    out = [0]
    for x in xs[1:]:
        lo = min(lo, x)
        hi = max(hi, x)
        out.append(hi - lo)
    return out


def _run_length_encode(xs: list[int]) -> list[int]:
    out: list[int] = []
    for x in xs:
        if out and out[-2] == x:
            out[-1] += 1
        else:
            out.extend((x, 1))
    return out


def _unimodal_flag(xs: list[int]) -> int:
    descending = False
    for left, right in pairwise(xs):
        if right < left:
            descending = True
        elif descending and left < right:
            return 0
    return 1


def _max_running_mean(xs: list[int]) -> int:
    return max(sum(xs[:end]) // end for end in range(1, len(xs) + 1))


def _longest_positive_balance_streak(xs: list[int]) -> int:
    balance = current = best = 0
    for x in xs:
        balance += x
        current = current + 1 if balance > 0 else 0
        best = max(best, current)
    return best


CATALOGUE: tuple[Behaviour, ...] = (
    # -- scalars, so the corpus has a floor that needs no combinator ---------
    Behaviour(
        name="add_pair",
        summary="add two numbers",
        witnesses=("add",),
        sample=_pair(SMALL),
        reference=lambda a, b: a + b,
        tags=frozenset({"scalar"}),
    ),
    Behaviour(
        name="max_pair",
        summary="the larger of two numbers",
        witnesses=("[ over lt? ] [ drop ] [ swap drop ] ifte",),
        sample=_pair(SMALL),
        reference=max,
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="abs_diff_pair",
        summary="how far apart two numbers are",
        witnesses=("sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte",),
        sample=_pair(SMALL),
        reference=lambda a, b: abs(a - b),
        tags=frozenset({"scalar", "branch"}),
    ),
    # -- list -> list, elementwise ------------------------------------------
    Behaviour(
        name="double_each",
        summary="double every element",
        witnesses=("[ dup add ] map", "[ 1 1 add mul ] map"),
        sample=_list_of(SMALL),
        reference=lambda xs: [x + x for x in xs],
        tags=frozenset({"map"}),
    ),
    Behaviour(
        name="increment_each",
        summary="add one to every element",
        witnesses=("[ 1 add ] map", "[ 1 swap add ] map"),
        sample=_list_of(SMALL),
        reference=lambda xs: [x + 1 for x in xs],
        tags=frozenset({"map"}),
    ),
    Behaviour(
        name="square_each",
        summary="square every element",
        witnesses=("[ dup mul ] map",),
        sample=_list_of(SMALL),
        reference=lambda xs: [x * x for x in xs],
        tags=frozenset({"map"}),
    ),
    Behaviour(
        name="negate_each",
        summary="flip the sign of every element",
        witnesses=("[ 0 swap sub ] map", "[ 0 1 sub mul ] map"),
        sample=_list_of(SMALL),
        reference=lambda xs: [-x for x in xs],
        tags=frozenset({"map"}),
    ),
    Behaviour(
        name="halve_each",
        summary="halve every element, rounding down",
        witnesses=("[ 1 1 add div ] map",),
        sample=_list_of(SMALL),
        reference=lambda xs: [x // 2 for x in xs],
        tags=frozenset({"map"}),
    ),
    Behaviour(
        name="abs_each",
        summary="take the magnitude of every element",
        witnesses=("[ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] map",),
        sample=_list_of(MIXED),
        reference=lambda xs: [abs(x) for x in xs],
        tags=frozenset({"map", "branch"}),
    ),
    Behaviour(
        name="parity_each",
        summary="replace every element by its remainder mod two",
        witnesses=("[ dup 1 1 add div 1 1 add mul sub ] map",),
        sample=_list_of(SMALL),
        reference=lambda xs: [x - 2 * (x // 2) for x in xs],
        tags=frozenset({"map"}),
    ),
    # -- list -> list, structural -------------------------------------------
    Behaviour(
        name="reverse",
        summary="reverse the list",
        witnesses=("nil [ cons ] fold",),
        sample=_list_of(SMALL),
        reference=lambda xs: list(reversed(xs)),
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="tail",
        summary="drop the first element",
        witnesses=("uncons drop",),
        sample=_list_of(SMALL),
        reference=lambda xs: xs[1:],
        tags=frozenset({"first-order"}),
    ),
    Behaviour(
        name="prefix_sums",
        summary="running total of the list so far, one entry per element",
        witnesses=(
            "nil "
            "[ [ over null? ] [ cons ] [ over uncons swap drop add cons ] ifte ] fold "
            "nil [ cons ] fold",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: [sum(xs[: i + 1]) for i in range(len(xs))],
        tags=frozenset({"fold", "branch"}),
    ),
    # -- list -> list, selection --------------------------------------------
    Behaviour(
        name="keep_zeros",
        summary="keep only the zeros",
        witnesses=("[ 0 eq? ] filter",),
        sample=_list_of(SPARSE),
        reference=lambda xs: [x for x in xs if x == 0],
        tags=frozenset({"filter"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="keep_negative",
        summary="keep only the negative numbers",
        witnesses=("[ 0 lt? ] filter",),
        sample=_list_of(MIXED),
        reference=lambda xs: [x for x in xs if x < 0],
        tags=frozenset({"filter"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="keep_positive",
        summary="keep only the positive numbers",
        witnesses=("[ 0 swap lt? ] filter",),
        sample=_list_of(MIXED),
        reference=lambda xs: [x for x in xs if x > 0],
        tags=frozenset({"filter"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="keep_evens",
        summary="keep only the even numbers",
        witnesses=("[ dup 1 1 add div 1 1 add mul sub 0 eq? ] filter",),
        sample=_list_of(SMALL),
        reference=lambda xs: [x for x in xs if _even(x)],
        tags=frozenset({"filter"}),
        accept=_shrinks,
    ),
    # -- list -> number, folds ----------------------------------------------
    Behaviour(
        name="sum",
        summary="add the list up",
        witnesses=("0 [ add ] fold", "uncons [ add ] fold"),
        sample=_list_of(SMALL),
        reference=sum,
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="product",
        summary="multiply the list together",
        witnesses=("1 [ mul ] fold", "uncons [ mul ] fold"),
        sample=_list_of(MIXED, lo=2, hi=4),
        reference=math.prod,
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="length",
        summary="how many elements the list has",
        witnesses=("0 [ drop 1 add ] fold", "[ drop 1 ] map 0 [ add ] fold"),
        sample=_list_of(SMALL, lo=1, hi=7),
        reference=len,
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="head",
        summary="the first element",
        witnesses=("uncons swap drop",),
        sample=_list_of(SMALL),
        reference=lambda xs: xs[0],
        tags=frozenset({"first-order"}),
    ),
    Behaviour(
        name="last",
        summary="the last element",
        witnesses=("0 [ swap drop ] fold", "nil [ cons ] fold uncons swap drop"),
        sample=_list_of(SMALL),
        reference=lambda xs: xs[-1],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="maximum",
        summary="the largest element",
        witnesses=("uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold",),
        sample=_list_of(SMALL),
        reference=max,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="minimum",
        summary="the smallest element",
        witnesses=("uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold",),
        sample=_list_of(SMALL),
        reference=min,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="mean",
        summary="the average of the list, rounded down",
        witnesses=("dup 0 [ add ] fold swap 0 [ drop 1 add ] fold div",),
        sample=_list_of(POSITIVE),
        reference=lambda xs: sum(xs) // len(xs),
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="span",
        summary="the gap between the largest and smallest element",
        witnesses=(
            "dup uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold "
            "swap uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold sub",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: max(xs) - min(xs),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="sum_of_squares",
        summary="square every element, then add them up",
        # The map/fold pair reads more naturally and is three ops longer; a
        # single fold that squares as it accumulates is the shorter program.
        witnesses=("0 [ dup mul add ] fold", "[ dup mul ] map 0 [ add ] fold"),
        sample=_list_of(SMALL),
        reference=lambda xs: sum(x * x for x in xs),
        tags=frozenset({"map", "fold"}),
    ),
    Behaviour(
        name="sum_positive",
        summary="add up only the positive numbers",
        witnesses=("[ 0 swap lt? ] filter 0 [ add ] fold",),
        sample=_list_of(MIXED),
        reference=lambda xs: sum(x for x in xs if x > 0),
        tags=frozenset({"filter", "fold"}),
    ),
    Behaviour(
        name="count_zeros",
        summary="how many zeros the list has",
        witnesses=("[ 0 eq? ] filter 0 [ drop 1 add ] fold",),
        sample=_list_of(SPARSE),
        reference=lambda xs: sum(1 for x in xs if x == 0),
        tags=frozenset({"filter", "fold"}),
    ),
    Behaviour(
        name="count_negative",
        summary="how many negative numbers the list has",
        witnesses=("[ 0 lt? ] filter 0 [ drop 1 add ] fold",),
        sample=_list_of(MIXED),
        reference=lambda xs: sum(1 for x in xs if x < 0),
        tags=frozenset({"filter", "fold"}),
    ),
    Behaviour(
        name="count_evens",
        summary="how many even numbers the list has",
        witnesses=(
            "[ dup 1 1 add div 1 1 add mul sub 0 eq? ] filter 0 [ drop 1 add ] fold",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: sum(1 for x in xs if _even(x)),
        tags=frozenset({"filter", "fold"}),
    ),
    # -- (list, number): the program must build a closure over the number ----
    Behaviour(
        name="scale_by",
        summary="multiply every element by the given number",
        witnesses=("[ mul ] swap cons map",),
        sample=_list_and_int(SMALL, POSITIVE),
        reference=lambda xs, n: [x * n for x in xs],
        tags=frozenset({"map", "closure"}),
    ),
    Behaviour(
        name="add_to_each",
        summary="add the given number to every element",
        witnesses=("[ add ] swap cons map",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, n: [x + n for x in xs],
        tags=frozenset({"map", "closure"}),
    ),
    Behaviour(
        name="n_minus_each",
        summary="subtract every element from the given number",
        witnesses=("[ swap sub ] swap cons map",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, n: [n - x for x in xs],
        tags=frozenset({"map", "closure"}),
    ),
    Behaviour(
        name="keep_below",
        summary="keep the elements smaller than the given number",
        witnesses=("[ lt? ] swap cons filter",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, n: [x for x in xs if x < n],
        tags=frozenset({"filter", "closure"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="keep_above",
        summary="keep the elements larger than the given number",
        witnesses=("[ swap lt? ] swap cons filter",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, n: [x for x in xs if x > n],
        tags=frozenset({"filter", "closure"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="count_equal",
        summary="how many times the given number appears",
        witnesses=("[ eq? ] swap cons filter 0 [ drop 1 add ] fold",),
        sample=_list_and_int(SPARSE, SPARSE),
        reference=lambda xs, n: sum(1 for x in xs if x == n),
        tags=frozenset({"filter", "fold", "closure"}),
    ),
    Behaviour(
        name="push_front",
        summary="put the given number on the front of the list",
        witnesses=("cons",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, n: [n, *xs],
        tags=frozenset({"first-order"}),
    ),
    # -- two lists -----------------------------------------------------------
    Behaviour(
        name="append",
        summary="join the two lists end to end",
        witnesses=("cat",),
        sample=_two_lists(SMALL),
        reference=lambda xs, ys: [*xs, *ys],
        tags=frozenset({"first-order"}),
    ),
    Behaviour(
        name="sum_both",
        summary="add up the numbers in both lists",
        witnesses=(
            "cat 0 [ add ] fold",
            "0 [ add ] fold swap 0 [ add ] fold add",
        ),
        sample=_two_lists(SMALL),
        reference=lambda xs, ys: sum(xs) + sum(ys),
        tags=frozenset({"fold"}),
    ),
    # -- stage 1: scalar and whole-list decisions ---------------------------
    Behaviour(
        name="min_pair",
        summary="the smaller of two numbers",
        witnesses=("[ over lt? ] [ swap drop ] [ drop ] ifte",),
        sample=_pair(SMALL),
        reference=min,
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="ordered_pair",
        summary="put two numbers in ascending order",
        witnesses=("[ over lt? ] [ swap quote swap cons ] [ quote swap cons ] ifte",),
        sample=_pair(SMALL),
        reference=lambda a, b: sorted((a, b)),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="sign",
        summary="encode a number as -1, 0, or 1 according to its sign",
        witnesses=(
            "[ dup 0 lt? ] [ drop 0 1 sub ] "
            "[ [ dup 0 eq? ] [ drop 0 ] [ drop 1 ] ifte ] ifte",
        ),
        sample=_scalar_of(SMALL),
        reference=lambda x: (x > 0) - (x < 0),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="median_three",
        summary="the middle of three numbers",
        witnesses=(
            "[ [ over lt? ] dip swap ] "
            "[ swap [ swap ] dip swap ] [ swap [ swap ] dip ] ifte "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip "
            "[ over lt? ] [ swap drop ] [ drop ] ifte",
        ),
        sample=_triple_of(SMALL),
        reference=lambda a, b, c: sorted((a, b, c))[1],
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="sum_negative",
        summary="add up only the negative numbers",
        witnesses=("0 [ [ dup 0 lt? ] [ add ] [ drop ] ifte ] fold",),
        sample=_list_of(MIXED),
        reference=lambda xs: sum(x for x in xs if x < 0),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="count_in_range",
        summary="count values strictly between the given bounds",
        witnesses=(
            "[ [ swap drop lt? ] "
            "[ [ drop swap lt? ] [ drop drop drop 1 add ] "
            "[ drop drop drop ] ifte ] "
            "[ drop drop drop ] ifte ] "
            "swap cons swap cons 0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(lo < x < hi for x in xs),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="replace_zeros",
        summary="replace every zero by the given number",
        witnesses=("[ [ over 0 eq? ] [ swap drop ] [ drop ] ifte ] swap cons map",),
        sample=_list_and_nonzero_replacement,
        reference=lambda xs, n: [n if x == 0 else x for x in xs],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="palindrome_indicator",
        summary="one if the list reads the same backward, otherwise zero",
        witnesses=("dup nil [ cons ] fold eq? [ ] [ drop 1 ] [ drop 0 ] ifte",),
        sample=_maybe_palindrome,
        reference=lambda xs: int(xs == list(reversed(xs))),
        tags=frozenset({"fold", "branch"}),
        accept=lambda _args, results: len(set(results)) > 1,
    ),
    # -- stage 1: closures and structural list problems --------------------
    Behaviour(
        name="raise_to_floor",
        summary="raise every element below the given floor",
        witnesses=("[ [ over lt? ] [ drop ] [ swap drop ] ifte ] swap cons map",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, floor: [max(x, floor) for x in xs],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="quotient_remainder",
        summary="the floor quotient and remainder",
        witnesses=("over over div [ over over div mul sub ] dip swap quote swap cons",),
        sample=_pair_nonzero_divisor,
        reference=lambda a, b: [a // b, a - b * (a // b)],
        tags=frozenset({"scalar"}),
    ),
    Behaviour(
        name="replace_value",
        summary="replace every occurrence of one value by another",
        witnesses=(
            "[ [ drop eq? ] [ swap drop swap drop ] [ drop drop ] ifte ] "
            "swap cons swap cons map",
        ),
        sample=_list_old_new,
        reference=lambda xs, old, new: [new if x == old else x for x in xs],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="clamp_each",
        summary="clamp every element to the given inclusive interval",
        witnesses=(
            "[ swap [ [ over lt? ] [ swap drop ] [ drop ] ifte ] dip "
            "[ over lt? ] [ drop ] [ swap drop ] ifte ] "
            "swap cons swap cons map",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: [min(max(x, lo), hi) for x in xs],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="remove_value",
        summary="remove every occurrence of the given value",
        witnesses=(
            "[ [ eq? ] [ drop drop false ] [ drop drop true ] ifte ] swap cons filter",
        ),
        sample=_list_existing_value,
        reference=lambda xs, target: [x for x in xs if x != target],
        tags=frozenset({"filter", "closure", "branch"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="push_back",
        summary="put the given number on the end of the list",
        witnesses=("nil swap cons cat",),
        sample=_list_and_int(SMALL, SMALL),
        reference=lambda xs, n: [*xs, n],
        tags=frozenset({"first-order"}),
    ),
    Behaviour(
        name="insert_after_first",
        summary="insert the given number after the first element",
        witnesses=("swap uncons [ swap cons ] dip cons",),
        sample=_list_and_int(SMALL, SMALL, lo=1),
        reference=lambda xs, n: [xs[0], n, *xs[1:]],
        tags=frozenset({"first-order"}),
    ),
    Behaviour(
        name="rotate_left",
        summary="move the first element to the end",
        witnesses=("uncons nil swap cons cat",),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: [*xs[1:], xs[0]],
        tags=frozenset({"first-order"}),
    ),
    Behaviour(
        name="rotate_right",
        summary="move the last element to the front",
        witnesses=("nil [ cons ] fold uncons [ nil [ cons ] fold ] dip cons",),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: [xs[-1], *xs[:-1]],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="drop_last",
        summary="drop the last element",
        witnesses=("nil [ cons ] fold uncons drop nil [ cons ] fold",),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: xs[:-1],
        tags=frozenset({"fold", "first-order"}),
    ),
    Behaviour(
        name="endpoints",
        summary="return the first and last elements",
        witnesses=(
            "dup uncons swap drop [ 0 [ swap drop ] fold ] dip quote swap quote cat",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: [xs[0], xs[-1]],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="adjacent_differences",
        summary="differences between each element and its predecessor",
        witnesses=(
            "uncons nil swap cons "
            "[ dup [ swap uncons swap [ sub ] dip swap cons ] dip cons ] fold "
            "uncons drop nil [ cons ] fold",
        ),
        sample=_list_of(SMALL, lo=2),
        reference=lambda xs: [b - a for a, b in pairwise(xs)],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="collapse_runs",
        summary="collapse each run of equal adjacent values to one value",
        witnesses=(
            "nil [ [ over null? ] [ cons ] "
            "[ [ over uncons swap drop eq? ] [ drop ] [ cons ] ifte ] ifte ] "
            "fold nil [ cons ] fold",
        ),
        sample=_list_of(SPARSE, lo=2, hi=7),
        reference=lambda xs: [
            x for index, x in enumerate(xs) if index == 0 or x != xs[index - 1]
        ],
        tags=frozenset({"fold", "branch"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="all_equal_flag",
        summary="one if every element is equal, otherwise zero",
        witnesses=(
            "dup dup uncons swap drop [ eq? ] swap cons filter "
            "[ eq? ] [ drop drop 1 ] [ drop drop 0 ] ifte",
        ),
        sample=_maybe_constant_list,
        reference=lambda xs: int(all(x == xs[0] for x in xs)),
        tags=frozenset({"filter", "closure", "branch"}),
        accept=lambda _args, results: len(set(results)) > 1,
    ),
    Behaviour(
        name="closest_to",
        summary="the element closest to the given number, first on ties",
        witnesses=(
            "[ uncons ] dip "
            "[ [ dup [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip "
            "swap [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip "
            "swap lt? ] [ drop swap drop ] [ drop drop ] ifte ] "
            "swap cons fold",
        ),
        sample=_list_and_int(SMALL, SMALL, lo=1),
        reference=lambda xs, n: min(xs, key=lambda x: abs(x - n)),
        tags=frozenset({"fold", "closure", "branch"}),
        accept=_brackets_target,
    ),
    Behaviour(
        name="nearest_bound",
        summary="the nearer interval endpoint, choosing the lower on ties",
        witnesses=(
            "[ [ over swap sub dup mul ] dip swap "
            "[ over swap sub dup mul ] dip lt? ] "
            "[ swap drop swap drop ] [ drop swap drop ] ifte",
        ),
        sample=_value_interval,
        reference=lambda x, lo, hi: hi if abs(x - hi) < abs(x - lo) else lo,
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="choose_longer",
        summary="the longer list, choosing the second on ties",
        witnesses=(
            "[ 0 [ drop 1 add ] fold swap 0 [ drop 1 add ] fold lt? ] "
            "[ drop ] [ swap drop ] ifte",
        ),
        sample=_two_lists(SMALL),
        reference=lambda xs, ys: xs if len(xs) > len(ys) else ys,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="choose_larger_total",
        summary="the list with the larger total, choosing the second on ties",
        witnesses=(
            "[ 0 [ add ] fold swap 0 [ add ] fold lt? ] [ drop ] [ swap drop ] ifte",
        ),
        sample=_two_lists(MIXED),
        reference=lambda xs, ys: xs if sum(xs) > sum(ys) else ys,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="sum_and_product",
        summary="the sum and product of two numbers",
        witnesses=("over over add [ mul ] dip swap quote swap cons",),
        sample=_pair(SMALL),
        reference=lambda a, b: [a + b, a * b],
        tags=frozenset({"scalar"}),
    ),
    Behaviour(
        name="list_totals",
        summary="return the total of each list in input order",
        witnesses=("0 [ add ] fold swap 0 [ add ] fold quote swap quote cat",),
        sample=_two_lists(SMALL),
        reference=lambda xs, ys: [sum(xs), sum(ys)],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="append_reversed_second",
        summary="append the second list in reverse order",
        witnesses=("nil [ cons ] fold cat",),
        sample=_two_lists(SMALL),
        reference=lambda xs, ys: [*xs, *reversed(ys)],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="dot_product",
        summary="the dot product of two equal-length lists",
        witnesses=(
            "0 cons [ swap uncons swap uncons swap "
            "[ [ swap ] dip mul add ] dip swap cons ] fold "
            "uncons swap drop",
        ),
        sample=_equal_length_lists,
        reference=lambda xs, ys: sum(x * y for x, y in zip(xs, ys, strict=True)),
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="hamming_distance",
        summary="how many positions differ between two equal-length lists",
        witnesses=(
            "0 cons [ swap uncons swap uncons swap "
            "[ [ swap ] dip [ eq? ] [ drop drop ] "
            "[ drop drop 1 add ] ifte ] dip swap cons ] fold "
            "uncons swap drop",
        ),
        sample=_equal_length_lists,
        reference=lambda xs, ys: sum(x != y for x, y in zip(xs, ys, strict=True)),
        tags=frozenset({"fold", "branch"}),
        accept=_has_matches_and_mismatches,
    ),
    Behaviour(
        name="magnitude_ordered_pair",
        summary="put two numbers in ascending order of magnitude",
        witnesses=(
            "[ over dup mul over dup mul swap lt? ] "
            "[ swap quote swap cons ] [ quote swap cons ] ifte",
        ),
        sample=_pair(SMALL),
        reference=lambda a, b: sorted((a, b), key=abs),
        tags=frozenset({"scalar", "branch"}),
    ),
    # -- stage 1: branch-bearing stateful folds -----------------------------
    Behaviour(
        name="sum_absolute_values",
        summary="add up the magnitudes of all elements",
        witnesses=("0 [ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte add ] fold",),
        sample=_list_of(SMALL),
        reference=lambda xs: sum(abs(x) for x in xs),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="strict_peak_count",
        summary="how many interior elements are larger than both neighbors",
        witnesses=(
            "uncons [ uncons ] dip swap 0 nil swap cons swap cons swap cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop lt? ] "
            "[ [ swap drop swap [ swap drop ] dip lt? ] "
            "[ swap [ swap [ swap drop ] dip swap ] dip 1 add "
            "nil swap cons swap cons swap cons ] "
            "[ swap [ swap [ swap drop ] dip swap ] dip "
            "nil swap cons swap cons swap cons ] ifte ] "
            "[ swap [ swap [ swap drop ] dip swap ] dip "
            "nil swap cons swap cons swap cons ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=2),
        reference=lambda xs: sum(
            xs[index - 1] < xs[index] and xs[index] > xs[index + 1]
            for index in range(1, len(xs) - 1)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="running_maxima",
        summary="the largest value seen so far at every position",
        witnesses=(
            "nil [ [ over null? ] [ cons ] "
            "[ over uncons swap drop [ over lt? ] [ drop ] "
            "[ swap drop ] ifte cons ] ifte ] fold nil [ cons ] fold",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: [max(xs[: index + 1]) for index in range(len(xs))],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="running_minima",
        summary="the smallest value seen so far at every position",
        witnesses=(
            "nil [ [ over null? ] [ cons ] "
            "[ over uncons swap drop [ over lt? ] [ swap drop ] "
            "[ drop ] ifte cons ] ifte ] fold nil [ cons ] fold",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: [min(xs[: index + 1]) for index in range(len(xs))],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="record_high_count",
        summary="how many elements set a new all-time high, counting the first",
        witnesses=(
            "uncons nil 1 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ [ drop ] dip lt? ] "
            "[ swap [ swap drop ] dip 1 add nil swap cons swap cons ] "
            "[ drop nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: (
            1 + sum(xs[index] > max(xs[:index]) for index in range(1, len(xs)))
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="positive_majority",
        summary="one if a strict majority of elements are positive, otherwise zero",
        witnesses=(
            "dup 0 [ [ 0 swap lt? ] [ drop 1 add ] [ drop ] ifte ] fold "
            "1 1 add mul swap 0 [ drop 1 add ] fold "
            "[ over lt? ] [ drop drop 1 ] [ drop drop 0 ] ifte",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: int(sum(x > 0 for x in xs) * 2 > len(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="adjacent_repetition_count",
        summary="how many adjacent pairs repeat the same value",
        witnesses=(
            "uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ [ drop ] dip eq? ] "
            "[ swap [ swap drop ] dip 1 add nil swap cons swap cons ] "
            "[ swap [ swap drop ] dip nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SPARSE, lo=1),
        reference=lambda xs: sum(a == b for a, b in pairwise(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="largest_adjacent_sum",
        summary="the largest sum of two neighboring elements",
        witnesses=(
            "uncons [ uncons ] dip swap dup [ add ] dip swap "
            "nil swap cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ swap [ add ] dip swap lt? ] "
            "[ [ drop ] dip dup [ add ] dip swap nil swap cons swap cons ] "
            "[ swap [ swap drop ] dip nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=2),
        reference=lambda xs: max(a + b for a, b in pairwise(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="largest_adjacent_gap",
        summary="the largest absolute difference between neighboring elements",
        witnesses=(
            "uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ swap [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip "
            "swap lt? ] "
            "[ swap drop dup [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] "
            "dip swap nil swap cons swap cons ] "
            "[ swap [ swap drop ] dip nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: max((abs(a - b) for a, b in pairwise(xs)), default=0),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="longest_positive_run",
        summary="the length of the longest consecutive run of positive elements",
        witnesses=(
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ 0 swap lt? ] "
            "[ drop swap 1 add dup "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip swap "
            "nil swap cons swap cons ] "
            "[ drop [ drop ] dip 0 swap nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: max(
            (
                end - start
                for start in range(len(xs))
                for end in range(start + 1, len(xs) + 1)
                if all(x > 0 for x in xs[start:end])
            ),
            default=0,
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="maximum_prefix_sum",
        summary="the largest sum of any nonempty prefix",
        witnesses=(
            "uncons dup nil swap cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ swap [ add ] dip swap lt? ] "
            "[ [ drop ] dip add dup nil swap cons swap cons ] "
            "[ swap [ add ] dip nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: max(sum(xs[:end]) for end in range(1, len(xs) + 1)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="maximum_subarray_sum",
        summary="the largest sum of any nonempty contiguous slice",
        witnesses=(
            "uncons dup nil swap cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "swap [ dup [ add ] dip "
            "[ over lt? ] [ drop ] [ swap drop ] ifte ] dip swap dup "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip swap "
            "nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: max(
            sum(xs[start:end])
            for start in range(len(xs))
            for end in range(start + 1, len(xs) + 1)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="sign_balance",
        summary="the number of positive elements minus the number of negative ones",
        witnesses=(
            "0 [ [ dup 0 lt? ] [ drop 1 sub ] "
            "[ [ 0 swap lt? ] [ drop 1 add ] [ drop ] ifte ] ifte ] fold",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: sum((x > 0) - (x < 0) for x in xs),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="nonnegative_running_balance",
        summary="add in order, resetting the balance to zero below zero",
        witnesses=("0 [ add [ dup 0 lt? ] [ drop 0 ] [ ] ifte ] fold",),
        sample=_list_of(SMALL),
        reference=lambda xs: max([0, *(sum(xs[start:]) for start in range(len(xs)))]),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="total_variation",
        summary="the sum of absolute changes between neighboring elements",
        witnesses=(
            "uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip swap "
            "[ dup [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip swap ] "
            "dip add nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: sum(abs(a - b) for a, b in pairwise(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="maximum_drawdown",
        summary="the largest drop from an earlier value to a later value",
        witnesses=(
            "uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip swap "
            "[ dup [ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip ] dip "
            "swap [ over ] dip sub "
            "[ over lt? ] [ drop ] [ swap drop ] ifte "
            "nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: max(
            [
                0,
                *(
                    xs[earlier] - xs[later]
                    for earlier in range(len(xs))
                    for later in range(earlier + 1, len(xs))
                ),
            ]
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="largest_magnitude",
        summary="the largest absolute value in the list",
        witnesses=(
            "0 [ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte "
            "[ over lt? ] [ drop ] [ swap drop ] ifte ] fold",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: max(abs(x) for x in xs),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="smallest_magnitude",
        summary="the smallest absolute value in the list",
        witnesses=(
            "uncons [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte "
            "[ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte "
            "[ over lt? ] [ swap drop ] [ drop ] ifte ] fold",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: min(abs(x) for x in xs),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="stable_unique",
        summary="remove repeated values while keeping their first occurrences",
        witnesses=(
            "nil [ [ [ eq? ] swap cons filter null? ] "
            "[ cons ] [ drop ] ifte ] fold nil [ cons ] fold",
        ),
        sample=_list_of(SPARSE),
        reference=lambda xs: [x for index, x in enumerate(xs) if x not in xs[:index]],
        tags=frozenset({"fold", "filter", "branch"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="leading_positive_count",
        summary="how many initial elements precede the first nonpositive one",
        witnesses=(
            "0 [ [ drop 0 lt? ] [ drop ] "
            "[ [ 0 swap lt? ] [ drop 1 add ] "
            "[ drop 1 add 0 swap sub ] ifte ] ifte ] fold "
            "[ dup 0 lt? ] [ 0 swap sub 1 sub ] [ ] ifte",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: next(
            (index for index, x in enumerate(xs) if x <= 0), len(xs)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    # -- stage 1: aggregate and multi-stage statistics ---------------------
    Behaviour(
        name="second_largest",
        summary="the second-largest element, counting duplicate values separately",
        witnesses=(
            "uncons swap uncons [ swap ] dip "
            "[ over lt? ] [ nil swap cons swap cons ] "
            "[ swap nil swap cons swap cons ] ifte "
            "[ [ over uncons swap drop swap lt? ] "
            "[ swap uncons swap drop nil swap cons swap cons ] "
            "[ [ over uncons drop uncons swap drop swap lt? ] "
            "[ swap uncons swap drop swap nil swap cons swap cons ] "
            "[ drop ] ifte ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=2, hi=7),
        reference=lambda xs: sorted(xs, reverse=True)[1],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="zero_base_by_minimum",
        summary="shift every value so the list minimum becomes zero",
        witnesses=(
            "dup uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold "
            "[ sub ] swap cons map",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: [x - min(xs) for x in xs],
        tags=frozenset({"map", "fold", "branch", "closure"}),
    ),
    Behaviour(
        name="mean_absolute_deviation",
        summary="rounded-down mean absolute distance from the rounded-down mean",
        witnesses=(
            "dup dup 0 [ add ] fold swap 0 [ drop 1 add ] fold div "
            "[ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] swap cons map "
            "dup 0 [ add ] fold swap 0 [ drop 1 add ] fold div",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: sum(abs(x - (sum(xs) // len(xs))) for x in xs) // len(xs),
        tags=frozenset({"map", "fold", "branch", "closure"}),
    ),
    Behaviour(
        name="compare_each_to_mean",
        summary="encode each value as -1, 0, or 1 relative to the rounded-down mean",
        witnesses=(
            "dup dup 0 [ add ] fold swap 0 [ drop 1 add ] fold div "
            "[ [ over eq? ] [ drop drop 0 ] "
            "[ [ lt? ] [ drop drop 0 1 sub ] [ drop drop 1 ] ifte ] ifte ] "
            "swap cons map",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: [
            -1 if x < (sum(xs) // len(xs)) else int(x > (sum(xs) // len(xs)))
            for x in xs
        ],
        tags=frozenset({"map", "fold", "branch", "closure"}),
    ),
    Behaviour(
        name="positive_part",
        summary="keep the positive part of a number",
        witnesses=("[ dup 0 lt? ] [ drop 0 ] [ ] ifte",),
        sample=_scalar_of(SMALL),
        reference=lambda x: max(x, 0),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="all_positive",
        summary="one if every element is positive, otherwise zero",
        witnesses=(
            "1 [ [ 0 swap lt? ] [ drop ] [ drop drop 0 ] ifte ] fold",
            "[ [ 0 swap lt? ] [ drop 1 ] [ drop 0 ] ifte ] map 1 [ mul ] fold",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: int(all(x > 0 for x in xs)),
        tags=frozenset({"map", "fold", "branch"}),
    ),
    Behaviour(
        name="capped_sum",
        summary="sum the values after capping each at the given ceiling",
        witnesses=(
            "[ [ lt? ] [ drop add ] [ swap drop add ] ifte ] swap cons 0 swap fold",
        ),
        sample=_list_and_int(MIXED, SMALL),
        reference=lambda xs, ceiling: sum(min(x, ceiling) for x in xs),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="squared_deviation_total",
        summary="sum squared distances from the rounded-down mean",
        witnesses=(
            "dup dup 0 [ add ] fold swap 0 [ drop 1 add ] fold div "
            "[ sub dup mul add ] swap cons 0 swap fold",
            "dup dup 0 [ add ] fold swap 0 [ drop 1 add ] fold div "
            "[ sub dup mul ] swap cons map 0 [ add ] fold",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: sum((x - (sum(xs) // len(xs))) ** 2 for x in xs),
        tags=frozenset({"map", "fold", "closure"}),
    ),
    # -- two lists, deliberately allowing unequal lengths -----------------
    Behaviour(
        name="truncated_dot_product",
        summary="dot product over the positions present in both lists",
        witnesses=(
            "0 cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ swap ] dip mul add ] dip swap cons ] ifte ] fold "
            "uncons swap drop",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: sum(x * y for x, y in zip(xs, ys, strict=False)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="paired_l1_distance",
        summary="total absolute difference over aligned positions in both lists",
        witnesses=(
            "0 cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ swap ] dip sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte add ] "
            "dip swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: sum(abs(x - y) for x, y in zip(xs, ys, strict=False)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="aligned_match_count",
        summary="how many aligned positions contain the same value",
        witnesses=(
            "0 cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ [ swap ] dip eq? ] [ drop swap drop 1 add ] "
            "[ drop swap drop ] ifte ] dip swap cons ] ifte ] fold "
            "uncons swap drop",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: sum(x == y for x, y in zip(xs, ys, strict=False)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="add_aligned_entries",
        summary="add corresponding entries, stopping when either list ends",
        witnesses=(
            "nil cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ swap ] dip add cons ] dip swap cons ] ifte ] fold "
            "uncons swap drop nil [ cons ] fold",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: [x + y for x, y in zip(xs, ys, strict=False)],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="aligned_maxima",
        summary="take the larger entry at each position shared by both lists",
        witnesses=(
            "nil cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ [ drop ] dip lt? ] [ [ [ drop ] dip ] dip cons ] "
            "[ drop swap cons ] ifte ] dip swap cons ] ifte ] fold "
            "uncons swap drop nil [ cons ] fold",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: [max(x, y) for x, y in zip(xs, ys, strict=False)],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="interleave_unequal",
        summary="alternate entries from two lists, then append the unmatched tail",
        witnesses=(
            "nil cons [ swap uncons [ drop null? ] "
            "[ swap [ swap cons ] dip swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ swap cons ] dip cons ] dip swap cons ] ifte ] fold "
            "uncons nil [ cons ] fold swap cat",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: [
            value
            for index in range(max(len(xs), len(ys)))
            for source in (xs, ys)
            if index < len(source)
            for value in (source[index],)
        ],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="common_prefix_length",
        summary="length of the equal prefix shared by two lists",
        witnesses=(
            "nil cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ [ drop ] dip eq? ] [ drop swap drop 1 cons ] "
            "[ drop swap drop 0 cons ] ifte ] dip swap cons ] ifte ] fold "
            "uncons swap drop nil [ cons ] fold "
            "0 [ [ drop 0 lt? ] [ drop ] "
            "[ [ 0 swap lt? ] [ drop 1 add ] "
            "[ drop 1 add 0 swap sub ] ifte ] ifte ] fold "
            "[ dup 0 lt? ] [ 0 swap sub 1 sub ] [ ] ifte",
        ),
        sample=_two_lists_with_shared_prefix,
        reference=lambda xs, ys: next(
            (index for index, (x, y) in enumerate(zip(xs, ys, strict=False)) if x != y),
            min(len(xs), len(ys)),
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="first_list_position_wins",
        summary="how many aligned positions are larger in the first list",
        witnesses=(
            "0 cons [ swap uncons [ drop null? ] "
            "[ swap drop swap drop nil swap cons ] "
            "[ [ uncons ] dip swap [ swap ] dip swap "
            "[ [ [ swap ] dip swap lt? ] [ drop swap drop 1 add ] "
            "[ drop swap drop ] ifte ] dip swap cons ] ifte ] fold "
            "uncons swap drop",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: sum(x > y for x, y in zip(xs, ys, strict=False)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="remove_values_seen_in_second",
        summary="remove first-list entries whose value appears in the second list",
        witnesses=("[ swap [ eq? ] swap cons filter null? ] swap cons filter",),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: [x for x in xs if x not in ys],
        tags=frozenset({"filter", "closure"}),
    ),
    Behaviour(
        name="shared_value_occurrences",
        summary="count first-list entries whose value appears in the second list",
        witnesses=(
            "[ swap [ eq? ] swap cons filter null? "
            "[ ] [ drop false ] [ drop true ] ifte ] swap cons filter "
            "0 [ drop 1 add ] fold",
        ),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: sum(x in ys for x in xs),
        tags=frozenset({"filter", "fold", "closure", "branch"}),
    ),
    Behaviour(
        name="disjoint_lists_flag",
        summary="one when the two lists have no value in common",
        witnesses=(
            "[ swap [ eq? ] swap cons filter null? "
            "[ ] [ drop false ] [ drop true ] ifte ] swap cons filter null? "
            "[ ] [ drop 1 ] [ drop 0 ] ifte",
        ),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: int(not any(x in ys for x in xs)),
        tags=frozenset({"filter", "closure", "branch"}),
    ),
    Behaviour(
        name="choose_wider_range",
        summary="the list with the wider value range, choosing the second on ties",
        witnesses=(
            "[ dup uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold "
            "swap uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold sub "
            "swap dup uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold "
            "swap uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold sub "
            "lt? ] [ drop ] [ swap drop ] ifte",
        ),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: xs if max(xs) - min(xs) > max(ys) - min(ys) else ys,
        tags=frozenset({"fold", "branch"}),
    ),
    # -- a list plus two runtime bounds -----------------------------------
    Behaviour(
        name="keep_inside_closed_interval",
        summary="keep values inside the inclusive interval",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop false ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop false ] "
            "[ drop drop drop true ] ifte ] ifte ] "
            "swap cons swap cons filter",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: [x for x in xs if lo <= x <= hi],
        tags=frozenset({"filter", "closure", "branch"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="closed_interval_membership_flags",
        summary="mark each value one inside the inclusive interval and zero outside",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop 0 ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop 0 ] "
            "[ drop drop drop 1 ] ifte ] ifte ] "
            "swap cons swap cons map",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: [int(lo <= x <= hi) for x in xs],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="sum_inside_closed_interval",
        summary="add values that lie inside the inclusive interval",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop ] "
            "[ drop drop add ] ifte ] ifte ] swap cons swap cons "
            "0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(x for x in xs if lo <= x <= hi),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="interval_violation_count",
        summary="count values outside the inclusive interval",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop 1 add ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop 1 add ] "
            "[ drop drop drop ] ifte ] ifte ] swap cons swap cons "
            "0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(x < lo or x > hi for x in xs),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="interval_side_balance",
        summary="number above the interval minus number below it",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop 0 1 sub add ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop 1 add ] "
            "[ drop drop drop ] ifte ] ifte ] swap cons swap cons "
            "0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum((x > hi) - (x < lo) for x in xs),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="total_distance_outside_interval",
        summary="total distance required to move all values into the interval",
        witnesses=(
            "[ [ drop lt? ] [ drop swap sub add ] "
            "[ [ [ drop ] dip swap lt? ] [ [ drop ] dip sub add ] "
            "[ drop drop drop ] ifte ] ifte ] swap cons swap cons "
            "0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(
            lo - x if x < lo else x - hi if x > hi else 0 for x in xs
        ),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="clamped_total",
        summary="sum the values after clamping each to the inclusive interval",
        witnesses=(
            "[ [ drop lt? ] [ drop swap drop add ] "
            "[ [ [ drop ] dip swap lt? ] [ swap drop swap drop add ] "
            "[ drop drop add ] ifte ] ifte ] swap cons swap cons "
            "0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(min(max(x, lo), hi) for x in xs),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="nearest_interval_endpoint_each",
        summary="replace each value by the nearer interval endpoint, lower on ties",
        witnesses=(
            "[ [ [ over swap sub dup mul ] dip swap "
            "[ over swap sub dup mul ] dip lt? ] "
            "[ swap drop swap drop ] [ drop swap drop ] ifte ] "
            "swap cons swap cons map",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: [
            hi if abs(x - hi) < abs(x - lo) else lo for x in xs
        ],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="interval_region_codes",
        summary="encode values below, inside, and above the interval as -1, 0, and 1",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop 0 1 sub ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop 1 ] "
            "[ drop drop drop 0 ] ifte ] ifte ] "
            "swap cons swap cons map",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: [-1 if x < lo else int(x > hi) for x in xs],
        tags=frozenset({"map", "closure", "branch"}),
    ),
    Behaviour(
        name="longest_interval_run",
        summary="length of the longest consecutive run inside the interval",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop 0 ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop 0 ] "
            "[ drop drop drop 1 ] ifte ] ifte ] "
            "swap cons swap cons map "
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ 0 swap lt? ] "
            "[ drop swap 1 add dup "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip swap "
            "nil swap cons swap cons ] "
            "[ drop [ drop ] dip 0 swap nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: max(
            (
                end - start
                for start in range(len(xs))
                for end in range(start + 1, len(xs) + 1)
                if all(lo <= x <= hi for x in xs[start:end])
            ),
            default=0,
        ),
        tags=frozenset({"map", "fold", "closure", "branch"}),
    ),
    Behaviour(
        name="span_inside_interval",
        summary="range from smallest to largest value inside the interval",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop false ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop false ] "
            "[ drop drop drop true ] ifte ] ifte ] "
            "swap cons swap cons filter "
            "dup uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold "
            "swap uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold sub",
        ),
        sample=_list_bounds_with_two_hits,
        reference=lambda xs, lo, hi: (
            max(x for x in xs if lo <= x <= hi) - min(x for x in xs if lo <= x <= hi)
        ),
        tags=frozenset({"filter", "fold", "closure", "branch"}),
    ),
    Behaviour(
        name="product_inside_interval",
        summary="multiply the values inside the inclusive interval",
        witnesses=(
            "[ [ drop lt? ] [ drop drop drop false ] "
            "[ [ [ drop ] dip swap lt? ] [ drop drop drop false ] "
            "[ drop drop drop true ] ifte ] ifte ] "
            "swap cons swap cons filter 1 [ mul ] fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: math.prod(x for x in xs if lo <= x <= hi),
        tags=frozenset({"filter", "fold", "closure", "branch"}),
    ),
    # -- three scalars: comparisons have to preserve all three values ------
    Behaviour(
        name="sort_three",
        summary="put three numbers in ascending order",
        witnesses=(
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "[ over lt? ] [ swap ] [ ] ifte "
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "nil swap cons swap cons swap cons",
        ),
        sample=_triple_of(SMALL),
        reference=lambda a, b, c: sorted((a, b, c)),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="three_number_range",
        summary="difference between the largest and smallest of three numbers",
        witnesses=(
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "[ over lt? ] [ swap ] [ ] ifte "
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "nil swap cons swap cons swap cons "
            "dup uncons swap drop swap 0 [ swap drop ] fold swap sub",
        ),
        sample=_triple_of(SMALL),
        reference=lambda a, b, c: max(a, b, c) - min(a, b, c),
        tags=frozenset({"scalar", "fold", "branch"}),
    ),
    Behaviour(
        name="smallest_pair_gap_three",
        summary="smallest absolute gap between any two of three numbers",
        witnesses=(
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "[ over lt? ] [ swap ] [ ] ifte "
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "over sub [ swap sub ] dip "
            "[ over lt? ] [ swap drop ] [ drop ] ifte",
        ),
        sample=_triple_of(SMALL),
        reference=lambda a, b, c: min(abs(a - b), abs(a - c), abs(b - c)),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="triangle_feasibility_flag",
        summary="one when three positive lengths can form a nondegenerate triangle",
        witnesses=(
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "[ over lt? ] [ swap ] [ ] ifte "
            "[ [ over lt? ] dip swap ] [ [ swap ] dip ] [ ] ifte "
            "[ [ add ] dip swap lt? ] "
            "[ drop drop drop 1 ] [ drop drop drop 0 ] ifte",
        ),
        sample=_three_positive_scalars,
        reference=lambda a, b, c: int(max(a, b, c) < a + b + c - max(a, b, c)),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="distance_to_nearer_choice",
        summary="distance from a target to the nearer of two choices",
        witnesses=(
            "[ [ over swap sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] "
            "dip swap [ over swap sub [ dup 0 lt? ] "
            "[ 0 swap sub ] [ ] ifte ] dip lt? ] "
            "[ swap drop sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] "
            "[ drop sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] ifte",
        ),
        sample=_triple_of(SMALL),
        reference=lambda target, first, second: min(
            abs(target - first), abs(target - second)
        ),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="farther_choice_from_target",
        summary="the farther of two choices from a target, choosing the second on ties",
        witnesses=(
            "[ [ over swap sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] "
            "dip swap [ over swap sub [ dup 0 lt? ] "
            "[ 0 swap sub ] [ ] ifte ] dip lt? ] "
            "[ drop swap drop ] [ swap drop swap drop ] ifte",
        ),
        sample=_triple_of(SMALL),
        reference=lambda target, first, second: (
            first if abs(target - second) < abs(target - first) else second
        ),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="distinct_count_three",
        summary="how many distinct values occur among three numbers",
        witnesses=(
            "[ [ eq? ] dip swap ] "
            "[ [ over eq? ] [ drop drop drop 1 ] "
            "[ drop drop drop 1 1 add ] ifte ] "
            "[ [ over eq? ] [ drop drop drop 1 1 add ] "
            "[ [ swap drop eq? ] [ drop drop drop 1 1 add ] "
            "[ drop drop drop 1 1 add 1 add ] ifte ] ifte ] ifte",
        ),
        sample=_triple_of(SMALL),
        reference=lambda a, b, c: len({a, b, c}),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="positive_majority_three",
        summary="one when at least two of three numbers are positive",
        witnesses=(
            "nil swap cons swap cons swap cons "
            "0 [ [ 0 swap lt? ] [ drop 1 add ] [ drop ] ifte ] fold "
            "1 swap lt? [ ] [ drop 1 ] [ drop 0 ] ifte",
        ),
        sample=_triple_of(SMALL),
        reference=lambda a, b, c: int(sum(x > 0 for x in (a, b, c)) >= 2),
        tags=frozenset({"scalar", "fold", "branch"}),
    ),
    # -- further widened configurations -----------------------------------
    Behaviour(
        name="concatenate_shorter_first",
        summary="join two lists with the shorter one first, preserving order on ties",
        witnesses=(
            "[ 0 [ drop 1 add ] fold swap 0 [ drop 1 add ] fold lt? ] "
            "[ swap cat ] [ cat ] ifte",
        ),
        sample=_two_unequal_lists,
        reference=lambda xs, ys: [*ys, *xs] if len(ys) < len(xs) else [*xs, *ys],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="combined_value_span",
        summary="range from smallest to largest across both lists",
        witnesses=(
            "cat dup uncons "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold "
            "swap uncons "
            "[ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold sub",
        ),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: max([*xs, *ys]) - min([*xs, *ys]),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="symmetric_value_difference",
        summary="values appearing only in one list, with first-list values first",
        witnesses=(
            "over over "
            "[ swap [ eq? ] swap cons filter null? ] swap cons filter "
            "[ swap [ swap [ eq? ] swap cons filter null? ] "
            "swap cons filter ] dip swap cat",
        ),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: [
            *[x for x in xs if x not in ys],
            *[y for y in ys if y not in xs],
        ],
        tags=frozenset({"filter", "closure"}),
    ),
    Behaviour(
        name="stable_distinct_shared_values",
        summary="distinct values shared by both lists, ordered by first appearance",
        witnesses=(
            "[ swap [ eq? ] swap cons filter null? "
            "[ ] [ drop false ] [ drop true ] ifte ] swap cons filter "
            "nil [ [ [ eq? ] swap cons filter null? ] "
            "[ cons ] [ drop ] ifte ] fold nil [ cons ] fold",
        ),
        sample=_two_lists(SMALL, lo=1, hi=7),
        reference=lambda xs, ys: [
            x for index, x in enumerate(xs) if x in ys and x not in xs[:index]
        ],
        tags=frozenset({"filter", "fold", "closure", "branch"}),
    ),
    Behaviour(
        name="closer_to_lower_bound_count",
        summary="count values at least as close to the lower as the upper bound",
        witnesses=(
            "[ [ [ over swap sub dup mul ] dip swap "
            "[ over swap sub dup mul ] dip lt? ] "
            "[ drop drop drop ] [ drop drop drop 1 add ] ifte ] "
            "swap cons swap cons 0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(abs(x - lo) <= abs(x - hi) for x in xs),
        tags=frozenset({"fold", "closure", "branch"}),
    ),
    Behaviour(
        name="bounds_cover_list_flag",
        summary="one when every list value lies inside the inclusive bounds",
        witnesses=(
            "[ [ dup ] dip ] dip "
            "[ [ uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold ] "
            "dip ] dip "
            "[ [ [ uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold ] "
            "dip ] dip ] dip "
            "[ drop lt? ] [ drop drop drop drop 0 ] "
            "[ [ [ drop drop ] dip swap lt? ] [ drop drop drop drop 0 ] "
            "[ drop drop drop drop 1 ] ifte ] ifte",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: int(all(lo <= x <= hi for x in xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="interval_overlap_length",
        summary="length of the overlap between two closed intervals",
        witnesses=(
            "[ [ drop swap drop ] dip lt? ] "
            "[ drop swap ] [ [ swap drop ] dip ] ifte "
            "[ [ over lt? ] dip swap ] "
            "[ [ drop ] dip ] [ [ swap drop ] dip ] ifte "
            "swap sub [ dup 0 lt? ] [ drop 0 ] [ ] ifte",
        ),
        sample=_two_intervals,
        reference=lambda lo1, hi1, lo2, hi2: max(0, min(hi1, hi2) - max(lo1, lo2)),
        tags=frozenset({"scalar", "branch"}),
    ),
    Behaviour(
        name="gap_between_intervals",
        summary="distance separating two intervals, or zero when they overlap",
        witnesses=(
            "[ [ drop swap drop ] dip lt? ] "
            "[ drop swap ] [ [ swap drop ] dip ] ifte "
            "[ [ over lt? ] dip swap ] "
            "[ [ drop ] dip ] [ [ swap drop ] dip ] ifte "
            "sub [ dup 0 lt? ] [ drop 0 ] [ ] ifte",
        ),
        sample=_two_intervals,
        reference=lambda lo1, hi1, lo2, hi2: max(0, max(lo1, lo2) - min(hi1, hi2)),
        tags=frozenset({"scalar", "branch"}),
    ),
    # -- stage 2: flat pair encodings and deeper stateful folds -------------
    Behaviour(
        name="pair_product_total",
        summary="sum the product within each pair in a flat pair list",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop mul add 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: sum(x * y for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="ascending_pair_count",
        summary="count pairs whose first value is smaller than their second",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ over over lt? ] [ drop drop 1 add ] "
            "[ drop drop ] ifte 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: sum(x < y for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="zero_straddling_pair_count",
        summary="count pairs with one negative and one positive endpoint",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ over over mul 0 lt? ] [ drop drop 1 add ] "
            "[ drop drop ] ifte 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(MIXED),
        reference=lambda xs: sum(x * y < 0 for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="pair_gap_total",
        summary="sum the absolute gaps within the pairs",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte add "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: sum(abs(x - y) for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="widest_pair",
        summary="the largest absolute gap within any pair",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte "
            "[ over lt? ] [ drop ] [ swap drop ] ifte "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: max(abs(x - y) for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="pair_sums",
        summary="replace each flat pair by its sum",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop add cons 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [x + y for x, y in _pairs(xs)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="pair_floor_averages",
        summary="replace each flat pair by its rounded-down average",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop add 1 1 add div cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [(x + y) // 2 for x, y in _pairs(xs)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="sorted_within_pairs",
        summary="sort the two values within every flat pair",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ over over lt? ] [ [ cons ] dip cons ] "
            "[ swap [ cons ] dip cons ] ifte "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [z for pair in _pairs(xs) for z in sorted(pair)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="swap_coordinates",
        summary="swap the two coordinates of every point in a flat point list",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop swap [ cons ] dip cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [z for x, y in _pairs(xs) for z in (y, x)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="reverse_point_order",
        summary="reverse the points while preserving each coordinate pair",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop swap [ cons ] dip cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [z for pair in reversed(_pairs(xs)) for z in pair],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="squared_point_norms",
        summary="compute squared distance from the origin for every flat point",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ dup mul ] dip dup mul add cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [x * x + y * y for x, y in _pairs(xs)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="furthest_squared_radius",
        summary="the greatest squared distance from the origin among flat points",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ dup mul ] dip dup mul add "
            "[ over lt? ] [ drop ] [ swap drop ] ifte "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: max(x * x + y * y for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="positive_quadrant_count",
        summary="count flat points whose two coordinates are positive",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ [ over 0 swap lt? ] [ 0 swap lt? ] "
            "[ drop drop false ] ifte ] [ drop drop 1 add ] "
            "[ drop drop ] ifte 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(MIXED),
        reference=lambda xs: sum(x > 0 and y > 0 for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="axis_point_count",
        summary="count flat points lying on either coordinate axis",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ [ over 0 eq? ] [ drop drop true ] [ 0 eq? ] ifte ] "
            "[ drop drop 1 add ] [ drop drop ] ifte "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(SPARSE),
        reference=lambda xs: sum(x == 0 or y == 0 for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="points_above_parabola",
        summary="count flat points lying strictly above y equals x squared",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ over dup mul swap lt? ] [ drop drop 1 add ] "
            "[ drop drop ] ifte 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: sum(y > x * x for x, y in _pairs(xs)),
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="manhattan_point_radii",
        summary="compute Manhattan distance from the origin for every flat point",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip "
            "[ dup 0 lt? ] [ 0 swap sub ] [ ] ifte add cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [abs(x) + abs(y) for x, y in _pairs(xs)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
    ),
    Behaviour(
        name="keep_ascending_pairs",
        summary="keep only flat pairs whose first value is smaller",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop [ over over lt? ] [ [ cons ] dip cons ] "
            "[ drop drop ] ifte 0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [z for x, y in _pairs(xs) if x < y for z in (x, y)],
        tags=frozenset({"fold", "branch", "flat-pairs"}),
        accept=_shrinks_flat,
    ),
    Behaviour(
        name="nearest_point_to_origin",
        summary="the flat point nearest the origin, keeping the first on ties",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop nil swap cons swap cons cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold uncons "
            "[ [ [ uncons swap uncons swap drop dup mul swap dup mul add ] dip "
            "uncons swap uncons swap drop dup mul swap dup mul add swap lt? ] "
            "[ swap drop ] [ drop ] ifte ] fold",
        ),
        sample=_flat_pairs(lo=1),
        reference=lambda xs: list(min(_pairs(xs), key=lambda p: p[0] ** 2 + p[1] ** 2)),
        tags=frozenset({"fold", "branch", "flat-pairs", "composed"}),
    ),
    Behaviour(
        name="point_centroid_floor",
        summary="the coordinate-wise rounded-down centroid of flat points",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop nil swap cons swap cons cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold "
            "dup dup [ uncons swap drop ] map 0 [ add ] fold "
            "swap [ uncons drop uncons swap drop ] map 0 [ add ] fold "
            "[ [ 0 [ drop 1 add ] fold ] dip ] dip "
            "[ over ] dip swap div [ swap div ] dip "
            "quote swap cons",
        ),
        sample=_flat_pairs(lo=1),
        reference=lambda xs: [
            sum(x for x, _ in _pairs(xs)) // len(_pairs(xs)),
            sum(y for _, y in _pairs(xs)) // len(_pairs(xs)),
        ],
        tags=frozenset({"map", "fold", "branch", "flat-pairs", "composed"}),
    ),
    Behaviour(
        name="polyline_manhattan_length",
        summary="the Manhattan length of a polyline encoded as flat points",
        witnesses=(
            "nil 0 cons 0 cons nil cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop [ drop ] dip 1 nil swap cons swap cons swap cons ] "
            "[ swap drop nil swap cons swap cons cons "
            "0 0 nil swap cons swap cons swap cons ] ifte ] "
            "fold uncons swap drop nil [ cons ] fold dup "
            "[ uncons swap drop ] map uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip swap "
            "[ dup [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip swap ] "
            "dip add nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop swap "
            "[ uncons drop uncons swap drop ] map uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip swap "
            "[ dup [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip swap ] "
            "dip add nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop add",
        ),
        sample=_flat_pairs(lo=1),
        reference=lambda xs: sum(
            abs(x1 - x2) + abs(y1 - y2) for (x1, y1), (x2, y2) in pairwise(_pairs(xs))
        ),
        tags=frozenset({"map", "fold", "branch", "flat-pairs", "composed"}),
    ),
    Behaviour(
        name="longest_strictly_increasing_run",
        summary="the longest contiguous strictly increasing run",
        witnesses=(
            "uncons nil swap cons "
            "[ dup [ swap uncons swap [ sub ] dip swap cons ] dip cons ] fold "
            "uncons drop nil [ cons ] fold "
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ 0 swap lt? ] "
            "[ drop swap 1 add dup "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip swap "
            "nil swap cons swap cons ] "
            "[ drop [ drop ] dip 0 swap nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop 1 add",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=_longest_strict_increase,
        tags=frozenset({"fold", "branch", "composed"}),
    ),
    Behaviour(
        name="best_single_trade_profit",
        summary="the largest profit from one earlier buy and one later sale",
        witnesses=(
            "uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip swap "
            "[ dup [ [ over lt? ] [ swap drop ] [ drop ] ifte ] dip ] dip "
            "swap [ over ] dip swap sub "
            "[ over lt? ] [ drop ] [ swap drop ] ifte "
            "nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: max(
            [0, *(xs[j] - xs[i] for i in range(len(xs)) for j in range(i + 1, len(xs)))]
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="first_maximum_index",
        summary="the zero-based position of the first maximum",
        witnesses=(
            "uncons nil 0 cons 0 cons swap cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ [ drop drop ] dip lt? ] "
            "[ swap 1 add dup nil swap cons swap cons swap cons "
            "[ drop drop ] dip ] "
            "[ drop 1 add nil swap cons swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: xs.index(max(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="record_extreme_count",
        summary="count values setting a new all-time high or low, including the first",
        witnesses=(
            "uncons dup nil 1 cons swap cons swap cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ [ drop drop ] dip swap lt? ] "
            "[ swap 1 add [ swap ] dip nil swap cons swap cons swap cons "
            "[ drop ] dip ] "
            "[ [ [ drop ] dip [ swap drop ] dip lt? ] "
            "[ swap 1 add [ swap drop ] dip "
            "nil swap cons swap cons swap cons ] "
            "[ drop nil swap cons swap cons swap cons ] ifte ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=_record_extreme_count,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="record_high_sum",
        summary="sum the values that establish a new strict record high",
        witnesses=(
            "uncons dup nil swap cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ [ drop ] dip lt? ] "
            "[ swap over add nil swap cons swap cons [ drop ] dip ] "
            "[ drop nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: (
            xs[0] + sum(xs[i] for i in range(1, len(xs)) if xs[i] > max(xs[:i]))
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="zero_balance_prefix_count",
        summary="count prefixes whose running sum returns to zero",
        witnesses=(
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip swap [ add ] dip "
            "[ over 0 eq? ] [ 1 add nil swap cons swap cons ] "
            "[ nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SPARSE),
        reference=lambda xs: sum(sum(xs[:end]) == 0 for end in range(1, len(xs) + 1)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="positive_balance_prefix_count",
        summary="count prefixes whose running sum is positive",
        witnesses=(
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip swap [ add ] dip "
            "[ drop 0 swap lt? ] [ 1 add nil swap cons swap cons ] "
            "[ nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: sum(sum(xs[:end]) > 0 for end in range(1, len(xs) + 1)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="largest_absolute_prefix_sum",
        summary="the greatest magnitude reached by any prefix sum",
        witnesses=(
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip swap [ add ] dip "
            "over [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte "
            "[ over lt? ] [ drop ] [ swap drop ] ifte "
            "nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: max(abs(sum(xs[:end])) for end in range(1, len(xs) + 1)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="sum_after_last_zero",
        summary="sum the suffix strictly after the final zero",
        witnesses=("0 [ [ dup 0 eq? ] [ drop drop 0 ] [ add ] ifte ] fold",),
        sample=_list_of(SPARSE),
        reference=lambda xs: (
            sum(xs[len(xs) - 1 - xs[::-1].index(0) + 1 :]) if 0 in xs else sum(xs)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="largest_zero_delimited_sum",
        summary="the largest running sum within blocks separated by zeros, allowing zero",
        witnesses=(
            "nil 0 cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ dup 0 eq? ] "
            "[ drop swap drop 0 swap nil swap cons swap cons ] "
            "[ swap [ add ] dip over [ over lt? ] [ drop ] "
            "[ swap drop ] ifte nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SPARSE),
        reference=_largest_zero_block_sum,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="running_positive_streaks",
        summary="the current positive-element streak length at every position",
        witnesses=(
            "nil nil cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ 0 swap lt? ] "
            "[ drop swap 1 add dup [ swap ] dip cons "
            "nil swap cons swap cons ] "
            "[ drop swap drop 0 dup [ swap ] dip cons "
            "nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_list_of(MIXED),
        reference=lambda xs: [
            next(
                (
                    length
                    for length in range(i + 1, 0, -1)
                    if all(x > 0 for x in xs[i + 1 - length : i + 1])
                ),
                0,
            )
            for i in range(len(xs))
        ],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="prefix_sum_signs",
        summary="encode each prefix sum as negative one, zero, or positive one",
        witnesses=(
            "nil nil cons 0 cons "
            "[ [ uncons swap uncons swap drop ] dip swap [ add ] dip over "
            "[ dup 0 lt? ] [ drop 0 1 sub ] "
            "[ [ dup 0 eq? ] [ drop 0 ] [ drop 1 ] ifte ] ifte "
            "cons nil swap cons swap cons ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: [
            (sum(xs[:end]) > 0) - (sum(xs[:end]) < 0) for end in range(1, len(xs) + 1)
        ],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="running_ranges",
        summary="the maximum minus the minimum in every prefix",
        witnesses=(
            "uncons dup nil nil swap cons swap cons swap cons "
            "[ over uncons drop uncons drop uncons swap drop nil swap cons "
            "[ over over swap uncons drop uncons swap drop swap "
            "[ over lt? ] [ drop ] [ swap drop ] ifte ] dip swap cons "
            "[ over over swap uncons swap drop swap "
            "[ over lt? ] [ swap drop ] [ drop ] ifte ] dip swap cons "
            "[ drop drop ] dip "
            "dup uncons swap uncons swap uncons swap drop "
            "[ swap sub ] dip swap cons "
            "[ uncons swap uncons swap uncons swap drop ] dip swap drop "
            "nil swap cons swap cons swap cons ] fold "
            "uncons drop uncons drop uncons swap drop nil [ cons ] fold 0 cons",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=_running_ranges,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="adjacent_floor_averages",
        summary="the rounded-down average of each neighboring pair",
        witnesses=(
            "uncons nil swap cons "
            "[ dup [ swap uncons swap [ add 1 1 add div ] dip swap cons ] "
            "dip cons ] fold uncons drop nil [ cons ] fold",
        ),
        sample=_list_of(SMALL, lo=2),
        reference=lambda xs: [(x + y) // 2 for x, y in pairwise(xs)],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="run_length_encoding",
        summary="encode adjacent runs as alternating value and count",
        witnesses=(
            "nil [ [ over null? ] [ swap drop nil swap cons 1 cons ] "
            "[ [ over uncons drop uncons swap drop eq? ] "
            "[ swap uncons 1 add cons swap drop ] "
            "[ cons 1 cons ] ifte ] ifte ] fold nil [ cons ] fold",
        ),
        sample=_list_of(SPARSE),
        reference=_run_length_encode,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="unimodal_indicator",
        summary="one if the list never rises after it first falls, otherwise zero",
        witnesses=(
            "uncons nil 1 cons 0 cons swap cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop 0 eq? ] "
            "[ swap drop swap 0 nil swap cons swap cons swap cons "
            "[ drop ] dip ] "
            "[ [ drop drop 0 eq? ] "
            "[ [ swap drop swap drop swap lt? ] "
            "[ [ drop drop drop ] dip 1 1 nil swap cons swap cons swap cons ] "
            "[ [ drop drop drop ] dip 0 1 nil swap cons swap cons swap cons ] "
            "ifte ] "
            "[ [ swap drop swap drop lt? ] "
            "[ [ drop drop drop ] dip 1 0 nil swap cons swap cons swap cons ] "
            "[ [ drop drop drop ] dip 1 1 nil swap cons swap cons swap cons ] "
            "ifte ] ifte ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=_unimodal_flag,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="running_peak_shortfall",
        summary="sum every value's shortfall from the highest value seen so far",
        witnesses=(
            "uncons nil 0 cons swap cons "
            "[ [ uncons swap uncons swap drop ] dip [ [ drop ] dip lt? ] "
            "[ swap nil swap cons swap cons [ drop ] dip ] "
            "[ [ over ] dip sub add nil swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=lambda xs: sum(max(xs[: i + 1]) - x for i, x in enumerate(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="prefix_record_count",
        summary="count strict new highs reached by the running sum",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip add "
            "[ [ drop ] dip lt? ] "
            "[ swap 1 add over nil swap cons swap cons swap cons "
            "[ drop ] dip ] "
            "[ nil swap cons swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SMALL),
        reference=lambda xs: sum(
            sum(xs[:end]) > max([0, *(sum(xs[:prior]) for prior in range(1, end))])
            for end in range(1, len(xs) + 1)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="maximum_running_mean",
        summary="the greatest rounded-down mean among all nonempty prefixes",
        witnesses=(
            "uncons dup nil swap cons 1 cons swap cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "add [ 1 add ] dip over over swap div "
            "[ [ drop drop ] dip lt? ] "
            "[ swap [ swap ] dip nil swap cons swap cons swap cons "
            "[ drop ] dip ] "
            "[ drop nil swap cons swap cons swap cons ] ifte ] fold "
            "uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1),
        reference=_max_running_mean,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="longest_positive_balance_streak",
        summary="the longest run of prefixes whose running sum stays positive",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip add "
            "[ 0 over lt? ] "
            "[ [ swap 1 add dup [ [ over lt? ] [ drop ] [ swap drop ] ifte ] "
            "dip swap ] dip nil swap cons swap cons swap cons ] "
            "[ [ swap drop 0 swap ] dip nil swap cons swap cons swap cons ] "
            "ifte ] fold uncons drop uncons swap drop",
        ),
        sample=_list_of(MIXED),
        reference=_longest_positive_balance_streak,
        tags=frozenset({"fold", "branch"}),
    ),
    # -- stage 2: frontier retries and cross-batch-distinct behaviours -----
    Behaviour(
        name="take_prefix_count",
        summary="take the requested number of initial elements",
        witnesses=(
            "nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ cons [ 1 sub ] dip quote swap cons ] "
            "[ drop quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_list_and_count,
        reference=lambda xs, count: xs[:count],
        tags=frozenset({"fold", "branch", "closure"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="drop_prefix_count",
        summary="drop the requested number of initial elements",
        witnesses=(
            "nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ drop [ 1 sub ] dip quote swap cons ] "
            "[ cons quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_list_and_count,
        reference=lambda xs, count: xs[count:],
        tags=frozenset({"fold", "branch", "closure"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="kth_order_statistic",
        summary="the value at a supplied zero-based rank in sorted order",
        witnesses=(
            "[ nil [ nil quote swap cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ [ drop ] dip lt? ] "
            "[ [ [ dup ] dip ] dip [ swap ] dip [ cons ] dip cons "
            "[ [ drop ] dip ] dip quote swap cons 1 cons ] "
            "[ cons quote swap cons swap cons ] ifte ] "
            "[ cons quote swap cons swap cons ] ifte ] fold "
            "uncons swap uncons swap uncons swap drop "
            "[ drop drop 0 eq? ] [ swap cons swap drop ] "
            "[ swap drop swap drop ] ifte nil [ cons ] fold ] fold ] dip "
            "nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ drop [ 1 sub ] dip quote swap cons ] "
            "[ cons quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold uncons swap drop",
        ),
        sample=_list_and_valid_index,
        reference=lambda xs, index: sorted(xs)[index],
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="sort_ascending",
        summary="sort the numbers into ascending order",
        witnesses=(
            "nil [ nil quote swap cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ [ drop ] dip lt? ] "
            "[ [ [ dup ] dip ] dip [ swap ] dip [ cons ] dip cons "
            "[ [ drop ] dip ] dip quote swap cons 1 cons ] "
            "[ cons quote swap cons swap cons ] ifte ] "
            "[ cons quote swap cons swap cons ] ifte ] fold "
            "uncons swap uncons swap uncons swap drop "
            "[ drop drop 0 eq? ] [ swap cons swap drop ] "
            "[ swap drop swap drop ] ifte nil [ cons ] fold ] fold",
        ),
        sample=_list_of(SMALL, lo=1, hi=7),
        reference=sorted,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="median_floor",
        summary="the median, averaging the middle pair and rounding down",
        witnesses=(
            "dup nil [ nil quote swap cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ [ drop ] dip lt? ] "
            "[ [ [ dup ] dip ] dip [ swap ] dip [ cons ] dip cons "
            "[ [ drop ] dip ] dip quote swap cons 1 cons ] "
            "[ cons quote swap cons swap cons ] ifte ] "
            "[ cons quote swap cons swap cons ] ifte ] fold "
            "uncons swap uncons swap uncons swap drop "
            "[ drop drop 0 eq? ] [ swap cons swap drop ] "
            "[ swap drop swap drop ] ifte nil [ cons ] fold ] fold "
            "swap 0 [ drop 1 add ] fold over over 1 1 add div "
            "nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ drop [ 1 sub ] dip quote swap cons ] "
            "[ cons quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold "
            "uncons swap drop "
            "[ 1 sub 1 1 add div nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ drop [ 1 sub ] dip quote swap cons ] "
            "[ cons quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold "
            "uncons swap drop ] dip add 1 1 add div",
        ),
        sample=_list_of(SMALL, lo=1, hi=7),
        reference=lambda xs: (
            (sorted(xs)[(len(xs) - 1) // 2] + sorted(xs)[len(xs) // 2]) // 2
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="stable_sort_flat_pairs",
        summary="stably sort flat key-value pairs by key",
        witnesses=(
            "nil quote 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ swap drop swap drop ] dip swap quote swap cons 1 cons ] "
            "[ [ swap ] dip quote swap cons cons swap drop "
            "quote 0 cons 0 cons ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop nil [ cons ] fold "
            "nil [ nil quote swap cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ uncons swap drop [ [ uncons swap drop ] dip ] dip "
            "[ drop ] dip lt? ] "
            "[ [ [ dup ] dip ] dip [ swap ] dip [ cons ] dip cons "
            "[ [ drop ] dip ] dip quote swap cons 1 cons ] "
            "[ cons quote swap cons swap cons ] ifte ] "
            "[ cons quote swap cons swap cons ] ifte ] fold "
            "uncons swap uncons swap uncons swap drop "
            "[ drop drop 0 eq? ] [ swap cons swap drop ] "
            "[ swap drop swap drop ] ifte nil [ cons ] fold ] fold "
            "nil [ cat ] fold",
        ),
        sample=_observable_stable_pairs,
        reference=_stable_sort_pairs,
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="longest_nondecreasing_run",
        summary="the longest consecutive run that never decreases",
        witnesses=(
            "uncons quote 1 cons 1 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ swap lt? ] "
            "[ [ drop drop ] dip quote 1 cons swap cons ] "
            "[ [ drop ] dip [ 1 add dup ] dip "
            "[ [ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip ] dip "
            "quote swap cons swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1, hi=8),
        reference=lambda xs: max(
            end - start
            for start in range(len(xs))
            for end in range(start + 1, len(xs) + 1)
            if all(a <= b for a, b in pairwise(xs[start:end]))
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="adjacent_sign_change_count",
        summary="count adjacent pairs whose nonzero signs differ",
        witnesses=(
            "uncons quote 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ mul 0 lt? ] "
            "[ [ drop ] dip [ 1 add ] dip quote swap cons ] "
            "[ [ drop ] dip quote swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_list_of(MIXED, lo=1, hi=8),
        reference=lambda xs: sum(a * b < 0 for a, b in pairwise(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="record_low_count",
        summary="how many elements set a new all-time low, counting the first",
        witnesses=(
            "uncons quote 1 cons "
            "[ [ uncons swap uncons swap drop ] dip [ swap lt? ] "
            "[ [ drop ] dip [ 1 add ] dip quote swap cons ] "
            "[ drop quote swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1, hi=8),
        reference=lambda xs: (
            1 + sum(xs[index] < min(xs[:index]) for index in range(1, len(xs)))
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="rotate_left_count",
        summary="rotate left by a supplied number of positions",
        witnesses=(
            "over over nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ drop [ 1 sub ] dip quote swap cons ] "
            "[ cons quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold "
            "[ nil quote swap cons "
            "[ [ uncons swap uncons swap drop ] dip "
            "[ drop drop 0 swap lt? ] "
            "[ cons [ 1 sub ] dip quote swap cons ] "
            "[ drop quote swap cons ] ifte ] fold "
            "uncons drop uncons swap drop nil [ cons ] fold ] dip swap cat",
        ),
        sample=_list_and_rotation,
        reference=lambda xs, count: [*xs[count:], *xs[:count]],
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="positive_step_total",
        summary="sum only the upward changes between adjacent values",
        witnesses=(
            "uncons quote 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ lt? ] "
            "[ dup [ swap sub ] dip [ add ] dip quote swap cons ] "
            "[ [ drop ] dip quote swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1, hi=8),
        reference=lambda xs: sum(max(0, right - left) for left, right in pairwise(xs)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="running_mean_floor",
        summary="the rounded-down mean of every prefix",
        witnesses=(
            "nil quote 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ swap ] dip [ [ swap ] dip ] dip swap [ add ] dip "
            "[ swap ] dip swap [ 1 add ] dip [ over over ] dip "
            "swap [ swap ] dip div cons quote swap cons swap cons ] fold "
            "uncons drop uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_list_of(SMALL, lo=1, hi=8),
        reference=lambda xs: [sum(xs[:end]) // end for end in range(1, len(xs) + 1)],
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="inversion_count",
        summary="count out-of-order index pairs",
        witnesses=(
            "nil quote 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ dup ] dip dup "
            "[ swap ] dip [ swap lt? ] swap cons filter "
            "0 [ drop 1 add ] fold swap [ swap [ add ] dip ] dip "
            "cons quote swap cons ] fold uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=1, hi=7),
        reference=lambda xs: sum(
            xs[left] > xs[right]
            for left in range(len(xs))
            for right in range(left + 1, len(xs))
        ),
        tags=frozenset({"fold", "filter", "closure"}),
    ),
    Behaviour(
        name="nondecreasing_indicator",
        summary="one if the list is in nondecreasing order, otherwise zero",
        witnesses=(
            "uncons quote 1 cons "
            "[ [ uncons swap uncons swap drop ] dip [ swap lt? ] "
            "[ [ drop drop 0 ] dip quote swap cons ] "
            "[ [ drop ] dip quote swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_maybe_sorted,
        reference=lambda xs: int(all(left <= right for left, right in pairwise(xs))),
        tags=frozenset({"fold", "branch"}),
        accept=lambda _args, results: len(set(results)) > 1,
    ),
    Behaviour(
        name="strict_turn_count",
        summary="count strict interior peaks and valleys",
        witnesses=(
            "uncons swap uncons swap [ quote swap cons 0 cons ] dip swap "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop lt? ] "
            "[ [ swap lt? ] "
            "[ [ [ drop ] dip ] dip [ [ 1 add ] dip ] dip "
            "quote swap cons swap cons ] "
            "[ [ [ drop ] dip ] dip quote swap cons swap cons ] ifte ] "
            "[ [ drop swap lt? ] "
            "[ [ lt? ] "
            "[ [ [ drop ] dip ] dip [ [ 1 add ] dip ] dip "
            "quote swap cons swap cons ] "
            "[ [ [ drop ] dip ] dip quote swap cons swap cons ] ifte ] "
            "[ [ [ drop ] dip ] dip quote swap cons swap cons ] ifte ] ifte ] fold "
            "uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=2, hi=8),
        reference=lambda xs: sum(
            (xs[index - 1] < xs[index] > xs[index + 1])
            or (xs[index - 1] > xs[index] < xs[index + 1])
            for index in range(1, len(xs) - 1)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="distinct_value_count",
        summary="how many different values occur",
        witnesses=(
            "nil [ [ [ eq? ] swap cons filter null? ] "
            "[ cons ] [ drop ] ifte ] fold nil [ cons ] fold "
            "0 [ drop 1 add ] fold",
        ),
        sample=_list_of(SPARSE, lo=2, hi=8),
        reference=lambda xs: len(set(xs)),
        tags=frozenset({"fold", "filter", "branch"}),
    ),
    Behaviour(
        name="mode_first_tie",
        summary="the most frequent value, choosing its first occurrence on ties",
        witnesses=(
            "dup nil 0 cons 0 cons swap "
            "[ [ [ uncons swap uncons swap drop ] dip ] dip "
            "[ dup ] dip swap [ eq? ] swap cons filter "
            "0 [ drop 1 add ] fold "
            "[ [ drop ] dip [ drop ] dip lt? ] "
            "[ swap [ [ drop drop ] dip ] dip quote swap cons ] "
            "[ drop drop quote swap cons ] ifte ] swap cons fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SPARSE, lo=1, hi=7),
        reference=lambda xs: max(dict.fromkeys(xs), key=xs.count),
        tags=frozenset({"fold", "filter", "closure", "branch"}),
    ),
    Behaviour(
        name="evaluate_polynomial_horner",
        summary="evaluate descending-order polynomial coefficients at a supplied value",
        witnesses=("[ [ swap ] dip mul add ] swap cons 0 swap fold",),
        sample=_list_and_int(SMALL, SMALL, lo=1, hi=7),
        reference=_evaluate_polynomial,
        tags=frozenset({"fold", "closure"}),
    ),
    Behaviour(
        name="aligned_absolute_gaps",
        summary="the absolute gap at each aligned position",
        witnesses=(
            "nil cons [ swap uncons swap uncons swap "
            "[ [ swap ] dip sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte cons ] dip "
            "swap cons ] fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_equal_length_lists,
        reference=lambda xs, ys: [abs(x - y) for x, y in zip(xs, ys, strict=True)],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="cartesian_pair_sums",
        summary="all sums formed by choosing one value from each list",
        witnesses=("[ swap [ add ] swap cons map cat ] swap cons nil swap fold",),
        sample=_two_lists(SMALL, lo=1, hi=6),
        reference=lambda xs, ys: [x + y for x in xs for y in ys],
        tags=frozenset({"map", "fold", "closure"}),
    ),
    Behaviour(
        name="maximum_aligned_gap",
        summary="the largest absolute gap at any aligned position",
        witnesses=(
            "0 cons [ swap uncons swap uncons swap "
            "[ [ swap ] dip sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte "
            "[ over lt? ] [ drop ] [ swap drop ] ifte ] dip "
            "swap cons ] fold uncons swap drop",
        ),
        sample=_equal_length_lists,
        reference=lambda xs, ys: max(abs(x - y) for x, y in zip(xs, ys, strict=True)),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="aligned_matches",
        summary="keep first-list values whose aligned partners are equal",
        witnesses=(
            "nil cons [ swap uncons swap uncons swap "
            "[ [ swap ] dip [ eq? ] [ drop cons ] [ drop drop ] ifte ] dip "
            "swap cons ] fold uncons swap drop nil [ cons ] fold",
        ),
        sample=_partly_matching_lists,
        reference=lambda xs, ys: [x for x, y in zip(xs, ys, strict=True) if x == y],
        tags=frozenset({"fold", "branch"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="minimum_cross_list_gap",
        summary="the smallest absolute gap between a value from each list",
        witnesses=(
            "[ swap [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] "
            "swap cons map cat ] swap cons nil swap fold uncons "
            "[ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold",
        ),
        sample=_two_lists(SMALL, lo=1, hi=6),
        reference=lambda xs, ys: min(abs(x - y) for x in xs for y in ys),
        tags=frozenset({"map", "fold", "closure", "branch"}),
    ),
    Behaviour(
        name="lexicographic_order",
        summary="minus one, zero, or one according to lexicographic list order",
        witnesses=(
            "quote 0 cons "
            "[ [ uncons swap uncons swap drop ] dip [ drop drop 0 eq? ] "
            "[ [ drop null? ] "
            "[ drop [ drop ] dip quote 1 cons ] "
            "[ [ uncons ] dip [ eq? ] [ drop drop quote swap cons ] "
            "[ [ swap lt? ] "
            "[ drop drop [ drop ] dip quote 0 1 sub cons ] "
            "[ drop drop [ drop ] dip quote 1 cons ] ifte ] ifte ] ifte ] "
            "[ drop quote swap cons ] ifte ] fold "
            "uncons swap uncons swap drop [ drop 0 eq? ] "
            "[ [ null? ] [ drop drop 0 ] [ drop drop 0 1 sub ] ifte ] "
            "[ drop ] ifte",
        ),
        sample=_two_lists(SMALL, lo=0, hi=6),
        reference=lambda xs, ys: -1 if xs < ys else int(xs > ys),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="merge_sorted_lists",
        summary="merge two ascending lists into one ascending list",
        witnesses=(
            "cat nil [ nil quote swap cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ [ drop ] dip lt? ] "
            "[ [ [ dup ] dip ] dip [ swap ] dip [ cons ] dip cons "
            "[ [ drop ] dip ] dip quote swap cons 1 cons ] "
            "[ cons quote swap cons swap cons ] ifte ] "
            "[ cons quote swap cons swap cons ] ifte ] fold "
            "uncons swap uncons swap uncons swap drop "
            "[ drop drop 0 eq? ] [ swap cons swap drop ] "
            "[ swap drop swap drop ] ifte nil [ cons ] fold ] fold",
        ),
        sample=_sorted_lists,
        reference=lambda xs, ys: sorted([*xs, *ys]),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="choose_largest_total_three",
        summary="the list with the largest total among three, choosing the latest on ties",
        witnesses=(
            "[ [ 0 [ add ] fold swap 0 [ add ] fold lt? ] "
            "[ drop ] [ swap drop ] ifte ] dip "
            "[ 0 [ add ] fold swap 0 [ add ] fold lt? ] "
            "[ drop ] [ swap drop ] ifte",
        ),
        sample=_three_lists,
        reference=lambda xs, ys, zs: max(
            enumerate((xs, ys, zs)), key=lambda item: (sum(item[1]), item[0])
        )[1],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="sign_histogram",
        summary="counts of negative, zero, and positive values in that order",
        witnesses=(
            "nil 0 cons 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ 0 lt? ] "
            "[ drop [ [ 1 add ] dip ] dip quote swap cons swap cons ] "
            "[ [ 0 eq? ] "
            "[ drop [ 1 add ] dip quote swap cons swap cons ] "
            "[ drop 1 add quote swap cons swap cons ] ifte ] ifte ] fold",
        ),
        sample=_list_of(SPARSE, lo=2, hi=8),
        reference=lambda xs: [
            sum(x < 0 for x in xs),
            sum(x == 0 for x in xs),
            sum(x > 0 for x in xs),
        ],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="first_repeat_index",
        summary="the first index whose value appeared earlier, or minus one",
        witnesses=(
            "nil quote 0 1 sub cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ [ drop ] dip drop 0 lt? ] "
            "[ [ [ dup ] dip [ eq? ] swap cons filter null? ] "
            "[ cons [ [ 1 add ] dip ] dip quote swap cons swap cons ] "
            "[ cons [ [ dup ] dip ] dip [ drop ] dip "
            "[ [ 1 add ] dip ] dip quote swap cons swap cons ] ifte ] "
            "[ drop [ [ 1 add ] dip ] dip quote swap cons swap cons ] ifte ] fold "
            "uncons drop uncons swap drop",
        ),
        sample=_list_of(SPARSE, lo=1, hi=8),
        reference=lambda xs: next(
            (index for index, x in enumerate(xs) if x in xs[:index]), -1
        ),
        tags=frozenset({"fold", "filter", "branch"}),
    ),
    Behaviour(
        name="flat_pair_maxima",
        summary="replace each consecutive flat pair by its larger value",
        witnesses=(
            "nil quote 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ swap drop swap drop ] dip swap quote swap cons 1 cons ] "
            "[ [ swap ] dip [ over lt? ] [ drop ] [ swap drop ] ifte cons "
            "swap drop quote 0 cons 0 cons ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [
            max(xs[index : index + 2]) for index in range(0, len(xs), 2)
        ],
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="longest_alternating_sign_run",
        summary="the longest consecutive run with strictly alternating nonzero signs",
        witnesses=(
            "uncons quote 1 cons 1 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip [ mul 0 lt? ] "
            "[ [ drop ] dip [ 1 add dup ] dip "
            "[ [ [ over lt? ] [ drop ] [ swap drop ] ifte ] dip ] dip "
            "quote swap cons swap cons ] "
            "[ [ drop drop ] dip quote 1 cons swap cons ] ifte ] fold "
            "uncons swap drop",
        ),
        sample=_list_of(MIXED, lo=1, hi=8),
        reference=lambda xs: max(
            end - start
            for start in range(len(xs))
            for end in range(start + 1, len(xs) + 1)
            if all(a * b < 0 for a, b in pairwise(xs[start:end]))
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="keep_positive_sum_pairs",
        summary="keep flat pairs whose two values have a positive sum",
        witnesses=(
            "nil quote 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ swap drop swap drop ] dip swap quote swap cons 1 cons ] "
            "[ [ [ drop ] dip add 0 swap lt? ] "
            "[ [ swap ] dip [ cons ] dip cons swap drop quote 0 cons 0 cons ] "
            "[ drop [ drop drop ] dip quote 0 cons 0 cons ] ifte ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop nil [ cons ] fold",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: [
            value
            for index in range(0, len(xs), 2)
            if xs[index] + xs[index + 1] > 0
            for value in xs[index : index + 2]
        ],
        tags=frozenset({"fold", "branch"}),
        accept=_shrinks,
    ),
    Behaviour(
        name="largest_flat_pair_sum",
        summary="the largest sum among consecutive flat pairs",
        witnesses=(
            "uncons swap uncons swap [ add ] dip swap quote 0 cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ swap drop swap drop ] dip swap quote swap cons 1 cons ] "
            "[ [ swap ] dip add [ over lt? ] [ drop ] [ swap drop ] ifte "
            "swap drop quote 0 cons 0 cons ] ifte ] fold "
            "uncons drop uncons drop uncons swap drop",
        ),
        sample=_flat_pairs(),
        reference=lambda xs: max(
            xs[index] + xs[index + 1] for index in range(0, len(xs), 2)
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="sum_inside_interval",
        summary="sum values strictly between two supplied bounds",
        witnesses=(
            "[ [ swap drop lt? ] "
            "[ [ drop swap lt? ] [ drop drop add ] [ drop drop drop ] ifte ] "
            "[ drop drop drop ] ifte ] swap cons swap cons 0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(x for x in xs if lo < x < hi),
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="count_outside_interval",
        summary="count values not strictly between two supplied bounds",
        witnesses=(
            "[ [ swap drop lt? ] "
            "[ [ drop swap lt? ] [ drop drop drop ] "
            "[ drop drop drop 1 add ] ifte ] "
            "[ drop drop drop 1 add ] ifte ] swap cons swap cons 0 swap fold",
        ),
        sample=_list_interval,
        reference=lambda xs, lo, hi: sum(not lo < x < hi for x in xs),
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="closest_pair_gap",
        summary="the smallest absolute gap between any two list elements",
        witnesses=(
            "nil [ nil quote swap cons 0 cons "
            "[ [ uncons swap uncons swap uncons swap drop ] dip "
            "[ drop drop drop 0 eq? ] "
            "[ [ [ drop ] dip lt? ] "
            "[ [ [ dup ] dip ] dip [ swap ] dip [ cons ] dip cons "
            "[ [ drop ] dip ] dip quote swap cons 1 cons ] "
            "[ cons quote swap cons swap cons ] ifte ] "
            "[ cons quote swap cons swap cons ] ifte ] fold "
            "uncons swap uncons swap uncons swap drop "
            "[ drop drop 0 eq? ] [ swap cons swap drop ] "
            "[ swap drop swap drop ] ifte nil [ cons ] fold ] fold "
            "uncons swap uncons swap "
            "[ dup [ swap sub ] dip quote swap cons ] dip swap "
            "[ [ uncons swap uncons swap drop ] dip [ swap sub swap lt? ] "
            "[ dup [ swap sub ] dip [ [ drop ] dip ] dip quote swap cons ] "
            "[ [ drop ] dip quote swap cons ] ifte ] fold uncons swap drop",
        ),
        sample=_list_of(SMALL, lo=2, hi=7),
        reference=lambda xs: min(
            abs(xs[left] - xs[right])
            for left in range(len(xs))
            for right in range(left + 1, len(xs))
        ),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="largest_inside_interval",
        summary="the largest value strictly between two supplied bounds",
        witnesses=(
            "[ [ swap drop lt? ] [ drop swap lt? ] "
            "[ drop drop drop false ] ifte ] "
            "swap cons swap cons filter uncons "
            "[ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold",
        ),
        sample=_interval_with_member,
        reference=lambda xs, lo, hi: max(x for x in xs if lo < x < hi),
        tags=frozenset({"filter", "fold", "branch", "closure"}),
    ),
    Behaviour(
        name="count_near_target",
        summary="count values within a strict supplied distance of a target",
        witnesses=(
            "[ [ [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip lt? ] "
            "[ drop drop drop 1 add ] [ drop drop drop ] ifte ] "
            "swap cons swap cons 0 swap fold",
        ),
        sample=_target_radius,
        reference=lambda xs, target, radius: sum(abs(x - target) < radius for x in xs),
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="distance_from_target_total",
        summary="sum all absolute distances from a supplied target",
        witnesses=(
            "[ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte add ] swap cons 0 swap fold",
        ),
        sample=_list_and_int(SMALL, SMALL, lo=1, hi=7),
        reference=lambda xs, target: sum(abs(x - target) for x in xs),
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="farthest_from_target_last_tie",
        summary="the farthest value from a target, choosing the last on ties",
        witnesses=(
            "[ uncons ] dip "
            "[ [ dup [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip "
            "swap [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip "
            "swap lt? ] [ drop drop ] [ drop swap drop ] ifte ] "
            "swap cons fold",
        ),
        sample=_list_and_int(SMALL, SMALL, lo=1, hi=7),
        reference=lambda xs, target: max(
            enumerate(xs), key=lambda item: (abs(item[1] - target), item[0])
        )[1],
        tags=frozenset({"fold", "branch", "closure"}),
    ),
    Behaviour(
        name="triangle_perimeter_or_zero",
        summary="the perimeter of three positive sides if they form a triangle, otherwise zero",
        witnesses=(
            "[ [ [ add ] dip swap lt? ] "
            "[ [ swap [ add ] dip swap lt? ] [ add lt? ] "
            "[ drop drop drop false ] ifte ] "
            "[ drop drop drop false ] ifte ] "
            "[ add add ] [ drop drop drop 0 ] ifte",
        ),
        sample=_triangle,
        reference=lambda a, b, c: (
            a + b + c if a + b > c and a + c > b and b + c > a else 0
        ),
        tags=frozenset({"scalar", "branch"}),
    ),
)

BY_NAME: dict[str, Behaviour] = {b.name: b for b in CATALOGUE}


# ---------------------------------------------------------------------------
# Agreement between the two statements of a behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    """One input on which the witness and the reference disagree."""

    behaviour: str
    args: Args
    expected: str
    got: str


def apply_witness(
    behaviour: Behaviour, args: Args, program: tuple[str, ...] | None = None
) -> tuple[Value, ...] | str:
    """Run a witness and require it to leave exactly one value."""
    outcome = run_program(program or behaviour.program, to_stack(args))
    if isinstance(outcome, str):
        return outcome
    stack, _fuel = outcome
    if len(stack) != 1:
        return f"left {len(stack)} values, want 1"
    return stack


def check(
    behaviour: Behaviour, args: Args, program: tuple[str, ...] | None = None
) -> Mismatch | None:
    """Whether a witness reproduces the reference on one input."""
    want = (to_value(behaviour.reference(*args)),)
    got = apply_witness(behaviour, args, program)
    if isinstance(got, str):
        return Mismatch(behaviour.name, args, format_stack(Config(want)), got)
    if tuple(value_key(v) for v in got) != tuple(value_key(v) for v in want):
        return Mismatch(
            behaviour.name, args, format_stack(Config(want)), format_stack(Config(got))
        )
    return None


def verify(behaviour: Behaviour, trials: int, seed: int) -> Mismatch | None:
    """Check witness against reference on fresh data. None means they agree.

    This is the whole guarantee. Neither statement of the behaviour is
    privileged: they were written separately, and agreement on data neither of
    them chose is the evidence that the task is labelled correctly. *Every*
    witness is checked, so an alternative program that is subtly wrong -- right
    on the shapes its author had in mind, wrong elsewhere -- cannot ride along
    on the canonical one.
    """
    rng = random.Random(seed)
    for _ in range(trials):
        args = behaviour.sample(rng)
        for program in behaviour.programs:
            if (bad := check(behaviour, args, program)) is not None:
                return bad
    return None


def _name_seed(name: str) -> int:
    """A per-behaviour seed that does not depend on `PYTHONHASHSEED`."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def verify_all(trials: int = 200, seed: int = 0) -> list[Mismatch]:
    return [
        bad
        for b in CATALOGUE
        if (bad := verify(b, trials, seed + _name_seed(b.name))) is not None
    ]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One PBE task: k example pairs, and a program that satisfies all of them.

    ``held_out_inputs`` / ``held_out_outputs`` are drawn from the same sampler
    and are *not* part of the prompt. A candidate program is scored on them,
    which is what makes overfitting impossible rather than merely unmeasured:
    a program that special-cases the shown examples has nothing to say about
    data it was never given. The shown examples are what the solver reasons
    from; the hidden ones are what it is graded on.
    """

    behaviour: str
    summary: str
    split: str
    task_hash_split: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    held_out_inputs: tuple[str, ...]
    held_out_outputs: tuple[str, ...]
    program: tuple[str, ...]
    length: int
    examples: int
    held_out: int
    fuel: int
    tags: tuple[str, ...]
    # Defaults so a corpus written under schema 1 still loads.
    skeleton: str = ""
    alternative_programs: tuple[tuple[str, ...], ...] = ()
    basis: str = BASIS
    witness_source: str = "authored"
    # The program as its author wrote it, when it was written over a macro
    # library and expanded to `program` (stack-pbe-world-v1, #120). Empty for
    # witnesses written directly in the basis.
    macro_program: str = ""
    schema_version: int = SCHEMA_VERSION


def _bucket(payload: str) -> str:
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return "train" if int(digest[:8], 16) % 100 < TRAIN_FRACTION else "eval"


def assign_split(skeleton_key: str) -> str:
    """Split on the *skeleton*, so no program shape appears in both halves.

    Three units were available and the choice matters more than it looks.
    Splitting on the *task* -- what `stack_pbe_v2` did -- leaves the same
    behaviour on both sides with different data, which asks only whether a
    solver can redo a problem it has seen. Splitting on the *behaviour* fixes
    that but still leaks: `keep_zeros` and `keep_negative` are one shape,
    `[ _ _ ] filter`, so learning either hands over the other.

    Splitting on the skeleton closes both. It is the conservative choice --
    `sum` and `product` land on the same side, which costs some eval mass --
    and it is the only one under which a held-out score means the solver met a
    shape it had not been shown. The task-level rule stays available as
    ``task_hash_split`` so the older protocol remains reproducible.
    """
    return _bucket(skeleton_key)


def assign_task_hash_split(inputs: tuple[str, ...], outputs: tuple[str, ...]) -> str:
    return _bucket(json.dumps([list(inputs), list(outputs)], sort_keys=True))


def encode_task(task: Task) -> dict[str, str]:
    """The text form a sequence model would be trained on."""
    pairs = " ; ".join(
        f"{given} -> {wanted}"
        for given, wanted in zip(task.inputs, task.outputs, strict=True)
    )
    return {"prompt": f"{pairs} =>", "completion": " " + " ".join(task.program)}


def _lengths_vary(stacks: list[tuple[Value, ...]]) -> bool:
    """Reject example sets where every list is the same length.

    Measured, not hypothetical. On `stack_pbe_v3` a `last` task drew three
    two-element lists, and BFS answered `uncons drop i` -- drop the head, then
    run the one-element tail *as a program*. That is `last` only at length two,
    and it was the corpus's single most-repeated memorisation. Examples that
    hold list length fixed cannot distinguish a program that uses the length
    from one that does not.

    Vacuously true when no input carries a list, which is the scalar
    behaviours.
    """
    shapes = {
        tuple(len(v.items) for v in stack if isinstance(v, Lst)) for stack in stacks
    }
    return len(next(iter(shapes))) == 0 or len(shapes) > 1


def _informative(
    args_list: list[Args], results: list[Result], stacks: list[tuple[Value, ...]]
) -> bool:
    """Reject example sets that a trivial program already satisfies.

    Two ways a drawn set can fail to constrain anything: every output is the
    same, so a constant fits; or every output is whatever sat at some fixed
    position of the input stack, so a projection fits. Both were filters on the
    old generator too -- they are about the *examples*, not about harvesting,
    so they survive the inversion.
    """
    keys = [value_key(to_value(r)) for r in results]
    if len(set(keys)) < 2:
        return False
    arity = len(stacks[0])
    for position in range(arity):
        if all(
            value_key(stack[position]) == key
            for stack, key in zip(stacks, keys, strict=True)
        ):
            return False
    return True


def _draw_distinct(
    behaviour: Behaviour,
    rng: random.Random,
    count: int,
    seen: set[str],
    attempts: int,
) -> list[Args] | None:
    """`count` fresh inputs, none of them already in `seen`."""
    drawn: list[Args] = []
    for _ in range(count):
        for _ in range(attempts):
            args = behaviour.sample(rng)
            key = json.dumps(args)
            if key not in seen:
                seen.add(key)
                drawn.append(args)
                break
        else:
            return None
    return drawn


@dataclass(frozen=True)
class Drawn:
    """The data for one task: what the solver sees, and what grades it."""

    shown: list[Args]
    shown_results: list[Result]
    held_out: list[Args]
    held_out_results: list[Result]


def draw_task(
    behaviour: Behaviour,
    rng: random.Random,
    examples: int,
    held_out: int = 8,
    attempts: int = 200,
) -> Drawn:
    """Draw an example set that is legal, distinct and non-trivial.

    The held-out inputs come from the same sampler and exclude the shown ones,
    so they are fresh in the only sense that matters: no program can have been
    fitted to them.
    """
    for _ in range(attempts):
        seen: set[str] = set()
        shown = _draw_distinct(behaviour, rng, examples, seen, attempts)
        if shown is None:
            continue
        stacks = [to_stack(a) for a in shown]
        if len({len(stack) for stack in stacks}) != 1:
            continue
        try:
            results = [behaviour.reference(*a) for a in shown]
        except (ValueError, ZeroDivisionError, IndexError):
            continue
        if not _informative(shown, results, stacks):
            continue
        if not _lengths_vary(stacks):
            continue
        if behaviour.accept is not None and not behaviour.accept(shown, results):
            continue
        hidden = _draw_distinct(behaviour, rng, held_out, seen, attempts)
        if hidden is None:
            continue
        hidden_results = [behaviour.reference(*a) for a in hidden]
        return Drawn(shown, results, hidden, hidden_results)
    raise MachineError(
        f"{behaviour.name}: no informative example set in {attempts} draws"
    )


def _label(
    behaviour: Behaviour, args_list: list[Args], results: list[Result]
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Run the witness on every input and check it against the reference.

    Every task is labelled by execution, never by assertion: the pair is
    re-derived here even though `verify` already checked the behaviour, because
    a task that shipped with an output its own witness does not produce would
    be a silent corpus bug.
    """
    inputs, outputs, fuel = [], [], 0
    for args, want in zip(args_list, results, strict=True):
        outcome = run_program(behaviour.program, to_stack(args))
        if isinstance(outcome, str):
            raise MachineError(f"{behaviour.name} on {args}: {outcome}")
        stack, spent = outcome
        if len(stack) != 1 or value_key(stack[0]) != value_key(to_value(want)):
            raise MachineError(
                f"{behaviour.name} on {args}: witness and reference disagree"
            )
        inputs.append(format_stack(Config(to_stack(args))))
        outputs.append(format_stack(Config(stack)))
        fuel += spent
    return tuple(inputs), tuple(outputs), fuel


def build_task(behaviour: Behaviour, drawn: Drawn) -> Task:
    """Make the record, re-running the witness on every example to label it."""
    encoded_in, encoded_out, fuel = _label(behaviour, drawn.shown, drawn.shown_results)
    hidden_in, hidden_out, _ = _label(behaviour, drawn.held_out, drawn.held_out_results)
    return Task(
        behaviour=behaviour.name,
        summary=behaviour.summary,
        split=assign_split(behaviour.skeleton),
        task_hash_split=assign_task_hash_split(encoded_in, encoded_out),
        inputs=encoded_in,
        outputs=encoded_out,
        held_out_inputs=hidden_in,
        held_out_outputs=hidden_out,
        program=behaviour.program,
        length=behaviour.length,
        examples=len(drawn.shown),
        held_out=len(drawn.held_out),
        fuel=fuel,
        tags=tuple(sorted(behaviour.tags)),
        skeleton=behaviour.skeleton,
        alternative_programs=behaviour.programs[1:],
    )


def generate(
    per_behaviour: int,
    seed: int,
    *,
    min_examples: int = 3,
    max_examples: int = 6,
    held_out: int = 8,
    catalogue: Sequence[Behaviour] = CATALOGUE,
) -> tuple[list[Task], dict]:
    """One corpus. Every task's data is drawn fresh; no seed table exists."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    stats: dict = {}
    for behaviour in catalogue:
        seen: set[str] = set()
        made = 0
        for _ in range(per_behaviour * 20):
            if made >= per_behaviour:
                break
            examples = rng.randint(min_examples, max_examples)
            drawn = draw_task(behaviour, rng, examples, held_out=held_out)
            key = json.dumps(drawn.shown, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(build_task(behaviour, drawn))
            made += 1
        stats[behaviour.name] = {
            "kept": made,
            "length": behaviour.length,
            "witnesses": len(behaviour.witnesses),
            "skeleton": behaviour.skeleton,
            "split": assign_split(behaviour.skeleton),
        }
    return tasks, stats


CORPUS_NAME = "stack_pbe_v4"
HUB_REPO_ID = "abhishekraok/wandering-light-stack-pbe-v4"


def catalogue_digest(catalogue: Sequence[Behaviour] = CATALOGUE) -> str:
    """A digest over what the catalogue *says*, so a payload cannot drift from it.

    Names, summaries and witnesses only. Changing a witness changes the labels
    in the corpus, so it must invalidate the manifest. What the catalogue
    *does* -- samplers, references and accept guards -- is
    `catalogue_source_digest`.
    """
    payload = json.dumps(
        [[b.name, b.summary, list(b.witnesses)] for b in catalogue], sort_keys=True
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _source_of(fn: object) -> str:
    """The text that decides what a behaviour computes, samples or accepts.

    ``inspect.getsource`` of a lambda is the line it sits on and of a closure
    the inner function only, so captured cells (a pool, a length range, another
    function) are appended: ``_list_of(SMALL)`` and ``_list_of(POSITIVE)``
    share one source line. Builtins have no source and are named instead.
    """
    if fn is None:
        return ""
    try:
        text = textwrap.dedent(inspect.getsource(fn))
    except (TypeError, OSError):
        module = getattr(fn, "__module__", "?")
        return f"<{module}.{getattr(fn, '__qualname__', repr(fn))}>"
    parts = [text, repr(getattr(fn, "__defaults__", None))]
    for cell in getattr(fn, "__closure__", None) or ():
        value = cell.cell_contents
        parts.append(_source_of(value) if callable(value) else repr(value))
    return "\n".join(parts)


def catalogue_source_digest(catalogue: Sequence[Behaviour] = CATALOGUE) -> str:
    """A digest over what the catalogue *does*: reference, sampler, accept guard.

    Those three decide whether a found program is genuine or an impostor, and
    a sampler edit changes every task of a behaviour without touching a
    witness (aligned-matches-sampler, #141), so `catalogue_digest` alone lets
    a corpus carry the digest of a catalogue it was not generated by (#142).
    This is source text, not behaviour: a rename or a reformat moves it, and
    helpers a reference calls are outside it.
    """
    payload = json.dumps(
        [
            [
                b.name,
                _source_of(b.reference),
                _source_of(b.sample),
                _source_of(b.accept),
            ]
            for b in catalogue
        ],
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_corpus(
    tasks: Sequence[Task],
    corpus_dir: str | Path,
    *,
    config: dict,
    name: str = CORPUS_NAME,
    catalogue: Sequence[Behaviour] = CATALOGUE,
) -> Path:
    """Write split payloads plus the manifest `corpus_hub` publishes and verifies.

    Payloads do not go in git. `wandering_light/corpus_hub.py` gives the reason:
    a generated corpus is a derived artifact, every regeneration is a fresh
    gzip, and gzips do not delta-compress, so committing them puts every
    superseded version in history forever. The manifest is what stays -- it
    pins the digests a later `fetch_corpus` is checked against, which is the
    whole point of the split.
    """
    root = Path(corpus_dir)
    root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict] = {}
    for split in sorted({task.split for task in tasks}):
        rows = [task for task in tasks if task.split == split]
        path = root / f"{name}_{split}.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for task in rows:
                handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
        splits[split] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "records": len(rows),
            "skeletons": len({task.skeleton for task in rows}),
            "behaviours": len({task.behaviour for task in rows}),
        }

    inputs = {i for task in tasks for i in (*task.inputs, *task.held_out_inputs)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": name,
        "basis": BASIS,
        "generator": "wandering_light.stack_pbe",
        "generator_description": (
            "Behaviours are hand-written and paired with an independent Python "
            "reference; inputs are sampled fresh per example; no task input "
            "contains a quotation."
        ),
        "config": config,
        "split_policy": "skeleton",
        "catalogue": {
            "behaviours": len(catalogue),
            "skeletons": len({b.skeleton for b in catalogue}),
            "digest": catalogue_digest(catalogue),
            "source_digest": catalogue_source_digest(catalogue),
        },
        "counts": {
            "tasks": len(tasks),
            "shown_examples": sum(task.examples for task in tasks),
            "held_out_examples": sum(task.held_out for task in tasks),
            "distinct_input_stacks": len(inputs),
        },
        "splits": splits,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


_TUPLE_FIELDS = (
    "inputs",
    "outputs",
    "held_out_inputs",
    "held_out_outputs",
    "program",
    "tags",
)


VERIFY_MODES = ("local", "hub", "off")


def load(
    path: str | Path,
    *,
    verify: str = "local",
    catalogue: Sequence[Behaviour] = CATALOGUE,
) -> list[Task]:
    """Read a corpus back, from a single payload or from a corpus directory.

    A directory is read through its manifest, so the split files it names are
    the ones loaded and nothing else in the directory is picked up by accident.
    Its payloads are not in git (see `corpus_hub`), so the directory is checked
    against its manifest before anything is read (#142):

    - ``"local"`` (default): `verify_corpus` without the Hub -- every split's
      sha256 matches the manifest (an error otherwise), and a manifest this
      module generated names ``catalogue``, the live one unless the corpus was
      written from another (`write_corpus` takes the same argument); a
      mismatch there warns.
    - ``"hub"``: the same, then `corpus_hub.verify_published_corpus` against
      the manifest's pinned Hub revision, the one check that tells a locally
      regenerated corpus from the published one. Needs the network or a warm
      Hub cache, and a manifest with a ``hub`` block.
    - ``"off"``: read whatever is there.

    A single payload file has no manifest and is read as it is.
    """
    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}, not {verify!r}")
    target = Path(path)
    if not target.is_dir():
        return _load_payload(target)
    if verify != "off":
        verify_corpus(target, hub=verify == "hub", catalogue=catalogue)
    manifest = json.loads((target / "manifest.json").read_text())
    return [
        task
        for split in sorted(manifest["splits"])
        for task in _load_payload(target / manifest["splits"][split]["path"])
    ]


def verify_corpus(
    corpus_dir: str | Path,
    *,
    hub: bool = False,
    catalogue: Sequence[Behaviour] = CATALOGUE,
) -> dict[str, str]:
    """Check a corpus directory against its manifest; return the digests checked.

    Every split payload must hash to what the manifest says, or this raises.
    A manifest this module wrote is also compared with ``catalogue``: its
    witness digest, and its source digest when it carries one (manifests
    written before #142 do not). A mismatch there is a warning, not an
    error: the catalogue legitimately grows past a published corpus, and the
    project records provenance rather than enforcing it. With ``hub``, the
    manifest and payloads must also be the ones published at the pinned
    revision.
    """
    root = Path(corpus_dir)
    manifest = json.loads((root / corpus_hub.MANIFEST_NAME).read_text())
    checked = corpus_hub.verify_corpus_files(root)
    if manifest.get("generator") == "wandering_light.stack_pbe":
        recorded = manifest.get("catalogue") or {}
        live = {
            "digest": catalogue_digest(catalogue),
            "source_digest": catalogue_source_digest(catalogue),
        }
        for key, value in live.items():
            if recorded.get(key) is None:
                continue
            if recorded[key] != value:
                warnings.warn(
                    f"{root}: manifest catalogue {key} is {recorded[key]}, "
                    f"the catalogue given is {value}; the corpus was not "
                    "generated from this catalogue",
                    stacklevel=3,
                )
            checked[f"catalogue:{key}"] = value
    if hub:
        location = manifest.get("hub") or {}
        if not location.get("revision"):
            raise ValueError(
                f"{root}: manifest pins no hub revision, so it cannot be checked "
                "against a published corpus"
            )
        checked.update(
            corpus_hub.verify_published_corpus(
                root, repo_id=location["repo_id"], revision=location["revision"]
            )
        )
    return checked


def _load_payload(path: Path) -> list[Task]:
    """JSON has no tuples, so restore them on the way in."""
    with (
        path.open("rb") as raw,
        gzip.GzipFile(fileobj=raw)
        if raw.peek(2).startswith(b"\x1f\x8b")
        else nullcontext(raw) as handle,
    ):
        return [
            Task(
                **{
                    **row,
                    **{f: tuple(row[f]) for f in _TUPLE_FIELDS},
                    "alternative_programs": tuple(
                        tuple(p) for p in row.get("alternative_programs", ())
                    ),
                }
            )
            for row in map(json.loads, handle)
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wandering_light.stack_pbe")
    parser.add_argument(
        "--corpus-dir",
        default=f"wandering_light/evals/data/{CORPUS_NAME}",
        help="written as split payloads plus a manifest; payloads are not committed",
    )
    parser.add_argument("--per-behaviour", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-examples", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=6)
    parser.add_argument("--held-out", type=int, default=8)
    parser.add_argument("--verify-trials", type=int, default=500)
    args = parser.parse_args(argv)

    started = time.time()
    bad = verify_all(trials=args.verify_trials)
    if bad:
        for mismatch in bad:
            print(
                f"MISMATCH {mismatch.behaviour} on {mismatch.args}: "
                f"want {mismatch.expected}, got {mismatch.got}",
                file=sys.stderr,
            )
        return 1
    print(
        f"{len(CATALOGUE)} behaviours agree with their witness on "
        f"{args.verify_trials} fresh inputs each",
        file=sys.stderr,
    )

    tasks, stats = generate(
        args.per_behaviour,
        args.seed,
        min_examples=args.min_examples,
        max_examples=args.max_examples,
        held_out=args.held_out,
    )
    manifest = write_corpus(
        tasks,
        args.corpus_dir,
        config={
            "per_behaviour": args.per_behaviour,
            "seed": args.seed,
            "min_examples": args.min_examples,
            "max_examples": args.max_examples,
            "held_out": args.held_out,
        },
    )

    inputs = {i for task in tasks for i in (*task.inputs, *task.held_out_inputs)}
    splits = Counter(task.split for task in tasks)
    print(
        f"wrote {len(tasks)} tasks under {args.corpus_dir} in "
        f"{time.time() - started:.1f}s; manifest {manifest}",
        file=sys.stderr,
    )
    print(
        f"  {len(inputs):,} distinct input stacks, "
        f"{sum(t.examples for t in tasks):,} shown + "
        f"{sum(t.held_out for t in tasks):,} held-out examples, "
        f"train {splits['train']} / eval {splits['eval']}",
        file=sys.stderr,
    )
    lengths = Counter(task.length for task in tasks)
    print(f"  witness lengths: {dict(sorted(lengths.items()))}", file=sys.stderr)
    print(
        f"  {len(CATALOGUE)} behaviours, "
        f"{len({b.skeleton for b in CATALOGUE})} skeletons",
        file=sys.stderr,
    )
    for name, s in stats.items():
        print(
            f"  {name:<18} {s['kept']:>3} tasks, witness {s['length']:>2} ops, "
            f"{s['witnesses']} witness(es), {s['split']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
