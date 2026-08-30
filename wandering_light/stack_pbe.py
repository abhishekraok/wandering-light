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
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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


def _pair(pool: Sequence[int]):
    return lambda rng: (rng.choice(pool), rng.choice(pool))


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


def _even(value: int) -> bool:
    # The witness computes `x - 2 * (x // 2)`, which is Python's floor
    # semantics, so the reference must floor too rather than use `abs`.
    return value - 2 * (value // 2) == 0


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
        reference=lambda a, b: max(a, b),
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
        reference=lambda xs: sum(xs),
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="product",
        summary="multiply the list together",
        witnesses=("1 [ mul ] fold", "uncons [ mul ] fold"),
        sample=_list_of(MIXED, lo=2, hi=4),
        reference=lambda xs: math.prod(xs),
        tags=frozenset({"fold"}),
    ),
    Behaviour(
        name="length",
        summary="how many elements the list has",
        witnesses=("0 [ drop 1 add ] fold", "[ drop 1 ] map 0 [ add ] fold"),
        sample=_list_of(SMALL, lo=1, hi=7),
        reference=lambda xs: len(xs),
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
        reference=lambda xs: max(xs),
        tags=frozenset({"fold", "branch"}),
    ),
    Behaviour(
        name="minimum",
        summary="the smallest element",
        witnesses=("uncons [ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold",),
        sample=_list_of(SMALL),
        reference=lambda xs: min(xs),
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
)


@dataclass(frozen=True)
class _ElementTransform:
    """One independently stated integer transform for composed behaviours."""

    name: str
    action: str
    body: str
    reference: Callable[[int], int]


@dataclass(frozen=True)
class _ElementPredicate:
    """One independently stated integer predicate for composed behaviours."""

    name: str
    description: str
    body: str
    reference: Callable[[int], bool]


_ELEMENT_TRANSFORMS = (
    _ElementTransform("increment", "increment", "1 add", lambda x: x + 1),
    _ElementTransform("double", "double", "dup add", lambda x: 2 * x),
    _ElementTransform("negate", "negate", "0 swap sub", lambda x: -x),
    _ElementTransform(
        "floor_half", "halve, rounding down", "1 1 add div", lambda x: x // 2
    ),
    _ElementTransform(
        "absolute",
        "take the magnitude of",
        "[ dup 0 lt? ] [ 0 swap sub ] [ ] ifte",
        abs,
    ),
    _ElementTransform("cube", "cube", "dup dup mul mul", lambda x: x**3),
    _ElementTransform(
        "fourth_power", "raise to the fourth power", "dup mul dup mul", lambda x: x**4
    ),
    _ElementTransform(
        "triangular",
        "replace by its triangular number",
        "dup 1 add mul 1 1 add div",
        lambda x: x * (x + 1) // 2,
    ),
    _ElementTransform(
        "ceiling_half",
        "halve, rounding up",
        "dup 1 1 add div sub",
        lambda x: x - x // 2,
    ),
    _ElementTransform(
        "parity",
        "replace by its remainder modulo two",
        "dup 1 1 add div 1 1 add mul sub",
        lambda x: x - 2 * (x // 2),
    ),
    _ElementTransform(
        "nonnegative_part",
        "replace by its nonnegative part",
        "[ dup 0 lt? ] [ drop 0 ] [ ] ifte",
        lambda x: max(x, 0),
    ),
    _ElementTransform(
        "nonpositive_part",
        "replace by its nonpositive part",
        "[ dup 0 swap lt? ] [ drop 0 ] [ ] ifte",
        lambda x: min(x, 0),
    ),
    _ElementTransform(
        "signed_square",
        "square while preserving its sign",
        "dup [ over 0 lt? ] [ mul 0 swap sub ] [ mul ] ifte",
        lambda x: (-1 if x < 0 else 1) * x * x,
    ),
    _ElementTransform(
        "distance_from_one",
        "replace by its distance from one",
        "1 sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte",
        lambda x: abs(x - 1),
    ),
    _ElementTransform(
        "successor_square",
        "increment, then square",
        "1 add dup mul",
        lambda x: (x + 1) ** 2,
    ),
    _ElementTransform(
        "zero_safe_reciprocal",
        "replace by one floor-divided by it, using zero at zero",
        "[ dup 0 eq? ] [ drop 0 ] [ 1 swap div ] ifte",
        lambda x: 0 if x == 0 else 1 // x,
    ),
    _ElementTransform(
        "positive_indicator",
        "replace by one when positive and zero otherwise",
        "[ dup 0 swap lt? ] [ drop 1 ] [ drop 0 ] ifte",
        lambda x: int(x > 0),
    ),
    _ElementTransform(
        "sign",
        "replace by its sign",
        "[ dup 0 lt? ] [ drop 0 1 sub ] "
        "[ [ dup 0 eq? ] [ drop 0 ] [ drop 1 ] ifte ] ifte",
        lambda x: int(x > 0) - int(x < 0),
    ),
)

_PARITY_BODY = "dup 1 1 add div 1 1 add mul sub"

_ELEMENT_PREDICATES = (
    _ElementPredicate("negative", "negative", "0 lt?", lambda x: x < 0),
    _ElementPredicate("positive", "positive", "0 swap lt?", lambda x: x > 0),
    _ElementPredicate("even", "even", f"{_PARITY_BODY} 0 eq?", lambda x: _even(x)),
    _ElementPredicate(
        "odd",
        "odd",
        f"[ {_PARITY_BODY} 0 eq? ] [ drop false ] [ drop true ] ifte",
        lambda x: not _even(x),
    ),
    _ElementPredicate(
        "nonzero",
        "nonzero",
        "[ 0 eq? ] [ drop false ] [ drop true ] ifte",
        lambda x: x != 0,
    ),
    _ElementPredicate(
        "nonnegative",
        "nonnegative",
        "[ dup 0 lt? ] [ drop false ] [ drop true ] ifte",
        lambda x: x >= 0,
    ),
    _ElementPredicate(
        "nonpositive",
        "nonpositive",
        "[ dup 0 swap lt? ] [ drop false ] [ drop true ] ifte",
        lambda x: x <= 0,
    ),
    _ElementPredicate(
        "unit_magnitude", "magnitude-one", "dup mul 1 eq?", lambda x: abs(x) == 1
    ),
    _ElementPredicate(
        "magnitude_below_three",
        "magnitude-below-three",
        "dup mul 1 1 add 1 add dup mul lt?",
        lambda x: abs(x) < 3,
    ),
    _ElementPredicate(
        "magnitude_above_one",
        "magnitude-above-one",
        "dup mul 1 swap lt?",
        lambda x: abs(x) > 1,
    ),
    _ElementPredicate(
        "multiple_of_three",
        "multiples-of-three",
        "dup 1 1 add 1 add div 1 1 add 1 add mul sub 0 eq?",
        lambda x: x - 3 * (x // 3) == 0,
    ),
    _ElementPredicate(
        "zero_or_one",
        "zero-or-one",
        "dup 1 sub mul 0 eq?",
        lambda x: x in (0, 1),
    ),
    _ElementPredicate(
        "negative_even",
        "negative-even",
        f"[ dup 0 lt? ] [ {_PARITY_BODY} 0 eq? ] [ drop false ] ifte",
        lambda x: x < 0 and _even(x),
    ),
    _ElementPredicate(
        "positive_odd",
        "positive-odd",
        f"[ dup 0 swap lt? ] "
        f"[ [ {_PARITY_BODY} 0 eq? ] [ drop false ] [ drop true ] ifte ] "
        "[ drop false ] ifte",
        lambda x: x > 0 and not _even(x),
    ),
)


def _filter_accept(
    transform: _ElementTransform | None, predicate: _ElementPredicate
) -> Callable[[list[Args], list[Result]], bool]:
    """Require a composed filter to both keep and reject sampled elements."""

    def accept(args_list: list[Args], _results: list[Result]) -> bool:
        decisions = [
            predicate.reference(
                transform.reference(value) if transform is not None else value
            )
            for args in args_list
            for value in args[0]
            if isinstance(value, int)
        ]
        return any(decisions) and not all(decisions)

    return accept


def _composed_tags(
    transform: _ElementTransform,
    predicate: _ElementPredicate,
    *,
    fold: bool = False,
) -> frozenset[str]:
    tags = {"filter", "map"}
    if fold:
        tags.add("fold")
    if "ifte" in transform.body or "ifte" in predicate.body:
        tags.add("branch")
    return frozenset(tags)


def _filter_then_map(
    transform: _ElementTransform, predicate: _ElementPredicate
) -> Behaviour:
    def reference(xs: list[int]) -> list[int]:
        return [transform.reference(x) for x in xs if predicate.reference(x)]

    return Behaviour(
        name=f"keep_{predicate.name}_then_{transform.name}",
        summary=(
            f"keep the {predicate.description} values, then {transform.action} each one"
        ),
        witnesses=(f"[ {predicate.body} ] filter [ {transform.body} ] map",),
        sample=_list_of(SMALL, hi=7),
        reference=reference,
        tags=_composed_tags(transform, predicate),
        accept=_shrinks,
    )


def _filter_map_sum(
    transform: _ElementTransform, predicate: _ElementPredicate
) -> Behaviour:
    def reference(xs: list[int]) -> int:
        return sum(transform.reference(x) for x in xs if predicate.reference(x))

    return Behaviour(
        name=f"sum_{transform.name}_of_{predicate.name}",
        summary=(
            f"keep the {predicate.description} values, {transform.action} each one, "
            "then sum them"
        ),
        witnesses=(
            f"[ {predicate.body} ] filter [ {transform.body} ] map 0 [ add ] fold",
        ),
        sample=_list_of(SMALL, hi=7),
        reference=reference,
        tags=_composed_tags(transform, predicate, fold=True),
        accept=_filter_accept(None, predicate),
    )


# Different pipelines can state the same function: absolute value after a
# positive filter is just the positive filter, for example. Such programs are
# alternative witnesses, not new behaviours. Keep one representative of each
# per-element function and omit the aliases explicitly.
_DUPLICATE_FILTER_MAPS = frozenset(
    {
        ("absolute", "negative"),
        ("nonpositive_part", "negative"),
        ("positive_indicator", "negative"),
        ("sign", "negative"),
        ("absolute", "positive"),
        ("nonnegative_part", "positive"),
        ("sign", "positive"),
        ("ceiling_half", "even"),
        ("nonnegative_part", "nonnegative"),
        ("sign", "nonnegative"),
        ("absolute", "nonpositive"),
        ("positive_indicator", "nonpositive"),
        ("sign", "nonpositive"),
        ("fourth_power", "unit_magnitude"),
        ("ceiling_half", "unit_magnitude"),
        ("parity", "unit_magnitude"),
        ("nonnegative_part", "unit_magnitude"),
        ("nonpositive_part", "unit_magnitude"),
        ("signed_square", "unit_magnitude"),
        ("zero_safe_reciprocal", "unit_magnitude"),
        ("positive_indicator", "unit_magnitude"),
        ("sign", "unit_magnitude"),
        ("cube", "zero_or_one"),
        ("fourth_power", "zero_or_one"),
        ("triangular", "zero_or_one"),
        ("ceiling_half", "zero_or_one"),
        ("parity", "zero_or_one"),
        ("nonnegative_part", "zero_or_one"),
        ("nonpositive_part", "zero_or_one"),
        ("signed_square", "zero_or_one"),
        ("zero_safe_reciprocal", "zero_or_one"),
        ("positive_indicator", "zero_or_one"),
        ("sign", "zero_or_one"),
        ("absolute", "negative_even"),
        ("ceiling_half", "negative_even"),
        ("nonnegative_part", "negative_even"),
        ("positive_indicator", "negative_even"),
        ("sign", "negative_even"),
        ("nonnegative_part", "positive_odd"),
        ("positive_indicator", "positive_odd"),
        ("sign", "positive_odd"),
    }
)

_DUPLICATE_FILTER_MAP_SUMS = frozenset(
    {
        ("absolute", "negative"),
        ("nonnegative_part", "negative"),
        ("positive_indicator", "negative"),
        ("sign", "negative"),
        ("absolute", "positive"),
        ("nonnegative_part", "positive"),
        ("nonpositive_part", "positive"),
        ("sign", "positive"),
        ("ceiling_half", "even"),
        ("parity", "even"),
        ("positive_indicator", "odd"),
        ("parity", "nonzero"),
        ("nonnegative_part", "nonzero"),
        ("nonpositive_part", "nonzero"),
        ("positive_indicator", "nonzero"),
        ("double", "nonnegative"),
        ("negate", "nonnegative"),
        ("floor_half", "nonnegative"),
        ("absolute", "nonnegative"),
        ("cube", "nonnegative"),
        ("fourth_power", "nonnegative"),
        ("triangular", "nonnegative"),
        ("ceiling_half", "nonnegative"),
        ("parity", "nonnegative"),
        ("nonnegative_part", "nonnegative"),
        ("nonpositive_part", "nonnegative"),
        ("signed_square", "nonnegative"),
        ("zero_safe_reciprocal", "nonnegative"),
        ("positive_indicator", "nonnegative"),
        ("sign", "nonnegative"),
        ("double", "nonpositive"),
        ("negate", "nonpositive"),
        ("floor_half", "nonpositive"),
        ("absolute", "nonpositive"),
        ("cube", "nonpositive"),
        ("fourth_power", "nonpositive"),
        ("triangular", "nonpositive"),
        ("ceiling_half", "nonpositive"),
        ("parity", "nonpositive"),
        ("nonnegative_part", "nonpositive"),
        ("nonpositive_part", "nonpositive"),
        ("signed_square", "nonpositive"),
        ("zero_safe_reciprocal", "nonpositive"),
        ("positive_indicator", "nonpositive"),
        ("sign", "nonpositive"),
        ("fourth_power", "unit_magnitude"),
        ("triangular", "unit_magnitude"),
        ("ceiling_half", "unit_magnitude"),
        ("parity", "unit_magnitude"),
        ("nonnegative_part", "unit_magnitude"),
        ("nonpositive_part", "unit_magnitude"),
        ("signed_square", "unit_magnitude"),
        ("zero_safe_reciprocal", "unit_magnitude"),
        ("positive_indicator", "unit_magnitude"),
        ("sign", "unit_magnitude"),
        ("parity", "magnitude_below_three"),
        ("double", "zero_or_one"),
        ("floor_half", "zero_or_one"),
        ("absolute", "zero_or_one"),
        ("cube", "zero_or_one"),
        ("fourth_power", "zero_or_one"),
        ("triangular", "zero_or_one"),
        ("ceiling_half", "zero_or_one"),
        ("parity", "zero_or_one"),
        ("nonnegative_part", "zero_or_one"),
        ("nonpositive_part", "zero_or_one"),
        ("signed_square", "zero_or_one"),
        ("distance_from_one", "zero_or_one"),
        ("zero_safe_reciprocal", "zero_or_one"),
        ("positive_indicator", "zero_or_one"),
        ("sign", "zero_or_one"),
        ("absolute", "negative_even"),
        ("ceiling_half", "negative_even"),
        ("parity", "negative_even"),
        ("nonnegative_part", "negative_even"),
        ("nonpositive_part", "negative_even"),
        ("zero_safe_reciprocal", "negative_even"),
        ("positive_indicator", "negative_even"),
        ("sign", "negative_even"),
        ("absolute", "positive_odd"),
        ("parity", "positive_odd"),
        ("nonnegative_part", "positive_odd"),
        ("nonpositive_part", "positive_odd"),
        ("zero_safe_reciprocal", "positive_odd"),
        ("positive_indicator", "positive_odd"),
        ("sign", "positive_odd"),
    }
)


def _composed_behaviours() -> tuple[Behaviour, ...]:
    pairs = tuple(
        (transform, predicate)
        for predicate in _ELEMENT_PREDICATES
        for transform in _ELEMENT_TRANSFORMS
    )
    filter_map_pairs = tuple(
        pair
        for pair in pairs
        if (pair[0].name, pair[1].name) not in _DUPLICATE_FILTER_MAPS
    )
    filter_map_sum_pairs = tuple(
        pair
        for pair in pairs
        if (pair[0].name, pair[1].name) not in _DUPLICATE_FILTER_MAP_SUMS
    )
    return (
        *(_filter_then_map(*pair) for pair in filter_map_pairs),
        *(_filter_map_sum(*pair) for pair in filter_map_sum_pairs),
    )


def _structural_behaviours() -> tuple[Behaviour, ...]:
    """Authored list rearrangements whose top-level shapes are not pipelines."""

    specs = (
        (
            "second",
            "the second element",
            "uncons drop uncons swap drop",
            2,
            lambda xs: xs[1],
        ),
        (
            "penultimate",
            "the next-to-last element",
            "nil [ cons ] fold uncons drop uncons swap drop",
            2,
            lambda xs: xs[-2],
        ),
        (
            "without_last",
            "all but the last element",
            "nil [ cons ] fold uncons drop nil [ cons ] fold",
            2,
            lambda xs: xs[:-1],
        ),
        (
            "without_ends",
            "all but the first and last elements",
            "uncons drop nil [ cons ] fold uncons drop nil [ cons ] fold",
            2,
            lambda xs: xs[1:-1],
        ),
        (
            "rotate_left",
            "move the first element to the end",
            "uncons quote cat",
            1,
            lambda xs: [*xs[1:], xs[0]],
        ),
        (
            "rotate_right",
            "move the last element to the front",
            "nil [ cons ] fold uncons swap nil [ cons ] fold swap cons",
            1,
            lambda xs: [xs[-1], *xs[:-1]],
        ),
        (
            "duplicate_first",
            "prepend a second copy of the first element",
            "dup uncons swap drop cons",
            1,
            lambda xs: [xs[0], *xs],
        ),
        (
            "prepend_last",
            "prepend a second copy of the last element",
            "dup nil [ cons ] fold uncons swap drop cons",
            1,
            lambda xs: [xs[-1], *xs],
        ),
        (
            "append_first",
            "append a second copy of the first element",
            "dup uncons swap drop quote cat",
            1,
            lambda xs: [*xs, xs[0]],
        ),
        (
            "append_last",
            "append a second copy of the last element",
            "dup nil [ cons ] fold uncons swap drop quote cat",
            1,
            lambda xs: [*xs, xs[-1]],
        ),
        (
            "repeat_list",
            "repeat the whole list twice",
            "dup cat",
            1,
            lambda xs: [*xs, *xs],
        ),
        (
            "palindrome_forward",
            "append the reverse of the list",
            "dup nil [ cons ] fold cat",
            1,
            lambda xs: [*xs, *reversed(xs)],
        ),
        (
            "palindrome_backward",
            "prepend the reverse of the list",
            "dup nil [ cons ] fold swap cat",
            1,
            lambda xs: [*reversed(xs), *xs],
        ),
        (
            "singleton_head",
            "a singleton list containing the first element",
            "uncons swap drop quote",
            1,
            lambda xs: xs[:1],
        ),
        (
            "singleton_last",
            "a singleton list containing the last element",
            "nil [ cons ] fold uncons swap drop quote",
            1,
            lambda xs: xs[-1:],
        ),
        (
            "repeat_each_twice",
            "repeat every element twice",
            "nil [ dup [ cons ] dip cons ] fold nil [ cons ] fold",
            1,
            lambda xs: [item for x in xs for item in (x, x)],
        ),
        (
            "zero_between",
            "put a zero between consecutive elements",
            "nil [ cons 0 cons ] fold nil [ cons ] fold "
            "nil [ cons ] fold uncons drop nil [ cons ] fold",
            1,
            lambda xs: [item for x in xs for item in (x, 0)][:-1],
        ),
        (
            "zero_wrapped",
            "put zeros before, after, and between the elements",
            "nil [ cons 0 cons ] fold nil [ cons ] fold 0 cons",
            1,
            lambda xs: [0, *[item for x in xs for item in (x, 0)]],
        ),
        (
            "reverse_without_first",
            "drop the first element, then reverse",
            "uncons drop nil [ cons ] fold",
            1,
            lambda xs: list(reversed(xs[1:])),
        ),
        (
            "reverse_without_last",
            "reverse, then drop the original last element",
            "nil [ cons ] fold uncons drop",
            1,
            lambda xs: list(reversed(xs))[1:],
        ),
        (
            "list_then_tail",
            "append the tail to the original list",
            "dup uncons drop cat",
            1,
            lambda xs: [*xs, *xs[1:]],
        ),
        (
            "tail_then_list",
            "prepend the tail to the original list",
            "dup uncons drop swap cat",
            1,
            lambda xs: [*xs[1:], *xs],
        ),
        (
            "head_and_last",
            "the first and last elements",
            "dup uncons swap drop swap 0 [ swap drop ] fold nil swap cons swap cons",
            1,
            lambda xs: [xs[0], xs[-1]],
        ),
        (
            "last_and_head",
            "the last and first elements",
            "dup uncons swap drop swap 0 [ swap drop ] fold "
            "swap nil swap cons swap cons",
            1,
            lambda xs: [xs[-1], xs[0]],
        ),
        (
            "reverse_repeat",
            "reverse the list, then repeat it twice",
            "nil [ cons ] fold dup cat",
            1,
            lambda xs: [*reversed(xs), *reversed(xs)],
        ),
    )
    return tuple(
        Behaviour(
            name=name,
            summary=summary,
            witnesses=(witness,),
            sample=_list_of(SMALL, lo=lo, hi=7),
            reference=reference,
            tags=frozenset(
                {"structural", "fold"} if "fold" in witness else {"structural"}
            ),
        )
        for name, summary, witness, lo, reference in specs
    )


def _closure_behaviours() -> tuple[Behaviour, ...]:
    regular = _list_and_int(SMALL, SMALL, hi=7)
    positive_n = _list_and_int(SMALL, POSITIVE, hi=7)
    return (
        Behaviour(
            name="square_plus_n_each",
            summary="square every element, then add the given number",
            witnesses=("[ swap dup mul swap add ] swap cons map",),
            sample=regular,
            reference=lambda xs, n: [x * x + n for x in xs],
            tags=frozenset({"map", "closure"}),
        ),
        Behaviour(
            name="shifted_square_each",
            summary="add the given number to every element, then square",
            witnesses=("[ add dup mul ] swap cons map",),
            sample=regular,
            reference=lambda xs, n: [(x + n) ** 2 for x in xs],
            tags=frozenset({"map", "closure"}),
        ),
        Behaviour(
            name="squared_distance_each",
            summary="the squared distance of every element from the given number",
            witnesses=(
                "[ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte dup mul ] swap cons map",
                "[ sub dup mul ] swap cons map",
            ),
            sample=regular,
            reference=lambda xs, n: [(x - n) ** 2 for x in xs],
            tags=frozenset({"map", "closure", "branch"}),
        ),
        Behaviour(
            name="remainder_by_n_each",
            summary="each element's remainder modulo the given positive number",
            witnesses=("[ over over div mul sub ] swap cons map",),
            sample=positive_n,
            reference=lambda xs, n: [x - n * (x // n) for x in xs],
            tags=frozenset({"map", "closure"}),
        ),
        Behaviour(
            name="distance_from_n_each",
            summary="the distance of every element from the given number",
            witnesses=("[ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] swap cons map",),
            sample=regular,
            reference=lambda xs, n: [abs(x - n) for x in xs],
            tags=frozenset({"map", "closure", "branch"}),
        ),
        Behaviour(
            name="minimum_with_n_each",
            summary="the smaller of every element and the given number",
            witnesses=(
                "[ [ over over lt? ] [ drop ] [ swap drop ] ifte ] swap cons map",
            ),
            sample=regular,
            reference=lambda xs, n: [min(x, n) for x in xs],
            tags=frozenset({"map", "closure", "branch"}),
        ),
        Behaviour(
            name="maximum_with_n_each",
            summary="the larger of every element and the given number",
            witnesses=(
                "[ [ over over lt? ] [ swap drop ] [ drop ] ifte ] swap cons map",
            ),
            sample=regular,
            reference=lambda xs, n: [max(x, n) for x in xs],
            tags=frozenset({"map", "closure", "branch"}),
        ),
        Behaviour(
            name="below_n_mask",
            summary="one where an element is below the given number, else zero",
            witnesses=(
                "[ [ over over lt? ] [ drop drop 1 ] [ drop drop 0 ] ifte ] "
                "swap cons map",
            ),
            sample=regular,
            reference=lambda xs, n: [int(x < n) for x in xs],
            tags=frozenset({"map", "closure", "branch"}),
        ),
        Behaviour(
            name="multiply_below_add_other",
            summary="multiply values below n by n and add n to the rest",
            witnesses=("[ [ over over lt? ] [ mul ] [ add ] ifte ] swap cons map",),
            sample=regular,
            reference=lambda xs, n: [x * n if x < n else x + n for x in xs],
            tags=frozenset({"map", "closure", "branch"}),
        ),
        Behaviour(
            name="keep_multiples_of_n",
            summary="keep the multiples of the given positive number",
            witnesses=("[ over over div mul sub 0 eq? ] swap cons filter",),
            sample=positive_n,
            reference=lambda xs, n: [x for x in xs if x - n * (x // n) == 0],
            tags=frozenset({"filter", "closure"}),
            accept=_shrinks,
        ),
        Behaviour(
            name="keep_abs_below_n",
            summary="keep values whose magnitude is below the given positive number",
            witnesses=(
                "[ swap [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte swap lt? ] "
                "swap cons filter",
            ),
            sample=positive_n,
            reference=lambda xs, n: [x for x in xs if abs(x) < n],
            tags=frozenset({"filter", "closure", "branch"}),
            accept=_shrinks,
        ),
        Behaviour(
            name="keep_square_above_n",
            summary="keep values whose square is above the given positive number",
            witnesses=("[ swap dup mul lt? ] swap cons filter",),
            sample=positive_n,
            reference=lambda xs, n: [x for x in xs if x * x > n],
            tags=frozenset({"filter", "closure"}),
            accept=_shrinks,
        ),
        Behaviour(
            name="sum_distances_from_n",
            summary="sum every element's distance from the given number",
            witnesses=(
                "[ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] "
                "swap cons map 0 [ add ] fold",
            ),
            sample=regular,
            reference=lambda xs, n: sum(abs(x - n) for x in xs),
            tags=frozenset({"map", "fold", "closure", "branch"}),
        ),
        Behaviour(
            name="sum_squared_distances_from_n",
            summary="sum every element's squared distance from the given number",
            witnesses=("[ sub dup mul ] swap cons map 0 [ add ] fold",),
            sample=regular,
            reference=lambda xs, n: sum((x - n) ** 2 for x in xs),
            tags=frozenset({"map", "fold", "closure"}),
        ),
        Behaviour(
            name="maximum_distance_from_n",
            summary="the largest distance from the given number",
            witnesses=(
                "[ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] swap cons map "
                "uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold",
            ),
            sample=regular,
            reference=lambda xs, n: max(abs(x - n) for x in xs),
            tags=frozenset({"map", "fold", "closure", "branch"}),
        ),
        Behaviour(
            name="keep_below_n_then_reverse",
            summary="keep values below the given number, then reverse them",
            witnesses=("[ lt? ] swap cons filter nil [ cons ] fold",),
            sample=regular,
            reference=lambda xs, n: list(reversed([x for x in xs if x < n])),
            tags=frozenset({"filter", "fold", "closure"}),
            accept=_shrinks,
        ),
        Behaviour(
            name="sum_squares_below_n",
            summary="sum the squares of values below the given number",
            witnesses=("[ lt? ] swap cons filter [ dup mul ] map 0 [ add ] fold",),
            sample=regular,
            reference=lambda xs, n: sum(x * x for x in xs if x < n),
            tags=frozenset({"filter", "map", "fold", "closure"}),
        ),
        Behaviour(
            name="below_n_centered",
            summary="keep values below n, then subtract n from each",
            witnesses=(
                "swap over [ lt? ] swap cons filter swap [ sub ] swap cons map",
            ),
            sample=regular,
            reference=lambda xs, n: [x - n for x in xs if x < n],
            tags=frozenset({"filter", "map", "closure"}),
            accept=_shrinks,
        ),
        Behaviour(
            name="multiples_n_quotients",
            summary="divide each multiple of n by the given positive n",
            witnesses=(
                "swap over [ over over div mul sub 0 eq? ] swap cons filter "
                "swap [ div ] swap cons map",
            ),
            sample=positive_n,
            reference=lambda xs, n: [x // n for x in xs if x - n * (x // n) == 0],
            tags=frozenset({"filter", "map", "closure"}),
            accept=_shrinks,
        ),
    )


def _scan_behaviours() -> tuple[Behaviour, ...]:
    return (
        Behaviour(
            name="prefix_square_sums",
            summary="running sum of the squares",
            witnesses=(
                "nil [ [ over null? ] [ dup mul cons ] "
                "[ over uncons swap drop swap dup mul add cons ] ifte ] fold "
                "nil [ cons ] fold",
            ),
            sample=_list_of(SMALL, lo=1, hi=7),
            reference=lambda xs: [
                sum(x * x for x in xs[:i]) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_magnitudes",
            summary="running sum of the magnitudes",
            witnesses=(
                "nil [ [ over null? ] "
                "[ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte cons ] "
                "[ over uncons swap drop swap [ dup 0 lt? ] "
                "[ 0 swap sub ] [ ] ifte add cons ] ifte ] fold nil [ cons ] fold",
            ),
            sample=_list_of(SMALL, lo=1, hi=7),
            reference=lambda xs: [
                sum(abs(x) for x in xs[:i]) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_maxima",
            summary="running maximum",
            witnesses=(
                "nil [ [ over null? ] [ cons ] [ over uncons swap drop "
                "[ over over lt? ] [ swap drop ] [ drop ] ifte cons ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(SMALL, lo=1, hi=7),
            reference=lambda xs: [max(xs[:i]) for i in range(1, len(xs) + 1)],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_minima",
            summary="running minimum",
            witnesses=(
                "nil [ [ over null? ] [ cons ] [ over uncons swap drop "
                "[ over over lt? ] [ drop ] [ swap drop ] ifte cons ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(SMALL, lo=1, hi=7),
            reference=lambda xs: [min(xs[:i]) for i in range(1, len(xs) + 1)],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_positive_counts",
            summary="running count of positive values",
            witnesses=(
                "nil [ [ over null? ] [ [ dup 0 swap lt? ] "
                "[ drop 1 cons ] [ drop 0 cons ] ifte ] "
                "[ over uncons swap drop [ over 0 swap lt? ] "
                "[ 1 add swap drop cons ] [ swap drop cons ] ifte ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(MIXED, lo=1, hi=7),
            reference=lambda xs: [
                sum(x > 0 for x in xs[:i]) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_negative_counts",
            summary="running count of negative values",
            witnesses=(
                "nil [ [ over null? ] [ [ dup 0 lt? ] "
                "[ drop 1 cons ] [ drop 0 cons ] ifte ] "
                "[ over uncons swap drop [ over 0 lt? ] "
                "[ 1 add swap drop cons ] [ swap drop cons ] ifte ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(MIXED, lo=1, hi=7),
            reference=lambda xs: [
                sum(x < 0 for x in xs[:i]) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_even_counts",
            summary="running count of even values",
            witnesses=(
                "nil [ [ over null? ] "
                "[ [ dup dup 1 1 add div 1 1 add mul sub 0 eq? ] "
                "[ drop 1 cons ] [ drop 0 cons ] ifte ] "
                "[ over uncons swap drop [ over dup 1 1 add div 1 1 add mul "
                "sub 0 eq? ] [ 1 add swap drop cons ] [ swap drop cons ] ifte ] "
                "ifte ] fold nil [ cons ] fold",
            ),
            sample=_list_of(SMALL, lo=1, hi=7),
            reference=lambda xs: [
                sum(_even(x) for x in xs[:i]) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_nonzero_counts",
            summary="running count of nonzero values",
            witnesses=(
                "nil [ [ over null? ] [ [ dup 0 eq? ] "
                "[ drop 0 cons ] [ drop 1 cons ] ifte ] "
                "[ over uncons swap drop [ over 0 eq? ] "
                "[ swap drop cons ] [ 1 add swap drop cons ] ifte ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(SPARSE, lo=1, hi=7),
            reference=lambda xs: [
                sum(x != 0 for x in xs[:i]) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_positive_sums",
            summary="running sum of positive values",
            witnesses=(
                "nil [ [ over null? ] [ [ dup 0 swap lt? ] "
                "[ cons ] [ drop 0 cons ] ifte ] "
                "[ over uncons swap drop [ over 0 swap lt? ] "
                "[ add cons ] [ swap drop cons ] ifte ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(MIXED, lo=1, hi=7),
            reference=lambda xs: [
                sum(x for x in xs[:i] if x > 0) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
        Behaviour(
            name="prefix_negative_magnitude_sums",
            summary="running sum of negative magnitudes",
            witnesses=(
                "nil [ [ over null? ] [ [ dup 0 lt? ] "
                "[ 0 swap sub cons ] [ drop 0 cons ] ifte ] "
                "[ over uncons swap drop [ over 0 lt? ] "
                "[ swap 0 swap sub add cons ] [ swap drop cons ] ifte ] ifte ] "
                "fold nil [ cons ] fold",
            ),
            sample=_list_of(MIXED, lo=1, hi=7),
            reference=lambda xs: [
                sum(-x for x in xs[:i] if x < 0) for i in range(1, len(xs) + 1)
            ],
            tags=frozenset({"fold", "branch", "scan"}),
        ),
    )


CATALOGUE += (
    *_composed_behaviours(),
    *_structural_behaviours(),
    *_closure_behaviours(),
    *_scan_behaviours(),
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

    Names, summaries and witnesses only -- samplers and references are Python
    objects and cannot be hashed stably. Changing a witness changes the labels
    in the corpus, so it must invalidate the manifest; changing a sampler
    changes the data, which the payload digest already covers.
    """
    payload = json.dumps(
        [[b.name, b.summary, list(b.witnesses)] for b in catalogue], sort_keys=True
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


def load(path: str | Path) -> list[Task]:
    """Read a corpus back, from a single payload or from a corpus directory.

    A directory is read through its manifest, so the split files it names are
    the ones loaded and nothing else in the directory is picked up by accident.
    """
    target = Path(path)
    if target.is_dir():
        manifest = json.loads((target / "manifest.json").read_text())
        return [
            task
            for split in sorted(manifest["splits"])
            for task in load(target / manifest["splits"][split]["path"])
        ]
    return _load_payload(target)


def _load_payload(path: Path) -> list[Task]:
    """JSON has no tuples, so restore them on the way in."""
    with gzip.open(path, "rt") as handle:
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
