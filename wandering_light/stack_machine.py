"""A guarded, concatenative stack machine — the substrate core of issue #42.

Design in four sentences:

1. A *value* is an arbitrary-precision integer, a boolean, or a list of values
   (``nil`` is the empty list). Nothing else.
2. A *config* is ``(stack, fuel)``: an immutable bottom-to-top tuple of values
   and the fuel spent so far.
3. An *op* is a total-by-construction pair — ``guard(config)`` returns ``None``
   when the op is legal and a short reason string when it is not, and
   ``apply(config)`` returns ``(new_stack, cost)``. Illegal transitions are
   *absent*, not exceptional.
4. *Identity is structural.* ``value_key`` maps a value to a nested tuple of
   plain ints, bools and str tags. Two configs are the same state exactly when
   their keys are equal. No ``hash()``, no ``repr()``, no dict ordering, no
   floats: the key is a value, not a digest, so it is identical in every
   process regardless of ``PYTHONHASHSEED``.

Quotations are ordinary list values. An op's *name* is a value too -- ``Sym``
-- so a program is a list of symbols, built with the same ``nil`` and ``cons``
that build any other list, and run with ``i``. Nothing here manipulates
unevaluated program *text*: pushing ``Sym("add")`` materialises a value like
any other push, which is what keeps every intermediate state a state.

Combinators can loop, so the transition function is no longer total by
construction. Every application that runs code carries a step budget, and
exhausting it is reported through the same channel as any other refusal --
``guard`` returns a reason. Totality is therefore restored by the guard rather
than assumed.

Deliberately out of scope: ``linrec`` (``ifte`` with a self-applying quotation
reaches the same programs), stack effect types, strings, and any wiring into
``TrajectoryGraph``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import lru_cache

# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


class MachineError(Exception):
    """Base class for stack machine errors."""


class IllegalOpError(MachineError):
    """Raised when an op is applied to a config its guard rejects."""


@dataclass(frozen=True, slots=True)
class Int:
    """An arbitrary-precision integer."""

    value: int

    def __post_init__(self) -> None:
        # `bool` subclasses `int`, so `Int(True) == Int(1)` would hold and we
        # would be back to two things that are equal but not the same. This is
        # the exact bug class #42 exists to remove; refuse it at construction.
        if self.value.__class__ is not int:
            raise MachineError(f"Int takes a plain int, got {self.value.__class__}")


@dataclass(frozen=True, slots=True)
class Bool:
    """A boolean, kept structurally distinct from Int(0)/Int(1)."""

    value: bool

    def __post_init__(self) -> None:
        if self.value.__class__ is not bool:
            raise MachineError(f"Bool takes a plain bool, got {self.value.__class__}")


@dataclass(frozen=True, slots=True)
class Sym:
    """The name of an op, as a value.

    This is the whole of homoiconicity here. A quotation is just a list whose
    items are symbols, so `nil` and `cons` build programs and `i` runs them --
    no separate program representation, and no stage where the machine is
    holding text instead of values.
    """

    name: str

    def __post_init__(self) -> None:
        if self.name.__class__ is not str:
            raise MachineError(f"Sym takes a plain str, got {self.name.__class__}")


@dataclass(frozen=True, slots=True)
class Lst:
    """A list of values. The empty list is `nil`; a quotation is one of these."""

    items: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Mark:
    """The open-quotation marker left by `[` (``mark-28`` basis only).

    While one sits anywhere on the stack, executable ops push their own names
    instead of running, and `]` collects everything above the topmost marker
    into one list. Quote construction is therefore part of the machine state,
    not a mode hidden beside it.
    """


Value = Int | Bool | Sym | Lst | Mark

NIL = Lst(())
TRUE = Bool(True)
FALSE = Bool(False)
ZERO = Int(0)
ONE = Int(1)
MARK = Mark()


def value_key(value: Value) -> tuple:
    """The one and only notion of sameness for values.

    Returns a nested tuple built from str tags, ints and bools. Equality of
    keys is structural equality of values, and the key is byte-identical across
    interpreters because it contains no hashes, no reprs and no dicts.
    """
    if isinstance(value, Int):
        return ("int", value.value)
    if isinstance(value, Bool):
        return ("bool", value.value)
    if isinstance(value, Sym):
        return ("sym", value.name)
    if isinstance(value, Mark):
        return ("mark",)
    return ("list", tuple(value_key(item) for item in value.items))


def same(left: Value, right: Value) -> bool:
    """Structural sameness — defined as key equality, by construction."""
    return value_key(left) == value_key(right)


def format_value(value: Value) -> str:
    """Human-readable rendering. Display only — never an identity."""
    if isinstance(value, Int):
        return str(value.value)
    if isinstance(value, Bool):
        return "true" if value.value else "false"
    if isinstance(value, Sym):
        return value.name
    if isinstance(value, Mark):
        return "MARK"
    if not value.items:
        return "nil"
    return "(" + " ".join(format_value(item) for item in value.items) + ")"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """(stack, fuel). The stack runs bottom-to-top: `stack[-1]` is the top."""

    stack: tuple[Value, ...] = ()
    fuel: int = 0
    fuel_limit: int | None = None

    @property
    def key(self) -> tuple:
        """Structural state identity: the stack, and nothing else.

        Fuel is deliberately excluded. Fuel is what a state *cost* to reach,
        not what the state *is*; two searches arriving at the same stack are at
        the same state and should be deduplicated.
        """
        return tuple(value_key(item) for item in self.stack)

    def peek(self, depth: int = 0) -> Value:
        """`peek(0)` is the top of the stack, `peek(1)` the one below it."""
        return self.stack[-1 - depth]


def format_stack(config: Config) -> str:
    """Display only. Top of stack on the right."""
    if not config.stack:
        return "[]"
    return "[ " + " ".join(format_value(item) for item in config.stack) + " ]"


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

Guard = Callable[[Config], str | None]
Apply = Callable[[Config], tuple[tuple[Value, ...], int]]
# An op that runs code decides legality and result together, because it cannot
# know either without running: (stack, budget, depth) -> (stack, cost) | reason.
Runner = Callable[[tuple, int, int], "tuple[tuple[Value, ...], int] | str"]

# Reached only when a config carries no explicit limit. Combinators can loop,
# so something must bound them or `guard` would not terminate.
DEFAULT_FUEL_LIMIT = 100_000
MAX_QUOTATION_DEPTH = 32


def remaining_fuel(config: Config) -> int:
    limit = config.fuel_limit if config.fuel_limit is not None else DEFAULT_FUEL_LIMIT
    return max(0, limit - config.fuel)


# Groups that keep their meaning inside an open quotation: literals and symbol
# pushes already *are* data, and the brackets are what open and close it.
UNQUOTED_GROUPS = frozenset({"literals", "symbols", "brackets"})


def quoting(stack: tuple[Value, ...]) -> bool:
    """True while some top-level stack item is an open-quotation `MARK`."""
    return any(isinstance(item, Mark) for item in stack)


def _quotes(op: Op, stack: tuple[Value, ...]) -> bool:
    """Whether `op` is captured by name rather than run on this stack."""
    return op.group not in UNQUOTED_GROUPS and quoting(stack)


@dataclass(frozen=True, eq=False)
class Op:
    """A single machine instruction: a guard and an application."""

    name: str
    group: str
    effect: str  # Factor-style ( inputs -- outputs ), documentation only.
    _guard: Guard = field(repr=False)
    _apply: Apply = field(repr=False)
    _run: Runner | None = field(repr=False, default=None)

    @property
    def runs_code(self) -> bool:
        """True for ops that execute a quotation, so their cost is not 1."""
        return self._run is not None

    def guard(self, config: Config) -> str | None:
        """None when legal, else a short reason. Fuel is one such reason."""
        if config.fuel_limit is not None and config.fuel >= config.fuel_limit:
            return "out of fuel"
        if _quotes(self, config.stack):
            return None
        if self._run is None:
            return self._guard(config)
        outcome = self._run(config.stack, remaining_fuel(config), 0)
        return outcome if isinstance(outcome, str) else None

    def apply(self, config: Config) -> tuple[tuple[Value, ...], int]:
        """The new stack and the fuel this application costs."""
        if _quotes(self, config.stack):
            return (*config.stack, Sym(self.name)), 1
        if self._run is None:
            return self._apply(config)
        outcome = self._run(config.stack, remaining_fuel(config), 0)
        if isinstance(outcome, str):
            raise IllegalOpError(f"{self.name}: {outcome}")
        return outcome

    def __repr__(self) -> str:
        return f"Op({self.name})"


# ---------------------------------------------------------------------------
# Running quotations
# ---------------------------------------------------------------------------


def step_raw(
    op: Op, stack: tuple[Value, ...], budget: int, depth: int
) -> tuple[tuple[Value, ...], int] | str:
    """Apply one op to a bare stack under a budget, or return why it cannot."""
    if budget <= 0:
        return "out of fuel"
    if _quotes(op, stack):
        return (*stack, Sym(op.name)), 1
    if op._run is not None:
        return op._run(stack, budget, depth)
    config = Config(stack)
    reason = op._guard(config)
    if reason is not None:
        return reason
    return op._apply(config)


@lru_cache(maxsize=1 << 17)
def run_body(
    body: tuple, stack: tuple, budget: int, depth: int = 0
) -> tuple[tuple[Value, ...], int] | str:
    """Run a quotation body against a stack, or return the reason it stopped.

    A body item that is a `Sym` names an op to apply; anything else is a value
    to push. Cached because `guard` and `apply` ask the same question of the
    same config, and the answer is expensive for a combinator.
    """
    if depth > MAX_QUOTATION_DEPTH:
        return "quotation nested too deep"
    cost = 0
    for item in body:
        if cost >= budget:
            return "out of fuel"
        if isinstance(item, Sym):
            op = OP_BY_NAME.get(item.name)
            if op is None:
                return f"unknown op {item.name!r}"
            outcome = step_raw(op, stack, budget - cost, depth + 1)
            if isinstance(outcome, str):
                # Name the op that refused, so a failure inside a quotation is
                # not reported as though the combinator itself was misapplied.
                return (
                    outcome if outcome.startswith(op.name) else f"{op.name}: {outcome}"
                )
            stack, spent = outcome
            cost += spent
        else:
            stack = (*stack, item)
            cost += 1
    return stack, cost


def _needs(config: Config, count: int) -> str | None:
    if len(config.stack) < count:
        return f"needs {count} value(s), stack has {len(config.stack)}"
    return None


def _literal(name: str, value: Value) -> Op:
    return Op(
        name=name,
        group="literals",
        effect=f"( -- {name} )",
        _guard=lambda config: None,
        _apply=lambda config: ((*config.stack, value), 1),
    )


def _int_binary(
    name: str,
    combine: Callable[[int, int], int],
    extra_guard: Callable[[int, int], str | None] | None = None,
) -> Op:
    def guard(config: Config) -> str | None:
        if (reason := _needs(config, 2)) is not None:
            return reason
        left, right = config.peek(1), config.peek(0)
        if not isinstance(left, Int) or not isinstance(right, Int):
            return "top two values are not both integers"
        if extra_guard is not None:
            return extra_guard(left.value, right.value)
        return None

    def apply(config: Config) -> tuple[tuple[Value, ...], int]:
        left, right = config.peek(1), config.peek(0)
        result = Int(combine(left.value, right.value))
        return (*config.stack[:-2], result), 1

    return Op(name, "arithmetic", "( a b -- c )", guard, apply)


def _plumbing(name: str, effect: str, count: int, rearrange) -> Op:
    return Op(
        name=name,
        group="plumbing",
        effect=effect,
        _guard=lambda config: _needs(config, count),
        _apply=lambda config: (rearrange(config), 1),
    )


def _cons_guard(config: Config) -> str | None:
    if (reason := _needs(config, 2)) is not None:
        return reason
    if not isinstance(config.peek(1), Lst):
        return "second value is not a list"
    return None


def _cons_apply(config: Config) -> tuple[tuple[Value, ...], int]:
    target, element = config.peek(1), config.peek(0)
    return (*config.stack[:-2], Lst((element, *target.items))), 1


def _uncons_guard(config: Config) -> str | None:
    if (reason := _needs(config, 1)) is not None:
        return reason
    top = config.peek(0)
    if not isinstance(top, Lst):
        return "top is not a list"
    if not top.items:
        return "list is empty"
    return None


def _uncons_apply(config: Config) -> tuple[tuple[Value, ...], int]:
    top = config.peek(0)
    head, tail = top.items[0], Lst(top.items[1:])
    return (*config.stack[:-1], tail, head), 1


def _null_guard(config: Config) -> str | None:
    if (reason := _needs(config, 1)) is not None:
        return reason
    if not isinstance(config.peek(0), Lst):
        return "top is not a list"
    return None


def _eq_apply(config: Config) -> tuple[tuple[Value, ...], int]:
    result = Bool(same(config.peek(1), config.peek(0)))
    return (*config.stack[:-2], result), 1


def _lt_guard(config: Config) -> str | None:
    if (reason := _needs(config, 2)) is not None:
        return reason
    if not isinstance(config.peek(1), Int) or not isinstance(config.peek(0), Int):
        return "top two values are not both integers"
    return None


def _nonzero_divisor(left: int, right: int) -> str | None:
    return "division by zero" if right == 0 else None


OPS: tuple[Op, ...] = (
    # literals
    _literal("0", ZERO),
    _literal("1", ONE),
    _literal("nil", NIL),
    _literal("true", TRUE),
    _literal("false", FALSE),
    # plumbing
    _plumbing("dup", "( a -- a a )", 1, lambda c: (*c.stack, c.peek(0))),
    _plumbing(
        "swap", "( a b -- b a )", 2, lambda c: (*c.stack[:-2], c.peek(0), c.peek(1))
    ),
    _plumbing("drop", "( a -- )", 1, lambda c: c.stack[:-1]),
    _plumbing("over", "( a b -- a b a )", 2, lambda c: (*c.stack, c.peek(1))),
    # structure
    Op("cons", "structure", "( lst a -- lst' )", _cons_guard, _cons_apply),
    Op("uncons", "structure", "( lst -- tail head )", _uncons_guard, _uncons_apply),
    Op(
        "null?",
        "structure",
        "( lst -- bool )",
        _null_guard,
        lambda c: ((*c.stack[:-1], Bool(not c.peek(0).items)), 1),
    ),
    # arithmetic
    _int_binary("add", lambda a, b: a + b),
    _int_binary("sub", lambda a, b: a - b),
    _int_binary("mul", lambda a, b: a * b),
    _int_binary("div", lambda a, b: a // b, extra_guard=_nonzero_divisor),
    # predicates
    Op(
        "eq?",
        "predicates",
        "( a b -- bool )",
        lambda c: _needs(c, 2),
        _eq_apply,
    ),
    Op(
        "lt?",
        "predicates",
        "( a b -- bool )",
        _lt_guard,
        lambda c: ((*c.stack[:-2], Bool(c.peek(1).value < c.peek(0).value)), 1),
    ),
)

# Name -> op. The *order* of OPS is the stable action ordering; this mapping is
# only ever used for lookup by name, never iterated.
OP_BY_NAME: dict[str, Op] = {op.name: op for op in OPS}

# ---------------------------------------------------------------------------
# Quotation and combinator builders
# ---------------------------------------------------------------------------


def _symbol(name: str) -> Op:
    """Push an op's name as a value. This is how a program becomes data."""
    return Op(
        name=f"'{name}",
        group="symbols",
        effect=f"( -- {name} )",
        _guard=lambda config: None,
        _apply=lambda config: ((*config.stack, Sym(name)), 1),
    )


def _quotations(stack: tuple, count: int) -> tuple | str:
    """The top `count` values, checked to be quotations (lists)."""
    if len(stack) < count:
        return f"needs {count} value(s), stack has {len(stack)}"
    top = stack[-count:]
    for value in top:
        if not isinstance(value, Lst):
            return "expects a quotation on top"
    return top


def _run_i(stack: tuple, budget: int, depth: int):
    got = _quotations(stack, 1)
    if isinstance(got, str):
        return got
    (body,) = got
    outcome = run_body(body.items, stack[:-1], budget - 1, depth + 1)
    if isinstance(outcome, str):
        return outcome
    return outcome[0], outcome[1] + 1


def _run_dip(stack: tuple, budget: int, depth: int):
    got = _quotations(stack, 1)
    if isinstance(got, str):
        return got
    if len(stack) < 2:
        return f"needs 2 value(s), stack has {len(stack)}"
    (body,) = got
    kept = stack[-2]
    outcome = run_body(body.items, stack[:-2], budget - 1, depth + 1)
    if isinstance(outcome, str):
        return outcome
    return (*outcome[0], kept), outcome[1] + 1


def _one_result(outcome, want_bool: bool = False):
    """A body used as a function must leave exactly one value."""
    if isinstance(outcome, str):
        return outcome
    stack, cost = outcome
    if len(stack) != 1:
        return f"body must leave exactly one value, left {len(stack)}"
    if want_bool and not isinstance(stack[0], Bool):
        return "body must leave a boolean"
    return stack[0], cost


def _run_map(stack: tuple, budget: int, depth: int):
    got = _quotations(stack, 1)
    if isinstance(got, str):
        return got
    if len(stack) < 2 or not isinstance(stack[-2], Lst):
        return "second value is not a list"
    (body,) = got
    source = stack[-2]
    out, cost = [], 1
    for item in source.items:
        result = _one_result(run_body(body.items, (item,), budget - cost, depth + 1))
        if isinstance(result, str):
            return result
        value, spent = result
        out.append(value)
        cost += spent
        if cost >= budget:
            return "out of fuel"
    return (*stack[:-2], Lst(tuple(out))), cost


def _run_filter(stack: tuple, budget: int, depth: int):
    got = _quotations(stack, 1)
    if isinstance(got, str):
        return got
    if len(stack) < 2 or not isinstance(stack[-2], Lst):
        return "second value is not a list"
    (body,) = got
    source = stack[-2]
    out, cost = [], 1
    for item in source.items:
        result = _one_result(
            run_body(body.items, (item,), budget - cost, depth + 1), want_bool=True
        )
        if isinstance(result, str):
            return result
        keep, spent = result
        if keep.value:
            out.append(item)
        cost += spent
        if cost >= budget:
            return "out of fuel"
    return (*stack[:-2], Lst(tuple(out))), cost


def _run_fold(stack: tuple, budget: int, depth: int):
    got = _quotations(stack, 1)
    if isinstance(got, str):
        return got
    if len(stack) < 3 or not isinstance(stack[-3], Lst):
        return "third value is not a list"
    (body,) = got
    source, acc = stack[-3], stack[-2]
    cost = 1
    for item in source.items:
        result = _one_result(
            run_body(body.items, (acc, item), budget - cost, depth + 1)
        )
        if isinstance(result, str):
            return result
        acc, spent = result
        cost += spent
        if cost >= budget:
            return "out of fuel"
    return (*stack[:-3], acc), cost


def _run_ifte(stack: tuple, budget: int, depth: int):
    got = _quotations(stack, 3)
    if isinstance(got, str):
        return got
    condition, then, otherwise = got
    rest = stack[:-3]
    outcome = run_body(condition.items, rest, budget - 1, depth + 1)
    if isinstance(outcome, str):
        return outcome
    tested, cost = outcome
    if not tested or not isinstance(tested[-1], Bool):
        return "condition must leave a boolean on top"
    chosen = then if tested[-1].value else otherwise
    outcome = run_body(chosen.items, rest, budget - cost - 1, depth + 1)
    if isinstance(outcome, str):
        return outcome
    return outcome[0], outcome[1] + cost + 1


def _combinator(name: str, effect: str, runner: Runner) -> Op:
    return Op(
        name=name,
        group="combinators",
        effect=effect,
        _guard=lambda config: None,
        _apply=lambda config: (config.stack, 1),
        _run=runner,
    )


def _quotation_op(name: str, effect: str, runner: Runner) -> Op:
    return Op(
        name=name,
        group="quotation",
        effect=effect,
        _guard=lambda config: None,
        _apply=lambda config: (config.stack, 1),
        _run=runner,
    )


def _cat_guard(config: Config) -> str | None:
    if (reason := _needs(config, 2)) is not None:
        return reason
    if not isinstance(config.peek(1), Lst) or not isinstance(config.peek(0), Lst):
        return "both values must be quotations"
    return None


FIRST_ORDER_OPS: tuple[Op, ...] = OPS

# The ops that do something, as opposed to naming something.
CORE_OPS: tuple[Op, ...] = (
    *FIRST_ORDER_OPS,
    # plumbing completed -- dip needs a quotation, so it could not exist before
    Op(
        "dip",
        "plumbing",
        "( x q -- ... x )",
        lambda config: None,
        lambda config: (config.stack, 1),
        _run_dip,
    ),
    # quotation
    Op(
        "quote",
        "quotation",
        "( a -- (a) )",
        lambda config: _needs(config, 1),
        lambda config: ((*config.stack[:-1], Lst((config.peek(0),))), 1),
    ),
    Op(
        "cat",
        "quotation",
        "( (a) (b) -- (a b) )",
        _cat_guard,
        lambda config: (
            (*config.stack[:-2], Lst(config.peek(1).items + config.peek(0).items)),
            1,
        ),
    ),
    _quotation_op("i", "( q -- ... )", _run_i),
    # combinators
    _combinator("map", "( lst q -- lst' )", _run_map),
    _combinator("filter", "( lst q -- lst' )", _run_filter),
    _combinator("fold", "( lst acc q -- x )", _run_fold),
    _combinator("ifte", "( qc qt qf -- ... )", _run_ifte),
)

# One symbol per core op -- *including* the combinators and `i`. Partial
# coverage would be a quiet restriction to first-order quotations, under which
# nothing can call itself, every program terminates, and the fuel guard could
# never fire. Self-application is the point: it is what lets a program name
# itself, and what makes the budget load-bearing rather than decorative.
OPS = (*CORE_OPS, *(_symbol(op.name) for op in CORE_OPS))

# Lookup for running quotation bodies. Deliberately the full-52 names only: the
# brackets never appear as symbols in a body, since nothing pushes `Sym("[")`.
OP_BY_NAME = {op.name: op for op in OPS}


def _close_mark_guard(config: Config) -> str | None:
    return None if quoting(config.stack) else "no open quotation"


def _close_mark_apply(config: Config) -> tuple[tuple[Value, ...], int]:
    stack = config.stack
    at = max(i for i, item in enumerate(stack) if isinstance(item, Mark))
    return (*stack[:at], Lst(stack[at + 1 :])), 1


OPEN_MARK = Op(
    "[",
    "brackets",
    "( -- MARK )",
    lambda config: None,
    lambda config: ((*config.stack, MARK), 1),
)
CLOSE_MARK = Op(
    "]",
    "brackets",
    "( MARK ... -- (...) )",
    _close_mark_guard,
    _close_mark_apply,
)

# The executable ops plus structural quotation: `[ dup mul ]` builds
# `(dup mul)` at one op per quoted op, instead of a `'name cons` pair each.
MARK_OPS: tuple[Op, ...] = (*CORE_OPS, OPEN_MARK, CLOSE_MARK)

DEFAULT_BASIS = "executable-26"
BASIS_OPS: dict[str, tuple[Op, ...]] = {
    "executable-26": CORE_OPS,
    "full-52": OPS,
    "mark-28": MARK_OPS,
}


def ops_for_basis(name: str = DEFAULT_BASIS) -> tuple[Op, ...]:
    """The op tuple for a named basis, in its stable action order."""
    ops = BASIS_OPS.get(name)
    if ops is None:
        raise MachineError(
            f"unknown basis {name!r}; choose one of: {', '.join(BASIS_OPS)}"
        )
    return ops


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def legal_mask(config: Config, ops: tuple[Op, ...] = OPS) -> tuple[bool, ...]:
    """One bool per op, in basis order. The action mask a policy head consumes."""
    return tuple(op.guard(config) is None for op in ops)


def legal_ops(config: Config, ops: tuple[Op, ...] = OPS) -> tuple[Op, ...]:
    """The ops applicable in this config, in basis order."""
    return tuple(op for op in ops if op.guard(config) is None)


def step(config: Config, op: Op) -> Config:
    """Apply `op`, or raise IllegalOpError with the guard's reason."""
    reason = op.guard(config)
    if reason is not None:
        raise IllegalOpError(f"{op.name}: {reason}")
    stack, cost = op.apply(config)
    return replace(config, stack=stack, fuel=config.fuel + cost)


def parse_stack(text: str, *, basis: str | None = None) -> tuple[Value, ...]:
    """The inverse of `format_stack`: read a printed stack back into values.

    Input data is given, not built -- only 0 and 1 are literals, so there is no
    program that produces the stack `[ 3 5 ]`. Anything that is not a number,
    `nil`, `true`, `false` or a parenthesised list is an op name, and becomes a
    `Sym`, which is what makes a printed quotation readable back in.

    Under ``basis="mark-28"`` the token `MARK` reads back as the open-quotation
    marker; under any other basis it stays the symbol `MARK`.
    """
    if basis is not None:
        ops_for_basis(basis)
    marks = basis == "mark-28"

    def read_value(token: str, tokens: list[str]) -> Value:
        if token == "(":
            items = []
            while True:
                if not tokens:
                    raise MachineError("unterminated list")
                head = tokens.pop(0)
                if head == ")":
                    return Lst(tuple(items))
                items.append(read_value(head, tokens))
        if token == ")":
            raise MachineError("unexpected )")
        if token == "nil":
            return NIL
        if token == "true":
            return TRUE
        if token == "false":
            return FALSE
        if marks and token == "MARK":
            return MARK
        try:
            return Int(int(token))
        except ValueError:
            return Sym(token)

    spaced = text.replace("(", " ( ").replace(")", " ) ")
    spaced = spaced.replace("[", " ").replace("]", " ")
    tokens = spaced.split()
    stack: list[Value] = []
    while tokens:
        stack.append(read_value(tokens.pop(0), tokens))
    return tuple(stack)


def parse_program(text: str, ops: tuple[Op, ...] = OPS) -> tuple[Op, ...]:
    """Whitespace-separated op names. Search never needs this; humans do."""
    by_name = {op.name: op for op in ops}
    program = []
    for token in text.split():
        op = by_name.get(token)
        if op is None:
            raise MachineError(f"unknown op {token!r}")
        program.append(op)
    return tuple(program)


def run(program: tuple[Op, ...], config: Config | None = None) -> Iterator[Config]:
    """Yield the config after each op. Raises IllegalOpError on a guarded step."""
    current = config if config is not None else Config()
    for op in program:
        current = step(current, op)
        yield current


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _trace_line(label: str, config: Config, ops: tuple[Op, ...] = OPS) -> str:
    legal = sum(legal_mask(config, ops))
    return (
        f"{label:<8}  {format_stack(config):<34}"
        f"  fuel {config.fuel:>3}  legal {legal:>2}/{len(ops)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.stack_machine",
        description="Run a concatenative program and trace the machine.",
    )
    parser.add_argument("program", help='e.g. "nil 1 cons uncons"')
    parser.add_argument(
        "--fuel-limit",
        type=int,
        default=None,
        help="refuse every op once this much fuel is spent",
    )
    parser.add_argument(
        "--basis",
        choices=tuple(BASIS_OPS),
        default="full-52",
        help="which op set the program is read against",
    )
    args = parser.parse_args(argv)
    ops = ops_for_basis(args.basis)

    try:
        program = parse_program(args.program, ops)
    except MachineError as error:
        print(f"error: {error}", file=sys.stderr)
        print(f"known ops: {' '.join(op.name for op in ops)}", file=sys.stderr)
        return 2

    config = Config(fuel_limit=args.fuel_limit)
    print("top of stack on the right")
    print(f"{'op':<8}  {'stack':<34}  {'fuel':<8}  legal")
    print(_trace_line("(start)", config, ops))
    for op in program:
        reason = op.guard(config)
        if reason is not None:
            print(f"{op.name:<8}  ILLEGAL: {reason}")
            return 1
        config = step(config, op)
        print(_trace_line(op.name, config, ops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
