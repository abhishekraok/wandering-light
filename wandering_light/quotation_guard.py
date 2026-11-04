"""A search-side guard for open quotations under the ``mark-28`` basis.

Under an open ``[`` the machine captures every op as a symbol, so every
``Op.guard`` returns None and all 28 ops are legal where 13 were. The machine is
right to do that -- a quotation is data until something runs it -- but a search
gains nothing from building a body that must fail whenever it is run. This
module reads the accumulating body *as a program* and says when that is so.

The machine is unchanged. Nothing here is consulted by ``stack_machine``; a
search asks ``blocked`` and treats a reason as a pruned transition.

What a body holds. Literals (``0 1 nil true false``) are not captured: they push
their value, so a body holds values as well as symbols. A closed nested
quotation is just a list value, and its contents are not judged here --
``[ [ 0 null? ] drop ]`` is legal, because nothing need ever run the inner one.

What "unrunnable" means. A body is unrunnable only when it must fail on *any*
stack: an op meets a value the body itself put there, of a type the op never
accepts. Values the consumer supplies are unconstrained. Once code runs (``i dip
map filter fold ifte``) nothing more is known about the stack, so nothing after
that point is judged. That keeps the verdict sound -- no witness is ever
blocked -- at the cost of missing failures a fuller inference would catch.

Both outputs of ``body_effect`` are monotone under extension: appending an item
never turns an unrunnable body runnable, and never lowers ``reach``.
"""

from __future__ import annotations

from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, Value

# What ``fold`` hands its body: the accumulator and the element. ``map`` and
# ``filter`` hand one. Recorded for callers comparing a body's reach against
# what its consumer will supply; ``blocked`` does not prune on reach.
FIXED_SUPPLY = 2

MODES = ("off", "body", "close", "both")

# Abstract value types. ANY is an unknown: it meets every requirement.
INT, BOOL, LIST, ANY = "int", "bool", "list", "any"

# Required types, top of stack first.
_REQUIRES: dict[str, tuple[str, ...]] = {
    "add": (INT, INT),
    "sub": (INT, INT),
    "mul": (INT, INT),
    "div": (INT, INT),
    "lt?": (INT, INT),
    "eq?": (ANY, ANY),
    "null?": (LIST,),
    "uncons": (LIST,),
    "cons": (ANY, LIST),
    "quote": (ANY,),
    "cat": (LIST, LIST),
    "i": (LIST,),
    "dip": (LIST, ANY),
    "map": (LIST, LIST),
    "filter": (LIST, LIST),
    "fold": (LIST, ANY, LIST),
    "ifte": (LIST, LIST, LIST),
}

# Results, bottom first, for ops whose output does not depend on their inputs.
_RESULTS: dict[str, tuple[str, ...]] = {
    "add": (INT,),
    "sub": (INT,),
    "mul": (INT,),
    "div": (INT,),
    "lt?": (BOOL,),
    "eq?": (BOOL,),
    "null?": (BOOL,),
    "uncons": (ANY, ANY),
    "cons": (LIST,),
    "quote": (LIST,),
    "cat": (LIST,),
}

# Rearrangements: how many values each pops, and which popped values (by index,
# top of stack first) it pushes back, bottom first.
_PLUMBING: dict[str, tuple[int, tuple[int, ...]]] = {
    "dup": (1, (0, 0)),
    "swap": (2, (0, 1)),
    "drop": (1, ()),
    "over": (2, (1, 0, 1)),
}

_RUNS_CODE = frozenset({"i", "dip", "map", "filter", "fold", "ifte"})


def _type_of(value: Value) -> str:
    if isinstance(value, Int):
        return INT
    if isinstance(value, Bool):
        return BOOL
    if isinstance(value, Lst):
        return LIST
    return ANY


def body_effect(items: tuple[Value, ...]) -> tuple[bool, int]:
    """``(runnable, reach)`` for a quotation body.

    ``runnable`` is False only when the body must fail on any stack. ``reach``
    is how many values the body pops beyond what it pushed, counted until code
    first runs: ``[ add add add`` reaches 4.
    """
    known: list[str] = []  # types the body pushed, bottom first
    reach = 0
    for item in items:
        if not isinstance(item, Sym):
            # A literal, or a closed nested quotation: a value pushed.
            known.append(_type_of(item))
            continue
        name = item.name
        if name in _PLUMBING:
            count, pushed = _PLUMBING[name]
        elif name in _REQUIRES:
            count, pushed = len(_REQUIRES[name]), ()
        else:
            return False, reach  # the machine refuses an unknown op outright
        popped: list[str] = []  # top first
        for _ in range(count):
            if known:
                popped.append(known.pop())
            else:
                popped.append(ANY)  # supplied by the consumer: unconstrained
                reach += 1
        if name in _PLUMBING:
            known.extend(popped[index] for index in pushed)
            continue
        for need, got in zip(_REQUIRES[name], popped, strict=True):
            if need != ANY and got not in (ANY, need):
                return False, reach
        if name in _RUNS_CODE:
            # Code runs: nothing more is known about the stack, so nothing
            # after this point can be judged, and reach stops counting.
            return True, reach
        known.extend(_RESULTS[name])
    return True, reach


def open_body(stack: tuple[Value, ...]) -> tuple[tuple[Value, ...], int] | None:
    """The items after the innermost ``Mark`` and how many values sit below it.

    None when no quotation is open.
    """
    for index in range(len(stack) - 1, -1, -1):
        if isinstance(stack[index], Mark):
            return tuple(stack[index + 1 :]), index
    return None


def blocked(stack: tuple[Value, ...], remaining: int, mode: str = "both") -> str | None:
    """Why a search should not extend this stack, or None.

    ``remaining`` is how many ops the search may still apply. Under ``close``
    or ``both``, more open marks than that cannot all be closed in time:
    ``"close"``. Under ``body`` or ``both``, an innermost body that must fail
    whenever it runs: ``"type"``. Reach is reported by ``body_effect`` but never
    pruned on, since ``cat`` can widen a body in bulk.
    """
    if mode not in MODES:
        raise ValueError(f"unknown guard mode {mode!r}; choose one of: {MODES}")
    if mode == "off":
        return None
    opened = open_body(stack)
    if opened is None:
        return None
    if mode in ("close", "both"):
        marks = sum(1 for value in stack if isinstance(value, Mark))
        if marks > remaining:
            return "close"
    if mode in ("body", "both"):
        runnable, _ = body_effect(opened[0])
        if not runnable:
            return "type"
    return None
