"""A search-side predicate over partially built quotations on ``mark-28``.

Under an open ``[`` the machine captures every executable op as a symbol, so
``Op.guard`` returns ``None`` for all of them and the legal set widens from 13
ops to 28. The machine is right to do so -- a quotation is data until it runs
-- but search is then free to accumulate bodies such as ``[ 0 null?`` that no
stack can ever run. This module reads the accumulating body *as a program* and
says, without touching the machine, whether it is still worth extending.

Two facts about the body make this a small abstract interpretation:

* Literals (``0 1 nil true false``) are not captured: they push their value. So
  a body holds values as well as symbols, and a ``Sym`` item is an op to apply
  while anything else is a value the body itself put on the stack.
* A closed nested quotation is a list value whose contents are not judged.
  ``[ [ 0 null? ] drop ]`` is legal: the inner body is never run by the outer.

The interpreter tracks the *types* of the values the body pushed. Values the
consumer supplies -- whatever sits on the stack when the quotation is finally
run -- are unconstrained, so a pop that reaches below the body's own pushes
yields an unknown value and is counted toward ``reach``. Once an op that runs
code has fired, nothing is known about the stack below and ``reach`` stops
counting. Both outputs are monotone under extension: once unrunnable a body
stays unrunnable, and reach never decreases.

Only *types* are judged. ``[ 0 div`` and ``[ nil uncons`` also fail on every
stack, but through a value rather than a type, and that distinction is left to
the machine.
"""

from __future__ import annotations

from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, Value

__all__ = [
    "FIXED_SUPPLY",
    "MODES",
    "blocked",
    "body_effect",
    "open_body",
]

#: The guard modes ``blocked`` accepts, in order of increasing strictness.
MODES: tuple[str, ...] = ("off", "body", "close", "both")

#: How many values ``fold`` hands a body: the accumulator and one element. The
#: other combinators hand one (``map``, ``filter``) or the whole stack (``i``,
#: ``dip``, ``ifte``). Reported beside ``reach`` by callers; never pruned on.
FIXED_SUPPLY = 2

# Abstract types. ``_ANY`` is a value about which nothing is known.
_INT, _BOOL, _LIST, _SYM, _MARK, _ANY = "int", "bool", "list", "sym", "mark", "any"

# Required input types, top of stack first, for every op that demands one.
# Ops absent here accept anything (``dup swap drop over eq? quote``) or are
# handled specially below.
_NEEDS: dict[str, tuple[str, ...]] = {
    "add": (_INT, _INT),
    "sub": (_INT, _INT),
    "mul": (_INT, _INT),
    "div": (_INT, _INT),
    "lt?": (_INT, _INT),
    "null?": (_LIST,),
    "uncons": (_LIST,),
    "i": (_LIST,),
    "cons": (_ANY, _LIST),
    "cat": (_LIST, _LIST),
    "dip": (_LIST, _ANY),
    "map": (_LIST, _LIST),
    "filter": (_LIST, _LIST),
    "fold": (_LIST, _ANY, _LIST),
    "ifte": (_LIST, _LIST, _LIST),
    "eq?": (_ANY, _ANY),
    "quote": (_ANY,),
    "dup": (_ANY,),
    "drop": (_ANY,),
    "swap": (_ANY, _ANY),
    "over": (_ANY, _ANY),
}

# What a first-order op leaves, bottom to top. Plumbing and code-runners are
# handled by hand in ``_apply``.
_LEAVES: dict[str, tuple[str, ...]] = {
    "add": (_INT,),
    "sub": (_INT,),
    "mul": (_INT,),
    "div": (_INT,),
    "lt?": (_BOOL,),
    "null?": (_BOOL,),
    "eq?": (_BOOL,),
    "cons": (_LIST,),
    "quote": (_LIST,),
    "cat": (_LIST,),
    "uncons": (_ANY, _ANY),
}

_RUNS_CODE = frozenset({"i", "dip", "map", "filter", "fold", "ifte"})


def _type_of(value: Value) -> str:
    if isinstance(value, Int):
        return _INT
    if isinstance(value, Bool):
        return _BOOL
    if isinstance(value, Lst):
        return _LIST
    if isinstance(value, Sym):
        return _SYM
    if isinstance(value, Mark):
        return _MARK
    raise TypeError(f"not a machine value: {value!r}")


class _Abstract:
    """The body's own pushes, typed, over an unconstrained consumer stack."""

    __slots__ = ("opaque", "reach", "runnable", "stack")

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.opaque = False  # code has run: nothing known below ``stack``
        self.reach = 0
        self.runnable = True

    def pop(self) -> str:
        if self.stack:
            return self.stack.pop()
        if not self.opaque:
            self.reach += 1
        return _ANY

    def push(self, *types: str) -> None:
        self.stack.extend(types)

    def take(self, needs: tuple[str, ...]) -> list[str]:
        """Pop the inputs an op needs, top first, judging each type."""
        got = []
        for need in needs:
            actual = self.pop()
            if need != _ANY and actual not in (_ANY, need):
                self.runnable = False
            got.append(actual)
        return got

    def run_code(self) -> None:
        self.stack.clear()
        self.opaque = True


def _apply(state: _Abstract, name: str) -> None:
    if name.startswith("'") and len(name) > 1:
        # A symbol op from the full basis: pushes a name as a value.
        state.push(_SYM)
        return
    needs = _NEEDS.get(name)
    if needs is None:
        # ``[``, ``]`` and anything unnamed: ``run_body`` refuses these on
        # every stack, so the body can never run.
        state.runnable = False
        return
    got = state.take(needs)
    if name == "dup":
        state.push(got[0], got[0])
    elif name == "drop":
        pass
    elif name == "swap":
        top, below = got
        state.push(top, below)
    elif name == "over":
        top, below = got
        state.push(below, top, below)
    elif name in _RUNS_CODE:
        state.run_code()
        if name == "dip":
            state.push(got[1])  # ( x q -- ... x )
        elif name in ("map", "filter"):
            state.push(_LIST)  # the machine always leaves a list here
        elif name == "fold":
            state.push(_ANY)
    else:
        state.push(*_LEAVES[name])


def body_effect(items: tuple[Value, ...]) -> tuple[bool, int]:
    """``(runnable, reach)`` for a quotation body read as a program.

    ``runnable`` is ``False`` only when the body must fail on any stack: an op
    meets a value the body itself put there of a type the op never accepts.
    ``reach`` is how many values the body pops beyond what it pushed, counted
    until code first runs. The empty body is ``(True, 0)``.
    """
    state = _Abstract()
    for item in items:
        if isinstance(item, Sym):
            _apply(state, item.name)
        else:
            state.push(_type_of(item))
    return state.runnable, state.reach


def open_body(stack: tuple[Value, ...]) -> tuple[tuple[Value, ...], int] | None:
    """The items after the innermost ``Mark`` and how many values sit below it.

    ``None`` when the stack holds no mark.
    """
    for index in range(len(stack) - 1, -1, -1):
        if isinstance(stack[index], Mark):
            return stack[index + 1 :], index
    return None


def blocked(stack: tuple[Value, ...], remaining: int, mode: str = "both") -> str | None:
    """Why search should not extend this stack, or ``None``.

    ``remaining`` is how many ops the program may still place. Under ``close``
    and ``both``, more open marks than that returns ``"close"``: the quotations
    can never all be closed. Then under ``body`` and ``both``, an innermost
    body that cannot run on any stack returns ``"type"``. ``off`` and a
    mark-free stack return ``None``. Reach is reported by ``body_effect`` and
    never pruned on: ``cat`` can widen a body in bulk.
    """
    if mode not in MODES:
        choices = ", ".join(MODES)
        raise ValueError(f"unknown guard mode {mode!r}; choose one of: {choices}")
    if mode == "off":
        return None
    found = open_body(stack)
    if found is None:
        return None
    if mode in ("close", "both"):
        marks = sum(isinstance(value, Mark) for value in stack)
        if marks > remaining:
            return "close"
    if mode in ("body", "both"):
        runnable, _reach = body_effect(found[0])
        if not runnable:
            return "type"
    return None
