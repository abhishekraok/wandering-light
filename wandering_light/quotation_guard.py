"""A guard for the states inside an open quotation, where the machine has none.

Under an open ``[`` the ``mark-28`` machine captures every op as a symbol rather
than applying it, so ``Op.guard`` returns ``None`` for all of them: on a typical
data stack 13 of 28 ops are legal, and 28 of 28 once a ``MARK`` is on top. That
is the mechanism behind ``mark-28``'s ``b_eff`` of 23.11 against
``executable-26``'s 13.38, and it fires exactly where a higher-order program
must pass -- 87.5% of v3 witnesses build a quotation.

Nothing here changes the machine. Capture stays as it is, every substrate
measurement stands, and this is consulted by *search* when it decides whether a
successor state is worth keeping. Two checks:

**type** -- an op meets a value whose type it can never accept. ``0 null?`` is
dead because ``null?`` wants a list; ``0 i`` is dead because ``i`` wants a
quotation. Only values the body itself put on the stack are judged; anything the
consumer will supply is unknown and therefore unconstrained.

**reach** -- how far the body dips below its own contribution. Computed and
reported, but **not pruned on**: there is no sound bound. A closed quotation can
be *widened* afterwards, and ``cat`` prepends a whole list in a single op, so
widening is not paid for one value at a time. From ``[ (1 1 1 1 1 1 1) ]`` the
prefix ``[ add add add add add add`` reaches seven deep with three ops left, and
``] cat i`` evaluates it to ``7``. ``dup cat`` then doubles a list per op, so no
bound linear in the remaining budget survives. Kept as a diagnostic because the
rate is worth knowing; see the report.

Both are monotone under extension: appending an op can only add a type error and
can only increase reach, so a body that fails either can never be repaired. That
is what makes them safe to *prune* rather than merely deprioritise, and it is the
whole difference between this and a prior over bodies.

What is deliberately not checked: a *closed* nested quotation. ``[ [ 0 null? ]
drop ]`` builds a body that cannot run and then throws it away, which is legal,
so a type error inside a completed inner quotation is not fatal to the outer one.
"""

from __future__ import annotations

from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, Value

INT, BOOL, LST, UNK = "int", "bool", "lst", "?"

# name -> (arity, {offset_from_top: required type}, produced types bottom-first)
# ``None`` produced marks an op that rearranges what it popped, or one that runs
# code and so leaves something this pass cannot know.
RULES: dict[str, tuple[int, dict[int, str], list[str] | None]] = {
    "0": (0, {}, [INT]),
    "1": (0, {}, [INT]),
    "nil": (0, {}, [LST]),
    "true": (0, {}, [BOOL]),
    "false": (0, {}, [BOOL]),
    "dup": (1, {}, None),
    "swap": (2, {}, None),
    "drop": (1, {}, None),
    "over": (2, {}, None),
    "cons": (2, {1: LST}, [LST]),
    "uncons": (1, {0: LST}, [UNK, UNK]),
    "null?": (1, {0: LST}, [BOOL]),
    "add": (2, {0: INT, 1: INT}, [INT]),
    "sub": (2, {0: INT, 1: INT}, [INT]),
    "mul": (2, {0: INT, 1: INT}, [INT]),
    "div": (2, {0: INT, 1: INT}, [INT]),
    "eq?": (2, {}, [BOOL]),
    "lt?": (2, {0: INT, 1: INT}, [BOOL]),
    "quote": (1, {}, [LST]),
    "cat": (2, {0: LST, 1: LST}, [LST]),
    # These run a quotation. Their arguments are still typed; their result is not.
    "i": (1, {0: LST}, None),
    "dip": (2, {0: LST}, None),
    "map": (2, {0: LST, 1: LST}, None),
    "filter": (2, {0: LST, 1: LST}, None),
    "fold": (3, {0: LST, 2: LST}, None),
    "ifte": (3, {0: LST, 1: LST, 2: LST}, None),
}
RUNS_CODE = frozenset({"i", "dip", "map", "filter", "fold", "ifte"})
PLUMBING = frozenset({"dup", "swap", "drop", "over"})

#: ``fold`` hands its body two values, which is the most any fixed-arity
#: consumer supplies. Reported against ``reach`` as a diagnostic; nothing prunes
#: on it, because ``cat`` can widen a body in bulk (see the module docstring).
FIXED_SUPPLY = 2

MODES = ("off", "body", "close", "both")

#: The five ops that are *not* captured under a MARK: they push their value for
#: real, so a body carries data as well as symbols.
LITERALS_BY_NAME: dict[str, Value] = {
    "0": Int(0),
    "1": Int(1),
    "nil": Lst(()),
    "true": Bool(True),
    "false": Bool(False),
}


def type_of(value: Value) -> str:
    if isinstance(value, Int):
        return INT
    if isinstance(value, Bool):
        return BOOL
    if isinstance(value, Lst):
        return LST
    return UNK


class _Frame:
    """One quotation's worth of abstract stack."""

    __slots__ = ("blind", "known", "reach")

    def __init__(self) -> None:
        self.known: list[str] = []  # types this body put there, top last
        self.reach = 0  # values a consumer has to supply
        self.blind = False  # code has run; the depth below is unknowable

    def pop(self) -> str:
        if self.known:
            return self.known.pop()
        if not self.blind:
            self.reach += 1
        return UNK


def body_effect(items: tuple[Value, ...]) -> tuple[bool, int]:
    """``(runnable, reach)`` for the body of the innermost open quotation.

    ``runnable`` is False only when executing the body must fail whatever stack
    it is handed, so a False here is a fact about the body and not a guess.
    """
    frames = [_Frame()]
    for value in items:
        top = frames[-1]
        if not isinstance(value, Sym):
            top.known.append(type_of(value))
            continue
        name = value.name
        if name == "[":
            frames.append(_Frame())
            continue
        if name == "]":
            if len(frames) > 1:
                frames.pop()
                frames[-1].known.append(LST)
            continue
        arity, checks, produced = RULES[name]
        for offset, want in checks.items():
            if offset >= len(top.known):
                continue  # the consumer supplies it, so any type is still open
            found = top.known[len(top.known) - 1 - offset]
            if found not in (want, UNK):
                return False, frames[0].reach
        popped = [top.pop() for _ in range(arity)][::-1]  # bottom-first
        if name in PLUMBING:
            if name == "dup":
                top.known.extend([popped[0], popped[0]])
            elif name == "swap":
                top.known.extend([popped[1], popped[0]])
            elif name == "over":
                top.known.extend([popped[0], popped[1], popped[0]])
            # drop pushes nothing back
        elif name in RUNS_CODE:
            top.known.clear()
            top.blind = True
        else:
            assert produced is not None
            top.known.extend(produced)
    return True, frames[0].reach


def open_body(stack: tuple[Value, ...]) -> tuple[tuple[Value, ...], int] | None:
    """The innermost unclosed body and how many values sit below its ``MARK``."""
    marks = [index for index, value in enumerate(stack) if isinstance(value, Mark)]
    if not marks:
        return None
    innermost = max(marks)
    return stack[innermost + 1 :], innermost


def blocked(stack: tuple[Value, ...], remaining: int, mode: str = "both") -> str | None:
    """Why this state cannot lead anywhere, or None to keep it.

    ``remaining`` is how many ops the search may still spend, which is what the
    ``close`` check needs: a state carrying more open marks than that has no way
    back to a mark-free stack, and every corpus target is mark-free.
    """
    if mode == "off":
        return None
    if mode not in MODES:
        raise ValueError(f"unknown guard mode {mode!r}; choose one of: {MODES}")
    marks = sum(1 for value in stack if isinstance(value, Mark))
    if not marks:
        return None
    if mode in ("close", "both") and marks > remaining:
        return "close"
    if mode in ("body", "both"):
        found = open_body(stack)
        assert found is not None
        runnable, _reach = body_effect(found[0])
        if not runnable:
            return "type"
    return None
