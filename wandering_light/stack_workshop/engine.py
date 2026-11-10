"""Compile and run a learner's program on the canonical `mark-28` machine.

Nothing here decides what an op does. Every step is one call to
``stack_machine.step_raw``; what happens inside a combinator is read off the
machine's own nested ``step_raw`` calls with a profile hook, not re-stated.
This module adds only what a workshop needs around the machine: source spans,
macro expansion, input validation, and a bounded display form of a stack.
"""

from __future__ import annotations

import difflib
import sys
import threading
from pathlib import Path
from typing import Any

from wandering_light.stack_machine import (
    MARK_OPS,
    Bool,
    Int,
    Lst,
    MachineError,
    Mark,
    Op,
    Sym,
    Value,
    format_value,
    parse_stack,
    quotation_open,
    run_body,
    step_raw,
    value_key,
)

OPS: dict[str, Op] = {op.name: op for op in MARK_OPS}
FUEL = 100_000
# Every token costs at least one fuel, so a program longer than the fuel can
# never run to its end. Refusing it at compile time also bounds what a macro
# can expand to.
MAX_PROGRAM_OPS = FUEL

MACRO_FILE = (
    Path(__file__).resolve().parents[2] / "experiments/world_behaviours/macros.txt"
)
GROUP_MARKER = "# ----"

# What a learner may type in. Outputs may be longer numbers than inputs,
# because a short program can square its way well past 10^12.
MAX_EXAMPLES = 16
MAX_VALUES = 16
MAX_ITEMS = 64
MAX_NESTING = 12
MAX_INPUT_INT = 10**12
MAX_EXPECTED_DIGITS = 512
# Stack notation is read by the machine's parser, which is quadratic in the
# number of words. Anything longer belongs in an array.
MAX_TEXT_WORDS = 10_000

# What one stack display may hold. The machine can build far more than a
# learner can type, so the display is bounded where the values are not.
STACK_SHOWN = 48
INT_BITS_SHOWN = 3400  # about 1,000 digits, well inside str()'s own limit
INSPECT_LIMIT = 350
# The canonical function's code, taken once: a profile hook recognises the
# machine's own frames by it, whatever name later points at the function.
STEP_CODE = step_raw.__code__

# The machine keeps one process-wide cache of quotation runs. Inspection has
# to clear it and then watch a run that nothing else re-warms, so everything
# in this package that steps the machine holds this lock while it does.
LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def load_macros(path: Path = MACRO_FILE) -> dict[str, dict[str, str]]:
    """``name: body`` lines, grouped by the ``# ----`` heading above them.

    A heading such as ``# ---- ranges: no recursion ... ----`` names the group
    ``ranges``; what follows the colon is a note on the group, not its name.
    The file sits outside the package, so an install without the repository
    simply has no macros.
    """
    macros: dict[str, dict[str, str]] = {}
    if not path.exists():
        return macros
    group = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(GROUP_MARKER):
            group = raw.strip("#- \t").split(":", 1)[0].strip()
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, colon, body = line.partition(":")
        name = name.strip()
        if not colon or not name or name in OPS:
            raise ValueError(f"{path.name}: not a macro definition: {raw!r}")
        macros[name] = {"body": body.strip(), "group": group}
    return macros


MACROS: dict[str, dict[str, str]] = load_macros()


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------


def _words(text: str) -> list[tuple[str, int, int]]:
    """``(word, start, end)`` for every word; brackets delimit themselves."""
    words = []
    position, size = 0, len(text)
    while position < size:
        char = text[position]
        if char == "#":
            while position < size and text[position] not in "\r\n":
                position += 1
        elif char.isspace():
            position += 1
        elif char in "[]":
            words.append((char, position, position + 1))
            position += 1
        else:
            end = position
            while end < size and not text[end].isspace() and text[end] not in "[]#":
                end += 1
            words.append((text[position:end], position, end))
            position = end
    return words


def _unknown(name: str) -> ValueError:
    message = f"Unknown word '{name}'. Use one of the 28 ops or a macro."
    if name.lstrip("-").isdigit():
        message += " Only 0 and 1 are ops; other numbers are built from them."
    elif close := difflib.get_close_matches(name, [*OPS, *MACROS], n=1):
        message += f" Did you mean '{close[0]}'?"
    return ValueError(message)


def _too_long() -> ValueError:
    return ValueError(
        f"The program expands to more than {MAX_PROGRAM_OPS} ops; every op "
        "costs fuel, so it could never finish."
    )


def _expand(name: str, active: list[str], out: list[str]) -> None:
    """Append the primitives ``name`` stands for; ``active`` is the path to it."""
    if name in OPS:
        if len(out) >= MAX_PROGRAM_OPS:
            raise _too_long()
        out.append(name)
        return
    macro = MACROS.get(name)
    if macro is None:
        raise _unknown(name)
    if name in active:
        raise ValueError(f"Recursive macro: {name}")
    body = macro["body"]
    inner = [word for word, _, _ in _words(body)] if isinstance(body, str) else body
    active.append(name)
    for word in inner:
        _expand(word, active, out)
    active.pop()


def compile_program(text: str) -> list[dict[str, Any]]:
    """Source text to primitive tokens, each tied to the word that produced it.

    A macro expands recursively, so one written word may yield many tokens.
    They all carry that word's source span and its name as ``macro``, the
    outermost one when macros nest; a primitive written directly has
    ``macro`` None.
    """
    if not isinstance(text, str):
        raise ValueError("A program is text.")
    tokens: list[dict[str, Any]] = []
    for word, start, end in _words(text):
        ops: list[str] = []
        _expand(word, [], ops)
        if len(tokens) + len(ops) > MAX_PROGRAM_OPS:
            raise _too_long()
        macro = None if word in OPS else word
        tokens.extend(
            {"op": op, "start": start, "end": end, "macro": macro} for op in ops
        )
    return tokens


# ---------------------------------------------------------------------------
# Stacks in
# ---------------------------------------------------------------------------


def _check_int(number: int, expected: bool) -> None:
    if expected:
        if abs(number) >= 10**MAX_EXPECTED_DIGITS:
            raise ValueError(
                f"An expected integer is longer than {MAX_EXPECTED_DIGITS} digits."
            )
    elif abs(number) > MAX_INPUT_INT:
        raise ValueError("An input integer is beyond 10^12.")


def _check_list(size: int, nesting: int) -> None:
    if nesting > MAX_NESTING:
        raise ValueError(f"Lists are nested more than {MAX_NESTING} deep.")
    if size > MAX_ITEMS:
        raise ValueError(f"A list has more than {MAX_ITEMS} items.")


def _from_json(item: Any, nesting: int, expected: bool) -> Value:
    # bool first: Python's True is an int, the machine's is not.
    if isinstance(item, bool):
        return Bool(item)
    if isinstance(item, int):
        _check_int(item, expected)
        return Int(item)
    if isinstance(item, list | tuple):
        _check_list(len(item), nesting + 1)
        return Lst(tuple(_from_json(inner, nesting + 1, expected) for inner in item))
    kind = "decimal number" if isinstance(item, float) else type(item).__name__
    raise ValueError(
        f"A stack holds integers, booleans and lists; found a {kind}: {item!r}"
    )


def _check_parsed(value: Value, nesting: int, expected: bool) -> None:
    if isinstance(value, Int):
        _check_int(value.value, expected)
    elif isinstance(value, Sym):
        raise ValueError(
            f"'{value.name}' is not a value. A stack holds integers, true, "
            "false, nil and ( lists )."
        )
    elif isinstance(value, Mark):
        raise ValueError("MARK is not a value an example may hold.")
    elif isinstance(value, Lst):
        _check_list(len(value.items), nesting + 1)
        for inner in value.items:
            _check_parsed(inner, nesting + 1, expected)


def stack_from(value: Any, *, expected: bool = False) -> tuple[Value, ...]:
    """A stack from a JSON array or from stack notation, bottom first.

    ``[3, [1, 2], true]`` and ``"3 (1 2) true"`` are the same stack. Anything
    a learner could not mean -- a decimal, a string inside an array, an op
    name or ``MARK`` in the text, or a value past the size limits -- is a
    ``ValueError``. ``expected`` relaxes only the integer bound.
    """
    if isinstance(value, str):
        # The canonical parser recurses per "(", so the nesting limit has to
        # be applied before it runs, not to its result.
        depth = 0
        for char in value:
            depth += (char == "(") - (char == ")")
            if depth > MAX_NESTING:
                raise ValueError(f"Lists are nested more than {MAX_NESTING} deep.")
        if len(value.replace("(", " ( ").replace(")", " ) ").split()) > MAX_TEXT_WORDS:
            raise ValueError("The stack text is too long; give it as an array.")
        try:
            stack = parse_stack(value, basis="mark-28")
        except MachineError as error:
            raise ValueError(f"Cannot read the stack: {error}") from None
        if len(stack) > MAX_VALUES:
            raise ValueError(f"A stack has more than {MAX_VALUES} values.")
        for item in stack:
            _check_parsed(item, 0, expected)
        return stack
    if isinstance(value, list | tuple):
        if len(value) > MAX_VALUES:
            raise ValueError(f"A stack has more than {MAX_VALUES} values.")
        return tuple(_from_json(item, 0, expected) for item in value)
    raise ValueError("A stack is an array of values or stack-notation text.")


# ---------------------------------------------------------------------------
# Stacks out
# ---------------------------------------------------------------------------


def _elision(hidden: int) -> dict[str, Any]:
    return {"type": "elision", "text": f"… {hidden} more"}


def _int_text(number: int) -> str:
    # str() of a very long int is slow and, past the interpreter's digit
    # limit, an error. 0.30103 is log10(2), so the digit count is approximate.
    if number.bit_length() <= INT_BITS_SHOWN:
        return str(number)
    digits = int(number.bit_length() * 0.30103) + 1
    sign = "-" if number < 0 else ""
    return f"{sign}…{abs(number) % 10**12:012d} (about {digits} digits)"


def _display(value: Value, nesting: int = 0) -> dict[str, Any]:
    if isinstance(value, Int):
        return {"type": "int", "text": _int_text(value.value)}
    if isinstance(value, Bool):
        return {"type": "bool", "text": format_value(value)}
    if isinstance(value, Sym):
        return {"type": "symbol", "text": format_value(value)}
    if isinstance(value, Mark):
        return {"type": "mark", "text": format_value(value)}
    if not value.items:
        return {"type": "list", "text": format_value(value), "items": []}
    if nesting >= MAX_NESTING:
        items = [_elision(len(value.items))]
    else:
        items = [_display(item, nesting + 1) for item in value.items[:MAX_ITEMS]]
        if len(value.items) > MAX_ITEMS:
            # Keep the head: `uncons` reads a list from the front.
            items[MAX_ITEMS - 1 :] = [_elision(len(value.items) - MAX_ITEMS + 1)]
    text = "(" + " ".join(item["text"] for item in items) + ")"
    return {"type": "list", "text": text, "items": items}


def snapshot(stack: tuple[Value, ...]) -> list[dict[str, Any]]:
    """One display dict per value, bottom first.

    A stack deeper than 48 shows its bottom value, an elision, then its top
    47. Inside a value the same idea applies at the sizes a learner can type:
    a list shows 64 items and 12 levels, and elides what the machine built
    beyond that. ``text`` is ``format_value`` for everything not elided.
    """
    if len(stack) > STACK_SHOWN:
        hidden = len(stack) - STACK_SHOWN
        kept = STACK_SHOWN - 1
        return [
            _display(stack[0]),
            _elision(hidden),
            *(_display(value) for value in stack[-kept:]),
        ]
    return [_display(value) for value in stack]


def same_stack(left: tuple[Value, ...], right: tuple[Value, ...]) -> bool:
    """Whether two stacks are the same state, by the machine's ``value_key``."""
    if len(left) != len(right):
        return False
    try:
        return tuple(map(value_key, left)) == tuple(map(value_key, right))
    except RecursionError:
        # Only a value nested past the interpreter's limit gets here, and an
        # expected output is at most MAX_NESTING deep.
        return False


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def read_examples(examples: Any) -> list[tuple[tuple[Value, ...], tuple[Value, ...]]]:
    """``[{"input", "output"}, ...]`` as ``(input stack, expected stack)`` pairs."""
    if not isinstance(examples, list | tuple) or not (
        1 <= len(examples) <= MAX_EXAMPLES
    ):
        raise ValueError(f"Give between 1 and {MAX_EXAMPLES} examples.")
    pairs = []
    for number, example in enumerate(examples, start=1):
        if (
            not isinstance(example, dict)
            or "input" not in example
            or "output" not in example
        ):
            raise ValueError(f"Example {number} needs an 'input' and an 'output'.")
        try:
            pairs.append(
                (
                    stack_from(example["input"]),
                    stack_from(example["output"], expected=True),
                )
            )
        except ValueError as error:
            raise ValueError(f"Example {number}: {error}") from None
    return pairs


def _frame(
    stack: tuple[Value, ...], fuel: int, op: str | None, error: str | None
) -> dict[str, Any]:
    return {
        "stack": snapshot(stack),
        "fuel": fuel,
        "op": op,
        "error": error,
        "quoting": quotation_open(stack),
    }


def machine_step(op: Op, stack: tuple[Value, ...], budget: int) -> tuple | str:
    """One top-level ``step_raw``. A crash inside the machine is a refusal here.

    The machine hashes and compares values recursively, so a value nested a
    few hundred deep (``quote`` in a loop) overflows the interpreter's stack
    in the middle of an op. That ends the example like any other refusal
    instead of taking the request down.
    """
    try:
        return step_raw(op, stack, budget, 0)
    except RecursionError:
        return "a value is nested too deep for the machine"


def _advance(
    tokens: list[dict[str, Any]], stack: tuple[Value, ...], spent: int = 0
) -> tuple[tuple[Value, ...], int, int, str | None]:
    """Step every token: ``(stack, fuel spent, tokens applied, error)``."""
    for applied, token in enumerate(tokens):
        outcome = machine_step(OPS[token["op"]], stack, FUEL - spent)
        if isinstance(outcome, str):
            return stack, spent, applied, f"{token['op']}: {outcome}"
        stack, cost = outcome
        spent += cost
    return stack, spent, len(tokens), None


def final_stacks(
    tokens: list[dict[str, Any]], inputs: list[tuple[Value, ...]]
) -> list[tuple[tuple[Value, ...], int]]:
    """``(stack, fuel spent)`` after the whole program, for each input.

    A program that cannot run to its end has no final stack, so a refusal is
    a ``ValueError`` naming the example.
    """
    finals = []
    with LOCK:
        for number, given in enumerate(inputs, start=1):
            stack, spent, _, error = _advance(tokens, given)
            if error is not None:
                raise ValueError(f"Example {number} stops at {error}")
            finals.append((stack, spent))
    return finals


def _run_example(
    tokens: list[dict[str, Any]],
    given: tuple[Value, ...],
    wanted: tuple[Value, ...],
    capture: bool,
) -> dict[str, Any]:
    stack, spent, error = given, 0, None
    frames = [_frame(stack, 0, None, None)]
    max_depth = len(stack)
    last = None
    for token in tokens:
        name = token["op"]
        outcome = machine_step(OPS[name], stack, FUEL - spent)
        if isinstance(outcome, str):
            error = f"{name}: {outcome}"
            last = (stack, spent, name, error)
        else:
            stack, cost = outcome
            spent += cost
            max_depth = max(max_depth, len(stack))
            last = (stack, spent, name, None)
        if capture:
            frames.append(_frame(*last))
        if error is not None:
            break
    if not capture and last is not None:
        frames.append(_frame(*last))
    return {
        "frames": frames,
        "matched": error is None and same_stack(stack, wanted),
        "error": error,
        "fuel": spent,
        "max_depth": max_depth,
        "output": snapshot(wanted),
    }


def execute(program: str, examples: Any, *, capture: bool = True) -> dict[str, Any]:
    """Run ``program`` on every example and report what the machine did.

    Each row's frame 0 is the input; every later frame is the stack after one
    token. A refusal ends that example with a frame holding the unchanged
    stack and the machine's reason, and the other examples still run. Without
    ``capture`` a row keeps its first and final frames only. Across rows,
    ``fuel`` is the total spent, as a corpus task's fuel is, and ``max_depth``
    is the deepest any example's stack got.
    """
    tokens = compile_program(program)
    pairs = read_examples(examples)
    with LOCK:
        rows = [_run_example(tokens, given, wanted, capture) for given, wanted in pairs]
    return {
        "tokens": tokens,
        "examples": rows,
        "matches": sum(row["matched"] for row in rows),
        "count": len(rows),
        "primitive_count": len(tokens),
        "fuel": sum(row["fuel"] for row in rows),
        "max_depth": max(row["max_depth"] for row in rows),
    }


# ---------------------------------------------------------------------------
# Looking inside one op
# ---------------------------------------------------------------------------


def _index(value: Any, size: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{what}' must be a whole number.")
    if not 0 <= value < size:
        raise ValueError(f"No {what} {value}; there are {size}.")
    return value


def inspect_op(
    program: str, examples: Any, step: int, example: int = 0
) -> dict[str, Any]:
    """Every nested op the machine runs while applying token ``step``.

    The tokens before ``step`` are replayed, then the machine applies that one
    token while a profile hook records each ``step_raw`` call it makes below
    the top level -- the body of a ``map`` once per item, the branch an
    ``ifte`` chose -- in the order the calls return. ``depth`` is the depth
    the machine passed to that call, and a refused call keeps its ``before``
    stack as ``after``. The run-body cache is cleared just before, because a
    cached quotation run returns its result without stepping anything.
    """
    tokens = compile_program(program)
    pairs = read_examples(examples)
    given, _ = pairs[_index(example, len(pairs), "example")]
    target = tokens[_index(step, len(tokens), "step")]

    pending: list[tuple[str, int, tuple[Value, ...]]] = []
    calls: list[tuple[str, int, tuple[Value, ...], Any]] = []
    truncated = False

    def watch(frame: Any, event: str, arg: Any) -> None:
        nonlocal truncated
        if frame.f_code is not STEP_CODE:
            return
        if event == "call":
            seen = frame.f_locals
            pending.append((seen["op"].name, seen["depth"], seen["stack"]))
        elif event == "return" and pending:
            name, depth, before = pending.pop()
            if depth == 0:
                return
            if len(calls) < INSPECT_LIMIT:
                calls.append((name, depth, before, arg))
            else:
                truncated = True

    with LOCK:
        stack, spent, applied, error = _advance(tokens[:step], given)
        if error is not None:
            raise ValueError(
                f"Step {step} is never reached: step {applied} stops at {error}"
            )
        run_body.cache_clear()
        previous = sys.getprofile()
        sys.setprofile(watch)
        try:
            outcome = machine_step(OPS[target["op"]], stack, FUEL - spent)
        finally:
            sys.setprofile(previous)

    frames = []
    for name, depth, before, result in calls:
        refused = not isinstance(result, tuple)
        frames.append(
            {
                "op": name,
                "depth": depth,
                "before": snapshot(before),
                "after": snapshot(before if refused else result[0]),
                "error": (str(result) if result is not None else "did not return")
                if refused
                else None,
            }
        )
    return {
        "frames": frames,
        "truncated": truncated,
        "error": f"{target['op']}: {outcome}" if isinstance(outcome, str) else None,
    }
