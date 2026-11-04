"""Signal House's view of the canonical ``mark-28`` machine.

Nothing here decides what an op does. Every transition is
``stack_machine.step_raw``; this module only compiles workshop text into op
names, validates the stacks a learner types, runs the canonical step over each
example, and turns the states it passes through into display dicts.

Standard library only: the policy adviser (``model.py``) is the one part of
the workshop that needs torch, and it imports it lazily.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from wandering_light import stack_machine
from wandering_light.stack_machine import (
    MARK_OPS,
    Bool,
    Int,
    Lst,
    MachineError,
    Mark,
    Sym,
    Value,
    format_value,
    parse_stack,
    quotation_open,
    run_body,
    step_raw,
    value_key,
)

OPS = {op.name: op for op in MARK_OPS}
FUEL = 100_000

MACRO_FILE = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "world_behaviours"
    / "macros.txt"
)

MAX_VALUES = 16
MAX_LIST_ITEMS = 64
MAX_NESTING = 12
MAX_INPUT_INT = 10**12
MAX_OUTPUT_DIGITS = 512
MAX_EXAMPLES = 16
DISPLAY_DEPTH = 48
MAX_INSPECT_FRAMES = 350

# step_raw and run_body's cache are process-global; inspection installs a
# profiler. Every machine operation the workshop performs takes this lock.
MACHINE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def load_macros(path: Path = MACRO_FILE) -> dict[str, dict[str, str | None]]:
    """``name: body`` lines; ``#`` comments; a ``# ----`` line opens a group."""
    macros: dict[str, dict[str, str | None]] = {}
    group: str | None = None
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("# ----"):
            group = stripped[len("# ----") :].strip(" -") or None
            continue
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, body = line.split(":", 1)
        macros[name.strip()] = {"body": " ".join(body.split()), "group": group}
    return macros


MACROS = load_macros()


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------


def _words(text: str) -> list[tuple[str, int, int]]:
    """``(word, start, end)`` for ``[``, ``]`` and whitespace-separated words."""
    words: list[tuple[str, int, int]] = []
    index, size = 0, len(text)
    while index < size:
        char = text[index]
        if char == "#":
            while index < size and text[index] != "\n":
                index += 1
        elif char.isspace():
            index += 1
        elif char in "[]":
            words.append((char, index, index + 1))
            index += 1
        else:
            start = index
            while (
                index < size and not text[index].isspace() and text[index] not in "[]#"
            ):
                index += 1
            words.append((text[start:index], start, index))
    return words


def _expand(name: str, active: tuple[str, ...]) -> list[str]:
    """A macro's primitive ops, expanding the macros its body names."""
    if name in active:
        raise ValueError(f"Recursive macro: {name}")
    out: list[str] = []
    for word, _start, _end in _words(MACROS[name]["body"]):
        if word in OPS:
            out.append(word)
        elif word in MACROS:
            out.extend(_expand(word, (*active, name)))
        else:
            raise ValueError(_unknown(word))
    return out


def _unknown(name: str) -> str:
    return (
        f"Unknown word '{name}'. Words are mark-28 ops "
        f"({' '.join(OPS)}) or macros from macros.txt."
    )


def compile_program(text: str) -> list[dict[str, Any]]:
    """One ``{op, start, end, macro}`` per primitive the text expands to."""
    tokens: list[dict[str, Any]] = []
    for word, start, end in _words(text):
        if word in OPS:
            tokens.append({"op": word, "start": start, "end": end, "macro": None})
        elif word in MACROS:
            for op in _expand(word, ()):
                tokens.append({"op": op, "start": start, "end": end, "macro": word})
        else:
            raise ValueError(_unknown(word))
    return tokens


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


def _from_json(item: Any, depth: int) -> Value:
    if item.__class__ is bool:
        return Bool(item)
    if item.__class__ is int:
        return Int(item)
    if isinstance(item, list):
        if depth >= MAX_NESTING:
            raise ValueError(f"Lists may nest at most {MAX_NESTING} deep")
        return Lst(tuple(_from_json(value, depth + 1) for value in item))
    raise ValueError(
        f"Stack values are ints, booleans or lists, not {type(item).__name__}"
    )


def _check(value: Value, depth: int, expected: bool) -> None:
    if isinstance(value, Int):
        if expected:
            if len(str(abs(value.value))) > MAX_OUTPUT_DIGITS:
                raise ValueError(
                    f"Expected integers may have at most {MAX_OUTPUT_DIGITS} digits"
                )
        elif abs(value.value) > MAX_INPUT_INT:
            raise ValueError("Input integers must lie within ±10^12")
    elif isinstance(value, Sym):
        raise ValueError(f"Stacks hold data, not the symbol '{value.name}'")
    elif isinstance(value, Mark):
        raise ValueError("Stacks hold data, not a quotation mark")
    elif isinstance(value, Lst):
        if depth >= MAX_NESTING:
            raise ValueError(f"Lists may nest at most {MAX_NESTING} deep")
        if len(value.items) > MAX_LIST_ITEMS:
            raise ValueError(f"Lists may hold at most {MAX_LIST_ITEMS} items")
        for item in value.items:
            _check(item, depth + 1, expected)


def stack_from(value: Any, *, expected: bool = False) -> tuple[Value, ...]:
    """A stack from a JSON array or stack-notation text, bottom first."""
    if isinstance(value, list):
        stack = tuple(_from_json(item, 0) for item in value)
    elif isinstance(value, str):
        try:
            stack = parse_stack(value, basis="mark-28")
        except (MachineError, ValueError) as error:
            raise ValueError(f"Cannot read stack: {error}") from None
    else:
        raise ValueError("A stack is a JSON array or stack-notation text")
    if len(stack) > MAX_VALUES:
        raise ValueError(f"A stack may hold at most {MAX_VALUES} values")
    for item in stack:
        _check(item, 0, expected)
    return stack


def _display(value: Value) -> dict[str, Any]:
    if isinstance(value, Int):
        return {"type": "int", "text": str(value.value)}
    if isinstance(value, Bool):
        return {"type": "bool", "text": format_value(value)}
    if isinstance(value, Sym):
        return {"type": "symbol", "text": value.name}
    if isinstance(value, Mark):
        return {"type": "mark", "text": "["}
    return {
        "type": "list",
        "text": format_value(value),
        "items": [_display(item) for item in value.items],
    }


def snapshot(stack: tuple[Value, ...]) -> list[dict[str, Any]]:
    """Display dicts, bottom first; a deep stack keeps its bottom and top 47."""
    if len(stack) <= DISPLAY_DEPTH:
        return [_display(value) for value in stack]
    hidden = len(stack) - DISPLAY_DEPTH
    return [
        _display(stack[0]),
        {"type": "elision", "text": f"… {hidden} more", "count": hidden},
        *(_display(value) for value in stack[-(DISPLAY_DEPTH - 1) :]),
    ]


def _keys(stack: tuple[Value, ...]) -> tuple:
    return tuple(value_key(value) for value in stack)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _tokens(program: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compile_program(program) if isinstance(program, str) else program


def _examples(examples: Any) -> list[tuple[tuple[Value, ...], tuple[Value, ...]]]:
    if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
        raise ValueError(f"Give between 1 and {MAX_EXAMPLES} examples")
    parsed = []
    for example in examples:
        if not isinstance(example, dict) or not {"input", "output"} <= example.keys():
            raise ValueError('Each example is {"input": ..., "output": ...}')
        parsed.append(
            (
                stack_from(example["input"]),
                stack_from(example["output"], expected=True),
            )
        )
    return parsed


def _frame(stack, fuel, op, error=None) -> dict[str, Any]:
    return {
        "stack": snapshot(stack),
        "fuel": fuel,
        "op": op,
        "error": error,
        "quoting": quotation_open(stack),
    }


def _run_row(tokens, start, output, capture: bool) -> dict[str, Any]:
    stack, spent, depth, error = start, 0, len(start), None
    frames = [_frame(stack, 0, None)]
    last_op = None
    for token in tokens:
        name = token["op"]
        outcome = step_raw(OPS[name], stack, FUEL - spent, 0)
        if isinstance(outcome, str):
            error = f"{name}: {outcome}"
            break
        stack, cost = outcome
        spent += cost
        depth = max(depth, len(stack))
        last_op = name
        if capture:
            frames.append(_frame(stack, spent, name))
    if error is not None:
        # The refused op leaves the stack unchanged and halts this example.
        frames.append(_frame(stack, spent, name, error))
    elif not capture and last_op is not None:
        frames.append(_frame(stack, spent, last_op))
    return {
        "frames": frames,
        "matched": error is None and _keys(stack) == _keys(output),
        "error": error,
        "fuel": spent,
        "max_depth": depth,
        "output": snapshot(output),
    }


def execute(
    program: str | list[dict[str, Any]], examples: Any, *, capture: bool = True
) -> dict[str, Any]:
    """Run the program over every example through ``step_raw``."""
    tokens = _tokens(program)
    parsed = _examples(examples)
    with MACHINE_LOCK:
        rows = [_run_row(tokens, start, output, capture) for start, output in parsed]
    return {
        "tokens": tokens,
        "examples": rows,
        "matches": sum(row["matched"] for row in rows),
        "count": len(rows),
        "primitive_count": len(tokens),
        "fuel": sum(row["fuel"] for row in rows),
        "max_depth": max(row["max_depth"] for row in rows),
    }


def inspect_op(
    program: str | list[dict[str, Any]], examples: Any, step: int, example: int = 0
) -> dict[str, Any]:
    """Every nested ``step_raw`` call that token ``step`` makes on one example."""
    tokens = _tokens(program)
    parsed = _examples(examples)
    if not isinstance(step, int) or not 0 <= step < len(tokens):
        raise ValueError(f"Step must lie in 0..{len(tokens) - 1}")
    if not isinstance(example, int) or not 0 <= example < len(parsed):
        raise ValueError(f"Example must lie in 0..{len(parsed) - 1}")
    stack, _output = parsed[example]
    with MACHINE_LOCK:
        spent = 0
        for token in tokens[:step]:
            outcome = step_raw(OPS[token["op"]], stack, FUEL - spent, 0)
            if isinstance(outcome, str):
                return {
                    "frames": [],
                    "truncated": False,
                    "error": f"{token['op']}: {outcome}",
                }
            stack, cost = outcome
            spent += cost
        return _profile(OPS[tokens[step]["op"]], stack, FUEL - spent)


def _profile(op, stack, budget) -> dict[str, Any]:
    code = step_raw.__code__
    active: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    state = {"truncated": False}

    def hook(frame, event, arg):
        if frame.f_code is not code:
            return
        if event == "call":
            # A warm cache would answer a repeated body without running it.
            run_body.cache_clear()
            local = frame.f_locals
            active.append({"op": local["op"].name, "before": local["stack"]})
        elif event == "return" and active:
            call = active.pop()
            if not active:
                return  # the token's own step, not a nested one
            if len(frames) >= MAX_INSPECT_FRAMES:
                state["truncated"] = True
                return
            failed = isinstance(arg, str)
            frames.append(
                {
                    "op": call["op"],
                    "depth": len(active),
                    "before": snapshot(call["before"]),
                    "after": None if failed or arg is None else snapshot(arg[0]),
                    "error": arg if failed else None,
                }
            )

    run_body.cache_clear()
    previous = sys.getprofile()
    sys.setprofile(hook)
    try:
        outcome = stack_machine.step_raw(op, stack, budget, 0)
    finally:
        sys.setprofile(previous)
    error = f"{op.name}: {outcome}" if isinstance(outcome, str) else None
    return {"frames": frames, "truncated": state["truncated"], "error": error}
