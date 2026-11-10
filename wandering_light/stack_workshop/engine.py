"""Compile workshop programs and observe the canonical machine running them.

Nothing here decides what an op does. Every transition is a call to
``stack_machine.step_raw`` with the op the basis defines, so the workshop
shows exactly what the corpus, the solver and the policy see. This module
adds only what a learner needs around that: macros with source spans, input
validation, display snapshots, per-token frames, and a view of the nested
calls a combinator makes.

Standard library only.
"""

from __future__ import annotations

import difflib
import re
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
    Sym,
    Value,
    parse_stack,
    quotation_open,
    run_body,
    step_raw,
    value_key,
)

OPS: dict[str, Any] = {op.name: op for op in MARK_OPS}
FUEL = 100_000

MACRO_FILE = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "world_behaviours"
    / "macros.txt"
)

# Input limits. Outputs may be larger integers: a lesson's answer can be a
# product the learner was never asked to type.
MAX_INPUT_INT = 10**12
MAX_OUTPUT_DIGITS = 512
MAX_STACK_VALUES = 16
MAX_LIST_ITEMS = 64
MAX_NESTING = 12
MAX_EXAMPLES = 16

# Display limits. Display only: identity is always ``value_key``.
SNAPSHOT_DEPTH = 48
SNAPSHOT_LIST_ITEMS = 256
SNAPSHOT_NESTING = 32
TEXT_LIMIT = 2000
DISPLAY_DIGITS = 4000
DISPLAY_BITS = int(DISPLAY_DIGITS * 3.3219)  # log2(10)
INSPECT_LIMIT = 350

# ``run_body`` keeps a process-wide cache, and ``inspect_op`` clears it to see
# nested execution. Every machine operation holds this lock so one request can
# neither warm the cache under another's inspection nor interleave with it.
MACHINE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def load_macros(path: Path = MACRO_FILE) -> dict[str, dict[str, str | None]]:
    """``name: body`` lines; a ``# ----`` line names the group that follows."""
    macros: dict[str, dict[str, str | None]] = {}
    if not path.exists():
        return macros
    group: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("# ----"):
            title = stripped.lstrip("#").strip().strip("-").strip()
            group = title.split(":", 1)[0].strip() or None
            continue
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, body = line.split(":", 1)
        name = name.strip()
        if not name or name in OPS:
            continue
        macros[name] = {"body": " ".join(body.split()), "group": group}
    return macros


MACROS = load_macros()


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"#[^\n]*|\[|\]|[^\s\[\]#]+")


def _words(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(), match.start(), match.end())
        for match in _TOKEN.finditer(text)
        if not match.group().startswith("#")
    ]


def _unknown(name: str) -> ValueError:
    known = [*OPS, *MACROS]
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    advice = (
        f" Did you mean {', '.join(repr(c) for c in close)}?"
        if close
        else " Use a mark-28 op or a macro from the library."
    )
    return ValueError(
        f"Unknown word '{name}'. It is neither a mark-28 op nor a macro.{advice}"
    )


def _expand(name: str, chain: tuple[str, ...]) -> list[str]:
    if name in chain:
        raise ValueError(f"Recursive macro: {name}")
    out: list[str] = []
    for word, _start, _end in _words(MACROS[name]["body"] or ""):
        if word in OPS:
            out.append(word)
        elif word in MACROS:
            out.extend(_expand(word, (*chain, name)))
        else:
            raise _unknown(word)
    return out


def compile_program(text: str) -> list[dict[str, Any]]:
    """Primitive tokens, each with the span of the word it came from.

    A written primitive carries ``macro=None``; every primitive a macro expands
    to carries the span of the macro's word and the outermost macro's name, so
    the frontend can highlight what the learner wrote, not what it became.
    """
    if not isinstance(text, str):
        raise ValueError("A program is text.")
    tokens: list[dict[str, Any]] = []
    for word, start, end in _words(text):
        if word in OPS:
            tokens.append({"op": word, "start": start, "end": end, "macro": None})
        elif word in MACROS:
            for op in _expand(word, ()):
                tokens.append({"op": op, "start": start, "end": end, "macro": word})
        else:
            raise _unknown(word)
    return tokens


def expansion(name: str) -> list[str]:
    """The primitive ops one macro stands for."""
    return _expand(name, ())


# ---------------------------------------------------------------------------
# Values in, values out
# ---------------------------------------------------------------------------


def _check_int(number: int, expected: bool) -> Int:
    if expected:
        if abs(number) >= 10**MAX_OUTPUT_DIGITS:
            raise ValueError(
                f"Integers in an expected output are limited to {MAX_OUTPUT_DIGITS} digits."
            )
    elif abs(number) > MAX_INPUT_INT:
        raise ValueError("Input integers are limited to 10^12 in magnitude.")
    return Int(number)


def _from_json(item: Any, depth: int, expected: bool) -> Value:
    if isinstance(item, bool):
        return Bool(item)
    if isinstance(item, int):
        return _check_int(int(item), expected)
    if isinstance(item, list | tuple):
        if depth >= MAX_NESTING:
            raise ValueError(f"Lists may nest at most {MAX_NESTING} deep.")
        if len(item) > MAX_LIST_ITEMS:
            raise ValueError(f"Lists are limited to {MAX_LIST_ITEMS} items.")
        return Lst(tuple(_from_json(inner, depth + 1, expected) for inner in item))
    raise ValueError(
        f"Stack values are integers, booleans or lists, not {type(item).__name__}."
    )


def _check_value(value: Value, depth: int, expected: bool) -> None:
    if isinstance(value, Sym):
        raise ValueError(f"'{value.name[:40]}' is not a value: stacks hold data.")
    if isinstance(value, Mark):
        raise ValueError("A MARK cannot be given as data.")
    if isinstance(value, Int):
        _check_int(value.value, expected)
    elif isinstance(value, Lst):
        if depth >= MAX_NESTING:
            raise ValueError(f"Lists may nest at most {MAX_NESTING} deep.")
        if len(value.items) > MAX_LIST_ITEMS:
            raise ValueError(f"Lists are limited to {MAX_LIST_ITEMS} items.")
        for item in value.items:
            _check_value(item, depth + 1, expected)


def stack_from(value: Any, *, expected: bool = False) -> tuple[Value, ...]:
    """A stack from a JSON array or from stack notation such as ``[ 1 (2 3) ]``.

    Bottom first, top last. ``expected`` relaxes the integer bound to 512
    digits, for outputs.
    """
    if isinstance(value, list | tuple):
        if len(value) > MAX_STACK_VALUES:
            raise ValueError(f"A stack is limited to {MAX_STACK_VALUES} values.")
        return tuple(_from_json(item, 0, expected) for item in value)
    if isinstance(value, str):
        if len(value) > 20_000:
            raise ValueError("Stack text is too long.")
        try:
            stack = parse_stack(value, basis="mark-28")
        except MachineError as error:
            raise ValueError(f"Cannot read stack: {error}") from None
        if len(stack) > MAX_STACK_VALUES:
            raise ValueError(f"A stack is limited to {MAX_STACK_VALUES} values.")
        for item in stack:
            _check_value(item, 0, expected)
        return stack
    raise ValueError("A stack is a JSON array or stack text.")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _int_text(number: int) -> str:
    """Decimal up to ``DISPLAY_DIGITS``, an approximate size beyond.

    Decided by bit length, not by ``sys.get_int_max_str_digits()``: the display
    must not depend on a process-wide setting another module may change.
    """
    approximate = number.bit_length() > DISPLAY_BITS
    if not approximate:
        try:
            return str(number)
        except ValueError:  # the process limit is lower still
            approximate = True
    digits = int(number.bit_length() * 0.30103) + 1
    return f"{'-' if number < 0 else ''}<about {digits} digits>"


def _text(value: Value) -> str:
    """``format_value``'s notation, iterative and capped at ``TEXT_LIMIT``."""
    parts: list[str] = []
    size = 0
    work: list[Value | str] = [value]
    while work:
        if size > TEXT_LIMIT:
            parts.append("…")
            break
        item = work.pop()
        if isinstance(item, str):
            piece = item
        elif isinstance(item, Int):
            piece = _int_text(item.value)
        elif isinstance(item, Bool):
            piece = "true" if item.value else "false"
        elif isinstance(item, Sym):
            piece = item.name
        elif isinstance(item, Mark):
            piece = "MARK"
        elif not item.items:
            piece = "nil"
        else:
            work.append(")")
            for index in range(len(item.items) - 1, -1, -1):
                work.append(item.items[index])
                if index:
                    work.append(" ")
            piece = "("
        parts.append(piece)
        size += len(piece)
    return "".join(parts)


def _display(value: Value, depth: int = 0) -> dict[str, Any]:
    if isinstance(value, Int):
        return {"type": "int", "text": _int_text(value.value)}
    if isinstance(value, Bool):
        return {"type": "bool", "text": "true" if value.value else "false"}
    if isinstance(value, Sym):
        return {"type": "symbol", "text": value.name}
    if isinstance(value, Mark):
        return {"type": "mark", "text": "MARK"}
    if depth >= SNAPSHOT_NESTING:
        items = [_elision(len(value.items))] if value.items else []
    else:
        shown = value.items[:SNAPSHOT_LIST_ITEMS]
        items = [_display(item, depth + 1) for item in shown]
        if len(value.items) > SNAPSHOT_LIST_ITEMS:
            items.append(_elision(len(value.items) - SNAPSHOT_LIST_ITEMS))
    return {"type": "list", "text": _text(value), "items": items}


def _elision(count: int) -> dict[str, Any]:
    return {"type": "elision", "text": f"… {count} more", "count": count}


def snapshot(stack: tuple[Value, ...]) -> list[dict[str, Any]]:
    """One display dict per value, bottom first.

    A stack deeper than 48 shows its bottom value, an elision, then its top 47,
    so both what the program started from and what it is working on stay
    visible.
    """
    if len(stack) <= SNAPSHOT_DEPTH:
        return [_display(value) for value in stack]
    hidden = len(stack) - SNAPSHOT_DEPTH
    top = stack[-(SNAPSHOT_DEPTH - 1) :]
    return [_display(stack[0]), _elision(hidden), *(_display(v) for v in top)]


def _same_stack(left: tuple[Value, ...], right: tuple[Value, ...]) -> bool:
    if len(left) != len(right):
        return False
    try:
        return all(
            value_key(a) == value_key(b) for a, b in zip(left, right, strict=True)
        )
    except RecursionError:  # far deeper than any accepted output
        return False


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _tokens(program: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(program, str):
        return compile_program(program)
    if isinstance(program, list) and all(
        isinstance(token, dict) and token.get("op") in OPS for token in program
    ):
        return program
    raise ValueError("A program is text or a list of compiled tokens.")


def parse_examples(examples: Any) -> list[tuple[tuple[Value, ...], tuple[Value, ...]]]:
    """``[{"input", "output"}, ...]`` -> ``[(input stack, expected stack), ...]``."""
    if not isinstance(examples, list | tuple) or not (
        1 <= len(examples) <= MAX_EXAMPLES
    ):
        raise ValueError(f"Give between 1 and {MAX_EXAMPLES} examples.")
    parsed = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or not {"input", "output"} <= set(example):
            raise ValueError(f"Example {index + 1} needs an input and an output.")
        try:
            parsed.append(
                (
                    stack_from(example["input"]),
                    stack_from(example["output"], expected=True),
                )
            )
        except ValueError as error:
            raise ValueError(f"Example {index + 1}: {error}") from None
    return parsed


def _frame(stack, fuel, op, error):
    return {
        "stack": snapshot(stack),
        "fuel": fuel,
        "op": op,
        "error": error,
        "quoting": quotation_open(stack),
    }


def run_tokens(
    tokens: list[dict[str, Any]], stack: tuple[Value, ...], on_step=None
) -> tuple[tuple[Value, ...], int, str | None, int]:
    """Run tokens on one stack: ``(final stack, fuel spent, error, max depth)``.

    ``on_step(stack, spent, op, error)`` sees every step, refusals included.
    Callers hold ``MACHINE_LOCK``.
    """
    spent, deepest = 0, len(stack)
    for token in tokens:
        name = token["op"]
        outcome = step_raw(OPS[name], stack, FUEL - spent, 0)
        if isinstance(outcome, str):
            error = f"{name}: {outcome}"
            if on_step is not None:
                on_step(stack, spent, name, error)
            return stack, spent, error, deepest
        stack, cost = outcome
        spent += cost
        deepest = max(deepest, len(stack))
        if on_step is not None:
            on_step(stack, spent, name, None)
    return stack, spent, None, deepest


def execute(
    program: str | list[dict[str, Any]], examples: Any, *, capture: bool = True
) -> dict[str, Any]:
    """Run a program on every example and keep a frame per token.

    A refusal halts its own example only. ``capture=False`` keeps the first and
    final frames of each row, which is all a verdict needs.
    """
    tokens = _tokens(program)
    parsed = parse_examples(examples)
    rows = []
    with MACHINE_LOCK:
        for given, wanted in parsed:
            frames = [_frame(given, 0, None, None)]
            last: list[dict[str, Any]] = []

            def record(stack, spent, op, error, frames=frames, last=last):
                if capture:
                    frames.append(_frame(stack, spent, op, error))
                else:
                    last[:] = [(stack, spent, op, error)]

            final, spent, error, deepest = run_tokens(tokens, given, record)
            if not capture and last:
                frames.append(_frame(*last[0]))
            rows.append(
                {
                    "frames": frames,
                    "matched": error is None and _same_stack(final, wanted),
                    "error": error,
                    "fuel": spent,
                    "max_depth": deepest,
                    "output": snapshot(wanted),
                }
            )
    return {
        "tokens": tokens,
        "examples": rows,
        "matches": sum(row["matched"] for row in rows),
        "count": len(rows),
        "primitive_count": len(tokens),
        "fuel": sum(row["fuel"] for row in rows),
        "max_depth": max(row["max_depth"] for row in rows),
    }


def final_states(
    program: str | list[dict[str, Any]], examples: Any
) -> list[tuple[tuple[Value, ...], int, tuple[Value, ...]]]:
    """``(stack, fuel spent, expected)`` after the whole program, per example.

    Raises ``ValueError`` naming the first refused step.
    """
    tokens = _tokens(program)
    parsed = parse_examples(examples)
    out = []
    with MACHINE_LOCK:
        for index, (given, wanted) in enumerate(parsed):
            final, spent, error, _ = run_tokens(tokens, given)
            if error is not None:
                raise ValueError(f"Example {index + 1} stopped at {error}")
            out.append((final, spent, wanted))
    return out


def try_step(
    op_name: str, stack: tuple[Value, ...], spent: int
) -> tuple[tuple[Value, ...], int] | str:
    """One canonical step under the workshop budget."""
    with MACHINE_LOCK:
        return step_raw(OPS[op_name], stack, FUEL - spent, 0)


# ---------------------------------------------------------------------------
# Looking inside one step
# ---------------------------------------------------------------------------


def inspect_op(
    program: str | list[dict[str, Any]],
    examples: Any,
    step: int,
    example: int = 0,
) -> dict[str, Any]:
    """Every nested ``step_raw`` call one token makes, in return order.

    The calls are observed, not re-derived: a profile hook watches the real
    ``step_raw`` frames while the token runs. ``run_body``'s cache is cleared
    first, because a cached body returns without running and would hide the
    very calls this is meant to show.
    """
    tokens = _tokens(program)
    parsed = parse_examples(examples)
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("step is a token index.")
    if isinstance(example, bool) or not isinstance(example, int):
        raise ValueError("example is an example index.")
    if not 0 <= step < len(tokens):
        raise ValueError(f"No step {step}; the program has {len(tokens)}.")
    if not 0 <= example < len(parsed):
        raise ValueError(f"No example {example}; there are {len(parsed)}.")

    with MACHINE_LOCK:
        stack, spent, error, _ = run_tokens(tokens[:step], parsed[example][0])
        if error is not None:
            return {
                "frames": [],
                "truncated": False,
                "error": f"The example halted before step {step}: {error}",
            }

        frames: list[dict[str, Any]] = []
        pending: dict[int, tuple[str, list, int]] = {}
        state = {"outer": None, "truncated": False}
        target = step_raw.__code__

        def profiler(frame, event, arg):
            if frame.f_code is not target or event not in ("call", "return"):
                return
            key = id(frame)
            if event == "call":
                if state["outer"] is None:
                    state["outer"] = key
                    return
                if state["truncated"]:
                    return
                local = frame.f_locals
                pending[key] = (
                    local["op"].name,
                    snapshot(local["stack"]),
                    local["depth"],
                )
                return
            if key == state["outer"] or key not in pending:
                return
            op, before, depth = pending.pop(key)
            if len(frames) >= INSPECT_LIMIT:
                state["truncated"] = True
                pending.clear()
                return
            if isinstance(arg, tuple):
                after, failure = snapshot(arg[0]), None
            else:
                after, failure = None, arg if isinstance(arg, str) else "raised"
            frames.append(
                {
                    "op": op,
                    "depth": depth,
                    "before": before,
                    "after": after,
                    "error": failure,
                }
            )

        name = tokens[step]["op"]
        run_body.cache_clear()
        previous = sys.getprofile()
        sys.setprofile(profiler)
        try:
            outcome = step_raw(OPS[name], stack, FUEL - spent, 0)
        finally:
            sys.setprofile(previous)
    return {
        "frames": frames,
        "truncated": state["truncated"],
        "error": f"{name}: {outcome}" if isinstance(outcome, str) else None,
    }
