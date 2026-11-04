"""Compile, run and inspect `mark-28` programs by watching the canonical machine.

Everything that changes a stack goes through ``stack_machine.step_raw``. This
module adds what a workshop needs around it: a tokenizer with source spans and
macro expansion, input validation for data typed by a learner, a display form
of a stack, a per-token trace with fuel, and a profiler-based view of the
nested steps a combinator takes. Standard library only.
"""

from __future__ import annotations

import json
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
    format_value,
    parse_stack,
    quotation_open,
    run_body,
    step_raw,
    value_key,
)

REPO = Path(__file__).resolve().parents[2]
MACRO_FILE = REPO / "experiments" / "world_behaviours" / "macros.txt"
BASIS = "mark-28"

OPS: dict[str, Any] = {op.name: op for op in MARK_OPS}
FUEL = 100_000

# Limits on learner-typed data. The machine itself has no such limits; these
# keep a browser page responsive and a mistyped value from becoming a mystery.
MAX_VALUES = 16
MAX_LIST_ITEMS = 64
MAX_NESTING = 12
MAX_INT = 10**12
MAX_EXPECTED_DIGITS = 512
MAX_EXAMPLES = 16
ELISION_DEPTH = 48
MAX_NESTED_CALLS = 350

# One lock for everything that touches the machine: ``run_body``'s cache is
# process-wide and ``inspect_op`` installs a profiler, so two requests must
# not interleave.
MACHINE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def _group_name(comment: str) -> str:
    """``# ---- lists -----`` names the group ``lists``."""
    text = comment.strip("#").strip().strip("-").strip()
    return text.split(":", 1)[0].strip()


def load_macros(path: Path = MACRO_FILE) -> dict[str, dict[str, str]]:
    """Read ``name: body`` lines; a ``# ----`` line names the group that follows."""
    macros: dict[str, dict[str, str]] = {}
    group = "macros"
    if not path.exists():
        return macros
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("# ----"):
            group = _group_name(stripped) or group
            continue
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, body = line.split(":", 1)
        name = name.strip()
        if not name:
            continue
        macros[name] = {"body": " ".join(body.split()), "group": group}
    return macros


MACROS: dict[str, dict[str, str]] = load_macros()


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\[|\]|[^\s\[\]]+")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """``(word, start, end)`` for every token; ``#`` to end of line is a comment."""
    out = []
    position = 0
    for line in text.split("\n"):
        code = line.split("#", 1)[0]
        for match in _TOKEN.finditer(code):
            out.append(
                (match.group(), position + match.start(), position + match.end())
            )
        position += len(line) + 1
    return out


def expand_word(name: str, _trail: tuple[str, ...] = ()) -> list[str]:
    """The primitive ops a word stands for, expanding macros recursively."""
    if name in OPS:
        return [name]
    if name in MACROS:
        if name in _trail:
            raise ValueError(f"Recursive macro: {name}")
        out: list[str] = []
        for word in MACROS[name]["body"].split():
            out.extend(expand_word(word, (*_trail, name)))
        return out
    raise ValueError(
        f"Unknown word '{name}'. Words are the {len(OPS)} ops of the machine "
        "and the macros in the library."
    )


def compile_program(text: str) -> list[dict]:
    """Primitive tokens with the source span of the written word each came from."""
    tokens = []
    for word, start, end in tokenize(text):
        macro = word if word in MACROS and word not in OPS else None
        for op in expand_word(word):
            tokens.append({"op": op, "start": start, "end": end, "macro": macro})
    return tokens


def expansion(name: str) -> str:
    """A macro's body as primitives, the way ``compile_program`` sees it."""
    return " ".join(expand_word(name))


# ---------------------------------------------------------------------------
# Values typed by a learner
# ---------------------------------------------------------------------------


def _check_int(number: int, *, expected: bool) -> None:
    if expected:
        if len(str(abs(number))) > MAX_EXPECTED_DIGITS:
            raise ValueError(
                f"Integers longer than {MAX_EXPECTED_DIGITS} digits are not allowed"
            )
    elif abs(number) > MAX_INT:
        raise ValueError(f"Integers beyond {MAX_INT:,} are not allowed as inputs")


def _from_json(item: Any, depth: int, *, expected: bool) -> Value:
    if isinstance(item, bool):
        return Bool(item)
    if isinstance(item, int):
        _check_int(item, expected=expected)
        return Int(item)
    if isinstance(item, list):
        if depth >= MAX_NESTING:
            raise ValueError(f"Lists nested deeper than {MAX_NESTING} are not allowed")
        if len(item) > MAX_LIST_ITEMS:
            raise ValueError(
                f"Lists longer than {MAX_LIST_ITEMS} items are not allowed"
            )
        return Lst(tuple(_from_json(x, depth + 1, expected=expected) for x in item))
    if isinstance(item, float):
        raise ValueError(f"{item!r} is not an integer; the machine has no floats")
    if isinstance(item, str):
        raise ValueError(f"{item!r} is a string; values are ints, bools and lists")
    raise ValueError(f"{item!r} is not a machine value")


def _check_value(value: Value, depth: int, *, expected: bool) -> None:
    if isinstance(value, Sym):
        raise ValueError(
            f"'{value.name}' is a symbol, not data; values are ints, bools and lists"
        )
    if isinstance(value, Mark):
        raise ValueError("MARK is not data; values are ints, bools and lists")
    if isinstance(value, Int):
        _check_int(value.value, expected=expected)
    if isinstance(value, Lst):
        if depth >= MAX_NESTING:
            raise ValueError(f"Lists nested deeper than {MAX_NESTING} are not allowed")
        if len(value.items) > MAX_LIST_ITEMS:
            raise ValueError(
                f"Lists longer than {MAX_LIST_ITEMS} items are not allowed"
            )
        for item in value.items:
            _check_value(item, depth + 1, expected=expected)


def stack_from(value: Any, *, expected: bool = False) -> tuple[Value, ...]:
    """A stack from a JSON array (ints, bools, nested lists) or stack-notation text."""
    if isinstance(value, str):
        text = value.strip()
        parsed: Any = None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
        if isinstance(parsed, list):
            value = parsed
        else:
            try:
                stack = parse_stack(text, basis=BASIS)
            except MachineError as error:
                raise ValueError(str(error)) from None
            for item in stack:
                _check_value(item, 0, expected=expected)
            if len(stack) > MAX_VALUES:
                raise ValueError(f"A stack holds at most {MAX_VALUES} values")
            return stack
    if not isinstance(value, list):
        raise ValueError("A stack is a JSON array or stack-notation text")
    if len(value) > MAX_VALUES:
        raise ValueError(f"A stack holds at most {MAX_VALUES} values")
    return tuple(_from_json(item, 0, expected=expected) for item in value)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _display(value: Value) -> dict:
    if isinstance(value, Int):
        return {"type": "int", "text": format_value(value)}
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


def snapshot(stack: tuple[Value, ...]) -> list[dict]:
    """One display dict per value, bottom first; a deep stack elides its middle."""
    if len(stack) > ELISION_DEPTH:
        hidden = len(stack) - ELISION_DEPTH
        return [
            _display(stack[0]),
            {"type": "elision", "text": f"... {hidden} more ..."},
            *(_display(value) for value in stack[-(ELISION_DEPTH - 1) :]),
        ]
    return [_display(value) for value in stack]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _tokens(program: str | list[dict]) -> list[dict]:
    return compile_program(program) if isinstance(program, str) else list(program)


def _examples(
    examples: list[dict],
) -> list[tuple[tuple[Value, ...], tuple[Value, ...]]]:
    if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
        raise ValueError(f"Give between 1 and {MAX_EXAMPLES} examples")
    pairs = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or not {"input", "output"} <= set(example):
            raise ValueError(f"Example {index + 1} needs an input and an output")
        pairs.append(
            (stack_from(example["input"]), stack_from(example["output"], expected=True))
        )
    return pairs


def _frame(
    stack: tuple[Value, ...], fuel: int, op: str | None, error: str | None
) -> dict:
    return {
        "stack": snapshot(stack),
        "fuel": fuel,
        "op": op,
        "error": error,
        "quoting": quotation_open(stack),
    }


def _run_example(
    tokens: list[dict],
    stack: tuple[Value, ...],
    output: tuple[Value, ...],
    *,
    capture: bool,
) -> dict:
    frames = [_frame(stack, 0, None, None)]
    spent = 0
    error = None
    max_depth = len(stack)
    for token in tokens:
        name = token["op"]
        outcome = step_raw(OPS[name], stack, FUEL - spent, 0)
        if isinstance(outcome, str):
            error = f"{name}: {outcome}"
            frames.append(_frame(stack, spent, name, error))
            break
        stack, cost = outcome
        spent += cost
        max_depth = max(max_depth, len(stack))
        frames.append(_frame(stack, spent, name, None))
    matched = error is None and [value_key(v) for v in stack] == [
        value_key(v) for v in output
    ]
    if not capture and len(frames) > 2:
        frames = [frames[0], frames[-1]]
    return {
        "frames": frames,
        "matched": matched,
        "error": error,
        "fuel": spent,
        "max_depth": max_depth,
        "output": snapshot(output),
    }


def execute(
    program: str | list[dict], examples: list[dict], *, capture: bool = True
) -> dict:
    """Run every token on every example and report one row per example."""
    tokens = _tokens(program)
    pairs = _examples(examples)
    with MACHINE_LOCK:
        rows = [
            _run_example(tokens, stack, output, capture=capture)
            for stack, output in pairs
        ]
    spans = {(token["start"], token["end"]) for token in tokens}
    return {
        "tokens": tokens,
        "examples": rows,
        "matches": sum(row["matched"] for row in rows),
        "count": len(spans),
        "primitive_count": len(tokens),
        "fuel": max(row["fuel"] for row in rows),
        "max_depth": max(row["max_depth"] for row in rows),
    }


# ---------------------------------------------------------------------------
# Inspecting one token
# ---------------------------------------------------------------------------


class _Recorder:
    """A ``sys.setprofile`` hook that records nested ``step_raw`` calls."""

    def __init__(self) -> None:
        self.pending: dict[int, dict] = {}
        self.frames: list[dict] = []
        self.truncated = False
        self.code = step_raw.__code__

    def __call__(self, frame, event, arg):
        if frame.f_code is not self.code:
            return
        if event == "call":
            self.pending[id(frame)] = {
                "op": frame.f_locals["op"].name,
                "depth": frame.f_locals["depth"],
                "before": frame.f_locals["stack"],
            }
        elif event == "return":
            record = self.pending.pop(id(frame), None)
            if record is None or self.truncated:
                return
            if len(self.frames) >= MAX_NESTED_CALLS:
                self.truncated = True
                return
            if isinstance(arg, str):
                after, error = None, arg
            else:
                after, error = snapshot(arg[0]), None
            self.frames.append(
                {
                    "op": record["op"],
                    "depth": record["depth"],
                    "before": snapshot(record["before"]),
                    "after": after,
                    "error": error,
                }
            )


def inspect_op(
    program: str | list[dict], examples: list[dict], step: int, example: int = 0
) -> dict:
    """Every nested step one token takes on one example, in return order."""
    tokens = _tokens(program)
    pairs = _examples(examples)
    if not 0 <= example < len(pairs):
        raise ValueError(f"No example {example}")
    if not 0 <= step < len(tokens):
        raise ValueError(f"No step {step}; the program has {len(tokens)} tokens")
    stack, _ = pairs[example]
    with MACHINE_LOCK:
        spent = 0
        for token in tokens[:step]:
            outcome = step_raw(OPS[token["op"]], stack, FUEL - spent, 0)
            if isinstance(outcome, str):
                raise ValueError(
                    f"Step {step} is never reached: {token['op']}: {outcome}"
                )
            stack, cost = outcome
            spent += cost
        run_body.cache_clear()
        recorder = _Recorder()
        name = tokens[step]["op"]
        previous = sys.getprofile()
        sys.setprofile(_outer_filter(recorder))
        try:
            outcome = step_raw(OPS[name], stack, FUEL - spent, 0)
        finally:
            sys.setprofile(previous)
    return {
        "frames": recorder.frames,
        "truncated": recorder.truncated,
        "error": f"{name}: {outcome}" if isinstance(outcome, str) else None,
    }


def _outer_filter(recorder: _Recorder):
    """Skip the outermost ``step_raw`` frame: the one ``inspect_op`` itself makes."""
    inspect_code = inspect_op.__code__

    def hook(frame, event, arg):
        if (
            frame.f_code is recorder.code
            and frame.f_back is not None
            and frame.f_back.f_code is inspect_code
        ):
            return
        recorder(frame, event, arg)

    return hook
