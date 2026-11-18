"""Compile workshop programs and observe the canonical machine."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, TypedDict

from wandering_light import stack_machine as machine
from wandering_light.stack_machine import (
    Bool,
    Int,
    Lst,
    Sym,
    Value,
    format_value,
    parse_stack,
    quotation_open,
    value_key,
)

if TYPE_CHECKING:
    from types import FrameType

OPS = {op.name: op for op in machine.MARK_OPS}
FUEL = 100_000
MACHINE_LOCK = Lock()
REPO = Path(__file__).resolve().parents[2]
MACRO_FILE = REPO / "experiments/world_behaviours/macros.txt"
_WORD = re.compile(r"#[^\r\n]*|\[|\]|[^\s\[\]#]+")
_MAX_INSPECT_FRAMES = 350


def _read_macros(path: Path) -> dict[str, dict[str, str]]:
    macros = {}
    group = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("# ----"):
            group = line[1:].strip(" -")
        elif definition := line.partition("#")[0].strip():
            name, body = definition.split(":", 1)
            macros[name.strip()] = {"body": body.strip(), "group": group}
    return macros


# The authoring library is available in a source checkout.
MACROS = _read_macros(MACRO_FILE) if MACRO_FILE.is_file() else {}


class Token(TypedDict):
    op: str
    start: int
    end: int
    macro: str | None


def compile_program(text: str) -> list[Token]:
    """Expand macros, retaining the source span of each written word."""
    if not isinstance(text, str):
        raise ValueError("Program must be text")

    def words(source: str):
        return (match for match in _WORD.finditer(source) if match[0][0] != "#")

    tokens = []

    def expand(name: str, start: int, end: int, chain: tuple[str, ...]) -> None:
        if name in OPS:
            tokens.append(
                Token(op=name, start=start, end=end, macro=chain[0] if chain else None)
            )
            return
        if name in chain:
            raise ValueError(f"Recursive macro: {name}")
        if name not in MACROS:
            raise ValueError(f"Unknown word '{name}'. Use a mark-28 op or a macro.")
        for word in words(MACROS[name]["body"]):
            expand(word[0], start, end, (*chain, name))

    for word in words(text):
        expand(word[0], *word.span(), ())
    return tokens


def stack_from(value: object, *, expected: bool = False) -> tuple[Value, ...]:
    """Read data stacks from JSON arrays or canonical stack notation."""
    limit = 10**512 - 1 if expected else 10**12
    notation = isinstance(value, str)
    if notation:
        try:
            items = parse_stack(value, basis="mark-28")
        except (machine.MachineError, RecursionError) as error:
            raise ValueError(f"Invalid stack notation: {error}") from error
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError("A stack must be a JSON array or stack-notation text")
    if len(items) > 16:
        raise ValueError("Stacks may contain at most 16 values")

    def convert(item: object, depth: int) -> Value:
        if notation:
            if isinstance(item, (Int, Bool)):
                item = item.value
            elif isinstance(item, Lst):
                item = list(item.items)
        if type(item) is bool:
            return Bool(item)
        if type(item) is int:
            if abs(item) > limit:
                bound = "512 digits" if expected else "10^12 in magnitude"
                raise ValueError(f"Integer exceeds the limit of {bound}")
            return Int(item)
        if isinstance(item, list):
            if depth > 12:
                raise ValueError("Lists may nest at most 12 levels")
            if len(item) > 64:
                raise ValueError("Lists may contain at most 64 items")
            return Lst(tuple(convert(child, depth + 1) for child in item))
        raise ValueError("Stacks may contain only integers, booleans and lists")

    return tuple(convert(item, 1) for item in items)


def _display(value: Value) -> dict:
    if isinstance(value, Lst):
        items = [_display(item) for item in value.items]
        body = " ".join(item["text"] for item in items)
        return {"type": "list", "text": f"({body})" if items else "nil", "items": items}
    if isinstance(value, Int):
        try:
            text = format_value(value)
        except ValueError:
            # Machine results can exceed Python's decimal conversion limit.
            text = str(Decimal(value.value))
        return {"type": "int", "text": text}
    kind = (
        "bool"
        if isinstance(value, Bool)
        else "symbol"
        if isinstance(value, Sym)
        else "mark"
    )
    return {"type": kind, "text": format_value(value)}


def snapshot(stack: tuple[Value, ...]) -> list[dict]:
    """Display the bottom and top of the stack, with the top on the right."""
    if len(stack) <= 48:
        return [_display(value) for value in stack]
    return [
        _display(stack[0]),
        {"type": "elision", "text": f"… {len(stack) - 48} omitted …"},
        *(_display(value) for value in stack[-47:]),
    ]


def _example_stacks(
    examples: object,
) -> list[tuple[tuple[Value, ...], tuple[Value, ...]]]:
    if not isinstance(examples, (list, tuple)) or not 1 <= len(examples) <= 16:
        raise ValueError("Provide 1 to 16 examples")
    stacks = []
    for example in examples:
        if not isinstance(example, dict) or not {"input", "output"} <= example.keys():
            raise ValueError("Each example needs an input and an output")
        stacks.append(
            (stack_from(example["input"]), stack_from(example["output"], expected=True))
        )
    return stacks


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


@dataclass
class _Run:
    stack: tuple[Value, ...]
    fuel: int
    max_depth: int
    error: str | None
    frames: list[dict]


def _run_tokens(
    tokens: list[Token], stack: tuple[Value, ...], *, capture: bool | None = None
) -> _Run:
    """Run under MACHINE_LOCK. With capture=None, omit display work entirely."""
    fuel, max_depth = 0, len(stack)
    error = name = None
    frames = [] if capture is None else [_frame(stack, 0, None, None)]
    for token in tokens:
        name = token["op"]
        outcome = machine.step_raw(OPS[name], stack, FUEL - fuel, 0)
        if isinstance(outcome, str):
            error = f"{name}: {outcome}"
        else:
            stack, cost = outcome
            fuel += cost
            max_depth = max(max_depth, len(stack))
        if capture:
            frames.append(_frame(stack, fuel, name, error))
        if error:
            break
    if capture is False and tokens:
        frames.append(_frame(stack, fuel, name, error))
    return _Run(stack, fuel, max_depth, error, frames)


def execute(program: str, examples: list[dict], *, capture: bool = True) -> dict:
    """Return one run per example, the total fuel, and the deepest observed stack."""
    tokens = compile_program(program)
    rows = []
    with MACHINE_LOCK:
        for initial, output in _example_stacks(examples):
            result = _run_tokens(tokens, initial, capture=capture)
            matched = result.error is None and tuple(
                map(value_key, result.stack)
            ) == tuple(map(value_key, output))
            rows.append(
                {
                    "frames": result.frames,
                    "matched": matched,
                    "error": result.error,
                    "fuel": result.fuel,
                    "max_depth": result.max_depth,
                    "output": snapshot(output),
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


def inspect_op(program: str, examples: list[dict], step: int, example: int = 0) -> dict:
    """Trace calls made by the selected token in the machine's return order."""
    tokens = compile_program(program)
    stacks = _example_stacks(examples)
    if type(step) is not int or not 0 <= step < len(tokens):
        raise ValueError("Step must be a zero-based primitive index")
    if type(example) is not int or not 0 <= example < len(stacks):
        raise ValueError("Example index out of range")

    with MACHINE_LOCK:
        replay = _run_tokens(tokens[:step], stacks[example][0])
        frames = []
        if replay.error:
            return {"frames": frames, "truncated": False, "error": replay.error}

        raw_code = machine.step_raw.__code__

        def profile(
            frame: FrameType,
            event: str,
            outcome: tuple[tuple[Value, ...], int] | str | None,
        ) -> None:
            if event != "return" or frame.f_code is not raw_code or outcome is None:
                return
            local = frame.f_locals
            if local["depth"] == 0 or len(frames) >= _MAX_INSPECT_FRAMES:
                return
            before = local["stack"]
            failed = isinstance(outcome, str)
            frames.append(
                {
                    "op": local["op"].name,
                    "depth": local["depth"],
                    "before": snapshot(before),
                    "after": snapshot(before if failed else outcome[0]),
                    "error": outcome if failed else None,
                }
            )

        op = tokens[step]["op"]
        machine.run_body.cache_clear()
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            outcome = machine.step_raw(OPS[op], replay.stack, FUEL - replay.fuel, 0)
        finally:
            sys.setprofile(previous)

    return {
        "frames": frames,
        "truncated": len(frames) == _MAX_INSPECT_FRAMES,
        "error": f"{op}: {outcome}" if isinstance(outcome, str) else None,
    }
