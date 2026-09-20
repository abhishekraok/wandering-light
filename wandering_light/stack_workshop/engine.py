"""Bounded execution and observation of the canonical machine (no second VM)."""

from __future__ import annotations

import re
import sys
import threading
from functools import wraps
from pathlib import Path

from wandering_light.stack_machine import (
    MARK_OPS,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    parse_stack,
    run_body,
    step_raw,
    value_key,
)

ROOT = Path(__file__).resolve().parents[2]
OPS = {op.name: op for op in MARK_OPS}
FUEL = 100_000
MAX_PROGRAM = 10_000
MACHINE_LOCK = threading.RLock()


def machine_operation(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with MACHINE_LOCK:
            return function(*args, **kwargs)

    return guarded


DESCRIPTIONS = {
    "0": "Push zero.",
    "1": "Push one.",
    "true": "Push true.",
    "false": "Push false.",
    "nil": "Push an empty list.",
    "dup": "Copy the top value; keep the original.",
    "drop": "Remove the top value.",
    "swap": "Exchange the top two values.",
    "over": "Copy the second value onto the top.",
    "add": "Consume two integers and leave their sum.",
    "sub": "Subtract the top integer from the integer below it.",
    "mul": "Consume two integers and leave their product.",
    "div": "Divide the deeper integer by the top integer, rounding down toward negative infinity.",
    "eq?": "Consume two values and test structural equality.",
    "lt?": "Is the deeper integer smaller than the top integer?",
    "null?": "Consume a list and test whether it is empty.",
    "cons": "Prepend the top value to the list below it.",
    "uncons": "Split a nonempty list into its tail, then its first value on top.",
    "[": "Begin a quotation. Instructions become symbols until the closing bracket.",
    "]": "Collect the open quotation into one list value.",
    "quote": "Wrap the top value in a one-element quotation.",
    "cat": "Concatenate two lists or quotations in stack order.",
    "i": "Execute the quotation on top of the stack.",
    "dip": "Shelter the value below the quotation, execute underneath it, then restore it.",
    "map": "Run the quotation on each element independently. Each call must leave one value.",
    "filter": "Run the predicate on each element. Keep the original element when it returns true.",
    "fold": "Run the body on (accumulator, item), carrying its single result to the next item.",
    "ifte": "Test the condition, then run the selected branch on the original starting stack.",
}


def macro_library():
    macros = {}
    group = "library"
    for line in (
        (ROOT / "experiments/world_behaviours/macros.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.startswith("# ----"):
            group = line.strip("# -")
        entry = line.split("#", 1)[0].strip()
        if entry:
            name, body = entry.split(":", 1)
            macros[name.strip()] = {"body": body.strip(), "group": group}
    return macros


MACROS = macro_library()


def compile_program(text):
    if not isinstance(text, str) or len(text) > 24000:
        raise ValueError("Programs must be text shorter than 24,000 characters.")
    tokens = []

    def expand(name, start, end, chain=()):
        if name in chain or len(chain) > 32:
            raise ValueError(f"Recursive macro: {name}")
        if name in MACROS:
            for token in MACROS[name]["body"].split():
                expand(token, start, end, (*chain, name))
        else:
            if name not in OPS:
                raise ValueError(
                    f"Unknown word '{name}'. Open the dictionary or use completion."
                )
            tokens.append(
                {
                    "op": name,
                    "start": start,
                    "end": end,
                    "macro": chain[0] if chain else None,
                }
            )
            if len(tokens) > MAX_PROGRAM:
                raise ValueError(
                    f"Expanded program exceeds {MAX_PROGRAM} instructions."
                )

    for match in re.finditer(r"#[^\n]*|\[|\]|[^\s\[\]#]+", text):
        if not match.group().startswith("#"):
            expand(match.group(), match.start(), match.end())
    return tokens


def to_value(value, depth=0, *, expected=False):
    if depth > 12:
        raise ValueError("Lists are nested too deeply.")
    if isinstance(value, bool):
        return Bool(value)
    if isinstance(value, int) and abs(value) <= (10**512 if expected else 10**12):
        return Int(value)
    if isinstance(value, list) and len(value) <= 64:
        return Lst(tuple(to_value(x, depth + 1, expected=expected) for x in value))
    raise ValueError(
        "Use integers up to 10^12, booleans, or lists of at most 64 values."
    )


def stack_from(value, *, expected=False):
    if isinstance(value, str):
        if len(value) > 3000:
            raise ValueError("Example is too large.")
        stack = parse_stack(value)

        # Text examples are data, not executable quotations.
        def check(v, depth=0):
            if depth > 12:
                raise ValueError("Lists are nested too deeply.")
            if isinstance(v, (Sym, Mark)):
                raise ValueError(
                    "Examples should contain data, not instruction symbols."
                )
            if isinstance(v, Int) and abs(v.value) > (10**512 if expected else 10**12):
                raise ValueError(
                    "Example integer is too large (inputs: 10^12; expected results: 512 digits)."
                )
            if isinstance(v, Lst):
                if len(v.items) > 64:
                    raise ValueError("Example lists may contain at most 64 values.")
                for item in v.items:
                    check(item, depth + 1)

        for v in stack:
            check(v)
    elif isinstance(value, list):
        stack = tuple(to_value(v, expected=expected) for v in value)
    else:
        raise ValueError(
            "Each input and output must be a stack (a JSON array or stack notation)."
        )
    if len(stack) > 16:
        raise ValueError("An example stack may contain at most 16 values.")
    return stack


def display(value, depth=0, budget=None):
    budget = [500] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 8:
        return {"type": "elision", "text": "… more"}
    if isinstance(value, Int):
        if value.value.bit_length() > 4000:
            return {
                "type": "elision",
                "text": f"integer · {value.value.bit_length()} bits",
            }
        return {"type": "int", "text": str(value.value)}
    if isinstance(value, Bool):
        return {"type": "bool", "text": "true" if value.value else "false"}
    if isinstance(value, Sym):
        return {"type": "symbol", "text": value.name}
    if isinstance(value, Mark):
        return {"type": "mark", "text": "["}
    items = list(value.items)
    if len(items) > 32:
        shown = [display(v, depth + 1, budget) for v in items[:24]]
        shown.append({"type": "elision", "text": f"… {len(items) - 28} more"})
        shown.extend(display(v, depth + 1, budget) for v in items[-4:])
    else:
        shown = [display(v, depth + 1, budget) for v in items]
    return {
        "type": "list",
        "text": "(" + " ".join(v["text"] for v in shown) + ")",
        "items": shown,
    }


def snapshot(stack):
    budget = [700]
    if len(stack) > 48:
        return [
            display(stack[0], budget=budget),
            {"type": "elision", "text": f"… {len(stack) - 48} deeper values"},
            *(display(value, budget=budget) for value in stack[-47:]),
        ]
    return [display(value, budget=budget) for value in stack]


def text_stack(stack):
    return "[ " + " ".join(v["text"] for v in snapshot(stack)) + " ]"


@machine_operation
def execute(program, examples, *, capture=True):
    tokens = compile_program(program)
    if not isinstance(examples, list) or not 1 <= len(examples) <= 16:
        raise ValueError("Provide between 1 and 16 input/output examples.")
    rows = []
    for example in examples:
        stack, target = (
            stack_from(example["input"]),
            stack_from(example["output"], expected=True),
        )
        frames = [{"stack": snapshot(stack), "fuel": 0, "op": None, "error": None}]
        spent, error, max_depth = 0, None, len(stack)
        for token in tokens:
            result = step_raw(OPS[token["op"]], stack, FUEL - spent, 0)
            if isinstance(result, str):
                error = f"{token['op']}: {result}"
                frames.append(
                    {
                        "stack": snapshot(stack),
                        "fuel": spent,
                        "op": token["op"],
                        "error": error,
                    }
                )
                break
            stack, cost = result
            spent += cost
            max_depth = max(max_depth, len(stack))
            if len(stack) > 4096 or any(
                isinstance(v, Int) and v.value.bit_length() > 16_000 for v in stack
            ):
                error = "The live stack is too large for the workbench. Try a smaller example."
            if capture:
                frames.append(
                    {
                        "stack": snapshot(stack),
                        "fuel": spent,
                        "op": token["op"],
                        "error": error,
                        "quoting": any(isinstance(v, Mark) for v in stack),
                    }
                )
            if error:
                break
        if not capture and tokens and error is None:
            frames.append(
                {
                    "stack": snapshot(stack),
                    "fuel": spent,
                    "op": tokens[-1]["op"],
                    "error": error,
                }
            )
        rows.append(
            {
                "input": snapshot(stack_from(example["input"])),
                "target": snapshot(target),
                "frames": frames,
                "matched": error is None
                and tuple(map(value_key, stack)) == tuple(map(value_key, target)),
                "error": error,
                "fuel": spent,
                "max_depth": max_depth,
                "output": text_stack(stack),
            }
        )
    return {
        "tokens": tokens,
        "examples": rows,
        "matches": sum(r["matched"] for r in rows),
        "count": len(rows),
        "primitive_count": len(tokens),
        "fuel": sum(r["fuel"] for r in rows),
        "max_depth": max(r["max_depth"] for r in rows),
    }


@machine_operation
def inspect_op(program, examples, step, example=0):
    """Observe real step_raw calls, rather than restating combinator semantics."""
    tokens = compile_program(program)
    if not 0 <= step < len(tokens) or not 0 <= example < len(examples):
        raise ValueError("Choose an instruction and an example to inspect.")
    stack = stack_from(examples[example]["input"])
    spent = 0
    for token in tokens[:step]:
        result = step_raw(OPS[token["op"]], stack, FUEL - spent, 0)
        if isinstance(result, str):
            raise ValueError(result)
        stack, cost = result
        spent += cost
    frames = []
    starts = {}

    def profile(frame, event, result):
        if frame.f_code is not step_raw.__code__:
            return
        local = frame.f_locals
        if event == "call":
            starts[id(frame)] = snapshot(local["stack"])
        elif event == "return":
            before = starts.pop(id(frame), [])
            if len(frames) < 350:
                frames.append(
                    {
                        "op": local["op"].name,
                        "depth": local["depth"],
                        "before": before,
                        "after": before
                        if isinstance(result, str) or result is None
                        else snapshot(result[0]),
                        "error": result if isinstance(result, str) else None,
                    }
                )

    old = sys.getprofile()
    try:
        # Evaluation warms this cache. Clear it so every actual nested call is
        # observable; all workshop machine operations share the same lock.
        run_body.cache_clear()
        sys.setprofile(profile)
        result = step_raw(OPS[tokens[step]["op"]], stack, FUEL - spent, 0)
    finally:
        sys.setprofile(old)
    return {
        "frames": frames,
        "truncated": len(frames) >= 350,
        "error": result if isinstance(result, str) else None,
    }


def op_catalog():
    return [
        {
            "name": op.name,
            "group": op.group,
            "effect": op.effect,
            "description": DESCRIPTIONS.get(op.name, op.effect),
        }
        for op in MARK_OPS
    ]
