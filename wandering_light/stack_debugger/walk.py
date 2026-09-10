"""Play a program through the machine and show the work inside each op.

Replay mode steps a program op by op with ``step_all_explained``, the call the
search makes, so what is shown is what the search would have seen. For an op
that runs code -- ``i``, ``dip``, ``map``, ``filter``, ``fold``, ``ifte`` --
the machine's own runner returns only the finished stack. The walker here
re-states that runner's sequencing one item at a time so the page can show a
``fold`` accumulating, a ``map`` visiting each element, an ``ifte`` choosing a
branch. Each leaf op inside a body is still applied by ``step_raw``; only the
loop structure is restated here, and the walker's final stack is checked
against ``step_raw``'s own result -- a disagreement is reported on the frame
rather than hidden.

The walker also records every *call* of a quotation body -- which combinator
ran it, in which role, on what stack, leaving what -- which is what the
program anatomy (``anatomy.py``) turns into per-section examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wandering_light.stack_beam import step_all_explained
from wandering_light.stack_debugger.trace import product_text, stack_text
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Bool,
    Lst,
    Sym,
    format_value,
    parse_stack,
    quotation_open,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_beam import Product
    from wandering_light.stack_machine import Op
    from wandering_light.stack_pbe import Task

FRAME_CAP = 600


class _OverflowError(Exception):
    pass


@dataclass
class Walk:
    """What one walk collects: display frames (optional) and body calls."""

    keep_frames: bool = True
    cap: int = FRAME_CAP
    frames: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def frame(self, level: int, kind: str, text: str, stack: tuple) -> None:
        if not self.keep_frames:
            return
        if len(self.frames) >= self.cap:
            raise _OverflowError
        self.frames.append(
            {"level": level, "kind": kind, "text": text, "stack": stack_text(stack)}
        )

    def call(self, role: str, body: Lst, stack: tuple, level: int) -> tuple | str:
        """Run one quotation body in a named role and record what it did."""
        out = _body(body.items, stack, level, self)
        self.calls.append(
            {
                "body": format_value(body),
                "role": role,
                "input": stack_text(stack),
                "output": None if isinstance(out, str) else stack_text(out),
                "refused": out if isinstance(out, str) else None,
            }
        )
        return out


def _body(items: tuple, stack: tuple, level: int, walk: Walk) -> tuple | str:
    for item in items:
        if isinstance(item, Sym):
            op = OP_BY_NAME.get(item.name)
            if op is None:
                return f"unknown op {item.name!r}"
            stack = _op(op, stack, level, walk)
            if isinstance(stack, str):
                return stack
        else:
            stack = (*stack, item)
            walk.frame(level, "push", format_value(item), stack)
    return stack


def _one(result: tuple | str, want_bool: bool = False):
    if isinstance(result, str):
        return result
    if len(result) != 1:
        return f"body must leave exactly one value, left {len(result)}"
    if want_bool and not isinstance(result[0], Bool):
        return "body must leave a boolean"
    return result[0]


def _op(op: Op, stack: tuple, level: int, walk: Walk) -> tuple | str:
    if not op.runs_code or quotation_open(stack):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            walk.frame(level, "refused", f"{op.name}: {outcome}", stack)
            return outcome
        kind = "captured" if quotation_open(stack) and op.name not in "[]" else "apply"
        walk.frame(level, kind, op.name, outcome[0])
        return outcome[0]

    name = op.name
    if len(stack) < 1 or not isinstance(stack[-1], Lst):
        walk.frame(level, "refused", f"{name}: expects a quotation on top", stack)
        return "expects a quotation on top"
    body = stack[-1]
    if name == "i":
        rest = stack[:-1]
        walk.frame(level, "enter", f"i · run {format_value(body)}", rest)
        return walk.call("i body", body, rest, level + 1)
    if name == "dip":
        if len(stack) < 2:
            return "needs 2 value(s)"
        kept, rest = stack[-2], stack[:-2]
        walk.frame(
            level,
            "enter",
            f"dip · hold {format_value(kept)} aside, run {format_value(body)} underneath",
            rest,
        )
        out = walk.call("dip body", body, rest, level + 1)
        if isinstance(out, str):
            return out
        result = (*out, kept)
        walk.frame(
            level, "leave", f"dip · put {format_value(kept)} back on top", result
        )
        return result
    if name in ("map", "filter"):
        if len(stack) < 2 or not isinstance(stack[-2], Lst):
            return "second value is not a list"
        source = stack[-2]
        out: list = []
        for index, item in enumerate(source.items):
            walk.frame(
                level,
                "iter",
                f"{name} · element {index + 1} of {len(source.items)}: {format_value(item)}",
                (item,),
            )
            got = _one(
                walk.call(f"{name} body", body, (item,), level + 1),
                want_bool=name == "filter",
            )
            if isinstance(got, str):
                walk.frame(level, "refused", f"{name}: {got}", (item,))
                return got
            if name == "map":
                out.append(got)
            elif got.value:
                out.append(item)
        result = (*stack[:-2], Lst(tuple(out)))
        walk.frame(
            level,
            "leave",
            f"{name} · collected {format_value(Lst(tuple(out)))}",
            result,
        )
        return result
    if name == "fold":
        if len(stack) < 3 or not isinstance(stack[-3], Lst):
            return "third value is not a list"
        source, acc = stack[-3], stack[-2]
        for index, item in enumerate(source.items):
            walk.frame(
                level,
                "iter",
                f"fold · element {index + 1} of {len(source.items)}: "
                f"acc {format_value(acc)}, item {format_value(item)}",
                (acc, item),
            )
            got = _one(walk.call("fold body", body, (acc, item), level + 1))
            if isinstance(got, str):
                walk.frame(level, "refused", f"fold: {got}", (acc, item))
                return got
            acc = got
        result = (*stack[:-3], acc)
        walk.frame(level, "leave", f"fold · result {format_value(acc)}", result)
        return result
    if name == "ifte":
        if len(stack) < 3 or not all(isinstance(v, Lst) for v in stack[-3:]):
            return "needs three quotations"
        condition, then, otherwise = stack[-3:]
        rest = stack[:-3]
        walk.frame(level, "enter", f"ifte · test {format_value(condition)}", rest)
        tested = walk.call("ifte condition", condition, rest, level + 1)
        if isinstance(tested, str):
            return tested
        if not tested or not isinstance(tested[-1], Bool):
            walk.frame(level, "refused", "ifte: condition must leave a boolean", tested)
            return "condition must leave a boolean on top"
        chosen = then if tested[-1].value else otherwise
        role = "ifte then" if tested[-1].value else "ifte else"
        walk.frame(
            level,
            "branch",
            f"ifte · {format_value(tested[-1])} → run {format_value(chosen)} "
            "on the stack the test saw",
            rest,
        )
        return walk.call(role, chosen, rest, level + 1)
    return f"no walker for {name}"  # pragma: no cover - the combinator set is closed


def walk_op(
    op: Op, stack: tuple, *, keep_frames: bool = True
) -> tuple[Walk, tuple | str, bool]:
    """Walk one op on one stack: the collected walk, its result, and whether it overflowed."""
    walk = Walk(keep_frames=keep_frames)
    try:
        walked = _op(op, stack, 0, walk)
    except _OverflowError:
        return walk, "truncated", True
    return walk, walked, False


def inner_frames(op: Op, stack: tuple) -> dict[str, Any]:
    """The work inside one op on one stack, checked against the machine."""
    walk, walked, truncated = walk_op(op, stack)
    machine = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
    agrees = None
    if not truncated and not isinstance(machine, str) and not isinstance(walked, str):
        agrees = tuple(map(value_key, walked)) == tuple(map(value_key, machine[0]))
    return {
        "frames": walk.frames,
        "truncated": truncated,
        "agrees_with_machine": agrees,
        "fuel": None if isinstance(machine, str) else machine[1],
    }


def play_program(task: Task, program: Sequence[str]) -> dict[str, Any]:
    """Step a program through every shown example, with inner work for the first."""
    product: Product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    target = [tuple(map(value_key, stack)) for stack in outputs]
    steps: list[dict[str, Any]] = []
    steps.append(
        {
            "index": 0,
            "op": None,
            "states": product_text(product),
            "matches": [
                tuple(map(value_key, s)) == t
                for s, t in zip(product, target, strict=True)
            ],
        }
    )
    refused = None
    for index, name in enumerate(program, start=1):
        op = OP_BY_NAME.get(name)
        if op is None:
            refused = f"unknown op {name!r}"
            break
        inner = (
            inner_frames(op, product[0])
            if op.runs_code and not quotation_open(product[0])
            else None
        )
        moved, fuel, _stepped, reason = step_all_explained(product, op)
        if moved is None:
            refused = reason
            steps.append(
                {
                    "index": index,
                    "op": name,
                    "refused": reason,
                    "states": product_text(product),
                    "inner": inner,
                }
            )
            break
        product = moved
        steps.append(
            {
                "index": index,
                "op": name,
                "states": product_text(product),
                "fuel": fuel,
                "inner": inner,
                "matches": [
                    tuple(map(value_key, s)) == t
                    for s, t in zip(product, target, strict=True)
                ],
            }
        )
    return {
        "program": list(program),
        "steps": steps,
        "refused": refused,
        "solved_shown": refused is None and all(steps[-1]["matches"]),
    }
