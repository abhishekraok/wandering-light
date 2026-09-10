"""A program's own tree, with what each section did on the shown examples.

A `mark-28` program carries its structure in its brackets: every ``[ … ]`` is
a quotation, and the combinator that consumes a quotation gives it a role --
the body of a ``fold``, the condition of an ``ifte``. Nothing here guesses:
the tree is the bracket nesting, a section's body value is what the machine
makes of that span from an empty stack, and its role and example rows come
from the walker's record of every body call while the program ran on every
shown example. A quotation the walker ran that matches no span was built at
runtime (by ``cons``, ``cat``, ``quote``) and is listed separately.

Two spans with the same body text are indistinguishable at runtime and
behave identically, so they share their roles and rows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from wandering_light.stack_debugger.walk import walk_op
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    format_value,
    parse_stack,
    quotation_open,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

EXAMPLE_ROWS = 8


def spans(program: Sequence[str]) -> list[dict[str, Any]]:
    """Every ``[ … ]`` span as (start, end, depth, parent), in program order."""
    nodes: list[dict[str, Any]] = []
    open_stack: list[int] = []
    for index, name in enumerate(program):
        if name == "[":
            nodes.append(
                {
                    "id": len(nodes),
                    "start": index,
                    "end": None,
                    "depth": len(open_stack),
                    "parent": open_stack[-1] if open_stack else None,
                }
            )
            open_stack.append(nodes[-1]["id"])
        elif name == "]" and open_stack:
            nodes[open_stack.pop()]["end"] = index
    return nodes


def span_body(program: Sequence[str], start: int, end: int | None) -> str | None:
    """What the machine makes of a span from an empty stack, printed."""
    if end is None:
        return None
    stack: tuple = ()
    for name in program[start : end + 1]:
        outcome = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None
        stack = outcome[0]
    if len(stack) == 1 and isinstance(stack[0], Lst):
        return format_value(stack[0])
    return None


def anatomy(task: Task, program: Sequence[str]) -> dict[str, Any]:
    """The tree plus, per section, its roles, run count and example rows."""
    nodes = spans(program)
    for node in nodes:
        node["body"] = span_body(program, node["start"], node["end"])

    calls: list[dict[str, Any]] = []
    for text in task.inputs:
        stack = parse_stack(text, basis=BASIS)
        for name in program:
            op = OP_BY_NAME.get(name)
            if op is None:
                break
            if op.runs_code and not quotation_open(stack):
                walk, _walked, _overflow = walk_op(op, stack, keep_frames=False)
                calls.extend(walk.calls)
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                break
            stack = outcome[0]

    by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_body[call["body"]].append(call)

    def summarise(body: str) -> dict[str, Any]:
        rows = by_body.get(body, [])
        roles = sorted({row["role"] for row in rows})
        examples: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for row in rows:
            key = (row["role"], row["input"], row["output"], row["refused"])
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "role": row["role"],
                    "input": row["input"],
                    "output": row["output"],
                    "refused": row["refused"],
                }
            )
            if len(examples) == EXAMPLE_ROWS:
                break
        return {"roles": roles, "runs": len(rows), "examples": examples}

    for node in nodes:
        node.update(
            summarise(node["body"])
            if node["body"]
            else {"roles": [], "runs": 0, "examples": []}
        )

    literal = {node["body"] for node in nodes if node["body"]}
    runtime = [
        {"body": body, **summarise(body)} for body in by_body if body not in literal
    ]
    return {"nodes": nodes, "runtime": runtime}
