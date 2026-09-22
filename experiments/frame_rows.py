"""Witness rows whose body steps show the policy the inner frame (inner-frames #160 E2).

Under an open mark the machine does not move: the policy is shown the stack
below the mark and, after it, the body written so far as captured symbols and
literals. inner-frames (#160) proposes committing to the consuming combinator
when the quotation opens, so the body can be run on the frame that combinator
will feed it. E1 showed that frame is computable for 70-77% of v4's under-mark
ops; this module is the training-side half of E2, which asks whether a policy
shown the frame pays less per step inside bodies.

A body step's state is rebuilt from the machine's own stack, split at its
marks. Each open quotation keeps its `Mark` and, where E1's walk defines its
frame, has its captured items replaced by `Sym(tag)` -- plus `Int(role)` for an
`ifte` body, whose three roles share one frame -- and the frame after the body
prefix. Where the frame is undefined (no consumer, a refused prefix, an empty
list) the level keeps the machine's items, so the row falls back to exactly the
state the flat policy sees. Top-level steps are untouched, so a witness with no
quotation yields the same rows as `stack_policy.witness_rows`.

Tags and frames are the witness's own, in hindsight, and capture-aware (looking
through `[ body ] swap cons filter`): what a search that had committed to the
right consumer would know. What that commitment costs is priced separately.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.inner_frames import (  # noqa: E402
    COMBINATORS,
    attribute,
    replay,
    walk_one_example,
)
from wandering_light.stack_machine import Int, Mark, Sym, parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    Row,
    encode_item,
    encode_targets,
    item_array,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_machine import Value
    from wandering_light.stack_pbe import Task

View = tuple[str, int, "tuple | None"]


ENCODINGS = ("frames", "tags")


def frame_state(
    state: tuple[Value, ...], views: Sequence[View], encoding: str = "frames"
) -> tuple[Value, ...]:
    """``state`` with each open level's items replaced by its tag and frame, where defined.

    ``encoding="tags"`` is the control: the same levels show the same tag
    tokens but keep the machine's items instead of the frame, so the frame's
    contribution is what ``"frames"`` adds over it.
    """
    marks = [i for i, value in enumerate(state) if isinstance(value, Mark)]
    if len(marks) != len(views):
        raise ValueError(
            f"{len(marks)} marks on the stack but {len(views)} open levels"
        )
    if not marks:
        return state
    out: list[Value] = list(state[: marks[0]])
    for k, (at, (tag, role, virtual)) in enumerate(zip(marks, views, strict=True)):
        end = marks[k + 1] if k + 1 < len(marks) else len(state)
        out.append(state[at])
        if virtual is None or tag not in COMBINATORS:
            out.extend(state[at + 1 : end])
            continue
        out.append(Sym(tag))
        if tag == "ifte":
            out.append(Int(role))
        out.extend(virtual if encoding == "frames" else state[at + 1 : end])
    return tuple(out)


def example_walks(task: Task) -> list[tuple[list[tuple], dict[int, tuple[View, ...]]]]:
    """Per shown example: the machine states along the witness and E1's per-step views."""
    tags = attribute(task.program)
    walks = []
    for text in task.inputs:
        states = replay(task.program, parse_stack(text, basis=BASIS))
        if states is None:
            raise ValueError(f"{task.behaviour}: witness refused")
        walk = walk_one_example(task.program, states, tags, capture_aware=True)
        walks.append((states, walk["views"]))
    return walks


def frame_product(
    task: Task, encoding: str = "frames"
) -> list[tuple[tuple[Value, ...], ...]]:
    """The product of frame states the policy is shown before each witness step."""
    walks = example_walks(task)
    return [
        tuple(
            frame_state(states[step], views.get(step, ()), encoding)
            for states, views in walks
        )
        for step in range(len(task.program))
    ]


def frame_witness_rows(tasks: Sequence[Task], encoding: str = "frames") -> list[Row]:
    """`stack_policy.witness_rows` with every body step re-encoded on its frame."""
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, not {encoding!r}")
    rows = []
    for index, task in enumerate(tasks):
        targets = encode_targets(
            [parse_stack(text, basis=BASIS) for text in task.outputs]
        )
        for step, product in enumerate(frame_product(task, encoding)):
            items = [
                encode_item(state, target)
                for state, target in zip(product, targets, strict=True)
            ]
            rows.append(
                Row(
                    item_array(items),
                    OP_INDEX[task.program[step]],
                    index,
                    step,
                    task.length - step,
                )
            )
    return rows
