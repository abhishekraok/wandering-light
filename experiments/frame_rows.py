"""Witness rows whose body steps show the policy the inner frame (inner-frames #160 E2).

Under an open mark the machine does not move: the policy is shown the stack
below the mark and, after it, the body written so far as captured symbols and
literals. inner-frames (#160) proposes committing to the consuming combinator
when the quotation opens, so the body can be run on the frame that combinator
will feed it. E1 showed that frame is computable for 70-77% of v4's under-mark
ops; this module is the training-side half of E2, which asks whether a policy
shown the frame pays less per step inside bodies.

A body step's state is rebuilt from the machine's own stack, split at its
marks. Each open quotation whose frame E1's capture-aware walk defines keeps its
`Mark` and its captured items, and gains, after a separator no machine state
can contain, its tag (plus `Int(role)` for `ifte`) and the frame after the body
prefix -- or only the tag, for the control. Undefined levels and top-level
steps are untouched, so a witness with no quotation yields exactly the rows of
`stack_policy.witness_rows`.

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

View = tuple[str, int, int, "tuple | None"]


ENCODINGS = ("frames", "tags", "replace")
# `]` collapses a quotation the moment it is written, so `Sym("]")` never sits
# on a machine stack: after it, everything in a level is metadata, unambiguously.
META = Sym("]")


def frame_state(
    state: tuple[Value, ...], views: Sequence[View], encoding: str = "frames"
) -> tuple[Value, ...]:
    """``state`` with each defined open level annotated with its tag and frame.

    A defined level keeps its `Mark` and the items the machine captured -- the
    body written so far, which a search always knows -- and then, after `META`:

    - ``"frames"``: the label and the frame the body has reached. The policy
      sees strictly more than the machine state.
    - ``"tags"``: the label only; the control, so the frame's contribution is
      what ``"frames"`` adds over it.

    The label is everything a search commits to at the `[`, always three items
    so it cannot run into the frame: ``Sym(tag)``, ``Int(role)`` (0 unless
    `ifte`) and ``Int(captures)``, the number of values the closure idiom will
    cons on. The capture count changes the frame, so a control without it
    would not isolate what executing the body adds.
    - ``"replace"``: the first design, which put the tag and frame *in place of*
      the captured items, losing the text of the body. Kept because its
      learning-rate probe is on record; the gate does not use it.

    An undefined level is left exactly as the machine has it.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, not {encoding!r}")
    marks = [i for i, value in enumerate(state) if isinstance(value, Mark)]
    if len(marks) != len(views):
        raise ValueError(
            f"{len(marks)} marks on the stack but {len(views)} open levels"
        )
    if not marks:
        return state
    out: list[Value] = list(state[: marks[0]])
    for k, (at, (tag, role, captures, virtual)) in enumerate(
        zip(marks, views, strict=True)
    ):
        end = marks[k + 1] if k + 1 < len(marks) else len(state)
        items = state[at + 1 : end]
        out.append(state[at])
        if virtual is None or tag not in COMBINATORS:
            out.extend(items)
            continue
        label = [Sym(tag), Int(role), Int(captures)]
        if encoding == "replace":
            out.extend([*label, *virtual])
        elif encoding == "tags":
            out.extend([*items, META, *label])
        else:
            out.extend([*items, META, *label, *virtual])
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
    """`stack_policy.witness_rows` with every defined body step annotated."""
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
