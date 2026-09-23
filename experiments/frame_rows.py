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
from wandering_light.stack_machine import Int, Lst, Mark, Sym, parse_stack  # noqa: E402
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


ENCODINGS = ("frames", "tags", "replace", "tags_inner", "frames_inner", "exit_inner")
# Encodings that annotate only the innermost defined level (#160, second
# campaign): the stacked ones gave the label a cost that grew with nesting.
INNER = ("tags_inner", "frames_inner", "exit_inner")
# `]` collapses a quotation the moment it is written, so `Sym("]")` never sits
# on a machine stack: after it, everything in a level is metadata, unambiguously.
META = Sym("]")
# `[` pushes a `Mark`, never a symbol, so `Sym("[")` cannot be on a stack
# either; it separates the frame from the exit frame in the oracle encoding.
EXIT = Sym("[")


def frame_state(
    state: tuple[Value, ...],
    views: Sequence[View],
    encoding: str = "frames",
    exits: Sequence[tuple | None] = (),
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
    - ``"tags_inner"`` / ``"frames_inner"``: as ``"tags"`` / ``"frames"``, but
      only on the innermost defined level; its ancestors keep the machine's
      items. The encoder pools local features over the token sequence, so
      stacked labels are not told apart by level.
    - ``"exit_inner"``: ``"frames_inner"`` plus, after `EXIT`, the frame the
      body must produce, from the authored witness (``exits``, one per level).
      An oracle: it prices a body that knows its specification. A quotation in
      the exit that holds ops and is not yet on the stack or in the frame is
      code the body has still to write, so it shows as `EXIT`, not its text.

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
    if encoding == "exit_inner" and len(exits) != len(views):
        raise ValueError("the oracle needs one exit frame per open level")
    annotated = set(range(len(views)))
    if encoding in INNER:
        defined = [
            k
            for k, (tag, _r, _c, virtual) in enumerate(views)
            if virtual is not None and tag in COMBINATORS
        ]
        annotated = set(defined[-1:])
    out: list[Value] = list(state[: marks[0]])
    for k, (at, (tag, role, captures, virtual)) in enumerate(
        zip(marks, views, strict=True)
    ):
        end = marks[k + 1] if k + 1 < len(marks) else len(state)
        items = state[at + 1 : end]
        out.append(state[at])
        if virtual is None or tag not in COMBINATORS or k not in annotated:
            out.extend(items)
            continue
        label = [Sym(tag), Int(role), Int(captures)]
        if encoding == "replace":
            out.extend([*label, *virtual])
        elif encoding in ("tags", "tags_inner"):
            out.extend([*items, META, *label])
        elif encoding == "exit_inner" and exits[k] is not None:
            known = {*state, *virtual}
            shown = tuple(
                EXIT if _is_code(v) and v not in known else v for v in exits[k]
            )
            out.extend([*items, META, *label, *virtual, EXIT, *shown])
        else:
            out.extend([*items, META, *label, *virtual])
    return tuple(out)


def _is_code(value: Value) -> bool:
    """A quotation that holds ops: the exit may show it only once it is written."""
    return isinstance(value, Lst) and any(
        isinstance(v, Sym) or _is_code(v) for v in value.items
    )


def level_exits(
    program: Sequence[str], views: dict[int, tuple[View, ...]]
) -> dict[int, tuple[tuple | None, ...]]:
    """Per under-mark step, each open level's frame once its body has run.

    That is the level's frame at the step that writes its `]`, where the whole
    body has been executed on it; ``None`` where that frame is undefined or the
    quotation never closes.
    """
    match, stack = {}, []
    for step, name in enumerate(program):
        if name == "[":
            stack.append(step)
        elif name == "]":
            match[stack.pop()] = step
    out, stack = {}, []
    for step, name in enumerate(program):
        if step in views:
            if len(views[step]) != len(stack):
                raise ValueError(f"step {step}: levels and open brackets disagree")
            out[step] = tuple(
                views[match[opened]][k][3] if opened in match else None
                for k, opened in enumerate(stack)
            )
        if name == "[":
            stack.append(step)
        elif name == "]":
            stack.pop()
    return out


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
    if encoding != "exit_inner":
        return [
            tuple(
                frame_state(states[step], views.get(step, ()), encoding)
                for states, views in walks
            )
            for step in range(len(task.program))
        ]
    exits = [level_exits(task.program, views) for _states, views in walks]
    return [
        tuple(
            frame_state(states[step], views.get(step, ()), encoding, ex.get(step, ()))
            for (states, views), ex in zip(walks, exits, strict=True)
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
