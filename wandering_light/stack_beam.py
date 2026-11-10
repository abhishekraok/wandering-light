"""Policy-guided beam search over ``mark-28`` products, beside ``bfs_solve``.

``experiments/stack_pbe_baseline.py``'s ``bfs_solve`` expands every node of a
layer before the next. This keeps only the ``beam_width`` successors a
one-step policy (``wandering_light.stack_policy``) ranks highest, scored by
the running sum of the policy's log-probabilities along each program, so it
reaches witness depths BFS cannot -- if the policy is any good.

Cost is accounted the way a compute-optimal comparison needs it: every
attempted op on every node is charged, kept or not, as *executions* (stacks
stepped, so a refusal on the first example costs one) and as machine fuel.
The optional ``budget`` caps executions and is checked before every attempt.

Like ``bfs_solve``, nothing here reads a task's held-out examples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from wandering_light.quotation_guard import MODES, blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Mark,
    Op,
    Value,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, OPS
from wandering_light.stack_policy import OP_INDEX, OP_NAMES, encode_targets, node_block

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

Product = tuple[tuple[Value, ...], ...]

TERMINATIONS = ("fit", "max_depth", "frontier_exhausted", "budget")


class Scorer(Protocol):
    def log_probs(
        self, blocks: Sequence[np.ndarray], legal: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class BeamResult:
    """What one search found and what it cost.

    ``depth`` is the program length for a ``fit``, the layer being attempted
    (current layer + 1) for ``budget``, and otherwise the depth of the deepest
    live beam.
    """

    program: tuple[str, ...] | None
    termination: str
    seconds: float
    executions: int
    fuel: int
    expansions: int
    depth: int


def key_of(product: Product) -> tuple:
    return tuple(tuple(value_key(v) for v in stack) for stack in product)


def step_all(product: Product, op: Op) -> tuple[Product | None, int, int]:
    """Run ``op`` on each example stack in turn, stopping at the first refusal.

    Returns ``(moved, fuel, stepped)``: the moved product (``None`` on a
    refusal), the fuel the successful steps spent, and how many stacks were
    stepped -- the refusing one included -- which is what the attempt cost.
    """
    moved = []
    fuel = 0
    stepped = 0
    for stack in product:
        stepped += 1
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None, fuel, stepped
        moved.append(outcome[0])
        fuel += outcome[1]
    return tuple(moved), fuel, stepped


@dataclass(slots=True)
class _Node:
    score: float
    product: Product
    program: tuple[str, ...]


def beam_solve(
    task: Task,
    scorer: Scorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
) -> BeamResult:
    """Beam search from the product of the shown inputs to the shown outputs.

    Each layer tries every op of ``OPS`` on every beam node (one expansion per
    node). A successor equal to the shown outputs ends the search. Otherwise
    one ``scorer.log_probs`` call scores every node that has a successor, over
    the ops the machine accepted there; successors are ranked by parent score
    plus log-probability (ties: parent order, then op order) and kept, up to
    ``beam_width``, unless the #70 quotation guard blocks them or their
    product has been kept before.

    ``budget`` caps executions: an attempt that could step more stacks than
    are left stops the search, so ``executions`` never exceeds it.
    """
    if beam_width <= 0 or max_depth <= 0:
        raise ValueError("beam_width and max_depth must be positive")
    if guard not in MODES:
        raise ValueError(f"unknown guard mode {guard!r}; choose one of: {MODES}")
    started = time.perf_counter()
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    # As in bfs_solve: "close" is sound only for mark-free targets.
    if guard in ("close", "both") and any(
        isinstance(v, Mark) for stack in outputs for v in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    target = key_of(outputs)
    targets = encode_targets(outputs)
    width = len(inputs)
    executions = fuel = expansions = 0

    def result(program, termination, depth) -> BeamResult:
        return BeamResult(
            program=program,
            termination=termination,
            seconds=time.perf_counter() - started,
            executions=executions,
            fuel=fuel,
            expansions=expansions,
            depth=depth,
        )

    beam = [_Node(0.0, inputs, ())]
    seen = {key_of(inputs)}
    for depth in range(max_depth):
        # successors[i]: (op index into OP_NAMES, moved product) for beam[i]
        successors: list[list[tuple[int, Product]]] = []
        for node in beam:
            expansions += 1
            found: list[tuple[int, Product]] = []
            for op in OPS:
                if budget is not None and executions + width > budget:
                    return result(None, "budget", depth + 1)
                moved, spent, stepped = step_all(node.product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                if key_of(moved) == target:
                    return result((*node.program, op.name), "fit", depth + 1)
                found.append((OP_INDEX[op.name], moved))
            successors.append(found)

        scored = [index for index, found in enumerate(successors) if found]
        if not scored:
            return result(None, "frontier_exhausted", depth)
        legal = np.zeros((len(scored), len(OP_NAMES)), dtype=bool)
        for row, index in enumerate(scored):
            for op_index, _ in successors[index]:
                legal[row, op_index] = True
        blocks = [node_block(beam[index].product, targets) for index in scored]
        log_probs = np.asarray(scorer.log_probs(blocks, legal), dtype=np.float64)

        ranked: list[tuple[float, int, int, Product]] = []
        for row, index in enumerate(scored):
            parent = beam[index].score
            for op_index, moved in successors[index]:
                ranked.append(
                    (parent + float(log_probs[row, op_index]), index, op_index, moved)
                )
        ranked.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))

        remaining = max_depth - depth - 1
        nxt: list[_Node] = []
        for score, index, op_index, moved in ranked:
            if len(nxt) >= beam_width:
                break
            if guard != "off" and blocked(moved[0], remaining, guard) is not None:
                continue
            moved_key = key_of(moved)
            if moved_key in seen:
                continue
            seen.add(moved_key)
            nxt.append(_Node(score, moved, (*beam[index].program, OP_NAMES[op_index])))
        if not nxt:
            return result(None, "frontier_exhausted", depth)
        beam = nxt
    return result(None, "max_depth", max_depth)
