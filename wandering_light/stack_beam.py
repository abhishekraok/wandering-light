"""Policy-guided beam search over ``mark-28`` product states.

The sibling of ``bfs_solve`` in ``experiments/stack_pbe_baseline.py``: same
state (the product of a task's shown inputs), same successor rule (every op of
``OPS`` stepped on every example, refusal on any example drops the successor),
same goal test and the same quotation guard. What differs is which successors
survive a layer: BFS keeps all of them, the beam keeps the ``beam_width`` with
the highest cumulative policy log-probability.

Cost is charged in the units ``bfs_solve`` is compared in: *executions*, one
per example stack stepped, whether or not the attempt is kept, and the fuel the
machine reports for each kept step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from wandering_light.quotation_guard import MODES, blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Mark,
    Op,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import OPS, Task
from wandering_light.stack_policy import (
    PolicyScorer,
    Product,
    encode_item,
    encode_targets,
    item_array,
    task_product,
)

TERMINATIONS = ("fit", "max_depth", "frontier_exhausted", "budget")


@dataclass(frozen=True)
class BeamResult:
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
    """``(product or None, fuel, stacks stepped)``.

    Stops at the first refusal, so ``stacks stepped`` counts the refused stack
    too: it is what the attempt cost. A refused step reports no fuel.
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
    product: Product
    program: tuple[str, ...]
    score: float


def beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
) -> BeamResult:
    """Search for a program fitting the shown examples, guided by ``scorer``.

    ``budget`` bounds executions (stacks stepped). It is checked before every
    attempt, so the search stops as soon as the spend reaches it.
    """
    if beam_width <= 0 or max_depth < 0:
        raise ValueError("beam_width must be positive and max_depth non-negative")
    if guard not in MODES:
        raise ValueError(f"unknown guard mode {guard!r}; choose one of: {MODES}")
    started = time.perf_counter()
    inputs, outputs = task_product(task)
    if guard in ("close", "both") and any(
        isinstance(v, Mark) for stack in outputs for v in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    target = key_of(outputs)
    targets = encode_targets(outputs)

    executions = fuel = expansions = 0

    def done(program, termination, depth) -> BeamResult:
        return BeamResult(
            program=program,
            termination=termination,
            seconds=time.perf_counter() - started,
            executions=executions,
            fuel=fuel,
            expansions=expansions,
            depth=depth,
        )

    beam = [_Node(inputs, (), 0.0)]
    seen = {key_of(inputs)}
    for depth in range(max_depth):
        remaining = max_depth - depth - 1
        successors: list[tuple[int, int, Product]] = []
        legal = np.zeros((len(beam), len(OPS)), dtype=bool)
        for node_index, node in enumerate(beam):
            expansions += 1
            for op_index, op in enumerate(OPS):
                if budget is not None and executions >= budget:
                    return done(None, "budget", depth + 1)
                moved, spent, stepped = step_all(node.product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                if key_of(moved) == target:
                    return done((*node.program, op.name), "fit", depth + 1)
                legal[node_index, op_index] = True
                successors.append((node_index, op_index, moved))
        if not successors:
            return done(None, "frontier_exhausted", depth)

        blocks = [
            item_array(
                [
                    encode_item(state, tokens)
                    for state, tokens in zip(node.product, targets, strict=True)
                ]
            )
            for node in beam
        ]
        log_probs = scorer.log_probs(blocks, legal)
        # Ties keep node order, then op order, so the search is deterministic.
        ranked = sorted(
            successors,
            key=lambda s: (
                -(beam[s[0]].score + float(log_probs[s[0], s[1]])),
                s[0],
                s[1],
            ),
        )
        nxt: list[_Node] = []
        for node_index, op_index, moved in ranked:
            if len(nxt) >= beam_width:
                break
            if guard != "off" and blocked(moved[0], remaining, guard) is not None:
                continue
            moved_key = key_of(moved)
            if moved_key in seen:
                continue
            seen.add(moved_key)
            parent = beam[node_index]
            nxt.append(
                _Node(
                    moved,
                    (*parent.program, OPS[op_index].name),
                    parent.score + float(log_probs[node_index, op_index]),
                )
            )
        if not nxt:
            return done(None, "frontier_exhausted", depth)
        beam = nxt
    return done(None, "max_depth", max_depth)
