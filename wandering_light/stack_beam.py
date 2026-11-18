"""Policy-guided beam search over the product of a task's shown stacks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wandering_light import quotation_guard
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    Mark,
    parse_stack,
    step_raw,
)
from wandering_light.stack_machine_corpus import behaviour_key
from wandering_light.stack_pbe import BASIS, OPS
from wandering_light.stack_policy import encode_targets, product_items

if TYPE_CHECKING:
    from wandering_light.stack_machine import Op
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer, Product, Stack


@dataclass(frozen=True)
class BeamResult:
    program: tuple[str, ...] | None
    termination: str
    seconds: float
    executions: int
    fuel: int
    expansions: int
    depth: int


def step_all(product: Product, op: Op) -> tuple[Product | None, int, int]:
    moved = []
    fuel = 0
    for stepped, stack in enumerate(product, start=1):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None, fuel, stepped
        next_stack, spent = outcome
        moved.append(next_stack)
        fuel += spent
    return tuple(moved), fuel, len(product)


def _holds_mark(stack: Stack) -> bool:
    return any(
        isinstance(v, Mark) or (isinstance(v, Lst) and _holds_mark(v.items))
        for v in stack
    )


@dataclass(frozen=True, slots=True)
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
    """Charge stacks stepped, checking the budget before each product transition."""
    started = time.perf_counter()
    executions = fuel = expansions = 0

    def result(termination: str, depth: int, program=None) -> BeamResult:
        return BeamResult(
            program,
            termination,
            time.perf_counter() - started,
            executions,
            fuel,
            expansions,
            depth,
        )

    if beam_width < 1 or max_depth < 0:
        raise ValueError("beam_width must be positive and max_depth non-negative")
    if guard not in quotation_guard.MODES:
        raise ValueError(f"Unknown quotation guard {guard!r}")
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and any(_holds_mark(stack) for stack in outputs):
        raise ValueError("The close guard needs mark-free targets")
    targets = encode_targets(outputs)
    beam = [_Node(inputs, (), 0.0)]
    seen = {behaviour_key(inputs)}

    for depth in range(max_depth):
        legal = np.zeros((len(beam), len(OPS)), dtype=bool)
        successors = []
        for parent, node in enumerate(beam):
            expansions += 1
            for action, op in enumerate(OPS):
                if budget is not None and executions >= budget:
                    return result("budget", depth + 1)
                moved, spent, stepped = step_all(node.product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                if moved == outputs:
                    return result("fit", depth + 1, (*node.program, op.name))
                legal[parent, action] = True
                successors.append((parent, action, moved))
        if not successors:
            return result("frontier_exhausted", depth)

        blocks = [product_items(node.product, targets) for node in beam]
        log_probs = scorer.log_probs(blocks, legal)
        scores = log_probs + np.array([node.score for node in beam])[:, None]
        # Stable ties follow beam order, then basis order. Deduplicate after
        # ranking so the highest-scoring path to each product wins.
        successors.sort(key=lambda move: scores[move[0], move[1]], reverse=True)
        next_beam = []
        for parent, action, moved in successors:
            if guard != "off" and quotation_guard.blocked(
                moved[0], max_depth - depth - 1, guard
            ):
                continue
            key = behaviour_key(moved)
            if key in seen:
                continue
            seen.add(key)
            program = (*beam[parent].program, OPS[action].name)
            next_beam.append(_Node(moved, program, float(scores[parent, action])))
            if len(next_beam) == beam_width:
                break
        if not next_beam:
            return result("frontier_exhausted", depth)
        beam = next_beam
    return result("max_depth", max_depth)
