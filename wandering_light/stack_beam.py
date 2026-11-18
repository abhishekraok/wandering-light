"""Policy-guided search over the product of shown stack PBE examples."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from wandering_light.quotation_guard import MODES, blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    parse_stack,
    quotation_open,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OPS
from wandering_light.stack_policy import encode_item, encode_targets, item_array

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_machine import Op
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import Product

type Termination = Literal["fit", "max_depth", "frontier_exhausted", "budget"]


@dataclass(frozen=True)
class BeamResult:
    program: tuple[str, ...] | None
    termination: Termination
    seconds: float
    executions: int
    fuel: int
    expansions: int
    depth: int


def step_all(product: Product, op: Op) -> tuple[Product | None, int, int]:
    """Try one op on the product, charging the first refusing stack as stepped."""
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


class _Scorer(Protocol):
    def log_probs(
        self, blocks: Sequence[np.ndarray], legal: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _Node:
    product: Product
    program: tuple[str, ...]
    score: float


def beam_solve(
    task: Task,
    scorer: _Scorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
) -> BeamResult:
    """Search for a nonempty program fitting every shown input/output pair.

    The budget counts stacks stepped and is checked between product attempts.
    A single attempt may step several stacks.
    """
    started = time.perf_counter()
    executions = fuel = expansions = 0

    def finish(
        termination: Termination, depth: int, program: tuple[str, ...] | None = None
    ) -> BeamResult:
        return BeamResult(
            program,
            termination,
            time.perf_counter() - started,
            executions,
            fuel,
            expansions,
            depth,
        )

    if beam_width <= 0 or max_depth < 0 or (budget is not None and budget < 0):
        raise ValueError(
            "Beam width must be positive and depth and budget non-negative"
        )
    if guard not in MODES:
        raise ValueError(f"unknown guard mode {guard!r}; choose one of: {MODES}")
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("A task must have at least one aligned shown example")
    if guard in ("close", "both") and any(map(quotation_open, outputs)):
        raise ValueError("the close guard needs mark-free targets")
    targets = encode_targets(outputs)
    beam = [_Node(inputs, (), 0.0)]
    seen = {inputs}

    for depth in range(max_depth):
        legal = np.zeros((len(beam), len(OPS)), dtype=np.bool_)
        successors: list[tuple[int, int, Product]] = []
        for parent_index, node in enumerate(beam):
            for action, op in enumerate(OPS):
                if budget is not None and executions >= budget:
                    return finish("budget", depth + 1)
                if action == 0:
                    expansions += 1
                moved, spent, stepped = step_all(node.product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                if moved == outputs:
                    return finish("fit", depth + 1, (*node.program, op.name))
                legal[parent_index, action] = True
                successors.append((parent_index, action, moved))
        if not successors:
            return finish("frontier_exhausted", depth)

        blocks = [
            item_array(
                [
                    encode_item(stack, target)
                    for stack, target in zip(node.product, targets, strict=True)
                ]
            )
            for node in beam
        ]
        log_probs = scorer.log_probs(blocks, legal)
        ranked = sorted(
            successors,
            key=lambda entry: (
                beam[entry[0]].score + float(log_probs[entry[0], entry[1]])
            ),
            reverse=True,
        )
        next_beam = []
        remaining = max_depth - depth - 1
        for parent_index, action, moved in ranked:
            if moved in seen:
                continue
            if guard != "off" and blocked(moved[0], remaining, guard) is not None:
                continue
            parent = beam[parent_index]
            next_beam.append(
                _Node(
                    moved,
                    (*parent.program, OPS[action].name),
                    parent.score + float(log_probs[parent_index, action]),
                )
            )
            seen.add(moved)
            if len(next_beam) == beam_width:
                break
        if not next_beam:
            return finish("frontier_exhausted", depth + 1)
        beam = next_beam

    return finish("max_depth", max_depth)
