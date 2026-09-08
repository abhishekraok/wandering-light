"""Policy-guided beam search over `mark-28` for a PBE task.

The frame is `bfs_solve` in `experiments/stack_pbe_baseline.py`: the state is
the *product* of the task's shown examples, every op is tried on every beam
node, a successor that reproduces every shown output is the hit, and nothing
here looks at the held-out examples. What the policy adds is the choice of
which successors survive: each layer keeps the ``beam_width`` successors with
the highest cumulative log-probability under the policy, normalised per node
over the ops the machine allows there.

Costs, per `docs/metrics.md`: wall-clock to the hit is primary and charges the
policy's forward passes. ``executions`` is the device-free diagnostic -- one per
op the machine ran on one example stack, the analogue of a type-compatible
basis-function execution; an op the guard refuses costs nothing, like a
type-incompatible function. It and ``fuel`` are charged where the work happens,
in expansion, so successors the beam then discards are still paid for, and the
budget is checked before every attempt. ``expansions`` counts beam nodes and
reads against the BFS rows' node budgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wandering_light.quotation_guard import blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Mark,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, OPS
from wandering_light.stack_policy import encode_item, encode_targets, item_array

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

Product = tuple[tuple, ...]  # one stack per shown example


@dataclass(frozen=True, slots=True)
class BeamResult:
    program: tuple[str, ...] | None
    termination: str  # fit | max_depth | frontier_exhausted | budget
    seconds: float  # to the hit, or to termination
    executions: int
    fuel: int
    expansions: int
    depth: int


def key_of(product: Product) -> tuple:
    return tuple(tuple(value_key(v) for v in stack) for stack in product)


def step_all(product: Product, op) -> tuple[Product | None, int, int]:
    """Apply ``op`` to every example: ``(product or None, fuel, stacks stepped)``.

    The machine is asked example by example and stops at the first refusal, so
    the last value is what the attempt cost even when it fails.
    """
    moved = []
    fuel = 0
    for stepped, stack in enumerate(product):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None, fuel, stepped
        moved.append(outcome[0])
        fuel += outcome[1]
    return tuple(moved), fuel, len(product)


def beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
) -> BeamResult:
    started = time.perf_counter()
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and any(
        isinstance(v, Mark) for stack in outputs for v in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    target = key_of(outputs)
    targets = encode_targets(outputs)
    seen = {key_of(inputs)}
    beam: list[tuple[Product, tuple[str, ...], float]] = [(inputs, (), 0.0)]
    executions = fuel = expansions = 0

    def result(program, termination, depth):
        return BeamResult(
            program,
            termination,
            time.perf_counter() - started,
            executions,
            fuel,
            expansions,
            depth,
        )

    for depth in range(max_depth):
        remaining = max_depth - depth - 1
        successors = []  # (node index, op index, product, program)
        legal = np.zeros((len(beam), len(OPS)), dtype=bool)
        for node, (product, program, _) in enumerate(beam):
            expansions += 1
            for index, op in enumerate(OPS):
                if budget is not None and executions >= budget:
                    return result(None, "budget", depth + 1)
                moved, spent, stepped = step_all(product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                legal[node, index] = True
                extended = (*program, op.name)
                if key_of(moved) == target:
                    return result(extended, "fit", depth + 1)
                successors.append((node, index, moved, extended))
        if not successors:
            return result(None, "frontier_exhausted", depth)
        blocks = [
            item_array(
                [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
            )
            for product, _, _ in beam
        ]
        log_probs = scorer.log_probs(blocks, legal)
        ranked = sorted(
            successors,
            key=lambda s: beam[s[0]][2] + log_probs[s[0], s[1]],
            reverse=True,
        )
        next_beam = []
        for node, index, product, program in ranked:
            if guard != "off" and blocked(product[0], remaining, guard) is not None:
                continue
            key = key_of(product)
            if key in seen:
                continue
            seen.add(key)
            next_beam.append((product, program, beam[node][2] + log_probs[node, index]))
            if len(next_beam) == beam_width:
                break
        if not next_beam:
            return result(None, "frontier_exhausted", depth + 1)
        beam = next_beam
    return result(None, "max_depth", max_depth)
