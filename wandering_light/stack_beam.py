"""Policy-guided beam search over ``mark-28`` product states.

``bfs_solve`` in ``experiments/stack_pbe_baseline.py`` keeps every state it
reaches, so what it can afford is set by how fast the frontier grows. This
keeps ``beam_width`` states per layer and lets a one-step policy from
``wandering_light/stack_policy.py`` choose which. Everything else is the same
search: the state is the product of the shown examples, a successor that
equals the shown outputs is the answer, and the held-out examples are never
read.

What a search *costs* is the reason this is a separate module. Both counters
are charged for every attempt, whether or not the machine accepted it:

* ``executions`` -- stacks stepped. An op tried on a five-example product is
  five executions, or fewer if an early example refuses it.
* ``fuel`` -- what the machine reports those steps spent, which is more than
  one per step once a combinator runs a quotation.

``budget`` is a cap on ``executions``, checked before each attempt, so a
search can finish at most one attempt over it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

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
from wandering_light.stack_pbe import BASIS, OPS, Task
from wandering_light.stack_policy import encode_item, encode_targets, item_array

Product = tuple[tuple[Value, ...], ...]


class Scorer(Protocol):
    """What the search needs from a policy; ``PolicyScorer`` is the real one."""

    def log_probs(self, blocks: list[np.ndarray], legal: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class BeamResult:
    """How one search ended and what it spent getting there.

    ``depth`` is the length of ``program`` on a ``fit``, the layer being built
    when a ``budget`` ran out, the last layer that held a live state on
    ``frontier_exhausted``, and ``max_depth`` otherwise.
    """

    program: tuple[str, ...] | None
    termination: str  # fit | max_depth | frontier_exhausted | budget
    seconds: float
    executions: int
    fuel: int
    expansions: int
    depth: int


def product_key(product: Product) -> tuple:
    return tuple(tuple(value_key(value) for value in stack) for stack in product)


def step_all(product: Product, op: Op) -> tuple[Product | None, int, int]:
    """Run ``op`` on every example: ``(product | None, fuel, stacks stepped)``.

    Stops at the first example that refuses, so the count is what the attempt
    cost rather than what it would have cost had it worked. The refusing step
    is counted -- it ran -- but reports no fuel, so it adds none.
    """
    moved = []
    fuel = 0
    for stepped, stack in enumerate(product, start=1):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None, fuel, stepped
        moved.append(outcome[0])
        fuel += outcome[1]
    return tuple(moved), fuel, len(moved)


def beam_solve(
    task: Task,
    scorer: Scorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
) -> BeamResult:
    """Search for a program fitting the *shown* examples, guided by ``scorer``.

    Each layer tries every op on every beam state. A successor matching the
    shown outputs ends the search. Otherwise the scorer is asked once for the
    whole layer, successors are ranked by their parent's score plus the op's
    log-probability among that parent's legal ops, and the best ``beam_width``
    that the guard allows and that no beam has held before go forward.

    ``guard`` is the open-quotation predicate of ``quotation_guard``, given
    the ops left under ``max_depth``. Its ``close`` arm is sound only for
    mark-free targets, so it refuses a task whose outputs carry one.
    """
    started = time.perf_counter()
    if guard not in MODES:
        raise ValueError(f"unknown guard mode {guard!r}; choose one of: {MODES}")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must not be negative")
    inputs: Product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs: Product = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("a task needs one shown output for each shown input")
    if guard in ("close", "both") and any(
        isinstance(value, Mark) for stack in outputs for value in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    target = product_key(outputs)
    targets = encode_targets(outputs)

    executions = fuel = expansions = 0

    def done(program: tuple[str, ...] | None, termination: str, depth: int):
        return BeamResult(
            program=program,
            termination=termination,
            seconds=time.perf_counter() - started,
            executions=executions,
            fuel=fuel,
            expansions=expansions,
            depth=depth,
        )

    # A beam entry is (product, program so far, summed log-probability).
    beam: list[tuple[Product, tuple[str, ...], float]] = [(inputs, (), 0.0)]
    seen = {product_key(inputs)}
    for depth in range(max_depth):
        # (parent, op) -> the product it leads to, in beam order then op order.
        successors: list[tuple[int, int, Product, tuple]] = []
        legal = np.zeros((len(beam), len(OPS)), dtype=bool)
        for parent, (product, program, _score) in enumerate(beam):
            expansions += 1
            for column, op in enumerate(OPS):
                if budget is not None and executions >= budget:
                    return done(None, "budget", depth + 1)
                moved, spent, stepped = step_all(product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                key = product_key(moved)
                if key == target:
                    return done((*program, op.name), "fit", depth + 1)
                legal[parent, column] = True
                successors.append((parent, column, moved, key))
        if not successors:
            return done(None, "frontier_exhausted", depth)

        blocks = [
            item_array(
                [
                    encode_item(stack, tokens)
                    for stack, tokens in zip(product, targets, strict=True)
                ]
            )
            for product, _program, _score in beam
        ]
        log_probs = scorer.log_probs(blocks, legal)
        scores = [
            beam[parent][2] + float(log_probs[parent, column])
            for parent, column, _moved, _key in successors
        ]
        # Best first; a tie goes to the earlier parent, then the earlier op,
        # which is the order the successors were generated in.
        ranked = sorted(range(len(successors)), key=lambda index: -scores[index])

        # Ops a kept successor may still spend under the depth cap, which is
        # what makes the close guard sound.
        remaining = max_depth - depth - 1
        kept: list[tuple[Product, tuple[str, ...], float]] = []
        for index in ranked:
            parent, column, moved, key = successors[index]
            # Every example shares one mark structure, so the first stack is
            # enough to judge the open body.
            if guard != "off" and blocked(moved[0], remaining, guard) is not None:
                continue
            if key in seen:
                continue
            seen.add(key)
            kept.append((moved, (*beam[parent][1], OPS[column].name), scores[index]))
            if len(kept) == beam_width:
                break
        if not kept:
            return done(None, "frontier_exhausted", depth)
        beam = kept
    return done(None, "max_depth", max_depth)
