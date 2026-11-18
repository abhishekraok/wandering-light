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

import gc
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
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS
from wandering_light.stack_policy import encode_item, encode_targets, item_array

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.stack_machine import Op
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

Product = tuple[tuple, ...]  # one stack per shown example
Node = tuple[Product, tuple[str, ...], float]  # (product, program, log-probability)


class BeamObserver:
    """Told what ``beam_solve`` decided, in the order it decided it.

    Every hook is a no-op here. Nothing an observer returns is read, so a
    recorded search and an unrecorded one are the same search; the stack
    debugger's trace recorder is the one implementation, and
    ``tests/test_stack_debugger_trace.py`` asserts the transparency.
    """

    def layer_started(self, depth: int, beam: Sequence[Node]) -> None:
        """The nodes about to be expanded, in beam order."""

    def expansion_attempted(
        self,
        depth: int,
        node: int,
        op_index: int,
        successor: Product | None,
        reason: str | None,
        fuel: int,
        stepped: int,
    ) -> None:
        """One op tried on one node: the product it made, or why it could not."""

    def layer_scored(self, depth: int, log_probs, legal) -> None:
        """The policy's normalized rows for the layer, one per node."""

    def layer_retained(
        self,
        depth: int,
        beam: Sequence[Node],
        verdicts: Mapping[tuple[int, int], str],
        parents: Sequence[tuple[int, int]],
    ) -> None:
        """What survived, and for each ranked successor why it did or did not.

        ``verdicts`` maps ``(node, op_index)`` to ``retained``, ``duplicate``,
        ``guard:close`` or ``guard:type``; a successor the ranking never
        reached, because the beam filled first, is absent. ``parents[rank]`` is
        the ``(node, op_index)`` that produced ``beam[rank]``.
        """

    def search_terminated(
        self,
        termination: str,
        depth: int,
        executions: int,
        fuel: int,
        expansions: int,
    ) -> None:
        """The last thing said."""


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


def step_all_explained(
    product: Product, op: Op
) -> tuple[Product | None, int, int, str | None]:
    """``step_all`` plus the machine's reason when an example refuses the op.

    The reason names the example that refused, because a product moves only
    when every example moves: an op the first stack accepts and the third
    rejects is refused, and the display should say which one.
    """
    moved = []
    fuel = 0
    for stepped, stack in enumerate(product):
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None, fuel, stepped, f"example {stepped}: {outcome}"
        moved.append(outcome[0])
        fuel += outcome[1]
    return tuple(moved), fuel, len(product), None


def step_all_quiet(product: Product, op: Op) -> tuple[Product | None, int, int]:
    """``step_all_explained`` without the reason string.

    The reason is an f-string built on every refusal, and about half of the
    28 x beam attempts in a layer refuse. Nothing reads it unless an observer
    is watching, so the unobserved search skips building it.
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


def step_all(product: Product, op) -> tuple[Product | None, int, int]:
    """Apply ``op`` to every example: ``(product or None, fuel, stacks stepped)``.

    The machine is asked example by example and stops at the first refusal, so
    the last value is what the attempt cost even when it fails.
    """
    moved, fuel, stepped, _reason = step_all_explained(product, op)
    return moved, fuel, stepped


def replay_program(
    program: Sequence[str], inputs: Product, outputs: Product
) -> tuple[bool, int, int]:
    """Does ``program`` take every shown input to its shown output, and at what cost?

    Charged like the beam: one execution per op run on one example stack, and an
    op the machine refuses ends the attempt. Returns ``(fits, executions, fuel)``.
    """
    executions = 0
    fuel = 0
    product = inputs
    for name in program:
        moved, spent, stepped = step_all_quiet(product, OP_BY_NAME[name])
        executions += stepped
        fuel += spent
        if moved is None:
            return False, executions, fuel
        product = moved
    return product == outputs, executions, fuel


def beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
    observer: BeamObserver | None = None,
) -> BeamResult:
    """Search one task. The cyclic collector is off for the duration.

    Every object the expansion loop makes -- a product tuple per op per example,
    a program tuple per legal successor, the visited set's keys -- is acyclic,
    so reference counting frees all of it and the cyclic collector only walks
    it. It walks a lot of it: at depth 48 the collector is about half the
    wall-clock, and it is also most of the run-to-run variance in the number
    `docs/metrics.md` makes primary. The collector is restored on the way out,
    including when the search raises, so a caller that had it off keeps it off
    and cycles made during the search are collected between tasks rather than
    never. Callers that solve many tasks should call ``gc.freeze()`` once first,
    which moves the corpus and the model out of the generations being scanned.
    """
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        return _beam_solve(
            task,
            scorer,
            beam_width=beam_width,
            max_depth=max_depth,
            guard=guard,
            budget=budget,
            observer=observer,
        )
    finally:
        if collecting:
            gc.enable()


def _beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
    observer: BeamObserver | None = None,
) -> BeamResult:
    started = time.perf_counter()
    watch = observer if observer is not None else BeamObserver()
    watching = observer is not None
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and any(
        isinstance(v, Mark) for stack in outputs for v in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    targets = encode_targets(outputs)
    # The visited set is keyed on the product itself. Membership is decided by
    # ``Value.__eq__``, which is structural and induces exactly ``value_key``'s
    # partition (the union is closed and each variant carries a distinct class);
    # the hash only buckets, never leaves the process and never decides a tie,
    # so the search this set drives is the same search ``key_of`` drove, about
    # three times cheaper. ``key_of`` stays the name for keys that are written
    # down or compared across processes.
    seen = {inputs}
    beam: list[Node] = [(inputs, (), 0.0)]
    executions = fuel = expansions = 0

    def result(program, termination, depth):
        watch.search_terminated(termination, depth, executions, fuel, expansions)
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
        watch.layer_started(depth, beam)
        for node, (product, program, _) in enumerate(beam):
            expansions += 1
            for index, op in enumerate(OPS):
                if budget is not None and executions >= budget:
                    return result(None, "budget", depth + 1)
                if watching:
                    moved, spent, stepped, reason = step_all_explained(product, op)
                    watch.expansion_attempted(
                        depth, node, index, moved, reason, spent, stepped
                    )
                else:
                    moved, spent, stepped = step_all_quiet(product, op)
                executions += stepped
                fuel += spent
                if moved is None:
                    continue
                legal[node, index] = True
                extended = (*program, op.name)
                # Structural equality on the closed ``Value`` union is exactly
                # ``value_key`` equality, and short-circuits at the first
                # difference instead of building a key for the whole product.
                # Almost every successor differs in its first stack's depth.
                if moved == outputs:
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
        watch.layer_scored(depth, log_probs, legal)
        ranked = sorted(
            successors,
            key=lambda s: beam[s[0]][2] + log_probs[s[0], s[1]],
            reverse=True,
        )
        next_beam = []
        verdicts: dict[tuple[int, int], str] = {}
        parents: list[tuple[int, int]] = []
        for node, index, product, program in ranked:
            if guard != "off":
                why = blocked(product[0], remaining, guard)
                if why is not None:
                    verdicts[node, index] = f"guard:{why}"
                    continue
            if product in seen:
                verdicts[node, index] = "duplicate"
                continue
            seen.add(product)
            verdicts[node, index] = "retained"
            parents.append((node, index))
            next_beam.append((product, program, beam[node][2] + log_probs[node, index]))
            if len(next_beam) == beam_width:
                break
        watch.layer_retained(depth, next_beam, verdicts, parents)
        if not next_beam:
            return result(None, "frontier_exhausted", depth + 1)
        beam = next_beam
    return result(None, "max_depth", max_depth)
