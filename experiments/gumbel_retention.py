"""Experiment-local fresh Gumbel retention (#183), not exact SBS.

The search body is copied from production stack_beam at 205a661. Only its
retention ordering changes; rng=None preserves native production arithmetic.
No data, models or reference implementations are loaded by this module.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import numpy as np

from wandering_light.quotation_guard import blocked
from wandering_light.stack_beam import (
    BeamObserver,
    BeamResult,
    atoms_exceed,
    block_encoder,
    spent_in,
    step_all_explained,
    step_all_quiet,
)
from wandering_light.stack_machine import Mark, parse_stack
from wandering_light.stack_pbe import BASIS, OPS

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer


def jitter_beam_solve(task, scorer, *, rng=None, **kwargs):
    """Ordinary beam GC lifecycle with a separate experimental retention key."""
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        return _search(task, scorer, rng=rng, **kwargs)
    finally:
        if collecting:
            gc.enable()


def _search(
    task: Task,
    scorer: PolicyScorer,
    *,
    rng=None,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
    observer: BeamObserver | None = None,
    max_state_atoms: int | None = None,
    deadline: float | None = None,
    budget_unit: str = "executions",
) -> BeamResult:
    started = time.perf_counter()
    spent_in(budget_unit, 0, 0)  # refuse an unknown unit before any work
    by_executions = budget_unit == "executions"
    watch = observer if observer is not None else BeamObserver()
    watching = observer is not None
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and any(
        isinstance(v, Mark) for stack in outputs for v in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    encode = block_encoder(scorer, outputs)
    # The visited set is keyed on the product itself. Membership is decided by
    # ``Value.__eq__``, which is structural and induces exactly ``value_key``'s
    # partition (the union is closed and each variant carries a distinct class);
    # the hash only buckets, never leaves the process and never decides a tie,
    # so the search this set drives is the same search ``key_of`` drove, about
    # three times cheaper. ``key_of`` stays the name for keys that are written
    # down or compared across processes.
    seen = {inputs}
    beam: list = [(inputs, (), 0.0)]
    executions = fuel = expansions = 0
    pruned = False

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
            pruned,
        )

    for depth in range(max_depth):
        remaining = max_depth - depth - 1
        successors = []  # (node index, op index, product, program)
        legal = np.zeros((len(beam), len(OPS)), dtype=bool)
        watch.layer_started(depth, beam)
        for node, (product, program, _) in enumerate(beam):
            expansions += 1
            for index, op in enumerate(OPS):
                if budget is not None and (
                    executions >= budget
                    if by_executions
                    else spent_in(budget_unit, executions, fuel) >= budget
                ):
                    return result(None, "budget", depth + 1)
                if deadline is not None and time.perf_counter() >= deadline:
                    return result(None, "wall_clock_cap", depth + 1)
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
        blocks = [encode(product) for product, _, _ in beam]
        log_probs = scorer.log_probs(blocks, legal)
        watch.layer_scored(depth, log_probs, legal)
        # Keys alone receive fresh noise. Retained path scores below stay native.
        scores = [beam[s[0]][2] + log_probs[s[0], s[1]] for s in successors]
        if rng is not None:
            scores64 = np.asarray(scores, dtype=np.float64)
            if not np.isfinite(scores64).all():
                raise ValueError("nonfinite executable cumulative score")
            priorities = scores64 + rng.gumbel(size=len(scores))
            if not np.isfinite(priorities).all():
                raise ValueError("nonfinite retention key")
        else:
            priorities = scores
        order = sorted(range(len(successors)), key=priorities.__getitem__, reverse=True)
        ranked = [successors[i] for i in order]
        next_beam = []
        verdicts: dict[tuple[int, int], str] = {}
        parents: list[tuple[int, int]] = []
        for rank, (node, index, product, program) in enumerate(ranked):
            if guard != "off":
                why = blocked(product[0], remaining, guard)
                if why is not None:
                    verdicts[node, index] = f"guard:{why}"
                    continue
            if max_state_atoms is not None and atoms_exceed(product, max_state_atoms):
                verdicts[node, index] = "guard:size"
                continue
            if product in seen:
                verdicts[node, index] = "duplicate"
                continue
            seen.add(product)
            verdicts[node, index] = "retained"
            parents.append((node, index))
            next_beam.append((product, program, beam[node][2] + log_probs[node, index]))
            if len(next_beam) == beam_width:
                pruned = pruned or rank + 1 < len(ranked)
                break
        watch.layer_retained(depth, next_beam, verdicts, parents)
        if not next_beam:
            return result(None, "frontier_exhausted", depth + 1)
        beam = next_beam
    return result(None, "max_depth", max_depth)
