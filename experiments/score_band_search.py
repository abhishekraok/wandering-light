"""Experiment-local score-band retention (#179), never wired into production.

A bounded copy of frozen stack_beam._beam_solve (f9ae90b) preserves the disabled
path. Real machine/OPS and low-level production primitives remain unchanged.
Only retained states enter seen; a band stop can change later first fits.
Diagnostic telemetry is untimed software QA, not coverage or clock evidence.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from wandering_light.quotation_guard import blocked
from wandering_light.stack_beam import (
    BeamObserver,
    BeamResult,
    Node,
    Product,
    atoms_exceed,
    block_encoder,
    spent_in,
    step_all_explained,
    step_all_quiet,
)
from wandering_light.stack_machine import Mark, parse_stack
from wandering_light.stack_pbe import BASIS, OPS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer


@dataclass(frozen=True, slots=True)
class BandSearchResult:
    beam_result: BeamResult
    band_pruned: bool

    @property
    def width_pruned(self) -> bool:
        return self.beam_result.pruned


@dataclass(frozen=True, slots=True)
class BandLayer:
    depth: int
    retained: tuple[Node, ...]
    parents: tuple[tuple[int, int], ...]
    verdicts: dict[tuple[int, int], str]
    # Complete ranked identities/scores, but eligibility verdicts ONLY for read rows.
    ranked: tuple[tuple[int, int, Product, tuple[str, ...], float], ...]
    old_seen: frozenset[Product]
    anchor: float | None
    tau: float | None
    stop: str
    first_band_rejected: tuple[int, int] | None
    all_inf_fallback: bool
    width_pruned: bool
    band_pruned: bool


class BandObserver(BeamObserver):
    """Ordinary hooks plus a separate band hook; no widening/debugger integration."""

    def band_retained(self, layer: BandLayer) -> None:
        """Retained identities, exact stop precedence, and unknown unread tail."""


def _validate_band(value: Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("score_band_nats must be None or a finite nonnegative real")
    try:
        tau = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("score_band_nats must be finite") from error
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("score_band_nats must be finite and nonnegative")
    return tau


def band_beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
    observer: BeamObserver | None = None,
    max_state_atoms: int | None = None,
    deadline: float | None = None,
    budget_unit: str = "executions",
    score_band_nats: Real | None = None,
) -> BandSearchResult:
    """Flat beam with optional cumulative-score band and reference GC restoration."""
    tau = _validate_band(score_band_nats)  # before any parser/scorer/machine work
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        return _band_beam_solve(
            task,
            scorer,
            beam_width=beam_width,
            max_depth=max_depth,
            guard=guard,
            budget=budget,
            observer=observer,
            max_state_atoms=max_state_atoms,
            deadline=deadline,
            budget_unit=budget_unit,
            score_band_nats=tau,
        )
    finally:
        if collecting:
            gc.enable()


def _retain(
    depth: int,
    ranked,
    beam: Sequence[Node],
    log_probs,
    seen: set[Product],
    *,
    beam_width: int,
    remaining: int,
    guard: str,
    max_state_atoms: int | None,
    score_band_nats: float | None,
    capture: bool = True,
) -> BandLayer:
    """Stable scan: only read eligible scores can anchor or trigger the band.

    This helper has no transition/parser/op injection. Table-only arithmetic
    units may call it directly; end-to-end solver fixtures always use real OPS.
    """
    old_seen = frozenset(seen) if capture else frozenset()
    retained = []
    parents = []
    verdicts = {}
    anchor = None
    stop = "exhausted"
    rejected = None
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
        score = beam[node][2] + log_probs[node, index]
        if anchor is None:
            anchor = score
        if score_band_nats is not None and np.float64(score) < (
            np.float64(anchor) - np.float64(score_band_nats)
        ):
            verdicts[node, index] = "score_band"
            rejected = (node, index)
            stop = "score_band"  # including a rejection on the very last row
            break
        seen.add(product)
        verdicts[node, index] = "retained"
        parents.append((node, index))
        retained.append((product, program, score))
        if len(retained) == beam_width:
            stop = "width" if rank + 1 < len(ranked) else "exhausted"
            break
    return BandLayer(
        depth,
        tuple(retained),
        tuple(parents),
        verdicts,
        tuple((n, i, p, pr, beam[n][2] + log_probs[n, i]) for n, i, p, pr in ranked)
        if capture
        else (),
        old_seen,
        anchor,
        score_band_nats,
        stop,
        rejected,
        score_band_nats is not None and anchor is not None and np.isneginf(anchor),
        stop == "width",
        stop == "score_band",
    )


def _band_beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
    observer: BeamObserver | None = None,
    max_state_atoms: int | None = None,
    deadline: float | None = None,
    budget_unit: str = "executions",
    score_band_nats: float | None = None,
) -> BandSearchResult:
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
    beam: list[Node] = [(inputs, (), 0.0)]
    executions = fuel = expansions = 0
    pruned = False
    band_pruned = False

    def result(program, termination, depth):
        watch.search_terminated(termination, depth, executions, fuel, expansions)
        return BandSearchResult(
            BeamResult(
                program,
                termination,
                time.perf_counter() - started,
                executions,
                fuel,
                expansions,
                depth,
                pruned,
            ),
            band_pruned,
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
        # Validate every executable non-fit score BEFORE sorting or seen changes.
        # Guarded, duplicate and future unread candidates are still checked.
        if score_band_nats is not None:
            for node, index, _product, _program in successors:
                score = beam[node][2] + log_probs[node, index]
                if np.isnan(score) or np.isposinf(score):
                    raise ValueError("invalid executable cumulative score: NaN or +inf")
        ranked = sorted(
            successors,
            key=lambda s: beam[s[0]][2] + log_probs[s[0], s[1]],
            reverse=True,
        )
        layer = _retain(
            depth,
            ranked,
            beam,
            log_probs,
            seen,
            beam_width=beam_width,
            remaining=remaining,
            guard=guard,
            max_state_atoms=max_state_atoms,
            score_band_nats=score_band_nats,
            capture=watching,
        )
        next_beam = list(layer.retained)
        verdicts = layer.verdicts
        parents = layer.parents
        pruned = pruned or layer.width_pruned
        band_pruned = band_pruned or layer.band_pruned
        watch.layer_retained(depth, next_beam, verdicts, parents)
        if isinstance(watch, BandObserver):
            watch.band_retained(layer)
        if not next_beam:
            return result(None, "frontier_exhausted", depth + 1)
        beam = next_beam
    return result(None, "max_depth", max_depth)
