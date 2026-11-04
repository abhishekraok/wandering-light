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

``max_state_atoms`` bounds the size of a state the beam will keep. ``dup`` and
``cat`` cost one fuel whatever they copy, so ``dup cat`` doubles a list for two
fuel and a policy that likes the pair can park the beam on states of millions
of atoms, where hashing, comparing and encoding them is all the search does
(beam-state-size-unbounded, #157). A successor past the bound is refused like a
guard verdict; a bound far above any witness's peak state changes no search that
stays small.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wandering_light.quotation_guard import RUNS_CODE, blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
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

OP_INDEX: dict[str, int] = {op.name: index for index, op in enumerate(OPS)}


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
        ``guard:close``, ``guard:type`` or ``guard:size``; a successor the ranking never
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


def replay_program(
    program: Sequence[str], inputs: Product, outputs: Product
) -> tuple[bool, int, int]:
    """Does ``program`` take every shown input to its shown output, and at what cost?

    Charged like the beam: one execution per op run on one example stack, and an
    op the machine refuses ends the attempt. Returns ``(fits, executions, fuel)``.
    A library of solved programs is replayed with this before a task is searched
    (solved-program-cache, #151); a fit here says nothing about the held-out
    examples, which `stack_pbe_eval.grade` checks afterwards.
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


def atoms_exceed(product: Product, cap: int) -> bool:
    """Does the product hold more than ``cap`` atoms? Stops counting at the cap.

    An atom is an int, bool, symbol or mark, or a list cell; a list of n ints is
    n + 1 atoms. Iterative, so a deeply nested value cannot overflow the stack,
    and early-exit, so a million-atom state costs ``cap`` steps to refuse.
    """
    count = 0
    pending = [value for stack in product for value in stack]
    while pending:
        value = pending.pop()
        count += 1
        if count > cap:
            return True
        if isinstance(value, Lst):
            pending.extend(value.items)
    return False


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


def beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    budget: int | None = None,
    observer: BeamObserver | None = None,
    max_state_atoms: int | None = None,
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
            max_state_atoms=max_state_atoms,
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
    max_state_atoms: int | None = None,
) -> BeamResult:
    started = time.perf_counter()
    watch = observer if observer is not None else BeamObserver()
    watching = observer is not None
    inputs, outputs, targets = _prepare(task, guard)
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
                break
        watch.layer_retained(depth, next_beam, verdicts, parents)
        if not next_beam:
            return result(None, "frontier_exhausted", depth + 1)
        beam = next_beam
    return result(None, "max_depth", max_depth)


# ---------------------------------------------------------------------------
# Segment-reset beam
# ---------------------------------------------------------------------------
#
# A *closed state* is what a top-level combinator application leaves: the
# successor of an op in ``RUNS_CODE`` applied to a product with no ``Mark`` on
# any stack. Between two closed states a witness builds a quotation and runs it,
# and the flat beam has to carry every half-built body across the whole depth
# alongside every finished one. The segment beam resets at closed states: each
# segment runs a short beam per root, collects the closed states it reaches, and
# the best of them are the next segment's roots.


def _has_mark(product: Product) -> bool:
    return any(isinstance(v, Mark) for stack in product for v in stack)


def closed_cuts(states: Sequence[Product], program: Sequence[str]) -> list[int]:
    """``t + 1`` for every step that leaves a closed state.

    ``states[t]`` is the product before ``program[t]``; a step closes when its
    op runs code and ``states[t]`` carries no mark.
    """
    return [
        t + 1
        for t, name in enumerate(program)
        if name in RUNS_CODE and not _has_mark(states[t])
    ]


@dataclass(slots=True)
class _Tally:
    executions: int = 0
    fuel: int = 0
    expansions: int = 0
    deepest: int = 0  # ops in the longest program any expansion produced


def _prepare(task: Task, guard: str) -> tuple[Product, Product, list]:
    """The task's parsed inputs and outputs and the encoded targets."""
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and _has_mark(outputs):
        raise ValueError("the close guard needs mark-free targets")
    return inputs, outputs, encode_targets(outputs)


def _run_segment(
    roots: Sequence[Node],
    outputs: Product,
    targets,
    scorer: PolicyScorer,
    tally: _Tally,
    *,
    beam_width: int,
    segment_depth: int,
    horizon: int,
    guard: str,
    budget: int | None,
    max_state_atoms: int | None,
    stop_at_fit: bool,
) -> tuple[str | None, tuple[str, ...] | None, dict[Product, tuple[float, tuple]]]:
    """One segment: a beam per root, layer by layer, collecting closed states.

    Returns ``(stop, program, closed)``: ``stop`` is ``"fit"``, ``"budget"`` or
    None, and ``closed`` maps each closed product reached to its best
    ``(score, program)``, in the order first reached.
    """
    beams: list[list[Node]] = [[root] for root in roots]
    visited: list[set[Product]] = [{root[0]} for root in roots]
    closed: dict[Product, tuple[float, tuple[str, ...]]] = {}
    for layer in range(segment_depth):
        last = layer == segment_depth - 1
        for which, beam in enumerate(beams):
            if not beam:
                continue
            successors = []  # (node, op index, product, program, closes)
            legal = np.zeros((len(beam), len(OPS)), dtype=bool)
            for node, (product, program, _) in enumerate(beam):
                tally.expansions += 1
                open_parent = not _has_mark(product)
                for index, op in enumerate(OPS):
                    if budget is not None and tally.executions >= budget:
                        return "budget", None, closed
                    moved, spent, stepped = step_all_quiet(product, op)
                    tally.executions += stepped
                    tally.fuel += spent
                    if moved is None:
                        continue
                    legal[node, index] = True
                    extended = (*program, op.name)
                    tally.deepest = max(tally.deepest, len(extended))
                    if stop_at_fit and moved == outputs:
                        return "fit", extended, closed
                    closes = open_parent and op.name in RUNS_CODE
                    successors.append((node, index, moved, extended, closes))
            next_beam: list[Node] = []
            if successors:
                blocks = [
                    item_array(
                        [
                            encode_item(s, t)
                            for s, t in zip(product, targets, strict=True)
                        ]
                    )
                    for product, _, _ in beam
                ]
                log_probs = scorer.log_probs(blocks, legal)
                ranked = sorted(
                    successors,
                    key=lambda s: beam[s[0]][2] + log_probs[s[0], s[1]],
                    reverse=True,
                )
                seen = visited[which]
                for node, index, product, program, closes in ranked:
                    if guard != "off" and (
                        blocked(product[0], horizon - len(program), guard) is not None
                    ):
                        continue
                    if max_state_atoms is not None and atoms_exceed(
                        product, max_state_atoms
                    ):
                        continue
                    score = beam[node][2] + log_probs[node, index]
                    if closes:
                        held = closed.get(product)
                        if held is None or score > held[0]:
                            closed[product] = (score, program)
                        continue
                    if last or len(next_beam) == beam_width or product in seen:
                        continue
                    seen.add(product)
                    next_beam.append((product, program, score))
            beams[which] = next_beam
    return None, None, closed


def _rank_closed(
    closed: Mapping[Product, tuple[float, tuple[str, ...]]], expanded: set[Product]
) -> list[Node]:
    """Unexpanded closed states, best score first, ties in merge order."""
    fresh = [
        (product, program, score)
        for product, (score, program) in closed.items()
        if product not in expanded
    ]
    fresh.sort(key=lambda node: node[2], reverse=True)
    return fresh


def segment_beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int = 64,
    roots: int = 16,
    segment_depth: int = 12,
    max_segments: int = 12,
    guard: str = "both",
    budget: int | None = None,
    max_state_atoms: int | None = None,
) -> BeamResult:
    """Search one task by segments that reset at closed states.

    The cyclic collector is off for the duration, as in ``beam_solve``.
    Terminations: ``fit``, ``budget`` (over the whole search),
    ``frontier_exhausted`` (no new closed state) and ``max_segments``.
    """
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        return _segment_beam_solve(
            task,
            scorer,
            beam_width=beam_width,
            roots=roots,
            segment_depth=segment_depth,
            max_segments=max_segments,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
        )
    finally:
        if collecting:
            gc.enable()


def _segment_beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    beam_width: int,
    roots: int,
    segment_depth: int,
    max_segments: int,
    guard: str,
    budget: int | None,
    max_state_atoms: int | None,
) -> BeamResult:
    started = time.perf_counter()
    inputs, outputs, targets = _prepare(task, guard)
    tally = _Tally()
    horizon = max_segments * segment_depth

    def result(program, termination):
        return BeamResult(
            program,
            termination,
            time.perf_counter() - started,
            tally.executions,
            tally.fuel,
            tally.expansions,
            len(program) if program is not None else tally.deepest,
        )

    frontier: list[Node] = [(inputs, (), 0.0)]
    expanded: set[Product] = set()
    for _segment in range(max_segments):
        expanded.update(product for product, _, _ in frontier)
        stop, program, closed = _run_segment(
            frontier,
            outputs,
            targets,
            scorer,
            tally,
            beam_width=beam_width,
            segment_depth=segment_depth,
            horizon=horizon,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
            stop_at_fit=True,
        )
        if stop == "fit":
            return result(program, "fit")
        if stop == "budget":
            return result(None, "budget")
        frontier = _rank_closed(closed, expanded)[:roots]
        if not frontier:
            return result(None, "frontier_exhausted")
    return result(None, "max_segments")


def _witness_log_prob(
    states: Sequence[Product],
    program: Sequence[str],
    *,
    start: int,
    stop: int,
    targets,
    scorer: PolicyScorer,
) -> float:
    """The policy's log-probability of ``program[start:stop]`` along its states.

    Normalised over the ops the machine allows at each state, as the beam
    scores. Diagnostic only: the executions spent here are not charged.
    """
    total = 0.0
    for t in range(start, stop):
        product = states[t]
        legal = np.zeros((1, len(OPS)), dtype=bool)
        for index, op in enumerate(OPS):
            legal[0, index] = step_all_quiet(product, op)[0] is not None
        block = item_array(
            [encode_item(s, g) for s, g in zip(product, targets, strict=True)]
        )
        total += float(scorer.log_probs([block], legal)[0, OP_INDEX[program[t]]])
    return total


def segment_witness_ranks(
    task: Task,
    scorer: PolicyScorer,
    program: Sequence[str] | None = None,
    *,
    beam_width: int = 64,
    roots: int = 16,
    segment_depth: int = 12,
    max_segments: int = 12,
    guard: str = "both",
    budget: int | None = None,
    max_state_atoms: int | None = None,
) -> list[dict]:
    """Where the witness's closed states rank, segment by segment (the K diagnostic).

    The search runs as ``segment_beam_solve`` does without stopping at a fit,
    one segment per closed cut of the witness. A witness root that fell out of
    the frontier is appended (``forced``) so the next segment still asks the
    question. One row per segment; see the keys below.
    """
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        return _segment_witness_ranks(
            task,
            scorer,
            task.program if program is None else tuple(program),
            beam_width=beam_width,
            roots=roots,
            segment_depth=segment_depth,
            max_segments=max_segments,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
        )
    finally:
        if collecting:
            gc.enable()


def _segment_witness_ranks(
    task: Task,
    scorer: PolicyScorer,
    program: tuple[str, ...],
    *,
    beam_width: int,
    roots: int,
    segment_depth: int,
    max_segments: int,
    guard: str,
    budget: int | None,
    max_state_atoms: int | None,
) -> list[dict]:
    inputs, outputs, targets = _prepare(task, guard)
    states: list[Product] = [inputs]
    for step, name in enumerate(program):
        moved, _spent, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None:
            raise ValueError(
                f"{task.behaviour}: the program refuses step {step} ({name})"
            )
        states.append(moved)
    cuts = closed_cuts(states, program)[:max_segments]
    tally = _Tally()
    horizon = max_segments * segment_depth
    frontier: list[Node] = [(inputs, (), 0.0)]
    witness: Node = frontier[0]
    expanded: set[Product] = set()
    rows: list[dict] = []
    previous = 0
    for segment, cut in enumerate(cuts):
        forced = all(product != witness[0] for product, _, _ in frontier)
        if forced:
            frontier = [*frontier, witness]
        expanded.update(product for product, _, _ in frontier)
        stop, _, closed = _run_segment(
            frontier,
            outputs,
            targets,
            scorer,
            tally,
            beam_width=beam_width,
            segment_depth=segment_depth,
            horizon=horizon,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
            stop_at_fit=False,
        )
        ranked = _rank_closed(closed, expanded)
        target = states[cut]
        rank = next(
            (at for at, (product, _, _) in enumerate(ranked, 1) if product == target),
            None,
        )
        complete = stop != "budget"
        rows.append(
            {
                "segment": segment,
                "ops": cut - previous,
                "roots": len(frontier),
                "forced": forced,
                "closed": len(ranked),
                "rank": rank,
                "score": float(ranked[rank - 1][2]) if rank is not None else None,
                "best": float(ranked[0][2]) if ranked else None,
                "executions": tally.executions,
                "complete": complete,
            }
        )
        if not complete:
            break
        if rank is not None:
            witness = ranked[rank - 1]
        else:
            witness = (
                target,
                program[:cut],
                witness[2]
                + _witness_log_prob(
                    states,
                    program,
                    start=previous,
                    stop=cut,
                    targets=targets,
                    scorer=scorer,
                ),
            )
        frontier = ranked[:roots]
        previous = cut
    return rows
