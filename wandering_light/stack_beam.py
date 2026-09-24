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

``segment_beam_solve`` is the segment-reset beam of short-segment-campaign
(#168). A **closed state** is what a top-level combinator application leaves: an
op that runs a quotation (``RUNS_CODE``) applied to a product with no mark on any
stack, the ``run`` step of `experiments/staged_transfer.step_kinds` and the cut
of segment-composite (#160). From each root a flat beam runs until its paths
close; equal closed states are merged across roots, and the best ``roots`` of
them by cumulative log-probability start the next segment, each with a fresh
beam. ``segment_witness_ranks`` is the diagnostic that sizes ``roots``.
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
    # fit | max_depth | frontier_exhausted | budget; max_segments for the segment beam
    termination: str
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
# Segment-reset search (short-segment-campaign, #168)
# ---------------------------------------------------------------------------

#: Ops whose successor is closed when the product before them has no mark open.
RUN_INDICES = frozenset(index for index, op in enumerate(OPS) if op.name in RUNS_CODE)
OP_INDEX: dict[str, int] = {op.name: index for index, op in enumerate(OPS)}

# Merged closed states: product -> (best cumulative log-probability, its program).
Closed = dict[Product, tuple[float, tuple[str, ...]]]


def has_mark(product: Product) -> bool:
    return any(isinstance(value, Mark) for stack in product for value in stack)


def closed_cuts(states: Sequence[Product], program: Sequence[str]) -> list[int]:
    """Op counts after each top-level combinator application along ``program``.

    ``states[t]`` is the product before ``program[t]``. A cut at ``t + 1``
    means ``states[t + 1]`` is closed, as `staged_transfer.step_kinds` labels
    ``program[t]`` ``run``.
    """
    return [
        step + 1
        for step, name in enumerate(program)
        if name in RUNS_CODE and not has_mark(states[step])
    ]


@dataclass(slots=True)
class _Tally:
    executions: int = 0
    fuel: int = 0
    expansions: int = 0
    deepest: int = 0  # ops in the longest program any expansion produced


def _prepare(task: Task, guard: str) -> tuple[Product, Product, list[list[int]]]:
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and has_mark(outputs):
        raise ValueError("the close guard needs mark-free targets")
    return inputs, outputs, encode_targets(outputs)


def _run_segment(
    roots: Sequence[Node],
    outputs: Product,
    targets: list[list[int]],
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
) -> tuple[str | None, tuple[str, ...] | None, Closed]:
    """One segment: a flat beam from every root, until its paths close.

    Each root has its own beam of ``beam_width`` and its own visited set, and
    each layer of it is ``_beam_solve``'s: every op on every node, successors
    ranked by cumulative log-probability, one policy call per root per layer.
    The beams advance together, a layer of every root in the order given before
    the next layer of any, so the first fit found is at the shallowest layer
    and a budget is spread over the roots; with no fit and no budget the closed
    states are those of running the roots one after another. A successor made
    by a top-level combinator is closed: it is recorded whatever its rank, as a
    hit would be, and takes no beam slot; the slots go to open successors. The
    guard's ``remaining`` is ``horizon`` less the successor's program length,
    the steps left in the whole search. Returns ``(stop, program, closed)``:
    ``stop`` is ``fit`` or ``budget`` when the search must end here, else None;
    ``closed`` merges equal closed states across roots, keeping the best score
    and its program.
    """
    closed: Closed = {}
    beams: list[list[Node]] = [[root] for root in roots]
    seens = [{root[0]} for root in roots]
    limits = (horizon, guard, max_state_atoms)
    for layer in range(segment_depth):
        for owner, beam in enumerate(beams):
            if not beam:
                continue
            opens = []  # (node index, op index, product, program)
            closes = []
            legal = np.zeros((len(beam), len(OPS)), dtype=bool)
            for node, (product, program, _) in enumerate(beam):
                tally.expansions += 1
                runs = not has_mark(product)
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
                    if runs and index in RUN_INDICES:
                        closes.append((node, index, moved, extended))
                    else:
                        opens.append((node, index, moved, extended))
            beams[owner] = []
            if not opens and not closes:
                continue
            blocks = [
                item_array(
                    [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
                )
                for product, _, _ in beam
            ]
            log_probs = scorer.log_probs(blocks, legal)
            for score, product, program in _ranked(closes, beam, log_probs):
                if not _kept(product, program, *limits):
                    continue
                best = closed.get(product)
                if best is None or score > best[0]:
                    closed[product] = (score, program)
            if layer == segment_depth - 1:
                continue  # an open path the segment cannot expand is not kept
            seen = seens[owner]
            for score, product, program in _ranked(opens, beam, log_probs):
                if len(beams[owner]) == beam_width:
                    break
                if not _kept(product, program, *limits) or product in seen:
                    continue
                seen.add(product)
                beams[owner].append((product, program, score))
    return None, None, closed


def _ranked(successors, beam: Sequence[Node], log_probs) -> list:
    """``(score, product, program)`` best first, scored as ``_beam_solve`` scores."""
    return sorted(
        (
            (beam[node][2] + log_probs[node, index], product, program)
            for node, index, product, program in successors
        ),
        key=lambda scored: scored[0],
        reverse=True,
    )


def _kept(
    product: Product,
    program: tuple[str, ...],
    horizon: int,
    guard: str,
    max_state_atoms: int | None,
) -> bool:
    """The guard, with ``remaining`` the steps left in the whole search, and the size cap."""
    if guard != "off" and blocked(product[0], horizon - len(program), guard):
        return False
    return max_state_atoms is None or not atoms_exceed(product, max_state_atoms)


def _next_roots(closed: Closed, expanded: set[Product]) -> list[Node]:
    """Closed states not yet expanded, best cumulative score first; ties keep merge order."""
    return sorted(
        (
            (product, program, score)
            for product, (score, program) in closed.items()
            if product not in expanded
        ),
        key=lambda node: node[2],
        reverse=True,
    )


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

    Starts from one root, the inputs, at score 0. Each segment runs a
    ``beam_width`` beam for at most ``segment_depth`` layers from every root
    (``_run_segment``); the ``roots`` best merged closed states become the next
    roots. A closed state that has already been a root is not one again, so the
    search cannot circle. A successor equal to the shown outputs ends the search
    with the whole program from the inputs. ``budget`` caps primitive
    executions over the whole search. Terminations: ``fit``, ``budget``,
    ``frontier_exhausted`` (a segment closed nothing new) and ``max_segments``.
    ``depth`` is the length of the returned program, or of the longest one any
    expansion produced. The cyclic collector is off, as in ``beam_solve``.
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
            horizon=max_segments * segment_depth,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
            stop_at_fit=True,
        )
        if stop is not None:
            return result(program, stop)
        frontier = _next_roots(closed, expanded)[:roots]
        if not frontier:
            return result(None, "frontier_exhausted")
    return result(None, "max_segments")


def _path_scores(
    states: Sequence[Product],
    program: Sequence[str],
    targets: list[list[int]],
    scorer: PolicyScorer,
) -> list[float]:
    """Cumulative log-probability of ``program``'s first t ops, t = 0..len."""
    scores = [0.0]
    if not program:
        return scores
    legal = np.zeros((len(program), len(OPS)), dtype=bool)
    for step in range(len(program)):
        for index, op in enumerate(OPS):
            legal[step, index] = step_all_quiet(states[step], op)[0] is not None
    blocks = [
        item_array([encode_item(s, t) for s, t in zip(state, targets, strict=True)])
        for state in states[: len(program)]
    ]
    rows = scorer.log_probs(blocks, legal)
    for step, name in enumerate(program):
        scores.append(scores[-1] + float(rows[step, OP_INDEX[name]]))
    return scores


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
    """Where the witness's closed states rank in the segment search: the K diagnostic.

    ``program`` (the task's witness by default) is cut at its top-level
    combinator applications. The segment search runs as ``segment_beam_solve``
    does, without stopping at a fit, and at each cut the witness's closed state
    there is forced in as a root if the ranking left it out. One row per witness
    segment, up to ``max_segments``; trailing ops after the last cut are not a
    segment. A row gives ``rank``, the 1-based position of the witness's next
    closed state among the merged closed states by cumulative log-probability
    (ties in merge order, so ``rank <= roots`` is "it would be a root"), or None
    if no root's beam produced it; ``closed``, how many merged closed states
    there were; ``ops``, the witness segment's length; and ``forced``, whether
    this segment's witness root had to be forced in.

    A forced root carries the best score the search found for its product,
    or, if the search never produced it, the forced root before it plus the
    witness's own log-probability over the segment between them. The rows stop
    early, with ``complete`` False on the last, if ``budget`` runs out.
    """
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        return _segment_witness_ranks(
            task,
            scorer,
            program,
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
    program: Sequence[str] | None,
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
    program = tuple(task.program if program is None else program)
    states = [inputs]
    for step, name in enumerate(program):
        moved, _spent, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None:
            raise ValueError(
                f"{task.behaviour}: the program refuses step {step} ({name})"
            )
        states.append(moved)
    own = _path_scores(states, program, targets, scorer)
    tally = _Tally()
    frontier: list[Node] = [(inputs, (), 0.0)]
    witness: Node = frontier[0]
    expanded: set[Product] = set()
    rows: list[dict] = []
    start = 0
    for segment, cut in enumerate(closed_cuts(states, program)[:max_segments]):
        forced = witness[0] not in {product for product, _, _ in frontier}
        if forced:
            frontier = [*frontier, witness]
        expanded.update(product for product, _, _ in frontier)
        stop, _program, closed = _run_segment(
            frontier,
            outputs,
            targets,
            scorer,
            tally,
            beam_width=beam_width,
            segment_depth=segment_depth,
            horizon=max_segments * segment_depth,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
            stop_at_fit=False,
        )
        ranked = _next_roots(closed, expanded)
        target = states[cut]
        rank = next(
            (place for place, node in enumerate(ranked, 1) if node[0] == target), None
        )
        rows.append(
            {
                "segment": segment,
                "ops": cut - start,
                "roots": len(frontier),
                "forced": forced,
                "closed": len(ranked),
                "rank": rank,
                "score": float(ranked[rank - 1][2]) if rank is not None else None,
                "best": float(ranked[0][2]) if ranked else None,
                "executions": tally.executions,
                "complete": stop is None,
            }
        )
        if stop is not None:
            break
        if rank is not None:
            witness = ranked[rank - 1]
        else:
            witness = (
                target,
                (*witness[1], *program[start:cut]),
                witness[2] + own[cut] - own[start],
            )
        frontier = ranked[:roots]
        start = cut
    return rows
