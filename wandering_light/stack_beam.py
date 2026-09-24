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

``budget`` caps primitive executions and ``deadline``, a ``time.perf_counter()``
value, caps wall-clock; both are checked before every op attempt, and a search
stopped by one ends ``budget`` or ``wall_clock_cap``. The flat beam's
``budget_unit`` can count ``fuel`` (the outer op's 1 plus every quotation-body
step) or ``executions+fuel`` against the budget instead (``BUDGET_UNITS``). Left None, neither is
read and the search is the uncapped one (``tests/test_short_segment_search.py``
pins that).

``segment_beam_solve`` is the segment-reset beam of short-segment-campaign
(#168). A **closed state** is what a top-level combinator application leaves: an
op that runs a quotation (``RUNS_CODE``) applied to a product with no mark on any
stack, the ``run`` step of `experiments/staged_transfer.step_kinds` and the cut
of segment-composite (#160). From each root a flat beam runs until its paths
close; equal closed states are merged across roots, and the best ``roots`` of
them by cumulative log-probability start the next segment, each with a fresh
beam. ``segment_witness_ranks`` is the diagnostic that sizes ``roots``.

Every search builds a node's block with ``block_encoder``: the scorer's own
``encoder``, so a ``difference`` policy (#168) reads the layout it trained on.
A scorer without one, such as a test stub that only has ``log_probs``, reads
the ``machine`` layout every published checkpoint uses.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from wandering_light.quotation_guard import RUNS_CODE, blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Config,
    Lst,
    Mark,
    format_stack,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS
from wandering_light.stack_policy import MACHINE, node_encoder

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from wandering_light.stack_machine import Op
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

BUDGET_UNITS = ("executions", "fuel", "executions+fuel")
# How the segment beam ranks closed states at a cut: by cumulative
# log-probability from the inputs, or by what each gained in the segment just
# finished, its root restarting at 0 (short-segment-campaign #168, exploratory).
SCORINGS = ("cumulative", "local")

Product = tuple[tuple, ...]  # one stack per shown example
Node = tuple[Product, tuple[str, ...], float]  # (product, program, log-probability)


def block_encoder(
    scorer: PolicyScorer, outputs: Product
) -> Callable[[Product], np.ndarray]:
    """``product -> block`` against ``outputs`` in the scorer's layout.

    A `PolicyScorer` (and any subclass) supplies ``encoder``; a duck-typed
    scorer without it gets ``machine`` blocks, what it was written against.
    The targets are encoded once, here, not once per node.
    """
    encoder = getattr(scorer, "encoder", None)
    return encoder(outputs) if encoder is not None else node_encoder(outputs, MACHINE)


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
    # fit | max_depth | frontier_exhausted | budget | wall_clock_cap; max_segments
    # for the segment beam
    termination: str
    seconds: float  # to the hit, or to termination
    executions: int
    fuel: int
    expansions: int
    depth: int
    # Did the width ever turn away a successor the guards, the size cap and the
    # visited set would have kept? Conservatively True whenever a layer filled
    # with ranked successors still unread. A flat search that never pruned is
    # the search every wider beam would run (short-segment-campaign #168).
    pruned: bool = False


def spent_in(unit: str, executions: int, fuel: int) -> int:
    """What a search has spent, counted in one of ``BUDGET_UNITS``."""
    if unit == "executions":
        return executions
    if unit == "fuel":
        return fuel
    if unit == "executions+fuel":
        return executions + fuel
    raise ValueError(f"unknown budget unit {unit!r}; choose one of {BUDGET_UNITS}")


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


def replay_capped(
    program: Sequence[str],
    inputs: Product,
    outputs: Product,
    *,
    budget: int | None = None,
    budget_unit: str = "executions",
    deadline: float | None = None,
) -> tuple[bool, int, int, str | None]:
    """``replay_program`` with the beams' caps, checked before every op.

    Returns ``(fits, executions, fuel, stop)``: ``stop`` is ``budget`` or
    ``wall_clock_cap`` when a cap ended the replay first, else None. ``budget``
    is what this replay may spend, in ``budget_unit``. With neither cap it is
    ``replay_program``.
    """
    executions = 0
    fuel = 0
    product = inputs
    for name in program:
        if budget is not None and spent_in(budget_unit, executions, fuel) >= budget:
            return False, executions, fuel, "budget"
        if deadline is not None and time.perf_counter() >= deadline:
            return False, executions, fuel, "wall_clock_cap"
        moved, spent, stepped = step_all_quiet(product, OP_BY_NAME[name])
        executions += stepped
        fuel += spent
        if moved is None:
            return False, executions, fuel, None
        product = moved
    return product == outputs, executions, fuel, None


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
    deadline: float | None = None,
    budget_unit: str = "executions",
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
            deadline=deadline,
            budget_unit=budget_unit,
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
    beam: list[Node] = [(inputs, (), 0.0)]
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
        ranked = sorted(
            successors,
            key=lambda s: beam[s[0]][2] + log_probs[s[0], s[1]],
            reverse=True,
        )
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


def widening_beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    budget: int,
    start_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
    max_state_atoms: int | None = None,
    deadline: float | None = None,
    budget_unit: str = "executions",
) -> tuple[BeamResult, dict]:
    """``beam_solve`` at ``start_width``, then twice as wide, from scratch, until one ends it.

    The matched control of short-segment-campaign (#168). Every restart draws
    on one execution counter capped at ``budget``, and the widening stops at
    the first shown fit, when the budget (counted in ``budget_unit``) or the
    ``deadline`` runs out, or when a
    restart ran to its own end without the width ever turning a successor away
    (``BeamResult.pruned`` False), so every wider beam is the same search
    (``saturated``). Equal spend at two widths proves nothing: a pruned beam
    can fail at the same cost as a narrower one and a wider one still fit. The
    budget is checked before every op attempt, so the last restart may overshoot
    it by less than one attempt; the overshoot is charged. Returns the combined
    result (termination of the last restart; ``depth`` the deepest restart's)
    and a record of the restarts: ``widths`` (width, termination, executions,
    each), ``largest_width_completed`` (the widest restart that ran to its own
    end, a fit included), ``fit_width`` and ``saturated``.
    """
    started = time.perf_counter()
    executions = fuel = expansions = depth = 0
    runs: list[dict] = []
    width = start_width
    saturated = False
    while True:
        run = beam_solve(
            task,
            scorer,
            beam_width=width,
            max_depth=max_depth,
            guard=guard,
            budget=max(0, budget - spent_in(budget_unit, executions, fuel)),
            max_state_atoms=max_state_atoms,
            deadline=deadline,
            budget_unit=budget_unit,
        )
        executions += run.executions
        fuel += run.fuel
        expansions += run.expansions
        depth = max(depth, run.depth)
        runs.append(
            {
                "width": width,
                "termination": run.termination,
                "executions": run.executions,
                "fuel": run.fuel,
                "pruned": run.pruned,
            }
        )
        if run.termination in ("fit", "budget", "wall_clock_cap"):
            break
        if not run.pruned:
            saturated = True
            break
        width *= 2
    completed = [
        r["width"] for r in runs if r["termination"] not in ("budget", "wall_clock_cap")
    ]
    result = BeamResult(
        run.program,
        run.termination,
        time.perf_counter() - started,
        executions,
        fuel,
        expansions,
        run.depth if run.program is not None else depth,
    )
    return result, {
        "widths": runs,
        "largest_width_completed": max(completed, default=None),
        "fit_width": width if run.termination == "fit" else None,
        "saturated": saturated,
        # Why the widening ended: fit, budget, wall_clock_cap or saturated.
        "stop": "saturated" if saturated else run.termination,
    }


# ---------------------------------------------------------------------------
# Segment-reset search (short-segment-campaign, #168)
# ---------------------------------------------------------------------------

#: Ops whose successor is closed when the product before them has no mark open.
RUN_INDICES = frozenset(index for index, op in enumerate(OPS) if op.name in RUNS_CODE)
OP_INDEX: dict[str, int] = {op.name: index for index, op in enumerate(OPS)}

# Merged closed states: product -> (best cumulative log-probability, its program).
Closed = dict[Product, tuple[float, tuple[str, ...]]]


class ArchivedSegment(NamedTuple):
    """One segment of a segment search, as ``segment_beam_solve`` archives it.

    ``roots`` are the segment's roots ``(product, program, score)`` in the
    order it ran them, the score the one the search gave the root (0 under
    local scoring); ``closed`` is the segment's merged closed states; ``stop``
    is ``_run_segment``'s.
    """

    segment: int
    roots: list[Node]
    closed: Closed
    stop: str | None


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


def _prepare(task: Task, guard: str) -> tuple[Product, Product]:
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    if guard in ("close", "both") and has_mark(outputs):
        raise ValueError("the close guard needs mark-free targets")
    return inputs, outputs


def _run_segment(
    roots: Sequence[Node],
    outputs: Product,
    encode: Callable[[Product], np.ndarray],
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
    deadline: float | None = None,
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
    hit would be, and takes no beam slot; the slots go to open successors. For an
    open successor the guard's ``remaining`` is the steps left in this segment,
    since a path that cannot close before the segment ends is dropped; for a
    closed one it is ``horizon`` less the program length. ``encode`` is
    ``block_encoder(scorer, outputs)``. Returns ``(stop, program, closed)``:
    ``stop`` is ``fit``, ``budget`` or ``wall_clock_cap`` when the search must
    end here, else None;
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
                    if deadline is not None and time.perf_counter() >= deadline:
                        return "wall_clock_cap", None, closed
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
            blocks = [encode(product) for product, _, _ in beam]
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
                # An open path must close within this segment or it is dropped
                # when the segment ends, so the guard counts the segment's steps.
                left = segment_depth - layer - 1
                if not _open_kept(product, left, guard, max_state_atoms):
                    continue
                if product in seen:
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


def _open_kept(
    product: Product, remaining: int, guard: str, max_state_atoms: int | None
) -> bool:
    """The guard with ``remaining`` the steps left in the segment, and the size cap."""
    if guard != "off" and blocked(product[0], remaining, guard):
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
    deadline: float | None = None,
    scoring: str = "cumulative",
    archive: list | None = None,
) -> BeamResult:
    """Search one task by segments that reset at closed states.

    Starts from one root, the inputs, at score 0. Each segment runs a
    ``beam_width`` beam for at most ``segment_depth`` layers from every root
    (``_run_segment``); the ``roots`` best merged closed states become the next
    roots. A closed state that has already been a root is not one again, so the
    search cannot circle. A successor equal to the shown outputs ends the search
    with the whole program from the inputs. ``budget`` caps primitive
    executions over the whole search and ``deadline`` its wall-clock, as in
    ``beam_solve``. ``scoring`` ``local`` starts every root at score 0, so a
    closed state's score is what it gained in its segment: the next roots are
    ranked by that gain and equal closed states merge keeping the best gain.
    Within a root's beam the ranking is the same either way, up to the root's
    constant. Terminations: ``fit``, ``budget``, ``wall_clock_cap``,
    ``frontier_exhausted`` (a segment closed nothing new) and ``max_segments``.
    ``depth`` is the length of the returned program, or of the longest one any
    expansion produced. The cyclic collector is off, as in ``beam_solve``.
    ``archive``, when given, gets one ``ArchivedSegment`` per segment run: its
    roots as ranked and every closed state it merged. Appending holds
    references only, so the search and its counters are the same with or
    without it; holding them defers freeing each segment's closed states from
    inside the clock to after it.
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
            deadline=deadline,
            scoring=scoring,
            archive=archive,
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
    deadline: float | None = None,
    scoring: str = "cumulative",
    archive: list | None = None,
) -> BeamResult:
    if scoring not in SCORINGS:
        raise ValueError(f"unknown scoring {scoring!r}; choose one of {SCORINGS}")
    started = time.perf_counter()
    inputs, outputs = _prepare(task, guard)
    encode = block_encoder(scorer, outputs)
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
    for segment in range(max_segments):
        expanded.update(product for product, _, _ in frontier)
        stop, program, closed = _run_segment(
            frontier,
            outputs,
            encode,
            scorer,
            tally,
            beam_width=beam_width,
            segment_depth=segment_depth,
            horizon=max_segments * segment_depth,
            guard=guard,
            budget=budget,
            max_state_atoms=max_state_atoms,
            stop_at_fit=True,
            deadline=deadline,
        )
        if archive is not None:  # references only: no copy inside the clock
            archive.append(ArchivedSegment(segment, frontier, closed, stop))
        if stop is not None:
            return result(program, stop)
        frontier = _next_roots(closed, expanded)[:roots]
        if scoring == "local":
            frontier = [(product, program, 0.0) for product, program, _ in frontier]
        if not frontier:
            return result(None, "frontier_exhausted")
    return result(None, "max_segments")


def _path_scores(
    states: Sequence[Product],
    program: Sequence[str],
    encode: Callable[[Product], np.ndarray],
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
    blocks = [encode(state) for state in states[: len(program)]]
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
    scoring: str = "cumulative",
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

    ``scoring`` ``local`` is ``segment_beam_solve``'s: every root, the forced
    witness root included, starts its segment at 0, so ``rank`` and ``score``
    are by the gain in the segment, and the roots were chosen by that gain too.
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
            scoring=scoring,
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
    scoring: str = "cumulative",
) -> list[dict]:
    if scoring not in SCORINGS:
        raise ValueError(f"unknown scoring {scoring!r}; choose one of {SCORINGS}")
    local = scoring == "local"
    inputs, outputs = _prepare(task, guard)
    encode = block_encoder(scorer, outputs)
    program = tuple(task.program if program is None else program)
    states = [inputs]
    for step, name in enumerate(program):
        moved, _spent, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None:
            raise ValueError(
                f"{task.behaviour}: the program refuses step {step} ({name})"
            )
        states.append(moved)
    # Replaying the witness and scoring its own path are bookkeeping for the
    # diagnostic, outside the search: they are not charged to `budget` or
    # counted in `executions`.
    own = _path_scores(states, program, encode, scorer)
    tally = _Tally()
    frontier: list[Node] = [(inputs, (), 0.0)]
    witness: Node = frontier[0]
    expanded: set[Product] = set()
    produced: set[Product] = set()  # closed in an earlier segment
    rows: list[dict] = []
    start = 0
    for segment, cut in enumerate(closed_cuts(states, program)[:max_segments]):
        forced = witness[0] not in {product for product, _, _ in frontier}
        if forced:
            # Under local scoring the forced root restarts at 0 like the others.
            frontier = [
                *frontier,
                (witness[0], witness[1], 0.0 if local else witness[2]),
            ]
        expanded.update(product for product, _, _ in frontier)
        stop, _program, closed = _run_segment(
            frontier,
            outputs,
            encode,
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
        # The rank is among every merged closed state, expanded or not: it says
        # whether the witness's state was produced and how it scored. Whether it
        # could be a next root is `eligible_rank`, among the unexpanded ones.
        every = _next_roots(closed, set())
        ranked = _next_roots(closed, expanded)
        target = states[cut]
        rank = next(
            (place for place, node in enumerate(every, 1) if node[0] == target), None
        )
        eligible = next(
            (place for place, node in enumerate(ranked, 1) if node[0] == target), None
        )
        rows.append(
            {
                "segment": segment,
                "ops": cut - start,
                "roots": len(frontier),
                "forced": forced,
                "closed": len(every),
                "rank": rank,
                "eligible_rank": eligible,
                "score": float(every[rank - 1][2]) if rank is not None else None,
                "best": float(every[0][2]) if every else None,
                "executions": tally.executions,
                "complete": stop is None,
                **_rank_variants(
                    every, frontier, produced | expanded, target, local_scoring=local
                ),
            }
        )
        produced.update(closed)
        if stop is not None:
            break
        if rank is not None:
            witness = every[rank - 1]
        else:
            # The witness root's score at the segment's start: 0 under local
            # scoring, where every root restarts, else what it carried.
            base = 0.0 if local else witness[2]
            witness = (
                target,
                (*witness[1], *program[start:cut]),
                base + own[cut] - own[start],
            )
        frontier = ranked[:roots]
        if local:
            frontier = [(product, path, 0.0) for product, path, _ in frontier]
        start = cut
    return rows


def _stack_tokens(product: Product) -> list[str]:
    """The first shown example's stack as tokens, for an edit distance."""
    return format_stack(Config(product[0])).split() if product else []


def _edit_distance(left: Sequence[str], right: Sequence[str], cap: int = 200) -> int:
    left, right = list(left[:cap]), list(right[:cap])
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def _rank_variants(
    every: Sequence[Node],
    roots: Sequence[Node],
    stale: set[Product],
    target: Product,
    *,
    local_scoring: bool = False,
) -> dict:
    """The K diagnostic's analysis columns beside the registered cumulative rank.

    ``local_rank`` ranks by the score gained in this segment (cumulative less
    the root's), the ranking a score reset at each cut would use; ``per_op_rank``
    by mean log-probability per op of the whole program under cumulative
    scoring, and under local scoring (``local_scoring``), where scores are
    segment gains, by the gain per op of the segment, so numerator and
    denominator cover the same span; ``fresh_rank`` by
    cumulative score among closed states never closed or expanded before (no
    revisits, no no-op segments). ``winner`` and ``local_winner`` describe the
    top state under the cumulative and local rankings: its segment length and
    the token edit distance of its first stack from its root's.
    """
    by_program = {program: (product, score) for product, program, score in roots}

    def root_of(program: tuple[str, ...]) -> tuple[str, ...]:
        for size in range(len(program), -1, -1):
            if program[:size] in by_program:
                return program[:size]
        raise ValueError("a closed state with no root")

    def local(node: Node) -> float:
        return float(node[2] - by_program[root_of(node[1])][1])

    def place(order: Sequence[Node]) -> int | None:
        return next((n for n, node in enumerate(order, 1) if node[0] == target), None)

    def describe(node: Node | None) -> dict | None:
        if node is None:
            return None
        root = root_of(node[1])
        return {
            "segment_ops": len(node[1]) - len(root),
            "edit_from_root": _edit_distance(
                _stack_tokens(by_program[root][0]), _stack_tokens(node[0])
            ),
            "local": local(node),
            "program": " ".join(node[1][len(root) :]),
        }

    locals_ = sorted(every, key=local, reverse=True)

    def per_op_score(node: Node) -> float:
        if local_scoring:
            ops = len(node[1]) - len(root_of(node[1]))
            return float(node[2]) / max(1, ops)
        return node[2] / max(1, len(node[1]))

    per_op = sorted(every, key=per_op_score, reverse=True)
    fresh = [node for node in every if node[0] not in stale]
    mine = next((node for node in every if node[0] == target), None)
    return {
        "local_rank": place(locals_),
        "per_op_rank": place(per_op),
        "fresh_rank": place(fresh),
        "fresh": len(fresh),
        "witness_local": local(mine) if mine is not None else None,
        "witness_segment": describe(mine),
        "winner": describe(every[0] if every else None),
        "local_winner": describe(locals_[0] if locals_ else None),
    }
