"""Synthetic lineage foundation; not the bounded #180 artifact recorder/runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from experiments.score_gap_profiles import Snapshot
from wandering_light.stack_beam import BeamObserver, key_of
from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, step_raw
from wandering_light.stack_pbe import OP_BY_NAME, OPS

if TYPE_CHECKING:
    from wandering_light.stack_beam import BeamResult, Node, Product


@dataclass(frozen=True)
class Retained:
    depth: int
    beam: tuple[Node, ...]
    parents: tuple[tuple[int, int], ...]


class SyntheticCapture(BeamObserver):
    """Exact in-memory observer for synthetic tests only; no attempts are saved."""

    def __init__(self):
        self.starts = []
        self.scored = {}
        self.retained = []
        self.terminal = None

    def layer_started(self, depth, beam):
        self.starts.append((depth, tuple(beam)))

    def layer_scored(self, depth, log_probs, legal):
        if depth in self.scored:
            raise ValueError("duplicate scored callback")
        self.scored[depth] = (
            np.array(log_probs, copy=True),
            np.array(legal, copy=True),
        )

    def layer_retained(self, depth, beam, verdicts, parents):
        self.retained.append(Retained(depth, tuple(beam), tuple(parents)))

    def search_terminated(self, termination, depth, executions, fuel, expansions):
        if self.terminal is not None:
            raise ValueError("duplicate terminal callback")
        self.terminal = (termination, depth, executions, fuel, expansions)


def replay_states(
    program: tuple[str, ...], initial: Product, *, check=None
) -> tuple[Product, ...]:
    """Full typed replay under the production per-call fuel/depth allocation."""
    states = [initial]
    for name in program:
        if check is not None:
            check()
        moved = []
        for stack in states[-1]:
            if check is not None:
                check()
            outcome = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                raise ValueError(f"known continuation refused: {outcome}")
            moved.append(outcome[0])
        states.append(tuple(moved))
    return tuple(states)


def same_node(left: Node, right: Node) -> bool:
    a, b = np.asarray(left[2]), np.asarray(right[2])
    return (
        key_of(left[0]) == key_of(right[0])
        and left[1] == right[1]
        and a.dtype == b.dtype
        and a.tobytes() == b.tobytes()
    )


def reconstruct_lineage(
    capture: SyntheticCapture,
    initial: Product,
    outputs: Product,
    result: BeamResult,
    *,
    check=None,
) -> tuple[Snapshot, ...]:
    expected_terminal = (
        result.termination,
        result.depth,
        result.executions,
        result.fuel,
        result.expansions,
    )
    if capture.terminal != expected_terminal:
        raise ValueError("terminal callback/result mismatch")
    if [depth for depth, _ in capture.starts] != list(range(len(capture.starts))):
        raise ValueError("started callback order mismatch")
    if (
        not capture.starts
        or len(capture.starts[0][1]) != 1
        or not same_node(capture.starts[0][1][0], (initial, (), 0.0))
    ):
        raise ValueError("initial product mismatch")
    depths = [layer.depth for layer in capture.retained]
    if depths != list(capture.scored) or depths != list(range(len(depths))):
        raise ValueError("exact scored/retained callback order mismatch")
    for layer in capture.retained:
        if len(layer.parents) != len(layer.beam):
            raise ValueError("retained parent count mismatch")
        if layer.depth >= len(capture.starts):
            raise ValueError("retention without parent layer")
        if layer.depth + 1 < len(capture.starts):
            following = capture.starts[layer.depth + 1][1]
            if len(following) != len(layer.beam) or any(
                not same_node(a, b) for a, b in zip(following, layer.beam, strict=True)
            ):
                raise ValueError("next started beam differs from recorded retention")
    if result.program is None:
        if result.termination == "fit":
            raise ValueError("fit without program")
        return tuple(
            Snapshot(tuple(node[2] for node in layer.beam))
            for layer in capture.retained
        )
    if result.termination != "fit" or result.depth != len(result.program):
        raise ValueError("program/fit depth mismatch")
    states = replay_states(result.program, initial, check=check)
    if key_of(states[-1]) != key_of(outputs):
        raise ValueError("returned program misses shown targets")
    if (
        len(capture.starts) != len(result.program)
        or len(depths) != len(result.program) - 1
    ):
        raise ValueError("fit scored/started lineage length mismatch")
    # Identify the actual terminal parent by its full program AND typed state.
    terminal_beam = capture.starts[-1][1]
    candidates = [
        i
        for i, node in enumerate(terminal_beam)
        if node[1] == result.program[:-1] and key_of(node[0]) == key_of(states[-2])
    ]
    if len(candidates) != 1:
        raise ValueError("returned program has no unique actual terminal parent")
    child = candidates[0]
    ranks = {}
    for layer in reversed(capture.retained):
        if check is not None:
            check()
        if not 0 <= child < len(layer.beam):
            raise ValueError("ancestor rank outside retained beam")
        node = layer.beam[child]
        if node[1] != result.program[: layer.depth + 1] or key_of(node[0]) != key_of(
            states[layer.depth + 1]
        ):
            raise ValueError("actual retained ancestor differs from replay")
        parent, op = layer.parents[child]
        started = capture.starts[layer.depth][1]
        if not 0 <= parent < len(started) or not 0 <= op < len(OPS):
            raise ValueError("invalid recorded parent identity")
        before = started[parent]
        if (
            before[1] != result.program[: layer.depth]
            or OPS[op].name != result.program[layer.depth]
            or key_of(before[0]) != key_of(states[layer.depth])
        ):
            raise ValueError("parent ancestry/typed state mismatch")
        scores, legal = capture.scored[layer.depth]
        if (
            scores.shape != (len(started), len(OPS))
            or legal.shape != scores.shape
            or not legal[parent, op]
        ):
            raise ValueError("parent score/legal row mismatch")
        expected_score = before[2] + scores[parent, op]
        if not same_node(node, (node[0], node[1], expected_score)):
            raise ValueError("cumulative score dtype/bits mismatch")
        # Replay the known suffix from this exact product, not a string display.
        suffix = replay_states(result.program[layer.depth + 1 :], node[0], check=check)
        if key_of(suffix[-1]) != key_of(outputs):
            raise ValueError("known retained continuation misses targets")
        ranks[layer.depth] = child + 1
        child = parent
    if child != 0:
        raise ValueError("ancestry does not terminate at root")
    return tuple(
        Snapshot(tuple(node[2] for node in layer.beam), ranks[layer.depth])
        for layer in capture.retained
    )
