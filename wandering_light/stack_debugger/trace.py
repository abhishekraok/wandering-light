"""Record one ``beam_solve`` run as a navigable trace.

``TraceRecorder`` is a ``BeamObserver``: it is told what the search decided
and stores it. The search never reads it back, so a recorded run and an
unrecorded one are the same run -- ``tests/test_stack_debugger_trace.py``
asserts that.

What is kept per layer is what the page needs to step without a round-trip:
every beam node with its program, its k example stacks and the policy's row
over the 28 ops; every op tried on every node with the machine's verdict; and
what the ranking then did with each successor. Successor stacks are recorded
for the first example only and truncated, because 64 nodes x 28 ops x 48
layers x k examples is tens of megabytes and the first example already shows
what the op did; the full product is one request away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wandering_light.stack_beam import BeamObserver
from wandering_light.stack_machine import Config, Sym, format_stack
from wandering_light.stack_pbe import OPS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.stack_beam import Node, Product

SCHEMA_VERSION = 1
SUCCESSOR_DISPLAY_LIMIT = 140
STATE_DISPLAY_LIMIT = 600
FLOAT_PLACES = 4
OP_NAMES: tuple[str, ...] = tuple(op.name for op in OPS)


def _round(value: float) -> float | None:
    if value is None or math.isinf(value) or math.isnan(value):
        return None
    return round(float(value), FLOAT_PLACES)


def stack_text(stack: tuple, limit: int = STATE_DISPLAY_LIMIT) -> str:
    """The machine's own printing, top of stack on the right, bounded."""
    text = format_stack(Config(stack))
    return text if len(text) <= limit else text[:limit] + " …"


def product_text(product: Product) -> list[str]:
    return [stack_text(stack) for stack in product]


def _captured(before: tuple, after: tuple, name: str) -> bool:
    """Whether the op was materialised as a symbol instead of run.

    Derived from the stacks rather than from the op's private flag: under an
    open ``[`` the machine appends ``Sym(name)`` and nothing else.
    """
    return len(after) == len(before) + 1 and after[-1] == Sym(name)


@dataclass
class ExpansionRecord:
    outcome: str  # refused | captured | executed
    reason: str | None
    successor: str | None
    verdict: str | None  # retained | duplicate | guard:close | guard:type | cut | fit
    fuel: int
    stepped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "successor": self.successor,
            "verdict": self.verdict,
            "fuel": self.fuel,
            "stepped": self.stepped,
        }


@dataclass
class LayerRecord:
    depth: int
    nodes: list[dict[str, Any]] = field(default_factory=list)
    expansions: list[list[ExpansionRecord | None]] = field(default_factory=list)
    scores: list[list[float | None]] | None = None
    retained: list[dict[str, Any]] = field(default_factory=list)
    retained_products: list[Product] = field(default_factory=list)
    scored: bool = False
    ranked: bool = False
    # The policy's pooled row per node, kept in memory for the map; not shipped.
    features: Any = None

    def to_dict(self) -> dict[str, Any]:
        attempted = sum(1 for row in self.expansions for e in row if e is not None)
        executed = sum(
            1
            for row in self.expansions
            for e in row
            if e is not None and e.outcome != "refused"
        )
        verdicts: dict[str, int] = {}
        for row in self.expansions:
            for e in row:
                if e is not None and e.verdict is not None:
                    verdicts[e.verdict] = verdicts.get(e.verdict, 0) + 1
        return {
            "depth": self.depth,
            "nodes": self.nodes,
            "expansions": [
                [None if e is None else e.to_dict() for e in row]
                for row in self.expansions
            ],
            "scores": self.scores,
            "scored": self.scored,
            "ranked": self.ranked,
            "retained": self.retained,
            "counts": {
                "attempted": attempted,
                "executed": executed,
                "verdicts": verdicts,
            },
        }


class TraceRecorder(BeamObserver):
    """Capture a complete, replayable record of one beam search."""

    def __init__(self, scorer: Any = None) -> None:
        # With a scorer, each scored layer also keeps the pooled features the
        # scorer computed for it, so the session map needs no second pass.
        self.scorer = scorer
        self.layers: list[LayerRecord] = []
        self.termination: str | None = None
        self.terminal_depth = 0
        self.executions = 0
        self.fuel = 0
        self.expansions = 0
        self._beam: Sequence[Node] = ()
        self._last: tuple[int, int, int] | None = None

    def _layer(self, depth: int) -> LayerRecord:
        if not self.layers or self.layers[-1].depth != depth:
            self.layers.append(LayerRecord(depth=depth))
        return self.layers[-1]

    def layer_started(self, depth: int, beam: Sequence[Node]) -> None:
        layer = self._layer(depth)
        self._beam = beam
        layer.nodes = [
            {
                "program": list(program),
                "log_probability": _round(log_probability),
                "states": product_text(product),
            }
            for product, program, log_probability in beam
        ]
        layer.expansions = [[None] * len(OPS) for _ in beam]

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
        layer = self._layer(depth)
        before = self._beam[node][0]
        if successor is None:
            outcome = "refused"
        elif _captured(before[0], successor[0], OP_NAMES[op_index]):
            outcome = "captured"
        else:
            outcome = "executed"
        layer.expansions[node][op_index] = ExpansionRecord(
            outcome=outcome,
            reason=reason,
            successor=None
            if successor is None
            else stack_text(successor[0], SUCCESSOR_DISPLAY_LIMIT),
            verdict=None,
            fuel=fuel,
            stepped=stepped,
        )
        self._last = (depth, node, op_index)

    def layer_scored(self, depth: int, log_probs, legal) -> None:
        layer = self._layer(depth)
        layer.scored = True
        layer.scores = [[_round(value) for value in row] for row in log_probs]
        features = getattr(self.scorer, "last_features", None)
        if features is not None and len(features) == len(layer.nodes):
            layer.features = features.copy()

    def layer_retained(
        self,
        depth: int,
        beam: Sequence[Node],
        verdicts: Mapping[tuple[int, int], str],
        parents: Sequence[tuple[int, int]],
    ) -> None:
        layer = self._layer(depth)
        layer.ranked = True
        for (node, op_index), verdict in verdicts.items():
            record = layer.expansions[node][op_index]
            if record is not None:
                record.verdict = verdict
        # A successor the ranking never reached was cut by width: the beam
        # filled before its turn. Distinct from "reached and dropped".
        for row in layer.expansions:
            for record in row:
                if (
                    record is not None
                    and record.outcome != "refused"
                    and record.verdict is None
                ):
                    record.verdict = "cut"
        layer.retained = [
            {
                "rank": rank,
                "program": list(program),
                "log_probability": _round(log_probability),
                "states": product_text(product),
                "parent": list(parents[rank]),
            }
            for rank, (product, program, log_probability) in enumerate(beam)
        ]
        layer.retained_products = [product for product, _, _ in beam]

    def search_terminated(
        self,
        termination: str,
        depth: int,
        executions: int,
        fuel: int,
        expansions: int,
    ) -> None:
        self.termination = termination
        self.terminal_depth = depth
        self.executions = executions
        self.fuel = fuel
        self.expansions = expansions
        if termination == "fit" and self._last is not None:
            layer_depth, node, op_index = self._last
            record = self._layer(layer_depth).expansions[node][op_index]
            if record is not None:
                record.verdict = "fit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ops": list(OP_NAMES),
            "termination": self.termination,
            "terminal_depth": self.terminal_depth,
            "executions": self.executions,
            "fuel": self.fuel,
            "expansions": self.expansions,
            "layers": [layer.to_dict() for layer in self.layers],
        }
