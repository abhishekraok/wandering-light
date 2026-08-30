"""Record one rollout search as a navigable trace.

``TraceRecorder`` is a ``RolloutObserver``: it is told what the search decided
and stores it. The search never reads it back, so a recorded run and an
unrecorded one are the same run -- ``tests/test_rollout_trace.py`` asserts
that on the golden tasks.

Two things are deliberately kept out of the default trace.  Raw logit rows are
118 floats per beam node per layer and are wanted only when someone opens the
logits panel, so the recorder keeps the type-compatible entries (the ones the
mask admits, which is what normalization uses) plus the unmasked top few, and
``capture_logits`` adds the full rows on demand.  Expansion records are capped
per layer, because a budget-terminated search attempts up to its whole budget
and the point of the cap is that the trace stays small enough to hand to a
browser whole -- which is what makes stepping through it feel instant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wandering_light.solver import RolloutObserver

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.function_def import FunctionDef
    from wandering_light.solver import _RolloutNode
    from wandering_light.typed_list import TypedList


SCHEMA_VERSION = 1
DEFAULT_UNMASKED_TOP_K = 10
DEFAULT_MAX_EXPANSIONS_PER_LAYER = 4_000
VALUE_DISPLAY_LIMIT = 512
FLOAT_PLACES = 6


def _round(value: float) -> float:
    """Trim display precision. A trace is read, and it is shipped whole."""

    return round(value, FLOAT_PLACES)


def _display_value(state: TypedList) -> dict[str, Any]:
    """Render a state for display, bounding how much of a huge list travels."""

    text = repr(state.items)
    truncated = len(text) > VALUE_DISPLAY_LIMIT
    return {
        "item_type": state.item_type.__name__,
        "length": len(state.items),
        "value": text[:VALUE_DISPLAY_LIMIT],
        "truncated": truncated,
    }


@dataclass
class ScoredNode:
    """One beam node as the policy saw it at the start of a layer."""

    parent_index: int
    state: dict[str, Any]
    path: tuple[str, ...]
    log_probability: float
    compatible: list[dict[str, float]] = field(default_factory=list)
    unmasked_top: list[dict[str, float]] = field(default_factory=list)
    log_normalizer: float | None = None
    logits: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "parent_index": self.parent_index,
            "state": self.state,
            "path": list(self.path),
            "log_probability": self.log_probability,
            "compatible": self.compatible,
            "unmasked_top": self.unmasked_top,
            "log_normalizer": self.log_normalizer,
            "mask_size": len(self.compatible),
        }
        if self.logits is not None:
            payload["logits"] = self.logits
        return payload


@dataclass
class ExpansionRecord:
    """One charged execution: an attempted type-compatible transition."""

    order: int
    parent_index: int
    function: str
    log_probability: float
    outcome: str
    successor: dict[str, Any] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "parent_index": self.parent_index,
            "function": self.function,
            "log_probability": self.log_probability,
            "outcome": self.outcome,
            "successor": self.successor,
            "error": self.error,
        }


@dataclass
class LayerRecord:
    """Everything one rollout depth did, in the order it did it."""

    depth: int
    nodes: list[ScoredNode] = field(default_factory=list)
    expansions: list[ExpansionRecord] = field(default_factory=list)
    expansions_recorded: int = 0
    expansions_attempted: int = 0
    unique_successors: int = 0
    retained: list[dict[str, Any]] = field(default_factory=list)
    pruned_by_width: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "nodes": [node.to_dict() for node in self.nodes],
            "expansions": [record.to_dict() for record in self.expansions],
            "expansions_recorded": self.expansions_recorded,
            "expansions_attempted": self.expansions_attempted,
            "expansions_elided": self.expansions_attempted - self.expansions_recorded,
            "unique_successors": self.unique_successors,
            "retained": self.retained,
            "pruned_by_width": self.pruned_by_width,
        }


class TraceRecorder(RolloutObserver):
    """Capture a complete, replayable record of one rollout search."""

    def __init__(
        self,
        *,
        capture_logits: bool = False,
        unmasked_top_k: int = DEFAULT_UNMASKED_TOP_K,
        max_expansions_per_layer: int = DEFAULT_MAX_EXPANSIONS_PER_LAYER,
    ) -> None:
        self.capture_logits = capture_logits
        self.unmasked_top_k = unmasked_top_k
        self.max_expansions_per_layer = max_expansions_per_layer
        self.layers: list[LayerRecord] = []
        self.termination: str | None = None
        self.executions: int = 0
        self.terminal_depth: int = 0
        # The current layer's raw rows, held between layer_scored and the
        # node_masked calls that turn them into per-node normalized entries.
        self._scores: list[dict[str, float]] = []

    def _layer(self, depth: int) -> LayerRecord:
        if not self.layers or self.layers[-1].depth != depth:
            self.layers.append(LayerRecord(depth=depth))
        return self.layers[-1]

    def layer_scored(
        self,
        depth: int,
        nodes: Sequence[_RolloutNode],
        score_rows: Sequence[dict[str, float]],
    ) -> None:
        layer = self._layer(depth)
        for parent_index, (node, row) in enumerate(zip(nodes, score_rows, strict=True)):
            ranked = sorted(row.items(), key=lambda item: (-item[1], item[0]))
            layer.nodes.append(
                ScoredNode(
                    parent_index=parent_index,
                    state=_display_value(node.state),
                    path=tuple(function.name for function in node.function_defs),
                    log_probability=_round(node.log_probability),
                    unmasked_top=[
                        {"function": name, "logit": _round(score)}
                        for name, score in ranked[: self.unmasked_top_k]
                    ],
                    logits=dict(row) if self.capture_logits else None,
                )
            )
        self._scores = [dict(row) for row in score_rows]

    def node_masked(
        self,
        depth: int,
        parent_index: int,
        compatible: Sequence[FunctionDef],
        log_normalizer: float,
    ) -> None:
        layer = self._layer(depth)
        node = layer.nodes[parent_index]
        node.log_normalizer = None if math.isinf(log_normalizer) else log_normalizer
        row = self._scores[parent_index]
        # Only the name, the raw logit and the normalized log-probability. The
        # function's types are the same for every node that admits it, so they
        # travel once in the session's basis table; the probability is
        # exp(log_probability) and is derived where it is displayed.
        entries = [
            {
                "function": function.name,
                "logit": _round(row[function.name]),
                "log_probability": _round(row[function.name] - log_normalizer),
            }
            for function in compatible
        ]
        entries.sort(key=lambda entry: (-entry["logit"], entry["function"]))
        node.compatible = entries

    def expansion_attempted(
        self,
        depth: int,
        order: int,
        parent_index: int,
        function: FunctionDef,
        log_probability: float,
        outcome: str,
        successor: TypedList | None,
        error: BaseException | None,
    ) -> None:
        layer = self._layer(depth)
        layer.expansions_attempted += 1
        if layer.expansions_recorded >= self.max_expansions_per_layer:
            return
        layer.expansions_recorded += 1
        layer.expansions.append(
            ExpansionRecord(
                order=order,
                parent_index=parent_index,
                function=function.name,
                log_probability=_round(log_probability),
                outcome=outcome,
                successor=None if successor is None else _display_value(successor),
                error=None if error is None else f"{type(error).__name__}: {error}",
            )
        )

    def layer_retained(
        self,
        depth: int,
        beam: Sequence[_RolloutNode],
        successors: Mapping[tuple, _RolloutNode],
    ) -> None:
        layer = self._layer(depth)
        layer.unique_successors = len(successors)
        retained_orders = {node.order for node in beam}
        layer.retained = [
            {
                "rank": rank,
                "state": _display_value(node.state),
                "path": [function.name for function in node.function_defs],
                "log_probability": _round(node.log_probability),
                "order": node.order,
            }
            for rank, node in enumerate(beam)
        ]
        # A unique successor that is not in the beam was pruned by width alone;
        # recording it separates "the search never reached this" from "the
        # search reached it and dropped it", which is the distinction the
        # prune-reason panel exists to make.
        layer.pruned_by_width = sorted(
            (
                {
                    "path": [function.name for function in node.function_defs],
                    "log_probability": _round(node.log_probability),
                    "order": node.order,
                }
                for node in successors.values()
                if node.order not in retained_orders
            ),
            key=lambda entry: (-entry["log_probability"], entry["order"]),
        )

    def search_terminated(self, termination: str, executions: int, depth: int) -> None:
        self.termination = termination
        self.executions = executions
        self.terminal_depth = depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "termination": self.termination,
            "executions": self.executions,
            "terminal_depth": self.terminal_depth,
            "capture_logits": self.capture_logits,
            "layers": [layer.to_dict() for layer in self.layers],
        }
