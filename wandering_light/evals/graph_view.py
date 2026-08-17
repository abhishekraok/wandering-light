"""Render a ``TrajectoryGraph`` expansion as Graphviz DOT.

Streamlit draws DOT in the browser, so this needs no local Graphviz install and
no networkx -- the notebook's matplotlib draw is replaced rather than ported.

Kept free of Streamlit so layout decisions (which nodes survive the cap, how a
path is highlighted) can be unit-tested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wandering_light.proposer_pilot.graph import ExpansionResult, TrajectoryGraph

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.function_def import FunctionDef, FunctionDefSet
    from wandering_light.typed_list import TypedList

# Depth 0 first; deeper layers get progressively cooler fills.
DEPTH_FILLS = ("#ffd8a8", "#ffe8cc", "#d0ebff", "#c5f6fa", "#d3f9d8", "#e5dbff")
HIGHLIGHT_EDGE = "#e03131"
DEFAULT_MAX_NODES = 60


@dataclass(frozen=True)
class GraphExpansion:
    """One breadth-first expansion, with the graph it was run on."""

    graph: TrajectoryGraph
    root_id: int
    expansion: ExpansionResult

    @property
    def depths(self) -> dict[int, int]:
        return dict(self.expansion.node_depths)


def expand_from(
    input_value: TypedList,
    functions: FunctionDefSet,
    *,
    max_depth: int = 2,
    max_states: int | None = 200,
    max_transitions: int | None = 20_000,
) -> GraphExpansion:
    """Expand every type-compatible function from one root, under budgets.

    The budgets matter: the full basis at depth three reaches hundreds of
    thousands of states, which is a corpus-generation run, not a UI action.
    """
    graph = TrajectoryGraph(functions=functions)
    root_id = graph.add_root(input_value)
    expansion = graph.expand(
        root_id,
        max_depth=max_depth,
        max_states=max_states,
        max_transitions=max_transitions,
    )
    return GraphExpansion(graph=graph, root_id=root_id, expansion=expansion)


def path_edges(
    view: GraphExpansion, destination_id: int
) -> list[tuple[int, FunctionDef, int]]:
    """Shortest ``(parent, function, child)`` edges from the root, or ``[]``.

    Walks the graph rather than the expansion's evidence so it also answers for
    nodes reached by a later ``apply``.
    """
    graph = view.graph
    if destination_id == view.root_id:
        return []
    parent_of: dict[int, tuple[int, FunctionDef]] = {}
    visited = {view.root_id}
    queue: deque[int] = deque([view.root_id])
    while queue:
        current = queue.popleft()
        if current == destination_id:
            break
        for function, child in graph.node(current).out_edges:
            if child in visited:
                continue
            visited.add(child)
            parent_of[child] = (current, function)
            queue.append(child)
    if destination_id not in parent_of:
        return []
    edges: list[tuple[int, FunctionDef, int]] = []
    current = destination_id
    while current != view.root_id:
        parent, function = parent_of[current]
        edges.append((parent, function, current))
        current = parent
    edges.reverse()
    return edges


def node_label(value: TypedList, *, max_chars: int = 34) -> str:
    """A one-line state label, truncated to keep boxes readable."""
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def visible_nodes(
    view: GraphExpansion,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    keep: Sequence[int] = (),
) -> list[int]:
    """Shallowest ``max_nodes`` nodes, with ``keep`` always retained.

    A capped drawing is still honest about the shape near the root, which is
    what a reader can actually take in; ``keep`` is how a highlighted path
    survives the cap.
    """
    depths = view.depths
    ordered = sorted(depths, key=lambda node_id: (depths[node_id], node_id))
    selected = list(ordered[:max_nodes])
    chosen = set(selected)
    known = {node.id for node in view.graph.nodes()}
    for node_id in keep:
        if node_id not in chosen and node_id in known:
            selected.append(node_id)
            chosen.add(node_id)
    return selected


def to_dot(
    view: GraphExpansion,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    highlight: Sequence[tuple[int, FunctionDef, int]] = (),
    label_chars: int = 34,
) -> str:
    """Graphviz source for the expansion, capped at ``max_nodes`` states."""
    highlight_nodes = {view.root_id}
    for parent, _function, child in highlight:
        highlight_nodes.update((parent, child))
    shown = visible_nodes(view, max_nodes=max_nodes, keep=sorted(highlight_nodes))
    shown_set = set(shown)
    highlighted_edges = {
        (parent, function.name, child) for parent, function, child in highlight
    }
    depths = view.depths

    lines = [
        "digraph trajectory {",
        "  rankdir=LR;",
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", color="#adb5bd", '
        'fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#868e96"];',
    ]
    for node_id in shown:
        node = view.graph.node(node_id)
        depth = depths.get(node_id)
        fill = (
            DEPTH_FILLS[min(depth, len(DEPTH_FILLS) - 1)]
            if depth is not None
            else "#f1f3f5"
        )
        prefix = f"#{node_id}" if depth is None else f"#{node_id} · d{depth}"
        # Escape each part first: ``\n`` is a DOT line break, not content, so it
        # must survive the backslash escaping applied to the state's own text.
        label = (
            f"{_escape(prefix)}\\n"
            f"{_escape(node_label(node.typed_list, max_chars=label_chars))}"
        )
        penwidth = 2 if node_id in highlight_nodes and highlight else 1
        lines.append(
            f'  {node_id} [label="{label}", fillcolor="{fill}", penwidth={penwidth}];'
        )
    for node_id in shown:
        for function, child in view.graph.node(node_id).out_edges:
            if child not in shown_set:
                continue
            is_highlighted = (node_id, function.name, child) in highlighted_edges
            attributes = f'label="{_escape(function.name)}"'
            if is_highlighted:
                attributes += f', color="{HIGHLIGHT_EDGE}", penwidth=2, fontcolor="{HIGHLIGHT_EDGE}"'
            lines.append(f"  {node_id} -> {child} [{attributes}];")
    lines.append("}")
    return "\n".join(lines)


def expansion_stats(view: GraphExpansion) -> dict[str, object]:
    """Headline numbers for one expansion."""
    result = view.expansion
    return {
        "nodes": view.graph.num_nodes(),
        "edges": view.graph.num_edges(),
        "reached_states": result.num_reached_states,
        "certified_depth": result.certified_depth,
        "attempted_transitions": result.attempted_transitions,
        "failed_transitions": result.failed_transitions,
        "skipped_self_loops": result.skipped_self_loops,
        "complete": result.complete,
        "stop_reason": result.stop_reason,
    }
