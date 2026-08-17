"""Build bounded, deterministic Plotly views of trajectory experiments.

The deep corpus stores sampled tasks, not the tens of millions of states visited
while certifying them.  Replaying stored witnesses into ``TrajectoryGraph``
therefore gives an honest and tractable projection: convergent paths, parallel
edges, and cycles are visible without pretending to reconstruct the exhaustive
generation graph. Live workspace paths and explicitly capped local expansions
cover the useful structural experiments from ``notebooks/proposer_pilot.ipynb``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wandering_light.basis_dataset import typed_list_from_builtin_str
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from plotly.graph_objects import Figure

    from wandering_light.evals.corpus_index import RecordDetail
    from wandering_light.evals.explorer_tree import TrajectoryTree
    from wandering_light.typed_list import TypedList


@dataclass(frozen=True, slots=True)
class GraphNodeView:
    node_id: int
    x: float
    y: float
    label: str
    hover: str
    role: str


@dataclass(frozen=True, slots=True)
class GraphEdgeView:
    source_id: int
    target_id: int
    function_names: tuple[str, ...]
    highlighted: bool


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    self_loop_groups: int
    parallel_function_groups: int
    convergent_nodes: int
    directed_cycle_groups: int


@dataclass(frozen=True, slots=True)
class GraphView:
    nodes: tuple[GraphNodeView, ...]
    edges: tuple[GraphEdgeView, ...]
    total_nodes: int
    total_edges: int
    truncated: bool
    diagnostics: GraphDiagnostics


@dataclass(frozen=True, slots=True)
class WitnessProjection:
    view: GraphView
    processed_records: int
    skipped_records: int
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpansionTaskView:
    node_id: int
    distance: int
    output: str
    function_names: tuple[str, ...]
    certified: bool


@dataclass(frozen=True, slots=True)
class LocalExpansionProjection:
    view: GraphView
    tasks: tuple[ExpansionTaskView, ...]
    attempted_transitions: int
    failed_transitions: int
    skipped_self_loops: int
    certified_depth: int
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class WorkspaceProjection:
    view: GraphView
    processed_edges: int
    skipped_edges: int
    errors: tuple[str, ...]


MAX_TYPED_LIST_JSON_BYTES = 65_536
MAX_TYPED_LIST_ITEMS = 1_000
MAX_NESTED_VALUES = 20_000
MAX_EXPANSION_UNITS = 20_000
MAX_VALUE_WIDTH = 10_000
MAX_VALUE_DEPTH = 16


def validate_typed_list_workload(value, serialized: str, *, label: str) -> None:
    """Reject compact values that registered functions could expand explosively."""
    if len(serialized.encode("utf-8")) > MAX_TYPED_LIST_JSON_BYTES:
        raise ValueError(f"{label} JSON exceeds {MAX_TYPED_LIST_JSON_BYTES:,} bytes")
    if len(value) > MAX_TYPED_LIST_ITEMS:
        raise ValueError(f"{label} is limited to {MAX_TYPED_LIST_ITEMS:,} outer items")

    stack = [(item, 0) for item in value.items]
    visited = 0
    expansion_units = 0
    while stack:
        item, depth = stack.pop()
        visited += 1
        expansion_units += 1
        if visited > MAX_NESTED_VALUES:
            raise ValueError(f"{label} exceeds {MAX_NESTED_VALUES:,} nested values")
        if depth > MAX_VALUE_DEPTH:
            raise ValueError(f"{label} nesting exceeds depth {MAX_VALUE_DEPTH}")
        if isinstance(item, range):
            try:
                width = len(item)
            except OverflowError as error:
                raise ValueError(f"{label} contains an oversized range") from error
            if width > MAX_VALUE_WIDTH:
                raise ValueError(
                    f"{label} range expands to {width:,} items; limit is "
                    f"{MAX_VALUE_WIDTH:,}"
                )
            expansion_units += width
        elif isinstance(item, str | bytes | bytearray):
            if len(item) > MAX_VALUE_WIDTH:
                raise ValueError(
                    f"{label} contains a value longer than {MAX_VALUE_WIDTH:,} items"
                )
            expansion_units += len(item)
        elif isinstance(item, dict):
            stack.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, int) and item.bit_length() > 4_096:
            raise ValueError(f"{label} contains an integer wider than 4,096 bits")
        if expansion_units > MAX_EXPANSION_UNITS:
            raise ValueError(
                f"{label} can expand beyond {MAX_EXPANSION_UNITS:,} total items"
            )


def resolve_witness_functions(
    record: RecordDetail, available_functions: FunctionDefSet
) -> FunctionDefList:
    """Resolve a witness by stable ID when available, validating its name."""
    by_id = {
        function.metadata.get("basis_function_id"): function
        for function in available_functions
        if function.metadata.get("basis_function_id")
    }
    resolved: list[FunctionDef] = []
    if record.witness_function_ids:
        if len(record.witness_function_ids) != len(record.witness_function_names):
            raise ValueError("witness function IDs and names have different lengths")
        for function_id, expected_name in zip(
            record.witness_function_ids,
            record.witness_function_names,
            strict=True,
        ):
            function = by_id.get(function_id)
            if function is None:
                raise ValueError(f"basis does not contain witness ID {function_id!r}")
            if function.name != expected_name:
                raise ValueError(
                    f"witness ID {function_id!r} resolves to {function.name!r}, "
                    f"not {expected_name!r}"
                )
            resolved.append(function)
        return FunctionDefList(resolved)

    for name in record.witness_function_names:
        function = available_functions.name_to_function.get(name)
        if function is None:
            raise ValueError(f"basis does not contain witness function {name!r}")
        resolved.append(function)
    return FunctionDefList(resolved)


def build_witness_projection(
    records: Sequence[RecordDetail],
    available_functions: FunctionDefSet,
    *,
    selected_task_id: str | None = None,
    max_records: int = 250,
    max_nodes: int = 600,
    max_edges: int = 1_200,
) -> WitnessProjection:
    """Replay stored witnesses into a graph and return a capped primitive view."""
    if max_records <= 0 or max_nodes <= 0 or max_edges <= 0:
        raise ValueError("graph caps must be positive")

    # Lazy import keeps the corpus browser light. On current main the package
    # initializer reaches the solver/transformer stack; PR #37 makes it lazy.
    from wandering_light.executor import Executor
    from wandering_light.proposer_pilot.graph import TrajectoryGraph

    graph = TrajectoryGraph(functions=available_functions)
    executor = Executor(available_functions)
    selected_nodes: set[int] = set()
    selected_edges: set[tuple[int, int, str]] = set()
    target_nodes: set[int] = set()
    errors: list[str] = []
    processed = 0

    ordered = sorted(
        records,
        key=lambda record: (
            record.task_id != selected_task_id,
            record.distance if record.distance is not None else -1,
            record.task_id,
        ),
    )
    for record in ordered[:max_records]:
        try:
            input_value = typed_list_from_builtin_str(record.input)
            validate_typed_list_workload(
                input_value, record.input, label=f"task {record.task_id} input"
            )
            functions = resolve_witness_functions(record, available_functions)
            execution = executor.execute_trajectory(
                TrajectorySpec(input_value, functions)
            )
            if not execution.success or execution.trajectory is None:
                raise ValueError(
                    execution.error_msg or "stored witness execution failed"
                )
            if record.output is not None:
                expected = typed_list_from_builtin_str(record.output)
                if execution.trajectory.output != expected:
                    raise ValueError("stored witness does not reproduce the target")

            # Only mutate the shared projection after a complete validation;
            # one corrupt row must not leave partial nodes or edges behind.
            root_id = graph.add_root(input_value)
            parent_id = root_id
            path_nodes = [root_id]
            path_edges: list[tuple[int, int, str]] = []
            for function in functions:
                child_id = graph.apply(parent_id, function)
                path_edges.append((parent_id, child_id, function.name))
                path_nodes.append(child_id)
                parent_id = child_id

            target_nodes.add(parent_id)
            if record.task_id == selected_task_id:
                selected_nodes.update(path_nodes)
                selected_edges.update(path_edges)
            processed += 1
        except Exception as error:
            errors.append(f"{record.task_id}: {error}")

    view = graph_to_view(
        graph,
        selected_nodes=selected_nodes,
        selected_edges=selected_edges,
        target_nodes=target_nodes,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    skipped = max(0, len(records) - processed)
    return WitnessProjection(
        view=view,
        processed_records=processed,
        skipped_records=skipped,
        errors=tuple(errors),
    )


def build_workspace_projection(
    trees: Sequence[TrajectoryTree],
    available_functions: FunctionDefSet,
    *,
    target: TypedList | None = None,
    max_nodes: int = 600,
    max_edges: int = 1_200,
) -> WorkspaceProjection:
    """Merge the active editor and solver trees into one bounded graph view."""
    if not trees:
        raise ValueError("at least one workspace tree is required")
    if max_nodes <= 0 or max_edges <= 0:
        raise ValueError("graph caps must be positive")

    from wandering_light.proposer_pilot.graph import TrajectoryGraph

    graph = TrajectoryGraph(functions=available_functions)
    selected_nodes: set[int] = set()
    selected_edges: set[tuple[int, int, str]] = set()
    target_nodes: set[int] = set()
    errors: list[str] = []
    processed_edges = 0
    skipped_edges = 0

    for tree_index, tree in enumerate(trees):
        root = tree.nodes.get(0)
        if root is None or root.get("typed_list") is None:
            errors.append(f"workspace tree {tree_index + 1}: missing root value")
            continue
        root_value = root["typed_list"]
        try:
            validate_typed_list_workload(
                root_value,
                root_value.to_string(),
                label=f"workspace tree {tree_index + 1} input",
            )
            graph_root = graph.add_root(root_value)
            if target is not None and root_value == target:
                target_nodes.add(graph_root)
        except Exception as error:
            errors.append(f"workspace tree {tree_index + 1}: {error}")
            continue

        graph_ids = {0: graph_root}
        selected_nodes.add(graph_root)
        pending = deque([0])
        while pending:
            parent_tree_id = pending.popleft()
            parent_graph_id = graph_ids[parent_tree_id]
            parent = tree.nodes[parent_tree_id]
            for child_tree_id in parent.get("children", ()):
                child = tree.nodes.get(child_tree_id)
                function = None if child is None else child.get("applied_fn_def")
                if child is None or function is None or child.get("typed_list") is None:
                    skipped_edges += 1
                    if child is not None and child.get("error"):
                        errors.append(
                            f"workspace tree {tree_index + 1}, node {child_tree_id}: "
                            f"{child['error']}"
                        )
                    continue
                try:
                    child_graph_id = graph.apply(parent_graph_id, function)
                except Exception as error:
                    skipped_edges += 1
                    errors.append(
                        f"workspace tree {tree_index + 1}, node {child_tree_id}: "
                        f"{error}"
                    )
                    continue
                graph_ids[child_tree_id] = child_graph_id
                selected_nodes.add(child_graph_id)
                selected_edges.add(
                    (parent_graph_id, child_graph_id, function.name)
                )
                if target is not None and graph.node(child_graph_id).typed_list == target:
                    target_nodes.add(child_graph_id)
                processed_edges += 1
                pending.append(child_tree_id)

    return WorkspaceProjection(
        view=graph_to_view(
            graph,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            target_nodes=target_nodes,
            max_nodes=max_nodes,
            max_edges=max_edges,
        ),
        processed_edges=processed_edges,
        skipped_edges=skipped_edges,
        errors=tuple(errors),
    )


def build_local_expansion(
    input_value: TypedList,
    available_functions: FunctionDefSet,
    *,
    max_depth: int = 2,
    max_states: int = 250,
    max_transitions: int = 2_500,
    skip_self_loops: bool = False,
    max_nodes: int = 600,
    max_edges: int = 1_200,
) -> LocalExpansionProjection:
    """Port the notebook's graph-mining loop with explicit resource bounds."""
    if not available_functions:
        raise ValueError("local expansion requires at least one function")
    if not 0 <= max_depth <= 3:
        raise ValueError("local expansion depth must be between 0 and 3")
    if not 1 <= max_states <= 1_000:
        raise ValueError("local expansion state cap must be between 1 and 1,000")
    if not 1 <= max_transitions <= 5_000:
        raise ValueError(
            "local expansion transition cap must be between 1 and 5,000"
        )
    if max_nodes <= 0 or max_edges <= 0:
        raise ValueError("graph caps must be positive")

    validate_typed_list_workload(
        input_value, input_value.to_string(), label="local expansion input"
    )

    from wandering_light.proposer_pilot.graph import TrajectoryGraph

    graph = TrajectoryGraph(functions=available_functions)
    root_id = graph.add_root(input_value)
    expansion = graph.expand(
        root_id,
        max_depth=max_depth,
        max_states=max_states,
        max_transitions=max_transitions,
        skip_self_loops=skip_self_loops,
    )
    tasks = tuple(
        sorted(
            (
                ExpansionTaskView(
                    node_id=task.dst_id,
                    distance=task.num_steps,
                    output=repr(task.trajectory.output),
                    function_names=tuple(
                        function.name for function in task.trajectory.function_defs
                    ),
                    certified=task.shortest_path_is_certified,
                )
                for task in graph.tasks_from_expansion(
                    expansion,
                    max_steps=max_depth,
                    require_certified=False,
                )
            ),
            key=lambda task: (task.distance, task.node_id),
        )
    )
    return LocalExpansionProjection(
        view=graph_to_view(
            graph,
            max_nodes=max_nodes,
            max_edges=max_edges,
        ),
        tasks=tasks,
        attempted_transitions=expansion.attempted_transitions,
        failed_transitions=expansion.failed_transitions,
        skipped_self_loops=expansion.skipped_self_loops,
        certified_depth=expansion.certified_depth,
        stop_reason=expansion.stop_reason,
    )


def graph_to_view(
    graph: Any,
    *,
    selected_nodes: Iterable[int] = (),
    selected_edges: Iterable[tuple[int, int, str]] = (),
    target_nodes: Iterable[int] = (),
    max_nodes: int = 600,
    max_edges: int = 1_200,
) -> GraphView:
    """Convert ``TrajectoryGraph`` to deterministic layered primitives."""
    selected_node_set = set(selected_nodes)
    selected_edge_set = set(selected_edges)
    target_node_set = set(target_nodes)
    roots = set(graph.roots)
    all_nodes = list(graph.nodes())
    total_nodes = len(all_nodes)
    total_edges = graph.num_edges()

    priority: list[int] = []
    for tier in (
        roots,
        selected_node_set & target_node_set,
        target_node_set,
        selected_node_set,
    ):
        priority.extend(sorted(node_id for node_id in tier if node_id not in priority))
    priority_set = set(priority)
    remaining = sorted(node.id for node in all_nodes if node.id not in priority_set)
    included_ids = set((priority + remaining)[:max_nodes])

    grouped_edges: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    highlighted_pairs: set[tuple[int, int]] = set()
    for node in all_nodes:
        if node.id not in included_ids:
            continue
        for function, child_id in node.out_edges:
            if child_id not in included_ids:
                continue
            grouped_edges[(node.id, child_id)].append(function.name)
            if (node.id, child_id, function.name) in selected_edge_set:
                highlighted_pairs.add((node.id, child_id))

    ordered_pairs = sorted(
        grouped_edges,
        key=lambda pair: (pair not in highlighted_pairs, pair[0], pair[1]),
    )
    included_pairs = ordered_pairs[:max_edges]
    edges = tuple(
        GraphEdgeView(
            source_id=source,
            target_id=target,
            function_names=tuple(sorted(set(grouped_edges[(source, target)]))),
            highlighted=(source, target) in highlighted_pairs,
        )
        for source, target in included_pairs
    )

    depths = _minimum_depths(roots, edges)
    incoming_sources: defaultdict[int, set[int]] = defaultdict(set)
    for edge in edges:
        incoming_sources[edge.target_id].add(edge.source_id)
    diagnostics = GraphDiagnostics(
        self_loop_groups=sum(edge.source_id == edge.target_id for edge in edges),
        parallel_function_groups=sum(
            len(edge.function_names) > 1 for edge in edges
        ),
        convergent_nodes=sum(
            len(sources) > 1 for sources in incoming_sources.values()
        ),
        directed_cycle_groups=_cycle_group_count(edges),
    )
    layers: defaultdict[int, list[int]] = defaultdict(list)
    for node_id in included_ids:
        layers[depths.get(node_id, max(depths.values(), default=-1) + 1)].append(
            node_id
        )
    coordinates: dict[int, tuple[float, float]] = {}
    for depth, node_ids in sorted(layers.items()):
        ordered_ids = sorted(node_ids)
        center = (len(ordered_ids) - 1) / 2
        for index, node_id in enumerate(ordered_ids):
            coordinates[node_id] = (float(depth), center - index)

    nodes_by_id = {node.id: node for node in all_nodes}
    nodes: list[GraphNodeView] = []
    for node_id in sorted(included_ids, key=lambda value: (*coordinates[value], value)):
        node = nodes_by_id[node_id]
        if node_id in selected_node_set and node_id in target_node_set:
            role = "selected_target"
        elif node_id in roots:
            role = "root"
        elif node_id in selected_node_set:
            role = "selected_path"
        elif node_id in target_node_set:
            role = "target"
        else:
            role = "state"
        value = repr(node.typed_list)
        label = value if len(value) <= 42 else value[:41] + "…"
        x, y = coordinates[node_id]
        nodes.append(
            GraphNodeView(
                node_id=node_id,
                x=x,
                y=y,
                label=label,
                hover=f"node {node_id}<br>{_html(value)}",
                role=role,
            )
        )

    return GraphView(
        nodes=tuple(nodes),
        edges=edges,
        total_nodes=total_nodes,
        total_edges=total_edges,
        truncated=(total_nodes > len(nodes) or len(ordered_pairs) > len(edges)),
        diagnostics=diagnostics,
    )


def graph_view_figure(view: GraphView) -> Figure:
    """Render graph primitives as an interactive Plotly figure."""
    import plotly.graph_objects as go

    nodes_by_id = {node.node_id: node for node in view.nodes}
    edge_pairs = {(edge.source_id, edge.target_id) for edge in view.edges}

    def geometry(edge: GraphEdgeView) -> tuple[float, float, float, float]:
        source = nodes_by_id[edge.source_id]
        target = nodes_by_id[edge.target_id]
        source_x, source_y = source.x, source.y
        target_x, target_y = target.x, target.y
        if (edge.target_id, edge.source_id) in edge_pairs:
            delta_x = target_x - source_x
            delta_y = target_y - source_y
            length = max((delta_x**2 + delta_y**2) ** 0.5, 1.0)
            offset_x = -delta_y / length * 0.08
            offset_y = delta_x / length * 0.08
            source_x += offset_x
            source_y += offset_y
            target_x += offset_x
            target_y += offset_y
        return source_x, source_y, target_x, target_y

    figure = go.Figure()
    loop_shapes: list[dict[str, Any]] = []
    for highlighted, color, width in (
        (False, "#94a3b8", 1.2),
        (True, "#f97316", 3.2),
    ):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for edge in view.edges:
            if edge.highlighted != highlighted:
                continue
            if edge.source_id == edge.target_id:
                node = nodes_by_id[edge.source_id]
                loop_shapes.append(
                    {
                        "type": "circle",
                        "x0": node.x - 0.16,
                        "x1": node.x + 0.16,
                        "y0": node.y + 0.08,
                        "y1": node.y + 0.4,
                        "line": {"color": color, "width": width},
                        "layer": "below",
                    }
                )
                continue
            source_x, source_y, target_x, target_y = geometry(edge)
            x_values.extend((source_x, target_x, None))
            y_values.extend((source_y, target_y, None))
        if x_values:
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"color": color, "width": width},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_hover: list[str] = []
    arrow_annotations: list[dict[str, Any]] = []
    for edge in view.edges:
        if edge.source_id == edge.target_id:
            node = nodes_by_id[edge.source_id]
            source_x, source_y = node.x - 0.14, node.y + 0.13
            target_x, target_y = node.x - 0.16, node.y + 0.24
            edge_x.append(node.x)
            edge_y.append(node.y + 0.28)
        else:
            source_x, source_y, target_x, target_y = geometry(edge)
            edge_x.append((source_x + target_x) / 2)
            edge_y.append((source_y + target_y) / 2)
        edge_hover.append(
            f"{edge.source_id} → {edge.target_id}<br>"
            + "<br>".join(_html(name) for name in edge.function_names)
        )
        # Plotly line traces have no direction. Shortened data-coordinate
        # annotations add arrowheads without covering the node markers.
        arrow_annotations.append(
            {
                "x": source_x + (target_x - source_x) * 0.82,
                "y": source_y + (target_y - source_y) * 0.82,
                "ax": source_x + (target_x - source_x) * 0.18,
                "ay": source_y + (target_y - source_y) * 0.18,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "text": "",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1,
                "arrowwidth": 3 if edge.highlighted else 1.2,
                "arrowcolor": "#f97316" if edge.highlighted else "#64748b",
            }
        )
    if edge_x:
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="markers",
                marker={"size": 9, "color": "rgba(0,0,0,0)"},
                hovertext=edge_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    colors = {
        "root": "#2563eb",
        "selected_target": "#dc2626",
        "selected_path": "#f97316",
        "target": "#16a34a",
        "state": "#cbd5e1",
    }
    figure.add_trace(
        go.Scatter(
            x=[node.x for node in view.nodes],
            y=[node.y for node in view.nodes],
            mode="markers+text" if len(view.nodes) <= 40 else "markers",
            marker={
                "size": [18 if node.role != "state" else 12 for node in view.nodes],
                "color": [colors[node.role] for node in view.nodes],
                "line": {"color": "#0f172a", "width": 1},
            },
            text=[node.label for node in view.nodes],
            textposition="top center",
            textfont={"size": 10, "color": "#334155"},
            hovertext=[node.hover for node in view.nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=max(440, min(900, 260 + len(view.nodes) * 3)),
        margin={"l": 20, "r": 20, "t": 20, "b": 40},
        plot_bgcolor="white",
        xaxis={
            "title": "minimum graph depth",
            "showgrid": True,
            "zeroline": False,
            "dtick": 1,
        },
        yaxis={"visible": False},
        hovermode="closest",
        annotations=arrow_annotations,
        shapes=loop_shapes,
    )
    return figure


def execute_witness(record: RecordDetail, available_functions: FunctionDefSet):
    """Safely decode and execute one registry-resolved stored witness."""
    from wandering_light.executor import Executor

    input_value = typed_list_from_builtin_str(record.input)
    functions = resolve_witness_functions(record, available_functions)
    return Executor(available_functions).execute_trajectory(
        TrajectorySpec(input_value, functions)
    )


def _minimum_depths(roots: set[int], edges: Sequence[GraphEdgeView]) -> dict[int, int]:
    outgoing: defaultdict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source_id].append(edge.target_id)
    depths = dict.fromkeys(roots, 0)
    queue: deque[int] = deque(sorted(roots))
    while queue:
        current = queue.popleft()
        for child in sorted(outgoing[current]):
            depth = depths[current] + 1
            if child not in depths or depth < depths[child]:
                depths[child] = depth
                queue.append(child)
    return depths


def _cycle_group_count(edges: Sequence[GraphEdgeView]) -> int:
    """Count non-trivial strongly connected components without networkx."""
    outgoing: defaultdict[int, list[int]] = defaultdict(list)
    incoming: defaultdict[int, list[int]] = defaultdict(list)
    node_ids: set[int] = set()
    for edge in edges:
        if edge.source_id == edge.target_id:
            continue
        outgoing[edge.source_id].append(edge.target_id)
        incoming[edge.target_id].append(edge.source_id)
        node_ids.update((edge.source_id, edge.target_id))

    visited: set[int] = set()
    finish_order: list[int] = []
    ordered_outgoing = {
        node_id: tuple(sorted(outgoing[node_id])) for node_id in node_ids
    }
    for start in sorted(node_ids):
        if start in visited:
            continue
        visited.add(start)
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            node_id, child_index = stack[-1]
            children = ordered_outgoing[node_id]
            if child_index >= len(children):
                stack.pop()
                finish_order.append(node_id)
                continue
            child_id = children[child_index]
            stack[-1] = (node_id, child_index + 1)
            if child_id not in visited:
                visited.add(child_id)
                stack.append((child_id, 0))

    assigned: set[int] = set()
    cycle_groups = 0
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: set[int] = set()
        assigned.add(start)
        stack = [start]
        while stack:
            node_id = stack.pop()
            component.add(node_id)
            for parent_id in incoming[node_id]:
                if parent_id not in assigned:
                    assigned.add(parent_id)
                    stack.append(parent_id)
        cycle_groups += int(len(component) > 1)
    return cycle_groups


def _html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
