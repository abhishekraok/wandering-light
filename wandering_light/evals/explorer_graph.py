"""Graph page: expand a root into a TrajectoryGraph and draw it.

This is the notebook's section 4 made interactive. Expansion is budgeted and
explicit -- a full-basis sweep is a corpus-generation job, not a page render --
and the drawing is capped, so the picture stays readable near the root.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wandering_light.evals import graph_view
from wandering_light.evals.explorer_corpus import SELECTED_TASK_KEY
from wandering_light.evals.explorer_playground import (
    editable_text,
    parse_typed_list,
    select_basis,
)
from wandering_light.function_def import FunctionDefSet

VIEW_KEY = "graph_view"
DEFAULT_ROOT = "TL<int>([1, 2, 3])"
PRESET_INT_FNS = [
    "inc",
    "dec",
    "double",
    "half",
    "square",
    "neg",
    "abs",
    "sign",
    "mod2",
    "int_to_str",
]


def _root_text_default() -> str:
    task = st.session_state.get(SELECTED_TASK_KEY)
    if task is None:
        return DEFAULT_ROOT
    return editable_text(task["input"])


def _select_palette(functions: FunctionDefSet) -> FunctionDefSet:
    """Which functions may be applied during expansion.

    Left at the full basis, one layer already fans out to a hundred-odd states
    and the drawing stops being a picture. The preset keeps it legible.
    """
    all_names = [fn.name for fn in functions]
    preset = st.radio(
        "Function palette",
        ["Integer preset", "Whole basis", "Pick functions"],
        horizontal=True,
        key="graph_palette_mode",
    )
    if preset == "Whole basis":
        return functions
    if preset == "Integer preset":
        chosen = [name for name in PRESET_INT_FNS if name in set(all_names)]
    else:
        chosen = st.multiselect(
            "Functions",
            all_names,
            default=[name for name in PRESET_INT_FNS if name in set(all_names)][:6],
            key="graph_functions",
        )
    return FunctionDefSet([functions.name_to_function[name] for name in chosen])


def _render_tasks(view: graph_view.GraphExpansion) -> None:
    tasks = list(
        view.graph.tasks_from_expansion(
            view.expansion, min_steps=1, require_certified=True
        )
    )
    if not tasks:
        st.info("No certified tasks from this expansion.")
        return
    rows = [
        {
            "node": task.dst_id,
            "distance": task.verified_shortest_num_steps,
            "path": " → ".join(fn.name for fn in task.trajectory.function_defs),
            "output": repr(task.trajectory.output)[:70],
        }
        for task in tasks
    ]
    frame = pd.DataFrame(rows).sort_values(["distance", "node"])
    st.dataframe(frame, width="stretch", hide_index=True, height=240)
    st.caption(
        f"{len(tasks)} certified tasks · every node first reached at depth k is "
        "at shortest distance exactly k."
    )


def render_graph_tab() -> None:
    st.caption(
        "Breadth-first expansion from one state. Node fill is depth; the "
        "highlighted path is the shortest route to the node you select."
    )
    basis = select_basis("graph_basis")
    if basis is None:
        return
    functions, _basis_set_id = basis

    if "graph_root" not in st.session_state:
        st.session_state["graph_root"] = _root_text_default()
    cols = st.columns([3, 1])
    with cols[0]:
        root_text = st.text_input("Root state", key="graph_root")
    with cols[1]:
        st.write("")
        st.write("")
        if st.button("↩ Use corpus task input", key="graph_use_corpus"):
            st.session_state["graph_root"] = _root_text_default()
            st.rerun()

    palette = _select_palette(functions)
    if len(palette) == 0:
        st.info("Select at least one function.")
        return

    budget_cols = st.columns(4)
    with budget_cols[0]:
        depth = st.number_input("Depth", 1, 6, 2, key="graph_depth")
    with budget_cols[1]:
        max_states = st.number_input(
            "Max states", 5, 5000, 120, step=5, key="graph_states"
        )
    with budget_cols[2]:
        max_transitions = st.number_input(
            "Max transitions", 10, 500_000, 20_000, step=1000, key="graph_transitions"
        )
    with budget_cols[3]:
        max_nodes = st.number_input(
            "Nodes to draw",
            5,
            200,
            graph_view.DEFAULT_MAX_NODES,
            step=5,
            key="graph_max_nodes",
        )

    if st.button("🌐 Expand", key="graph_expand"):
        try:
            root_value = parse_typed_list(root_text)
        except Exception as error:
            st.error(f"Could not parse root: {error}")
            return
        with st.spinner("Expanding…"):
            st.session_state[VIEW_KEY] = graph_view.expand_from(
                root_value,
                palette,
                max_depth=int(depth),
                max_states=int(max_states),
                max_transitions=int(max_transitions),
            )

    view: graph_view.GraphExpansion | None = st.session_state.get(VIEW_KEY)
    if view is None:
        st.info("Press **Expand** to build the graph.")
        return

    stats = graph_view.expansion_stats(view)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Nodes", stats["nodes"])
    metric_cols[1].metric("Edges", stats["edges"])
    metric_cols[2].metric("Certified depth", stats["certified_depth"])
    metric_cols[3].metric("Transitions tried", f"{stats['attempted_transitions']:,}")
    metric_cols[4].metric("Failed", f"{stats['failed_transitions']:,}")
    if not stats["complete"]:
        st.warning(
            f"Expansion stopped early ({stats['stop_reason']}); distances beyond "
            f"depth {stats['certified_depth']} are not certified."
        )

    depths = view.depths
    targets = sorted(depths, key=lambda node_id: (depths[node_id], node_id))
    highlight_target = st.selectbox(
        "Highlight path to",
        targets,
        format_func=lambda node_id: (
            f"#{node_id} · d{depths[node_id]} · "
            f"{graph_view.node_label(view.graph.node(node_id).typed_list)}"
        ),
        key="graph_highlight",
    )
    highlight = graph_view.path_edges(view, highlight_target)
    if highlight:
        st.markdown(
            "**Shortest path:** `"
            + " → ".join(function.name for _p, function, _c in highlight)
            + "`"
        )

    st.graphviz_chart(
        graph_view.to_dot(view, max_nodes=int(max_nodes), highlight=highlight),
        width="stretch",
    )
    if stats["nodes"] > int(max_nodes):
        st.caption(
            f"Drawing the {int(max_nodes)} shallowest of {stats['nodes']} states."
        )

    st.divider()
    st.subheader("Certified tasks from this expansion")
    _render_tasks(view)
