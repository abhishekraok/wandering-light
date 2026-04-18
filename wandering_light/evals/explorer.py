import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.trajectory import Trajectory
from wandering_light.typed_list import TypedList

DEFAULT_EVAL_FILE = "wandering_light/evals/data/random_inputs_500.py"
WIDGET_PREFIXES = ("edge_sel_", "add_sel_")


@st.cache_resource(show_spinner=False)
def load_eval(eval_file: str):
    return load_eval_data_as_trajectories(eval_file)


def _type_str(item_type: type) -> str:
    return f"{item_type.__module__}.{item_type.__qualname__}"


def _tree() -> TrajectoryTree:
    return st.session_state.tree


def _purge_widget_state(node_ids: list[int] | None = None) -> None:
    if node_ids is None:
        for key in list(st.session_state.keys()):
            if key.startswith(WIDGET_PREFIXES):
                del st.session_state[key]
        return
    for nid in node_ids:
        for prefix in WIDGET_PREFIXES:
            st.session_state.pop(f"{prefix}{nid}", None)


def _init_from_trajectory(traj: Trajectory, executor: Executor) -> None:
    _purge_widget_state()
    st.session_state.tree = TrajectoryTree.from_trajectory(traj, executor)


def _render_typed_list(tl: TypedList) -> None:
    st.markdown(f"**TL&lt;{tl.item_type.__name__}&gt;** · {len(tl)} items")
    preview_items = tl.items if len(tl) <= 10 else [*tl.items[:10], "…"]
    st.code(repr(preview_items), language="python")


def _render_edit_edge(
    node_id: int,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    tree = _tree()
    node = tree.nodes[node_id]
    parent = tree.nodes[node["parent"]]
    parent_tl: TypedList | None = parent["typed_list"]
    current_fn: FunctionDef = node["applied_fn_def"]

    if parent_tl is None:
        compatible = [current_fn]
    else:
        parent_type = _type_str(parent_tl.item_type)
        compatible = [f for f in available_functions if f.input_type == parent_type]
        if current_fn.name not in {f.name for f in compatible}:
            compatible = [current_fn, *compatible]

    fn_names = [f.name for f in compatible]
    try:
        default_idx = fn_names.index(current_fn.name)
    except ValueError:
        default_idx = 0

    col_icon, col_sel, col_btn = st.columns([0.6, 5, 1])
    with col_icon:
        st.markdown("⤷")
    with col_sel:
        selected = st.selectbox(
            "Edge function",
            fn_names,
            index=default_idx,
            key=f"edge_sel_{node_id}",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Apply", key=f"edge_btn_{node_id}"):
            new_fn = next(f for f in compatible if f.name == selected)
            deleted = tree.replace_edge(node_id, new_fn, executor)
            _purge_widget_state(deleted)
            st.rerun()


def _render_add_step(
    node_id: int,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    tree = _tree()
    node = tree.nodes[node_id]
    tl: TypedList = node["typed_list"]
    tl_type = _type_str(tl.item_type)
    compatible = [f for f in available_functions if f.input_type == tl_type]

    if not compatible:
        st.caption(f"No compatible functions for type `{tl_type}`")
        return

    fn_names = [f.name for f in compatible]
    col_sel, col_btn = st.columns([5, 1])
    with col_sel:
        selected = st.selectbox(
            "Add step",
            fn_names,
            key=f"add_sel_{node_id}",
            label_visibility="collapsed",
            placeholder="Add step…",
        )
    with col_btn:
        if st.button("Add", key=f"add_btn_{node_id}"):
            fn = next(f for f in compatible if f.name == selected)
            tree.append_child(node_id, fn, executor)
            st.rerun()


def _render_node(
    node_id: int,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    tree = _tree()
    node = tree.nodes[node_id]

    with st.container(border=True):
        if node["applied_fn_def"] is not None:
            _render_edit_edge(node_id, available_functions, executor)

        if node["error"]:
            st.error(node["error"])
        elif node["typed_list"] is not None:
            _render_typed_list(node["typed_list"])
            if not node["children"]:
                _render_add_step(node_id, available_functions, executor)

    for child_id in node["children"]:
        _render_node(child_id, available_functions, executor)


def main() -> None:
    st.set_page_config(page_title="Trajectory Explorer", page_icon="🌳")
    st.title("🌳 Trajectory Explorer")
    st.caption(
        "Full trajectory is shown by default. Change any edge's function and hit "
        "Apply to recompute downstream."
    )

    eval_file = st.sidebar.text_input("Eval file", DEFAULT_EVAL_FILE)
    with st.spinner(f"Loading {eval_file}…"):
        trajectories, available_functions = load_eval(eval_file)
    st.sidebar.success(
        f"{len(trajectories)} inputs · {len(available_functions)} functions"
    )

    options = [f"#{i}: {t.input!r}" for i, t in enumerate(trajectories)]
    selected_idx = st.sidebar.selectbox(
        "Sample",
        range(len(options)),
        format_func=lambda i: options[i],
    )
    executor = Executor(available_functions)
    selected_traj: Trajectory = trajectories[selected_idx]
    sample_key = (selected_idx, repr(selected_traj.input))

    if st.session_state.get("sample_key") != sample_key:
        _init_from_trajectory(selected_traj, executor)
        st.session_state.sample_key = sample_key

    if st.sidebar.button("🔄 Reset to original trajectory"):
        _init_from_trajectory(selected_traj, executor)
        st.rerun()

    _render_node(ROOT_ID, available_functions, executor)


if __name__ == "__main__":
    main()
