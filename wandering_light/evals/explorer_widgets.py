"""Streamlit widgets shared by the explorer's pages.

The editable trajectory tree is the explorer's one real interaction, and every
page wants it: an eval sample, a solver run, a corpus task, a hand-typed input.
It lives here so the pages hold layout only.
"""

import json

import streamlit as st

from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.evals.run_evaluation import (
    is_packaged_legacy_eval_file,
    load_eval_data_as_trajectories,
)
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.typed_list import TypedList


@st.cache_resource(show_spinner=False)
def load_eval(eval_file: str):
    return load_eval_data_as_trajectories(
        eval_file,
        trusted_legacy_python=is_packaged_legacy_eval_file(eval_file),
    )


@st.cache_resource(show_spinner=False)
def load_json_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def type_str(item_type: type) -> str:
    return f"{item_type.__module__}.{item_type.__qualname__}"


def purge_widget_state(ns: str, node_ids: list[int] | None = None) -> None:
    prefixes = (f"{ns}_edge_sel_", f"{ns}_add_sel_")
    if node_ids is None:
        for key in list(st.session_state.keys()):
            if key.startswith(prefixes):
                del st.session_state[key]
        return
    for nid in node_ids:
        for prefix in prefixes:
            st.session_state.pop(f"{prefix}{nid}", None)


def render_typed_list(tl: TypedList) -> None:
    st.markdown(f"**TL&lt;{tl.item_type.__name__}&gt;** · {len(tl)} items")
    preview_items = tl.items if len(tl) <= 10 else [*tl.items[:10], "…"]
    st.code(repr(preview_items), language="python")


def _render_edit_edge(
    tree: TrajectoryTree,
    node_id: int,
    ns: str,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    node = tree.nodes[node_id]
    parent = tree.nodes[node["parent"]]
    parent_tl: TypedList | None = parent["typed_list"]
    current_fn: FunctionDef = node["applied_fn_def"]

    if parent_tl is None:
        compatible = [current_fn]
    else:
        parent_type = type_str(parent_tl.item_type)
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
            key=f"{ns}_edge_sel_{node_id}",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Apply", key=f"{ns}_edge_btn_{node_id}"):
            new_fn = next(f for f in compatible if f.name == selected)
            deleted = tree.replace_edge(node_id, new_fn, executor)
            purge_widget_state(ns, deleted)
            st.rerun()


def _render_add_step(
    tree: TrajectoryTree,
    node_id: int,
    ns: str,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    node = tree.nodes[node_id]
    tl: TypedList = node["typed_list"]
    tl_type = type_str(tl.item_type)
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
            key=f"{ns}_add_sel_{node_id}",
            label_visibility="collapsed",
            placeholder="Add step…",
        )
    with col_btn:
        if st.button("Add", key=f"{ns}_add_btn_{node_id}"):
            fn = next(f for f in compatible if f.name == selected)
            tree.append_child(node_id, fn, executor)
            st.rerun()


def render_node(
    tree: TrajectoryTree,
    node_id: int,
    ns: str,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    node = tree.nodes[node_id]

    with st.container(border=True):
        if node["applied_fn_def"] is not None:
            _render_edit_edge(tree, node_id, ns, available_functions, executor)

        if node["error"]:
            st.error(node["error"])
        elif node["typed_list"] is not None:
            render_typed_list(node["typed_list"])
            if not node["children"]:
                _render_add_step(tree, node_id, ns, available_functions, executor)

    for child_id in node["children"]:
        render_node(tree, child_id, ns, available_functions, executor)


def resolve_fn_names(
    names: list[str], available: FunctionDefSet
) -> tuple[FunctionDefList, list[str]]:
    resolved = FunctionDefList()
    missing: list[str] = []
    for name in names:
        fn = available.name_to_function.get(name)
        if fn is None:
            missing.append(name)
        else:
            resolved.append(fn)
    return resolved, missing


def build_tree_from_names(
    input_tl: TypedList,
    names: list[str],
    available: FunctionDefSet,
    executor: Executor,
) -> tuple[TrajectoryTree | None, list[str]]:
    resolved, missing = resolve_fn_names(names, available)
    if missing:
        return None, missing
    tree = TrajectoryTree.with_root(input_tl)
    parent_id = ROOT_ID
    for fn in resolved:
        parent_id = tree.append_child(parent_id, fn, executor)
    return tree, missing


def tree_function_names(tree: TrajectoryTree) -> list[str]:
    """Function names along the tree's first branch, root outwards."""
    names: list[str] = []
    node_id = ROOT_ID
    while True:
        children = tree.nodes[node_id]["children"]
        if not children:
            return names
        node_id = children[0]
        names.append(tree.nodes[node_id]["applied_fn_def"].name)


def tree_leaf_value(tree: TrajectoryTree) -> TypedList | None:
    """The value at the end of the tree's first branch, if it executed."""
    node_id = ROOT_ID
    while tree.nodes[node_id]["children"]:
        node_id = tree.nodes[node_id]["children"][0]
    return tree.nodes[node_id]["typed_list"]
