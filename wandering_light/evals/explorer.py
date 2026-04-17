import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.typed_list import TypedList

DEFAULT_EVAL_FILE = "wandering_light/evals/data/random_inputs_500.py"


@st.cache_resource(show_spinner=False)
def load_eval(eval_file: str):
    return load_eval_data_as_trajectories(eval_file)


def _type_str(item_type: type) -> str:
    return f"{item_type.__module__}.{item_type.__qualname__}"


def _init_tree(input_tl: TypedList) -> None:
    st.session_state.tree = {
        0: {
            "typed_list": input_tl,
            "error": None,
            "parent": None,
            "applied_fn": None,
            "children": [],
        }
    }
    st.session_state.next_id = 1


def _apply_function(parent_id: int, fn: FunctionDef, executor: Executor) -> None:
    parent = st.session_state.tree[parent_id]
    new_id = st.session_state.next_id
    st.session_state.next_id += 1
    try:
        result = executor.execute(fn, parent["typed_list"])
        node = {
            "typed_list": result,
            "error": None,
            "parent": parent_id,
            "applied_fn": fn.name,
            "children": [],
        }
    except Exception as e:
        node = {
            "typed_list": None,
            "error": f"{type(e).__name__}: {e}",
            "parent": parent_id,
            "applied_fn": fn.name,
            "children": [],
        }
    st.session_state.tree[new_id] = node
    parent["children"].append(new_id)


def _render_typed_list(tl: TypedList) -> None:
    st.markdown(f"**TL&lt;{tl.item_type.__name__}&gt;** · {len(tl)} items")
    preview_items = tl.items if len(tl) <= 10 else [*tl.items[:10], "…"]
    st.code(repr(preview_items), language="python")


def _render_node(
    node_id: int,
    depth: int,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    node = st.session_state.tree[node_id]

    if depth == 0:
        target = st.container()
    else:
        d = min(depth, 6)
        cs = st.columns([d, 16 - d])
        target = cs[1]

    with target:
        with st.container(border=True):
            if node["applied_fn"]:
                st.caption(f"⤷ via `{node['applied_fn']}`")

            if node["error"]:
                st.error(node["error"])
            else:
                tl: TypedList = node["typed_list"]
                _render_typed_list(tl)

                tl_type = _type_str(tl.item_type)
                compatible = [f for f in available_functions if f.input_type == tl_type]
                if compatible:
                    fn_names = [f.name for f in compatible]
                    col_sel, col_btn = st.columns([4, 1])
                    with col_sel:
                        selected = st.selectbox(
                            "Apply function",
                            fn_names,
                            key=f"sel_{node_id}",
                            label_visibility="collapsed",
                        )
                    with col_btn:
                        if st.button("Apply", key=f"btn_{node_id}"):
                            fn = next(f for f in compatible if f.name == selected)
                            _apply_function(node_id, fn, executor)
                            st.rerun()
                else:
                    st.caption(f"No compatible functions for type `{tl_type}`")

    for child_id in node["children"]:
        _render_node(child_id, depth + 1, available_functions, executor)


def main() -> None:
    st.set_page_config(
        page_title="Trajectory Explorer", page_icon="🌳", layout="wide"
    )
    st.title("🌳 Trajectory Explorer")
    st.caption("Pick an input, then branch by applying functions at any node.")

    eval_file = st.sidebar.text_input("Eval file", DEFAULT_EVAL_FILE)
    with st.spinner(f"Loading {eval_file}…"):
        trajectories, available_functions = load_eval(eval_file)
    st.sidebar.success(
        f"{len(trajectories)} inputs · {len(available_functions)} functions"
    )

    options = [f"#{i}: {t.input!r}" for i, t in enumerate(trajectories)]
    selected_idx = st.sidebar.selectbox(
        "Input TypedList",
        range(len(options)),
        format_func=lambda i: options[i],
    )
    selected_input = trajectories[selected_idx].input
    selected_repr = repr(selected_input)

    if (
        "tree" not in st.session_state
        or st.session_state.get("root_input_repr") != selected_repr
    ):
        _init_tree(selected_input)
        st.session_state.root_input_repr = selected_repr

    if st.sidebar.button("🔄 Reset tree"):
        _init_tree(selected_input)
        st.rerun()

    executor = Executor(available_functions)
    _render_node(0, 0, available_functions, executor)


if __name__ == "__main__":
    main()
