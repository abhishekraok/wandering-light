"""Playground page: take a task, change the functions, run a solver on it.

Sources are a corpus task sent over from the Corpus tab, or an input/output pair
typed by hand. Either way the trajectory is editable and the solver is run
against the *target*, so a predicted path that differs from the witness still
counts -- that is how the solver is graded elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from wandering_light.basis_dataset import typed_list_from_builtin_str
from wandering_light.basis_set import (
    available_basis_set_aliases,
    available_basis_sets,
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.evals.explorer_corpus import SELECTED_TASK_KEY
from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.evals.explorer_widgets import (
    build_tree_from_names,
    purge_widget_state,
    render_node,
    tree_function_names,
    tree_leaf_value,
)
from wandering_light.executor import Executor
from wandering_light.solver import get_solver_by_name
from wandering_light.typed_list import TypedList

if TYPE_CHECKING:
    from wandering_light.function_def import FunctionDefSet

DEFAULT_BASIS = "wl-core-v1"
NS = "play"
TREE_KEY = "tree_play"
SIGNATURE_KEY = "play_signature"
RESULT_KEY = "play_solver_result"
DEFAULT_INPUT = "TL<int>([1, 2, 3])"
DEFAULT_OUTPUT = "TL<int>([3, 5, 7])"


@st.cache_resource(show_spinner=False)
def load_functions(basis_set_id: str) -> tuple[FunctionDefSet, str | None]:
    """The basis palette, plus any hash-seed requirement it imposes."""
    basis_set = load_basis_set(basis_set_id)
    seed = require_reproducible_basis_runtime(basis_set)
    return basis_set.as_function_set(), seed


def parse_typed_list(text: str) -> TypedList:
    """Accept either a ``TL<int>([1, 2])`` repr or the JSON wire format."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return typed_list_from_builtin_str(stripped)
    return TypedList.parse_from_repr(stripped)


def editable_text(serialized: str) -> str:
    """Prefer the readable repr, but only when it parses back to the value.

    Not every builtin round-trips through ``repr``; the JSON the record stores
    always does, so that is the fallback rather than a broken text box.
    """
    value = typed_list_from_builtin_str(serialized)
    text = repr(value)
    try:
        if TypedList.parse_from_repr(text) == value:
            return text
    except (ValueError, TypeError):
        pass
    return serialized


def select_basis(key: str) -> tuple[FunctionDefSet, str] | None:
    """Basis picker shared by the pages that execute functions."""
    ids = list(available_basis_sets())
    aliases = available_basis_set_aliases()
    reverse = {resolved: alias for alias, resolved in aliases.items()}
    index = ids.index(DEFAULT_BASIS) if DEFAULT_BASIS in ids else 0
    basis_set_id = st.selectbox(
        "Basis set",
        ids,
        index=index,
        format_func=lambda i: f"{i} ({reverse[i]})" if i in reverse else i,
        key=key,
    )
    try:
        functions, _seed = load_functions(basis_set_id)
    except RuntimeError as error:
        # A palette with randomized hashing is unusable without a fixed seed.
        st.error(f"{error}")
        return None
    return functions, basis_set_id


def _sync_source() -> dict[str, Any] | None:
    """Refill the input, target and function list when the source changes.

    Streamlit widgets keep their own state once created, so a task sent over
    from the Corpus tab has to be written into session state *before* the text
    areas are instantiated -- passing it as a default would be ignored.
    """
    task = st.session_state.get(SELECTED_TASK_KEY)
    options = ["Custom"] if task is None else ["Corpus task", "Custom"]
    source = st.radio("Source", options, horizontal=True, key="play_source")
    from_corpus = source == "Corpus task" and task is not None

    marker = (source, task["task_id"] if task else None)
    if st.session_state.get("play_source_marker") != marker:
        st.session_state["play_source_marker"] = marker
        if from_corpus:
            st.session_state["play_input"] = editable_text(task["input"])
            st.session_state["play_output"] = editable_text(task["output"])
            st.session_state["play_names"] = list(task["witness"])
        else:
            st.session_state["play_input"] = DEFAULT_INPUT
            st.session_state["play_output"] = DEFAULT_OUTPUT
            st.session_state["play_names"] = []

    if task is None:
        st.caption(
            "Pick a task in the Corpus tab and press *Send to playground* to "
            "load one from the corpus."
        )
    return task if from_corpus else None


def _reset_tree(
    input_value: TypedList,
    names: list[str],
    functions: FunctionDefSet,
    executor: Executor,
) -> None:
    purge_widget_state(NS)
    tree, missing = build_tree_from_names(input_value, names, functions, executor)
    st.session_state[TREE_KEY] = tree or TrajectoryTree.with_root(input_value)
    st.session_state[RESULT_KEY] = None
    if missing:
        st.warning(f"Not in this basis, dropped: {', '.join(missing)}")


def _render_solver_controls(
    input_value: TypedList, target: TypedList, functions: FunctionDefSet
) -> None:
    st.subheader("Run a solver")
    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        solver_name = st.selectbox("Solver", ["bfs", "random"], key="play_solver")
    with cols[1]:
        budget = st.number_input(
            "Budget",
            min_value=1,
            max_value=200_000,
            value=2000,
            step=100,
            key="play_budget",
        )
    with cols[2]:
        depth = st.number_input(
            "Max depth / path length",
            min_value=1,
            max_value=8,
            value=2,
            key="play_depth",
            help="BFS over the full basis grows fast: depth 3 is seconds, "
            "depth 6 is minutes per task.",
        )
    with cols[3]:
        st.write("")
        st.write("")
        run = st.button("▶ Solve", key="play_run")

    if run:
        solver = get_solver_by_name(
            solver_name, budget=int(budget), path_length=int(depth)
        )
        with st.spinner(f"Solving with {solver_name}…"):
            result = solver.solve(input_value, target, functions)
        st.session_state[RESULT_KEY] = {
            "success": result.success,
            "functions": (
                [fn.name for fn in result.trajectory.function_defs]
                if result.trajectory is not None
                else []
            ),
            "output": (
                repr(result.trajectory.output)
                if result.trajectory is not None
                else None
            ),
            "error": result.error_msg,
            "solver": f"{solver_name} (budget {int(budget)}, depth {int(depth)})",
        }

    outcome = st.session_state.get(RESULT_KEY)
    if not outcome:
        return
    if outcome["success"]:
        st.success(
            f"Solved by {outcome['solver']} in {len(outcome['functions'])} step(s): "
            f"`{' → '.join(outcome['functions'])}`"
        )
    else:
        st.error(f"{outcome['solver']} failed: {outcome['error'] or 'no path found'}")
    if outcome["output"]:
        st.code(outcome["output"], language="python")


def render_playground_tab() -> None:
    st.caption(
        "Edit a trajectory's functions and watch every intermediate state "
        "recompute, then ask a solver for its own path to the same target."
    )
    basis = select_basis("play_basis")
    if basis is None:
        return
    functions, basis_set_id = basis
    executor = Executor(functions)

    task = _sync_source()
    seed_names: list[str] = st.session_state.get("play_names", [])

    left, right = st.columns(2)
    with left:
        input_text = st.text_area("Input", height=90, key="play_input")
    with right:
        output_text = st.text_area("Target output", height=90, key="play_output")
    st.caption("Accepts `TL<int>([1, 2, 3])` or the JSON wire format.")

    try:
        input_value = parse_typed_list(input_text)
    except Exception as error:
        st.error(f"Could not parse input: {error}")
        return
    try:
        target = parse_typed_list(output_text)
    except Exception as error:
        st.error(f"Could not parse target output: {error}")
        return

    if task is not None:
        st.caption(
            f"Corpus task `{task['task_id'][:12]}` from `{task['corpus']}` · "
            f"certified distance {task['distance']}"
        )

    signature = (basis_set_id, input_text, tuple(seed_names))
    if st.session_state.get(SIGNATURE_KEY) != signature:
        _reset_tree(input_value, seed_names, functions, executor)
        st.session_state[SIGNATURE_KEY] = signature

    if st.button("🔄 Reset trajectory", key="play_reset"):
        _reset_tree(input_value, seed_names, functions, executor)
        st.rerun()

    tree: TrajectoryTree = st.session_state[TREE_KEY]
    render_node(tree, ROOT_ID, NS, functions, executor)

    leaf = tree_leaf_value(tree)
    names = tree_function_names(tree)
    if leaf is None:
        st.warning("The current trajectory does not execute.")
    elif leaf == target:
        st.success(f"Reaches the target in {len(names)} step(s).")
    else:
        st.info("Current trajectory does not reach the target output.")

    st.divider()
    _render_solver_controls(input_value, target, functions)
