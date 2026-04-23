import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from wandering_light.constants import DEFAULT_EVAL_FILE as PROPOSER_EVAL_FILE
from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.trajectory import Trajectory, TrajectorySpec
from wandering_light.typed_list import TypedList

DEFAULT_EVAL_FILE = "wandering_light/evals/data/random_inputs_500.py"
RESULTS_ROOT = "results"
RESULTS_PROPOSER_DIR = "results/proposer"


@st.cache_resource(show_spinner=False)
def load_eval(eval_file: str):
    return load_eval_data_as_trajectories(eval_file)


@st.cache_resource(show_spinner=False)
def load_json_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _type_str(item_type: type) -> str:
    return f"{item_type.__module__}.{item_type.__qualname__}"


def _purge_widget_state(ns: str, node_ids: list[int] | None = None) -> None:
    prefixes = (f"{ns}_edge_sel_", f"{ns}_add_sel_")
    if node_ids is None:
        for key in list(st.session_state.keys()):
            if key.startswith(prefixes):
                del st.session_state[key]
        return
    for nid in node_ids:
        for prefix in prefixes:
            st.session_state.pop(f"{prefix}{nid}", None)


def _render_typed_list(tl: TypedList) -> None:
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
            key=f"{ns}_edge_sel_{node_id}",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Apply", key=f"{ns}_edge_btn_{node_id}"):
            new_fn = next(f for f in compatible if f.name == selected)
            deleted = tree.replace_edge(node_id, new_fn, executor)
            _purge_widget_state(ns, deleted)
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
            key=f"{ns}_add_sel_{node_id}",
            label_visibility="collapsed",
            placeholder="Add step…",
        )
    with col_btn:
        if st.button("Add", key=f"{ns}_add_btn_{node_id}"):
            fn = next(f for f in compatible if f.name == selected)
            tree.append_child(node_id, fn, executor)
            st.rerun()


def _render_node(
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
            _render_typed_list(node["typed_list"])
            if not node["children"]:
                _render_add_step(tree, node_id, ns, available_functions, executor)

    for child_id in node["children"]:
        _render_node(tree, child_id, ns, available_functions, executor)


def _init_eval_tree(traj: Trajectory, executor: Executor) -> None:
    _purge_widget_state("eval")
    st.session_state.tree_eval = TrajectoryTree.from_trajectory(traj, executor)


def _render_eval_tab() -> None:
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.caption(
            "Full trajectory is shown by default. Change any edge's function "
            "and hit Apply to recompute downstream."
        )

        eval_file = st.text_input(
            "Eval file", DEFAULT_EVAL_FILE, key="eval_file_input"
        )
        with st.spinner(f"Loading {eval_file}…"):
            trajectories, available_functions = load_eval(eval_file)
        st.caption(
            f"{len(trajectories)} inputs · {len(available_functions)} functions"
        )

        options = [f"#{i}: {t.input!r}" for i, t in enumerate(trajectories)]
        selected_idx = st.selectbox(
            "Sample",
            range(len(options)),
            format_func=lambda i: options[i],
            key="eval_sample_idx",
        )
        executor = Executor(available_functions)
        selected_traj: Trajectory = trajectories[selected_idx]
        sample_key = (eval_file, selected_idx, repr(selected_traj.input))

        if st.session_state.get("eval_sample_key") != sample_key:
            _init_eval_tree(selected_traj, executor)
            st.session_state.eval_sample_key = sample_key

        if st.button("🔄 Reset to original trajectory", key="eval_reset"):
            _init_eval_tree(selected_traj, executor)
            st.rerun()

        # Sample-specific namespace prevents widget state from one sample
        # leaking into another when switching samples.
        _render_node(
            st.session_state.tree_eval,
            ROOT_ID,
            f"eval_{selected_idx}",
            available_functions,
            executor,
        )


def _find_solver_runs() -> list[tuple[str, str]]:
    """Return (display_name, path) for every solver JSON under results/.

    Only considers direct subdirectories of `results/` that contain a
    `summary.json` — this excludes siblings like `results/proposer/`.
    """
    runs: list[tuple[str, str]] = []
    if not os.path.exists(RESULTS_ROOT):
        return runs
    for entry in os.listdir(RESULTS_ROOT):
        dir_path = os.path.join(RESULTS_ROOT, entry)
        if not os.path.isdir(dir_path):
            continue
        if not os.path.exists(os.path.join(dir_path, "summary.json")):
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".json") and fname != "summary.json":
                path = os.path.join(dir_path, fname)
                rel = os.path.relpath(path, RESULTS_ROOT)
                runs.append((rel, path))
    runs.sort(key=lambda r: r[0], reverse=True)
    return runs


def _resolve_fn_names(
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


def _build_tree_from_names(
    input_tl: TypedList,
    names: list[str],
    available: FunctionDefSet,
    executor: Executor,
) -> tuple[TrajectoryTree | None, list[str]]:
    resolved, missing = _resolve_fn_names(names, available)
    if missing:
        return None, missing
    tree = TrajectoryTree.with_root(input_tl)
    parent_id = ROOT_ID
    for fn in resolved:
        parent_id = tree.append_child(parent_id, fn, executor)
    return tree, missing


def _init_solver_trees(
    input_tl: TypedList,
    golden_names: list[str],
    predicted_names: list[str],
    available: FunctionDefSet,
    executor: Executor,
) -> None:
    _purge_widget_state("gold")
    _purge_widget_state("pred")
    gold_tree, gold_missing = _build_tree_from_names(
        input_tl, golden_names, available, executor
    )
    pred_tree, pred_missing = _build_tree_from_names(
        input_tl, predicted_names, available, executor
    )
    st.session_state.tree_gold = gold_tree
    st.session_state.tree_pred = pred_tree
    st.session_state.solver_missing_gold = gold_missing
    st.session_state.solver_missing_pred = pred_missing


def _render_solver_tree_column(
    header: str,
    tree_key: str,
    missing_key: str,
    ns: str,
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    st.subheader(header)
    missing = st.session_state.get(missing_key) or []
    tree = st.session_state.get(tree_key)
    if missing:
        st.error(f"Unknown functions: {', '.join(missing)}")
        return
    if tree is None:
        st.info("No trajectory.")
        return
    _render_node(tree, ROOT_ID, ns, available_functions, executor)


def _render_solver_tab() -> None:
    st.caption(
        "Compare golden vs. predicted trajectories for a solver run. "
        "Both trees are fully editable."
    )

    runs = _find_solver_runs()
    if not runs:
        st.warning(f"No solver JSON files found under `{RESULTS_ROOT}/`.")
        return

    top_l, top_r = st.columns([3, 1])
    with top_l:
        run_idx = st.selectbox(
            "Solver run",
            range(len(runs)),
            format_func=lambda i: runs[i][0],
            key="solver_run_idx",
        )
    solver_json_path = runs[run_idx][1]
    run_dir = os.path.dirname(solver_json_path)
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        st.error(f"`summary.json` missing from `{run_dir}`.")
        return

    summary = load_json_file(summary_path)
    eval_file = summary.get("eval_file")
    if not eval_file or not os.path.exists(eval_file):
        st.error(f"`eval_file` from summary not found: `{eval_file}`")
        return

    with st.spinner(f"Loading {eval_file}…"):
        _, available_functions = load_eval(eval_file)
    executor = Executor(available_functions)

    solver_data = load_json_file(solver_json_path)
    details = solver_data.get("detailed_results") or []
    if not details:
        st.warning(
            "This solver JSON has no `detailed_results` (older run format)."
        )
        return

    # Run-level metadata
    m1, m2, m3, m4 = st.columns(4)
    success_rate = solver_data.get("success_rate")
    with m1:
        st.metric(
            "Success rate",
            f"{success_rate:.1%}" if success_rate is not None else "—",
        )
    with m2:
        st.metric(
            "Successes",
            f"{solver_data.get('success_count', 0)} / "
            f"{solver_data.get('total_samples', 0)}",
        )
    with m3:
        avg_len = solver_data.get("avg_solution_length")
        st.metric(
            "Avg solution length",
            f"{avg_len:.2f}" if avg_len is not None else "—",
        )
    with m4:
        st.metric("Budget", summary.get("budget", "—"))

    meta_bits = [
        f"**Timestamp:** `{summary.get('timestamp', '—')}`",
        f"**Eval file:** `{eval_file}`",
    ]
    if summary.get("num_samples") is not None:
        meta_bits.append(f"**num_samples:** `{summary['num_samples']}`")
    st.markdown(" · ".join(meta_bits))

    command = summary.get("command")
    if command:
        with st.expander("Command", expanded=False):
            st.code(command, language="bash")

    st.divider()

    sample_col, random_col = st.columns([6, 1])
    # Render the random button first so it can mutate solver_sample_idx
    # before the selectbox below is instantiated.
    with random_col:
        st.write("")
        st.write("")
        if st.button("🎲 Random sample", key="solver_random_sample"):
            st.session_state.solver_sample_idx = random.randrange(len(details))
            st.rerun()
    with sample_col:
        sample_idx = st.selectbox(
            "Sample",
            range(len(details)),
            format_func=lambda i: (
                f"{'✅' if details[i].get('success') else '❌'} "
                f"#{i} · {details[i].get('input', '')[:90]}"
            ),
            key="solver_sample_idx",
        )
    sample = details[sample_idx]
    sample_key = (solver_json_path, sample_idx)

    try:
        input_tl = TypedList.parse_from_repr(sample["input"])
    except Exception as e:
        st.error(f"Failed to parse input repr: {e}")
        return

    if st.session_state.get("solver_sample_key") != sample_key:
        _init_solver_trees(
            input_tl,
            sample.get("golden_functions") or [],
            sample.get("predicted_functions") or [],
            available_functions,
            executor,
        )
        st.session_state.solver_sample_key = sample_key

    with top_r:
        st.write("")  # vertical alignment with selectbox
        st.write("")
        if st.button("🔄 Reset trees", key="solver_reset"):
            _init_solver_trees(
                input_tl,
                sample.get("golden_functions") or [],
                sample.get("predicted_functions") or [],
                available_functions,
                executor,
            )
            st.rerun()

    # Sample metadata
    status = "✅ Success" if sample.get("success") else "❌ Failure"
    st.markdown(
        f"**{status}** · solution length: `{sample.get('solution_length')}` · "
        f"golden: `{len(sample.get('golden_functions') or [])}` fn(s) · "
        f"predicted: `{len(sample.get('predicted_functions') or [])}` fn(s)"
    )

    exp_col, act_col = st.columns(2)
    with exp_col:
        st.caption("Expected output")
        st.code(sample.get("expected_output") or "(none)", language="python")
    with act_col:
        st.caption("Actual output")
        st.code(sample.get("actual_output") or "(none)", language="python")

    if sample.get("error"):
        st.error(sample["error"])

    st.divider()

    gold_col, pred_col = st.columns(2)
    with gold_col:
        _render_solver_tree_column(
            "Golden",
            "tree_gold",
            "solver_missing_gold",
            f"gold_{sample_idx}",
            available_functions,
            executor,
        )
    with pred_col:
        _render_solver_tree_column(
            "Predicted",
            "tree_pred",
            "solver_missing_pred",
            f"pred_{sample_idx}",
            available_functions,
            executor,
        )


def _find_proposer_runs() -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    if not os.path.exists(RESULTS_PROPOSER_DIR):
        return runs
    for fname in os.listdir(RESULTS_PROPOSER_DIR):
        if fname.endswith(".json"):
            path = os.path.join(RESULTS_PROPOSER_DIR, fname)
            runs.append((fname, path))
    runs.sort(key=lambda r: r[0], reverse=True)
    return runs


def _group_attempts(
    attempts: list[list[str]],
) -> tuple[list[tuple[tuple[str, ...], int]], int]:
    """Group non-empty attempts by sequence. Returns (groups, failure_count)."""
    non_empty = [tuple(a) for a in attempts if a]
    failures = len(attempts) - len(non_empty)
    counter = Counter(non_empty)
    groups = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return groups, failures


def _prop_sample_label(sample: dict, i: int) -> str:
    solve_rate = sample.get("solve_rate") or 0.0
    parse_ok = sample.get("parse_success", False)
    if not parse_ok:
        badge = "🚫"
    elif solve_rate == 1.0:
        badge = "✅"
    elif solve_rate > 0:
        badge = "⚖️"
    else:
        badge = "❌"
    spec = sample.get("problem_spec") or sample.get("raw_response") or ""
    return f"{badge} #{i} · solve={solve_rate:.2f} · {spec[:70]}"


def _prop_status_badge(solve_rate: float, parse_ok: bool) -> str:
    if not parse_ok:
        return "🚫 Parse failed"
    if solve_rate == 1.0:
        return "✅ Fully solved"
    if solve_rate > 0:
        return "⚖️ Partially solved"
    return "❌ Unsolved"


def _init_proposer_trees(
    problem_spec_str: str | None,
    attempts: list[list[str]],
    available_functions: FunctionDefSet,
    executor: Executor,
) -> None:
    input_tl: TypedList | None = None
    parse_error: str | None = None
    tree_golden = None
    missing_golden: list[str] = []

    if problem_spec_str:
        try:
            spec = TrajectorySpec.parse_from_string(
                problem_spec_str, available_functions
            )
            input_tl = spec.input
            fn_names = [fn.name for fn in spec.function_defs]
            tree_golden, missing_golden = _build_tree_from_names(
                input_tl, fn_names, available_functions, executor
            )
        except Exception as e:
            parse_error = str(e)

    st.session_state.tree_prop_golden = tree_golden
    st.session_state.prop_missing_golden = missing_golden
    st.session_state.prop_parse_error = parse_error
    st.session_state.prop_input_tl = input_tl

    groups, failures = _group_attempts(attempts)
    st.session_state.prop_groups = groups
    st.session_state.prop_failures = failures

    # Force the attempt tree to be (re)built on the next render.
    st.session_state.tree_prop_attempt = None
    st.session_state.prop_missing_attempt = []
    st.session_state.prop_attempt_key = None


def _render_proposer_tab() -> None:
    st.caption(
        "Explore problems generated by a proposer model and the solver's "
        "attempts on each. Both trees are fully editable."
    )

    runs = _find_proposer_runs()
    if not runs:
        st.warning(
            f"No proposer JSON files found under `{RESULTS_PROPOSER_DIR}/`."
        )
        return

    run_idx = st.selectbox(
        "Proposer run",
        range(len(runs)),
        format_func=lambda i: runs[i][0],
        key="proposer_run_idx",
    )
    run_path = runs[run_idx][1]
    run_data = load_json_file(run_path)
    samples = run_data.get("sample_results") or []
    if not samples:
        st.warning("No `sample_results` in this JSON.")
        return

    # Proposer JSON doesn't record which eval file was used; assume the default.
    eval_file = PROPOSER_EVAL_FILE
    if not os.path.exists(eval_file):
        st.error(f"Default eval file not found: `{eval_file}`")
        return
    with st.spinner(f"Loading {eval_file}…"):
        _, available_functions = load_eval(eval_file)
    executor = Executor(available_functions)

    # Run-level metadata
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Parse rate", f"{run_data.get('parse_rate', 0):.1%}")
    with m2:
        st.metric(
            "Solver success rate",
            f"{run_data.get('solver_success_rate', 0):.1%}",
        )
    with m3:
        st.metric("Num samples", run_data.get("num_samples", 0))
    with m4:
        st.metric(
            "Avg fn count", f"{run_data.get('avg_function_count', 0):.2f}"
        )
    with m5:
        st.metric(
            "Non-zero solve frac",
            f"{run_data.get('frac_non_zero_std', 0):.1%}",
        )
    st.markdown(
        f"**Eval file (assumed):** `{eval_file}` · "
        f"**avg_function_count_ratio:** "
        f"`{run_data.get('avg_function_count_ratio', 0):.2f}`"
    )

    st.divider()

    sample_col, random_col = st.columns([6, 1])
    with random_col:
        st.write("")
        st.write("")
        if st.button("🎲 Random sample", key="prop_random_sample"):
            st.session_state.prop_sample_idx = random.randrange(len(samples))
            st.rerun()
    with sample_col:
        sample_idx = st.selectbox(
            "Sample",
            range(len(samples)),
            format_func=lambda i: _prop_sample_label(samples[i], i),
            key="prop_sample_idx",
        )
    sample = samples[sample_idx]
    sample_key = (run_path, sample_idx)

    if st.session_state.get("prop_sample_key") != sample_key:
        _init_proposer_trees(
            sample.get("problem_spec"),
            sample.get("attempted_function_deflists") or [],
            available_functions,
            executor,
        )
        st.session_state.prop_sample_key = sample_key

    # Sample status
    solve_rate = sample.get("solve_rate") or 0.0
    parse_success = sample.get("parse_success", False)
    attempts = sample.get("attempted_function_deflists") or []
    successes = sum(1 for a in attempts if a)
    st.markdown(
        f"**{_prop_status_badge(solve_rate, parse_success)}** · "
        f"solve_rate: `{solve_rate:.2f}` · parse_success: `{parse_success}` · "
        f"{successes}/{len(attempts)} attempts succeeded"
    )

    with st.expander("Raw response", expanded=False):
        st.code(sample.get("raw_response") or "", language="text")

    if st.session_state.get("prop_parse_error"):
        st.error(
            f"Failed to parse problem_spec: {st.session_state.prop_parse_error}"
        )

    st.divider()

    gol_col, att_col = st.columns(2)
    with gol_col:
        st.subheader("Proposed problem")
        missing_g = st.session_state.get("prop_missing_golden") or []
        tree_g = st.session_state.get("tree_prop_golden")
        if missing_g:
            st.error(f"Unknown functions: {', '.join(missing_g)}")
        elif tree_g is None:
            st.info("No tree available.")
        else:
            _render_node(
                tree_g,
                ROOT_ID,
                f"prop_gol_{sample_idx}",
                available_functions,
                executor,
            )

    with att_col:
        st.subheader("Solver attempt")
        groups = st.session_state.get("prop_groups") or []
        failures = st.session_state.get("prop_failures", 0)
        if failures:
            st.markdown(f"❌ Failed attempts: **{failures}**")
        if not groups:
            st.info("No successful attempts to display.")
        else:
            group_idx = st.selectbox(
                "Attempt group",
                range(len(groups)),
                format_func=lambda i: (
                    f"✓ {', '.join(groups[i][0]) or '(empty)'} × {groups[i][1]}"
                ),
                key=f"prop_group_idx_{sample_idx}",
            )
            input_tl = st.session_state.get("prop_input_tl")
            attempt_key = (run_path, sample_idx, group_idx)
            if (
                st.session_state.get("prop_attempt_key") != attempt_key
                and input_tl is not None
            ):
                names = list(groups[group_idx][0])
                tree_a, missing_a = _build_tree_from_names(
                    input_tl, names, available_functions, executor
                )
                st.session_state.tree_prop_attempt = tree_a
                st.session_state.prop_missing_attempt = missing_a
                st.session_state.prop_attempt_key = attempt_key

            missing_a = st.session_state.get("prop_missing_attempt") or []
            tree_a = st.session_state.get("tree_prop_attempt")
            if missing_a:
                st.error(f"Unknown functions: {', '.join(missing_a)}")
            elif tree_a is None:
                st.info("No tree available.")
            else:
                _render_node(
                    tree_a,
                    ROOT_ID,
                    f"prop_att_{sample_idx}_{group_idx}",
                    available_functions,
                    executor,
                )


def main() -> None:
    st.set_page_config(
        page_title="Trajectory Explorer", page_icon="🌳", layout="wide"
    )
    st.title("🌳 Trajectory Explorer")

    eval_tab, solver_tab, proposer_tab = st.tabs(
        ["Eval file", "Solver run", "Proposer run"]
    )
    with eval_tab:
        _render_eval_tab()
    with solver_tab:
        _render_solver_tab()
    with proposer_tab:
        _render_proposer_tab()


if __name__ == "__main__":
    main()
