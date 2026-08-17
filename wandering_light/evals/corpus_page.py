"""Streamlit corpus browser, task workbench, and witness-graph viewer."""

from __future__ import annotations

import os
import random
import time
from inspect import signature
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from wandering_light.basis_dataset import typed_list_from_builtin_str
from wandering_light.basis_set import (
    BasisSetDigestMismatchError,
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.evals.corpus_index import (
    FUNCTION_ROLES,
    CorpusFilters,
    CorpusIndex,
    CorpusSource,
    RecordDetail,
    corpus_source,
    discover_corpus_sources,
    ensure_corpus_index,
    source_signature,
)
from wandering_light.evals.explorer_graph import (
    build_local_expansion,
    build_witness_projection,
    build_workspace_projection,
    graph_view_figure,
    resolve_witness_functions,
    validate_typed_list_workload,
)
from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefSet

if TYPE_CHECKING:
    from collections.abc import Callable


DEFAULT_CORPUS_ROOTS = (
    "wandering_light/training/data",
    "wandering_light/evals/data",
)
_ROLE_LABELS = {
    "witness": "Witness",
    "optimal_first": "Optimal first action",
    "optimal_last": "Optimal last action",
}


def render_corpus_page(
    *,
    render_node: Callable[[TrajectoryTree, int, str, FunctionDefSet, Executor], None],
) -> None:
    st.caption(
        "Compare corpora, filter tasks by basis-function role, edit stored "
        "witnesses, run bounded solvers, and inspect witness projections."
    )
    sources = _discover_sources()
    if not sources:
        st.warning("No `.jsonl.gz` corpus files or corpus manifests were found.")
        _manual_source_help()
        return

    source_by_key = {str(source.path): source for source in sources}
    labels = {key: _source_label(source) for key, source in source_by_key.items()}
    default_keys = [
        key
        for key, source in source_by_key.items()
        if source.name
        in {
            "deep_corpus_v1",
            "induction_shortest_v1",
            "random_inputs_500_shortest_v1",
        }
    ]
    if not default_keys:
        default_keys = [next(iter(source_by_key))]
    if "corpus_source_selection" in st.session_state:
        valid_selection = [
            key
            for key in st.session_state.corpus_source_selection
            if key in source_by_key
        ]
        if valid_selection != st.session_state.corpus_source_selection:
            st.session_state.corpus_source_selection = valid_selection

    active_defaults = [
        key
        for key in st.session_state.get("corpus_active_sources", ())
        if key in source_by_key
    ]
    selection_defaults = {}
    if "corpus_source_selection" not in st.session_state:
        selection_defaults["default"] = active_defaults or default_keys
    selected_keys = st.multiselect(
        "Corpora to compare",
        options=list(source_by_key),
        format_func=lambda key: labels[key],
        key="corpus_source_selection",
        **selection_defaults,
    )
    action_col, path_col = st.columns([1, 3])
    with action_col:
        if st.button("Load / refresh", type="primary", key="corpus_load"):
            st.session_state.corpus_active_sources = selected_keys
            _invalidate_corpus_workspace()
            st.rerun()
    with path_col:
        st.caption(
            "The first load streams each source into a persistent SQLite index; "
            "later reruns query the index and do not retain corpus rows in memory."
        )

    active_keys = st.session_state.get("corpus_active_sources", [])
    active_sources = [source_by_key[key] for key in active_keys if key in source_by_key]
    if not active_sources:
        st.info("Choose one or more corpora, then click **Load / refresh**.")
        _render_missing_sources([source_by_key[key] for key in selected_keys])
        _manual_source_help()
        return

    _render_missing_sources(active_sources)
    indices = _open_indices([source for source in active_sources if source.ready])
    if not indices:
        return

    st.divider()
    _render_comparison(indices)
    st.divider()
    selected_index, filters = _render_browser(indices)
    if selected_index is None or filters is None:
        return
    selected_record = _render_record_picker(selected_index, filters)
    if selected_record is None:
        return
    st.divider()
    _render_task_workspace(selected_index, selected_record, render_node=render_node)


def _configured_roots() -> tuple[str, ...]:
    configured = os.environ.get("WANDERING_LIGHT_CORPUS_PATHS")
    if configured:
        return tuple(path for path in configured.split(os.pathsep) if path)
    return DEFAULT_CORPUS_ROOTS


def _discover_sources() -> tuple[CorpusSource, ...]:
    discovery_errors: list[tuple[Any, Exception]] = []
    sources = list(
        discover_corpus_sources(
            _configured_roots(),
            on_error=lambda path, error: discovery_errors.append((path, error)),
        )
    )
    for path, error in discovery_errors:
        st.warning(f"Ignored invalid corpus source `{path}`: {error}")
    manual_widget_key = "corpus_manual_path"
    if manual_widget_key not in st.session_state:
        st.session_state[manual_widget_key] = st.session_state.get(
            "corpus_manual_path_value", ""
        )
    manual = st.text_input(
        "Additional corpus file or directory",
        placeholder="/path/to/corpus or /path/to/tasks.jsonl.gz",
        key=manual_widget_key,
    ).strip()
    st.session_state.corpus_manual_path_value = manual
    if manual:
        try:
            extra = corpus_source(manual)
            if all(source.path != extra.path for source in sources):
                sources.append(extra)
        except Exception as error:
            st.warning(f"Could not add `{manual}`: {error}")
    return tuple(sorted(sources, key=lambda source: (source.name, str(source.path))))


def _source_label(source: CorpusSource) -> str:
    state = "" if source.ready else " · payload missing"
    return f"{source.name}{state} — {source.path}"


def _manual_source_help() -> None:
    st.caption(
        "Discovery searches `wandering_light/training/data` and "
        "`wandering_light/evals/data`. Set `WANDERING_LIGHT_CORPUS_PATHS` or "
        "enter another local path above to add a source."
    )


def _render_missing_sources(sources: list[CorpusSource]) -> None:
    for source in sources:
        if source.ready:
            continue
        missing = ", ".join(path.name for path in source.missing_files) or "split files"
        st.warning(
            f"**{source.name}** is described by a manifest but is missing {missing}."
        )
        if not source.hub_repo_id or source.manifest_path is None:
            continue
        st.caption(
            f"Hub source: `{source.hub_repo_id}` at pinned revision "
            f"`{source.hub_revision or 'manifest default'}`."
        )
        if st.button(f"Fetch {source.name}", key=f"fetch_{_source_key(source)}"):
            try:
                from wandering_light.corpus_hub import fetch_corpus

                with st.spinner(f"Fetching and verifying {source.name}…"):
                    fetch_corpus(source.manifest_path)
                _invalidate_corpus_workspace()
                st.success("Corpus fetched and digest-verified.")
                st.rerun()
            except ImportError:
                st.error(
                    "The corpus Hub helper arrives with PR #37 and is not present yet."
                )
            except Exception as error:
                st.error(f"Fetch failed: {error}")


def _open_indices(sources: list[CorpusSource]) -> list[CorpusIndex]:
    indices: list[CorpusIndex] = []
    cache_dir = os.environ.get("WANDERING_LIGHT_CORPUS_CACHE")
    progress_line = st.empty()

    def report(progress) -> None:
        progress_line.caption(
            f"Indexing {progress.source_name}: file {progress.file_index}/"
            f"{progress.file_count}, {progress.records_indexed:,} records…"
        )

    for source in sources:
        try:
            with st.spinner(f"Opening {source.name}…"):
                index = ensure_corpus_index(
                    source,
                    cache_dir=cache_dir,
                    progress=report,
                )
            indices.append(index)
        except Exception as error:
            st.error(f"Could not index **{source.name}**: {error}")
    progress_line.empty()
    return indices


def _render_comparison(indices: list[CorpusIndex]) -> None:
    st.subheader("Corpus comparison")
    stats_rows: list[dict[str, Any]] = []
    for index in indices:
        stats = index.stats()
        stats_rows.append(
            {
                "corpus": index.source.name,
                "tasks": stats.records,
                "certified": stats.certified_records,
                "roots": stats.roots or None,
                "min distance": stats.min_distance,
                "max distance": stats.max_distance,
                "mean distance": stats.mean_distance,
            }
        )

    total_tasks = sum(row["tasks"] for row in stats_rows)
    max_distance = max(
        (row["max distance"] for row in stats_rows if row["max distance"] is not None),
        default=None,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Corpora", len(indices))
    metric_cols[1].metric("Indexed tasks", f"{total_tasks:,}")
    metric_cols[2].metric(
        "Largest distance", max_distance if max_distance is not None else "—"
    )
    metric_cols[3].metric(
        "Manifest-backed",
        sum(index.source.manifest_path is not None for index in indices),
    )
    st.dataframe(
        pd.DataFrame(stats_rows), hide_index=True, **_stretch_kwargs(st.dataframe)
    )

    distance_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    for index in indices:
        total = max(1, index.stats().records)
        distance_rows.extend(
            {
                "corpus": index.source.name,
                "distance": distance,
                "tasks": count,
                "share": count / total,
            }
            for distance, count in index.counts("distance")
        )
        type_rows.extend(
            {
                "corpus": index.source.name,
                "input type": str(input_type).removeprefix("builtins."),
                "tasks": count,
                "share": count / total,
            }
            for input_type, count in index.counts("input_type")
        )
        function_rows.extend(
            {
                "corpus": index.source.name,
                "function": count.function_name,
                "tasks": count.records,
                "share": count.records / total,
            }
            for count in index.function_counts(roles=("witness",), limit=20)
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        if distance_rows:
            figure = px.bar(
                pd.DataFrame(distance_rows),
                x="distance",
                y="tasks",
                color="corpus",
                barmode="group",
                title="Task distance",
            )
            st.plotly_chart(figure, **_stretch_kwargs(st.plotly_chart))
    with chart_right:
        if type_rows:
            figure = px.bar(
                pd.DataFrame(type_rows),
                x="input type",
                y="share",
                color="corpus",
                barmode="group",
                title="Input-type share",
            )
            figure.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(figure, **_stretch_kwargs(st.plotly_chart))
    if function_rows:
        with st.expander("Top witness-function frequencies", expanded=False):
            figure = px.bar(
                pd.DataFrame(function_rows),
                x="function",
                y="share",
                color="corpus",
                barmode="group",
            )
            figure.update_layout(yaxis_tickformat=".1%")
            st.plotly_chart(figure, **_stretch_kwargs(st.plotly_chart))


def _render_browser(
    indices: list[CorpusIndex],
) -> tuple[CorpusIndex | None, CorpusFilters | None]:
    st.subheader("Filter tasks")
    index_by_key = {str(index.source.path): index for index in indices}
    if st.session_state.get("corpus_browse_source") not in index_by_key:
        st.session_state.pop("corpus_browse_source", None)
    selected_key = st.selectbox(
        "Corpus to browse",
        options=list(index_by_key),
        format_func=lambda key: index_by_key[key].source.name,
        key="corpus_browse_source",
    )
    index = index_by_key[selected_key]
    namespace = _source_key(index.source)

    splits = index.values("split")
    distances = index.values("distance")
    input_types = index.values("input_type")
    output_types = index.values("output_type")

    first, second, third = st.columns(3)
    with first:
        split_key = f"corpus_splits_{namespace}"
        selected_splits = st.multiselect(
            "Split",
            splits,
            key=split_key,
            **_initial_widget_value(split_key, "default", list(splits)),
        )
    with second:
        if distances and min(distances) != max(distances):
            distance_key = f"corpus_distance_{namespace}"
            distance_range = st.slider(
                "Distance",
                min_value=int(min(distances)),
                max_value=int(max(distances)),
                key=distance_key,
                **_initial_widget_value(
                    distance_key,
                    "value",
                    (int(min(distances)), int(max(distances))),
                ),
            )
        elif distances:
            only_distance = int(distances[0])
            st.metric("Distance", only_distance)
            distance_range = (only_distance, only_distance)
        else:
            distance_range = (None, None)
    with third:
        match_mode = st.radio(
            "Selected functions",
            ("any", "all"),
            horizontal=True,
            key=f"corpus_match_{namespace}",
        )

    type_left, type_right = st.columns(2)
    with type_left:
        selected_inputs = st.multiselect(
            "Input type",
            input_types,
            format_func=lambda value: str(value).removeprefix("builtins."),
            key=f"corpus_inputs_{namespace}",
        )
    with type_right:
        selected_outputs = st.multiselect(
            "Output type",
            output_types,
            format_func=lambda value: str(value).removeprefix("builtins."),
            key=f"corpus_outputs_{namespace}",
        )

    roles_key = f"corpus_roles_{namespace}"
    selected_roles = st.multiselect(
        "Function role",
        FUNCTION_ROLES,
        format_func=lambda role: _ROLE_LABELS[role],
        key=roles_key,
        **_initial_widget_value(roles_key, "default", ("witness",)),
    )
    roles = tuple(selected_roles or FUNCTION_ROLES)
    counts = index.function_totals(roles=roles)
    function_names: dict[str, str] = {}
    function_records: dict[str, int] = {}
    for count in counts:
        function_names[count.function_key] = count.function_name
        function_records[count.function_key] = count.records
    selected_functions = st.multiselect(
        "Basis functions",
        options=list(function_names),
        format_func=lambda key: (
            f"{function_names[key]} · {function_records[key]:,} associated tasks"
        ),
        key=f"corpus_functions_{namespace}",
    )

    filters = CorpusFilters(
        splits=tuple(selected_splits),
        min_distance=distance_range[0],
        max_distance=distance_range[1],
        input_types=tuple(selected_inputs),
        output_types=tuple(selected_outputs),
        function_keys=tuple(selected_functions),
        function_match=match_mode,
        function_roles=roles,
    )
    matched = index.count_records(filters)
    st.caption(f"**{matched:,}** tasks match these filters.")
    return index, filters


def _render_record_picker(
    index: CorpusIndex, filters: CorpusFilters
) -> RecordDetail | None:
    namespace = _source_key(index.source)
    total = index.count_records(filters)
    if total == 0:
        st.info("No tasks match the current filters.")
        return None

    control_left, control_mid, control_right = st.columns([2, 1, 1])
    with control_left:
        task_prefix = st.text_input(
            "Jump to task ID prefix",
            key=f"corpus_task_prefix_{namespace}",
        ).strip()
    with control_mid:
        page_size_key = f"corpus_page_size_{namespace}"
        page_size = st.selectbox(
            "Rows per page",
            (25, 50, 100),
            key=page_size_key,
            **_initial_widget_value(page_size_key, "index", 1),
        )
    page_count = max(1, (total + page_size - 1) // page_size)
    page_key = f"corpus_page_{namespace}"
    if st.session_state.get(page_key, 1) > page_count:
        st.session_state[page_key] = page_count
    with control_right:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=page_count,
            key=page_key,
            **_initial_widget_value(page_key, "value", 1),
        )

    widget_key = f"corpus_selected_row_{namespace}"
    jump_col, random_col, _ = st.columns([1, 1, 3])
    with jump_col:
        if st.button("Jump", key=f"corpus_jump_{namespace}", disabled=not task_prefix):
            matches = index.find_task(task_prefix, filters=filters)
            if matches:
                st.session_state[widget_key] = matches[0].row_id
            else:
                st.warning("No matching task ID begins with that prefix.")
    with random_col:
        if st.button("Random match", key=f"corpus_random_{namespace}"):
            offset = random.randrange(total)
            random_row = index.records(filters, limit=1, offset=offset)[0]
            st.session_state[widget_key] = random_row.row_id

    rows = list(
        index.records(filters, limit=page_size, offset=(int(page) - 1) * page_size)
    )
    table = pd.DataFrame(
        [
            {
                "task": row.task_id[:12],
                "split": row.split,
                "distance": row.distance,
                "input type": row.input_type.removeprefix("builtins."),
                "output type": row.output_type.removeprefix("builtins."),
                "witness": " → ".join(row.witness_function_names),
                "input": row.input_preview,
                "output": row.output_preview,
            }
            for row in rows
        ]
    )
    st.dataframe(table, hide_index=True, **_stretch_kwargs(st.dataframe))

    row_ids = [row.row_id for row in rows]
    current = st.session_state.get(widget_key)
    if current is not None and current not in row_ids:
        # Preserve off-page jump/random selections only while they still
        # satisfy every active filter. A stale selection must never drive a
        # workspace whose task is absent from the displayed result set.
        if index.record_matches(current, filters):
            row_ids.insert(0, current)
        else:
            st.session_state.pop(widget_key, None)
            current = None
    if current is None:
        st.session_state[widget_key] = row_ids[0]

    label_cache = {row.row_id: row for row in rows}

    def label(row_id: int) -> str:
        row = label_cache.get(row_id)
        if row is not None:
            witness = " → ".join(row.witness_function_names) or "identity"
            return f"{row.task_id[:12]} · d={row.distance} · {witness}"
        detail = index.get_record(row_id)
        witness = " → ".join(detail.witness_function_names) or "identity"
        return f"{detail.task_id[:12]} · d={detail.distance} · {witness}"

    selected_row_id = st.selectbox(
        "Task workspace",
        row_ids,
        format_func=label,
        key=widget_key,
    )
    return index.get_record(selected_row_id)


def _render_task_workspace(
    index: CorpusIndex,
    record: RecordDetail,
    *,
    render_node: Callable[[TrajectoryTree, int, str, FunctionDefSet, Executor], None],
) -> None:
    st.subheader("Task workspace")
    try:
        basis, available_functions, assumed = _basis_for_record(record)
    except Exception as error:
        st.error(f"Cannot resolve this task's basis: {error}")
        return

    provenance = "assumed for legacy row" if assumed else "exact, digest-verified"
    meta_cols = st.columns(5)
    meta_cols[0].metric(
        "Distance", record.distance if record.distance is not None else "—"
    )
    meta_cols[1].metric("Split", record.split)
    meta_cols[2].metric("Input", record.input_type.removeprefix("builtins."))
    meta_cols[3].metric("Output", record.output_type.removeprefix("builtins."))
    meta_cols[4].metric(
        "Root", record.root_index if record.root_index is not None else "—"
    )
    st.caption(
        f"Task `{record.task_id}` · basis `{basis.basis_set_id}` · {provenance} · "
        f"certification `{record.certification or 'unknown'}`"
    )
    if assumed:
        st.warning(
            "This legacy row stores function names but no immutable basis ID or "
            "digest. Execution uses the current `default` basis as an explicit assumption."
        )

    try:
        require_reproducible_basis_runtime(basis)
    except RuntimeError as error:
        st.error(str(error))
        st.caption(
            "Corpus filtering and raw metadata remain available; execution is disabled."
        )
        with st.expander("Raw record"):
            st.json(record.raw)
        return

    workspace_key = (
        str(index.source.path),
        source_signature(index.source),
        record.row_id,
        record.task_id,
    )
    if st.session_state.get("corpus_workspace_key") != workspace_key:
        try:
            _initialize_workspace(record, available_functions, workspace_key)
        except Exception as error:
            st.error(f"Cannot open this task safely: {error}")
            with st.expander("Raw record"):
                st.json(record.raw)
            return

    input_key = "corpus_custom_input"
    output_key = "corpus_custom_output"
    if st.button("Reset stored task", key="corpus_reset_task"):
        _initialize_workspace(record, available_functions, workspace_key, force=True)
        st.rerun()
    # Widget-owned keys disappear while another explorer page is rendered.
    # Restore them from durable, non-widget state when the user comes back.
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state.corpus_applied_input
    if output_key not in st.session_state:
        st.session_state[output_key] = st.session_state.corpus_applied_output
    with st.form("corpus_custom_io_form"):
        input_col, output_col = st.columns(2)
        with input_col:
            st.text_area(
                "Input TypedList JSON",
                key=input_key,
                height=130,
            )
        with output_col:
            st.text_area(
                "Target TypedList JSON",
                key=output_key,
                height=130,
            )
        apply_io = st.form_submit_button("Apply custom I/O to stored witness")
    if apply_io:
        try:
            input_json = st.session_state[input_key]
            output_json = st.session_state[output_key]
            if len(input_json.encode("utf-8")) > 65_536:
                raise ValueError("input JSON exceeds 65,536 bytes")
            if len(output_json.encode("utf-8")) > 65_536:
                raise ValueError("target JSON exceeds 65,536 bytes")
            input_value = typed_list_from_builtin_str(input_json)
            target = typed_list_from_builtin_str(output_json)
            validate_typed_list_workload(input_value, input_json, label="input")
            validate_typed_list_workload(target, output_json, label="target")
            tree = _tree_for_record(
                record, available_functions, input_value=input_value
            )
            st.session_state.corpus_tree = tree
            st.session_state.corpus_target = target
            st.session_state.corpus_applied_input = st.session_state[input_key]
            st.session_state.corpus_applied_output = st.session_state[output_key]
            st.session_state.corpus_editor_generation += 1
            st.session_state.pop("corpus_solver_result", None)
            st.rerun()
        except Exception as error:
            st.error(f"Invalid custom I/O: {error}")

    _render_edited_status(
        st.session_state.corpus_tree,
        st.session_state.corpus_target,
    )
    st.caption(
        "Edit an edge by choosing another registered basis function and applying it; "
        "definitions stay immutable so corpus provenance remains reproducible."
    )

    editor_ns = (
        f"corpus_edit_{_source_key(index.source)}_{record.row_id}_"
        f"{st.session_state.corpus_editor_generation}"
    )
    render_node(
        st.session_state.corpus_tree,
        ROOT_ID,
        editor_ns,
        available_functions,
        Executor(available_functions),
    )

    with st.expander("Basis definitions", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "name": function.name,
                        "function ID": function.function_id,
                        "input": function.input_type,
                        "output": function.output_type,
                        "code": function.code,
                    }
                    for function in basis.functions
                ]
            ),
            hide_index=True,
            **_stretch_kwargs(st.dataframe),
        )

    st.divider()
    _render_solver(
        available_functions,
        input_value=st.session_state.corpus_tree.nodes[ROOT_ID]["typed_list"],
        target=st.session_state.corpus_target,
        render_node=render_node,
    )
    st.divider()
    _render_graph(
        index,
        record,
        available_functions,
        active_tree=st.session_state.corpus_tree,
        input_value=st.session_state.corpus_tree.nodes[ROOT_ID]["typed_list"],
        target=st.session_state.corpus_target,
    )
    with st.expander("Raw record", expanded=False):
        st.json(record.raw)


def _basis_for_record(record: RecordDetail):
    assumed = record.basis_set_id is None
    basis = load_basis_set(record.basis_set_id or "default")
    if record.basis_set_digest is not None and basis.digest != record.basis_set_digest:
        raise BasisSetDigestMismatchError(
            f"record says {record.basis_set_digest}, installed basis is {basis.digest}"
        )
    return basis, basis.as_function_set(), assumed


def _initialize_workspace(
    record: RecordDetail,
    available_functions: FunctionDefSet,
    workspace_key: tuple[str, str, int, str],
    *,
    force: bool = False,
) -> None:
    if force or st.session_state.get("corpus_workspace_key") != workspace_key:
        input_value = typed_list_from_builtin_str(record.input)
        target = (
            typed_list_from_builtin_str(record.output)
            if record.output is not None
            else input_value
        )
        validate_typed_list_workload(input_value, record.input, label="task input")
        validate_typed_list_workload(
            target, record.output or record.input, label="task target"
        )
        st.session_state.corpus_workspace_key = workspace_key
        st.session_state.corpus_custom_input = record.input
        st.session_state.corpus_custom_output = record.output or record.input
        st.session_state.corpus_applied_input = record.input
        st.session_state.corpus_applied_output = record.output or record.input
        st.session_state.corpus_tree = _tree_for_record(
            record, available_functions, input_value=input_value
        )
        st.session_state.corpus_target = target
        st.session_state.corpus_editor_generation = (
            st.session_state.get("corpus_editor_generation", 0) + 1
        )
        _clear_corpus_results()


def _clear_corpus_results() -> None:
    for key in (
        "corpus_solver_result",
        "corpus_graph_projection",
        "corpus_workspace_projection",
        "corpus_local_expansion",
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("_corpus_expand_palette_context", None)


def _invalidate_corpus_workspace() -> None:
    _clear_corpus_results()
    for key in (
        "corpus_workspace_key",
        "corpus_tree",
        "corpus_target",
        "corpus_applied_input",
        "corpus_applied_output",
        "corpus_custom_input",
        "corpus_custom_output",
    ):
        st.session_state.pop(key, None)


def _tree_for_record(
    record: RecordDetail,
    available_functions: FunctionDefSet,
    *,
    input_value,
) -> TrajectoryTree:
    functions = resolve_witness_functions(record, available_functions)
    tree = TrajectoryTree.with_root(input_value)
    executor = Executor(available_functions)
    parent_id = ROOT_ID
    for function in functions:
        parent_id = tree.append_child(parent_id, function, executor)
    return tree


def _tree_leaf(tree: TrajectoryTree) -> tuple[list, dict[str, Any]]:
    functions = []
    node = tree.nodes[ROOT_ID]
    seen = {ROOT_ID}
    while node["children"]:
        child_id = node["children"][0]
        if child_id in seen:
            break
        seen.add(child_id)
        node = tree.nodes[child_id]
        functions.append(node["applied_fn_def"])
    return functions, node


def _render_edited_status(tree: TrajectoryTree, target) -> None:
    functions, leaf = _tree_leaf(tree)
    if leaf["error"]:
        st.error(f"Edited path fails after {len(functions)} steps: {leaf['error']}")
    elif leaf["typed_list"] == target:
        st.success(f"Edited path reaches the target in {len(functions)} steps.")
    else:
        st.warning(f"Edited path does not reach the target ({len(functions)} steps).")


def _render_solver(
    available_functions: FunctionDefSet,
    *,
    input_value,
    target,
    render_node: Callable[[TrajectoryTree, int, str, FunctionDefSet, Executor], None],
) -> None:
    st.subheader("Run a bounded solver")
    st.caption(
        "Solver runs happen only when submitted. Inputs have expansion guards and "
        "the transition budget is capped because exhaustive failures on the deep "
        "corpus can still take time."
    )
    names = [function.name for function in available_functions]
    with st.form("corpus_solver_form"):
        control_a, control_b, control_c = st.columns(3)
        with control_a:
            solver_kind = st.selectbox(
                "Solver", ("BFS", "Random"), key="corpus_solver_kind"
            )
        with control_b:
            budget = st.number_input(
                "Transition / attempt budget",
                min_value=1,
                max_value=50_000,
                step=1_000,
                key="corpus_solver_budget",
                **_initial_widget_value("corpus_solver_budget", "value", 20_000),
            )
        with control_c:
            max_depth = st.number_input(
                "Maximum depth / path length",
                min_value=1,
                max_value=8,
                key="corpus_solver_depth",
                **_initial_widget_value("corpus_solver_depth", "value", 4),
            )
        palette_key = "corpus_solver_palette_" + (
            (
                next(iter(available_functions)).metadata.get("basis_set_id", "legacy")
                or "legacy"
            )
            if len(available_functions)
            else "empty"
        )
        palette_names = st.multiselect(
            "Execution palette",
            names,
            key=palette_key,
            **_initial_widget_value(palette_key, "default", names),
            help="Only registered functions from the task's exact basis can be selected.",
        )
        run_solver = st.form_submit_button("Run solver", type="primary")

    if max_depth >= 6:
        st.warning(
            "Depth 6+ exhaustive failures can take tens of seconds even with a "
            "moderate palette; the budget remains the hard stop."
        )
    if run_solver:
        solver_generation = st.session_state.get("corpus_solver_generation", 0) + 1
        st.session_state.corpus_solver_generation = solver_generation
        try:
            from wandering_light.solver import create_bfs_solver, create_random_solver

            validate_typed_list_workload(
                input_value, input_value.to_string(), label="solver input"
            )
            validate_typed_list_workload(
                target, target.to_string(), label="solver target"
            )
            selected_palette = FunctionDefSet(
                [available_functions.name_to_function[name] for name in palette_names]
            )
            if solver_kind == "BFS":
                solver = create_bfs_solver(
                    budget=int(budget),
                    max_depth=int(max_depth),
                    track_function_usage=False,
                )
            else:
                solver = create_random_solver(
                    budget=int(budget),
                    path_length=int(max_depth),
                    track_function_usage=False,
                )
            started = time.perf_counter()
            with st.spinner(f"Running {solver_kind}…"):
                result = solver.solve(input_value, target, selected_palette)
            elapsed = time.perf_counter() - started
            tree = None
            if result.trajectory is not None:
                tree = TrajectoryTree.from_trajectory(
                    result.trajectory, Executor(available_functions)
                )
            st.session_state.corpus_solver_result = {
                "success": result.success,
                "error": result.error_msg,
                "elapsed": elapsed,
                "tree": tree,
                "kind": solver_kind,
                "generation": solver_generation,
            }
        except Exception as error:
            st.session_state.corpus_solver_result = {
                "success": False,
                "error": str(error),
                "elapsed": 0.0,
                "tree": None,
                "kind": solver_kind,
                "generation": solver_generation,
            }

    result = st.session_state.get("corpus_solver_result")
    if not result:
        return
    if result["success"]:
        st.success(f"{result['kind']} solved the task in {result['elapsed']:.3f}s.")
    else:
        st.error(
            f"{result['kind']} did not solve the task in {result['elapsed']:.3f}s: "
            f"{result['error'] or 'no solution'}"
        )
    if result["tree"] is not None:
        render_node(
            result["tree"],
            ROOT_ID,
            f"corpus_solver_{st.session_state.corpus_editor_generation}_"
            f"{result['generation']}",
            available_functions,
            Executor(available_functions),
        )


def _render_graph(
    index: CorpusIndex,
    record: RecordDetail,
    available_functions: FunctionDefSet,
    *,
    active_tree: TrajectoryTree,
    input_value,
    target,
) -> None:
    st.subheader("TrajectoryGraph lab")
    st.caption(
        "Visualize the live workspace, replay stored witnesses, or mine a tightly "
        "bounded local expansion. None of these reconstructs the exhaustive search "
        "used to certify the corpus, which may contain millions of states per root."
    )
    mode = st.radio(
        "Graph source",
        ("Active workspace", "Stored corpus witnesses", "Bounded local expansion"),
        horizontal=True,
        key="corpus_graph_mode",
    )

    cap_a, cap_b = st.columns(2)
    with cap_a:
        max_nodes = st.number_input(
            "Max rendered nodes",
            min_value=10,
            max_value=2_000,
            key="corpus_graph_max_nodes",
            **_initial_widget_value("corpus_graph_max_nodes", "value", 600),
        )
    with cap_b:
        max_edges = st.number_input(
            "Max rendered edge groups",
            min_value=10,
            max_value=4_000,
            key="corpus_graph_max_edges",
            **_initial_widget_value("corpus_graph_max_edges", "value", 1_200),
        )

    if mode == "Active workspace":
        _render_workspace_graph(
            active_tree,
            available_functions,
            target=target,
            max_nodes=int(max_nodes),
            max_edges=int(max_edges),
        )
    elif mode == "Stored corpus witnesses":
        _render_stored_graph(
            index,
            record,
            available_functions,
            max_nodes=int(max_nodes),
            max_edges=int(max_edges),
        )
    else:
        _render_local_expansion(
            index,
            record,
            available_functions,
            input_value=input_value,
            max_nodes=int(max_nodes),
            max_edges=int(max_edges),
        )


def _render_workspace_graph(
    active_tree: TrajectoryTree,
    available_functions: FunctionDefSet,
    *,
    target,
    max_nodes: int,
    max_edges: int,
) -> None:
    trees = [active_tree]
    solver_result = st.session_state.get("corpus_solver_result")
    if solver_result and solver_result.get("tree") is not None:
        trees.append(solver_result["tree"])
    graph_key = (
        st.session_state.get("corpus_workspace_key"),
        tuple(_tree_graph_signature(tree) for tree in trees),
        target.to_string(),
        max_nodes,
        max_edges,
    )
    if st.button("Build workspace graph", key="corpus_build_workspace_graph"):
        with st.spinner("Merging the active editor and solver paths…"):
            projection = build_workspace_projection(
                trees,
                available_functions,
                target=target,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
        st.session_state.corpus_workspace_projection = (graph_key, projection)

    stored = st.session_state.get("corpus_workspace_projection")
    if not stored or stored[0] != graph_key:
        return
    projection = stored[1]
    view = projection.view
    sources = "edited tree"
    if len(trees) > 1:
        sources += " + latest solver path"
    st.caption(
        f"Merged {sources} · {projection.processed_edges:,} path edges · "
        f"{view.total_nodes:,} states"
        + (" · rendered view is capped" if view.truncated else "")
    )
    st.caption("Blue = input · orange = active path · red = matching target")
    if projection.errors:
        with st.expander(f"{len(projection.errors)} workspace projection errors"):
            st.code("\n".join(projection.errors), language="text")
    _render_graph_diagnostics(view)
    st.plotly_chart(graph_view_figure(view), **_stretch_kwargs(st.plotly_chart))


def _render_stored_graph(
    index: CorpusIndex,
    record: RecordDetail,
    available_functions: FunctionDefSet,
    *,
    max_nodes: int,
    max_edges: int,
) -> None:
    scope_options = ["Selected witness"]
    if record.root_index is not None:
        scope_options.append(f"All stored witnesses for root {record.root_index}")
    scope_key = "corpus_graph_scope"
    if st.session_state.get(scope_key) not in (None, *scope_options):
        st.session_state[scope_key] = scope_options[0]
    scope = st.radio(
        "Graph scope",
        scope_options,
        horizontal=True,
        key=scope_key,
    )
    with st.container():
        max_records = st.number_input(
            "Max witnesses",
            min_value=1,
            max_value=500,
            key="corpus_graph_max_records",
            **_initial_widget_value("corpus_graph_max_records", "value", 250),
        )

    graph_key = (
        str(index.source.path),
        record.row_id,
        record.task_id,
        scope,
        int(max_records),
        max_nodes,
        max_edges,
    )
    if st.button("Build witness graph", key="corpus_build_graph"):
        records = [record]
        if scope != "Selected witness" and record.root_index is not None:
            summaries = index.records(
                CorpusFilters(root_indices=(record.root_index,)),
                limit=int(max_records),
            )
            records = [record]
            records.extend(
                index.get_record(summary.row_id)
                for summary in summaries
                if summary.row_id != record.row_id
            )
        with st.spinner(f"Replaying {len(records):,} stored witnesses…"):
            projection = build_witness_projection(
                records,
                available_functions,
                selected_task_id=record.task_id,
                max_records=int(max_records),
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
        st.session_state.corpus_graph_projection = (graph_key, projection)

    stored = st.session_state.get("corpus_graph_projection")
    if not stored or stored[0] != graph_key:
        return
    projection = stored[1]
    view = projection.view
    st.caption(
        f"Replayed {projection.processed_records:,} witnesses · "
        f"graph has {view.total_nodes:,} nodes / {view.total_edges:,} edges"
        + (" · rendered view is capped" if view.truncated else "")
    )
    st.caption(
        "Blue = root · orange = selected path · red = selected target · "
        "green = another stored target"
    )
    if projection.errors:
        with st.expander(f"{len(projection.errors)} witness replay errors"):
            st.code("\n".join(projection.errors), language="text")
    _render_graph_diagnostics(view)
    st.plotly_chart(graph_view_figure(view), **_stretch_kwargs(st.plotly_chart))


def _render_local_expansion(
    index: CorpusIndex,
    record: RecordDetail,
    available_functions: FunctionDefSet,
    *,
    input_value,
    max_nodes: int,
    max_edges: int,
) -> None:
    st.caption(
        "This ports the notebook's graph-expansion and task-mining loop. It starts "
        "from the current custom input and applies only the selected registered "
        "basis functions. Shortest-path certification is relative to that palette, "
        "not the full corpus basis."
    )
    all_names = [function.name for function in available_functions]
    defaults = list(dict.fromkeys(record.witness_function_names))
    defaults = [name for name in defaults if name in set(all_names)]
    if not defaults:
        defaults = all_names[: min(8, len(all_names))]
    palette_key = "corpus_expand_palette"
    palette_context = st.session_state.get("corpus_workspace_key")
    if st.session_state.get("_corpus_expand_palette_context") != palette_context:
        st.session_state[palette_key] = defaults
        st.session_state._corpus_expand_palette_context = palette_context
    elif palette_key in st.session_state:
        valid_palette = [
            name for name in st.session_state[palette_key] if name in set(all_names)
        ]
        if valid_palette != st.session_state[palette_key]:
            st.session_state[palette_key] = valid_palette or defaults
    palette_names = st.multiselect(
        "Expansion palette",
        all_names,
        key=palette_key,
        **_initial_widget_value(palette_key, "default", defaults),
        help="A small palette keeps branching predictable; the stored witness is the default.",
    )
    control_a, control_b, control_c, control_d = st.columns(4)
    with control_a:
        max_depth = st.number_input(
            "Expansion depth",
            min_value=1,
            max_value=3,
            key="corpus_expand_depth",
            **_initial_widget_value("corpus_expand_depth", "value", 2),
        )
    with control_b:
        max_states = st.number_input(
            "State cap",
            min_value=10,
            max_value=1_000,
            step=10,
            key="corpus_expand_states",
            **_initial_widget_value("corpus_expand_states", "value", 250),
        )
    with control_c:
        max_transitions = st.number_input(
            "Transition cap",
            min_value=10,
            max_value=5_000,
            step=100,
            key="corpus_expand_transitions",
            **_initial_widget_value("corpus_expand_transitions", "value", 2_500),
        )
    with control_d:
        include_self_loops = st.checkbox(
            "Keep self-loops",
            key="corpus_expand_self_loops",
            **_initial_widget_value("corpus_expand_self_loops", "value", True),
            help="Shows effective identities, as in the original notebook experiments.",
        )

    expansion_key = (
        str(index.source.path),
        record.row_id,
        input_value.to_string(),
        tuple(palette_names),
        int(max_depth),
        int(max_states),
        int(max_transitions),
        bool(include_self_loops),
        max_nodes,
        max_edges,
    )
    if st.button("Build local expansion", key="corpus_build_expansion"):
        try:
            selected = FunctionDefSet(
                [available_functions.name_to_function[name] for name in palette_names]
            )
            with st.spinner("Expanding the current input within hard caps…"):
                projection = build_local_expansion(
                    input_value,
                    selected,
                    max_depth=int(max_depth),
                    max_states=int(max_states),
                    max_transitions=int(max_transitions),
                    skip_self_loops=not include_self_loops,
                    max_nodes=max_nodes,
                    max_edges=max_edges,
                )
            st.session_state.corpus_local_expansion = (expansion_key, projection)
        except Exception as error:
            st.error(f"Cannot build local expansion: {error}")

    stored = st.session_state.get("corpus_local_expansion")
    if not stored or stored[0] != expansion_key:
        return
    projection = stored[1]
    view = projection.view
    summary_a, summary_b, summary_c, summary_d = st.columns(4)
    summary_a.metric("Reached states", f"{view.total_nodes:,}")
    summary_b.metric("Transitions tried", f"{projection.attempted_transitions:,}")
    summary_c.metric(
        "Palette-certified depth", f"{projection.certified_depth:,}"
    )
    summary_d.metric("Candidate tasks", f"{len(projection.tasks):,}")
    stop = projection.stop_reason or "complete through requested depth"
    st.caption(
        f"Stop: {stop} · {projection.failed_transitions:,} failed applications · "
        f"{projection.skipped_self_loops:,} self-loops skipped"
        + (" · rendered view is capped" if view.truncated else "")
    )
    _render_graph_diagnostics(view)
    st.plotly_chart(graph_view_figure(view), **_stretch_kwargs(st.plotly_chart))
    if projection.tasks:
        task_rows = [
            {
                "node": task.node_id,
                "distance": task.distance,
                "palette-shortest certified": task.certified,
                "functions": " → ".join(task.function_names),
                "output": task.output,
            }
            for task in projection.tasks[:200]
        ]
        st.dataframe(
            pd.DataFrame(task_rows),
            hide_index=True,
            **_stretch_kwargs(st.dataframe),
        )
        if len(projection.tasks) > len(task_rows):
            st.caption(
                f"Showing {len(task_rows):,} of {len(projection.tasks):,} candidate tasks."
            )


def _render_graph_diagnostics(view) -> None:
    diagnostics = view.diagnostics
    cols = st.columns(4)
    cols[0].metric("Self-loop groups", f"{diagnostics.self_loop_groups:,}")
    cols[1].metric(
        "Parallel alternatives", f"{diagnostics.parallel_function_groups:,}"
    )
    cols[2].metric("Convergent states", f"{diagnostics.convergent_nodes:,}")
    cols[3].metric("Directed cycle groups", f"{diagnostics.directed_cycle_groups:,}")
    if view.truncated:
        st.caption("Structural diagnostics describe the capped rendered view only.")


def _tree_graph_signature(tree: TrajectoryTree) -> tuple:
    return tuple(
        (
            node_id,
            node.get("parent"),
            getattr(node.get("applied_fn_def"), "name", None),
            (
                node["typed_list"].to_string()
                if node.get("typed_list") is not None
                else None
            ),
            node.get("error"),
            tuple(node.get("children", ())),
        )
        for node_id, node in sorted(tree.nodes.items())
    )


def _source_key(source: CorpusSource) -> str:
    import hashlib

    return hashlib.sha256(str(source.path).encode()).hexdigest()[:12]


def _stretch_kwargs(component) -> dict[str, Any]:
    """Use the current width API while retaining Streamlit 1.28 compatibility."""
    if "width" in signature(component).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def _initial_widget_value(key: str, argument: str, value: Any) -> dict[str, Any]:
    """Avoid Streamlit's default-plus-Session-State warning on page restore."""
    if key in st.session_state:
        return {}
    return {argument: value}
