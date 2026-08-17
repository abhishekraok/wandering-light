"""Corpus page: what a generated corpus holds, and how it compares to older data.

The manifest is enough for every overview number, so this page opens without
touching the payload -- which a fresh clone has to fetch from the Hub before any
task can be browsed at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from wandering_light.evals import corpus_view

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord

SELECTED_TASK_KEY = "explorer_selected_task"
BROWSE_LIMIT = 4000

LEGACY_SOURCES = {
    "random_inputs_500_shortest_v1": corpus_view.LEGACY_EVAL_RELABEL,
    "induction_shortest_v1": corpus_view.LEGACY_TRAIN_RELABEL,
}


@st.cache_data(show_spinner=False)
def load_manifest_cached(manifest_path: str) -> dict[str, Any]:
    return corpus_view.read_manifest(manifest_path)


@st.cache_resource(show_spinner=False)
def load_records_cached(
    directory: str, name: str, split: str, limit: int
) -> list[BasisTaskRecord]:
    ref = corpus_view.CorpusRef(name=name, directory=Path(directory))
    return corpus_view.load_records(
        ref, corpus_view.load_manifest(ref), split, limit=limit
    )


@st.cache_data(show_spinner=False)
def legacy_profiles_cached(
    path: str, name: str
) -> tuple[corpus_view.DistanceProfile, corpus_view.DistanceProfile]:
    records = corpus_view.read_relabel_records(path)
    return corpus_view.relabel_distance_profiles(records, name=name)


def select_corpus() -> tuple[corpus_view.CorpusRef, dict[str, Any]] | None:
    """Corpus picker plus manifest. ``None`` when the repo has no manifest."""
    refs = corpus_view.discover_corpora()
    if not refs:
        st.warning(
            f"No corpus manifest under `{corpus_view.CORPUS_ROOT}/*/manifest.json`."
        )
        return None
    names = [ref.name for ref in refs]
    index = names.index("deep_corpus_v1") if "deep_corpus_v1" in names else 0
    chosen = st.selectbox("Corpus", names, index=index, key="corpus_name")
    ref = refs[names.index(chosen)]
    return ref, load_manifest_cached(str(ref.manifest_path))


def _render_headline(manifest: dict[str, Any]) -> None:
    headline = corpus_view.corpus_headline(manifest)
    cols = st.columns(5)
    cols[0].metric("Tasks", f"{headline['tasks']:,}")
    cols[1].metric("Roots", headline["roots"])
    cols[2].metric("Splits", len(headline["splits"]))
    cols[3].metric(
        "States expanded",
        f"{headline['reached_states']:,}" if headline["reached_states"] else "—",
    )
    wall = headline["wall_seconds"]
    cols[4].metric("Expansion", f"{wall / 60:.0f} min" if wall else "—")
    st.markdown(
        f"**Basis:** `{headline['basis_set_id']}` · **Generator:** "
        f"`{headline['generator']}`"
    )
    if headline["distance_semantics"]:
        st.caption(f"Distance: {headline['distance_semantics']}")


def _render_payload_status(
    ref: corpus_view.CorpusRef, manifest: dict[str, Any]
) -> bool:
    """Report and offer to fetch a missing payload. True when tasks are local."""
    missing = corpus_view.missing_splits(ref, manifest)
    if not missing:
        return True
    st.warning(
        f"Payload not on disk for: {', '.join(missing)}. The manifest is "
        "committed, the split files are fetched from the Hub."
    )
    hub = manifest.get("hub", {})
    st.caption(
        f"Hub: `{hub.get('repo_id', '—')}` at revision `{hub.get('revision', '—')[:12]}`"
    )
    if st.button("⬇️ Download and verify payload", key="corpus_download"):
        with st.spinner("Downloading…"):
            try:
                corpus_view.download_corpus(ref)
            except Exception as error:  # network, auth, digest mismatch
                st.error(f"{type(error).__name__}: {error}")
                return False
        load_records_cached.clear()
        st.rerun()
    return False


def _render_comparison(manifest: dict[str, Any], corpus_name: str) -> None:
    st.subheader("Distance distribution")
    st.caption(
        "Share of tasks at each certified shortest-path distance. The older "
        "eval sets top out at four; that ceiling is what the forward generator "
        "was built to clear."
    )
    profiles = [
        corpus_view.corpus_distance_profile(manifest, name=corpus_name),
    ]
    chosen = st.multiselect(
        "Compare against",
        list(LEGACY_SOURCES),
        default=["random_inputs_500_shortest_v1"],
        key="corpus_compare_sources",
    )
    show_nominal = st.checkbox(
        "Show nominal random-walk lengths too",
        value=False,
        key="corpus_show_nominal",
        help="What the random-walk generator claimed, before relabelling proved "
        "most of those walks had a shorter route.",
    )
    for name in chosen:
        path = LEGACY_SOURCES[name]
        if not path.exists():
            st.caption(f"Missing `{path}` — skipped.")
            continue
        certified, nominal = legacy_profiles_cached(str(path), name)
        profiles.append(certified)
        if show_nominal:
            profiles.append(nominal)

    rows = corpus_view.profile_rows(profiles)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    figure = px.bar(
        frame,
        x="distance",
        y="share",
        color="dataset",
        barmode="group",
        labels={"share": "share of tasks", "distance": "certified distance"},
    )
    figure.update_layout(yaxis_tickformat=".0%", height=380)
    st.plotly_chart(figure, width="stretch")

    st.dataframe(
        frame.pivot_table(
            index="distance", columns="dataset", values="tasks", fill_value=0
        ),
        width="stretch",
    )
    for profile in profiles:
        if profile.note:
            st.caption(f"**{profile.name}** — {profile.note}")


def _render_split_details(manifest: dict[str, Any]) -> None:
    with st.expander("Per-split summary", expanded=False):
        rows = []
        for split, metadata in manifest["splits"].items():
            rows.append(
                {
                    "split": split,
                    "tasks": metadata["size"],
                    "roots": metadata["roots"],
                    "max distance": max(
                        int(d) for d in metadata["by_certified_distance"]
                    ),
                    "frontier-certified": metadata["by_certification"].get(
                        "frontier-extension", 0
                    ),
                    "mean optimal first actions": round(
                        metadata["optimal_first_actions"]["mean"] or 0, 2
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Expansion and filters", expanded=False):
        expansion = manifest.get("expansion", {})
        st.json(
            {
                key: expansion[key]
                for key in (
                    "reached_states",
                    "attempted_transitions",
                    "failed_transitions",
                    "skipped_self_loops",
                    "certified_depth_histogram",
                    "stop_reasons",
                    "rejections",
                    "wall_seconds",
                )
                if key in expansion
            }
        )
        st.json(manifest.get("filters", {}))


def _sort_unique(values) -> list:
    return sorted(set(values))


def _render_browser(ref: corpus_view.CorpusRef, manifest: dict[str, Any]) -> None:
    st.subheader("Task browser")
    splits = list(manifest["splits"])
    split = st.selectbox("Split", splits, key="corpus_split")
    limit = st.slider(
        "Records to load",
        min_value=200,
        max_value=BROWSE_LIMIT,
        value=1000,
        step=200,
        key="corpus_limit",
        help="Splits run to tens of thousands of tasks; loading a prefix keeps "
        "the page responsive.",
    )
    with st.spinner(f"Reading {split}…"):
        records = load_records_cached(str(ref.directory), ref.name, split, limit)
    if not records:
        st.info("No records in this split.")
        return

    rows = [corpus_view.record_row(record) for record in records]
    filter_cols = st.columns(4)
    with filter_cols[0]:
        distances = st.multiselect(
            "Distance", _sort_unique(r["distance"] for r in rows), key="corpus_f_dist"
        )
    with filter_cols[1]:
        input_types = st.multiselect(
            "Input type", _sort_unique(r["input_type"] for r in rows), key="corpus_f_in"
        )
    with filter_cols[2]:
        output_types = st.multiselect(
            "Output type",
            _sort_unique(r["output_type"] for r in rows),
            key="corpus_f_out",
        )
    with filter_cols[3]:
        function_name = st.text_input(
            "Uses function", key="corpus_f_fn", placeholder="e.g. int_to_str"
        ).strip()

    selected = corpus_view.filter_records(
        records,
        distances=distances,
        input_types=input_types,
        output_types=output_types,
        function_name=function_name or None,
    )
    st.caption(f"{len(selected)} of {len(records)} loaded tasks match.")
    if not selected:
        return

    st.dataframe(
        pd.DataFrame([corpus_view.record_row(r) for r in selected[:500]]).drop(
            columns=["task_id"]
        ),
        width="stretch",
        hide_index=True,
        height=260,
    )

    index = st.selectbox(
        "Inspect task",
        range(len(selected)),
        format_func=lambda i: (
            f"d{selected[i].metadata['certified_distance']} · "
            f"{selected[i].metadata['input_type'].split('.')[-1]}→"
            f"{selected[i].metadata['output_type'].split('.')[-1]} · "
            f"{', '.join(selected[i].witness_function_names)}"
        ),
        key="corpus_task_idx",
    )
    _render_task_detail(selected[index], ref.name)


def _render_task_detail(record: BasisTaskRecord, corpus_name: str) -> None:
    metadata = record.metadata
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Certified distance", metadata["certified_distance"])
        cols[1].metric("Certification", metadata["certification"])
        cols[2].metric(
            "Optimal first actions", len(metadata["optimal_first_action_names"])
        )
        cols[3].metric("Shell size", f"{metadata['distance_shell_size']:,}")

        left, right = st.columns(2)
        with left:
            st.caption("Input")
            st.code(repr(record.input_value), language="python")
        with right:
            st.caption("Output")
            st.code(repr(record.output_value), language="python")

        st.markdown(
            f"**Witness:** `{' → '.join(record.witness_function_names)}`  \n"
            f"**Optimal first actions:** "
            f"`{', '.join(metadata['optimal_first_action_names'])}`"
            + ("" if metadata["optimal_first_actions_complete"] else " *(partial)*")
        )
        st.caption(f"task_id `{record.task_id}` · root #{metadata['root_index']}")

        if st.button("▶ Send to playground", key="corpus_send_playground"):
            st.session_state[SELECTED_TASK_KEY] = {
                "corpus": corpus_name,
                "task_id": record.task_id,
                "input": record.input,
                "output": record.output,
                "witness": list(record.witness_function_names),
                "distance": metadata["certified_distance"],
            }
            st.success("Loaded into the Playground tab.")


def render_corpus_tab() -> None:
    st.caption(
        "Generated corpora described by a committed manifest. Overview numbers "
        "come from the manifest; browsing tasks needs the payload on disk."
    )
    chosen = select_corpus()
    if chosen is None:
        return
    ref, manifest = chosen

    _render_headline(manifest)
    st.divider()
    _render_comparison(manifest, ref.name)
    st.divider()
    _render_split_details(manifest)
    if _render_payload_status(ref, manifest):
        st.divider()
        _render_browser(ref, manifest)
