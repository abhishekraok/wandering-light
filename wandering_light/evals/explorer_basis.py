"""Basis page: pick a palette, then see everything a corpus says about it.

A basis set is the action vocabulary every artifact is bound to, so the useful
question is not "what functions exist" but "what does the data do with them" --
how often each one appears in a witness, how often it is an optimal first move,
and at what distances it shows up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from wandering_light.basis_set import (
    available_basis_set_aliases,
    available_basis_sets,
    load_basis_set,
)
from wandering_light.evals import corpus_view
from wandering_light.evals.explorer_corpus import (
    load_manifest_cached,
    load_records_cached,
)

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.basis_set import BasisSet

STATS_LIMIT = 4000


@st.cache_resource(show_spinner=False)
def load_basis_cached(basis_set_id: str) -> BasisSet:
    return load_basis_set(basis_set_id)


@st.cache_data(show_spinner=False)
def function_rows(basis_set_id: str) -> list[dict[str, Any]]:
    basis_set = load_basis_cached(basis_set_id)
    return [
        {
            "function": function.name,
            "input": function.input_type.removeprefix("builtins."),
            "output": function.output_type.removeprefix("builtins."),
            "function_id": function.function_id,
            "code": function.code,
        }
        for function in basis_set.functions
    ]


def _corpus_records(
    ref: corpus_view.CorpusRef, manifest: dict[str, Any], splits: list[str], limit: int
) -> list[BasisTaskRecord]:
    records: list[BasisTaskRecord] = []
    for split in splits:
        records.extend(load_records_cached(str(ref.directory), ref.name, split, limit))
    return records


def _attach_corpus_stats(
    rows: list[dict[str, Any]],
    records: list[BasisTaskRecord],
    id_to_name: dict[str, str],
) -> pd.DataFrame:
    stats = corpus_view.function_stats(records, id_to_name=id_to_name)
    frame = pd.DataFrame(rows)
    counts = pd.DataFrame([entry.row() for entry in stats.values()])
    if counts.empty:
        return frame
    merged = frame.merge(counts, on="function", how="left")
    numeric = ["witness_uses", "witness_tasks", "optimal_first", "optimal_last"]
    merged[numeric] = merged[numeric].fillna(0).astype(int)
    return merged.sort_values("witness_uses", ascending=False)


def _render_function_detail(
    name: str,
    rows: list[dict[str, Any]],
    records: list[BasisTaskRecord],
) -> None:
    definition = next(row for row in rows if row["function"] == name)
    st.markdown(
        f"**`{name}`** · `{definition['input']}` → `{definition['output']}`  \n"
        f"`{definition['function_id']}`"
    )
    st.code(f"def {name}(x):\n    {definition['code']}", language="python")

    if not records:
        return
    using = corpus_view.filter_records(records, function_name=name)
    first = [r for r in records if name in r.metadata["optimal_first_action_names"]]
    cols = st.columns(3)
    cols[0].metric("Tasks with it in the witness", len(using))
    cols[1].metric("Tasks where it is an optimal first action", len(first))
    cols[2].metric(
        "Share of loaded tasks", f"{len(using) / len(records):.1%}" if records else "—"
    )
    if not using:
        st.info("This function appears in no loaded witness.")
        return

    histogram = (
        pd.DataFrame([{"distance": r.metadata["certified_distance"]} for r in using])
        .value_counts("distance")
        .sort_index()
    )
    st.bar_chart(histogram, height=200)
    st.dataframe(
        pd.DataFrame([corpus_view.record_row(r) for r in using[:200]])[
            ["split", "distance", "input_type", "output_type", "witness"]
        ],
        width="stretch",
        hide_index=True,
        height=240,
    )


def render_basis_tab() -> None:
    st.caption(
        "Immutable, content-addressed function palettes, and how a corpus uses them."
    )
    ids = list(available_basis_sets())
    reverse = {
        resolved: alias for alias, resolved in available_basis_set_aliases().items()
    }
    basis_set_id = st.selectbox(
        "Basis set",
        ids,
        format_func=lambda i: f"{i} ({reverse[i]})" if i in reverse else i,
        key="basis_id",
    )
    basis_set = load_basis_cached(basis_set_id)
    cols = st.columns(3)
    cols[0].metric("Functions", len(basis_set))
    cols[1].metric("Parent", basis_set.parent_basis_set_id or "—")
    cols[2].metric("Digest", basis_set.digest.removeprefix("sha256:")[:12])
    st.caption(basis_set.description)

    rows = function_rows(basis_set_id)
    id_to_name = {row["function_id"]: row["function"] for row in rows}

    records: list[BasisTaskRecord] = []
    refs = corpus_view.discover_corpora()
    matching = []
    for ref in refs:
        manifest = load_manifest_cached(str(ref.manifest_path))
        if manifest["basis_set_id"] == basis_set_id:
            matching.append((ref, manifest))

    if not matching:
        st.info(f"No local corpus is bound to `{basis_set_id}`.")
    else:
        names = [ref.name for ref, _ in matching]
        chosen = st.selectbox("Corpus", names, key="basis_corpus")
        ref, manifest = matching[names.index(chosen)]
        missing = corpus_view.missing_splits(ref, manifest)
        if missing:
            st.warning(
                f"Payload missing for {', '.join(missing)} — fetch it on the "
                "Corpus tab to see usage counts."
            )
        else:
            splits = st.multiselect(
                "Splits",
                list(manifest["splits"]),
                default=["test"] if "test" in manifest["splits"] else None,
                key="basis_splits",
            )
            limit = st.slider(
                "Records per split", 200, STATS_LIMIT, 1000, step=200, key="basis_limit"
            )
            if splits:
                with st.spinner("Reading corpus…"):
                    records = _corpus_records(ref, manifest, splits, limit)
                st.caption(f"{len(records)} tasks loaded.")

    table = (
        _attach_corpus_stats(rows, records, id_to_name)
        if records
        else pd.DataFrame(rows)
    )
    search = st.text_input("Filter functions", key="basis_search").strip()
    if search:
        table = table[table["function"].str.contains(search, case=False)]
    st.dataframe(
        table.drop(columns=["code", "function_id"]),
        width="stretch",
        hide_index=True,
        height=320,
    )

    if table.empty:
        return
    st.divider()
    selected = st.selectbox(
        "Inspect function", list(table["function"]), key="basis_function"
    )
    _render_function_detail(selected, rows, records)
