"""Focused AppTest coverage for the lazy Corpus workspace."""

import gzip
import json
import os

from streamlit.testing.v1 import AppTest

from wandering_light.typed_list import TypedList

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPLORER_PATH = os.path.join(REPO_ROOT, "wandering_light/evals/explorer.py")


def _legacy_row(index: int, functions: list[str], output: int) -> dict:
    return {
        "schema_version": 1,
        "split": "eval",
        "source_index": index,
        "input": TypedList([index], item_type=int).to_string(),
        "output": TypedList([output], item_type=int).to_string(),
        "original_functions": functions,
        "relabeled_functions": functions,
        "original_length": len(functions),
        "relabeled_length": len(functions),
        "lower_bound": len(functions),
        "upper_bound": len(functions),
        "certified": True,
        "method": "bfs",
    }


def _write_tiny_corpus(path) -> None:
    rows = [
        _legacy_row(1, ["inc"], 2),
        _legacy_row(2, ["inc", "double"], 6),
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_corpus_page_loads_edits_solves_and_graphs(tmp_path, monkeypatch):
    corpus_path = tmp_path / "tiny.jsonl.gz"
    _write_tiny_corpus(corpus_path)

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("WANDERING_LIGHT_CORPUS_PATHS", str(corpus_path))
    monkeypatch.setenv("WANDERING_LIGHT_CORPUS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    at = AppTest.from_file(EXPLORER_PATH, default_timeout=120).run()
    next(button for button in at.button if button.key == "corpus_load").click().run()

    assert not at.exception
    assert "Task workspace" in [header.value for header in at.subheader]
    assert any("reaches the target in 2 steps" in item.value for item in at.success)
    basis_filter = next(
        widget for widget in at.multiselect if widget.label == "Basis functions"
    )
    assert any(option.startswith("inc ·") for option in basis_filter.options)

    stored_edges = [
        widget
        for widget in at.selectbox
        if (widget.key or "").startswith("corpus_edit_") and "_edge_sel_" in widget.key
    ]
    edge = stored_edges[0]
    alternative = next(option for option in edge.options if option != edge.value)
    edge.set_value(alternative)
    namespace, _, node_id = edge.key.partition("_edge_sel_")
    next(
        button
        for button in at.button
        if button.key == f"{namespace}_edge_btn_{node_id}"
    ).click().run()
    assert any("does not reach the target" in item.value for item in at.warning)

    next(
        button for button in at.button if button.key == "corpus_reset_task"
    ).click().run()
    assert not at.exception
    assert any("reaches the target in 2 steps" in item.value for item in at.success)

    next(
        widget
        for widget in at.number_input
        if widget.label == "Transition / attempt budget"
    ).set_value(100)
    next(
        widget
        for widget in at.number_input
        if widget.label == "Maximum depth / path length"
    ).set_value(2)
    next(
        widget for widget in at.multiselect if widget.label == "Execution palette"
    ).set_value(["inc", "double"])
    next(button for button in at.button if button.label == "Run solver").click().run()

    assert not at.exception
    assert any("BFS solved the task" in item.value for item in at.success)

    result_edges = [
        widget
        for widget in at.selectbox
        if (widget.key or "").startswith("corpus_solver_")
        and "_edge_sel_" in widget.key
    ]
    if result_edges:
        edge = result_edges[0]
        alternative = next(option for option in edge.options if option != edge.value)
        edge.set_value(alternative).run()
        next(
            button for button in at.button if button.label == "Run solver"
        ).click().run()

        result = at.session_state["corpus_solver_result"]
        tree = result["tree"]
        generation = result["generation"]
        editor_generation = at.session_state["corpus_editor_generation"]
        widget_values = {
            widget.key: widget.value
            for widget in at.selectbox
            if (widget.key or "").startswith(
                f"corpus_solver_{editor_generation}_{generation}_"
            )
        }
        for node_id, node in tree.nodes.items():
            applied = node.get("applied_fn_def")
            if applied is not None:
                assert (
                    widget_values[
                        f"corpus_solver_{editor_generation}_{generation}_edge_sel_{node_id}"
                    ]
                    == applied.name
                )

    next(
        button
        for button in at.button
        if button.key == "corpus_build_workspace_graph"
    ).click().run()
    assert any(
        "Merged edited tree + latest solver path" in item.value for item in at.caption
    )

    graph_source = next(
        radio for radio in at.radio if radio.label == "Graph source"
    )
    graph_source.set_value("Stored corpus witnesses").run()
    next(
        button for button in at.button if button.key == "corpus_build_graph"
    ).click().run()

    assert not at.exception
    assert any("Replayed 1 witnesses" in item.value for item in at.caption)

    graph_source = next(
        radio for radio in at.radio if radio.label == "Graph source"
    )
    graph_source.set_value("Bounded local expansion").run()
    next(
        widget for widget in at.multiselect if widget.label == "Expansion palette"
    ).set_value(["inc", "double"])
    next(
        button for button in at.button if button.key == "corpus_build_expansion"
    ).click().run()

    assert not at.exception
    assert any(metric.label == "Candidate tasks" for metric in at.metric)
    assert any("complete through requested depth" in item.value for item in at.caption)

    next(
        widget
        for widget in at.number_input
        if widget.label == "Max rendered nodes"
    ).set_value(10).run()
    assert not any(metric.label == "Candidate tasks" for metric in at.metric)

    bad_input = json.dumps({"type": "attacker.Payload", "items": []})
    next(area for area in at.text_area if area.key == "corpus_custom_input").set_value(
        bad_input
    )
    next(
        button
        for button in at.button
        if button.label == "Apply custom I/O to stored witness"
    ).click().run()

    assert any("unsupported basis-task item type" in item.value for item in at.error)

    oversized_range = json.dumps(
        {
            "type": "builtins.range",
            "items": [{"__range__": [0, 1_000_000_000, 1]}],
        }
    )
    next(area for area in at.text_area if area.key == "corpus_custom_input").set_value(
        oversized_range
    )
    next(
        button
        for button in at.button
        if button.label == "Apply custom I/O to stored witness"
    ).click().run()

    assert any("range expands" in item.value for item in at.error)

    next(button for button in at.button if button.key == "corpus_load").click().run()

    assert not at.exception
    assert (
        next(
            area for area in at.text_area if area.key == "corpus_custom_input"
        ).value
        != oversized_range
    )
    assert "corpus_solver_result" not in at.session_state
    assert "corpus_graph_projection" not in at.session_state
    assert "corpus_workspace_projection" not in at.session_state
    assert "corpus_local_expansion" not in at.session_state


def test_corpus_filter_and_custom_workspace_survive_page_navigation(
    tmp_path, monkeypatch
):
    corpus_path = tmp_path / "tiny.jsonl.gz"
    _write_tiny_corpus(corpus_path)
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("WANDERING_LIGHT_CORPUS_PATHS", str(corpus_path))
    monkeypatch.setenv("WANDERING_LIGHT_CORPUS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    at = AppTest.from_file(EXPLORER_PATH, default_timeout=120).run()
    next(button for button in at.button if button.key == "corpus_load").click().run()

    distance = next(slider for slider in at.slider if slider.label == "Distance")
    distance.set_value((1, 1)).run()
    assert (
        next(metric for metric in at.metric if metric.label == "Distance").value == "1"
    )

    custom_input = TypedList([3], item_type=int).to_string()
    custom_output = TypedList([4], item_type=int).to_string()
    next(area for area in at.text_area if area.key == "corpus_custom_input").set_value(
        custom_input
    )
    next(area for area in at.text_area if area.key == "corpus_custom_output").set_value(
        custom_output
    )
    next(
        button
        for button in at.button
        if button.label == "Apply custom I/O to stored witness"
    ).click().run()

    view = next(radio for radio in at.radio if radio.key == "explorer_view")
    view.set_value("Eval file").run()
    assert not at.exception
    assert any("PYTHONHASHSEED" in item.value for item in at.error)

    view = next(radio for radio in at.radio if radio.key == "explorer_view")
    view.set_value("Corpus").run()

    assert not at.exception
    assert next(slider for slider in at.slider if slider.label == "Distance").value == (
        1,
        1,
    )
    assert (
        next(metric for metric in at.metric if metric.label == "Distance").value == "1"
    )
    assert (
        next(area for area in at.text_area if area.key == "corpus_custom_input").value
        == custom_input
    )
    assert (
        next(area for area in at.text_area if area.key == "corpus_custom_output").value
        == custom_output
    )


def test_single_distance_corpus_uses_a_fixed_filter(tmp_path, monkeypatch):
    corpus_path = tmp_path / "one.jsonl.gz"
    with gzip.open(corpus_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_legacy_row(1, ["inc"], 2)) + "\n")
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("WANDERING_LIGHT_CORPUS_PATHS", str(corpus_path))
    monkeypatch.setenv("WANDERING_LIGHT_CORPUS_CACHE", str(tmp_path / "cache"))

    at = AppTest.from_file(EXPLORER_PATH, default_timeout=120).run()
    next(button for button in at.button if button.key == "corpus_load").click().run()

    assert not at.exception
    assert not any(slider.label == "Distance" for slider in at.slider)
    assert [metric.value for metric in at.metric if metric.label == "Distance"] == [
        "1",
        "1",
    ]
