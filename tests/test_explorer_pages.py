"""AppTest coverage for the explorer's corpus, playground and graph pages.

The pages are mostly layout; what is worth pinning down is that they render at
all, that a corpus task reaches the playground intact, and that expanding a
graph produces something drawable.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

from wandering_light.basis_dataset import typed_list_from_builtin_str
from wandering_light.evals import corpus_view, graph_view
from wandering_light.evals.explorer_playground import parse_typed_list

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPLORER_PATH = os.path.join(REPO_ROOT, "wandering_light/evals/explorer.py")


@pytest.fixture
def app(monkeypatch):
    if os.environ.get("PYTHONHASHSEED") != "0":
        pytest.skip("requires interpreter startup with PYTHONHASHSEED=0")
    monkeypatch.chdir(REPO_ROOT)
    at = AppTest.from_file(EXPLORER_PATH, default_timeout=300)
    at.run()
    assert [e.value for e in at.exception] == []
    return at


def _keys(elements):
    return {element.key for element in elements}


def _corpus_payload_present() -> bool:
    refs = corpus_view.discover_corpora()
    return any(
        not corpus_view.missing_splits(ref, corpus_view.load_manifest(ref))
        for ref in refs
    )


def test_every_tab_renders(app):
    labels = {metric.label for metric in app.metric}
    assert {"Tasks", "Roots", "Functions"} <= labels
    assert {"corpus_name", "basis_id", "play_basis", "graph_basis"} <= _keys(
        app.selectbox
    )


def test_playground_solves_its_default_task(app):
    app.button(key="play_run").click().run()
    assert [e.value for e in app.exception] == []
    result = app.session_state["play_solver_result"]
    assert result["success"], result
    # double then inc maps [1, 2, 3] to [3, 5, 7]; any path scoring equal counts.
    assert len(result["functions"]) <= 2


def test_graph_expansion_produces_a_drawable_dot(app):
    app.button(key="graph_expand").click().run()
    assert [e.value for e in app.exception] == []
    view = app.session_state["graph_view"]
    stats = graph_view.expansion_stats(view)
    assert stats["nodes"] > 1
    assert stats["certified_depth"] == 2
    assert "digraph trajectory {" in graph_view.to_dot(view)


def test_graph_root_button_rewrites_the_widget_value(app):
    """Writing a widget's key after it exists raises; the button uses a callback."""
    if not _corpus_payload_present():
        pytest.skip("corpus payload not downloaded")
    assert app.button(key="graph_use_corpus").proto.disabled is True

    app.button(key="corpus_send_playground").click().run()
    app.button(key="graph_use_corpus").click().run()
    assert [e.value for e in app.exception] == []

    task = app.session_state["explorer_selected_task"]
    expected = typed_list_from_builtin_str(task["input"])
    assert parse_typed_list(app.session_state["graph_root"]) == expected


def test_graph_tab_reports_an_unparseable_root(app):
    app.session_state["graph_root"] = "not a typed list"
    app.run()
    assert [e.value for e in app.exception] == []
    assert any("Could not parse root" in error.value for error in app.error)


def _edge_widget_violations(at) -> list[str]:
    """Playground edge selectboxes that disagree with the tree they edit."""
    tree = at.session_state["tree_play"]
    values = {
        selectbox.key: selectbox.value
        for selectbox in at.selectbox
        if (selectbox.key or "").startswith("play_") and "_edge_sel_" in selectbox.key
    }
    return [
        f"{key}={values[key]!r} but tree[{node_id}]={node['applied_fn_def'].name!r}"
        for node_id, node in tree.nodes.items()
        if node["applied_fn_def"] is not None
        and (key := f"play_edge_sel_{node_id}") in values
        and values[key] != node["applied_fn_def"].name
    ]


def test_playground_widgets_track_the_tree_across_tasks(app):
    """Switching tasks must not leave a previous task's edges on screen."""
    if not _corpus_payload_present():
        pytest.skip("corpus payload not downloaded")
    app.button(key="corpus_send_playground").click().run()
    assert _edge_widget_violations(app) == []

    app.selectbox(key="corpus_task_idx").set_value(7).run()
    app.button(key="corpus_send_playground").click().run()
    assert [e.value for e in app.exception] == []
    assert _edge_widget_violations(app) == []


def test_corpus_task_reaches_the_playground(app):
    if not _corpus_payload_present():
        pytest.skip("corpus payload not downloaded")
    app.button(key="corpus_send_playground").click().run()
    assert [e.value for e in app.exception] == []

    task = app.session_state["explorer_selected_task"]
    assert task["witness"]
    assert app.session_state["play_names"] == task["witness"]
    # The trajectory is seeded from the witness, so it already hits the target.
    assert any("Reaches the target" in message.value for message in app.success), [
        m.value for m in app.success
    ]
