"""Graph expansion and its DOT rendering."""

import re

import pytest

from wandering_light.common_functions import basic_fns
from wandering_light.evals import graph_view
from wandering_light.function_def import FunctionDefSet
from wandering_light.typed_list import TypedList


@pytest.fixture
def int_fns() -> FunctionDefSet:
    return FunctionDefSet(
        [basic_fns.name_to_function[name] for name in ("inc", "double", "neg")]
    )


@pytest.fixture
def view(int_fns) -> graph_view.GraphExpansion:
    return graph_view.expand_from(
        TypedList([1, 2, 3], item_type=int), int_fns, max_depth=2
    )


def test_expand_from_certifies_depth_when_complete(view):
    stats = graph_view.expansion_stats(view)
    assert stats["complete"] is True
    assert stats["certified_depth"] == 2
    assert view.depths[view.root_id] == 0
    assert stats["nodes"] > 1


def test_expand_from_records_the_budget_that_stopped_it(int_fns):
    capped = graph_view.expand_from(
        TypedList([1, 2, 3], item_type=int), int_fns, max_depth=3, max_states=4
    )
    stats = graph_view.expansion_stats(capped)
    assert stats["complete"] is False
    assert stats["stop_reason"] == "max_states"
    assert stats["certified_depth"] < 3


def test_path_edges_returns_the_shortest_route(view):
    inc = next(fn for fn in view.graph.functions if fn.name == "inc")
    target = view.graph.find(TypedList([3, 4, 5], item_type=int))
    edges = [fn.name for _p, fn, _c in graph_view.path_edges(view, target)]
    assert edges == ["inc", "inc"]
    assert graph_view.path_edges(view, view.root_id) == []
    assert view.graph.apply(view.root_id, inc) in view.depths


def test_to_dot_draws_visible_nodes_and_labels_edges(view):
    dot = graph_view.to_dot(view)
    assert dot.startswith("digraph trajectory {")
    assert dot.rstrip().endswith("}")
    assert '[label="inc"' in dot
    assert "TL<int>" in dot


def test_to_dot_caps_node_count(view):
    dot = graph_view.to_dot(view, max_nodes=3)
    drawn = [line for line in dot.splitlines() if "fillcolor=" in line]
    assert len(drawn) == 3
    # Only edges between drawn nodes survive.
    drawn_ids = {int(line.strip().split(" ")[0]) for line in drawn}
    for line in dot.splitlines():
        if "->" in line:
            source, target = line.strip().split(" -> ")
            assert int(source) in drawn_ids
            assert int(target.split(" ")[0]) in drawn_ids


def test_to_dot_keeps_the_highlighted_path_past_the_cap(view):
    target = view.graph.find(TypedList([3, 4, 5], item_type=int))
    highlight = graph_view.path_edges(view, target)
    dot = graph_view.to_dot(view, max_nodes=1, highlight=highlight)
    for parent, function, child in highlight:
        assert f"  {parent} -> {child} " in dot
        assert function.name in dot
    assert graph_view.HIGHLIGHT_EDGE in dot


def test_node_label_truncates_long_states():
    long_value = TypedList(list(range(50)), item_type=int)
    label = graph_view.node_label(long_value, max_chars=20)
    assert len(label) == 20
    assert label.endswith("…")


def test_to_dot_escapes_quotes_but_keeps_the_line_break(int_fns):
    quoted = graph_view.expand_from(
        TypedList(['a"b'], item_type=str),
        FunctionDefSet([basic_fns.name_to_function["upper"]]),
        max_depth=1,
    )
    dot = graph_view.to_dot(quoted)
    labels = re.findall(r'label="((?:[^"\\]|\\.)*)"', dot)
    # A quote inside the state is escaped; the label's own newline is not.
    assert any('\\"' in label for label in labels)
    assert all(label.count("\\n") == 1 for label in labels if label.startswith("#"))
    assert "\\\\n" not in dot
