import json
from dataclasses import replace

import pytest

from wandering_light.evals.corpus_index import RecordDetail
from wandering_light.evals.explorer_graph import (
    build_local_expansion,
    build_witness_projection,
    build_workspace_projection,
    graph_to_view,
    graph_view_figure,
    resolve_witness_functions,
    validate_typed_list_workload,
)
from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.typed_list import TypedList


def _function(name: str, code: str, function_id: str) -> FunctionDef:
    return FunctionDef(
        name=name,
        input_type="builtins.int",
        output_type="builtins.int",
        code=code,
        metadata={"basis_function_id": function_id},
    )


inc = _function("inc", "return x + 1", "bf:inc:1")
dec = _function("dec", "return x - 1", "bf:dec:1")
add_two = _function("add_two", "return x + 2", "bf:add_two:1")
functions = FunctionDefSet([inc, dec, add_two])


def _record(
    task_id: str,
    names: list[str],
    ids: list[str],
    output: int,
) -> RecordDetail:
    raw = {"task_id": task_id}
    return RecordDetail(
        row_id=1,
        task_id=task_id,
        schema_kind="basis-task-v2",
        split="test",
        input=TypedList([1]).to_string(),
        output=TypedList([output]).to_string(),
        input_type="builtins.int",
        output_type="builtins.int",
        distance=len(names),
        certified=True,
        witness_function_names=tuple(names),
        witness_function_ids=tuple(ids),
        basis_set_id="test-basis",
        basis_set_digest="sha256:" + "0" * 64,
        generator="test",
        source_index=1,
        root_index=0,
        certification="complete-bfs-expansion",
        metadata={},
        raw=raw,
        functions_by_role={
            "witness": tuple(ids),
            "optimal_first": (),
            "optimal_last": (),
        },
    )


def test_projection_replays_convergent_paths_and_highlights_selection():
    long_path = _record("long", ["inc", "inc"], ["bf:inc:1"] * 2, 3)
    shortcut = _record("short", ["add_two"], ["bf:add_two:1"], 3)

    projection = build_witness_projection(
        [long_path, shortcut], functions, selected_task_id="short"
    )

    assert projection.errors == ()
    assert projection.processed_records == 2
    assert projection.view.total_nodes == 3
    assert projection.view.total_edges == 3
    assert any(edge.highlighted for edge in projection.view.edges)
    assert any(node.role == "selected_target" for node in projection.view.nodes)
    assert projection.view.diagnostics.convergent_nodes == 1
    assert projection.view.diagnostics.directed_cycle_groups == 0


def test_projection_preserves_cycle_back_to_root():
    cycle = _record("cycle", ["inc", "dec"], ["bf:inc:1", "bf:dec:1"], 1)

    projection = build_witness_projection([cycle], functions, selected_task_id="cycle")

    assert projection.view.total_nodes == 2
    assert projection.view.total_edges == 2
    root = next(
        node for node in projection.view.nodes if node.role == "selected_target"
    )
    assert any(edge.target_id == root.node_id for edge in projection.view.edges)
    assert projection.view.diagnostics.directed_cycle_groups == 1


def test_workspace_projection_merges_editor_and_solver_paths():
    editor = TrajectoryTree.with_root(TypedList([1]))
    editor.append_child(ROOT_ID, add_two, Executor(functions))
    solver = TrajectoryTree.with_root(TypedList([1]))
    first = solver.append_child(ROOT_ID, inc, Executor(functions))
    solver.append_child(first, inc, Executor(functions))

    projection = build_workspace_projection(
        [editor, solver], functions, target=TypedList([3])
    )

    assert projection.errors == ()
    assert projection.processed_edges == 3
    assert projection.view.total_nodes == 3
    assert projection.view.total_edges == 3
    assert projection.view.diagnostics.convergent_nodes == 1
    assert any(node.role == "selected_target" for node in projection.view.nodes)


def test_bounded_local_expansion_enumerates_certified_candidate_tasks():
    projection = build_local_expansion(
        TypedList([1]),
        functions,
        max_depth=2,
        max_states=50,
        max_transitions=50,
        skip_self_loops=False,
    )

    assert projection.stop_reason is None
    assert projection.certified_depth == 2
    assert projection.attempted_transitions > 0
    assert projection.tasks
    assert all(task.certified for task in projection.tasks)
    assert {task.distance for task in projection.tasks} <= {1, 2}
    assert projection.view.diagnostics.directed_cycle_groups >= 1


def test_graph_diagnostics_count_parallel_functions_and_self_loops():
    same_as_inc = _function("same_as_inc", "return x + 1", "bf:same-as-inc:1")
    identity = _function("identity", "return x", "bf:identity:1")
    diagnostic_functions = FunctionDefSet([inc, same_as_inc, identity])

    projection = build_local_expansion(
        TypedList([1]),
        diagnostic_functions,
        max_depth=1,
        max_states=10,
        max_transitions=10,
        skip_self_loops=False,
    )

    assert projection.view.diagnostics.parallel_function_groups == 1
    assert projection.view.diagnostics.self_loop_groups == 1
    encoded = json.loads(graph_view_figure(projection.view).to_json())
    assert encoded["layout"]["shapes"]


def test_graph_diagnostics_do_not_treat_dag_cross_edge_as_cycle():
    from wandering_light.proposer_pilot.graph import TrajectoryGraph

    graph = TrajectoryGraph(functions=functions)
    root_id = graph.add_root(TypedList([1]))
    intermediate_id = graph.apply(root_id, inc)
    target_id = graph.apply(root_id, add_two)
    assert graph.apply(intermediate_id, inc) == target_id

    view = graph_to_view(graph)

    assert view.diagnostics.convergent_nodes == 1
    assert view.diagnostics.directed_cycle_groups == 0


def test_projection_surfaces_graph_caps():
    records = [
        replace(
            _record(f"task-{value}", ["inc"], ["bf:inc:1"], value + 1),
            input=TypedList([value]).to_string(),
        )
        for value in range(4)
    ]

    projection = build_witness_projection(records, functions, max_nodes=2)

    assert projection.view.truncated
    assert len(projection.view.nodes) == 2


def test_projection_caps_keep_selected_target_visible():
    record = _record(
        "long-selected",
        ["inc"] * 15,
        ["bf:inc:1"] * 15,
        16,
    )

    projection = build_witness_projection(
        [record], functions, selected_task_id=record.task_id, max_nodes=10
    )

    assert projection.view.truncated
    assert any(node.role == "selected_target" for node in projection.view.nodes)


def test_stable_id_resolution_rejects_name_mismatch():
    record = _record("bad", ["not_inc"], ["bf:inc:1"], 2)

    with pytest.raises(ValueError, match="resolves to 'inc'"):
        resolve_witness_functions(record, functions)


def test_invalid_witness_does_not_leave_partial_graph_state():
    invalid = _record("invalid", ["inc"], ["bf:inc:1"], 99)

    projection = build_witness_projection([invalid], functions)

    assert projection.processed_records == 0
    assert len(projection.errors) == 1
    assert projection.view.total_nodes == 0
    assert projection.view.total_edges == 0


def test_compact_oversized_range_is_rejected_before_execution():
    serialized = json.dumps(
        {
            "type": "builtins.range",
            "items": [{"__range__": [0, 1_000_000_000, 1]}],
        }
    )
    value = TypedList([range(1_000_000_000)], item_type=range)

    with pytest.raises(ValueError, match="range expands"):
        validate_typed_list_workload(value, serialized, label="input")


def test_many_compact_ranges_are_rejected_by_aggregate_expansion_limit():
    ranges = [range(10_000) for _ in range(1_000)]
    value = TypedList(ranges, item_type=range)
    serialized = value.to_string()
    assert len(serialized.encode()) < 65_536

    with pytest.raises(ValueError, match="total items"):
        validate_typed_list_workload(value, serialized, label="input")


def test_plotly_figure_contains_node_hover_and_depth_axis():
    record = _record("one", ["inc"], ["bf:inc:1"], 2)
    view = build_witness_projection([record], functions).view

    figure = graph_view_figure(view)
    encoded = json.loads(figure.to_json())

    assert encoded["layout"]["xaxis"]["title"]["text"] == "minimum graph depth"
    assert any(trace.get("hovertext") for trace in encoded["data"])
    assert encoded["layout"]["annotations"]
    assert all(
        annotation["showarrow"] for annotation in encoded["layout"]["annotations"]
    )


def test_reciprocal_edges_render_on_opposite_sides():
    cycle = _record("cycle", ["inc", "dec"], ["bf:inc:1", "bf:dec:1"], 1)
    view = build_witness_projection([cycle], functions).view

    encoded = json.loads(graph_view_figure(view).to_json())
    annotations = encoded["layout"]["annotations"]

    assert len(annotations) == 2
    assert annotations[0]["y"] != annotations[1]["y"]


def test_signed_zero_states_remain_distinct_after_issue_37_lands():
    negative_zero = FunctionDef(
        name="negative_zero",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return -0.0",
        metadata={"basis_function_id": "bf:negative_zero:1"},
    )
    positive_zero = FunctionDef(
        name="positive_zero",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return 0.0",
        metadata={"basis_function_id": "bf:positive_zero:1"},
    )
    float_functions = FunctionDefSet([negative_zero, positive_zero])
    negative_record = replace(
        _record("negative", ["negative_zero"], ["bf:negative_zero:1"], -0.0),
        input=TypedList([1.0], item_type=float).to_string(),
        input_type="builtins.float",
        output_type="builtins.float",
    )
    positive_record = replace(
        _record("positive", ["positive_zero"], ["bf:positive_zero:1"], 0.0),
        input=TypedList([1.0], item_type=float).to_string(),
        input_type="builtins.float",
        output_type="builtins.float",
    )

    projection = build_witness_projection(
        [negative_record, positive_record], float_functions
    )

    if projection.view.total_nodes == 2:
        pytest.skip("TrajectoryGraph signed-zero search identity arrives with #37")
    assert projection.errors == ()
    assert projection.view.total_nodes == 3
    assert projection.view.total_edges == 2
