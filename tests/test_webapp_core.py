"""The explorer's domain layer: states, successors, expansion, solving."""

import pytest

from wandering_light.basis_set import load_basis_set
from wandering_light.typed_list import TypedList
from wandering_light.webapp import core


@pytest.fixture(scope="module")
def functions():
    return load_basis_set("wl-core-v1").as_function_set()


@pytest.fixture(scope="module")
def ints(functions):
    return core.palette(functions, ["inc", "dec", "double", "half", "neg", "square"])


def test_state_view_carries_every_form_the_ui_needs():
    view = core.state_view(TypedList([1, 2, 3], item_type=int))
    assert view.label == "TL<int>([1, 2, 3])"
    assert view.type == "int"
    assert view.size == 3
    assert view.wire.startswith('{"type": "builtins.int"')
    assert len(view.id) == 16


def test_state_view_ids_follow_the_value_not_the_object():
    first = core.state_view(TypedList([1, 2], item_type=int))
    second = core.state_view(TypedList([1, 2], item_type=int))
    other = core.state_view(TypedList([2, 1], item_type=int))
    assert first.id == second.id
    assert first.id != other.id


def test_parse_state_accepts_both_text_forms():
    from_repr = core.parse_state("TL<int>([1, 2, 3])")
    from_wire = core.parse_state(from_repr.to_string())
    assert from_repr == from_wire


def test_run_trajectory_reports_each_intermediate_state(ints):
    steps = core.run_trajectory(TypedList([1, 2, 3], int), ["double", "inc"], ints)
    assert [step.function for step in steps] == ["double", "inc"]
    assert all(step.ok for step in steps)
    assert steps[-1].state.label == "TL<int>([3, 5, 7])"


def test_run_trajectory_stops_at_the_first_failure(functions):
    steps = core.run_trajectory(
        TypedList([1, 2, 3], int), ["int_to_str", "inc", "double"], functions
    )
    assert [step.ok for step in steps] == [True, False]
    # `inc` is an int function, so it never applies to the strings before it.
    assert "not in this basis" not in (steps[1].error or "")
    assert len(steps) == 2


def test_run_trajectory_rejects_a_function_outside_the_basis(ints):
    steps = core.run_trajectory(TypedList([1, 2], int), ["nonesuch"], ints)
    assert steps[0].ok is False
    assert steps[0].error == "not in this basis"


def test_successors_keep_failures_and_self_loops(functions):
    results = core.successors(TypedList([0, 1, 0, 1], item_type=int), functions)
    by_name = {item.function: item for item in results}
    assert by_name["mod2"].ok and by_name["mod2"].self_loop
    assert by_name["inc"].ok and not by_name["inc"].self_loop
    # Only type-compatible functions are considered at all.
    assert "upper" not in by_name


def test_successors_of_a_dead_end_type(functions):
    results = core.successors(TypedList([1, 2], item_type=int), functions)
    assert results, "int states have successors"
    assert all(item.function in {f.name for f in functions} for item in results)


def test_expand_lays_out_reachable_states_by_depth(ints):
    view = core.expand(TypedList([1, 2, 3], int), ints, max_depth=2)
    assert view.stats["complete"] is True
    assert view.stats["certified_depth"] == 2
    assert view.stats["by_depth"]["0"] == 1
    assert {node["depth"] for node in view.nodes} <= {0, 1, 2}
    # Every edge joins two nodes that are actually drawn.
    drawn = {node["node_id"] for node in view.nodes}
    assert all(e["source"] in drawn and e["target"] in drawn for e in view.edges)


def test_expand_reports_which_functions_never_fired(functions):
    palette = core.palette(functions, ["inc", "upper"])
    view = core.expand(TypedList([1, 2, 3], int), palette, max_depth=1)
    assert view.function_edges == {"inc": 1}
    assert view.idle_functions == ["upper"]
    assert view.types == {"int": 2}


def test_expand_records_the_budget_that_stopped_it(ints):
    view = core.expand(TypedList([1, 2, 3], int), ints, max_depth=4, max_states=5)
    assert view.stats["complete"] is False
    assert view.stats["stop_reason"] == "max_states"
    assert view.stats["certified_depth"] < 4


def test_solve_finds_a_path_and_times_it(ints):
    attempt = core.solve(
        TypedList([1, 2, 3], int),
        TypedList([3, 5, 7], int),
        ints,
        max_depth=2,
    )
    assert attempt.success
    assert attempt.functions == ["double", "inc"]
    assert attempt.output.label == "TL<int>([3, 5, 7])"
    assert attempt.elapsed_seconds >= 0


def test_solve_reports_a_miss_rather_than_raising(ints):
    attempt = core.solve(
        TypedList([1, 2, 3], int),
        TypedList([999], item_type=int),
        ints,
        max_depth=1,
        budget=5,
    )
    assert attempt.success is False
    assert attempt.functions == [] or attempt.output is not None


def test_solve_rejects_an_unknown_solver(ints):
    with pytest.raises(ValueError, match="unknown solver"):
        core.solve(TypedList([1], int), TypedList([2], int), ints, solver="oracle")


def test_palette_restricts_without_copying_the_whole_basis(functions):
    assert len(core.palette(functions, ["inc", "dec"])) == 2
    assert core.palette(functions, None) is functions
