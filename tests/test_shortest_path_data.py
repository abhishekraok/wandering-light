from pathlib import Path

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.shortest_path_data import (
    apply_solver_candidate,
    bounded_relabel,
    certified_specs,
    read_jsonl_gz,
    recertify_shortest_path,
    recertify_shortest_v1_record,
    write_jsonl_gz,
)
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList

inc = FunctionDef(
    name="inc",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 1",
)
add_two = FunctionDef(
    name="add_two",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 2",
)
identity = FunctionDef(
    name="identity",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x",
)
to_negative_zero = FunctionDef(
    name="to_negative_zero",
    input_type="builtins.float",
    output_type="builtins.float",
    code="return -0.0",
)
absolute = FunctionDef(
    name="absolute",
    input_type="builtins.float",
    output_type="builtins.float",
    code="return abs(x)",
)


def _spec(functions):
    return TrajectorySpec(TypedList([0]), FunctionDefList(functions))


def test_bfs_replaces_random_walk_with_global_shortcut():
    record = bounded_relabel(
        _spec([inc, inc]),
        0,
        split="test",
        available_functions=FunctionDefSet([inc, add_two]),
    )

    assert record["certified"]
    assert record["method"] == "bfs"
    assert record["relabeled_functions"] == ["add_two"]
    assert record["lower_bound"] == record["upper_bound"] == 1


def test_bounded_relabel_reads_distance_in_answer_equality_space():
    record = bounded_relabel(
        TrajectorySpec(
            TypedList([1.0]),
            FunctionDefList([to_negative_zero, absolute]),
        ),
        0,
        split="test",
        available_functions=FunctionDefSet([to_negative_zero, absolute]),
    )

    assert record["certified"]
    assert record["method"] == "bfs"
    assert record["relabeled_functions"] == ["to_negative_zero"]
    assert record["relabeled_length"] == 1


def test_original_upper_bound_certifies_length_four_after_depth_three():
    record = bounded_relabel(
        _spec([inc] * 4),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )

    assert record["certified"]
    assert record["method"] == "original_after_bfs"
    assert record["lower_bound"] == record["upper_bound"] == 4


def test_length_four_subsequence_certifies_length_five_proposal():
    record = bounded_relabel(
        _spec([inc, inc, identity, inc, inc]),
        0,
        split="test",
        available_functions=FunctionDefSet([inc, identity]),
    )

    assert record["certified"]
    assert record["method"] == "subsequence_after_bfs"
    assert record["relabeled_functions"] == ["inc"] * 4


def test_length_five_without_length_four_witness_remains_bracketed():
    record = bounded_relabel(
        _spec([inc] * 5),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )

    assert not record["certified"]
    assert (record["lower_bound"], record["upper_bound"]) == (4, 5)


def test_valid_solver_candidate_tightens_bound_and_certifies():
    record = bounded_relabel(
        _spec([inc] * 5),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )
    record["output"] = TypedList([4]).to_string()

    updated = apply_solver_candidate(
        record,
        FunctionDefList([inc] * 4),
        available_functions=FunctionDefSet([inc]),
        budget=8,
    )

    assert updated["certified"]
    assert updated["method"] == "rl_after_bfs"
    assert updated["rl_budget"] == 8


def test_identity_is_certified_but_excluded_from_dataset():
    record = bounded_relabel(
        _spec([identity]),
        0,
        split="test",
        available_functions=FunctionDefSet([identity]),
    )

    assert record["certified"] and record["relabeled_length"] == 0
    assert (
        len(certified_specs([record], available_functions=FunctionDefSet([identity])))
        == 0
    )


def test_compressed_records_are_deterministic_and_round_trip(tmp_path):
    records = [{"certified": True, "source_index": 3}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    write_jsonl_gz(records, first)
    write_jsonl_gz(records, second)

    assert first.read_bytes() == second.read_bytes()
    assert read_jsonl_gz(first) == records


def test_recertification_distinguishes_pass_inflation_and_inconclusive():
    functions = FunctionDefSet([inc, add_two])
    passed = recertify_shortest_path(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=FunctionDefSet([inc]),
    )
    inflated = recertify_shortest_path(
        TypedList([0]), TypedList([2]), 2, available_functions=functions
    )
    inconclusive = recertify_shortest_path(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=FunctionDefSet([inc]),
        max_transitions=0,
    )
    exact_budget = recertify_shortest_path(
        TypedList([0]),
        TypedList([99]),
        2,
        available_functions=FunctionDefSet([inc]),
        max_transitions=1,
    )

    assert passed.outcome == "certified" and passed.ok
    assert inflated.outcome == "inflated" and inflated.shorter_distance == 1
    assert inconclusive.outcome == "inconclusive"
    assert inconclusive.stop_reason == "max_transitions"
    assert not inconclusive.complete_expansion
    assert exact_budget.outcome == "certified"
    assert exact_budget.complete_expansion


def test_partial_recertification_can_still_prove_inflation():
    result = recertify_shortest_path(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=FunctionDefSet([add_two, inc]),
        max_transitions=1,
    )

    assert result.outcome == "inflated"
    assert result.shorter_distance == 1
    assert not result.complete_expansion


def test_recertification_target_uses_answer_equality_not_search_identity():
    result = recertify_shortest_path(
        TypedList([1.0]),
        TypedList([0.0]),
        2,
        available_functions=FunctionDefSet([to_negative_zero, absolute]),
    )

    assert result.outcome == "inflated"
    assert result.shorter_distance == 1


def test_shortest_v1_recertification_checks_stored_witness():
    record = bounded_relabel(
        _spec([inc, inc]),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )
    assert recertify_shortest_v1_record(
        record, available_functions=FunctionDefSet([inc])
    ).ok

    record["output"] = TypedList([99]).to_string()
    with pytest.raises(ValueError, match="stored witness"):
        recertify_shortest_v1_record(record, available_functions=FunctionDefSet([inc]))


def test_committed_eval_shortest_paths_recertify():
    path = (
        Path(__file__).parents[1]
        / "wandering_light/evals/data/random_inputs_500_shortest_v1.jsonl.gz"
    )
    records = read_jsonl_gz(path)
    results = [recertify_shortest_v1_record(record) for record in records]
    bad = [
        (record["source_index"], result.outcome, result.stop_reason)
        for record, result in zip(records, results, strict=True)
        if not result.ok
    ]

    assert len(records) == 480
    assert not bad
