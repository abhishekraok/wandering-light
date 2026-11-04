from pathlib import Path

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.shortest_path_data import (
    ShortestPathRecertification,
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


to_neg_zero = FunctionDef(
    name="to_neg_zero",
    input_type="builtins.float",
    output_type="builtins.float",
    code="return -0.0 * x",
)
absolute = FunctionDef(
    name="absolute",
    input_type="builtins.float",
    output_type="builtins.float",
    code="return abs(x)",
)
SIGNED_ZERO_FNS = FunctionDefSet([to_neg_zero, absolute])

SHORTEST_V1 = (
    Path(__file__).resolve().parents[1]
    / "wandering_light/evals/data/random_inputs_500_shortest_v1.jsonl.gz"
)


def test_target_reached_through_signed_zero_relabels_to_length_one():
    spec = TrajectorySpec(TypedList([1.0]), FunctionDefList([to_neg_zero, absolute]))

    record = bounded_relabel(spec, 0, split="test", available_functions=SIGNED_ZERO_FNS)

    assert record["output"] == TypedList([0.0]).to_string()
    assert record["certified"]
    assert record["method"] == "bfs"
    assert record["relabeled_functions"] == ["to_neg_zero"]
    assert record["relabeled_length"] == 1


def test_recertify_certifies_true_distance():
    result = recertify_shortest_path(
        TypedList([0]),
        TypedList([3]),
        3,
        available_functions=FunctionDefSet([inc]),
    )

    assert isinstance(result, ShortestPathRecertification)
    assert result.ok and result.outcome == "certified"
    assert result.shorter_distance is None
    assert result.search_depth == result.certified_search_depth == 2
    assert result.complete_expansion and result.stop_reason is None
    assert result.states_searched == 3
    assert result.transitions_attempted == 2


def test_recertify_flags_inflated_distance():
    result = recertify_shortest_path(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=FunctionDefSet([inc, add_two]),
    )

    assert not result.ok
    assert result.outcome == "inflated"
    assert result.shorter_distance == 1


def test_recertify_judges_target_by_answer_equality():
    result = recertify_shortest_path(
        TypedList([1.0]),
        TypedList([0.0]),
        2,
        available_functions=SIGNED_ZERO_FNS,
    )

    assert result.outcome == "inflated"
    assert result.shorter_distance == 1


def test_recertify_input_equal_target_with_positive_distance_is_inflated():
    result = recertify_shortest_path(
        TypedList([0.0]),
        TypedList([-0.0]),
        1,
        available_functions=SIGNED_ZERO_FNS,
    )

    assert result.outcome == "inflated"
    assert result.shorter_distance == 0


def test_recertify_zero_distance_identity_is_certified():
    result = recertify_shortest_path(
        TypedList([5]), TypedList([5]), 0, available_functions=FunctionDefSet([inc])
    )

    assert result.ok
    assert result.search_depth == 0
    assert result.transitions_attempted == 0


def test_recertify_budget_stop_is_inconclusive():
    result = recertify_shortest_path(
        TypedList([0]),
        TypedList([4]),
        4,
        available_functions=FunctionDefSet([inc, identity]),
        max_transitions=3,
    )

    assert result.outcome == "inconclusive"
    assert not result.ok
    assert not result.complete_expansion
    assert result.stop_reason == "max_transitions"
    assert result.shorter_distance is None
    assert result.certified_search_depth < result.search_depth == 3


def test_recertify_partial_expansion_still_detects_inflation():
    result = recertify_shortest_path(
        TypedList([0]),
        TypedList([2]),
        4,
        available_functions=FunctionDefSet([inc, add_two]),
        max_transitions=2,
    )

    assert not result.complete_expansion
    assert result.outcome == "inflated"
    assert result.shorter_distance == 1


@pytest.mark.parametrize("distance", [-1, 1.0, "2", True])
def test_recertify_rejects_bad_distance(distance):
    with pytest.raises(ValueError):
        recertify_shortest_path(TypedList([0]), TypedList([1]), distance)


def test_recertify_rejects_zero_distance_to_different_target():
    with pytest.raises(ValueError):
        recertify_shortest_path(TypedList([0]), TypedList([1]), 0)


def _certified_record():
    return bounded_relabel(
        _spec([inc, inc]),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )


def test_recertify_record_round_trip():
    record = _certified_record()

    result = recertify_shortest_v1_record(
        record, available_functions=FunctionDefSet([inc])
    )

    assert result.ok
    assert result.recorded_distance == 2


@pytest.mark.parametrize(
    "corruption",
    [
        {"certified": False},
        {"output": None},
        {"relabeled_length": 3},
        {"relabeled_functions": ["inc"]},
        {"schema_version": 2},
    ],
)
def test_recertify_record_rejects_malformed_records(corruption):
    record = {**_certified_record(), **corruption}

    with pytest.raises(ValueError):
        recertify_shortest_v1_record(record, available_functions=FunctionDefSet([inc]))


def test_recertify_record_rejects_witness_that_misses_output():
    record = {**_certified_record(), "output": TypedList([5]).to_string()}

    with pytest.raises(
        ValueError, match="stored witness does not reproduce the recorded output"
    ):
        recertify_shortest_v1_record(record, available_functions=FunctionDefSet([inc]))


def test_recertify_record_flags_inflated_record():
    record = {
        **_certified_record(),
        "relabeled_functions": ["inc", "inc"],
        "relabeled_length": 2,
    }

    result = recertify_shortest_v1_record(
        record, available_functions=FunctionDefSet([inc, add_two])
    )

    assert result.outcome == "inflated"
    assert result.shorter_distance == 1


def test_committed_shortest_v1_dataset_recertifies():
    records = read_jsonl_gz(SHORTEST_V1)
    assert records

    failures = []
    for record in records:
        result = recertify_shortest_v1_record(record)
        if not result.ok:
            failures.append((record["source_index"], result))

    assert failures == []
