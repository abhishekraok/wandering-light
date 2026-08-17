from pathlib import Path

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.shortest_path_data import (
    RecertificationStatus,
    apply_solver_candidate,
    bounded_relabel,
    certified_specs,
    read_jsonl_gz,
    recertify_distance,
    recertify_shortest_record,
    recertify_shortest_records,
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

DATA_ROOT = Path(__file__).parents[1] / "wandering_light"


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


def test_recertification_distinguishes_certified_inflated_and_inconclusive():
    functions = FunctionDefSet([inc, add_two])

    certified = recertify_distance(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=FunctionDefSet([inc]),
    )
    inflated = recertify_distance(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=functions,
    )
    inconclusive = recertify_distance(
        TypedList([0]),
        TypedList([2]),
        2,
        available_functions=functions,
        max_transitions=0,
    )

    assert certified.status is RecertificationStatus.CERTIFIED
    assert inflated.status is RecertificationStatus.INFLATED
    assert inflated.shorter_path_length == 1
    assert inconclusive.status is RecertificationStatus.INCONCLUSIVE
    assert inconclusive.stop_reason == "max_transitions"
    assert not inconclusive.complete_expansion


def test_recertification_matches_targets_with_answer_equality():
    negative_zero = FunctionDef(
        name="negative_zero",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return -0.0",
    )

    result = recertify_distance(
        TypedList([1.0]),
        TypedList([0.0]),
        2,
        available_functions=FunctionDefSet([negative_zero]),
    )

    assert result.status is RecertificationStatus.INFLATED
    assert result.shorter_path_length == 1


def test_shortest_record_recertification_validates_its_witness():
    record = {
        "certified": True,
        "input": TypedList([0]).to_string(),
        "output": TypedList([2]).to_string(),
        "relabeled_functions": ["inc", "inc"],
        "relabeled_length": 2,
    }

    result = recertify_shortest_record(
        record, available_functions=FunctionDefSet([inc])
    )

    assert result.status is RecertificationStatus.CERTIFIED


def test_all_committed_random_input_distances_recertify():
    records = read_jsonl_gz(
        DATA_ROOT / "evals/data/random_inputs_500_shortest_v1.jsonl.gz"
    )

    results = recertify_shortest_records(records)
    failures = [
        (record["source_index"], result)
        for record, result in zip(records, results, strict=True)
        if result.status is not RecertificationStatus.CERTIFIED
    ]

    assert len(results) == 480
    assert failures == []
