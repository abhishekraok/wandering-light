from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.shortest_path_data import (
    apply_solver_candidate,
    bounded_relabel,
    certified_specs,
    read_jsonl_gz,
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
