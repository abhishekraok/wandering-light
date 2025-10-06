import os
import tempfile

import pytest

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.trajectory import (
    TrajectoryList,
    TrajectorySpec,
    TrajectorySpecList,
)
from wandering_light.typed_list import TypedList


def make_function(name, input_type, output_type, code):
    return FunctionDef(
        name=name, input_type=input_type, output_type=output_type, code=code
    )


def test_empty_trajectory():
    tl = TypedList([1, 2, 3])
    available_functions = []
    executor = Executor(available_functions)
    spec = TrajectorySpec(tl, FunctionDefList([]))
    result = executor.execute_trajectory(spec)
    assert result.success is True
    traj = result.trajectory
    assert traj.input == tl
    assert traj.output == tl
    assert repr(traj) == f"Trajectory({tl} ->  -> {tl})"


def test_simple_trajectory():
    tl = TypedList([1, 2, 3])
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    available_functions = [f1]
    executor = Executor(available_functions)
    spec = TrajectorySpec(tl, FunctionDefList([f1]))
    result = executor.execute_trajectory(spec)
    assert result.success is True
    traj = result.trajectory
    expected = TypedList([2, 3, 4])
    assert traj.output == expected
    # usage count incremented
    assert f1.usage_count == 1


def test_multi_step_trajectory():
    tl = TypedList([1, 2])
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 2")
    f2 = make_function("square", "builtins.int", "builtins.int", "return x * x")
    available_functions = [f1, f2]
    executor = Executor(available_functions)
    spec = TrajectorySpec(tl, FunctionDefList([f1, f2]))
    result = executor.execute_trajectory(spec)
    assert result.success is True
    traj = result.trajectory
    # inc: [1+2,2+2] -> [3,4], square: [9,16]
    assert traj.output == TypedList([9, 16])
    assert f1.usage_count == 1
    assert f2.usage_count == 1


def test_type_mismatch_before():
    tl = TypedList([1, 2, 3])
    f1 = make_function("toStr", "builtins.str", "builtins.str", "return str(x)")
    executor = Executor([f1])
    result = executor.execute_trajectory(TrajectorySpec(tl, FunctionDefList([f1])))
    assert result.success is False
    assert "Type mismatch" in result.error_msg
    assert result.failed_at_step == 0


def test_type_mismatch_after():
    tl = TypedList([1, 2])
    # Function returns wrong type (string) but declares int output
    f = make_function("bad", "builtins.int", "builtins.int", "return str(x)")
    executor = Executor([f])
    result = executor.execute_trajectory(TrajectorySpec(tl, FunctionDefList([f])))
    assert result.success is False
    assert "Type mismatch" in result.error_msg or "failed" in result.error_msg
    assert result.failed_at_step == 0


# Tests for create_random_walk
def test_random_walk_single_fn():
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    available_functions = [f1]
    executor = Executor(available_functions)
    tl = TypedList([1, 2, 3])
    spec = TrajectorySpec.create_random_walk(
        tl, path_length=2, available_functions=available_functions
    )
    result = executor.execute_trajectory(spec)
    assert result.success is True
    traj = result.trajectory
    expected = TypedList([3, 4, 5])
    assert traj.input == tl
    assert traj.function_defs == FunctionDefList([f1, f1])
    assert traj.output == expected
    assert f1.usage_count == 2


def test_random_walk_no_candidates():
    available_functions = []
    executor = Executor(available_functions)
    tl = TypedList([1, 2])
    spec = TrajectorySpec.create_random_walk(
        tl, path_length=3, available_functions=available_functions
    )
    result = executor.execute_trajectory(spec)
    assert result.success is True
    traj = result.trajectory
    assert traj.function_defs == FunctionDefList([])
    assert traj.output == tl


def test_random_walk_two_fns():
    f1 = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
    f2 = make_function("square", "builtins.int", "builtins.int", "return x * x")
    available_functions = [f1, f2]
    executor = Executor(available_functions)
    tl = TypedList([1, 2, 3])
    spec = TrajectorySpec.create_random_walk(
        tl, path_length=3, available_functions=available_functions
    )
    result = executor.execute_trajectory(spec)
    assert result.success is True
    traj = result.trajectory
    assert len(traj.function_defs) == 3


def test_trajectory_spec_parse_from_string():
    """Test TrajectorySpec.parse_from_string method."""

    # Create some test functions
    inc_func = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    double_func = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )

    available_functions = FunctionDefSet([inc_func, double_func])

    # Test parsing a simple TrajectorySpec
    spec = TrajectorySpec(
        input_list=TypedList([1, 2, 3], item_type=int),
        function_defs=FunctionDefList([inc_func, double_func]),
    )

    # Get the string representation
    repr_str = repr(spec)  # "TrajectorySpec(TL<int>([1, 2, 3]) -> inc, double)"

    # Parse it back
    parsed_spec = TrajectorySpec.parse_from_string(repr_str, available_functions)

    # Verify it matches the original
    assert parsed_spec.input == spec.input
    assert len(parsed_spec.function_defs) == len(spec.function_defs)

    for original, parsed in zip(
        spec.function_defs, parsed_spec.function_defs, strict=False
    ):
        assert original.name == parsed.name
        assert original.input_type == parsed.input_type
        assert original.output_type == parsed.output_type


def test_trajectory_spec_parse_from_string_empty_functions():
    """Test parsing TrajectorySpec with no functions."""

    available_functions = FunctionDefSet([])

    # Create spec with no functions
    spec = TrajectorySpec(
        input_list=TypedList([1, 2, 3], item_type=int),
        function_defs=FunctionDefList([]),
    )

    repr_str = repr(spec)  # "TrajectorySpec(TL<int>([1, 2, 3]) -> )"
    parsed_spec = TrajectorySpec.parse_from_string(repr_str, available_functions)

    assert parsed_spec.input == spec.input
    assert len(parsed_spec.function_defs) == 0


def test_trajectory_spec_parse_from_string_invalid_format():
    """Test that parse_from_string raises ValueError for invalid formats."""
    import pytest

    available_functions = FunctionDefSet([])

    # Test invalid format strings
    with pytest.raises(ValueError, match="Invalid TrajectorySpec format"):
        TrajectorySpec.parse_from_string("invalid", available_functions)

    with pytest.raises(ValueError, match="Invalid TrajectorySpec format"):
        TrajectorySpec.parse_from_string("NotTrajectorySpec(...)", available_functions)

    with pytest.raises(ValueError, match="Expected format 'input -> functions'"):
        TrajectorySpec.parse_from_string(
            "TrajectorySpec(no_arrow)", available_functions
        )


def test_trajectory_spec_parse_from_string_unknown_function():
    """Test parsing with unknown function names."""
    import pytest

    # Create available functions
    inc_func = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    available_functions = FunctionDefSet([inc_func])

    # Try to parse with unknown function
    repr_str = "TrajectorySpec(TL<int>([1, 2, 3]) -> inc, unknown_func)"

    with pytest.raises(ValueError, match="No functions found"):
        TrajectorySpec.parse_from_string(repr_str, available_functions)


def test_trajectory_spec_parse_roundtrip():
    """Test that repr -> parse_from_string -> repr produces the same result."""

    # Create test functions
    inc_func = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    double_func = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )

    available_functions = FunctionDefSet([inc_func, double_func])

    test_cases = [
        TrajectorySpec(
            TypedList([1, 2, 3], item_type=int), FunctionDefList([inc_func])
        ),
        TrajectorySpec(
            TypedList([1, 2, 3], item_type=int),
            FunctionDefList([inc_func, double_func]),
        ),
        TrajectorySpec(
            TypedList([1, 2, 3], item_type=int),
            FunctionDefList([double_func, inc_func, double_func]),
        ),  # duplicates
        TrajectorySpec(
            TypedList([], item_type=int), FunctionDefList([])
        ),  # empty functions
    ]

    for original in test_cases:
        repr_str = repr(original)
        parsed = TrajectorySpec.parse_from_string(repr_str, available_functions)
        assert parsed.input == original.input
        assert len(parsed.function_defs) == len(original.function_defs)

        # Check function names match (since they may be different object instances)
        original_names = [f.name for f in original.function_defs]
        parsed_names = [f.name for f in parsed.function_defs]
        assert parsed_names == original_names


def test_trajectory_list_from_file():
    """Test TrajectoryList.from_file method with successful execution."""
    # Create test functions
    inc_func = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
        usage_count=0,
        metadata={},
    )

    double_func = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
        usage_count=0,
        metadata={},
    )

    # Create test specs
    input_list1 = TypedList([1, 2, 3])
    input_list2 = TypedList([5])

    spec1 = TrajectorySpec(input_list1, FunctionDefList([inc_func]))
    spec2 = TrajectorySpec(input_list2, FunctionDefList([inc_func, double_func]))

    spec_list = TrajectorySpecList([spec1, spec2])

    # Create temporary file and serialize the specs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        spec_list.to_py_file(temp_path, "eval_trajectory_specs")

        # Test the from_file method
        trajectory_list = TrajectoryList.from_file(temp_path)

        # Verify we got the correct number of trajectories
        assert len(trajectory_list) == 2

        # Verify first trajectory
        traj1 = trajectory_list[0]
        assert traj1.input.items == [1, 2, 3]
        assert len(traj1.function_defs) == 1
        assert traj1.function_defs[0].name == "increment"
        assert traj1.output.items == [2, 3, 4]  # [1,2,3] + 1 each

        # Verify second trajectory
        traj2 = trajectory_list[1]
        assert traj2.input.items == [5]
        assert len(traj2.function_defs) == 2
        assert traj2.function_defs[0].name == "increment"
        assert traj2.function_defs[1].name == "double"
        assert traj2.output.items == [12]  # (5 + 1) * 2 = 12

    finally:
        os.unlink(temp_path)


def test_trajectory_list_from_file_not_found():
    """Test TrajectoryList.from_file with non-existent file."""
    with pytest.raises((FileNotFoundError, ImportError)):
        TrajectoryList.from_file("non_existent_file.py")


def test_trajectory_list_from_file_empty():
    """Test TrajectoryList.from_file with empty spec list."""
    empty_list = TrajectorySpecList()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        empty_list.to_py_file(temp_path, "eval_trajectory_specs")
        trajectory_list = TrajectoryList.from_file(temp_path)

        assert len(trajectory_list) == 0
        assert len(trajectory_list.trajectories) == 0

    finally:
        os.unlink(temp_path)


def test_trajectory_spec_parse_from_string_range_type():
    """Test TrajectorySpec.parse_from_string with range type."""

    # Create test functions that work with ranges
    range_list_func = FunctionDef(
        name="range_list",
        input_type="range",
        output_type="builtins.list",
        code="return list(x)",
    )
    list_max_func = FunctionDef(
        name="list_max",
        input_type="builtins.list",
        output_type="builtins.int",
        code="return max(x) if x else 0",
    )
    int_to_str_func = FunctionDef(
        name="int_to_str",
        input_type="builtins.int",
        output_type="builtins.str",
        code="return str(x)",
    )

    available_functions = FunctionDefSet(
        [range_list_func, list_max_func, int_to_str_func]
    )

    # Test the problematic string from the bug report
    repr_str = "TrajectorySpec(TL<range>([range(3, 5), range(3, 5), range(3, 5), range(3, 5)]) -> range_list, list_max, int_to_str)"

    # This should not raise a ValueError
    parsed_spec = TrajectorySpec.parse_from_string(repr_str, available_functions)

    # Verify the parsed spec
    assert parsed_spec.input.item_type == range
    assert len(parsed_spec.input.items) == 4
    assert all(isinstance(item, range) for item in parsed_spec.input.items)
    assert len(parsed_spec.function_defs) == 3
    assert parsed_spec.function_defs[0].name == "range_list"
    assert parsed_spec.function_defs[1].name == "list_max"
    assert parsed_spec.function_defs[2].name == "int_to_str"
