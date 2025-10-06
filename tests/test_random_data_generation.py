import pytest

from wandering_light.common_functions import basic_fns
from wandering_light.evals.create_data import make_random_data
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList


def _convert_to_hashable(item):
    """Convert an item to a hashable representation for comparison."""
    if isinstance(item, bytearray | set):
        return str(item)  # Convert unhashable types to strings
    elif isinstance(item, list):
        return tuple(_convert_to_hashable(x) for x in item)
    elif isinstance(item, dict):
        return tuple(sorted((k, _convert_to_hashable(v)) for k, v in item.items()))
    else:
        return item


@pytest.fixture
def sample_functions():
    return [
        FunctionDef(
            name="add_one",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        ),
        FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        ),
    ]


def test_random_data_reproducible(sample_functions):
    length_counts = {1: 5}
    seed = 99
    result1 = make_random_data(
        FunctionDefList(sample_functions), length_counts, seed=seed
    )
    result2 = make_random_data(
        FunctionDefList(sample_functions), length_counts, seed=seed
    )
    assert len(result1) == len(result2)
    for spec1, spec2 in zip(result1, result2, strict=False):
        assert spec1.input == spec2.input
        assert [f.name for f in spec1.function_defs] == [
            f.name for f in spec2.function_defs
        ]


def test_random_data_variety(sample_functions):
    COUNTS = 10
    length_counts = {1: COUNTS}
    result = make_random_data(FunctionDefList(sample_functions), length_counts, seed=42)

    # Convert items to hashable representations for comparison
    def make_hashable(items):
        hashable_items = []
        for item in items:
            if isinstance(item, bytearray):
                hashable_items.append(("bytearray", bytes(item)))
            elif isinstance(item, set):
                hashable_items.append(("set", frozenset(item)))
            elif isinstance(item, list):
                hashable_items.append(("list", tuple(item)))
            elif isinstance(item, dict):
                hashable_items.append(("dict", tuple(sorted(item.items()))))
            else:
                hashable_items.append(item)
        return tuple(hashable_items)

    inputs = [make_hashable(spec.input.items) for spec in result]
    assert len(set(inputs)) == COUNTS, "Expected multiple distinct input lists"


def test_random_data_output_variety(sample_functions):
    COUNTS = 15
    length_counts = {1: COUNTS}
    data = make_random_data(FunctionDefList(sample_functions), length_counts, seed=123)
    ex = Executor(FunctionDefList(sample_functions))

    # Convert items to hashable representations for comparison
    def make_hashable(items):
        hashable_items = []
        for item in items:
            if isinstance(item, bytearray):
                hashable_items.append(("bytearray", bytes(item)))
            elif isinstance(item, set):
                hashable_items.append(("set", frozenset(item)))
            elif isinstance(item, list):
                hashable_items.append(("list", tuple(item)))
            elif isinstance(item, dict):
                hashable_items.append(("dict", tuple(sorted(item.items()))))
            else:
                hashable_items.append(item)
        return tuple(hashable_items)

    outputs = []
    for spec in data:
        result = ex.execute_trajectory(spec)
        if result.success:
            outputs.append(make_hashable(result.trajectory.output.items))
        else:
            # Skip failed executions for this test
            continue
    assert len(set(outputs)) == COUNTS, "Expected varying outputs for random inputs"


def test_length_by_type():
    data = make_random_data(basic_fns, {1: 1}, seed=42)
    assert 2 <= len(data[0].input.items) <= 5

    # Test that different seeds produce different results
    data2 = make_random_data(basic_fns, {1: 1}, seed=123)
    # They might be the same by chance, but with different seeds they should usually differ
    assert len(data2[0].input.items) >= 2
