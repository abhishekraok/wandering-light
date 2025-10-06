import pytest

from wandering_light.evals.create_data import make_data
from wandering_light.function_def import FunctionDef
from wandering_light.typed_list import TypedList


@pytest.fixture
def sample_functions():
    """Fixture providing sample function definitions for testing."""
    return [
        FunctionDef(
            name="add_one",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        ),
        FunctionDef(
            name="multiply_by_two",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        ),
    ]


@pytest.fixture
def sample_input_lists():
    """Fixture providing sample input lists for testing."""
    return [
        TypedList([1, 2, 3], int),
        TypedList([4, 5, 6], int),
    ]


def test_make_data_reproducibility(sample_functions, sample_input_lists):
    """Test that make_data is reproducible with the same seed."""
    length_counts = {2: 3}  # Generate 3 trajectories of length 2
    seed = 123

    # Generate results with the same seed
    result1 = make_data(sample_functions, length_counts, sample_input_lists, seed=seed)
    result2 = make_data(sample_functions, length_counts, sample_input_lists, seed=seed)

    # Verify results are identical
    assert len(result1) == len(result2), "Result lengths differ"
    for t1, t2 in zip(result1, result2, strict=False):
        assert t1.input == t2.input, "Input lists differ"
        assert len(t1.function_defs) == len(t2.function_defs), (
            "Function sequence lengths differ"
        )
        for f1, f2 in zip(t1.function_defs, t2.function_defs, strict=False):
            assert f1.name == f2.name, "Function names differ"


def test_make_data_output_structure(sample_functions, sample_input_lists):
    """Test the structure of the generated data."""
    length_counts = {2: 2, 3: 1}  # 2 trajectories of length 2, 1 of length 3
    result = make_data(sample_functions, length_counts, sample_input_lists)

    # Check total number of trajectories
    assert len(result) == 3, "Should generate 3 examples in total"

    # Check each trajectory
    for traj in result:
        assert hasattr(traj, "input"), "Trajectory missing input"
        assert hasattr(traj, "function_defs"), "Trajectory missing function_defs"
        assert len(traj.function_defs) in [2, 3], "Unexpected trajectory length"
        for fn in traj.function_defs:
            assert fn.name in ["add_one", "multiply_by_two"], "Unexpected function name"


def test_make_data_without_seed(sample_functions, sample_input_lists):
    """Test that make_data works without a seed."""
    length_counts = {2: 2}
    result = make_data(sample_functions, length_counts, sample_input_lists)
    assert len(result) == 2, "Should generate 2 examples"
