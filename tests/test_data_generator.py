import pytest
from datasets import Dataset

from wandering_light.common_functions import basic_fns
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.llm_utils import LLMTrainingExample
from wandering_light.training.data_generator import (
    generate_proposer_training_data,
    generate_training_data,
    proposer_dataset,
)
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList


@pytest.fixture
def sample_functions():
    """Fixture providing sample function definitions for testing."""
    return [
        FunctionDef(
            name="add",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        ),
        FunctionDef(
            name="multiply",
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


def test_generate_training_data_reproducibility(sample_functions, sample_input_lists):
    """Test that generate_training_data is reproducible with the same seed."""
    length_counts = {2: 3}  # Generate 3 trajectories of length 2
    seed = 123

    # Generate results with the same seed
    result1, _, _ = generate_training_data(
        FunctionDefSet(sample_functions), length_counts, seed=seed
    )
    result2, _, _ = generate_training_data(
        FunctionDefSet(sample_functions), length_counts, seed=seed
    )

    # Verify results are identical
    assert len(result1) == len(result2), "Result lengths differ"
    for i, (ex1, ex2) in enumerate(zip(result1, result2, strict=False)):
        assert ex1.input_text == ex2.input_text, f"Input text differs at index {i}"
        assert ex1.output_text == ex2.output_text, f"Output text differs at index {i}"
        assert ex1.metadata is not None and ex2.metadata is not None, (
            f"Metadata is None at index {i}"
        )


def test_generate_training_data_output_structure(sample_functions, sample_input_lists):
    """Test the structure of the generated training data."""
    length_counts = {1: 2, 2: 2}  # Generate 2 trajectories of length 1 and 2
    examples, _, _ = generate_training_data(
        FunctionDefSet(sample_functions), length_counts
    )

    # Verify we got the expected number of examples
    assert len(examples) == 4, f"Expected 4 examples, got {len(examples)}"

    # Verify each example has the expected structure
    for example in examples:
        assert isinstance(example, LLMTrainingExample)
        assert hasattr(example, "input_text")
        assert isinstance(example.input_text, str)
        assert hasattr(example, "output_text")
        assert isinstance(example.output_text, str)
        assert hasattr(example, "metadata")
        assert isinstance(example.metadata, dict)
        assert "verifier_id" in example.metadata


def test_generate_training_data_random_order(sample_functions, sample_input_lists):
    """Generated examples should not be sorted by trajectory length."""
    # Use basic_fns instead of limited sample_functions to get meaningful trajectories
    length_counts = {1: 3, 2: 3, 3: 3}
    examples, _, _ = generate_training_data(basic_fns, length_counts, seed=42)

    lengths = [len(ex.metadata["function_def_list"]) for ex in examples]
    # Only check if we have trajectories with different lengths
    if len(set(lengths)) > 1:
        assert lengths != sorted(lengths), "Examples appear to be length sorted"
    else:
        # If all trajectories have the same length, we can't test ordering
        raise ValueError(
            "All generated trajectories have the same length, cannot test ordering"
        )


def test_generate_proposer_training_data_basic_functionality():
    """Test that generate_proposer_training_data returns only training examples."""
    # Create test functions
    test_functions = [
        FunctionDef(
            name="inc",
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

    length_counts = {1: 2, 2: 2}  # Generate 2 trajectories each of length 1 and 2

    # Call the function (this will fail until implemented)
    result = generate_proposer_training_data(
        available_functions=FunctionDefSet(test_functions),
        length_counts=length_counts,
        seed=42,
    )

    # Verify it returns only the training examples (not a tuple)
    assert isinstance(result, list), "Should return a list of training examples"
    assert len(result) == 4, f"Expected 4 examples, got {len(result)}"

    # Verify each example is an LLMTrainingExample
    for example in result:
        assert isinstance(example, LLMTrainingExample)
        assert isinstance(example.input_text, str)
        assert isinstance(example.output_text, str)
        assert isinstance(example.metadata, dict)
        # output should be a parseable trajectory spec
        assert isinstance(
            TrajectorySpec.parse_from_string(
                example.output_text, available_functions=FunctionDefSet(test_functions)
            ),
            TrajectorySpec,
        )


def test_proposer_dataset():
    """Test that proposer_dataset returns a properly formatted Dataset."""

    test_functions = basic_fns[:9]  # Use first 3 basic functions
    length_counts = {1: 2, 2: 2}  # Small counts for testing

    # Call the function
    result = proposer_dataset(
        length_counts=length_counts, function_pallete=test_functions
    )

    # Verify it returns a Dataset
    assert isinstance(result, Dataset)
    assert len(result) > 0  # Should have some examples

    # Verify the dataset structure
    assert "prompt" in result.column_names
    assert "completion" in result.column_names
    assert "verifier_id" not in result.column_names  # Proposer doesn't need verifier_id

    # Verify each example has the expected structure
    for example in result:
        assert isinstance(example["prompt"], str)
        assert isinstance(example["completion"], str)

        # Proposer format: input should contain "Example" and completion should be TrajectorySpec
        assert "Example" in example["prompt"]
        assert "TrajectorySpec(" in example["completion"]

        # Verify the completion can be parsed as a TrajectorySpec
        parsed_spec = TrajectorySpec.parse_from_string(
            example["completion"], available_functions=FunctionDefSet(test_functions)
        )
        assert isinstance(parsed_spec, TrajectorySpec)


def test_proposer_dataset_with_defaults():
    """Test proposer_dataset with default parameters."""

    # Test with larger counts to ensure we have enough trajectories (need at least 4)
    length_counts = {1: 3, 2: 3}  # Enough to generate 6 trajectories

    result = proposer_dataset(length_counts=length_counts)

    assert isinstance(result, Dataset)
    assert len(result) > 0  # Should have examples since we have enough trajectories
    assert "prompt" in result.column_names
    assert "completion" in result.column_names
