from wandering_light.function_def import FunctionDef
from wandering_light.training.data_generator import induction_dataset


def test_induction_dataset_custom_functions():
    """Test that custom function palette works as expected."""
    # Arrange
    custom_functions = [
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
    custom_lengths = {1: 2, 2: 3}  # 2 length-1 and 3 length-2 trajectories

    # Act
    result = induction_dataset(
        function_pallete=custom_functions, length_counts=custom_lengths
    )

    # Assert
    assert len(result) == 5, f"Expected 5 examples, got {len(result)}"

    # Verify that the examples use our custom functions
    custom_fn_names = {f.name for f in custom_functions}
    non_empty_examples = [ex for ex in result if ex["completion"].strip()]

    # At least some examples should have non-empty completions
    assert len(non_empty_examples) > 0, (
        "Expected at least some examples with non-empty completions"
    )

    for example in non_empty_examples:
        # Check if any custom function name appears in the completion
        # The completion should contain comma-separated function names
        completion_functions = set(example["completion"].split(","))
        assert any(name in completion_functions for name in custom_fn_names), (
            f"Example doesn't use custom functions: {example['completion']}, expected one of {custom_fn_names}"
        )

    # Check structure of each example
    for example in result:
        assert isinstance(example, dict), "Each example should be a dictionary"
        assert "prompt" in example, "Example should have 'prompt' key"
        assert "completion" in example, "Example should have 'completion' key"
        assert isinstance(example["prompt"], str), "Prompt should be a string"
        assert isinstance(example["completion"], str), "Completion should be a string"
