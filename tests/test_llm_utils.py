from pathlib import Path

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.llm_utils import (
    LLMTrainingExample,
    generate_eval_prompt,
    generate_proposer_training_prompt,
    generate_train_prompt,
)
from wandering_light.trajectory import Trajectory, TrajectorySpec
from wandering_light.typed_list import TypedList


@pytest.mark.parametrize("include_available_functions", [True, False])
def test_generate_prompt(include_available_functions: bool):
    # Create actual FunctionDef objects
    func1 = FunctionDef(
        name="add_one",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )

    func2 = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )

    available_functions = FunctionDefList([func1, func2])
    input_list = TypedList([1, 2, 3], item_type=int)
    output_list = TypedList([2, 4, 6], item_type=int)

    # Call the function
    result = generate_eval_prompt(
        input_list, output_list, available_functions, include_available_functions
    )

    # Read the expected prompt from file and normalize line endings
    expected_prompt_path = (
        Path(__file__).parent
        / "test_data"
        / f"expected_eval_prompt{'' if include_available_functions else '_no_available_functions'}.txt"
    )
    with open(expected_prompt_path, newline="") as f:
        expected_prompt = f.read()

    # Normalize line endings and remove extra newlines
    result = result.replace("\r\n", "\n").replace("\n\n\n", "\n\n")
    expected_prompt = expected_prompt.replace("\r\n", "\n")

    # Compare the generated prompt with the expected one
    if result != expected_prompt:
        print("\n=== ACTUAL PROMPT ===")
        print(repr(result))
        print("\n=== EXPECTED PROMPT ===")
        print(repr(expected_prompt))
        print("\n")
        assert result == expected_prompt, "Generated prompt does not match expected"


def test_generate_prompt_empty_functions():
    # Test with empty functions list
    result = generate_eval_prompt(
        TypedList([1, 2, 3], item_type=int),
        TypedList([1, 2, 3], item_type=int),
        FunctionDefList([]),
        include_available_functions=True,
    )
    assert "Available functions" in result
    assert "Input: [1, 2, 3]" in result
    assert "Target Output: [1, 2, 3]" in result


@pytest.mark.parametrize("include_available_functions", [True, False])
def test_generate_train_prompt(include_available_functions: bool):
    # Create actual FunctionDef objects
    func1 = FunctionDef(
        name="add_one",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )

    func2 = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )

    available_functions = FunctionDefList([func1, func2])
    input_list = TypedList([1, 2, 3], item_type=int)
    output_list = TypedList([2, 4, 6], item_type=int)

    # Create a solution as a list of FunctionDef objects
    solution = FunctionDefList([func1, func2])

    # Call the function
    result = generate_train_prompt(
        input_list,
        output_list,
        available_functions,
        solution,
        include_available_functions=include_available_functions,
    )

    # Verify the result is an LLMTrainingExample
    assert isinstance(result, LLMTrainingExample)

    # Read the expected prompt from file and normalize line endings
    expected_prompt_path = (
        Path(__file__).parent
        / "test_data"
        / f"expected_train_prompt{'' if include_available_functions else '_no_available_functions'}.txt"
    )
    with open(expected_prompt_path, newline="") as f:
        expected_prompt = f.read()

    # Normalize line endings and remove extra newlines
    actual_prompt = (
        result.input_text.replace("\r\n", "\n").replace("\n\n\n", "\n\n").rstrip()
    )
    expected_prompt = expected_prompt.replace("\r\n", "\n").rstrip()

    # Compare the generated prompt with the expected one
    if actual_prompt != expected_prompt:
        print("\n=== ACTUAL PROMPT ===")
        print(repr(actual_prompt))
        print("\n=== EXPECTED PROMPT ===")
        print(repr(expected_prompt))
        print("\n")
        assert actual_prompt == expected_prompt, (
            "Generated training prompt does not match expected"
        )

    # Verify the output text matches the solution
    assert result.output_text == "add_one,double"

    # Verify metadata
    assert result.metadata is not None
    assert result.metadata["function_def_list"] == solution
    assert result.metadata["input_list"] == input_list
    assert result.metadata["output_list"] == output_list
    assert result.metadata["available_functions"] == available_functions


def test_generate_train_prompt_empty_functions():
    # Test with empty functions list
    input_list = TypedList([], item_type=int)
    output_list = TypedList([], item_type=int)
    solution = FunctionDefList([])  # Empty list for solution

    result = generate_train_prompt(
        input_list, output_list, FunctionDefList([]), solution
    )

    assert isinstance(result, LLMTrainingExample)
    assert "Available functions:" in result.input_text
    assert "Input: []" in result.input_text
    assert "Target Output: []" in result.input_text
    assert result.output_text == ""


def test_generate_proposer_training_prompt():
    # Create actual FunctionDef objects
    f_add_1 = FunctionDef(
        name="add_one",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )

    f_double = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )

    f_subtract_1 = FunctionDef(
        name="subtract_one",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x - 1",
    )

    available_functions = FunctionDefList([f_add_1, f_double, f_subtract_1])

    # Create example trajectories for training input
    example_trajectories = [
        Trajectory(
            spec=TrajectorySpec(
                input_list=TypedList([1, 2, 3], item_type=int),
                function_defs=FunctionDefList([f_add_1]),
            ),
            output=TypedList([2, 3, 4], item_type=int),
        ),
        Trajectory(
            spec=TrajectorySpec(
                input_list=TypedList([2, 4, 6], item_type=int),
                function_defs=FunctionDefList([f_double]),
            ),
            output=TypedList([4, 8, 12], item_type=int),
        ),
        Trajectory(
            spec=TrajectorySpec(
                input_list=TypedList([2, 4, 6], item_type=int),
                function_defs=FunctionDefList([f_subtract_1]),
            ),
            output=TypedList([1, 3, 5], item_type=int),
        ),
    ]

    # Create target trajectory spec for the expected output
    target_spec = TrajectorySpec(
        input_list=TypedList([5, 10, 15], item_type=int),
        function_defs=FunctionDefList([f_add_1, f_double]),
    )

    # Call the function
    result = generate_proposer_training_prompt(
        example_trajectories=example_trajectories,
        target_spec=target_spec,
        available_functions=available_functions,
        include_available_functions=False,
    )

    # Verify the result is an LLMTrainingExample
    assert isinstance(result, LLMTrainingExample)

    # Verify the input text contains trajectory examples
    assert (
        result.input_text
        == f"""Example 1:
{example_trajectories[0].__repr__()}

Example 2:
{example_trajectories[1].__repr__()}

Example 3:
{example_trajectories[2].__repr__()}
"""
    )
    assert result.output_text == f"""{target_spec.__repr__()}"""
