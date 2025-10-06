from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.typed_list import TypedList

if TYPE_CHECKING:
    from wandering_light.trajectory import Trajectory, TrajectorySpec


class LLMTrainingExample(BaseModel):
    """A single training example for LLM supervised fine-tuning."""

    input_text: str
    output_text: str
    metadata: dict[str, Any] = {}


def generate_eval_prompt(
    input_list: TypedList,
    output_list: TypedList,
    available_functions: FunctionDefSet,
    include_available_functions: bool = True,
) -> str:
    """Generate the evaluation prompt for the LLM.

    Args:
        input_list: The input list for the transformation
        output_list: The target output list
        available_functions: Set of available FunctionDef objects
        include_available_functions: Whether to include available functions in the prompt

    Returns:
        str: The formatted prompt string
    """
    function_descs = [fn.executable_code().lstrip("\n") for fn in available_functions]

    newline = "\n"
    if include_available_functions:
        prefix = (
            "You are an expert at solving sequence transformation problems.\n"
            "Given an input list and a target output list, you need to find a sequence of functions\n"
            "that transforms the input to the output.\n\n"
            "Available functions:\n\n"
            f"{(newline * 2).join(function_descs)}\n\n"
        )
        suffix = (
            "\nProvide the sequence of function names (comma-separated) that transforms the input to the output.\n"
            "Do not output any additional text other than the sequence of function names.\n"
            "Only use the function names from the available functions list above.\n"
            "Example: function1,function2,function3\n\n"
            "Answer:"
        )
    else:
        prefix = ""
        suffix = "Answer:"

    prompt = (
        f"{prefix}"
        f"Input: {input_list.items}\n"
        f"Target Output: {output_list.items}\n"
        f"{suffix}"
    )
    return prompt


def generate_train_prompt(
    input_list: TypedList,
    output_list: TypedList,
    available_functions: FunctionDefSet,
    solution: FunctionDefList,
    include_available_functions: bool = True,
) -> LLMTrainingExample:
    """Generate the training prompt for the LLM.

    Args:
        input_list: The input list for the transformation
        output_list: The target output list
        available_functions: All available functions
        solution: The solution sequence

    Returns:
        LLMTrainingExample: The formatted training example
    """
    function_descs = [fn.executable_code().lstrip("\n") for fn in available_functions]
    solution_str = ",".join(fn.name for fn in solution)

    newline = "\n"
    available_functions_str = (
        f"Available functions:\n\n{(newline * 2).join(function_descs)}\n\n"
        if include_available_functions
        else ""
    )
    prompt = (
        f"{available_functions_str}Input: {input_list.items}\n"
        f"Target Output: {output_list.items}\n\n"
        "Answer:"
    )
    return LLMTrainingExample(
        input_text=prompt,
        output_text=solution_str,
        metadata={
            "function_def_list": solution,
            "input_list": input_list,
            "output_list": output_list,
            "available_functions": available_functions,
        },
    )


def generate_proposer_training_prompt(
    example_trajectories: list["Trajectory"],
    target_spec: "TrajectorySpec",
    available_functions: FunctionDefSet,
    include_available_functions: bool = False,
) -> LLMTrainingExample:
    """Generate training prompt for proposing trajectory specs based on examples.

    Args:
        example_trajectories: List of example Trajectory objects to learn from
        target_spec: The target TrajectorySpec to predict
        available_functions: Set of available FunctionDef objects
        include_available_functions: Whether to include available functions in the prompt

    Returns:
        LLMTrainingExample: The formatted training example
    """
    if include_available_functions:
        raise ValueError(
            "include_available_functions is not supported for proposer training currently."
        )

    # Build the input text with example trajectories
    input_parts = []

    for i, trajectory in enumerate(example_trajectories, 1):
        input_parts.append(f"Example {i}:")
        input_parts.append(f"{trajectory.__repr__()}")
        input_parts.append("")  # Empty line between examples

    input_text = "\n".join(input_parts)  # Keep trailing newline

    # Output text is the target spec representation
    output_text = target_spec.__repr__()

    return LLMTrainingExample(
        input_text=input_text,
        output_text=output_text,
        metadata={
            "example_trajectories": example_trajectories,
            "target_spec": target_spec,
            "available_functions": available_functions,
        },
    )
