import argparse
import os
import random
import string
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from wandering_light.common_functions import SAMPLE_INPUTS, basic_fns
from wandering_light.function_def import FunctionDefList
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList

# Define the types supported by _random_value_for_type
SUPPORTED_RANDOM_TYPES = [
    int,
    float,
    str,
    bool,
    list,
    tuple,
    set,
    dict,
    bytes,
    bytearray,
    complex,
    range,
]


def _random_value_for_type(t: type[Any]) -> Any:
    """Generate a random value for a builtin type."""
    if t is int:
        return random.randint(-10, 10)
    if t is float:
        return random.uniform(-10.0, 10.0)
    if t is str:
        length = random.randint(0, 6)
        alphabet = string.ascii_letters + string.digits
        return "".join(random.choices(alphabet, k=length))
    if t is bool:
        return random.choice([True, False])
    if t is list:
        return [random.randint(-5, 5) for _ in range(random.randint(0, 3))]
    if t is tuple:
        return tuple(random.randint(-5, 5) for _ in range(random.randint(0, 3)))
    if t is set:
        return {random.randint(-5, 5) for _ in range(random.randint(0, 3))}
    if t is dict:
        return {chr(97 + i): random.randint(-5, 5) for i in range(random.randint(0, 3))}
    if t is bytes:
        return bytes(random.getrandbits(8) for _ in range(random.randint(0, 4)))
    if t is bytearray:
        return bytearray(random.getrandbits(8) for _ in range(random.randint(0, 4)))
    if t is complex:
        return complex(random.randint(-5, 5), random.randint(-5, 5))
    if t is range:
        start = random.randint(0, 3)
        stop = start + random.randint(0, 5)
        return range(start, stop)

    # Fallback: try SAMPLE_INPUTS
    type_str = f"{t.__module__}.{t.__qualname__}"
    if type_str in SAMPLE_INPUTS:
        return random.choice(SAMPLE_INPUTS[type_str])
    raise ValueError(f"No sample input for type {type_str}")


def make_data(
    available_functions: FunctionDefList,
    length_counts: Mapping[int, int],
    input_lists: list[TypedList],
    seed: int | None = None,
) -> TrajectorySpecList:
    """
    Creates a TrajectorySpecList with lengths specified in length_counts,
    using available functions.

    Args:
        available_functions: List of function definitions
        length_counts: Mapping of length to count of trajectories to generate
        input_lists: List of possible input lists to choose from
        seed: Optional random seed for reproducibility. If None, uses system time.

    Returns:
        TrajectorySpecList object
    """
    if seed is not None:
        random.seed(seed)

    trajectories = TrajectorySpecList()

    for length, count in length_counts.items():
        # Create a random typed list to use as input
        for _ in range(count):
            input_list = random.choice(input_lists)

            # Generate a random trajectory of the specified length
            trajectory = TrajectorySpec.create_random_walk(
                input_list=input_list,
                path_length=length,
                available_functions=available_functions,
            )

            trajectories.append(trajectory)

    return trajectories


def make_random_data(
    available_functions: FunctionDefList,
    length_counts: Mapping[int, int],
    seed: int | None = None,
) -> TrajectorySpecList:
    """Like :func:`make_data` but generates new random input lists with randomly chosen types."""
    if seed is not None:
        random.seed(seed)

    trajectories = TrajectorySpecList()

    for length, count in length_counts.items():
        for _ in range(count):
            # Randomly choose a type from supported types
            input_type = random.choice(SUPPORTED_RANDOM_TYPES)

            list_len = 2 if input_type is bool else random.randint(3, 5)
            items = [_random_value_for_type(input_type) for _ in range(list_len)]
            input_list = TypedList(items, item_type=input_type)

            trajectory = TrajectorySpec.create_random_walk(
                input_list=input_list,
                path_length=length,
                available_functions=available_functions,
            )

            trajectories.append(trajectory)

    return trajectories


def make_common_fns_data(
    save_to_file: bool = False,
    output_dir: str = "evals/data",
    length_counts: Mapping[int, int] | None = None,
    random_inputs: bool = False,
) -> TrajectorySpecList:
    """
    Generate trajectory spec data using basic functions.

    Args:
        save_to_file: Whether to save the data to a .py file
        output_dir: Directory to save the file (relative to project root)
        length_counts: Mapping of length to count of trajectories to generate.
                      If None, defaults to {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
        random_inputs: Whether to generate random input lists instead of using SAMPLE_INPUTS

    Returns:
        TrajectorySpecList object
    """
    if length_counts is None:
        length_counts = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}

    if random_inputs:
        data = make_random_data(
            FunctionDefList(basic_fns),
            length_counts,
        )
    else:
        input_lists = [TypedList(items=v) for v in SAMPLE_INPUTS.values()]
        data = make_data(
            FunctionDefList(basic_fns),
            length_counts,
            input_lists,
        )

    if save_to_file:
        # Create output directory if it doesn't exist
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        full_output_dir = os.path.join(project_root, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_data_v{timestamp}.py"
        file_path = os.path.join(full_output_dir, filename)

        # Save to file
        data.to_py_file(file_path, "eval_trajectory_specs")
        print(f"Saved {len(data)} trajectory specs to {file_path}")

        # Also create a latest symlink/copy
        latest_path = os.path.join(full_output_dir, "eval_data_latest.py")
        data.to_py_file(latest_path, "eval_trajectory_specs")
        print(f"Also saved as {latest_path}")

    return data


def parse_length_counts(csv_string: str) -> Mapping[int, int]:
    """
    Parse a CSV string of length:count pairs into a dictionary.

    Args:
        csv_string: String like "1:10,2:20,3:30" (spaces around separators are allowed)

    Returns:
        Dictionary mapping length to count
    """
    length_counts = {}
    for pair in csv_string.strip().split(","):
        trimmed_pair = pair.strip()
        if ":" not in trimmed_pair:
            raise ValueError(
                f"Invalid format: '{trimmed_pair}'. Expected 'length:count'"
            )

        length_str, count_str = trimmed_pair.split(":", 1)  # Split only on first ':'
        try:
            length_counts[int(length_str.strip())] = int(count_str.strip())
        except ValueError as e:
            raise ValueError(f"Invalid number in pair '{trimmed_pair}': {e}") from e
    return length_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation trajectory data")
    parser.add_argument("--save", action="store_true", help="Save the data to a file")
    parser.add_argument(
        "--length-counts",
        type=str,
        default="1:100,2:100,3:100,4:100,5:100",
        help="Length counts as CSV (e.g., '1:10,2:20,3:30')",
    )
    parser.add_argument(
        "--random-inputs",
        action="store_true",
        help="Generate random input lists instead of using predefined SAMPLE_INPUTS",
    )

    args = parser.parse_args()

    length_counts = parse_length_counts(args.length_counts)
    data = make_common_fns_data(
        save_to_file=args.save,
        length_counts=length_counts,
        random_inputs=args.random_inputs,
    )

    if args.save:
        print(f"Generated {len(data)} trajectory specifications")
    else:
        print(data)
