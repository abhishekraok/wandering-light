import random
from collections.abc import Callable, Mapping

import numpy as np
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer

from wandering_light.common_functions import basic_fns
from wandering_light.evals.create_data import make_random_data
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.llm_utils import (
    LLMTrainingExample,
    generate_proposer_training_prompt,
    generate_train_prompt,
)
from wandering_light.trajectory import Trajectory, TrajectorySpec


def generate_training_data(
    available_functions: FunctionDefSet,
    length_counts: Mapping[int, int],
    seed: int = 42,
) -> tuple[
    list[LLMTrainingExample],
    dict[int, Callable[[str], tuple[float, int]]],
    dict[int, int],
]:
    """
    Generate reproducible training data for LLM fine-tuning.

    Args:
        available_functions: List of function definitions
        length_counts: Mapping of length to count of trajectories to generate
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of LLMTrainingExample objects containing input-output pairs
        Dict mapping verifier_id to verifier function that returns (correctness, function_count)
        Dict mapping verifier_id to ground truth function count
    """
    executor = Executor(available_functions)
    trajectory_spec_list = make_random_data(
        available_functions=available_functions,
        length_counts=length_counts,
        seed=seed,
    )
    trajectory_spec_list.shuffle(seed)
    print(f"Generated {len(trajectory_spec_list):,} trajectories")
    training_examples = []
    verifier_id_to_fn = {}
    verifier_id_to_ground_truth_length = {}
    error_count = 0
    for verifier_id, traj_spec in enumerate(trajectory_spec_list):
        result = executor.execute_trajectory(traj_spec)
        if result.success:
            trajectory = result.trajectory
            train_ex = generate_train_prompt(
                input_list=traj_spec.input,
                output_list=trajectory.output,
                available_functions=available_functions,
                solution=traj_spec.function_defs,
                include_available_functions=False,
            )
            verifier_id_to_fn[verifier_id] = create_verifier(
                trajectory, available_functions
            )
            verifier_id_to_ground_truth_length[verifier_id] = len(
                traj_spec.function_defs
            )
            train_ex.metadata["verifier_id"] = verifier_id
            training_examples.append(train_ex)
        else:
            error_count += 1
            continue
    if error_count:
        print(f"Skipped {error_count} invalid trajectories")
    print(f"Generated {len(training_examples):,} training examples")
    return training_examples, verifier_id_to_fn, verifier_id_to_ground_truth_length


def generate_proposer_training_data(
    available_functions: FunctionDefSet,
    length_counts: Mapping[int, int],
    seed: int = 42,
) -> list[LLMTrainingExample]:
    """Generate training data for the proposer model."""
    executor = Executor(available_functions)
    trajectory_spec_list = make_random_data(
        available_functions=available_functions,
        length_counts=length_counts,
        seed=seed,
    )
    trajectory_spec_list.shuffle(seed)
    print(f"Generated {len(trajectory_spec_list):,} trajectories")
    training_examples = []
    error_count = 0
    trajectories = []
    for traj_spec in trajectory_spec_list:
        result = executor.execute_trajectory(traj_spec)
        if result.success:
            trajectory = result.trajectory
            trajectories.append(trajectory)
        else:
            error_count += 1
            continue
    if error_count:
        print(f"Skipped {error_count} invalid trajectories")
    if len(trajectories) < 4:
        raise ValueError(
            f"Not enough trajectories to generate training examples. Need at least 4, got {len(trajectories)}"
        )

    random.seed(seed)
    for _ in range(len(trajectories)):
        # Pick 4 random trajectories to use as examples
        example_trajectories = random.sample(trajectories, 4)
        train_ex = generate_proposer_training_prompt(
            example_trajectories=example_trajectories[:3],
            target_spec=example_trajectories[3].to_spec(),
            available_functions=available_functions,
            include_available_functions=False,
        )
        training_examples.append(train_ex)

    print(f"Generated {len(training_examples):,} training examples")
    return training_examples


def induction_dataset_rl(
    length_counts: Mapping[int, int] | None = None,
    function_pallete: list[FunctionDef] | None = None,
) -> tuple[Dataset, dict[int, Callable[[str], tuple[float, int]]], dict[int, int]]:
    """Generate a dataset for function induction in the TRL prompt completion format."""
    available_functions = FunctionDefSet(function_pallete or basic_fns)
    if not length_counts:
        length_counts = {
            1: len(basic_fns),
            2: (len(basic_fns) ** 2) // 2,
            3: (len(basic_fns) ** 2) // 3,
            4: len(basic_fns),
            5: len(basic_fns),
        }
    print(f"Final length counts: {length_counts}")

    training_data, verifier_id_to_fn, verifier_id_to_ground_truth_length = (
        generate_training_data(
            available_functions=available_functions,
            length_counts=length_counts,
            seed=42,
        )
    )

    dataset = []
    for llm_train_ex in training_data:
        dataset.append(
            {
                "prompt": llm_train_ex.input_text,
                "completion": llm_train_ex.output_text,
                "verifier_id": llm_train_ex.metadata["verifier_id"],
            }
        )
    print(f"Generated {len(dataset):,} examples for induction")
    return (
        Dataset.from_list(dataset),
        verifier_id_to_fn,
        verifier_id_to_ground_truth_length,
    )


def induction_dataset(
    length_counts: Mapping[int, int] | None = None,
    function_pallete: list[FunctionDef] | None = None,
) -> Dataset:
    return induction_dataset_rl(length_counts, function_pallete)[0]


def proposer_dataset(
    length_counts: Mapping[int, int] | None = None,
    function_pallete: list[FunctionDef] | None = None,
) -> Dataset:
    """Generate a dataset for trajectory proposal training."""
    available_functions = FunctionDefSet(function_pallete or basic_fns)
    if not length_counts:
        length_counts = {
            1: len(basic_fns),
            2: (len(basic_fns) ** 2) // 2,
            3: (len(basic_fns) ** 2) // 3,
            4: len(basic_fns),
            5: len(basic_fns),
        }
    print(f"Final length counts: {length_counts}")

    training_data = generate_proposer_training_data(
        available_functions=available_functions,
        length_counts=length_counts,
        seed=42,
    )

    dataset = []
    for llm_train_ex in training_data:
        dataset.append(
            {
                "prompt": llm_train_ex.input_text,
                "completion": llm_train_ex.output_text,
            }
        )
    print(f"Generated {len(dataset):,} examples for trajectory proposal")
    return Dataset.from_list(dataset)


def proposer_dataset_rl(
    length_counts: Mapping[int, int] | None,
    function_pallete: FunctionDefSet,
):
    """Generate a dataset for trajectory proposer in the TRL prompt completion format."""
    dataset = proposer_dataset(length_counts, function_pallete)
    rl_dataset = []
    for item in dataset:
        rl_dataset.append(
            {
                "prompt": item["prompt"],
                "completion": item["completion"],
            }
        )

    print(f"Generated {len(rl_dataset):,} examples for proposer RL training")
    return Dataset.from_list(rl_dataset)


def create_verifier(
    actual_trajectory: Trajectory, available_functions: FunctionDefSet
) -> Callable[[str], tuple[float, int]]:
    """Create a verifier function from a trajectory that returns (correctness, function_count)."""
    executor = Executor(available_functions)

    def reward_fn(completion: str) -> tuple[float, int]:
        parsed_functions = available_functions.parse_string(completion)
        function_count = len(parsed_functions)

        parsed_traj_spec = TrajectorySpec(
            input_list=actual_trajectory.input,
            function_defs=parsed_functions,
        )
        result = executor.execute_trajectory(parsed_traj_spec)

        if not result.success:
            # If execution failed, return 0.0 reward with high penalty count
            return (0.0, 10)  # High function count as penalty for failed execution

        # Compare with expected output
        if result.trajectory.output.items == actual_trajectory.output.items:
            return (1.0, function_count)  # Correct
        else:
            return (
                0.0,
                function_count,
            )  # Incorrect but we still have function count

    return reward_fn


def token_statistics(model_name: str):
    BATCH_SIZE = 64  # tokenise in chunks to avoid RAM spikes
    PERCENTILE = 95  # which percentile you care about

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    ds = induction_dataset()

    lengths = []

    def length_mapper(batch):
        # concat prompt + completion the same way SFTTrainer does (no extra BOS)
        texts = [
            p + c + tok.eos_token
            for p, c in zip(batch["prompt"], batch["completion"], strict=False)
        ]
        enc = tok(
            texts, add_special_tokens=False
        )  # faster, we already added EOS manually
        # note: enc["input_ids"] is a list of lists
        lengths.extend([len(ids) for ids in enc["input_ids"]])
        return {}  # we don't need to keep anything

    if isinstance(ds, IterableDataset):
        for batch in ds.batch(batch_size=BATCH_SIZE):
            length_mapper(batch)
    else:
        ds.map(
            length_mapper,
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=ds.column_names,
        )

    arr = np.array(lengths)
    mean_len = arr.mean()
    max_len = arr.max()
    p95_len = np.percentile(arr, PERCENTILE)

    print(f"# samples          : {len(arr):,}")
    print(f"Mean length        : {mean_len:,.2f} tokens")
    print(f"{PERCENTILE}th percentile : {p95_len:,.0f} tokens")
    print(f"Maximum length     : {max_len:,} tokens")

    suggested = int(p95_len) + 32  # a small buffer; adjust to taste
    print(
        f"\nSuggested max_seq_length ≈ {suggested} (95th percentile + 32-token head-room)"
    )


if __name__ == "__main__":
    token_statistics("facebook/opt-125m")
