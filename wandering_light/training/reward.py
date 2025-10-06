import math
from collections.abc import Callable
from typing import Protocol

from wandering_light.evals.evaluate_proposer import evaluate_responses
from wandering_light.function_def import FunctionDefList, FunctionDefSet
from wandering_light.solver import TrajectorySolver

LENGTH_REWARD_NAME = "LengthReward"
INDUCTION_REWARD_NAME = "InductionReward"
MAX_FUNCTIONS = 5


class RewardMetricsObserver(Protocol):
    """Observer interface for receiving metrics from reward functions."""

    def on_batch_processed(
        self,
        success_rate: float,
        avg_function_count: float,
        function_counts: list[int],
        correctness_scores: list[float],
        avg_function_count_ratio: float,
        function_count_ratios: list[float],
    ) -> None:
        """Called when a batch is processed by the reward function."""
        ...


class ProposerMetricsObserver(Protocol):
    """Observer interface for receiving proposer-specific metrics from reward functions."""

    def on_proposer_batch_processed(
        self,
        parse_rate: float,
        solver_success_rate: float,
        frac_non_zero_std: float,
        rewards: list[float],
        avg_function_count: float,
    ) -> None:
        """Called when a batch is processed by the proposer reward function."""
        ...


def length_reward_single(parsed_function_list: FunctionDefList) -> float:
    if len(parsed_function_list) == 0:
        # Most likely parsing failed due to improper output from the model
        return -1
    return -len(parsed_function_list) / MAX_FUNCTIONS


class LengthReward:
    """Reward fewer functions."""

    def __init__(self, available_functions: FunctionDefSet) -> None:
        self.available_functions = available_functions
        self.__name__ = LENGTH_REWARD_NAME

    def __call__(
        self,
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        parsed_function_lists = [
            self.available_functions.parse_string(completion)
            for completion in completions
        ]
        rewards = [
            length_reward_single(parsed_function_list)
            for parsed_function_list in parsed_function_lists
        ]
        return rewards


class InductionReward:
    """Length-dependent accuracy reward for GRPO training."""

    def __init__(
        self,
        verifier_id_to_fn: dict[int, Callable[[str], tuple[float, int]]],
        verifier_id_to_ground_truth_length: dict[int, int],
        length_penalty_strength: float = 0.1,
        observer: RewardMetricsObserver | None = None,
    ) -> None:
        self.verifier_id_to_fn = verifier_id_to_fn
        self.verifier_id_to_ground_truth_length = verifier_id_to_ground_truth_length
        self.length_penalty_strength = length_penalty_strength
        self.observer = observer
        # Add __name__ attribute for compatibility with GRPOTrainer
        self.__name__ = INDUCTION_REWARD_NAME

    def __call__(
        self,
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Make the class callable to match GRPOTrainer interface.

        GRPOTrainer calls reward functions with completions and any additional
        dataset columns as keyword arguments. We extract verifier_id from kwargs.
        """
        # Extract verifier_id from kwargs (passed by GRPOTrainer from dataset)
        verifier_ids = kwargs.get("verifier_id")
        if verifier_ids is None:
            raise ValueError("verifier_id not found in dataset or kwargs")

        return self.induction_reward(completions, verifier_ids, **kwargs)

    def induction_reward(
        self,
        completions: list[str],
        verifier_ids: list[int],
        **kwargs,
    ) -> list[float]:
        """
        Length-dependent accuracy reward based on GRPO-LEAD paper.

        For correct responses, applies exponential decay based on standardized length.
        For incorrect responses, applies explicit negative penalty.
        """
        # Get correctness scores and function counts from verifier functions
        verifier_results = [
            self.verifier_id_to_fn[verifier_id](completion)
            for verifier_id, completion in zip(verifier_ids, completions, strict=False)
        ]

        # Unpack the tuples
        correctness_scores = [result[0] for result in verifier_results]
        function_counts = [result[1] for result in verifier_results]

        # Get ground truth function counts and calculate ratios
        ground_truth_counts = [
            self.verifier_id_to_ground_truth_length[verifier_id]
            for verifier_id in verifier_ids
        ]
        function_count_ratios = [
            pred_count / gt_count if gt_count > 0 else 0.0
            for pred_count, gt_count in zip(
                function_counts, ground_truth_counts, strict=False
            )
        ]

        # Calculate success rate and average function count for observer
        num_correct = sum(1 for score in correctness_scores if score > 0.5)
        success_rate = (
            num_correct / len(correctness_scores) if correctness_scores else 0.0
        )
        avg_function_count = (
            sum(function_counts) / len(function_counts) if function_counts else 0.0
        )
        avg_function_count_ratio = (
            sum(function_count_ratios) / len(function_count_ratios)
            if function_count_ratios
            else 0.0
        )

        # Notify observer if present
        if self.observer:
            self.observer.on_batch_processed(
                success_rate=success_rate,
                avg_function_count=avg_function_count,
                function_counts=function_counts,
                correctness_scores=correctness_scores,
                avg_function_count_ratio=avg_function_count_ratio,
                function_count_ratios=function_count_ratios,
            )

        # Get correct responses and their lengths
        correct_indices = [
            i for i, score in enumerate(correctness_scores) if score > 0.5
        ]

        if len(correct_indices) == 0:
            # No correct responses, apply penalty to all
            return [-1.0 for _ in correctness_scores]

        # Calculate statistics for correct responses only
        correct_lengths = [function_counts[i] for i in correct_indices]
        mean_length = sum(correct_lengths) / len(correct_lengths)

        # Calculate standard deviation
        if len(correct_lengths) > 1:
            variance = sum((l - mean_length) ** 2 for l in correct_lengths) / len(
                correct_lengths
            )
            std_length = math.sqrt(variance)
        else:
            std_length = 1.0  # Default when only one correct response

        rewards = []
        for _i, (score, length) in enumerate(
            zip(correctness_scores, function_counts, strict=False)
        ):
            if score > 0.5:  # Correct response
                # Standardized length deviation
                z = (length - mean_length) / (std_length + 1e-8)
                # Exponential decay for length penalty
                length_factor = math.exp(-self.length_penalty_strength * z)
                rewards.append(score * length_factor)
            else:  # Incorrect response
                rewards.append(-1.0)  # Explicit negative penalty

        return rewards


class ProposerReward:
    def __init__(
        self,
        trajectory_solver: TrajectorySolver,
        solver_attempts: int,
        available_functions: FunctionDefSet,
        observer: ProposerMetricsObserver | None = None,
    ):
        self.trajectory_solver = trajectory_solver
        self.solver_attempts = solver_attempts
        if not available_functions:
            raise ValueError("available_functions must be provided")
        self.available_functions = available_functions
        self.observer = observer
        # Add __name__ attribute for compatibility with GRPOTrainer
        self.__name__ = "ProposerReward"
        # Store latest sample results for logging
        self.latest_sample_results = []

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        eval_result = evaluate_responses(
            responses=completions,
            available_functions=self.available_functions,
            solver_model=self.trajectory_solver,
            solver_attempts=self.solver_attempts,
            num_samples=None,
        )

        # Store sample results for logging
        self.latest_sample_results = eval_result.sample_results

        rewards = []
        parse_successes = 0
        solver_successes = 0
        non_zero_variance_count = 0
        function_counts = []

        for sample_result in eval_result.sample_results:
            if sample_result.parse_success:
                parse_successes += 1
                function_counts.append(
                    len(sample_result.attempted_function_deflists[0])
                    if sample_result.attempted_function_deflists
                    else 0
                )

                # Check if solver succeeded on this proposal
                if hasattr(sample_result, "solve_rate"):
                    if sample_result.solve_rate > 0:
                        solver_successes += 1
                    # Reward proposals with intermediate difficulty (0 < solve_rate < 1.0)
                    if 0 < sample_result.solve_rate < 1.0:
                        non_zero_variance_count += 1
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(-1.0)
                function_counts.append(0)

        # Calculate metrics for observer
        total_samples = len(completions)
        parse_rate = parse_successes / total_samples if total_samples > 0 else 0.0
        solver_success_rate = (
            solver_successes / total_samples if total_samples > 0 else 0.0
        )
        frac_non_zero_std = (
            non_zero_variance_count / total_samples if total_samples > 0 else 0.0
        )
        avg_function_count = (
            sum(function_counts) / len(function_counts) if function_counts else 0.0
        )

        # Notify observer if present
        if self.observer:
            self.observer.on_proposer_batch_processed(
                parse_rate=parse_rate,
                solver_success_rate=solver_success_rate,
                frac_non_zero_std=frac_non_zero_std,
                rewards=rewards,
                avg_function_count=avg_function_count,
            )

        return rewards
