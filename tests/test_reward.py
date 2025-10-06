import pytest

from tests.test_evaluate_proposer import MockTokenGenerator
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import create_token_solver
from wandering_light.training.data_generator import induction_dataset_rl
from wandering_light.training.reward import (
    INDUCTION_REWARD_NAME,
    InductionReward,
    LengthReward,
    ProposerReward,
)
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList


class TestLengthReward:
    """Test suite for LengthReward class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample functions for testing
        func1 = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )
        func2 = FunctionDef(
            name="dec",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x - 1",
        )
        func3 = FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        )

        self.available_functions = FunctionDefSet([func1, func2, func3])
        self.length_reward = LengthReward(self.available_functions)

    def test_empty_completion(self):
        """Test reward for empty completion."""
        completions = [""]
        rewards = self.length_reward(completions)

        assert len(rewards) == 1
        assert (
            rewards[0] == -1
        )  # Empty completion indicates parsing failure, so reward is -1
        assert isinstance(rewards[0], int | float)

    def test_batch_completions(self):
        """Test reward for batch of completions."""
        completions = ["", "inc", "inc, dec", "inc, dec, double"]
        rewards = self.length_reward(completions)

        assert len(rewards) == 4
        assert rewards == [
            -1,  # Empty completion (parsing failure)
            -0.2,  # 1 function: -1/5 = -0.2
            -0.4,  # 2 functions: -2/5 = -0.4
            -0.6,  # 3 functions: -3/5 = -0.6
        ]  # Rewards decrease with more functions, empty completion penalized most
        assert all(isinstance(r, int | float) for r in rewards)

    def test_invalid_function_name(self):
        """Test reward for completion with invalid function name."""
        completions = ["invalid_function"]
        rewards = self.length_reward(completions)

        assert len(rewards) == 1
        assert (
            rewards[0] == -1
        )  # Invalid function causes parsing failure, so reward is -1
        assert isinstance(rewards[0], int | float)

    def test_mixed_valid_invalid_functions(self):
        """Test reward for completion with mix of valid and invalid functions."""
        completions = ["inc, invalid_function, dec"]
        rewards = self.length_reward(completions)

        assert len(rewards) == 1
        # If any function is invalid, parse_string returns empty FunctionDefList, so reward is -1
        assert rewards[0] == -1
        assert isinstance(rewards[0], int | float)

    def test_kwargs_ignored(self):
        """Test that additional kwargs are ignored."""
        completions = ["inc"]
        rewards = self.length_reward(completions, extra_param="ignored")

        assert len(rewards) == 1
        assert rewards[0] == -0.2
        assert isinstance(rewards[0], int | float)


class TestInductionReward:
    """Test suite for InductionReward class."""

    def setup_method(self):
        """Set up test fixtures."""

        # Create mock verifier functions that return (correctness, function_count) tuples
        def mock_verifier_0(completion):
            if completion == "correct_0":
                return (1.0, 1)  # Correct, 1 function
            else:
                return (0.0, 10)  # Incorrect, high penalty count

        def mock_verifier_1(completion):
            if completion == "correct_1":
                return (1.0, 2)  # Correct, 2 functions
            else:
                return (0.0, 10)  # Incorrect, high penalty count

        def mock_verifier_2(completion):
            if "partial" in completion:
                return (0.5, 3)  # Partially correct, 3 functions
            else:
                return (0.0, 10)  # Incorrect, high penalty count

        self.verifier_id_to_fn = {
            0: mock_verifier_0,
            1: mock_verifier_1,
            2: mock_verifier_2,
        }

        # Create ground truth lengths for each verifier
        self.verifier_id_to_ground_truth_length = {
            0: 2,  # Ground truth for verifier 0 has 2 functions
            1: 3,  # Ground truth for verifier 1 has 3 functions
            2: 4,  # Ground truth for verifier 2 has 4 functions
        }

        # Test with different reward strategies
        self.induction_reward = InductionReward(
            self.verifier_id_to_fn,
            self.verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
        )

    def test_init(self):
        """Test InductionReward initialization."""
        assert self.induction_reward.verifier_id_to_fn == self.verifier_id_to_fn
        assert self.induction_reward.__name__ == INDUCTION_REWARD_NAME

    def test_call_interface(self):
        """Test the callable interface matches GRPOTrainer expectations."""
        completions = ["correct_0", "wrong"]
        verifier_ids = [0, 1]

        rewards = self.induction_reward(completions, verifier_id=verifier_ids)

        assert len(rewards) == 2
        assert rewards[0] == 1.0  # correct_0 is correct for verifier 0
        assert rewards[1] == -1.0  # wrong is incorrect (explicit penalty)
        assert all(isinstance(r, float) for r in rewards)

    def test_missing_verifier_id_raises_error(self):
        """Test that missing verifier_id raises ValueError."""
        completions = ["test"]

        with pytest.raises(ValueError, match="verifier_id not found"):
            self.induction_reward(completions)

    def test_induction_reward_direct(self):
        """Test the induction_reward method directly."""
        completions = ["correct_0", "correct_1", "wrong", "partial_match"]
        verifier_ids = [0, 1, 0, 2]

        rewards = self.induction_reward.induction_reward(completions, verifier_ids)

        assert len(rewards) == 4
        # With length_dependent_accuracy, correct completions get normalized rewards
        # based on standardized length within the group
        assert rewards[0] > 0  # correct_0 should be positive
        assert rewards[1] > 0  # correct_1 should be positive
        assert rewards[2] == -1.0  # wrong gets explicit negative penalty
        assert rewards[3] == -1.0  # partial (0.5) is below threshold, gets penalty
        assert all(isinstance(r, float) for r in rewards)

    def test_length_penalty_strength(self):
        """Test that different length penalty strengths affect rewards correctly."""
        completions = ["correct_0", "correct_1"]
        verifier_ids = [0, 1]

        # Test with different penalty strengths
        weak_penalty = InductionReward(
            self.verifier_id_to_fn,
            self.verifier_id_to_ground_truth_length,
            length_penalty_strength=0.01,
        )
        strong_penalty = InductionReward(
            self.verifier_id_to_fn,
            self.verifier_id_to_ground_truth_length,
            length_penalty_strength=0.5,
        )

        weak_rewards = weak_penalty(completions, verifier_id=verifier_ids)
        strong_rewards = strong_penalty(completions, verifier_id=verifier_ids)

        assert len(weak_rewards) == 2
        assert len(strong_rewards) == 2
        assert all(isinstance(r, float) for r in weak_rewards)
        assert all(isinstance(r, float) for r in strong_rewards)
        # Both should be positive since they're correct
        assert all(r > 0 for r in weak_rewards)
        assert all(r > 0 for r in strong_rewards)

    def test_global_function_count_tracking(self):
        """Test that function count tracking works correctly via observer pattern."""

        # Create a mock observer to capture metrics
        class MockObserver:
            def __init__(self):
                self.last_success_rate = None
                self.last_avg_function_count = None
                self.last_function_counts = None
                self.last_correctness_scores = None

            def on_batch_processed(
                self,
                success_rate,
                avg_function_count,
                function_counts,
                correctness_scores,
                avg_function_count_ratio,
                function_count_ratios,
            ):
                self.last_success_rate = success_rate
                self.last_avg_function_count = avg_function_count
                self.last_function_counts = function_counts
                self.last_correctness_scores = correctness_scores

        # Create reward function with observer
        mock_observer = MockObserver()
        induction_reward_with_observer = InductionReward(
            self.verifier_id_to_fn,
            self.verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=mock_observer,
        )

        completions = ["correct_0", "correct_1", "wrong"]
        verifier_ids = [0, 1, 0]

        # Before calling reward function, observer should have no data
        assert mock_observer.last_avg_function_count is None

        # Call reward function
        induction_reward_with_observer(completions, verifier_id=verifier_ids)

        # Observer should now have received the metrics
        assert mock_observer.last_avg_function_count is not None
        assert mock_observer.last_avg_function_count > 0
        assert mock_observer.last_success_rate is not None
        assert mock_observer.last_function_counts is not None
        assert len(mock_observer.last_function_counts) == len(completions)

        print(f"Observer avg function count: {mock_observer.last_avg_function_count}")
        print(f"Observer function counts: {mock_observer.last_function_counts}")

    def test_success_rate_tracking(self):
        """Test that success rate tracking works correctly via observer pattern."""

        # Create a mock observer to capture metrics
        class MockObserver:
            def __init__(self):
                self.last_success_rate = None
                self.last_avg_function_count = None

            def on_batch_processed(
                self,
                success_rate,
                avg_function_count,
                function_counts,
                correctness_scores,
                avg_function_count_ratio,
                function_count_ratios,
            ):
                self.last_success_rate = success_rate
                self.last_avg_function_count = avg_function_count

        # Create reward function with observer
        mock_observer = MockObserver()
        induction_reward_with_observer = InductionReward(
            self.verifier_id_to_fn,
            self.verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=mock_observer,
        )

        # Test with mixed correct/incorrect responses
        completions = ["correct_0", "correct_1", "wrong"]
        verifier_ids = [0, 1, 0]

        # Call reward function
        induction_reward_with_observer(completions, verifier_id=verifier_ids)

        # Success rate should be 2/3 (2 correct out of 3)
        expected_success_rate = 2.0 / 3.0
        assert abs(mock_observer.last_success_rate - expected_success_rate) < 0.001

        # Test with all incorrect responses
        completions_all_wrong = ["wrong", "wrong", "wrong"]
        verifier_ids_all_wrong = [0, 0, 0]
        induction_reward_with_observer(
            completions_all_wrong, verifier_id=verifier_ids_all_wrong
        )

        # Success rate should be 0
        assert mock_observer.last_success_rate == 0.0

        # Test with all correct responses
        completions_all_correct = ["correct_0", "correct_1", "correct_0"]
        verifier_ids_all_correct = [0, 1, 0]
        induction_reward_with_observer(
            completions_all_correct, verifier_id=verifier_ids_all_correct
        )

        # Success rate should be 1.0
        assert mock_observer.last_success_rate == 1.0

        print(f"Mixed success rate: {expected_success_rate}")
        print("All wrong success rate: 0.0")
        print("All correct success rate: 1.0")

    def test_observer_pattern_end_to_end(self):
        """Test that the observer pattern works end-to-end with callback."""
        # Import here to avoid circular imports in testing
        from wandering_light.training.rl_grpo import RewardEvaluationCallback

        # Create a callback that will act as observer
        callback = RewardEvaluationCallback(use_wandb=False)

        # Create ground truth lengths mapping
        verifier_id_to_ground_truth_length = {
            0: 2,  # Ground truth for verifier 0 has 2 functions
            1: 3,  # Ground truth for verifier 1 has 3 functions
        }

        # Create reward function with callback as observer
        induction_reward = InductionReward(
            self.verifier_id_to_fn,
            verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=callback,
        )

        # Process a batch
        completions = ["correct_0", "correct_1", "wrong"]
        verifier_ids = [0, 1, 0]
        induction_reward(completions, verifier_id=verifier_ids)

        # Verify callback received the metrics
        assert callback._last_success_rate > 0
        assert callback._last_avg_function_count > 0

        # Verify the success rate is correct (2/3)
        expected_success_rate = 2.0 / 3.0
        assert abs(callback._last_success_rate - expected_success_rate) < 0.001

        print(
            f"End-to-end test: success_rate={callback._last_success_rate}, avg_function_count={callback._last_avg_function_count}"
        )

    def test_dual_level_metrics_logging(self):
        """Test that both immediate and interval-averaged metrics are tracked correctly."""
        # Import here to avoid circular imports
        from wandering_light.training.rl_grpo import RewardEvaluationCallback

        # Create callback with wandb disabled for testing
        callback = RewardEvaluationCallback(
            use_wandb=False, eval_steps=3
        )  # Log every 3 steps

        # Create ground truth lengths mapping
        verifier_id_to_ground_truth_length = {
            0: 2,  # Ground truth for verifier 0 has 2 functions
            1: 3,  # Ground truth for verifier 1 has 3 functions
        }

        # Create reward function with callback as observer
        induction_reward = InductionReward(
            self.verifier_id_to_fn,
            verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=callback,
        )

        # Simulate multiple batches with different success rates
        batch_1_completions = [
            "correct_0",
            "wrong",
            "wrong",
        ]  # 1/3 = 0.333 success rate
        batch_2_completions = [
            "correct_0",
            "correct_1",
            "wrong",
        ]  # 2/3 = 0.667 success rate
        batch_3_completions = [
            "correct_0",
            "correct_1",
            "correct_0",
        ]  # 3/3 = 1.0 success rate

        verifier_ids = [0, 1, 0]

        # Process batches (this will accumulate metrics)
        induction_reward(batch_1_completions, verifier_id=verifier_ids)
        induction_reward(batch_2_completions, verifier_id=verifier_ids)
        induction_reward(batch_3_completions, verifier_id=verifier_ids)

        # Verify accumulation
        assert len(callback._batch_metrics.success_rates) == 3
        assert len(callback._batch_metrics.function_counts) == 3

        # Check individual batch success rates
        assert (
            abs(callback._batch_metrics.success_rates[0] - (1.0 / 3.0)) < 0.001
        )  # Batch 1: 33.3%
        assert (
            abs(callback._batch_metrics.success_rates[1] - (2.0 / 3.0)) < 0.001
        )  # Batch 2: 66.7%
        assert (
            abs(callback._batch_metrics.success_rates[2] - 1.0) < 0.001
        )  # Batch 3: 100%

        # Check that immediate metrics reflect the last batch
        assert (
            abs(callback._last_success_rate - 1.0) < 0.001
        )  # Should be 100% from batch 3

        # Calculate expected interval average: (0.333 + 0.667 + 1.0) / 3 = 0.667
        expected_interval_avg = (1.0 / 3.0 + 2.0 / 3.0 + 1.0) / 3.0

        # Manually trigger the interval calculation like on_log would do
        if callback._batch_metrics.success_rates:
            interval_avg_success_rate = sum(
                callback._batch_metrics.success_rates
            ) / len(callback._batch_metrics.success_rates)
            interval_avg_function_count = sum(
                callback._batch_metrics.function_counts
            ) / len(callback._batch_metrics.function_counts)

            # Verify interval averages
            assert abs(interval_avg_success_rate - expected_interval_avg) < 0.001
            assert interval_avg_function_count > 0

            print(f"Immediate success rate: {callback._last_success_rate:.3f}")
            print(f"Interval avg success rate: {interval_avg_success_rate:.3f}")
            print(f"Expected interval avg: {expected_interval_avg:.3f}")
            print(
                f"Individual batch rates: {[round(r, 3) for r in callback._batch_metrics.success_rates]}"
            )

    def test_function_count_ratio_tracking(self):
        """Test that function count ratio tracking works correctly via observer pattern."""

        # Create a mock observer to capture metrics
        class MockObserver:
            def __init__(self):
                self.last_success_rate = None
                self.last_avg_function_count = None
                self.last_avg_function_count_ratio = None
                self.last_function_count_ratios = None

            def on_batch_processed(
                self,
                success_rate,
                avg_function_count,
                function_counts,
                correctness_scores,
                avg_function_count_ratio,
                function_count_ratios,
            ):
                self.last_success_rate = success_rate
                self.last_avg_function_count = avg_function_count
                self.last_avg_function_count_ratio = avg_function_count_ratio
                self.last_function_count_ratios = function_count_ratios

        # Create ground truth lengths mapping
        verifier_id_to_ground_truth_length = {
            0: 2,  # Ground truth for verifier 0 has 2 functions
            1: 3,  # Ground truth for verifier 1 has 3 functions
        }

        # Create reward function with observer
        mock_observer = MockObserver()
        induction_reward_with_observer = InductionReward(
            self.verifier_id_to_fn,
            verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=mock_observer,
        )

        # Test with known function counts
        # correct_0 has 1 function, ground truth is 2 -> ratio = 0.5
        # correct_1 has 2 functions, ground truth is 3 -> ratio = 0.67
        # wrong has 10 functions (penalty), ground truth is 2 -> ratio = 5.0
        completions = ["correct_0", "correct_1", "wrong"]
        verifier_ids = [0, 1, 0]

        # Call reward function
        induction_reward_with_observer(completions, verifier_id=verifier_ids)

        # Observer should now have received the metrics including ratios
        assert mock_observer.last_avg_function_count_ratio is not None
        assert mock_observer.last_function_count_ratios is not None
        assert len(mock_observer.last_function_count_ratios) == len(completions)

        # Check that ratios are calculated correctly
        expected_ratios = [0.5, 2.0 / 3.0, 5.0]  # [1/2, 2/3, 10/2]
        for i, expected_ratio in enumerate(expected_ratios):
            actual_ratio = mock_observer.last_function_count_ratios[i]
            assert abs(actual_ratio - expected_ratio) < 0.01

        # Check average ratio
        expected_avg_ratio = sum(expected_ratios) / len(expected_ratios)
        assert (
            abs(mock_observer.last_avg_function_count_ratio - expected_avg_ratio) < 0.01
        )

        print(f"Function count ratios: {mock_observer.last_function_count_ratios}")
        print(
            f"Average function count ratio: {mock_observer.last_avg_function_count_ratio}"
        )

    def test_function_count_ratio_edge_cases(self):
        """Test function count ratio handles edge cases correctly."""

        class MockObserver:
            def __init__(self):
                self.last_avg_function_count_ratio = None
                self.last_function_count_ratios = None

            def on_batch_processed(
                self,
                success_rate,
                avg_function_count,
                function_counts,
                correctness_scores,
                avg_function_count_ratio,
                function_count_ratios,
            ):
                self.last_avg_function_count_ratio = avg_function_count_ratio
                self.last_function_count_ratios = function_count_ratios

        # Create ground truth lengths mapping with zero length (edge case)
        verifier_id_to_ground_truth_length = {
            0: 0,  # Edge case: zero ground truth length
            1: 1,  # Normal case
        }

        mock_observer = MockObserver()
        induction_reward_with_observer = InductionReward(
            self.verifier_id_to_fn,
            verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=mock_observer,
        )

        completions = ["correct_0", "correct_1"]
        verifier_ids = [0, 1]

        # Call reward function
        induction_reward_with_observer(completions, verifier_id=verifier_ids)

        # Check that zero ground truth length is handled correctly (should give ratio of 0.0)
        assert mock_observer.last_function_count_ratios[0] == 0.0  # 1/0 -> 0.0
        assert mock_observer.last_function_count_ratios[1] == 2.0  # 2/1 = 2.0

        print(f"Edge case ratios: {mock_observer.last_function_count_ratios}")

    def test_end_to_end_with_function_count_ratio(self):
        """Test that the function count ratio works end-to-end with callback."""
        # Import here to avoid circular imports in testing
        from wandering_light.training.rl_grpo import RewardEvaluationCallback

        # Create a callback that will act as observer
        callback = RewardEvaluationCallback(use_wandb=False)

        # Create ground truth lengths mapping
        verifier_id_to_ground_truth_length = {
            0: 2,  # Ground truth for verifier 0 has 2 functions
            1: 3,  # Ground truth for verifier 1 has 3 functions
        }

        # Create reward function with callback as observer
        induction_reward = InductionReward(
            self.verifier_id_to_fn,
            verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
            observer=callback,
        )

        # Process a batch
        completions = ["correct_0", "correct_1", "wrong"]
        verifier_ids = [0, 1, 0]
        induction_reward(completions, verifier_id=verifier_ids)

        # Verify callback received the metrics including function count ratio
        assert callback._last_success_rate > 0
        assert callback._last_avg_function_count > 0
        assert callback._last_avg_function_count_ratio > 0

        # Verify the success rate is correct (2/3)
        expected_success_rate = 2.0 / 3.0
        assert abs(callback._last_success_rate - expected_success_rate) < 0.001

        print(
            f"End-to-end test: success_rate={callback._last_success_rate}, "
            f"avg_function_count={callback._last_avg_function_count}, "
            f"avg_function_count_ratio={callback._last_avg_function_count_ratio}"
        )


def test_induction_reward_0_invalid_completion():
    """Test InductionReward with invalid completions (legacy test)."""
    completions = ["double", "inc", "inc"]
    dataset, verifier_id_to_fn, verifier_id_to_ground_truth_length = (
        induction_dataset_rl(length_counts={2: 3})
    )
    induction_reward = InductionReward(
        verifier_id_to_fn, verifier_id_to_ground_truth_length
    )

    # Extract verifier_ids correctly from the dataset
    subset = dataset.select(range(3))
    verifier_ids = subset["verifier_id"]
    rewards = induction_reward(completions, verifier_id=verifier_ids)

    # Check that we get the right number of rewards
    assert len(rewards) == len(completions), (
        f"Expected {len(completions)} rewards, got {len(rewards)}"
    )

    # With the new reward system, incorrect completions get -1.0 penalty
    assert all(reward == -1.0 for reward in rewards), (
        f"Expected all rewards to be -1.0, got {rewards}"
    )

    # Check that return type is list of floats
    assert isinstance(rewards, list), "Rewards should be a list"
    assert all(isinstance(reward, float) for reward in rewards), (
        "All rewards should be floats"
    )


def test_induction_reward_1_correct_completion():
    """Test InductionReward with correct completions (legacy test)."""
    dataset, verifier_id_to_fn, verifier_id_to_ground_truth_length = (
        induction_dataset_rl(length_counts={1: 3})
    )
    induction_reward = InductionReward(
        verifier_id_to_fn, verifier_id_to_ground_truth_length
    )

    # Extract verifier_ids and completions correctly from the dataset
    subset = dataset.select(range(3))
    verifier_ids = subset["verifier_id"]
    completions = subset["completion"]
    rewards = induction_reward(completions, verifier_id=verifier_ids)

    # Check that we get the right number of rewards
    assert len(rewards) == len(completions), (
        f"Expected {len(completions)} rewards, got {len(rewards)}"
    )

    # Check that all rewards are positive for correct completions
    # (they should be 1.0 or slightly different due to length normalization)
    assert all(reward > 0 for reward in rewards), (
        f"Expected all rewards to be positive, got {rewards}"
    )

    # Check that return type is list of floats
    assert isinstance(rewards, list), "Rewards should be a list"
    assert all(isinstance(reward, float) for reward in rewards), (
        "All rewards should be floats"
    )


class TestProposerReward:
    """Test suite for ProposerReward class."""

    def test_proposer_reward_with_correct_completion(self):
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )
        first_trajectory_function_names = inc_func.name
        trajectory_spec = TrajectorySpec(
            input_list=TypedList([1, 2, 3], int),
            function_defs=FunctionDefList([inc_func]),
        )
        completions = ["invalid gibberish"] + [trajectory_spec.__repr__()] * 4
        solver_response_2 = [
            first_trajectory_function_names
        ] * 3  # valid data generation, but no variance, too easy
        solver_response_3 = [
            first_trajectory_function_names,
            first_trajectory_function_names,
            "invalid gibberish",
        ]
        solver_response_4 = [
            "invalid gibberish",
            first_trajectory_function_names,
            "invalid gibberish",
        ]
        solver_response_5 = [
            "invalid gibberish"
        ] * 3  # valid data generation, but no variance, too hard
        trajectory_solver = create_token_solver(
            MockTokenGenerator(
                solver_response_2
                + solver_response_3
                + solver_response_4
                + solver_response_5
            )
        )
        rewarder = ProposerReward(
            trajectory_solver,
            solver_attempts=3,
            available_functions=FunctionDefSet([inc_func]),
        )
        rewards = rewarder(completions)
        assert rewards == [-1.0, 0.0, 1.0, 1.0, 0.0]
