"""
Test for rl_grpo_main with model instance support.
Uses a small transformer model and runs for a few steps to verify everything works.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerControl,
)

# Import the GroundTruthTokenGenerator from the existing test file
# Import what we need
from wandering_light.training.rl_grpo import RewardEvaluationCallback, rl_grpo_main


class TestRLGRPOMain:
    """Test suite for rl_grpo_main function with model instance support."""

    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        """Create a tiny model and tokenizer for testing."""
        # Use a very small model for testing
        model_name = "roneneldan/TinyStories-1M"

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Ensure we have a pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer, model_name
        except Exception as e:
            pytest.skip(f"Could not load model {model_name}: {e}")

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_rl_grpo_main_with_model_instance(
        self, tiny_model_and_tokenizer, temp_output_dir
    ):
        """Test rl_grpo_main with a provided model instance."""
        model, tokenizer, model_name = tiny_model_and_tokenizer

        # Mock the heavy components to avoid config and training issues
        with (
            patch("wandering_light.training.rl_grpo.wandb") as mock_wandb,
            patch("wandering_light.training.rl_grpo.GRPOConfig") as mock_grpo_config,
            patch("wandering_light.training.rl_grpo.GRPOTrainer") as mock_trainer_class,
        ):
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.log = MagicMock()
            mock_wandb.finish = MagicMock()

            # Create a mock config
            mock_config = MagicMock()
            mock_grpo_config.return_value = mock_config

            # Create a mock trainer that accepts our model
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = None
            mock_trainer.save_model.return_value = None
            mock_trainer.model = model  # Set the model attribute
            mock_trainer_class.return_value = mock_trainer

            # Run rl_grpo_main with model instance for minimal steps
            trainer = rl_grpo_main(
                model_name=model_name,  # Still needed for tokenizer
                model=model,  # Provide the actual model instance
                full_run=False,  # Use minimal dataset
                eval_steps=2,  # Evaluate very frequently
                eval_samples=5,  # Use very few evaluation samples
                num_train_epochs=0.1,  # Train for minimal time
                learning_rate=1e-5,
                max_length=32,  # Short sequences
                per_device_train_batch_size=2,  # Small batch size
                wandb_project="test-project",
                wandb_run_name="test-run",
                length_penalty_strength=0.1,
            )

            # Verify trainer was created successfully
            assert trainer is not None

            # Verify the GRPOTrainer was called with the model instance (not model_name)
            mock_trainer_class.assert_called_once()
            call_args = mock_trainer_class.call_args
            assert call_args[1]["model"] is model

            # Verify wandb was called with correct config
            mock_wandb.init.assert_called_once()
            call_args = mock_wandb.init.call_args
            config = call_args[1]["config"]
            assert config["model_provided"] is True
            assert config["model_name"] == model_name

            print("✅ rl_grpo_main successfully ran with model instance")

    def test_rl_grpo_main_model_name_only(self, temp_output_dir):
        """Test rl_grpo_main with model_name only (existing functionality)."""
        model_name = "roneneldan/TinyStories-1M"

        # Mock all the heavy components to avoid config and training issues
        with (
            patch("wandering_light.training.rl_grpo.wandb") as mock_wandb,
            patch("wandering_light.training.rl_grpo.GRPOConfig") as mock_grpo_config,
            patch("wandering_light.training.rl_grpo.GRPOTrainer") as mock_trainer_class,
            patch(
                "wandering_light.training.rl_grpo.AutoTokenizer"
            ) as mock_tokenizer_class,
        ):
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.log = MagicMock()
            mock_wandb.finish = MagicMock()

            # Create a mock config
            mock_config = MagicMock()
            mock_grpo_config.return_value = mock_config

            # Create a mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Create a mock trainer
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = None
            mock_trainer.save_model.return_value = None
            mock_trainer_class.return_value = mock_trainer

            try:
                # Run rl_grpo_main with model_name only
                trainer = rl_grpo_main(
                    model_name=model_name,
                    model=None,  # Don't provide model instance
                    full_run=False,
                    eval_steps=2,
                    eval_samples=5,
                    num_train_epochs=0.1,
                    learning_rate=1e-5,
                    max_length=32,
                    per_device_train_batch_size=2,
                    wandb_project="test-project",
                    wandb_run_name="test-run",
                    length_penalty_strength=0.1,
                )

                # Verify trainer was created successfully
                assert trainer is not None

                # Verify the GRPOTrainer was called with model_name (not a model instance)
                mock_trainer_class.assert_called_once()
                call_args = mock_trainer_class.call_args
                assert call_args[1]["model"] == model_name

                # Verify wandb was called with correct config
                mock_wandb.init.assert_called_once()
                call_args = mock_wandb.init.call_args
                config = call_args[1]["config"]
                assert config["model_provided"] is False
                assert config["model_name"] == model_name

                print("✅ rl_grpo_main successfully ran with model_name only")

            except Exception as e:
                if "model" in str(e).lower() or "checkpoint" in str(e).lower():
                    pytest.skip(f"Model loading failed: {e}")
                else:
                    raise

    def test_reward_callback_integration(self):
        """Test that RewardEvaluationCallback works correctly."""
        # Create callback with wandb disabled
        callback = RewardEvaluationCallback(
            eval_steps=1,
            num_samples=5,
            budget=1,
            use_wandb=False,
        )

        # Verify callback was created successfully
        assert callback is not None
        assert callback.eval_steps == 1
        assert callback.num_samples == 5
        assert callback.budget == 1
        assert callback.use_wandb is False

        # Test the observer interface
        callback.on_batch_processed(
            success_rate=0.8,
            avg_function_count=2.5,
            function_counts=[2, 3, 2, 3, 2],
            correctness_scores=[1.0, 1.0, 1.0, 1.0, 0.0],
            avg_function_count_ratio=1.25,
            function_count_ratios=[1.0, 1.5, 1.0, 1.5, 1.0],
        )

        # Verify metrics were stored
        assert callback._last_success_rate == 0.8
        assert callback._last_avg_function_count == 2.5
        assert len(callback._batch_metrics.success_rates) == 1
        assert len(callback._batch_metrics.function_counts) == 1

        print("✅ RewardEvaluationCallback integration test passed")

    def test_training_infrastructure_components(self):
        """Test that all training infrastructure components can be imported and initialized."""
        from wandering_light.training.data_generator import induction_dataset_rl
        from wandering_light.training.reward import InductionReward
        from wandering_light.training.rl_grpo_config import RLMetrics

        # Test RLMetrics dataclass
        metrics = RLMetrics(
            step=100,
            avg_reward=0.75,
            success_rate=0.60,
            avg_function_count=2.2,
            avg_function_count_ratio=1.1,
            kl_divergence=0.1,
            policy_loss=0.05,
            frac_reward_zero_std=0.2,
        )
        assert metrics.step == 100
        assert metrics.avg_reward == 0.75

        # Test dataset generation
        dataset, verifier_fn_dict, verifier_id_to_ground_truth_length = (
            induction_dataset_rl(length_counts={2: 10})
        )
        assert len(dataset) == 10
        assert len(verifier_fn_dict) > 0

        # Test reward function creation
        reward_fn = InductionReward(
            verifier_id_to_fn=verifier_fn_dict,
            verifier_id_to_ground_truth_length=verifier_id_to_ground_truth_length,
            length_penalty_strength=0.1,
        )
        assert reward_fn is not None
        assert callable(reward_fn)

        print("✅ All training infrastructure components working correctly")

    @pytest.mark.slow
    def test_minimal_training_run(self, tiny_model_and_tokenizer):
        """Test an actual minimal training run to verify end-to-end functionality."""
        model, tokenizer, model_name = tiny_model_and_tokenizer

        # This test actually runs training for a few steps
        with patch("wandering_light.training.rl_grpo.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.log = MagicMock()
            mock_wandb.finish = MagicMock()

            try:
                trainer = rl_grpo_main(
                    model_name=model_name,
                    model=model,
                    full_run=False,
                    eval_steps=1,  # Evaluate after every step
                    eval_samples=3,  # Minimal evaluation
                    num_train_epochs=0.01,  # Very short training
                    learning_rate=1e-6,  # Small learning rate
                    max_length=16,  # Very short sequences
                    per_device_train_batch_size=8,  # Minimal batch size
                    wandb_project=None,
                    wandb_run_name=None,
                    length_penalty_strength=0.1,
                )

                # Verify training completed without errors
                assert trainer is not None
                assert hasattr(trainer, "model")

                # Check that callbacks were registered
                assert len(trainer.callback_handler.callbacks) > 0

                # Check that the reward evaluation callback is present
                reward_callbacks = [
                    cb
                    for cb in trainer.callback_handler.callbacks
                    if isinstance(cb, RewardEvaluationCallback)
                ]
                assert len(reward_callbacks) > 0

                print("✅ Minimal training run completed successfully")

            except Exception as e:
                if any(
                    keyword in str(e).lower()
                    for keyword in ["cuda", "memory", "gpu", "device"]
                ):
                    pytest.skip(f"Hardware/memory limitation: {e}")
                else:
                    # Re-raise other exceptions as they indicate real problems
                    raise

    def test_reward_callback_runs_evaluation_on_save(self, tmp_path):
        """Ensure RewardEvaluationCallback performs evaluation when a checkpoint is saved."""
        from wandering_light.function_def import (
            FunctionDef,
            FunctionDefList,
            FunctionDefSet,
        )
        from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
        from wandering_light.typed_list import TypedList

        inc = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        spec_list = TrajectorySpecList(
            [TrajectorySpec(TypedList([1]), FunctionDefList([inc]))]
        )
        fn_set = FunctionDefSet([inc])

        with (
            patch(
                "wandering_light.training.rl_grpo.load_eval_data_as_trajectories",
                return_value=(spec_list, fn_set),
            ) as mock_load,
            patch(
                "wandering_light.training.rl_grpo.evaluate_model_checkpoint_with_trajectories"
            ) as mock_eval,
        ):
            mock_result = MagicMock()
            mock_result.success_rate = 1.0
            mock_result.avg_solution_length = 1.0
            mock_result.success_count = 1
            mock_result.total_samples = 1
            mock_eval.return_value = mock_result

            callback = RewardEvaluationCallback(
                eval_steps=1, num_samples=1, budget=1, use_wandb=False
            )

            output_dir = tmp_path
            ckpt_dir = output_dir / "checkpoint-1"
            ckpt_dir.mkdir()

            args = MagicMock()
            args.output_dir = str(output_dir)
            state = MagicMock()
            state.global_step = 1
            control = TrainerControl()

            callback.on_save(args, state, control)

            mock_load.assert_called_once()
            mock_eval.assert_called_once()
            assert len(callback.eval_results) == 1
            assert callback.eval_results[0]["success_rate"] == 1.0

    def test_reward_callback_no_eval_data(self, tmp_path):
        """Callback should skip evaluation gracefully if eval data fails to load."""
        with patch(
            "wandering_light.training.rl_grpo.load_eval_data_as_trajectories",
            side_effect=Exception("fail"),
        ):
            callback = RewardEvaluationCallback(
                eval_steps=1, num_samples=1, budget=1, use_wandb=False
            )

        output_dir = tmp_path
        ckpt_dir = output_dir / "checkpoint-1"
        ckpt_dir.mkdir()

        args = MagicMock()
        args.output_dir = str(output_dir)
        state = MagicMock()
        state.global_step = 1
        control = TrainerControl()

        with patch(
            "wandering_light.training.rl_grpo.evaluate_model_checkpoint_with_trajectories"
        ) as mock_eval:
            callback.on_save(args, state, control)
            mock_eval.assert_not_called()

    def test_training_metrics_logged_with_correct_prefix(self):
        """RewardEvaluationCallback should log training metrics with the 'training/' prefix."""
        from wandering_light.trajectory import TrajectorySpecList

        dummy_specs = TrajectorySpecList([])
        dummy_fn_set = []

        with (
            patch(
                "wandering_light.training.rl_grpo.load_eval_data_as_trajectories",
                return_value=(dummy_specs, dummy_fn_set),
            ),
            patch("wandering_light.training.rl_grpo.wandb") as mock_wandb,
        ):
            mock_wandb.log = MagicMock()
            callback = RewardEvaluationCallback(
                eval_steps=1, num_samples=1, budget=1, use_wandb=True
            )

            args = MagicMock()
            state = MagicMock()
            state.global_step = 1
            control = TrainerControl()

            callback.on_batch_processed(
                success_rate=0.8,
                avg_function_count=2.0,
                function_counts=[],
                correctness_scores=[],
                avg_function_count_ratio=1.0,
                function_count_ratios=[],
            )

            logs = {
                "reward": 0.5,
                "completions/mean_length": 2.0,
                "kl": 0.1,
                "loss": 0.2,
                "frac_reward_zero_std": 0.0,
            }

            callback.on_log(args, state, control, logs)

            assert any(
                "training/avg_reward" in call.args[0]
                for call in mock_wandb.log.call_args_list
            )
            assert any(
                "training/success_rate" in call.args[0]
                for call in mock_wandb.log.call_args_list
            )

    def test_eval_metrics_logged_with_correct_prefix(self, tmp_path):
        """RewardEvaluationCallback should log eval metrics with the 'eval/' prefix."""
        from wandering_light.function_def import (
            FunctionDef,
            FunctionDefList,
            FunctionDefSet,
        )
        from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
        from wandering_light.typed_list import TypedList

        inc = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        spec_list = TrajectorySpecList(
            [TrajectorySpec(TypedList([1]), FunctionDefList([inc]))]
        )
        fn_set = FunctionDefSet([inc])

        with (
            patch(
                "wandering_light.training.rl_grpo.load_eval_data_as_trajectories",
                return_value=(spec_list, fn_set),
            ),
            patch(
                "wandering_light.training.rl_grpo.evaluate_model_checkpoint_with_trajectories"
            ) as mock_eval,
            patch("wandering_light.training.rl_grpo.wandb") as mock_wandb,
        ):
            mock_result = MagicMock()
            mock_result.success_rate = 1.0
            mock_result.avg_solution_length = 1.0
            mock_result.success_count = 1
            mock_result.total_samples = 1
            mock_eval.return_value = mock_result
            mock_wandb.log = MagicMock()

            callback = RewardEvaluationCallback(
                eval_steps=1, num_samples=1, budget=1, use_wandb=True
            )

            output_dir = tmp_path
            ckpt_dir = output_dir / "checkpoint-1"
            ckpt_dir.mkdir()

            args = MagicMock()
            args.output_dir = str(output_dir)
            state = MagicMock()
            state.global_step = 1
            control = TrainerControl()

            callback.on_save(args, state, control)

            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            assert "eval/success_rate" in logged
            assert "eval/avg_function_count" in logged
