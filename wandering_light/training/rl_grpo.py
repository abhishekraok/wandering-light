import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any

from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import GRPOConfig, GRPOTrainer

import wandb
from wandering_light.constants import DEFAULT_EVAL_FILE, DEFAULT_SOLVER_CHECKPOINT, Task
from wandering_light.evals.evaluate_proposer import evaluate_proposer
from wandering_light.evals.model_eval import evaluate_model_checkpoint_with_trajectories
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.function_def import FunctionDefSet
from wandering_light.solver import TrainedLLMTokenGenerator, create_token_solver
from wandering_light.training.data_generator import (
    induction_dataset_rl,
    proposer_dataset_rl,
)
from wandering_light.training.reward import (
    InductionReward,
    ProposerReward,
)
from wandering_light.training.rl_grpo_config import (
    BatchMetrics,
    InductionMetricsObserver,
    ProposerMetricsObserver,
    RLGRPOConfig,
    RLMetrics,
)
from wandering_light.training.wandb_utils import WandbRunLinkCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingSampleLogger:
    """Logs training samples (prompts and completions) to files in training_logs directory."""

    def __init__(self, log_interval: int = 10, task: Task = Task.INDUCTION):
        if task != Task.PROPOSER:
            raise ValueError(
                f"TrainingSampleLogger only supports proposer task, got {task}"
            )
        self.log_interval = log_interval
        self.task = task
        self.step_counter = 0

        # Create log directory in repo
        self.log_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "training_logs",
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.log_dir,
            f"rl_samples_{task.value}_{timestamp}_{os.getpid()}.jsonl",
        )
        logger.info(f"Training samples will be logged to: {self.log_file}")

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step > 0 and step % self.log_interval == 0

    def log_samples(
        self,
        step: int,
        completions: list[str],
        sample_results: list | None = None,
        rewards: list[float] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        """Log training samples to temporary file."""
        if not self.should_log(step):
            return

        timestamp = datetime.now().isoformat()

        # Create log entries
        log_entries = []
        for i in range(min(5, len(completions))):  # Log up to 5 samples
            entry = {
                "step": step,
                "timestamp": timestamp,
                "sample_index": i,
                "completion": completions[i] if i < len(completions) else None,
                "reward": rewards[i] if rewards and i < len(rewards) else None,
            }

            # Add sample result data if available
            if sample_results and i < len(sample_results):
                sample_result = sample_results[i]
                entry["parse_success"] = sample_result.parse_success
                entry["solve_rate"] = sample_result.solve_rate
                entry["attempted_function_deflists"] = [
                    [f.name for f in deflist]
                    for deflist in sample_result.attempted_function_deflists
                ]

            # Add any additional metrics
            if metrics:
                entry["metrics"] = metrics

            log_entries.append(entry)

        # Append to file
        try:
            with open(self.log_file, "a") as f:
                for entry in log_entries:
                    f.write(json.dumps(entry) + "\n")
            logger.info(
                f"Step {step}: Appended {len(log_entries)} training samples to {self.log_file}"
            )
        except Exception as e:
            logger.warning(f"Failed to log training samples: {e}")


class RewardEvaluationCallback(
    TrainerCallback, InductionMetricsObserver, ProposerMetricsObserver
):
    """Callback to compute custom rewards and track metrics during RL training."""

    def __init__(
        self,
        eval_steps: int = 500,
        num_samples: int | None = None,
        budget: int = 1,
        eval_file: str = DEFAULT_EVAL_FILE,
        use_wandb: bool = True,
        task: Task = Task.INDUCTION,
        sample_logger: TrainingSampleLogger | None = None,
    ):
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.budget = budget
        self.eval_file = eval_file
        self.trajectories = None
        self.available_functions = None
        self.metrics_history = []
        self.eval_results = []
        self.use_wandb = use_wandb
        self.task = task
        self.solver_model = None
        self.sample_logger = sample_logger

        # Store metrics received from reward function observers
        # Induction metrics
        self._last_success_rate = 0.0
        self._last_avg_function_count = 0.0
        self._last_avg_function_count_ratio = 0.0

        # Proposer metrics
        self._last_parse_rate = 0.0
        self._last_solver_success_rate = 0.0
        self._last_frac_non_zero_std = 0.0

        # Use BatchMetrics for cleaner accumulation
        self._batch_metrics = BatchMetrics()

        self._current_step = 0

        # Store latest batch data for logging
        self._latest_prompts = []
        self._latest_completions = []
        self._latest_rewards = []

        # Load evaluation data once
        self._load_eval_data()

    def _load_eval_data(self):
        """Load the evaluation data once at initialization and pre-compute trajectories."""
        try:
            self.trajectories, self.available_functions = (
                load_eval_data_as_trajectories(self.eval_file)
            )
            logger.info(
                f"Pre-computed {len(self.trajectories)} trajectories for RL eval"
            )
            logger.info(
                f"Found {len(self.available_functions)} unique functions for RL eval"
            )

        except Exception as e:
            logger.warning(f"Could not load evaluation data: {e}")
            self.trajectories = None
            self.available_functions = None

        # Load solver model for proposer task
        if self.task == Task.PROPOSER:
            if os.path.exists(DEFAULT_SOLVER_CHECKPOINT):
                self.solver_model = create_token_solver(
                    TrainedLLMTokenGenerator(
                        DEFAULT_SOLVER_CHECKPOINT, temperature=0.8
                    ),
                    budget=1,
                )
                logger.info(f"Loaded solver model from {DEFAULT_SOLVER_CHECKPOINT}")
            else:
                logger.warning(
                    "Solver checkpoint not found, skipping solver based evaluation for proposer task"
                )

    def _run_evaluation(self, model_path: str):
        """Run evaluation on the current model using pre-computed trajectories."""
        if self.trajectories is None or self.available_functions is None:
            return None

        try:
            if self.task == Task.INDUCTION:
                return evaluate_model_checkpoint_with_trajectories(
                    model_path,
                    self.trajectories,
                    self.available_functions,
                    num_samples=self.num_samples,
                    budget=self.budget,
                )
            elif self.task == Task.PROPOSER:
                # Create the model for proposer evaluation
                model = TrainedLLMTokenGenerator(model_path)
                return evaluate_proposer(
                    model,
                    trajectory_solver=self.solver_model,
                    trajectories=self.trajectories,
                    num_samples=self.num_samples or len(self.trajectories),
                )
            else:
                logger.error(f"Unknown task type: {self.task}")
                return None
        except Exception as e:
            logger.exception(f"Error during RL evaluation: {e}")
            return None

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        """Called after metrics are logged."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            # Use the logs parameter which contains the current step's metrics
            latest_logs = logs or {}

            # Only process if we have reward metrics (skip summary/runtime logs)
            if "reward" not in latest_logs:
                return control

            # Get reward metrics
            avg_reward = latest_logs.get("reward", 0.0)

            # Get KL divergence
            kl_divergence = latest_logs.get("kl", 0.0)

            # GRPO doesn't log separate policy/value losses like PPO
            # The main loss is logged as 'loss'
            policy_loss = latest_logs.get("loss", 0.0)

            # Get fraction of rewards with zero standard deviation
            frac_reward_zero_std = latest_logs.get("frac_reward_zero_std", 0.0)

            if self.task == Task.INDUCTION:
                # Calculate interval averages
                interval_avg_success_rate = (
                    self._batch_metrics.calculate_average(
                        self._batch_metrics.success_rates
                    )
                    or self._last_success_rate
                )
                interval_avg_function_count = (
                    self._batch_metrics.calculate_average(
                        self._batch_metrics.function_counts
                    )
                    or self._last_avg_function_count
                )
                interval_avg_function_count_ratio = (
                    self._batch_metrics.calculate_average(
                        self._batch_metrics.function_count_ratios
                    )
                    or self._last_avg_function_count_ratio
                )
                num_batches_in_interval = len(self._batch_metrics.success_rates) or 1

                metrics = RLMetrics(
                    step=state.global_step,
                    avg_reward=avg_reward,
                    kl_divergence=kl_divergence,
                    policy_loss=policy_loss,
                    frac_reward_zero_std=frac_reward_zero_std,
                    success_rate=interval_avg_success_rate,
                    avg_function_count=interval_avg_function_count,
                    avg_function_count_ratio=interval_avg_function_count_ratio,
                )

                # Clear accumulated metrics for next interval
                self._batch_metrics.clear_induction_metrics()

                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log(
                        {
                            # Interval averages (primary training metrics)
                            "training/avg_reward": metrics.avg_reward,
                            "training/success_rate": metrics.success_rate,
                            "training/avg_function_count": metrics.avg_function_count,
                            "training/avg_function_count_ratio": metrics.avg_function_count_ratio,
                            "training/kl_divergence": metrics.kl_divergence,
                            "training/policy_loss": metrics.policy_loss,
                            "training/frac_reward_zero_std": metrics.frac_reward_zero_std,
                            # Interval metadata
                            "interval_metadata/num_batches": num_batches_in_interval,
                            "step": state.global_step,
                        }
                    )

                logger.info(f"\n{'=' * 60}")
                logger.info(f"RL TRAINING METRICS AT STEP {state.global_step}")
                logger.info(f"{'=' * 60}")
                logger.info(f"  Average Reward: {metrics.avg_reward:.4f}")
                logger.info(f"  Success Rate: {metrics.success_rate:.2%}")
                logger.info(f"  Avg Function Count: {metrics.avg_function_count:.2f}")
                logger.info(
                    f"  Avg Function Count Ratio: {metrics.avg_function_count_ratio:.3f}"
                )
                logger.info(f"  KL Divergence: {metrics.kl_divergence:.4f}")
                logger.info(f"  Policy Loss: {metrics.policy_loss:.4f}")
                logger.info(
                    f"  Frac Reward Zero Std: {metrics.frac_reward_zero_std:.4f}"
                )
                logger.info(f"  Batches in interval: {num_batches_in_interval}")
                logger.info(f"{'=' * 60}\n")

            elif self.task == Task.PROPOSER:
                # Calculate interval averages
                interval_avg_parse_rate = (
                    self._batch_metrics.calculate_average(
                        self._batch_metrics.parse_rates
                    )
                    or self._last_parse_rate
                )
                interval_avg_solver_success_rate = (
                    self._batch_metrics.calculate_average(
                        self._batch_metrics.solver_success_rates
                    )
                    or self._last_solver_success_rate
                )
                interval_avg_frac_non_zero_std = (
                    self._batch_metrics.calculate_average(
                        self._batch_metrics.frac_non_zero_stds
                    )
                    or self._last_frac_non_zero_std
                )
                num_batches_in_interval = len(self._batch_metrics.parse_rates) or 1

                metrics = RLMetrics(
                    step=state.global_step,
                    avg_reward=avg_reward,
                    kl_divergence=kl_divergence,
                    policy_loss=policy_loss,
                    frac_reward_zero_std=frac_reward_zero_std,
                    parse_rate=interval_avg_parse_rate,
                    solver_success_rate=interval_avg_solver_success_rate,
                    frac_non_zero_std=interval_avg_frac_non_zero_std,
                )

                # Clear accumulated metrics for next interval
                self._batch_metrics.clear_proposer_metrics()

                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log(
                        {
                            # Interval averages (primary training metrics)
                            "training/avg_reward": metrics.avg_reward,
                            "training/parse_rate": metrics.parse_rate,
                            "training/solver_success_rate": metrics.solver_success_rate,
                            "training/intermediate_difficulty": metrics.frac_non_zero_std,
                            "training/kl_divergence": metrics.kl_divergence,
                            "training/policy_loss": metrics.policy_loss,
                            "training/frac_reward_zero_std": metrics.frac_reward_zero_std,
                            # Interval metadata
                            "interval_metadata/num_batches": num_batches_in_interval,
                            "step": state.global_step,
                        }
                    )

                logger.info(f"\n{'=' * 60}")
                logger.info(f"RL TRAINING METRICS AT STEP {state.global_step}")
                logger.info(f"{'=' * 60}")
                logger.info(f"  Average Reward: {metrics.avg_reward:.4f}")
                logger.info(f"  Parse Rate: {metrics.parse_rate:.2%}")
                logger.info(f"  Solver Success Rate: {metrics.solver_success_rate:.2%}")
                logger.info(
                    f"  Proposer Frac Intermediate Difficulty: {metrics.frac_non_zero_std:.2%}"
                )
                logger.info(f"  KL Divergence: {metrics.kl_divergence:.4f}")
                logger.info(f"  Policy Loss: {metrics.policy_loss:.4f}")
                logger.info(
                    f"  TRL Frac Reward Zero Std: {metrics.frac_reward_zero_std:.4f}"
                )
                logger.info(f"  Batches in interval: {num_batches_in_interval}")
                logger.info(f"{'=' * 60}\n")

            self.metrics_history.append(metrics)

        return control

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Evaluate saved checkpoints similar to SFT online evaluation."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            if os.path.exists(checkpoint_path):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"RL EVALUATION METRICS AT STEP {state.global_step}")
                logger.info(f"{'=' * 60}")

                result = self._run_evaluation(checkpoint_path)
                if result:
                    if self.task == Task.INDUCTION:
                        logger.info(
                            f"Step {state.global_step} RL Eval Success Rate: {result.success_rate:.2%}"
                        )
                        logger.info(
                            f"Avg Function Count: {result.avg_solution_length:.2f}"
                        )

                        eval_result = {
                            "step": state.global_step,
                            "success_rate": result.success_rate,
                            "avg_function_count": result.avg_solution_length,
                            "total_samples": result.total_samples,
                            "success_count": result.success_count,
                        }
                        self.eval_results.append(eval_result)

                        if self.use_wandb:
                            wandb.log(
                                {
                                    "eval/success_rate": result.success_rate,
                                    "eval/avg_function_count": result.avg_solution_length,
                                    "eval/success_count": result.success_count,
                                    "eval/total_samples": result.total_samples,
                                    "step": state.global_step,
                                }
                            )

                    elif self.task == Task.PROPOSER:
                        logger.info(f"Step {state.global_step} Results:")
                        logger.info(f"  Parse Rate: {result.parse_rate:.2%}")
                        logger.info(
                            f"  Avg Function Count: {result.avg_function_count:.2f}"
                        )
                        logger.info(
                            f"  Solver Success Rate: {result.solver_success_rate:.2%}"
                        )
                        logger.info(
                            f"  Eval Frac Intermediate Difficulty: {result.frac_non_zero_std:.2%}"
                        )

                        eval_result = {
                            "step": state.global_step,
                            "parse_rate": result.parse_rate,
                            "avg_function_count": result.avg_function_count,
                            "avg_function_count_ratio": result.avg_function_count_ratio,
                            "solver_success_rate": result.solver_success_rate,
                            "frac_non_zero_std": result.frac_non_zero_std,
                            "num_samples": result.num_samples,
                        }
                        self.eval_results.append(eval_result)

                        if self.use_wandb:
                            wandb.log(
                                {
                                    "eval/parse_rate": result.parse_rate,
                                    "eval/avg_function_count": result.avg_function_count,
                                    "eval/avg_function_count_ratio": result.avg_function_count_ratio,
                                    "eval/solver_success_rate": result.solver_success_rate,
                                    "eval/intermediate_difficulty": result.frac_non_zero_std,
                                    "eval/num_samples": result.num_samples,
                                    "step": state.global_step,
                                }
                            )
                else:
                    logger.warning(f"Step {state.global_step}: Evaluation failed")

                logger.info(f"{'=' * 60}\n")

        return control

    def on_batch_processed(
        self,
        success_rate: float,
        avg_function_count: float,
        function_counts: list[int],
        correctness_scores: list[float],
        avg_function_count_ratio: float,
        function_count_ratios: list[float],
    ) -> None:
        """Observer method called by induction reward function when batch is processed."""
        # Store for immediate access (used by on_log)
        self._last_success_rate = success_rate
        self._last_avg_function_count = avg_function_count
        self._last_avg_function_count_ratio = avg_function_count_ratio

        # Accumulate for interval averaging
        self._batch_metrics.add_induction_batch(
            success_rate, avg_function_count, avg_function_count_ratio
        )

        # Increment step counter (approximation since we don't have direct access to trainer step)
        self._current_step += 1

    def on_proposer_batch_processed(
        self,
        parse_rate: float,
        solver_success_rate: float,
        frac_non_zero_std: float,
        rewards: list[float],
        avg_function_count: float,
    ) -> None:
        """Observer method called by proposer reward function when batch is processed."""
        # Store for immediate access (used by on_log)
        self._last_parse_rate = parse_rate
        self._last_solver_success_rate = solver_success_rate
        self._last_frac_non_zero_std = frac_non_zero_std
        self._last_avg_function_count = avg_function_count

        # Accumulate for interval averaging
        self._batch_metrics.add_proposer_batch(
            parse_rate, solver_success_rate, frac_non_zero_std
        )

        # Increment step counter (approximation since we don't have direct access to trainer step)
        self._current_step += 1


class ProposerRewardWithLogging:
    """Wrapper for ProposerReward that captures prompts and completions."""

    def __init__(self, base_reward: ProposerReward, callback: RewardEvaluationCallback):
        self.base_reward = base_reward
        self.callback = callback
        self.__name__ = self.base_reward.__name__
        self.call_count = 0

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        self.call_count += 1

        # Debug logging (only in debug mode)
        if (
            self.callback.sample_logger
            and self.callback.sample_logger.log_interval == 1
        ):
            logger.debug(
                f"ProposerRewardWithLogging called: count={self.call_count}, num_completions={len(completions)}"
            )

        # Store completions for logging
        self.callback._latest_completions = completions

        # Calculate rewards
        rewards = self.base_reward(completions, **kwargs)
        self.callback._latest_rewards = rewards

        # Log samples if logger is available using call count as step
        if self.callback.sample_logger:
            # Get sample results from the base reward
            sample_results = self.base_reward.latest_sample_results

            metrics = {
                "task": "proposer",
                "avg_function_count": self.callback._last_avg_function_count,
            }
            self.callback.sample_logger.log_samples(
                step=self.call_count,
                completions=completions,
                sample_results=sample_results,
                rewards=rewards,
                metrics=metrics,
            )

        return rewards


def setup_wandb(
    wandb_run_name: str, wandb_project: str, task: Task, config: dict[str, Any]
) -> str | None:
    """Initialize wandb if enabled.

    Args:
        wandb_run_name: Run name for wandb (if None, wandb will be disabled)
        wandb_project: Project name for wandb
        task: Task type (induction or proposer)
        config: Configuration dictionary for wandb

    Returns:
        Wandb URL if enabled, None otherwise
    """
    if wandb_run_name is None:
        return None

    wandb.init(
        project=wandb_project + "_" + task.value,
        name=wandb_run_name,
        config=config,
        tags=["rl", "grpo", task],
    )
    return str(wandb.run.get_url())


def create_dataset(
    task: Task,
    full_run: bool,
    available_functions: FunctionDefSet | None = None,
) -> tuple:
    """Create dataset based on task type.

    Args:
        task: Task type (induction or proposer)
        full_run: Whether to use full dataset
        available_functions: Available functions for proposer task

    Returns:
        Tuple of (dataset, verifier_id_to_fn, verifier_id_to_ground_truth_length)
    """
    length_counts = (
        {1: 100, 2: 10_000, 3: 100_000, 4: 10_000, 5: 1_000} if full_run else {2: 10}
    )

    if task == Task.INDUCTION:
        return induction_dataset_rl(length_counts=length_counts)
    elif task == Task.PROPOSER:
        dataset = proposer_dataset_rl(
            length_counts=length_counts,
            function_pallete=available_functions,
        )
        return dataset, None, None
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: {list(Task)}")


def create_reward_function(
    task: Task,
    callback: RewardEvaluationCallback,
    verifier_id_to_fn: dict | None = None,
    verifier_id_to_ground_truth_length: dict | None = None,
    length_penalty_strength: float = 0.1,
    solver_attempts: int = 8,
):
    """Create appropriate reward function for task.

    Args:
        task: Task type (induction or proposer)
        callback: Reward evaluation callback observer
        verifier_id_to_fn: Verifier functions for induction task
        verifier_id_to_ground_truth_length: Ground truth lengths for induction task
        length_penalty_strength: Strength of length penalty for induction task
        solver_attempts: Number of solver attempts for proposer task

    Returns:
        Reward function instance
    """
    if task == Task.INDUCTION:
        reward_func = InductionReward(
            verifier_id_to_fn=verifier_id_to_fn,
            verifier_id_to_ground_truth_length=verifier_id_to_ground_truth_length,
            length_penalty_strength=length_penalty_strength,
            observer=callback,
        )
        logger.info(f"Length penalty strength: {length_penalty_strength}")
    elif task == Task.PROPOSER:
        # Load solver model for proposer reward
        if os.path.exists(DEFAULT_SOLVER_CHECKPOINT):
            solver_model = create_token_solver(
                TrainedLLMTokenGenerator(DEFAULT_SOLVER_CHECKPOINT, temperature=0.8),
                budget=1,
            )
            logger.info(f"Loaded solver model from {DEFAULT_SOLVER_CHECKPOINT}")
        else:
            raise ValueError(
                f"Solver checkpoint not found at {DEFAULT_SOLVER_CHECKPOINT}. "
                "ProposerReward requires a trained solver model."
            )

        # Get available functions from evaluation data for reward function
        if callback.available_functions is None:
            raise ValueError("Could not load available functions for proposer reward")

        base_reward = ProposerReward(
            trajectory_solver=solver_model,
            solver_attempts=solver_attempts,
            available_functions=callback.available_functions,
            observer=callback,
        )
        # Wrap with logging if sample logger is available
        reward_func = (
            ProposerRewardWithLogging(base_reward, callback)
            if callback.sample_logger
            else base_reward
        )
        logger.info(f"Solver attempts: {solver_attempts}")
    else:
        raise ValueError(f"Unknown task: {task}")

    return reward_func


def rl_grpo_main(
    model_name: str = "checkpoints/latest",
    model=None,
    full_run: bool = False,
    eval_steps: int = 500,
    eval_samples: int | None = None,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-6,
    max_length: int = 256,
    per_device_train_batch_size: int = 8,
    wandb_project: str = "wandering-light-rl",
    wandb_run_name: str | None = None,
    length_penalty_strength: float = 0.1,
    task: Task = Task.INDUCTION,
    solver_attempts: int = 8,
    sample_log_interval: int = 0,
):
    """
    Main function for RL training using GRPO with task-specific rewards.

    Args:
        model_name: Either a HuggingFace model name or path to local model (used if model is None)
        model: Optional model instance to use instead of loading from model_name
        full_run: Whether to use full dataset
        eval_steps: Steps between evaluations
        eval_samples: Number of samples for evaluation
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for GRPO
        max_length: Maximum length of generated responses
        per_device_train_batch_size: Per-device batch size for training
        wandb_project: Project name for wandb logging
        wandb_run_name: Run name for wandb (if None, wandb will be disabled)
        length_penalty_strength: Strength of length penalty (induction task only)
        task: Task to train on ('induction' or 'proposer')
        solver_attempts: Number of solver attempts for proposer reward evaluation
        sample_log_interval: Log training samples every N steps (0 to disable)
    """
    logger.info(f"Starting RL training with GRPO for {task} task...")

    # Set use_wandb based on whether wandb_run_name is provided
    use_wandb = wandb_run_name is not None

    # Initialize wandb if enabled
    wandb_config = {
        "model_name": model_name,
        "model_provided": model is not None,
        "task": task,
        "full_run": full_run,
        "eval_steps": eval_steps,
        "eval_samples": eval_samples,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "per_device_train_batch_size": per_device_train_batch_size,
        "length_penalty_strength": length_penalty_strength,
        "solver_attempts": solver_attempts,
        "sample_log_interval": sample_log_interval,
    }
    wandb_url = setup_wandb(wandb_run_name, wandb_project, task, wandb_config)

    # Create sample logger if interval is specified
    sample_logger = None
    if sample_log_interval > 0:
        sample_logger = TrainingSampleLogger(
            log_interval=sample_log_interval, task=task
        )
        logger.info(
            f"Training sample logging enabled every {sample_log_interval} steps"
        )

    # Create callback for reward evaluation
    reward_callback = RewardEvaluationCallback(
        eval_steps=eval_steps,
        num_samples=eval_samples,
        budget=1,
        use_wandb=use_wandb,
        task=task,
        sample_logger=sample_logger,
    )

    # Generate dataset
    completion_dataset, verifier_id_to_fn, verifier_id_to_ground_truth_length = (
        create_dataset(task, full_run, reward_callback.available_functions)
    )
    logger.info(f"Generated {task} dataset with {len(completion_dataset)} examples")

    # Debug: Check dataset columns
    if hasattr(completion_dataset, "column_names"):
        logger.info(f"Dataset columns: {completion_dataset.column_names}")
    if len(completion_dataset) > 0:
        logger.info(f"First example keys: {list(completion_dataset[0].keys())}")

    # Determine which model to use
    model_to_use = model if model is not None else model_name
    tokenizer_name = model_name if model is not None else model_name

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create reward function
    reward_func = create_reward_function(
        task=task,
        callback=reward_callback,
        verifier_id_to_fn=verifier_id_to_fn,
        verifier_id_to_ground_truth_length=verifier_id_to_ground_truth_length,
        length_penalty_strength=length_penalty_strength,
        solver_attempts=solver_attempts,
    )

    reward_funcs = [reward_func]
    logger.info(f"Using reward function: {reward_func.__name__}")

    # GRPO Configuration
    if model is not None:
        output_dir = f"./checkpoints/rl/grpo/{task}/custom_model"
    else:
        output_dir = f"./checkpoints/rl/grpo/{task}/{model_name.replace('checkpoints/', '_').replace('/', '_')}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=wandb_run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        logging_steps=eval_steps,
        report_to=["wandb"] if use_wandb else [],
        max_completion_length=max_length,
    )

    # Initialize GRPO Trainer
    wandb_callback = WandbRunLinkCallback(wandb_url)
    trainer = GRPOTrainer(
        model=model_to_use,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=completion_dataset,
        callbacks=[reward_callback, wandb_callback],
    )

    logger.info(f"Starting GRPO training with {len(completion_dataset)} examples...")
    logger.info(f"Training epochs: {num_train_epochs}")
    logger.info(f"Evaluation every {eval_steps} steps with {eval_samples} samples")
    logger.info(f"Output directory: {output_dir}")

    # Log sample logging configuration
    if sample_logger:
        logger.info(f"Sample logging enabled: {sample_logger.log_file}")
        logger.info(f"First log expected at step: {sample_logger.log_interval}")

    # Start training
    trainer.train()

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    wandb_callback.write_link(final_model_path)

    logger.info(f"\nGRPO training for {task} task completed successfully!")
    logger.info(f"Final model path: {final_model_path}")

    if hasattr(reward_callback, "metrics_history") and reward_callback.metrics_history:
        final_metrics = reward_callback.metrics_history[-1]
        logger.info(f"Final average reward: {final_metrics.avg_reward:.4f}")
        best_reward = max(m.avg_reward for m in reward_callback.metrics_history)
        logger.info(f"Best average reward: {best_reward:.4f}")

        if task == Task.INDUCTION:
            logger.info(
                f"Final avg function count: {final_metrics.avg_function_count:.2f}"
            )
            logger.info(
                f"Final avg function count ratio: {final_metrics.avg_function_count_ratio:.3f}"
            )
        elif task == Task.PROPOSER:
            logger.info(f"Final parse rate: {final_metrics.parse_rate:.2%}")
            logger.info(
                f"Final solver success rate: {final_metrics.solver_success_rate:.2%}"
            )
            logger.info(
                f"Final proposer frac intermediate difficulty: {final_metrics.frac_non_zero_std:.2%}"
            )

        # Log final summary to wandb
        if use_wandb:
            final_logs = {
                "final/avg_reward": final_metrics.avg_reward,
                "final/best_reward": best_reward,
            }
            if task == Task.INDUCTION:
                final_logs.update(
                    {
                        "final/success_rate": final_metrics.success_rate,
                        "final/avg_function_count": final_metrics.avg_function_count,
                        "final/avg_function_count_ratio": final_metrics.avg_function_count_ratio,
                    }
                )
            elif task == Task.PROPOSER:
                final_logs.update(
                    {
                        "final/parse_rate": final_metrics.parse_rate,
                        "final/solver_success_rate": final_metrics.solver_success_rate,
                        "final/frac_non_zero_std": final_metrics.frac_non_zero_std,
                    }
                )
            wandb.log(final_logs)

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    return trainer


if __name__ == "__main__":
    config = RLGRPOConfig()
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning training with GRPO for induction and proposer tasks"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roneneldan/TinyStories-1M",
        help="Model name from HuggingFace or path to local model",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="If set, use full dataset",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=config.DEFAULT_EVAL_STEPS,
        help=f"Run evaluation every N training steps (default: {config.DEFAULT_EVAL_STEPS})",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Number of samples to use for evaluation (default: None - use all)",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=config.DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs (default: {config.DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.DEFAULT_LEARNING_RATE,
        help=f"Learning rate for GRPO (default: {config.DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=config.DEFAULT_MAX_LENGTH,
        help=f"Maximum length of generated responses (default: {config.DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help=f"Per-device batch size for training (default: {config.DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="wandering-light-rl",
        help="Wandb project name (default: wandering-light-rl)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Run name for wandb (if None, wandb will be disabled)",
    )
    parser.add_argument(
        "--length-penalty-strength",
        type=float,
        default=config.DEFAULT_LENGTH_PENALTY,
        help=f"Strength of length penalty for induction task (default: {config.DEFAULT_LENGTH_PENALTY})",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=Task.INDUCTION,
        choices=list(Task),
        help="Task to run: 'induction' for function induction or 'proposer' for trajectory proposal (default: induction)",
    )
    parser.add_argument(
        "--solver-attempts",
        type=int,
        default=config.DEFAULT_SOLVER_ATTEMPTS,
        help=f"Number of solver attempts for proposer reward evaluation (default: {config.DEFAULT_SOLVER_ATTEMPTS})",
    )
    parser.add_argument(
        "--sample-log-interval",
        type=int,
        default=0,
        help="Log training samples every N steps (0 to disable)",
    )

    args = parser.parse_args()
    rl_grpo_main(
        model_name=args.model_name,
        full_run=args.full_run,
        eval_steps=args.eval_steps,
        eval_samples=args.eval_samples,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate * args.batch_size / 8,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        length_penalty_strength=args.length_penalty_strength,
        task=Task(args.task),
        solver_attempts=args.solver_attempts,
        sample_log_interval=args.sample_log_interval,
    )
