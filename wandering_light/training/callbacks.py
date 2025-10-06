import json
import logging
import os
from datetime import datetime
from typing import Any

from transformers import (
    PreTrainedModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

import wandb
from wandering_light.constants import DEFAULT_EVAL_FILE, DEFAULT_SOLVER_CHECKPOINT, Task
from wandering_light.evals.evaluate_proposer import evaluate_proposer
from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.function_def import FunctionDefSet
from wandering_light.trajectory import TrajectoryList
from wandering_light.solver import (
    TokenGenerator,
    create_token_solver,
)
from wandering_light.training.reward import (
    ProposerReward,
)
from wandering_light.training.rl_grpo_config import (
    BatchMetrics,
    InductionMetricsObserver,
    ProposerMetricsObserver,
    RLMetrics,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingSampleLogger:
    """Logs training samples (prompts and completions) to files in training_logs directory."""

    def __init__(self, log_interval: int = 10, task: Task = Task.PROPOSER):
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
        trajectories: TrajectoryList | None = None,
        available_functions: FunctionDefSet | None = None,
    ):
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.budget = budget
        self.eval_file = eval_file
        self.metrics_history = []
        self.use_wandb = use_wandb
        self.task = task
        self.solver_model = None
        self.sample_logger = sample_logger
        self.current_step = 0

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

        # Store latest batch data for logging
        self._latest_prompts = []
        self._latest_completions = []
        self._latest_rewards = []

        # Load evaluation data once
        if available_functions is None or trajectories is None:
            self._load_eval_data()
        else:
            self.trajectories = trajectories
            self.available_functions = available_functions

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

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        """Called after metrics are logged."""
        # Global step is not the true global step as trainer.train() resets it
        if (
            state.global_step > 0
            and (state.global_step + self.current_step) % self.eval_steps == 0
        ):
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

            # TODO: Separate metrics for induction and proposer
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
                    step=state.global_step + self.current_step,
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
                            "step": state.global_step + self.current_step,
                        }
                    )

                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"RL TRAINING METRICS AT STEP {state.global_step + self.current_step}"
                )
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
                    step=state.global_step + self.current_step,
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
                            "step": state.global_step + self.current_step,
                        }
                    )

                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"RL TRAINING METRICS AT STEP {state.global_step + self.current_step}"
                )
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


class OnlineEvaluator:
    def __init__(
        self,
        task: Task,
        trajectories: TrajectoryList,
        available_functions: FunctionDefSet,
        use_wandb: bool,
        num_samples: int | None,
        budget: int,
        solver_model: PreTrainedModel | None = None,
    ):
        self.task = task
        self.trajectories = trajectories
        self.available_functions = available_functions
        self.num_samples = num_samples
        self.budget = budget
        self.solver_model = solver_model
        self.eval_results = []
        self.use_wandb = use_wandb

    def evaluate_model(self, token_generator: TokenGenerator, step: int):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RL EVALUATION METRICS AT STEP {step}")
        logger.info(f"{'=' * 60}")

        if self.task == Task.INDUCTION:
            trajectory_solver = create_token_solver(
                token_generator,
                budget=self.budget,
            )
            result = EvaluateSolver.evaluate_using_trajectories(
                trajectory_solver,
                trajectories=self.trajectories,
                available_functions=self.available_functions,
                num_samples=self.num_samples,
            )
            logger.info(f"Step {step} RL Eval Success Rate: {result.success_rate:.2%}")
            logger.info(f"Avg Function Count: {result.avg_solution_length:.2f}")

            eval_result = {
                "step": step,
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
                        "step": step,
                    }
                )
        elif self.task == Task.PROPOSER:
            # Create the model for proposer evaluation
            trajectory_solver = create_token_solver(
                token_generator,
                budget=self.budget,
            )
            result = evaluate_proposer(
                token_generator,
                trajectory_solver=trajectory_solver,
                trajectories=self.trajectories,
                num_samples=self.num_samples or len(self.trajectories),
            )
            logger.info(f"Step {step} Results:")
            logger.info(f"  Parse Rate: {result.parse_rate:.2%}")
            logger.info(f"  Avg Function Count: {result.avg_function_count:.2f}")
            logger.info(f"  Solver Success Rate: {result.solver_success_rate:.2%}")
            logger.info(
                f"  Eval Frac Intermediate Difficulty: {result.frac_non_zero_std:.2%}"
            )

            eval_result = {
                "step": step,
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
                        "step": step,
                    }
                )
        else:
            logger.error(f"Unknown task type: {self.task}")
            return None


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
