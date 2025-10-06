import argparse
import os
import shutil
from typing import Any

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
)
from trl import GRPOConfig, GRPOTrainer

import wandb
from wandering_light.constants import Task
from wandering_light.function_def import FunctionDefSet
from wandering_light.solver import (
    TrainedLLMTokenGenerator,
    create_token_solver,
)
from wandering_light.training.data_generator import (
    induction_dataset_rl,
    proposer_dataset_rl,
)
from wandering_light.training.reward import (
    InductionReward,
    ProposerReward,
)
from wandering_light.training.rl_grpo_config import (
    RLGRPOConfig,
)
from wandering_light.training.wandb_utils import WandbRunLinkCallback
from wandering_light.training.callbacks import (
    RewardEvaluationCallback,
    TrainingSampleLogger,
    OnlineEvaluator,
    ProposerRewardWithLogging,
    logger,
)


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
    solver_model: PreTrainedModel | None = None,
    tokenizer: AutoTokenizer | None = None,
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
        if solver_model is None:
            raise ValueError("Solver model is required for proposer reward")
        # Load solver model for proposer reward
        trajectory_solver = create_token_solver(
            TrainedLLMTokenGenerator(solver_model, tokenizer=tokenizer),
            budget=1,
        )

        # Get available functions from evaluation data for reward function
        if callback.available_functions is None:
            raise ValueError("Could not load available functions for proposer reward")

        base_reward = ProposerReward(
            trajectory_solver=trajectory_solver,
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


def dual_rl_grpo_main(
    induction_model: str,
    proposer_model: str,
    full_run: bool = False,
    eval_steps: int = 500,
    eval_samples: int | None = None,
    num_train_steps: int = 12_000,
    learning_rate: float = 1e-6,
    max_length: int = 256,
    per_device_train_batch_size: int = 8,
    wandb_project: str = "wandering-light-rl",
    wandb_run_name: str | None = None,
    length_penalty_strength: float = 0.1,
    task: Task = Task.INDUCTION,
    solver_attempts: int = 8,
    sample_log_interval: int = 0,
    training_interval_steps: int = 32,
):
    """
    Main function for RL training using GRPO with task-specific rewards.

    Args:
        induction_model: Induction model name from HuggingFace or path to local model
        proposer_model: Proposer model name from HuggingFace or path to local model
        full_run: Whether to use full dataset
        eval_steps: Steps between evaluations
        eval_samples: Number of samples for evaluation
        num_train_steps: Number of training steps
        learning_rate: Learning rate for GRPO
        max_length: Maximum length of generated responses
        per_device_train_batch_size: Per-device batch size for training
        wandb_project: Project name for wandb logging
        wandb_run_name: Run name for wandb (if None, wandb will be disabled)
        length_penalty_strength: Strength of length penalty (induction task only)
        task: Task to train on ('induction' or 'proposer')
        solver_attempts: Number of solver attempts for proposer reward evaluation
        sample_log_interval: Log training samples every N steps (0 to disable)
        training_interval_steps: Number of training steps to run continuously before changing task
    """
    logger.info(f"Starting RL training with GRPO for {task} task...")

    # Set use_wandb based on whether wandb_run_name is provided
    use_wandb = wandb_run_name is not None

    # Initialize wandb if enabled
    wandb_config = {
        "induction_model": induction_model,
        "proposer_model": proposer_model,
        "task": task,
        "full_run": full_run,
        "eval_steps": eval_steps,
        "eval_samples": eval_samples,
        "num_train_steps": num_train_steps,
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
    induction_reward_callback = RewardEvaluationCallback(
        eval_steps=eval_steps,
        num_samples=eval_samples,
        budget=1,
        use_wandb=use_wandb,
        task=Task.INDUCTION,
        sample_logger=None,
    )
    induction_online_evaluator = OnlineEvaluator(
        task=Task.INDUCTION,
        trajectories=induction_reward_callback.trajectories,
        available_functions=induction_reward_callback.available_functions,
        num_samples=eval_samples,
        budget=1,
        use_wandb=use_wandb,
    )
    proposer_reward_callback = RewardEvaluationCallback(
        eval_steps=eval_steps,
        num_samples=eval_samples,
        budget=1,
        use_wandb=use_wandb,
        task=Task.PROPOSER,
        sample_logger=sample_logger,
        trajectories=induction_reward_callback.trajectories,
        available_functions=induction_reward_callback.available_functions,
    )

    # Generate dataset
    induction_dataset, verifier_id_to_fn, verifier_id_to_ground_truth_length = (
        create_dataset(
            Task.INDUCTION, full_run, induction_reward_callback.available_functions
        )
    )
    logger.info(
        f"Generated {Task.INDUCTION} dataset with {len(induction_dataset)} examples"
    )
    proposer_dataset, _, _ = create_dataset(
        Task.PROPOSER, full_run, proposer_reward_callback.available_functions
    )
    logger.info(
        f"Generated {Task.PROPOSER} dataset with {len(proposer_dataset)} examples"
    )

    tokenizer_name = induction_model

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create reward function
    induction_reward_func = create_reward_function(
        task=Task.INDUCTION,
        callback=induction_reward_callback,
        verifier_id_to_fn=verifier_id_to_fn,
        verifier_id_to_ground_truth_length=verifier_id_to_ground_truth_length,
        length_penalty_strength=length_penalty_strength,
        solver_attempts=solver_attempts,
    )

    # GRPO Configuration
    output_dir_induction = f"./checkpoints/rl/grpo/{Task.DUAL}/{induction_model.replace('checkpoints/saved/rl', '').replace('/', '_')}/induction"
    output_dir_proposer = f"./checkpoints/rl/grpo/{Task.DUAL}/{proposer_model.replace('checkpoints/saved/rl', '').replace('/', '_')}/proposer"

    # Initialize GRPO Trainer
    wandb_callback = WandbRunLinkCallback(wandb_url)

    num_samples = len(induction_dataset)
    logger.info(f"Starting GRPO training with {num_samples} examples...")
    logger.info(f"Training steps: {num_train_steps}")
    logger.info(f"Evaluation every {eval_steps} steps with {eval_samples} samples")
    logger.info(f"Output directory induction: {output_dir_induction}")
    logger.info(f"Output directory proposer: {output_dir_proposer}")

    # Log sample logging configuration
    if sample_logger:
        logger.info(f"Sample logging enabled: {sample_logger.log_file}")
        logger.info(f"First log expected at step: {sample_logger.log_interval}")

    logger.info(f"Total training steps: {num_train_steps:,}")
    if eval_steps % training_interval_steps != 0:
        raise ValueError(
            f"eval_steps ({eval_steps}) must be a multiple of {training_interval_steps}"
        )
    num_repeats = num_train_steps * per_device_train_batch_size // num_samples + 1
    num_checkpoints_to_keep = 5
    retained_checkpoints = []
    logger.info(f"Number of repeats: {num_repeats}")

    induction_base_training_args = GRPOConfig(
        output_dir=output_dir_induction,  # Will be updated per iteration
        run_name=wandb_run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=1,
        learning_rate=learning_rate,
        save_total_limit=1,
        report_to=["wandb"] if use_wandb else [],
        max_completion_length=max_length,
        logging_strategy="steps",
        logging_steps=training_interval_steps,  # Log at the end of the interval
        save_steps=training_interval_steps,
        save_strategy="no",  # Will be updated per iteration
    )
    # Create base dataset that will be sliced per iteration
    induction_base_dataset = induction_dataset.repeat(num_repeats)
    induction_trainer = GRPOTrainer(
        model=induction_model,
        reward_funcs=[induction_reward_func],
        args=induction_base_training_args,
        train_dataset=induction_base_dataset,  # Will be updated per iteration
        callbacks=[induction_reward_callback, wandb_callback],
    )

    proposer_online_evaluator = OnlineEvaluator(
        task=Task.PROPOSER,
        trajectories=proposer_reward_callback.trajectories,
        available_functions=proposer_reward_callback.available_functions,
        num_samples=eval_samples,
        budget=1,
        use_wandb=use_wandb,
        solver_model=induction_trainer.model,
    )
    proposer_reward_func = create_reward_function(
        task=Task.PROPOSER,
        callback=proposer_reward_callback,
        solver_model=induction_trainer.model,
        tokenizer=tokenizer,
    )
    proposer_base_training_args = GRPOConfig(
        output_dir=output_dir_proposer,  # Will be updated per iteration
        run_name=wandb_run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=1,
        learning_rate=learning_rate,
        save_total_limit=1,
        report_to=["wandb"] if use_wandb else [],
        max_completion_length=max_length,
        logging_strategy="steps",
        logging_steps=training_interval_steps,  # Log at the end of the interval
        save_steps=training_interval_steps,
        save_strategy="no",  # Will be updated per iteration
    )
    proposer_base_dataset = proposer_dataset.repeat(num_repeats)
    proposer_trainer = GRPOTrainer(
        model=proposer_model,
        reward_funcs=[proposer_reward_func],
        args=proposer_base_training_args,
        train_dataset=proposer_base_dataset,  # Will be updated per iteration
        callbacks=[proposer_reward_callback, wandb_callback],
    )
    for step in range(0, num_train_steps, training_interval_steps):
        induction_trainer.args.output_dir = os.path.join(
            output_dir_induction, f"checkpoint_{step}"
        )
        os.makedirs(induction_trainer.args.output_dir, exist_ok=True)
        proposer_trainer.args.output_dir = os.path.join(
            output_dir_proposer, f"checkpoint_{step}"
        )
        os.makedirs(proposer_trainer.args.output_dir, exist_ok=True)

        if step > 0 and step % eval_steps == 0:
            token_generator = TrainedLLMTokenGenerator(
                induction_trainer.model, tokenizer=tokenizer
            )
            induction_online_evaluator.evaluate_model(
                token_generator=token_generator,
                step=induction_reward_callback.current_step,
            )
            proposer_token_generator = TrainedLLMTokenGenerator(
                proposer_trainer.model, tokenizer=tokenizer
            )
            proposer_online_evaluator.evaluate_model(
                token_generator=proposer_token_generator,
                step=proposer_reward_callback.current_step,
            )
            wandb_callback.write_link(induction_trainer.args.output_dir)
            wandb_callback.write_link(proposer_trainer.args.output_dir)
            logger.info(f"Saving checkpoint: {induction_trainer.args.output_dir}")
            logger.info(f"Saving checkpoint: {proposer_trainer.args.output_dir}")
            induction_trainer.args.save_strategy = "steps"
            proposer_trainer.args.save_strategy = "steps"
            retained_checkpoints.append(induction_trainer.args.output_dir)
            retained_checkpoints.append(proposer_trainer.args.output_dir)

            if len(retained_checkpoints) > num_checkpoints_to_keep:
                shutil.rmtree(retained_checkpoints.pop(0))
        else:
            induction_trainer.args.save_strategy = "no"
            proposer_trainer.args.save_strategy = "no"
        induction_trainer.train_dataset = induction_base_dataset.select(
            range(
                (step * per_device_train_batch_size)
                // induction_trainer.args.num_generations,
                (step + training_interval_steps)
                * per_device_train_batch_size
                // induction_trainer.args.num_generations,
            )
        )
        proposer_trainer.train_dataset = proposer_base_dataset.select(
            range(
                (step * per_device_train_batch_size)
                // proposer_trainer.args.num_generations,
                (step + training_interval_steps)
                * per_device_train_batch_size
                // proposer_trainer.args.num_generations,
            )
        )

        logger.info(f"Training steps: {step}:{step + training_interval_steps}")
        induction_trainer.train()
        proposer_trainer.train()
        induction_reward_callback.current_step += induction_trainer.state.global_step
        proposer_reward_callback.current_step += proposer_trainer.state.global_step

    logger.info(f"\nGRPO training for {task} task completed successfully!")

    if (
        hasattr(induction_reward_callback, "metrics_history")
        and induction_reward_callback.metrics_history
        and hasattr(proposer_reward_callback, "metrics_history")
        and proposer_reward_callback.metrics_history
    ):
        induction_final_metrics = induction_reward_callback.metrics_history[-1]
        logger.info(
            f"Final Induction average reward: {induction_final_metrics.avg_reward:.4f}"
        )
        best_reward = max(
            m.avg_reward for m in induction_reward_callback.metrics_history
        )
        logger.info(f"Best Induction average reward: {best_reward:.4f}")

        logger.info(
            f"Final Induction avg function count: {induction_final_metrics.avg_function_count:.2f}"
        )
        logger.info(
            f"Final Induction avg function count ratio: {induction_final_metrics.avg_function_count_ratio:.3f}"
        )

        # Proposer
        proposer_final_metrics = proposer_reward_callback.metrics_history[-1]
        logger.info(
            f"Final Proposer parse rate: {proposer_final_metrics.parse_rate:.2%}"
        )
        logger.info(
            f"Final Proposer solver success rate: {proposer_final_metrics.solver_success_rate:.2%}"
        )
        logger.info(
            f"Final proposer frac intermediate difficulty: {proposer_final_metrics.frac_non_zero_std:.2%}"
        )

        # Log final summary to wandb
        if use_wandb:
            final_logs = {
                "final/induction_avg_reward": induction_final_metrics.avg_reward,
                "final/induction_best_reward": best_reward,
            }
            final_logs.update(
                {
                    "final/induction_success_rate": induction_final_metrics.success_rate,
                    "final/induction_avg_function_count": induction_final_metrics.avg_function_count,
                    "final/induction_avg_function_count_ratio": induction_final_metrics.avg_function_count_ratio,
                }
            )
            final_logs.update(
                {
                    "final/proposer_parse_rate": proposer_final_metrics.parse_rate,
                    "final/proposer_solver_success_rate": proposer_final_metrics.solver_success_rate,
                    "final/proposer_frac_non_zero_std": proposer_final_metrics.frac_non_zero_std,
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
        "--induction-model",
        type=str,
        default="checkpoints/saved/sft/long_sft_opt_125m_r1_s434k/",
        help="Induction model name from HuggingFace or path to local model",
    )
    parser.add_argument(
        "--proposer-model",
        type=str,
        default="checkpoints/saved/rl/proposer/temp_opt_125m_s2k_s8k/",
        help="Proposer model name from HuggingFace or path to local model",
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
        "--num-train-steps",
        type=int,
        default=32,
        help=f"Number of training steps (default: 12,000)",
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
    parser.add_argument(
        "--training-interval-steps",
        type=int,
        default=config.TRAINING_INTERVAL_STEPS,
        help=f"Training interval steps (default: {config.TRAINING_INTERVAL_STEPS})",
    )

    args = parser.parse_args()
    dual_rl_grpo_main(
        induction_model=args.induction_model,
        proposer_model=args.proposer_model,
        full_run=args.full_run,
        eval_steps=args.eval_steps,
        eval_samples=args.eval_samples,
        num_train_steps=args.num_train_steps,
        learning_rate=args.learning_rate * args.batch_size / 8,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        length_penalty_strength=args.length_penalty_strength,
        task=Task(args.task),
        solver_attempts=args.solver_attempts,
        sample_log_interval=args.sample_log_interval,
        training_interval_steps=args.training_interval_steps,
    )
