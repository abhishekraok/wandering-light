import argparse
import os

from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import SFTConfig, SFTTrainer

import wandb
from wandering_light.constants import DEFAULT_EVAL_FILE, DEFAULT_SOLVER_CHECKPOINT, Task
from wandering_light.evals.evaluate_proposer import evaluate_proposer
from wandering_light.evals.model_eval import evaluate_model_checkpoint_with_trajectories
from wandering_light.evals.run_evaluation import load_eval_data_as_trajectories
from wandering_light.solver import TrainedLLMTokenGenerator, create_token_solver
from wandering_light.training.data_generator import induction_dataset, proposer_dataset
from wandering_light.training.wandb_utils import WandbRunLinkCallback


class OnlineEvaluationCallback(TrainerCallback):
    """Callback to run online evaluation every N steps during training."""

    def __init__(
        self,
        eval_steps: int = 500,
        num_samples: int | None = None,
        budget: int = 1,
        eval_file: str = DEFAULT_EVAL_FILE,
        use_wandb: bool = True,
        task: Task = Task.INDUCTION,
    ):
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.budget = budget
        self.eval_file = eval_file
        self.trajectories = None
        self.available_functions = None
        self.eval_results = []
        self.use_wandb = use_wandb
        self.task = task
        self.solver_model = None
        self._load_eval_data()

    def _load_eval_data(self):
        """Load the evaluation data once at initialization and pre-compute trajectories."""
        try:
            self.trajectories, self.available_functions = (
                load_eval_data_as_trajectories(self.eval_file)
            )
            print(f"Pre-computed {len(self.trajectories)} trajectories for online eval")
            print(
                f"Found {len(self.available_functions)} unique functions for online eval"
            )

        except Exception as e:
            print(f"Warning: Could not load evaluation data: {e}")
            self.trajectories = None
            self.available_functions = None
        if self.task == Task.PROPOSER:
            if os.path.exists(DEFAULT_SOLVER_CHECKPOINT):
                self.solver_model = create_token_solver(
                    TrainedLLMTokenGenerator(DEFAULT_SOLVER_CHECKPOINT), budget=1
                )
            else:
                print(
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
                print(f"Unknown task type: {self.task}")
                return None
        except Exception as e:
            print(f"Error during online evaluation: {e}")
            return None

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when a checkpoint is saved - this is when we can evaluate."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n{'=' * 60}")
            print(f"ONLINE EVALUATION AT STEP {state.global_step}")
            print(f"{'=' * 60}")

            # Use the checkpoint that was just saved
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )

            if os.path.exists(checkpoint_path):
                try:
                    # Run evaluation on the checkpoint
                    result = self._run_evaluation(checkpoint_path)

                    if result:
                        print(f"Step {state.global_step} Results:")

                        if self.task == Task.INDUCTION:
                            print(f"  Success Rate: {result.success_rate:.2%}")
                            print(
                                f"  Avg Solution Length: {result.avg_solution_length:.2f}"
                            )
                            print(
                                f"  Successes: {result.success_count}/{result.total_samples}"
                            )

                            # Store results for later analysis
                            eval_result = {
                                "step": state.global_step,
                                "success_rate": result.success_rate,
                                "avg_solution_length": result.avg_solution_length,
                                "total_samples": result.total_samples,
                                "success_count": result.success_count,
                            }
                            self.eval_results.append(eval_result)

                            # Log to wandb if enabled
                            if self.use_wandb:
                                wandb.log(
                                    {
                                        "eval/success_rate": result.success_rate,
                                        "eval/avg_solution_length": result.avg_solution_length,
                                        "eval/success_count": result.success_count,
                                        "eval/total_samples": result.total_samples,
                                        "step": state.global_step,
                                    }
                                )

                            # Log to state if possible
                            logs = {
                                "eval_success_rate": result.success_rate,
                                "eval_avg_solution_length": result.avg_solution_length,
                            }
                            state.log_history.append(
                                {**logs, "step": state.global_step}
                            )

                        elif self.task == Task.PROPOSER:
                            print(f"  Parse Rate: {result.parse_rate:.2%}")
                            print(
                                f"  Avg Function Count: {result.avg_function_count:.2f}"
                            )
                            print(
                                f"  Solver Success Rate: {result.solver_success_rate:.2%}"
                            )
                            print(
                                f"  Frac Non-Zero Std: {result.frac_non_zero_std:.2%}"
                            )
                            print(f"  Samples: {result.num_samples}")

                            # Store results for later analysis
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

                            # Log to wandb if enabled
                            if self.use_wandb:
                                wandb.log(
                                    {
                                        "eval/parse_rate": result.parse_rate,
                                        "eval/avg_function_count": result.avg_function_count,
                                        "eval/avg_function_count_ratio": result.avg_function_count_ratio,
                                        "eval/solver_success_rate": result.solver_success_rate,
                                        "eval/frac_non_zero_std": result.frac_non_zero_std,
                                        "eval/num_samples": result.num_samples,
                                        "step": state.global_step,
                                    }
                                )

                            # Log to state if possible
                            logs = {
                                "eval_parse_rate": result.parse_rate,
                                "eval_avg_function_count": result.avg_function_count,
                                "eval_frac_non_zero_std": result.frac_non_zero_std,
                            }
                            state.log_history.append(
                                {**logs, "step": state.global_step}
                            )

                    else:
                        print(f"Step {state.global_step}: Evaluation failed")

                except Exception as e:
                    print(f"Step {state.global_step}: Error during evaluation: {e}")
            else:
                print(
                    f"Step {state.global_step}: Checkpoint not found at {checkpoint_path}"
                )

            print(f"{'=' * 60}\n")

        return control


def sft_main(
    model_name: str = "facebook/opt-125m",
    full_run: bool = False,
    run_eval: bool = True,
    online_eval_steps: int = 500,
    online_eval_samples: int | None = None,
    wandb_project: str = "wandering-light-sft",
    wandb_run_name: str | None = None,
    task: Task = Task.INDUCTION,
):
    # Set use_wandb based on whether wandb_run_name is provided
    use_wandb = wandb_run_name is not None

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "task": task,
                "full_run": full_run,
                "run_eval": run_eval,
                "online_eval_steps": online_eval_steps,
                "online_eval_samples": online_eval_samples,
                "num_train_epochs": 4,
                "max_length": 256,
                "completion_only_loss": True,
            },
            tags=["sft", task],
        )
        wandb_url = str(wandb.run.get_url())
    else:
        wandb_url = None
    # Training phase
    print(f"Starting SFT training for {task} task...")
    length_counts = (
        {1: 100, 2: 10_000, 3: 100_000, 4: 10_000, 5: 1_000}
        if full_run
        else {1: 10, 2: 6}
    )

    # Use appropriate dataset based on task
    if task == Task.INDUCTION:
        dataset = induction_dataset(length_counts=length_counts)
        print(f"Using induction dataset with {len(dataset)} examples")
    elif task == Task.PROPOSER:
        dataset = proposer_dataset(length_counts=length_counts)
        print(f"Using proposer dataset with {len(dataset)} examples")
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: {list(Task)}")

    # Update training args to enable more frequent checkpointing for online eval
    max_length = 256
    processing_class = None
    model_id = model_name

    training_args = SFTConfig(
        max_length=max_length,
        output_dir=f"./checkpoints/{model_name.replace('/', '_')}",
        save_strategy="steps",
        save_steps=online_eval_steps,
        save_total_limit=3,
        logging_steps=500,
        num_train_epochs=4,
        completion_only_loss=True,
        report_to=["wandb"] if use_wandb else [],
        run_name=wandb_run_name if use_wandb else None,
    )

    # Create the online evaluation callback
    eval_callback = OnlineEvaluationCallback(
        eval_steps=online_eval_steps,
        num_samples=online_eval_samples,
        budget=1,
        use_wandb=use_wandb,
        task=task,
    )

    wandb_callback = WandbRunLinkCallback(wandb_url)
    callbacks = [wandb_callback]
    if run_eval:
        callbacks.insert(0, eval_callback)

    trainer = SFTTrainer(
        model_id,
        args=training_args,
        train_dataset=dataset,
        processing_class=processing_class,
        callbacks=callbacks,
    )

    print("Training model...")
    if run_eval:
        print(
            f"Online evaluation will run every {online_eval_steps} steps with {online_eval_samples} samples"
        )

    trainer.train()

    # Save the final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    wandb_callback.write_link(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Print summary of online evaluation results if available
    if run_eval:
        print("\nTraining completed successfully!")
        print(f"Final model path: {final_model_path}")

        if hasattr(trainer, "callback_handler"):
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, OnlineEvaluationCallback):
                    print("\nOnline Evaluation Summary:")
                    print(f"Total evaluations run: {len(callback.eval_results)}")
                    if callback.eval_results:
                        final_result = callback.eval_results[-1]

                        if task == Task.INDUCTION:
                            best_success_rate = max(
                                r["success_rate"] for r in callback.eval_results
                            )
                            print(
                                f"Final success rate: {final_result['success_rate']:.2%}"
                            )
                            print(f"Best success rate: {best_success_rate:.2%}")

                            # Log final summary to wandb
                            if use_wandb:
                                wandb.log(
                                    {
                                        "final/success_rate": final_result[
                                            "success_rate"
                                        ],
                                        "final/best_success_rate": best_success_rate,
                                        "final/avg_solution_length": final_result[
                                            "avg_solution_length"
                                        ],
                                        "final/total_evaluations": len(
                                            callback.eval_results
                                        ),
                                    }
                                )
                        elif task == Task.PROPOSER:
                            best_parse_rate = max(
                                r["parse_rate"] for r in callback.eval_results
                            )
                            best_frac_non_zero_std = max(
                                r["frac_non_zero_std"] for r in callback.eval_results
                            )
                            print(f"Final parse rate: {final_result['parse_rate']:.2%}")
                            print(f"Best parse rate: {best_parse_rate:.2%}")
                            print(
                                f"Final avg function count: {final_result['avg_function_count']:.2f}"
                            )
                            print(
                                f"Final frac non-zero std: {final_result['frac_non_zero_std']:.2%}"
                            )
                            print(
                                f"Best frac non-zero std: {best_frac_non_zero_std:.2%}"
                            )

                            # Log final summary to wandb
                            if use_wandb:
                                wandb.log(
                                    {
                                        "final/parse_rate": final_result["parse_rate"],
                                        "final/best_parse_rate": best_parse_rate,
                                        "final/avg_function_count": final_result[
                                            "avg_function_count"
                                        ],
                                        "final/solver_success_rate": final_result[
                                            "solver_success_rate"
                                        ],
                                        "final/frac_non_zero_std": final_result[
                                            "frac_non_zero_std"
                                        ],
                                        "final/best_frac_non_zero_std": best_frac_non_zero_std,
                                        "final/total_evaluations": len(
                                            callback.eval_results
                                        ),
                                    }
                                )
                    break
    else:
        print(f"Training completed. Model saved to {final_model_path}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for induction dataset with online evaluation"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roneneldan/TinyStories-1M",  # other options: "facebook/opt-125m"
        help="Model name to use for training",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="If set, use full dataset (length_counts=None)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="If set, skip online evaluation during and after training",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=2000,
        help="Run online evaluation every N training steps (default: 2000)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Number of samples to use for online evaluation (default: None - use all)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="wandering-light-sft",
        help="Wandb project name (default: wandering-light-sft)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Run name for wandb (if None, wandb will be disabled)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=Task.INDUCTION,
        choices=list(Task),
        help="Task to run: 'induction' for function induction or 'proposer' for trajectory proposal (default: induction)",
    )

    args = parser.parse_args()
    sft_main(
        model_name=args.model_name,
        full_run=args.full_run,
        run_eval=not args.no_eval,
        online_eval_steps=args.eval_steps,
        online_eval_samples=args.eval_samples,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        task=Task(args.task),
    )
