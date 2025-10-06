from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.function_def import FunctionDefSet
from wandering_light.solver import TokenGeneratorSolver, TrainedLLMTokenGenerator
from wandering_light.trajectory import TrajectoryList


def evaluate_model_checkpoint_with_trajectories(
    model_path: str,
    trajectories: TrajectoryList,
    available_functions: FunctionDefSet,
    num_samples: int | None = None,
    budget: int = 1,
):
    """
    Evaluate a saved model checkpoint using pre-computed trajectories.

    This is more efficient than the trajectory_specs version since trajectories
    already have computed outputs.
    """
    token_generator = TrainedLLMTokenGenerator(model_path)
    solver = TokenGeneratorSolver(budget=budget, token_generator=token_generator)

    return EvaluateSolver.evaluate_using_trajectories(
        solver,
        trajectories,
        available_functions=available_functions,
        num_samples=num_samples,
        save_results=False,
    )
