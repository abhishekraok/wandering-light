from wandering_light.evals.evaluate_solver import EvaluateSolver
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.solver import FunctionPredictor, MaybeTrajectory, TrajectorySolver
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList


class DummyPredictor(FunctionPredictor):
    """Predictor that returns predefined function sequences."""

    def __init__(self, function_sequences):
        self.function_sequences = list(function_sequences)
        self.calls = 0

    def predict_functions_batch(
        self, problems, available_functions
    ) -> list[FunctionDefList]:
        results = []
        for i in range(len(problems)):
            if self.calls + i < len(self.function_sequences):
                results.append(self.function_sequences[self.calls + i])
            else:
                results.append(FunctionDefList())
        self.calls += len(problems)
        return results


class DummySolver(TrajectorySolver):
    """Solver that returns predefined results sequentially."""

    def __init__(self, results):
        # Create a dummy predictor that doesn't matter since we override solve_batch
        super().__init__(DummyPredictor([]))
        self._results = list(results)
        self.calls = 0

    def solve_batch(self, problems, available_functions):
        """Override batch solve to return predefined results."""
        results = []
        for i in range(len(problems)):
            if self.calls + i < len(self._results):
                results.append(self._results[self.calls + i])
            else:
                results.append(
                    MaybeTrajectory(success=False, error_msg="No more results")
                )
        self.calls += len(problems)
        return results

    def solve(self, input_list, output_list, available_functions):
        result = self._results[self.calls]
        self.calls += 1
        return result


def test_avg_solution_length_counts_only_successes():
    inc = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )

    # Specs: one requiring a single increment (success) and another requiring two
    spec_success = TrajectorySpec(TypedList([1], int), FunctionDefList([inc]))
    spec_failure = TrajectorySpec(TypedList([1], int), FunctionDefList([inc, inc]))
    spec_list = TrajectorySpecList([spec_success, spec_failure])

    executor = Executor(FunctionDefList([inc]))
    result_success = executor.execute_trajectory(spec_success)
    traj_success = result_success.trajectory
    # Solver will attempt only one increment for the failure case
    result_failure = executor.execute_trajectory(
        TrajectorySpec(spec_failure.input, FunctionDefList([inc]))
    )
    traj_failure = result_failure.trajectory

    results = [
        MaybeTrajectory(success=True, trajectory=traj_success),
        MaybeTrajectory(success=False, trajectory=traj_failure, error_msg="fail"),
    ]
    solver = DummySolver(results)

    eval_result = EvaluateSolver.evaluate_using_trajectory_specs(
        solver, spec_list, available_functions=FunctionDefList([inc])
    )

    assert eval_result.success_count == 1
    assert eval_result.total_samples == 2
    # Average length should only consider successful trajectory (length 1)
    assert eval_result.avg_solution_length == 1.0
