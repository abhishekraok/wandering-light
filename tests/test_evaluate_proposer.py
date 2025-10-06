from wandering_light.constants import DEFAULT_EVAL_FILE
from wandering_light.evals.evaluate_proposer import EvalResult, evaluate_proposer
from wandering_light.function_def import FunctionDefList
from wandering_light.solver import TokenGenerator, create_token_solver
from wandering_light.trajectory import TrajectoryList


class MockTokenGenerator(TokenGenerator):
    """Mock model that returns valid trajectory spec for first sample and gibberish for second."""

    def __init__(self, responses: list[str]):
        self.llm_io_history = []
        self.call_count = 0
        self.responses = responses

    def generate(self, prompt: str) -> str:
        """Return valid function names for first call, gibberish for second."""
        self.call_count += 1

        response = self.responses[self.call_count - 1]

        self.llm_io_history.append((prompt, response))
        return response

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate batch responses by calling generate for each prompt."""
        return [self.generate(prompt) for prompt in prompts]


def test_evaluate_proposer():
    """Test evaluate_proposer with mock model that succeeds on first sample, fails on second."""

    trajectories = TrajectoryList.from_file(DEFAULT_EVAL_FILE)

    # Create mock model
    mock_model = MockTokenGenerator(
        [trajectories[-1].to_spec().__repr__(), "invalid gibberish"]
    )

    # Test the function
    result = evaluate_proposer(
        mock_model, trajectory_solver=None, trajectories=trajectories, num_samples=2
    )

    # Verify the result
    assert isinstance(result, EvalResult)
    assert result.num_samples == 2

    # Parse rate: only first sample should parse successfully
    assert result.parse_rate == 0.5  # 1 out of 2 samples parsed

    # Function count: only first sample contributes
    assert result.avg_function_count == len(trajectories[-1].function_defs)

    # Other metrics are 0 for now.
    assert result.solver_success_rate == 0.0
    assert result.avg_function_count_ratio == 0.0

    # Verify mock was called twice
    assert mock_model.call_count == 2
    assert len(mock_model.llm_io_history) == 2


def test_evaluate_proposer_empty_trajectories():
    """Test evaluate_proposer with empty trajectory list."""
    mock_model = MockTokenGenerator([])

    result = evaluate_proposer(
        mock_model, trajectory_solver=None, trajectories=[], num_samples=0
    )

    assert result.num_samples == 0
    assert result.parse_rate == 0.0
    assert result.avg_function_count == 0.0
    assert result.avg_function_count_ratio == 0.0
    assert result.solver_success_rate == 0.0


def test_evaluate_proposer_all_unparsable():
    """Test evaluate_proposer when all model outputs are unparsable."""
    trajectories = TrajectoryList.from_file(DEFAULT_EVAL_FILE)

    mock_model = MockTokenGenerator(["invalid gibberish"])
    result = evaluate_proposer(
        mock_model, trajectory_solver=None, trajectories=trajectories, num_samples=1
    )

    assert result.num_samples == 1
    assert result.parse_rate == 0.0  # No samples parsed
    assert result.avg_function_count == 0.0  # No parsed samples to average
    assert result.avg_function_count_ratio == 0.0  # No parsed samples
    assert result.solver_success_rate == 0.0  # No samples solved


def test_evaluate_proposer_with_solver():
    """Test evaluate_proposer with solver model and check per sample results."""

    trajectories = TrajectoryList.from_file(DEFAULT_EVAL_FILE)
    first_trajectory = trajectories[0]
    second_trajectory = trajectories[1]

    # Create mock model - 4 successful parses + 1 failure = 5 samples total
    mock_model = MockTokenGenerator(
        [first_trajectory.to_spec().__repr__()] * 4 + ["invalid gibberish"]
    )

    # Extract function names for solver responses
    first_trajectory_function_names = FunctionDefList(
        first_trajectory.function_defs
    ).to_string()
    second_trajectory_function_names = FunctionDefList(
        second_trajectory.function_defs
    ).to_string()

    solver_model = create_token_solver(
        MockTokenGenerator(
            [first_trajectory_function_names] * 3
            + [
                first_trajectory_function_names,
                first_trajectory_function_names,
                "invalid gibberish",
            ]
            + [
                "invalid gibberish",
                first_trajectory_function_names,
                "invalid gibberish",
            ]
            + [
                "invalid gibberish",
                second_trajectory_function_names,  # Different function sequence
                "invalid gibberish",
            ]
        )
    )

    # Test the function with 5 samples to match mock model responses
    result = evaluate_proposer(
        mock_model,
        trajectory_solver=solver_model,
        trajectories=trajectories,
        num_samples=5,
        solver_attempts=3,
    )
    # Per sample results
    assert result.sample_results[0].parse_success
    assert result.sample_results[0].problem_spec == first_trajectory.to_spec()
    assert (
        result.sample_results[0].attempted_function_deflists
        == [first_trajectory.function_defs] * 3
    )
    assert result.sample_results[1].parse_success
    assert result.sample_results[1].problem_spec == first_trajectory.to_spec()
    assert result.sample_results[1].attempted_function_deflists == [
        first_trajectory.function_defs
    ] * 2 + [FunctionDefList([])]
    assert not result.sample_results[4].parse_success

    # Check solve rates
    assert result.sample_results[0].solve_rate == 1.0
    assert result.sample_results[1].solve_rate == 2 / 3
    assert result.sample_results[2].solve_rate == 1 / 3
    assert result.sample_results[3].solve_rate == 0.0
    assert result.sample_results[4].solve_rate == 0.0

    # Other results
    assert isinstance(result, EvalResult)
    assert result.num_samples == 5

    # Parse rate: 4 out of 5 samples should parse successfully
    assert result.parse_rate == 4 / 5

    # Function count: only parsed samples contribute
    assert result.avg_function_count == len(first_trajectory.function_defs)

    # Other metrics
    assert result.solver_success_rate == 6 / 12
    assert result.avg_function_count_ratio == 1.0
    assert result.frac_non_zero_std == 2 / 4  # Count only on parsed samples
