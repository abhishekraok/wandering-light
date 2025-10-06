import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import (
    FunctionPredictor,
    TokenGenerator,
    TokenGeneratorPredictor,
    TrainedLLMTokenGenerator,
    TrajectorySolver,
)
from wandering_light.typed_list import TypedList


class MockTokenGenerator(TokenGenerator):
    """Mock token generator for testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.llm_io_history = []

    def generate(self, prompt: str) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        self.llm_io_history.append((prompt, response))
        return response

    def generate_batch(self, prompts: list[str]) -> list[str]:
        responses = []
        for prompt in prompts:
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            responses.append(response)
            self.llm_io_history.append((prompt, response))
        return responses


@pytest.fixture
def sample_functions():
    """Create sample functions for testing."""
    increment = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )

    double = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )

    return FunctionDefSet([increment, double])


@pytest.fixture
def sample_trajectory_specs(sample_functions):
    """Create sample trajectory specs for testing."""
    specs = [
        (TypedList([1, 2, 3]), TypedList([2, 3, 4])),  # increment
        (TypedList([1, 2, 3]), TypedList([2, 4, 6])),  # double
        (TypedList([1, 2, 3]), TypedList([4, 6, 8])),  # increment, double
    ]
    return specs


class TestBatchProcessing:
    """Test the new composition-based batch processing architecture."""

    def test_shared_process_response_logic(self, sample_functions):
        """Test that the shared _process_response logic works correctly."""

        # Create a mock predictor that returns specific function sequences
        class MockPredictor(FunctionPredictor):
            def predict_functions_batch(self, problems, available_functions):
                return [
                    available_functions.parse_string("increment"),
                    available_functions.parse_string("double"),
                    available_functions.parse_string("increment,double"),
                ]

        solver = TrajectorySolver(MockPredictor())

        input_list = TypedList([1, 2, 3])
        output_list = TypedList([2, 3, 4])  # increment result

        # Test successful execution
        function_def_list = sample_functions.parse_string("increment")
        result = solver._process_response(
            function_def_list, input_list, output_list, sample_functions
        )
        assert result.success is True
        assert result.trajectory is not None
        assert result.trajectory.output == output_list

        # Test failed execution (wrong output)
        function_def_list = sample_functions.parse_string("double")
        result = solver._process_response(
            function_def_list, input_list, output_list, sample_functions
        )
        assert result.success is False
        assert result.trajectory is not None
        assert "Output mismatch" in result.error_msg

        # Test no-function response processing
        function_def_list = FunctionDefList()
        result = solver._process_response(
            function_def_list, input_list, output_list, sample_functions
        )
        assert result.success is False
        assert (
            "No solution found" in result.error_msg
        )  # Empty function list = no solution found

    def test_token_generator_batch_vs_individual(self):
        """Test that batch generation produces the same results as individual generation."""
        responses = ["increment", "double", "increment,double"]
        mock_generator = MockTokenGenerator(responses)

        prompts = ["prompt1", "prompt2", "prompt3"]

        # Individual generation
        individual_responses = []
        for prompt in prompts:
            response = mock_generator.generate(prompt)
            individual_responses.append(response)

        # Reset call count
        mock_generator.call_count = 0
        mock_generator.llm_io_history = []

        # Batch generation
        batch_responses = mock_generator.generate_batch(prompts)

        assert individual_responses == batch_responses
        assert len(mock_generator.llm_io_history) == len(prompts)

    def test_solver_batch_vs_individual(self, sample_functions):
        """Test that solver batch processing uses shared logic and produces correct results."""
        # Create mock predictor with predictable responses
        responses = ["increment", "double", "increment,double"]
        mock_generator = MockTokenGenerator(responses)
        predictor = TokenGeneratorPredictor(mock_generator, budget=1)
        solver = TrajectorySolver(predictor)

        # Create test problems
        test_problems = [
            (TypedList([1, 2, 3]), TypedList([2, 3, 4])),  # increment
            (TypedList([1, 2, 3]), TypedList([2, 4, 6])),  # double
            (TypedList([1, 2, 3]), TypedList([4, 6, 8])),  # increment,double
        ]

        # Test batch solving directly
        batch_results = solver.solve_batch(test_problems, sample_functions)

        # Verify results
        assert len(batch_results) == 3
        assert batch_results[0].success is True  # increment should work
        assert batch_results[1].success is True  # double should work
        assert batch_results[2].success is True  # increment,double should work

        # Verify that the shared logic was used by checking trajectories
        assert batch_results[0].trajectory.output == TypedList([2, 3, 4])
        assert batch_results[1].trajectory.output == TypedList([2, 4, 6])
        assert batch_results[2].trajectory.output == TypedList([4, 6, 8])

        # Verify that batch generation was called
        assert (
            mock_generator.call_count == 3
        )  # Should have called generate_batch once with 3 prompts

    def test_individual_solve_uses_shared_logic(self, sample_functions):
        """Test that individual solve() method also uses the shared _process_response logic."""
        mock_generator = MockTokenGenerator(["increment"])
        predictor = TokenGeneratorPredictor(mock_generator, budget=1)
        solver = TrajectorySolver(predictor)

        input_list = TypedList([1, 2, 3])
        output_list = TypedList([2, 3, 4])  # increment result

        # Test individual solve
        result = solver.solve(input_list, output_list, sample_functions)

        # Should be successful and use shared logic
        assert result.success is True
        assert result.trajectory is not None
        assert result.trajectory.output == output_list
        assert len(result.trajectory.function_defs) == 1
        assert result.trajectory.function_defs[0].name == "increment"

    def test_batch_processing_with_empty_batch(self, sample_functions):
        """Test that batch processing handles empty batches correctly."""
        mock_generator = MockTokenGenerator([])
        predictor = TokenGeneratorPredictor(mock_generator, budget=1)
        solver = TrajectorySolver(predictor)

        # Test empty batch
        results = solver.solve_batch([], sample_functions)
        assert results == []

        # Test empty prompts in predictor
        predictions = predictor.predict_functions_batch([], sample_functions)
        assert predictions == []

    def test_batch_processing_efficiency(self, sample_functions):
        """Test that batch processing is more efficient than individual calls."""

        class CountingTokenGenerator(TokenGenerator):
            def __init__(self):
                self.llm_io_history = []
                self.individual_calls = 0
                self.batch_calls = 0

            def generate(self, prompt: str) -> str:
                self.individual_calls += 1
                response = "increment"
                self.llm_io_history.append((prompt, response))
                return response

            def generate_batch(self, prompts: list[str]) -> list[str]:
                self.batch_calls += 1
                responses = ["increment"] * len(prompts)
                for prompt, response in zip(prompts, responses, strict=False):
                    self.llm_io_history.append((prompt, response))
                return responses

        counting_generator = CountingTokenGenerator()
        predictor = TokenGeneratorPredictor(counting_generator, budget=1)
        solver = TrajectorySolver(predictor)

        test_problems = [
            (TypedList([1, 2, 3]), TypedList([2, 3, 4])),
            (TypedList([2, 3, 4]), TypedList([3, 4, 5])),
            (TypedList([3, 4, 5]), TypedList([4, 5, 6])),
        ]

        # Test batch processing
        results = solver.solve_batch(test_problems, sample_functions)

        assert len(results) == 3
        assert counting_generator.batch_calls == 1  # Only one batch call
        assert counting_generator.individual_calls == 0  # No individual calls

    def test_trained_llm_batch_generation(self):
        """Test that TrainedLLMTokenGenerator has proper batch generation."""
        # This is a unit test for the batch generation method specifically
        # We'll mock the pipeline to avoid needing actual model files

        class MockPipeline:
            def __init__(self):
                self.call_count = 0

                class MockTokenizer:
                    eos_token_id = 0

                self.tokenizer = MockTokenizer()

            def __call__(self, prompts, **kwargs):
                self.call_count += 1
                if isinstance(prompts, str):
                    # Single prompt
                    return [{"generated_text": f"{prompts} response"}]
                else:
                    # Batch of prompts - each result is a list containing a dict
                    return [
                        [{"generated_text": f"{prompt} response"}] for prompt in prompts
                    ]

        # Create a mock pipeline
        mock_pipeline = MockPipeline()

        # Create token generator and replace the pipeline
        generator = TrainedLLMTokenGenerator.__new__(
            TrainedLLMTokenGenerator
        )  # Skip __init__
        generator.model_or_path = "mock"
        generator.pipeline = mock_pipeline
        generator.llm_io_history = []
        generator.inference_batch_size = 64
        generator.temperature = 0.1
        generator.use_live_model = False

        # Test batch generation
        prompts = ["prompt1", "prompt2", "prompt3"]
        responses = generator.generate_batch(prompts)

        assert len(responses) == 3
        assert all("response" in response for response in responses)
        assert mock_pipeline.call_count == 1  # Should be called once for the batch
        assert len(generator.llm_io_history) == 3  # Should record all interactions
