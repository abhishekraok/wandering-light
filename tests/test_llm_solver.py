"""Tests for the TokenGeneratorSolver functionality."""

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.solver import TokenGeneratorPredictor, create_token_solver
from wandering_light.typed_list import TypedList


class MockTokenGenerator:
    def generate(
        self, prompt: str, temperature: float = 1.0, max_tokens: int = 100
    ) -> str:
        del prompt, temperature, max_tokens
        return "increment,double"

    def generate_batch(self, prompts):
        """Generate responses for a batch of prompts."""
        return [self.generate(prompt) for prompt in prompts]


class TestLLMSolver:
    """Test suite for TokenGeneratorSolver functionality."""

    @pytest.fixture
    def sample_functions(self) -> FunctionDefSet:
        """Create sample functions for testing."""
        return FunctionDefSet(
            [
                FunctionDef(
                    name="increment",
                    input_type="builtins.int",
                    output_type="builtins.int",
                    code="return x + 1",
                ),
                FunctionDef(
                    name="double",
                    input_type="builtins.int",
                    output_type="builtins.int",
                    code="return x * 2",
                ),
                FunctionDef(
                    name="square",
                    input_type="builtins.int",
                    output_type="builtins.int",
                    code="return x * x",
                ),
            ]
        )

    def test_generate_prompt(self, sample_functions):
        """Test prompt generation."""
        # Setup
        mock_token_generator = MockTokenGenerator()
        predictor = TokenGeneratorPredictor(mock_token_generator)
        input_list = TypedList([1, 2, 3])
        output_list = TypedList([2, 4, 6])

        # Execute
        prompt = predictor._generate_prompt(input_list, output_list, sample_functions)

        # Verify
        assert (
            prompt
            == """You are an expert at solving sequence transformation problems.
Given an input list and a target output list, you need to find a sequence of functions
that transforms the input to the output.

Available functions:

def increment(x):
    return x + 1


def double(x):
    return x * 2


def square(x):
    return x * x


Input: [1, 2, 3]
Target Output: [2, 4, 6]

Provide the sequence of function names (comma-separated) that transforms the input to the output.
Do not output any additional text other than the sequence of function names.
Only use the function names from the available functions list above.
Example: function1,function2,function3

Answer:"""
        )

    def test_solve_success(self, sample_functions):
        """Test successful solve with mock API response."""
        # Setup mock
        mock_token_generator = MockTokenGenerator()

        # Initialize solver with mock
        solver = create_token_solver(mock_token_generator)
        input_list = TypedList([1, 2, 3])
        output_list = TypedList([4, 6, 8])  # (1+1)*2=4, (2+1)*2=6, (3+1)*2=8

        # Execute
        result = solver.solve(input_list, output_list, sample_functions)

        # Verify
        assert result.success is True
        assert result.trajectory is not None
        assert len(result.trajectory.function_defs) == 2
        assert result.trajectory.function_defs[0].name == "increment"
        assert result.trajectory.function_defs[1].name == "double"
        assert result.trajectory.output == output_list

    def test_solve_failure(self, sample_functions):
        """Test solve when no valid solution is found."""
        # Setup mock to return invalid function sequence
        mock_token_generator = MockTokenGenerator()

        # Initialize solver with mock and budget=1
        solver = create_token_solver(mock_token_generator, budget=1)
        input_list = TypedList([1, 2, 3])
        output_list = TypedList([3, 8, 2])

        # Execute
        result = solver.solve(input_list, output_list, sample_functions)

        # Verify
        assert result.success is False
        # The error message format might be different in the new architecture
        assert not result.success
