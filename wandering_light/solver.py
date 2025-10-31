"""Solver implementations for discovering function trajectories.

This module provides several strategies for finding a sequence of functions
that maps an input list to a target output list.  Available approaches include
random search (``RandomSolve``), breadth-first search (``BFSSolve``) and
LLM-based solvers using ``TokenGenerator`` classes.
"""

import abc
import os
from dataclasses import dataclass

import google.generativeai as genai
import openai
from dotenv import load_dotenv
from ollama import ChatResponse, Client
from transformers import pipeline as hf_pipeline, PreTrainedModel

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.llm_utils import generate_eval_prompt
from wandering_light.trajectory import Trajectory, TrajectorySpec
from wandering_light.typed_list import TypedList


@dataclass
class MaybeTrajectory:
    success: bool
    trajectory: Trajectory | None = None
    error_msg: str | None = None


class FunctionPredictor(abc.ABC):
    """
    Base class for function prediction strategies.
    Focuses solely on predicting function sequences without execution.
    """

    @abc.abstractmethod
    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """
        Predict function sequences for a batch of input-output problems.

        Args:
            problems: List of (input_list, output_list) pairs
            available_functions: Available functions for solving

        Returns:
            List of FunctionDefList predictions corresponding to each problem
        """


class TrajectorySolver:
    """
    Handles trajectory execution and comparison using a FunctionPredictor for function prediction.
    """

    def __init__(self, predictor: FunctionPredictor):
        self.predictor = predictor

    def solve_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[MaybeTrajectory]:
        """
        Solve multiple problems in batch using the predictor and shared execution logic.

        Args:
            problems: List of (input_list, output_list) pairs to solve
            available_functions: Available functions for solving

        Returns:
            List of MaybeTrajectory results corresponding to each input problem
        """
        if not problems:
            return []

        # Get predictions for all problems
        predictions = self.predictor.predict_functions_batch(
            problems, available_functions
        )

        # Execute and compare using shared logic
        return [
            self._process_response(pred, input_list, output_list, available_functions)
            for pred, (input_list, output_list) in zip(
                predictions, problems, strict=True
            )
        ]

    def solve(
        self,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
    ) -> MaybeTrajectory:
        """
        Solve a single problem. Convenience wrapper around solve_batch.
        """
        results = self.solve_batch([(input_list, output_list)], available_functions)
        return results[0]

    def _process_response(
        self,
        function_def_list: FunctionDefList,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
    ) -> MaybeTrajectory:
        """
        Process a FunctionDefList into a MaybeTrajectory by executing and comparing results.
        This shared logic is used by solve_batch().
        """
        executor = Executor(available_functions)
        spec = TrajectorySpec(input_list, function_def_list)
        result = executor.execute_trajectory(spec)

        if not result.success:
            # Execution failed due to type mismatch or function error
            return MaybeTrajectory(
                success=False,
                trajectory=None,
                error_msg=result.error_msg,
            )

        traj = result.trajectory
        if traj.output == output_list:
            return MaybeTrajectory(success=True, trajectory=traj)
        # Check if this is an empty function list (no solution found case)
        elif len(function_def_list.functions) == 0 and traj.output == input_list:
            budget = getattr(self.predictor, "budget", "unknown")
            return MaybeTrajectory(
                success=False,
                trajectory=traj,
                error_msg=f"No solution found for output {output_list} within budget {budget}",
            )
        else:
            return MaybeTrajectory(
                success=False,
                trajectory=traj,
                error_msg=f"Output mismatch: expected {output_list}, got {traj.output}",
            )

    def save(self, directory: str):
        """Save any predictor-specific data."""
        if hasattr(self.predictor, "save"):
            self.predictor.save(directory)


class RandomPredictor(FunctionPredictor):
    """
    Randomly predicts function sequences of fixed length.
    """

    def __init__(self, budget: int = 100, path_length: int = 1):
        self.budget = budget
        self.path_length = path_length

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Predict random function sequences for each problem."""
        results = []
        available_functions_list = available_functions.functions

        for input_list, _output_list in problems:
            if available_functions_list:
                spec = TrajectorySpec.create_random_walk(
                    input_list, self.path_length, available_functions_list
                )
                function_def_list = FunctionDefList(spec.function_defs)
                results.append(function_def_list)
            else:
                results.append(FunctionDefList())

        return results


class BFSPredictor(FunctionPredictor):
    """
    Performs breadth-first search through function combinations up to a maximum depth.
    """

    def __init__(self, budget: int = 100, max_depth: int = 3):
        self.budget = budget
        self.max_depth = max_depth

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Predict function sequences using BFS for each problem."""
        results = []

        for input_list, output_list in problems:
            solution = self._bfs_search(input_list, output_list, available_functions)
            results.append(solution)

        return results

    def _bfs_search(
        self,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
    ) -> FunctionDefList:
        """
        Perform BFS to find a function sequence that transforms input_list to output_list.

        Fixed implementation that addresses:
        1. Missing visited state tracking
        2. Incorrect state progression (now builds incrementally)
        3. Inefficient execution (now executes incrementally)
        4. Missing type compatibility checks
        """
        if not available_functions.functions:
            return FunctionDefList()

        # Check if input already equals output (no functions needed)
        if input_list == output_list:
            return FunctionDefList()

        executor = Executor(available_functions)

        # Queue stores tuples of (current_list, function_sequence)
        queue = [(input_list, [])]
        # Track visited states to avoid redundant exploration
        visited = {self._state_key(input_list)}
        trajectories_tried = 0

        while queue and trajectories_tried < self.budget:
            current_list, function_sequence = queue.pop(0)

            # Don't explore beyond max depth
            if len(function_sequence) >= self.max_depth:
                continue

            # Filter functions for type compatibility with current list
            compatible_functions = [
                func
                for func in available_functions.functions
                if self._is_type_compatible(func, current_list)
            ]

            # Try applying each compatible function
            for func in compatible_functions:
                # Execute function incrementally from current_list
                new_list = executor.execute(func, current_list)
                new_sequence = [*function_sequence, func]
                trajectories_tried += 1

                # Check if this reaches the target
                if new_list == output_list:
                    return FunctionDefList(new_sequence)

                # Check budget after each execution
                if trajectories_tried >= self.budget:
                    return FunctionDefList()

                # Check if this state has been visited before
                state_key = self._state_key(new_list)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_list, new_sequence))

        # No solution found within budget/depth
        return FunctionDefList()

    def _state_key(self, typed_list: TypedList) -> tuple:
        """Create a hashable key representing the state of a TypedList."""
        return (typed_list.item_type, tuple(typed_list.items))

    def _is_type_compatible(self, func: FunctionDef, typed_list: TypedList) -> bool:
        """Check if a function's input type is compatible with the typed list's type."""
        return typed_list.item_type == func.input_type_cls()


class TokenGenerator(abc.ABC):
    """Interface for token generators."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a response from the LLM."""

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate responses for a batch of prompts. Default implementation calls generate sequentially.

        Args:
            prompts: List of prompts to generate responses for

        Returns:
            List of responses corresponding to each prompt
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt)
            responses.append(response)
        return responses


class OpenAITokenGenerator(TokenGenerator):
    def __init__(self, model: str):
        self.model = model
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_io_history = []

    def generate(
        self,
        prompt: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        output_text = response.choices[0].message.content
        self.llm_io_history.append((prompt, output_text))
        return output_text

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts. Uses sequential processing for API-based generation."""
        return super().generate_batch(prompts)


class GeminiTokenGenerator(TokenGenerator):
    def __init__(self, model: str = "models/gemini-2.5-flash"):
        self.model = model
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(self.model)
        self.llm_io_history = []

    def generate(
        self,
        prompt: str,
    ) -> str:
        response = self.client.generate_content(prompt)
        output = response.text
        self.llm_io_history.append((prompt, output))
        return output

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts. Uses sequential processing for API-based generation."""
        return super().generate_batch(prompts)


def remove_thinking(response: str) -> str:
    return response.split("</think>")[-1].strip()


class OllamaTokenGenerator(TokenGenerator):
    def __init__(
        self,
        model: str = "phi3:mini",
    ):
        self.model = model
        load_dotenv()
        self.client = Client(os.getenv("OLLAMA_BASE_URL"))
        self.llm_io_history = []

    def generate(self, prompt: str) -> str:
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.message.content
        self.llm_io_history.append((prompt, output))
        return remove_thinking(output)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts. Uses sequential processing for API-based generation."""
        return super().generate_batch(prompts)


class TrainedLLMTokenGenerator(TokenGenerator):
    """Token generator that uses a trained model from SFT training."""

    def __init__(
        self,
        model_or_path: str | PreTrainedModel,
        temperature: float = 0.1,
        tokenizer=None,
    ):
        self.model_or_path = model_or_path
        self.temperature = temperature

        # Handle both model objects and string paths
        if isinstance(model_or_path, str):
            # String path - use the original approach (hf_pipeline auto-detects tokenizer)
            self.pipeline = hf_pipeline("text-generation", model=self.model_or_path)
            self.model = None
            self.tokenizer = None
        elif isinstance(model_or_path, PreTrainedModel):
            # Model object - must provide tokenizer explicitly
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer must be provided when passing a model object. "
                    "Use: TrainedLLMTokenGenerator(model, tokenizer=tokenizer)"
                )

            # Use live model directly without hf_pipeline
            self.model = model_or_path
            self.tokenizer = tokenizer
            self.pipeline = None

            # Set pad_token_id for batch processing if not already set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Set left padding for decoder-only models for better generation results
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
        else:
            raise ValueError(f"Unknown model type: {type(model_or_path)}")

        # Set pad_token_id for batch processing if not already set (for pipeline mode)
        if self.pipeline is not None and self.pipeline.tokenizer.pad_token_id is None:
            self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id

        # Set left padding for decoder-only models for better generation results (for pipeline mode)
        if self.pipeline is not None and hasattr(
            self.pipeline.tokenizer, "padding_side"
        ):
            self.pipeline.tokenizer.padding_side = "left"

        self.llm_io_history = []
        self.inference_batch_size = 64

    def generate(self, prompt: str) -> str:
        if self.pipeline is None and self.model is not None:
            # Use live model directly
            return self._generate_with_live_model([prompt])[0]
        elif self.pipeline is not None:
            # Use pipeline
            result = self.pipeline(
                prompt,
                max_new_tokens=128,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Extract only the completion (remove the original prompt)
            completion = result[len(prompt) :].strip()
            self.llm_io_history.append((prompt, completion))
            return remove_thinking(completion)
        else:
            raise ValueError("No pipeline or model provided")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts using efficient batch inference."""
        if not prompts:
            return []

        if self.pipeline is None and self.model is not None:
            # Use live model directly
            return self._generate_with_live_model(prompts)
        elif self.pipeline is not None:
            # Use HuggingFace pipeline batch inference
            results = self.pipeline(
                prompts,
                max_new_tokens=128,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                batch_size=self.inference_batch_size,
            )

            responses = []
            for prompt, result in zip(prompts, results, strict=True):
                # Extract only the completion (remove the original prompt)
                completion = result[0]["generated_text"][len(prompt) :].strip()
                completion = remove_thinking(completion)
                responses.append(completion)
                self.llm_io_history.append((prompt, completion))

            return responses
        else:
            raise ValueError("No pipeline or model provided")

    def _generate_with_live_model(self, prompts: list[str]) -> list[str]:
        """Generate responses using the live model directly."""
        import torch

        responses = []

        for prompt in prompts:
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )

            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the completion (remove the original prompt)
            completion = generated_text[len(prompt) :].strip()
            completion = remove_thinking(completion)
            responses.append(completion)
            self.llm_io_history.append((prompt, completion))

        return responses


class TokenGeneratorPredictor(FunctionPredictor):
    """
    Function predictor that uses a TokenGenerator (LLM) to predict function sequences.
    """

    def __init__(self, token_generator: TokenGenerator, budget: int = 1):
        """
        Initialize the TokenGenerator predictor.

        Args:
            token_generator: a generic LLM interface
            budget: Number of attempts to make before giving up
        """
        self.token_generator = token_generator
        self.budget = budget

    def _generate_prompt(
        self,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
    ) -> str:
        """Generate the prompt for the LLM."""
        # For trained models, don't include available functions in the prompt
        include_functions = not isinstance(
            self.token_generator, TrainedLLMTokenGenerator
        )
        return generate_eval_prompt(
            input_list, output_list, available_functions, include_functions
        )

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Predict function sequences for a batch of problems using the token generator."""
        if not problems:
            return []

        # For batch processing, we try once per problem (budget=1 per problem)
        # Generate all prompts
        prompts = [
            self._generate_prompt(input_list, output_list, available_functions)
            for input_list, output_list in problems
        ]

        # Get batch responses
        responses = self.token_generator.generate_batch(prompts)

        # Parse responses into FunctionDefList objects
        results = []
        for response in responses:
            function_def_list = available_functions.parse_string(response)
            results.append(function_def_list)
        return results

    def save(self, directory: str):
        """Save the LLM IO histories to the given directory."""
        os.makedirs(directory, exist_ok=True)
        # Each llm input output is saved to as a numbered <number>_input.txt and <number>_output.txt file
        for i, (input, output) in enumerate(self.token_generator.llm_io_history):
            with open(os.path.join(directory, f"{i}_input.txt"), "w") as f:
                f.write(input)
            with open(os.path.join(directory, f"{i}_output.txt"), "w") as f:
                f.write(output)
        print(
            f"Saved {len(self.token_generator.llm_io_history)} LLM IO pairs to {directory}"
        )


# Factory functions for creating common solver configurations
def create_random_solver(budget: int = 100, path_length: int = 1) -> TrajectorySolver:
    """Create a solver with random function prediction."""
    predictor = RandomPredictor(budget=budget, path_length=path_length)
    return TrajectorySolver(predictor)


def create_bfs_solver(budget: int = 100, max_depth: int = 3) -> TrajectorySolver:
    """Create a solver with breadth-first search function prediction."""
    predictor = BFSPredictor(budget=budget, max_depth=max_depth)
    return TrajectorySolver(predictor)


def create_token_solver(
    token_generator: TokenGenerator, budget: int = 1
) -> TrajectorySolver:
    """Create a solver with token generator function prediction."""
    predictor = TokenGeneratorPredictor(token_generator, budget=budget)
    return TrajectorySolver(predictor)


def get_solver_by_name(
    name: str,
    budget: int = 10,
    path_length: int = 4,
    model_name: str = "checkpoints/latest",
) -> TrajectorySolver:
    if name == "random":
        return create_random_solver(budget=budget, path_length=path_length)
    elif name == "bfs":
        return create_bfs_solver(budget=budget, max_depth=path_length)
    elif name == "openai":
        return create_token_solver(
            OpenAITokenGenerator(model="o4-mini-2025-04-16"), budget=budget
        )
    elif name == "gemini":
        return create_token_solver(GeminiTokenGenerator(), budget=budget)
    elif name == "ollama_phi3_mini":
        return create_token_solver(
            OllamaTokenGenerator(model="phi3:mini"), budget=budget
        )
    elif name == "ollama_deepseek_r1_32b":
        return create_token_solver(
            OllamaTokenGenerator(model="deepseek-r1:32b-qwen-distill-q4_K_M"),
            budget=budget,
        )
    elif name == "trained_local":
        return create_token_solver(
            TrainedLLMTokenGenerator(model_or_path=model_name), budget=budget
        )
    else:
        raise ValueError(f"Unknown solver name: {name}")


# Backward compatibility aliases
RandomSolve = create_random_solver
BFSSolve = create_bfs_solver
TokenGeneratorSolver = create_token_solver


if __name__ == "__main__":
    token_generator = TrainedLLMTokenGenerator(model_or_path="checkpoints/latest")
    print(token_generator.generate("Hi"))
    print(token_generator.llm_io_history)
