"""Solver implementations for discovering function trajectories.

This module provides several strategies for finding a sequence of functions
that maps an input list to a target output list.  Available approaches include
random search (``RandomSolve``), breadth-first search (``BFSSolve``), a
materialized-state beam rollout (``RolloutPredictor``) and LLM-based solvers
using ``TokenGenerator`` classes.
"""

import abc
import math
import numbers
import os
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import google.generativeai as genai
import openai
from dotenv import load_dotenv
from ollama import ChatResponse, Client
from transformers import PreTrainedModel
from transformers import pipeline as hf_pipeline

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.function_usage import FunctionUsageTracker
from wandering_light.llm_utils import generate_eval_prompt
from wandering_light.trajectory import Trajectory, TrajectorySpec
from wandering_light.typed_list import TypedList


@dataclass
class MaybeTrajectory:
    success: bool
    trajectory: Trajectory | None = None
    error_msg: str | None = None


@dataclass(frozen=True)
class BFSSearchResult:
    """A BFS prediction together with the search cost and stop reason."""

    function_defs: FunctionDefList
    executions: int
    termination: Literal["solved", "budget", "max_depth", "frontier_exhausted"]


class FunctionPredictor(abc.ABC):
    """
    Base class for function prediction strategies.
    Focuses solely on predicting function sequences without execution.
    """

    track_function_usage = True

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

    def __init__(
        self,
        predictor: FunctionPredictor,
        usage_tracker: FunctionUsageTracker | None = None,
        track_function_usage: bool | None = None,
    ):
        self.predictor = predictor
        self.function_usage = (
            usage_tracker if usage_tracker is not None else FunctionUsageTracker()
        )
        if track_function_usage is None:
            track_function_usage = getattr(predictor, "track_function_usage", True)
        self.track_function_usage = track_function_usage

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

        if self.track_function_usage:
            basis_set_id, basis_digest = FunctionUsageTracker.basis_provenance(
                available_functions
            )
            self.function_usage.bind_basis(basis_set_id, basis_digest)

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
            if self.track_function_usage:
                self.function_usage.record_solution(
                    traj.function_defs,
                    basis_set_id=self.function_usage.basis_set_id,
                    basis_digest=self.function_usage.basis_digest,
                )
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
        """Save predictor-specific data and successful-solution usage."""
        if hasattr(self.predictor, "save"):
            self.predictor.save(directory)
        self.function_usage.save(
            os.path.join(directory, FunctionUsageTracker.FILE_NAME)
        )


class RandomPredictor(FunctionPredictor):
    """
    Randomly predicts function sequences of fixed length.
    """

    track_function_usage = False

    def __init__(self, budget: int = 100, path_length: int = 1):
        self.budget = budget
        self.path_length = path_length

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Try up to ``budget`` random sequences for each problem."""
        results = []
        available_functions_list = available_functions.functions
        executor = Executor(available_functions)

        for input_list, output_list in problems:
            prediction = FunctionDefList()
            for _ in range(self.budget if available_functions_list else 0):
                spec = TrajectorySpec.create_random_walk(
                    input_list, self.path_length, available_functions_list
                )
                execution = executor.execute_trajectory(spec)
                if execution.success and execution.trajectory.output == output_list:
                    prediction = FunctionDefList(list(spec.function_defs))
                    break
            results.append(prediction)

        return results


class BFSPredictor(FunctionPredictor):
    """
    Performs breadth-first search through function combinations up to a maximum depth.
    """

    def __init__(self, budget: int = 100, max_depth: int = 3):
        """`budget` counts **basis executions**, not trajectories.

        Every type-compatible function tried costs one, including attempts
        whose function raises. This changed with #50: it previously counted
        trajectories dequeued, and executions are always the larger number, so
        the same value is now a strictly tighter bound. Figures recorded before
        that commit -- notably the reference curve in
        `reports/deep-corpus-20260816/` -- are in the older unit and are not
        directly comparable.
        """
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
        """Return only the predicted functions for the legacy predictor API."""
        return self.search_with_cost(
            input_list, output_list, available_functions
        ).function_defs

    def search_with_cost(
        self,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
    ) -> BFSSearchResult:
        """
        Find a function sequence and report the compatible executions attempted.

        Fixed implementation that addresses:
        1. Missing visited state tracking
        2. Incorrect state progression (now builds incrementally)
        3. Inefficient execution (now executes incrementally)
        4. Missing type compatibility checks
        """
        # Check if input already equals output (no functions needed)
        if input_list == output_list:
            return BFSSearchResult(FunctionDefList(), 0, "solved")

        if not available_functions.functions:
            return BFSSearchResult(FunctionDefList(), 0, "frontier_exhausted")

        executor = Executor(available_functions)

        # Queue stores tuples of (current_list, function_sequence). A deque keeps
        # the frontier pop O(1); a list makes deep searches quadratic in the
        # number of visited states, which dominates past depth four.
        queue = deque([(input_list, [])])
        # Track visited states to avoid redundant exploration
        visited = {self._state_key(input_list)}
        executions = 0
        reached_max_depth = False

        while queue and executions < self.budget:
            current_list, function_sequence = queue.popleft()

            # Don't explore beyond max depth
            if len(function_sequence) >= self.max_depth:
                reached_max_depth = True
                continue

            # Filter functions for type compatibility with current list
            compatible_functions = [
                func
                for func in available_functions.functions
                if self._is_type_compatible(func, current_list)
            ]

            # Try applying each compatible function
            for func in compatible_functions:
                new_sequence = [*function_sequence, func]
                executions += 1

                # Execute function incrementally from current_list.
                # _is_type_compatible only checks the container's item type, so a
                # function can accept this list and still raise on its contents
                # (list_median over a list of strings). Treat a raising expansion
                # as an absent transition, matching TrajectoryGraph.expand.
                try:
                    new_list = executor.execute(func, current_list)
                except Exception:
                    if executions >= self.budget:
                        return BFSSearchResult(FunctionDefList(), executions, "budget")
                    continue

                # Check if this reaches the target
                if new_list == output_list:
                    return BFSSearchResult(
                        FunctionDefList(new_sequence), executions, "solved"
                    )

                # Check budget after each execution
                if executions >= self.budget:
                    return BFSSearchResult(FunctionDefList(), executions, "budget")

                # Check if this state has been visited before
                state_key = self._state_key(new_list)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_list, new_sequence))

        if executions >= self.budget:
            return BFSSearchResult(FunctionDefList(), executions, "budget")
        if reached_max_depth:
            return BFSSearchResult(FunctionDefList(), executions, "max_depth")
        return BFSSearchResult(FunctionDefList(), executions, "frontier_exhausted")

    def _state_key(self, typed_list: TypedList) -> tuple:
        """Create a hashable key representing the state of a TypedList."""
        return typed_list.search_key()

    def _is_type_compatible(self, func: FunctionDef, typed_list: TypedList) -> bool:
        """Check if a function's input type is compatible with the typed list's type."""
        return typed_list.item_type == func.input_type_cls()


class ActionScorer(abc.ABC):
    """Scores the next basis function to apply from a materialized state.

    A rollout policy sees the concrete intermediate list it has reached rather
    than only the original input, so each problem is a ``(state, target)``
    pair. Scores are unnormalized logits: ``RolloutPredictor`` log-softmaxes
    each row over the functions that are type-compatible with the state.
    """

    @abc.abstractmethod
    def score_actions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[dict[str, float]]:
        """Return one row of scores per problem, keyed by every function name."""


@dataclass(frozen=True)
class RolloutDepthTrace:
    """The beam retained after one complete rollout layer."""

    depth: int
    retained_paths: tuple[tuple[str, ...], ...]
    unique_successors: int


@dataclass(frozen=True)
class RolloutSearchResult:
    """A rollout prediction with its cost, stop reason and per-layer trace.

    ``trace`` is diagnostic only and is excluded from equality, so results
    compare like ``BFSSearchResult``.
    """

    function_defs: FunctionDefList
    executions: int
    termination: Literal["solved", "budget", "max_depth", "frontier_exhausted"]
    trace: tuple[RolloutDepthTrace, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class _RolloutNode:
    state: TypedList
    path: tuple[FunctionDef, ...]
    log_prob: float


def _validated_limit(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _log_softmax(scores: list[float]) -> list[float]:
    """Numerically stable log-softmax of a non-empty list of finite scores."""
    peak = max(scores)
    log_total = peak + math.log(sum(math.exp(score - peak) for score in scores))
    return [score - log_total for score in scores]


class RolloutPredictor(FunctionPredictor):
    """Beam search whose policy observes every materialized intermediate state.

    Each layer asks the ``ActionScorer`` to score every available function
    from every beam state, expands the beam with type-compatible functions in
    order of descending path log-probability, and keeps the ``beam_width`` most
    probable distinct successors. Like ``BFSPredictor``, ``budget`` counts
    basis executions: every attempted transition costs one, including attempts
    whose function raises.
    """

    def __init__(
        self,
        action_scorer: ActionScorer,
        beam_width: int = 8,
        max_depth: int = 8,
        budget: int = 100_000,
    ):
        self.action_scorer = action_scorer
        self.beam_width = _validated_limit("beam_width", beam_width, 1)
        self.max_depth = _validated_limit("max_depth", max_depth, 0)
        self.budget = _validated_limit("budget", budget, 0)

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Search each problem independently, each with its own budget."""
        return [
            self.search_with_cost(
                input_list, output_list, available_functions
            ).function_defs
            for input_list, output_list in problems
        ]

    def search_with_cost(
        self,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
    ) -> RolloutSearchResult:
        """Find a function sequence by beam rollout and report what it cost."""
        if input_list == output_list:
            return RolloutSearchResult(FunctionDefList(), 0, "solved")
        functions = list(available_functions.functions)
        if not functions:
            return RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
        if self.budget == 0:
            return RolloutSearchResult(FunctionDefList(), 0, "budget")
        if self.max_depth == 0:
            return RolloutSearchResult(FunctionDefList(), 0, "max_depth")

        executor = Executor(available_functions)
        names = [function.name for function in functions]
        input_types = [function.input_type_cls() for function in functions]
        beam = [_RolloutNode(input_list, (), 0.0)]
        trace: list[RolloutDepthTrace] = []
        executions = 0

        def unsolved(termination: str) -> RolloutSearchResult:
            return RolloutSearchResult(
                FunctionDefList(), executions, termination, tuple(trace)
            )

        for depth in range(1, self.max_depth + 1):
            rows = self.action_scorer.score_actions_batch(
                [(node.state, output_list) for node in beam], available_functions
            )
            scores = self._validated_rows(rows, len(beam), names)

            # (path log-probability, parent position, function position, parent)
            transitions = []
            for parent_index, (node, row) in enumerate(zip(beam, scores, strict=True)):
                compatible = [
                    index
                    for index, input_type in enumerate(input_types)
                    if node.state.item_type is input_type
                ]
                if not compatible:
                    continue
                log_probs = _log_softmax([row[index] for index in compatible])
                for index, log_prob in zip(compatible, log_probs, strict=True):
                    transitions.append(
                        (node.log_prob + log_prob, parent_index, index, node)
                    )
            if not transitions:
                return unsolved("frontier_exhausted")
            transitions.sort(key=lambda item: (-item[0], item[1], item[2]))

            # search_key -> (attempt order, successor node)
            successors: dict[tuple, tuple[int, _RolloutNode]] = {}
            for attempt, (log_prob, _, index, parent) in enumerate(transitions):
                function = functions[index]
                executions += 1
                try:
                    state = executor.execute(function, parent.state)
                except Exception:
                    # Type compatibility only checks the container's item type;
                    # a function can still raise on the contents. That attempt
                    # is paid for and then treated as an absent transition.
                    if executions >= self.budget:
                        return unsolved("budget")
                    continue
                path = (*parent.path, function)
                if state == output_list:
                    return RolloutSearchResult(
                        FunctionDefList(list(path)), executions, "solved", tuple(trace)
                    )
                if executions >= self.budget:
                    return unsolved("budget")
                key = state.search_key()
                previous = successors.get(key)
                # Attempts run in descending log-probability, so a later
                # duplicate cannot normally beat the first; the strict
                # comparison keeps the earliest attempt on ties.
                if previous is None or log_prob > previous[1].log_prob:
                    successors[key] = (attempt, _RolloutNode(state, path, log_prob))

            ranked = sorted(
                successors.values(), key=lambda item: (-item[1].log_prob, item[0])
            )
            beam = [node for _, node in ranked[: self.beam_width]]
            trace.append(
                RolloutDepthTrace(
                    depth=depth,
                    retained_paths=tuple(
                        tuple(function.name for function in node.path) for node in beam
                    ),
                    unique_successors=len(successors),
                )
            )
            if not beam:
                return unsolved("frontier_exhausted")

        return unsolved("max_depth")

    @staticmethod
    def _validated_rows(
        rows: object, expected_rows: int, names: list[str]
    ) -> list[list[float]]:
        """Check a scorer's output and return each row in basis order."""
        if not isinstance(rows, list) or len(rows) != expected_rows:
            returned = (
                len(rows) if isinstance(rows, list) else f"a {type(rows).__name__}, not"
            )
            raise ValueError(
                f"action scorer returned {returned} rows; expected {expected_rows}"
            )
        expected_names = set(names)
        validated = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected_names:
                got = sorted(map(str, row)) if isinstance(row, Mapping) else type(row)
                raise ValueError(
                    f"action scorer row {row_index} has wrong function names: "
                    f"expected {sorted(expected_names)}, got {got}"
                )
            values = []
            for name in names:
                score = row[name]
                value = math.nan
                if isinstance(score, numbers.Real) and not isinstance(score, bool):
                    try:
                        value = float(score)
                    except (OverflowError, TypeError, ValueError):
                        value = math.nan
                if not math.isfinite(value):
                    raise ValueError(
                        f"action scorer row {row_index} has a non-finite or "
                        f"non-numeric score for {name!r}: {score!r}"
                    )
                values.append(value)
            validated.append(values)
        return validated


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
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        self.llm_io_history.append((prompt, response.output_text))
        return response.output_text

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
    return response.rsplit("</think>", maxsplit=1)[-1].strip()


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
            for prompt, result in zip(prompts, results, strict=False):
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
        """Return the shortest valid prediction from ``budget`` attempts."""
        if not problems:
            return []
        if self.budget <= 0:
            return [FunctionDefList() for _ in problems]

        prompts = [
            self._generate_prompt(input_list, output_list, available_functions)
            for input_list, output_list in problems
        ]
        responses = self.token_generator.generate_batch(prompts * self.budget)
        expected = len(prompts) * self.budget
        if len(responses) != expected:
            raise ValueError(
                f"token generator returned {len(responses)} responses; expected {expected}"
            )

        executor = Executor(available_functions)
        predictions = [FunctionDefList() for _ in problems]
        best_lengths = [float("inf") for _ in problems]
        for response_index, response in enumerate(responses):
            problem_index = response_index % len(problems)
            input_list, output_list = problems[problem_index]
            candidate = available_functions.parse_string(response)
            execution = executor.execute_trajectory(
                TrajectorySpec(input_list, candidate)
            )
            if (
                execution.success
                and execution.trajectory.output == output_list
                and len(candidate) < best_lengths[problem_index]
            ):
                predictions[problem_index] = candidate
                best_lengths[problem_index] = len(candidate)
        return predictions

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
def create_random_solver(
    budget: int = 100,
    path_length: int = 1,
    usage_tracker: FunctionUsageTracker | None = None,
    track_function_usage: bool | None = None,
) -> TrajectorySolver:
    """Create a solver with random function prediction."""
    predictor = RandomPredictor(budget=budget, path_length=path_length)
    return TrajectorySolver(predictor, usage_tracker, track_function_usage)


def create_bfs_solver(
    budget: int = 100,
    max_depth: int = 3,
    usage_tracker: FunctionUsageTracker | None = None,
    track_function_usage: bool | None = None,
) -> TrajectorySolver:
    """Create a solver with breadth-first search function prediction."""
    predictor = BFSPredictor(budget=budget, max_depth=max_depth)
    return TrajectorySolver(predictor, usage_tracker, track_function_usage)


def create_token_solver(
    token_generator: TokenGenerator,
    budget: int = 1,
    usage_tracker: FunctionUsageTracker | None = None,
    track_function_usage: bool | None = None,
) -> TrajectorySolver:
    """Create a solver with token generator function prediction."""
    predictor = TokenGeneratorPredictor(token_generator, budget=budget)
    return TrajectorySolver(predictor, usage_tracker, track_function_usage)


def get_solver_by_name(
    name: str,
    budget: int = 10,
    path_length: int = 4,
    model_name: str = "checkpoints/latest",
    usage_tracker: FunctionUsageTracker | None = None,
    track_function_usage: bool | None = None,
) -> TrajectorySolver:
    if name == "random":
        return create_random_solver(
            budget=budget,
            path_length=path_length,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
        )
    elif name == "bfs":
        return create_bfs_solver(
            budget=budget,
            max_depth=path_length,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
        )
    elif name == "openai":
        return create_token_solver(
            OpenAITokenGenerator(model="o4-mini-2025-04-16"),
            budget=budget,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
        )
    elif name == "gemini":
        return create_token_solver(
            GeminiTokenGenerator(),
            budget=budget,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
        )
    elif name == "ollama_phi3_mini":
        return create_token_solver(
            OllamaTokenGenerator(model="phi3:mini"),
            budget=budget,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
        )
    elif name == "ollama_deepseek_r1_32b":
        return create_token_solver(
            OllamaTokenGenerator(model="deepseek-r1:32b-qwen-distill-q4_K_M"),
            budget=budget,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
        )
    elif name == "trained_local":
        return create_token_solver(
            TrainedLLMTokenGenerator(model_or_path=model_name),
            budget=budget,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
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
