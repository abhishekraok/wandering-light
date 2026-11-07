"""Solver implementations for discovering function trajectories.

This module provides several strategies for finding a sequence of functions
that maps an input list to a target output list.  Available approaches include
random search (``RandomSolve``), breadth-first search (``BFSSolve``) and
LLM-based solvers using ``TokenGenerator`` classes.
"""

import abc
import hashlib
import math
import os
import random
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import anthropic
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

BASIS_EXECUTION_COST_UNIT = "basis_function_executions"
BFS_COST_DEFINITION = (
    "One execution is charged for every type-compatible basis function tried, "
    "including a function that raises. The successful path is not charged again "
    "when the common solver wrapper verifies it."
)
SAMPLED_PROGRAM_COST_DEFINITION = (
    "Each syntactically valid candidate program is charged its full function "
    "sequence length, including candidates that fail during execution. Malformed "
    "candidates cost zero and cannot solve."
)
WALL_CLOCK_COST_UNIT = "wall_clock_seconds"
WALL_CLOCK_COST_DEFINITION = (
    "Wall-clock seconds of the solve call on the evaluation host, including "
    "model inference and search bookkeeping. This is the primary comparable "
    "cost across solvers; charged executions remain the hardware-independent "
    "diagnostic. Only comparable between runs on the same host."
)


@dataclass(frozen=True)
class PredictionCost:
    """Comparable search cost for one prediction."""

    charged_executions: int
    candidate_programs: int | None
    termination: str
    seconds: float | None = None


@dataclass(frozen=True)
class FunctionPrediction:
    """A predicted program plus optional cost telemetry."""

    function_defs: FunctionDefList
    cost: PredictionCost | None = None


@dataclass
class MaybeTrajectory:
    success: bool
    trajectory: Trajectory | None = None
    error_msg: str | None = None
    cost: PredictionCost | None = None


@dataclass(frozen=True)
class BFSSearchResult:
    """A BFS prediction together with the search cost and stop reason."""

    function_defs: FunctionDefList
    executions: int
    termination: Literal["solved", "budget", "max_depth", "frontier_exhausted"]


@dataclass(frozen=True)
class RolloutDepthTrace:
    """The paths retained after one complete rollout layer."""

    depth: int
    retained_paths: tuple[tuple[str, ...], ...]
    unique_successors: int


@dataclass(frozen=True)
class RolloutSearchResult:
    """A rollout prediction together with the search cost and stop reason."""

    function_defs: FunctionDefList
    executions: int
    termination: Literal["solved", "budget", "max_depth", "frontier_exhausted"]
    trace: tuple[RolloutDepthTrace, ...] = field(default=(), compare=False)


class FunctionPredictor(abc.ABC):
    """
    Base class for function prediction strategies.
    Focuses solely on predicting function sequences without execution.
    """

    track_function_usage = True
    cost_unit: str | None = None
    cost_definition: str | None = None
    budget_unit: str | None = None

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

    def predict_functions_batch_with_cost(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionPrediction]:
        """Predict with telemetry, preserving custom predictors without it."""
        return [
            FunctionPrediction(function_defs)
            for function_defs in self.predict_functions_batch(
                problems,
                available_functions,
            )
        ]

    def cost_metadata(self) -> dict[str, object]:
        """Describe the predictor's budget and common reported cost."""
        return {
            "cost_unit": self.cost_unit,
            "cost_definition": self.cost_definition,
            "budget_unit": self.budget_unit,
            "budget_value": getattr(self, "budget", None),
        }


class ActionScorer(abc.ABC):
    """Score the next basis function from each materialized state and target."""

    @abc.abstractmethod
    def score_actions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[dict[str, float]]:
        """Return one unnormalized score per available function and problem."""


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
        predictions = self.predictor.predict_functions_batch_with_cost(
            problems, available_functions
        )

        # Execute and compare using shared logic
        return [
            self._process_response(
                prediction.function_defs,
                input_list,
                output_list,
                available_functions,
                prediction.cost,
            )
            for prediction, (input_list, output_list) in zip(
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
        prediction_cost: PredictionCost | None = None,
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
                cost=prediction_cost,
            )

        traj = result.trajectory
        if traj.output == output_list:
            if self.track_function_usage:
                self.function_usage.record_solution(
                    traj.function_defs,
                    basis_set_id=self.function_usage.basis_set_id,
                    basis_digest=self.function_usage.basis_digest,
                )
            return MaybeTrajectory(
                success=True,
                trajectory=traj,
                cost=prediction_cost,
            )
        # Check if this is an empty function list (no solution found case)
        elif len(function_def_list.functions) == 0 and traj.output == input_list:
            budget = getattr(self.predictor, "budget", "unknown")
            return MaybeTrajectory(
                success=False,
                trajectory=traj,
                error_msg=f"No solution found for output {output_list} within budget {budget}",
                cost=prediction_cost,
            )
        else:
            return MaybeTrajectory(
                success=False,
                trajectory=traj,
                error_msg=f"Output mismatch: expected {output_list}, got {traj.output}",
                cost=prediction_cost,
            )

    def cost_metadata(self) -> dict[str, object]:
        """Return the predictor's declared budget and cost semantics."""
        return self.predictor.cost_metadata()

    def save(self, directory: str):
        """Save predictor-specific data and successful-solution usage."""
        if hasattr(self.predictor, "save"):
            self.predictor.save(directory)
        self.function_usage.save(
            os.path.join(directory, FunctionUsageTracker.FILE_NAME)
        )


class RandomPredictor(FunctionPredictor):
    """
    Randomly predicts programs with lengths sampled from an explicit set.
    """

    track_function_usage = False
    cost_unit = BASIS_EXECUTION_COST_UNIT
    cost_definition = SAMPLED_PROGRAM_COST_DEFINITION
    budget_unit = "candidate_programs"

    def __init__(
        self,
        budget: int = 100,
        path_length: int = 1,
        *,
        path_lengths: Sequence[int] | None = None,
        seed: int | None = None,
    ):
        if budget < 0:
            raise ValueError("budget must be non-negative")
        sampled_lengths = (
            (path_length,) if path_lengths is None else tuple(path_lengths)
        )
        if not sampled_lengths:
            raise ValueError("path_lengths must contain at least one length")
        if any(length < 0 for length in sampled_lengths):
            raise ValueError("path lengths must be non-negative")
        self.budget = budget
        self.path_length = path_length
        self.path_lengths = sampled_lengths
        self.seed = seed
        self._base_seed = (
            seed if seed is not None else random.SystemRandom().getrandbits(128)
        )

    def _rng_for_problem(
        self,
        input_list: TypedList,
        output_list: TypedList,
    ) -> random.Random:
        payload = "\0".join(
            (
                str(self._base_seed),
                input_list.to_string(),
                output_list.to_string(),
            )
        ).encode()
        derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
        return random.Random(derived_seed)

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Return only programs for the legacy predictor API."""
        return [
            prediction.function_defs
            for prediction in self.predict_functions_batch_with_cost(
                problems,
                available_functions,
            )
        ]

    def predict_functions_batch_with_cost(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionPrediction]:
        """Try at most ``budget`` candidates per problem and report their cost."""
        results: list[FunctionPrediction] = []
        available_functions_list = available_functions.functions
        executor = Executor(available_functions)

        for input_list, output_list in problems:
            started = time.perf_counter()
            if input_list == output_list:
                results.append(
                    FunctionPrediction(
                        FunctionDefList(),
                        PredictionCost(0, 0, "solved", time.perf_counter() - started),
                    )
                )
                continue
            if not available_functions_list:
                results.append(
                    FunctionPrediction(
                        FunctionDefList(),
                        PredictionCost(
                            0,
                            0,
                            "frontier_exhausted",
                            time.perf_counter() - started,
                        ),
                    )
                )
                continue

            prediction = FunctionDefList()
            rng = self._rng_for_problem(input_list, output_list)
            charged_executions = 0
            candidates = 0
            termination = "candidate_budget"
            for _ in range(self.budget):
                path_length = rng.choice(self.path_lengths)
                spec = TrajectorySpec.create_random_walk(
                    input_list,
                    path_length,
                    available_functions_list,
                    rng=rng,
                )
                candidates += 1
                charged_executions += len(spec.function_defs)
                execution = executor.execute_trajectory(spec)
                if execution.success and execution.trajectory.output == output_list:
                    prediction = FunctionDefList(list(spec.function_defs))
                    termination = "solved"
                    break
            results.append(
                FunctionPrediction(
                    prediction,
                    PredictionCost(
                        charged_executions,
                        candidates,
                        termination,
                        time.perf_counter() - started,
                    ),
                )
            )

        return results

    def cost_metadata(self) -> dict[str, object]:
        metadata = super().cost_metadata()
        metadata.update(
            {
                "path_length_sampling": "uniform",
                "path_lengths": list(self.path_lengths),
                "seed": self.seed,
            }
        )
        return metadata


class BFSPredictor(FunctionPredictor):
    """
    Performs breadth-first search through function combinations up to a maximum depth.
    """

    cost_unit = BASIS_EXECUTION_COST_UNIT
    cost_definition = BFS_COST_DEFINITION
    budget_unit = BASIS_EXECUTION_COST_UNIT

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
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self.budget = budget
        self.max_depth = max_depth

    def cost_metadata(self) -> dict[str, object]:
        metadata = super().cost_metadata()
        metadata["max_depth"] = self.max_depth
        return metadata

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Return only programs for the legacy predictor API."""
        return [
            prediction.function_defs
            for prediction in self.predict_functions_batch_with_cost(
                problems,
                available_functions,
            )
        ]

    def predict_functions_batch_with_cost(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionPrediction]:
        """Predict with independent per-problem execution counts and timings."""
        predictions = []
        for input_list, output_list in problems:
            started = time.perf_counter()
            result = self.search_with_cost(
                input_list,
                output_list,
                available_functions,
            )
            predictions.append(
                FunctionPrediction(
                    result.function_defs,
                    PredictionCost(
                        result.executions,
                        None,
                        result.termination,
                        time.perf_counter() - started,
                    ),
                )
            )
        return predictions

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


@dataclass(frozen=True)
class _RolloutNode:
    state: TypedList
    function_defs: tuple[FunctionDef, ...]
    log_probability: float
    order: int


class RolloutObserver:
    """Watch one rollout search without being able to steer it.

    Every method is a no-op hook. ``search_with_cost`` reports what it already
    decided; nothing an observer returns is read, so instrumenting a search
    cannot change its result. Used by the interactive rollout debugger to
    record a trace from the production search rather than a reimplementation.
    """

    def layer_scored(
        self,
        depth: int,
        nodes: Sequence[_RolloutNode],
        score_rows: Sequence[dict[str, float]],
    ) -> None:
        """One scorer call has produced raw logits for the whole beam."""

    def node_masked(
        self,
        depth: int,
        parent_index: int,
        compatible: Sequence[FunctionDef],
        log_normalizer: float,
    ) -> None:
        """The type mask and per-node normalizer for one beam node."""

    def expansion_attempted(
        self,
        depth: int,
        order: int,
        parent_index: int,
        function: FunctionDef,
        log_probability: float,
        outcome: str,
        successor: TypedList | None,
        error: BaseException | None,
    ) -> None:
        """One type-compatible transition was charged one execution.

        ``outcome`` is ``raised``, ``solved``, ``retained``, ``duplicate`` or
        ``budget``: the fate of this single attempt, before beam truncation.
        """

    def layer_retained(
        self,
        depth: int,
        beam: Sequence[_RolloutNode],
        successors: Mapping[tuple, _RolloutNode],
    ) -> None:
        """The layer's unique successors, and the prefix of them retained."""

    def search_terminated(self, termination: str, executions: int, depth: int) -> None:
        """The search stopped, for the reason it reports to its caller."""


class RolloutPredictor(FunctionPredictor):
    """Beam search whose policy observes every materialized intermediate state."""

    cost_unit = BASIS_EXECUTION_COST_UNIT
    cost_definition = BFS_COST_DEFINITION
    budget_unit = BASIS_EXECUTION_COST_UNIT

    def __init__(
        self,
        action_scorer: ActionScorer,
        beam_width: int = 8,
        max_depth: int = 8,
        budget: int = 100_000,
    ):
        self._validate_integer("beam_width", beam_width, minimum=1)
        self._validate_integer("max_depth", max_depth, minimum=0)
        self._validate_integer("budget", budget, minimum=0)
        self.action_scorer = action_scorer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.budget = budget

    def cost_metadata(self) -> dict[str, object]:
        metadata = super().cost_metadata()
        metadata.update(
            {
                "beam_width": self.beam_width,
                "max_depth": self.max_depth,
            }
        )
        return metadata

    @staticmethod
    def _validate_integer(name: str, value: int, *, minimum: int) -> None:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def predict_functions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionDefList]:
        """Run an independent execution-budgeted rollout for each problem."""
        return [
            prediction.function_defs
            for prediction in self.predict_functions_batch_with_cost(
                problems,
                available_functions,
            )
        ]

    def predict_functions_batch_with_cost(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionPrediction]:
        """Run independent rollouts and preserve their exact costs and timings.

        The per-problem seconds include the action scorer's forward passes,
        so wall-clock is the one cost that charges policy inference.
        """
        predictions = []
        for input_list, output_list in problems:
            started = time.perf_counter()
            result = self.search_with_cost(
                input_list,
                output_list,
                available_functions,
            )
            predictions.append(
                FunctionPrediction(
                    result.function_defs,
                    PredictionCost(
                        result.executions,
                        None,
                        result.termination,
                        time.perf_counter() - started,
                    ),
                )
            )
        return predictions

    def search_with_cost(
        self,
        input_list: TypedList,
        output_list: TypedList,
        available_functions: FunctionDefSet,
        observer: RolloutObserver | None = None,
    ) -> RolloutSearchResult:
        """Search with a per-problem budget measured in basis executions.

        The scorer supplies logits for the whole basis. The rollout applies the
        exact input-type mask, normalizes each beam node separately, and orders
        all transitions in a layer by cumulative log probability. Every
        type-compatible transition attempted costs one, including functions
        that raise on their input values.

        An optional ``observer`` is told what each step decided. It is reported
        to, never consulted, so a search runs identically with and without one.
        """
        if input_list == output_list:
            if observer is not None:
                observer.search_terminated("solved", 0, 0)
            return RolloutSearchResult(FunctionDefList(), 0, "solved")
        if not available_functions.functions:
            if observer is not None:
                observer.search_terminated("frontier_exhausted", 0, 0)
            return RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
        if self.budget == 0:
            if observer is not None:
                observer.search_terminated("budget", 0, 0)
            return RolloutSearchResult(FunctionDefList(), 0, "budget")
        if self.max_depth == 0:
            if observer is not None:
                observer.search_terminated("max_depth", 0, 0)
            return RolloutSearchResult(FunctionDefList(), 0, "max_depth")

        executor = Executor(available_functions)
        grouped_functions: dict[type, list[tuple[int, FunctionDef]]] = {}
        for function_index, function in enumerate(available_functions.functions):
            grouped_functions.setdefault(function.input_type_cls(), []).append(
                (function_index, function)
            )
        compatible_by_input_type: dict[type, tuple[tuple[int, FunctionDef], ...]] = {
            input_type: tuple(compatible)
            for input_type, compatible in grouped_functions.items()
        }
        beam = [_RolloutNode(input_list, (), 0.0, 0)]
        executions = 0
        trace: list[RolloutDepthTrace] = []

        for depth in range(1, self.max_depth + 1):
            score_rows = self.action_scorer.score_actions_batch(
                [(node.state, output_list) for node in beam],
                available_functions,
            )
            scores = self._validate_scores(score_rows, len(beam), available_functions)
            if observer is not None:
                observer.layer_scored(depth, tuple(beam), tuple(scores))

            expansions = []
            for parent_index, (node, row) in enumerate(zip(beam, scores, strict=True)):
                compatible = compatible_by_input_type.get(node.state.item_type, ())
                if not compatible:
                    if observer is not None:
                        observer.node_masked(depth, parent_index, (), -math.inf)
                    continue
                top_score = max(row[function.name] for _, function in compatible)
                log_normalizer = top_score + math.log(
                    sum(
                        math.exp(row[function.name] - top_score)
                        for _, function in compatible
                    )
                )
                if observer is not None:
                    observer.node_masked(
                        depth,
                        parent_index,
                        tuple(function for _, function in compatible),
                        log_normalizer,
                    )
                for function_index, function in compatible:
                    log_probability = (
                        node.log_probability + row[function.name] - log_normalizer
                    )
                    expansions.append(
                        (
                            log_probability,
                            parent_index,
                            function_index,
                            node,
                            function,
                        )
                    )

            if not expansions:
                trace.append(RolloutDepthTrace(depth, (), 0))
                if observer is not None:
                    observer.search_terminated("frontier_exhausted", executions, depth)
                return RolloutSearchResult(
                    FunctionDefList(),
                    executions,
                    "frontier_exhausted",
                    tuple(trace),
                )

            # The ordering matters if a solution is encountered or the budget
            # binds partway through a layer. Basis order is the final stable
            # tie-break, after the already deterministic beam order.
            expansions.sort(key=lambda item: (-item[0], item[1], item[2]))
            successors: dict[tuple, _RolloutNode] = {}
            for order, (log_probability, parent_index, _, node, function) in enumerate(
                expansions
            ):
                executions += 1
                function_defs = (*node.function_defs, function)
                try:
                    state = executor.execute(function, node.state)
                except Exception as error:
                    if observer is not None:
                        observer.expansion_attempted(
                            depth,
                            order,
                            parent_index,
                            function,
                            log_probability,
                            "raised",
                            None,
                            error,
                        )
                    if executions >= self.budget:
                        if observer is not None:
                            observer.search_terminated("budget", executions, depth)
                        return RolloutSearchResult(
                            FunctionDefList(), executions, "budget", tuple(trace)
                        )
                    continue

                if state == output_list:
                    if observer is not None:
                        observer.expansion_attempted(
                            depth,
                            order,
                            parent_index,
                            function,
                            log_probability,
                            "solved",
                            state,
                            None,
                        )
                        observer.search_terminated("solved", executions, depth)
                    return RolloutSearchResult(
                        FunctionDefList(list(function_defs)),
                        executions,
                        "solved",
                        tuple(trace),
                    )
                if executions >= self.budget:
                    if observer is not None:
                        observer.expansion_attempted(
                            depth,
                            order,
                            parent_index,
                            function,
                            log_probability,
                            "budget",
                            state,
                            None,
                        )
                        observer.search_terminated("budget", executions, depth)
                    return RolloutSearchResult(
                        FunctionDefList(), executions, "budget", tuple(trace)
                    )

                state_key = state.search_key()
                previous = successors.get(state_key)
                if previous is None or log_probability > previous.log_probability:
                    successors[state_key] = _RolloutNode(
                        state, function_defs, log_probability, order
                    )
                    outcome = "retained"
                else:
                    outcome = "duplicate"
                if observer is not None:
                    observer.expansion_attempted(
                        depth,
                        order,
                        parent_index,
                        function,
                        log_probability,
                        outcome,
                        state,
                        None,
                    )

            beam = sorted(
                successors.values(),
                key=lambda node: (-node.log_probability, node.order),
            )[: self.beam_width]
            trace.append(
                RolloutDepthTrace(
                    depth=depth,
                    retained_paths=tuple(
                        tuple(function.name for function in node.function_defs)
                        for node in beam
                    ),
                    unique_successors=len(successors),
                )
            )
            if observer is not None:
                observer.layer_retained(depth, tuple(beam), dict(successors))
            if not beam:
                if observer is not None:
                    observer.search_terminated("frontier_exhausted", executions, depth)
                return RolloutSearchResult(
                    FunctionDefList(),
                    executions,
                    "frontier_exhausted",
                    tuple(trace),
                )

        if observer is not None:
            observer.search_terminated("max_depth", executions, self.max_depth)
        return RolloutSearchResult(
            FunctionDefList(), executions, "max_depth", tuple(trace)
        )

    @staticmethod
    def _validate_scores(
        score_rows: list[dict[str, float]],
        expected_rows: int,
        available_functions: FunctionDefSet,
    ) -> list[dict[str, float]]:
        if not isinstance(score_rows, list) or len(score_rows) != expected_rows:
            actual = (
                len(score_rows) if isinstance(score_rows, list) else type(score_rows)
            )
            raise ValueError(
                f"action scorer returned {actual} rows; expected {expected_rows}"
            )

        expected_names = {function.name for function in available_functions}
        validated = []
        for row_index, row in enumerate(score_rows):
            if not isinstance(row, dict) or set(row) != expected_names:
                actual_names = set(row) if isinstance(row, dict) else set()
                missing = sorted(expected_names - actual_names)
                unexpected = sorted(actual_names - expected_names)
                raise ValueError(
                    f"action scorer row {row_index} has wrong function names; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            clean_row = {}
            for name, score in row.items():
                if (
                    isinstance(score, bool)
                    or not isinstance(score, Real)
                    or not math.isfinite(float(score))
                ):
                    raise ValueError(
                        f"action scorer row {row_index} has non-finite or "
                        f"non-numeric score for {name!r}: {score!r}"
                    )
                clean_row[name] = float(score)
            validated.append(clean_row)
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
    def __init__(self, model: str, effort: str | None = None):
        """`effort` is passed only when set, since it is rejected by models
        that do not reason. Leaving it None keeps the original behaviour."""
        self.model = model
        self.effort = effort
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_io_history = []

    def generate(
        self,
        prompt: str,
    ) -> str:
        extra = {"reasoning": {"effort": self.effort}} if self.effort else {}
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            **extra,
        )
        self.llm_io_history.append((prompt, response.output_text))
        return response.output_text

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts. Uses sequential processing for API-based generation."""
        return super().generate_batch(prompts)


class AnthropicTokenGenerator(TokenGenerator):
    """Claude, through the official Anthropic SDK.

    Deliberately no `fallbacks` parameter. Server-side fallback would let a
    refusal be answered by a different model inside the same call, which is
    right for a product and wrong here: these generators exist to produce
    benchmark numbers, and the model is the variable being measured. A silent
    substitution would put two models' answers in one column. Refusals are
    reported through `stop_reason` instead, so a run can say so.
    """

    def __init__(self, model: str = "claude-opus-5", effort: str = "high"):
        self.model = model
        self.effort = effort
        load_dotenv()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.llm_io_history = []

    def generate(
        self,
        prompt: str,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            output = ""
        else:
            output = "".join(
                block.text for block in response.content if block.type == "text"
            )
        self.llm_io_history.append((prompt, output))
        return output

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts. Uses sequential processing for API-based generation."""
        return super().generate_batch(prompts)


class GeminiTokenGenerator(TokenGenerator):
    """Gemini over REST, so `effort` can set `thinkingLevel`.

    The pinned `google-generativeai` SDK predates thinking levels. `effort`
    (low / medium / high on gemini-3.8-flash and gemini-4-argon) goes straight
    to `thinkingConfig.thinkingLevel`; left unset the model's default applies.
    Gemini 2.x takes a token budget, not a level, and rejects one, so `effort`
    is dropped for those (the baseline CLI always passes one).
    `last_meta` records the served `modelVersion`, the usage (output tokens
    include thinking tokens, which are billed as output) and the finish
    reason. The key travels in a header so an error message never carries it.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
    RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        effort: str | None = None,
        timeout: float = 900.0,
        retries: int = 4,
    ):
        self.model = model.removeprefix("models/")
        self.effort = None if self.model.startswith("gemini-2") else effort
        self.timeout = timeout
        self.retries = retries
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.llm_io_history = []
        self.last_meta: dict | None = None

    def generate(
        self,
        prompt: str,
    ) -> str:
        import httpx

        self.last_meta = None
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if self.effort:
            body["generationConfig"] = {
                "thinkingConfig": {"thinkingLevel": self.effort}
            }
        started = time.time()
        # A 429, a 5xx or a dropped connection is the service, not the model:
        # retry it rather than score a task the model never saw as unsolved.
        # A deadline is not retried; it is the bound on a runaway generation.
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    self.URL.format(self.model),
                    headers={"x-goog-api-key": self.api_key or ""},
                    json=body,
                    timeout=self.timeout,
                )
            except httpx.TimeoutException:
                raise
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            else:
                if (
                    response.status_code not in self.RETRY_STATUS
                    or attempt == self.retries
                ):
                    break
            time.sleep(min(60, 5 * 2**attempt))
        response.raise_for_status()
        data = response.json()
        candidate = (data.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        output = "".join(
            part.get("text", "") for part in parts if not part.get("thought")
        )
        usage = data.get("usageMetadata", {})
        thinking = usage.get("thoughtsTokenCount", 0)
        self.last_meta = {
            "model": [data["modelVersion"]] if data.get("modelVersion") else [],
            "usage": {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0) + thinking,
                "thinking_tokens": thinking,
            },
            "stop_reason": candidate.get("finishReason"),
            "attempts": attempt + 1,
            "seconds": round(time.time() - started, 1),
        }
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
        timeout: float = 300.0,
        think: bool | str | None = None,
        num_ctx: int | None = None,
    ):
        """Bound a runaway generation by wall clock, not by token count.

        `think` is passed straight to ollama when set: a bool toggles a
        model's thinking, a level (`"low"`, `"medium"`, `"high"`) is the
        reasoning effort for models that grade it (gpt-oss). Left unset the
        model's own default applies, which is what every run before
        2026-09-03 used.

        `num_ctx` sets ollama's context window. The server default (32,768
        tokens on 0.33) is a second, unstated bound: gpt-oss:20b at effort
        high filled it on 23 of 34 v4 tasks and every truncated answer was
        empty, so the deadline was not the binding limit. Unset keeps the
        server default.

        Some local models answer in two seconds and then run for ten minutes on
        the next prompt, stalling a whole evaluation. Capping `num_predict` does
        stop that -- and also truncates a reasoning model mid-thought: measured
        2026-08-30, `qwen3:30b-a3b` scores 25% on 40 tasks unbounded, 22.5%
        under a 300 s deadline, and **5%** at a 4,096-token cap. A deadline
        aborts the pathological case without shortening a legitimate answer.
        """
        self.model = model
        self.timeout = timeout
        self.think = think
        self.num_ctx = num_ctx
        load_dotenv()
        self.client = Client(os.getenv("OLLAMA_BASE_URL"), timeout=timeout)
        self.llm_io_history = []

    def generate(self, prompt: str) -> str:
        extra = {} if self.think is None else {"think": self.think}
        if self.num_ctx is not None:
            extra["options"] = {"num_ctx": self.num_ctx}
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        output = response.message.content
        self.llm_io_history.append((prompt, output))
        return remove_thinking(output)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts. Uses sequential processing for API-based generation."""
        return super().generate_batch(prompts)


CODEX_EXEC_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode_host",
    "multi_agent",
    "plugins",
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "view_image",
    "sleep_tool",
    "tool_suggest",
    "skill_search",
    "goals",
    "hooks",
    "personality",
)

CODEX_EXEC_INSTRUCTIONS = "Follow the instructions in the message.\n"


class CodexExecTokenGenerator(TokenGenerator):
    """An OpenAI model reached through `codex exec` on a subscription.

    Trimmed to the API protocol, as measured in
    codex-exec-frontier-effort-curve (#123): a private `CODEX_HOME` holding
    only `auth.json` (no global AGENTS.md, skills or plugins), every tool
    feature disabled so the model cannot run code, the agent instructions
    replaced by one line, an empty working directory, `--ephemeral` so no
    session is written, and `--strict-config` so an unknown key errors instead
    of silently no-op-ing. What is left above the prompt is tool schema Codex
    sends regardless, which fails closed. The `--json` event stream is kept per
    call so `last_meta` can say whether the model attempted a tool call and
    what it cost in tokens.
    """

    def __init__(
        self,
        model: str,
        effort: str | None = None,
        timeout: float = 900.0,
        codex_home: str | os.PathLike | None = None,
        instructions: str = CODEX_EXEC_INSTRUCTIONS,
    ):
        import shutil
        import tempfile
        from pathlib import Path

        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.llm_io_history = []
        self.last_meta: dict | None = None
        home = (
            Path(codex_home)
            if codex_home
            else Path(tempfile.mkdtemp(prefix="codex-exec-"))
        )
        home.mkdir(parents=True, exist_ok=True)
        source = Path(os.getenv("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
        if not (home / "auth.json").exists():
            shutil.copy(source, home / "auth.json")
        (home / "instructions.md").write_text(instructions)
        (home / "workdir").mkdir(exist_ok=True)
        self.home = home
        # The private home is only private if the subprocess is told to use it.
        # Without this the child inherits the caller's CODEX_HOME (or the
        # default `~/.codex`) and Codex loads that home's `AGENTS.md`, config
        # and plugins on top of the prompt -- which is what every run in
        # reports/stack-pbe-v4-codex-exec-20260910 did before this line existed
        # (Codex review on PR #126). Separate from `command()` so a test can
        # assert on it without running anything.
        self.env = self.environment()

    def environment(self) -> dict:
        return {**os.environ, "CODEX_HOME": str(self.home)}

    def command(self) -> list[str]:
        argv = [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--strict-config",
            "--sandbox",
            "read-only",
            "-C",
            str(self.home / "workdir"),
            "-m",
            self.model,
            "-c",
            f"model_instructions_file={json_dumps(str(self.home / 'instructions.md'))}",
            "-c",
            "tools.web_search=false",
            "--json",
        ]
        if self.effort:
            argv += ["-c", f"model_reasoning_effort={json_dumps(self.effort)}"]
        for feature in CODEX_EXEC_DISABLED_FEATURES:
            argv += ["--disable", feature]
        argv.append("-")
        return argv

    @staticmethod
    def parse_events(lines: Sequence[str]) -> tuple[str, dict]:
        """The last agent message and what the run cost, from `--json` output.

        Tool attempts are counted from every non-message item the agent
        completed -- with the features off they error rather than run, and the
        count is the point: a row where it is nonzero is not the API protocol.
        """
        import json

        message = ""
        meta: dict = {"tool_calls": 0, "errors": []}
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item") or {}
            kind = item.get("type")
            if event.get("type") == "item.completed":
                if kind == "agent_message":
                    message = item.get("text", "")
                elif kind == "error":
                    meta["errors"].append(item.get("message", "")[:120])
                elif kind:
                    meta["tool_calls"] += 1
            usage = event.get("usage")
            if usage:
                meta["usage"] = usage
        return message, meta

    @staticmethod
    def kill_group(process) -> None:
        """SIGKILL the whole process group, not just the child we spawned.

        The npm `codex` is a Node launcher that spawns the native binary, so a
        kill aimed at the child reaches the launcher and leaves inference
        running past the deadline, letting requests pile up across tasks
        (Codex review on PR #126). Falls back to killing the child alone if
        the group is already gone.
        """
        import os
        import signal

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()

    def abandon(self, process, started: float, reason: str) -> None:
        """Kill the child's process group, reap it, and record why."""
        self.kill_group(process)
        process.communicate()
        self.last_meta = {
            "tool_calls": 0,
            "errors": [reason],
            "seconds": round(time.time() - started, 1),
        }

    def generate(self, prompt: str) -> str:
        import subprocess

        started = time.time()
        # Cleared before the call rather than written only after it. A caller
        # that catches the exception reads `last_meta` on the failure path too
        # -- `solve_with_llm` does -- and a failed call must not be handed the
        # previous task's token counts (Codex review on PR #126).
        self.last_meta = None
        # `start_new_session` puts the CLI in its own process group so a
        # timeout can take the whole group; see `kill_group`.
        process = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(prompt, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.abandon(process, started, f"timeout after {self.timeout:.0f}s")
            self.last_meta["timed_out"] = True
            raise
        except BaseException:
            # Ctrl-C or SIGTERM (the baseline turns it into KeyboardInterrupt
            # and writes partial results). The child sits in its own session
            # and would outlive the run, still spending tokens (Codex review
            # on PR #139).
            self.abandon(process, started, "interrupted")
            raise
        message, meta = self.parse_events(stdout.splitlines())
        meta["seconds"] = round(time.time() - started, 1)
        self.last_meta = meta
        if process.returncode != 0:
            raise RuntimeError(
                f"codex exec exited {process.returncode}: {stderr.strip()[-200:]}"
            )
        self.llm_io_history.append((prompt, message))
        return message


# Variables that would send `claude -p` somewhere other than the logged-in
# subscription: an API key or token bills the API, the provider switches route
# to Bedrock / Vertex / Foundry, a base URL to anything at all.
CLAUDE_EXEC_DROPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

CLAUDE_EXEC_SYSTEM_PROMPT = (
    "Follow the instructions in the message. You have no tools and cannot run "
    "commands or code; answer from the message alone.\n"
)


class ClaudeExecTokenGenerator(TokenGenerator):
    """An Anthropic model reached through `claude -p` on a Claude subscription.

    Trimmed to the API protocol the way `CodexExecTokenGenerator` is
    (opus-effort-curve, #137): every built-in tool off (`--tools ""`), no MCP
    servers, no user / project / local settings, no session written, an empty
    working directory so no `CLAUDE.md` is discovered, and Claude Code's system
    prompt replaced by two lines. `--bare` would strip more but disables OAuth,
    and the subscription is the point. `--output-format json` returns the
    answer with per-call usage (input, cache, output and thinking tokens), the
    canonical model that served the call, the turn count and permission
    denials; all of it lands in `last_meta`.

    With no tools the model can still *write* a tool call as text: the probe
    for #137 saw Opus at effort high answer a task with a Bash call to look for
    the interpreter. `parse_result` flags that as `text_tool_call`; the grader
    scores the answer as illegal, and the count is what lets a row say whether
    it stayed the API protocol.
    """

    def __init__(
        self,
        model: str,
        effort: str | None = None,
        timeout: float = 1800.0,
        workdir: str | os.PathLike | None = None,
        system_prompt: str = CLAUDE_EXEC_SYSTEM_PROMPT,
    ):
        import tempfile
        from pathlib import Path

        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.llm_io_history = []
        self.last_meta: dict | None = None
        self.workdir = (
            Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="claude-exec-"))
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = self.environment()

    def environment(self) -> dict:
        """The caller's environment minus anything that re-routes auth.

        `--setting-sources ""` drops settings files, not variables: an exported
        `ANTHROPIC_API_KEY` would make `claude -p` bill the API instead of the
        subscription, silently (Codex review on PR #139). OAuth lives in the
        home directory, which stays.
        """
        return {k: v for k, v in os.environ.items() if k not in CLAUDE_EXEC_DROPPED_ENV}

    def command(self) -> list[str]:
        argv = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--system-prompt",
            self.system_prompt,
        ]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv

    @staticmethod
    def looks_like_tool_call(text: str) -> bool:
        """A tool call the model wrote into its answer, having no tool to call.

        Claude Code's JSON shape (`{"name": ..., "input": ...}`) on the last
        line, or a function-call tag anywhere in the reply.
        """
        import json

        if "<function_calls>" in text or "<invoke " in text:
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or not lines[-1].startswith("{"):
            return False
        try:
            call = json.loads(lines[-1])
        except ValueError:
            return False
        return isinstance(call, dict) and {"name", "input"} <= set(call)

    @classmethod
    def parse_result(cls, stdout: str) -> tuple[str, dict]:
        """The answer and what the call cost, from `--output-format json`.

        Usage is renamed to the keys the codex rows use (`input_tokens` is
        every input token, cached or not; thinking tokens are
        `reasoning_output_tokens`) so one analysis reads both. `model` lists
        every model that served the call: more than one would mean a fallback
        answered, and that row is not one model's.
        """
        import json

        try:
            data = json.loads(stdout)
        except ValueError:
            return "", {
                "tool_calls": 0,
                "text_tool_call": False,
                "errors": [f"unparseable output: {stdout.strip()[:120]!r}"],
            }
        message = data.get("result") or ""
        usage = data.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        cached = usage.get("cache_creation_input_tokens", 0) + usage.get(
            "cache_read_input_tokens", 0
        )
        meta = {
            "tool_calls": max(0, int(data.get("num_turns") or 1) - 1)
            + len(data.get("permission_denials") or []),
            "text_tool_call": cls.looks_like_tool_call(message),
            "errors": [],
            "usage": {
                "input_tokens": usage.get("input_tokens", 0) + cached,
                "cache_creation_input_tokens": usage.get(
                    "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_output_tokens": details.get("thinking_tokens", 0),
            },
            "model": sorted(data.get("modelUsage") or {}),
            "stop_reason": data.get("stop_reason"),
            "api_ms": data.get("duration_api_ms"),
        }
        if data.get("is_error"):
            meta["errors"].append(str(message)[:120])
        return message, meta

    def abandon(self, process, started: float, reason: str) -> None:
        """Kill the child's process group, reap it, and record why."""
        CodexExecTokenGenerator.kill_group(process)
        process.communicate()
        self.last_meta = {
            "tool_calls": 0,
            "text_tool_call": False,
            "errors": [reason],
            "seconds": round(time.time() - started, 1),
        }

    def generate(self, prompt: str) -> str:
        import subprocess

        started = time.time()
        # Cleared before the call, as in `CodexExecTokenGenerator.generate`: a
        # failed call must not be handed the previous task's token counts.
        self.last_meta = None
        process = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.workdir,
            env=self.env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(prompt, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.abandon(process, started, f"timeout after {self.timeout:.0f}s")
            self.last_meta["timed_out"] = True
            raise
        except BaseException:
            # Same as the codex generator: an interrupted run must not leave
            # the model working in its own session (Codex review on PR #139).
            self.abandon(process, started, "interrupted")
            raise
        message, meta = self.parse_result(stdout)
        meta["seconds"] = round(time.time() - started, 1)
        self.last_meta = meta
        if process.returncode != 0 or meta["errors"]:
            raise RuntimeError(
                f"claude -p exited {process.returncode}: "
                f"{(meta['errors'] or [stderr.strip()[-200:]])[0]}"
            )
        self.llm_io_history.append((prompt, message))
        return message


def json_dumps(value: str) -> str:
    import json

    return json.dumps(value)


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

    cost_unit = BASIS_EXECUTION_COST_UNIT
    cost_definition = SAMPLED_PROGRAM_COST_DEFINITION
    budget_unit = "candidate_programs"

    def __init__(self, token_generator: TokenGenerator, budget: int = 1):
        """
        Initialize the TokenGenerator predictor.

        Args:
            token_generator: a generic LLM interface
            budget: Number of attempts to make before giving up
        """
        if budget < 0:
            raise ValueError("budget must be non-negative")
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
        """Return only programs for the legacy predictor API."""
        return [
            prediction.function_defs
            for prediction in self.predict_functions_batch_with_cost(
                problems,
                available_functions,
            )
        ]

    @staticmethod
    def _parse_candidate(
        response: str,
        available_functions: FunctionDefSet,
    ) -> FunctionDefList | None:
        """Strictly parse a comma-separated program; malformed text is not empty."""
        response = remove_thinking(response).strip()
        if not response:
            return None
        names = tuple(part.strip() for part in response.split(","))
        if any(not name for name in names):
            return None
        if any(name not in available_functions.name_to_function for name in names):
            return None
        return FunctionDefList(
            [available_functions.name_to_function[name] for name in names]
        )

    def predict_functions_batch_with_cost(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[FunctionPrediction]:
        """Return the shortest solution and charge every valid sampled program.

        Generation is batched across problems, so per-problem ``seconds`` stays
        ``None``; the evaluation layer times the whole batch instead.
        """
        if not problems:
            return []

        predictions: list[FunctionPrediction | None] = [None] * len(problems)
        active_indices = []
        for problem_index, (input_list, output_list) in enumerate(problems):
            if input_list == output_list:
                predictions[problem_index] = FunctionPrediction(
                    FunctionDefList(),
                    PredictionCost(0, 0, "solved"),
                )
            else:
                active_indices.append(problem_index)

        if not active_indices:
            assert all(prediction is not None for prediction in predictions)
            return [prediction for prediction in predictions if prediction is not None]

        if self.budget == 0:
            for problem_index in active_indices:
                predictions[problem_index] = FunctionPrediction(
                    FunctionDefList(),
                    PredictionCost(0, 0, "candidate_budget"),
                )
            assert all(prediction is not None for prediction in predictions)
            return [prediction for prediction in predictions if prediction is not None]

        prompts = [
            self._generate_prompt(
                problems[problem_index][0],
                problems[problem_index][1],
                available_functions,
            )
            for problem_index in active_indices
        ]
        responses = self.token_generator.generate_batch(prompts * self.budget)
        expected = len(prompts) * self.budget
        if len(responses) != expected:
            raise ValueError(
                f"token generator returned {len(responses)} responses; expected {expected}"
            )

        executor = Executor(available_functions)
        best_predictions = [FunctionDefList() for _ in problems]
        best_lengths = [float("inf") for _ in problems]
        charged_executions = [0 for _ in problems]
        candidates = [0 for _ in problems]
        for response_index, response in enumerate(responses):
            active_index = response_index % len(active_indices)
            problem_index = active_indices[active_index]
            input_list, output_list = problems[problem_index]
            candidates[problem_index] += 1
            candidate = self._parse_candidate(response, available_functions)
            if candidate is None:
                continue
            charged_executions[problem_index] += len(candidate)
            execution = executor.execute_trajectory(
                TrajectorySpec(input_list, candidate)
            )
            if (
                execution.success
                and execution.trajectory.output == output_list
                and len(candidate) < best_lengths[problem_index]
            ):
                best_predictions[problem_index] = candidate
                best_lengths[problem_index] = len(candidate)

        for problem_index in active_indices:
            solved = best_lengths[problem_index] != float("inf")
            predictions[problem_index] = FunctionPrediction(
                best_predictions[problem_index],
                PredictionCost(
                    charged_executions[problem_index],
                    candidates[problem_index],
                    "solved" if solved else "candidate_budget",
                ),
            )

        assert all(prediction is not None for prediction in predictions)
        return [prediction for prediction in predictions if prediction is not None]

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
    path_lengths: Sequence[int] | None = None,
    seed: int | None = None,
) -> TrajectorySolver:
    """Create a solver with random function prediction."""
    predictor = RandomPredictor(
        budget=budget,
        path_length=path_length,
        path_lengths=path_lengths,
        seed=seed,
    )
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
    path_lengths: Sequence[int] | None = None,
    max_depth: int | None = None,
    seed: int | None = None,
) -> TrajectorySolver:
    if name == "random":
        return create_random_solver(
            budget=budget,
            path_length=path_length,
            usage_tracker=usage_tracker,
            track_function_usage=track_function_usage,
            path_lengths=path_lengths,
            seed=seed,
        )
    elif name == "bfs":
        return create_bfs_solver(
            budget=budget,
            max_depth=path_length if max_depth is None else max_depth,
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
