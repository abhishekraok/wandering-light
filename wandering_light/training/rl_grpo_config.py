"""Configuration and protocol definitions for RL GRPO training."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from typing_extensions import runtime_checkable

# Type aliases
type RewardFunction = Callable[[list[str]], list[float]]

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RLGRPOConfig:
    """Configuration for RL GRPO training for a quick run."""

    DEFAULT_EVAL_STEPS: int = 16
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_LEARNING_RATE: float = 1e-6
    DEFAULT_MAX_LENGTH: int = 256
    DEFAULT_NUM_EPOCHS: int = 3
    DEFAULT_LENGTH_PENALTY: float = 0.1
    DEFAULT_SOLVER_ATTEMPTS: int = 8
    INFERENCE_BATCH_SIZE: int = 64
    TRAINING_INTERVAL_STEPS: int = 8


@runtime_checkable
class InductionMetricsObserver(Protocol):
    """Protocol for induction task metrics observers."""

    def on_batch_processed(
        self,
        success_rate: float,
        avg_function_count: float,
        function_counts: list[int],
        correctness_scores: list[float],
        avg_function_count_ratio: float,
        function_count_ratios: list[float],
    ) -> None:
        """Called when a batch is processed for induction task."""
        ...


@runtime_checkable
class ProposerMetricsObserver(Protocol):
    """Protocol for proposer task metrics observers."""

    def on_proposer_batch_processed(
        self,
        parse_rate: float,
        solver_success_rate: float,
        frac_non_zero_std: float,
        rewards: list[float],
        avg_function_count: float,
    ) -> None:
        """Called when a batch is processed for proposer task."""
        ...


@dataclass
class BatchMetrics:
    """Container for batch-level metrics accumulation."""

    # Induction metrics
    success_rates: list[float] = field(default_factory=list)
    function_counts: list[float] = field(default_factory=list)
    function_count_ratios: list[float] = field(default_factory=list)

    # Proposer metrics
    parse_rates: list[float] = field(default_factory=list)
    solver_success_rates: list[float] = field(default_factory=list)
    frac_non_zero_stds: list[float] = field(default_factory=list)

    def add_induction_batch(
        self,
        success_rate: float,
        avg_function_count: float,
        avg_function_count_ratio: float,
    ):
        """Add metrics from an induction batch."""
        self.success_rates.append(success_rate)
        self.function_counts.append(avg_function_count)
        self.function_count_ratios.append(avg_function_count_ratio)

    def add_proposer_batch(
        self, parse_rate: float, solver_success_rate: float, frac_non_zero_std: float
    ):
        """Add metrics from a proposer batch."""
        self.parse_rates.append(parse_rate)
        self.solver_success_rates.append(solver_success_rate)
        self.frac_non_zero_stds.append(frac_non_zero_std)

    def clear_induction_metrics(self):
        """Clear induction-specific metrics."""
        self.success_rates.clear()
        self.function_counts.clear()
        self.function_count_ratios.clear()

    def clear_proposer_metrics(self):
        """Clear proposer-specific metrics."""
        self.parse_rates.clear()
        self.solver_success_rates.clear()
        self.frac_non_zero_stds.clear()

    def calculate_average(self, values: list[float]) -> float:
        """Calculate average of a list of values."""
        return sum(values) / len(values) if values else 0.0


@dataclass
class RLMetrics:
    """Track RL training metrics."""

    step: int
    avg_reward: float
    kl_divergence: float
    policy_loss: float
    frac_reward_zero_std: float

    # Task-specific metrics (will be None for the other task)
    # Induction metrics
    success_rate: float | None = None
    avg_function_count: float | None = None
    avg_function_count_ratio: float | None = None

    # Proposer metrics
    parse_rate: float | None = None
    solver_success_rate: float | None = None
    frac_non_zero_std: float | None = None
