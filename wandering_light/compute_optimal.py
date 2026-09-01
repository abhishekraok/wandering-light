"""Metrics and exact state labels for compute-optimal rollout evaluation.

The functions here are deliberately independent of report and CLI code.  They
turn first-hit costs into the metric defined in :mod:`docs.metrics`, score a
policy distribution against a certified optimal-action set, record exactly the
states a production rollout scores, and certify those sets with bounded BFS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.solver import RolloutObserver

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from wandering_light.typed_list import TypedList

BITS_EPSILON = 0.01


@dataclass(frozen=True)
class SolveRatePoint:
    """One point of an empirical first-hit solve-rate curve."""

    budget: float
    solved: int
    tasks: int
    solve_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "budget": self.budget,
            "solved": self.solved,
            "tasks": self.tasks,
            "solve_rate": self.solve_rate,
        }


@dataclass(frozen=True)
class LogBudgetAUC:
    """Exact area under an empirical solve-rate versus log2-budget curve."""

    unit: str
    min_budget: float
    max_budget: float
    log2_budget_span: float
    tasks: int
    finite_hits: int
    solved_by_max: int
    area: float
    normalized_area: float
    curve: tuple[SolveRatePoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "log2_budget_span": self.log2_budget_span,
            "tasks": self.tasks,
            "finite_hits": self.finite_hits,
            "solved_by_max": self.solved_by_max,
            "area": self.area,
            "normalized_area": self.normalized_area,
            "integration": "empirical-first-passage-step",
            "log_base": 2,
            "curve": [point.to_dict() for point in self.curve],
        }


def _positive_finite_budget(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _hit_budget(value: Real | None, *, index: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"hit budget {index} must be a non-negative number or None")
    result = float(value)
    if math.isnan(result) or result == -math.inf or result < 0:
        raise ValueError(
            f"hit budget {index} must be non-negative or positive infinity"
        )
    return None if result == math.inf else result


def log_budget_auc(
    hit_budgets: Sequence[Real | None],
    *,
    min_budget: Real,
    max_budget: Real,
    unit: str,
) -> LogBudgetAUC:
    """Integrate the empirical first-hit CDF exactly on a fixed log2 range.

    ``hit_budgets`` has one entry per cohort task.  A miss is ``None`` (positive
    infinity is accepted as an equivalent input).  A hit below the lower bound
    earns the full area and a hit at the upper bound earns zero area, while it
    still appears in the final solve rate.
    """

    if not isinstance(unit, str) or not unit:
        raise ValueError("unit must be a non-empty string")
    lower = _positive_finite_budget(min_budget, name="min_budget")
    upper = _positive_finite_budget(max_budget, name="max_budget")
    if upper <= lower:
        raise ValueError("max_budget must be greater than min_budget")
    if not hit_budgets:
        raise ValueError("hit_budgets must contain one entry per task")

    hits = tuple(
        _hit_budget(value, index=index) for index, value in enumerate(hit_budgets)
    )
    span = math.log2(upper / lower)
    contributions = [
        max(0.0, math.log2(upper / max(lower, hit)))
        for hit in hits
        if hit is not None and hit < upper
    ]
    area = math.fsum(contributions) / len(hits)

    curve_budgets = sorted(
        {
            lower,
            upper,
            *(hit for hit in hits if hit is not None and lower < hit < upper),
        }
    )
    curve = tuple(
        SolveRatePoint(
            budget=budget,
            solved=sum(hit is not None and hit <= budget for hit in hits),
            tasks=len(hits),
            solve_rate=sum(hit is not None and hit <= budget for hit in hits)
            / len(hits),
        )
        for budget in curve_budgets
    )
    return LogBudgetAUC(
        unit=unit,
        min_budget=lower,
        max_budget=upper,
        log2_budget_span=span,
        tasks=len(hits),
        finite_hits=sum(hit is not None for hit in hits),
        solved_by_max=sum(hit is not None and hit <= upper for hit in hits),
        area=area,
        normalized_area=area / span,
        curve=curve,
    )


def wall_clock_auc(
    hit_seconds: Sequence[Real | None],
    *,
    min_seconds: Real,
    max_seconds: Real,
) -> LogBudgetAUC:
    """Compute the primary AUC in wall-clock seconds."""

    return log_budget_auc(
        hit_seconds,
        min_budget=min_seconds,
        max_budget=max_seconds,
        unit="wall_clock_seconds",
    )


def execution_auc(
    hit_executions: Sequence[int | float | None],
    *,
    min_executions: int,
    max_executions: int,
) -> LogBudgetAUC:
    """Compute the hardware-independent diagnostic AUC in executions."""

    if type(min_executions) is not int or type(max_executions) is not int:
        raise ValueError("execution bounds must be integers")
    for index, value in enumerate(hit_executions):
        if value is not None and value != math.inf and type(value) is not int:
            raise ValueError(f"hit execution {index} must be an integer or None")
    return log_budget_auc(
        hit_executions,
        min_budget=min_executions,
        max_budget=max_executions,
        unit="basis_function_executions",
    )


@dataclass(frozen=True)
class MaskedOptimalBits:
    """Optimal-set probability and code length under one masked policy row."""

    epsilon: float
    mask_width: int
    optimal_count: int
    log_optimal_mass: float
    optimal_mass: float
    raw_bits: float
    smoothed_optimal_mass: float
    bits: float
    uniform_bits: float
    bits_saved: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "epsilon": self.epsilon,
            "mask_width": self.mask_width,
            "optimal_count": self.optimal_count,
            "log_optimal_mass": self.log_optimal_mass,
            "optimal_mass": self.optimal_mass,
            "raw_bits": self.raw_bits,
            "smoothed_optimal_mass": self.smoothed_optimal_mass,
            "bits": self.bits,
            "uniform_bits": self.uniform_bits,
            "bits_saved": self.bits_saved,
        }


def _unique_action_names(actions: Iterable[str], *, label: str) -> tuple[str, ...]:
    names = tuple(actions)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain non-empty action names")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate action names")
    return names


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    peak = max(left, right)
    return peak + math.log(math.exp(left - peak) + math.exp(right - peak))


def masked_optimal_set_bits(
    logits: Mapping[str, Real],
    compatible_actions: Iterable[str],
    optimal_actions: Iterable[str],
    *,
    epsilon: float = BITS_EPSILON,
) -> MaskedOptimalBits:
    """Score total masked probability mass on every certified optimal action."""

    compatible = _unique_action_names(compatible_actions, label="compatible_actions")
    optimal = _unique_action_names(optimal_actions, label="optimal_actions")
    unknown = sorted(set(optimal) - set(compatible))
    if unknown:
        raise ValueError(
            f"optimal actions are outside the compatibility mask: {unknown}"
        )
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be in [0, 1]")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")

    scores: dict[str, float] = {}
    for name in compatible:
        value = logits.get(name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"logit for compatible action {name!r} is missing or invalid"
            )
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"logit for compatible action {name!r} is not finite")
        scores[name] = score

    log_denominator = _logsumexp([scores[name] for name in compatible])
    log_numerator = _logsumexp([scores[name] for name in optimal])
    log_mass = min(0.0, log_numerator - log_denominator)
    raw_bits = max(0.0, -log_mass / math.log(2))
    uniform_mass = len(optimal) / len(compatible)
    uniform_bits = -math.log2(uniform_mass)

    policy_term = -math.inf if epsilon == 1 else math.log1p(-epsilon) + log_mass
    uniform_term = (
        -math.inf if epsilon == 0 else math.log(epsilon) + math.log(uniform_mass)
    )
    log_smoothed = _logaddexp(policy_term, uniform_term)
    bits = max(0.0, -log_smoothed / math.log(2))
    return MaskedOptimalBits(
        epsilon=epsilon,
        mask_width=len(compatible),
        optimal_count=len(optimal),
        log_optimal_mass=log_mass,
        optimal_mass=math.exp(log_mass),
        raw_bits=raw_bits,
        smoothed_optimal_mass=math.exp(log_smoothed),
        bits=bits,
        uniform_bits=uniform_bits,
        bits_saved=uniform_bits - bits,
    )


@dataclass(frozen=True)
class ScoredVisit:
    """One full beam-node input and score row observed by production rollout."""

    depth: int
    parent_index: int
    state_serialized: str
    path: tuple[str, ...]
    path_log_probability: float
    scores: dict[str, float]
    compatible_action_names: tuple[str, ...]
    log_normalizer: float

    @property
    def logits(self) -> dict[str, float]:
        """Alias for policy evaluators that call the raw scores logits."""

        return self.scores

    @property
    def compatible_actions(self) -> tuple[str, ...]:
        return self.compatible_action_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "parent_index": self.parent_index,
            "state": self.state_serialized,
            "path": list(self.path),
            "path_log_probability": self.path_log_probability,
            "scores": dict(self.scores),
            "compatible_actions": list(self.compatible_action_names),
            "log_normalizer": (
                self.log_normalizer if math.isfinite(self.log_normalizer) else None
            ),
        }


@dataclass(frozen=True)
class _PendingVisit:
    state_serialized: str
    path: tuple[str, ...]
    path_log_probability: float
    scores: dict[str, float]


class ScoredVisitObserver(RolloutObserver):
    """Capture scored states through observer hooks without steering the search."""

    def __init__(self) -> None:
        self._pending: dict[tuple[int, int], _PendingVisit] = {}
        self._visits: list[ScoredVisit] = []
        self.termination: str | None = None
        self.executions = 0
        self.terminal_depth = 0

    @property
    def visits(self) -> tuple[ScoredVisit, ...]:
        return tuple(self._visits)

    def layer_scored(
        self,
        depth: int,
        nodes: Sequence[Any],
        score_rows: Sequence[dict[str, float]],
    ) -> None:
        if len(nodes) != len(score_rows):
            raise ValueError("observer received different node and score-row counts")
        for parent_index, (node, scores) in enumerate(
            zip(nodes, score_rows, strict=True)
        ):
            key = (depth, parent_index)
            if key in self._pending:
                raise ValueError(f"duplicate scored rollout visit {key}")
            self._pending[key] = _PendingVisit(
                state_serialized=node.state.to_string(),
                path=tuple(function.name for function in node.function_defs),
                path_log_probability=float(node.log_probability),
                scores={name: float(value) for name, value in scores.items()},
            )

    def node_masked(
        self,
        depth: int,
        parent_index: int,
        compatible: Sequence[FunctionDef],
        log_normalizer: float,
    ) -> None:
        key = (depth, parent_index)
        try:
            pending = self._pending.pop(key)
        except KeyError as error:
            raise ValueError(
                f"mask arrived without a scored rollout visit {key}"
            ) from error
        self._visits.append(
            ScoredVisit(
                depth=depth,
                parent_index=parent_index,
                state_serialized=pending.state_serialized,
                path=pending.path,
                path_log_probability=pending.path_log_probability,
                scores=pending.scores,
                compatible_action_names=tuple(function.name for function in compatible),
                log_normalizer=float(log_normalizer),
            )
        )

    def search_terminated(self, termination: str, executions: int, depth: int) -> None:
        self.termination = termination
        self.executions = executions
        self.terminal_depth = depth


OracleStatus = Literal["certified", "already_solved", "unreachable", "unresolved"]
OracleStopReason = Literal["max_depth", "max_states", "max_executions"]


@dataclass(frozen=True)
class OracleLimits:
    """Hard bounds for one exact optimal-first-action certification."""

    max_depth: int
    max_states: int | None = None
    max_executions: int | None = None

    def __post_init__(self) -> None:
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be an integer >= 0")
        if self.max_states is not None and (
            type(self.max_states) is not int or self.max_states < 1
        ):
            raise ValueError("max_states must be an integer >= 1")
        if self.max_executions is not None and (
            type(self.max_executions) is not int or self.max_executions < 0
        ):
            raise ValueError("max_executions must be an integer >= 0")


@dataclass(frozen=True)
class OptimalActionOracleResult:
    """Certified optimal first actions, or an explicit reason they are unknown."""

    status: OracleStatus
    stop_reason: OracleStopReason | None
    distance: int | None
    observed_target_distance: int | None
    optimal_action_names: tuple[str, ...]
    optimal_actions_complete: bool
    certified_depth: int
    reached_states: int
    attempted_executions: int
    failed_executions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "distance": self.distance,
            "observed_target_distance": self.observed_target_distance,
            "optimal_action_names": list(self.optimal_action_names),
            "optimal_actions_complete": self.optimal_actions_complete,
            "certified_depth": self.certified_depth,
            "reached_states": self.reached_states,
            "attempted_executions": self.attempted_executions,
            "failed_executions": self.failed_executions,
        }


@dataclass
class _OracleNode:
    state: TypedList
    first_action_names: set[str]


def _oracle_functions(functions: FunctionDefSet) -> FunctionDefSet:
    """Clone the executable palette so oracle accounting cannot mutate callers."""

    return FunctionDefSet(
        [
            FunctionDef(
                name=function.name,
                input_type=function.input_type,
                output_type=function.output_type,
                code=function.code,
                metadata=dict(function.metadata or {}),
            )
            for function in functions
        ]
    )


def certify_shortest_first_actions(
    source: TypedList,
    target: TypedList,
    functions: FunctionDefSet,
    *,
    limits: OracleLimits,
) -> OptimalActionOracleResult:
    """Run bounded BFS and return every shortest first action only when complete.

    Search states use :meth:`TypedList.search_key`, exactly as production search
    does.  Goal checks use ``TypedList.__eq__`` and therefore canonical answer
    equality.  A target found partway through a budget-truncated layer is
    reported as observed but unresolved: the missing suffix could contain more
    equally short first actions.
    """

    if not isinstance(limits, OracleLimits):
        raise ValueError("limits must be an OracleLimits instance")
    if source == target:
        return OptimalActionOracleResult(
            status="already_solved",
            stop_reason=None,
            distance=0,
            observed_target_distance=0,
            optimal_action_names=(),
            optimal_actions_complete=True,
            certified_depth=0,
            reached_states=1,
            attempted_executions=0,
            failed_executions=0,
        )

    palette = _oracle_functions(functions)
    executor = Executor(palette)
    action_order = {
        function.name: index for index, function in enumerate(palette.functions)
    }
    root_key = source.search_key()
    seen_depth = {root_key: 0}
    frontier = {root_key: _OracleNode(source, set())}
    attempted = 0
    failed = 0
    certified_depth = 0

    def result(
        status: OracleStatus,
        *,
        stop_reason: OracleStopReason | None = None,
        distance: int | None = None,
        observed: int | None = None,
        actions: Iterable[str] = (),
        complete: bool = False,
    ) -> OptimalActionOracleResult:
        ordered_actions = tuple(sorted(set(actions), key=action_order.__getitem__))
        return OptimalActionOracleResult(
            status=status,
            stop_reason=stop_reason,
            distance=distance,
            observed_target_distance=observed,
            optimal_action_names=ordered_actions,
            optimal_actions_complete=complete,
            certified_depth=certified_depth,
            reached_states=len(seen_depth),
            attempted_executions=attempted,
            failed_executions=failed,
        )

    for parent_depth in range(limits.max_depth):
        child_depth = parent_depth + 1
        next_frontier: dict[tuple, _OracleNode] = {}
        target_first_actions: set[str] = set()
        target_observed = False

        for node in frontier.values():
            compatible = [
                function
                for function in palette
                if node.state.item_type is function.input_type_cls()
            ]
            for function in compatible:
                if (
                    limits.max_executions is not None
                    and attempted >= limits.max_executions
                ):
                    return result(
                        "unresolved",
                        stop_reason="max_executions",
                        observed=child_depth if target_observed else None,
                    )
                attempted += 1
                try:
                    child = executor.execute(function, node.state)
                except Exception:
                    failed += 1
                    continue

                first_actions = (
                    {function.name}
                    if parent_depth == 0
                    else set(node.first_action_names)
                )
                if child == target:
                    target_observed = True
                    target_first_actions.update(first_actions)

                child_key = child.search_key()
                earlier_depth = seen_depth.get(child_key)
                if earlier_depth is not None and earlier_depth < child_depth:
                    continue
                existing = next_frontier.get(child_key)
                if existing is not None:
                    existing.first_action_names.update(first_actions)
                    continue
                if earlier_depth == child_depth:
                    raise RuntimeError(
                        "same-depth oracle state is missing from frontier"
                    )
                if (
                    limits.max_states is not None
                    and len(seen_depth) >= limits.max_states
                ):
                    return result(
                        "unresolved",
                        stop_reason="max_states",
                        observed=child_depth if target_observed else None,
                    )
                seen_depth[child_key] = child_depth
                next_frontier[child_key] = _OracleNode(child, first_actions)

        certified_depth = child_depth
        if target_observed:
            return result(
                "certified",
                distance=child_depth,
                observed=child_depth,
                actions=target_first_actions,
                complete=True,
            )
        if not next_frontier:
            return result("unreachable")
        frontier = next_frontier

    return result("unresolved", stop_reason="max_depth")
