import math

import pytest

from wandering_light.compute_optimal import (
    OracleLimits,
    PreparedOptimalActionOracle,
    ScoredVisitObserver,
    certify_shortest_first_actions,
    execution_auc,
    masked_optimal_set_bits,
    wall_clock_auc,
)
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.solver import ActionScorer, RolloutPredictor
from wandering_light.typed_list import TypedList


def function(
    name: str,
    code: str,
    *,
    input_type: str = "builtins.int",
    output_type: str = "builtins.int",
) -> FunctionDef:
    return FunctionDef(
        name=name,
        input_type=input_type,
        output_type=output_type,
        code=code,
    )


def test_log_budget_auc_integrates_the_empirical_step_curve_exactly():
    # On log2 [1, 16], the four contributions are 4, 2, 0 and 0 bits.
    result = wall_clock_auc(
        [0.5, 4.0, 16.0, None],
        min_seconds=1.0,
        max_seconds=16.0,
    )

    assert result.area == pytest.approx(1.5)
    assert result.normalized_area == pytest.approx(0.375)
    assert result.tasks == 4
    assert result.solved_by_max == 3
    assert [(point.budget, point.solve_rate) for point in result.curve] == [
        (1.0, 0.25),
        (4.0, 0.5),
        (16.0, 0.75),
    ]
    # A trapezoid over those display points would be a different metric.
    trapezoid = (0.25 + 0.5) / 2 * 2 + (0.5 + 0.75) / 2 * 2
    assert trapezoid != pytest.approx(result.area)


def test_log_budget_auc_handles_misses_boundaries_and_common_rescaling():
    misses = wall_clock_auc([None, math.inf], min_seconds=0.01, max_seconds=10.0)
    early = wall_clock_auc([0.0, 0.001], min_seconds=0.01, max_seconds=10.0)
    original = wall_clock_auc([0.2, 3.0, None], min_seconds=0.1, max_seconds=10.0)
    scaled = wall_clock_auc([2.0, 30.0, None], min_seconds=1.0, max_seconds=100.0)

    assert (misses.area, misses.normalized_area) == (0.0, 0.0)
    assert early.area == pytest.approx(early.log2_budget_span)
    assert early.normalized_area == pytest.approx(1.0)
    assert scaled.area == pytest.approx(original.area)
    assert scaled.normalized_area == pytest.approx(original.normalized_area)


def test_execution_auc_uses_the_same_metric_and_requires_integral_costs():
    result = execution_auc([1, 4, 16, None], min_executions=1, max_executions=16)
    assert result.unit == "basis_function_executions"
    assert result.area == pytest.approx(1.5)

    with pytest.raises(ValueError, match="integer"):
        execution_auc([1, 2.5], min_executions=1, max_executions=16)


@pytest.mark.parametrize(
    ("hits", "lower", "upper", "message"),
    [
        ([], 1.0, 2.0, "one entry per task"),
        ([1.0], 0.0, 2.0, "positive finite"),
        ([1.0], 2.0, 2.0, "greater than"),
        ([float("nan")], 1.0, 2.0, "non-negative"),
        ([-1.0], 1.0, 2.0, "non-negative"),
    ],
)
def test_log_budget_auc_rejects_undefined_inputs(hits, lower, upper, message):
    with pytest.raises(ValueError, match=message):
        wall_clock_auc(hits, min_seconds=lower, max_seconds=upper)


def test_masked_bits_sum_the_optimal_set_and_ignore_incompatible_logits():
    logits = {
        "a": math.log(0.5),
        "b": math.log(0.25),
        "c": math.log(0.25),
        "incompatible": 1_000.0,
    }
    result = masked_optimal_set_bits(logits, ["a", "b", "c"], ["a", "b"])
    expected_smoothed_mass = 0.99 * 0.75 + 0.01 * (2 / 3)

    assert result.optimal_mass == pytest.approx(0.75)
    assert result.raw_bits == pytest.approx(-math.log2(0.75))
    assert result.smoothed_optimal_mass == pytest.approx(expected_smoothed_mass)
    assert result.bits == pytest.approx(-math.log2(expected_smoothed_mass))
    assert result.uniform_bits == pytest.approx(math.log2(3 / 2))
    assert result.bits_saved == pytest.approx(result.uniform_bits - result.bits)


def test_masked_bits_are_stable_for_extreme_logits_and_anchor_uniform():
    extreme = masked_optimal_set_bits(
        {"likely": 1_000.0, "optimal": -1_000.0},
        ["likely", "optimal"],
        ["optimal"],
    )
    uniform = masked_optimal_set_bits(
        dict.fromkeys(["a", "b", "c", "d"], 0.0),
        ["a", "b", "c", "d"],
        ["a", "b"],
    )

    assert math.isfinite(extreme.raw_bits)
    assert extreme.raw_bits > 2_000
    assert extreme.bits == pytest.approx(-math.log2(0.01 / 2))
    assert uniform.bits == pytest.approx(1.0)
    assert uniform.uniform_bits == pytest.approx(1.0)
    assert uniform.bits_saved == pytest.approx(0.0)


def test_masked_bits_require_a_complete_nonempty_subset_of_the_mask():
    with pytest.raises(ValueError, match="outside"):
        masked_optimal_set_bits({"a": 0.0, "b": 0.0}, ["a"], ["b"])
    with pytest.raises(ValueError, match="non-empty"):
        masked_optimal_set_bits({"a": 0.0}, ["a"], [])
    with pytest.raises(ValueError, match="missing or invalid"):
        masked_optimal_set_bits({"a": 0.0}, ["a", "b"], ["a"])


class StaticScorer(ActionScorer):
    def score_actions_batch(self, problems, available_functions):
        return [
            {
                function.name: float(index)
                for index, function in enumerate(available_functions)
            }
            for _ in problems
        ]


def test_scored_visit_observer_is_inert_and_keeps_full_unrounded_rows():
    functions = FunctionDefSet(
        [
            function("inc", "return x + 1"),
            function("double", "return x * 2"),
        ]
    )
    predictor = RolloutPredictor(StaticScorer(), beam_width=2, max_depth=2, budget=100)
    source = TypedList([1])
    target = TypedList([3])

    plain = predictor.search_with_cost(source, target, functions)
    observer = ScoredVisitObserver()
    observed = predictor.search_with_cost(source, target, functions, observer=observer)

    assert observed == plain
    assert observer.termination == observed.termination
    assert observer.executions == observed.executions
    assert len(observer.visits) == 2
    first = observer.visits[0]
    assert TypedList.from_str(first.state_serialized) == source
    assert first.path == ()
    assert first.scores == {"inc": 0.0, "double": 1.0}
    assert first.logits == first.scores
    assert first.compatible_action_names == ("inc", "double")
    expected_normalizer = 1.0 + math.log1p(math.exp(-1.0))
    assert first.log_normalizer == pytest.approx(expected_normalizer)


def test_oracle_sums_alternate_first_actions_after_state_deduplication():
    functions = FunctionDefSet(
        [
            function(
                "route_a",
                "return float(x)",
                output_type="builtins.float",
            ),
            function(
                "route_b",
                "return float(x)",
                output_type="builtins.float",
            ),
            function(
                "finish",
                "return str(int(x + 1))",
                input_type="builtins.float",
                output_type="builtins.str",
            ),
        ]
    )
    result = certify_shortest_first_actions(
        TypedList([0]),
        TypedList(["1"]),
        functions,
        limits=OracleLimits(max_depth=2),
    )

    assert result.status == "certified"
    assert result.distance == 2
    assert result.optimal_actions_complete is True
    assert result.optimal_action_names == ("route_a", "route_b")


def test_prepared_oracle_matches_one_shot_and_reuses_compiled_palette(monkeypatch):
    functions = FunctionDefSet(
        [
            function("inc", "return x + 1"),
            function("double", "return x * 2"),
        ]
    )
    source = TypedList([1])
    target = TypedList([3])
    limits = OracleLimits(max_depth=2)
    expected = certify_shortest_first_actions(
        source,
        target,
        functions,
        limits=limits,
    )
    compile_misses = []
    compile_function = Executor._compile

    def count_compile_misses(executor, function_def):
        cache_key = (
            function_def.name,
            function_def.input_type,
            function_def.output_type,
            function_def.code,
        )
        if cache_key not in executor._compiled:
            compile_misses.append(cache_key)
        return compile_function(executor, function_def)

    monkeypatch.setattr(Executor, "_compile", count_compile_misses)
    oracle = PreparedOptimalActionOracle(functions)
    first = oracle.certify(source, target, limits=limits)
    misses_after_first = list(compile_misses)
    second = oracle(source, target, functions, limits=limits)

    assert first == second == expected
    assert compile_misses == misses_after_first
    assert compile_misses
    assert all(function_def.usage_count == 0 for function_def in functions)


def test_prepared_oracle_resolves_palette_types_once_not_per_node(monkeypatch):
    functions = FunctionDefSet(
        [
            function("inc", "return x + 1"),
            function("double", "return x * 2"),
        ]
    )
    input_type_calls = 0
    resolve_input_type = FunctionDef.input_type_cls

    def count_input_type_calls(function_def):
        nonlocal input_type_calls
        input_type_calls += 1
        return resolve_input_type(function_def)

    monkeypatch.setattr(FunctionDef, "input_type_cls", count_input_type_calls)
    oracle = PreparedOptimalActionOracle(functions)
    first = oracle.certify(
        TypedList([1]),
        TypedList([999]),
        limits=OracleLimits(max_depth=3),
    )
    second = oracle.certify(
        TypedList([2]),
        TypedList([999]),
        limits=OracleLimits(max_depth=3),
    )

    assert input_type_calls == (
        len(functions) + first.attempted_executions + second.attempted_executions
    )


def test_prepared_oracle_rejects_definition_mutation_but_not_usage_counts():
    step = function("step", "return x + 1")
    functions = FunctionDefSet([step])
    oracle = PreparedOptimalActionOracle(functions)
    limits = OracleLimits(max_depth=1)

    step.increment_usage()
    assert oracle.certify(TypedList([0]), TypedList([1]), limits=limits).status == (
        "certified"
    )

    step.code = "return x + 2"
    with pytest.raises(ValueError, match="source palette changed"):
        oracle(TypedList([0]), TypedList([1]), functions, limits=limits)
    with pytest.raises(ValueError, match="source palette changed"):
        oracle.certify(TypedList([0]), TypedList([1]), limits=limits)


def test_oracle_uses_canonical_answer_equality_but_search_safe_state_keys():
    functions = FunctionDefSet(
        [
            function(
                "positive_zero",
                "return 0.0",
                input_type="builtins.float",
                output_type="builtins.float",
            ),
            function(
                "negative_zero",
                "return -0.0",
                input_type="builtins.float",
                output_type="builtins.float",
            ),
        ]
    )
    result = certify_shortest_first_actions(
        TypedList([1.0]),
        TypedList([0.0]),
        functions,
        limits=OracleLimits(max_depth=1),
    )

    assert result.status == "certified"
    assert result.optimal_action_names == ("positive_zero", "negative_zero")
    # +0.0 and -0.0 are answer-equal but remain distinct search states.
    assert result.reached_states == 3


def test_oracle_rejects_a_target_seen_in_an_incomplete_layer():
    functions = FunctionDefSet(
        [
            function("first_route", "return x + 1"),
            function("second_route", "return x + 1"),
        ]
    )
    incomplete = certify_shortest_first_actions(
        TypedList([0]),
        TypedList([1]),
        functions,
        limits=OracleLimits(max_depth=1, max_executions=1),
    )
    exact_boundary = certify_shortest_first_actions(
        TypedList([0]),
        TypedList([1]),
        functions,
        limits=OracleLimits(max_depth=1, max_executions=2),
    )

    assert incomplete.status == "unresolved"
    assert incomplete.stop_reason == "max_executions"
    assert incomplete.observed_target_distance == 1
    assert incomplete.distance is None
    assert incomplete.optimal_actions_complete is False
    assert incomplete.optimal_action_names == ()
    assert incomplete.certified_depth == 0

    assert exact_boundary.status == "certified"
    assert exact_boundary.optimal_action_names == ("first_route", "second_route")


def test_oracle_distinguishes_depth_censoring_from_exhausted_reachability():
    depth_limited = certify_shortest_first_actions(
        TypedList([0]),
        TypedList([2]),
        FunctionDefSet([function("inc", "return x + 1")]),
        limits=OracleLimits(max_depth=1),
    )
    exhausted = certify_shortest_first_actions(
        TypedList([0]),
        TypedList([2]),
        FunctionDefSet([function("identity", "return x")]),
        limits=OracleLimits(max_depth=5),
    )

    assert depth_limited.status == "unresolved"
    assert depth_limited.stop_reason == "max_depth"
    assert depth_limited.certified_depth == 1
    assert exhausted.status == "unreachable"
    assert exhausted.stop_reason is None
    assert exhausted.certified_depth == 1
