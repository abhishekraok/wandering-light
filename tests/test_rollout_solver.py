import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import (
    ActionScorer,
    FunctionPredictor,
    RolloutDepthTrace,
    RolloutPredictor,
    RolloutSearchResult,
    TrajectorySolver,
)
from wandering_light.typed_list import TypedList


def make_function(name: str, input_type: str, output_type: str, code: str):
    return FunctionDef(
        name=name, input_type=input_type, output_type=output_type, code=code
    )


INC = make_function("inc", "builtins.int", "builtins.int", "return x + 1")
DBL = make_function("dbl", "builtins.int", "builtins.int", "return x * 2")
NEG = make_function("neg", "builtins.int", "builtins.int", "return -x")
SQ = make_function("sq", "builtins.int", "builtins.int", "return x * x")
BOOM = make_function("boom", "builtins.int", "builtins.int", "return x + None")
UPPER = make_function("upper", "builtins.str", "builtins.str", "return x.upper()")
TO_STR = make_function("to_str", "builtins.int", "builtins.str", "return str(x)")


class ScriptedScorer(ActionScorer):
    """Scores with ``policy(state, target, names)`` and records every call."""

    def __init__(self, policy=None):
        self.policy = policy or (lambda state, target, names: dict.fromkeys(names, 0))
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append((list(problems), available_functions))
        names = [function.name for function in available_functions.functions]
        return [self.policy(state, target, names) for state, target in problems]


class FixedRowsScorer(ActionScorer):
    def __init__(self, rows):
        self.rows = rows

    def score_actions_batch(self, problems, available_functions):
        return self.rows


def by_name(**scores):
    """A policy giving each named function a fixed score and the rest 0."""

    def policy(state, target, names):
        return {name: scores.get(name, 0.0) for name in names}

    return policy


@pytest.fixture
def executions(monkeypatch):
    """Record every basis execution as ``(function name, input items)``."""
    recorded = []
    execute = Executor.execute

    def record(self, fn_def, inputs):
        recorded.append((fn_def.name, list(inputs.items)))
        return execute(self, fn_def, inputs)

    monkeypatch.setattr(Executor, "execute", record)
    return recorded


# --- construction -----------------------------------------------------------


def test_is_a_function_predictor_with_documented_defaults():
    predictor = RolloutPredictor(ScriptedScorer())

    assert isinstance(predictor, FunctionPredictor)
    assert (predictor.beam_width, predictor.max_depth, predictor.budget) == (
        8,
        8,
        100_000,
    )


def test_action_scorer_is_abstract():
    with pytest.raises(TypeError):
        ActionScorer()


@pytest.mark.parametrize(
    ("name", "minimum", "value"),
    [
        ("beam_width", 1, 0),
        ("beam_width", 1, -3),
        ("beam_width", 1, True),
        ("beam_width", 1, 2.0),
        ("max_depth", 0, -1),
        ("max_depth", 0, False),
        ("max_depth", 0, "3"),
        ("budget", 0, -1),
        ("budget", 0, True),
        ("budget", 0, None),
    ],
)
def test_invalid_limits_are_rejected(name, minimum, value):
    with pytest.raises(ValueError) as error:
        RolloutPredictor(ScriptedScorer(), **{name: value})

    assert str(error.value) == f"{name} must be an integer >= {minimum}"


def test_minimum_limits_are_accepted():
    predictor = RolloutPredictor(ScriptedScorer(), beam_width=1, max_depth=0, budget=0)

    assert (predictor.beam_width, predictor.max_depth, predictor.budget) == (1, 0, 0)


# --- checks before any scoring ----------------------------------------------


def test_zero_distance_is_solved_before_anything_else():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=0)

    result = predictor.search_with_cost(
        TypedList([1]), TypedList([1]), FunctionDefSet()
    )

    assert result == RolloutSearchResult(FunctionDefList(), 0, "solved")
    assert result.trace == ()
    assert scorer.calls == []


def test_no_functions_is_frontier_exhausted_before_budget_and_depth():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=0)

    result = predictor.search_with_cost(
        TypedList([1]), TypedList([2]), FunctionDefSet()
    )

    assert result == RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
    assert scorer.calls == []


def test_zero_budget_is_reported_before_zero_depth():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=0)

    result = predictor.search_with_cost(
        TypedList([1]), TypedList([2]), FunctionDefSet([INC])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 0, "budget")
    assert scorer.calls == []


def test_zero_depth_is_max_depth_without_scoring():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=5)

    result = predictor.search_with_cost(
        TypedList([1]), TypedList([2]), FunctionDefSet([INC])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 0, "max_depth")
    assert scorer.calls == []


# --- scorer contract ----------------------------------------------------------


def test_scorer_sees_each_materialized_beam_state_in_beam_order():
    functions = FunctionDefSet([INC, DBL])
    scorer = ScriptedScorer(by_name(inc=1.0))
    target = TypedList([100])

    result = RolloutPredictor(scorer, beam_width=2, max_depth=2).search_with_cost(
        TypedList([3]), target, functions
    )

    assert result.termination == "max_depth"
    assert len(scorer.calls) == 2
    first_problems, first_functions = scorer.calls[0]
    assert first_problems == [(TypedList([3]), target)]
    assert first_functions is functions
    second_problems, _ = scorer.calls[1]
    # inc scored higher, so [4] precedes [6] in the beam.
    assert second_problems == [(TypedList([4]), target), (TypedList([6]), target)]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "action scorer returned 0 rows; expected 1"),
        (
            [{"inc": 0.0}, {"inc": 0.0}],
            "action scorer returned 2 rows; expected 1",
        ),
    ],
)
def test_wrong_row_count_is_rejected(rows, message):
    predictor = RolloutPredictor(FixedRowsScorer(rows))

    with pytest.raises(ValueError) as error:
        predictor.search_with_cost(
            TypedList([1]), TypedList([5]), FunctionDefSet([INC])
        )

    assert str(error.value) == message


def test_non_list_rows_are_rejected():
    predictor = RolloutPredictor(FixedRowsScorer(({"inc": 0.0},)))

    with pytest.raises(
        ValueError, match=r"^action scorer returned .* rows; expected 1$"
    ):
        predictor.search_with_cost(
            TypedList([1]), TypedList([5]), FunctionDefSet([INC])
        )


@pytest.mark.parametrize(
    "row",
    [
        {"inc": 0.0},
        {"inc": 0.0, "dbl": 0.0, "neg": 0.0},
        {"inc": 0.0, "neg": 0.0},
        {},
        [0.0, 0.0],
    ],
)
def test_rows_must_be_keyed_by_exactly_the_available_names(row):
    predictor = RolloutPredictor(FixedRowsScorer([row]))

    with pytest.raises(ValueError, match="wrong function names"):
        predictor.search_with_cost(
            TypedList([1]), TypedList([5]), FunctionDefSet([INC, DBL])
        )


@pytest.mark.parametrize(
    "score", [True, False, math.nan, math.inf, -math.inf, "1.0", None, 10**400, 1j]
)
def test_bool_non_real_and_non_finite_scores_are_rejected(score):
    # The bad score belongs to an incompatible function: every score is checked.
    predictor = RolloutPredictor(FixedRowsScorer([{"inc": 0.0, "upper": score}]))

    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        predictor.search_with_cost(
            TypedList([1]), TypedList([5]), FunctionDefSet([INC, UPPER])
        )


@pytest.mark.parametrize("score", [0, 3, Fraction(1, 3), np.float32(0.5), np.int64(2)])
def test_real_finite_scores_are_accepted(score):
    predictor = RolloutPredictor(FixedRowsScorer([{"inc": score}]), max_depth=1)

    result = predictor.search_with_cost(
        TypedList([1]), TypedList([2]), FunctionDefSet([INC])
    )

    assert result.termination == "solved"


# --- ordering ---------------------------------------------------------------


def test_scores_are_normalized_per_node_over_compatible_functions(executions):
    # upper is never compatible; its huge score must not enter the softmax.
    # After layer 1 the beam is [4] (via inc, log p = -0.31) and [6] (via dbl,
    # log p = -1.31).  [4] scores its children evenly and [6] strongly prefers
    # inc, so with per-node normalization both children of [4] come first even
    # though [6] -> inc has the larger raw score.
    def policy(state, target, names):
        scores = {"inc": 0.0, "dbl": 0.0, "upper": 1000.0}
        if state == TypedList([3]):
            scores["inc"] = 1.0
        if state == TypedList([6]):
            scores["inc"] = 10.0
        return scores

    result = RolloutPredictor(
        ScriptedScorer(policy), beam_width=2, max_depth=2
    ).search_with_cost(
        TypedList([3]), TypedList([100]), FunctionDefSet([INC, UPPER, DBL])
    )

    assert executions == [
        ("inc", [3]),
        ("dbl", [3]),
        ("inc", [4]),
        ("dbl", [4]),
        ("inc", [6]),
        ("dbl", [6]),
    ]
    assert result.executions == 6
    assert result.termination == "max_depth"
    assert result.trace[1].retained_paths == (("inc", "inc"), ("inc", "dbl"))


def test_ties_break_by_parent_position_then_basis_position(executions):
    result = RolloutPredictor(
        ScriptedScorer(), beam_width=3, max_depth=2
    ).search_with_cost(
        TypedList([3]), TypedList([100]), FunctionDefSet([NEG, INC, DBL])
    )

    assert executions == [
        ("neg", [3]),
        ("inc", [3]),
        ("dbl", [3]),
        ("neg", [-3]),
        ("inc", [-3]),
        ("dbl", [-3]),
        ("neg", [4]),
        ("inc", [4]),
        ("dbl", [4]),
        ("neg", [6]),
        ("inc", [6]),
        ("dbl", [6]),
    ]
    assert result.trace == (
        RolloutDepthTrace(1, (("neg",), ("inc",), ("dbl",)), 3),
        # Nine attempts reach 3, -2, -6, -4, 5, 8, -6, 7, 12: eight unique.
        RolloutDepthTrace(2, (("neg", "neg"), ("neg", "inc"), ("neg", "dbl")), 8),
    )


def test_higher_probability_successors_fill_the_beam_first(executions):
    result = RolloutPredictor(
        ScriptedScorer(by_name(dbl=2.0, inc=1.0)), beam_width=2, max_depth=1
    ).search_with_cost(
        TypedList([3]), TypedList([100]), FunctionDefSet([NEG, INC, DBL])
    )

    assert [name for name, _ in executions] == ["dbl", "inc", "neg"]
    assert result.trace == (RolloutDepthTrace(1, (("dbl",), ("inc",)), 3),)


# --- execution accounting ---------------------------------------------------


def test_raising_attempt_costs_an_execution_and_is_skipped(executions):
    result = RolloutPredictor(
        ScriptedScorer(by_name(boom=5.0)), max_depth=1
    ).search_with_cost(TypedList([1]), TypedList([2]), FunctionDefSet([BOOM, INC]))

    assert result == RolloutSearchResult(FunctionDefList([INC]), 2, "solved")
    assert [name for name, _ in executions] == ["boom", "inc"]


def test_raise_on_the_last_affordable_execution_ends_with_budget():
    result = RolloutPredictor(
        ScriptedScorer(by_name(boom=5.0)), budget=1
    ).search_with_cost(TypedList([1]), TypedList([2]), FunctionDefSet([BOOM, INC]))

    assert result == RolloutSearchResult(FunctionDefList(), 1, "budget")
    assert result.trace == ()


def test_solution_on_the_last_affordable_execution_wins():
    result = RolloutPredictor(ScriptedScorer(), budget=1).search_with_cost(
        TypedList([1]), TypedList([2]), FunctionDefSet([INC])
    )

    assert result == RolloutSearchResult(FunctionDefList([INC]), 1, "solved")


def test_budget_is_counted_across_layers_and_beats_max_depth():
    functions = FunctionDefSet([INC])

    spent = RolloutPredictor(ScriptedScorer(), max_depth=3, budget=3).search_with_cost(
        TypedList([0]), TypedList([10]), functions
    )
    spare = RolloutPredictor(ScriptedScorer(), max_depth=3, budget=4).search_with_cost(
        TypedList([0]), TypedList([10]), functions
    )

    assert spent == RolloutSearchResult(FunctionDefList(), 3, "budget")
    assert [entry.depth for entry in spent.trace] == [1, 2]
    assert spare == RolloutSearchResult(FunctionDefList(), 3, "max_depth")
    assert [entry.depth for entry in spare.trace] == [1, 2, 3]


def test_multi_step_solution_returns_the_full_path():
    result = RolloutPredictor(
        ScriptedScorer(by_name(dbl=1.0)), beam_width=1, max_depth=4
    ).search_with_cost(TypedList([1]), TypedList([8]), FunctionDefSet([INC, DBL]))

    assert result == RolloutSearchResult(FunctionDefList([DBL, DBL, DBL]), 5, "solved")
    assert [entry.retained_paths for entry in result.trace] == [
        (("dbl",),),
        (("dbl", "dbl"),),
    ]


# --- deduplication, beam and trace ------------------------------------------


def test_successors_are_deduplicated_keeping_the_earlier_tie():
    # inc(1) == dbl(1) == [2]; sq(1) == [1].
    result = RolloutPredictor(ScriptedScorer(), max_depth=1).search_with_cost(
        TypedList([1]), TypedList([100]), FunctionDefSet([INC, DBL, SQ])
    )

    assert result.executions == 3
    assert result.trace == (RolloutDepthTrace(1, (("inc",), ("sq",)), 2),)


def test_beam_is_truncated_but_trace_counts_all_unique_successors():
    result = RolloutPredictor(
        ScriptedScorer(), beam_width=1, max_depth=1
    ).search_with_cost(TypedList([1]), TypedList([100]), FunctionDefSet([INC, DBL, SQ]))

    assert result.trace == (RolloutDepthTrace(1, (("inc",),), 2),)


def test_deduplication_uses_search_key_not_equality():
    # -0.0 == 0.0, but a basis function can tell them apart, so both survive.
    neg_float = make_function("negf", "builtins.float", "builtins.float", "return -x")
    zero = make_function("zero", "builtins.float", "builtins.float", "return 0.0")

    result = RolloutPredictor(ScriptedScorer(), max_depth=1).search_with_cost(
        TypedList([0.0]), TypedList([1.0]), FunctionDefSet([neg_float, zero])
    )

    assert result.trace == (RolloutDepthTrace(1, (("negf",), ("zero",)), 2),)


def test_layer_where_every_attempt_raises_ends_frontier_exhausted():
    result = RolloutPredictor(ScriptedScorer()).search_with_cost(
        TypedList([1]), TypedList([2]), FunctionDefSet([BOOM])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert result.trace == (RolloutDepthTrace(1, (), 0),)


def test_layer_without_compatible_transitions_ends_frontier_exhausted():
    # Layer 1 reaches ["1"]; no basis function accepts a str in layer 2.
    scorer = ScriptedScorer()

    result = RolloutPredictor(scorer).search_with_cost(
        TypedList([1]), TypedList(["x"]), FunctionDefSet([TO_STR])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert result.trace == (RolloutDepthTrace(1, (("to_str",),), 1),)
    assert len(scorer.calls) == 2
    assert scorer.calls[1][0] == [(TypedList(["1"]), TypedList(["x"]))]


def test_incompatible_beam_nodes_are_scored_but_not_expanded(executions):
    # The beam holds ["1"] and [2]; only [2] has compatible transitions.
    scorer = ScriptedScorer(by_name(to_str=1.0))

    result = RolloutPredictor(scorer, max_depth=2).search_with_cost(
        TypedList([1]), TypedList(["x"]), FunctionDefSet([TO_STR, INC])
    )

    assert scorer.calls[1][0] == [
        (TypedList(["1"]), TypedList(["x"])),
        (TypedList([2]), TypedList(["x"])),
    ]
    assert executions[2:] == [("to_str", [2]), ("inc", [2])]
    assert result == RolloutSearchResult(FunctionDefList(), 4, "max_depth")


def test_no_compatible_transition_at_the_first_layer_spends_nothing():
    scorer = ScriptedScorer()

    result = RolloutPredictor(scorer).search_with_cost(
        TypedList(["a"]), TypedList(["b"]), FunctionDefSet([INC])
    )

    assert result == RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
    assert result.trace == ()
    assert len(scorer.calls) == 1


# --- result type ------------------------------------------------------------


def test_trace_is_excluded_from_equality_and_result_is_frozen():
    with_trace = RolloutSearchResult(
        FunctionDefList(), 3, "max_depth", (RolloutDepthTrace(1, (("inc",),), 1),)
    )
    without_trace = RolloutSearchResult(FunctionDefList(), 3, "max_depth")

    assert with_trace == without_trace
    assert without_trace.trace == ()
    with pytest.raises(FrozenInstanceError):
        with_trace.executions = 4
    with pytest.raises(FrozenInstanceError):
        with_trace.trace[0].depth = 2


# --- batch API --------------------------------------------------------------


def test_batch_gives_each_problem_its_own_budget(monkeypatch):
    predictor = RolloutPredictor(ScriptedScorer(), beam_width=1, max_depth=3, budget=2)
    recorded = []
    search_with_cost = predictor.search_with_cost

    def record(input_list, output_list, available_functions):
        result = search_with_cost(input_list, output_list, available_functions)
        recorded.append(result)
        return result

    monkeypatch.setattr(predictor, "search_with_cost", record)
    predictions = predictor.predict_functions_batch(
        [
            (TypedList([0]), TypedList([1])),
            (TypedList([0]), TypedList([2])),
            (TypedList([0]), TypedList([3])),
            (TypedList([0]), TypedList([1])),
        ],
        FunctionDefSet([INC]),
    )

    assert [(r.executions, r.termination) for r in recorded] == [
        (1, "solved"),
        (2, "solved"),
        (2, "budget"),
        (1, "solved"),
    ]
    assert predictions == [
        FunctionDefList([INC]),
        FunctionDefList([INC, INC]),
        FunctionDefList(),
        FunctionDefList([INC]),
    ]
    assert predictor.predict_functions_batch([], FunctionDefSet([INC])) == []


def test_predictor_plugs_into_trajectory_solver():
    solver = TrajectorySolver(RolloutPredictor(ScriptedScorer(by_name(dbl=1.0))))

    result = solver.solve(TypedList([1]), TypedList([4]), FunctionDefSet([INC, DBL]))

    assert result.success
    assert result.trajectory.function_defs == FunctionDefList([DBL, DBL])
