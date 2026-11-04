import math
from dataclasses import FrozenInstanceError

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import (
    ActionScorer,
    RolloutDepthTrace,
    RolloutPredictor,
    RolloutSearchResult,
)
from wandering_light.typed_list import TypedList


def make_function(name: str, input_type: str, output_type: str, code: str):
    return FunctionDef(
        name=name, input_type=input_type, output_type=output_type, code=code
    )


INT = "builtins.int"
STR = "builtins.str"

inc = make_function("inc", INT, INT, "return x + 1")
double = make_function("double", INT, INT, "return x * 2")
neg = make_function("neg", INT, INT, "return -x")
exploding = make_function("exploding", INT, INT, "return x + None")
to_str = make_function("to_str", INT, STR, "return str(x)")
upper = make_function("upper", STR, STR, "return x.upper()")
length = make_function("length", STR, INT, "return len(x)")


class ScriptedScorer(ActionScorer):
    """Scores every row with the same fixed preferences and records calls.

    ``preferences`` maps function names to scores; names missing from it get
    ``0.0``. ``rows`` lets a test hand back an arbitrary response instead.
    """

    def __init__(self, preferences=None, rows=None):
        self.preferences = preferences or {}
        self.rows = rows
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append((list(problems), available_functions))
        if self.rows is not None:
            return self.rows
        return [
            {
                func.name: float(self.preferences.get(func.name, 0.0))
                for func in available_functions
            }
            for _ in problems
        ]


class PerStateScorer(ActionScorer):
    """Looks scores up per materialized state so tests can steer each node."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append([state for state, _ in problems])
        rows = []
        for state, _ in problems:
            overrides = self.table.get(tuple(state.items), {})
            rows.append(
                {
                    func.name: float(overrides.get(func.name, 0.0))
                    for func in available_functions
                }
            )
        return rows


def search(predictor, input_items, output_items, functions):
    return predictor.search_with_cost(
        TypedList(input_items), TypedList(output_items), FunctionDefSet(functions)
    )


# --- interface and construction -------------------------------------------


def test_action_scorer_is_abstract():
    with pytest.raises(TypeError):
        ActionScorer()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beam_width": 0}, "beam_width must be an integer >= 1"),
        ({"beam_width": True}, "beam_width must be an integer >= 1"),
        ({"beam_width": 2.0}, "beam_width must be an integer >= 1"),
        ({"max_depth": -1}, "max_depth must be an integer >= 0"),
        ({"max_depth": False}, "max_depth must be an integer >= 0"),
        ({"budget": -1}, "budget must be an integer >= 0"),
        ({"budget": "10"}, "budget must be an integer >= 0"),
    ],
)
def test_limits_must_be_non_bool_integers_at_least_the_minimum(kwargs, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        RolloutPredictor(ScriptedScorer(), **kwargs)


def test_default_limits_and_zero_limits_are_accepted():
    predictor = RolloutPredictor(ScriptedScorer())
    assert (predictor.beam_width, predictor.max_depth, predictor.budget) == (
        8,
        8,
        100_000,
    )
    zero = RolloutPredictor(ScriptedScorer(), beam_width=1, max_depth=0, budget=0)
    assert (zero.max_depth, zero.budget) == (0, 0)


def test_result_is_frozen_and_trace_is_excluded_from_equality():
    trace = (RolloutDepthTrace(1, (("inc",),), 1),)
    with_trace = RolloutSearchResult(FunctionDefList(), 3, "max_depth", trace)
    without_trace = RolloutSearchResult(FunctionDefList(), 3, "max_depth")

    assert with_trace == without_trace
    assert with_trace != RolloutSearchResult(FunctionDefList(), 4, "max_depth")
    assert with_trace.trace == trace
    with pytest.raises(FrozenInstanceError):
        with_trace.executions = 1
    with pytest.raises(FrozenInstanceError):
        trace[0].depth = 2


# --- pre-scoring terminations ---------------------------------------------


def test_equal_input_and_output_is_solved_before_anything_else():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=0)

    result = search(predictor, [1], [1], [])

    assert result == RolloutSearchResult(FunctionDefList(), 0, "solved")
    assert result.trace == ()
    assert scorer.calls == []


def test_no_available_functions_is_frontier_exhausted_before_budget():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=0)

    result = search(predictor, [1], [2], [])

    assert result == RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
    assert scorer.calls == []


def test_zero_budget_is_reported_before_zero_depth():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=0)

    result = search(predictor, [1], [2], [inc])

    assert result == RolloutSearchResult(FunctionDefList(), 0, "budget")
    assert scorer.calls == []


def test_zero_depth_is_reported_without_scoring():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=0, budget=5)

    result = search(predictor, [1], [2], [inc])

    assert result == RolloutSearchResult(FunctionDefList(), 0, "max_depth")
    assert scorer.calls == []


# --- the scorer sees every materialized state -----------------------------


def test_scorer_is_called_once_per_layer_with_materialized_states_in_beam_order():
    scorer = ScriptedScorer({"double": 1.0, "inc": 0.0})
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=2, budget=100)
    functions = FunctionDefSet([inc, double])
    target = TypedList([100])

    result = predictor.search_with_cost(TypedList([3]), target, functions)

    assert result.termination == "max_depth"
    assert len(scorer.calls) == 2
    first_problems, first_functions = scorer.calls[0]
    assert first_problems == [(TypedList([3]), target)]
    assert first_functions is functions
    # Depth 2 scores the children of depth 1 in beam order: double is preferred.
    second_problems, _ = scorer.calls[1]
    assert second_problems == [(TypedList([6]), target), (TypedList([4]), target)]
    assert result.trace == (
        RolloutDepthTrace(1, (("double",), ("inc",)), 2),
        RolloutDepthTrace(2, (("double", "double"), ("double", "inc")), 4),
    )


def test_scorer_preference_steers_the_search_to_a_solution():
    scorer = ScriptedScorer({"double": 1.0})
    predictor = RolloutPredictor(scorer, beam_width=1, max_depth=3, budget=100)

    result = search(predictor, [1], [8], [inc, double])

    assert result.function_defs == FunctionDefList([double, double, double])
    assert result.termination == "solved"
    # Each layer attempts double first; the solving attempt is the fifth one.
    assert result.executions == 5
    assert [tuple(t.retained_paths) for t in result.trace] == [
        (("double",),),
        (("double", "double"),),
    ]


def test_a_path_the_scorer_dislikes_is_still_found_when_the_beam_is_wide():
    scorer = ScriptedScorer({"double": 10.0, "inc": 0.0})
    functions = [inc, double]

    wide = RolloutPredictor(scorer, beam_width=2, max_depth=2, budget=100)
    result = search(wide, [2], [6], functions)
    assert result.function_defs == FunctionDefList([inc, double])
    assert result.termination == "solved"
    assert result.executions == 5

    narrow = RolloutPredictor(scorer, beam_width=1, max_depth=2, budget=100)
    assert search(narrow, [2], [6], functions) == RolloutSearchResult(
        FunctionDefList(), 4, "max_depth"
    )


# --- type filtering and per-node normalization ----------------------------


def test_incompatible_functions_are_never_attempted():
    scorer = ScriptedScorer({"upper": 100.0})
    predictor = RolloutPredictor(scorer, beam_width=4, max_depth=1, budget=100)

    result = search(predictor, [1], [5], [upper, inc])

    assert result == RolloutSearchResult(FunctionDefList(), 1, "max_depth")
    assert result.trace == (RolloutDepthTrace(1, (("inc",),), 1),)


def test_scores_are_normalized_over_compatible_functions_only():
    # Depth 1 from ["a"]: upper and length tie, so the beam is [["A"], [1]].
    # Depth 2: inc from [1] should outrank upper from ["A"] only if the huge
    # scores given to the str functions on [1] are ignored in its softmax.
    table = {
        ("A",): {"upper": 1.0, "length": 0.0, "inc": 0.0, "neg": 0.0},
        (1,): {"inc": 5.0, "neg": 0.0, "upper": 100.0, "length": 100.0},
    }
    scorer = PerStateScorer(table)
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=2, budget=100)

    result = search(predictor, ["a"], [2], [upper, length, inc, neg])

    assert result.function_defs == FunctionDefList([length, inc])
    assert result.termination == "solved"
    assert result.executions == 3
    assert scorer.calls == [[TypedList(["a"])], [TypedList(["A"]), TypedList([1])]]


def test_single_compatible_function_has_log_probability_zero():
    # [1] has only inc compatible; its log-prob is 0 regardless of score, so
    # the str branch (two-way tie, log 1/2) is attempted after it at depth 2.
    scorer = PerStateScorer(
        {
            ("a",): {"upper": 0.0, "length": 0.0},
            (1,): {"inc": -50.0, "upper": 60.0, "length": 60.0},
            ("A",): {"upper": 3.0, "length": 3.0},
        }
    )
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=2, budget=100)

    result = search(predictor, ["a"], [2], [upper, length, inc])

    assert result.function_defs == FunctionDefList([length, inc])
    assert result.executions == 3
    assert result.termination == "solved"


# --- attempt ordering --------------------------------------------------------


def test_ties_are_broken_by_basis_position():
    scorer = ScriptedScorer()

    forward = RolloutPredictor(scorer, max_depth=1, budget=10)
    assert search(forward, [2], [4], [inc, double]).executions == 2

    reversed_basis = RolloutPredictor(scorer, max_depth=1, budget=10)
    assert search(reversed_basis, [2], [4], [double, inc]).executions == 1


def test_ties_are_broken_by_parent_beam_position_before_basis_position():
    # Depth 1 from [1]: neg -> [-1] and double -> [2] tie; neg is attempted
    # first (basis order) so it holds beam position 0. Depth 2 ties everywhere,
    # so attempts run neg's children (neg, double) before double's children.
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=2, budget=100)

    result = search(predictor, [1], [4], [neg, double])

    assert result.function_defs == FunctionDefList([double, double])
    assert result.executions == 6
    assert result.trace == (RolloutDepthTrace(1, (("neg",), ("double",)), 2),)


def test_transitions_are_attempted_in_descending_cumulative_log_probability():
    # Depth 1 from [0]: double and neg both give [0] (deduplicated), inc gives
    # [1]; double is preferred so the beam is [[0], [1]]. At depth 2 the
    # cumulative log-probabilities interleave the two parents: double(0) first,
    # then double(1) from the lower-ranked parent ahead of inc(0) and neg(0).
    table = {
        (0,): {"inc": 0.0, "double": 1.0, "neg": -100.0},
        (1,): {"inc": 0.0, "double": 5.0, "neg": 0.0},
    }
    scorer = PerStateScorer(table)
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=2, budget=100)

    result = search(predictor, [0], [2], [inc, double, neg])

    assert result.function_defs == FunctionDefList([inc, double])
    assert result.termination == "solved"
    assert result.executions == 5
    assert result.trace == (RolloutDepthTrace(1, (("double",), ("inc",)), 2),)


# --- deduplication and beam truncation ------------------------------------


def test_successors_are_deduplicated_by_search_key_within_a_layer():
    scorer = ScriptedScorer({"inc": 1.0, "double": 0.0})
    predictor = RolloutPredictor(scorer, beam_width=4, max_depth=1, budget=100)

    result = search(predictor, [1], [5], [inc, double])

    assert result.termination == "max_depth"
    assert result.executions == 2
    assert result.trace == (RolloutDepthTrace(1, (("inc",),), 1),)


def test_signed_zero_states_are_kept_apart():
    neg_float = make_function("neg", "builtins.float", "builtins.float", "return -x")
    identity = make_function("identity", "builtins.float", "builtins.float", "return x")
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=4, max_depth=1, budget=100)

    result = search(predictor, [0.0], [1.0], [neg_float, identity])

    assert result.trace == (RolloutDepthTrace(1, (("neg",), ("identity",)), 2),)


def test_beam_keeps_the_best_beam_width_successors():
    scorer = ScriptedScorer({"inc": 2.0, "double": 1.0, "neg": 0.0})
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=1, budget=100)

    result = search(predictor, [3], [100], [neg, double, inc])

    assert result == RolloutSearchResult(FunctionDefList(), 3, "max_depth")
    assert result.trace == (RolloutDepthTrace(1, (("inc",), ("double",)), 3),)


def test_beam_truncation_ties_are_broken_by_attempt_order():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=1, budget=100)

    result = search(predictor, [3], [100], [neg, double, inc])

    assert result.trace == (RolloutDepthTrace(1, (("neg",), ("double",)), 3),)


# --- budget accounting --------------------------------------------------------


def test_raising_transition_costs_an_execution_and_can_exhaust_the_budget():
    scorer = ScriptedScorer({"exploding": 1.0})
    predictor = RolloutPredictor(scorer, max_depth=1, budget=1)

    result = search(predictor, [0], [1], [exploding, inc])

    assert result == RolloutSearchResult(FunctionDefList(), 1, "budget")
    assert result.trace == ()


def test_raising_transition_is_skipped_when_budget_remains():
    scorer = ScriptedScorer({"exploding": 1.0})
    predictor = RolloutPredictor(scorer, max_depth=1, budget=2)

    result = search(predictor, [0], [1], [exploding, inc])

    assert result.function_defs == FunctionDefList([inc])
    assert result.executions == 2
    assert result.termination == "solved"


def test_solution_on_the_last_affordable_execution_wins():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=1, budget=1)

    result = search(predictor, [0], [1], [inc])

    assert result == RolloutSearchResult(FunctionDefList([inc]), 1, "solved")


def test_budget_exhausted_by_a_non_solving_execution():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, max_depth=3, budget=1)

    result = search(predictor, [0], [5], [inc])

    assert result == RolloutSearchResult(FunctionDefList(), 1, "budget")
    assert result.trace == ()


def test_budget_runs_out_mid_layer_keeps_earlier_trace():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=3, budget=3)

    result = search(predictor, [3], [100], [inc, double])

    assert result == RolloutSearchResult(FunctionDefList(), 3, "budget")
    assert result.trace == (RolloutDepthTrace(1, (("inc",), ("double",)), 2),)


def test_no_execution_is_attempted_once_the_budget_is_reached():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=5, budget=2)

    result = search(predictor, [3], [100], [inc, double])

    assert result.executions == 2
    assert result.termination == "budget"
    assert len(scorer.calls) == 1


# --- depth and frontier terminations ------------------------------------------


def test_max_depth_after_the_last_layer_returns_the_full_trace():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=1, max_depth=2, budget=100)

    result = search(predictor, [0], [5], [inc])

    assert result == RolloutSearchResult(FunctionDefList(), 2, "max_depth")
    assert result.trace == (
        RolloutDepthTrace(1, (("inc",),), 1),
        RolloutDepthTrace(2, (("inc", "inc"),), 1),
    )
    assert len(scorer.calls) == 2


def test_no_compatible_transition_in_a_layer_is_frontier_exhausted():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=3, budget=100)

    result = search(predictor, [1], [100], [to_str])

    # Depth 1 reaches ["1"]; no str function exists, so depth 2 has nothing.
    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert result.trace == (RolloutDepthTrace(1, (("to_str",),), 1),)
    assert len(scorer.calls) == 2


def test_all_transitions_raising_empties_the_beam():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=2, max_depth=3, budget=100)

    result = search(predictor, [0], [1], [exploding])

    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert result.trace == (RolloutDepthTrace(1, (), 0),)


# --- scorer response validation ---------------------------------------------


def test_wrong_number_of_rows_is_rejected():
    rows = [{"inc": 0.0}, {"inc": 0.0}]
    predictor = RolloutPredictor(ScriptedScorer(rows=rows), max_depth=1)

    with pytest.raises(
        ValueError, match=r"^action scorer returned 2 rows; expected 1$"
    ):
        search(predictor, [0], [5], [inc])


def test_non_list_response_is_rejected():
    predictor = RolloutPredictor(ScriptedScorer(rows=({"inc": 0.0},)), max_depth=1)

    with pytest.raises(ValueError, match=r"action scorer returned .* rows; expected 1"):
        search(predictor, [0], [5], [inc])


@pytest.mark.parametrize(
    "row",
    [
        {"inc": 0.0},
        {"inc": 0.0, "double": 0.0, "extra": 0.0},
        {"inc": 0.0, "upper": 0.0},
        [0.0, 0.0],
    ],
)
def test_rows_must_be_keyed_by_exactly_the_available_names(row):
    predictor = RolloutPredictor(ScriptedScorer(rows=[row]), max_depth=1)

    with pytest.raises(ValueError, match="wrong function names"):
        search(predictor, [0], [5], [inc, double])


@pytest.mark.parametrize(
    "score", [math.nan, math.inf, -math.inf, True, "1.0", None, 1 + 1j]
)
def test_scores_must_be_finite_real_numbers(score):
    predictor = RolloutPredictor(
        ScriptedScorer(rows=[{"inc": 0.0, "double": score}]), max_depth=1
    )

    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        search(predictor, [0], [5], [inc, double])


def test_incompatible_functions_scores_are_validated_too():
    predictor = RolloutPredictor(
        ScriptedScorer(rows=[{"inc": 0.0, "upper": math.nan}]), max_depth=1
    )

    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        search(predictor, [0], [5], [inc, upper])


def test_integer_scores_are_accepted():
    predictor = RolloutPredictor(
        ScriptedScorer(rows=[{"inc": 3, "double": 0}]), max_depth=1
    )

    result = search(predictor, [0], [1], [double, inc])

    assert result == RolloutSearchResult(FunctionDefList([inc]), 1, "solved")


# --- predictor API --------------------------------------------------------------


def test_predict_functions_batch_runs_each_problem_with_its_own_budget():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer, beam_width=1, max_depth=3, budget=2)
    functions = FunctionDefSet([inc])
    recorded = []
    search_with_cost = predictor.search_with_cost

    def record(input_list, output_list, available_functions):
        result = search_with_cost(input_list, output_list, available_functions)
        recorded.append(result)
        return result

    predictor.search_with_cost = record
    predictions = predictor.predict_functions_batch(
        [
            (TypedList([0]), TypedList([1])),
            (TypedList([0]), TypedList([2])),
            (TypedList([0]), TypedList([3])),
            (TypedList([0]), TypedList([1])),
        ],
        functions,
    )

    assert predictions == [
        FunctionDefList([inc]),
        FunctionDefList([inc, inc]),
        FunctionDefList(),
        FunctionDefList([inc]),
    ]
    assert [r.executions for r in recorded] == [1, 2, 2, 1]
    assert [r.termination for r in recorded] == ["solved", "solved", "budget", "solved"]


def test_predict_functions_batch_with_no_problems_does_not_score():
    scorer = ScriptedScorer()
    predictor = RolloutPredictor(scorer)

    assert predictor.predict_functions_batch([], FunctionDefSet([inc])) == []
    assert scorer.calls == []
