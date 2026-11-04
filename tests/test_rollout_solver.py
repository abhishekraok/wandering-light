import math
from dataclasses import FrozenInstanceError

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.solver import (
    ActionScorer,
    FunctionPredictor,
    RolloutDepthTrace,
    RolloutPredictor,
    RolloutSearchResult,
)
from wandering_light.typed_list import TypedList


def make_function(name: str, code: str, input_type="builtins.int", output_type=None):
    return FunctionDef(
        name=name,
        input_type=input_type,
        output_type=output_type or input_type,
        code=code,
    )


INC = make_function("inc", "return x + 1")
DOUBLE = make_function("double", "return x * 2")
NEG = make_function("neg", "return -x")
BOOM = make_function("boom", "return x + None")
UPPER = make_function("upper", "return x.upper()", input_type="builtins.str")
TOSTR = make_function(
    "tostr", "return str(x)", input_type="builtins.int", output_type="builtins.str"
)


class FixedScorer(ActionScorer):
    """Returns the same scores for every state; records every call."""

    def __init__(self, scores=None, rows=None):
        self.scores = scores or {}
        self.rows = rows
        self.calls = []

    def score_actions_batch(self, problems, available_functions):
        self.calls.append(list(problems))
        if self.rows is not None:
            return self.rows
        return [
            {f.name: self.scores.get(f.name, 0.0) for f in available_functions}
            for _ in problems
        ]


class StateScorer(FixedScorer):
    """Scores keyed by the state's first item; missing entries score 0."""

    def __init__(self, by_state):
        super().__init__()
        self.by_state = by_state

    def score_actions_batch(self, problems, available_functions):
        self.calls.append(list(problems))
        return [
            {
                f.name: self.by_state.get(state[0], {}).get(f.name, 0.0)
                for f in available_functions
            }
            for state, _ in problems
        ]


def search(scorer, functions, start, target, **limits):
    return RolloutPredictor(scorer, **limits).search_with_cost(
        TypedList(start), TypedList(target), FunctionDefSet(functions)
    )


def test_is_a_function_predictor():
    assert isinstance(RolloutPredictor(FixedScorer()), FunctionPredictor)


@pytest.mark.parametrize(
    ("name", "value", "minimum"),
    [
        ("beam_width", 0, 1),
        ("beam_width", True, 1),
        ("beam_width", 2.0, 1),
        ("max_depth", -1, 0),
        ("max_depth", False, 0),
        ("budget", -1, 0),
        ("budget", "3", 0),
    ],
)
def test_limits_are_validated(name, value, minimum):
    with pytest.raises(ValueError) as error:
        RolloutPredictor(FixedScorer(), **{name: value})
    assert str(error.value) == f"{name} must be an integer >= {minimum}"


def test_zero_limits_are_allowed_where_documented():
    RolloutPredictor(FixedScorer(), beam_width=1, max_depth=0, budget=0)


def test_preconditions_are_checked_in_order_before_scoring():
    scorer = FixedScorer()
    assert search(scorer, [], [1], [1], budget=0, max_depth=0) == (
        RolloutSearchResult(FunctionDefList(), 0, "solved")
    )
    assert search(scorer, [], [1], [2], budget=0, max_depth=0).termination == (
        "frontier_exhausted"
    )
    assert search(scorer, [INC], [1], [2], budget=0, max_depth=0).termination == (
        "budget"
    )
    assert search(scorer, [INC], [1], [2], max_depth=0).termination == "max_depth"
    assert scorer.calls == []


def test_solves_by_following_scores_and_counts_executions():
    scorer = FixedScorer({"double": 5.0})
    result = search(scorer, [INC, DOUBLE], [3], [12], beam_width=1)

    assert result == RolloutSearchResult(FunctionDefList([DOUBLE, DOUBLE]), 3, "solved")
    # Layer 1 tried double then inc; beam_width=1 kept only double.
    assert result.trace == (RolloutDepthTrace(1, (("double",),), 2),)
    assert [len(call) for call in scorer.calls] == [1, 1]
    assert scorer.calls[1][0][0] == TypedList([6])
    assert scorer.calls[1][0][1] == TypedList([12])


def test_scorer_receives_one_problem_per_beam_node_in_beam_order():
    scorer = FixedScorer({"double": 2.0, "inc": 1.0})
    result = search(scorer, [INC, DOUBLE], [3], [100], beam_width=2, max_depth=2)

    assert result.termination == "max_depth"
    assert scorer.calls[0] == [(TypedList([3]), TypedList([100]))]
    assert scorer.calls[1] == [
        (TypedList([6]), TypedList([100])),
        (TypedList([4]), TypedList([100])),
    ]


def test_trace_excluded_from_equality_and_result_is_frozen():
    a = RolloutSearchResult(FunctionDefList(), 1, "budget")
    b = RolloutSearchResult(
        FunctionDefList(), 1, "budget", (RolloutDepthTrace(1, (), 0),)
    )
    assert a == b
    with pytest.raises(FrozenInstanceError):
        a.executions = 2
    with pytest.raises(FrozenInstanceError):
        RolloutDepthTrace(1, (), 0).depth = 2


def test_successors_deduplicated_and_beam_ranked_by_log_probability():
    # From [0]: double -> [0], neg -> [0], inc -> [1]. Two unique successors.
    scorer = FixedScorer({"double": 3.0, "neg": 2.0, "inc": 1.0})
    result = search(scorer, [INC, DOUBLE, NEG], [0], [99], beam_width=8, max_depth=1)

    assert result.termination == "max_depth"
    assert result.executions == 3
    assert result.trace == (RolloutDepthTrace(1, (("double",), ("inc",)), 2),)


def test_ties_broken_by_parent_position_then_basis_position():
    scorer = FixedScorer()  # all scores equal
    result = search(scorer, [NEG, INC, DOUBLE], [3], [99], beam_width=3, max_depth=2)

    assert result.termination == "max_depth"
    assert result.executions == 12
    first, second = result.trace
    assert first == RolloutDepthTrace(1, (("neg",), ("inc",), ("double",)), 3)
    # Depth 2 attempts parents in beam order, functions in basis order:
    # [-3] -> 3, -2, -6; [4] -> -4, 5, 8; [6] -> -6 (duplicate), 7, 12.
    assert second == RolloutDepthTrace(
        2, (("neg", "neg"), ("neg", "inc"), ("neg", "double")), 8
    )


def test_equal_log_probability_duplicate_keeps_the_first_attempt():
    # From [1], inc and double both reach [2] with equal log-probability.
    result = search(FixedScorer(), [INC, DOUBLE], [1], [99], max_depth=1)
    assert result.trace == (RolloutDepthTrace(1, (("inc",),), 1),)


def test_log_softmax_is_over_compatible_functions_only():
    # Equal scores everywhere. Layer 1 from [3]: inc -> [4], tostr -> ["3"],
    # each at log(1/2). At layer 2 the str node has one compatible function,
    # so upper gets probability 1 there and outranks the int node's children
    # (1/4 each). Normalizing over all three names would tie everything and
    # put the int node's children first by parent position.
    scorer = FixedScorer()
    result = search(scorer, [INC, TOSTR, UPPER], [3], [99], beam_width=3, max_depth=2)
    assert result.trace[0].retained_paths == (("inc",), ("tostr",))
    assert result.trace[1].retained_paths == (
        ("tostr", "upper"),
        ("inc", "inc"),
        ("inc", "tostr"),
    )


def test_cumulative_log_probability_reranks_across_parents():
    # Layer 1 from [3]: inc -> [4] at p=0.6, double -> [6] at p=0.4.
    # Layer 2: [4] splits evenly (0.3, 0.3); [6] gives inc 0.9 -> 0.36.
    scorer = StateScorer(
        {
            3: {"inc": math.log(0.6), "double": math.log(0.4)},
            6: {"inc": math.log(9.0)},
        }
    )
    result = search(scorer, [INC, DOUBLE], [3], [99], beam_width=2, max_depth=2)
    assert result.trace[1].retained_paths == (("double", "inc"), ("inc", "inc"))
    assert result.trace[1].unique_successors == 4


def test_raising_transition_costs_an_execution_and_is_skipped():
    scorer = FixedScorer({"boom": 5.0})
    result = search(scorer, [BOOM, INC], [0], [1])
    assert result == RolloutSearchResult(FunctionDefList([INC]), 2, "solved")


def test_raise_on_last_affordable_execution_ends_budget():
    scorer = FixedScorer({"boom": 5.0})
    result = search(scorer, [BOOM, INC], [0], [1], budget=1)
    assert result == RolloutSearchResult(FunctionDefList(), 1, "budget")
    assert result.trace == ()


def test_solution_on_last_affordable_execution_wins():
    scorer = FixedScorer({"double": 5.0})
    result = search(scorer, [DOUBLE, INC], [2], [3], budget=2)
    assert result == RolloutSearchResult(FunctionDefList([INC]), 2, "solved")


def test_budget_exhausted_mid_layer_keeps_trace_of_complete_layers():
    scorer = FixedScorer({"double": 5.0})
    result = search(scorer, [DOUBLE, INC], [3], [99], budget=3, beam_width=1)
    assert result == RolloutSearchResult(FunctionDefList(), 3, "budget")
    assert result.trace == (RolloutDepthTrace(1, (("double",),), 2),)


def test_no_compatible_transition_is_frontier_exhausted():
    scorer = FixedScorer()
    result = search(scorer, [UPPER], [1], [2])
    assert result == RolloutSearchResult(FunctionDefList(), 0, "frontier_exhausted")
    assert len(scorer.calls) == 1


def test_all_transitions_raising_empties_the_beam():
    scorer = FixedScorer()
    result = search(scorer, [BOOM], [1], [2])
    assert result == RolloutSearchResult(FunctionDefList(), 1, "frontier_exhausted")
    assert result.trace == (RolloutDepthTrace(1, (), 0),)


def test_wrong_row_count_is_rejected():
    scorer = FixedScorer(rows=[])
    with pytest.raises(ValueError) as error:
        search(scorer, [INC], [1], [3])
    assert str(error.value) == "action scorer returned 0 rows; expected 1"

    with pytest.raises(ValueError, match="rows; expected 1"):
        search(FixedScorer(rows=({"inc": 0.0},)), [INC], [1], [3])


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"inc": 0.0},
        {"inc": 0.0, "extra": 0.0},
        {"inc": 0.0, "double": 0.0, "extra": 0.0},
        ["inc", "double"],
    ],
)
def test_wrong_function_names_are_rejected(row):
    with pytest.raises(ValueError, match="wrong function names"):
        search(FixedScorer(rows=[row]), [INC, DOUBLE], [1], [3])


@pytest.mark.parametrize("score", [True, math.nan, math.inf, -math.inf, "1", None])
def test_bad_scores_are_rejected_even_for_incompatible_functions(score):
    row = {"inc": 0.0, "upper": score}
    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        search(FixedScorer(rows=[row]), [INC, UPPER], [1], [3])


def test_integer_scores_are_accepted():
    row = {"inc": 1, "double": 0}
    result = search(FixedScorer(rows=[row]), [INC, DOUBLE], [1], [2])
    assert result.termination == "solved"


def test_predict_functions_batch_gives_each_problem_its_own_budget():
    scorer = FixedScorer({"inc": 1.0})
    predictor = RolloutPredictor(scorer, beam_width=1, budget=2)
    predictions = predictor.predict_functions_batch(
        [
            (TypedList([3]), TypedList([99])),  # spends its whole budget
            (TypedList([3]), TypedList([5])),  # inc, double, then inc: costs 3
            (TypedList([5]), TypedList([6])),  # solved on the first execution
        ],
        FunctionDefSet([INC, DOUBLE]),
    )
    assert predictions == [FunctionDefList(), FunctionDefList(), FunctionDefList([INC])]
