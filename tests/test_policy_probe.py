"""The probe's conclusion rests on its ranking metric. Pin that."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from policy_probe import evaluate, rank_of_first_optimal


def test_uniform_scores_rank_worst_not_best():
    """The tie-break must be pessimistic.

    An earlier version sorted optimal actions first among ties, which made a
    uniform predictor score 100% top-1. Any predictor that carries no
    information must land at the bottom.
    """
    flat = dict.fromkeys("abcdef", 0.0)
    assert rank_of_first_optimal(flat, {"d"}) == 6
    assert rank_of_first_optimal(flat, {"a", "f"}) == 5


def test_rank_follows_the_score_order():
    scores = {"a": 0.1, "b": 0.9, "c": 0.0}
    assert rank_of_first_optimal(scores, {"b"}) == 1
    assert rank_of_first_optimal(scores, {"a"}) == 2
    assert rank_of_first_optimal(scores, {"c"}) == 3


def test_any_optimal_action_counts():
    """Tasks can have several optimal first actions; hitting one is enough."""
    scores = {"a": 0.3, "b": 0.2, "c": 0.1}
    assert rank_of_first_optimal(scores, {"b", "c"}) == 2


def test_missing_optimal_action_ranks_past_the_end():
    scores = {"a": 1.0, "b": 0.5}
    assert rank_of_first_optimal(scores, {"zzz"}) == 3


FIRST = "optimal_first_action_names"


def _task(distance, optimal, field=FIRST):
    return {"metadata": {"certified_distance": distance, field: optimal}}


def test_evaluate_reports_top1_and_distance_breakdown():
    tasks = [_task(1, ["a"]), _task(2, ["b"]), _task(2, ["a"])]
    perfect = evaluate(lambda t: {"a": 1.0, "b": 0.0}, tasks, "x", FIRST)
    assert perfect["top1"] == pytest.approx(2 / 3)
    assert perfect["by_distance"][1] == pytest.approx(1.0)
    assert perfect["by_distance"][2] == pytest.approx(1.5)


def test_evaluate_is_not_fooled_by_a_flat_predictor():
    tasks = [_task(3, ["a"]) for _ in range(4)]
    flat = evaluate(lambda t: {"a": 0.0, "b": 0.0, "c": 0.0}, tasks, "flat", FIRST)
    assert flat["top1"] == 0.0
    assert flat["mean_rank"] == 3.0


def test_last_action_masks_on_output_type_not_input():
    """The two ends are not symmetric.

    A first action must consume the task's input type; a last action must
    produce its output type. Masking a last-action probe on input type would
    compare predictors against a mask that does not apply to them.
    """
    from policy_probe import action_field

    names_field, complete_flag = action_field(last=True)
    assert names_field == "optimal_last_action_names"
    assert complete_flag == "optimal_last_actions_complete"
    assert action_field(last=False) == (
        "optimal_first_action_names",
        "optimal_first_actions_complete",
    )


def test_type_mask_uses_the_field_it_is_given():
    from policy_probe import make_type_mask

    type_of = {"a": "builtins.int", "b": "builtins.str"}
    task = {"metadata": {"input_type": "builtins.int", "output_type": "builtins.str"}}
    assert make_type_mask(["a", "b"], type_of, "input_type")(task) == {
        "a": 1.0,
        "b": 0.0,
    }
    assert make_type_mask(["a", "b"], type_of, "output_type")(task) == {
        "a": 0.0,
        "b": 1.0,
    }
