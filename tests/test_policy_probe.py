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


def _task(distance, optimal):
    return {
        "metadata": {
            "certified_distance": distance,
            "optimal_first_action_names": optimal,
        }
    }


def test_evaluate_reports_top1_and_distance_breakdown():
    tasks = [_task(1, ["a"]), _task(2, ["b"]), _task(2, ["a"])]
    perfect = evaluate(lambda t: {"a": 1.0, "b": 0.0}, tasks, "x")
    assert perfect["top1"] == pytest.approx(2 / 3)
    assert perfect["by_distance"][1] == pytest.approx(1.0)
    assert perfect["by_distance"][2] == pytest.approx(1.5)


def test_evaluate_is_not_fooled_by_a_flat_predictor():
    tasks = [_task(3, ["a"]) for _ in range(4)]
    flat = evaluate(lambda t: {"a": 0.0, "b": 0.0, "c": 0.0}, tasks, "flat")
    assert flat["top1"] == 0.0
    assert flat["mean_rank"] == 3.0
