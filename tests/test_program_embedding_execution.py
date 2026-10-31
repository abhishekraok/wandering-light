"""Regression tests for satisfying programs with different training-record IDs."""

import pytest
import torch

from wandering_light import program_embedding_execution as execution
from wandering_light.program_embedding import symmetric_multi_positive_contrastive_loss
from wandering_light.program_embedding_eval import reciprocal_rank_bounds
from wandering_light.program_embedding_train import _validation_score


def test_positive_masks_follow_execution_and_change_when_examples_are_added():
    matcher = execution.ExecutionMatcher()
    # Identity and doubling have different behavior but agree at zero.
    queries = [[("0", "0")], [("0", "0"), ("3", "3")], [("2", "4")]]
    positives, known = matcher.masks(queries, ["0 add", "dup add", "dup 0 add add"])
    assert positives.tolist() == [
        [True, True, True],
        [True, False, False],
        [False, True, True],
    ]
    assert known.all()
    before = matcher.statistics()
    again, _ = matcher.masks(queries, ["0 add", "dup add", "dup 0 add add"])
    assert torch.equal(again, positives)
    assert matcher.statistics()["cache_misses"] == before["cache_misses"]
    assert matcher.statistics()["cache_hits"] > before["cache_hits"]


def test_incomplete_execution_is_unknown_but_completed_failure_is_negative(monkeypatch):
    def grade(program, examples, seconds):
        given, wanted = examples[0]
        return None if given == "slow" else given == wanted

    monkeypatch.setattr(execution, "_matches", grade)
    matcher = execution.ExecutionMatcher(cache_size=2)
    positives, known = matcher.masks(
        [[("slow", "x")], [("slow", "x"), ("wrong", "wanted")], [("ok", "ok")]],
        ["program"],
    )
    assert positives.tolist() == [[False], [False], [True]]
    assert known.tolist() == [[False], [True], [True]]
    assert matcher.statistics()["cache_entries"] <= 2
    with pytest.raises(ValueError, match="shown examples"):
        matcher.matches("program", [])


def test_unknown_candidates_have_no_effect_on_loss_or_gradient():
    torch.manual_seed(4)
    left = torch.randn(2, 5, requires_grad=True)
    right = torch.randn(3, 5, requires_grad=True)
    positive = torch.tensor([[True, False, False], [False, True, False]])
    known = torch.tensor([[True, True, False], [True, True, False]])
    actual = symmetric_multi_positive_contrastive_loss(
        left, right, positive_mask=positive, valid_mask=known
    )
    expected = symmetric_multi_positive_contrastive_loss(left, right[:2])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.equal(right.grad[2], torch.zeros(5))
    assert torch.isfinite(left.grad).all()


def test_no_positive_anchors_and_all_unknown_batches_are_safe():
    left = torch.randn(2, 4, requires_grad=True)
    right = torch.randn(2, 4, requires_grad=True)
    positive = torch.tensor([[True, False], [False, False]])
    known = torch.tensor([[True, False], [False, False]])
    loss = symmetric_multi_positive_contrastive_loss(
        left, right, positive_mask=positive, valid_mask=known
    )
    loss.backward()
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()
    assert torch.equal(left.grad[1], torch.zeros(4))
    left.grad = right.grad = None
    zero = symmetric_multi_positive_contrastive_loss(
        left,
        right,
        positive_mask=torch.zeros_like(positive),
        valid_mask=torch.zeros_like(known),
    )
    assert zero.item() == 0
    zero.backward()
    assert torch.equal(left.grad, torch.zeros_like(left))
    assert torch.equal(right.grad, torch.zeros_like(right))
    with pytest.raises(ValueError, match="known"):
        symmetric_multi_positive_contrastive_loss(
            left, right, positive_mask=positive, valid_mask=torch.zeros_like(known)
        )


@pytest.mark.parametrize(
    ("outcomes", "bounds"),
    [
        ([False, True], [0.5, 0.5]),
        ([None, True], [0.5, 1.0]),
        ([False, None, True], [1 / 3, 0.5]),
        ([None, None], [0.0, 1.0]),
        ([False, False], [0.0, 0.0]),
    ],
)
def test_execution_rank_bounds_keep_unknowns_explicit(outcomes, bounds):
    assert reciprocal_rank_bounds(outcomes) == bounds


def test_validation_selection_uses_execution_instead_of_record_identity():
    report = {
        "partitions": {
            "validation": {
                "io_to_program": {"mrr": 0.0},
                "execution": {"shown_mrr@10_bounds": [1.0, 1.0]},
            }
        }
    }
    assert _validation_score(report) == 1.0


@pytest.mark.parametrize(("seconds", "size"), [(0, 2), (float("nan"), 2), (1, -1)])
def test_invalid_execution_limits(seconds, size):
    with pytest.raises(ValueError):
        execution.ExecutionMatcher(seconds=seconds, cache_size=size)
