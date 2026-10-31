"""Regression tests for satisfying programs with different training-record IDs."""

import pytest
import torch

from wandering_light import program_embedding_execution as execution
from wandering_light.program_embedding import symmetric_multi_positive_contrastive_loss
from wandering_light.program_embedding_eval import reciprocal_rank_bounds
from wandering_light.program_embedding_train import _validation_score, main


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


def test_program_positives_follow_mutual_execution_not_record_identity():
    matcher = execution.ExecutionMatcher()
    queries_a = [[("0", "0")], [("2", "4")]]
    queries_b = [[("0", "0")], [("3", "6")]]
    programs_a, programs_b = ["0 add", "dup add"], ["dup 0 add add", "dup add"]
    positive, known = matcher.program_masks(
        queries_a, programs_a, queries_b, programs_b
    )
    # Differently authored doubling programs agree across records. Identity only
    # agrees with doubling while both sampled sets contain the ambiguous zero.
    assert positive.tolist() == [[True, False], [True, True]]
    assert known.all()
    queries_a[0].append(("3", "3"))
    distinguished, known = matcher.program_masks(
        queries_a, programs_a, queries_b, programs_b
    )
    assert distinguished.tolist() == [[False, False], [True, True]]
    assert known.all()


def test_program_masks_reuse_own_diagonals_keep_order_and_require_both_crosses(
    monkeypatch,
):
    matcher = execution.ExecutionMatcher()
    queries_a = [[(f"a{i}", "out")] for i in range(3)]
    queries_b = [[(f"b{i}", "out")] for i in range(2)]
    grades = {
        ("B0", "a0"): None,
        ("A0", "b0"): False,
        ("B1", "a0"): True,
        ("A0", "b1"): None,
        ("B0", "a2"): True,
        ("A2", "b0"): True,
        ("B1", "a2"): False,
    }
    calls = []

    def matches(program, query):
        key = (program, query[0][0])
        calls.append(key)
        return grades[key]

    monkeypatch.setattr(matcher, "matches", matches)
    own_a = (torch.diag(torch.tensor([True, False, True])), torch.ones(3, 3).bool())
    own_b = (torch.eye(2).bool(), torch.ones(2, 2).bool())
    positive, known = matcher.program_masks(
        queries_a,
        ["A0", "A1", "A2"],
        queries_b,
        ["B0", "B1"],
        own_a=own_a,
        own_b=own_b,
    )
    assert positive.tolist() == [[False, False], [False, False], [True, False]]
    assert known.tolist() == [[True, False], [False, False], [True, True]]
    assert set(calls) == set(grades)
    stats = matcher.statistics()
    assert [stats[f"program_{key}"] for key in ("positive", "negative", "unknown")] == [
        1,
        2,
        3,
    ]


def test_unreliable_own_programs_exclude_rows_and_columns(monkeypatch):
    matcher = execution.ExecutionMatcher()
    grades = {("A0", "a0"): False, ("A1", "a1"): True, ("B0", "b0"): None}
    calls = []

    def matches(program, query):
        key = (program, query[0][0])
        calls.append(key)
        return grades[key]

    monkeypatch.setattr(matcher, "matches", matches)
    positive, known = matcher.program_masks(
        [[("a0", "out")], [("a1", "out")]],
        ["A0", "A1"],
        [[("b0", "out")]],
        ["B0"],
    )
    assert not positive.any()
    assert not known.any()
    assert set(calls) == set(grades)


@pytest.mark.parametrize(
    ("io", "program"), [(True, True), (True, False), (False, True)]
)
def test_spawned_batch_labels_match_serial_and_stats_do_not_double_count(io, program):
    queries_a = [[("0", "0")], [("2", "4")], [("3", "6")], [("1", "1")]]
    programs_a = ["0 add", "dup add", "dup 0 add add", "invalid"]
    queries_b = [[("0", "0")], [("3", "3")], [("4", "8")], [("1", "1")]]
    programs_b = ["dup add", "0 add", "dup 0 add add", "invalid"]
    keys = (
        "positive",
        "negative",
        "unknown",
        "program_positive",
        "program_negative",
        "program_unknown",
    )
    with (
        execution.ExecutionLabeler() as serial,
        execution.ExecutionLabeler(workers=2) as spawned,
    ):
        first_stats = None
        for _ in range(2):
            expected = serial.batch_masks(
                queries_a, programs_a, queries_b, programs_b, io=io, program=program
            )
            actual = spawned.batch_masks(
                queries_a, programs_a, queries_b, programs_b, io=io, program=program
            )
            for serial_pair, spawned_pair in zip(expected, actual, strict=True):
                if serial_pair is None:
                    assert spawned_pair is None
                else:
                    for a, b in zip(serial_pair, spawned_pair, strict=True):
                        assert torch.equal(a, b)
            stats = spawned.statistics()
            assert {key: stats[key] for key in keys} == {
                key: serial.statistics()[key] for key in keys
            }
            if first_stats is None:
                first_stats = stats
            else:
                assert all(stats[key] == 2 * first_stats[key] for key in keys)
    with pytest.raises(RuntimeError, match="closed"):
        spawned.batch_masks(
            queries_a, programs_a, queries_b, programs_b, io=io, program=program
        )


def test_alignment_does_not_grade_program_pairs_and_program_only_avoids_io_masks(
    monkeypatch,
):
    queries, programs = [[("2", "4")]], ["dup add"]
    with execution.ExecutionLabeler() as labeler:

        def unexpected(*args, **kwargs):
            pytest.fail("Disabled label path must not run")

        with monkeypatch.context() as patch:
            patch.setattr(labeler.matcher, "program_masks", unexpected)
            assert (
                labeler.batch_masks(
                    queries, programs, queries, programs, io=True, program=False
                )[2]
                is None
            )
        with monkeypatch.context() as patch:
            patch.setattr(labeler.matcher, "masks", unexpected)
            own_a, own_b, pair = labeler.batch_masks(
                queries, programs, queries, programs, io=False, program=True
            )
            assert own_a is own_b is None
            assert pair[0].all() and pair[1].all()


def test_negative_worker_count_is_rejected_by_cli(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["--out", str(tmp_path), "--execution-label-workers", "-1"])
    assert caught.value.code == 2


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
