"""The step-3 report reads verifier cells written before and after #142."""

from __future__ import annotations

from experiments.step3_report import verified_in


def test_verified_counts_draw_tested_and_witness_equal_solves_together():
    old = {"verified": {"1-5": 4, "6-10": 2}, "verified_total": 6, "graded_total": 9}
    new = {
        "verified": {"1-5": 3, "6-10": 0},
        "witness": {"1-5": 1, "6-10": 2},
        "verified_total": 3,
        "witness_total": 3,
        "graded_total": 9,
    }
    assert verified_in(old) == verified_in(new) == 6
    assert verified_in(old, "6-10") == verified_in(new, "6-10") == 2
    assert verified_in(old, witness_only=True) == 0
    assert verified_in(new, witness_only=True) == 3
    assert verified_in(new, "1-5", witness_only=True) == 1
