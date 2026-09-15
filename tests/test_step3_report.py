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


def test_the_verified_table_names_the_budget_the_file_was_rescored_at(tmp_path):
    """A beam-32 verification file must not be headed beam 64 (Codex review on #143)."""
    import json

    from experiments import step3_report as report

    for name, beam in (("v4only", 64), ("mixed", 32)):
        run = tmp_path / name
        (run / "timing").mkdir(parents=True)
        (run / "timing" / "1m-s0-cpu.json").write_text(
            json.dumps(
                {
                    "seed": 0,
                    "eval_corpus": "stack_pbe_v4",
                    "beam_width": beam,
                    "max_depth": 48,
                    "rows": [],
                }
            )
        )
    picked = report.cells(tmp_path / "mixed", "1m", "stack_pbe_v4", (32, 48))
    assert list(picked) == [0]
    assert report.cells(tmp_path / "mixed", "1m", "stack_pbe_v4") == {}
