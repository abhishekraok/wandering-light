"""The matched-step pairing in the step-3 epoch reader (world-plus-v4-policy, #134)."""

from experiments.step3_epochs_report import counts, step_pairs


def test_counts_report_the_witness_share_of_verified_solves():
    def row(behaviour, program, solved=True, bucket="1-5"):
        return {
            "behaviour": behaviour,
            "program": program.split(),
            "solved": solved,
            "bucket": bucket,
        }

    cell = {
        "rows": [
            row("keep_below", "[ lt? ] swap cons filter", bucket="6-10"),
            row("add_pair", "swap add"),
            row("all_positive", "bogus"),
            row("drop_last", "x", solved=False),
        ]
    }
    witness = {("keep_below", "[ lt? ] swap cons filter")}
    passing = witness | {("add_pair", "swap add")}
    out = counts(cell, passing, witness)
    assert (out["graded"], out["verified"], out["witness"]) == (3, 2, 1)
    assert out["buckets"]["6-10"] == 1
    assert counts(cell, None)["verified"] == 3 and counts(cell, None)["witness"] == 0


def test_pairs_follow_the_nearest_step_count_for_this_run():
    # v4-only 1,225 steps per epoch, v4+world 1,677 (batch 256).
    assert step_pairs(1225, 1677, [1, 2, 3, 4, 5, 6]) == [
        (1, 1, -452),
        (2, 3, 321),
        (3, 4, -131),
        (4, 5, -583),
    ]


def test_epochs_past_the_end_of_the_shorter_arm_stay_unpaired():
    pairs = step_pairs(1225, 1677, [1, 2, 3, 4])
    assert [b for b, _, _ in pairs] == [1, 2, 3]
    assert all(2 * abs(gap) <= 1225 for _, _, gap in pairs)


def test_equal_step_counts_pair_epoch_with_itself():
    assert step_pairs(100, 100, [1, 2, 3]) == [(1, 1, 0), (2, 2, 0), (3, 3, 0)]
