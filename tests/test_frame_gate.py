"""The E2 gate's verdict is read off the measured cells mechanically."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.frame_gate as gate


def cell(defined_total, both, top1=0.5, corpora=gate.CORPORA):
    """A measured cell as `measure_cells` writes it, at temperature 1."""
    body = {
        key: {
            "total": total,
            "calibrated_total": total,
            "ops": 10,
            "mean": total / 10,
            "calibrated_mean": total / 10,
            "top1": top1,
            "mean_rank": 2.0,
        }
        for key, total in (("defined", defined_total), ("top", 5.0))
    }
    composite = {"flat": 0, "segmented": 0, "quotations": 0, "both": both}
    variant = {
        "body": body,
        "composite": composite,
        "composite_with_commitment": composite,
    }
    return {c: {"machine_mask": variant} for c in corpora}


def write(run_dir, arm, seed, epoch, payload):
    path = run_dir / "measure" / f"{arm}-s{seed}-e{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": payload}))


@pytest.mark.parametrize(
    ("frames_defined", "frames_both", "tags_defined", "verdict"),
    [
        (5.0, 19, 10.0, "PASS"),  # half the world arm, and half the tag control
        (5.0, 19, 5.5, "TAG"),  # cheap bodies, but the tag alone got most of it
        (5.0, 18, 10.0, "WEAK"),  # the ratio passes, the clears do not
        (9.0, 12, 9.5, "FAIL_AT_TUNED_RATE_ONLY"),  # no matched arm written
        (7.0, 12, 9.0, "WEAK"),  # no gain in clears, but the ratio moved
    ],
)
def test_the_gate_needs_the_clears_the_ratio_and_the_frame(
    tmp_path, frames_defined, frames_both, tags_defined, verdict
):
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, "world", seed, epoch, cell(10.0, 12))
        write(tmp_path, "tags", seed, epoch, cell(tags_defined, 13))
        write(
            tmp_path, "frames", seed, epoch, cell(frames_defined, frames_both, top1=0.6)
        )
    out = gate.report(tmp_path, lambda line: None)
    assert out["pairs"] == 3
    assert out["verdict"] == verdict
    assert out["primary"]["defined_ratio"] == frames_defined / 10.0
    assert out["primary"]["frame_over_tag_ratio"] == round(
        frames_defined / tags_defined, 4
    )


def test_a_pass_without_the_tag_control_says_so(tmp_path):
    write(tmp_path, "world", 0, 5, cell(10.0, 12))
    write(tmp_path, "frames", 0, 5, cell(5.0, 19, top1=0.6))
    assert gate.report(tmp_path, lambda line: None)["verdict"] == (
        "PASS_WITHOUT_TAG_CONTROL"
    )


def test_an_unpaired_cell_is_not_counted(tmp_path):
    write(tmp_path, "world", 0, 5, cell(10.0, 12))
    write(tmp_path, "frames", 0, 5, cell(5.0, 19, top1=0.6))
    write(tmp_path, "frames", 1, 5, cell(1.0, 40, top1=0.6))
    write(tmp_path, "tags", 0, 5, cell(10.0, 12))
    out = gate.report(tmp_path, lambda line: None)
    assert out["pairs"] == 1 and out["verdict"] == "PASS"


def test_a_flatter_policy_without_a_rank_gain_does_not_pass(tmp_path):
    """Halving every deficit by being less confident is not ranking better."""
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, "world", seed, epoch, cell(10.0, 12, top1=0.5))
        write(tmp_path, "tags", seed, epoch, cell(10.0, 12, top1=0.5))
        write(tmp_path, "frames", seed, epoch, cell(5.0, 19, top1=0.5))
    assert gate.report(tmp_path, lambda line: None)["verdict"] == "WEAK"


def test_the_temperature_fit_undoes_a_sharpened_policy():
    """Logits twice as sharp as the labels' own distribution calibrate to T = 2."""
    from types import SimpleNamespace

    import numpy as np

    rng = np.random.default_rng(0)
    true = rng.normal(size=(4000, 6))
    probs = np.exp(true) / np.exp(true).sum(axis=1, keepdims=True)
    labels = [int(rng.choice(6, p=p)) for p in probs]
    rows = [SimpleNamespace(items=i, label=label) for i, label in enumerate(labels)]

    class Scorer:
        def logits(self, blocks):
            return 2.0 * true[np.array(blocks)]

    got = gate.fit_temperature(Scorer(), rows)
    assert got["temperature"] == pytest.approx(2.0, rel=0.08)
    assert got["calibrated_validation_ce"] < got["validation_ce"]


def test_a_commitment_costs_its_smoothed_surprise_given_the_shape():
    from collections import Counter

    counts = Counter(
        {
            ("list", ("map", 0, 0)): 8,
            ("list", ("filter", 0, 1)): 2,
            ("int", ("i", 0, 0)): 5,
        }
    )
    model = gate.CommitmentModel(counts)
    assert model.vocabulary == 4  # three choices seen, plus one unseen slot
    assert model.cost("list", ("map", 0, 0)) == pytest.approx(-math.log(9 / 14))
    # A capture count never seen with this shape is charged as unseen, not free.
    assert model.cost("list", ("filter", 0, 2)) == pytest.approx(-math.log(1 / 14))
    # An unseen shape is uniform over the vocabulary.
    assert model.cost("bool", ("map", 0, 0)) == pytest.approx(math.log(4))


def test_fail_must_hold_at_the_matched_rate_too(tmp_path):
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, "world", seed, epoch, cell(10.0, 12))
        write(tmp_path, "tags", seed, epoch, cell(9.5, 12))
        write(tmp_path, "frames", seed, epoch, cell(9.0, 12))
    # Without the matched arm a FAIL is only a failure at the tuned rate.
    assert gate.report(tmp_path, lambda line: None)["verdict"] == (
        "FAIL_AT_TUNED_RATE_ONLY"
    )
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, gate.MATCHED, seed, epoch, cell(9.2, 12))
    assert gate.report(tmp_path, lambda line: None)["verdict"] == "FAIL"
    # A matched arm that did better turns the FAIL into WEAK.
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, gate.MATCHED, seed, epoch, cell(6.0, 12))
    assert gate.report(tmp_path, lambda line: None)["verdict"] == "WEAK"


def test_measure_refuses_an_unregistered_rate(tmp_path):
    with pytest.raises(ValueError, match="no registered cell"):
        gate.measure(tmp_path, "tags", [(0, 5)], lambda line: None, 5e-4)
