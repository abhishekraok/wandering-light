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
    out = {c: {"machine_mask": variant} for c in corpora}
    out["calibration"] = {"temperature": 1.0, "at_boundary": False}
    return out


THREE = ((0, 5), (1, 5), (2, 5))


def write(run_dir, arm, seed, epoch, payload):
    path = run_dir / "measure" / f"{arm}-s{seed}-e{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": payload}))


@pytest.mark.parametrize(
    ("frames_defined", "frames_both", "tags_defined", "verdict"),
    [
        (5.0, 19, 10.0, "PASS"),  # half the world arm, and half the tag control
        (5.0, 19, 5.5, "TAG"),  # cheap bodies, but the tag alone got most of it
        (5.0, 17, 10.0, "WEAK"),  # the ratio passes, the clears do not
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
    out = gate.report(tmp_path, lambda line: None, required=THREE)
    assert out["pairs"] == 3
    assert out["verdict"] == verdict
    assert out["primary"]["defined_ratio"] == frames_defined / 10.0
    assert out["primary"]["frame_over_tag_ratio"] == round(
        frames_defined / tags_defined, 4
    )


def test_no_verdict_until_every_registered_cell_is_measured(tmp_path):
    write(tmp_path, "world", 0, 5, cell(10.0, 12))
    write(tmp_path, "frames", 0, 5, cell(5.0, 19, top1=0.6))
    write(tmp_path, "tags", 0, 5, cell(10.0, 12))
    out = gate.report(tmp_path, lambda line: None, required=THREE)
    assert out["verdict"] == "INCOMPLETE"
    assert out["missing"]["frames"] == [(1, 5), (2, 5)]
    # The default is the full registered grid: three seeds by six epochs.
    assert len(gate.REQUIRED) == 18


def test_no_verdict_on_a_temperature_at_the_edge_of_the_search(tmp_path):
    for seed, epoch in THREE:
        write(tmp_path, "world", seed, epoch, cell(10.0, 12))
        write(tmp_path, "tags", seed, epoch, cell(10.0, 12))
        edge = cell(5.0, 19, top1=0.6)
        edge["calibration"]["at_boundary"] = seed == 1
        write(tmp_path, "frames", seed, epoch, edge)
    out = gate.report(tmp_path, lambda line: None, required=THREE)
    assert out["verdict"] == "CALIBRATION_AT_BOUNDARY"
    assert out["calibration_at_boundary"] == ["frames-s1-e5"]


def test_an_unpaired_cell_is_not_counted(tmp_path):
    write(tmp_path, "world", 0, 5, cell(10.0, 12))
    write(tmp_path, "frames", 0, 5, cell(5.0, 19, top1=0.6))
    write(tmp_path, "frames", 1, 5, cell(1.0, 40, top1=0.6))
    write(tmp_path, "tags", 0, 5, cell(10.0, 12))
    out = gate.report(tmp_path, lambda line: None, required=((0, 5),))
    assert out["pairs"] == 1 and out["verdict"] == "PASS"


def test_a_flatter_policy_without_a_rank_gain_does_not_pass(tmp_path):
    """Halving every deficit by being less confident is not ranking better."""
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, "world", seed, epoch, cell(10.0, 12, top1=0.5))
        write(tmp_path, "tags", seed, epoch, cell(10.0, 12, top1=0.5))
        write(tmp_path, "frames", seed, epoch, cell(5.0, 19, top1=0.5))
    assert gate.report(tmp_path, lambda line: None, required=THREE)["verdict"] == "WEAK"


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
    # Captures seen up to 1, so K = 2: none, 5 consumers x 3 counts, ifte x 3 roles x 3.
    assert model.vocabulary == 1 + 5 * 3 + 3 * 3
    assert model.cost("list", ("map", 0, 0)) == pytest.approx(-math.log(9 / 35))
    # A capture count never seen with this shape is charged as unseen, not free.
    assert model.cost("list", ("filter", 0, 2)) == pytest.approx(-math.log(1 / 35))
    # An unseen shape is uniform over the vocabulary, which sums to one.
    assert model.cost("bool", ("map", 0, 0)) == pytest.approx(math.log(25))
    with pytest.raises(ValueError, match="vocabulary"):
        model.cost("list", ("map", 0, 3))


def test_fail_must_hold_at_the_matched_rate_too(tmp_path):
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, "world", seed, epoch, cell(10.0, 12))
        write(tmp_path, "tags", seed, epoch, cell(9.5, 12))
        write(tmp_path, "frames", seed, epoch, cell(9.0, 12))
    # Without the matched arm a FAIL is only a failure at the tuned rate.
    assert gate.report(tmp_path, lambda line: None, required=THREE)["verdict"] == (
        "FAIL_AT_TUNED_RATE_ONLY"
    )
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, gate.MATCHED, seed, epoch, cell(9.2, 12))
    assert gate.report(tmp_path, lambda line: None, required=THREE)["verdict"] == "FAIL"
    # A matched arm that did better turns the FAIL into WEAK.
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, gate.MATCHED, seed, epoch, cell(6.0, 12))
    assert gate.report(tmp_path, lambda line: None, required=THREE)["verdict"] == "WEAK"


def test_measure_refuses_an_unregistered_rate(tmp_path):
    with pytest.raises(ValueError, match="no registered cell"):
        gate.measure(tmp_path, "tags", [(0, 5)], lambda line: None, 5e-4)


def test_an_illegal_logit_does_not_move_the_temperature():
    """The fit runs under the same legal mask the deficits do."""
    from types import SimpleNamespace

    import numpy as np

    rng = np.random.default_rng(1)
    true = rng.normal(size=(3000, 5))
    probs = np.exp(true) / np.exp(true).sum(axis=1, keepdims=True)
    labels = [int(rng.choice(5, p=p)) for p in probs]
    legal = np.ones((3000, 6), dtype=bool)
    legal[:, 5] = False
    rows = [SimpleNamespace(items=i, label=label) for i, label in enumerate(labels)]

    def scorer(illegal):
        class Scorer:
            def logits(self, blocks):
                z = 1.5 * true[np.array(blocks)]
                return np.concatenate([z, np.full((len(z), 1), illegal)], axis=1)

        return Scorer()

    low = gate.fit_temperature(scorer(-50.0), rows, legal)
    high = gate.fit_temperature(scorer(50.0), rows, legal)
    assert low["temperature"] == pytest.approx(high["temperature"], rel=1e-6)
    assert low["temperature"] == pytest.approx(1.5, rel=0.1)
    assert not low["at_boundary"]
