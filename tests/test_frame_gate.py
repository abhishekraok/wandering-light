"""The E2 gate's verdict is read off the measured cells mechanically."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.frame_gate as gate


def cell(defined_total, both, corpora=gate.CORPORA):
    body = {
        key: {"total": total, "ops": 10, "mean": total / 10, "median": 0.0}
        for key, total in (
            ("body_defined", defined_total),
            ("body_undefined", 5.0),
            ("top_level", 5.0),
        )
    }
    composite = {"flat": 0, "segmented": 0, "quotations": 0, "both": both}
    variant = {
        "body": body,
        "composite": composite,
        "composite_with_commitment": composite,
    }
    return {"cells": {c: {"machine_mask": variant} for c in corpora}}


def write(run_dir, arm, seed, epoch, payload):
    path = run_dir / "measure" / f"{arm}-s{seed}-e{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    ("frames_defined", "frames_both", "tags_defined", "verdict"),
    [
        (5.0, 17, 10.0, "PASS"),  # half the world arm, and half the tag control
        (5.0, 17, 5.5, "TAG"),  # cheap bodies, but the tag alone got most of it
        (5.0, 16, 10.0, "WEAK"),  # the ratio passes, the clears do not
        (9.0, 13, 9.5, "FAIL"),  # ratio 0.9 and no gain
        (7.0, 13, 9.0, "WEAK"),  # no gain in clears, but the ratio moved
    ],
)
def test_the_gate_needs_the_clears_the_ratio_and_the_frame(
    tmp_path, frames_defined, frames_both, tags_defined, verdict
):
    for seed, epoch in ((0, 5), (1, 5), (2, 5)):
        write(tmp_path, "world", seed, epoch, cell(10.0, 13))
        write(tmp_path, "tags", seed, epoch, cell(tags_defined, 13))
        write(tmp_path, "frames", seed, epoch, cell(frames_defined, frames_both))
    out = gate.report(tmp_path, lambda line: None)
    assert out["pairs"] == 3
    assert out["verdict"] == verdict
    assert out["primary"]["defined_ratio"] == frames_defined / 10.0
    assert out["primary"]["frame_over_tag_ratio"] == round(
        frames_defined / tags_defined, 4
    )


def test_a_pass_without_the_tag_control_says_so(tmp_path):
    write(tmp_path, "world", 0, 5, cell(10.0, 13))
    write(tmp_path, "frames", 0, 5, cell(5.0, 17))
    assert gate.report(tmp_path, lambda line: None)["verdict"] == (
        "PASS_WITHOUT_TAG_CONTROL"
    )


def test_an_unpaired_cell_is_not_counted(tmp_path):
    write(tmp_path, "world", 0, 5, cell(10.0, 13))
    write(tmp_path, "frames", 0, 5, cell(5.0, 17))
    write(tmp_path, "frames", 1, 5, cell(1.0, 40))
    write(tmp_path, "tags", 0, 5, cell(10.0, 13))
    out = gate.report(tmp_path, lambda line: None)
    assert out["pairs"] == 1 and out["verdict"] == "PASS"
