"""Whole144 scored driver fixture, fresh dummy bytes/mock CPU policy only."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import score_gap_provenance as provenance
from experiments import score_gap_run as run
from experiments.score_gap_storage import canonical_json
from tests.test_score_gap_run import MockScorer, driver_fixture, review_fixture
from wandering_light.stack_pbe import OPS


class LayerScriptedScorer(MockScorer):
    def log_probs(self, blocks, legal):
        # Address only layer shape in this manually typed fixture. These are
        # literal float32 matrices, never trained-model inference or calibration.
        values = np.full(legal.shape, -20.0, dtype=np.float32)
        op = "mul" if legal.shape[0] == 1 else "dup"
        values[0, next(i for i, item in enumerate(OPS) if item.name == op)] = 0.0
        return np.where(legal, values, np.nan)


def test_whole144_scored_positive_support_then_null_headroom_lock(
    tmp_path, monkeypatch
):
    spec_path, repo, out, _ = driver_fixture(tmp_path, monkeypatch)
    spec = json.loads(spec_path.read_bytes())
    train = Path(spec["inputs"]["train"]["path"])
    rows = [json.loads(line) for line in train.read_bytes().splitlines()]
    for row in rows:
        row.update(inputs=["2 3", "3 4"], outputs=["12", "24"])
    train.write_bytes(b"".join(canonical_json(row) for row in rows))
    spec["inputs"]["train"]["sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    spec_path.write_bytes(canonical_json(spec))
    # driver_fixture explicitly mocks source/runtime acquisition. Update that
    # dummy source/spec pin and make another explicitly synthetic review fixture;
    # this establishes neither actual review acceptance nor runtime provenance.
    source = provenance.capture_source(repo, None)
    source["files"]["spec.json"] = {
        "bytes": spec_path.stat().st_size,
        "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }
    review_base = tmp_path / "scored-review-fixture"
    review_base.mkdir()
    review, _ = review_fixture(review_base, source)
    assert (
        run.run_diagnostic(
            spec_path, repo, out, review, loader=lambda _: LayerScriptedScorer()
        )
        == 0
    )
    result = json.loads((out / "run.json").read_bytes())
    assert result["reading"] == "NO_HEADROOM_OVER_PREFIX32"
    assert len(result["cells"]) == 144
    assert result["progress"]["cells_sealed"] == 144
    assert result["progress"]["cells_replayed"] == 144
    assert all(
        ref["result"]["program"] == ["mul", "dup", "add"] for ref in result["cells"]
    )
    assert all(ref["result"]["depth"] == 3 for ref in result["cells"])
    rows = [
        json.loads(line) for line in (out / "analysis.jsonl").read_bytes().splitlines()
    ]
    support = rows[0]
    assert support == {
        "kind": "support",
        "both_halves_pass": True,
        "support": {
            half: {str(seed): 24 for seed in range(3)}
            for half in ("selection", "confirmation")
        },
    }
    profiles = [row for row in rows if row["kind"] == "selection_profile"]
    assert len(profiles) == 24
    assert all(
        row["informative"] == 24 and row["prefix_protected"] == 24 for row in profiles
    )
    lock = json.loads((out / "selection-lock.json").read_bytes())["metadata"]
    assert lock["terminal"] == "NO_HEADROOM_OVER_PREFIX32"
    assert lock["tau"] is None
    prefix = (out / "analysis.jsonl").read_bytes()[: lock["selection_prefix_bytes"]]
    assert hashlib.sha256(prefix).hexdigest() == lock["selection_prefix_sha256"]
    assert not any(row["kind"] == "confirmation_profile" for row in rows)
    layers = [row for row in rows if row["kind"] == "retained_layer"]
    assert len(layers) == 288
    assert [row["C"] for row in layers] == [17, 64] * 144
    assert all(row["ancestor_rank"] == 1 and row["informative_cell"] for row in layers)
    assert len([row for row in rows if row["kind"] == "prefix_curve"]) == 384
    assert result["profile"]["confirmation_checks"] == {}
    assert result["independent_verification"] == "REQUIRED_AFTER_RUN_NOT_YET_PERFORMED"
