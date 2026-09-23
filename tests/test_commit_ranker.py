"""The learned commitment ranker's vocabulary, head and calibration (#160)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.commit_ranker as cr
from experiments.frame_gate import CommitmentModel
from experiments.inner_frames import COMBINATORS
from wandering_light.stack_policy import OP_NAMES


def test_the_vocabulary_is_the_one_the_prior_smooths_over():
    model = CommitmentModel({(("s",), ("dip", 0, 2)): 1})
    choices = cr.vocabulary(model.max_captures)
    assert len(choices) == model.vocabulary == 1 + (len(COMBINATORS) + 2) * 4
    assert len(set(choices)) == len(choices)


def test_the_head_keeps_the_ops_first_and_round_trips_the_choices():
    choices = cr.vocabulary(2)
    names = cr.actions(choices)
    assert names[: len(OP_NAMES)] == OP_NAMES

    class Stub:
        actions = names

    assert cr.choices_of(Stub()) == choices


def test_surprisal_is_the_negative_log_softmax_at_the_temperature():
    logits = np.array([[2.0, 0.0, -1.0]])
    t = 2.0
    z = np.exp(logits[0] / t)
    assert cr._nll(logits, np.array([1]), t)[0] == pytest.approx(
        -math.log(z[1] / z.sum())
    )


def test_the_temperature_fit_undoes_a_sharpened_ranker():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(4000, 5))
    labels = np.array([rng.choice(5, p=np.exp(r) / np.exp(r).sum()) for r in true])
    fit = cr.fit_temperature(true * 3.0, labels)
    assert fit["temperature"] == pytest.approx(3.0, rel=0.1)
    assert not fit["at_boundary"]


def test_a_free_correct_commitment_still_counts(monkeypatch):
    """A zero charge at a `[` is a data point, not a missing one."""

    class Task:
        def __init__(self, behaviour, program):
            self.behaviour, self.program = behaviour, program

    witnesses = [Task("a", ("[", "1", "]", "map", "[", "]", "i", "[", "]", "i"))]
    monkeypatch.setattr(cr, "eval_witnesses", lambda corpus: witnesses)
    charged = {"a": [0.0, 0, 0, 0, 0.0, 0, 0, 0.693, 0, 0]}
    costs = cr.at_opens(charged, "any")
    assert costs == [0.0, 0.0, 0.693]
    assert cr.summarise(costs)["median"] == 0.0


def test_a_ranker_temperature_at_the_edge_blocks_the_readings(tmp_path, monkeypatch):
    import json

    import experiments.frame_campaign2 as fc

    (tmp_path / "measure").mkdir()
    for seed in (0, 1, 2):
        (tmp_path / "measure" / f"ranker-s{seed}.json").write_text(
            json.dumps({"calibration": {"at_boundary": seed == 1}, "charges": {}})
        )
    ranker = fc.ranker_measures(tmp_path)
    assert sorted(ranker) == [0, 1, 2]
    flagged = [s for s, m in ranker.items() if m["calibration"]["at_boundary"]]
    assert flagged == [1]
    out = {
        "missing": {},
        "ranker_seeds": [0, 1, 2],
        "calibration_at_boundary": ["ranker-s1"],
    }
    assert fc.readings(out) == {"status": "CALIBRATION_AT_BOUNDARY"}
