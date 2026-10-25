"""Training-fit telemetry uses paired rows and never publishes incomplete scores."""

import math
from types import SimpleNamespace

import pytest
import torch

from experiments.pbe_scale import diagnose_fit as diagnostic


def test_samples_are_fixed_without_replacement():
    first = diagnostic.sample_indices(100, 20, 0)
    assert first == diagnostic.sample_indices(100, 20, 0)
    assert len(set(first)) == 20
    assert first != diagnostic.sample_indices(100, 20, 1)


def test_score_weights_uneven_batches_by_rows(monkeypatch):
    monkeypatch.setattr(
        diagnostic.stack_policy,
        "_batch",
        lambda rows, device: ((torch.zeros(len(rows), 2),), torch.tensor(rows)),
    )
    result = diagnostic.score(lambda logits: logits, [0, 1, 0], [2, 1, 0], 2, math.inf)
    assert result["teacher_label_ce"] == pytest.approx(math.log(2))
    assert result["teacher_label_top1"] == pytest.approx(2 / 3)
    assert result["rows"] == 3


@pytest.fixture
def run(tmp_path, monkeypatch):
    for name in ("baseline-fit-packed32", "new-fit-packed32"):
        pool = tmp_path / name
        pool.mkdir()
        for f in ("manifest.json", "index.bin", "tokens.bin"):
            (pool / f).write_text("{}")
    for name in ("old-staged-1m-s0", "mixed-1m-s0"):
        checkpoint = tmp_path / "policies" / name / "epoch-003"
        checkpoint.mkdir(parents=True)
        for f in (
            "model.safetensors",
            "policy_config.json",
            "trainer_state.json",
            "training_manifest.json",
            "wandb_run.url",
        ):
            (checkpoint / f).write_text("{}")

    class Rows(list):
        def __init__(self, directory):
            super().__init__(range(12))
            self.manifest = {"encoding": "machine"}

    monkeypatch.setattr(diagnostic.train, "PackedRows", Rows)
    monkeypatch.setattr(
        diagnostic.stack_policy,
        "load_policy",
        lambda *a, **kw: SimpleNamespace(eval=object),
    )
    return tmp_path


def test_both_policies_receive_identical_indices(run, monkeypatch):
    calls = []

    def score(model, rows, indices, batch_size, deadline):
        calls.append(list(indices))
        return {"rows": len(indices), "teacher_label_ce": 1, "teacher_label_top1": 0}

    monkeypatch.setattr(diagnostic, "score", score)
    result = diagnostic.diagnose(run, count=10)
    assert calls[0] == calls[2] and calls[1] == calls[3]
    assert result["status"] == "complete"
    assert (run / "fit-diagnostic/metrics.json").exists()


def test_timeout_never_publishes_complete_metrics(run, monkeypatch):
    def timeout(*args):
        raise TimeoutError("interrupted diagnostic")

    monkeypatch.setattr(diagnostic, "score", timeout)
    with pytest.raises(TimeoutError):
        diagnostic.diagnose(run, count=10)
    assert (run / "fit-diagnostic/input_manifest.json").exists()
    assert not (run / "fit-diagnostic/metrics.json").exists()
