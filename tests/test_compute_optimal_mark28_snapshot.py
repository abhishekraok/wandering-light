"""Epoch snapshots carry the training run's W&B link (world-plus-v4-policy, #134).

``archive`` reads ``manifest["wandb_url"]`` for every policy directory, so a
snapshot saved without the link would break archiving of a W&B run that
used ``--save-epochs`` (Codex review on #135).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from experiments import compute_optimal_mark28 as driver
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_policy import MANIFEST_FILE, StackPairPolicy

if TYPE_CHECKING:
    from pathlib import Path


def test_snapshot_manifest_records_the_run_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "SAVE_EPOCHS", frozenset({2}))
    monkeypatch.setattr(driver, "provenance", dict)
    url = "https://wandb.ai/abhishekraok-na/wandering-light/runs/abc123"
    save = driver.snapshotter(tmp_path, "1m", 0, 1e-3, ([], []), url)
    model = StackPairPolicy(PolicyArchitecture(8, 8, 8))
    save(1, model, {"epoch": 1})
    save(2, model, {"epoch": 2})
    assert not driver.policy_dir(tmp_path, "1m", 0, 1).exists()
    manifest = json.loads(
        (driver.policy_dir(tmp_path, "1m", 0, 2) / MANIFEST_FILE).read_text()
    )
    assert manifest["wandb_url"] == url
    assert manifest["snapshot_epoch"] == 2
