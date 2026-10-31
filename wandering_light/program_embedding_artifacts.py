"""Save self-contained experiment provenance without changing checkpoint formats."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def git_metadata(repo_root: Path) -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_training_history(path: Path, history: list[dict], metadata: dict) -> None:
    """Save loss/validation entries with the training run's independent lineage."""
    write_json(
        path,
        {
            "format_version": 1,
            "history": history,
            "provenance": {
                **metadata,
                "wandb_run.url": metadata.get("wandb_url")
                or metadata.get("wandb_run.url"),
            },
        },
    )


def load_training_history(path: Path) -> tuple[list[dict], dict | None]:
    """Read new envelopes or legacy entry lists, whose provenance is unknown."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload, None
    return payload["history"], payload.get("provenance")


def model_objectives(model, *, checkpoint: bool) -> dict:
    """Describe evaluated weights, never a future training recipe."""
    missing = "unknown" if checkpoint else "not-trained"
    return {
        "io_supervision": model.metadata.get("io_supervision", {"kind": missing}),
        "io_io_supervision": model.metadata.get("io_io_supervision", missing),
        "checkpoint_selection": model.metadata.get("checkpoint_selection", missing),
    }


def save_evaluation(
    path: Path, report: dict, metadata: dict, model, *, checkpoint: Path | None = None
) -> None:
    """Attach an independent JSON snapshot of evaluator and evaluated-model lineage.

    Run metadata describes the current code/data/arguments; checkpoint metadata
    describes the training run, possibly with different code/data/seed. Keep both
    intact. Missing legacy metadata is recorded as missing, never a load gate.
    The digest identifies the checkpoint actually evaluated, including best-step
    selection; initial_checkpoint_* in run metadata still identifies its parent.
    """
    provenance = {
        **metadata,
        "config": asdict(model.config),
        "evaluation_source": {
            key: metadata.get(key) for key in ("commit", "dirty", "source_root")
        },
        "evaluation_config": report["evaluation_config"],
        "initialization": "checkpoint" if checkpoint is not None else "random",
        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_digest": (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if checkpoint is not None
            else None
        ),
        "checkpoint_metadata": model.metadata,
        **model_objectives(model, checkpoint=checkpoint is not None),
        "wandb_run.url": metadata.get("wandb_url")
        or model.metadata.get("wandb_url")
        or model.metadata.get("wandb_run.url"),
    }
    # Later training metadata/model mutations must not rewrite an earlier report.
    report["provenance"] = json.loads(json.dumps(provenance, allow_nan=False))
    write_json(path, report)
