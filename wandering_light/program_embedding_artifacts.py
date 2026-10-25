"""Save self-contained evaluation provenance without changing checkpoint formats."""

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
        "wandb_run.url": metadata.get("wandb_url")
        or model.metadata.get("wandb_url")
        or model.metadata.get("wandb_run.url"),
    }
    # Later training metadata/model mutations must not rewrite an earlier report.
    report["provenance"] = json.loads(json.dumps(provenance, allow_nan=False))
    write_json(path, report)
