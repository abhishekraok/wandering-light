"""Load the v4 eval split and one `mark-28` policy once, and hold them warm.

The checkpoint is local by default -- the compute-optimal-mark-28 (#101)
weights on the machine that trained them -- with the W&B model artifact the
campaign archived as the portable path. Either way the session records what
it read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from wandering_light.corpus_hub import fetch_corpus
from wandering_light.stack_debugger.prior import TrainPrior
from wandering_light.stack_pbe import load
from wandering_light.stack_policy import MANIFEST_FILE, PolicyScorer, load_policy

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_MANIFEST = REPO / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
DEFAULT_SPLIT = "eval"
DEFAULT_ARM = "2m-s1"
DEFAULT_ARTIFACT = (
    f"abhishekraok-na/wandering-light-rollout-policy/mark28-policy-{DEFAULT_ARM}:v0"
)
DEFAULT_CACHE = Path.home() / ".cache" / "wandering-light" / "stack-policies"
# Where the #101 campaign left its weights, relative to the main checkout.
LOCAL_CHECKPOINTS = (
    ".claude/worktrees/standby-e5a35f/checkpoints/mark28-co-20260903/policies"
)
CHECKPOINT_ENV = "STACK_POLICY_CHECKPOINT"
DEFAULT_THREADS = 8


def _main_checkout() -> Path:
    """The repository the worktree belongs to; the campaign wrote under it."""
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return REPO
    return Path(common).parent


def resolve_checkpoint(
    checkpoint: str | Path | None,
    *,
    artifact: str | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> tuple[Path, dict[str, Any]]:
    """A local directory if one is given or known; else the archived artifact."""
    if checkpoint is None and os.environ.get(CHECKPOINT_ENV):
        checkpoint = os.environ[CHECKPOINT_ENV]
    if checkpoint is not None:
        path = Path(checkpoint)
        if not (path / MANIFEST_FILE).exists():
            raise FileNotFoundError(f"{path} is not a stack policy checkpoint")
        return path, {"source": "local", "path": str(path.resolve())}
    if artifact is None:
        local = _main_checkout() / LOCAL_CHECKPOINTS / DEFAULT_ARM
        if (local / MANIFEST_FILE).exists():
            return local, {"source": "local", "path": str(local.resolve())}
        artifact = DEFAULT_ARTIFACT
    target = cache_dir / artifact.replace("/", "__").replace(":", "__")
    if not (target / MANIFEST_FILE).exists():
        import wandb

        api = wandb.Api()
        api.artifact(artifact, type="model").download(root=str(target))
    return target, {"source": "wandb", "artifact": artifact, "path": str(target)}


@dataclass
class DebugContext:
    tasks: list[Task]
    scorer: PolicyScorer
    provenance: dict[str, Any]
    prior: TrainPrior | None = None

    @classmethod
    def load(
        cls,
        *,
        corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
        split: str = DEFAULT_SPLIT,
        checkpoint: str | Path | None = None,
        artifact: str | None = None,
        device: str = "cpu",
        threads: int = DEFAULT_THREADS,
    ) -> DebugContext:
        if device == "cpu":
            torch.set_num_threads(threads)
        # The train split is loaded only for the prior the drive table shows.
        corpus_dir = fetch_corpus(corpus_manifest, splits=[split, "train"])
        manifest = json.loads(Path(corpus_manifest).read_text())
        tasks = load(corpus_dir / manifest["splits"][split]["path"])
        prior = TrainPrior(load(corpus_dir / manifest["splits"]["train"]["path"]))
        path, policy_provenance = resolve_checkpoint(checkpoint, artifact=artifact)
        training = json.loads((path / MANIFEST_FILE).read_text())
        scorer = PolicyScorer(load_policy(path, device=device), device)
        return cls(
            tasks=tasks,
            scorer=scorer,
            prior=prior,
            provenance={
                "corpus": {
                    "name": manifest["corpus"],
                    "split": split,
                    "hub": manifest.get("hub", {}),
                    "sha256": manifest["splits"][split]["sha256"],
                    "tasks": len(tasks),
                    "train_tasks_for_prior": prior.tasks,
                },
                "policy": {
                    **policy_provenance,
                    "arm": training.get("arm"),
                    "seed": training.get("seed"),
                    "wandb_url": training.get("wandb_url"),
                    "parameters": scorer.model.parameter_count,
                },
                "device": device,
            },
        )
