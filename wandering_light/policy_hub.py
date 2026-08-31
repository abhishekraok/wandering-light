"""Fetch published rollout-policy checkpoints from the HuggingFace Hub.

A rollout-policy checkpoint is exactly five files, and
``assert_weights_only_checkpoint`` rejects a directory containing anything
else.  A Hub repository is not that directory: it also carries ``README.md``
and ``.gitattributes``, so a plain ``snapshot_download`` produces a tree the
loader refuses.  This module downloads only the five artifact files into a
clean directory, which ``RolloutPolicy.load`` then verifies against the
manifest's own hashes.

Downloads are pinned to an immutable revision.  ``fetch_policy_checkpoint``
resolves a branch name to the commit it pointed at and returns it, so a
debugging session records the revision it actually read rather than a name
that may move.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from wandering_light.rollout_policy import (
    CONFIG_FILE,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    TRAINING_MANIFEST_FILE,
    WANDB_LINK_FILE,
    assert_weights_only_checkpoint,
)

DEFAULT_POLICY_REPO_ID = "abhishekraok/wandering-light-rollout-policy-issue-55"

CHECKPOINT_FILES = (
    MODEL_FILE,
    CONFIG_FILE,
    TRAINING_MANIFEST_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
)


def resolve_policy_revision(
    *, repo_id: str = DEFAULT_POLICY_REPO_ID, revision: str | None = None
) -> str:
    """Return the immutable commit SHA a repo or branch name currently names."""

    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise ValueError(f"Hub returned no revision for {repo_id}")
    return info.sha


def fetch_policy_checkpoint(
    destination: str | Path,
    *,
    repo_id: str = DEFAULT_POLICY_REPO_ID,
    revision: str | None = None,
) -> tuple[Path, str]:
    """Download one checkpoint into ``destination`` and return it with its revision.

    Authentication is whatever ``huggingface_hub`` already resolves -- an
    ``HF_TOKEN`` in the environment or the token cache written by ``hf auth
    login``.  The private issue-55 checkpoint needs one of those to be present.
    """

    from huggingface_hub import hf_hub_download

    resolved = resolve_policy_revision(repo_id=repo_id, revision=revision)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_FILES:
        cached = Path(
            hf_hub_download(repo_id=repo_id, filename=name, revision=resolved)
        )
        local = target / name
        if cached.resolve() != local.resolve():
            shutil.copyfile(cached, local)

    # Raises if the download left anything the weights-only contract forbids.
    assert_weights_only_checkpoint(target)
    return target, resolved


def ensure_policy_checkpoint(
    checkpoint: str | Path | None,
    *,
    cache_dir: str | Path,
    repo_id: str = DEFAULT_POLICY_REPO_ID,
    revision: str | None = None,
) -> tuple[Path, dict[str, str]]:
    """Use a local checkpoint if given, otherwise fetch the published one.

    Returns the directory and the provenance a session records: either the
    local path, or the repo and the immutable revision that was downloaded.
    """

    if checkpoint is not None:
        path = Path(checkpoint)
        assert_weights_only_checkpoint(path)
        return path, {"source": "local", "path": str(path.resolve())}

    cache = Path(cache_dir)
    resolved = resolve_policy_revision(repo_id=repo_id, revision=revision)
    target = cache / repo_id.replace("/", "__") / resolved
    provenance = {"source": "hub", "repo_id": repo_id, "revision": resolved}
    if target.is_dir():
        try:
            assert_weights_only_checkpoint(target)
        except (ValueError, FileNotFoundError):
            shutil.rmtree(target, ignore_errors=True)
        else:
            return target, provenance
    path, _ = fetch_policy_checkpoint(target, repo_id=repo_id, revision=resolved)
    return path, provenance
