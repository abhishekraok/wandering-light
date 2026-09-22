"""Publish and fetch `mark-28` next-op policy checkpoints on the Hugging Face Hub.

A stack policy checkpoint is what `stack_policy.save_policy` writes: the
weights, the architecture, the trainer state and the training manifest, plus
a `wandb_run.url` when the run had one. `fetch_stack_policy` downloads
exactly those files into a clean directory, pinned to an immutable revision,
so the checkpoint `wandering_light.solve` runs is the one its record names
and not whatever a branch points at today. `publish_stack_policy` is the
other direction: one commit carrying the checkpoint and a model card, whose
commit id is the revision a record then pins.

The policy repos are private, like the rollout-policy ones of `policy_hub`,
so both directions need the token `hf auth login` writes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from wandering_light.stack_policy import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
)

DEFAULT_REPO_ID = "abhishekraok/wandering-light-stack-policy"
REQUIRED_FILES = (MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE, MANIFEST_FILE)
OPTIONAL_FILES = (WANDB_LINK_FILE,)
CARD_FILE = "README.md"


def default_cache_dir() -> Path:
    """Where fetched checkpoints live: outside every worktree, so one download serves them all."""
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: Path | None = None
) -> Path:
    return (cache_dir or default_cache_dir()) / repo_id.replace("/", "__") / revision


def is_checkpoint(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: Path | None = None
) -> Path:
    """Return a directory holding the checkpoint at ``revision``, downloading it once.

    ``revision`` must be a commit id: a branch name would make the same
    record name different weights on different days.
    """
    if len(revision) != 40 or set(revision) - set("0123456789abcdef"):
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    target.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        try:
            cached = Path(hf_hub_download(repo_id, name, revision=revision))
        except EntryNotFoundError:
            if name in REQUIRED_FILES:
                raise
            continue
        local = target / name
        if cached.resolve() != local.resolve():
            shutil.copyfile(cached, local)
    return target


def publish_stack_policy(
    checkpoint: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    card: str,
    commit_message: str,
    private: bool = True,
) -> str:
    """Upload one checkpoint and its model card in one commit; return the commit id."""
    from huggingface_hub import CommitOperationAdd, HfApi

    source = Path(checkpoint)
    if not is_checkpoint(source):
        raise FileNotFoundError(f"{source} is not a stack policy checkpoint")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(source / name))
        for name in REQUIRED_FILES + OPTIONAL_FILES
        if (source / name).is_file()
    ]
    operations.append(
        CommitOperationAdd(path_in_repo=CARD_FILE, path_or_fileobj=card.encode())
    )
    info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
    )
    return info.oid
