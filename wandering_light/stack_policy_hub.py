"""Fetch and publish weights-only stack-policy checkpoints on the Hub."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

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


def default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    cache = Path(root) if root else Path.home() / ".cache"
    return cache / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    cache = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return cache / repo_id.replace("/", "__") / revision


def is_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    return all((root / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """Cache the checkpoint files from one immutable Hub revision."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staging = Path(temporary)
        for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
            try:
                downloaded = hf_hub_download(repo_id, name, revision=revision)
            except EntryNotFoundError:
                if name in REQUIRED_FILES:
                    raise
                continue
            shutil.copyfile(downloaded, staging / name)

        # Stage every download first, then install the optional files before
        # the cache can pass the required-file check. Each copy is atomic.
        target.mkdir(exist_ok=True)
        for name in (*OPTIONAL_FILES, *REQUIRED_FILES):
            staged = staging / name
            if staged.is_file():
                staged.replace(target / name)
    return target


def publish_stack_policy(
    checkpoint: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    card: str,
    commit_message: str,
    private: bool = True,
) -> str:
    """Publish the checkpoint and its model card in a single Hub commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(checkpoint)
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    files = (
        *REQUIRED_FILES,
        *(name for name in OPTIONAL_FILES if (root / name).is_file()),
    )
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
        for name in files
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")
        )
    )
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    return api.create_commit(
        repo_id=repo_id, operations=operations, commit_message=commit_message
    ).oid
