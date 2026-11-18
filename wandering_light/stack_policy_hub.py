"""Publish and cache weights-only stack policies at immutable Hub revisions."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

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
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / repo_id.replace("/", "__") / revision


def is_checkpoint(path: str | Path) -> bool:
    return all((Path(path) / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(target.with_suffix(".lock")):
        if is_checkpoint(target):
            return target
        # Finish all downloads before exposing any files to another reader.
        with TemporaryDirectory(dir=target.parent) as tmp:
            stage = Path(tmp)
            for name in REQUIRED_FILES + OPTIONAL_FILES:
                try:
                    source = hf_hub_download(repo_id, name, revision=revision)
                except EntryNotFoundError:
                    if name in OPTIONAL_FILES:
                        continue
                    raise
                shutil.copyfile(source, stage / name)
            target.mkdir(exist_ok=True)
            for name in OPTIONAL_FILES + REQUIRED_FILES:
                artifact = stage / name
                if artifact.is_file():
                    artifact.replace(target / name)
    return target


def publish_stack_policy(
    checkpoint: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    card: str,
    commit_message: str,
    private: bool = True,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(checkpoint)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=root / name)
        for name in REQUIRED_FILES + OPTIONAL_FILES
        if name in REQUIRED_FILES or (root / name).is_file()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")
        )
    )
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
    ).oid
