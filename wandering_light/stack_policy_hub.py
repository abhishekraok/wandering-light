"""Fetch and publish stack-policy checkpoints on the HuggingFace Hub.

A checkpoint is fetched pinned to a commit id, never a branch, and once: the
files are copied into ``<cache>/<repo>/<revision>`` and a later fetch of the
same revision reads them from there without touching the network. A branch
name would make "the best system" mean whatever the branch points at today.

Authentication is whatever ``huggingface_hub`` resolves (``HF_TOKEN`` or the
``hf auth login`` cache); a private repository needs one.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from wandering_light.stack_policy import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
)

__all__ = [
    "CONFIG_FILE",
    "DEFAULT_REPO_ID",
    "MANIFEST_FILE",
    "MODEL_FILE",
    "OPTIONAL_FILES",
    "REQUIRED_FILES",
    "TRAINER_STATE_FILE",
    "WANDB_LINK_FILE",
    "checkpoint_dir",
    "default_cache_dir",
    "fetch_stack_policy",
    "is_checkpoint",
    "publish_stack_policy",
]

DEFAULT_REPO_ID = "abhishekraok/wandering-light-stack-policy"
REQUIRED_FILES = (MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE, MANIFEST_FILE)
OPTIONAL_FILES = (WANDB_LINK_FILE,)

_COMMIT_ID = re.compile(r"[0-9a-f]{40}")


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/wandering-light/stack-policy``, else under ``~/.cache``."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / repo_id.replace("/", "__") / revision


def is_checkpoint(path: str | Path) -> bool:
    """Every required file is present (the optional W&B link may not be)."""
    root = Path(path)
    return all((root / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """The checkpoint directory for ``repo_id`` at commit ``revision``."""
    if not _COMMIT_ID.fullmatch(revision or ""):
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    target.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        try:
            cached = huggingface_hub.hf_hub_download(repo_id, name, revision=revision)
        except EntryNotFoundError:
            if name in OPTIONAL_FILES:
                continue
            raise
        local = target / name
        if Path(cached).resolve() != local.resolve():
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
    """Upload a checkpoint and its card in one Hub commit; return the commit id."""
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(checkpoint)
    if not is_checkpoint(root):
        missing = [n for n in REQUIRED_FILES if not (root / n).is_file()]
        raise FileNotFoundError(f"{root} is not a stack-policy checkpoint: {missing}")
    names = [n for n in (*REQUIRED_FILES, *OPTIONAL_FILES) if (root / n).is_file()]
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
        for name in names
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())
    )
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    commit = api.create_commit(
        repo_id=repo_id, operations=operations, commit_message=commit_message
    )
    revision = getattr(commit, "oid", None)
    if revision is None:
        revision = api.model_info(repo_id).sha
    return revision
