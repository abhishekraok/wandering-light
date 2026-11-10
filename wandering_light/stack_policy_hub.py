"""Fetch and publish stack-policy checkpoints pinned to a Hub commit.

A stack-policy checkpoint (`wandering_light.stack_policy.save_policy`) is four
files plus an optional W&B link. A Hub repository also carries ``README.md`` and
``.gitattributes``, so this module downloads only the checkpoint's own files
into a clean directory that `load_policy` reads.

A fetch is pinned to a 40-hex commit id, never a branch: the solver a result was
measured with has to be the solver a later run gets, and a branch name moves.
Because the revision is immutable the download happens once; the directory it
lands in is named after the revision, and a later fetch that finds every
required file there returns it without touching the network.

Authentication is whatever ``huggingface_hub`` resolves -- ``HF_TOKEN`` or the
token cache ``hf auth login`` writes -- which a private repository needs.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
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
CARD_FILE = "README.md"

_COMMIT_ID = re.compile(r"[0-9a-f]{40}")


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME`` (or ``~/.cache``), then ``wandering-light/stack-policy``."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """Where one revision of one repository lives in the cache."""
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / repo_id.replace("/", "__") / revision


def is_checkpoint(path: str | Path) -> bool:
    """Every required file is present (the optional W&B link may be absent)."""
    root = Path(path)
    return root.is_dir() and all((root / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """The checkpoint at ``repo_id@revision``, downloaded on first use.

    Files are staged in a sibling directory and moved into place together, so
    an interrupted download never leaves a directory that looks like a
    checkpoint but lacks a file.
    """
    if not isinstance(revision, str) or not _COMMIT_ID.fullmatch(revision):
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target

    # Looked up at call time, so importing this module needs no network stack
    # and a test can stand in for the Hub.
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{revision}.{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        for name in REQUIRED_FILES:
            cached = huggingface_hub.hf_hub_download(repo_id, name, revision=revision)
            shutil.copyfile(cached, staging / name)
        for name in OPTIONAL_FILES:
            try:
                cached = huggingface_hub.hf_hub_download(
                    repo_id, name, revision=revision
                )
            except EntryNotFoundError:
                continue
            shutil.copyfile(cached, staging / name)
        if is_checkpoint(target):  # another process finished first
            return target
        if target.exists():  # an incomplete directory from an older layout
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target


def publish_stack_policy(
    checkpoint: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    card: str,
    commit_message: str,
    private: bool = True,
) -> str:
    """Upload a checkpoint and its model card as one Hub commit; return its id.

    The returned id is what `fetch_stack_policy` takes, and what a `System`
    pins.
    """
    root = Path(checkpoint)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} is not a stack-policy checkpoint: {missing}")

    import huggingface_hub

    api = huggingface_hub.HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    names = [
        *REQUIRED_FILES,
        *(name for name in OPTIONAL_FILES if (root / name).is_file()),
    ]
    operations = [
        huggingface_hub.CommitOperationAdd(
            path_in_repo=name, path_or_fileobj=str(root / name)
        )
        for name in names
    ]
    operations.append(
        huggingface_hub.CommitOperationAdd(
            path_in_repo=CARD_FILE, path_or_fileobj=card.encode("utf-8")
        )
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
    )
    revision = getattr(commit, "oid", None)
    if not revision:
        revision = api.model_info(repo_id).sha
    return revision
