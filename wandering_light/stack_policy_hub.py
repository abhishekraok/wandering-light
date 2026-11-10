"""Fetch and publish `stack_policy` checkpoints on the HuggingFace Hub.

A stack-policy checkpoint is the directory `stack_policy.save_policy` writes:
the weights, the config `load_policy` checks them against, the trainer state
and the training manifest, and a link to the W&B run when there was one. The
policies behind the reports live under `checkpoints/` on whichever host trained
them, which is not somewhere a second machine can read, so the one a command
depends on is published and fetched from here.

A fetch is pinned to a **commit id**, never a branch or a tag: a name can be
moved to different weights and a commit cannot, so the revision in the source
is the checkpoint that was measured. The cache directory is keyed on that id,
which is why a directory that already holds the files is returned without
asking the Hub anything: the same id cannot come to mean different files.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from huggingface_hub.errors import EntryNotFoundError

from wandering_light.stack_policy import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
)

DEFAULT_REPO_ID = "abhishekraok/wandering-light-stack-policy"

REQUIRED_FILES = (MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE, MANIFEST_FILE)
# `save_policy` writes the run link only when the run had one.
OPTIONAL_FILES = (WANDB_LINK_FILE,)
CARD_FILE = "README.md"

_COMMIT_ID = re.compile(r"[0-9a-f]{40}")


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME`` or ``~/.cache``, then ``wandering-light/stack-policy``."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """Where ``revision`` of ``repo_id`` is kept: one directory per commit."""
    cache = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return cache / repo_id.replace("/", "__") / revision


def is_checkpoint(path: str | Path) -> bool:
    """Every required file is there. The contents are `load_policy`'s to judge."""
    root = Path(path)
    return all((root / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """The checkpoint at ``revision``, downloaded once and then read from disk.

    Authentication is whatever ``huggingface_hub`` already resolves: an
    ``HF_TOKEN`` in the environment or the token ``hf auth login`` cached.
    """
    if not isinstance(revision, str) or _COMMIT_ID.fullmatch(revision) is None:
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target

    # Looked up here, not at import, so the call goes to whatever
    # `huggingface_hub.hf_hub_download` is when the fetch happens.
    import huggingface_hub

    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        try:
            cached = huggingface_hub.hf_hub_download(repo_id, name, revision=revision)
        except EntryNotFoundError:
            if name in REQUIRED_FILES:
                raise
            continue
        # Copied under another name and renamed, so a copy that is interrupted
        # never leaves a truncated file `is_checkpoint` would accept next time.
        target.mkdir(parents=True, exist_ok=True)
        partial = target / f"{name}.partial"
        shutil.copyfile(cached, partial)
        partial.replace(target / name)
    return target


def publish_stack_policy(
    checkpoint: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    card: str,
    commit_message: str,
    private: bool = True,
) -> str:
    """Upload a checkpoint and its model card in one commit; return the commit id.

    The id returned is what `fetch_stack_policy` takes, so it is the thing to
    write down beside whatever was measured with these weights.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(checkpoint)
    if not is_checkpoint(root):
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        raise FileNotFoundError(f"{root} is not a stack-policy checkpoint: {missing}")
    names = [
        *REQUIRED_FILES,
        *(name for name in OPTIONAL_FILES if (root / name).is_file()),
    ]
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
        for name in names
    ]
    operations.append(
        CommitOperationAdd(path_in_repo=CARD_FILE, path_or_fileobj=card.encode("utf-8"))
    )
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
    )
    revision = getattr(commit, "oid", None)
    if revision is None:
        revision = api.model_info(repo_id).sha
    return revision
