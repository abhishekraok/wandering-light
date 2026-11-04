"""Fetch and publish stack-policy checkpoints on the HuggingFace Hub.

A stack-policy checkpoint is the four files `stack_policy.save_policy` writes,
plus `wandb_run.url` when the run had one. A Hub repository also carries a
model card and `.gitattributes`, so instead of a `snapshot_download` this
module copies the checkpoint's own files into a clean directory keyed by the
repository and the commit it was read at.

Revisions are commit ids, never branch names. A branch moves; the number a
checkpoint produced is only reproducible if the code can say which bytes it
loaded, so `fetch_stack_policy` refuses anything but a 40-hex commit id and
caches under it. The cache is `default_cache_dir()` unless the caller says
otherwise, so a second run on the same host downloads nothing.
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

_COMMIT_ID = re.compile(r"^[0-9a-f]{40}$")


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME`` or ``~/.cache``, then ``wandering-light/stack-policy``."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "wandering-light" / "stack-policy"


def checkpoint_dir(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """Where one ``(repo_id, revision)`` lives in the cache."""
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return cache / repo_id.replace("/", "__") / revision


def is_checkpoint(path: str | Path) -> bool:
    """Every required file is present under ``path``."""
    root = Path(path)
    return root.is_dir() and all((root / name).is_file() for name in REQUIRED_FILES)


def fetch_stack_policy(
    repo_id: str, revision: str, *, cache_dir: str | Path | None = None
) -> Path:
    """Return a directory holding the checkpoint at ``repo_id@revision``.

    The directory is returned at once if it is already a checkpoint. Otherwise
    each required file, then each optional one, is fetched with
    ``huggingface_hub.hf_hub_download`` pinned to ``revision`` and copied in.
    An optional file the repository lacks is skipped; a required one raises.
    Authentication is whatever ``huggingface_hub`` resolves on its own.
    """
    if not isinstance(revision, str) or not _COMMIT_ID.fullmatch(revision):
        raise ValueError(f"revision must be a 40-hex commit id, not {revision!r}")
    target = checkpoint_dir(repo_id, revision, cache_dir=cache_dir)
    if is_checkpoint(target):
        return target

    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    target.mkdir(parents=True, exist_ok=True)
    for name, required in [(n, True) for n in REQUIRED_FILES] + [
        (n, False) for n in OPTIONAL_FILES
    ]:
        try:
            cached = Path(
                huggingface_hub.hf_hub_download(repo_id, name, revision=revision)
            )
        except EntryNotFoundError:
            if required:
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
    """Upload a checkpoint plus a ``README.md`` card in one commit; return its id.

    The commit id is what `fetch_stack_policy` pins to, so it is the thing to
    record beside any number the checkpoint produces.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(checkpoint)
    if not is_checkpoint(root):
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        raise FileNotFoundError(f"{root} is not a stack-policy checkpoint: {missing}")
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
        for name in (*REQUIRED_FILES, *OPTIONAL_FILES)
        if (root / name).is_file()
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())
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
