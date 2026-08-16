"""Publish and fetch generated corpora from the HuggingFace Hub.

Generated corpora are derived artifacts: a change to the basis, the generator,
or a state-deduplication key invalidates them, and each regeneration is a fresh
multi-megabyte payload.  Committing them puts every superseded version in git
history permanently, because gzipped payloads do not delta-compress.

So the payload lives on the Hub and the manifest stays in the repository.  The
manifest already records what was generated, from which basis and digest, under
which config, and the sha256 of every file -- everything needed to review a
corpus, and to verify a download really is the corpus the manifest describes.

``fetch_corpus`` refuses any file whose digest disagrees with the manifest.
That check is the point of the split: without it, moving the payload off disk
would trade an artifact git guarantees for one it does not.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

MANIFEST_NAME = "manifest.json"
DEFAULT_REPO_ID = "abhishekraok/wandering-light-deep-corpus-v1"
_HUB_KEY = "hub"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _split_files(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """``(split, filename)`` pairs, rejecting anything that is not a bare name.

    A manifest is data, and a fetched one is data from the network. A ``path``
    of ``../escape`` would otherwise read and write outside the corpus
    directory, so the check belongs on every path in, not just the upload one.
    """
    pairs = []
    for split, metadata in manifest["splits"].items():
        name = metadata["path"]
        if not isinstance(name, str) or Path(name).name != name or name in ("", "."):
            raise ValueError(f"unsafe split path for {split!r}: {name!r}")
        pairs.append((split, name))
    return pairs


def _payload_names(manifest: dict[str, Any]) -> list[str]:
    """The split files a manifest describes, as plain names."""
    return [name for _, name in _split_files(manifest)]


def verify_corpus_files(corpus_dir: str | Path) -> dict[str, str]:
    """Check every split file against the manifest digest. Raises on mismatch."""
    root = Path(corpus_dir)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    checked: dict[str, str] = {}
    for split, name in _split_files(manifest):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing corpus file for {split!r}: {path}")
        actual = _sha256_file(path)
        if actual != manifest["splits"][split]["sha256"]:
            raise ValueError(
                f"digest mismatch for {name}: manifest says "
                f"{manifest['splits'][split]['sha256']}, file is {actual}"
            )
        checked[split] = actual
    return checked


def publish_corpus(
    corpus_dir: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    private: bool = False,
    extra_files: Sequence[str | Path] = (),
    commit_message: str | None = None,
) -> str:
    """Upload a corpus to the Hub and return the immutable commit revision.

    The manifest is uploaded alongside the payload so the Hub copy is
    self-describing, but the repository keeps its own copy: that is what pins
    the digests a later fetch is checked against.
    """
    from huggingface_hub import HfApi

    root = Path(corpus_dir)
    verify_corpus_files(root)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
    )
    for name in [*_payload_names(manifest), MANIFEST_NAME]:
        api.upload_file(
            path_or_fileobj=str(root / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    for extra in extra_files:
        path = Path(extra)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[],
        commit_message=commit_message or f"Publish {root.name}",
    )
    revision = getattr(commit, "oid", None)
    if revision is None:
        info = api.dataset_info(repo_id)
        revision = info.sha
    return revision


def record_hub_location(
    corpus_dir: str | Path, *, repo_id: str, revision: str
) -> Path:
    """Note where a corpus was published, inside the manifest.

    Written under a separate key rather than into the digested body, so the
    manifest digest keeps describing the corpus rather than its hosting.
    """
    root = Path(corpus_dir)
    path = root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[_HUB_KEY] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def fetch_corpus(
    manifest_path: str | Path,
    *,
    destination: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download the payload a manifest describes and verify it against the manifest.

    ``manifest_path`` is the copy in the repository. Its ``hub`` block supplies
    the repo and the pinned revision unless they are given explicitly.
    """
    from huggingface_hub import hf_hub_download

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    hub = manifest.get(_HUB_KEY, {})
    repo_id = repo_id or hub.get("repo_id")
    revision = revision or hub.get("revision")
    if not repo_id:
        raise ValueError(
            f"{manifest_file} has no hub.repo_id; pass repo_id explicitly"
        )

    target = Path(destination) if destination else manifest_file.parent
    target.mkdir(parents=True, exist_ok=True)
    if manifest_file.resolve() != (target / MANIFEST_NAME).resolve():
        shutil.copyfile(manifest_file, target / MANIFEST_NAME)

    for name in _payload_names(manifest):
        cached = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=name,
            revision=revision,
        )
        destination_path = target / name
        if Path(cached).resolve() != destination_path.resolve():
            shutil.copyfile(cached, destination_path)

    # Raises if anything downloaded is not what the manifest describes.
    verify_corpus_files(target)
    return target
