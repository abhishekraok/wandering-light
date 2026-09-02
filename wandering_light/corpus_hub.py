"""Publish and fetch generated corpora from the HuggingFace Hub.

Generated corpora are derived artifacts: a change to the basis, the generator,
or a state-deduplication key invalidates them, and each regeneration is a fresh
multi-megabyte payload.  Committing them puts every superseded version in git
history permanently, because gzipped payloads do not delta-compress.

So payloads and per-root/per-record telemetry live on the Hub, while a compact
manifest stays in the repository.  The manifest records what was generated,
from which basis and digest, under which config, and the sha256 of every file --
everything needed to review corpus identity and to verify that a download is
the corpus and detail the manifest describes.

``fetch_corpus`` refuses any file whose digest disagrees with the manifest.
That check is the point of the split: without it, moving the payload off disk
would trade an artifact git guarantees for one it does not.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

MANIFEST_NAME = "manifest.json"
DEFAULT_REPO_ID = "abhishekraok/wandering-light-deep-corpus-v1"
EXPANSION_DETAIL_NAME = "expansion_detail.json"
EXPANSION_DETAIL_ARTIFACT_TYPE = "wandering-light-corpus-expansion-detail"
_HUB_KEY = "hub"
_DETAIL_FILES_KEY = "detail_files"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compact_manifest_details(corpus_dir: str | Path) -> dict[str, Any]:
    """Detach legacy per-root telemetry before a generated corpus is published.

    The corpus generator's commit remains the provenance for the rows. This
    packaging step changes only where its telemetry is stored: a compact
    manifest pins the detached file by byte count and SHA-256.
    """

    root = Path(corpus_dir)
    path = root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expansion = manifest.get("expansion")
    if not isinstance(expansion, dict):
        return manifest
    roots = expansion.get("roots")
    if roots is None:
        return manifest
    if not isinstance(roots, list):
        raise ValueError("manifest expansion.roots must be an array")
    if manifest.get(_DETAIL_FILES_KEY):
        raise ValueError("manifest mixes embedded roots with detached detail")

    detail = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": EXPANSION_DETAIL_ARTIFACT_TYPE,
        "basis_set_id": manifest.get("basis_set_id"),
        "basis_set_digest": manifest.get("basis_set_digest"),
        "generator": manifest.get("generator"),
        "seed": manifest.get("seed"),
        "roots": roots,
    }
    detail_path = root / EXPANSION_DETAIL_NAME
    _write_json(detail_path, detail)
    expansion.pop("roots")
    manifest[_DETAIL_FILES_KEY] = {
        "expansion": {
            "path": detail_path.name,
            "bytes": detail_path.stat().st_size,
            "sha256": _sha256_file(detail_path),
            "records": len(roots),
            "contents": "per-root generation and expansion telemetry",
        }
    }

    manifest.pop("manifest_digest", None)
    hub = manifest.pop(_HUB_KEY, None)
    manifest["manifest_digest"] = (
        "sha256:"
        + hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    )
    if hub is not None:
        manifest[_HUB_KEY] = hub
    _write_json(path, manifest)
    return manifest


def _safe_bare_name(name: Any, *, label: str) -> str:
    if not isinstance(name, str) or Path(name).name != name or name in ("", "."):
        raise ValueError(f"unsafe {label}: {name!r}")
    return name


def corpus_split_files(
    manifest: dict[str, Any], *, splits: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """Return selected ``(split, filename)`` pairs after validating their schema.

    A manifest is data, and a fetched one is data from the network. A ``path``
    of ``../escape`` would otherwise read and write outside the corpus
    directory, so the check belongs on every path in, not just the upload one.
    """
    split_metadata = manifest.get("splits")
    if not isinstance(split_metadata, dict):
        raise ValueError("corpus manifest 'splits' must be an object")
    selected = tuple(split_metadata) if splits is None else tuple(splits)
    if not selected:
        raise ValueError("at least one corpus split must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError(f"duplicate corpus split selection: {selected!r}")
    unknown = [split for split in selected if split not in split_metadata]
    if unknown:
        raise ValueError(f"unknown corpus splits: {unknown}")

    pairs = []
    for split in selected:
        metadata = split_metadata[split]
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata for corpus split {split!r} must be an object")
        name = _safe_bare_name(metadata.get("path"), label=f"split path for {split!r}")
        pairs.append((split, name))
    return pairs


def _detail_files(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """``(label, filename)`` pairs for detached, content-addressed telemetry."""

    details = manifest.get(_DETAIL_FILES_KEY, {})
    if not isinstance(details, dict):
        raise ValueError("manifest detail_files must be an object")
    pairs = []
    for label, metadata in details.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"detail metadata for {label!r} must be an object")
        name = _safe_bare_name(metadata.get("path"), label=f"detail path for {label!r}")
        pairs.append((label, name))
    return pairs


def _payload_names(
    manifest: dict[str, Any],
    *,
    splits: Sequence[str] | None = None,
    include_details: bool = True,
) -> list[str]:
    """The split and detail files a manifest describes, as plain names."""
    details = _detail_files(manifest) if include_details else ()
    return [
        *(name for _, name in corpus_split_files(manifest, splits=splits)),
        *(name for _, name in details),
    ]


def verify_corpus_files(
    corpus_dir: str | Path,
    *,
    splits: Sequence[str] | None = None,
    include_details: bool = True,
) -> dict[str, str]:
    """Check selected split files against the manifest; default to every split."""
    root = Path(corpus_dir)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    checked: dict[str, str] = {}
    for split, name in corpus_split_files(manifest, splits=splits):
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
    detail_files = _detail_files(manifest) if include_details else ()
    for label, name in detail_files:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing corpus detail for {label!r}: {path}")
        actual = _sha256_file(path)
        expected = manifest[_DETAIL_FILES_KEY][label].get("sha256")
        if actual != expected:
            raise ValueError(
                f"digest mismatch for {name}: manifest says {expected}, "
                f"file is {actual}"
            )
        expected_bytes = manifest[_DETAIL_FILES_KEY][label].get("bytes")
        if expected_bytes != path.stat().st_size:
            raise ValueError(
                f"size mismatch for {name}: manifest says {expected_bytes}, "
                f"file is {path.stat().st_size}"
            )
        checked[f"detail:{label}"] = actual
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
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(corpus_dir)
    compact_manifest_details(root)
    verify_corpus_files(root)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
    )
    files = [root / name for name in [*_payload_names(manifest), MANIFEST_NAME]]
    files.extend(Path(extra) for extra in extra_files)
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise ValueError("published corpus files must have unique bare names")
    operations = [
        CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=str(path))
        for path in files
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message or f"Publish {root.name}",
    )
    revision = getattr(commit, "oid", None)
    if revision is None:
        info = api.dataset_info(repo_id)
        revision = info.sha
    return revision


def verify_published_corpus(
    corpus_dir: str | Path,
    *,
    repo_id: str,
    revision: str,
    splits: Sequence[str] | None = None,
    include_details: bool = True,
) -> dict[str, str]:
    """Fetch one immutable Hub revision and verify selected payload splits."""
    from huggingface_hub import hf_hub_download

    root = Path(corpus_dir)
    local_manifest = root / MANIFEST_NAME
    cached_manifest = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=MANIFEST_NAME,
            revision=revision,
        )
    )
    if cached_manifest.read_bytes() != local_manifest.read_bytes():
        raise ValueError("published manifest differs from the local manifest")
    with tempfile.TemporaryDirectory(prefix="wandering-light-corpus-") as directory:
        target = Path(directory)
        shutil.copyfile(cached_manifest, target / MANIFEST_NAME)
        fetch_corpus(
            target / MANIFEST_NAME,
            destination=target,
            repo_id=repo_id,
            revision=revision,
            splits=splits,
            include_details=include_details,
        )
        return verify_corpus_files(
            target, splits=splits, include_details=include_details
        )


def record_hub_location(corpus_dir: str | Path, *, repo_id: str, revision: str) -> Path:
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
    splits: Sequence[str] | None = None,
    include_details: bool = True,
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
        raise ValueError(f"{manifest_file} has no hub.repo_id; pass repo_id explicitly")

    target = Path(destination) if destination else manifest_file.parent
    target.mkdir(parents=True, exist_ok=True)
    if manifest_file.resolve() != (target / MANIFEST_NAME).resolve():
        shutil.copyfile(manifest_file, target / MANIFEST_NAME)

    for name in _payload_names(
        manifest, splits=splits, include_details=include_details
    ):
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
    verify_corpus_files(target, splits=splits, include_details=include_details)
    return target
