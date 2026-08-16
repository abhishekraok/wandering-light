"""Tests for publishing and fetching corpora from the Hub."""

import gzip
import hashlib
import json

import pytest

from wandering_light import corpus_hub


def _corpus(tmp_path, *, payload=b'{"a": 1}\n'):
    root = tmp_path / "corpus"
    root.mkdir()
    digests = {}
    for split in ("discovery", "validation", "test"):
        path = root / f"{split}.jsonl.gz"
        path.write_bytes(gzip.compress(payload))
        digests[split] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "basis_set_id": "wl-core-v1",
        "manifest_digest": "sha256:deadbeef",
        "splits": {
            split: {
                "path": f"{split}.jsonl.gz",
                "sha256": digests[split],
                "size": 1,
            }
            for split in ("discovery", "validation", "test")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_verify_accepts_a_matching_corpus(tmp_path):
    root = _corpus(tmp_path)

    checked = corpus_hub.verify_corpus_files(root)

    assert set(checked) == {"discovery", "validation", "test"}


def test_verify_rejects_a_modified_payload(tmp_path):
    # The whole point of keeping the manifest in git: a payload that is not the
    # one the manifest describes must not load.
    root = _corpus(tmp_path)
    (root / "validation.jsonl.gz").write_bytes(gzip.compress(b'{"a": 2}\n'))

    with pytest.raises(ValueError, match="digest mismatch"):
        corpus_hub.verify_corpus_files(root)


def test_verify_rejects_a_missing_payload(tmp_path):
    root = _corpus(tmp_path)
    (root / "test.jsonl.gz").unlink()

    with pytest.raises(FileNotFoundError):
        corpus_hub.verify_corpus_files(root)


def test_payload_names_reject_path_traversal(tmp_path):
    root = _corpus(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["splits"]["test"]["path"] = "../escape.jsonl.gz"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe split path"):
        corpus_hub.verify_corpus_files(root)


def test_record_hub_location_keeps_the_manifest_digest(tmp_path):
    # Hosting is not part of what the corpus *is*, so noting it must not change
    # the digest that describes the corpus.
    root = _corpus(tmp_path)
    before = json.loads((root / "manifest.json").read_text())["manifest_digest"]

    corpus_hub.record_hub_location(root, repo_id="acct/name", revision="abc123")

    after = json.loads((root / "manifest.json").read_text())
    assert after["manifest_digest"] == before
    assert after["hub"] == {
        "repo_id": "acct/name",
        "repo_type": "dataset",
        "revision": "abc123",
    }


def test_recorded_hub_location_does_not_break_the_digest_check(tmp_path):
    # Asserting the stored string is unchanged is not enough: the reader
    # recomputes the digest over the manifest body, so an added key breaks
    # loading even though the stored digest still matches itself. Publishing a
    # corpus must not invalidate its own manifest.
    from experiments import generate_deep_corpus as deep

    root = tmp_path / "corpus"
    root.mkdir()
    manifest = {"basis_set_id": "wl-core-v1", "splits": {}}
    body = deep._canonical_json(manifest).encode()
    manifest["manifest_digest"] = deep._sha256_bytes(body)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    corpus_hub.record_hub_location(root, repo_id="acct/name", revision="abc123")

    stored = json.loads((root / "manifest.json").read_text())
    payload = {k: v for k, v in stored.items() if k not in ("manifest_digest", "hub")}
    assert deep._sha256_bytes(deep._canonical_json(payload).encode()) == (
        stored["manifest_digest"]
    )


def test_fetch_verifies_what_it_downloaded(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    corpus_hub.record_hub_location(root, repo_id="acct/name", revision="abc123")
    # A hub serving a payload that does not match the manifest must not be
    # accepted just because the download succeeded.
    rogue = tmp_path / "rogue.jsonl.gz"
    rogue.write_bytes(gzip.compress(b'{"a": 99}\n'))

    def fake_download(*, repo_id, repo_type, filename, revision):
        assert repo_id == "acct/name"
        assert revision == "abc123"
        return str(rogue)

    monkeypatch.setattr(corpus_hub, "hf_hub_download", fake_download, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("m", (), {"hf_hub_download": staticmethod(fake_download)}),
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        corpus_hub.fetch_corpus(
            root / "manifest.json", destination=tmp_path / "out"
        )


def test_fetch_requires_a_repo_id(tmp_path):
    root = _corpus(tmp_path)

    with pytest.raises(ValueError, match=r"no hub\.repo_id"):
        corpus_hub.fetch_corpus(root / "manifest.json")
