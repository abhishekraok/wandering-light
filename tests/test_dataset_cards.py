"""Keep Hub dataset cards aligned with their pinned corpus manifests."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
CARD_MANIFEST_PAIRS = (
    (
        "wandering_light/evals/data/stack_pbe_v3/README.md",
        "wandering_light/evals/data/stack_pbe_v3/manifest.json",
    ),
    (
        "wandering_light/evals/data/stack_pbe_v4/README.md",
        "wandering_light/evals/data/stack_pbe_v4/manifest.json",
    ),
    (
        "reports/deep-corpus-20260816/README.md",
        "wandering_light/training/data/deep_corpus_v1/manifest.json",
    ),
    (
        "wandering_light/training/data/deep_corpus_v2/README.md",
        "wandering_light/training/data/deep_corpus_v2/manifest.json",
    ),
)


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    metadata, separator, _body = text[4:].partition("\n---\n")
    assert separator, f"{path} has unterminated YAML frontmatter"
    return yaml.safe_load(metadata)


@pytest.mark.parametrize(("card_path", "manifest_path"), CARD_MANIFEST_PAIRS)
def test_dataset_card_lists_exact_manifest_splits(card_path, manifest_path):
    card = _frontmatter(ROOT / card_path)
    manifest = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))

    assert len(card["configs"]) == 1
    config = card["configs"][0]
    assert config["config_name"] == "default"
    expected = {
        split: metadata["path"] for split, metadata in manifest["splits"].items()
    }
    items = config["data_files"]
    data_files = {item["split"]: item["path"] for item in items}
    assert len(items) == len(data_files)
    assert data_files == expected


@pytest.mark.parametrize(("card_path", "manifest_path"), CARD_MANIFEST_PAIRS)
def test_dataset_card_exposes_only_jsonl_payloads(card_path, manifest_path):
    card = _frontmatter(ROOT / card_path)
    manifest = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))

    paths = {
        item["path"] for config in card["configs"] for item in config["data_files"]
    }
    detail_paths = {
        metadata["path"] for metadata in manifest.get("detail_files", {}).values()
    }
    assert all(path.endswith(".jsonl.gz") for path in paths)
    assert "manifest.json" not in paths
    assert paths.isdisjoint(detail_paths)
