"""Read-only views over generated corpora and their older counterparts.

Deliberately free of Streamlit: loading, summarising and filtering are the parts
worth unit-testing, and the explorer pages should hold nothing but layout.

Two record shapes are covered. Generated corpora such as ``deep_corpus_v1`` are
``BasisTaskRecord`` JSONL described by a ``manifest.json``; the older data is
``shortest_path_data`` relabel evidence, which carries both the nominal
random-walk length and the certified shortest length, so one file yields the
"before" and "after" of relabelling.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.basis_dataset import iter_basis_task_records
from wandering_light.corpus_hub import MANIFEST_NAME, fetch_corpus

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from wandering_light.basis_dataset import BasisTaskRecord

CORPUS_ROOT = Path("wandering_light/training/data")
LEGACY_EVAL_RELABEL = Path(
    "wandering_light/evals/data/random_inputs_500_shortest_v1.jsonl.gz"
)
LEGACY_TRAIN_RELABEL = Path(
    "wandering_light/training/data/induction_shortest_v1.jsonl.gz"
)


@dataclass(frozen=True)
class CorpusRef:
    """A corpus directory: a manifest in the repo, payload alongside it."""

    name: str
    directory: Path

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME


def discover_corpora(root: str | Path = CORPUS_ROOT) -> list[CorpusRef]:
    """Every directory under ``root`` that carries a corpus manifest."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    refs = [
        CorpusRef(name=path.parent.name, directory=path.parent)
        for path in root_path.glob(f"*/{MANIFEST_NAME}")
    ]
    return sorted(refs, key=lambda ref: ref.name)


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_manifest(ref: CorpusRef) -> dict[str, Any]:
    return read_manifest(ref.manifest_path)


def split_paths(ref: CorpusRef, manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        split: ref.directory / metadata["path"]
        for split, metadata in manifest["splits"].items()
    }


def missing_splits(ref: CorpusRef, manifest: dict[str, Any]) -> list[str]:
    """Splits whose payload is not on disk. The manifest is committed; the
    multi-megabyte payload is not, so a fresh clone has to fetch it."""
    return sorted(
        split for split, path in split_paths(ref, manifest).items() if not path.exists()
    )


def download_corpus(ref: CorpusRef) -> Path:
    """Fetch the payload the manifest describes, verifying every digest."""
    return fetch_corpus(ref.manifest_path)


def load_records(
    ref: CorpusRef,
    manifest: dict[str, Any],
    split: str,
    *,
    limit: int | None = None,
) -> list[BasisTaskRecord]:
    """Read one split, holding records to the manifest's basis provenance."""
    path = split_paths(ref, manifest)[split]
    records = iter_basis_task_records(
        path,
        expected_basis_set_id=manifest["basis_set_id"],
        expected_basis_set_digest=manifest["basis_set_digest"],
    )
    return list(records if limit is None else islice(records, limit))


@dataclass(frozen=True)
class DistanceProfile:
    """How a dataset's task mass is spread over shortest-path distance."""

    name: str
    counts: dict[int, int]
    note: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def max_distance(self) -> int:
        return max(self.counts, default=0)

    def share(self) -> dict[int, float]:
        total = self.total
        if total == 0:
            return {}
        return {distance: count / total for distance, count in self.counts.items()}


def corpus_distance_profile(
    manifest: dict[str, Any],
    *,
    name: str,
    splits: Sequence[str] | None = None,
) -> DistanceProfile:
    """Certified-distance histogram, read straight off the manifest."""
    selected = splits if splits is not None else list(manifest["splits"])
    counts: Counter[int] = Counter()
    for split in selected:
        for distance, count in manifest["splits"][split][
            "by_certified_distance"
        ].items():
            counts[int(distance)] += count
    return DistanceProfile(
        name=name,
        counts=dict(sorted(counts.items())),
        note=f"certified distance · splits: {', '.join(selected)}",
    )


def corpus_headline(manifest: dict[str, Any]) -> dict[str, Any]:
    """The few numbers worth putting above the fold."""
    expansion = manifest.get("expansion", {})
    splits = manifest["splits"]
    return {
        "tasks": manifest.get(
            "global_task_count", sum(s["size"] for s in splits.values())
        ),
        "splits": {split: metadata["size"] for split, metadata in splits.items()},
        "roots": len(
            {
                root
                for roots in manifest.get("split_roots", {}).values()
                for root in roots
            }
        ),
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "generator": manifest.get("generator", "—"),
        "distance_semantics": manifest.get("distance_semantics", ""),
        "reached_states": expansion.get("reached_states"),
        "wall_seconds": expansion.get("wall_seconds"),
        "certified_depth_histogram": expansion.get("certified_depth_histogram", {}),
    }


def read_relabel_records(path: str | Path) -> list[dict[str, Any]]:
    """Load ``shortest_path_data`` relabel evidence (gzip JSONL)."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def relabel_distance_profiles(
    records: Sequence[dict[str, Any]], *, name: str
) -> tuple[DistanceProfile, DistanceProfile]:
    """Certified-shortest and nominal-walk profiles for one relabelled file.

    The nominal profile is what a random-walk generator claimed; the certified
    one is what survived proof. The gap between them is the reason the forward
    generator exists.
    """
    certified: Counter[int] = Counter()
    nominal: Counter[int] = Counter()
    for record in records:
        nominal[record["original_length"]] += 1
        if record["certified"] and record["relabeled_length"] > 0:
            certified[record["relabeled_length"]] += 1
    return (
        DistanceProfile(
            name=f"{name} (certified)",
            counts=dict(sorted(certified.items())),
            note=f"{sum(certified.values())} of {len(records)} records certified non-identity",
        ),
        DistanceProfile(
            name=f"{name} (nominal walk)",
            counts=dict(sorted(nominal.items())),
            note="length of the generating random walk, before relabelling",
        ),
    )


def profile_rows(profiles: Iterable[DistanceProfile]) -> list[dict[str, Any]]:
    """Long-form rows for charting several profiles together."""
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        shares = profile.share()
        for distance, count in sorted(profile.counts.items()):
            rows.append(
                {
                    "dataset": profile.name,
                    "distance": distance,
                    "tasks": count,
                    "share": shares.get(distance, 0.0),
                }
            )
    return rows


def record_row(record: BasisTaskRecord) -> dict[str, Any]:
    """Flat summary of one task, for tables and filtering."""
    metadata = record.metadata
    return {
        "task_id": record.task_id,
        "split": record.split,
        "distance": metadata["certified_distance"],
        "certification": metadata["certification"],
        "input_type": metadata["input_type"],
        "output_type": metadata["output_type"],
        "witness": ", ".join(record.witness_function_names),
        "optimal_first": ", ".join(metadata["optimal_first_action_names"]),
        "optimal_first_complete": metadata["optimal_first_actions_complete"],
        "root_index": metadata["root_index"],
        "shell_size": metadata["distance_shell_size"],
    }


def filter_records(
    records: Sequence[BasisTaskRecord],
    *,
    distances: Sequence[int] | None = None,
    input_types: Sequence[str] | None = None,
    output_types: Sequence[str] | None = None,
    certifications: Sequence[str] | None = None,
    function_name: str | None = None,
) -> list[BasisTaskRecord]:
    """Narrow a split. ``function_name`` matches the witness path."""
    selected = list(records)
    if distances:
        wanted = set(distances)
        selected = [r for r in selected if r.metadata["certified_distance"] in wanted]
    if input_types:
        wanted_in = set(input_types)
        selected = [r for r in selected if r.metadata["input_type"] in wanted_in]
    if output_types:
        wanted_out = set(output_types)
        selected = [r for r in selected if r.metadata["output_type"] in wanted_out]
    if certifications:
        wanted_cert = set(certifications)
        selected = [r for r in selected if r.metadata["certification"] in wanted_cert]
    if function_name:
        selected = [r for r in selected if function_name in r.witness_function_names]
    return selected


@dataclass
class FunctionStats:
    """What a corpus says about one basis function."""

    name: str
    witness_uses: int = 0
    witness_tasks: int = 0
    optimal_first: int = 0
    optimal_last: int = 0
    by_distance: Counter[int] = field(default_factory=Counter)

    def row(self) -> dict[str, Any]:
        return {
            "function": self.name,
            "witness_uses": self.witness_uses,
            "witness_tasks": self.witness_tasks,
            "optimal_first": self.optimal_first,
            "optimal_last": self.optimal_last,
            "mean_distance": (
                sum(d * c for d, c in self.by_distance.items())
                / sum(self.by_distance.values())
                if self.by_distance
                else None
            ),
        }


def function_stats(
    records: Iterable[BasisTaskRecord],
    *,
    id_to_name: dict[str, str] | None = None,
) -> dict[str, FunctionStats]:
    """Per-function witness and optimal-action counts across a set of records.

    ``witness_uses`` counts every occurrence in a witness path, ``witness_tasks``
    counts distinct tasks -- a function applied twice in one witness is one task
    but two uses. Optimal *last* actions are stored by function ID, so they are
    counted only when ``id_to_name`` (a basis set's ID map) resolves them.
    """
    stats: dict[str, FunctionStats] = {}

    def entry(name: str) -> FunctionStats:
        return stats.setdefault(name, FunctionStats(name=name))

    for record in records:
        distance = record.metadata["certified_distance"]
        for name in record.witness_function_names:
            item = entry(name)
            item.witness_uses += 1
            item.by_distance[distance] += 1
        for name in set(record.witness_function_names):
            entry(name).witness_tasks += 1
        for name in record.metadata["optimal_first_action_names"]:
            entry(name).optimal_first += 1
        for function_id in record.metadata["optimal_last_action_ids"]:
            name = (id_to_name or {}).get(function_id)
            if name is not None:
                entry(name).optimal_last += 1
    return stats
