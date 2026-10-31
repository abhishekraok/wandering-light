"""Prepare the public stack PBE release offline, without executing programs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO_ID = "abhishekraok/wandering-light-stack-pbe-scale-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_metadata(
    accepted: Path, excluded_path: Path, tasks: Path, output: Path
) -> dict:
    """Join verified records to the original tasks; preserve reference source as text."""
    exclusions = json.loads(excluded_path.read_text())
    excluded = set(exclusions["excluded_hashes"])
    excluded_ids = set(exclusions["excluded_ids"])
    task_digests = {}
    with gzip.open(tasks, "rt") as stream:
        for line in stream:
            task = json.loads(line)
            key = task["behaviour"]
            if not isinstance(key, str) or not key or key in task_digests:
                raise ValueError(f"invalid or repeated task behavior: {key}")
            if key in excluded_ids or task["split"] != "train":
                raise ValueError(f"excluded or non-train task: {key}")
            task_digests[key] = hashlib.sha256(_canonical(task)).hexdigest()
    seen_ids, seen_hashes, omitted_ids, omitted_hashes, kept = (
        set(),
        set(),
        set(),
        set(),
        set(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with (
            accepted.open() as source,
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as target,
        ):
            for line in source:
                row = json.loads(line)
                key, digest = row["candidate"]["id"], row["candidate_digest"]
                if (
                    not isinstance(key, str)
                    or not key
                    or key in seen_ids
                    or not isinstance(digest, str)
                    or not digest
                    or digest in seen_hashes
                ):
                    raise ValueError(f"invalid or repeated metadata identity: {key}")
                seen_ids.add(key)
                seen_hashes.add(digest)
                if row["status"] != "accepted" or row["task"]["behaviour"] != key:
                    raise ValueError(f"wrong task association: {key}")
                if digest in excluded:
                    omitted_ids.add(key)
                    omitted_hashes.add(digest)
                    continue
                if key in excluded_ids:
                    raise ValueError(f"excluded ID has an unexpected digest: {key}")
                if (
                    task_digests.get(key)
                    != hashlib.sha256(_canonical(row["task"])).hexdigest()
                ):
                    raise ValueError(f"missing or different original task: {key}")
                kept.add(key)
                metadata = {
                    name: value
                    for name, value in row.items()
                    if name not in {"task", "merge_attempts", "source"}
                }
                target.write(_canonical(metadata) + b"\n")
        if kept != task_digests.keys():
            raise ValueError("metadata behavior IDs do not equal original task IDs")
        if omitted_hashes != excluded or omitted_ids != excluded_ids:
            raise ValueError("record exclusions do not match the saved audit")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "accepted_records": len(seen_ids),
        "metadata_records": len(kept),
        "excluded_ids": sorted(omitted_ids),
        "excluded_hashes": sorted(omitted_hashes),
        "behavior_ids_sha256": hashlib.sha256(_canonical(sorted(kept))).hexdigest(),
        "metadata_ids_equal_original_task_ids": True,
        "candidate_ids_equal_task_behaviors": True,
        "original_tasks_match_verified_records": True,
        "excluded_ids_absent": True,
    }


def prepare(run_dir: Path, output: Path, card: Path) -> dict:
    run_dir, output = run_dir.resolve(), output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("release output must be fresh or empty")
    if not output.is_relative_to(run_dir):
        raise ValueError("release output must live inside the persistent run directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def copy(source: Path, destination: str):
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[str(source.resolve())] = _sha(target)

    original = run_dir / "audit-v4-eval/tasks_train_filtered.jsonl.gz"
    expanded = run_dir / "tasks_train_16views.jsonl.gz"
    accepted = run_dir / "corpus-v3/accepted.jsonl"
    copy(original, "data/tasks_train.jsonl.gz")
    copy(expanded, "data/tasks_train_16views.jsonl.gz")
    invariants = extract_metadata(
        accepted,
        run_dir / "audit-v4-eval/exclusions.json",
        original,
        output / "metadata/behaviors.jsonl.gz",
    )
    inputs[str(accepted)] = _sha(accepted)
    copy(card, "README.md")
    copy(repo / "LICENSE", "LICENSE")
    augmentation_path = run_dir / "tasks_train_16views.jsonl.gz.manifest.json"
    copy(augmentation_path, "metadata/augmentation-manifest.json")
    augmentation = json.loads(augmentation_path.read_text())
    copy(run_dir / "corpus-v3/summary.json", "metadata/corpus-summary.json")
    copy(run_dir / "config/campaign.json", "metadata/authoring-config.json")
    for path in sorted((run_dir / "corpus-v3/provenance").glob("*.json")):
        copy(path, f"metadata/provenance/{path.name}")
    for name in ("exclusions.json", "interpretation-correction.json"):
        copy(run_dir / "audit-v4-eval" / name, f"audits/{name}")
    copy(run_dir / "provenance-correction.json", "audits/provenance-correction.json")
    for name in (
        "structural-summary.json",
        "prompt-coverage.json",
        "semantic-summary.json",
        "semantic-sensitivity.json",
        "diversity.png",
        "diversity.provenance.json",
    ):
        copy(run_dir / "diversity-audit" / name, f"audits/diversity/{name}")
    copy(
        run_dir / "diversity-audit/review/final-review-record.json",
        "audits/diversity/review.json",
    )
    sources = [
        *(
            f"experiments/pbe_scale/{name}.py"
            for name in (
                "generate",
                "generate_families",
                "campaign",
                "anchors",
                "validate",
                "merge",
                "audit_eval",
                "augment",
                "publish",
            )
        ),
        "experiments/world_behaviours/macros.txt",
        "experiments/world_behaviours/verify.py",
        "wandering_light/stack_machine.py",
        "wandering_light/stack_pbe.py",
        "pyproject.toml",
        "uv.lock",
    ]
    for name in sources:
        copy(repo / name, f"sources/{name}")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    count = invariants["metadata_records"]
    if (
        augmentation["distinct_behaviors"] != count
        or augmentation["task_views"] != count * augmentation["views_per_behavior"]
    ):
        raise ValueError("augmentation counts do not match the original behavior count")
    manifest = {
        "repo_id": REPO_ID,
        "configs": {
            "default": {
                "train": {"path": "data/tasks_train.jsonl.gz", "records": count}
            },
            "views16": {
                "train": {
                    "path": "data/tasks_train_16views.jsonl.gz",
                    "records": augmentation["task_views"],
                }
            },
        },
        "metadata_records": count,
        "canonical_invariants": invariants,
        "augmentation_hash_checks": {
            "source_matches_historical_manifest": inputs[str(original)]
            == augmentation["source_sha256"],
            "output_matches_historical_manifest": inputs[str(expanded)]
            == augmentation["output_sha256"],
            "mismatch_policy": "Recorded as caveats; historical provenance is never enforced.",
        },
        "source_snapshot_policy": "Release-time snapshots; not exact historical generation source.",
        "known_provenance_issue": "Historical git stamps, including augmentation, resolve the wrong checkout; see audits/provenance-correction.json. Original bytes are preserved.",
        "source_data_digests": inputs,
        "current_source_sha256": {name: inputs[str(repo / name)] for name in sources},
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "seed": augmentation["seed"],
        "config": {
            "validation_seed": json.loads(
                (run_dir / "corpus-v3/summary.json").read_text()
            )["provenance"]["config"]["seed"],
            "views_per_behavior": augmentation["views_per_behavior"],
            "execution_performed": False,
            "train_only": True,
        },
        "wandb_run.url": None,
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "output", "card"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    manifest = prepare(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {
                "repo_id": manifest["repo_id"],
                "metadata_records": manifest["metadata_records"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
