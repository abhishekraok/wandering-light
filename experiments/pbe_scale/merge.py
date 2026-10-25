"""Merge verified campaign journals without executing references or programs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.pbe_scale.validate import (
    atomic_text,
    canonical,
    digest,
    provenance,
    snapshot,
)


def merge(paths: list[Path], output_dir: Path) -> dict:
    """Keep the shortest verified witness per behavior, breaking ties by digest.

    Fingerprint namespaces use only configuration affecting shared probes.
    """
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise ValueError("at least one verified input directory is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("merge output directory must be fresh or empty")
    origins, chosen, attempts, configs = {}, {}, {}, {}
    summary_paths, source_counts, caveats = [], {}, []
    first_config = None
    for directory in paths:
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary_path.exists():
            summary_paths.append(summary_path)
        fallback = summary.get("provenance")
        for path in sorted((directory / "provenance").glob("*.json")):
            origins[path.stem] = json.loads(path.read_text())
        journal = sorted((directory / "records").glob("*.json"))
        source_counts[str(directory)] = len(journal)
        for path in journal:
            raw = path.read_bytes()
            record = json.loads(raw)
            if "provenance" in record:
                origin = record.pop("provenance")
                record["provenance_id"] = digest(origin)
                origins[record["provenance_id"]] = origin
            origin = origins.get(record.get("provenance_id"))
            if origin is None:
                if fallback is None:
                    raise ValueError(f"missing fingerprint configuration: {path}")
                origin = fallback
                key = digest(origin)
                origins[key] = origin
                record["provenance_id"] = key
                caveats.append(f"Used summary provenance for {path}; manifest missing.")
            config = origin["config"]
            if first_config is None:
                first_config = config
            probe_config = {
                key: config[key]
                for key in (
                    "seed",
                    "shared_probes",
                    "scalar_domain",
                    "list_domain",
                    "probe_policy",
                )
                if key in config
            }
            namespace = digest(probe_config)
            configs[namespace] = probe_config
            record["fingerprint_namespace"] = namespace
            key = record["candidate_digest"]
            attempt = {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": record["status"],
                "stage": record["stage"],
                "provenance_id": record["provenance_id"],
            }
            if "reason" in record:
                attempt["reason"] = record["reason"]
            attempts.setdefault(key, []).append(attempt)
            if key not in chosen or (
                record["status"] == "accepted"
                and (
                    chosen[key]["status"] != "accepted"
                    or record["task"]["length"] < chosen[key]["task"]["length"]
                )
            ):
                chosen[key] = record
    if not chosen:
        raise ValueError("input journals contain no candidate records")
    if len(configs) > 1:
        caveats.append(
            "Shared-probe configurations differ; behavioral fingerprints are compared "
            "only within matching configuration namespaces. Cross-namespace "
            "duplicates may remain; original verified task splits are preserved."
        )
    records, accepted_ids, signatures = list(chosen.values()), set(), {}
    for record in records:
        record["merge_attempts"] = attempts[record["candidate_digest"]]
    accepted = sorted(
        (record for record in records if record["status"] == "accepted"),
        key=lambda record: (record["task"]["length"], record["candidate_digest"]),
    )
    for record in accepted:
        candidate_id = record["candidate"]["id"]
        signature = (record["fingerprint_namespace"], record["signature"])
        reason = None
        if candidate_id in accepted_ids:
            reason = "candidate id already accepted with different content"
            record["stage"] = "schema"
        elif signature in signatures:
            representative = signatures[signature]
            shorter = representative["task"]["length"] < record["task"]["length"]
            reason = "behavioral duplicate on shared probes: " + (
                "shorter known verified witness"
                if shorter
                else "equal-length verified witness; candidate-digest tie-break"
            )
            record["stage"] = "dedup"
            record["duplicate_of"] = representative["candidate"]["id"]
        if reason:
            record["verified_witness_length"] = record["task"]["length"]
            record.update(status="rejected", reason=reason)
            record.pop("task", None)
        else:
            accepted_ids.add(candidate_id)
            signatures[signature] = record
    origin = provenance(
        summary_paths, first_config["seed"], first_config.get("dev_share", 0.0)
    )
    origin["config"] = dict(first_config)
    origin["config"].setdefault("dev_share", 0.0)
    origin["resume_caveats"] = list(dict.fromkeys(caveats))
    origin["merge_sources"] = source_counts
    origin["fingerprint_namespaces"] = configs
    origin_id = digest(origin)
    origins[origin_id] = origin
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance").mkdir(exist_ok=True)
    (output_dir / "records").mkdir(exist_ok=True)
    for key, manifest in origins.items():
        atomic_text(
            output_dir / "provenance" / f"{key}.json", canonical(manifest) + "\n"
        )
    for index, record in enumerate(records):
        name = f"{index:08d}-{record['candidate_digest']}.json"
        atomic_text(output_dir / "records" / name, canonical(record) + "\n")
    report = snapshot(output_dir, records, origin, origins)
    report["merge"] = {
        "source_records": sum(source_counts.values()),
        "unique_candidates": len(records),
        "source_directories": source_counts,
        "fingerprint_namespaces": len(configs),
        "attempts_by_status": dict(
            Counter(
                attempt["status"] for group in attempts.values() for attempt in group
            )
        ),
        "replay_performed": False,
        "representative_policy": "shortest verified witness, then candidate digest",
    }
    atomic_text(
        output_dir / "summary.json", json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = merge(args.inputs, args.output)
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "rejected": report["rejected"],
                "output": str(args.output),
                "caveats": report["resume_caveats"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
