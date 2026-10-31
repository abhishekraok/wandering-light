"""Regrade saved top-ten rankings; no checkpoint or encoder is needed."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from wandering_light import program_embedding_eval as ev
from wandering_light.program_embedding_artifacts import git_metadata, write_json
from wandering_light.program_embedding_data import load_records, manifest_for
from wandering_light.program_embedding_train import exclude_programs

BAD_SHA = "0f80c1f45e2d148534defa33e2b6efce5fddaa9dc2bf6837ce97e62fad537a43"


def shown_queries(records, config):
    groups = defaultdict(list)
    for record in records:
        if record.split == config["split"] and record.examples and record.programs:
            groups[ev._partition(record)].append(record)
    for partition, full_group in sorted(groups.items()):
        group = full_group
        rng = random.Random(config["seed"])
        limit = config["max_eval_records"]
        if limit and len(group) > limit:
            group = sorted(rng.sample(group, limit), key=lambda r: r.key)
        # evaluate samples every shown view before it samples any hidden views.
        inputs = [
            ev._sample_examples(r.examples, config["io_examples"], rng) for r in group
        ]
        yield from (
            (partition, r.key, shown) for r, shown in zip(group, inputs, strict=True)
        )


def replay(records, source, config):
    originals = {(r["partition"], r["key"]): r for r in source["records"]}
    rows, deviations, groups = [], [], defaultdict(list)
    for partition, key, shown in shown_queries(records, config):
        identity = {"partition": partition, "key": key}
        original = originals.pop((partition, key), None)
        if original is None:
            deviations.append({**identity, "missing_saved_ranking": True})
            continue
        grades = [ev._matches(program, shown) for program in original["top10"]]
        result = {
            **identity,
            "shown_examples": shown,
            "shown_pass": grades,
            "shown@1": grades[0],
            "shown@10": ev._known_any(grades),
            "shown_mrr@10_bounds": ev.reciprocal_rank_bounds(grades),
            "execution_timeouts": grades.count(None),
        }
        for metric in ("shown@1", "shown@10"):
            expected = original.get("execution", {}).get(metric)
            if expected != result[metric]:
                deviations.append(
                    {
                        **identity,
                        "metric": metric,
                        "historical": expected,
                        "replayed": result[metric],
                    }
                )
        rows.append(result)
        groups[partition].append(result)
    deviations.extend(
        {"partition": p, "key": k, "missing_reconstructed_query": True}
        for p, k in originals
    )
    summary = {}
    for partition, selected in groups.items():
        n = len(selected)
        summary[partition] = {
            "queries": n,
            "shown_mrr@10_bounds": [
                sum(r["shown_mrr@10_bounds"][i] for r in selected) / n for i in (0, 1)
            ],
            "execution_timeouts": sum(r["execution_timeouts"] for r in selected),
        }
        for metric in ("shown@1", "shown@10"):
            values = [r[metric] for r in selected]
            summary[partition][metric + "_bounds"] = [
                values.count(True) / n,
                (n - values.count(False)) / n,
            ]
    return {"partitions": summary, "records": rows, "validation_deviations": deviations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    historical = source.get("provenance", {})
    config = {
        "split": source["split"],
        "seed": historical.get("seed", 0),
        "io_examples": historical.get("io_examples", 5),
        "max_eval_records": historical.get("max_io_queries_per_partition", 128),
        "seconds_per_candidate_set": 1.0,
        "top_k": 10,
    }
    extras = [
        args.input.parent.parent / "data" / name
        for name in ("validated-v2.jsonl", "author-validation-v2.jsonl")
    ]
    records, excluded = exclude_programs(
        load_records(args.repo_root, extra_task_paths=extras), [BAD_SHA]
    )
    manifest = manifest_for(records, args.repo_root)
    manifest.update(
        excluded_program_sha256=[BAD_SHA], excluded_program_variants=excluded
    )
    caveats = [
        "Saved top-ten rankings, shown examples only; no weights or hidden outputs."
    ]
    old_data = historical.get("training", {}).get("data", {})
    if old_data.get("records_digest") != manifest["records_digest"]:
        caveats.append("Records digest differs from historical provenance.")
    changed = {
        p: digest
        for p, digest in old_data.get("source_data_digests", {}).items()
        if manifest["source_data_digests"].get(p) != digest
    }
    if changed:
        caveats.append(f"{len(changed)} source digests differ or are missing.")
    result = replay(records, source, config)
    if result["validation_deviations"]:
        caveats.append("Shown success or query membership differs from history.")
    result["evaluation_config"], result["caveats"] = config, caveats
    result["provenance"] = {
        **git_metadata(Path(__file__).resolve().parents[2]),
        "config": config,
        "seed": config["seed"],
        "data": manifest,
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical_provenance": historical,
        "wandb_run.url": historical.get("training", {}).get("wandb_url"),
        "changed_historical_sources": changed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result["partitions"]), flush=True)
    print(f"{len(result['validation_deviations'])} validation deviations", flush=True)


if __name__ == "__main__":
    main()
