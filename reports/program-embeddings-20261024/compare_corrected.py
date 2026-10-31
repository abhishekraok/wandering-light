"""Compare corrected retrieval, rebuild its index, and describe held-out geometry."""

import argparse
import hashlib
import itertools
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_results import COLORS, _bars
from replay_execution_rankings import BAD_SHA, shown_queries

from wandering_light import program_embedding_eval as ev
from wandering_light.program_embedding import ProgramEmbedding, ProgramIndex
from wandering_light.program_embedding_artifacts import git_metadata, write_json
from wandering_light.program_embedding_data import load_records, manifest_for
from wandering_light.program_embedding_train import exclude_programs

PARTITIONS = ["cross_style_holdout", "official_eval", "official_eval_overlap"]
LABELS = ["Grouped holdout", "Official v4", "Eval-linked"]


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def compact_provenance(build):
    keys = (
        "commit",
        "dirty",
        "seed",
        "config",
        "index_config",
        "checkpoint_digest",
        "wandb_run.url",
    )
    return {
        **{key: build.get(key) for key in keys},
        "records_digest": build["data"]["records_digest"],
    }


def distribution(values):
    values = [x for x in values if x is not None]
    if not values:
        return {"count": 0}
    quantiles = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist()
    result = dict(
        zip(
            ("min", "p05", "p25", "median", "p75", "p95", "max"), quantiles, strict=True
        )
    )
    return {"count": len(values), "mean": float(np.mean(values)), **result}


def margin(vector, positives, negatives, vectors, key):
    positive = min(float(vector @ vectors[i]) for i in positives)
    negative = max((float(vector @ vectors[i]) for i in negatives), default=None)
    return {
        "key": key,
        "positive_min_cosine": positive,
        "negative_max_cosine": negative,
        "margin": positive - negative if negative is not None else None,
        "verified_negative_count": len(negatives),
    }


def geometry(index, records, queries):
    bank = index.programs
    ids = {p: i for i, p in enumerate(bank)}
    vectors = index.vectors.detach().cpu()
    lexical = ev.lexical_vectors(bank)
    shown = {key: examples for _, key, examples in queries}
    pairs, program_margins, io_margins = [], [], []
    grades = {}
    for record in records:
        if record.split != "test" or not record.programs or not record.examples:
            continue
        positives = {ids[p] for p in record.programs}
        distinct = [
            (a, b)
            for a, b in itertools.combinations(range(len(record.programs)), 2)
            if record.styles[a] != record.styles[b]
            and record.programs[a] != record.programs[b]
        ]
        if not distinct and record.key not in shown:
            continue
        examples = shown.get(record.key) or ev._sample_examples(
            record.examples, 5, random.Random(0)
        )
        for a, b in distinct:
            pa, pb = record.programs[a], record.programs[b]
            ta, tb = set(pa.split()), set(pb.split())
            pairs.append(
                {
                    "key": record.key,
                    "partition": ev._partition(record),
                    "programs": [ids[pa], ids[pb]],
                    "styles": [record.styles[a], record.styles[b]],
                    "cosine": float(vectors[ids[pa]] @ vectors[ids[pb]]),
                    "token_jaccard": len(ta & tb) / len(ta | tb),
                }
            )
        query_ids = {ids[record.programs[a]] for pair in distinct for a in pair}
        negatives = set()
        for query in sorted(query_ids | (positives if record.key in shown else set())):
            order = torch.argsort(
                lexical[query] @ lexical.T, descending=True, stable=True
            ).tolist()
            nearest = list(
                itertools.islice((i for i in order if i not in positives), 20)
            )
            failing = []
            for candidate in nearest:
                cache_key = (record.key, candidate)
                if cache_key not in grades:
                    grade = ev._matches(bank[candidate], examples)
                    grades[cache_key] = grade
                if grades[cache_key] is False:
                    failing.append(candidate)
            negatives.update(failing)
            if query in query_ids:
                targets = {
                    ids[record.programs[b]]
                    for a, b in itertools.permutations(range(len(record.programs)), 2)
                    if ids[record.programs[a]] == query
                    and record.styles[a] != record.styles[b]
                }
                program_margins.append(
                    {
                        "query": query,
                        **margin(vectors[query], targets, failing, vectors, record.key),
                    }
                )
        if record.key in shown:
            with torch.inference_mode():
                io = index.model.encode_io([examples]).detach().cpu()[0]
            io_margins.append(margin(io, positives, negatives, vectors, record.key))
    low = [r for r in pairs if r["token_jaccard"] <= 0.3]
    return {
        "positive_pairs": pairs,
        "program_margins": program_margins,
        "io_margins": io_margins,
        "distributions": {
            "positive_cosine": distribution([r["cosine"] for r in pairs]),
            "low_lexical_positive_cosine": distribution([r["cosine"] for r in low]),
            "program_margin": distribution([r["margin"] for r in program_margins]),
            "io_minimum_positive_cosine": distribution(
                [r["positive_min_cosine"] for r in io_margins]
            ),
            "io_margin": distribution([r["margin"] for r in io_margins]),
        },
        "negative_execution_counts": {
            "passed": list(grades.values()).count(True),
            "failed": list(grades.values()).count(False),
            "unknown": list(grades.values()).count(None),
        },
        "protocol": "Descriptive, unweighted pairs; low lexical means token-set Jaccard <=0.3. Known positives are record-linked, not exhaustive semantic classes. Negatives fail shown examples among the 20 highest token-bag-cosine nonpositive candidates per program; incomplete executions are skipped. Program margin uses minimum different-style positive cosine minus maximum verified negative cosine. I/O margin uses minimum known-positive cosine minus maximum cosine across that record's pooled verified negatives. Five shown examples: exact evaluator views for sampled I/O queries; otherwise an independent Random(0) sample per record. No hidden examples or tuning.",
    }


def plot(comparison, path, provenance):
    COLORS.update(Historical="#8b939c", Corrected="#127a85")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = ("Historical", "Corrected")
    labels = [
        f"{label}\nn={comparison['Historical'][p]['behaviours']}"
        for label, p in zip(LABELS, PARTITIONS, strict=True)
    ]
    titles = (
        "Shown execution MRR@10",
        "Any shown-passing top-10",
        "First shown-passing choice succeeds on hidden",
    )
    for ax, metric, title in zip(
        axes.flat,
        ("shown_mrr@10", "shown@10", "selected_hidden@10"),
        titles,
        strict=False,
    ):
        bounds = {
            n: [comparison[n][p]["execution"][metric + "_bounds"] for p in PARTITIONS]
            for n in names
        }
        _bars(ax, labels, {n: [b[0] for b in bounds[n]] for n in names}, bounds=bounds)
        ax.set_ylim(
            0, max(0.35, max(b[1] for rows in bounds.values() for b in rows) + 0.08)
        )
        ax.set_title(title, fontsize=11)
    program = {n: comparison[n][PARTITIONS[0]]["cross_style_program"] for n in names}
    _bars(
        axes.flat[3],
        ["Recall@1", "Recall@10"],
        {n: [program[n][f"recall@{k}"] for k in (1, 10)] for n in names},
    )
    axes.flat[3].set_title(
        "Alternate programs · 65 behaviors · equal behavior weight", fontsize=11
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in names]
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False)
    fig.text(
        0.5,
        0.02,
        "Historical vs corrected training + selection; one seed, fixed answer-containing bank.\nWhiskers bound incomplete executions. Random/lexical context remains in comparison.json.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    fig.savefig(
        path,
        dpi=180,
        metadata={
            "Description": json.dumps(compact_provenance(provenance), sort_keys=True)
        },
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root, old = args.run_root, args.historical
    paths = [
        old / "test-trained.json",
        old / "test-untrained.json",
        root / "historical-execution.json",
        root / "test-trained/evaluation.json",
    ]
    historical, untrained, replay, corrected = [
        json.loads(p.read_text()) for p in paths
    ]
    records, excluded = exclude_programs(
        load_records(
            args.repo_root,
            extra_task_paths=[
                old.parent / "data" / n
                for n in ("validated-v2.jsonl", "author-validation-v2.jsonl")
            ],
        ),
        [BAD_SHA],
    )
    data = manifest_for(records, args.repo_root)
    data.update(excluded_program_sha256=[BAD_SHA], excluded_program_variants=excluded)
    checkpoint = root / "aligned-s0/best.pt"
    torch.set_num_threads(4)
    model = ProgramEmbedding.load(checkpoint, device=args.device).eval()
    build = {
        **git_metadata(Path(__file__).resolve().parents[2]),
        "seed": 0,
        "config": asdict(model.config),
        "data": data,
        "checkpoint_metadata": model.metadata,
        "checkpoint_path": str(checkpoint),
        "checkpoint_digest": digest(checkpoint),
        "source_data_digests": {str(p): digest(p) for p in paths},
        "script_digest": digest(Path(__file__)),
        "wandb_run.url": model.metadata.get("wandb_url"),
        "index_config": {"batch_size": 32, "device": args.device, "threads": 4},
    }
    model.metadata = {**model.metadata, "index_build": build}
    bank = sorted({p for r in records for p in r.programs})
    index = ProgramIndex(model).fit(bank, batch_size=32)
    index_path = root / "library.pt"
    index.save(index_path)
    library = root / "library.jsonl"
    library.write_text(
        "".join(
            json.dumps(
                {
                    "program": p,
                    "sha256": hashlib.sha256(p.encode()).hexdigest(),
                    "provenance": compact_provenance(build),
                }
            )
            + "\n"
            for p in bank
        )
    )
    config = {"split": "test", "seed": 0, "io_examples": 5, "max_eval_records": 128}
    queries = list(shown_queries(records, config))
    geometry_result = geometry(index, records, queries)
    geometry_result["provenance"] = build
    write_json(root / "geometry.json", geometry_result)
    cpu = ProgramIndex.load(index_path, device="cpu")
    official = [shown for p, _, shown in queries if p == "official_eval"]
    cpu.search_io(official[0], k=10)
    times = []
    for shown in official:
        start = time.perf_counter()
        cpu.search_io(shown, k=10)
        times.append((time.perf_counter() - start) * 1000)
    manifest = {
        "programs": len(bank),
        "dim": model.config.dim,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_step": model.metadata.get("step"),
        "checkpoint_sha256": digest(checkpoint),
        "index_sha256": digest(index_path),
        "index_bytes": index_path.stat().st_size,
        "library_sha256": digest(library),
        "wandb_url": model.metadata.get("wandb_url"),
        "cpu_warm_io_query_ms": {
            "median": float(np.median(times)),
            "p95": float(np.quantile(times, 0.95)),
            "n": len(times),
            "threads": 4,
        },
        "provenance": build,
    }
    write_json(root / "library_manifest.json", manifest)
    comparison = {
        "Historical": historical["partitions"],
        "Corrected": corrected["partitions"],
        "Random": untrained["partitions"],
    }
    comparison["Historical"] = json.loads(json.dumps(comparison["Historical"]))
    for p in PARTITIONS:
        comparison["Historical"][p]["execution"]["shown_mrr@10_bounds"] = replay[
            "partitions"
        ][p]["shown_mrr@10_bounds"]
    comparison["Lexical"] = {
        p: {"cross_style_program": historical["partitions"][p]["lexical_cross_style"]}
        for p in PARTITIONS
    }
    caveats = [
        "Historical versus corrected training and checkpoint selection; no isolated causal claim about the loss.",
        "Fixed complete bank includes held-out answers; retrieval is not synthesis or training-library coverage.",
        "One seed; pairwise geometry is descriptive and uses record-linked known positives.",
        "The program-only stage selects by record-linked alternate-text MRR, not execution satisfaction.",
        "Training labels use per-program/example deadlines; evaluation, historical replay and geometry use a one-second deadline per candidate example set. Deadline grouping can produce different unknown outcomes.",
    ]
    old_bank = [
        json.loads(line)["program"]
        for line in (old / "library.jsonl").read_text().splitlines()
    ]
    if old_bank != bank:
        caveats.append("Current bank differs from the saved historical library.")
    if (
        historical["provenance"]["training"]["data"]["records_digest"]
        != data["records_digest"]
    ):
        caveats.append("Current records digest differs from historical data.")
    if {(r["partition"], r["key"]) for r in historical["records"]} != {
        (r["partition"], r["key"]) for r in corrected["records"]
    }:
        caveats.append("Historical and corrected I/O query memberships differ.")
    result = {
        "metrics": comparison,
        "geometry": geometry_result["distributions"],
        "library": manifest,
        "provenance": build,
        "evaluation_provenance": {
            "historical": historical.get("provenance"),
            "corrected": corrected.get("provenance"),
            "historical_replay": replay.get("provenance"),
        },
        "caveats": caveats,
    }
    write_json(root / "comparison.json", result)
    plot(comparison, root / "comparison.png", build)
    print(
        json.dumps(
            {
                "library": len(bank),
                "geometry": geometry_result["distributions"],
                "caveats": caveats,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
