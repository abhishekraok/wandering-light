"""Exhaustive saved-IO agreement audit; never evaluates or generates programs."""

import argparse
import collections
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

THRESHOLDS = (0.99, 0.95, 0.90, 0.80)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retained_rows(path, excluded):
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            if row["candidate_digest"] not in excluded:
                yield row


def saved_io(row):
    task = row["task"]
    return dict(
        zip(
            task["inputs"] + task["held_out_inputs"],
            task["outputs"] + task["held_out_outputs"],
            strict=True,
        )
    )


def encode_corpus(path, excluded):
    """Recover shared probes by intersection of saved inputs, without regenerating."""
    common, shared_counts = {}, collections.defaultdict(set)
    for row in retained_rows(path, excluded):
        kind = row["candidate"]["input_kind"]
        inputs = set(saved_io(row))
        common[kind] = common.get(kind, inputs) & inputs
        shared_counts[kind].add(row["shared_count"])
    probes = {k: sorted(v) for k, v in common.items()}
    for kind, shared in probes.items():
        if shared_counts[kind] != {len(shared)}:
            raise ValueError(f"{kind}: input intersection does not match shared_count")
    vocab = {k: [{} for _ in v] for k, v in probes.items()}
    encoded, metadata = collections.defaultdict(list), collections.defaultdict(list)
    for row in retained_rows(path, excluded):
        candidate = row["candidate"]
        kind = candidate["input_kind"]
        io = saved_io(row)
        codes = []
        for probe, column in zip(probes[kind], vocab[kind], strict=True):
            value = io[probe]
            codes.append(column.setdefault(value, len(column)))
        encoded[kind].append(codes)
        metadata[kind].append(
            {
                "id": candidate["id"],
                "candidate_digest": row["candidate_digest"],
                "family": candidate["family"],
                "description": candidate["description"],
                "program": candidate["program"],
            }
        )
    matrices = {k: np.asarray(v, dtype=np.float32) for k, v in encoded.items()}
    return matrices, metadata, probes, vocab


def exact_neighbors(matrix, block_size=256):
    """Hamming mismatches for every pair; exclude self; bound live pair matrix."""
    x = torch.from_numpy(matrix)
    count, width = matrix.shape
    nearest = np.empty(count, dtype=np.int64)
    mismatches = np.empty(count, dtype=np.int64)
    edges = dict.fromkeys(THRESHOLDS, 0)
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        distance = torch.cdist(x[start:stop], x, p=0)
        distance[torch.arange(stop - start), torch.arange(start, stop)] = width + 1
        values, indices = distance.min(dim=1)
        nearest[start:stop] = indices.numpy()
        mismatches[start:stop] = values.numpy().astype(np.int64)
        for threshold in THRESHOLDS:
            # Only upper triangle: each unordered pair counted once.
            eligible = distance <= int(np.floor(width * (1 - threshold) + 1e-8))
            for offset in range(stop - start):
                edges[threshold] += int(eligible[offset, start + offset + 1 :].sum())
    return nearest, mismatches, edges


def quantiles(values):
    return {
        str(q): float(np.quantile(values, q))
        for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
    }


def audit(path, exclusions_path, output, seed=188, block_size=256, threads=8):
    start = time.monotonic()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    excluded = set(json.loads(exclusions_path.read_text())["excluded_hashes"])
    matrices, metadata, probes, vocab = encode_corpus(path, excluded)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "exhaustive same-input-kind saved shared-IO Hamming agreement",
        "thresholds": list(THRESHOLDS),
        "groups": {},
        "examples": [],
    }
    all_agreements = []
    rng = np.random.default_rng(seed)
    with (output / "semantic-neighbors.jsonl").open("w") as neighbors:
        for kind, matrix in matrices.items():
            n, width = matrix.shape
            sample = matrix[rng.integers(0, n, min(64, n))]
            direct = (sample[:, None, :] != sample[None, :, :]).sum(axis=-1)
            computed = torch.cdist(
                torch.from_numpy(sample), torch.from_numpy(sample), p=0
            ).numpy()
            np.testing.assert_array_equal(computed, direct)
            nearest, mismatches, edges = exact_neighbors(matrix, block_size)
            agreement = 1 - mismatches / width
            all_agreements.extend(agreement.tolist())
            modal, entropy, expected = [], [], []
            for col in matrix.T:
                _, counts = np.unique(col, return_counts=True)
                p = counts / n
                modal.append(float(p.max()))
                entropy.append(float(-(p * np.log2(p)).sum()))
                expected.append(float((counts * (counts - 1)).sum() / (n * (n - 1))))
            # Per-column labels cannot measure repeated output across probes: recover strings.
            inverse = [{v: k for k, v in column.items()} for column in vocab[kind]]
            row_mode = np.array(
                [
                    max(
                        collections.Counter(
                            inverse[j][int(v)] for j, v in enumerate(row)
                        ).values()
                    )
                    / width
                    for row in matrix
                ]
            )
            a = rng.integers(0, n, 10000)
            b = (a + rng.integers(1, n, 10000)) % n
            random_agreement = (matrix[a] == matrix[b]).mean(axis=1)
            informative = np.array(modal) < 0.9
            informative_nn = (
                (matrix[:, informative] == matrix[nearest][:, informative]).mean(axis=1)
                if informative.any()
                else np.full(n, np.nan)
            )
            group = {
                "rows": n,
                "shared_probes": width,
                "unordered_pairs": n * (n - 1) // 2,
                "nearest_agreement_quantiles": quantiles(agreement),
                "nearest_mismatch_quantiles": quantiles(mismatches),
                "rows_with_neighbor": {
                    str(t): int((agreement + 1e-12 >= t).sum()) for t in THRESHOLDS
                },
                "unordered_pairs_at_threshold": {str(t): edges[t] for t in THRESHOLDS},
                "random_pair_count": len(a),
                "random_pair_agreement_quantiles": quantiles(random_agreement),
                "exact_random_pair_expected_agreement": float(np.mean(expected)),
                "probe_modal_output_share_quantiles": quantiles(modal),
                "probe_output_entropy_bits_quantiles": quantiles(entropy),
                "probes_modal_share_at_least_90pct": int(
                    (np.array(modal) >= 0.9).sum()
                ),
                "row_most_common_output_share_quantiles": quantiles(row_mode),
                "nearest_agreement_on_probes_with_modal_share_below_90pct_quantiles": quantiles(
                    informative_nn
                )
                if informative.any()
                else None,
            }
            summary["groups"][kind] = group
            for i, j in enumerate(nearest):
                neighbors.write(
                    json.dumps(
                        {
                            "id": metadata[kind][i]["id"],
                            "kind": kind,
                            "neighbor_id": metadata[kind][j]["id"],
                            "agreement": float(agreement[i]),
                            "mismatches": int(mismatches[i]),
                            "probes": width,
                            "most_common_output_share": float(row_mode[i]),
                        }
                    )
                    + "\n"
                )
            # Representative different-description nearest pairs near each threshold, plus closest.
            used = set()
            for target in (1, 0.99, 0.95, 0.90, 0.80):
                candidates = np.argsort(abs(agreement - target), kind="stable")
                for i in candidates:
                    j = int(nearest[i])
                    pair = tuple(sorted((int(i), j)))
                    if (
                        pair in used
                        or metadata[kind][i]["description"]
                        == metadata[kind][j]["description"]
                    ):
                        continue
                    used.add(pair)
                    diff = np.flatnonzero(matrix[i] != matrix[j])
                    summary["examples"].append(
                        {
                            "kind": kind,
                            "agreement": float(agreement[i]),
                            "mismatches": len(diff),
                            "probes": width,
                            "a": metadata[kind][i],
                            "b": metadata[kind][j],
                            "disagreements_first_3": [
                                {
                                    "input": probes[kind][q],
                                    "a": inverse[q][int(matrix[i, q])],
                                    "b": inverse[q][int(matrix[j, q])],
                                }
                                for q in diff[:3]
                            ],
                        }
                    )
                    break
            print(json.dumps({"kind": kind, **group}), flush=True)
    summary["rows"] = len(all_agreements)
    summary["nearest_agreement_quantiles"] = quantiles(all_agreements)
    summary["rows_with_neighbor"] = {
        str(t): int((np.array(all_agreements) + 1e-12 >= t).sum()) for t in THRESHOLDS
    }
    summary["limitations"] = [
        "Raw output equality on existing shared probes approximates similarity; it does not prove useful diversity.",
        "Probe distribution is finite and includes repeated edge structures; list and scalar kinds have different coverage.",
        "High modal-output probabilities and mostly constant behaviors can inflate agreement; per-type baselines and entropy are supplied.",
        "Nearest-neighbor ties use the first row in corpus order. No generated execution, fresh probes, eval contents or model calls.",
        "Input intersection alignment is checked against saved shared_count; saved outputs are trusted as already verified.",
    ]
    summary["provenance"] = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "seed": seed,
        "config": {
            "block_size": block_size,
            "cpu_threads": threads,
            "random_pairs_per_kind": 10000,
        },
        "source_data_digests": {
            str(p): sha256(p) for p in (path, exclusions_path, Path(__file__))
        },
        "wandb_run.url": None,
        "elapsed_seconds": time.monotonic() - start,
    }
    (output / "semantic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def max_list_length(saved_input):
    lists = re.findall(r"\([^()]*\)|\bnil\b", saved_input)
    return max((len(x[1:-1].split()) if x != "nil" else 0 for x in lists), default=0)


def source_group(candidate):
    if candidate["family"] in {
        "series-pipeline",
        "baseline-pipeline",
        "paired-series-pipeline",
        "original-comparison",
        "conditional-summary",
    }:
        return "pipeline"
    if "algorithm" in candidate.get("parameters", {}) or candidate["id"].startswith(
        "pbe-family-"
    ):
        return "family"
    if candidate["id"].startswith("scale-"):
        return "codex"
    return "anchors/bootstrap/other"


def sensitivity(path, exclusions_path, output):
    """Reuse exact all-bank neighbor IDs; compare those same pairs on long-list probes."""
    excluded = set(json.loads(exclusions_path.read_text())["excluded_hashes"])
    matrices, metadata, probes, vocab = encode_corpus(path, excluded)
    source = {
        r["candidate"]["id"]: source_group(r["candidate"])
        for r in retained_rows(path, excluded)
    }
    saved = {
        r["id"]: r
        for r in map(json.loads, (output / "semantic-neighbors.jsonl").open())
    }
    result, pooled = {}, collections.defaultdict(list)
    for kind, matrix in matrices.items():
        ids = {r["id"]: i for i, r in enumerate(metadata[kind])}
        nearest = np.array([ids[saved[r["id"]]["neighbor_id"]] for r in metadata[kind]])
        lengths = np.array([max_list_length(p) for p in probes[kind]])
        eligible = lengths >= 4
        conditional = (
            (matrix[:, eligible] == matrix[nearest][:, eligible]).mean(axis=1)
            if eligible.any()
            else None
        )
        all_agreement = np.array([saved[r["id"]]["agreement"] for r in metadata[kind]])
        groups = {}
        for group in sorted({source[r["id"]] for r in metadata[kind]}):
            mask = np.array([source[r["id"]] == group for r in metadata[kind]])
            values = all_agreement[mask]
            pooled[group].extend(values.tolist())
            groups[group] = {
                "rows": len(values),
                "all_bank_nn_agreement_quantiles": quantiles(values),
                "rows_with_neighbor": {
                    str(t): int((values + 1e-12 >= t).sum()) for t in THRESHOLDS
                },
            }
        output_lengths = collections.Counter()
        for column, labels in zip(matrix.T, vocab[kind], strict=True):
            counts = np.bincount(column.astype(np.int64))
            for value, code in labels.items():
                inner = value.strip()[1:-1].strip()
                shape = (
                    f"list:{max_list_length(value)}"
                    if inner == "nil" or inner.startswith("(")
                    else "scalar"
                )
                output_lengths[shape] += int(counts[code])
        result[kind] = {
            "output_shape_length_histogram_shared_cases": dict(
                sorted(output_lengths.items())
            ),
            "rows": len(matrix),
            "shared_probes": len(lengths),
            "probe_max_list_length_histogram": dict(
                sorted(collections.Counter(map(int, lengths)).items())
            ),
            "max_list_length_at_least_4_probes": int(eligible.sum()),
            "same_selected_pair_long_probe_agreement_quantiles": quantiles(conditional)
            if conditional is not None
            else None,
            "same_selected_pair_long_probe_rows_at_threshold": {
                str(t): int((conditional + 1e-12 >= t).sum()) for t in THRESHOLDS
            }
            if conditional is not None
            else None,
            "source_groups": groups,
        }
    payload = {
        "groups": result,
        "pooled_source_groups": {
            group: {
                "rows": len(values),
                "all_bank_nn_agreement_quantiles": quantiles(values),
                "rows_with_neighbor": {
                    str(t): int((np.array(values) + 1e-12 >= t).sum())
                    for t in THRESHOLDS
                },
            }
            for group, values in pooled.items()
        },
        "limitation": "Long-list results reuse the nearest pair selected on ALL shared probes; these are not nearest neighbors recomputed on long-list probes. Source-group neighbors come from the full same-kind corpus, including other sources.",
        "provenance": {
            "seed": 188,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "config": {"min_max_input_list_length": 4, "reuse_all_bank_neighbor": True},
            "source_data_digests": {
                str(p): sha256(p)
                for p in (
                    path,
                    exclusions_path,
                    Path(__file__),
                    output / "semantic-neighbors.jsonl",
                    output / "semantic-source.py",
                    output / "semantic-summary.json",
                )
            },
            "wandb_run.url": None,
        },
    }
    (output / "semantic-sensitivity.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--sensitivity-only", action="store_true")
    args = parser.parse_args()
    if args.sensitivity_only:
        sensitivity(args.accepted, args.exclusions, args.output)
        return
    audit(args.accepted, args.exclusions, args.output, args.seed)


if __name__ == "__main__":
    main()
