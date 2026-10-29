"""Fixed, post-hoc syntax baselines for the saved semantic-retrieval test.

No program execution, model fitting, or test-driven hyperparameter selection.
"""

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from wandering_light.program_embedding_eval import lexical_vectors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()
RUN, OUT = args.run_dir, args.out_dir
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
trained = json.loads((RUN / "test-trained.json").read_text())
random = json.loads((RUN / "test-untrained.json").read_text())
bank = [
    json.loads(line)["program"]
    for line in (RUN / "library.jsonl").read_text().splitlines()
]
assert bank == sorted(set(bank))
indices = {p: i for i, p in enumerate(bank)}
queries = trained["program_records"]
random_by_key = {row["key"]: row for row in random["program_records"]}


def features(program, mode):
    units = program if mode == "char" else program.split()
    return Counter(
        units[i : i + n] if mode == "char" else tuple(units[i : i + n])
        for n in ((3, 4, 5) if mode == "char" else (1, 2, 3))
        for i in range(len(units) - n + 1)
    )


def make_scorer(mode):
    rows = [features(program, mode) for program in bank]
    df = Counter(feature for row in rows for feature in row)
    idf = {f: math.log((1 + len(bank)) / (1 + count)) + 1 for f, count in df.items()}
    postings = defaultdict(list)
    for i, row in enumerate(rows):
        weights = {f: (1 + math.log(count)) * idf[f] for f, count in row.items()}
        norm = math.sqrt(sum(v * v for v in weights.values()))
        rows[i] = {f: w / norm for f, w in weights.items()}
        for f, w in rows[i].items():
            postings[f].append((i, w))
    postings = {
        f: (np.array([i for i, _ in pairs]), np.array([w for _, w in pairs]))
        for f, pairs in postings.items()
    }

    def score(i):
        scores = np.zeros(len(bank))
        for f, w in rows[i].items():
            ids, weights = postings[f]
            scores[ids] += w * weights
        return scores

    return score, len(df)


def ranks_for(row, score):
    ranks = []
    positives = {indices[q["program"]] for q in row["cross_style"]}
    for query in row["cross_style"]:
        other = {
            indices[q["program"]]
            for q in row["cross_style"]
            if q["style"] != query["style"]
        }
        excluded = positives - other
        order = np.argsort(-score(indices[query["program"]]), kind="stable")
        order = [i for i in order if i not in excluded]
        ranks.append(next(j + 1 for j, i in enumerate(order) if i in other))
    return ranks


lexical = lexical_vectors(bank)
lex_mismatches = []
rows = []
for row in queries:
    replay = ranks_for(row, lambda i: (lexical[i] @ lexical.T).numpy())
    expected = [q["lexical_rank"] for q in row["cross_style"]]
    if replay != expected:
        lex_mismatches.append(
            {"key": row["key"], "expected": expected, "actual": replay}
        )
    rows.append(
        {
            "key": row["key"],
            "partition": row["partition"],
            "independent_authors": {q["style"] for q in row["cross_style"]}
            == {"sol", "opus"},
            "trained": [q["rank"] for q in row["cross_style"]],
            "random": [q["rank"] for q in random_by_key[row["key"]]["cross_style"]],
            "token_bag": expected,
        }
    )
assert not lex_mismatches, lex_mismatches
dimensions = {}
for mode in ("char", "token"):
    scorer, dimensions[mode] = make_scorer(mode)
    for source, result in zip(queries, rows, strict=True):
        result[mode] = ranks_for(source, scorer)
    print(mode, "complete", flush=True)


def summarize(selected):
    metrics = {}
    for method in ("trained", "random", "token_bag", "char", "token"):
        values = np.array(
            [
                [np.mean([r <= k for r in row[method]]) for k in (1, 10)]
                + [np.mean([1 / r for r in row[method]])]
                for row in selected
            ]
        )
        metrics[method] = dict(
            zip(
                ("recall@1", "recall@10", "mrr"),
                values.mean(axis=0).tolist(),
                strict=True,
            )
        )
    intervals = {}
    rng = np.random.default_rng(0)
    sample = rng.integers(0, len(selected), size=(10000, len(selected)))
    for baseline in ("random", "token_bag", "char", "token"):
        delta = np.array(
            [
                np.mean([r <= 10 for r in row["trained"]])
                - np.mean([r <= 10 for r in row[baseline]])
                for row in selected
            ]
        )
        intervals[baseline] = {
            "mean": float(delta.mean()),
            "paired_behavior_bootstrap_95_percent": np.quantile(
                delta[sample].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    return {
        "behaviors": len(selected),
        "queries": sum(len(row["trained"]) for row in selected),
        "metrics": metrics,
        "trained_minus_baseline_recall10": intervals,
    }


primary = [row for row in rows if row["partition"] == "cross_style_holdout"]
report = {
    "protocol": "Exploratory fixed baselines after test was opened. TF-IDF with smoothed IDF, sublinear TF, L2 cosine; char 3-5 grams and whitespace-token 1-3 grams. IDF fits the candidate bank without labels. Stable index-order ties, self/same-style-equivalent exclusion identical to original. No parameter sweep. Bootstrap resamples behaviors, not training seeds.",
    "provenance": {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "seed": 0,
        "wandb_run.url": json.loads((RUN / "library_manifest.json").read_text())[
            "wandb_url"
        ],
        "data_sha256": {
            name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
            for name in ("library.jsonl", "test-trained.json", "test-untrained.json")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {
            "char_ngrams": [3, 5],
            "token_ngrams": [1, 3],
            "sublinear_tf": True,
            "smooth_idf": True,
            "bootstrap_replicates": 10000,
        },
        "embedding_run": json.loads((RUN / "library_manifest.json").read_text()),
    },
    "bank_size": len(bank),
    "features": dimensions,
    "lexical_replay_mismatches": lex_mismatches,
    "primary": summarize(primary),
    "independent_authors": summarize(
        [row for row in primary if row["independent_authors"]]
    ),
    "records": rows,
}
(OUT / "syntax-baselines.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            k: report[k]
            for k in ("bank_size", "features", "primary", "independent_authors")
        },
        indent=2,
    )
)

# The primary evaluation and its independently authored subset.
plt.switch_backend("Agg")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
methods = ["trained", "random", "token_bag", "char", "token"]
labels = ["Learned", "Untrained", "Token bag", "Char n-gram", "Token n-gram"]
colors = ["#087f8c", "#9a9a9a", "#b2b2b2", "#b2b2b2", "#b2b2b2"]
for ax, key, title in zip(
    axes,
    ["primary", "independent_authors"],
    ["All 65 held-out behaviors", "47 independently authored behaviors"],
    strict=True,
):
    values = [100 * report[key]["metrics"][m]["recall@10"] for m in methods]
    ax.bar(range(5), values, color=colors, width=0.7)
    ax.set_xticks(range(5), labels, rotation=25, ha="right")
    ax.set_ylim(0, 85)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
axes[0].set_ylabel("Equivalent program in top 10 (%)")
fig.suptitle("Keep the program retriever; PBE utility remains unmeasured", fontsize=13)
fig.text(
    0.5,
    0.015,
    "12,837 candidates, correct answers included. One training seed. Fixed post-hoc syntax baselines.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(
    OUT / "syntax-baselines.png",
    dpi=170,
    metadata={"Description": json.dumps(report["provenance"])},
)
plt.close(fig)
