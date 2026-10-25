"""Plot saved behavior-weighted test metrics and export compact ledger evidence.

    uv run python reports/program-embeddings-20261024/plot_results.py --run-dir CHECKPOINT_DIR

The main program panel uses the broad held-out partition. Tiny official-eval
alternate-text strata remain separate in metrics.json. No per-query averaging
or program execution occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

COLORS = {"Trained": "#176B87", "Random": "#9A9FA6", "Lexical": "#D2902E"}
PRIMARY = "cross_style_holdout"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(report: dict) -> dict:
    fields = (
        "behaviours",
        "full_partition_behaviours",
        "bank_programs",
        "io_to_program",
        "cross_style_program",
        "lexical_cross_style",
        "execution",
        "by_query_style",
    )
    return {
        key: {field: values[field] for field in fields if field in values}
        for key, values in report["partitions"].items()
    }


def _independent_authors(report: dict) -> dict:
    groups = [
        row["cross_style"]
        for row in report["program_records"]
        if row["partition"] == PRIMARY
        and {item["style"] for item in row["cross_style"]} == {"sol", "opus"}
    ]
    result = {"behaviors": len(groups), "queries": sum(map(len, groups))}
    for field in ("rank", "lexical_rank"):
        result[field] = {
            f"recall@{k}": sum(
                sum(item[field] <= k for item in group) / len(group) for group in groups
            )
            / len(groups)
            if groups
            else None
            for k in (1, 10)
        }
    return result


def _bars(ax, labels, series, *, counts=None, bounds=None):
    width = 0.78 / len(series)
    for index, (name, values) in enumerate(series.items()):
        centers = np.arange(len(labels)) + (index - (len(series) - 1) / 2) * width
        for column, (center, value) in enumerate(zip(centers, values, strict=True)):
            if value is None:
                continue
            ax.bar(center, value, width * 0.94, color=COLORS[name])
            top = value
            if bounds and bounds[name][column]:
                lo, hi = bounds[name][column]
                top = max(value, hi)
                ax.errorbar(
                    center,
                    value,
                    yerr=[[max(0, value - lo)], [max(0, hi - value)]],
                    fmt="none",
                    color="#242424",
                    capsize=4,
                )
            label = f"{value:.1%}"
            if counts:
                total = counts[name][column]
                label += f"\n{round(value * total)}/{total}"
            ax.text(center, top + 0.018, label, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)


def plot(run_dir: Path, out: Path) -> Path:
    paths = [
        run_dir / name
        for name in (
            "test-trained.json",
            "test-untrained.json",
            "library_manifest.json",
        )
    ]
    trained, random, manifest = [_read(path) for path in paths]
    reports = {"Trained": trained, "Random": random}
    partitions = list(trained["partitions"])
    main = trained["partitions"][PRIMARY]
    figure, (program_ax, io_ax) = plt.subplots(1, 2, figsize=(12, 5.2))
    _bars(
        program_ax,
        ["Recall@1", "Recall@10"],
        {
            **{
                name: [
                    report["partitions"][PRIMARY]["cross_style_program"][f"recall@{k}"]
                    for k in (1, 10)
                ]
                for name, report in reports.items()
            },
            "Lexical": [main["lexical_cross_style"][f"recall@{k}"] for k in (1, 10)],
        },
    )
    program_ax.set_title(
        f"Alternate programs · {main['cross_style_program']['behaviours']} behaviors"
    )
    program_ax.set_ylabel("Recall, equal weight per behavior")
    metric = "selected_hidden@10"
    values, counts, bounds = {}, {}, {}
    unknowns = 0
    for name, report in reports.items():
        executions = [
            report["partitions"][partition]["execution"] for partition in partitions
        ]
        values[name] = [execution[metric] for execution in executions]
        counts[name] = [
            execution[metric + "_scored_queries"] for execution in executions
        ]
        bounds[name] = [execution[metric + "_bounds"] for execution in executions]
        unknowns += sum(
            execution[metric + "_unknown_queries"] for execution in executions
        )
    labels = {
        PRIMARY: "Grouped holdout",
        "official_eval": "Official v4",
        "official_eval_overlap": "Eval-linked",
    }
    _bars(
        io_ax,
        [labels.get(partition, partition) for partition in partitions],
        values,
        counts=counts,
        bounds=bounds,
    )
    io_ax.set_title("I/O · first shown-passing candidate in top 10")
    io_ax.set_ylabel("Selected program passes hidden examples")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS.values()]
    figure.legend(
        handles,
        COLORS,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
    )
    figure.suptitle(
        f"Program / I/O retrieval · {manifest['programs']:,}-program bank", fontsize=15
    )
    small = [
        f"{labels.get(p, p)} {trained['partitions'][p]['cross_style_program']['behaviours']}"
        for p in partitions
        if p != PRIMARY
    ]
    figure.text(
        0.5,
        0.035,
        "Known answers are in the bank; finite-domain hidden checks; one seed. "
        f"Unknown execution outcomes: {unknowns}.\n"
        f"Tiny alternate-text strata ({'; '.join(small)} behaviors) stay separate in metrics.json.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.83))
    model = manifest["provenance"]
    summary = {
        "primary_program_partition": PRIMARY,
        "metric_weighting": "Program: mean within each behavior, then equal behavior mean. I/O: one query per behavior.",
        "execution_definition": main["execution_definition"],
        "selected_run": str(run_dir),
        "library": {
            key: value for key, value in manifest.items() if key != "provenance"
        },
        "test_trained": _compact(trained),
        "test_untrained": _compact(random),
        "validation_runs": {},
        "independent_two_author_subset": {
            "selection": "Descriptive subset: primary test behaviors with exactly Sol and Opus query styles; not a checkpoint-selection metric.",
            "trained": _independent_authors(trained),
            "untrained": _independent_authors(random),
        },
        "provenance": {
            key: model.get(key)
            for key in (
                "commit",
                "dirty",
                "seed",
                "config",
                "training",
                "data_digest",
                "train_keys_digest",
                "step",
                "validation_mrr",
            )
        },
        # Historical reports predate embedded evaluator provenance. Do not
        # mistake the library checkpoint's commit for the evaluator's commit.
        "evaluation_source": trained.get("provenance", {}).get("evaluation_source"),
        "evaluation_provenance": {
            name: report.get("provenance") for name, report in reports.items()
        },
        "source_data_digests": {str(path): _digest(path) for path in paths},
        "caveats": [
            "Held-out answers included in bank; synthesis/coverage not measured.",
            "Finite-domain equivalence; one seed; grouped skeleton/template splits.",
            "Tiny official alternate-text strata excluded from main program headline.",
        ],
    }
    for name, report in reports.items():
        if not report.get("provenance", {}).get("evaluation_source"):
            summary["caveats"].append(
                f"{name}: legacy evaluation lacks embedded evaluator provenance; "
                "evaluation source is unknown. Original artifacts are unchanged."
            )
    for name in ("joint-s0", "program-s0", "aligned-s0"):
        path = run_dir.parent / name / "validation.json"
        if path.exists():
            summary["validation_runs"][name] = {
                "source": str(path),
                "partitions": _compact(_read(path)),
            }
            summary["source_data_digests"][str(path)] = _digest(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        out,
        dpi=180,
        metadata={
            "Description": json.dumps(
                {
                    "provenance": summary["provenance"],
                    "evaluation_provenance": summary["evaluation_provenance"],
                    "source_data_digests": summary["source_data_digests"],
                    "wandb_run.url": manifest.get("wandb_url"),
                },
                sort_keys=True,
            )
        },
    )
    plt.close(figure)
    out.with_name("metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("metrics.png")
    )
    args = parser.parse_args()
    print(plot(args.run_dir, args.out))


if __name__ == "__main__":
    main()
