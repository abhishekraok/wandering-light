"""Plot saved-data composition counts and input-length sensitivity."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

from experiments.pbe_scale.eval_compare import provenance

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(audit: Path, output: Path) -> None:
    paths = [
        audit / name
        for name in (
            "structural-summary.json",
            "semantic-summary.json",
            "semantic-sensitivity.json",
            "prompt-coverage.json",
        )
    ]
    structure, semantic, sensitivity, prompts = [
        json.loads(p.read_text()) for p in paths
    ]
    n = structure["kept_rows"]
    shown = prompts["groups"]["list_int"]
    long_share = (
        sum(
            int(length) * count
            for length, count in shown["shown_long_case_counts_per_task"].items()
        )
        / shown["shown_pairs"]
    )
    counts = structure["counts"]["source_class"]
    authored = sensitivity["pooled_source_groups"]["codex"]["rows"]
    values = [
        counts["pipeline"],
        counts["explicit_algorithm"],
        authored,
        counts["other_authored_or_anchor"] - authored,
    ]
    labels = [
        "Pipelines: 23 stages, 2-4 per program",
        "Compositions: 39 named algorithms",
        "Codex-authored",
        "Anchors / bootstrap",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    bars = axes[0].barh(labels, np.array(values) / n * 100, color="#0284c7")
    axes[0].bar_label(bars, labels=[f"{v:,} ({v / n:.1%})" for v in values], padding=4)
    axes[0].invert_yaxis()
    axes[0].set(
        title="Most volume comes from compositions",
        xlabel="Share of retained behaviors (%)",
        xlim=(0, 100),
    )
    kinds = ["list", "list_int", "two_lists"]
    all_inputs = [
        semantic["groups"][k]["nearest_agreement_quantiles"]["0.5"] * 100 for k in kinds
    ]
    longer = [
        sensitivity["groups"][k]["same_selected_pair_long_probe_agreement_quantiles"][
            "0.5"
        ]
        * 100
        for k in kinds
    ]
    y = np.arange(3)
    for shift, data, color, label in [
        (-0.18, all_inputs, "#64748b", "All shared inputs"),
        (0.18, longer, "#0284c7", "Same pairs; list length ≥4"),
    ]:
        bars = axes[1].barh(y + shift, data, height=0.32, color=color, label=label)
        axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in data], padding=3)
    axes[1].set(
        yticks=y,
        yticklabels=["One list", "List + integer", "Two lists"],
        title="Median agreement with closest function",
        xlabel="Equal outputs on saved test inputs (%)",
        xlim=(0, 113),
    )
    axes[1].invert_yaxis()
    axes[1].set_ylim(3.1, -0.6)
    axes[1].legend(loc="lower left", fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "68,499 tested behaviors: composition reuse and input sensitivity", fontsize=15
    )
    figure.supxlabel(
        "31.2% have a neighbor matching ≥95% of shared outputs; this is not a duplicate-function count.\n"
        "Length sensitivity keeps the original neighbor pairs. Longer-list subsets: 134 / 17 / 81 inputs; finite tests.\n"
        f"Training prompts have broader length coverage: {long_share:.0%} of shown list+integer inputs have list length ≥4.",
        fontsize=8,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)
    evidence = {
        **provenance(),
        "seed": None,
        "config": {"dpi": 170, "size_inches": [12, 5.2]},
        "inputs_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*paths, Path(__file__)]
        },
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "wandb_run.url": None,
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.audit_dir, args.output)
