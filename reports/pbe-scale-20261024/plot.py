"""Render the completed, reference-checked fixed-epoch comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

from experiments.pbe_scale.eval_compare import provenance

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(metrics_path: Path, verified_path: Path, output: Path) -> None:
    metrics = json.loads(metrics_path.read_text())
    verified = json.loads(verified_path.read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for axis, key, title, total in zip(
        axes,
        ("accepted_behaviors", "accepted_tasks"),
        ("Verified behaviors", "Tasks solved by verified programs"),
        (45, 1800),
        strict=True,
    ):
        values = [verified["counts"][label][key] for label in ("old", "mixed")]
        bars = axis.bar(
            ("Old data", "Old + new data"), values, color=("#64748b", "#0284c7")
        )
        axis.bar_label(bars, padding=4, fontsize=12)
        axis.set(title=title, ylim=(0, total * 1.1), ylabel=f"Count / {total}")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{metrics['retained_behaviors']:,} tested behaviors · "
        f"{metrics['views']['task_views']:,} PBE task views\n"
        "First fixed-epoch training comparison",
        fontsize=16,
    )
    ratio = (
        metrics["training"]["mixed-1m-s0"]["optimizer_steps"]
        / metrics["training"]["old-staged-1m-s0"]["optimizer_steps"]
    )
    figure.supxlabel(
        f"Seed 0 · epoch 3 · beam 64 x depth 48. Mixed training used {ratio:.2f}x optimizer updates.\n"
        "Verified = catalogue-equal or passes fresh sampler / perturbed / dense checks; finite tests.\n"
        "The augmented view export was not used in this first training run. Longer-problem generalization remains untested.",
        fontsize=8,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)
    evidence = {
        **provenance(),
        "seed": 0,
        "config": {"dpi": 170, "size_inches": [11, 5]},
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (metrics_path, verified_path, Path(__file__))
        },
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "wandb_run.url": {
            arm: data["wandb_run.url"] for arm, data in metrics["training"].items()
        },
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=Path(__file__).with_name("metrics.json")
    )
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.metrics, args.verified, args.output)
