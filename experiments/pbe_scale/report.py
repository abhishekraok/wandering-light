"""Render a corpus snapshot, witness lengths, validation yield, and quota history."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

PACIFIC = ZoneInfo("America/Los_Angeles")


def report(
    corpus: Path,
    campaign: Path,
    output_dir: Path,
    audit: Path | None = None,
    provisional: bool = False,
    view_manifest: Path | None = None,
) -> dict:
    summary_path, accepted_path = corpus / "summary.json", corpus / "accepted.jsonl"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    status_path = campaign / "campaign_status.json"
    status_bytes = status_path.read_bytes() if status_path.exists() else b"{}"
    state = json.loads(status_bytes)
    audit_bytes = audit.read_bytes() if audit else b"{}"
    audit_data = json.loads(audit_bytes)
    excluded = set(audit_data.get("excluded_ids", []))
    rows, accepted_digest = [], hashlib.sha256()
    with accepted_path.open("rb") as handle:
        for line in handle:
            accepted_digest.update(line)
            record = json.loads(line)
            task = record["task"]
            rows.append(
                {
                    **{key: task[key] for key in ("split", "length", "examples")},
                    "id": record["candidate"]["id"],
                    "kind": record["candidate"]["input_kind"],
                    "checks": record["verification_count"],
                }
            )
    kept = [row for row in rows if row["id"] not in excluded]
    train = [row for row in kept if row["split"] == "train"]
    lengths = [row["length"] for row in train]
    quota_path = campaign / "quota.jsonl"
    quota_bytes = quota_path.read_bytes() if quota_path.exists() else b""
    quota = [json.loads(line) for line in quota_bytes.splitlines() if line.strip()]
    points = [
        (
            datetime.fromisoformat(row["at"]).astimezone(PACIFIC),
            row["quota"]["remaining_percent"],
        )
        for row in quota
        if row.get("quota") and row["quota"].get("remaining_percent") is not None
    ]
    deduplicated = sum(
        count
        for reason, count in summary.get("rejections_by_reason", {}).items()
        if "duplicate" in reason.lower()
    )
    origin = summary.get("provenance", {})

    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    metrics = {
        "generated_at": datetime.now(PACIFIC).isoformat(),
        "provisional": provisional,
        "corpus": str(corpus.resolve()),
        "usable_train_behaviors": len(train),
        "accepted_before_audit": len(rows),
        "accepted_after_audit": len(kept),
        "audit_excluded": len(rows) - len(kept),
        "deduplicated": deduplicated,
        "other_rejected": summary.get("rejected", 0) - deduplicated,
        "known_witness_lengths": dict(Counter(str(value) for value in lengths)),
        "estimated_full_prefix_rows": sum(lengths),
        "shown_training_examples": sum(row["examples"] for row in train),
        "verification_counts_by_kind": {
            kind: dict(
                Counter(str(row["checks"]) for row in kept if row["kind"] == kind)
            )
            for kind in sorted({row["kind"] for row in kept})
        },
        "verification_policy": {
            "list_kinds": 208,
            "int": 21,
            "two_ints": 441,
            "shown_per_task": 6,
        },
        "author_campaign": {
            key: state.get(key)
            for key in ("status", "candidate_rows", "quota", "updated_at")
        },
        "composition_components": {"explicit_new_algorithms": 39, "typed_stages": 23},
        "composition_note": "Generated compositions are not independently authored algorithms.",
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "seed": origin.get("config", {}).get("seed"),
        "wandb_run.url": None,
        "corpus_hashes": {
            "accepted.jsonl": accepted_digest.hexdigest(),
            "summary.json": hashlib.sha256(summary_bytes).hexdigest(),
        },
        "input_hashes": origin.get("source_data_digests", {}),
        "additional_input_hashes": {
            str(status_path): hashlib.sha256(status_bytes).hexdigest(),
            str(quota_path): hashlib.sha256(quota_bytes).hexdigest(),
            **({str(audit): hashlib.sha256(audit_bytes).hexdigest()} if audit else {}),
        },
        "caveats": [
            "Behavior fingerprints approximate equivalence on shared probes.",
            "Known witness length is neither certified minimal length nor a difficulty measure.",
        ],
    }
    if view_manifest:
        view_bytes = view_manifest.read_bytes()
        metrics["augmentation"] = json.loads(view_bytes)
        metrics["additional_input_hashes"][str(view_manifest)] = hashlib.sha256(
            view_bytes
        ).hexdigest()
    panels = 3 if points else 2
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1, panels, figsize=(5 * panels, 4.6), constrained_layout=True
    )
    fig.suptitle(
        ("PROVISIONAL · " if provisional else "")
        + f"{len(train):,} usable training behaviors",
        fontsize=20,
        fontweight="bold",
    )
    axes[0].hist(lengths, bins=30, color="#315b96", edgecolor="white", linewidth=0.4)
    axes[0].set(
        xlabel="Known witness primitive operations",
        ylabel="Training behaviors",
        title="Verified witness lengths",
    )
    values = [
        len(kept),
        deduplicated,
        metrics["other_rejected"],
        metrics["audit_excluded"],
    ]
    axes[1].barh(
        ["Accepted", "Deduplicated", "Other rejected", "Audit excluded"],
        values,
        color=["#287d70", "#8a99ae", "#be8067", "#955a7a"],
    )
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Candidate behaviors", title="Validation and audit yield")
    for index, value in enumerate(values):
        axes[1].text(value, index, f" {value:,}", va="center", fontsize=9)
    axes[1].set_xlim(0, max(values, default=1) * 1.2 or 1)
    if points:
        axes[2].plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#9b6c25",
        )
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=PACIFIC))
        axes[2].set(
            xlabel="Pacific time",
            ylabel="Weekly quota remaining (%)",
            title="Authoring quota",
            ylim=(0, 100),
        )
        axes[2].grid(alpha=0.2)
    footer = (
        f"Source: {corpus.name} · Author campaign: {state.get('status', 'unknown')} · "
        f"Raw authored rows: {state.get('candidate_rows', 0):,} · 39 explicit algorithms + 23 typed stages\n"
        "Witness length is not certified minimal length or task difficulty."
    )
    fig.supxlabel(footer, fontsize=9)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "corpus.png", dpi=170)
    plt.close(fig)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--view-manifest", type=Path)
    args = parser.parse_args(argv)
    result = report(**vars(args))
    print(
        json.dumps(
            {
                "usable_train_behaviors": result["usable_train_behaviors"],
                "output": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
