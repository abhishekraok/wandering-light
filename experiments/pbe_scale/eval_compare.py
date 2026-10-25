"""Freeze only v4 eval and compare existing solve-CLI results; no model calls."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

V4 = (
    Path(__file__).resolve().parents[2]
    / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
)
KNOBS = (
    "beam_width",
    "max_depth",
    "guard",
    "library",
    "max_state_atoms",
    "search",
    "roots",
    "segment_depth",
    "max_segments",
    "scoring",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)


def task_keys(payload: Path) -> list[tuple[int, str]]:
    """Read identities only; never execute a witness or expose task contents."""
    with gzip.open(payload, "rt") as stream:
        return [(i, json.loads(line)["behaviour"]) for i, line in enumerate(stream)]


def provenance() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": sha(Path(__file__)),
    }


def freeze(output: Path) -> dict:
    from huggingface_hub import hf_hub_download

    manifest = read(V4)
    split = manifest["splits"]["eval"]
    source = V4.parent / split["path"]
    if not source.exists():
        source = Path(
            hf_hub_download(
                **manifest["hub"], filename=split["path"], local_files_only=True
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    target = output / "tasks.jsonl.gz"
    shutil.copyfile(source, target)
    keys = task_keys(target)
    result = {
        "source": str(source),
        "payload": str(target),
        "sha256": sha(target),
        "expected_sha256": split["sha256"].removeprefix("sha256:"),
        "manifest_sha256": sha(V4),
        "tasks": len(keys),
        "behaviors": len({key[1] for key in keys}),
        "expected_tasks": split["records"],
        "expected_behaviors": split["behaviours"],
        "hub": manifest["hub"],
        "provenance": provenance(),
    }
    write(output / "input_manifest.json", result)
    return result


def compare(old_path: Path, mixed_path: Path, output: Path) -> dict:
    old, mixed = read(old_path), read(mixed_path)
    frozen = read(output / "input_manifest.json")
    expected = task_keys(output / "tasks.jsonl.gz")
    ordered = [
        [(row["index"], row["behaviour"]) for row in run["rows"]]
        for run in (old, mixed)
    ]
    if any(
        len(keys) != frozen["tasks"] or len(set(keys)) != len(keys) for keys in ordered
    ) or any(set(keys) != set(expected) for keys in ordered):
        raise ValueError(
            "incomplete or unpaired evaluation rows; missing tasks are not scored as zero"
        )
    caveats = [
        "Held-out-example grading, not fresh-reference semantic verification; one training seed.",
        "Behavior count means at least one graded task solved; fixed epochs do not match training compute.",
    ]
    if any(
        type(row["solved"]) is not bool for run in (old, mixed) for row in run["rows"]
    ):
        raise ValueError("evaluation rows need explicit boolean held-out grades")
    knobs = [{key: run["system"][key] for key in KNOBS} for run in (old, mixed)]
    if knobs[0] != knobs[1]:
        caveats.append(
            "Search settings differ; this is not a matched search-budget comparison."
        )
    if ordered[0] != ordered[1]:
        caveats.append(
            "Task order differs; solved-program-library exposure may differ."
        )
    payload_sha = sha(output / "tasks.jsonl.gz")
    if payload_sha != frozen["sha256"] or payload_sha != frozen["expected_sha256"]:
        caveats.append(
            "Frozen payload digest differs from its recorded or pinned manifest digest."
        )
    if (frozen["tasks"], frozen["behaviors"]) != (
        frozen.get("expected_tasks", frozen["tasks"]),
        frozen.get("expected_behaviors", frozen["behaviors"]),
    ):
        caveats.append("Frozen payload counts differ from the pinned manifest counts.")
    successes, sources, counts = [], {}, {}
    for label, path, run in (("old", old_path, old), ("mixed", mixed_path, mixed)):
        solved = {
            (row["index"], row["behaviour"]) for row in run["rows"] if row["solved"]
        }
        successes.append(solved)
        counts[label] = {
            "tasks_solved": len(solved),
            "behaviors_with_graded_solution": len({key[1] for key in solved}),
            "tasks_attempted": len(run["rows"]),
            "seconds": run["summary"]["seconds"],
            "executions": run["summary"]["executions"],
        }
        checkpoint = Path(run["checkpoint"])
        state, manifest = (
            read(checkpoint / "trainer_state.json"),
            read(checkpoint / "training_manifest.json"),
        )
        sources[label] = {
            "result": str(path),
            "result_sha256": sha(path),
            "checkpoint": str(checkpoint),
            "model_sha256": sha(checkpoint / "model.safetensors"),
            "policy_config": read(checkpoint / "policy_config.json"),
            "trainer_state": state,
            "training_manifest": manifest,
            "host": run["host"],
            "commit": run["commit"],
            "wandb_run.url": manifest.get("wandb_run.url", manifest.get("wandb_url")),
            "search": knobs[len(successes) - 1],
        }
        if Path(run["tasks_file"]).resolve() != (output / "tasks.jsonl.gz").resolve():
            caveats.append(f"{label} result reports a different task payload path.")
        if (state["seed"], state["epoch"]) != (0, 3):
            caveats.append(
                f"{label} checkpoint differs from the planned seed 0, epoch 3."
            )
    result = {
        "status": "complete",
        "counts": counts,
        "task_gain": len(successes[1] - successes[0]),
        "task_loss": len(successes[0] - successes[1]),
        "task_delta": len(successes[1]) - len(successes[0]),
        "behavior_delta": counts["mixed"]["behaviors_with_graded_solution"]
        - counts["old"]["behaviors_with_graded_solution"],
        "input": frozen,
        "actual_payload_sha256": payload_sha,
        "sources": sources,
        "caveats": caveats,
        "provenance": provenance(),
    }
    plot(result, output / "comparison.png")
    write(output / "comparison.json", result)
    return result


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, key, title, total in zip(
        axes,
        ("tasks_solved", "behaviors_with_graded_solution"),
        ("Tasks passing held-out examples", "Behaviors with ≥1 graded solution"),
        (result["input"]["tasks"], result["input"]["behaviors"]),
        strict=True,
    ):
        values = [result["counts"][label][key] for label in ("old", "mixed")]
        bars = axis.bar(
            ("Old staged", "Old + new"), values, color=("#64748b", "#0284c7")
        )
        axis.bar_label(bars)
        axis.set(title=title, ylim=(0, total * 1.1), ylabel=f"Count / {total}")
    checkpoints = [
        result["sources"][label]["trainer_state"] for label in ("old", "mixed")
    ]
    settings = {(state["seed"], state["epoch"]) for state in checkpoints}
    setting = (
        f"seed {checkpoints[0]['seed']} · fixed epoch {checkpoints[0]['epoch']}"
        if len(settings) == 1
        else "different training seeds or epochs; see comparison.json"
    )
    figure.suptitle(f"Frozen v4 eval · {setting}")
    figure.supxlabel(
        "Held-out-example grade; no fresh semantic proof. Mixture uses more training updates.",
        fontsize=8,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "compare"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--mixed", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze(args.output_dir)
    else:
        if args.old is None or args.mixed is None:
            parser.error("compare requires --old and --mixed")
        result = compare(args.old, args.mixed, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("counts", "task_delta", "behavior_delta")
                if key in result
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
