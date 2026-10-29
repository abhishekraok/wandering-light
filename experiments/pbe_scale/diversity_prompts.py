"""Count shown-example coverage from saved tasks, without executing references."""

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.pbe_scale import diversity_semantic


def summarize(source: Path, exclusions: Path) -> dict:
    excluded = set(json.loads(exclusions.read_text())["excluded_hashes"])
    groups, digest = {}, hashlib.sha256()
    for line in source.open("rb"):
        digest.update(line)
        row = json.loads(line)
        if row["candidate_digest"] in excluded:
            continue
        kind, task = row["candidate"]["input_kind"], row["task"]
        group = groups.setdefault(
            kind,
            {
                "rows": 0,
                "shown_pairs": 0,
                "input_max_list_lengths": collections.Counter(),
                "distinct_shown_outputs": collections.Counter(),
                "shown_long_case_counts_per_task": collections.Counter(),
                "rows_with_fewer_than_two_shown_outputs": 0,
            },
        )
        lengths = [diversity_semantic.max_list_length(s) for s in task["inputs"]]
        different = len(set(task["outputs"]))
        group["rows"] += 1
        group["shown_pairs"] += len(lengths)
        group["input_max_list_lengths"].update(lengths)
        group["distinct_shown_outputs"][different] += 1
        group["shown_long_case_counts_per_task"][sum(n >= 4 for n in lengths)] += 1
        group["rows_with_fewer_than_two_shown_outputs"] += different < 2
    repo = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "groups": groups,
        "method": "Original one-view training prompts; maximum list length across arguments; no programs executed.",
        "provenance": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "seed": None,
            "config": {"min_list_length": 4, "source_view": "original"},
            "data_sha256": {
                str(source.resolve()): digest.hexdigest(),
                str(exclusions.resolve()): hashlib.sha256(
                    exclusions.read_bytes()
                ).hexdigest(),
            },
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__).resolve(),
                    Path(diversity_semantic.__file__).resolve(),
                )
            },
            "wandb_run.url": None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.accepted, args.exclusions)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved original training-prompt coverage.")


if __name__ == "__main__":
    main()
