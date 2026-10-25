"""Export multiple six-example views of existing verified train behaviors."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from wandering_light.stack_pbe import Task, assign_task_hash_split


def views_for_task(row: dict, count: int, seed: int):
    Task(**row)  # Reject missing/extra schema fields; this never executes a program.
    if row["split"] != "train" or row["length"] != len(row["program"]):
        raise ValueError("expected a train task with consistent program length")
    for prefix, size in (("", row["examples"]), ("held_out_", row["held_out"])):
        if len(row[prefix + "inputs"]) != size or len(row[prefix + "outputs"]) != size:
            raise ValueError("inconsistent example counts")
    inputs = row["inputs"] + row["held_out_inputs"]
    outputs = row["outputs"] + row["held_out_outputs"]
    if len(inputs) != len(outputs) or len(set(inputs)) != len(inputs):
        raise ValueError("verified inputs must be unique with matching output counts")
    if len(inputs) not in {21, 208, 441} or not all(
        isinstance(value, str) for value in inputs + outputs
    ):
        raise ValueError("expected 21/208/441 verified string-encoded pairs")
    hidden_count = 5 if len(inputs) == 21 else 32
    rng = random.Random(f"pbe-views:{seed}:{row['behaviour']}")
    groups = defaultdict(list)
    for index, value in enumerate(outputs):
        groups[value].append(index)
    ordered = list(groups.values())
    rng.shuffle(ordered)
    ordered.sort(key=len)
    protected = {rng.choice(group) for group in ordered[:2]}
    indices = list(range(len(inputs)))
    rng.shuffle(indices)
    hidden = [index for index in indices if index not in protected][:hidden_count]
    hidden_set = set(hidden)
    pool = [index for index in indices if index not in hidden_set]
    if count < 1:
        raise ValueError("views must be positive")
    seen = set()
    for view_index in range(count):
        for _ in range(10000):
            shown = rng.sample(pool, 6)
            if len(groups) > 1 and len({outputs[index] for index in shown}) == 1:
                shown[-1] = rng.choice(
                    [index for index in pool if outputs[index] != outputs[shown[0]]]
                )
            key = tuple(sorted(shown))
            if key not in seen:
                seen.add(key)
                break
        else:
            raise ValueError("could not construct another distinct shown set")
        rng.shuffle(shown)
        given = [inputs[index] for index in shown]
        wanted = [outputs[index] for index in shown]
        yield row | {
            "inputs": given,
            "outputs": wanted,
            "examples": 6,
            "held_out_inputs": [inputs[index] for index in hidden],
            "held_out_outputs": [outputs[index] for index in hidden],
            "held_out": hidden_count,
            "task_hash_split": assign_task_hash_split(tuple(given), tuple(wanted)),
            "fuel": 0,
            "tags": [*row["tags"], "fuel-unmeasured", f"view-{view_index:02d}"],
        }


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def augment(tasks: Path, output: Path, views: int = 16, seed: int = 0) -> dict:
    if output.exists() or output.resolve() == tasks.resolve():
        raise FileExistsError("augmentation requires a new output path")
    if views < 1:
        raise ValueError("views must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    behaviors, programs, prefixes, emitted = set(), set(), 0, 0
    with (
        gzip.open(tasks, "rt") as source,
        temporary.open("wb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1
        ) as stream,
    ):
        for line in source:
            row = json.loads(line)
            if row["behaviour"] in behaviors:
                raise ValueError(f"repeated behavior: {row['behaviour']}")
            behaviors.add(row["behaviour"])
            encoded = json.dumps(row["program"]).encode()
            programs.add(hashlib.sha256(encoded).hexdigest())
            prefixes += row["length"]
            for view in views_for_task(row, views, seed):
                payload = json.dumps(view, sort_keys=True, separators=(",", ":"))
                stream.write((payload + "\n").encode())
                emitted += 1
    temporary.replace(output)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), *args],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    manifest = {
        "distinct_behaviors": len(behaviors),
        "distinct_programs": len(programs),
        "task_views": emitted,
        "views_per_behavior": views,
        "shown_examples": 6 * emitted,
        "full_prefix_rows": views * prefixes,
        "original_full_prefix_rows": prefixes,
        "seed": seed,
        "train_only": True,
        "execution_performed": False,
        "new_function_claim": False,
        "source_sha256": _sha(tasks),
        "output_sha256": _sha(output),
        "source": str(tasks.resolve()),
        "output": str(output.resolve()),
        "hidden_policy": "fixed across all views; 5 of 21 scalar pairs, otherwise 32 of 208/441",
        "fuel_measurement": "not measured for reshuffled shown inputs; zero is sentinel, not execution cost",
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "wandb_run.url": None,
    }
    Path(str(output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    result = augment(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {key: value for key, value in result.items() if isinstance(value, int)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
