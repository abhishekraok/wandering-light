"""Stream frozen train tasks to packed rows, then train fixed-epoch policies.

Prepare executes stack programs: run inside systemd MemoryMax=12G / MemorySwapMax=0.
Use the same prepare/train settings on old and new train payloads for comparisons.
No evaluation corpus is opened and no held-out score selects a checkpoint.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from experiments.compute_optimal_mark28 import ARMS
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_machine import parse_stack, run_body
from wandering_light.stack_pbe import _TUPLE_FIELDS, BASIS, Task
from wandering_light.stack_policy import (
    ENCODINGS,
    OP_INDEX,
    STATE_CAP,
    TARGET_CAP,
    Row,
    _encode_value,
    node_encoder,
    replay_witness,
    save_policy,
    train_policy,
)

INDEX = np.dtype(
    [
        ("offset", "<u8"),
        ("count", "<u2"),
        ("width", "<u2"),
        ("label", "<u2"),
        ("task", "<u4"),
        ("step", "<u4"),
        ("remaining", "<u4"),
    ]
)


def full_token_count(stack) -> int:
    """Count shared-encoder tokens without allocating a full token sequence."""

    class Counter:
        count = 0

        def append(self, token):
            self.count += 1

        def extend(self, tokens):
            self.count += sum(1 for _ in tokens)

    counter = Counter()
    for value in stack:
        _encode_value(value, counter)
    return counter.count


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def provenance(config: dict, sources: list[Path]) -> dict:
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), *args],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "config": config,
        "seed": config["seed"],
        "data_sha256": {str(path.resolve()): sha256(path) for path in sources},
        "wandb_run.url": None,
    }


def tasks(paths: list[Path]):
    for path in paths:
        with path.open("rb") as raw:
            zipped = raw.read(2) == b"\x1f\x8b"
        with (gzip.open if zipped else open)(path, "rt") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != "train":
                    continue
                for key in _TUPLE_FIELDS:
                    row[key] = tuple(row[key])
                row["alternative_programs"] = tuple(
                    tuple(p) for p in row.get("alternative_programs", ())
                )
                yield Task(**row)


def prefix_steps(task: Task, count: int, seed: int) -> set[int]:
    length = len(task.program)
    if count == 0 or count >= length:
        return set(range(length))
    rng = random.Random(f"pbe-scale-prefix:{seed}:{task.behaviour}")
    return {0, length - 1, *rng.sample(range(1, length - 1), count - 2)}


def prepare(
    paths: list[Path],
    output: Path,
    *,
    prefixes: int = 32,
    seed: int = 0,
    encoding: str = "machine",
    limit: int | None = None,
) -> dict:
    if prefixes != 0 and prefixes < 2:
        raise ValueError("prefixes must be 0 (all) or at least 2")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = {
        "prefixes_per_task": prefixes,
        "seed": seed,
        "encoding": encoding,
        "task_limit": limit,
    }
    manifest = provenance(config, paths)
    row_count = token_count = task_count = full_prefixes = 0
    stats = dict.fromkeys(
        (
            "state_examples",
            "state_overflow_examples",
            "max_state_tokens",
            "target_examples",
            "target_overflow_examples",
            "max_target_tokens",
        ),
        0,
    )

    def count_stacks(stacks, kind, cap):
        for stack in stacks:
            size = full_token_count(stack)
            stats[f"{kind}_examples"] += 1
            stats[f"{kind}_overflow_examples"] += size > cap
            stats[f"max_{kind}_tokens"] = max(stats[f"max_{kind}_tokens"], size)

    with (
        (output / "tokens.bin.part").open("wb") as data,
        (output / "index.bin.part").open("wb") as index,
    ):
        for task in tasks(paths):
            if limit is not None and task_count >= limit:
                break
            selected = prefix_steps(task, prefixes, seed)
            targets = [parse_stack(text, basis=BASIS) for text in task.outputs]
            count_stacks(targets, "target", TARGET_CAP)
            encode = node_encoder(targets, encoding)
            # Complete each replay before publishing its bounded sample of rows.
            buffered = []
            for step, states, name in replay_witness(task):
                if step in selected:
                    count_stacks(states, "state", STATE_CAP)
                    buffered.append((step, name, encode(states)))
            for step, name, block in buffered:
                packed = np.asarray(block, dtype="<u2")
                data.write(packed.tobytes())
                entry = (
                    token_count,
                    *packed.shape,
                    OP_INDEX[name],
                    task_count,
                    step,
                    len(task.program) - step,
                )
                index.write(np.array([entry], dtype=INDEX).tobytes())
                token_count += packed.size
                row_count += 1
            full_prefixes += len(task.program)
            task_count += 1
            run_body.cache_clear()
            if task_count % 100 == 0:
                print(
                    f"Prepared {task_count} tasks / {row_count} rows in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    if not row_count:
        raise ValueError("no training rows found")
    for name in ("tokens.bin", "index.bin"):
        (output / f"{name}.part").replace(output / name)
    seconds = time.perf_counter() - started
    manifest.update(
        tasks=task_count,
        rows=row_count,
        stored_tokens=token_count,
        full_prefixes=full_prefixes,
        prepare_seconds=seconds,
        tasks_per_second=task_count / seconds,
        index_dtype=INDEX.descr,
        files={name: sha256(output / name) for name in ("tokens.bin", "index.bin")},
    )
    stats.update(state_cap=STATE_CAP, target_cap=TARGET_CAP)
    for kind in ("state", "target"):
        stats[f"{kind}_overflow_fraction"] = (
            stats[f"{kind}_overflow_examples"] / stats[f"{kind}_examples"]
        )
    manifest["encoding_stats"] = stats
    write_json(output / "manifest.json", manifest)
    return manifest


class PackedRows(Sequence):
    """Only an index and memory maps stay live; Row objects are created per batch."""

    def __init__(self, directory: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.index = np.memmap(directory / "index.bin", dtype=INDEX, mode="r")
        self.tokens = np.memmap(directory / "tokens.bin", dtype="<u2", mode="r")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        item = self.index[key]
        offset, count, width = (int(item[k]) for k in ("offset", "count", "width"))
        block = self.tokens[offset : offset + count * width].reshape(count, width)
        return Row(
            block, *(int(item[k]) for k in ("label", "task", "step", "remaining"))
        )


class MixedRows(Sequence):
    """Visit new rows once; replay old rows through a fixed modular permutation."""

    def __init__(self, new: PackedRows, old: PackedRows, share: float, seed: int):
        if not 0 < share < 1:
            raise ValueError("mix share must be strictly between 0 and 1")
        self.new, self.old = new, old
        self.old_count = math.ceil(len(new) * share / (1 - share))
        rng = random.Random(seed)
        self.offset, self.stride = (
            rng.randrange(len(old)),
            rng.randrange(1, len(old) + 1),
        )
        while math.gcd(self.stride, len(old)) != 1:
            self.stride = rng.randrange(1, len(old) + 1)
        tokens = sum(
            int(old.index[self.old_index(i)]["count"])
            * int(old.index[self.old_index(i)]["width"])
            for i in range(self.old_count)
        )
        self.manifest = {
            **new.manifest,
            "rows": len(self),
            "stored_tokens": new.manifest["stored_tokens"] + tokens,
            "old_preparation": old.manifest,
            "mixture": {
                "new_rows": len(new),
                "old_rows": self.old_count,
                "old_underlying_rows": len(old),
                "new_full_prefixes": new.manifest["full_prefixes"],
                "old_full_prefixes": old.manifest["full_prefixes"],
                "requested_old_share": share,
                "actual_old_share": self.old_count / len(self),
                "sampling": "fixed modular old-row permutation with cycling; heuristic share, not selected from eval",
            },
        }

    def old_index(self, position):
        return (self.offset + position * self.stride) % len(self.old)

    def __len__(self):
        return len(self.new) + self.old_count

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        return (
            self.new[key]
            if key < len(self.new)
            else self.old[self.old_index(key - len(self.new))]
        )


def train(
    directory: Path,
    output: Path,
    *,
    arm: str = "1m",
    epochs: int = 3,
    seed: int = 0,
    device: str = "cpu",
    learning_rate: float | None = None,
    batch_size: int = 256,
    wandb_mode: str = "online",
    threads: int = 8,
    mix_data: Path | None = None,
    mix_share: float = 0.2,
) -> dict:
    import torch
    import wandb

    if epochs < 1 or batch_size < 1 or threads < 1:
        raise ValueError("epochs, batch size and threads must be positive")
    torch.set_num_threads(threads)
    rows = PackedRows(directory)
    sources = [
        directory / name for name in ("manifest.json", "tokens.bin", "index.bin")
    ]
    if mix_data is not None:
        rows = MixedRows(rows, PackedRows(mix_data), mix_share, seed)
        sources += [
            mix_data / name for name in ("manifest.json", "tokens.bin", "index.bin")
        ]
    learning_rate = (
        learning_rate if learning_rate is not None else {"1m": 2e-3, "10m": 5e-4}[arm]
    )
    config = {
        "arm": arm,
        "epochs": epochs,
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": device,
        "encoding": rows.manifest["config"]["encoding"],
        "threads": threads,
        "checkpoint_selection": "every fixed epoch; no held-out selection",
        "mix_data": str(mix_data.resolve()) if mix_data else None,
        "mixture": rows.manifest.get("mixture"),
    }
    output.mkdir(parents=True, exist_ok=False)
    origin = provenance(config, sources)
    origin["preparation"] = rows.manifest
    # This probe contains training rows only; it is telemetry, not model selection.
    probe = [
        rows[i]
        for i in random.Random(seed).sample(range(len(rows)), min(256, len(rows)))
    ]
    run = wandb.init(
        entity="abhishekraok-na",
        project="wandering-light-rollout-policy",
        name=output.name,
        config=config | {"train_rows": len(rows)},
        mode=wandb_mode,
    )
    url = run.url if wandb_mode == "online" else None
    origin["wandb_run.url"] = url
    write_json(output / "run_manifest.json", origin)
    steps_per_epoch = math.ceil(len(rows) / batch_size)

    def budget(epoch):
        return {
            "optimizer_steps": epoch * steps_per_epoch,
            "rows_seen": epoch * len(rows),
            "stored_tokens_seen": epoch * rows.manifest["stored_tokens"],
        }

    def snapshot(epoch, model, row):
        state = {
            **row,
            **config,
            "epoch": epoch,
            "train_rows": len(rows),
            "parameters": model.parameter_count,
            **budget(epoch),
            "train_probe_cross_entropy": row["validation_cross_entropy"],
        }
        state.pop("validation_cross_entropy")
        save_policy(
            output / f"epoch-{epoch:03d}",
            model,
            trainer_state=state,
            manifest=origin,
            wandb_url=url,
        )
        run.log(state, step=epoch)
        print(json.dumps(state), flush=True)

    try:
        _, state = train_policy(
            rows,
            probe,
            PolicyArchitecture(*ARMS[arm]["dims"]),
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_batch_size=batch_size,
            seed=seed,
            device=device,
            epoch_end=snapshot,
            encoding=config["encoding"],
        )
        # train_policy's best-CE return is deliberately unused. Epoch checkpoints are the deliverable.
        result = {
            "final_checkpoint": str(output / f"epoch-{epochs:03d}"),
            "epochs": epochs,
            **budget(epochs),
            "training_seconds": state["training_seconds"],
            "wandb_run.url": url,
            "provenance": origin,
        }
        write_json(output / "training_summary.json", result)
        return result
    finally:
        run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--input", dest="paths", nargs="+", required=True, type=Path)
    prep.add_argument("--output-dir", dest="output", required=True, type=Path)
    prep.add_argument(
        "--prefixes-per-task",
        "--prefixes",
        dest="prefixes",
        type=int,
        default=32,
        help="0 keeps every prefix; default 32",
    )
    prep.add_argument("--encoding", choices=ENCODINGS, default="machine")
    prep.add_argument("--limit", type=int)
    fit = commands.add_parser("train")
    fit.add_argument("--data", dest="directory", required=True, type=Path)
    fit.add_argument("--output-dir", dest="output", required=True, type=Path)
    fit.add_argument("--arm", choices=("1m", "10m"), default="1m")
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--learning-rate", type=float)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--mix-data", type=Path)
    fit.add_argument("--mix-share", type=float, default=0.2)
    for command in (prep, fit):
        command.add_argument("--seed", type=int, default=0)
    options = vars(parser.parse_args())
    function = {"prepare": prepare, "train": train}[options.pop("stage")]
    result = function(**options)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
