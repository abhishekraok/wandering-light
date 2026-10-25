"""Compare fixed policies on identical sampled teacher labels from training pools."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from experiments.pbe_scale import eval_compare, train
from wandering_light import rollout_policy, stack_policy

SEED, BATCH, THREADS, SECONDS = 0, 256, 8, 300


def sample_indices(size: int, count: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(size), count)


def file_hashes(paths) -> dict:
    return {str(path.resolve()): train.sha256(path) for path in paths}


def score(model, rows, indices, batch_size: int, deadline: float) -> dict:
    loss, correct = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            if time.monotonic() >= deadline:
                raise TimeoutError("training-fit diagnostic exceeded its time budget")
            block = [rows[i] for i in indices[start : start + batch_size]]
            inputs, labels = stack_policy._batch(block, torch.device("cpu"))
            logits = model(*inputs)
            loss += torch.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
    return {
        "rows": len(indices),
        "teacher_label_ce": loss / len(indices),
        "teacher_label_top1": correct / len(indices),
    }


def diagnose(run: Path, count: int = 10_000) -> dict:
    started = time.monotonic()
    deadline = started + SECONDS
    if count <= 0:
        raise ValueError("positive row count required")
    torch.set_num_threads(THREADS)
    output = run / "fit-diagnostic"
    output.mkdir(parents=True, exist_ok=False)
    pools = {"old": run / "baseline-fit-packed32", "new": run / "new-fit-packed32"}
    rows = {name: train.PackedRows(path) for name, path in pools.items()}
    indices = {
        name: sample_indices(len(pool), count, SEED) for name, pool in rows.items()
    }
    names = {"old": "old-staged-1m-s0", "mixed": "mixed-1m-s0"}
    checkpoints = {k: run / "policies" / v / "epoch-003" for k, v in names.items()}
    modules = (train, stack_policy, rollout_policy, eval_compare)
    sources = [Path(m.__file__) for m in modules] + [Path(__file__)]
    provenance = {**eval_compare.provenance(), "source_sha256": file_hashes(sources)}
    provenance["pools"] = {
        name: {
            "files_sha256": file_hashes(path.iterdir()),
            "sample_indices": indices[name],
            "sample_indices_sha256": hashlib.sha256(
                json.dumps(indices[name]).encode()
            ).hexdigest(),
        }
        for name, path in pools.items()
    }
    provenance["checkpoints"] = {}
    for name, path in checkpoints.items():
        manifest = eval_compare.read(path / "training_manifest.json")
        provenance["checkpoints"][name] = {
            "files_sha256": file_hashes(path.iterdir()),
            "trainer_state": eval_compare.read(path / "trainer_state.json"),
            "wandb_run.url": manifest.get("wandb_run.url", manifest.get("wandb_url")),
        }
    settings = {
        "rows_per_pool": count,
        "seed": SEED,
        "batch_size": BATCH,
        "threads": THREADS,
        "device": "cpu",
        "seconds_cap": SECONDS,
    }
    result = {"settings": settings, "provenance": provenance}
    eval_compare.write(output / "input_manifest.json", result)
    results = {}
    for name, path in checkpoints.items():
        model = stack_policy.load_policy(path, device="cpu").eval()
        results[name] = {
            pool: score(model, data, indices[pool], BATCH, deadline)
            for pool, data in rows.items()
        }
        del model
    if time.monotonic() >= deadline:
        raise TimeoutError("training-fit diagnostic exceeded its time budget")
    result.update(
        status="complete", results=results, seconds=time.monotonic() - started
    )
    result["caveats"] = [
        "Training-pool teacher-label fit, not semantic generalization or held-out evaluation.",
        "Uniform packed-row sample; not uniform over behaviors or full prefixes.",
        "Raw logits without legality mask. Both epoch-003 models use identical samples.",
        "Original training git fields need the provenance-correction.json sidecar.",
    ]
    eval_compare.write(output / "metrics.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    result = diagnose(parser.parse_args().run_dir.resolve())
    print(json.dumps({"results": result["results"], "seconds": result["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
