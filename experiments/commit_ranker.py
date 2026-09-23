"""A learned commitment ranker for inner-frames (#160), second campaign.

At every `[` a committing search chooses what the quotation is for: its
consumer, the `ifte` role, and how many values the closure idiom will cons on.
The first gate charged that choice from a shape-only prior counted on train
(`frame_gate.CommitmentModel`: 1.24 nats a `[` on v4 eval, 1.00 on world-v3),
and with it charged even free defined bodies cleared only 23 of 45 v4
witnesses. This replaces the prior with the policy's own encoder and a head
over the same closed vocabulary, trained on every `[` of the world arm's fit
tasks from what a search has there:

- the state, in the ``frames_inner`` encoding, so a nested `[` sees its
  enclosing level's label and frame (a search that commits has them);
- the task's outputs, as every policy row does.

Its surprisal of the witness's own choice, at a temperature fitted on the v4
holdout's `[` rows, is the charge when clears are recomputed.

Stages, under ``--run-dir`` (default `checkpoints/frame-campaign-2`): ``rows``
(counts, no training), ``train``, ``measure``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
import experiments.world_arm as wa  # noqa: E402
from experiments.frame_gate import (  # noqa: E402
    CORPORA,
    NO_COMMITMENT,
    T_BOUNDS,
    commitments,
    eval_witnesses,
)
from experiments.frame_rows import frame_product  # noqa: E402
from experiments.inner_frames import COMBINATORS  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_NAMES,
    Row,
    StackPairPolicy,
    collate,
    encode_item,
    encode_targets,
    item_array,
    load_policy,
    seed_everything,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

RUN_DIR = REPO / "checkpoints" / "frame-campaign-2"
ARM = "ranker"
# About a seventh of the policy's rows, so six epochs would be ~1,500 steps.
# Thirty epochs at the policy's rate, and each seed is measured at its epoch of
# lowest validation cross-entropy (the v4 holdout's `[` rows, not eval).
EPOCHS = 30
ENCODING = "frames_inner"
PREFIX = "commit:"

Choice = tuple[str, int, int]


def vocabulary(max_captures: int) -> list[Choice]:
    """Every choice a search could make at a `[`, as `CommitmentModel` smooths over."""
    out: list[Choice] = [(NO_COMMITMENT, 0, 0)]
    for tag in sorted(COMBINATORS):
        for role in range(3) if tag == "ifte" else (0,):
            out.extend((tag, role, c) for c in range(max_captures + 1))
    return out


def actions(choices: Sequence[Choice]) -> tuple[str, ...]:
    """The head's action space: the 28 ops (a checkpoint contract), then the choices."""
    return (*OP_NAMES, *(f"{PREFIX}{t}:{r}:{c}" for t, r, c in choices))


def choices_of(model: StackPairPolicy) -> list[Choice]:
    out = []
    for name in model.actions[len(OP_NAMES) :]:
        tag, role, captures = name[len(PREFIX) :].rsplit(":", 2)
        out.append((tag, int(role), int(captures)))
    return out


def ranker_rows(tasks: Sequence[Task], choices: Sequence[Choice]) -> list[Row]:
    """One row per `[` of each task's witness, labelled with its commitment."""
    index = {c: i for i, c in enumerate(choices)}
    rows = []
    for number, task in enumerate(tasks):
        committed = commitments(task.program, task.inputs[0])
        if not committed:
            continue
        targets = encode_targets(
            [parse_stack(text, basis=BASIS) for text in task.outputs]
        )
        product = frame_product(task, ENCODING)
        for step, (_shape, choice) in sorted(committed.items()):
            if choice not in index:
                raise ValueError(f"{task.behaviour} step {step}: {choice} not in vocab")
            items = [
                encode_item(state, target)
                for state, target in zip(product[step], targets, strict=True)
            ]
            rows.append(
                Row(item_array(items), index[choice], number, step, len(task.program))
            )
    return rows


def split_rows() -> tuple[list[Row], list[Row], list[Choice]]:
    """Fit and validation rows on the world arm's own task split."""
    fit_tasks, held_tasks = co.split_tasks(wa.configure())
    most = max(
        (
            choice[2]
            for task in fit_tasks
            for _shape, choice in commitments(task.program, task.inputs[0]).values()
        ),
        default=0,
    )
    choices = vocabulary(most + 1)
    return ranker_rows(fit_tasks, choices), ranker_rows(held_tasks, choices), choices


def _choice_logits(model: StackPairPolicy, rows: Sequence[Row], device) -> torch.Tensor:
    return model(*collate([row.items for row in rows], device))[:, len(OP_NAMES) :]


def _labels(rows: Sequence[Row], device) -> torch.Tensor:
    return torch.tensor([row.label for row in rows], dtype=torch.long, device=device)


def validation_ce(model, rows, device, batch_size: int = wa.VAL_BATCH) -> float:
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            total += nn.functional.cross_entropy(
                _choice_logits(model, chunk, device),
                _labels(chunk, device),
                reduction="sum",
            ).item()
    return total / len(rows)


def train(run_dir: Path, device: str, seeds: Sequence[int], log) -> None:
    """The world arm's Adam, LR and batch, `EPOCHS` epochs, every epoch snapshotted."""
    fit, held, choices = split_rows()
    log(
        f"RESULT ranker rows: fit {len(fit)} validation {len(held)}, {len(choices)} choices"
    )
    target = torch.device(device)
    for seed in seeds:
        if hs.policy_dir(run_dir / ARM, wa.ARM, seed, EPOCHS).exists():
            log(f"skip train ranker s{seed} (snapshots exist)")
            continue
        seed_everything(seed)
        model = StackPairPolicy(co.architecture(wa.ARM), actions(choices)).to(target)
        optimizer = torch.optim.Adam(model.parameters(), lr=wa.LEARNING_RATE)
        save = hs.snapshotter(
            run_dir / ARM,
            wa.ARM,
            seed,
            {
                "arm": ARM,
                "encoding": f"{ENCODING} at each `[` (experiments/commit_ranker.py)",
                "choices": len(choices),
                "epochs": EPOCHS,
                "learning_rate": wa.LEARNING_RATE,
                "seed": seed,
                "provenance": co.provenance(),
            },
        )
        order = list(range(len(fit)))
        rng = random.Random(seed)
        history = []
        started = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            rng.shuffle(order)
            total = 0.0
            for start in range(0, len(order), wa.BATCH):
                chunk = [fit[i] for i in order[start : start + wa.BATCH]]
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(
                    _choice_logits(model, chunk, target),
                    _labels(chunk, target),
                    reduction="sum",
                )
                (loss / len(chunk)).backward()
                optimizer.step()
                total += loss.item()
            model.eval()
            row = {
                "epoch": epoch,
                "train_cross_entropy": total / len(fit),
                "validation_cross_entropy": validation_ce(model, held, target),
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            save(epoch, model, dict(row))
        co.write_json(
            run_dir / ARM / "training" / f"s{seed}.json",
            {"history": history, "fit_rows": len(fit), "validation_rows": len(held)},
        )
        curve = " ".join(
            f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}" for r in history
        )
        log(
            f"RESULT train ranker s{seed}: [{curve}] {time.perf_counter() - started:.0f}s"
        )


def _all_logits(model, rows, device, batch_size: int = 512) -> np.ndarray:
    with torch.inference_mode():
        return np.concatenate(
            [
                _choice_logits(model, rows[i : i + batch_size], device)
                .float()
                .cpu()
                .numpy()
                for i in range(0, len(rows), batch_size)
            ]
        )


def _nll(logits: np.ndarray, labels: np.ndarray, t: float) -> np.ndarray:
    scaled = logits / t
    peak = scaled.max(axis=1, keepdims=True)
    norm = peak[:, 0] + np.log(np.exp(scaled - peak).sum(axis=1))
    return norm - scaled[np.arange(len(labels)), labels]


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Golden section on log T over `T_BOUNDS`, as the policies are calibrated."""
    lo, hi = (math.log(b) for b in T_BOUNDS)
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if (
            _nll(logits, labels, math.exp(a)).mean()
            < _nll(logits, labels, math.exp(b)).mean()
        ):
            hi = b
        else:
            lo = a
    t = math.exp((lo + hi) / 2)
    return {
        "temperature": round(t, 5),
        "validation_ce": round(float(_nll(logits, labels, 1.0).mean()), 5),
        "calibrated_validation_ce": round(float(_nll(logits, labels, t).mean()), 5),
        "at_boundary": min(abs(math.log(t) - math.log(b)) for b in T_BOUNDS) < 1e-6,
    }


def charges(model, corpus: str, temperature: float, device) -> dict[str, list[float]]:
    """Per eval behaviour, per witness step: the learned charge, nonzero at a `[`."""
    choices = choices_of(model)
    out = {}
    for task in eval_witnesses(corpus):
        rows = ranker_rows([task], choices)
        step_charge = [0.0] * len(task.program)
        if rows:
            logits = _all_logits(model, rows, device)
            labels = np.array([row.label for row in rows])
            for row, cost in zip(rows, _nll(logits, labels, temperature), strict=True):
                step_charge[row.step] = float(cost)
        out[task.behaviour] = step_charge
    return out


def best_epoch(run_dir: Path, seed: int) -> int:
    history = json.loads((run_dir / ARM / "training" / f"s{seed}.json").read_text())
    return min(history["history"], key=lambda r: r["validation_cross_entropy"])["epoch"]


def measure(run_dir: Path, seeds: Sequence[int], log) -> list[Path]:
    """Calibrate each seed's best-validation snapshot on the v4 holdout; charge eval."""
    _fit, held, _choices = split_rows()
    labels = np.array([row.label for row in held])
    paths = []
    for seed in seeds:
        epoch = best_epoch(run_dir, seed)
        out = run_dir / "measure" / f"{ARM}-s{seed}.json"
        if out.exists():
            paths.append(out)
            continue
        path = hs.policy_dir(run_dir / ARM, wa.ARM, seed, epoch)
        model = load_policy(path).eval()
        calibration = fit_temperature(_all_logits(model, held, "cpu"), labels)
        per = {
            corpus: charges(model, corpus, calibration["temperature"], "cpu")
            for corpus in CORPORA
        }
        summary = {
            corpus: {
                "charged": sum(1 for c in v.values() for x in c if x),
                "mean": round(
                    statistics.fmean(x for c in v.values() for x in c if x), 4
                ),
                "median": round(
                    statistics.median(x for c in v.values() for x in c if x), 4
                ),
            }
            for corpus, v in per.items()
        }
        log(f"RESULT ranker s{seed} e{epoch}: {calibration} {summary}")
        co.write_json(
            out,
            {
                "policy": str(path),
                "epoch": epoch,
                "calibration": calibration,
                "summary": summary,
                "charges": per,
            },
        )
        paths.append(out)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("rows", "train", "measure"))
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default=",".join(str(s) for s in wa.SEEDS))
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with open(args.run_dir / "log.txt", "a") as handle:
            handle.write(stamped + "\n")

    torch.set_num_threads(8)
    if args.stage == "rows":
        fit, held, choices = split_rows()
        counts = dict.fromkeys(choices, 0)
        for row in fit:
            counts[choices[row.label]] += 1
        log(
            f"RESULT ranker rows: fit {len(fit)} validation {len(held)}, "
            f"{len(choices)} choices; most common "
            f"{sorted(counts.items(), key=lambda kv: -kv[1])[:6]}"
        )
    elif args.stage == "train":
        train(args.run_dir, args.device, [int(s) for s in args.seeds.split(",")], log)
    else:
        measure(args.run_dir, [int(s) for s in args.seeds.split(",")], log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
