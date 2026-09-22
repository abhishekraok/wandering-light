"""A clean v4 + full world-v3 policy, measured on world data (clear-world-v3 #161).

clear-world-v3 (#161) found that `solve.BEST`, trained on `stack_pbe_v4` alone,
sits a median **4.19 nats per step** behind its own top choice along world-v3's
eval witnesses, and that the two world-trained checkpoints that happen to exist
— the `random` and `ablated` arms of keep-below-ablation (#136) — sit at 0.76
and 1.10. Those two were built for a different ablation and each dropped 159
world behaviours from training, so the size of the effect is measured but its
ceiling is not. This trains the arm that should have been the baseline: v4 plus
the **whole** world-v3 train split.

It also fixes the metric. Coverage on this split is 1 of 44 for every policy
measured, so it cannot see a fivefold change in how close a policy is; the
per-step deficit can. This driver reports deficit per epoch as the steering
signal and runs the search once, at the chosen epoch, as the acceptance test.

    uv run python experiments/world_arm.py train  --run-dir checkpoints/world-arm-<date>
    uv run python experiments/world_arm.py deficit --run-dir ...
    uv run python experiments/world_arm.py solve  --run-dir ... --epoch <n>

Recipe is compute-optimal-mark-28 (#101)'s 1m arm, which every comparable
number in the repository uses: LR 1.25e-4, batch 256, six epochs, three seeds,
every epoch snapshotted. Validation is pinned to the **v4** holdout, as
world-plus-v4-policy (#134) pinned it, so `best_validation_cross_entropy` means
the same thing across corpus mixes and world-v3's own holdout tasks stay a
holdout. The three behaviours #134 found sharing a witness with a v4 eval
behaviour are dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
from experiments.staged_transfer import score  # noqa: E402
from experiments.witness_deficit import summarise  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.solve import (  # noqa: E402
    BEST,
    System,
    solve_tasks,
)
from wandering_light.solve import summarise as run_summary  # noqa: E402
from wandering_light.stack_pbe import load  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    PolicyScorer,
    load_policy,
    train_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DATA = REPO / "wandering_light" / "evals" / "data"
ARM = "1m"
SEEDS = (0, 1, 2)
EPOCHS = 6
LEARNING_RATE = 1.25e-4  # #101's probe choice for the 1m arm
BATCH, VAL_BATCH = 256, 512
# world-plus-v4-policy (#134): their witness equals a v4 eval behaviour's.
CONTAMINATED = {"codearc-128", "mbpp-404", "mbpp-970"}
TOLERANCE_WIDTH = 64


def configure() -> list:
    """Point `compute_optimal_mark28`'s module state at both corpora."""
    co.CORPUS_MANIFESTS = [
        DATA / "stack_pbe_v4" / "manifest.json",
        DATA / "stack_pbe_world_v3" / "manifest.json",
    ]
    co.DROP_BEHAVIOURS = set(CONTAMINATED)
    co.VALIDATION_CORPUS = "stack_pbe_v4"
    co.EVAL_CORPUS = "stack_pbe_v4"
    return co.corpus()


def world_eval() -> list:
    """One task per world-v3 eval behaviour; the witness is the object."""
    tasks = [
        t
        for t in load(fetch_corpus(DATA / "stack_pbe_world_v3" / "manifest.json"))
        if t.split == "eval"
    ]
    seen, unique = set(), []
    for t in tasks:
        if t.behaviour in seen:
            continue
        seen.add(t.behaviour)
        unique.append(t)
    return unique


def train(run_dir: Path, device: str, seeds: Sequence[int], log) -> None:
    tasks = configure()
    fit, held = co.training_rows(tasks)
    log(
        f"RESULT rows: fit {len(fit)} validation {len(held)} "
        f"over {len({t.behaviour for t in tasks})} behaviours, "
        f"dropped {sorted(CONTAMINATED)}"
    )
    for seed in seeds:
        if hs.policy_dir(run_dir, ARM, seed, EPOCHS).exists():
            log(f"skip train s{seed} (snapshots exist)")
            continue
        manifest = {
            "arm": ARM,
            "corpora": ["stack_pbe_v4", "stack_pbe_world_v3"],
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": seed,
            "dropped_behaviours": sorted(CONTAMINATED),
            "provenance": co.provenance(),
        }
        started = time.perf_counter()
        _model, state = train_policy(
            fit,
            held,
            co.architecture(ARM),
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH,
            validation_batch_size=VAL_BATCH,
            seed=seed,
            device=device,
            epoch_end=hs.snapshotter(run_dir, ARM, seed, manifest),
        )
        co.write_json(run_dir / "training" / f"s{seed}.json", state)
        curve = " ".join(
            f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}"
            for r in state["history"]
        )
        log(f"RESULT train s{seed}: [{curve}] {time.perf_counter() - started:.0f}s")


def deficit(run_dir: Path, device: str, seeds: Sequence[int], epochs, log) -> Path:
    """Per-step deficit along world-v3 eval witnesses, per (seed, epoch)."""
    out = run_dir / "deficit.json"
    known = json.loads(out.read_text()) if out.exists() else {}
    tasks = world_eval()
    tolerance = math.log(TOLERANCE_WIDTH)
    for seed in seeds:
        for epoch in epochs:
            key = f"s{seed}-e{epoch}"
            if key in known:
                continue
            path = hs.policy_dir(run_dir, ARM, seed, epoch)
            if not path.exists():
                continue
            scorer = PolicyScorer(load_policy(path, device=device), device)
            rows = [
                {"behaviour": t.behaviour, **got}
                for t in tasks
                if (got := score(t, scorer)) is not None
            ]
            known[key] = summarise(rows, tolerance)
            out.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
            cell = known[key]
            log(
                f"RESULT deficit {key}: median {cell['per_step_deficit']['median']} nats, "
                f"{cell['over_budget_factor_at_median']}x over, top-1 {cell['top1_share']}, "
                f"within budget {cell['within_budget']}/{cell['witnesses']}"
            )
    return out


def solve(run_dir: Path, device: str, seed: int, epoch: int, log) -> Path:
    """The acceptance test: the search, once, at the chosen epoch."""
    out = run_dir / f"solve-s{seed}-e{epoch}.json"
    if out.exists():
        log(f"skip solve {out.name}")
        return out
    if device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    path = hs.policy_dir(run_dir, ARM, seed, epoch)
    scorer = PolicyScorer(load_policy(path, device=device), device)
    system = System(
        name=f"world-arm-s{seed}-e{epoch}",
        repo_id="local",
        revision=BEST.revision,
        beam_width=BEST.beam_width,
        max_depth=BEST.max_depth,
        guard=BEST.guard,
        library=BEST.library,
        measured="clear-world-v3 (#161): v4 + full world-v3, this run",
        max_state_atoms=BEST.max_state_atoms,
    )
    rows = solve_tasks(world_eval(), scorer, system)
    report = run_summary(rows)
    co.write_json(
        out,
        {
            "seed": seed,
            "epoch": epoch,
            "system": system.name,
            "summary": {
                "tasks": report.tasks,
                "solved": report.solved,
                "coverage": report.coverage,
                "seconds": report.seconds,
                "executions": report.executions,
                "memorised": report.memorised,
            },
            "rows": [r.record() for r in rows],
            "host": co.host(),
            "provenance": co.provenance(),
        },
    )
    log(f"RESULT solve s{seed} e{epoch}: {report.line()}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("train", "deficit", "solve"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--epochs", default=",".join(str(e) for e in range(1, EPOCHS + 1))
    )
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with (args.run_dir / "log.txt").open("a") as handle:
            handle.write(stamped + "\n")

    seeds = [int(s) for s in args.seeds.split(",") if s]
    epochs = [int(e) for e in args.epochs.split(",") if e]
    if args.stage == "train":
        train(args.run_dir, args.device, seeds, log)
    elif args.stage == "deficit":
        deficit(args.run_dir, "cpu", seeds, epochs, log)
    else:
        solve(args.run_dir, "cpu", args.seed, args.epoch, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
