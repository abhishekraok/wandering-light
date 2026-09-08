"""Compute-optimal policy size on `mark-28` / `stack_pbe_v4`.

compute-optimal-mark-28 (#101): the #80 question on the stack machine. Arms are
rows in ``ARMS``; every stage writes its provenance (commit, dirty flag, corpus
digest, seed, W&B URL) beside its result and nothing here refuses to run on a
mismatch. Stages, each skipped when its output exists so a run resumes::

    uv run python experiments/compute_optimal_mark28.py campaign --run-dir checkpoints/mark28-co

    ... or one at a time:
    probe-lr   one epoch per candidate LR per arm, seed 0; picks by validation CE
    train      the chosen LR, 12 epochs, best-validation weights, both seeds
    time       beam search over the eval cohort, one (arm, seed, device) cell
    replay     re-run recorded hits under the current counters (see `replay`)
    archive    log each checkpoint as a W&B artifact on its training run
    aggregate  per-bucket AUC tables, paired over seeds

Difficulty is witness length, an upper bound -- v4 has no certified distance.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from wandering_light.compute_optimal import log_budget_auc, wall_clock_auc
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_pbe import Task, load
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import (
    MANIFEST_FILE,
    PolicyScorer,
    load_policy,
    save_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

REPO = Path(__file__).resolve().parents[1]
CORPUS_MANIFEST = REPO / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
WANDB_ENTITY = "abhishekraok-na"
WANDB_PROJECT = "wandering-light-rollout-policy"

# arm -> (architecture, LR candidates for the probe, train batch, validation batch)
ARMS: dict[str, dict[str, Any]] = {
    "1m": {
        "dims": (128, 320, 640),
        "lrs": (1e-3, 2e-3, 4e-3),
        "batch": 256,
        "val_batch": 512,
    },
    "2m": {
        "dims": (224, 576, 1152),
        "lrs": (5e-4, 1e-3, 2e-3),
        "batch": 256,
        "val_batch": 512,
    },
    "10m": {
        "dims": (384, 1024, 2048),
        "lrs": (2.5e-4, 5e-4, 1e-3),
        "batch": 256,
        "val_batch": 256,
    },
}
SEEDS = (0, 1)
DEVICES = ("cpu", "cuda")
EPOCHS = 12
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))
TASKS_PER_BUCKET = 60
COHORT_SEED = 17
BEAM_WIDTH = 64
MAX_DEPTH = 48
GUARD = "both"
# Wall-clock range fixed before any cell ran; it brackets a beam hit on either
# device. The executions range was set with the counter fix (see `replay`): it
# starts below the cheapest possible hit and ends at 2^19, above the hard cap of
# depth 48 x beam 64 x 28 ops x 6 examples, so no hit is censored on that axis.
WALL_RANGE = (0.01, 100.0)
EXECUTIONS_RANGE = (16, 524_288)
CPU_THREADS = 8


# ---------------------------------------------------------------------------
# Provenance and data
# ---------------------------------------------------------------------------


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    manifest = json.loads(CORPUS_MANIFEST.read_text())
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "corpus": {
            "name": manifest["corpus"],
            "hub": manifest["hub"],
            "splits": {k: v["sha256"] for k, v in manifest["splits"].items()},
        },
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }


def host() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def corpus() -> list[Task]:
    return load(fetch_corpus(CORPUS_MANIFEST))


def bucket_of(length: int) -> str:
    for low, high in BUCKETS:
        if low <= length <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    raise ValueError(length)


def cohort(tasks: list[Task]) -> list[tuple[int, Task]]:
    """The same eval tasks for every cell: seed-17 shuffle, first N per bucket."""
    eval_tasks = [(i, t) for i, t in enumerate(tasks) if t.split == "eval"]
    random.Random(COHORT_SEED).shuffle(eval_tasks)
    taken: dict[str, list] = defaultdict(list)
    for index, task in eval_tasks:
        bucket = bucket_of(task.length)
        if len(taken[bucket]) < TASKS_PER_BUCKET:
            taken[bucket].append((index, task))
    return [pair for low, high in BUCKETS for pair in taken[bucket_of(low)]]


TRAIN_TASK_CAP = 0


def training_rows(tasks: list[Task]):
    train = [t for t in tasks if t.split == "train"]
    if TRAIN_TASK_CAP:
        train = random.Random(COHORT_SEED).sample(train, TRAIN_TASK_CAP)
    fit = witness_rows([t for t in train if not validation_skeleton(t)])
    validation = witness_rows([t for t in train if validation_skeleton(t)])
    return fit, validation


def architecture(arm: str) -> PolicyArchitecture:
    return PolicyArchitecture(*ARMS[arm]["dims"])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def wandb_run(name: str, config: dict[str, Any], tags: list[str], enabled: bool):
    if not enabled:
        return None
    import wandb

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=name,
        config=config,
        tags=["stack_pbe_v4", "mark-28", *tags],
        mode="online",
    )


def _fit(arm, seed, lr, epochs, rows, device, run_name, use_wandb, log):
    run = wandb_run(
        run_name,
        {
            "arm": arm,
            "seed": seed,
            "learning_rate": lr,
            "epochs": epochs,
            "architecture": architecture(arm).to_config(),
            "train_rows": len(rows[0]),
            "validation_rows": len(rows[1]),
        },
        ["lr-probe" if epochs == 1 else "policy"],
        use_wandb,
    )
    try:
        model, state = train_policy(
            rows[0],
            rows[1],
            architecture(arm),
            epochs=epochs,
            learning_rate=lr,
            batch_size=ARMS[arm]["batch"],
            validation_batch_size=ARMS[arm]["val_batch"],
            seed=seed,
            device=device,
            epoch_callback=lambda row: (
                log(
                    f"  {arm} s{seed} lr={lr:g} epoch {row['epoch']} "
                    f"val_ce={row['validation_cross_entropy']:.4f} {row['seconds']:.0f}s"
                ),
                run.log(row, step=row["epoch"]) if run else None,
            ),
        )
        return model, state, (run.url if run else None)
    finally:
        if run:
            run.finish()


def probe_lr(run_dir: Path, tasks, device, arms, use_wandb, log) -> dict[str, float]:
    """One epoch per candidate LR; the grid extends downward while its edge wins."""
    out = run_dir / "lr_probe.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    chosen = previous.get("chosen", {})
    probes = previous.get("probes", [])
    if all(arm in chosen for arm in arms):
        return chosen
    rows = training_rows(tasks)

    def probe(arm: str, lr: float) -> float:
        for done in probes:
            if done["arm"] == arm and done["learning_rate"] == lr:
                return done["validation_cross_entropy"]
        _, state, url = _fit(
            arm, 0, lr, 1, rows, device, f"mark28-probe-{arm}-lr{lr:g}", use_wandb, log
        )
        probes.append(
            {
                "arm": arm,
                "learning_rate": lr,
                "wandb_url": url,
                "validation_cross_entropy": state["best_validation_cross_entropy"],
                "seconds": state["training_seconds"],
            }
        )
        return probes[-1]["validation_cross_entropy"]

    for arm in arms:
        if arm in chosen:
            continue
        scores = {lr: probe(arm, lr) for lr in ARMS[arm]["lrs"]}
        # A win at the lowest candidate says nothing about where the optimum
        # is; halve until an interior point wins, three halvings at most.
        for _ in range(3):
            lowest = min(scores)
            if min(scores, key=scores.get) != lowest:
                break
            scores[lowest / 2] = probe(arm, lowest / 2)
        chosen[arm] = min(scores, key=scores.get)
        log(f"RESULT probe-lr {arm} -> {chosen[arm]:g} (grid {sorted(scores)})")
        write_json(
            out, {"chosen": chosen, "probes": probes, "provenance": provenance()}
        )
    return chosen


def train(run_dir: Path, tasks, device, arms, seeds, chosen, use_wandb, log) -> None:
    rows = None
    for arm in arms:
        for seed in seeds:
            out = run_dir / "policies" / f"{arm}-s{seed}"
            if out.exists():
                continue
            rows = rows or training_rows(tasks)
            lr = chosen[arm]
            model, state, url = _fit(
                arm,
                seed,
                lr,
                EPOCHS,
                rows,
                device,
                f"mark28-{arm}-s{seed}-lr{lr:g}",
                use_wandb,
                log,
            )
            save_policy(
                out,
                model,
                trainer_state=state,
                wandb_url=url,
                manifest={
                    "arm": arm,
                    "seed": seed,
                    "learning_rate": lr,
                    "epochs": EPOCHS,
                    "validation": "1 in 10 train skeletons by sha256",
                    "provenance": provenance(),
                },
            )
            log(
                f"RESULT train {arm} s{seed} best_val_ce={state['best_validation_cross_entropy']:.4f} "
                f"epoch {state['best_epoch']} params={state['parameters']} {url}"
            )


def time_cell(run_dir: Path, tasks, arm, seed, device, log) -> Path:
    out = run_dir / "timing" / f"{arm}-s{seed}-{device}.json"
    if out.exists():
        return out
    if device == "cpu":
        torch.set_num_threads(CPU_THREADS)
    scorer = PolicyScorer(
        load_policy(run_dir / "policies" / f"{arm}-s{seed}", device=device), device
    )
    rows = []
    started = time.perf_counter()
    for index, task in cohort(tasks):
        result = beam_solve(
            task, scorer, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH, guard=GUARD
        )
        score = grade(result.program, task) if result.program else None
        rows.append(
            {
                "task_index": index,
                "behaviour": task.behaviour,
                "length": task.length,
                "bucket": bucket_of(task.length),
                "examples": task.examples,
                "solved": bool(score and score.solved),
                "shown_fit": bool(score and score.shown),
                "program": list(result.program) if result.program else None,
                "termination": result.termination,
                "seconds": result.seconds,
                "executions": result.executions,
                "fuel": result.fuel,
                "expansions": result.expansions,
                "depth": result.depth,
            }
        )
    solved = sum(r["solved"] for r in rows)
    write_json(
        out,
        {
            "arm": arm,
            "seed": seed,
            "device": device,
            "beam_width": BEAM_WIDTH,
            "max_depth": MAX_DEPTH,
            "guard": GUARD,
            "cpu_threads": CPU_THREADS,
            "host": host(),
            "provenance": provenance(),
            "cell_seconds": time.perf_counter() - started,
            "rows": rows,
        },
    )
    log(
        f"RESULT time {arm} s{seed} {device}: {solved}/{len(rows)} solved, "
        f"{time.perf_counter() - started:.0f}s"
    )
    return out


def _executions(row: dict[str, Any]) -> int:
    """The corrected counter; a cell timed before it existed needs `replay`."""
    if "executions" not in row:
        raise KeyError(f"task {row['task_index']} has no executions; run replay")
    return row["executions"]


def replay(run_dir: Path, tasks, log) -> None:
    """Re-run every hit under the current counters and record its executions.

    The campaign's cells were timed with a counter that charged 28 per node
    regardless of legality and summed fuel only for retained successors (Codex
    review on the PR). A miss contributes nothing to a log-budget AUC and a hit
    takes well under a second, so the hits are replayed rather than the cells:
    same weights, same device, same search. The program must come back
    identical; the row then gains ``executions``, ``fuel_replayed`` and
    ``seconds_replayed`` beside the values as first recorded.
    """
    by_index = dict(cohort(tasks))
    for path in sorted((run_dir / "timing").glob("*.json")):
        cell = json.loads(path.read_text())
        if cell.get("replay"):
            continue
        device = cell["device"]
        if device == "cpu":
            torch.set_num_threads(CPU_THREADS)
        policy = load_policy(
            run_dir / "policies" / f"{cell['arm']}-s{cell['seed']}", device=device
        )
        scorer = PolicyScorer(policy, device)
        started = time.perf_counter()
        hits = 0
        mismatches = []
        for row in cell["rows"]:
            if row["program"] is None:
                continue
            hits += 1
            result = beam_solve(
                by_index[row["task_index"]],
                scorer,
                beam_width=cell["beam_width"],
                max_depth=cell["max_depth"],
                guard=cell["guard"],
            )
            if list(result.program or ()) != row["program"]:
                mismatches.append(row["task_index"])
                continue
            row["executions"] = result.executions
            row["fuel_replayed"] = result.fuel
            row["seconds_replayed"] = result.seconds
        cell["replay"] = {
            "hits": hits,
            "mismatches": mismatches,
            "seconds": time.perf_counter() - started,
            "provenance": provenance(),
            "note": (
                "executions and fuel_replayed count per op executed on one example "
                "stack; op_applications (28 per node) and fuel (retained successors "
                "only) are the values as first recorded"
            ),
        }
        write_json(path, cell)
        log(
            f"RESULT replay {path.stem}: {hits} hits, {len(mismatches)} mismatches, "
            f"{cell['replay']['seconds']:.0f}s"
        )


def archive(run_dir: Path, log) -> None:
    """Log each checkpoint as a W&B model artifact on its own training run."""
    import wandb

    out = run_dir / "archive.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    for path in sorted(p for p in (run_dir / "policies").iterdir() if p.is_dir()):
        if path.name in done:
            continue
        manifest = json.loads((path / MANIFEST_FILE).read_text())
        run_id = manifest["wandb_url"].rstrip("/").rsplit("/", 1)[-1]
        run = wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT, id=run_id, resume="must"
        )
        try:
            artifact = wandb.Artifact(
                name=f"mark28-policy-{path.name}",
                type="model",
                metadata={
                    "arm": manifest["arm"],
                    "seed": manifest["seed"],
                    "provenance": manifest["provenance"],
                },
            )
            artifact.add_dir(str(path))
            logged = run.log_artifact(artifact)
            logged.wait()
            done[path.name] = {
                "artifact": f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact.name}:{logged.version}",
                "wandb_url": manifest["wandb_url"],
            }
        finally:
            run.finish()
        write_json(out, done)
        log(f"RESULT archive {path.name} -> {done[path.name]['artifact']}")


def aggregate(run_dir: Path, log) -> Path:
    cells = [
        json.loads(p.read_text()) for p in sorted((run_dir / "timing").glob("*.json"))
    ]
    arms = sorted({c["arm"] for c in cells}, key=lambda a: list(ARMS).index(a))
    buckets = [bucket_of(low) for low, _ in BUCKETS]
    table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for device in DEVICES:
        for bucket in buckets:
            for arm in arms:
                per_seed = {}
                for cell in (
                    c for c in cells if c["device"] == device and c["arm"] == arm
                ):
                    rows = [r for r in cell["rows"] if r["bucket"] == bucket]
                    if not rows:
                        continue
                    per_seed[cell["seed"]] = {
                        "wall": wall_clock_auc(
                            [r["seconds"] if r["solved"] else None for r in rows],
                            min_seconds=WALL_RANGE[0],
                            max_seconds=WALL_RANGE[1],
                        ).normalized_area,
                        "ops": log_budget_auc(
                            [_executions(r) if r["solved"] else None for r in rows],
                            min_budget=EXECUTIONS_RANGE[0],
                            max_budget=EXECUTIONS_RANGE[1],
                            unit="op_executions",
                        ).normalized_area,
                        "solve_rate": mean(r["solved"] for r in rows),
                        "shown_rate": mean(r["shown_fit"] for r in rows),
                        # define-cost-to-clear (#95): the pair, never collapsed.
                        # Cost is over the covered subset; total is what the
                        # cell actually spent, misses included.
                        "cost_covered_seconds": sum(
                            r["seconds"] for r in rows if r["solved"]
                        ),
                        "total_seconds": sum(r["seconds"] for r in rows),
                        "tasks": len(rows),
                    }
                if per_seed:
                    table.setdefault(device, {}).setdefault(bucket, {})[arm] = per_seed
    reference = arms[0]
    summary = {}
    lines = []
    clear = {}
    for cell in cells:
        rows = cell["rows"]
        clear.setdefault(cell["device"], {}).setdefault(cell["arm"], {})[
            cell["seed"]
        ] = {
            "coverage": mean(r["solved"] for r in rows),
            "cost_covered_seconds": sum(r["seconds"] for r in rows if r["solved"]),
            "total_seconds": sum(r["seconds"] for r in rows),
            "tasks": len(rows),
        }
    lines.append(
        "### Cost to clear the cohort (#95 pair): coverage, wall-clock over the covered subset, total spent; mean over seeds\n"
    )
    lines.append(
        "| device | "
        + " | ".join(f"{a} coverage | {a} covered s | {a} total s" for a in arms)
        + " |"
    )
    lines.append("|---|" + "---|" * (3 * len(arms)))
    for device, by_arm in clear.items():
        out = []
        for arm in arms:
            seeds = by_arm.get(arm, {})
            out += [
                f"{mean(s[k] for s in seeds.values()):.{'3' if k == 'coverage' else '0'}f}"
                if seeds
                else "—"
                for k in ("coverage", "cost_covered_seconds", "total_seconds")
            ]
        lines.append(f"| {device} | " + " | ".join(out) + " |")
    for device, by_bucket in table.items():
        lines.append(
            f"\n### {device}: normalized AUC, mean over seeds (seed SD); paired minus {reference}\n"
        )
        lines.append(
            "| bucket | "
            + " | ".join(f"{a} wall | {a} ops | {a} solve" for a in arms)
            + " |"
        )
        lines.append("|---|" + "---|" * (3 * len(arms)))
        for bucket, by_arm in by_bucket.items():
            cells_out = []
            for arm in arms:
                seeds = by_arm.get(arm, {})
                if not seeds:
                    cells_out += ["—"] * 3
                    continue
                stats = {}
                for metric in ("wall", "ops", "solve_rate"):
                    values = [s[metric] for s in seeds.values()]
                    paired = [
                        seeds[k][metric] - by_arm[reference][k][metric]
                        for k in seeds
                        if k in by_arm.get(reference, {})
                    ]
                    stats[metric] = {
                        "mean": mean(values),
                        "sd": pstdev(values) if len(values) > 1 else 0.0,
                        "paired_mean": mean(paired) if paired else None,
                        "paired_sd": pstdev(paired) if len(paired) > 1 else 0.0,
                        "seeds": len(values),
                    }
                summary.setdefault(device, {}).setdefault(bucket, {})[arm] = stats
                cells_out += [
                    f"{stats[m]['mean']:.3f} ({stats[m]['sd']:.3f})"
                    + (
                        f" Δ{stats[m]['paired_mean']:+.3f}"
                        if arm != reference and stats[m]["paired_mean"] is not None
                        else ""
                    )
                    for m in ("wall", "ops", "solve_rate")
                ]
            lines.append(f"| {bucket} | " + " | ".join(cells_out) + " |")
    out = run_dir / "aggregate.json"
    write_json(
        out,
        {
            "wall_range_seconds": WALL_RANGE,
            "executions_range": EXECUTIONS_RANGE,
            "reference_arm": reference,
            "per_seed": table,
            "summary": summary,
            "cost_to_clear": clear,
            "provenance": provenance(),
        },
    )
    (run_dir / "tables.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"RESULT aggregate {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    global EPOCHS, TASKS_PER_BUCKET, TRAIN_TASK_CAP
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=(
            "probe-lr",
            "train",
            "time",
            "replay",
            "archive",
            "aggregate",
            "campaign",
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--devices", nargs="+", default=list(DEVICES), choices=DEVICES)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tasks-per-bucket", type=int, default=TASKS_PER_BUCKET)
    parser.add_argument("--train-tasks", type=int, default=0, help="cap, 0 = all")
    args = parser.parse_args(argv)
    EPOCHS, TASKS_PER_BUCKET, TRAIN_TASK_CAP = (
        args.epochs,
        args.tasks_per_bucket,
        args.train_tasks,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (args.run_dir / "runner.log").open("a")

    def log(line: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        print(f"{stamp} {line}", flush=True)
        log_file.write(f"{stamp} {line}\n")
        log_file.flush()

    tasks = corpus() if args.stage not in ("aggregate", "archive") else None
    try:
        if args.stage in ("probe-lr", "campaign"):
            chosen = probe_lr(
                args.run_dir,
                tasks,
                args.train_device,
                args.arms,
                not args.no_wandb,
                log,
            )
        if args.stage in ("train", "campaign"):
            chosen = json.loads((args.run_dir / "lr_probe.json").read_text())["chosen"]
            train(
                args.run_dir,
                tasks,
                args.train_device,
                args.arms,
                args.seeds,
                chosen,
                not args.no_wandb,
                log,
            )
        if args.stage in ("time", "campaign"):
            for device in args.devices:
                for arm in args.arms:
                    for seed in args.seeds:
                        time_cell(args.run_dir, tasks, arm, seed, device, log)
        if args.stage == "replay":
            replay(args.run_dir, tasks, log)
        if args.stage == "archive":
            archive(args.run_dir, log)
        if args.stage in ("aggregate", "campaign"):
            aggregate(args.run_dir, log)
    except Exception as error:
        log(f"FAILED {args.stage}: {error!r}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
