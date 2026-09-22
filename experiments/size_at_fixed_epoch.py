"""model-size-at-fixed-epoch: does policy size buy behaviours once the checkpoint is not CE-selected?

compute-optimal-mark-28 (#101) found the size axis flat, 1m / 2m / 10m all
near 0.18 coverage of the 300-task cohort, but every arm was selected on
validation cross-entropy, which stopped them at epoch 1 or 2;
epoch-selection-in-behaviours (#154) then showed the 1m arm at epoch 3
verifies 11 behaviours against 6 at its CE-selected epoch. The size question
has never been asked of a checkpoint that was allowed to train. This driver
asks it on the staged text of staged-witness-reauthoring (#155), the training
data behind `solve.BEST`: the same rows, optimiser, batch and seeds as the
staged arm, three sizes, every epoch snapshotted, every snapshot searched.

Stages, each skipped where its output exists so a run resumes::

    uv run python experiments/size_at_fixed_epoch.py train  --run-dir checkpoints/size-20260922 --sizes 1m,2m,10m
    uv run python experiments/size_at_fixed_epoch.py time   --run-dir checkpoints/size-20260922 --epochs 3,4,5,6
    uv run python experiments/size_at_fixed_epoch.py report --run-dir checkpoints/size-20260922 --out reports/size-20260922

Learning rates are #101's probe choices, read from its `lr_probe.json`, so the
recipe is #101's at every size. Search is beam 64 x depth 48, guard `both`,
library off (the published cohort protocol), with the state-size guard of
beam-state-size-unbounded (#157) at `solve.BEST`'s cap so a runaway staged
policy costs seconds, not minutes. Cells are written in the shape
`experiments/step3_verify_solves.py` reads; `report` draw-tests every distinct
solved program through it and counts verified behaviours per cell.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
import experiments.staged_arms as sa  # noqa: E402
import experiments.step3_verify_solves as verify  # noqa: E402
from wandering_light.solve import BEST  # noqa: E402
from wandering_light.stack_beam import beam_solve  # noqa: E402
from wandering_light.stack_pbe import CATALOGUE  # noqa: E402
from wandering_light.stack_pbe_eval import grade  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    PolicyScorer,
    load_policy,
    train_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LR_PROBE = REPO / "reports" / "compute-optimal-mark28-20260903" / "lr_probe.json"
SIZES = ("1m", "2m", "10m")
SEEDS = (0, 1, 2)
EPOCHS = 6
TEXT_ARM = "staged"  # the #155 arm behind solve.BEST
VERIFIED = ("genuine", "equals_catalogue_witness")


def learning_rates() -> dict[str, float]:
    return json.loads(LR_PROBE.read_text())["chosen"]


def train(run_dir: Path, tasks, device: str, sizes, seeds, log) -> None:
    lrs = learning_rates()
    fit = held = facts = None
    for size in sizes:
        for seed in seeds:
            if hs.policy_dir(run_dir, size, seed, EPOCHS).exists():
                log(f"skip train {size} s{seed} (snapshots exist)")
                continue
            if fit is None:
                fit, held, facts = sa.arm_rows(TEXT_ARM, tasks, keep_contrived=False)
                log(
                    f"RESULT rows {TEXT_ARM}: {json.dumps(facts)} validation {len(held)}"
                )
            arm = co.ARMS[size]
            manifest = {
                "arm": TEXT_ARM,
                "policy_arm": size,
                "epochs": EPOCHS,
                "learning_rate": lrs[size],
                "seed": seed,
                "sample_seed": sa.SAMPLE_SEED,
                "keep_contrived": False,
                "side_file": str(sa.SIDE_FILE.relative_to(REPO)),
                **facts,
                "provenance": co.provenance(),
            }
            started = time.perf_counter()
            _model, state = train_policy(
                fit,
                held,
                co.architecture(size),
                epochs=EPOCHS,
                learning_rate=lrs[size],
                batch_size=arm["batch"],
                validation_batch_size=arm["val_batch"],
                seed=seed,
                device=device,
                epoch_end=hs.snapshotter(run_dir, size, seed, manifest),
            )
            co.write_json(run_dir / "training" / f"{size}-s{seed}.json", state)
            curve = " ".join(
                f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}"
                for r in state["history"]
            )
            log(
                f"RESULT train {size} s{seed}: [{curve}] {time.perf_counter() - started:.0f}s"
            )


def time_cell(run_dir: Path, tasks, size: str, seed: int, epoch: int, device, log):
    out = hs.cell_path(run_dir, size, seed, device, epoch, co.BEAM_WIDTH, co.MAX_DEPTH)
    if out.exists():
        log(f"skip time {out.name}")
        return out
    if device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    scorer = PolicyScorer(
        load_policy(hs.policy_dir(run_dir, size, seed, epoch), device=device), device
    )
    rows = []
    gc.freeze()
    started = time.perf_counter()
    for index, task in co.cohort(tasks):
        result = beam_solve(
            task,
            scorer,
            beam_width=co.BEAM_WIDTH,
            max_depth=co.MAX_DEPTH,
            guard=co.GUARD,
            max_state_atoms=BEST.max_state_atoms,
        )
        score = grade(result.program, task) if result.program else None
        rows.append(
            {
                "task_index": index,
                "behaviour": task.behaviour,
                "length": task.length,
                "bucket": co.bucket_of(task.length),
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
    cell_seconds = time.perf_counter() - started
    co.write_json(
        out,
        {
            "arm": size,
            "text_arm": TEXT_ARM,
            "seed": seed,
            "device": device,
            "policy_epoch": epoch,
            "beam_width": co.BEAM_WIDTH,
            "max_depth": co.MAX_DEPTH,
            "guard": co.GUARD,
            "max_state_atoms": BEST.max_state_atoms,
            "cpu_threads": co.CPU_THREADS,
            "eval_corpus": co.primary_corpus(),
            "host": co.host(),
            "provenance": co.provenance(),
            "cell_seconds": cell_seconds,
            "rows": rows,
        },
    )
    solved = sum(r["solved"] for r in rows)
    log(
        f"RESULT time {size} s{seed} e{epoch}: {solved}/{len(rows)} solved, "
        f"{sum(r['seconds'] for r in rows):.0f}s timed, "
        f"{max(r['seconds'] for r in rows):.1f}s slowest"
    )
    return out


def verified_programs(run_dir: Path, cells: Sequence[dict[str, Any]], log) -> dict:
    path = run_dir / "verified.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    by_name = {b.name: b for b in CATALOGUE}
    for cell in cells:
        for row in cell["rows"]:
            if not row["solved"]:
                continue
            key = f"{row['behaviour']} :: {' '.join(row['program'])}"
            if key in known:
                continue
            known[key] = verify.check(
                tuple(row["program"]), by_name[row["behaviour"]], verify.DRAWS, 134
            )
            log(f"{known[key]['verdict']:24s} {key}")
            path.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
    return known


def report(run_dir: Path, out_dir: Path, log) -> None:
    cells = [
        json.loads(p.read_text()) for p in sorted((run_dir / "timing").glob("*.json"))
    ]
    verdicts = verified_programs(run_dir, cells, log)
    table = []
    for cell in cells:
        solved = [r for r in cell["rows"] if r["solved"]]
        verified = {
            r["behaviour"]
            for r in solved
            if verdicts[f"{r['behaviour']} :: {' '.join(r['program'])}"]["verdict"]
            in VERIFIED
        }
        table.append(
            {
                "size": cell["arm"],
                "seed": cell["seed"],
                "epoch": cell["policy_epoch"],
                "device": cell["device"],
                "tasks": len(solved),
                "graded_behaviours": len({r["behaviour"] for r in solved}),
                "verified_behaviours": len(verified),
                "verified": sorted(verified),
                "verified_by_bucket": {
                    b: len(
                        {r["behaviour"] for r in solved if r["bucket"] == b} & verified
                    )
                    for b in verify.BUCKETS
                },
                "timed_seconds": sum(r["seconds"] for r in cell["rows"]),
                "slowest_task_seconds": max(r["seconds"] for r in cell["rows"]),
                "executions": sum(r["executions"] for r in cell["rows"]),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cells.json").write_text(json.dumps(table, indent=2) + "\n")
    grid: dict[tuple[str, int], list[int]] = defaultdict(list)
    tasks_grid: dict[tuple[str, int], list[int]] = defaultdict(list)
    for t in table:
        grid[t["size"], t["epoch"]].append(t["verified_behaviours"])
        tasks_grid[t["size"], t["epoch"]].append(t["tasks"])
    sizes = [s for s in SIZES if any(k[0] == s for k in grid)]
    epochs = sorted({k[1] for k in grid})
    lines = [
        "Verified behaviours of 43, mean over seeds (per seed in parentheses); "
        "graded tasks of 300 in the second table.",
        "",
        "| epoch | " + " | ".join(sizes) + " |",
        "|---|" + "---:|" * len(sizes),
    ]
    for e in epochs:
        cells_ = [
            f"{mean(grid[s, e]):.2f} ({' / '.join(map(str, grid[s, e]))})"
            if grid[s, e]
            else ""
            for s in sizes
        ]
        lines.append(f"| {e} | " + " | ".join(cells_) + " |")
    lines += [
        "",
        "| epoch | " + " | ".join(sizes) + " |",
        "|---|" + "---:|" * len(sizes),
    ]
    for e in epochs:
        cells_ = [
            f"{mean(tasks_grid[s, e]):.1f}" if tasks_grid[s, e] else "" for s in sizes
        ]
        lines.append(f"| {e} | " + " | ".join(cells_) + " |")
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("train", "time", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--epochs", default=",".join(str(e) for e in range(1, EPOCHS + 1))
    )
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.run_dir / "log.txt"

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with logfile.open("a") as handle:
            handle.write(stamped + "\n")

    sizes = [s for s in args.sizes.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    epochs = [int(e) for e in args.epochs.split(",") if e]
    if args.stage == "report":
        report(args.run_dir, args.out or args.run_dir, log)
        return 0
    tasks = co.corpus()
    if args.stage == "train":
        train(args.run_dir, tasks, args.device, sizes, seeds, log)
    else:
        for size in sizes:
            for seed in seeds:
                for epoch in epochs:
                    time_cell(args.run_dir, tasks, size, seed, epoch, args.device, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
