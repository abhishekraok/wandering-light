"""Hindsight sub-targets: does the target a training row carries bind the policy?

`compute_optimal_mark-28` (#101) found the label set binds -- 313,560 rows come
from **175 distinct programs**, and 1m / 2m / 10m are flat. Every row a witness
replay produces carries the *same* target, the task's final output, so a task of
length n gives the policy n rows that differ only in how far along the witness
they are. A policy that memorises "this behaviour's target means replay this
witness" fits all n of them without ever comparing the state it is in against the
state it is asked for.

This script changes one thing and nothing else. Arm A is the #101/#134 recipe.
Arm B relabels each row's **target** with a state drawn uniformly from the ones
still ahead of it on the same witness, so `(state_i, s_j)` asks "you are here,
get to there" and the label, the witness's op at i, is still what the author did.
The op distribution, the row count, the batch size, the optimiser, the seeds and
the validation set are identical, so a difference is the target distribution and
not compute. Sub-targets are restricted to mark-free states because every corpus
target is mark-free and `beam_solve`'s close guard requires it.

Arm C draws from every state ahead, not only the mark-free ones: only 19.5% of
the states along a v4 witness are mark-free, so arm B's draw is sparse and its
nearest sub-target a median 15 ops away. Arm C is denser and also asks "finish
building this quotation", at the price of targets no eval task ever asks for.

Stages, each skipped when its output exists so a run resumes::

    uv run python experiments/hindsight_subtargets.py train --run-dir checkpoints/hindsight
    uv run python experiments/hindsight_subtargets.py time  --run-dir checkpoints/hindsight --epochs 3,4,5
    uv run python experiments/hindsight_subtargets.py report --run-dir checkpoints/hindsight --out reports/...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from wandering_light.stack_beam import beam_solve  # noqa: E402
from wandering_light.stack_machine import (  # noqa: E402
    DEFAULT_FUEL_LIMIT,
    Mark,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME  # noqa: E402
from wandering_light.stack_pbe_eval import grade  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    TARGET_CAP,
    PolicyScorer,
    Row,
    encode_item,
    encode_stack,
    item_array,
    load_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_machine import Value
    from wandering_light.stack_pbe import Task

ARM = "1m"
SEEDS = (0, 1, 2)
EPOCHS = 6
LEARNING_RATE = 1.25e-4  # #101's probe choice for the 1m arm; see its lr_probe.json
BATCH = 256
VAL_BATCH = 512
ARMS = ("witness", "hindsight", "hindsight-all")
MIXED_FINAL_PROBABILITY = 0.5
ROW_SEED = 20260919  # the sub-target draw; fixed so two arms differ only in policy seed


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def replay_states(task: Task) -> list[tuple[tuple[Value, ...], ...]]:
    """Every product state along the witness, ``s_0`` (inputs) to ``s_n``.

    ``stack_policy.replay_witness`` yields the same walk but stops at ``s_{n-1}``
    because it pairs each state with the op that leaves it. The final state is
    the task's output and is exactly what a sub-target is drawn against.
    """
    states = [tuple(parse_stack(text, basis=BASIS) for text in task.inputs)]
    for name in task.program:
        moved = []
        for stack in states[-1]:
            outcome = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                raise ValueError(f"{task.behaviour}: witness refused at {name}: {outcome}")
            moved.append(outcome[0])
        states.append(tuple(moved))
    return states


def _mark_free(product: tuple[tuple[Value, ...], ...]) -> bool:
    return not any(isinstance(v, Mark) for stack in product for v in stack)


def subtarget_rows(
    tasks: Sequence[Task], *, final_probability: float, seed: int, mark_free: bool = True
) -> list[Row]:
    """One row per (task, witness step), its target drawn from the states ahead.

    Exactly ``witness_rows``' shape and label; only the target differs. The draw
    is uniform over the states strictly ahead of the row's own, with the final
    output always among them, so the row count and the label distribution are
    those of the control arm.

    ``mark_free`` restricts the draw to targets that look like a corpus target.
    It costs a lot of density: only 19.5% of the states along a v4 witness are
    mark-free, because a witness spends most of its length inside an open ``[``,
    and the nearest mark-free state ahead of a row is a median 15 ops away. The
    unrestricted arm keeps every state and so also asks "finish building this
    quotation", which is the sub-skill the 6-10 op bucket fails at; what it gives
    up is that most of its targets are shapes no eval task ever asks for.
    """
    rng = random.Random(seed)
    rows: list[Row] = []
    for index, task in enumerate(tasks):
        states = replay_states(task)
        last = len(states) - 1
        ahead = [
            j for j in range(1, last + 1) if not mark_free or _mark_free(states[j])
        ]
        if last not in ahead:  # every corpus output is mark-free; keep the invariant loud
            raise ValueError(f"{task.behaviour}: final state carries a mark")
        for step, name in enumerate(task.program):
            options = [j for j in ahead if j > step]
            if final_probability and rng.random() < final_probability:
                pick = last
            else:
                pick = rng.choice(options) if options else last
            target = [encode_stack(stack, TARGET_CAP) for stack in states[pick]]
            items = [
                encode_item(state, tokens)
                for state, tokens in zip(states[step], target, strict=True)
            ]
            rows.append(
                Row(item_array(items), OP_INDEX[name], index, step, task.length - step)
            )
    return rows


def arm_rows(arm: str, tasks: Sequence[Task]) -> tuple[list[Row], list[Row]]:
    """``(fit, validation)`` for one arm.

    Validation is the control arm's rows in every arm: it is the pinned v4
    holdout with final targets, the same quantity #134 reports, so the three
    arms' validation cross-entropies compare and none of them is scored on its
    own training condition.
    """
    train = [t for t in tasks if t.split == "train"]
    fit_tasks = [t for t in train if not validation_skeleton(t)]
    held = [t for t in train if validation_skeleton(t)]
    if arm == "witness":
        fit = witness_rows(fit_tasks)
    elif arm == "hindsight":
        fit = subtarget_rows(fit_tasks, final_probability=0.0, seed=ROW_SEED)
    elif arm == "hindsight-all":
        fit = subtarget_rows(
            fit_tasks, final_probability=0.0, seed=ROW_SEED, mark_free=False
        )
    elif arm == "mixed":
        fit = subtarget_rows(
            fit_tasks, final_probability=MIXED_FINAL_PROBABILITY, seed=ROW_SEED
        )
    else:
        raise ValueError(f"unknown arm {arm!r}; choose one of {ARMS}")
    return fit, witness_rows(held)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def policy_dir(run_dir: Path, arm: str, seed: int, epoch: int | None) -> Path:
    name = f"{arm}-s{seed}" + (f"-e{epoch}" if epoch is not None else "")
    return run_dir / "policies" / name


def snapshotter(run_dir: Path, arm: str, seed: int, manifest: dict[str, Any]):
    def save(epoch: int, model, row: dict[str, Any]) -> None:
        path = policy_dir(run_dir, arm, seed, epoch)
        if path.exists():
            return
        from wandering_light.stack_policy import save_policy

        save_policy(
            path,
            model,
            trainer_state={**row, "snapshot_epoch": epoch, "parameters": model.parameter_count},
            manifest=manifest,
            wandb_url=None,
        )

    return save


def train(run_dir: Path, tasks, device: str, arms: Sequence[str], seeds, log) -> None:
    for arm in arms:
        fit, held = arm_rows(arm, tasks)
        log(f"RESULT rows {arm}: fit {len(fit)} validation {len(held)}")
        for seed in seeds:
            if policy_dir(run_dir, arm, seed, EPOCHS).exists():
                log(f"skip train {arm} s{seed} (snapshots exist)")
                continue
            manifest = {
                "arm": arm,
                "policy_arm": ARM,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "seed": seed,
                "fit_rows": len(fit),
                "validation_rows": len(held),
                "row_seed": ROW_SEED,
                "mixed_final_probability": MIXED_FINAL_PROBABILITY,
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
                epoch_end=snapshotter(run_dir, arm, seed, manifest),
            )
            co.write_json(run_dir / "training" / f"{arm}-s{seed}.json", state)
            curve = " ".join(
                f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}" for r in state["history"]
            )
            log(
                f"RESULT train {arm} s{seed}: best e{state['best_epoch']} "
                f"CE {state['best_validation_cross_entropy']:.4f} "
                f"[{curve}] {time.perf_counter() - started:.0f}s"
            )


def cell_path(run_dir: Path, arm: str, seed: int, device: str, epoch: int, beam: int, depth: int) -> Path:
    suffix = "" if (beam, depth) == (co.BEAM_WIDTH, co.MAX_DEPTH) else f"-b{beam}-d{depth}"
    return run_dir / "timing" / f"{arm}-s{seed}-{device}-e{epoch}{suffix}.json"


def time_cell(
    run_dir: Path, tasks, arm: str, seed: int, device: str, epoch: int, beam: int, depth: int, log
) -> Path:
    out = cell_path(run_dir, arm, seed, device, epoch, beam, depth)
    if out.exists():
        log(f"skip time {out.name}")
        return out
    if device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    scorer = PolicyScorer(load_policy(policy_dir(run_dir, arm, seed, epoch), device=device), device)
    rows = []
    started = time.perf_counter()
    for index, task in co.cohort(tasks):
        result = beam_solve(task, scorer, beam_width=beam, max_depth=depth, guard=co.GUARD)
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
            "arm": arm,
            "policy_arm": ARM,
            "seed": seed,
            "device": device,
            "policy_epoch": epoch,
            "beam_width": beam,
            "max_depth": depth,
            "guard": co.GUARD,
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
        f"RESULT time {arm} s{seed} e{epoch} beam {beam} depth {depth}: "
        f"{solved}/{len(rows)} solved, {cell_seconds:.0f}s"
    )
    return out


def report(run_dir: Path, out_dir: Path, log) -> Path:
    cells = [json.loads(p.read_text()) for p in sorted((run_dir / "timing").glob("*.json"))]
    by = defaultdict(dict)
    for cell in cells:
        by[cell["arm"]][cell["seed"], cell["policy_epoch"]] = cell
    epochs = sorted({e for arm in by.values() for _, e in arm})
    seeds = sorted({s for arm in by.values() for s, _ in arm})
    lines = ["# hindsight-subtargets: solves on the 300-task v4 cohort", ""]
    lines.append("| arm | " + " | ".join(f"e{e}" for e in epochs) + " | mean |")
    lines.append("|---|" + "---:|" * (len(epochs) + 1))
    for arm in sorted(by):
        cells_of = []
        row = [f"| `{arm}` "]
        for e in epochs:
            got = [by[arm][s, e]["rows"] for s in seeds if (s, e) in by[arm]]
            counts = [sum(r["solved"] for r in rs) for rs in got]
            cells_of += counts
            row.append(
                f"| {mean(counts):.1f} ± {pstdev(counts):.1f} " if counts else "| — "
            )
        row.append(f"| {mean(cells_of):.1f} |" if cells_of else "| — |")
        lines.append("".join(row))
    lines += ["", "Per cell:", "", "| arm | seed | epoch | solved | shown-fit | wall (s) | executions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in sorted(by):
        for (seed, epoch), cell in sorted(by[arm].items()):
            rs = cell["rows"]
            lines.append(
                f"| `{arm}` | {seed} | {epoch} | {sum(r['solved'] for r in rs)} | "
                f"{sum(r['shown_fit'] for r in rs)} | {cell['cell_seconds']:.0f} | "
                f"{sum(r['executions'] for r in rs):,} |"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "solves.md"
    path.write_text("\n".join(lines) + "\n")
    log(f"RESULT report {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rows", "train", "time", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--epochs", default="3,4,5")
    parser.add_argument("--beam", type=int, default=co.BEAM_WIDTH)
    parser.add_argument("--depth", type=int, default=co.MAX_DEPTH)
    args = parser.parse_args(argv)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.run_dir / "log.txt"

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with logfile.open("a") as handle:
            handle.write(stamped + "\n")

    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    epochs = [int(e) for e in args.epochs.split(",") if e]
    tasks = co.corpus()

    if args.stage == "rows":
        for arm in arms:
            fit, held = arm_rows(arm, tasks)
            labels = defaultdict(int)
            for r in fit:
                labels[r.label] += 1
            log(f"RESULT rows {arm}: fit {len(fit)} validation {len(held)} labels {len(labels)}")
    elif args.stage == "train":
        train(args.run_dir, tasks, args.device, arms, seeds, log)
    elif args.stage == "time":
        for arm in arms:
            for seed in seeds:
                for epoch in epochs:
                    time_cell(
                        args.run_dir, tasks, arm, seed, args.device, epoch, args.beam, args.depth, log
                    )
    else:
        report(args.run_dir, args.out or args.run_dir, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
