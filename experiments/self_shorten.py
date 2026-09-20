"""Let the solver relabel its own training data with the shorter programs it finds.

This is the README's loop -- take an action, learn from its outcome -- aimed at
the quantity that actually binds. The beam keeps the ``W`` highest cumulative
log-probabilities, tolerating about ``ln W`` nats, so a witness of length ``L``
must hold its deficit under ``ln W / L`` per step. Shortening ``L`` raises that
allowance for every step at once, and it does so more cheaply than width: a 5-op
witness solved in 3 ops is a 67% larger per-step budget, where a x4 beam is
``ln 4`` = 1.386 nats total.

The solver already finds such programs. Over 3,257 verified solves on disk,
**37.5% are strictly shorter than the authored witness** -- 44% in the 1-5
bucket saving a median 2 ops, 51% in 11-20 saving 1. Nothing has ever fed them
back.

**Shorter is not the same as correct.** Held-out grading draws from the same
sampler as the shown examples, and `docs/STATUS.md` records two behaviours whose
samplers leak; the 41+ bucket's "100% shorter, median 46 ops saved" is those
shortcuts, not insight. So a harvested program is accepted only if it agrees
with the authored witness on **every shown input of every train task of that
behaviour** -- a median 160 inputs the program never saw, all train-side, no
sampler and no reference implementation needed.

Only the train split is searched. The eval split is never read.

    uv run python experiments/self_shorten.py harvest --policy <ckpt> --out harvest.json
    uv run python experiments/self_shorten.py train   --harvest harvest.json --run-dir <dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
from experiments.solve_with_cache import replay_program  # noqa: E402
from wandering_light.stack_beam import beam_solve  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    PolicyScorer,
    Row,
    encode_item,
    encode_targets,
    item_array,
    load_policy,
    validation_skeleton,
    witness_rows,
)

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task

HARVEST_SEED = 4242


def equivalent_on_behaviour(
    program: tuple[str, ...], tasks_of_behaviour: list[Task]
) -> tuple[bool, int]:
    """Does ``program`` match the authored witness on every train input available?

    The witness's own output for an input is the label, so this needs neither the
    behaviour's sampler nor its Python reference -- only other train tasks of the
    same behaviour, whose shown pairs the program has never seen.
    """
    checked = 0
    for task in tasks_of_behaviour:
        inputs = tuple(parse_stack(t, basis=BASIS) for t in task.inputs)
        outputs = tuple(parse_stack(t, basis=BASIS) for t in task.outputs)
        fits, _executions, _fuel = replay_program(program, inputs, outputs)
        checked += len(inputs)
        if not fits:
            return False, checked
    return True, checked


def harvest(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"skip {args.out} (exists)")
        return 0
    if args.device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    tasks = co.corpus()
    train = [t for t in tasks if t.split == "train" and not validation_skeleton(t)]
    by_behaviour: dict[str, list[Task]] = defaultdict(list)
    for t in train:
        by_behaviour[t.behaviour].append(t)

    rng = random.Random(HARVEST_SEED)
    sample = rng.sample(train, min(args.tasks, len(train)))
    scorer = PolicyScorer(load_policy(args.policy, device=args.device), args.device)
    gc.freeze()

    found: dict[str, dict[str, Any]] = {}  # behaviour -> best accepted program
    rows = []
    started = time.perf_counter()
    for n, task in enumerate(sample, 1):
        result = beam_solve(
            task, scorer, beam_width=args.beam, max_depth=args.depth, guard=co.GUARD
        )
        record: dict[str, Any] = {
            "behaviour": task.behaviour,
            "witness_length": task.length,
            "termination": result.termination,
            "seconds": result.seconds,
        }
        if result.program is not None:
            saving = task.length - len(result.program)
            ok, checked = equivalent_on_behaviour(result.program, by_behaviour[task.behaviour])
            record |= {
                "program": list(result.program),
                "found_length": len(result.program),
                "saving": saving,
                "equivalent": ok,
                "inputs_checked": checked,
            }
            best = found.get(task.behaviour)
            if ok and saving > 0 and (best is None or len(result.program) < best["found_length"]):
                found[task.behaviour] = {
                    "program": list(result.program),
                    "found_length": len(result.program),
                    "witness_length": task.length,
                    "saving": saving,
                    "inputs_checked": checked,
                }
        rows.append(record)
        if n % 50 == 0:
            print(
                f"  {n}/{len(sample)} searched, {len(found)} behaviours shortened, "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )

    report = {
        "policy": str(args.policy),
        "tasks_searched": len(sample),
        "beam_width": args.beam,
        "max_depth": args.depth,
        "seconds": time.perf_counter() - started,
        "shortened_behaviours": found,
        "rows": rows,
        "provenance": co.provenance(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    solved = sum(1 for r in rows if r.get("program"))
    shorter = [r for r in rows if r.get("saving", 0) > 0]
    accepted = [r for r in shorter if r.get("equivalent")]
    print(
        f"RESULT harvest: {solved}/{len(sample)} searched tasks returned a program, "
        f"{len(shorter)} shorter than the witness, {len(accepted)} of those equivalent "
        f"({len(accepted) / max(len(shorter), 1):.2f}), "
        f"{len(found)} behaviours shortened, {report['seconds']:.0f}s"
    )
    return 0


def relabelled_rows(tasks: list[Task], shortened: dict[str, dict[str, Any]]) -> list[Row]:
    """Witness rows, except a shortened behaviour replays its harvested program."""
    rows: list[Row] = []
    for index, task in enumerate(tasks):
        entry = shortened.get(task.behaviour)
        if entry is None or entry["found_length"] >= task.length:
            rows.extend(witness_rows([task]))
            continue
        program = tuple(entry["program"])
        replacement = task.__class__(
            **{
                **{f: getattr(task, f) for f in task.__dataclass_fields__},
                "program": list(program),
                "length": len(program),
            }
        )
        try:
            states = hs.replay_states(replacement)
        except ValueError:
            rows.extend(witness_rows([task]))  # refuses on this task's inputs
            continue
        targets = encode_targets(states[-1])
        for step, name in enumerate(program):
            items = [
                encode_item(s, t) for s, t in zip(states[step], targets, strict=True)
            ]
            rows.append(
                Row(item_array(items), OP_INDEX[name], index, step, len(program) - step)
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--policy", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--tasks", type=int, default=1200)
    h.add_argument("--beam", type=int, default=co.BEAM_WIDTH)
    h.add_argument("--depth", type=int, default=co.MAX_DEPTH)
    h.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    t = sub.add_parser("train")
    t.add_argument("--harvest", type=Path, required=True)
    t.add_argument("--run-dir", type=Path, required=True)
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)
    if args.stage == "harvest":
        return harvest(args)

    shortened = json.loads(args.harvest.read_text())["shortened_behaviours"]
    tasks = co.corpus()
    train = [t for t in tasks if t.split == "train"]
    fit_tasks = [t for t in train if not validation_skeleton(t)]
    held = [t for t in train if validation_skeleton(t)]
    fit = relabelled_rows(fit_tasks, shortened)
    validation = witness_rows(held)
    print(
        f"RESULT rows shortened: fit {len(fit)} (witness baseline 313560), "
        f"validation {len(validation)}, {len(shortened)} behaviours relabelled"
    )

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        (args.run_dir / "log.txt").open("a").write(stamped + "\n")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",") if s):
        if hs.policy_dir(args.run_dir, "shortened", seed, hs.EPOCHS).exists():
            log(f"skip train shortened s{seed}")
            continue
        manifest = {
            "arm": "shortened",
            "harvest": str(args.harvest),
            "behaviours_relabelled": len(shortened),
            "fit_rows": len(fit),
            "seed": seed,
            "provenance": co.provenance(),
        }
        _model, state = hs.train_policy(
            fit,
            validation,
            co.architecture(hs.ARM),
            epochs=hs.EPOCHS,
            learning_rate=hs.LEARNING_RATE,
            batch_size=hs.BATCH,
            validation_batch_size=hs.VAL_BATCH,
            seed=seed,
            device=args.device,
            epoch_end=hs.snapshotter(args.run_dir, "shortened", seed, manifest),
        )
        curve = " ".join(
            f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}" for r in state["history"]
        )
        log(f"RESULT train shortened s{seed}: best e{state['best_epoch']} [{curve}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
