"""short-segment-campaign (#168): the closed-state archive on train tasks, for a learned ranker.

Runs the segment search (`search.py`'s segment cell: width 64, K roots, 12
segments of the arm's depth, guard both, the #157 state cap, 120 s a task) on
capped train tasks, the first ``--sets`` example sets of each of the 148
capped train behaviours, and writes `search.archive_record` per task with each
kept closed state flagged ``on_witness`` when it equals a closed state along
the behaviour's authored capped witness (`train_capped.jsonl`). Train tasks give
positives on every task from the witness, where eval searches mostly fail;
branch-and-check (#164) step 2 trains on this, never on the eval-cohort
archive. The library is off, so every task is its own search. Appends one gzip
member per task and resumes from what the file holds::

    uv run python experiments/short_segment/archive_train.py \\
        --run-dir checkpoints/short-segment-difference-20260923 --arm capped --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.hindsight_subtargets as hs  # noqa: E402
from experiments.segment_k_diagnostic import CAPPED, _jsonl  # noqa: E402
from experiments.short_segment import search as driver  # noqa: E402
from experiments.staged_witnesses import check as staged  # noqa: E402
from wandering_light import solve  # noqa: E402
from wandering_light.stack_beam import (  # noqa: E402
    _prepare,
    closed_cuts,
    step_all_quiet,
)
from wandering_light.stack_pbe import OP_BY_NAME  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402


def witness_closed(task, program: tuple[str, ...]) -> frozenset | None:
    """The closed states along ``program`` on ``task``'s shown examples; None if it refuses a step."""
    inputs, _ = _prepare(task, "both")
    states = [inputs]
    for name in program:
        moved, _spent, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None:
            return None
        states.append(moved)
    return frozenset(states[cut] for cut in closed_cuts(states, program))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", default="capped")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roots", type=int, default=driver.ROOTS)
    parser.add_argument(
        "--sets", type=int, default=4, help="example sets per behaviour"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seconds-cap", type=float, default=driver.TASK_SECONDS_CAP)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    depth = driver.segment_depth_for(args.arm)
    out = args.out or (
        driver.DEFAULT_OUT
        / args.run_dir.name
        / "archive"
        / f"train-{args.arm}-s{args.seed}-e{driver.EPOCH}-segment-k{args.roots}-d{depth}.jsonl.gz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with gzip.open(out, "rt") as lines:
            done = {(r["behaviour"], r["example_set"]) for r in map(json.loads, lines)}
    tasks = staged.v4_tasks()
    work = []
    for row in _jsonl(CAPPED):
        program = tuple(row["program"].split())
        for number, task in enumerate(tasks[row["behaviour"]][: args.sets]):
            if (row["behaviour"], number) not in done:
                work.append((number, dataclasses.replace(task, program=program)))
    print(f"{len(work)} tasks to run, {len(done)} done; archive {out}", flush=True)
    if not work:
        return 0
    system = dataclasses.replace(
        driver.system_for("segment", roots=args.roots, segment_depth=depth),
        library=False,
    )
    torch.set_num_threads(args.threads)
    policy = hs.policy_dir(args.run_dir, args.arm, args.seed, driver.EPOCH)
    scorer = PolicyScorer(load_policy(policy, device=args.device), args.device)
    print(f"{system.describe()}; policy {policy} ({scorer.encoding})", flush=True)

    def record(row: solve.Row, segments: list) -> None:
        number, task = work[row.index]
        witness = witness_closed(task, task.program)
        line = driver.archive_record(row, row.index, segments, witness=witness)
        line.update(
            example_set=number,
            witness=" ".join(task.program),
            witness_closed=None if witness is None else len(witness),
            witness_seen=sum(
                s.get("on_witness", False)
                for segment in line["segments"]
                for s in segment["states"]
            ),
            policy=str(policy),
            system=dataclasses.asdict(system),
            seconds=row.seconds,
            executions=row.executions,
            fuel=row.fuel,
        )
        del line["task_index"]
        with gzip.open(out, "at") as handle:
            handle.write(json.dumps(line) + "\n")
        print(
            f"{row.index + 1}/{len(work)} {task.behaviour}#{number} {row.termination} "
            f"witness {line['witness_seen']}/{line['witness_closed']} {row.seconds:.1f}s",
            flush=True,
        )

    solve.solve_tasks(
        [task for _, task in work],
        scorer,
        system,
        seconds_cap=args.seconds_cap,
        archive=record,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
