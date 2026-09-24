"""A beam whose path score floors each step's log-probability at ``-tau``.

`beam_solve` ranks on a plain cumulative sum of masked log-softmax values, so a
prefix that takes one low-probability step carries that whole deficit for the
rest of the search: at 1-8% the authored op costs 2.5-4.6 nats, and the beam's
tolerance is ``ln 64`` = 4.159 in total. One bad step is the whole budget.

Flooring replaces ``log p`` with ``max(log p, -tau)`` in the *ranking key only*.
A path is then charged at most ``tau`` for any single step, so a witness that is
right everywhere except one op is no longer eliminated by that op alone. Nothing
else changes: the machine, the guard, the dedup, the hit test and the budget
counters are `stack_beam`'s, imported not copied, and at ``tau = inf`` this is
`beam_solve` exactly -- asserted by `--tau inf` reproducing the unfloored cell.

Why it is worth a cell. An adversarial review refuted this on the grounds that
the competitor is floored too, so the relative per-step gain is only
``min(log(p_C/p_W), tau + log p_C)`` ~ 0.30 nats at ``tau = 1.5``, "below a width
step already recorded as buying nothing". The second half of that is now false:
beam 256 measured +2 non-leaky behaviours, and a x4 width step is worth
``ln 4`` = 1.386 nats *in total*. A per-step gain of 0.30 nats over a 16-op
witness is 4.8 nats in total, 3.5x the width step. The refutation compared a
per-step quantity against a total one.

    uv run python experiments/floored_beam.py --policy <ckpt> --tau 1.5 --out cell.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from wandering_light.quotation_guard import blocked  # noqa: E402
from wandering_light.stack_beam import (  # noqa: E402
    BeamResult,
    block_encoder,
    step_all_quiet,
)
from wandering_light.stack_machine import Mark, parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS, OPS  # noqa: E402
from wandering_light.stack_pbe_eval import grade  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    PolicyScorer,
    load_policy,
)

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task


def floored_beam_solve(
    task: Task,
    scorer: PolicyScorer,
    *,
    tau: float,
    beam_width: int = 64,
    max_depth: int = 48,
    guard: str = "both",
) -> BeamResult:
    """`beam_solve` with ``max(log p, -tau)`` as the per-step ranking charge."""
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        started = time.perf_counter()
        inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
        outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
        if guard in ("close", "both") and any(
            isinstance(v, Mark) for stack in outputs for v in stack
        ):
            raise ValueError("the close guard needs mark-free targets")
        encode = block_encoder(scorer, outputs)
        seen = {inputs}
        beam = [(inputs, (), 0.0)]
        executions = fuel = expansions = 0

        def result(program, termination, depth):
            return BeamResult(
                program,
                termination,
                time.perf_counter() - started,
                executions,
                fuel,
                expansions,
                depth,
            )

        for depth in range(max_depth):
            remaining = max_depth - depth - 1
            successors = []
            legal = np.zeros((len(beam), len(OPS)), dtype=bool)
            for node, (product, program, _) in enumerate(beam):
                expansions += 1
                for index, op in enumerate(OPS):
                    moved, spent, stepped = step_all_quiet(product, op)
                    executions += stepped
                    fuel += spent
                    if moved is None:
                        continue
                    legal[node, index] = True
                    extended = (*program, op.name)
                    if moved == outputs:
                        return result(extended, "fit", depth + 1)
                    successors.append((node, index, moved, extended))
            if not successors:
                return result(None, "frontier_exhausted", depth)
            blocks = [encode(product) for product, _, _ in beam]
            log_probs = scorer.log_probs(blocks, legal)
            charged = np.maximum(log_probs, -tau) if math.isfinite(tau) else log_probs
            ranked = sorted(
                successors,
                key=lambda s: beam[s[0]][2] + charged[s[0], s[1]],
                reverse=True,
            )
            next_beam = []
            for node, index, product, program in ranked:
                if guard != "off" and blocked(product[0], remaining, guard) is not None:
                    continue
                if product in seen:
                    continue
                seen.add(product)
                next_beam.append(
                    (product, program, beam[node][2] + charged[node, index])
                )
                if len(next_beam) == beam_width:
                    break
            if not next_beam:
                return result(None, "frontier_exhausted", depth + 1)
            beam = next_beam
        return result(None, "max_depth", max_depth)
    finally:
        if collecting:
            gc.enable()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tau", type=float, required=True, help="inf reproduces beam_solve"
    )
    parser.add_argument("--beam", type=int, default=co.BEAM_WIDTH)
    parser.add_argument("--depth", type=int, default=co.MAX_DEPTH)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"skip {args.out} (exists)")
        return 0
    if args.device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    tasks = co.corpus()
    scorer = PolicyScorer(load_policy(args.policy, device=args.device), args.device)
    gc.freeze()
    rows = []
    started = time.perf_counter()
    for index, task in co.cohort(tasks):
        r = floored_beam_solve(
            task, scorer, tau=args.tau, beam_width=args.beam, max_depth=args.depth
        )
        rows.append(
            {
                "task_index": index,
                "behaviour": task.behaviour,
                "length": task.length,
                "bucket": co.bucket_of(task.length),
                "examples": task.examples,
                "program": list(r.program) if r.program else None,
                "_program": r.program,
                "task": task,
                "termination": r.termination,
                "seconds": r.seconds,
                "executions": r.executions,
                "fuel": r.fuel,
                "expansions": r.expansions,
                "depth": r.depth,
                "source": "search",
            }
        )
    cell_seconds = time.perf_counter() - started
    for row in rows:
        found, task = row.pop("_program"), row.pop("task")
        score = grade(found, task) if found else None
        row["solved"] = bool(score and score.solved)
        row["shown_fit"] = bool(score and score.shown)
    cell: dict[str, Any] = {
        "cache": "off",
        "tau": args.tau,
        "beam_width": args.beam,
        "max_depth": args.depth,
        "guard": co.GUARD,
        "policy": str(args.policy),
        "device": args.device,
        "eval_corpus": co.primary_corpus(),
        "cell_seconds": cell_seconds,
        "timed_seconds": sum(r["seconds"] for r in rows),
        "cache_programs": [],
        "host": co.host(),
        "provenance": co.provenance(),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    solved = sum(r["solved"] for r in rows)
    behaviours = len({r["behaviour"] for r in rows if r["solved"]})
    print(
        f"RESULT {args.out.name} tau {args.tau} beam {args.beam}: {solved}/{len(rows)} tasks, "
        f"{behaviours}/43 behaviours, {cell_seconds:.0f}s, "
        f"{sum(r['executions'] for r in rows):,} executions",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
