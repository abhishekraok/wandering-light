"""The registered prediction for segment-reset search: which witnesses fit its budget.

A beam ranked on cumulative log-probability spends `ln W` once over a whole
witness (`docs/STATUS.md`, the budget law). A beam whose ranking resets at each
**closed state** -- after a top-level combinator has consumed its quotation, so
no mark is open -- spends `ln W` once per **segment** instead. A segment is the
full span of ops from one closed state to the next, so it carries every body
token written inside it; nothing gives a quotation body a budget of its own.

For each witness this walks the program under a policy exactly as
`staged_transfer.score` does, keeps the per-step deficit (nats behind the
policy's best legal op), and asks two questions:

- **flat**: is the witness's summed deficit within `ln W`? The control arm's
  prediction, and the `within_budget` count `witness_deficit.py` reports.
- **segmented**: is every segment's summed deficit within `ln W`? The arm's
  prediction.

It prices the budget law and nothing else: that a correct closed state stays in
the top `W` at every stage is an assumption no path score can guarantee, and a
search may return a different program than the authored witness.

Two further rows say where a decomposition *would* pay, and are not predictions
for segment-reset search. Each op is assigned to the innermost quotation open
when it is written -- a `[` to the level that opens it, the `]` to the body it
closes -- and **quotations** gives every quotation body its own `ln W` with the
top level as one sub-search, the shape of inner-frames (#160) E2; **both**
also segments the top level. They charge a body the deficit the *flat* policy
pays for it with the whole stack in view, which is not what a sub-search over a
frame would pay; that is the E2 question, and these rows only bound it.

    uv run python experiments/segment_composite.py \\
        --checkpoint checkpoints/world-arm-20260922/policies/1m-s0-e5 --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from experiments.search_proxy import TOLERANCE  # noqa: E402
from experiments.staged_transfer import step_kinds  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.solve import BEST  # noqa: E402
from wandering_light.stack_beam import block_encoder, step_all_quiet  # noqa: E402
from wandering_light.stack_pbe import OPS, load  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    PolicyScorer,
    load_policy,
)
from wandering_light.stack_policy_hub import fetch_stack_policy  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

DATA = REPO / "wandering_light" / "evals" / "data"


def step_deficits(task: Task, scorer: PolicyScorer) -> list[float] | None:
    """Nats behind the best legal op at each witness step; None if the witness refuses."""
    states = replay_states(task)
    encode = block_encoder(scorer, states[-1])
    out = []
    for step, name in enumerate(task.program):
        product = states[step]
        legal = np.zeros((1, len(OPS)), dtype=bool)
        for index, op in enumerate(OPS):
            if step_all_quiet(product, op)[0] is not None:
                legal[0, index] = True
        want = OP_INDEX[name]
        if not legal[0, want]:
            return None
        row = scorer.log_probs([encode(product)], legal)[0]
        out.append(float(np.max(row[legal[0]])) - float(row[want]))
    return out


def segments(deficits: Sequence[float], kinds: Sequence[str]) -> list[float]:
    """Summed deficit per segment, cutting after every top-level combinator.

    `step_kinds` labels a combinator applied with no mark open as ``run``; the
    state after it is closed. Body ops, the quotation's own brackets and any
    top-level glue before the combinator all fall in the segment it closes.
    """
    # Whether ops are pending is tracked apart from their sum: a tail of ops the
    # policy gets exactly right sums to zero and is still a segment.
    sums, current, pending = [], 0.0, False
    for d, kind in zip(deficits, kinds, strict=True):
        current += d
        pending = True
        if kind == "run":
            sums.append(current)
            current, pending = 0.0, False
    if pending or not sums:
        sums.append(current)
    return sums


def levels(
    program: Sequence[str], deficits: Sequence[float], kinds: Sequence[str]
) -> tuple[list[float], list[float]]:
    """(top-level segment sums, one sum per quotation body) for the nested rows.

    An op belongs to the innermost quotation open when it is written. The top
    level keeps only its own ops and is cut after each ``run``, as `segments`.
    """
    top: list[float] = []
    current, pending = 0.0, False
    open_bodies: list[float] = []
    bodies: list[float] = []
    for name, d, kind in zip(program, deficits, kinds, strict=True):
        if name == "[":
            if open_bodies:
                open_bodies[-1] += d
            else:
                current += d
                pending = True
            open_bodies.append(0.0)
        elif name == "]" and open_bodies:
            open_bodies[-1] += d
            bodies.append(open_bodies.pop())
        elif open_bodies:
            open_bodies[-1] += d
        else:
            current += d
            pending = True
            if kind == "run":
                top.append(current)
                current, pending = 0.0, False
    bodies.extend(open_bodies)
    if pending or not top:
        top.append(current)
    return top, bodies


def witness_rows(tasks: Sequence[Task], scorer: PolicyScorer, log) -> list[dict]:
    rows = []
    for task in tasks:
        deficits = step_deficits(task, scorer)
        if deficits is None:
            log(f"skipped {task.behaviour}: witness refused")
            continue
        kinds = step_kinds(task)
        segs = segments(deficits, kinds)
        top, bodies = levels(task.program, deficits, kinds)
        worst_body = max(bodies, default=0.0)
        rows.append(
            {
                "behaviour": task.behaviour,
                "length": len(deficits),
                "accumulated": sum(deficits),
                "segments": [round(s, 4) for s in segs],
                "top_level_segments": [round(s, 4) for s in top],
                "bodies": [round(b, 4) for b in bodies],
                "flat": sum(deficits) <= TOLERANCE,
                "segmented": max(segs) <= TOLERANCE,
                "quotations": max(sum(top), worst_body) <= TOLERANCE,
                "both": max(*top, worst_body) <= TOLERANCE,
            }
        )
    return rows


def tally(rows: Sequence[dict], behaviours: set[str] | None = None) -> dict[str, Any]:
    rows = [r for r in rows if behaviours is None or r["behaviour"] in behaviours]
    names = {
        key: sorted(r["behaviour"] for r in rows if r[key])
        for key in ("flat", "segmented", "quotations", "both")
    }
    return {
        "witnesses": len(rows),
        **{key: len(v) for key, v in names.items()},
        "delta": len(names["segmented"]) - len(names["flat"]),
        "segmented_only": sorted(set(names["segmented"]) - set(names["flat"])),
        "flat_only": sorted(set(names["flat"]) - set(names["segmented"])),
        "names": names,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--corpora", default="stack_pbe_v4,stack_pbe_world_v3")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    def log(line: str) -> None:
        print(line, flush=True)

    torch.set_num_threads(8)
    policy_dir = args.checkpoint or fetch_stack_policy(BEST.repo_id, BEST.revision)
    scorer = PolicyScorer(load_policy(policy_dir, device="cpu"), "cpu")
    log(f"policy {policy_dir}; budget {TOLERANCE:.3f} nats per sub-search")
    report: dict[str, Any] = {
        "policy": str(policy_dir),
        "policy_is_best": args.checkpoint is None,
        "budget_nats": TOLERANCE,
        "split": args.split,
        "cells": {},
    }
    for name in args.corpora.split(","):
        tasks = load(fetch_corpus(DATA / name / "manifest.json"))
        # One task per behaviour, as `witness_deficit.py` does: the witness is the unit.
        seen, unique = set(), []
        for t in tasks:
            if t.split == args.split and t.behaviour not in seen:
                seen.add(t.behaviour)
                unique.append(t)
        rows = witness_rows(unique, scorer, log)
        cell = {"all": tally(rows), "rows": rows}
        if name == "stack_pbe_v4" and args.split == "eval":
            # The #101 cohort is the denominator every verified count is on.
            cohort = {t.behaviour for _i, t in co.cohort(tasks)}
            cell["cohort_101"] = tally(rows, cohort)
        report["cells"][f"{name}:{args.split}"] = cell
        for scope in ("all", "cohort_101"):
            if scope in cell:
                t = cell[scope]
                log(
                    f"{name}:{args.split} [{scope}] n={t['witnesses']} "
                    f"flat={t['flat']} segmented={t['segmented']} delta={t['delta']:+d} "
                    f"| quotations={t['quotations']} both={t['both']}"
                )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "prediction.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        log(f"wrote {args.out / 'prediction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
