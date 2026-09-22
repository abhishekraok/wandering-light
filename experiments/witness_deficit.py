"""What a policy pays per step along a corpus's witnesses (clear-world-v3 #161, #99).

`docs/STATUS.md` carries the budget law: a beam of width `W` ranked on
cumulative log-probability tolerates about `ln W` nats of accumulated deficit
**over the whole witness**, so a witness of length `L` must hold its per-step
deficit under `ln W / L`. Every use of that law so far has taken `L` from a
corpus and the deficit from a *different* corpus's measurement -- #161's first
reading of the world-v3 depth ladder took the 0.075 nats that
staged-witness-reauthoring (#155) measured on `stack_pbe_v4` witnesses and
applied it at world-v3's median length of 150 ops. That is an extrapolation
across corpora, and it is the load-bearing step in the argument that only
shortening `L` can help.

This measures the deficit where the law is being applied. For every witness it
walks the program under the policy, recording at each step how many nats behind
the policy's top legal choice the witness's own op is, and reports `L`, the
per-step deficit, their product against `ln W`, and the depth at which the
accumulated deficit first exceeds the budget -- the step where a beam of that
width would have dropped the witness.

    uv run python experiments/witness_deficit.py --out reports/<dir>

Corpora are given as manifest directory names; `--split` picks the split. The
policy defaults to whatever `wandering_light.solve.BEST` pins, so the number
describes the system the owner actually runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from experiments.staged_transfer import score  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.solve import BEST  # noqa: E402
from wandering_light.stack_pbe import load  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402
from wandering_light.stack_policy_hub import fetch_stack_policy  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

DATA = REPO / "wandering_light" / "evals" / "data"
# `search_proxy.TOLERANCE` is ln(BEAM_WIDTH); restate it here so the report can
# say which width the budget belongs to.
from experiments.search_proxy import TOLERANCE  # noqa: E402


def summarise(rows: Sequence[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    if not rows:
        return {"witnesses": 0}
    lengths = [r["length"] for r in rows]
    deficits = [r["mean_deficit"] for r in rows]
    products = [r["accumulated"] for r in rows]
    within = [r for r in rows if r["accumulated"] <= tolerance]
    return {
        "witnesses": len(rows),
        "length": {"median": median(lengths), "min": min(lengths), "max": max(lengths)},
        "per_step_deficit": {
            "median": round(median(deficits), 4),
            "min": round(min(deficits), 4),
            "max": round(max(deficits), 4),
        },
        "accumulated_deficit": {"median": round(median(products), 2)},
        "budget_nats": round(tolerance, 3),
        # What a witness of the median length is ALLOWED per step, against what
        # it actually costs. The ratio is how far out of reach the split is.
        "budget_per_step_at_median_length": round(tolerance / median(lengths), 4),
        "over_budget_factor_at_median": round(
            median(deficits) / (tolerance / median(lengths)), 1
        ),
        "within_budget": len(within),
        "within_budget_share": round(len(within) / len(rows), 3),
        "survival_depth": {
            "median": median(r["survival_depth"] for r in rows),
            "max": max(r["survival_depth"] for r in rows),
        },
        "top1_share": round(median(r["top1"] for r in rows), 3),
    }


def run(tasks: Sequence[Task], scorer: PolicyScorer, log) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        got = score(task, scorer)
        if got is None:  # a witness its own machine refuses; none expected
            log(f"skipped {task.behaviour}: witness refused")
            continue
        rows.append({"behaviour": task.behaviour, **got})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpora", default="stack_pbe_world_v3,stack_pbe_v4")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    def log(line: str) -> None:
        print(line, flush=True)

    if args.device == "cpu":
        torch.set_num_threads(8)
    policy_dir = args.checkpoint or fetch_stack_policy(BEST.repo_id, BEST.revision)
    scorer = PolicyScorer(load_policy(policy_dir, device=args.device), args.device)
    log(f"policy {policy_dir}")
    log(f"budget {TOLERANCE:.3f} nats (ln of beam width {BEST.beam_width})")

    report: dict[str, Any] = {
        "policy": str(policy_dir),
        "policy_is_best": args.checkpoint is None,
        "beam_width": BEST.beam_width,
        "budget_nats": TOLERANCE,
        "split": args.split,
        "cells": {},
    }
    for name in args.corpora.split(","):
        tasks = [
            t
            for t in load(fetch_corpus(DATA / name / "manifest.json"))
            if t.split == args.split
        ]
        # One task per behaviour: the witness is the object, not the task.
        seen, unique = set(), []
        for t in tasks:
            if t.behaviour in seen:
                continue
            seen.add(t.behaviour)
            unique.append(t)
        log(f"== {name}:{args.split}: {len(unique)} witnesses")
        rows = run(unique, scorer, log)
        cell = summarise(rows, TOLERANCE)
        report["cells"][f"{name}:{args.split}"] = {**cell, "rows": rows}
        log(json.dumps(cell, indent=1))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "deficit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        log(f"wrote {args.out / 'deficit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
