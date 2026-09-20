"""How the fused-trained policy scores STAGED witness text (staged-witness-reauthoring, #155).

**This measures transfer, not difficulty.** The checkpoint was trained only on
`stack_pbe_v4` witnesses, which are write-then-run: one long quotation, executed
once at the end. A staged witness closes a short body every few ops and runs it
immediately, so the policy has never seen a `]` four tokens in followed by a
combinator and another `[`. Low probability at those steps is unfamiliarity as
much as difficulty, and the two cannot be separated with a fused-trained policy.

So `L x deficit` here is an **out-of-distribution** number. It bounds how well
today's policy transfers to staged text; it does **not** estimate the deficit a
staged-trained policy would show, which is what the `ln W` break-even argument of
clear-stack-pbe-v4 (#99) needs. A staged number worse than its fused twin must
not be read as "staging raises deficit". The real test needs a staged-trained
policy, and 20 witnesses cannot train one.

What *is* readable is the split by step kind. If the deficit concentrates at the
cycle boundary (`]`, the combinator, the next `[`) while tokens inside a body
stay cheap, that is weak evidence short bodies are cheaper steps -- the mechanism
staging is supposed to exploit. If body tokens are expensive too, it is only OOD.

    uv run python experiments/staged_transfer.py --policy <ckpt> \
        --pilot reports/staged-pilot-20260920/assessed.jsonl --out transfer.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from experiments.search_proxy import TOLERANCE  # noqa: E402
from wandering_light.stack_beam import step_all_quiet  # noqa: E402
from wandering_light.stack_machine import Mark  # noqa: E402
from wandering_light.stack_pbe import OPS  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    PolicyScorer,
    encode_item,
    encode_targets,
    item_array,
    load_policy,
)

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task

RUNNERS = ("map", "filter", "fold", "ifte", "i", "dip")
# The three kinds that make up a staged cycle boundary. `body` is everything
# written under an open mark; `direct` is a top-level op that is none of these.
BOUNDARY = ("close", "run", "open")


def step_kinds(task: Task) -> list[str]:
    """One label per op: what kind of step the policy is being asked for."""
    states = replay_states(task)
    kinds = []
    for step, name in enumerate(task.program):
        under_mark = any(isinstance(v, Mark) for stack in states[step] for v in stack)
        if name == "[":
            kinds.append("open")
        elif name == "]":
            kinds.append("close")
        elif under_mark:
            kinds.append("body")
        elif name in RUNNERS:
            kinds.append("run")
        else:
            kinds.append("direct")
    return kinds


def score(task: Task, scorer: PolicyScorer) -> dict[str, Any] | None:
    """Per-step rank and deficit along ``task.program``, labelled by step kind.

    The same walk as `search_proxy.score_task`; it carries the kind labels and
    accepts any program that replays, not only a corpus witness.
    """
    states = replay_states(task)
    targets = encode_targets(states[-1])
    kinds = step_kinds(task)
    deficits: list[float] = []
    ranks: list[int] = []
    for step, name in enumerate(task.program):
        product = states[step]
        legal = np.zeros((1, len(OPS)), dtype=bool)
        for index, op in enumerate(OPS):
            moved, _fuel, _stepped = step_all_quiet(product, op)
            if moved is not None:
                legal[0, index] = True
        want = OP_INDEX[name]
        if not legal[0, want]:
            return None  # a program cannot refuse its own op
        block = item_array(
            [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
        )
        row = scorer.log_probs([block], legal)[0]
        best = float(np.max(row[legal[0]]))
        mine = float(row[want])
        deficits.append(best - mine)
        ranks.append(int((row[legal[0]] > mine).sum()) + 1)
    cumulative = 0.0
    survival = len(deficits)
    for i, d in enumerate(deficits):
        cumulative += d
        if cumulative > TOLERANCE:
            survival = i
            break
    by_kind: dict[str, list[float]] = defaultdict(list)
    for kind, d in zip(kinds, deficits, strict=True):
        by_kind[kind].append(d)
    return {
        "length": len(deficits),
        "mean_deficit": sum(deficits) / len(deficits),
        "accumulated": sum(deficits),
        "budget_ratio": sum(deficits) / TOLERANCE,
        "top1": sum(1 for r in ranks if r == 1) / len(ranks),
        "survival_depth": survival,
        "deficit_by_kind": {
            k: {"n": len(v), "mean": sum(v) / len(v)}
            for k, v in sorted(by_kind.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--control",
        type=int,
        default=40,
        help=(
            "fused witnesses per split to score as a familiarity control. The "
            "pilot behaviours are v4 TRAIN behaviours, so their fused witnesses "
            "are in this policy's training set: without an eval-split control "
            "the staged/fused gap conflates unseen style with unseen content."
        ),
    )
    parser.add_argument("--control-seed", type=int, default=155)
    args = parser.parse_args(argv)

    if args.device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    pilot = [
        json.loads(line) for line in args.pilot.read_text().splitlines() if line.strip()
    ]
    staged = {r["behaviour"]: r for r in pilot if r.get("verdict") == "staged"}
    print(f"{len(staged)} staged of {len(pilot)} pilot behaviours", flush=True)

    # One train task per behaviour, so fused and staged are scored against the
    # same inputs and the same final target.
    tasks: dict[str, Task] = {}
    for task in co.corpus():
        if task.split == "train" and task.behaviour in staged:
            tasks.setdefault(task.behaviour, task)
    print(f"matched {len(tasks)} behaviours to a v4 train task", flush=True)

    scorer = PolicyScorer(load_policy(args.policy, device=args.device), args.device)
    rows = []
    for behaviour, task in sorted(tasks.items()):
        program = tuple(staged[behaviour]["program"].split())
        fused = score(task, scorer)
        staged_task = replace(task, program=program, length=len(program))
        stg = score(staged_task, scorer)
        if fused is None or stg is None:
            print(f"  SKIP {behaviour} (a program refused its own op)", flush=True)
            continue
        rows.append({"behaviour": behaviour, "fused": fused, "staged": stg})
        print(
            f"  {behaviour:28s} fused L={fused['length']:3d} d={fused['mean_deficit']:.3f} "
            f"Ld={fused['accumulated']:7.1f} | staged L={stg['length']:3d} "
            f"d={stg['mean_deficit']:.3f} Ld={stg['accumulated']:7.1f}",
            flush=True,
        )

    def agg(side: str, field: str) -> float:
        return median(r[side][field] for r in rows)

    kinds: dict[str, dict[str, list[float]]] = {
        "fused": defaultdict(list),
        "staged": defaultdict(list),
    }
    for r in rows:
        for side in ("fused", "staged"):
            for kind, cell in r[side]["deficit_by_kind"].items():
                kinds[side][kind].extend([cell["mean"]] * cell["n"])

    # The familiarity control: fused witnesses the policy DID train on, and
    # fused witnesses it did not, at the pilot's length profile. It prices how
    # much of the staged penalty is "unseen" rather than "staged".
    control: dict[str, dict[str, float]] = {}
    if args.control:
        rng = random.Random(args.control_seed)
        low = min(r["fused"]["length"] for r in rows)
        high = max(r["fused"]["length"] for r in rows)
        for split in ("train", "eval"):
            pool = [
                t for t in co.corpus() if t.split == split and low <= t.length <= high
            ]
            picked = rng.sample(pool, min(args.control, len(pool)))
            scored = [c for c in (score(t, scorer) for t in picked) if c]
            control[f"fused_{split}"] = {
                "n": len(scored),
                "median_length": median(c["length"] for c in scored),
                "median_deficit": median(c["mean_deficit"] for c in scored),
                "median_accumulated": median(c["accumulated"] for c in scored),
                "within_budget": sum(
                    1 for c in scored if c["accumulated"] <= TOLERANCE
                ),
            }
            cell = control[f"fused_{split}"]
            print(
                f"control fused/{split}: n={cell['n']} "
                f"median d={cell['median_deficit']:.3f} "
                f"median L*d={cell['median_accumulated']:.1f} "
                f"within lnW={cell['within_budget']}",
                flush=True,
            )

    summary = {
        "n": len(rows),
        "tolerance": TOLERANCE,
        "familiarity_control": control,
        "median": {
            side: {
                f: agg(side, f)
                for f in ("length", "mean_deficit", "accumulated", "top1")
            }
            for side in ("fused", "staged")
        },
        "deficit_by_kind": {
            side: {
                k: {"steps": len(v), "mean": sum(v) / len(v)}
                for k, v in sorted(kinds[side].items())
            }
            for side in ("fused", "staged")
        },
        "within_budget": {
            side: sum(1 for r in rows if r[side]["accumulated"] <= TOLERANCE)
            for side in ("fused", "staged")
        },
    }
    out = {
        "caveat": "OUT OF DISTRIBUTION: transfer of a fused-trained policy to staged text, not a difficulty estimate",
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{'':10s} {'median L':>9s} {'median d':>9s} {'median L*d':>11s} {'within ln W':>12s}"
    )
    for side in ("fused", "staged"):
        m = summary["median"][side]
        print(
            f"{side:10s} {m['length']:9.1f} {m['mean_deficit']:9.3f} "
            f"{m['accumulated']:11.1f} {summary['within_budget'][side]:12d}"
        )
    print(
        f"\n{'step kind':10s} {'fused steps':>12s} {'fused mean d':>13s} {'staged steps':>13s} {'staged mean d':>14s}"
    )
    for kind in ("body", "open", "close", "run", "direct"):
        f = summary["deficit_by_kind"]["fused"].get(kind)
        s = summary["deficit_by_kind"]["staged"].get(kind)
        print(
            f"{kind:10s} {(f['steps'] if f else 0):12d} {(f['mean'] if f else float('nan')):13.3f} "
            f"{(s['steps'] if s else 0):13d} {(s['mean'] if s else float('nan')):14.3f}"
        )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
