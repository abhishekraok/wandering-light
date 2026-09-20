"""A minute-scale stand-in for a 15-minute search cell, and the check of whether it works.

Iterating on this policy currently costs a 300-task cohort cell, and the one
cheap signal there is -- validation cross-entropy -- is *anti-correlated* with
solves after epoch 2 (`docs/STATUS.md`, Superseded). So there is no fast way to
tell a better checkpoint from a worse one, and every idea costs a quarter of an
hour to score.

What the beam actually needs is not a calibrated distribution over next ops. It
needs the authored op to stay near the top of the ranking, layer after layer,
because the beam keeps the ``W`` highest *cumulative* log-probabilities. Falling
behind the best available path by ``d`` nats puts roughly ``e^d`` paths ahead of
you, so a width-``W`` beam tolerates about ``ln W`` nats of accumulated deficit
-- 4.16 nats at ``W = 64``. That is a heuristic, not a bound (it assumes the
paths ahead are roughly uniform in log-space), but it is the quantity the
recorded width sweep behaves like: `reports/materialized-rollout-20260829`
measured 64x width buying exactly one depth of witness survival, i.e. ~4 nats
per depth.

So this replays authored witnesses and measures, per step, where the authored op
ranks among the ops the machine allows and how far behind the best one it is.
The headline is **survival depth**: how deep the witness gets before its
accumulated deficit passes ``ln W``.

Nothing here runs a beam, and nothing here reads the eval split.

Two task sets, because the model-selection holdout is not fit for this purpose:
``validation_skeleton`` holds out one skeleton in ten by hash, and on v4 that
holdout contains **zero** rows from 1-5-op witnesses and 81.7% from 41+ ones,
while 62% of cohort solves come from 1-5 and none from 41+. So:

- ``holdout``   -- unseen skeletons, honest, but all long witnesses;
- ``fit``       -- a bucket-stratified sample of training tasks, which covers the
                   lengths that produce solves but has been trained on.

If the honest set predicts solves, the proxy is deployable as it stands. If only
the contaminated one does, the recipe needs a bucket-stratified holdout, which is
a change to `validation_skeleton` and a finding in its own right.

    uv run python experiments/search_proxy.py score --policy <ckpt> --out p.json
    uv run python experiments/search_proxy.py correlate --proxy 'dir/*.json' --truth v4only
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from wandering_light.stack_beam import step_all_quiet  # noqa: E402
from wandering_light.stack_pbe import OPS  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    PolicyScorer,
    encode_item,
    encode_targets,
    item_array,
    load_policy,
    validation_skeleton,
)

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41+")
BEAM_WIDTH = 64
TOLERANCE = math.log(BEAM_WIDTH)  # 4.159 nats
TASKS_PER_BUCKET = 24
SAMPLE_SEED = 991

# Verified solves on the 300-task cohort for the #101 1m arm trained on v4 alone,
# from reports/world-plus-v4-policy-20260912/epochs.md. The checkpoints this
# scores reproduce those runs bitwise (same seeds, same recipe, same LR), so
# these are ground truth for the proxy with no search of any kind.
# Verified solves AND behaviours covered on the 300-task cohort for the #101 1m
# arm trained on v4 alone, as (tasks, behaviours of 43). Fifteen cells come from
# reports/world-plus-v4-policy-20260912/epochs/v4only/timing/ and the three
# CE-selected ones (s0 e1, s1 e2, s2 e1) from that report's best-checkpoint
# study. The checkpoints scored here reproduce those runs bitwise, so this is
# ground truth for a selection rule with no search of any kind.
#
# What it shows, and why this file exists: validation CE picks epochs 1/2/1,
# which cover 6, 7 and 6 behaviours. The best epoch of the same run covers 13,
# 11 and 12. Model selection alone is worth 1.89x behaviour coverage.
TRUTH = {
    "v4only": {
        (0, 1): (48, 6),  (0, 2): (55, 8),  (0, 3): (85, 12),
        (0, 4): (94, 12), (0, 5): (84, 12), (0, 6): (91, 13),
        (1, 1): (56, 7),  (1, 2): (63, 7),  (1, 3): (77, 11),
        (1, 4): (74, 10), (1, 5): (80, 11), (1, 6): (72, 10),
        (2, 1): (48, 6),  (2, 2): (54, 7),  (2, 3): (80, 11),
        (2, 4): (89, 11), (2, 5): (92, 12), (2, 6): (76, 12),
    }
}
CE_EPOCH = {0: 1, 1: 2, 2: 1}  # what validation CE selects, per seed


def task_sets(tasks: list[Task]) -> dict[str, list[Task]]:
    train = [t for t in tasks if t.split == "train"]
    holdout = [t for t in train if validation_skeleton(t)]
    fit = [t for t in train if not validation_skeleton(t)]
    rng = random.Random(SAMPLE_SEED)

    def stratify(pool: list[Task]) -> list[Task]:
        by_bucket: dict[str, list[Task]] = defaultdict(list)
        for t in pool:
            by_bucket[co.bucket_of(t.length)].append(t)
        out: list[Task] = []
        for bucket in BUCKETS:
            group = by_bucket.get(bucket, [])
            out += rng.sample(group, min(TASKS_PER_BUCKET, len(group)))
        return out

    return {"holdout": stratify(holdout), "fit": stratify(fit)}


def score_task(task: Task, scorer: PolicyScorer) -> dict[str, Any] | None:
    """Rank and deficit of the authored op at every step of one witness."""
    states = replay_states(task)
    targets = encode_targets(states[-1])
    ranks: list[int] = []
    deficits: list[float] = []
    for step, name in enumerate(task.program):
        product = states[step]
        legal = np.zeros((1, len(OPS)), dtype=bool)
        for index, op in enumerate(OPS):
            moved, _fuel, _stepped = step_all_quiet(product, op)
            if moved is not None:
                legal[0, index] = True
        want = OP_INDEX[name]
        if not legal[0, want]:
            return None  # the witness cannot refuse its own op; a corpus bug
        block = item_array(
            [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
        )
        row = scorer.log_probs([block], legal)[0]
        best = float(np.max(row[legal[0]]))
        mine = float(row[want])
        ranks.append(int((row[legal[0]] > mine).sum()) + 1)
        deficits.append(best - mine)
    cumulative = 0.0
    survival = len(deficits)
    for i, d in enumerate(deficits):
        cumulative += d
        if cumulative > TOLERANCE:
            survival = i
            break
    return {
        "behaviour": task.behaviour,
        "length": task.length,
        "bucket": co.bucket_of(task.length),
        "ranks": ranks,
        "mean_rank": sum(ranks) / len(ranks),
        "top1": sum(1 for r in ranks if r == 1) / len(ranks),
        "mean_deficit": sum(deficits) / len(deficits),
        "first_step_deficit": deficits[0],
        "survival_depth": survival,
        "survives_5": survival >= 5,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {
        "tasks": len(rows),
        "mean_rank": sum(r["mean_rank"] for r in rows) / len(rows),
        "top1": sum(r["top1"] for r in rows) / len(rows),
        "mean_deficit": sum(r["mean_deficit"] for r in rows) / len(rows),
        "mean_survival_depth": sum(r["survival_depth"] for r in rows) / len(rows),
        "frac_survives_5": sum(r["survives_5"] for r in rows) / len(rows),
        "mean_first_step_deficit": sum(r["first_step_deficit"] for r in rows) / len(rows),
    }
    for bucket in BUCKETS:
        group = [r for r in rows if r["bucket"] == bucket]
        if group:
            out[f"survival_{bucket}"] = sum(r["survival_depth"] for r in group) / len(group)
            out[f"top1_{bucket}"] = sum(r["top1"] for r in group) / len(group)
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            share = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = share
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("score")
    s.add_argument("--policy", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--sets", default="holdout,fit")

    c = sub.add_parser("correlate")
    c.add_argument("--proxy", required=True, help="glob over score outputs")
    c.add_argument("--truth", default="v4only")
    c.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.stage == "score":
        if args.out.exists():
            print(f"skip {args.out} (exists)")
            return 0
        if args.device == "cpu":
            torch.set_num_threads(co.CPU_THREADS)
        tasks = co.corpus()
        sets = task_sets(tasks)
        scorer = PolicyScorer(load_policy(args.policy, device=args.device), args.device)
        report: dict[str, Any] = {"policy": str(args.policy), "beam_width": BEAM_WIDTH}
        for name in args.sets.split(","):
            rows = [r for t in sets[name] if (r := score_task(t, scorer)) is not None]
            report[name] = aggregate(rows)
            report[f"{name}_rows"] = rows
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for name in args.sets.split(","):
            a = report[name]
            print(
                f"{args.policy.name} {name}: survival {a['mean_survival_depth']:.2f} "
                f"top1 {a['top1']:.3f} rank {a['mean_rank']:.2f} "
                f"deficit {a['mean_deficit']:.3f}"
            )
        return 0

    # correlate
    truth = TRUTH[args.truth]
    points: list[tuple[int, int, dict[str, Any]]] = []
    for path in sorted(glob.glob(args.proxy)):
        report = json.loads(Path(path).read_text())
        name = Path(report["policy"]).name  # witness-s<seed>-e<epoch>
        try:
            seed = int(name.split("-s")[1].split("-")[0])
            epoch = int(name.split("-e")[1])
        except (IndexError, ValueError):
            continue
        if (seed, epoch) in truth:
            points.append((seed, epoch, report))
    if not points:
        raise SystemExit("no proxy files matched a (seed, epoch) with known solves")

    solves = [truth[(s, e)][0] for s, e, _ in points]
    behaviours = [truth[(s, e)][1] for s, e, _ in points]
    table: dict[str, float] = {}
    table_b: dict[str, float] = {}
    for task_set in ("holdout", "fit"):
        for metric in (
            "mean_survival_depth",
            "frac_survives_5",
            "top1",
            "mean_rank",
            "mean_deficit",
            "mean_first_step_deficit",
            *(f"survival_{b}" for b in BUCKETS),
            *(f"top1_{b}" for b in BUCKETS),
        ):
            vals = [r.get(task_set, {}).get(metric) for _s, _e, r in points]
            if any(v is None for v in vals):
                continue
            table[f"{task_set}.{metric}"] = spearman(vals, solves)
            table_b[f"{task_set}.{metric}"] = spearman(vals, behaviours)

    print(
        f"{len(points)} checkpoints with known outcomes: "
        f"{min(solves)}-{max(solves)} tasks, {min(behaviours)}-{max(behaviours)} behaviours"
    )
    print(f"\n{'signal':42s} {'rho vs tasks':>13s} {'rho vs behaviours':>18s}")
    for key, rho in sorted(table.items(), key=lambda kv: -abs(kv[1])):
        print(f"{key:42s} {rho:>13.3f} {table_b.get(key, float('nan')):>18.3f}")

    # What a selection rule built on each signal would actually have picked.
    by_seed: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for seed, epoch, report in points:
        by_seed[seed].append((epoch, report))
    print(f"\n{'selection rule':42s} {'tasks':>8s} {'behaviours/43':>14s}")
    ce = [truth[(s, CE_EPOCH[s])] for s in by_seed if (s, CE_EPOCH[s]) in truth]
    if ce:
        print(f"{'validation cross-entropy (the status quo)':42s} "
              f"{sum(t for t, _ in ce) / len(ce):>8.1f} {sum(b for _, b in ce) / len(ce):>14.1f}")
    selections: dict[str, tuple[float, float]] = {}
    for key in table:
        task_set, metric = key.split(".", 1)
        lower_is_better = ("rank" in metric) or ("deficit" in metric)
        picks = []
        for seed, entries in by_seed.items():
            scored = [
                (e, r.get(task_set, {}).get(metric))
                for e, r in entries
                if r.get(task_set, {}).get(metric) is not None and (seed, e) in truth
            ]
            if not scored:
                continue
            pick = min(scored, key=lambda x: x[1])[0] if lower_is_better else max(scored, key=lambda x: x[1])[0]
            picks.append(truth[(seed, pick)])
        if picks:
            selections[key] = (
                sum(t for t, _ in picks) / len(picks),
                sum(b for _, b in picks) / len(picks),
            )
    for key, (t, b) in sorted(selections.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"{key:42s} {t:>8.1f} {b:>14.1f}")
    oracle = [
        max((truth[(s, e)] for e, _ in entries if (s, e) in truth), key=lambda v: v[1])
        for s, entries in by_seed.items()
    ]
    print(f"{'ORACLE (peeks at the cohort; upper bound)':42s} "
          f"{sum(t for t, _ in oracle) / len(oracle):>8.1f} "
          f"{sum(b for _, b in oracle) / len(oracle):>14.1f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "checkpoints": [(s, e, *truth[(s, e)]) for s, e, _ in points],
                    "spearman_tasks": table,
                    "spearman_behaviours": table_b,
                    "selection": selections,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
