"""Does the front/back asymmetry persist one step inward?

`reports/policy-probe-20260824/` measured the two ends of a program and found
they disagree. At the front a dict on `(input type, output type)` matches a
trained encoder (3.18 vs 2.86 mean rank, 42.1% vs 42.2% top-1); at the back the
encoder nearly doubles top-1 (22.3% -> 42.1%). The type mask explains it: 9.87
functions consume a given input type but 25.88 produce a given output type, so
the back has more left to learn.

That argues for bidirectional search — but only if the asymmetry survives
inward. If it collapses after one step, neither direction has much to offer and
#33 should shrink to "BFS ordered by a lookup table". This is the cheap check
that decides which.

**Peeling, and why it is symmetric.** A witness `w_1..w_d` certified shortest
gives a sub-task at each end, each with an exact distance:

    front, depth k:  (s_k, output)      distance d-k, predict its FIRST action
    back,  depth k:  (input, s_{d-k})   distance d-k, predict its LAST action

`s_j` is the witness replayed `j` steps. Both sub-tasks are genuine tasks with
certified distance `d-k` by the prefix argument (see `value_probe.py`), both
carry the type mask that applies to their end, and at `k = 0` both reduce
exactly to the published probes. Peeling the *target* end is what makes the
back comparable: a second-to-last action must produce `s_{d-1}`, and masking it
on the original output type would be masking on the wrong field.

**The labels are witness-derived, so they are a subset.** Only the two outermost
actions ship a certified-complete optimal set; deeper in, all that is known is
that the witness's op is optimal. Measured on test, the complete set is a
singleton for 93.0% of first actions and 86.2% of last ones, and the witness op
is in the set every time — so a witness label is exactly right most of the time
and never wrong, only incomplete. Incompleteness can only *understate* a
predictor, so `k = 0` is rerun with witness labels alongside the published
complete-label numbers to size the gap before any `k = 1` number is read.

All predictors and the scorer are imported from `policy_probe`, unchanged, so
these numbers sit on the same scale as the published ones.

**Both encoders, several seeds.** The learned rung runs twice over: once with
the published truncating encoder and once with the per-item pair encoder that
replaces it (see `policy_probe`), so the old/new difference is measured within
one run on one device rather than against a published table produced on other
hardware. Each is trained once per seed, because a ~90k-parameter model at 12
epochs can have seed spread comparable to the effect being measured -- and if
the gap between the encoders falls inside that spread, there is no effect to
report.

    uv run python experiments/policy_probe_depth.py [corpus-dir] [--epochs=N]
        [--depths=0,1,2] [--seeds=0,1,2] [--device=cuda]
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_probe import (
    BITS_EPS,
    baseline_ladder,
    basis_names,
    evaluate,
    resolve_corpus,
    train_learned,
    truncation_rates,
    type_table,
)

SPLITS = ("discovery", "validation", "test")
FIRST = "optimal_first_action_names"
LAST = "optimal_last_action_names"


def type_name(typed_list) -> str:
    """`builtins.int`, matching the corpus metadata and the basis type fields."""
    item = typed_list.item_type
    return f"{item.__module__}.{item.__qualname__}"


def peel(corpus: Path, split: str, by_name, executor, depth: int, min_distance: int):
    """Sub-tasks obtained by peeling `depth` steps off each end of a witness.

    `min_distance` holds the task population fixed across depths. Without it a
    `k = 0` row would include short tasks that `k = 1` cannot form, and the
    comparison would confound depth with task length.
    """
    from wandering_light.typed_list import TypedList

    front, back = [], []
    with gzip.open(corpus / f"{split}.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            meta = row["metadata"]
            names = row["witness_function_names"]
            distance = meta["certified_distance"]
            if distance < min_distance:
                continue
            states = [TypedList.from_str(row["input"])]
            for name in names:
                states.append(executor.execute(by_name[name], states[-1]))

            head = states[depth]
            front.append(
                {
                    "input": head.to_string(),
                    "output": row["output"],
                    "metadata": {
                        "input_type": type_name(head),
                        "output_type": meta["output_type"],
                        "certified_distance": distance - depth,
                        FIRST: [names[depth]],
                    },
                }
            )
            tail = states[distance - depth]
            back.append(
                {
                    "input": row["input"],
                    "output": tail.to_string(),
                    "metadata": {
                        "input_type": meta["input_type"],
                        "output_type": type_name(tail),
                        "certified_distance": distance - depth,
                        LAST: [names[distance - depth - 1]],
                    },
                }
            )
    return front, back


def run(names, splits, last: bool, label: str, epochs: int, seeds, device: str):
    """The `policy_probe` ladder, on one end at one depth, both encoders.

    Each encoder is trained once per seed. A ~90k-parameter model at 12 epochs
    can have seed spread comparable to the effect being measured, so a single
    run of each would not distinguish "the encoder sees more" from "this seed
    landed better".
    """
    type_of, match_key = type_table(last)
    names_field = LAST if last else FIRST
    train, valid, test = splits
    rows = baseline_ladder(names, train, test, type_of, match_key, names_field)
    for encoder in ("flat", "pairs"):
        for seed in seeds:
            print(f"  training {label} {encoder} seed {seed}...", file=sys.stderr)
            row = evaluate(
                train_learned(
                    names,
                    train,
                    valid,
                    type_of,
                    match_key,
                    names_field,
                    epochs=epochs,
                    seed=seed,
                    conv=True,
                    encoder=encoder,
                    device=device,
                ),
                test,
                f"learned ({encoder})",
                names_field,
                names=names,
                type_of=type_of,
                match_key=match_key,
                normalize="softmax",
            )
            row["seed"] = seed
            rows.append(row)
    for row in rows:
        row["probe"] = label
    return rows


def spread(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation, for cells averaged over seeds."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def collapse(rows: list[dict]) -> list[dict]:
    """One line per (probe, predictor), with seed spread where there is any."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["probe"], row["predictor"]), []).append(row)
    out = []
    for (probe, predictor), group in grouped.items():
        cell = {"probe": probe, "predictor": predictor, "seeds": len(group)}
        for key in ("top1", "mean_rank"):
            cell[key], cell[key + "_sd"] = spread([g[key] for g in group])
        if group[0].get("bits"):
            cell["bits"] = {}
            for eps in BITS_EPS:
                name = f"{eps:g}"
                mean, sd = spread([g["bits"][name]["mean"] for g in group])
                median, _ = spread([g["bits"][name]["median"] for g in group])
                unreachable, _ = spread([g["bits"][name]["unreachable"] for g in group])
                cell["bits"][name] = {
                    "mean": mean,
                    "mean_sd": sd,
                    "median": median,
                    "unreachable": unreachable,
                }
            cell["mask_width"] = group[0]["mask_width"]
        out.append(cell)
    return out


def main() -> None:
    argv = [a for a in sys.argv if not a.startswith("--")]
    flags = [a for a in sys.argv if a.startswith("--")]
    epochs = next(
        (int(a.split("=")[1]) for a in flags if a.startswith("--epochs=")), 12
    )
    depths = next(
        (
            [int(d) for d in a.split("=")[1].split(",")]
            for a in flags
            if a.startswith("--depths=")
        ),
        [0, 1, 2],
    )
    seeds = next(
        (
            [int(s) for s in a.split("=")[1].split(",")]
            for a in flags
            if a.startswith("--seeds=")
        ),
        [0, 1, 2],
    )
    device = next((a.split("=")[1] for a in flags if a.startswith("--device=")), "cpu")
    # Every depth is scored on the same tasks, so a difference between rows is
    # depth and not task length.
    min_distance = max(depths) + 1

    corpus = resolve_corpus(argv)
    from wandering_light.basis_set import load_basis_set
    from wandering_light.executor import Executor

    functions = load_basis_set("wl-core-v1").to_function_def_set()
    by_name = {f.name: f for f in functions.functions}
    executor = Executor(functions)
    names = basis_names()

    all_rows, budget = [], {}
    for depth in depths:
        peeled = [
            peel(corpus, split, by_name, executor, depth, min_distance)
            for split in SPLITS
        ]
        for last, end in ((False, "front"), (True, "back")):
            splits = tuple(p[1 if last else 0] for p in peeled)
            label = f"{end} k={depth}"
            rates = truncation_rates(splits[2])
            budget[label] = rates
            print(
                f"\n== {label}: {len(splits[0])} train / {len(splits[2])} test"
                f"   over budget: flat {rates['flat_tasks_over_budget']:.1%} of "
                f"tasks, pairs {rates['pairs_items_over_budget']:.2%} of items",
                file=sys.stderr,
            )
            all_rows.extend(run(names, splits, last, label, epochs, seeds, device))

    cells = collapse(all_rows)
    print(
        f"\nDepth policy probe -- witness-derived labels, "
        f"tasks with d >= {min_distance}, {len(seeds)} seeds\n"
    )
    header = f"{'probe':<14} {'predictor':<20} {'top-1':>14} {'mean rank':>14}"
    header += "".join(f"{('H e=' + f'{e:g}'):>9}" for e in BITS_EPS)
    print(header)
    for cell in cells:
        top1 = f"{cell['top1']:.1%} ±{cell['top1_sd']:.1%}"
        rank = f"{cell['mean_rank']:.2f} ±{cell['mean_rank_sd']:.2f}"
        bits = "".join(
            f"{cell['bits'][f'{e:g}']['mean']:>9.2f}" if cell.get("bits") else ""
            for e in BITS_EPS
        )
        print(
            f"{cell['probe']:<14} {cell['predictor']:<20} {top1:>14} {rank:>14}{bits}"
        )
    Path("policy_probe_depth.json").write_text(
        json.dumps(
            {"truncation": budget, "cells": cells, "rows": all_rows},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
