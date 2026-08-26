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

    uv run python experiments/policy_probe_depth.py [corpus-dir] [--epochs=N]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_probe import (
    basis_names,
    evaluate,
    make_conditional,
    make_marginal,
    make_type_mask,
    make_uniform,
    resolve_corpus,
    train_learned,
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


def run(names, splits, last: bool, label: str, epochs: int):
    """The `policy_probe` ladder, on one end at one depth."""
    type_of, match_key = type_table(last)
    names_field = LAST if last else FIRST
    train, valid, test = splits
    rows = [
        evaluate(make_uniform(names), test, "uniform over all", names_field),
        evaluate(
            make_type_mask(names, type_of, match_key), test, "type mask", names_field
        ),
        evaluate(
            make_marginal(names, train, names_field),
            test,
            "marginal frequency",
            names_field,
        ),
        evaluate(
            make_conditional(names, train, type_of, match_key, names_field),
            test,
            "conditional lookup",
            names_field,
        ),
    ]
    print(f"  training {label} conv encoder...", file=sys.stderr)
    rows.append(
        evaluate(
            train_learned(
                names,
                train,
                valid,
                type_of,
                match_key,
                names_field,
                epochs=epochs,
                conv=True,
            ),
            test,
            "learned (conv)",
            names_field,
        )
    )
    for row in rows:
        row["probe"] = label
    return rows


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

    all_rows = []
    for depth in depths:
        peeled = [
            peel(corpus, split, by_name, executor, depth, min_distance)
            for split in SPLITS
        ]
        for last, end in ((False, "front"), (True, "back")):
            splits = tuple(p[1 if last else 0] for p in peeled)
            print(
                f"\n== {end} k={depth}: {len(splits[0])} train / {len(splits[2])} test",
                file=sys.stderr,
            )
            all_rows.extend(run(names, splits, last, f"{end} k={depth}", epochs))

    print(
        f"\nDepth policy probe -- witness-derived labels, "
        f"tasks with d >= {min_distance}\n"
    )
    print(f"{'probe':<14} {'predictor':<22} {'top-1':>8} {'mean rank':>10}")
    for row in all_rows:
        print(
            f"{row['probe']:<14} {row['predictor']:<22} {row['top1']:>7.1%} "
            f"{row['mean_rank']:>10.2f}"
        )
    Path("policy_probe_depth.json").write_text(json.dumps(all_rows, indent=1))


if __name__ == "__main__":
    main()
