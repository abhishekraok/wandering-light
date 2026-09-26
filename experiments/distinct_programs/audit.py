"""Signature audit of new behaviours against v4 and each other (distinct-programs, #174).

A new behaviour is compared with every v4 catalogue behaviour whose arguments
have the same shape (ints and lists, in order), on the new behaviour's own
tests plus 40 inputs drawn from the v4 behaviour's sampler. Agreement is the
share of those inputs on which both references return the same value.

    duplicate        agreement 1.0 with any v4 behaviour: dropped
    near_eval        agreement >= 0.8 with a behaviour of the 43-behaviour eval
                     cohort: dropped and recorded, since solving it would be
                     retrieval by construction
    same_as <id>     computes the same function as an earlier new behaviour on
                     the pooled inputs of its shape group: dropped
    keep             otherwise

    uv run python experiments/distinct_programs/audit.py B.jsonl --out audit.jsonl --keep kept.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from experiments import compute_optimal_mark28 as co
from experiments.world_behaviours.extract import as_plain, run_solution
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.stack_pbe import CATALOGUE, _name_seed, load

SAMPLES = 40
NEAR = 0.8
V4 = Path("wandering_light/evals/data/stack_pbe_v4/manifest.json")


def shape(args) -> tuple[str, ...]:
    return tuple("list" if isinstance(a, list) else "int" for a in args)


def plain_args(args) -> list:
    return [as_plain(a) for a in args]


def cohort_names() -> set[str]:
    tasks = load(fetch_corpus(V4))
    return {task.behaviour for _, task in co.cohort(tasks)}


def v4_reference(behaviour, args):
    try:
        return as_plain(behaviour.reference(*args))
    except Exception:  # a reference refusing an input skips it
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("behaviours", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(x) for x in args.behaviours.read_text().splitlines() if x]
    eval_names = cohort_names()

    v4_inputs = {}
    for b in CATALOGUE:
        rng = random.Random(_name_seed(b.name) + 7)
        drawn = [plain_args(b.sample(rng)) for _ in range(SAMPLES)]
        v4_inputs[b.name] = (shape(drawn[0]), drawn)

    verdicts, groups = [], defaultdict(list)
    for row in rows:
        own = [t["args"] for t in row["tests"]]
        sig = shape(own[0])
        groups[sig].append(row)
        best: tuple[float, str] = (0.0, "")
        best_eval: tuple[float, str] = (0.0, "")
        matched = [b for b in CATALOGUE if v4_inputs[b.name][0] == sig]
        inputs = own + [a for b in matched for a in v4_inputs[b.name][1]]
        mine = run_solution(row["solution"], row["entry"], inputs) if matched else []
        mine_own, offset = mine[: len(own)], len(own)
        for b in matched:
            drawn = v4_inputs[b.name][1]
            mine_b = mine_own + mine[offset : offset + len(drawn)]
            offset += len(drawn)
            theirs = [v4_reference(b, a) for a in own + drawn]
            pairs = [
                (m[1], t)
                for m, t in zip(mine_b, theirs, strict=True)
                if m[0] == "ok" and t is not None
            ]
            if len(pairs) < 10:
                continue
            agree = sum(m == t for m, t in pairs) / len(pairs)
            best = max(best, (agree, b.name))
            if b.name in eval_names:
                best_eval = max(best_eval, (agree, b.name))
        verdict = "keep"
        if best[0] == 1.0:
            verdict = "duplicate"
        elif best_eval[0] >= NEAR:
            verdict = "near_eval"
        verdicts.append(
            {
                "id": row["id"],
                "verdict": verdict,
                "closest_v4": best[1],
                "agreement": round(best[0], 3),
                "closest_eval": best_eval[1],
                "eval_agreement": round(best_eval[0], 3),
            }
        )

    # New against new: identical outputs on the pooled inputs of a shape group.
    by_id = {v["id"]: v for v in verdicts}
    for group in groups.values():
        pooled = [t["args"] for row in group for t in row["tests"]][:400]
        seen: dict[str, str] = {}
        for row in group:
            if by_id[row["id"]]["verdict"] != "keep":
                continue
            outs = run_solution(row["solution"], row["entry"], pooled)
            key = json.dumps(outs)
            if key in seen:
                by_id[row["id"]]["verdict"] = f"same_as {seen[key]}"
            else:
                seen[key] = row["id"]

    args.out.write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    kept = [r for r in rows if by_id[r["id"]]["verdict"] == "keep"]
    args.keep.write_text("".join(json.dumps(r) + "\n" for r in kept))
    counts = defaultdict(int)
    for v in verdicts:
        counts[v["verdict"].split()[0]] += 1
    print(json.dumps({"rows": len(rows), "kept": len(kept), **counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
