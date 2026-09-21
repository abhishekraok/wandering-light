"""mine-v4-library (#148): where along a staged witness does the policy's deficit sit?

A cycle library compresses staged v4 witnesses 1.7x in length, but a beam pays in
summed per-step deficit (`search_proxy.py`), not in length. This replays staged
witnesses under the staged-trained policies of staged-witness-reauthoring (#155)
and asks how much of each witness's summed deficit a macro could remove.

A **cycle** is a maximal run of adjacent top-level quotations immediately
followed by a combinator (`[ body ] fold`, `[ c ] [ t ] [ e ] ifte`); nested
cycles are ignored. The **library** is the top ``--library`` cycles by ops saved,
selected on fit behaviours only. A macro replaces a cycle's steps by one action,
so per occurrence it saves at most the cycle's summed deficit (``sum``: the macro
is always ranked first) and, if choosing *which* cycle stays as hard as the
cycle's hardest step, only ``sum - max`` (``tail``).

Analysis, not a method: nothing here trains, searches, or selects a checkpoint.

    uv run python experiments/cycle_deficit.py \
        --run-dir checkpoints/staged-arms-20260920 --out reports/cycle-deficit-20260920
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from experiments.staged_arms import EPOCHS, SEEDS, treated, with_program  # noqa: E402
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

COMBINATORS = {"map", "filter", "fold", "i", "dip", "ifte"}
TOLERANCE = math.log(64)


def cycle_spans(tokens: tuple[str, ...]) -> list[tuple[int, int]]:
    """``(start, end)`` token spans, end exclusive, of every top-level cycle."""
    spans, depth, run_start = [], 0, None
    for i, tok in enumerate(tokens):
        if tok == "[":
            if depth == 0 and run_start is None:
                run_start = i
            depth += 1
        elif tok == "]":
            depth -= 1
        elif depth == 0:
            if tok in COMBINATORS and run_start is not None:
                spans.append((run_start, i + 1))
            run_start = None
    return spans


def library(programs: list[tuple[str, ...]], size: int) -> list[tuple[str, ...]]:
    count = Counter(p[a:b] for p in programs for a, b in cycle_spans(p))
    ranked = sorted(count.items(), key=lambda kv: (-(len(kv[0]) - 1) * kv[1], kv[0]))
    return [cycle for cycle, _ in ranked[:size]]


def deficits(task, scorer: PolicyScorer) -> list[float] | None:
    """Per-step deficit along ``task.program``; the `search_proxy.score_task` walk, batched."""
    states = replay_states(task)
    targets = encode_targets(states[-1])
    legal = np.zeros((len(task.program), len(OPS)), dtype=bool)
    blocks = []
    for step, name in enumerate(task.program):
        for index, op in enumerate(OPS):
            moved, _fuel, _stepped = step_all_quiet(states[step], op)
            legal[step, index] = moved is not None
        if not legal[step, OP_INDEX[name]]:
            return None
        blocks.append(
            item_array(
                [encode_item(s, t) for s, t in zip(states[step], targets, strict=True)]
            )
        )
    rows = scorer.log_probs(blocks, legal)
    return [
        float(np.max(rows[i][legal[i]]) - rows[i][OP_INDEX[name]])
        for i, name in enumerate(task.program)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="staged")
    parser.add_argument("--library", type=int, default=35)
    parser.add_argument(
        "--tasks", type=int, default=4, help="train tasks per behaviour"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)

    side = treated(keep_contrived=False)
    picked: dict[str, list] = defaultdict(list)
    for task in co.corpus():
        wanted = task.split == "train" and task.behaviour in side
        if wanted and len(picked[task.behaviour]) < args.tasks:
            picked[task.behaviour].append(
                with_program(task, side[task.behaviour]["program"])
            )
    held = {b for b, ts in picked.items() if validation_skeleton(ts[0])}
    fit_programs = [ts[0].program for b, ts in picked.items() if b not in held]
    lib = set(library(fit_programs, args.library))
    print(
        f"{len(picked)} behaviours ({len(held)} held out), library {len(lib)}",
        flush=True,
    )

    occurrences: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    glue_ops: dict[tuple[bool, str], list[float]] = defaultdict(list)
    for seed in SEEDS:
        policy = args.run_dir / "policies" / f"{args.arm}-s{seed}-e{EPOCHS}"
        scorer = PolicyScorer(load_policy(policy, device=args.device), args.device)
        for behaviour, tasks in sorted(picked.items()):
            for task in tasks:
                d = deficits(task, scorer)
                if d is None:
                    print(
                        f"  SKIP {behaviour}: a program refused its own op", flush=True
                    )
                    continue
                total = sum(d)
                where = {"library": 0.0, "other_cycle": 0.0, "glue": total}
                steps = {"library": 0, "other_cycle": 0, "glue": len(d)}
                tail = 0.0
                in_cycle = [False] * len(d)
                for a, b in cycle_spans(task.program):
                    in_cycle[a:b] = [True] * (b - a)
                    kind = "library" if task.program[a:b] in lib else "other_cycle"
                    inside = d[a:b]
                    where[kind] += sum(inside)
                    where["glue"] -= sum(inside)
                    steps[kind] += b - a
                    steps["glue"] -= b - a
                    if kind == "library":
                        tail += sum(inside) - max(inside)
                        occurrences.append(
                            {
                                "seed": seed,
                                "held": behaviour in held,
                                "cycle": " ".join(task.program[a:b]),
                                "deficits": inside,
                            }
                        )
                for op, cost, covered in zip(task.program, d, in_cycle, strict=True):
                    if not covered:
                        glue_ops[(behaviour in held, op)].append(cost)
                witnesses.append(
                    {
                        "seed": seed,
                        "behaviour": behaviour,
                        "held": behaviour in held,
                        "length": len(d),
                        "total": total,
                        "where": where,
                        "steps": steps,
                        "library_tail": tail,
                    }
                )
        print(f"  seed {seed} scored", flush=True)

    summary: dict[str, Any] = {"library": sorted(" ".join(c) for c in lib)}
    for name, keep in (("fit", False), ("held", True)):
        rows = [w for w in witnesses if w["held"] is keep]
        total = sum(w["total"] for w in rows)
        length = sum(w["length"] for w in rows)
        lib_sum = sum(w["where"]["library"] for w in rows)
        lib_tail = sum(w["library_tail"] for w in rows)
        lib_steps = sum(w["steps"]["library"] for w in rows)
        lib_uses = sum(1 for o in occurrences if o["held"] is keep)
        # One value per behaviour: the median over its tasks and seeds.
        per: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for w in rows:
            per[w["behaviour"]].append(w)

        def inside(field, per=per) -> int:
            return sum(median(field(w) for w in ws) < TOLERANCE for ws in per.values())

        summary[name] = {
            "behaviours": len(per),
            "witness_scores": len(rows),
            "median_total_deficit": median(w["total"] for w in rows),
            "share_of_steps": {
                k: sum(w["steps"][k] for w in rows) / length
                for k in ("library", "other_cycle", "glue")
            },
            "share_of_deficit": {
                k: sum(w["where"][k] for w in rows) / total
                for k in ("library", "other_cycle", "glue")
            },
            "mean_deficit_per_step": {
                k: sum(w["where"][k] for w in rows)
                / max(1, sum(w["steps"][k] for w in rows))
                for k in ("library", "other_cycle", "glue")
            },
            "compression_in_steps": length / (length - lib_steps + lib_uses),
            "compression_in_deficit_tail": total / (total - lib_tail),
            "compression_in_deficit_upper": total / (total - lib_sum),
            "behaviours_inside_budget": {
                "raw": inside(lambda w: w["total"]),
                "minus_tail": inside(lambda w: w["total"] - w["library_tail"]),
                "minus_sum": inside(lambda w: w["total"] - w["where"]["library"]),
            },
        }
    profile = defaultdict(list)
    for o in occurrences:
        profile[(o["held"], o["cycle"])].append(o["deficits"])
    summary["profiles"] = [
        {
            "held": held_flag,
            "cycle": cycle,
            "uses": len(rows),
            "mean_by_position": [
                round(mean(col), 3) for col in zip(*rows, strict=True)
            ],
        }
        for (held_flag, cycle), rows in sorted(
            profile.items(), key=lambda kv: -len(kv[1])
        )
    ]
    summary["glue_ops"] = [
        {
            "held": flag,
            "op": op,
            "steps": len(costs),
            "sum": sum(costs),
            "mean": mean(costs),
        }
        for (flag, op), costs in sorted(glue_ops.items(), key=lambda kv: -sum(kv[1]))
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cycle_deficit.json").write_text(json.dumps(summary, indent=1))
    (args.out / "witnesses.json").write_text(json.dumps(witnesses))
    for name in ("fit", "held"):
        print(name, json.dumps(summary[name], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
