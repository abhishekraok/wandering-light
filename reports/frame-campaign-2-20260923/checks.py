"""Checks on campaign 2 asked for after its readings, before the results are posted.

1. Retrieval or generalisation: the learned charge per `[`, split by whether the
   eval witness's skeleton (its sequence of commitments) occurs verbatim among
   the ranker's train-split witnesses.
2. Charge per sub-search: under decomposition a `[` is paid by the search that
   writes it, the top level or the enclosing body, not by the witness as a whole.
3. The exchange table under the learned charge: clears at `s0-e5` as calibrated
   defined-body deficit is scaled, commitment charged by ranker seed 0. It reads
   per-op rows from the run dirs under `checkpoints/`, which are not committed.
4. The last op before `]`, v4: at how many states is the witness op the only
   op that maps every example's frame to its exit, and what does the oracle
   pay there against the world arm (median over the 18 paired cells)?
5. The one-level control, frames_inner over stacked frames on identical
   inputs, without the first gate's outlier, and how many cells sit above 1.
6. World-v3's stacking ratio at 2 or more levels with each witness left out.
7. Where the world arm fails at `s0-e5` once defined-body deficit is removed:
   in a top-level segment or inside a body, with commitment free or learned.
8. The share of interior defined-body deficit in bodies of 1-2, 3-6 and 7+
   ops, per arm, median over cells.
9. Clears at epoch 5 of each seed under the learned charge, world and oracle.

Check 2's segmented rows cut the top level after each top-level combinator,
as the clears composite does; its unsegmented row sums the whole top level.

Run from the repository root: `uv run python reports/frame-campaign-2-20260923/checks.py`.
Reads `ranker.json` beside it; writes `checks.json`.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.frame_campaign2 as fc  # noqa: E402
import experiments.world_arm as wa  # noqa: E402
from experiments.frame_gate import (  # noqa: E402
    STRUCTURAL,
    commit_charges,
    commitments,
    eval_witnesses,
    exchange_rate,
    step_classes,
)
from experiments.frame_rows import example_walks, level_exits  # noqa: E402
from experiments.inner_frames import BUDGET as RUN_BUDGET  # noqa: E402
from experiments.inner_frames import PREVIEW_DEPTH  # noqa: E402
from experiments.segment_composite import levels  # noqa: E402
from wandering_light.stack_machine import Sym, run_body  # noqa: E402
from wandering_light.stack_pbe import OPS  # noqa: E402

CORPORA = ("stack_pbe_v4", "stack_pbe_world_v3")
BUDGET = math.log(64)
CHECKPOINTS = HERE.parents[1] / "checkpoints"
EXCHANGE = {
    "world": CHECKPOINTS / "frame-arm-20260922/measure/world-s0-e5.json",
    "frames_inner": CHECKPOINTS / "frame-campaign-2/measure/frames_inner-s0-e5.json",
    "exit_inner": CHECKPOINTS / "frame-campaign-2/measure/exit_inner-s0-e5.json",
}


def skeleton(task) -> tuple:
    return tuple(
        choice
        for _step, (_shape, choice) in sorted(
            commitments(task.program, task.inputs[0]).items()
        )
    )


def fmt(s: dict) -> str:
    return f"{s.get('median', math.nan):.3f}/{s.get('mean', math.nan):.3f} n{s['n']}"


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "over_budget": sum(v > BUDGET for v in values) / len(values),
    }


def groups(program) -> dict[int | str, list[int]]:
    """Each `[` step, keyed by the sub-search that writes it: 'top' or its enclosing `[`."""
    out: dict[int | str, list[int]] = defaultdict(list)
    stack: list[int] = []
    for step, name in enumerate(program):
        if name == "[":
            out[stack[-1] if stack else "top"].append(step)
            stack.append(step)
        elif name == "]" and stack:
            stack.pop()
    return out


def unique_last_ops(corpus: str) -> tuple[dict, set[tuple[str, int]]]:
    """Last-op states by how many ops map every example's frame to its exit."""
    counts: dict[str, int] = defaultdict(int)
    unique = set()
    for task in eval_witnesses(corpus):
        walks = example_walks(task)
        classes, _ = step_classes(task, walks)
        exits = [level_exits(task.program, views) for _s, views in walks]
        prog = task.program
        for step, name in enumerate(prog):
            if classes[step] != "defined" or name in STRUCTURAL:
                continue
            if not (step + 1 < len(prog) and prog[step + 1] == "]"):
                continue
            frames = [views[step][-1][3] for _s, views in walks]
            ends = [e[step][-1] if e.get(step) else None for e in exits]
            if any(x is None for x in ends):
                counts["exit undefined"] += 1
                continue

            def maps(op: str, frames=frames, ends=ends) -> bool:
                for frame, end in zip(frames, ends, strict=True):
                    out = run_body((Sym(op),), frame, RUN_BUDGET, PREVIEW_DEPTH)
                    if isinstance(out, str) or tuple(out[0]) != tuple(end):
                        return False
                return True

            counts["states"] += 1
            counts["witness maps"] += maps(name)
            mapping = [o.name for o in OPS if o.name not in STRUCTURAL and maps(o.name)]
            if mapping == [name]:
                counts["witness op is the only one"] += 1
                unique.add((task.behaviour, step))
    return dict(counts), unique


def paired_ratio(cells, num: str, den: str, corpus: str, keep) -> dict:
    """Median over paired cells of calibrated deficit totals on the steps ``keep`` selects."""

    def total(cell) -> float:
        t = cell["calibration"]["temperature"]
        return sum(
            d / t
            for row in cell[corpus]["machine_mask"]["rows"]
            for step, d in enumerate(row["deficits"])
            if d is not None and keep(row["behaviour"], step, row["classes"][step])
        )

    ratios = [
        total(cells[num][k]) / total(cells[den][k])
        for k in sorted(set(cells[num]) & set(cells[den]))
    ]
    return {
        "median": statistics.median(ratios),
        "range": [min(ratios), max(ratios)],
        "cells_above_1": sum(r > 1 for r in ratios),
        "cells": len(ratios),
    }


def main() -> None:
    ranker = json.loads((HERE / "ranker.json").read_text())
    fit, held = co.split_tasks(wa.configure())
    train = [*fit, *held]
    seen_skeletons = {skeleton(t) for t in train}
    seen_programs = {tuple(t.program) for t in train}
    out: dict = {}
    world = json.loads(EXCHANGE["world"].read_text())["cells"]
    for corpus in CORPORA:
        kinds = {
            r["behaviour"]: r["kinds"] for r in world[corpus]["machine_mask"]["rows"]
        }
        tasks = eval_witnesses(corpus)
        prior = commit_charges(corpus, tasks)
        charged = {"prior": prior, **{s: ranker[s]["charges"][corpus] for s in ranker}}
        rows = []
        for t in tasks:
            sk = skeleton(t)
            if not sk:
                continue
            rows.append(
                {
                    "behaviour": t.behaviour,
                    "opens": len(sk),
                    "skeleton_seen": sk in seen_skeletons,
                    "program_seen": tuple(t.program) in seen_programs,
                }
            )
        c: dict = {
            "witnesses_with_opens": len(rows),
            "skeleton_seen": sum(r["skeleton_seen"] for r in rows),
            "program_seen": sum(r["program_seen"] for r in rows),
            "opens_in_seen": sum(r["opens"] for r in rows if r["skeleton_seen"]),
            "opens_in_unseen": sum(r["opens"] for r in rows if not r["skeleton_seen"]),
            "per_open": {},
            "per_witness": {},
            "per_sub_search": {},
        }
        seen_of = {r["behaviour"]: r["skeleton_seen"] for r in rows}
        for name, charge in charged.items():
            per_open = {"seen": [], "unseen": []}
            per_witness = {"seen": [], "unseen": []}
            top, bodies, nested_bodies, worst = [], [], [], []
            top_segment, worst_segmented = [], []
            for t in tasks:
                if t.behaviour not in seen_of:
                    continue
                key = "seen" if seen_of[t.behaviour] else "unseen"
                steps = charge[t.behaviour]
                opens = [steps[s] for s, n in enumerate(t.program) if n == "["]
                per_open[key] += opens
                per_witness[key].append(sum(opens))
                g = groups(t.program)
                sums = {k: sum(steps[s] for s in v) for k, v in g.items()}
                top.append(sums.get("top", 0.0))
                # Every quotation is a body; one with no `[` inside pays nothing.
                for s, n in enumerate(t.program):
                    if n == "[":
                        bodies.append(sums.get(s, 0.0))
                        if s in sums:
                            nested_bodies.append(sums[s])
                worst.append(max(sums.values()))
                segs, per_body = levels(t.program, steps, kinds[t.behaviour])
                top_segment.append(max(segs))
                worst_segmented.append(max(*segs, *per_body))
            c["per_open"][name] = {k: stats(v) for k, v in per_open.items()}
            c["per_witness"][name] = {k: stats(v) for k, v in per_witness.items()}
            c["per_sub_search"][name] = {
                "top_level": stats(top),
                "worst_top_segment": stats(top_segment),
                "worst_sub_search_segmented": stats(worst_segmented),
                "every_body": stats(bodies),
                "bodies_with_nested_opens": stats(nested_bodies),
                "worst_sub_search_per_witness": stats(worst),
            }
        c["unseen_behaviours"] = sorted(
            r["behaviour"] for r in rows if not r["skeleton_seen"]
        )
        out[corpus] = c
    out["exchange_learned"] = {}
    for arm, path in EXCHANGE.items():
        cells = json.loads(path.read_text())["cells"]
        t = cells["calibration"]["temperature"]
        out["exchange_learned"][arm] = {
            corpus: [
                {
                    "scale": r["scale"],
                    "free": r["free_commitment"]["both"],
                    "learned": r["with_commitment"]["both"],
                }
                for r in exchange_rate(
                    cells[corpus]["machine_mask"]["rows"],
                    t,
                    ranker["s0"]["charges"][corpus],
                )
            ]
            for corpus in CORPORA
        }
        print(
            "exchange",
            arm,
            {
                c: [(r["scale"], r["learned"], r["free"]) for r in v]
                for c, v in out["exchange_learned"][arm].items()
            },
        )
    cells = fc.load_cells(fc.RUN_DIR, fc.GATE_DIR)
    counts, unique = unique_last_ops("stack_pbe_v4")
    out["last_op_v4"] = {
        **counts,
        "exit_inner/world, unique": paired_ratio(
            cells,
            "exit_inner",
            "world",
            "stack_pbe_v4",
            lambda b, s, _c: (b, s) in unique,
        ),
        "frames_inner/world, unique": paired_ratio(
            cells,
            "frames_inner",
            "world",
            "stack_pbe_v4",
            lambda b, s, _c: (b, s) in unique,
        ),
    }
    print("last op", out["last_op_v4"])
    flags = fc.op_flags("stack_pbe_v4")

    def one_level(b, s, c, drop=None):
        return c == "defined" and flags[b][s]["levels"] == 1 and b != drop

    out["control_v4"] = {
        "1 level": paired_ratio(
            cells, "frames_inner", "frames", "stack_pbe_v4", one_level
        ),
        f"1 level, without {fc.OUTLIER}": paired_ratio(
            cells,
            "frames_inner",
            "frames",
            "stack_pbe_v4",
            lambda b, s, c: one_level(b, s, c, fc.OUTLIER),
        ),
        "top-level ops": paired_ratio(
            cells,
            "frames_inner",
            "frames",
            "stack_pbe_v4",
            lambda _b, _s, c: c == "top",
        ),
    }
    print("control", out["control_v4"])
    w3 = fc.op_flags("stack_pbe_world_v3")

    def deep(b, s, c, drop=None):
        return c == "defined" and w3[b][s]["levels"] >= 2 and b != drop

    loo = {
        b: paired_ratio(
            cells,
            "frames_inner",
            "frames",
            "stack_pbe_world_v3",
            lambda x, s, c, b=b: deep(x, s, c, b),
        )["median"]
        for b in w3
    }
    worst = max(loo, key=loo.get)
    out["stacking_world_v3_leave_one_out"] = {
        "all": paired_ratio(
            cells, "frames_inner", "frames", "stack_pbe_world_v3", deep
        ),
        "highest": [worst, loo[worst]],
    }
    print("leave one out", out["stacking_world_v3_leave_one_out"])
    t = world["calibration"]["temperature"]
    floor = {}
    for corpus in CORPORA:
        counts: dict[str, int] = defaultdict(int)
        for r in world[corpus]["machine_mask"]["rows"]:
            if r["killed"]:
                counts["killed"] += 1
                continue
            counts["witnesses"] += 1
            for name in ("free", "learned"):
                charges = (
                    ranker["s0"]["charges"][corpus][r["behaviour"]]
                    if name == "learned"
                    else [0.0] * len(r["deficits"])
                )
                d = [
                    (x / t) * (0.0 if cls == "defined" else 1.0) + c
                    for x, cls, c in zip(
                        r["deficits"], r["classes"], charges, strict=True
                    )
                ]
                segs, per_body = levels(r["program"], d, r["kinds"])
                top_over = max(segs) > BUDGET
                body_over = max(per_body, default=0.0) > BUDGET
                counts[f"{name}: fails in a top-level segment"] += top_over
                counts[f"{name}: fails inside a body"] += body_over
                counts[f"{name}: clears"] += not (top_over or body_over)
        floor[corpus] = dict(counts)
    out["floor_world_s0_e5_defined_free"] = floor
    print("floor", floor)
    shares = {}
    for corpus in CORPORA:
        flags = fc.op_flags(corpus)
        shares[corpus] = {}
        for arm in ("world", "frames_inner", "exit_inner"):
            per_cell = []
            for cell in cells[arm].values():
                t = cell["calibration"]["temperature"]
                by = defaultdict(float)
                for row in cell[corpus]["machine_mask"]["rows"]:
                    for step, d in enumerate(row["deficits"]):
                        f = flags[row["behaviour"]][step]
                        if (
                            d is not None
                            and row["classes"][step] == "defined"
                            and f["position"] in ("first", "later")
                        ):
                            by[f["body"]] += d / t
                total = sum(by.values())
                per_cell.append({b: v / total for b, v in by.items()})
            shares[corpus][arm] = {
                b: statistics.median(c.get(b, 0.0) for c in per_cell)
                for b in ("1-2", "3-6", "7+")
            }
    out["interior_share_by_body_length"] = shares
    print("interior shares", shares)
    per_seed = {}
    for arm in ("world", "exit_inner"):
        per_seed[arm] = {
            corpus: [
                fc.clears(
                    cells[arm][seed, 5], corpus, ranker[f"s{seed}"]["charges"][corpus]
                )
                for seed in (0, 1, 2)
            ]
            for corpus in CORPORA
        }
    out["clears_e5_learned_by_seed"] = per_seed
    print("clears e5 by seed", per_seed)
    (HERE / "checks.json").write_text(json.dumps(out, indent=1))
    for corpus in CORPORA:
        c = out[corpus]
        print(
            corpus,
            {
                k: c[k]
                for k in (
                    "witnesses_with_opens",
                    "skeleton_seen",
                    "program_seen",
                    "opens_in_seen",
                    "opens_in_unseen",
                )
            },
        )
        for name in c["per_open"]:
            po, ss = c["per_open"][name], c["per_sub_search"][name]
            print(
                f"  {name}: open seen {fmt(po['seen'])} unseen {fmt(po['unseen'])}"
                f" | top {fmt(ss['top_level'])} >B {ss['top_level'].get('over_budget', 0):.2f}"
                f" | nested bodies {fmt(ss['bodies_with_nested_opens'])} >B {ss['bodies_with_nested_opens'].get('over_budget', 0):.2f}"
                f" | worst {fmt(ss['worst_sub_search_per_witness'])} >B {ss['worst_sub_search_per_witness'].get('over_budget', 0):.2f}"
            )


if __name__ == "__main__":
    main()
