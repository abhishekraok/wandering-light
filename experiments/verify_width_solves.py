"""Verify the width-sweep solves against their behaviours, not just held-out examples (#99, #151).

Every number the solved-program-cache campaign reports is a *graded* solve: the
program fits the task's held-out examples, which come from the same sampler as
the shown ones. `world-plus-v4-policy (#134)` found 7 of its 36 distinct solved
programs were shortcuts that pass grading and fail the behaviour. So a headline
of the form "beam 512 buys +4 behaviours" is not a coverage claim until the new
programs are checked.

This reuses `step3_verify_solves.check` unchanged -- same three draw modes, same
10,000 draws, same 500-tested floor, same verdicts -- over the distinct
(behaviour, program) pairs of the cells, and rescores each cell as graded /
verified / witness-equal / shortcut / inconclusive behaviours.

`verified` is the union of the draw-tested `genuine` verdicts and the
`equals_catalogue_witness` ones, which is the convention the reports use
(`step3_verify_solves.partition`). **A program equal to a catalogue witness is
not draw-tested at all**: it returns early and inherits the corpus's own
verification, a different and weaker standard that #142 counts apart. So
"verified" never means "every program survived 10,000 draws", and the
`draw_tested_behaviours` and `witness_equal_behaviours` columns carry the split
so a reader can see which standard each count rests on.

    uv run python experiments/verify_width_solves.py --out reports/.../verified_width.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.step3_verify_solves import DRAWS, MIN_TESTED, check  # noqa: E402
from wandering_light.stack_pbe import CATALOGUE  # noqa: E402

CELLS = (
    "off-s0-e4",
    "on-s0-e4",
    "off-s0-e4-b128",
    "on-s0-e4-b128",
    "off-s0-e4-b256",
    "on-s0-e4-b256",
    "off-s0-e4-b512",
    "on-s0-e4-b512",
)
# The 18 (seed, epoch) cells of the #101 1m v4-only arm, which is what
# epoch-selection-in-behaviours (#154) selects over. Its tables count graded
# behaviours; these give the verified ones.
EPOCH_GRID = tuple(
    f"off-witness-s{seed}-e{epoch}" for seed in (0, 1, 2) for epoch in range(1, 7)
)
# The self-shortening arm and its matched controls (#151). Reported as
# name-filtered graded counts at first, which is the practice `docs/STATUS.md`
# now records as superseded, so they are draw-tested here too.
SHORTENED = tuple(
    f"off-{arm}-s{seed}-e{epoch}"
    for arm in ("shortened", "witness")
    for seed in (0, 1, 2)
    for epoch in (3, 4, 5)
)
# docs/STATUS.md records both samplers as leaking; aligned_matches is
# aligned-matches-sampler (#141), open, same defect not yet recorded in STATUS.
LEAKY = ("all_positive", "bounds_cover_list_flag")
SUSPECT = (*LEAKY, "aligned_matches")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("checkpoints/cache-20260919")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cells",
        choices=("width", "epoch-grid", "shortened", "all"),
        default="width",
        help="which cell family to rescore",
    )
    args = parser.parse_args(argv)

    by_name = {b.name: b for b in CATALOGUE}
    wanted = {
        "width": CELLS,
        "epoch-grid": EPOCH_GRID,
        "shortened": SHORTENED,
        "all": CELLS + EPOCH_GRID + SHORTENED,
    }[args.cells]
    cells: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    missing: list[str] = []
    for name in wanted:
        path = args.run_dir / f"{name}.json"
        if not path.exists():
            # Never drop a cell in silence: a table missing a row reads as a
            # cell that was run and scored, not one that failed to run.
            missing.append(name)
            print(f"  MISSING {name}", flush=True)
            continue
        rows = json.loads(path.read_text())["rows"]
        cells[name] = [
            (r["behaviour"], tuple(r["program"]))
            for r in rows
            if r["solved"] and r["program"]
        ]

    distinct = sorted({p for pairs in cells.values() for p in pairs})
    print(
        f"{len(distinct)} distinct (behaviour, program) pairs over {len(cells)} cells",
        flush=True,
    )

    verdicts: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for behaviour, program in distinct:
        record = check(
            program, by_name[behaviour], args.draws, args.seed, min_tested=MIN_TESTED
        )
        verdicts[behaviour, program] = record
        print(f"  {record['verdict']:26s} {behaviour} ({len(program)} ops)", flush=True)

    def score(pairs: list[tuple[str, tuple[str, ...]]]) -> dict[str, Any]:
        good: set[str] = set()
        kinds: dict[str, set[str]] = {
            k: set()
            for k in ("genuine", "equals_catalogue_witness", "shortcut", "inconclusive")
        }
        for pair in pairs:
            v = verdicts[pair]["verdict"]
            kinds[v].add(pair[0])
            if v in ("genuine", "equals_catalogue_witness"):
                good.add(pair[0])
        return {
            "graded_behaviours": len({b for b, _ in pairs}),
            "verified_behaviours": len(good),
            # NOT a quality column. A behaviour here survived 10,000 draws in
            # three modes; this drops it anyway because its *sampler* leaks,
            # which `docs/STATUS.md` records as superseded practice
            # (`bounds_cover_list_flag` verifies genuinely at beam 512). Kept
            # only so the older name-filtered numbers can be reproduced.
            "verified_minus_suspect_samplers_superseded": len(good - set(SUSPECT)),
            "draw_tested_behaviours": len(kinds["genuine"] | kinds["shortcut"]),
            "witness_equal_behaviours": len(kinds["equals_catalogue_witness"]),
            "graded_tasks_note": "tasks are in the cell file; this table is behaviour-level",
            **{f"{k}_behaviours": sorted(v) for k, v in kinds.items()},
            "verified": sorted(good),
        }

    table = {name: score(pairs) for name, pairs in cells.items()}
    out = {
        "draws": args.draws,
        "seed": args.seed,
        "min_tested": MIN_TESTED,
        "suspect_samplers": list(SUSPECT),
        "missing_cells": missing,
        "cells": table,
        "programs": [
            {"behaviour": b, "program": list(p), "length": len(p), **verdicts[b, p]}
            for b, p in distinct
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{'cell':26s} {'graded':>7s} {'verified':>9s} {'draw-tested':>12s} {'witness-equal':>14s}"
    )
    for name in wanted:
        if name in table:
            t = table[name]
            print(
                f"{name:26s} {t['graded_behaviours']:7d} {t['verified_behaviours']:9d} "
                f"{t['draw_tested_behaviours']:12d} {t['witness_equal_behaviours']:14d}"
            )
    if missing:
        print(f"\nMISSING {len(missing)} cell(s): {', '.join(missing)}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
