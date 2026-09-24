"""Score the capped-body programs under `BEST` (short-body-authoring #167).

For each unsolved behaviour, the program with the smallest cap met is taken
from every `programs_*.jsonl` in the report directory. It is re-verified with
`check.py`, then given best-system-command (#158)'s 10,000-draw verdict, and
scored beside the authored witness under `BEST`:

- **raw** log-probabilities, the beam as it runs (the registered primary);
- **calibrated** deficits at `BEST`'s fitted temperature.

Clears come from `frame_gate.composite`:
- **flat**: one `ln 64` for the whole witness.
- **segmented**: reset at each closed state after a top-level combinator.
- **quotations**: a budget per body.
- **both**.

    uv run python experiments/short_body/score.py --out reports/short-body-20260923

``--checkpoint`` scores under another policy in place of `BEST`, a
``difference`` one (short-segment-campaign #168) included: blocks and the
calibration rows are built in the checkpoint's own encoding. ``--name`` keeps
its output apart from `BEST`'s ``scores.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.step3_verify_solves as verify  # noqa: E402
from experiments.frame_gate import (  # noqa: E402
    composite,
    eval_witnesses,
    fit_temperature,
    validation_legal,
    validation_rows,
    walk,
)
from experiments.short_body.check import assess, shape, unsolved  # noqa: E402
from experiments.staged_witnesses import check as staged  # noqa: E402
from wandering_light.solve import BEST  # noqa: E402
from wandering_light.stack_pbe import CATALOGUE  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402
from wandering_light.stack_policy_hub import fetch_stack_policy  # noqa: E402

RANK = {"cap4": 0, "cap8": 1, "best_attempt": 2}
DECOMPOSITIONS = ("flat", "segmented", "quotations", "both")
# Registered on #167: segmented search is the lever if cap-4 programs exist for
# at least 15 behaviours and segment-reset clears at least 5 more of them than a
# flat beam does, raw; long bodies are needed if cap-8 programs exist for at most 12.
SEGMENTED_MIN_CAP4, SEGMENTED_MIN_GAIN, LONG_BODIES_MAX_CAP8 = 15, 5, 12


def chosen(report_dir: Path, tasks) -> dict[str, dict[str, Any]]:
    """Per behaviour, the verified program with the smallest cap met, shortest first."""
    best: dict[str, dict[str, Any]] = {}
    for path in sorted(report_dir.glob("programs_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("program") or row.get("status") not in RANK:
                continue
            checked = assess(row["behaviour"], row["program"], tasks, 8)
            if not checked["verified"]:
                continue
            largest = checked["metrics"]["largest_quotation"]
            status = (
                "cap4" if largest <= 4 else "cap8" if largest <= 8 else "best_attempt"
            )
            key = (RANK[status], len(row["program"].split()))
            old = best.get(row["behaviour"])
            if old is None or key < old["key"]:
                best[row["behaviour"]] = {
                    "key": key,
                    "status": status,
                    "program": row["program"],
                    "author": path.stem.removeprefix("programs_"),
                    "largest_quotation": largest,
                    # Unrolled to the sampler's longest input: verified on
                    # every sample and task, but not a program for any length.
                    "bounded": "unroll" in row.get("notes", "").lower(),
                }
    return best


def scored(row: dict, t: float) -> dict[str, Any]:
    d = list(row["deficits"])
    body = sum(x for x, c in zip(d, row["classes"], strict=True) if c != "top")
    return {
        "raw_total": sum(d),
        "raw_per_op": sum(d) / len(d),
        "body_share": body / sum(d) if sum(d) else 0.0,
        "raw": {k: composite([row], temperature=1.0)[k] for k in DECOMPOSITIONS},
        "calibrated": {k: composite([row], temperature=t)[k] for k in DECOMPOSITIONS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="a local policy in place of BEST's"
    )
    parser.add_argument("--name", default="scores.json", help="output file in --out")
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    tasks = staged.v4_tasks()
    picks = chosen(args.out, tasks)
    by_name = {b.name: b for b in CATALOGUE}
    authored = {t.behaviour: t for t in eval_witnesses("stack_pbe_v4")}
    policy = args.checkpoint or fetch_stack_policy(BEST.repo_id, BEST.revision)
    scorer = PolicyScorer(load_policy(policy), "cpu")
    calibration = fit_temperature(
        scorer, validation_rows(None, scorer.encoding), validation_legal()
    )
    t = calibration["temperature"]
    rows = []
    for name in unsolved():
        base = authored[name]
        entry: dict[str, Any] = {
            "behaviour": name,
            "authored": {
                **shape(base),
                **scored(walk(base, scorer, encoding=None, mask=False), t),
            },
        }
        pick = picks.get(name)
        if pick is not None:
            program = tuple(pick["program"].split())
            mine = dataclasses.replace(base, program=program)
            entry.update(
                status=pick["status"],
                author=pick["author"],
                largest_quotation=pick["largest_quotation"],
                bounded=pick["bounded"],
                program=pick["program"],
                draw_verdict=verify.check(program, by_name[name], verify.DRAWS, 134)[
                    "verdict"
                ],
                capped={
                    **shape(mine),
                    **scored(walk(mine, scorer, encoding=None, mask=False), t),
                },
            )
        else:
            entry["status"] = "none"
        rows.append(entry)
    have = [r for r in rows if "capped" in r]
    cap4 = [r for r in have if r["status"] == "cap4"]
    cap8 = [r for r in have if r["status"] in ("cap4", "cap8")]
    general = [r for r in have if not r["bounded"]]
    summary = {
        "behaviours": len(rows),
        "cap4": len(cap4),
        "cap4_not_bounded": sum(r["status"] == "cap4" for r in general),
        "bounded": sorted(r["behaviour"] for r in have if r["bounded"]),
        "cap8_or_better": len(cap8),
        "genuine": sum(
            r["draw_verdict"] in ("genuine", "equals_catalogue_witness") for r in have
        ),
        "median_ops_cap4": statistics.median(r["capped"]["ops"] for r in cap4)
        if cap4
        else None,
        "median_longest_segment_cap4": statistics.median(
            r["capped"]["longest_segment"] for r in cap4
        )
        if cap4
        else None,
        "longer_than_authored": sum(
            r["capped"]["ops"] > r["authored"]["ops"] for r in have
        ),
        "clears": {
            side: {
                scale: {k: sum(r[side][scale][k] for r in have) for k in DECOMPOSITIONS}
                for scale in ("raw", "calibrated")
            }
            for side in ("authored", "capped")
        },
        "clears_not_bounded": {
            side: {k: sum(r[side]["raw"][k] for r in general) for k in DECOMPOSITIONS}
            for side in ("authored", "capped")
        },
    }
    gain = (
        summary["clears"]["capped"]["raw"]["segmented"]
        - summary["clears"]["capped"]["raw"]["flat"]
    )
    if len(cap4) >= SEGMENTED_MIN_CAP4 and gain >= SEGMENTED_MIN_GAIN:
        reading = "SEGMENTED_SEARCH"
    elif len(cap8) <= LONG_BODIES_MAX_CAP8:
        reading = "NEEDS_LONG_BODIES"
    else:
        reading = "BETWEEN"
    out = {
        "calibration": calibration,
        "summary": summary,
        "segmented_gain_raw": gain,
        "reading": reading,
        "rows": rows,
    }
    if args.checkpoint is not None:  # `BEST`'s output stays as it was
        out = {
            "checkpoint": str(args.checkpoint),
            "encoding": scorer.encoding,
            **out,
        }
    (args.out / args.name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1), "\nRESULT", reading, "gain", gain, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
