"""Does short-body style transfer to behaviours a policy never saw? (short-body-authoring #167)

Scores the genuine capped programs, and their authored witnesses, under
staged-witness-reauthoring (#155)'s `fused`, `refused` and `staged` arms at
epoch 3 (seeds 0-2) and under the world arm `1m-s0-e5`. Everything is raw: the
beam as it runs.
- raw nats per op, and segment-reset clears;
- per op by class: structural (`[`, `]` and the consumers) against content;
- per op by position within its closed-state segment.

The registered reading is the median over seeds of staged/fused raw nats per
op on the capped programs: TRANSFERS at 0.85 or less, NO_TRANSFER at 0.95 or
more, PARTIAL between.

    uv run python experiments/short_body/transfer.py --out reports/short-body-20260923
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from experiments.frame_gate import composite, eval_witnesses, walk  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402

ARMS = ("fused", "refused", "staged")
SEEDS = (0, 1, 2)
STAGED = REPO / "checkpoints" / "staged-arms-20260920" / "policies"
WORLD = REPO / "checkpoints" / "world-arm-20260922" / "policies" / "1m-s0-e5"
STRUCTURAL = {"[", "]", "map", "filter", "fold", "ifte", "dip", "i"}
TRANSFERS, NO_TRANSFER = 0.85, 0.95


def positions(kinds) -> list[str]:
    """Each step's place in its segment, cut after each top-level combinator."""
    out, start = [], 0
    for step, kind in enumerate(kinds):
        if kind == "run" or step == len(kinds) - 1:
            length = step - start + 1
            out += [
                "only"
                if length == 1
                else "first"
                if i == 0
                else "last"
                if i == length - 1
                else "middle"
                for i in range(length)
            ]
            start = step + 1
    return out


def profile(rows: list[dict]) -> dict[str, Any]:
    """Pooled raw nats per op overall, by class and by segment position; segment clears."""
    sums: dict[str, list[float]] = defaultdict(list)
    per_program = []
    for row in rows:
        d = row["deficits"]
        per_program.append(sum(d) / len(d))
        for name, x, where in zip(
            row["program"], d, positions(row["kinds"]), strict=True
        ):
            sums["all"].append(x)
            sums["structural" if name in STRUCTURAL else "content"].append(x)
            sums[f"position {where}"].append(x)
    return {
        "median_per_op": statistics.median(per_program),
        "pooled": {
            k: {"ops": len(v), "per_op": statistics.fmean(v)}
            for k, v in sorted(sums.items())
        },
        "segmented_clears": sum(
            composite([r], temperature=1.0)["segmented"] for r in rows
        ),
        "flat_clears": sum(composite([r], temperature=1.0)["flat"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    scores = json.loads((args.out / "scores.json").read_text())
    genuine = [
        r
        for r in scores["rows"]
        if r.get("draw_verdict") in ("genuine", "equals_catalogue_witness")
    ]
    authored = {t.behaviour: t for t in eval_witnesses("stack_pbe_v4")}
    tasks = {
        "capped": [
            dataclasses.replace(
                authored[r["behaviour"]], program=tuple(r["program"].split())
            )
            for r in genuine
        ],
        "authored": [authored[r["behaviour"]] for r in genuine],
    }
    checkpoints = {
        f"{arm}-s{s}": STAGED / f"{arm}-s{s}-e3" for arm in ARMS for s in SEEDS
    }
    checkpoints["world"] = WORLD
    out: dict[str, Any] = {"programs": len(genuine), "checkpoints": {}}
    for name, path in checkpoints.items():
        scorer = PolicyScorer(load_policy(path), "cpu")
        out["checkpoints"][name] = {
            side: profile([walk(t, scorer, encoding=None, mask=False) for t in group])
            for side, group in tasks.items()
        }
        c = out["checkpoints"][name]
        print(name, {s: round(c[s]["median_per_op"], 3) for s in c}, flush=True)
    ratios = [
        out["checkpoints"][f"staged-s{s}"]["capped"]["median_per_op"]
        / out["checkpoints"][f"fused-s{s}"]["capped"]["median_per_op"]
        for s in SEEDS
    ]
    ratio = statistics.median(ratios)
    out["staged_over_fused"] = {"per_seed": ratios, "median": ratio}
    out["reading"] = (
        "TRANSFERS"
        if ratio <= TRANSFERS
        else "NO_TRANSFER"
        if ratio >= NO_TRANSFER
        else "PARTIAL"
    )
    (args.out / "transfer.json").write_text(json.dumps(out, indent=1) + "\n")
    print("RESULT", out["reading"], ratios, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
