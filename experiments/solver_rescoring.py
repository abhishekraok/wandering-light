"""Does the body-cost diagnosis hold on programs the solver finds? (solver-program-rescoring #165)

Every campaign so far scores frontier-authored witnesses, so its finding that
quotation bodies carry the deficit could be a fact about how Codex and Opus
write programs rather than about the tasks. This scores every verified program
the solver itself has found that is *not* the catalogue witness (a `genuine`
verdict in best-system-command (#158)'s 10,000-draw test) beside its
behaviour's authored v4 eval witness:
- on that witness's task inputs;
- under the same checkpoint;
- with the attribution the inner-frames (#160) campaigns use: `frame_gate.walk`
  under the machine mask, calibrated deficits, `composite` clears.

Programs are read from the verdict cache of the tempered-beam run, which holds
#158's verdicts plus that run's. Two checkpoints:
- the world arm `1m-s0-e5`, the campaigns' reference;
- `BEST`, which found most of the programs.

    uv run python experiments/solver_rescoring.py --out reports/solver-rescoring-20260923
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.commit_ranker as cr  # noqa: E402
from experiments.frame_gate import (  # noqa: E402
    CommitmentModel,
    composite,
    eval_witnesses,
    fit_temperature,
    validation_legal,
    validation_rows,
    walk,
)
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from experiments.segment_composite import levels  # noqa: E402
from wandering_light.solve import BEST  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import BASIS  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402
from wandering_light.stack_policy_hub import fetch_stack_policy  # noqa: E402

CORPUS = "stack_pbe_v4"
VERDICTS = REPO / "checkpoints" / "tempered-beam-20260923" / "verified.json"
WORLD = REPO / "checkpoints" / "world-arm-20260922" / "policies" / "1m-s0-e5"
RANKER = REPO / "checkpoints" / "frame-campaign-2" / "ranker" / "policies" / "1m-s0-e3"
RANKER_T = 1.129489004610947  # ranker seed 0's fitted temperature (campaign 2)
BUDGET = math.log(64)
# The registered reading: the diagnosis survives on the solved set if the median
# body share of calibrated deficit on solver programs is within this of the
# authored witnesses' median, under both checkpoints.
SURVIVES_WITHIN = 0.10


def solver_programs() -> list[tuple[str, tuple[str, ...]]]:
    verdicts = json.loads(VERDICTS.read_text())
    return sorted(
        (key.split(" :: ")[0], tuple(key.split(" :: ")[1].split()))
        for key, v in verdicts.items()
        if v["verdict"] == "genuine"
    )


def fits(task) -> bool:
    """Does the program produce the task's outputs on its inputs?"""
    try:
        final = replay_states(task)[-1]
    except ValueError:
        return False
    return final == tuple(parse_stack(text, basis=BASIS) for text in task.outputs)


def learned_charges(model, task) -> list[float] | None:
    """Ranker seed 0's calibrated charge at each `[` of ``task``'s program."""
    choices = cr.choices_of(model)
    try:
        rows = cr.ranker_rows([task], choices)
    except ValueError:  # a commitment outside the ranker's vocabulary
        return None
    out = [0.0] * len(task.program)
    if rows:
        logits = cr._all_logits(model, rows, "cpu")
        labels = np.array([row.label for row in rows])
        for row, cost in zip(rows, cr._nll(logits, labels, RANKER_T), strict=True):
            out[row.step] = float(cost)
    return out


def body_lengths(program) -> list[int]:
    """Ops strictly inside each quotation, nested quotations included."""
    out, stack = [], []
    for step, name in enumerate(program):
        if name == "[":
            stack.append(step)
        elif name == "]" and stack:
            out.append(step - stack.pop() - 1)
    return out


def metrics(row: dict, t: float, prior: list[float], learned) -> dict[str, Any]:
    d = [x / t for x in row["deficits"]]
    body = sum(x for x, c in zip(d, row["classes"], strict=True) if c != "top")
    defined = sum(x for x, c in zip(d, row["classes"], strict=True) if c == "defined")
    _top, bodies = levels(row["program"], d, row["kinds"])
    total = sum(d)
    clears = {
        "free": composite([row], temperature=t),
        "prior": composite([row], temperature=t, commit={row["behaviour"]: prior}),
    }
    if learned is not None:
        clears["learned"] = composite(
            [row], temperature=t, commit={row["behaviour"]: learned}
        )
    return {
        "ops": len(d),
        "body_ops": sum(c != "top" for c in row["classes"]),
        "total": total,
        "per_op": total / len(d),
        "body_share": body / total if total else 0.0,
        "defined_share": defined / total if total else 0.0,
        "worst_body": max(bodies, default=0.0),
        "quotations": len(body_lengths(row["program"])),
        "longest_body_ops": max(body_lengths(row["program"]), default=0),
        "clears": {k: {m: v[m] for m in ("flat", "both")} for k, v in clears.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    authored = {t.behaviour: t for t in eval_witnesses(CORPUS)}
    prior_model = CommitmentModel.fit(CORPUS)
    ranker = load_policy(RANKER).eval()
    pairs = []
    for behaviour, program in solver_programs():
        base = authored[behaviour]
        found = dataclasses.replace(base, program=program)
        pairs.append((behaviour, base, found, fits(found)))
    out: dict[str, Any] = {
        "programs": len(pairs),
        "behaviours": len({p[0] for p in pairs}),
        "unfit_on_the_witness_task": [
            f"{b} :: {' '.join(f.program)}" for b, _a, f, ok in pairs if not ok
        ],
        "checkpoints": {},
    }
    for name, path in (
        ("world", WORLD),
        ("best", fetch_stack_policy(BEST.repo_id, BEST.revision)),
    ):
        scorer = PolicyScorer(load_policy(path), "cpu")
        calibration = fit_temperature(scorer, validation_rows(None), validation_legal())
        t = calibration["temperature"]
        per_pair = []
        for behaviour, base, found, ok in pairs:
            if not ok:
                continue
            entry = {"behaviour": behaviour, "program": " ".join(found.program)}
            for side, task in (("authored", base), ("solver", found)):
                row = walk(task, scorer, encoding=None, mask=False)
                entry[side] = metrics(
                    row, t, prior_model.per_step(task), learned_charges(ranker, task)
                )
            per_pair.append(entry)
        # One value per behaviour, so a behaviour the solver found four ways
        # counts once: its authored witness, and the median of its programs.
        by_behaviour: dict[str, list[dict]] = {}
        for p in per_pair:
            by_behaviour.setdefault(p["behaviour"], []).append(p)
        summary = {}
        for side in ("authored", "solver"):
            summary[side] = {
                key: statistics.median(
                    statistics.median(p[side][key] for p in group)
                    for group in by_behaviour.values()
                )
                for key in (
                    "ops",
                    "per_op",
                    "total",
                    "body_share",
                    "defined_share",
                    "worst_body",
                    "quotations",
                    "longest_body_ops",
                )
            }
            # Clears count programs on the solver side, witnesses on the other.
            counted = (
                per_pair if side == "solver" else [g[0] for g in by_behaviour.values()]
            )
            summary[side]["counted"] = len(counted)
            summary[side]["worst_body_over_budget"] = sum(
                p[side]["worst_body"] > BUDGET for p in counted
            )
            summary[side]["clears"] = {
                charge: {
                    m: sum(
                        p[side]["clears"][charge][m]
                        for p in counted
                        if charge in p[side]["clears"]
                    )
                    for m in ("flat", "both")
                }
                for charge in ("free", "prior", "learned")
            }
        gap = summary["authored"]["body_share"] - summary["solver"]["body_share"]
        out["checkpoints"][name] = {
            "policy": str(path),
            "calibration": calibration,
            "pairs": per_pair,
            "summary": summary,
            "reading": "SURVIVES" if gap <= SURVIVES_WITHIN else "WEAKENED",
        }
        print(
            name, json.dumps(summary), out["checkpoints"][name]["reading"], flush=True
        )
    readings = {c["reading"] for c in out["checkpoints"].values()}
    out["reading"] = "SURVIVES" if readings == {"SURVIVES"} else "WEAKENED"
    out["seconds"] = round(time.time() - started, 1)
    (args.out / "rescoring.json").write_text(json.dumps(out, indent=1) + "\n")
    print("RESULT", out["reading"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
