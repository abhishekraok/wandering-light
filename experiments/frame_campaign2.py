"""inner-frames (#160), second campaign: innermost-only frames, an exit oracle, a ranker.

The first gate came back WEAK (`reports/frame-gate-20260922`). A frame cut
calibrated defined-body deficit to 0.62 of the world arm's at one open level on
v4, but to 0.93 deeper, where the encoding stacks a label and frame per open
level, and the label alone went from helping to hurting. This campaign asks
three things with one set of GPU runs, each arm on the world arm's rows and
recipe (`frame_gate.train`), measured paired against its 18 cells:

- **innermost-only** (``tags_inner``, ``frames_inner``): was the depth cost the
  stacking? Only the innermost defined level is annotated.
- **exit-frame oracle** (``exit_inner``): ``frames_inner`` plus the frame the
  body must produce. Is a body short of a specification, or is the policy the
  bottleneck? It licenses nothing directly.
- **commitment ranker** (`experiments/commit_ranker.py`): does a learned
  charge at each `[` beat the shape-only prior that bound the first gate?

``report`` reads every measured cell and writes ``report.json`` with the
registered readings (#160). World cells come from the first gate's run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments.commit_ranker as cr  # noqa: E402
import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.world_arm as wa  # noqa: E402
from experiments.frame_gate import (  # noqa: E402
    CORPORA,
    PRIMARY,
    _token_starts,
    commit_charges,
    composite,
    eval_witnesses,
)
from experiments.frame_rows import example_walks, frame_state  # noqa: E402
from wandering_light.stack_policy import STATE_CAP  # noqa: E402

RUN_DIR = REPO / "checkpoints" / "frame-campaign-2"
GATE_DIR = REPO / "checkpoints" / "frame-arm-20260922"
ARMS = ("tags_inner", "frames_inner", "exit_inner")
STACKED = ("tags", "frames")
CELLS = [(s, e) for s in wa.SEEDS for e in range(1, wa.EPOCHS + 1)]
# One v4 witness is a fifth of the world arm's v4 defined-body deficit; the
# first gate's v4 label cost was that witness (`reports/frame-gate-20260922`).
OUTLIER = "bounds_cover_list_flag"

# Registered readings (#160, before training). Oracle: exit/world on v4 defined
# bodies under SKELETON with clears at PRIMARY, learned charge, at or above
# SKELETON_CLEARS points to skeleton-first search; above SPEC_NOT_LEVER says a
# specification is not what bodies lack. Ranker: median learned surprisal per
# `[` on v4 eval at or under RANKER_TARGET nats.
SKELETON, SKELETON_CLEARS, SPEC_NOT_LEVER = 0.5, 18, 0.8
RANKER_TARGET = 0.4
# Innermost-only against stacked frames, paired cell by cell, v4 defined: the
# stacking was a cost if the 2+ level ratio is at or under STACKING while one
# level stays within ONE_LEVEL; depth is intrinsic at or above INTRINSIC.
STACKING, ONE_LEVEL, INTRINSIC = 0.90, (0.92, 1.08), 0.97


def _cell(run_dir: Path, arm: str, seed: int, epoch: int) -> dict | None:
    path = run_dir / "measure" / f"{arm}-s{seed}-e{epoch}.json"
    return json.loads(path.read_text())["cells"] if path.exists() else None


def load_cells(run_dir: Path, gate_dir: Path) -> dict[str, dict[tuple, dict]]:
    out: dict[str, dict[tuple, dict]] = {}
    for arm, source in (
        ("world", gate_dir),
        *((a, gate_dir) for a in STACKED),
        *((a, run_dir) for a in ARMS),
    ):
        out[arm] = {}
        for seed, epoch in CELLS:
            cell = _cell(source, arm, seed, epoch)
            if cell is not None:
                out[arm][seed, epoch] = cell
    return out


def op_flags(corpus: str) -> dict[str, list[dict[str, Any]]]:
    """Per witness step: open levels, body position, and tags-encoded clipping."""
    out = {}
    for task in eval_witnesses(corpus):
        walks = example_walks(task)
        opened: list[int] = []
        flags = []
        for step, name in enumerate(task.program):
            per = {
                "levels": len(walks[0][1].get(step, ())),
                "position": (
                    "close"
                    if name == "]" and opened
                    else "first"
                    if opened and step == opened[-1] + 1
                    else "later"
                    if opened
                    else "top"
                ),
                "tags_clip": any(
                    _token_starts(frame_state(s[step], v.get(step, ()), "tags"))[1]
                    > STATE_CAP
                    for s, v in walks
                ),
            }
            flags.append(per)
            if name == "[":
                opened.append(step)
            elif name == "]" and opened:
                opened.pop()
        out[task.behaviour] = flags
    return out


SUBSETS = {
    "defined": lambda op: op["cls"] == "defined",
    "defined, 1 level": lambda op: op["cls"] == "defined" and op["levels"] == 1,
    "defined, 2+ levels": lambda op: op["cls"] == "defined" and op["levels"] >= 2,
    f"defined, without {OUTLIER}": lambda op: (
        op["cls"] == "defined" and op["name"] != OUTLIER
    ),
    f"defined, 2+ levels, without {OUTLIER}": lambda op: (
        op["cls"] == "defined" and op["levels"] >= 2 and op["name"] != OUTLIER
    ),
    "defined, 2+ levels, tags unclipped": lambda op: (
        op["cls"] == "defined" and op["levels"] >= 2 and not op["tags_clip"]
    ),
    "defined, first op": lambda op: (
        op["cls"] == "defined" and op["position"] == "first"
    ),
    "defined, later ops": lambda op: (
        op["cls"] == "defined" and op["position"] == "later"
    ),
    "defined, close": lambda op: op["cls"] == "defined" and op["position"] == "close",
    "top": lambda op: op["cls"] == "top",
    "all body": lambda op: op["cls"] != "top",
}


def subset_totals(cell: dict, corpus: str, flags) -> dict[str, float]:
    t = cell["calibration"]["temperature"]
    totals = dict.fromkeys(SUBSETS, 0.0)
    for row in cell[corpus]["machine_mask"]["rows"]:
        for step, (d, cls) in enumerate(
            zip(row["deficits"], row["classes"], strict=True)
        ):
            if d is None:
                continue
            op = {**flags[row["behaviour"]][step], "cls": cls, "name": row["behaviour"]}
            for name, keep in SUBSETS.items():
                if keep(op):
                    totals[name] += d / t
    return totals


def _summary(values: list[float]) -> list[float] | None:
    if not values:
        return None
    # Full precision: `readings` compares these with its thresholds.
    return [statistics.median(values), min(values), max(values)]


def ranker_charges(run_dir: Path) -> dict[int, dict[str, dict[str, list[float]]]]:
    out = {}
    for seed in wa.SEEDS:
        path = run_dir / "measure" / f"{cr.ARM}-s{seed}.json"
        if path.exists():
            out[seed] = json.loads(path.read_text())["charges"]
    return out


def clears(cell: dict, corpus: str, commit) -> int:
    rows = cell[corpus]["machine_mask"]["rows"]
    t = cell["calibration"]["temperature"]
    return composite(rows, temperature=t, commit=commit)["both"]


def report(run_dir: Path, gate_dir: Path, log) -> dict[str, Any]:
    cells = load_cells(run_dir, gate_dir)
    learned = ranker_charges(run_dir)
    missing = {arm: sorted(set(CELLS) - set(cells[arm])) for arm in ("world", *ARMS)}
    out: dict[str, Any] = {"missing": missing, "ranker_seeds": sorted(learned)}
    boundary = sorted(
        f"{arm}-s{k[0]}-e{k[1]}"
        for arm in ARMS
        for k, cell in cells[arm].items()
        if cell["calibration"].get("at_boundary")
    )
    out["calibration_at_boundary"] = boundary
    for corpus in CORPORA:
        flags = op_flags(corpus)
        prior = commit_charges(corpus, eval_witnesses(corpus))
        totals = {
            arm: {k: subset_totals(c, corpus, flags) for k, c in cells[arm].items()}
            for arm in cells
        }
        corpus_out: dict[str, Any] = {"ratios": {}, "clears": {}, "top1_gain": {}}
        for num, den in (
            *((a, "world") for a in (*ARMS, *STACKED)),
            ("frames_inner", "frames"),
            ("tags_inner", "tags"),
            ("exit_inner", "frames_inner"),
        ):
            pairs = sorted(set(totals[num]) & set(totals[den]))
            corpus_out["ratios"][f"{num}/{den}"] = {
                name: _summary(
                    [
                        totals[num][k][name] / totals[den][k][name]
                        for k in pairs
                        if totals[den][k][name] > 0
                    ]
                )
                for name in SUBSETS
            }
        for arm in ("world", *ARMS, *STACKED):
            per_cell = defaultdict(list)
            for (seed, _epoch), cell in sorted(cells[arm].items()):
                per_cell["prior"].append(clears(cell, corpus, prior))
                per_cell["free"].append(clears(cell, corpus, None))
                if seed in learned:
                    per_cell["learned"].append(
                        clears(cell, corpus, learned[seed][corpus])
                    )
            primary = cells[arm].get(PRIMARY)
            corpus_out["clears"][arm] = {
                key: {
                    "median": statistics.median(v),
                    "range": [min(v), max(v)],
                }
                for key, v in per_cell.items()
                if v
            }
            if primary is not None:
                corpus_out["clears"][arm]["primary"] = {
                    "prior": clears(primary, corpus, prior),
                    "free": clears(primary, corpus, None),
                    **(
                        {
                            "learned": clears(
                                primary, corpus, learned[PRIMARY[0]][corpus]
                            )
                        }
                        if PRIMARY[0] in learned
                        else {}
                    ),
                }
            if arm != "world":
                gains = [
                    cells[arm][k][corpus]["machine_mask"]["body"]["defined"]["top1"]
                    - cells["world"][k][corpus]["machine_mask"]["body"]["defined"][
                        "top1"
                    ]
                    for k in sorted(set(cells[arm]) & set(cells["world"]))
                ]
                corpus_out["top1_gain"][arm] = _summary(gains)
        corpus_out["commitment"] = {
            "prior": cr.summarise(cr.at_opens(prior, corpus)),
            "learned": {
                seed: cr.summarise(cr.at_opens(v[corpus], corpus))
                for seed, v in learned.items()
            },
        }
        out[corpus] = corpus_out
    out["readings"] = readings(out)
    co.write_json(run_dir / "report.json", out)
    log(f"RESULT campaign 2 readings: {out['readings']}")
    return out


def readings(out: dict[str, Any]) -> dict[str, Any]:
    """The registered readings, or why there is none yet."""
    if any(out["missing"].values()) or len(out["ranker_seeds"]) < len(wa.SEEDS):
        return {"status": "INCOMPLETE"}
    if out["calibration_at_boundary"]:
        return {"status": "CALIBRATION_AT_BOUNDARY"}
    v4 = out["stack_pbe_v4"]
    oracle = v4["ratios"]["exit_inner/world"]["defined"][0]
    oracle_clears = v4["clears"]["exit_inner"]["primary"]["learned"]
    if oracle < SKELETON and oracle_clears >= SKELETON_CLEARS:
        spec = "SKELETON_FIRST"
    elif oracle > SPEC_NOT_LEVER:
        spec = "SPEC_NOT_THE_LEVER"
    else:
        spec = "BETWEEN"
    surprisal = statistics.median(
        s["median"] for s in v4["commitment"]["learned"].values()
    )
    inner = v4["ratios"]["frames_inner/frames"]
    deep, shallow = inner["defined, 2+ levels"][0], inner["defined, 1 level"][0]
    if deep <= STACKING and ONE_LEVEL[0] <= shallow <= ONE_LEVEL[1]:
        stacking = "STACKING_WAS_A_COST"
    elif deep >= INTRINSIC:
        stacking = "DEPTH_IS_INTRINSIC"
    else:
        stacking = "UNCLEAR"
    return {
        "status": "READ",
        "oracle": spec,
        "oracle_ratio": oracle,
        "oracle_clears_learned": oracle_clears,
        "ranker_median_surprisal": surprisal,
        "ranker_target_met": surprisal <= RANKER_TARGET,
        "stacking": stacking,
        "inner_over_stacked": {"2+ levels": deep, "1 level": shallow},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("report",))
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--gate-dir", type=Path, default=GATE_DIR)
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with open(args.run_dir / "log.txt", "a") as handle:
            handle.write(stamped + "\n")

    report(args.run_dir, args.gate_dir, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
