"""Part 2 of the distinct-programs (#174) falsifier: composition or retrieval.

For each arm and seed, a verified behaviour counts as **novel** when at least
one of its verified solving programs occurs nowhere, as a contiguous run of
ops, inside that arm's own training text; otherwise it was only ever solved by
a **contained** program. This is the split staged-arms (#155) reported (staged:
4 / 6 / 3 novel). An arm's training text is the programs its fit tasks are
replayed along: the staged witness where a behaviour was treated, the fused
witness otherwise, plus the new behaviours' witnesses in the ``staged_new``
arms.

    uv run python experiments/distinct_programs/containment.py \
        --run-dir checkpoints/distinct-programs-20260926 \
        --verdicts reports/distinct-programs-20260926/verified_solves.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import experiments.compute_optimal_mark28 as co
import experiments.staged_arms as sa
from wandering_light.stack_policy import validation_skeleton


def train_text(arm: str, tasks) -> set[str]:
    side = sa.treated(keep_contrived=False)
    fit = [t for t in tasks if t.split == "train" and not validation_skeleton(t)]
    fused = {" ".join(t.program) for t in fit}
    key = "refused_program" if arm == "refused" else "program"
    rewritten = {
        side[t.behaviour][key] if t.behaviour in side else " ".join(t.program)
        for t in fit
    }
    programs = {"fused": fused, "both": fused | rewritten}.get(arm, rewritten)
    if arm in ("staged_new", "staged_new_fixed"):
        programs |= {" ".join(t.program) for t in sa.new_tasks()}
    return {f" {p} " for p in programs}


def contained(program: str, text: set[str]) -> bool:
    needle = f" {program} "
    return any(needle in p for p in text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    args = parser.parse_args(argv)
    verdicts = {
        (row["behaviour"], row["program"]): row["verdict"]
        for row in json.loads(args.verdicts.read_text())["programs"]
    }
    tasks = co.corpus()
    texts: dict[str, set[str]] = {}
    per_arm: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((args.run_dir / "timing").glob("*.json")):
        cell = json.loads(path.read_text())
        arm = cell["arm"]
        if arm not in texts:
            texts[arm] = train_text(arm, tasks)
        solving: dict[str, set[str]] = defaultdict(set)
        for r in cell["rows"]:
            text = " ".join(r["program"] or ())
            if r["solved"] and verdicts.get((r["behaviour"], text)) in sa.VERIFIED:
                solving[r["behaviour"]].add(text)
        novel = {
            b
            for b, ps in solving.items()
            if any(not contained(p, texts[arm]) for p in ps)
        }
        per_arm[arm].append(
            {
                "seed": cell["seed"],
                "verified": len(solving),
                "novel": len(novel),
                "novel_behaviours": sorted(novel),
            }
        )
    print(
        "| arm | verified behaviours | novel (no contained program) | only contained |"
    )
    print("|---|---|---|---|")
    out = {}
    for arm, cells in per_arm.items():
        cells.sort(key=lambda c: c["seed"])
        v = [c["verified"] for c in cells]
        n = [c["novel"] for c in cells]
        print(
            f"| {arm} | {mean(v):.2f} ({' / '.join(map(str, v))}) | "
            f"{mean(n):.2f} ({' / '.join(map(str, n))}) | {mean(v) - mean(n):.2f} |"
        )
        out[arm] = cells
    (args.run_dir / "containment.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
