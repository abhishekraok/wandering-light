"""Assemble experiments/staged_witnesses/v4.jsonl from the pilot and full-run author files.

    PYTHONPATH=. uv run python experiments/staged_witnesses/build_side_file.py

Every row is re-assessed here, so the side file never carries a verdict an
author reported. Overrides and flags are reviewer judgements (#155) and are
listed below rather than buried in the data.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.staged_witnesses.check import (
    assess,
    assess_refused,
    skeleton,
    v4_tasks,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "reports" / "staged-pilot-20260920"
FULL = ROOT / "reports" / "staged-full-20260920"
OUT = Path(__file__).with_name("v4.jsonl")

# An approximation that only verifies because the sampler's lists are short is
# not a witness of the function. Replaced by the exact program the author named.
OVERRIDES = {
    "running_minima": (
        "uncons quote [ over uncons swap drop [ lt? ] [ ] [ swap ] ifte drop cons ] fold"
        " nil [ cons ] fold"
    ),
}
# Within the rules only through a trick a person would not give as the
# decomposition: kept and verified, flagged so an arm can drop them.
CONTRIVED = {
    "mean": "shifted mean, to reach +2 under the pilot's classifier",
    "max_pair": "selection as the blend a + s*(b-a) in place of a branch",
    "compare_each_to_mean": "sign as d // (d*d + 1)",
    "sorted_within_pairs": "pair sort as (s+g)/2 and s-max, 141 ops",
    "largest_zero_delimited_sum": "data turned into programs and run with `i`",
    "evaluate_polynomial_horner": "Horner replaced by an explicit power sum, 2.85x",
}
# Correct on the sampler's domain and not beyond it, like the fused witness.
PRECONDITION = {
    "interval_violation_count": "lo < hi",
    "longest_interval_run": "lo <= hi",
    "take_prefix_count": "count >= 0",
    "strict_turn_count": "at least two elements",
    "count_outside_interval": "lo <= hi",
    "interval_region_codes": "lo < hi",
    "total_variation": "non-empty list",
}


def author_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(PILOT.glob("programs_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row["behaviour"]] = {**row, "author": f"pilot-{path.stem[-1]}"}
    for path in sorted(FULL.glob("programs_*.jsonl")):
        name = path.stem.replace("programs_", "")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if name.startswith("r"):  # pilot re-fusers: keep the pilot author's program
                rows[row["behaviour"]]["refused_program"] = row["refused_program"]
                rows[row["behaviour"]]["refuser"] = name
            else:
                rows[row["behaviour"]] = {**row, "author": name}
    return rows


def main() -> None:
    tasks = v4_tasks()
    eval_skeletons = {
        skeleton(g[0].program) for g in tasks.values() if g[0].split != "train"
    }
    out = []
    for name, row in sorted(author_rows().items()):
        program, refused = row["program"], row["refused_program"]
        if name in OVERRIDES:
            program = refused = OVERRIDES[name]
        result = assess(name, program, tasks, eval_skeletons)
        twin = assess_refused(name, refused, tasks)
        assert result["verified"] and twin["verified"], name
        flags = []
        if name in CONTRIVED:
            flags.append(f"contrived: {CONTRIVED[name]}")
        if name in PRECONDITION:
            flags.append(f"precondition: {PRECONDITION[name]}")
        if name in OVERRIDES:
            flags.append(
                "override: author's approximate program replaced by the exact one"
            )
        out.append(
            {
                "behaviour": name,
                "verdict": result["verdict"],
                "program": program,
                "refused_program": refused,
                "fused_length": result["fused"]["length"],
                "length_ratio": result["length_ratio"],
                "post_execution_gain": result["post_execution_gain"],
                "broken_rules": result["broken_rules"],
                "fused": result["fused"],
                "staged": result["staged"],
                "refused": {k: v for k, v in twin.items() if k != "verified"},
                "flags": flags,
                "author": row["author"],
                "author_model": "claude-opus-5",
                "notes": row.get("notes", ""),
            }
        )
    OUT.write_text("".join(json.dumps(row) + "\n" for row in out))
    print(len(out), "rows ->", OUT)


if __name__ == "__main__":
    main()
