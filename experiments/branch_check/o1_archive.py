"""branch-and-check (#164) O1-A: audit training continuations, without training.

Preregistered in issuecomment-5965483510. Recorded suffix lengths are upper
bounds, not optimal distances. Timeout/off-witness states get no labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.staged_cycles import closed_product  # noqa: E402
from wandering_light.stack_beam import (  # noqa: E402
    Product,
    closed_cuts,
    step_all_quiet,
)
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import (  # noqa: E402
    BASIS,
    BY_NAME,
    OP_BY_NAME,
    assign_split,
)


def replay_path(start: Product, program: tuple[str, ...]) -> list[Product]:
    states = [start]
    for index, name in enumerate(program):
        moved, _, _ = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None:
            raise ValueError(f"refused op {index}: {name}")
        states.append(moved)
    return states


def candidate_indices(states: list[Product], goal: Product) -> list[int]:
    """Unique closed non-goal states, at their latest recorded occurrence.

    Equal states cannot be assigned conflicting suffix lengths. Dropping the
    goal also removes terminal-state equality as a trivial ordering cue.
    """
    latest = {
        state: index
        for index, state in enumerate(states)
        if state != goal and closed_product(state)
    }
    return sorted(latest.values())


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    name = row["behaviour"]
    if row["split"] != "train" or assign_split(BY_NAME[name].skeleton) != "train":
        raise ValueError("row or live catalogue is not training-only")
    start = tuple(parse_stack(text, basis=BASIS) for text in row["inputs"])
    goal = tuple(parse_stack(text, basis=BASIS) for text in row["outputs"])
    if not start or len(start) != len(goal):
        raise ValueError("empty or mismatched shown example axes")
    program = tuple(row["witness"].split())
    states = replay_path(start, program)
    if states[-1] != goal:
        raise ValueError("witness does not reach shown target")
    cuts = closed_cuts(states, program)
    if cuts != row["witness_cuts"]:
        raise ValueError("saved combinator cuts differ from replay")
    indices = candidate_indices(states, goal)
    cut_states = {states[cut] for cut in cuts}
    flags = kept = on_path = 0
    flagged_closed = 0
    for segment in row["segments"]:
        for record in segment["states"]:
            kept += 1
            on_path += bool(record["on_path"])
            if not record.get("on_witness", False):
                continue
            flags += 1
            flagged = replay_path(start, tuple(record["program"].split()))[-1]
            if flagged not in cut_states:
                raise ValueError("on_witness prefix misses all witness cut states")
            flagged_closed += closed_product(flagged)
    fit_length = None
    if row["program"] is not None:
        fit_program = tuple(row["program"].split())
        if replay_path(start, fit_program)[-1] != goal:
            raise ValueError("returned fit does not reach shown target")
        fit_length = len(fit_program)
    return {
        "behaviour": name,
        "example_set": row["example_set"],
        "witness_ops": len(program),
        "saved_cuts": len(cuts),
        "saved_cuts_not_closed": sum(not closed_product(states[i]) for i in cuts),
        "all_closed_occurrences": sum(closed_product(state) for state in states),
        "candidate_indices": indices,
        "recorded_suffix_ops": [len(program) - i for i in indices],
        "candidate_pairs": len(indices) * (len(indices) - 1) // 2,
        "kept_records": kept,
        "on_witness_records": flags,
        "on_witness_closed_records": flagged_closed,
        "on_path_records": on_path,
        "shown_fit_ops": fit_length,
        "termination": row["termination"],
    }


def audit(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, errors = [], []
    seen = set()
    with gzip.open(path, "rt") as source:
        for position, line in enumerate(source):
            row = json.loads(line)
            key = (row.get("behaviour"), row.get("example_set"))
            try:
                if key in seen:
                    raise ValueError("duplicate behaviour/example-set row")
                seen.add(key)
                rows.append(audit_row(row))
            except (ValueError, KeyError, TypeError, IndexError) as error:
                errors.append({"row": position, "key": key, "reason": str(error)})
    eligible = sorted({r["behaviour"] for r in rows if r["candidate_pairs"]})
    totals = {
        key: sum(r[key] for r in rows)
        for key in (
            "kept_records",
            "on_witness_records",
            "on_witness_closed_records",
            "on_path_records",
            "candidate_pairs",
            "saved_cuts_not_closed",
        )
    }
    return {
        "schema_version": 1,
        "input": str(path.resolve()),
        "sha256": digest,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "preregistration": "https://github.com/abhishekraok/wandering-light/issues/164#issuecomment-5965483510",
        "tasks": len(rows),
        "behaviours": len({r["behaviour"] for r in rows}),
        "eligible_behaviours": eligible,
        "eligible_tasks": sum(bool(r["candidate_pairs"]) for r in rows),
        "shown_fit_tasks": sum(r["shown_fit_ops"] is not None for r in rows),
        "terminations": dict(Counter(r["termination"] for r in rows)),
        "totals": totals,
        "errors": errors,
        "gate_passed": not errors and len(eligible) >= 100,
        "label": "ordering of recorded suffix upper bounds, not optimal distance",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
