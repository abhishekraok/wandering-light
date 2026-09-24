"""Does authored body length separate what BEST solves from what it does not? (#165)

For each #101-cohort behaviour: whether `BEST`'s `b64` cell (best-system-command,
#158) verifies or grades it, beside its authored v4 eval witness's longest body
and quotation count. Registered on #165 before it was computed.

    uv run python reports/solver-rescoring-20260923/separation.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.frame_gate import eval_witnesses  # noqa: E402
from experiments.solver_rescoring import body_lengths  # noqa: E402

CELLS = HERE.parents[1] / "reports" / "best-system-20260922" / "cells.json"


def main() -> None:
    cell = next(c for c in json.loads(CELLS.read_text()) if c["cell"] == "b64")
    verified, graded = (
        set(cell["verified"]),
        set(cell["verified"]) | set(cell["shortcut_only"]),
    )
    witnesses = {t.behaviour: t for t in eval_witnesses("stack_pbe_v4")}
    cohort = json.loads(
        (
            HERE.parents[1]
            / "checkpoints"
            / "best-system-20260922"
            / "timing"
            / "b64.json"
        ).read_text()
    )["rows"]
    behaviours = sorted({r["behaviour"] for r in cohort})
    rows = []
    for b in behaviours:
        lengths = body_lengths(witnesses[b].program)
        rows.append(
            {
                "behaviour": b,
                "status": "verified"
                if b in verified
                else "graded"
                if b in graded
                else "unsolved",
                "ops": len(witnesses[b].program),
                "quotations": len(lengths),
                "longest_body": max(lengths, default=0),
            }
        )
    out = {"rows": rows, "groups": {}}
    for status in ("verified", "graded", "unsolved"):
        group = [r for r in rows if r["status"] == status]
        if group:
            out["groups"][status] = {
                "behaviours": len(group),
                **{
                    key: statistics.median(r[key] for r in group)
                    for key in ("longest_body", "quotations", "ops")
                },
            }
    solved = [r for r in rows if r["status"] != "unsolved"]
    unsolved = [r for r in rows if r["status"] == "unsolved"]
    gap = statistics.median(r["longest_body"] for r in unsolved) - statistics.median(
        r["longest_body"] for r in [r for r in rows if r["status"] == "verified"]
    )
    out["gap_unsolved_minus_verified"] = gap
    out["reading"] = (
        "SEPARATES" if gap >= 8 else "DOES_NOT_SEPARATE" if gap < 4 else "BETWEEN"
    )
    out["solved_count"] = len(solved)
    (HERE / "separation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["groups"]), out["reading"], gap)
    for r in sorted(rows, key=lambda r: (r["status"], r["longest_body"])):
        print(
            f"{r['status']:9s} {r['longest_body']:3d} {r['quotations']:2d} {r['ops']:3d} {r['behaviour']}"
        )


if __name__ == "__main__":
    main()
