"""The report for solved-program-cache (#151) and hindsight-subtargets (#152).

Reads the campaign's cells and writes one markdown file. Tasks and behaviours
are reported side by side everywhere, because the cohort is 43 behaviours behind
300 tasks and a cache moves the first without moving the second.

    uv run python experiments/cache_report.py --run-dir checkpoints/cache-20260919 \
        --sim reports/.../order_sim.json --out reports/solved-program-cache-20260919
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41+")
# docs/STATUS.md records both samplers as leaking; a "behaviour" credited only
# through one of them is a sampler defect, not coverage.
LEAKY = ("all_positive", "bounds_cover_list_flag")


def load(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Every cell in the directory; anything that is not one is skipped."""
    cells = {}
    for path in sorted(run_dir.glob("*.json")):
        try:
            cell = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(cell, dict) and "rows" in cell and "cell_seconds" in cell:
            cells[path.stem] = cell
    return cells


def stats(cell: dict[str, Any]) -> dict[str, Any]:
    rows = cell["rows"]
    solved = [r for r in rows if r["solved"]]
    behaviours = {r["behaviour"] for r in solved}
    per_bucket = defaultdict(int)
    for r in solved:
        per_bucket[r["bucket"]] += 1
    return {
        "tasks": len(solved),
        "behaviours": len(behaviours),
        "behaviours_clean": len(behaviours - set(LEAKY)),
        "seconds": cell.get("timed_seconds", cell["cell_seconds"]),
        "cell_seconds": cell["cell_seconds"],
        "executions": sum(r["executions"] for r in rows),
        "from_cache": sum(1 for r in rows if r.get("source") == "cache"),
        "shown_fit": sum(1 for r in rows if r["shown_fit"]),
        "buckets": {b: per_bucket[b] for b in BUCKETS},
        "n": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sim", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cells = load(args.run_dir)
    s = {name: stats(cell) for name, cell in cells.items()}
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# solved-program-cache (#151) and hindsight-subtargets (#152)")
    w()
    w("The #101 300-task `stack_pbe_v4` cohort, beam 64 x depth 48, guard `both`, one")
    w("host (`UltraDesk`, RTX 4090, 8 CPU threads), one commit. Every cell here was")
    w("timed with the faster `beam_solve`, so none of them compares with a recorded")
    w("pre-speedup wall-clock; the arms compare with each other. **timed** is the sum")
    w("of the per-task timers, which is the quantity `docs/metrics.md` defines and the")
    w("quantity the recorded cells aggregate; **elapsed** is the whole cell and is kept")
    w("beside it. Grading runs after the timers are sealed, because it warms the")
    w("executor's process-global `run_body` cache that later searches read.")
    w()
    w("The cohort is **43 distinct behaviours behind 300 tasks**, so tasks and")
    w("behaviours are reported side by side. `behaviours*` excludes")
    w(f"{' and '.join(f'`{b}`' for b in LEAKY)}, whose samplers `docs/STATUS.md` records as leaking.")
    w()

    if "off-s0-e4" in s and "on-s0-e4" in s:
        w("## The cache")
        w()
        w("| cell | tasks /300 | behaviours /43 | behaviours* | timed (s) | elapsed (s) | executions | served from cache |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name in (
            "off-s0-e4",
            "off-s0-e4-order1",
            "on-s0-e4",
            "on-s0-e4-order1",
            "on-s0-e4-order2",
            "disjoint-s0-e4",
        ):
            if name not in s:
                continue
            c = s[name]
            order = cells[name].get("order_seed")
            label = f"`{name}`" + (f" (order {order})" if order is not None else "")
            if cells[name].get("cache") == "disjoint":
                label += " — diagnostic, cross-behaviour reuse only"
            w(
                f"| {label} | {c['tasks']} | {c['behaviours']} | {c['behaviours_clean']} | "
                f"{c['seconds']:.0f} | {c['cell_seconds']:.0f} | {c['executions']:,} "
                f"| {c['from_cache']} |"
            )
        w()
        w("Per witness-length bucket:")
        w()
        w("| cell | " + " | ".join(BUCKETS) + " |")
        w("|---|" + "---:|" * len(BUCKETS))
        for name in ("off-s0-e4", "on-s0-e4", "disjoint-s0-e4"):
            if name in s:
                w(f"| `{name}` | " + " | ".join(f"{s[name]['buckets'][b]}/60" for b in BUCKETS) + " |")
        w()

    if args.sim and args.sim.exists():
        sim = json.loads(args.sim.read_text())
        arm = sim["arms"]["shortest-first"]
        r = arm["random_order"]
        w("## Over 100 task orders, simulated exactly")
        w()
        w("The cache never changes what a search returns, only whether a task is searched,")
        w("so one cache-off cell plus a fits matrix determines every order. `cohort()` is")
        w("bucket-major and witness-length bucket is not visible to a solver that sees only")
        w("shown examples, so bucket-major is not an order the system could choose.")
        w()
        w("| | tasks /300 | behaviours /43 | searches | regressions |")
        w("|---|---:|---:|---:|---:|")
        w(
            f"| cache off | {sim['baseline']['solved']} | {sim['baseline']['behaviours']} "
            f"| 300 | — |"
        )
        w(
            f"| cache on, random order | {r['solved_mean']:.1f} ± {r['solved_sd']:.1f} "
            f"[{r['solved_min']}-{r['solved_max']}] | {r['behaviours_mean']:.1f} "
            f"[{r['behaviours_min']}-{r['behaviours_max']}] | {r['searches_mean']:.0f} "
            f"| {r['regressions_mean']:.1f} (max {r['regressions_max']}) |"
        )
        b = arm["bucket_major"]
        w(
            f"| cache on, bucket-major (privileged) | {b['solved']} | {b['behaviours']} "
            f"| {b['searches']} | {arm['bucket_major_regressions']} |"
        )
        w()
        w(
            f"{r['verified_cache_hits_mean']:.0f} of {r['cache_hits_mean']:.0f} cache hits "
            "survive held-out grading. The replay rule does not matter: "
            + ", ".join(
                f"{rule} {sim['arms'][rule]['random_order']['solved_mean']:.1f}"
                for rule in sim["arms"]
            )
            + "."
        )
        w()

    arms = {
        name: c
        for name, c in s.items()
        if name.startswith("off-") and name.endswith("-e4")
    }
    if len(arms) > 1:
        w("## Training arms at a matched epoch (#152), cache off")
        w()
        w("Identical row counts, labels, optimiser and seeds; only the target each")
        w("training row carries differs.")
        w()
        by_arm: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for name, c in arms.items():
            stem = name[len("off-"):-len("-e4")]
            arm = "witness" if stem == "s0" else stem.rsplit("-s", 1)[0]
            by_arm[arm].append((c["tasks"], c["behaviours"]))
        w("Behaviours are reported beside tasks and are the more stable of the two:")
        w("within one arm the seed spread on tasks is several times the spread on")
        w("behaviours, so a task difference of under ~15 says nothing.")
        w()
        w("| arm | seeds | tasks /300, mean ± sd | behaviours /43, mean ± sd | task range |")
        w("|---|---:|---:|---:|---|")
        for arm in ("witness", "hindsight", "hindsight-all", "mixed"):
            vals = by_arm.get(arm)
            if not vals:
                continue
            t = [v[0] for v in vals]
            b = [v[1] for v in vals]
            w(
                f"| `{arm}` | {len(vals)} | {mean(t):.1f} ± {pstdev(t):.1f} "
                f"| {mean(b):.1f} ± {pstdev(b):.1f} | {min(t)}-{max(t)} |"
            )
        w()

    extras = [n for n in s if n.startswith("on-") and ("-b" in n or "-d" in n)]
    if extras:
        w("## Search shape, on top of the cache")
        w()
        w("| cell | tasks /300 | behaviours /43 | wall (s) | executions |")
        w("|---|---:|---:|---:|---:|")
        for name in ["on-s0-e4", *sorted(extras)]:
            if name not in s:
                continue
            c = s[name]
            w(
                f"| `{name}` | {c['tasks']} | {c['behaviours']} | {c['seconds']:.0f} "
                f"| {c['executions']:,} |"
            )
        w()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    (args.out / "cells.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
