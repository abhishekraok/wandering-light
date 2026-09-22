"""best-system-command (#158): the pinned system's cells on the v4 cohort.

`wandering_light.solve` runs one system, `solve.BEST`, on a task file. This
driver runs variants of it that differ in one setting on the 300-task cohort
of compute-optimal-mark-28 (#101), so the setting `BEST` pins is a measured
choice: each cell is one `solve_file` run, written as a timing cell in the
shape world-plus-v4-policy (#134)'s verifier reads, and the report counts
graded and **verified** behaviours per cell, verification being the 10,000-draw
check of `experiments/step3_verify_solves.py` on every distinct solved program.

Stages, each skipped where its output exists so a run resumes::

    uv run python experiments/best_system_cells.py run --run-dir checkpoints/best-system-20260922 --cells b64,b128,b256,b512
    uv run python experiments/best_system_cells.py report --run-dir checkpoints/best-system-20260922 --out reports/best-system-20260922

`run` writes `cohort.jsonl.gz` (the #101 cohort, in cohort order) and one
`timing/<cell>.json` per cell; `report` draw-tests the distinct solved programs
once (cached in `verified.json`) and writes `cells.json` and `tables.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.step3_verify_solves as verify  # noqa: E402
from wandering_light import solve  # noqa: E402
from wandering_light.stack_pbe import CATALOGUE  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from wandering_light.solve import System

# Each cell is one setting away from BEST, so a difference reads as that setting.
CELLS: dict[str, Callable[[System], System]] = {
    "b64": lambda best: dataclasses.replace(best, beam_width=64),
    "b128": lambda best: dataclasses.replace(best, beam_width=128),
    "b256": lambda best: dataclasses.replace(best, beam_width=256),
    "b512": lambda best: dataclasses.replace(best, beam_width=512),
    # The controls: the state-size guard off (#157), and the library off (#151).
    "nocap-b64": lambda best: dataclasses.replace(
        best, beam_width=64, max_state_atoms=None
    ),
    "nolib-b64": lambda best: dataclasses.replace(best, beam_width=64, library=False),
    "nolib-b128": lambda best: dataclasses.replace(best, beam_width=128, library=False),
    "nolib-b256": lambda best: dataclasses.replace(best, beam_width=256, library=False),
    "nolib-b512": lambda best: dataclasses.replace(best, beam_width=512, library=False),
    # Depth, for a split whose witnesses are longer than the search
    # (clear-world-v3, #161). The #157 state cap is what makes a deep search
    # safe to run: without it a policy that likes `dup cat` has four times as
    # many layers in which to park the beam on a million-atom state.
    "d96": lambda best: dataclasses.replace(best, max_depth=96),
    "d192": lambda best: dataclasses.replace(best, max_depth=192),
    "d384": lambda best: dataclasses.replace(best, max_depth=384),
}
DEFAULT_CELLS = ("b64", "b128", "b256", "b512")
SEED = 0  # the checkpoint's training seed, which the verifier's cell label wants
VERIFIED = ("genuine", "equals_catalogue_witness")
# The verdict for a behaviour outside the v4 catalogue: graded, never verified,
# and not a shortcut either, since nothing was drawn to catch one.
UNVERIFIABLE = "no_sampler"


def cohort_file(run_dir: Path, log) -> Path:
    path = run_dir / "cohort.jsonl.gz"
    if path.exists():
        return path
    tasks = co.corpus()
    cohort = co.cohort(tasks)
    run_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for _, task in cohort:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    (run_dir / "cohort_index.json").write_text(
        json.dumps([index for index, _ in cohort]) + "\n"
    )
    log(f"wrote {path}: {len(cohort)} tasks")
    return path


def run(
    run_dir: Path,
    cells: list[str],
    checkpoint: Path | None,
    log,
    tasks_file: Path | None = None,
) -> None:
    """Run each named cell. ``tasks_file`` overrides the #101 cohort.

    The cohort is the default because every published `mark-28` number is on
    it, but the objective in `goals/g1.md` covers world data too, whose tasks
    are not in it (clear-world-v3, #161).
    """
    tasks = tasks_file if tasks_file is not None else cohort_file(run_dir, log)
    index_file = run_dir / "cohort_index.json"
    indices = (
        json.loads(index_file.read_text())
        if tasks_file is None and index_file.exists()
        else None
    )
    for name in cells:
        out = run_dir / "timing" / f"{name}.json"
        if out.exists():
            log(f"skip {name}: {out} exists")
            continue
        system = CELLS[name](solve.BEST)
        log(f"run {name}: {system.describe()}")
        with (
            open(run_dir / f"{name}.txt", "w") as table,
            open(run_dir / f"{name}.log", "w") as progress,
        ):
            summary = solve.solve_file(
                tasks,
                system=system,
                checkpoint=checkpoint,
                json_path=out,
                out=table,
                log=progress,
            )
        record = json.loads(out.read_text())
        record.update(
            {
                "cell": name,
                "beam_width": system.beam_width,
                "max_depth": system.max_depth,
                "guard": system.guard,
                "eval_corpus": co.primary_corpus(),
                "seed": SEED,
                "policy_epoch": None,
                "provenance": co.provenance(),
            }
        )
        for position, row in enumerate(record["rows"]):
            row["task_index"] = indices[position] if indices else position
            row["bucket"] = co.bucket_of(row["length"])
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        log(f"RESULT {name}: {summary.line()}")


def verified_programs(run_dir: Path, cells: list[dict[str, Any]], log) -> dict:
    """Every distinct solved (behaviour, program) with its draw-test verdict, cached."""
    path = run_dir / "verified.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    by_name = {b.name: b for b in CATALOGUE}
    for cell in cells:
        for row in cell["rows"]:
            if not row["solved"]:
                continue
            key = f"{row['behaviour']} :: {' '.join(row['program'])}"
            if key in known:
                continue
            behaviour = by_name.get(row["behaviour"])
            if behaviour is None:
                # A world task (clear-world-v3, #161) has examples but no
                # sampler, so there is nothing to draw fresh tasks from and the
                # program can be graded but never verified.
                known[key] = {"verdict": UNVERIFIABLE}
            else:
                known[key] = verify.check(
                    tuple(row["program"]), behaviour, verify.DRAWS, 134
                )
            log(f"{known[key]['verdict']:24s} {key}")
            path.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
    return known


def report(run_dir: Path, out_dir: Path, log) -> None:
    cells = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "timing").glob("*.json"))
    ]
    verdicts = verified_programs(run_dir, cells, log)
    table = []
    for cell in cells:
        rows = cell["rows"]
        solved = [r for r in rows if r["solved"]]
        graded = {r["behaviour"] for r in solved}

        def verdict(r):
            return verdicts[f"{r['behaviour']} :: {' '.join(r['program'])}"]["verdict"]

        verified = {r["behaviour"] for r in solved if verdict(r) in VERIFIED}
        unverifiable = {r["behaviour"] for r in solved if verdict(r) == UNVERIFIABLE}
        buckets = {b: sum(r["bucket"] == b for r in solved) for b in verify.BUCKETS}
        table.append(
            {
                "cell": cell["cell"],
                "system": cell["system"],
                "tasks": len(solved),
                # Under a library most solved tasks are replays of one search's
                # program, so a cell's verified count rests on this many
                # independent searches, not on `tasks`.
                "search_solved": sum(r["source"] == "search" for r in solved),
                "graded_behaviours": len(graded),
                "verified_behaviours": len(verified),
                "verified": sorted(verified),
                "shortcut_only": sorted(graded - verified - unverifiable),
                "unverifiable": sorted(unverifiable),
                "cohort_tasks": len(rows),
                "cohort_behaviours": len({r["behaviour"] for r in rows}),
                "from_library": cell["summary"]["from_library"],
                "memorised": cell["summary"]["memorised"],
                "seconds": cell["summary"]["seconds"],
                "executions": cell["summary"]["executions"],
                "buckets": buckets,
                "slowest_task_seconds": max(r["seconds"] for r in rows),
                "host": cell["host"],
                "commit": cell["commit"],
            }
        )
    # Filename order puts b128 before b64; read the ladder in width order with
    # the one-setting controls after it.
    table.sort(key=lambda t: (not t["cell"].startswith("b"), t["system"]["beam_width"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cells.json").write_text(json.dumps(table, indent=2) + "\n")
    # Every cell in a run dir runs the same task file, so the first one's totals
    # are the table's: /300 and /43 on the #101 cohort, /44 on world-v3 eval.
    n_tasks = table[0]["cohort_tasks"] if table else 0
    n_behaviours = table[0]["cohort_behaviours"] if table else 0
    lines = [
        f"| cell | beam | state cap | library | tasks /{n_tasks} | search-solved | "
        f"graded behaviours /{n_behaviours} | "
        f"**verified /{n_behaviours}** | timed (s) | executions | from library | "
        "slowest task (s) | " + " / ".join(verify.BUCKETS) + " |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in table:
        s = t["system"]
        lines.append(
            f"| {t['cell']} | {s['beam_width']} | {s['max_state_atoms']} | "
            f"{'on' if s['library'] else 'off'} | {t['tasks']} | {t['search_solved']} | "
            f"{t['graded_behaviours']} | "
            f"**{t['verified_behaviours']}** | {t['seconds']:.0f} | {t['executions']:,} | "
            f"{t['from_library']} | {t['slowest_task_seconds']:.2f} | "
            + " / ".join(str(t["buckets"][b]) for b in verify.BUCKETS)
            + " |"
        )
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a local checkpoint in place of BEST's, recorded as such",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="a task file to run instead of the #101 cohort",
    )
    args = parser.parse_args(argv)

    def log(line: str) -> None:
        print(line, flush=True)
        with open(args.run_dir / "log.txt", "a") as handle:
            handle.write(line + "\n")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        run(args.run_dir, args.cells.split(","), args.checkpoint, log, args.tasks)
    else:
        report(args.run_dir, args.out or args.run_dir, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
