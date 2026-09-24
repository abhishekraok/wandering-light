"""staged-witness-reauthoring (#155): does the shape of the witness text bind the policy?

Arms of the #101 / #134 1m recipe. Tasks, examples, splits, labels-per-row
rule, optimiser, batch, seeds and the validation set are the same in every arm;
what differs is which program each train task is replayed along:

    fused     the v4 witness (the current recipe, control)
    refused   the re-fused twin: the staged program's algorithm, cycles merged back
    staged    the staged witness: short write-run cycles with closed states between
    both      every task contributes its fused and its staged witness
    capped    short-segment-campaign (#168): bodies <= 4 atoms, segments <= 8 ops

`experiments/staged_witnesses/v4.jsonl` supplies the programs of the first four.
A behaviour gets its staged / re-fused program only when its verdict is
``staged`` and it is not flagged ``contrived`` (``--keep-contrived`` keeps
those); every other behaviour keeps its fused witness in every arm, so arms
differ on the treated behaviours alone. ``capped`` reads
`reports/short-segment-20260923/train_capped.jsonl` instead, lines
``{"behaviour", "program"}``: every behaviour listed takes that program and every
other keeps its fused witness, as ``staged`` treats its untreated behaviours.
``task.skeleton`` is never touched, so `validation_skeleton` membership is the
fused corpus's in every arm.

**Encoding.** ``--encoding`` is the policy's item layout (`stack_policy`'s
``machine``, every published checkpoint's, or ``difference``, #168's comparing
encoder) for every arm a run trains; rows and model are built in it and the
checkpoint's config and manifest record it. One run directory holds one
encoding: a snapshot already there in the other encoding is an error, not a
skip.

**Sampling.** Replaying longer witnesses yields more rows, which under the
recipe's row-uniform epochs would re-weight behaviours by witness length and
double the `both` arm. Every arm therefore trains on the same number of rows
(the fused arm's count), drawn task-uniform, then witness-uniform, then
row-uniform, once, with a fixed seed. Same rows per epoch, same batch, same
epochs: same optimiser updates. The draw is shared by an arm's three training
seeds, as the recipe's fixed row set is, so the seed spread is initialisation
and shuffle noise, not resampling noise; arms get independent draws (per-task
counts differ by about 2% between arms).

Stages, each skipped when its output exists so a run resumes::

    uv run python experiments/staged_arms.py train --run-dir checkpoints/staged-arms-20260920
    uv run python experiments/staged_arms.py train --arms fused,staged,capped --encoding difference --run-dir ...
    uv run python experiments/staged_arms.py time  --run-dir checkpoints/staged-arms-20260920
    uv run python experiments/staged_arms.py report --run-dir ... --out reports/...
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
from wandering_light.stack_beam import beam_solve  # noqa: E402
from wandering_light.stack_pbe_eval import grade  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    CONFIG_FILE,
    ENCODINGS,
    MACHINE,
    PolicyScorer,
    load_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import Row

SIDE_FILE = REPO / "experiments" / "staged_witnesses" / "v4.jsonl"
# short-segment-campaign (#168): the capped witnesses, written when authoring ends.
CAPPED_FILE = REPO / "reports" / "short-segment-20260923" / "train_capped.jsonl"
ARMS = ("fused", "refused", "staged", "both", "capped")
SEEDS = (0, 1, 2)
# epoch-selection-in-behaviours (#154): the best fixed epoch of the fused arm in
# verified behaviours. Fixed in advance for every arm; validation CE is not used.
EPOCHS = 3
SAMPLE_SEED = 20260920
# A search on one task normally ends in under 2 s (the slowest of 2,100 in the
# first seven cells took 1.6 s). One staged-arm cell sat on a single task for
# over 20 minutes at 100% CPU: that policy's beam wanders into `dup cat dup cat`,
# which doubles a list for 2 fuel, so states reach millions of atoms and the time
# goes into hashing and encoding them (the beam has no state-size bound). A task
# is abandoned as unsolved after this long and recorded ``wall_clock_cap``. At
# 60 s the cell capped eleven tasks in a row, so the cap is 10 s, still six times
# the slowest normal search.
TASK_SECONDS_CAP = 10


class TaskTimeoutError(Exception):
    pass


def _raise_timeout(signum, frame):
    raise TaskTimeoutError


def treated(keep_contrived: bool) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in SIDE_FILE.read_text().splitlines():
        row = json.loads(line)
        contrived = any(flag.startswith("contrived") for flag in row["flags"])
        if row["verdict"] == "staged" and (keep_contrived or not contrived):
            rows[row["behaviour"]] = row
    return rows


def capped_programs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The ``capped`` arm's side file, ``behaviour -> row``; every line is treated."""
    path = CAPPED_FILE if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"the capped arm needs {path}, which does not exist yet: it is written "
            "when short-segment-campaign (#168) authoring ends, one "
            '{"behaviour", "program"} line per re-authored train behaviour'
        )
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        behaviour, program = row.get("behaviour"), row.get("program")
        if not behaviour or not program or not str(program).split():
            raise ValueError(f"{path}:{number}: a line needs a behaviour and a program")
        if behaviour in rows:
            raise ValueError(f"{path}:{number}: {behaviour} is listed twice")
        rows[behaviour] = row
    return rows


def side_file(arm: str) -> Path:
    return CAPPED_FILE if arm == "capped" else SIDE_FILE


def _repo_path(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def with_program(task: Task, text: str) -> Task:
    program = tuple(text.split())
    return dataclasses.replace(task, program=program, length=len(program))


def balanced(groups: Sequence[Sequence[Sequence[Row]]], size: int) -> list[Row]:
    """``size`` rows: a task uniformly, one of its witnesses, one of its rows."""
    rng = random.Random(SAMPLE_SEED)
    out = []
    for _ in range(size):
        witnesses = groups[rng.randrange(len(groups))]
        rows = witnesses[rng.randrange(len(witnesses))]
        out.append(rows[rng.randrange(len(rows))])
    return out


def by_task(rows: Sequence[Row], count: int) -> list[list[Row]]:
    grouped: list[list[Row]] = [[] for _ in range(count)]
    for row in rows:
        grouped[row.task].append(row)
    return grouped


def arm_rows(
    arm: str, tasks: Sequence[Task], keep_contrived: bool, encoding: str = MACHINE
) -> tuple[list[Row], list[Row], dict[str, Any]]:
    """``(fit, validation, facts)``. Validation is the fused holdout in every arm.

    Rows are built in ``encoding``; which rows are drawn does not depend on it.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {ARMS}")
    side = capped_programs() if arm == "capped" else treated(keep_contrived)
    train = [t for t in tasks if t.split == "train"]
    fit_tasks = [t for t in train if not validation_skeleton(t)]
    held = [t for t in train if validation_skeleton(t)]
    fused = by_task(witness_rows(fit_tasks, encoding), len(fit_tasks))
    size = sum(len(rows) for rows in fused)

    def rewritten(key: str) -> list[list[Row]]:
        swapped = [
            with_program(t, side[t.behaviour][key]) if t.behaviour in side else t
            for t in fit_tasks
        ]
        return by_task(witness_rows(swapped, encoding), len(swapped))

    if arm == "fused":
        groups = [[rows] for rows in fused]
    elif arm == "both":
        staged = rewritten("program")
        groups = [
            [f, s] if t.behaviour in side else [f]
            for t, f, s in zip(fit_tasks, fused, staged, strict=True)
        ]
    else:
        groups = [
            [rows]
            for rows in rewritten("refused_program" if arm == "refused" else "program")
        ]
    path = side_file(arm)
    facts = {
        "side_file": _repo_path(path),
        "side_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "side_behaviours": len(side),
        "fit_tasks": len(fit_tasks),
        "treated_behaviours": len(
            {t.behaviour for t in fit_tasks if t.behaviour in side}
        ),
        "treated_tasks": sum(t.behaviour in side for t in fit_tasks),
        "distinct_rows_available": sum(len(w) for g in groups for w in g),
        "rows_per_epoch": size,
    }
    return balanced(groups, size), witness_rows(held, encoding), facts


def check_arms(arms: Sequence[str]) -> None:
    """Refuse an unknown arm, or ``capped`` without its side file, before any work."""
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {ARMS}")
    if "capped" in arms:
        capped_programs()


def snapshot_encoding(path: Path) -> str:
    """The encoding a saved snapshot was trained in; none recorded is ``machine``."""
    return json.loads((path / CONFIG_FILE).read_text()).get("encoding", MACHINE)


def _existing(run_dir: Path, arm: str, seed: int, encoding: str) -> bool:
    """Is this seed's final snapshot there? One in another encoding is an error."""
    path = hs.policy_dir(run_dir, arm, seed, EPOCHS)
    if not path.exists():
        return False
    found = snapshot_encoding(path)
    if found != encoding:
        raise ValueError(
            f"{path} was trained with encoding {found!r}, not {encoding!r}; "
            "use a separate --run-dir per encoding"
        )
    return True


def train(
    run_dir: Path,
    tasks,
    device: str,
    arms,
    seeds,
    keep_contrived: bool,
    log,
    encoding: str = MACHINE,
) -> None:
    check_arms(arms)
    for arm in arms:
        if all(_existing(run_dir, arm, s, encoding) for s in seeds):
            log(f"skip train {arm} (snapshots exist)")
            continue
        fit, held, facts = arm_rows(arm, tasks, keep_contrived, encoding)
        log(f"RESULT rows {arm} {encoding}: {json.dumps(facts)} validation {len(held)}")
        for seed in seeds:
            if _existing(run_dir, arm, seed, encoding):
                log(f"skip train {arm} s{seed} (snapshots exist)")
                continue
            manifest = {
                "arm": arm,
                "policy_arm": hs.ARM,
                "encoding": encoding,
                "epochs": EPOCHS,
                "learning_rate": hs.LEARNING_RATE,
                "seed": seed,
                "sample_seed": SAMPLE_SEED,
                "keep_contrived": keep_contrived,
                **facts,
                "provenance": co.provenance(),
            }
            started = time.perf_counter()
            _model, state = train_policy(
                fit,
                held,
                co.architecture(hs.ARM),
                epochs=EPOCHS,
                learning_rate=hs.LEARNING_RATE,
                batch_size=hs.BATCH,
                validation_batch_size=hs.VAL_BATCH,
                seed=seed,
                device=device,
                epoch_end=hs.snapshotter(run_dir, arm, seed, manifest),
                encoding=encoding,
            )
            co.write_json(run_dir / "training" / f"{arm}-s{seed}.json", state)
            curve = " ".join(
                f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}"
                for r in state["history"]
            )
            log(
                f"RESULT train {arm} {encoding} s{seed}: [{curve}] "
                f"{time.perf_counter() - started:.0f}s"
            )


def time_cell(
    run_dir: Path, tasks, arm: str, seed: int, device: str, beam: int, depth: int, log
):
    """`hindsight_subtargets.time_cell` with a per-task wall-clock cap; same cell format."""
    out = hs.cell_path(run_dir, arm, seed, device, EPOCHS, beam, depth)
    if out.exists():
        log(f"skip time {out.name}")
        return out
    if device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    scorer = PolicyScorer(
        load_policy(hs.policy_dir(run_dir, arm, seed, EPOCHS), device=device), device
    )
    rows = []
    gc.freeze()
    signal.signal(signal.SIGALRM, _raise_timeout)
    started = time.perf_counter()
    for index, task in co.cohort(tasks):
        base = {
            "task_index": index,
            "behaviour": task.behaviour,
            "length": task.length,
            "bucket": co.bucket_of(task.length),
            "examples": task.examples,
        }
        task_started = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, TASK_SECONDS_CAP)
        try:
            result = beam_solve(
                task, scorer, beam_width=beam, max_depth=depth, guard=co.GUARD
            )
        except TaskTimeoutError:
            seconds = time.perf_counter() - task_started
            log(
                f"RESULT capped {arm} s{seed}: task {index} {task.behaviour} after {seconds:.0f}s"
            )
            rows.append(
                {
                    **base,
                    "solved": False,
                    "shown_fit": False,
                    "program": None,
                    "termination": "wall_clock_cap",
                    "seconds": seconds,
                    "executions": None,
                    "fuel": None,
                    "expansions": None,
                    "depth": None,
                }
            )
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        score = grade(result.program, task) if result.program else None
        rows.append(
            {
                **base,
                "solved": bool(score and score.solved),
                "shown_fit": bool(score and score.shown),
                "program": list(result.program) if result.program else None,
                "termination": result.termination,
                "seconds": result.seconds,
                "executions": result.executions,
                "fuel": result.fuel,
                "expansions": result.expansions,
                "depth": result.depth,
            }
        )
    cell_seconds = time.perf_counter() - started
    co.write_json(
        out,
        {
            "arm": arm,
            "policy_arm": hs.ARM,
            "encoding": scorer.encoding,
            "seed": seed,
            "device": device,
            "policy_epoch": EPOCHS,
            "beam_width": beam,
            "max_depth": depth,
            "guard": co.GUARD,
            "cpu_threads": co.CPU_THREADS,
            "task_seconds_cap": TASK_SECONDS_CAP,
            "eval_corpus": co.primary_corpus(),
            "host": co.host(),
            "provenance": co.provenance(),
            "cell_seconds": cell_seconds,
            "rows": rows,
        },
    )
    solved = sum(r["solved"] for r in rows)
    capped = sum(r["termination"] == "wall_clock_cap" for r in rows)
    log(
        f"RESULT time {arm} s{seed} e{EPOCHS} beam {beam} depth {depth}: "
        f"{solved}/{len(rows)} solved, {capped} capped, {cell_seconds:.0f}s"
    )
    return out


def report(run_dir: Path, out_dir: Path, log) -> Path:
    """Graded solves per arm, by task and by distinct behaviour. Verification is separate."""
    cells = defaultdict(list)
    for path in sorted((run_dir / "timing").glob("*.json")):
        cell = json.loads(path.read_text())
        key = (
            cell["arm"],
            cell["device"],
            cell["policy_epoch"],
            cell["beam_width"],
            cell["max_depth"],
        )
        cells[key].append(cell)
    lines = [
        "| arm | device | epoch | beam | depth | seeds | graded tasks /300 | graded behaviours /43 | by bucket (tasks) | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (arm, device, epoch, beam, depth), group in sorted(cells.items()):
        tasks_solved = [sum(r["solved"] for r in c["rows"]) for c in group]
        behaviours = [
            len({r["behaviour"] for r in c["rows"] if r["solved"]}) for c in group
        ]
        buckets = defaultdict(list)
        for c in group:
            per = defaultdict(int)
            for r in c["rows"]:
                per[r["bucket"]] += r["solved"]
            for bucket, n in per.items():
                buckets[bucket].append(n)
        by_bucket = " ".join(
            f"{b}:{mean(v):.1f}"
            for b, v in sorted(
                buckets.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+"))
            )
        )
        lines.append(
            f"| {arm} | {device} | {epoch} | {beam} | {depth} | {len(group)} | "
            f"{mean(tasks_solved):.1f} ({' / '.join(map(str, tasks_solved))}) | "
            f"{mean(behaviours):.2f} ({' / '.join(map(str, behaviours))}) | {by_bucket} | "
            f"{mean(c['cell_seconds'] for c in group):.0f} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "graded.md"
    out.write_text("\n".join(lines) + "\n")
    log(f"wrote {out}")
    print("\n".join(lines))
    return out


VERIFIED = {"genuine", "equals_catalogue_witness"}


def verified_report(run_dir: Path, verdicts_path: Path, out_dir: Path, log) -> Path:
    """Per arm: verified tasks and distinct verified behaviours, from `step3_verify_solves`.

    A solve counts when its program's verdict is ``genuine`` (10,000 draws in
    each mode, no disagreement) or it equals a catalogue witness; ``shortcut``
    and ``inconclusive`` solves do not. Behaviours are the unit that bounds
    coverage: the cohort's 300 tasks are 43 behaviours (#151).
    """
    verdicts = {
        (row["behaviour"], row["program"]): row["verdict"]
        for row in json.loads(verdicts_path.read_text())["programs"]
    }
    per_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((run_dir / "timing").glob("*.json")):
        cell = json.loads(path.read_text())
        good = [
            r
            for r in cell["rows"]
            if r["solved"]
            and verdicts.get((r["behaviour"], " ".join(r["program"]))) in VERIFIED
        ]
        per_bucket: dict[str, set[str]] = defaultdict(set)
        for r in good:
            per_bucket[r["bucket"]].add(r["behaviour"])
        per_arm[cell["arm"]].append(
            {
                "seed": cell["seed"],
                "graded": sum(r["solved"] for r in cell["rows"]),
                "tasks": len(good),
                "behaviours": {r["behaviour"] for r in good},
                "per_bucket": per_bucket,
                "longest": max((len(r["program"]) for r in good), default=0),
            }
        )
    order = [arm for arm in ARMS if arm in per_arm]
    buckets = [co.bucket_of(low) for low, _ in co.BUCKETS]
    lines = [
        "| arm | graded tasks | verified tasks | verified behaviours /43 | union over seeds | "
        + " | ".join(f"{b} (beh.)" for b in buckets)
        + " | longest verified program |",
        "|---|---|---|---|---|" + "---|" * (len(buckets) + 1),
    ]
    for arm in order:
        cells = sorted(per_arm[arm], key=lambda c: c["seed"])
        union = set().union(*(c["behaviours"] for c in cells))

        def trio(key, cells=cells):
            values = [len(c[key]) if isinstance(c[key], set) else c[key] for c in cells]
            return f"{mean(values):.2f} ({' / '.join(map(str, values))})"

        by_bucket = " | ".join(
            f"{mean(len(c['per_bucket'][b]) for c in cells):.2f}" for b in buckets
        )
        lines.append(
            f"| {arm} | {trio('graded')} | {trio('tasks')} | {trio('behaviours')} | "
            f"{len(union)} | {by_bucket} | {max(c['longest'] for c in cells)} ops |"
        )
    if "fused" in per_arm:
        base = set().union(*(c["behaviours"] for c in per_arm["fused"]))
        lines += [
            "",
            "Behaviours verified by some seed of an arm and by no seed of `fused`, and the reverse:",
            "",
        ]
        for arm in order:
            if arm == "fused":
                continue
            union = set().union(*(c["behaviours"] for c in per_arm[arm]))
            lines.append(
                f"- `{arm}`: gains {sorted(union - base) or 'none'}; loses {sorted(base - union) or 'none'}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "verified.md"
    out.write_text("\n".join(lines) + "\n")
    log(f"wrote {out}")
    print("\n".join(lines))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("rows", "train", "time", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--beam", type=int, default=co.BEAM_WIDTH)
    parser.add_argument("--depth", type=int, default=co.MAX_DEPTH)
    parser.add_argument("--keep-contrived", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=ENCODINGS,
        default=MACHINE,
        help="the policy's item layout, for every arm this run trains",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        help="step3_verify_solves output; report verified counts",
    )
    args = parser.parse_args(argv)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.run_dir / "log.txt"

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with logfile.open("a") as handle:
            handle.write(stamped + "\n")

    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.stage == "report":
        report(args.run_dir, args.out or args.run_dir, log)
        if args.verdicts:
            verified_report(args.run_dir, args.verdicts, args.out or args.run_dir, log)
        return 0
    if args.stage in ("rows", "train"):
        check_arms(arms)
    tasks = co.corpus()
    if args.stage == "rows":
        for arm in arms:
            fit, held, facts = arm_rows(arm, tasks, args.keep_contrived, args.encoding)
            labels = defaultdict(int)
            for row in fit:
                labels[row.label] += 1
            log(
                f"RESULT rows {arm} {args.encoding}: {json.dumps(facts)} validation {len(held)} labels {len(labels)}"
            )
    elif args.stage == "train":
        train(
            args.run_dir,
            tasks,
            args.device,
            arms,
            seeds,
            args.keep_contrived,
            log,
            args.encoding,
        )
    else:
        for arm in arms:
            for seed in seeds:
                time_cell(
                    args.run_dir,
                    tasks,
                    arm,
                    seed,
                    args.device,
                    args.beam,
                    args.depth,
                    log,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
