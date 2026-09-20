"""staged-witness-reauthoring (#155), prediction 2: can a beam write one cycle?

`staged_arms.py` asks whether a staged corpus is easier to search end to end.
This driver asks the narrower question the prediction was registered on. Stand
on a closed state of a witness, hand the policy **the next closed state as the
target** -- the oracle sub-goal a value function would have to supply -- and let
a beam of 64 write the ops between the two. A staged witness closes every few
ops, so its cycles are short; the fused witness of the same behaviour closes at
the inputs and at the output and rarely in between, so its one "cycle" is most
of the program. Registered in #155: staged at least 0.5 of transitions solved,
fused under 0.1.

Two arms, and the arm is also the policy the evaluation runs under. Scoring
staged text with a fused-trained policy measures transfer and not learnability
(`staged_transfer.py` bounds how badly), so each witness kind is asked of the
policy trained on that kind of text:

    staged-subgoal   rows replay the staged witness where `v4.jsonl` has one
    fused-subgoal    rows replay the v4 witness everywhere (control)

Both arms relabel every row's **target** with the nearest closed product state
strictly ahead of it on the same witness: `hindsight_subtargets` (#152)'s row
with the uniform draw replaced by the oracle the cycles stage then asks the beam
to reach. Both train on the fused arm's 313,560 rows, drawn task-uniform then
row-uniform exactly as `staged_arms.balanced` draws them, so the arms differ in
the witness text and in nothing else.

A transition is solved when `beam_solve` returns a program that **replays** the
start stacks onto the target stacks on every shown example. `grade` is not used:
it wants held-out examples and a single-value output, and a sub-goal is neither.

One property of the search decides how the result must be read. `beam_solve`
tests the hit *while it expands*, before the policy ranks anything, so a one-op
transition is solved by exhaustive one-step search whatever the policy says. A
staged witness is mostly one-op cycles and a fused witness is a handful of them
around one enormous one, so the two kinds' overall shares are not comparable and
the report carries the share over cycles of two ops or more beside every one.

Stages, each skipped when its output exists so a run resumes::

    uv run python experiments/staged_cycles.py rows   --run-dir checkpoints/staged-cycles-20260920
    uv run python experiments/staged_cycles.py train  --run-dir checkpoints/staged-cycles-20260920
    uv run python experiments/staged_cycles.py cycles --run-dir checkpoints/staged-cycles-20260920
    uv run python experiments/staged_cycles.py report --run-dir ... --out reports/...
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
from experiments.staged_arms import (  # noqa: E402
    balanced,
    by_task,
    treated,
    with_program,
)
from experiments.witness_trajectory_shape import (  # noqa: E402
    closed,
    replay,
    run_lengths,
)
from wandering_light.stack_beam import beam_solve, step_all_quiet  # noqa: E402
from wandering_light.stack_machine import (  # noqa: E402
    Config,
    format_stack,
    parse_stack,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    TARGET_CAP,
    PolicyScorer,
    Row,
    encode_item,
    encode_stack,
    item_array,
    load_policy,
    train_policy,
    validation_skeleton,
    witness_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_beam import Product
    from wandering_light.stack_pbe import Task

SIDE_FILE = REPO / "experiments" / "staged_witnesses" / "v4.jsonl"
ARMS = ("staged-subgoal", "fused-subgoal")
#: Which witness each arm's policy is asked to re-write one cycle of.
WITNESS_OF = {"staged-subgoal": "staged", "fused-subgoal": "fused"}
SEEDS = (0, 1, 2)
EPOCHS = 3  # `staged_arms`' fixed epoch, from epoch-selection-in-behaviours (#154)
TASKS_PER_BEHAVIOUR = 5
#: Ops the beam may spend beyond the cycle's own length. Small on purpose: the
#: question is whether the policy writes *this* cycle, not whether some longer
#: detour reaches the same state.
DEPTH_SLACK = 4
#: Cycle lengths, in ops. `1` is a single direct op or one combinator.
CYCLE_BUCKETS = ((1, 1), (2, 5), (6, 12), (13, 10**9))


# ---------------------------------------------------------------------------
# Closed states and the cycles between them
# ---------------------------------------------------------------------------


def closed_product(product: Product) -> bool:
    """A state a sub-goal may name: every example's stack is closed.

    `witness_trajectory_shape.closed` is the per-stack test -- no open ``[`` and
    no finished-but-unexecuted quotation parked on top. A product state is only
    nameable when every example agrees, because the policy is handed one target
    per example and a search must reach all of them at once.
    """
    return all(closed(stack) for stack in product)


def closed_indices(states: Sequence[Product]) -> list[int]:
    """Where along the witness the product state is closed, ``s_0`` included."""
    return [step for step, product in enumerate(states) if closed_product(product)]


def next_closed(states: Sequence[Product]) -> list[int]:
    """``pick[j]``: the nearest closed state strictly after step ``j``.

    The final state is the task's output, which is data, so it is always closed
    and every step therefore has a target; a witness that broke the invariant
    would silently relabel its rows with a partial quotation, so it raises.
    """
    steps = len(states) - 1
    if not closed_product(states[steps]):
        raise ValueError("the final state is not closed")
    picks = [steps] * steps
    nearest = steps
    for step in range(steps - 1, -1, -1):
        picks[step] = nearest
        if closed_product(states[step]):
            nearest = step
    return picks


@dataclasses.dataclass(frozen=True, slots=True)
class Cycle:
    """One transition: the ops between two consecutive closed product states."""

    index: int
    ops: tuple[str, ...]
    kinds: str  # the W / D / X run-lengths of `witness_trajectory_shape`
    start: Product
    target: Product


def cycles(task: Task) -> list[Cycle]:
    """Every transition along ``task.program``, the root inputs counting as one end.

    Nothing is skipped: consecutive closed states are adjacent by construction,
    so the cycles partition the witness and their lengths sum to its length.
    """
    states = hs.replay_states(task)
    kinds = "".join(
        step["kind"]
        for step in replay(task.program, parse_stack(task.inputs[0], basis=BASIS))
    )
    cuts = closed_indices(states)
    return [
        Cycle(
            index,
            tuple(task.program[start:stop]),
            run_lengths(kinds[start:stop]),
            states[start],
            states[stop],
        )
        for index, (start, stop) in enumerate(pairwise(cuts))
    ]


def cycle_task(task: Task, cycle: Cycle) -> Task:
    """``task`` rewritten to ask for one cycle: start stacks in, target stacks out.

    Closed states carry no ``MARK`` and no symbol, so `format_stack` round-trips
    them through `parse_stack` and `beam_solve`'s own close guard accepts them
    as targets. Held-out examples are emptied rather than resampled: a sub-goal
    is a statement about these stacks and there is no reference to draw more
    from, and nothing downstream grades a cycle on them.
    """
    return dataclasses.replace(
        task,
        inputs=tuple(format_stack(Config(stack)) for stack in cycle.start),
        outputs=tuple(format_stack(Config(stack)) for stack in cycle.target),
        held_out_inputs=(),
        held_out_outputs=(),
        held_out=0,
        program=cycle.ops,
        length=len(cycle.ops),
        alternative_programs=(),
    )


def replays_to(program: Sequence[str], start: Product, target: Product) -> bool:
    """Does ``program`` carry every example's start stack onto its target stack?

    The beam's own hit test is ``moved == outputs`` on the product, and
    `step_all_quiet` is its ``step_raw`` loop, so this re-decides the hit on the
    stacks themselves rather than trusting the search that reported it.
    """
    product: Product | None = start
    for name in program:
        op = OP_BY_NAME.get(name)
        if op is None:
            return False
        product, _fuel, _stepped = step_all_quiet(product, op)
        if product is None:
            return False
    return product == target


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def cycle_rows(tasks: Sequence[Task]) -> list[Row]:
    """One row per (task, witness step), its target the next closed state ahead.

    `hindsight_subtargets.subtarget_rows`' row exactly -- same items, same
    label, same ``(task, step, remaining)`` -- with the uniform draw over the
    states ahead replaced by the oracle sub-goal. The row count and the label
    distribution are `witness_rows`', one row per op of the witness.
    """
    rows: list[Row] = []
    for index, task in enumerate(tasks):
        states = hs.replay_states(task)
        picks = next_closed(states)
        for step, name in enumerate(task.program):
            target = [encode_stack(stack, TARGET_CAP) for stack in states[picks[step]]]
            items = [
                encode_item(state, tokens)
                for state, tokens in zip(states[step], target, strict=True)
            ]
            rows.append(
                Row(item_array(items), OP_INDEX[name], index, step, task.length - step)
            )
    return rows


def replayed_along(
    arm: str, tasks: Sequence[Task], side: dict[str, dict[str, Any]]
) -> list[Task]:
    """Each task carrying the witness its arm replays; `task.skeleton` untouched."""
    if arm == "fused-subgoal":
        return list(tasks)
    return [
        with_program(task, side[task.behaviour]["program"])
        if task.behaviour in side
        else task
        for task in tasks
    ]


def arm_rows(
    arm: str, tasks: Sequence[Task], keep_contrived: bool, limit: int = 0
) -> tuple[list[Row], list[Row], dict[str, Any]]:
    """``(fit, validation, facts)``. Validation is the fused holdout in both arms."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {ARMS}")
    side = treated(keep_contrived)
    train = [t for t in tasks if t.split == "train"]
    fit_tasks = [t for t in train if not validation_skeleton(t)]
    held = [t for t in train if validation_skeleton(t)]
    if limit:
        fit_tasks, held = fit_tasks[:limit], held[:limit]
    # The fused arm's row count, which is one row per op of every fused witness.
    # Every arm draws this many, so the arms' optimiser updates match.
    size = sum(len(task.program) for task in fit_tasks)
    swapped = replayed_along(arm, fit_tasks, side)
    groups = [[rows] for rows in by_task(cycle_rows(swapped), len(swapped))]
    facts = {
        "fit_tasks": len(fit_tasks),
        "treated_behaviours": len(
            {t.behaviour for t in fit_tasks if t.behaviour in side}
        ),
        "treated_tasks": sum(t.behaviour in side for t in fit_tasks),
        "distinct_rows_available": sum(len(w) for g in groups for w in g),
        "rows_per_epoch": size,
    }
    return balanced(groups, size), witness_rows(held), facts


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def train(
    run_dir: Path, tasks, device: str, arms, seeds, keep_contrived: bool, limit, log
) -> None:
    for arm in arms:
        if all(hs.policy_dir(run_dir, arm, s, EPOCHS).exists() for s in seeds):
            log(f"skip train {arm} (snapshots exist)")
            continue
        fit, held, facts = arm_rows(arm, tasks, keep_contrived, limit)
        log(f"RESULT rows {arm}: {json.dumps(facts)} validation {len(held)}")
        for seed in seeds:
            if hs.policy_dir(run_dir, arm, seed, EPOCHS).exists():
                log(f"skip train {arm} s{seed} (snapshots exist)")
                continue
            manifest = {
                "arm": arm,
                "policy_arm": hs.ARM,
                "target": "next closed product state",
                "epochs": EPOCHS,
                "learning_rate": hs.LEARNING_RATE,
                "seed": seed,
                "keep_contrived": keep_contrived,
                "side_file": str(SIDE_FILE.relative_to(REPO)),
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
            )
            co.write_json(run_dir / "training" / f"{arm}-s{seed}.json", state)
            curve = " ".join(
                f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}"
                for r in state["history"]
            )
            log(
                f"RESULT train {arm} s{seed}: [{curve}] {time.perf_counter() - started:.0f}s"
            )


def holdout_tasks(
    tasks: Sequence[Task], side: dict[str, dict[str, Any]], per_behaviour: int
) -> list[tuple[int, Task]]:
    """The first ``per_behaviour`` holdout tasks of each staged behaviour.

    The validation-skeleton holdout is what neither arm fitted on, and a
    behaviour is only comparable across the two when it has a staged witness to
    compare the fused one against; 14 of the holdout's 16 behaviours do.
    """
    taken: dict[str, list[tuple[int, Task]]] = defaultdict(list)
    for index, task in enumerate(tasks):
        if (
            task.split == "train"
            and validation_skeleton(task)
            and task.behaviour in side
            and len(taken[task.behaviour]) < per_behaviour
        ):
            taken[task.behaviour].append((index, task))
    return [pair for behaviour in sorted(taken) for pair in taken[behaviour]]


def cell_path(
    run_dir: Path, arm: str, seed: int, device: str, beam: int, per_behaviour: int
) -> Path:
    name = f"{arm}-s{seed}-{device}-b{beam}-t{per_behaviour}.json"
    return run_dir / "cycles" / name


def cycles_cell(
    run_dir: Path,
    tasks,
    arm: str,
    seed: int,
    device: str,
    beam: int,
    per_behaviour: int,
    keep_contrived: bool,
    log,
) -> Path:
    """One (arm, seed) cell: every transition of every chosen witness, solved or not."""
    out = cell_path(run_dir, arm, seed, device, beam, per_behaviour)
    if out.exists():
        log(f"skip cycles {out.name}")
        return out
    if device == "cpu":
        torch.set_num_threads(co.CPU_THREADS)
    side = treated(keep_contrived)
    chosen = holdout_tasks(tasks, side, per_behaviour)
    kind = WITNESS_OF[arm]
    scorer = PolicyScorer(
        load_policy(hs.policy_dir(run_dir, arm, seed, EPOCHS), device=device), device
    )
    rows = []
    gc.freeze()  # see `solve_with_cache.run_cell`
    started = time.perf_counter()
    for index, task in chosen:
        witness = (
            task
            if kind == "fused"
            else with_program(task, side[task.behaviour]["program"])
        )
        for cycle in cycles(witness):
            depth = len(cycle.ops) + DEPTH_SLACK
            result = beam_solve(
                cycle_task(witness, cycle),
                scorer,
                beam_width=beam,
                max_depth=depth,
                guard=co.GUARD,
            )
            solved = bool(result.program) and replays_to(
                result.program, cycle.start, cycle.target
            )
            rows.append(
                {
                    "behaviour": task.behaviour,
                    "task_index": index,
                    "witness": kind,
                    "transition": cycle.index,
                    "ops": len(cycle.ops),
                    "kinds": cycle.kinds,
                    "cycle": list(cycle.ops),
                    "max_depth": depth,
                    "solved": solved,
                    "program": list(result.program) if result.program else None,
                    "termination": result.termination,
                    "executions": result.executions,
                    "seconds": result.seconds,
                }
            )
    cell_seconds = time.perf_counter() - started
    co.write_json(
        out,
        {
            "arm": arm,
            "witness": kind,
            "policy_arm": hs.ARM,
            "seed": seed,
            "device": device,
            "policy_epoch": EPOCHS,
            "beam_width": beam,
            "depth_slack": DEPTH_SLACK,
            "guard": co.GUARD,
            "tasks_per_behaviour": per_behaviour,
            "witnesses": len(chosen),
            "keep_contrived": keep_contrived,
            "cpu_threads": co.CPU_THREADS,
            "eval_corpus": co.primary_corpus(),
            "host": co.host(),
            "provenance": co.provenance(),
            "cell_seconds": cell_seconds,
            "rows": rows,
        },
    )
    solved = sum(r["solved"] for r in rows)
    log(
        f"RESULT cycles {arm} s{seed} beam {beam}: {solved}/{len(rows)} transitions "
        f"({solved / max(1, len(rows)):.3f}), {cell_seconds:.0f}s"
    )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def cycle_bucket(ops: int) -> str:
    for low, high in CYCLE_BUCKETS:
        if low <= ops <= high:
            if high >= 10**9:
                return f"{low}+"
            return str(low) if low == high else f"{low}-{high}"
    raise ValueError(ops)


BUCKET_LABELS = tuple(cycle_bucket(low) for low, _ in CYCLE_BUCKETS)


def share(rows: Sequence[dict[str, Any]]) -> float | None:
    """Solved share, or None when the selection is empty."""
    return sum(r["solved"] for r in rows) / len(rows) if rows else None


def mean_share(cells: Sequence[dict[str, Any]], pick) -> float | None:
    """Mean over cells of each cell's solved share on the rows ``pick`` keeps."""
    kept = [share([r for r in c["rows"] if pick(r)]) for c in cells]
    kept = [v for v in kept if v is not None]
    return mean(kept) if kept else None


def whole_witness_share(rows: Sequence[dict[str, Any]]) -> float | None:
    """Share of witnesses whose every transition the beam wrote."""
    per: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for row in rows:
        per[row["behaviour"], row["task_index"]].append(row["solved"])
    if not per:
        return None
    return sum(all(v) for v in per.values()) / len(per)


def cell(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def report(run_dir: Path, out_dir: Path, log) -> Path:
    cells = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "cycles").glob("*.json"))
    ]
    if not cells:
        raise FileNotFoundError(f"no cycle cells under {run_dir / 'cycles'}")
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_kind[c["witness"]].append(c)
    beams = sorted({c["beam_width"] for c in cells})
    lines = [
        "# staged-cycles: one cycle to the next closed state as target",
        "",
        f"Beam {', '.join(map(str, beams))}, depth = cycle length + {DEPTH_SLACK}, "
        f"guard `{cells[0]['guard']}`, policy epoch {EPOCHS}. A transition is solved "
        "when the returned program replays the start stacks onto the target stacks "
        "on every shown example. Shares are the mean over seeds.",
        "",
        "**Read the buckets, not the overall share.** `beam_solve` tests the hit "
        "while it expands, before the policy ranks anything, so every one-op "
        "transition is solved by exhaustive one-step search whatever the policy "
        "says. The two witness kinds have very different shares of one-op cycles, "
        "so the overall column mixes that in; `2+` is the same share over the "
        "cycles the policy actually has to write.",
        "",
        "| witness | seeds | witnesses | transitions | solved | share | 2+ | "
        + " | ".join(BUCKET_LABELS)
        + " | whole witnesses |",
        "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * (len(BUCKET_LABELS) + 1),
    ]
    for kind in sorted(by_kind):
        group = by_kind[kind]
        counts = [len(c["rows"]) for c in group]
        solved = [sum(r["solved"] for r in c["rows"]) for c in group]
        buckets = [
            cell(
                mean_share(
                    group, lambda r, label=label: cycle_bucket(r["ops"]) == label
                )
            )
            for label in BUCKET_LABELS
        ]
        whole = [whole_witness_share(c["rows"]) for c in group]
        lines.append(
            f"| {kind} | {len(group)} | {mean(c['witnesses'] for c in group):.0f} | "
            f"{mean(counts):.0f} | {mean(solved):.1f} | "
            f"{cell(mean_share(group, lambda r: True))} | "
            f"{cell(mean_share(group, lambda r: r['ops'] > 1))} | "
            + " | ".join(buckets)
            + f" | {cell(mean(w for w in whole if w is not None))} |"
        )
    lines += [
        "",
        "Per cell:",
        "",
        "| arm | seed | transitions | solved | share | 2+ | whole witnesses "
        "| wall (s) | executions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(cells, key=lambda c: (c["arm"], c["seed"])):
        rows = c["rows"]
        lines.append(
            f"| `{c['arm']}` | {c['seed']} | {len(rows)} | "
            f"{sum(r['solved'] for r in rows)} | {cell(share(rows))} | "
            f"{cell(share([r for r in rows if r['ops'] > 1]))} | "
            f"{cell(whole_witness_share(rows))} | {c['cell_seconds']:.0f} | "
            f"{sum(r['executions'] for r in rows):,} |"
        )
    kinds = sorted(by_kind)
    lines += [
        "",
        "By behaviour, transitions and solved share (all cycles, then 2+ ops):",
        "",
        "| behaviour | " + " | ".join(f"{k} n / share / 2+" for k in kinds) + " |",
        "|---|" + "---:|" * len(kinds),
    ]
    behaviours = sorted({r["behaviour"] for c in cells for r in c["rows"]})
    for behaviour in behaviours:
        cols = []
        for kind in kinds:
            group = by_kind[kind]

            def mine(row, behaviour=behaviour):
                return row["behaviour"] == behaviour

            count = mean(sum(mine(r) for r in c["rows"]) for c in group)
            cols.append(
                f"{count:.0f} / {cell(mean_share(group, mine))} / "
                f"{cell(mean_share(group, lambda r: mine(r) and r['ops'] > 1))}"
            )
        lines.append(f"| `{behaviour}` | " + " | ".join(cols) + " |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cycles.md"
    path.write_text("\n".join(lines) + "\n")
    log(f"RESULT report {path}")
    print("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("rows", "train", "cycles", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--tasks-per-behaviour", type=int, default=TASKS_PER_BEHAVIOUR)
    parser.add_argument("--beam", type=int, default=co.BEAM_WIDTH)
    parser.add_argument("--keep-contrived", action="store_true")
    # Smoke tests only: cut the fit and holdout task lists down so `rows` runs
    # in seconds. Never use it for a cell anyone reads.
    parser.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
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
        return 0
    tasks = co.corpus()
    if args.stage == "rows":
        for arm in arms:
            fit, held, facts = arm_rows(arm, tasks, args.keep_contrived, args.limit)
            labels = defaultdict(int)
            for row in fit:
                labels[row.label] += 1
            log(
                f"RESULT rows {arm}: {json.dumps(facts)} "
                f"validation {len(held)} labels {len(labels)}"
            )
    elif args.stage == "train":
        train(
            args.run_dir,
            tasks,
            args.device,
            arms,
            seeds,
            args.keep_contrived,
            args.limit,
            log,
        )
    else:
        for arm in arms:
            for seed in seeds:
                cycles_cell(
                    args.run_dir,
                    tasks,
                    arm,
                    seed,
                    args.device,
                    args.beam,
                    args.tasks_per_behaviour,
                    args.keep_contrived,
                    log,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
