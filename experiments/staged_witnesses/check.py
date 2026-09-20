"""Author and acceptance tool for staged-witness-reauthoring (#155).

    uv run python experiments/staged_witnesses/check.py --pilot
    uv run python experiments/staged_witnesses/check.py --show running_minima
    uv run python experiments/staged_witnesses/check.py --try running_minima "nil swap [ ... ] fold"
    uv run python experiments/staged_witnesses/check.py --file reports/<dir>/programs_a.jsonl

A staged witness is verified against the behaviour's Python reference on fresh
samples and on every v4 task's shown and held-out pairs, then measured with the
step labels of witness-trajectory-shape (#150) and classified against the
fused witness it replaces.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

from experiments.witness_trajectory_shape import (
    DATA,
    QUOTATIONS,
    atoms,
    closed,
    replay,
    run_lengths,
)
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.stack_machine import Config, format_stack, parse_stack
from wandering_light.stack_pbe import (
    BASIS,
    CATALOGUE,
    _name_seed,
    check,
    load,
    run_program,
    skeleton,
    to_stack,
)

MANIFEST = DATA / "stack_pbe_v4" / "manifest.json"
MAX_BODY = 8  # atoms in any one quotation, nested ones counted
MAX_WRITE_RUN = 12
MAX_PARKED = 4  # top-level steps a finished quotation may wait before it runs
MIN_FUSED = 12  # shorter fused witnesses are "already short" by definition
PILOT_STRATA = (((12, 20), 5), ((21, 40), 5), ((41, 10**9), 10))
PILOT_SEED = 17
BY_NAME = {behaviour.name: behaviour for behaviour in CATALOGUE}


def measure(program: tuple[str, ...], stack) -> dict[str, Any]:
    """The #150 labels plus the three rule quantities of the staging brief."""
    steps, state = replay(program, stack), stack
    kinds = "".join(step["kind"] for step in steps)
    largest, parked, longest_park = 0, 0, 0
    for name, step in zip(program, steps, strict=True):
        if step["kind"] == "X":
            taken = state[-QUOTATIONS.get(name, 1) :]
            largest = max(largest, *(atoms(q) - 1 for q in taken))
        state = step["stack"]
        # Parked: no mark open, yet code is still waiting on the stack.
        waiting = step["kind"] != "W" and not closed(state)
        parked = parked + 1 if waiting else 0
        longest_park = max(longest_park, parked)
    write_runs = [
        len(run) for run in "".join(kinds).replace("D", " ").replace("X", " ").split()
    ]
    executions = [i for i, kind in enumerate(kinds) if kind == "X"]
    return {
        "length": len(program),
        "shape": run_lengths(kinds),
        "executions": len(executions),
        "post_execution_states": sum(closed(steps[i]["stack"]) for i in executions),
        "closed_states": sum(closed(step["stack"]) for step in steps),
        "largest_quotation": largest,
        "longest_write_run": max(write_runs, default=0),
        "longest_park": longest_park,
    }


def within_rules(m: dict[str, Any]) -> list[str]:
    broken = []
    if m["largest_quotation"] > MAX_BODY:
        broken.append(f"a quotation of {m['largest_quotation']} atoms (max {MAX_BODY})")
    if m["longest_write_run"] > MAX_WRITE_RUN:
        broken.append(
            f"a write run of {m['longest_write_run']} steps (max {MAX_WRITE_RUN})"
        )
    if m["longest_park"] > MAX_PARKED:
        broken.append(
            f"a quotation parked for {m['longest_park']} steps (max {MAX_PARKED})"
        )
    return broken


def v4_tasks() -> dict[str, list]:
    tasks: dict[str, list] = {}
    for task in load(fetch_corpus(MANIFEST)):
        tasks.setdefault(task.behaviour, []).append(task)
    return tasks


def assess(
    name: str, text: str, tasks: dict[str, list], eval_skeletons: set[str]
) -> dict[str, Any]:
    behaviour, program = BY_NAME[name], tuple(text.split())
    result: dict[str, Any] = {"behaviour": name, "program": text, "verified": False}
    rng = random.Random(_name_seed(name) + 1)
    for _ in range(500):
        bad = check(behaviour, behaviour.sample(rng), program)
        if bad is not None:
            result["failure"] = (
                f"reference: args {bad.args} want {bad.expected} got {bad.got}"
            )
            return result
    for task in tasks[name]:
        pairs = zip(
            (*task.inputs, *task.held_out_inputs),
            (*task.outputs, *task.held_out_outputs),
            strict=True,
        )
        for given, want in pairs:
            outcome = run_program(program, parse_stack(given, basis=BASIS))
            got = (
                outcome
                if isinstance(outcome, str)
                else format_stack(Config(outcome[0]))
            )
            if got != format_stack(Config(parse_stack(want, basis=BASIS))):
                result["failure"] = f"task pair: {given} want {want} got {got}"
                return result
    first = tasks[name][0]
    root = parse_stack(first.inputs[0], basis=BASIS)
    fused, staged = measure(first.program, root), measure(program, root)
    broken = within_rules(staged)
    gained = staged["post_execution_states"] - fused["post_execution_states"]
    if skeleton(program) in eval_skeletons:
        verdict = "collides with an eval skeleton"
    elif not broken and (gained >= 2 or within_rules(fused)):
        # Within the rules, and either two more post-execution states or the
        # fused witness it replaces was outside them (owner's call, 2026-09-20:
        # a bare +2 bar pushed short witnesses into contrived rewrites).
        verdict = "staged"
    elif fused["length"] < MIN_FUSED:
        verdict = "already short"
    elif broken:
        verdict = "not staged"
    else:
        # Within the rules, under +2, and the fused witness was within the rules
        # too: nothing honest to stage. Untreated, not a failure.
        verdict = "fused already within rules"
    result.update(
        verified=True,
        verdict=verdict,
        broken_rules=broken,
        post_execution_gain=gained,
        length_ratio=round(staged["length"] / fused["length"], 2),
        staged=staged,
        fused=fused,
    )
    return result


def assess_refused(name: str, text: str, tasks: dict[str, list]) -> dict[str, Any]:
    """The re-fused control: same algorithm, cycles merged; only verified and measured."""
    behaviour, program = BY_NAME[name], tuple(text.split())
    rng = random.Random(_name_seed(name) + 2)
    for _ in range(500):
        bad = check(behaviour, behaviour.sample(rng), program)
        if bad is not None:
            return {
                "verified": False,
                "failure": f"args {bad.args} want {bad.expected} got {bad.got}",
            }
    first = tasks[name][0]
    for task in tasks[name]:
        for given, want in zip(
            (*task.inputs, *task.held_out_inputs),
            (*task.outputs, *task.held_out_outputs),
            strict=True,
        ):
            outcome = run_program(program, parse_stack(given, basis=BASIS))
            got = (
                outcome
                if isinstance(outcome, str)
                else format_stack(Config(outcome[0]))
            )
            if got != format_stack(Config(parse_stack(want, basis=BASIS))):
                return {
                    "verified": False,
                    "failure": f"task pair: {given} want {want} got {got}",
                }
    root = parse_stack(first.inputs[0], basis=BASIS)
    return {"verified": True, **measure(program, root)}


def remaining(tasks: dict[str, list]) -> list[str]:
    """Train behaviours of 12+ fused ops outside the pilot, shortest first."""
    done = set(pilot(tasks))
    names = [
        name
        for name, group in tasks.items()
        if group[0].split == "train"
        and group[0].length >= MIN_FUSED
        and name not in done
    ]
    return sorted(names, key=lambda name: (tasks[name][0].length, name))


def pilot(tasks: dict[str, list]) -> list[str]:
    train = sorted(
        (name for name, group in tasks.items() if group[0].split == "train"),
    )
    rng, chosen = random.Random(PILOT_SEED), []
    for (low, high), count in PILOT_STRATA:
        pool = [name for name in train if low <= tasks[name][0].length <= high]
        chosen += rng.sample(pool, count)
    return chosen


def show(name: str, tasks: dict[str, list]) -> None:
    behaviour, first = BY_NAME[name], tasks[name][0]
    print(f"# {name}: {behaviour.summary}\n")
    try:
        print(inspect.getsource(behaviour.reference).strip(), "\n")
    except TypeError:  # a builtin such as `max` has no source
        print(f"reference = {behaviour.reference!r}\n")
    rng = random.Random(0)
    for _ in range(6):
        args = behaviour.sample(rng)
        want = format_stack(Config(to_stack((behaviour.reference(*args),))))
        print(f"  {format_stack(Config(to_stack(args)))}  ->  {want}")
    print(f"\nfused witness ({first.length} ops):\n  {' '.join(first.program)}")
    root = parse_stack(first.inputs[0], basis=BASIS)
    print("fused metrics:", json.dumps(measure(first.program, root)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pilot", action="store_true", help="list the pilot sample")
    parser.add_argument(
        "--remaining", action="store_true", help="list the full-run behaviours"
    )
    parser.add_argument("--show", metavar="BEHAVIOUR")
    parser.add_argument(
        "--try", dest="attempt", nargs=2, metavar=("BEHAVIOUR", "PROGRAM")
    )
    parser.add_argument("--file", type=Path, help="jsonl of {behaviour, program}")
    args = parser.parse_args(argv)
    tasks = v4_tasks()
    eval_skeletons = {
        skeleton(group[0].program)
        for group in tasks.values()
        if group[0].split != "train"
    }
    if args.pilot:
        for name in pilot(tasks):
            print(f"{name}\t{tasks[name][0].length}")
    if args.remaining:
        for name in remaining(tasks):
            print(f"{name}\t{tasks[name][0].length}")
    if args.show:
        show(args.show, tasks)
    if args.attempt:
        print(json.dumps(assess(*args.attempt, tasks, eval_skeletons), indent=1))
    if args.file:
        rows = [
            json.loads(line)
            for line in args.file.read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            out = assess(row["behaviour"], row["program"], tasks, eval_skeletons)
            if row.get("refused_program"):
                out["refused"] = assess_refused(
                    row["behaviour"], row["refused_program"], tasks
                )
            print(json.dumps({**row, **out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
