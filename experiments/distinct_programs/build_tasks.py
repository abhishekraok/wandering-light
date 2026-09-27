"""Training tasks for the new behaviours of distinct-programs (#174).

A new behaviour gets the task layout of a v4 train behaviour, so one new
program weighs what one v4 program weighs under the task-uniform draw of
staged-arms (#155): **40 tasks**, each with 3-6 shown examples and 8 held-out
ones drawn without overlap from the behaviour's tests, seeded by its id. Every
row is labelled by running the witness and re-graded with ``grade``.

Kept: behaviours the audit kept, whose witness verified, and whose witness
skeleton is not the skeleton of any v4 eval task (that would put an eval
program's shape into training).

    uv run python experiments/distinct_programs/build_tasks.py \
        --behaviours B.jsonl --audit audit.jsonl --programs 'programs_*.jsonl' \
        --out tasks.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from experiments.distinct_programs.check import assess
from experiments.world_behaviours.build_corpus import label, summary_of
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.stack_pbe import Task, assign_task_hash_split, load, skeleton
from wandering_light.stack_pbe_eval import grade

TASKS_PER_BEHAVIOUR = 40  # every v4 train behaviour has 40
SHOWN = (3, 6)
HELD_OUT = 8
V4 = Path("wandering_light/evals/data/stack_pbe_v4/manifest.json")


def distinct_tests(behaviour: dict) -> list[dict]:
    seen, out = set(), []
    for test in behaviour["tests"]:
        key = json.dumps(test["args"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(test)
    return out


def tasks_for(behaviour: dict, program: tuple[str, ...], verdict: str) -> list[Task]:
    tests = distinct_tests(behaviour)
    rng = random.Random(f"distinct-programs:{behaviour['id']}")
    out = []
    for _ in range(TASKS_PER_BEHAVIOUR):
        drawn = rng.sample(tests, min(len(tests), SHOWN[1] + HELD_OUT))
        k = rng.randint(*SHOWN)
        shown, held = drawn[:k], drawn[k : k + HELD_OUT]
        inputs, outputs, fuel = label(program, shown)
        hidden_in, hidden_out, _ = label(program, held)
        task = Task(
            behaviour=behaviour["id"],
            summary=summary_of(behaviour),
            split="train",
            task_hash_split=assign_task_hash_split(inputs, outputs),
            inputs=inputs,
            outputs=outputs,
            held_out_inputs=hidden_in,
            held_out_outputs=hidden_out,
            program=program,
            length=len(program),
            examples=len(shown),
            held_out=len(held),
            fuel=fuel,
            tags=(behaviour["source"], f"witness:{verdict}"),
            skeleton=skeleton(program),
            witness_source="authored",
        )
        if not grade(task.program, task).solved:
            raise ValueError(f"{behaviour['id']}: witness does not grade as solved")
        out.append(task)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--behaviours", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--programs", required=True, help="glob of programs_*.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    behaviours = {
        (b := json.loads(x))["id"]: b
        for x in args.behaviours.read_text().splitlines()
        if x
    }
    kept = {
        v["id"]
        for v in map(json.loads, args.audit.read_text().splitlines())
        if v["verdict"] == "keep"
    }
    eval_skeletons = {t.skeleton for t in load(fetch_corpus(V4)) if t.split == "eval"}
    counts = {
        "tasks": 0,
        "behaviours": 0,
        "unsolved": 0,
        "unverified": 0,
        "eval_skeleton": 0,
        "not_kept": 0,
    }
    verdicts: dict[str, int] = {}
    with args.out.open("w") as handle:
        for path in sorted(glob.glob(args.programs)):
            for row in map(json.loads, Path(path).read_text().splitlines()):
                if row["id"] not in kept:
                    counts["not_kept"] += 1
                    continue
                if not row.get("program"):
                    counts["unsolved"] += 1
                    continue
                result = assess(behaviours[row["id"]], row["program"])
                if not result["verified"]:
                    counts["unverified"] += 1
                    continue
                program = tuple(row["program"].split())
                if skeleton(program) in eval_skeletons:
                    counts["eval_skeleton"] += 1
                    continue
                verdicts[result["verdict"]] = verdicts.get(result["verdict"], 0) + 1
                for task in tasks_for(
                    behaviours[row["id"]], program, result["verdict"]
                ):
                    handle.write(json.dumps(asdict(task)) + "\n")
                    counts["tasks"] += 1
                counts["behaviours"] += 1
    print(json.dumps({**counts, "verdicts": verdicts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
