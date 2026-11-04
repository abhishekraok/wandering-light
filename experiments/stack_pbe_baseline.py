"""First baselines on the stack PBE corpus, graded on the held-out examples.

``docs/STATUS.md`` records that no solver has been run on ``stack_pbe_v3``.
This is the runner that changes that. Three solvers, each of which sees only
a task's *shown* examples; ``wandering_light.stack_pbe_eval`` then grades the
answer on the eight hidden ones.

* ``bfs_solve`` -- breadth-first over programs, the shown inputs' stacks
  carried as one product state so that every example moves together. A
  budget in expanded nodes, and ``max_depth`` in ops. This is the auditor
  from ``stack_pbe_search_audit.py`` turned into a solver: it is expected to
  reach the 1-5 op floor and nothing beyond it.
* ``random_solve`` -- random walks over whatever is legal, the floor under
  any search that claims to be directed.
* ``llm_prompt`` / ``parse_response`` -- the text a language model is shown
  and the program read back from its reply. Parsing never repairs: a model
  that invents an op is scored as having invented it.

    uv run python experiments/stack_pbe_baseline.py --solver bfs --tasks 40
    uv run python experiments/stack_pbe_baseline.py --solver anthropic \\
        --model claude-opus-5 --effort high --tasks 40 --out results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from typing import TYPE_CHECKING

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Op,
    Value,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, OPS, load
from wandering_light.stack_pbe_eval import Score, grade, summarise

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Every shown example's stack, side by side: one op moves all of them.
State = tuple[tuple[Value, ...], ...]


def key_of(state: State) -> tuple:
    return tuple(tuple(value_key(v) for v in stack) for stack in state)


def step_all(state: State, op: Op) -> State | None:
    """One op across every example, or None if any of them refuses it."""
    moved = []
    for stack in state:
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None
        moved.append(outcome[0])
    return tuple(moved)


def shown(task) -> tuple[State, tuple]:
    """The solver's whole view of a task: shown inputs, and the target key."""
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    return inputs, key_of(outputs)


def bfs_solve(task, budget: int, max_depth: int = 8) -> tuple[str, ...] | None:
    """The first program, in breadth-first basis order, that fits the shown examples.

    Nodes are product states, deduplicated on their structural key. ``budget``
    bounds the number of nodes *expanded*; once more than that many would be,
    the answer is None.
    """
    start, target = shown(task)
    if key_of(start) == target:
        return ()
    seen = {key_of(start)}
    queue: deque[tuple[State, tuple[str, ...]]] = deque([(start, ())])
    expanded = 0
    while queue:
        if expanded >= budget:
            return None
        state, program = queue.popleft()
        expanded += 1
        if len(program) >= max_depth:
            continue
        for op in OPS:
            moved = step_all(state, op)
            if moved is None:
                continue
            extended = (*program, op.name)
            moved_key = key_of(moved)
            if moved_key == target:
                return extended
            if moved_key in seen:
                continue
            seen.add(moved_key)
            queue.append((moved, extended))
    return None


def legal_ops(state: State) -> list[tuple[Op, State]]:
    """The ops every shown example accepts, with where each one leads."""
    moves = []
    for op in OPS:
        moved = step_all(state, op)
        if moved is not None:
            moves.append((op, moved))
    return moves


def random_solve(
    task, budget: int, rng: random.Random, max_length: int = 8
) -> tuple[str, ...] | None:
    """Up to ``budget`` random walks over legal ops; the first fitting prefix wins."""
    start, target = shown(task)
    for _ in range(budget):
        state, program = start, ()
        for _ in range(max_length):
            moves = legal_ops(state)
            if not moves:
                break
            op, state = rng.choice(moves)
            program = (*program, op.name)
            if key_of(state) == target:
                return program
    return None


def llm_prompt(task) -> str:
    """What a language model is told: the machine, the ops, the shown pairs."""
    ops = "\n".join(f"  {op.name:<8} {op.effect}" for op in OPS)
    pairs = "\n".join(
        f"  {given} -> {wanted}"
        for given, wanted in zip(task.inputs, task.outputs, strict=True)
    )
    return f"""You are writing a program for a small stack machine.

A program is a sequence of ops separated by spaces. Each op pops its inputs
from the top of the stack and pushes its outputs. Stacks are printed bottom to
top as [ a b c ], so c is the top. Lists are printed as ( 1 2 3 ). Each op's
stack effect is written ( inputs -- outputs ) with the top of the stack on the
right. The only literals are 0, 1, nil, true and false; any other number has to
be built with arithmetic. Comparisons and div take their arguments in the order
written: a b lt? is a < b, a b div is a // b.

The ops are:
{ops}

Quotations are built with brackets. The op [ opens a quotation and ] closes
it; every op written between them is captured as code instead of being run,
and the closed quotation is pushed on the stack as a list. Quotations are what
the combinators consume: i runs one, dip runs one under the top item, map,
filter and fold apply one to each element of a list, and ifte takes a
condition, a then-branch and an else-branch. For example `[ dup add ] map`
doubles every element of a list, and `0 [ add ] fold` sums it.

Here are examples of the behaviour to implement, as input stack -> output
stack. The program must turn every input into exactly its output, and it must
do the same for other inputs of the same kind that are not shown here.

{pairs}

Reply with the program on the last line of your answer, ops separated by
spaces, and nothing else on that line."""


def _is_fence(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") and " " not in stripped


def parse_response(text: str) -> str:
    """The program in a model's reply, taken as written.

    Thinking and code fences are stripped, the last non-empty line is taken,
    backticks are removed and brackets spaced out so ``[add]`` reads
    ``[ add ]``. Unknown tokens are kept: the grader is what decides whether a
    token is an op, and a parser that repaired them would hide that.
    """
    from wandering_light.solver import remove_thinking

    lines = [line for line in remove_thinking(text).splitlines() if not _is_fence(line)]
    kept = [line for line in lines if line.strip()]
    if not kept:
        return ""
    last = kept[-1].replace("`", "")
    spaced = last.replace("[", " [ ").replace("]", " ] ")
    return " ".join(spaced.split())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _bfs(budget: int, max_depth: int):
    return lambda task: bfs_solve(task, budget, max_depth=max_depth)


def _random(budget: int, max_length: int, seed: int):
    rng = random.Random(seed)
    return lambda task: random_solve(task, budget, rng, max_length=max_length)


def _anthropic(model: str, effort: str):
    from wandering_light.solver import AnthropicTokenGenerator

    generator = AnthropicTokenGenerator(model=model, effort=effort)
    return lambda task: parse_response(generator.generate(llm_prompt(task)))


def verdict(score: Score) -> str:
    if score.solved:
        return "solved"
    if score.memorised:
        return "memorised"
    if not score.parsed:
        return "unparsed"
    if not score.legal:
        return "illegal"
    return "wrong"


def evaluate(tasks: Sequence, solve: Callable, *, log=None) -> list[tuple[Score, dict]]:
    """Solve each task from its shown half, grade on both halves, keep a row."""
    results = []
    for task in tasks:
        started = time.time()
        candidate = solve(task)
        score = grade(candidate if candidate is not None else (), task)
        row = {
            "behaviour": task.behaviour,
            "split": task.split,
            "witness": " ".join(task.program),
            "program": " ".join(score.program),
            "verdict": verdict(score),
            "parsed": score.parsed,
            "legal": score.legal,
            "shown": score.shown,
            "held_out": score.held_out,
            "seconds": round(time.time() - started, 3),
        }
        results.append((score, row))
        if log is not None:
            print(f"{task.behaviour:<18}{row['verdict']:<10}{row['program']}", file=log)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", default="wandering_light/evals/data/stack_pbe_v3.jsonl.gz"
    )
    parser.add_argument(
        "--solver", choices=["bfs", "random", "anthropic"], default="bfs"
    )
    parser.add_argument("--split", default="eval", help="train, eval or all")
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    tasks = load(args.corpus)
    if args.split != "all":
        tasks = [task for task in tasks if task.split == args.split]
    random.Random(args.seed).shuffle(tasks)
    sample = tasks[: args.tasks]

    if args.solver == "bfs":
        solve = _bfs(args.budget, args.max_depth)
    elif args.solver == "random":
        solve = _random(args.budget, args.max_depth, args.seed)
    else:
        solve = _anthropic(args.model, args.effort)

    started = time.time()
    print(f"{args.solver}: {len(sample)} {args.split} tasks, budget {args.budget}\n")
    results = evaluate(sample, solve, log=sys.stdout)
    report = summarise(score for score, _row in results)
    rows = [row for _score, row in results]
    print(f"\n{report}")
    print(f"{report.illegal} illegal, {time.time() - started:.0f}s")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "solver": args.solver,
                    "split": args.split,
                    "budget": args.budget,
                    "max_depth": args.max_depth,
                    "model": args.model if args.solver == "anthropic" else None,
                    "effort": args.effort if args.solver == "anthropic" else None,
                    "report": {
                        "tasks": report.tasks,
                        "solved": report.solved,
                        "memorised": report.memorised,
                        "shown_fit": report.shown_fit,
                        "illegal": report.illegal,
                        "solve_rate": report.solve_rate,
                        "shown_rate": report.shown_rate,
                        "overstatement": report.overstatement,
                    },
                    "rows": rows,
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
