"""First baselines on the stack-PBE corpus, graded on the held-out half.

A solver sees only the shown examples. Whatever it returns is graded by
``wandering_light.stack_pbe_eval.grade`` on both halves, so the report states
the solve rate and how far a shown-only grader would have overstated it.

    uv run python experiments/stack_pbe_baseline.py --corpus corpus.jsonl.gz \\
        --solver bfs --budget 20000
    uv run python experiments/stack_pbe_baseline.py --corpus corpus.jsonl.gz \\
        --solver llm --model claude-opus-5 --effort high
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Op,
    ops_for_basis,
    step_raw,
)
from wandering_light.stack_pbe_eval import grade, read, stack_key, summarise

if TYPE_CHECKING:
    from wandering_light.stack_pbe import Task

# Product state: one (stack, fuel spent) per shown input.
State = tuple[tuple, ...]


def _start(task: Task) -> tuple[tuple, ...]:
    return tuple(read(text, task.basis) for text in task.inputs)


def _goal(task: Task) -> tuple:
    return tuple(stack_key(read(text, task.basis)) for text in task.outputs)


def _advance(
    op: Op, stacks: tuple[tuple, ...], fuels: tuple[int, ...]
) -> tuple[tuple[tuple, ...], tuple[int, ...]] | None:
    """Apply ``op`` to every shown stack, or None if any of them refuses."""
    new_stacks, new_fuels = [], []
    for stack, fuel in zip(stacks, fuels, strict=True):
        try:
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - fuel, 0)
        except (RecursionError, ArithmeticError, MemoryError):
            return None
        if isinstance(outcome, str):
            return None
        new_stacks.append(outcome[0])
        new_fuels.append(fuel + outcome[1])
    return tuple(new_stacks), tuple(new_fuels)


def _key(stacks: tuple[tuple, ...]) -> tuple:
    return tuple(stack_key(stack) for stack in stacks)


def bfs_solve(task: Task, budget: int, max_depth: int = 8) -> tuple[str, ...] | None:
    """Breadth-first over the shown stacks as one product state.

    Ops are tried in basis order and states are deduplicated on the stacks
    alone, so the first hit is a shortest program fitting the shown examples.
    Returns None once more than ``budget`` nodes are expanded or the depth
    bound is exhausted.
    """
    ops = ops_for_basis(task.basis)
    goal = _goal(task)
    stacks = _start(task)
    if _key(stacks) == goal:
        return ()
    frontier = deque([(stacks, (0,) * len(stacks), ())])
    seen = {_key(stacks)}
    expanded = 0
    while frontier:
        stacks, fuels, program = frontier.popleft()
        if len(program) >= max_depth:
            continue
        expanded += 1
        if expanded > budget:
            return None
        for op in ops:
            moved = _advance(op, stacks, fuels)
            if moved is None:
                continue
            key = _key(moved[0])
            if key in seen:
                continue
            child = (*program, op.name)
            if key == goal:
                return child
            seen.add(key)
            frontier.append((*moved, child))
    return None


def random_solve(
    task: Task, budget: int, rng: random.Random, max_length: int = 8
) -> tuple[str, ...] | None:
    """Up to ``budget`` random walks over ops legal on every shown stack."""
    ops = ops_for_basis(task.basis)
    goal = _goal(task)
    start = _start(task)
    for _ in range(budget):
        stacks, fuels, program = start, (0,) * len(start), ()
        for _ in range(max_length):
            options = [
                (op, moved)
                for op in ops
                if (moved := _advance(op, stacks, fuels)) is not None
            ]
            if not options:
                break
            op, (stacks, fuels) = rng.choice(options)
            program = (*program, op.name)
            if _key(stacks) == goal:
                return program
    return None


def llm_prompt(task: Task) -> str:
    """Everything a model needs: the ops, the brackets, the shown pairs."""
    ops = ops_for_basis(task.basis)
    lines = [
        "You are writing a program for a small concatenative stack machine.",
        "A program is a sequence of op names separated by spaces, run left to "
        "right. Stacks are written bottom first, top of stack on the right; "
        "lists are written in parentheses and `nil` is the empty list.",
        "",
        "Ops (stack effect, top on the right):",
    ]
    lines += [f"  {op.name:<7} {op.effect}" for op in ops if op.group != "brackets"]
    lines += [
        "",
        "Quotations: `[` opens a quotation and `]` closes it. Ops between the "
        "brackets are not run; they are collected, in order, into a list that "
        "is pushed as one value, e.g. `[ 1 add ]` pushes the quotation "
        "(1 add). Brackets nest. `i`, `dip`, `map`, `filter`, `fold` and "
        "`ifte` run quotations; a `map` or `filter` body sees only the item, a "
        "`fold` body sees (acc item), and each must leave exactly one value "
        "(a boolean for `filter`). Integer literals other than 0 and 1 must be "
        "built, e.g. `1 1 add`.",
        "",
        "Find one program that turns every input stack into its output stack. "
        "It will also be checked on inputs you are not shown, so write the "
        "general rule, not a special case of these examples.",
        "",
        "Examples:",
    ]
    lines += [
        f"  {given} -> {wanted}"
        for given, wanted in zip(task.inputs, task.outputs, strict=True)
    ]
    lines += ["", "Answer with the program alone on the last line."]
    return "\n".join(lines)


_FENCE = re.compile(r"^\s*```[\w-]*\s*$")


def parse_response(text: str) -> str:
    """The program a model's reply ends with, untouched beyond spacing.

    Unknown tokens are kept: the grader, not the parser, decides what counts.
    """
    from wandering_light.solver import remove_thinking

    if not text:
        return ""
    body = remove_thinking(text)
    lines = [
        line.strip().strip("`").strip()
        for line in body.splitlines()
        if not _FENCE.match(line)
    ]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    spaced = lines[-1].replace("[", " [ ").replace("]", " ] ")
    return " ".join(spaced.split())


def _solve(args, task: Task, rng: random.Random, generator) -> str | tuple | None:
    if args.solver == "bfs":
        return bfs_solve(task, args.budget, args.max_depth)
    if args.solver == "random":
        return random_solve(task, args.budget, rng, args.max_depth)
    return parse_response(generator.generate(llm_prompt(task)))


def main(argv: list[str] | None = None) -> int:
    from wandering_light.stack_pbe import load

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--solver", choices=("bfs", "random", "llm"), default="bfs")
    parser.add_argument("--split", default="eval", help="train, eval or all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--out", default=None, help="write per-task JSONL here")
    args = parser.parse_args(argv)

    tasks = [t for t in load(args.corpus) if args.split in ("all", t.split)]
    tasks = tasks[: args.limit] if args.limit is not None else tasks
    generator = None
    if args.solver == "llm":
        from wandering_light.solver import AnthropicTokenGenerator

        generator = AnthropicTokenGenerator(model=args.model, effort=args.effort)
    rng = random.Random(args.seed)

    scores, rows = [], []
    for task in tasks:
        found = _solve(args, task, rng, generator)
        score = grade(found if found is not None else (), task)
        scores.append(score)
        rows.append(
            {
                "behaviour": task.behaviour,
                "program": list(score.program),
                "parsed": score.parsed,
                "legal": score.legal,
                "shown": score.shown,
                "held_out": score.held_out,
            }
        )
    report = summarise(scores)
    print(f"{args.solver} on {args.split}: {report}")
    if args.out:
        with Path(args.out).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
