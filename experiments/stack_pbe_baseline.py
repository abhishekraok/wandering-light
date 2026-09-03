"""First solver numbers on the behaviour-first corpus.

The protocol is the point. A solver sees **only the shown examples** of a task.
Whatever it returns is then graded on that task's **held-out** examples, which
it never saw. Two numbers come out:

* `solved` -- right on data it was not given. The only number worth quoting.
* `shown-only` -- what a grader without held-out examples would have reported.

The gap between them is how much the old protocol would have flattered the
solver, and it is the whole reason `stack_pbe_v3` carries hidden examples.
`stack_pbe_v2` could not measure it: its data came from a seventeen-row table,
so there was nothing left to hold out.

Backends:

* `bfs` -- breadth-first over the product behaviour of the shown examples,
  under a node budget, stopping at the first program that fits them. This is
  the strongest thing that needs no model, and #25 asks for baselines quoted
  against a stated budget.
* `random` -- uniform random legal programs. The chance floor.
* `llm` -- a language model, through the token generators in
  `wandering_light/solver.py`. Needs credentials.

    uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 10000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import Counter

from wandering_light.quotation_guard import blocked
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Mark,
    ops_for_basis,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import BASIS, Task, load
from wandering_light.stack_pbe_eval import Score, best, summarise

OPS = ops_for_basis(BASIS)


def key_of(behaviour) -> tuple:
    return tuple(tuple(value_key(v) for v in stack) for stack in behaviour)


def step_all(behaviour, op):
    moved = []
    for stack in behaviour:
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            return None
        moved.append(outcome[0])
    return tuple(moved)


def bfs_solve(
    task: Task, budget: int, max_depth: int = 8, guard: str = "off"
) -> tuple[str, ...] | None:
    """Shortest program fitting the *shown* examples, within a node budget.

    Nothing here looks at the held-out examples; that is what makes the score
    downstream mean something.

    `guard` is the search-side predicate from #70 over open-quotation states:
    "off" is the machine as it stands, where capture makes every op legal
    under an open `[`; "body", "close" or "both" drop successors whose open
    body can never run. It is sound -- no witness prefix is pruned -- so it
    changes what a node budget *reaches*, never which programs exist at a
    depth. "off" reproduces every unpruned number.
    """
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    target = key_of(outputs)
    # "close" is sound only because every target is mark-free: a state with
    # more open marks than ops left can never become one. A corpus that hands
    # the solver a quotation in its data would break this.
    if guard in ("close", "both") and any(
        isinstance(v, Mark) for stack in outputs for v in stack
    ):
        raise ValueError("the close guard needs mark-free targets")
    seen = {key_of(inputs)}
    frontier = [(inputs, ())]
    expanded = 0
    for depth in range(max_depth):
        # Ops a successor may still spend under the depth cap. The cap, not
        # the node budget, is what makes "close" and "reach" sound.
        remaining = max_depth - depth - 1
        nxt = []
        for behaviour, program in frontier:
            expanded += 1
            if expanded > budget:
                return None
            for op in OPS:
                moved = step_all(behaviour, op)
                if moved is None:
                    continue
                extended = (*program, op.name)
                moved_key = key_of(moved)
                if moved_key == target:
                    return extended
                # Every example shares the mark structure, so one stack is
                # enough to judge the open body.
                if guard != "off" and blocked(moved[0], remaining, guard) is not None:
                    continue
                if moved_key in seen:
                    continue
                seen.add(moved_key)
                nxt.append((moved, extended))
        if not nxt:
            break
        frontier = nxt
    return None


def random_solve(
    task: Task, budget: int, rng: random.Random, max_length: int = 8
) -> tuple[str, ...] | None:
    """Uniform random legal programs. The chance floor, for scale."""
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    target = key_of(tuple(parse_stack(text, basis=BASIS) for text in task.outputs))
    for _ in range(budget):
        behaviour, program = inputs, ()
        for _step in range(rng.randint(1, max_length)):
            legal = [op for op in OPS if step_all(behaviour, op) is not None]
            if not legal:
                break
            op = rng.choice(legal)
            behaviour = step_all(behaviour, op)
            program = (*program, op.name)
            if key_of(behaviour) == target:
                return program
    return None


def list_models(provider: str) -> int:
    """Ask a provider what it actually serves.

    Model names move faster than anything written down, and a run against a
    guessed name either errors or silently hits a different model than the one
    reported. Ask, then pick.
    """
    if provider == "openai":
        import openai
        from dotenv import load_dotenv

        load_dotenv()
        names = sorted(m.id for m in openai.OpenAI().models.list())
    elif provider == "gemini":
        import google.generativeai as genai
        from dotenv import load_dotenv

        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        names = sorted(
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        )
    elif provider == "ollama":
        from ollama import Client

        names = sorted(m["model"] for m in Client().list()["models"])
    else:
        import anthropic

        names = sorted(m.id for m in anthropic.Anthropic().models.list())
    for name in names:
        print(f"  {name}")
    print(f"{len(names)} models on {provider}")
    return 0


def op_reference() -> str:
    return "\n".join(f"  {op.name:<8} {op.effect}" for op in OPS)


def llm_prompt(task: Task) -> str:
    pairs = "\n".join(
        f"  {given}  ->  {wanted}"
        for given, wanted in zip(task.inputs, task.outputs, strict=True)
    )
    return f"""You are writing a program for a concatenative stack machine.

The ops available, with their stack effects:
{op_reference()}

`[` and `]` build a quotation: `[ dup add ]` pushes the program `(dup add)`,
which `map`, `filter`, `fold` and `i` can then run. Only `0` and `1` are
literals; any other number must be computed. `map` runs its body on one element
at a time and the body must leave exactly one value. `fold` runs its body on
(accumulator, element).

Find one program that turns every input stack below into its output stack.

{pairs}

Answer with the program only: op names separated by spaces, nothing else."""


def parse_response(text: str) -> str:
    """Pull a program out of a model's reply.

    Normalisation only -- strip thinking, code fences and surrounding prose,
    and space out brackets, which the corpus format always separates. It does
    not drop unknown tokens: an answer that names an op that does not exist is
    a wrong answer, and `grade` should say so rather than have it quietly
    repaired here.
    """
    from wandering_light.solver import remove_thinking

    body = remove_thinking(text).strip()
    body = body.replace("```", "\n")
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    program = lines[-1].strip("`").strip()
    return program.replace("[", " [ ").replace("]", " ] ").strip()


def solve_with_llm(task: Task, generator, samples: int) -> list[str]:
    """One candidate per sample, with a failed call scored as a failed answer.

    Deliberately broad: the four backends raise unrelated exception types --
    an httpx timeout, a Google 503, an OpenAI rate limit -- and a run over
    hundreds of tasks must not die because one call did. Without this the
    deadline added to bound a runaway generation would itself end the
    evaluation before any summary or output file is written, which is worse
    than the hang it replaced.
    """
    prompt = llm_prompt(task)
    candidates = []
    for _ in range(samples):
        try:
            candidates.append(parse_response(generator.generate(prompt)))
        except Exception as error:  # broad on purpose - see docstring
            print(
                f"  {task.behaviour}: generation failed, scoring as unsolved "
                f"({type(error).__name__}: {str(error)[:80]})",
                file=sys.stderr,
            )
            candidates.append("")
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v4")
    parser.add_argument("--backend", choices=("bfs", "random", "llm"), default="bfs")
    parser.add_argument("--split", default="eval", choices=("eval", "train", "all"))
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument(
        "--guard",
        choices=("off", "body", "close", "both"),
        default="off",
        help="bfs only: the #70 open-quotation guard; off reproduces unpruned numbers",
    )
    parser.add_argument("--tasks", type=int, default=0, help="0 means every task")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini", "ollama", "anthropic"),
        default="anthropic",
    )
    parser.add_argument(
        "--effort",
        default="high",
        help="reasoning effort, for providers and models that accept one",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=300.0,
        help="ollama only: per-request deadline, so a runaway answer cannot stall a run",
    )
    parser.add_argument(
        "--ollama-think",
        action="store_true",
        help="ollama only: pass --effort as the model's think level (gpt-oss grades it)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="ask --provider what it serves, then exit",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.list_models:
        return list_models(args.provider)

    tasks = load(args.corpus)
    if args.split != "all":
        tasks = [t for t in tasks if t.split == args.split]
    rng = random.Random(args.seed)
    if args.tasks:
        rng.shuffle(tasks)
        tasks = tasks[: args.tasks]

    generator = None
    if args.backend == "llm":
        import wandering_light.solver as solver_module

        builders = {
            "openai": lambda: solver_module.OpenAITokenGenerator(
                args.model, effort=args.effort
            ),
            "gemini": lambda: solver_module.GeminiTokenGenerator(args.model),
            "ollama": lambda: solver_module.OllamaTokenGenerator(
                args.model,
                timeout=args.ollama_timeout,
                think=args.effort if args.ollama_think else None,
            ),
            "anthropic": lambda: solver_module.AnthropicTokenGenerator(
                args.model, effort=args.effort
            ),
        }
        generator = builders[args.provider]()

    started = time.time()
    scores: list[Score] = []
    planned = len(tasks)

    # SIGTERM (a `timeout`, a scheduler) becomes KeyboardInterrupt so a run cut
    # short still summarises and writes what it scored, marked partial.
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index, task in enumerate(tasks, 1):
            if args.backend == "bfs":
                found = bfs_solve(task, args.budget, guard=args.guard)
                candidates = [found] if found else []
            elif args.backend == "random":
                found = random_solve(task, args.budget, rng)
                candidates = [found] if found else []
            else:
                candidates = solve_with_llm(task, generator, args.samples)
            # `best` rather than the first candidate: with more than one sample a
            # run that took candidate zero would report pass@1 while paying for k.
            scores.append(
                best(candidates, task)
                if candidates
                else Score((), parsed=False, legal=False, shown=False, held_out=False)
            )
            print(
                f"  [{index}/{planned}] {task.behaviour} ({task.length} ops): "
                f"{'solved' if scores[-1].solved else 'memorised' if scores[-1].memorised else 'failed'}, "
                f"{sum(s.solved for s in scores)} solved so far, "
                f"{time.time() - started:.0f}s",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print(f"\ninterrupted after {len(scores)} of {planned} tasks", file=sys.stderr)
    partial = len(scores) < planned
    tasks = tasks[: len(scores)]
    if not scores:
        return 1

    report = summarise(scores)
    print(
        f"{args.backend} on {args.split}, budget {args.budget:,}"
        f"{', guard ' + args.guard if args.backend == 'bfs' and args.guard != 'off' else ''}: "
        f"{len(tasks)}{' of ' + str(planned) + ' planned (PARTIAL)' if partial else ''} tasks, "
        f"{time.time() - started:.0f}s"
    )
    print(f"  {report}")

    by_behaviour: dict[str, list[Score]] = {}
    for task, score in zip(tasks, scores, strict=True):
        by_behaviour.setdefault(task.behaviour, []).append(score)
    print(f"\n  {'behaviour':<18}{'wit':>4}{'solved':>9}{'memorised':>11}")
    lengths = {t.behaviour: t.length for t in tasks}
    for name in sorted(by_behaviour, key=lambda n: lengths[n]):
        group = by_behaviour[name]
        solved = sum(s.solved for s in group)
        memo = sum(s.memorised for s in group)
        print(f"  {name:<18}{lengths[name]:>4}{solved:>5}/{len(group):<3}{memo:>11}")

    if args.out:
        payload = {
            "backend": args.backend,
            "split": args.split,
            "budget": args.budget,
            "guard": args.guard,
            "partial": partial,
            "tasks_planned": planned,
            "tasks": report.tasks,
            "solved": report.solved,
            "memorised": report.memorised,
            "shown_fit": report.shown_fit,
            "illegal": report.illegal,
            "solve_rate": report.solve_rate,
            "shown_rate": report.shown_rate,
            "by_behaviour": {
                name: {
                    "witness_length": lengths[name],
                    "tasks": len(group),
                    "solved": sum(s.solved for s in group),
                    "memorised": sum(s.memorised for s in group),
                }
                for name, group in by_behaviour.items()
            },
            "solved_programs": Counter(
                " ".join(s.program) for s in scores if s.solved
            ).most_common(),
            # Every candidate, with its verdict. The memorised ones are the
            # point of the corpus and a count alone cannot show what went
            # wrong -- `uncons drop i` only reads as a near-miss for `last`
            # once you can see it.
            "candidates": [
                {
                    "behaviour": task.behaviour,
                    "witness": " ".join(task.program),
                    "candidate": " ".join(score.program),
                    "verdict": (
                        "solved"
                        if score.solved
                        else "memorised"
                        if score.memorised
                        else "failed"
                    ),
                }
                for task, score in zip(tasks, scores, strict=True)
            ],
        }
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
