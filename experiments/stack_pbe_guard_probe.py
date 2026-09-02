"""What the open-quotation guard is worth, measured two ways.

Under an open ``[`` the ``mark-28`` machine captures every op as a symbol
instead of applying it, so every guard returns ``None`` and all 28 ops are
legal. `wandering_light.quotation_guard` reinstates a check at search level.
This measures it.

``lengths`` samples random bodies and reports how much of the space is provably
dead at each body length. It answers the question a depth-bounded search cannot:
a depth-5 budget cannot afford a body longer than about three ops, but
``stack_pbe_v4`` has a median witness of 43.

``states`` runs the real enumeration over corpus tasks and reports states per
task under each arm, which is what a search actually pays.

``witnesses`` replays every authored witness op by op and checks the guard never
fires on a prefix. This is the reachability half of the gate, and it has to be
done this way: a depth-5 sweep only reaches 7 of v3's 40 witnesses, so it cannot
speak for the other 33. Replay reaches all of them, and all 220 of v4's.

    uv run python experiments/stack_pbe_guard_probe.py lengths --max-length 16
    uv run python experiments/stack_pbe_guard_probe.py states --depth 5 --tasks 3
    uv run python experiments/stack_pbe_guard_probe.py witnesses --corpus <path>
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time

from experiments.stack_pbe_search_audit import key_of, stacks, step_all
from wandering_light.quotation_guard import (
    FIXED_SUPPLY,
    LITERALS_BY_NAME,
    blocked,
    body_effect,
)
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    Mark,
    Sym,
    Value,
    format_value,
    ops_for_basis,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, load

ARMS = ("off", "body", "close", "both")


#: A body that finished a nested quotation carries it as data. One such value
#: stands in for that, so the alphabet is not made entirely of symbols.
NESTED = Lst((Sym("dup"), Sym("add")))


def body_alphabet(nested: bool = False) -> tuple[Value, ...]:
    """What one step under an open MARK can append to the innermost body.

    Brackets are excluded, and that is not a simplification. `[` is not
    captured -- it pushes another `Mark`, which sits *below* the innermost body
    rather than in it -- and `]` closes the body, ending the open-body state
    that `blocked` inspects. A real innermost body therefore holds only
    literals, captured op symbols, and any quotation it finished building;
    never `[` or `]`.

    Omitting the finished quotations biases the rate *upward*, since a
    quotation satisfies the checks on `i`, `map`, `cat` and the rest.
    ``--nested`` adds one back to measure how much.
    """
    body_ops = tuple(
        LITERALS_BY_NAME.get(op.name) or Sym(op.name)
        for op in ops_for_basis("mark-28")
        if op.name not in {"[", "]"}
    )
    return (*body_ops, NESTED) if nested else body_ops


def measure_lengths(args) -> dict:
    alpha = body_alphabet(nested=args.nested)
    rng = random.Random(args.seed)
    print(
        f"alphabet {len(alpha)}, supply {args.supply}, "
        f"{args.samples:,} random bodies per length\n"
    )
    print(f"{'len':>4}{'pruned':>9}{'type':>9}{'deep':>9}{'surviving b_eff':>17}")
    rows = []
    for length in range(1, args.max_length + 1):
        type_dead = reach_deep = 0
        for _ in range(args.samples):
            body = tuple(rng.choice(alpha) for _ in range(length))
            runnable, reach = body_effect(body)
            if not runnable:
                type_dead += 1
            elif reach > args.supply:
                reach_deep += 1
        dead = type_dead  # only the type check prunes; reach is a diagnostic
        live = 1 - dead / args.samples
        rows.append(
            {
                "length": length,
                "dead": dead / args.samples,
                "type": type_dead / args.samples,
                "reach_deep": reach_deep / args.samples,
                "b_eff": len(alpha) * live ** (1 / length),
            }
        )
        print(
            f"{length:>4}{rows[-1]['dead']:>8.1%}{rows[-1]['type']:>9.1%}"
            f"{rows[-1]['reach_deep']:>9.1%}{rows[-1]['b_eff']:>17.2f}"
        )
    return {"kind": "lengths", "supply": args.supply, "rows": rows}


def walk(inputs, ops, depth: int, arm: str) -> int:
    """States reached under one arm, deduplicated by stack identity.

    ``seen`` holds 64-bit hashes rather than the keys themselves: at depth 6 the
    key set is ~26M nested tuples and does not fit comfortably in RAM. Expected
    collisions at that size are ~1e-5 states, far below the precision of
    anything reported from this.
    """
    seen = {hash(key_of(inputs))}
    frontier = [inputs]
    for step in range(depth):
        remaining = depth - step - 1
        last = step == depth - 1
        nxt = []
        for behaviour in frontier:
            for op in ops:
                moved = step_all(behaviour, op)
                if moved is None:
                    continue
                moved_key = hash(key_of(moved))
                if moved_key in seen:
                    continue
                if blocked(moved[0], remaining, arm) is not None:
                    continue
                seen.add(moved_key)
                if not last:  # the final ply is counted, never expanded
                    nxt.append(moved)
        frontier = nxt
    return len(seen)


def measure_states(args) -> dict:
    ops = ops_for_basis("mark-28")
    tasks = load(args.corpus)
    random.Random(args.seed).shuffle(tasks)
    arms = args.arms.split(",")
    totals: collections.Counter = collections.Counter()
    print(f"depth {args.depth}, {args.tasks} tasks, corpus {args.corpus}\n")
    rows = []
    for task in tasks[: args.tasks]:
        inputs = stacks(task.inputs)
        counts, seconds = {}, {}
        for arm in arms:
            started = time.time()
            counts[arm] = walk(inputs, ops, args.depth, arm)
            seconds[arm] = time.time() - started
            totals[arm] += counts[arm]
            totals[arm + "_s"] += seconds[arm]
        rows.append({"behaviour": task.behaviour, "states": counts, "seconds": seconds})
        print(
            f"{task.behaviour:<18}"
            + "  ".join(
                f"{arm}={counts[arm]:>10,}"
                + (f" ({counts['off'] / counts[arm]:.2f}x)" if arm != "off" else "")
                for arm in arms
            ),
            flush=True,
        )
    print()
    summary = {}
    for arm in arms:
        ratio = totals["off"] / totals[arm] if "off" in arms else 1.0
        summary[arm] = {
            "mean_states": totals[arm] / args.tasks,
            "ratio": ratio,
            "seconds": totals[arm + "_s"],
        }
        print(
            f"{arm:<8} mean states/task {totals[arm] // args.tasks:>11,}"
            f"   {ratio:>6.2f}x   {totals[arm + '_s']:.0f}s"
        )
    return {
        "kind": "states",
        "depth": args.depth,
        "corpus": args.corpus,
        "seed": args.seed,
        "rows": rows,
        "summary": summary,
    }


def measure_witnesses(args) -> dict:
    """Every authored witness, replayed, with the guard consulted at each prefix.

    ``remaining`` is set as if the search had a budget exactly long enough for
    this witness, which is the tightest a sound search could ever set it.
    """
    ops = {op.name: op for op in ops_for_basis(BASIS)}
    by_behaviour: dict = {}
    for task in load(args.corpus):
        by_behaviour.setdefault(task.behaviour, task)
    checked = 0
    lost: list[dict] = []
    for name, task in by_behaviour.items():
        # v3 stores alternatives as strings, v4 as tuples of op names.
        programs = [
            task.program,
            *(
                tuple(p.split()) if isinstance(p, str) else tuple(p)
                for p in task.alternative_programs
            ),
        ]
        for program in programs:
            for source in task.inputs[: args.examples]:
                stack = parse_stack(source, basis=BASIS)
                for index, op_name in enumerate(program):
                    outcome = step_raw(ops[op_name], stack, DEFAULT_FUEL_LIMIT, 0)
                    if isinstance(outcome, str):
                        break
                    stack = outcome[0]
                    if not any(isinstance(value, Mark) for value in stack):
                        continue
                    checked += 1
                    reason = blocked(stack, len(program) - index - 1, args.arm)
                    if reason is not None:
                        lost.append(
                            {
                                "behaviour": name,
                                "program": " ".join(program),
                                "prefix": " ".join(program[: index + 1]),
                                "stack": " ".join(format_value(v) for v in stack),
                                "reason": reason,
                            }
                        )
    print(f"corpus {args.corpus}, arm {args.arm}")
    print(
        f"behaviours {len(by_behaviour)}, open-mark prefix states checked {checked:,}"
    )
    print(f"witness prefixes the guard would prune: {len(lost)}")
    for row in lost[:10]:
        print(f"  {row['reason']:<6} {row['behaviour']}: {row['prefix']}")
    return {
        "kind": "witnesses",
        "corpus": args.corpus,
        "arm": args.arm,
        "behaviours": len(by_behaviour),
        "checked": checked,
        "lost": lost,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    sub = parser.add_subparsers(dest="mode", required=True)

    lengths = sub.add_parser("lengths")
    lengths.add_argument("--max-length", type=int, default=16)
    lengths.add_argument("--samples", type=int, default=50_000)
    lengths.add_argument("--supply", type=int, default=FIXED_SUPPLY)
    lengths.add_argument("--seed", type=int, default=17)
    lengths.add_argument("--nested", action="store_true")

    states = sub.add_parser("states")
    states.add_argument("--depth", type=int, default=5)
    states.add_argument("--tasks", type=int, default=3)
    states.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v3")
    states.add_argument("--seed", type=int, default=17)
    states.add_argument("--arms", default=",".join(ARMS))

    witnesses = sub.add_parser("witnesses")
    witnesses.add_argument(
        "--corpus", default="wandering_light/evals/data/stack_pbe_v3"
    )
    witnesses.add_argument("--arm", default="both", choices=ARMS)
    witnesses.add_argument("--examples", type=int, default=3)

    args = parser.parse_args(argv)
    result = {
        "lengths": measure_lengths,
        "states": measure_states,
        "witnesses": measure_witnesses,
    }[args.mode](args)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
