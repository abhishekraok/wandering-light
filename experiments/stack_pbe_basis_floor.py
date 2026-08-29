"""Which basis can host a corpus whose inputs carry no quotations?

`stack-corpus-no-higher-order (#59)` measured that the deleted corpus handed a
quotation to the solver in 36.6% of tasks and that 76.8% of those never ran it.
Removing that crutch is the point of `wandering_light/stack_pbe.py`. This
script measures what removing it costs each basis.

The argument to check is structural. No op in `executable-26` puts an op *name*
on the stack, and `quote`/`cons`/`cat` only combine values that are already
there. So from a data-only stack, every reachable quotation is a list of data,
and a data-only body pushes values without consuming any -- which the
combinator guards reject, because `map`, `filter` and `fold` all require a body
that leaves exactly one value (and `filter` a boolean).

Measured rather than asserted: expand the ball around data-only roots under
each basis, and count how many reachable quotations are executable, how often
each combinator is legal, and how often `map` does anything.

    uv run python experiments/stack_pbe_basis_floor.py --depth 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    Op,
    Sym,
    Value,
    format_stack,
    Config,
    ops_for_basis,
    quotation_open,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import CATALOGUE, to_stack

BASES = ("executable-26", "mark-28")
RUNS_CODE = ("i", "dip", "map", "filter", "fold", "ifte")


def roots(count: int, seed: int) -> list[tuple[Value, ...]]:
    """Data-only starting stacks, drawn from the catalogue's own samplers.

    Using the samplers rather than a fixed table is the same discipline the
    generator follows: the measurement should not depend on a hand-picked
    stack any more than a task should.
    """
    rng = random.Random(seed)
    shapes = {}
    for behaviour in CATALOGUE:
        args = behaviour.sample(rng)
        shapes.setdefault(tuple(type(a).__name__ for a in args), args)
    picked = list(shapes.values())[:count]
    return [to_stack(args) for args in picked]


def executable(value: Value) -> bool:
    """A quotation that could compute: a list containing an op name."""
    if isinstance(value, Sym):
        return True
    if isinstance(value, Lst):
        return any(executable(item) for item in value.items)
    return False


# Where each code-running op finds its body, and the list it iterates, counted
# down from the top of the stack.
BODY_AT = {"i": 1, "dip": 1, "map": 1, "filter": 1, "fold": 1, "ifte": 3}
SOURCE_AT = {"map": 2, "filter": 2, "fold": 3}


def ran_an_op(stack: tuple[Value, ...], name: str) -> bool:
    """Whether this application actually executes an op from its body.

    A legal application is not the same as a working one. `filter` over an
    empty list is legal whatever the body is, because the body is never
    reached -- and under `executable-26` that vacuous case is the *only* case,
    which is exactly the thing being measured.
    """
    if quotation_open(stack) or len(stack) < BODY_AT[name]:
        return False
    body = stack[-BODY_AT[name]]
    bodies = stack[-3:] if name == "ifte" else (body,)
    if not any(executable(q) for q in bodies):
        return False
    if name in SOURCE_AT:
        source = stack[-SOURCE_AT[name]]
        return isinstance(source, Lst) and bool(source.items)
    return True


def sweep(root: tuple[Value, ...], basis: str, depth: int) -> dict:
    """BFS the ball around one data-only root, counting what the ops do."""
    ops = ops_for_basis(basis)
    legal: Counter[str] = Counter()
    ran: Counter[str] = Counter()
    seen = {tuple(value_key(v) for v in root)}
    frontier = [root]
    quotations, executable_quotations = set(), set()

    for _ in range(depth):
        nxt = []
        for stack in frontier:
            for op in ops:
                outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
                if isinstance(outcome, str):
                    continue
                moved, _cost = outcome
                if op.name in RUNS_CODE and not quotation_open(stack):
                    # Under an open MARK the op is captured as a symbol, not
                    # applied, so counting it as legal would overstate reach.
                    legal[op.name] += 1
                    if ran_an_op(stack, op.name):
                        ran[op.name] += 1
                for value in moved:
                    if isinstance(value, Lst):
                        quotations.add(value_key(value))
                        if executable(value):
                            executable_quotations.add(value_key(value))
                key = tuple(value_key(v) for v in moved)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(moved)
        frontier = nxt
    return {
        "states": len(seen),
        "quotations": len(quotations),
        "executable_quotations": len(executable_quotations),
        "legal": dict(legal),
        "ran": dict(ran),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    starts = roots(args.roots, args.seed)
    print(f"{len(starts)} data-only roots, depth {args.depth}")
    for stack in starts:
        print(f"  {format_stack(Config(stack))}")

    summary = {}
    for basis in BASES:
        started = time.time()
        totals: Counter[str] = Counter()
        legal: Counter[str] = Counter()
        ran: Counter[str] = Counter()
        for stack in starts:
            result = sweep(stack, basis, args.depth)
            totals["states"] += result["states"]
            totals["quotations"] += result["quotations"]
            totals["executable_quotations"] += result["executable_quotations"]
            legal.update(result["legal"])
            ran.update(result["ran"])
        summary[basis] = {
            "states": totals["states"],
            "quotations": totals["quotations"],
            "executable_quotations": totals["executable_quotations"],
            "legal": dict(legal),
            "ran_an_op": dict(ran),
            "seconds": round(time.time() - started, 1),
        }
        print(f"\n{basis}: {totals['states']:,} states in {summary[basis]['seconds']}s")
        print(
            f"  distinct quotations reached {totals['quotations']:,}, "
            f"of which executable {totals['executable_quotations']:,}"
        )
        print(f"  {'op':<8}{'legal':>10}{'ran an op':>14}")
        for name in RUNS_CODE:
            print(f"  {name:<8}{legal.get(name, 0):>10,}{ran.get(name, 0):>14,}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {"depth": args.depth, "roots": len(starts), "bases": summary},
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
