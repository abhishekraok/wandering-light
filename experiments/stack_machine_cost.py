"""Description length of the same targets in both bases.

Reachability says what a basis can express; this says what it costs. Both
bases start holding the integer 1 and we record the shortest program that
leaves integer n behind -- `[ n ]` on the stack machine, `TypedList([n])` on
wl-core-v1. Same targets, same units (ops), so the numbers compare directly.

    uv run python experiments/stack_machine_cost.py > cost.json
"""

import json
import sys
import time

from wandering_light.stack_machine import OPS, Config, Int, parse_program, run

TARGETS = [*range(33), 36, 48, 63, 64, 100, 128, 255, 256, 1000, 1024]


# Integer arithmetic never needs a deep stack, and capping the stack keeps the
# search finite enough to reach the depths where the interesting targets live.
# A cost reported here is therefore the shortest program using at most this many
# stack cells: an upper bound on the true shortest program.
MAX_CELLS = 4


def stack_costs(max_depth: int, deadline: float) -> dict:
    config = Config()
    for reached in run(parse_program("1"), config):
        config = reached
    root = config.stack

    seen = {root}
    frontier = [root]
    cost = {}
    for depth in range(1, max_depth + 1):
        nxt = []
        for stack in frontier:
            current = Config(stack)
            for op in OPS:
                if op.guard(current) is not None:
                    continue
                new, _ = op.apply(current)
                if len(new) > MAX_CELLS or new in seen:
                    continue
                seen.add(new)
                nxt.append(new)
                if len(new) == 1 and isinstance(new[0], Int):
                    value = new[0].value
                    if value in TARGETS and value not in cost:
                        cost[value] = depth
        frontier = nxt
        if not frontier or time.time() > deadline:
            break
    if 1 not in cost:
        cost[1] = 0
    return {
        "searched_depth": depth,
        "states": len(seen),
        "max_cells": MAX_CELLS,
        "cost": {str(k): v for k, v in sorted(cost.items())},
    }


def old_costs(max_depth: int, max_states: int) -> dict:
    from wandering_light.basis_set import load_basis_set
    from wandering_light.proposer_pilot.graph import TrajectoryGraph
    from wandering_light.typed_list import TypedList

    functions = load_basis_set("wl-core-v1").to_function_def_set()
    graph = TrajectoryGraph(functions=functions)
    root = graph.add_root(TypedList([1]))
    result = graph.expand(root, max_depth=max_depth, max_states=max_states)

    cost = {}
    for node_id, depth in result.node_depths.items():
        tl = graph.node(node_id).typed_list
        items = list(tl.items)
        if (
            len(items) == 1
            and isinstance(items[0], int)
            and not isinstance(items[0], bool)
        ):
            value = items[0]
            if value in TARGETS and (value not in cost or depth < cost[value]):
                cost[value] = depth
    return {
        "searched_depth": max_depth,
        "states": len(result.node_depths),
        "complete": result.complete,
        "stop_reason": str(result.stop_reason),
        "cost": {str(k): v for k, v in sorted(cost.items())},
    }


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    out = {"targets": TARGETS}
    out["stack_machine"] = stack_costs(16, time.time() + budget * 0.55)
    print(
        f"stack machine: depth {out['stack_machine']['searched_depth']}",
        file=sys.stderr,
    )
    out["old_basis"] = old_costs(max_depth=6, max_states=250_000)
    print(f"old basis: {out['old_basis']['states']:,} states", file=sys.stderr)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
