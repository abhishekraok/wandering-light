"""BFS over the stack machine state graph, to measure what the basis reaches.

Three measurements, all from `wandering_light.stack_machine`:

* `deep`     -- how fast the reachable set grows with depth, and what values
                appear, from several roots.
* `collapse` -- executable programs vs distinct states at each depth. This is
                the composition-monoid question #42 raises about the old basis.
* `graph`    -- in-degree, reversibility and hubs on the fully expanded graph
                out to a small depth.

States are keyed on the stack tuple. `tests/test_stack_machine.py` pins
dataclass equality to `value_key`, so that key is the structural one.

    uv run python experiments/stack_machine_bfs.py > results.json
"""

import json
import math
import sys
import time
from collections import Counter, defaultdict

from wandering_light.stack_machine import (
    OPS,
    Config,
    Int,
    Lst,
    format_stack,
    parse_program,
    run,
)

ROOTS = {
    "empty": "",
    "one list": "nil 1 cons 0 cons 1 cons",
    "two ints": "1 1",
    "list and int": "nil 1 cons 0 cons 1",
}


def root_stack(program: str) -> tuple:
    config = Config()
    for reached in run(parse_program(program), config):
        config = reached
    return config.stack


def nesting(value) -> int:
    if isinstance(value, Lst):
        return 1 + max((nesting(item) for item in value.items), default=0)
    return 0


def shape(stack: tuple) -> dict:
    ints = [abs(v.value) for v in stack if isinstance(v, Int)]
    lists = [v for v in stack if isinstance(v, Lst)]
    return {
        "max_int": max(ints, default=0),
        "max_list_len": max((len(v.items) for v in lists), default=0),
        "max_nesting": max((nesting(v) for v in lists), default=0),
    }


def deep_bfs(root: tuple, max_depth: int, deadline: float) -> dict:
    """Grow the reachable set depth by depth until time or depth runs out."""
    seen = {root}
    frontier = [root]
    levels, legal_hist, discovery = [], Counter(), Counter()
    refusal = defaultdict(Counter)
    int_cost, list_cost = {}, {}
    reach = {"max_int": 0, "max_list_len": 0, "max_nesting": 0}

    for depth in range(1, max_depth + 1):
        started = time.time()
        nxt, legal_total, edges = [], 0, 0
        for stack in frontier:
            config = Config(stack)
            legal_here = 0
            for op in OPS:
                reason = op.guard(config)
                if reason is not None:
                    refusal[op.name][reason] += 1
                    continue
                legal_here += 1
                edges += 1
                new, _ = op.apply(config)
                if new not in seen:
                    seen.add(new)
                    nxt.append(new)
                    discovery[op.name] += 1
                    if len(new) == 1:
                        only = new[0]
                        if isinstance(only, Int) and only.value not in int_cost:
                            int_cost[str(only.value)] = depth
                        elif isinstance(only, Lst) and only.items:
                            key = format_stack(Config(new))
                            if key not in list_cost:
                                list_cost[key] = depth
            legal_total += legal_here
            legal_hist[legal_here] += 1

        for stack in nxt:
            s = shape(stack)
            for k, v in s.items():
                reach[k] = max(reach[k], v)

        levels.append(
            {
                "depth": depth,
                "expanded": len(frontier),
                "edges": edges,
                "new": len(nxt),
                "cumulative": len(seen),
                "b_eff": legal_total / max(1, len(frontier)),
                "seconds": round(time.time() - started, 2),
                "reach": dict(reach),
            }
        )
        frontier = nxt
        if not frontier or time.time() > deadline:
            break

    return {
        "levels": levels,
        "legal_hist": dict(sorted(legal_hist.items())),
        "discovery": dict(discovery.most_common()),
        "refusal": {k: dict(v.most_common()) for k, v in refusal.items()},
        "int_cost": int_cost,
        "list_cost": list_cost,
        "completed_depth": levels[-1]["depth"] if levels else 0,
        "exhausted": not frontier,
    }


def collapse(root: tuple, max_depth: int) -> list[dict]:
    """Executable programs of length d against distinct states reached in d."""
    counts = {root: 1}
    rows = []
    for depth in range(1, max_depth + 1):
        nxt = defaultdict(int)
        for stack, n in counts.items():
            config = Config(stack)
            for op in OPS:
                if op.guard(config) is None:
                    new, _ = op.apply(config)
                    nxt[new] += n
        if not nxt:
            break
        counts = dict(nxt)
        programs = sum(counts.values())
        rows.append(
            {
                "depth": depth,
                "programs": programs,
                "states": len(counts),
                "collapse": programs / len(counts),
            }
        )
    return rows


def graph(root: tuple, max_depth: int) -> dict:
    """Full adjacency out to max_depth: in-degree, reversibility, hubs."""
    seen = {root: 0}
    frontier = [root]
    out_edges = defaultdict(set)
    indeg = Counter()
    for depth in range(1, max_depth + 1):
        nxt = []
        for stack in frontier:
            config = Config(stack)
            for op in OPS:
                if op.guard(config) is None:
                    new, _ = op.apply(config)
                    out_edges[stack].add(new)
                    indeg[new] += 1
                    if new not in seen:
                        seen[new] = depth
                        nxt.append(new)
        frontier = nxt
    interior = [s for s in seen if s in out_edges]
    mutual = sum(
        1
        for a in interior
        for b in out_edges[a]
        if b in out_edges and a in out_edges[b]
    )
    total_out = sum(len(out_edges[a]) for a in interior)
    hubs = [
        {"stack": format_stack(Config(s)), "in_degree": n, "depth": seen[s]}
        for s, n in indeg.most_common(12)
    ]
    return {
        "nodes": len(seen),
        "distinct_edges": total_out,
        "reversible_fraction": mutual / max(1, total_out),
        "indeg_hist": dict(sorted(Counter(indeg.values()).items())[:20]),
        "hubs": hubs,
    }


def reachable(ops, root: tuple, max_depth: int) -> int:
    """Size of the reachable set within max_depth, using only `ops`."""
    seen = {root}
    frontier = [root]
    for _ in range(max_depth):
        nxt = []
        for stack in frontier:
            config = Config(stack)
            for op in ops:
                if op.guard(config) is None:
                    new, _ = op.apply(config)
                    if new not in seen:
                        seen.add(new)
                        nxt.append(new)
        frontier = nxt
        if not frontier:
            break
    return len(seen)


def ablation(root: tuple, max_depth: int) -> list[dict]:
    """Drop one op at a time and measure what the basis stops reaching.

    An op whose removal costs nothing is dead weight; one whose removal
    collapses the space is load-bearing. This is the measurement #41 needs to
    decide what a basis should contain.
    """
    full = reachable(OPS, root, max_depth)
    rows = []
    for op in OPS:
        kept = [o for o in OPS if o is not op]
        without = reachable(kept, root, max_depth)
        rows.append(
            {
                "op": op.name,
                "group": op.group,
                "reachable_without": without,
                "fraction_lost": 1 - without / full,
            }
        )
    rows.sort(key=lambda r: -r["fraction_lost"])
    return {"full": full, "max_depth": max_depth, "ops": rows}


def old_basis(max_depth: int, max_states: int) -> dict:
    """The same growth measurement on wl-core-v1, for comparison."""
    from collections import Counter as _Counter

    from wandering_light.basis_set import load_basis_set
    from wandering_light.proposer_pilot.graph import TrajectoryGraph
    from wandering_light.typed_list import TypedList

    basis = load_basis_set("wl-core-v1")
    functions = basis.to_function_def_set()
    out = {"basis": basis.basis_set_id, "n_functions": len(functions), "roots": {}}
    for label, tl in [
        ("[1, 0, 1]", TypedList([1, 0, 1])),
        ("[1, 2, 3]", TypedList([1, 2, 3])),
    ]:
        graph = TrajectoryGraph(functions=functions)
        root = graph.add_root(tl)
        started = time.time()
        result = graph.expand(root, max_depth=max_depth, max_states=max_states)
        counts = _Counter(result.node_depths.values())
        out["roots"][label] = {
            "levels": [{"depth": d, "new": counts[d]} for d in sorted(counts)],
            "total": len(result.node_depths),
            "complete": result.complete,
            "certified_depth": result.certified_depth,
            "stop_reason": str(result.stop_reason),
            "attempted_transitions": result.attempted_transitions,
            "seconds": round(time.time() - started, 1),
        }
    return out


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 540.0
    started = time.time()
    out = {"budget_seconds": budget, "roots": {}}

    share = {"empty": 0.55, "one list": 0.15, "two ints": 0.15, "list and int": 0.15}
    for name, program in ROOTS.items():
        root = root_stack(program)
        deadline = started + budget * sum(
            share[k] for k in list(ROOTS)[: list(ROOTS).index(name) + 1]
        )
        out["roots"][name] = {
            "program": program,
            "stack": format_stack(Config(root)),
            "bfs": deep_bfs(root, max_depth=14, deadline=deadline),
        }
        print(
            f"[{time.time() - started:6.1f}s] {name}: "
            f"depth {out['roots'][name]['bfs']['completed_depth']}, "
            f"{out['roots'][name]['bfs']['levels'][-1]['cumulative']:,} states",
            file=sys.stderr,
        )

    out["collapse"] = collapse(root_stack(""), max_depth=7)
    print(f"[{time.time() - started:6.1f}s] collapse done", file=sys.stderr)
    out["graph"] = graph(root_stack(""), max_depth=5)
    print(f"[{time.time() - started:6.1f}s] graph done", file=sys.stderr)
    out["ablation"] = ablation(root_stack(""), max_depth=7)
    print(f"[{time.time() - started:6.1f}s] ablation done", file=sys.stderr)
    try:
        out["old_basis"] = old_basis(max_depth=4, max_states=400_000)
    except Exception as error:
        out["old_basis"] = {"error": f"{type(error).__name__}: {error}"}
    print(f"[{time.time() - started:6.1f}s] old basis done", file=sys.stderr)
    out["bits"] = {
        "uniform": math.log2(len(OPS)),
        "ops": len(OPS),
    }
    print(f"[{time.time() - started:6.1f}s] done", file=sys.stderr)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
