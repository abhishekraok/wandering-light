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

The bounded ``mark-28`` basis comparison is a separate mode:

    uv run python experiments/stack_machine_bfs.py \
      --summary --basis mark-28 > mark-28.json
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict

from wandering_light.stack_machine import (
    BASIS_OPS,
    DEFAULT_BASIS,
    OPS,
    Config,
    Int,
    Lst,
    format_stack,
    ops_for_basis,
    parse_program,
    run,
    value_key,
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


def quote_is_open(stack: tuple) -> bool:
    """Whether the stack contains the structural MARK used by ``mark-28``."""
    return any(value_key(value) == ("mark",) for value in stack)


def deep_bfs(root: tuple, max_depth: int, deadline: float, ops=OPS) -> dict:
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
            for op in ops:
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


def collapse(root: tuple, max_depth: int, ops=OPS) -> list[dict]:
    """Executable programs of length d against distinct states reached in d."""
    counts = {root: 1}
    rows = []
    for depth in range(1, max_depth + 1):
        nxt = defaultdict(int)
        for stack, n in counts.items():
            config = Config(stack)
            for op in ops:
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


def graph(root: tuple, max_depth: int, ops=OPS) -> dict:
    """Full adjacency out to max_depth: in-degree, reversibility, hubs."""
    seen = {root: 0}
    frontier = [root]
    out_edges = defaultdict(set)
    indeg = Counter()
    for depth in range(1, max_depth + 1):
        nxt = []
        for stack in frontier:
            config = Config(stack)
            for op in ops:
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


def ablation(root: tuple, max_depth: int, ops=OPS) -> list[dict]:
    """Drop one op at a time and measure what the basis stops reaching.

    An op whose removal costs nothing is dead weight; one whose removal
    collapses the space is load-bearing. This is the measurement #41 needs to
    decide what a basis should contain.
    """
    full = reachable(ops, root, max_depth)
    rows = []
    for op in ops:
        kept = [o for o in ops if o is not op]
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


SUMMARY_DEPTH = 4
GROWTH_DEPTH = {
    "executable-26": 5,
    "full-52": 4,
    "mark-28": 5,
}


def measure_frontiers(
    root: tuple, max_depth: int, ops, *, track_quote_closed: bool = False
) -> list[dict]:
    """The bounded subset of ``deep_bfs`` needed for basis comparison.

    Avoid retaining refusal histograms and reachable-value examples here. At
    depth 4, ``full-52`` discovers more than a million states per root; those
    diagnostics otherwise dominate the memory cost of this small measurement.
    """
    seen = {root}
    frontier = [root]
    rows = []
    for depth in range(1, max_depth + 1):
        nxt = []
        edges = 0
        for stack in frontier:
            config = Config(stack)
            for op in ops:
                if op.guard(config) is not None:
                    continue
                edges += 1
                new, _ = op.apply(config)
                if new not in seen:
                    seen.add(new)
                    nxt.append(new)
        rows.append(
            {
                "depth": depth,
                "expanded": len(frontier),
                "edges": edges,
                "new": len(nxt),
                "cumulative": len(seen),
                "quote_closed_new": (
                    sum(not quote_is_open(stack) for stack in nxt)
                    if track_quote_closed
                    else len(nxt)
                ),
            }
        )
        frontier = nxt
        if not frontier:
            break
    return rows


def basis_summary(basis: str, ops) -> dict:
    """Return the bounded branching and quotation-frontier measurements.

    ``b_eff`` pools every state expanded at depths 1--4 from all four roots.
    Growth is the arithmetic mean of the last two frontier ratios from the
    ``one list`` root. The full basis stops at depth 4; its depth-5 frontier is
    tens of millions of states. The two smaller bases stop at depth 5.
    """
    growth_depth = GROWTH_DEPTH[basis]
    roots = {}
    pooled_edges = 0
    pooled_expanded = 0

    for name, program in ROOTS.items():
        max_depth = growth_depth if name == "one list" else SUMMARY_DEPTH
        levels = measure_frontiers(
            root_stack(program),
            max_depth,
            ops,
            track_quote_closed=basis == "mark-28",
        )
        roots[name] = {
            "program": program,
            "levels": [
                {
                    key: row[key]
                    for key in (
                        "depth",
                        "expanded",
                        "edges",
                        "new",
                        "cumulative",
                        "quote_closed_new",
                    )
                }
                for row in levels
            ],
        }
        for row in levels[:SUMMARY_DEPTH]:
            pooled_edges += row["edges"]
            pooled_expanded += row["expanded"]

    b_eff = pooled_edges / pooled_expanded
    growth_levels = roots["one list"]["levels"][-2:]
    growth_factors = [row["new"] / row["expanded"] for row in growth_levels]
    growth = sum(growth_factors) / len(growth_factors)

    out = {
        "mode": "basis-summary",
        "basis": basis,
        "ops": len(ops),
        "protocol": {
            "b_eff": (
                "legal edges / expanded states, pooled over all four roots "
                "and depths 1 through 4"
            ),
            "bits_per_step": "log2(b_eff)",
            "growth_per_depth": (
                "arithmetic mean of the final two new/expanded frontier ratios "
                "from the one-list root"
            ),
            "growth_max_depth": growth_depth,
            "state_space_split": (
                "new frontier states at the one-list root; quote-closed means "
                "no structural MARK on the stack"
            ),
        },
        "b_eff": b_eff,
        "bits_per_step": math.log2(b_eff),
        "growth_per_depth": growth,
        "growth_factors": [
            {
                "depth": row["depth"],
                "new": row["new"],
                "expanded": row["expanded"],
                "factor": factor,
            }
            for row, factor in zip(growth_levels, growth_factors, strict=True)
        ],
        "pooled": {
            "max_depth": SUMMARY_DEPTH,
            "edges": pooled_edges,
            "expanded": pooled_expanded,
        },
        "roots": roots,
    }
    if basis == "mark-28":
        by_depth = {row["depth"]: row for row in roots["one list"]["levels"]}
        out["state_space_split"] = [
            {
                "depth": depth,
                "all_states": by_depth[depth]["new"],
                "quote_closed": by_depth[depth]["quote_closed_new"],
                "closed_fraction": (
                    by_depth[depth]["quote_closed_new"] / by_depth[depth]["new"]
                ),
            }
            for depth in (3, 4)
        ]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "budget",
        nargs="?",
        type=float,
        default=540.0,
        help="legacy full-analysis wall-clock budget in seconds",
    )
    parser.add_argument(
        "--basis",
        choices=sorted(BASIS_OPS),
        default=None,
        help=(
            "stack-machine action basis (default: executable-26 for --summary; "
            "full-52 for the legacy analysis)"
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="run only the bounded deterministic basis measurements",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    basis = args.basis or (DEFAULT_BASIS if args.summary else "full-52")
    ops = ops_for_basis(basis)
    if args.summary:
        json.dump(basis_summary(basis, ops), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return

    budget = args.budget
    started = time.time()
    out = {
        "basis": basis,
        "ops": len(ops),
        "budget_seconds": budget,
        "roots": {},
    }

    share = {"empty": 0.55, "one list": 0.15, "two ints": 0.15, "list and int": 0.15}
    for name, program in ROOTS.items():
        root = root_stack(program)
        deadline = started + budget * sum(
            share[k] for k in list(ROOTS)[: list(ROOTS).index(name) + 1]
        )
        out["roots"][name] = {
            "program": program,
            "stack": format_stack(Config(root)),
            "bfs": deep_bfs(root, max_depth=14, deadline=deadline, ops=ops),
        }
        print(
            f"[{time.time() - started:6.1f}s] {name}: "
            f"depth {out['roots'][name]['bfs']['completed_depth']}, "
            f"{out['roots'][name]['bfs']['levels'][-1]['cumulative']:,} states",
            file=sys.stderr,
        )

    out["collapse"] = collapse(root_stack(""), max_depth=7, ops=ops)
    print(f"[{time.time() - started:6.1f}s] collapse done", file=sys.stderr)
    out["graph"] = graph(root_stack(""), max_depth=5, ops=ops)
    print(f"[{time.time() - started:6.1f}s] graph done", file=sys.stderr)
    out["ablation"] = ablation(root_stack(""), max_depth=7, ops=ops)
    print(f"[{time.time() - started:6.1f}s] ablation done", file=sys.stderr)
    try:
        out["old_basis"] = old_basis(max_depth=4, max_states=400_000)
    except Exception as error:
        out["old_basis"] = {"error": f"{type(error).__name__}: {error}"}
    print(f"[{time.time() - started:6.1f}s] old basis done", file=sys.stderr)
    out["bits"] = {
        "uniform": math.log2(len(ops)),
        "ops": len(ops),
    }
    print(f"[{time.time() - started:6.1f}s] done", file=sys.stderr)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
