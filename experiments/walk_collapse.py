"""Does undirected expansion reach past the BFS horizon?

Everything measured on this substrate so far is scoped to distances 1-8, where
BFS gives a baseline. The far regime is the target, and the intuition offered
for it is *outer space with sparse stars*: most states useless and unvisited,
so getting the general direction right over long travel matters more than being
accurate over short travel. That intuition implies a cheap way to reach far
states -- walk a long way. This checks whether walking works.

It does not. Two measurements:

**Collapse.** 300 uniform-random walks of length 40 from each of six root types
land on 43-56 *distinct* endpoint states, with an endpoint type distribution
near-identical whatever the walk started from. The walk forgets its origin.

**Distance.** Against a depth-6 ball around one root, an 80-step walk lands
within 6 of the root 99% of the time, modal true distance 2 -- the same
distribution a 4-step walk gives. Walking twenty times further travels no
further at all.

The mechanism is in the type flow: 37 of 118 ops output `int` and 27 output
`bool`, and the ops applicable to a state span 3 (`bytearray`) to 29 (`str`).
Undirected motion runs downhill into low-out-degree scalar types and stays
there. The volume is real -- 233,585 states at depth 6, still growing ~6x per
level -- it is just not reachable by drifting.

**What this does and does not license.** The walk policy is uniform over
applicable ops, so this bounds what *undirected* expansion reaches, not what a
trained policy reaches. It says a directed mechanism is required rather than
optional. It does not say any particular directed mechanism works; nothing here
measures that.

    uv run python experiments/walk_collapse.py [--walks=N] [--length=N] [--depth=N]
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter

from wandering_light.basis_set import load_basis_set
from wandering_light.executor import Executor
from wandering_light.proposer_pilot.graph import TrajectoryGraph
from wandering_light.typed_list import TypedList

# One root per type, matching the corpus root generator's spirit: a handful of
# items spanning sign, emptiness and magnitude rather than a single value.
ROOTS = {
    "int": TypedList([3, -7, 0, 12, 5], item_type=int),
    "str": TypedList(["ab", "Zed", "", "hi there", "Q"], item_type=str),
    "list": TypedList([[1, 2], [], [3, 3, 3], [0], [5, 1]], item_type=list),
    "float": TypedList([1.5, -0.25, 3.0, 99.9, 0.0], item_type=float),
    "bytes": TypedList([b"ab", b"", b"\xff\x01", b"hi", b"Z"], item_type=bytes),
    "dict": TypedList([{"a": 1}, {}, {"b": 2, "c": 3}, {"d": 4}, {}], item_type=dict),
}


def walk(start: TypedList, steps: int, functions, executor, rng) -> TypedList | None:
    """Uniformly random applicable op at each step, skipping self-loops.

    Self-loops are skipped so the walk cannot sit still and report having
    travelled; a state with no non-identity successor returns None rather than
    padding the sample with a stuck walk.
    """
    state = start
    for _ in range(steps):
        candidates = [
            fn for fn in functions.functions if fn.input_type_cls() is state.item_type
        ]
        rng.shuffle(candidates)
        for fn in candidates:
            try:
                nxt = executor.execute(fn, state)
            except Exception:
                # A runtime failure is an absent transition, matching
                # TrajectoryGraph.expand and BFSPredictor.
                continue
            if nxt.search_key() != state.search_key():
                state = nxt
                break
        else:
            return None
    return state


def collapse(functions, executor, walks: int, length: int) -> None:
    """How many distinct states do many long walks actually reach?"""
    print(f"\n== Collapse: {walks} walks of length {length}, per root type\n")
    print(f"{'root type':>10} {'distinct endpoints':>20} {'endpoint types':>44}")
    for name, root in ROOTS.items():
        rng = random.Random(11)
        ends = [walk(root, length, functions, executor, rng) for _ in range(walks)]
        ends = [e for e in ends if e is not None]
        distinct = len({e.search_key() for e in ends})
        types = Counter(e.item_type.__name__ for e in ends)
        top = " ".join(f"{k}:{v}" for k, v in types.most_common(4))
        print(f"{name:>10} {f'{distinct} of {len(ends)}':>20} {top:>44}")


def travel(functions, executor, depth: int, walks: int) -> None:
    """Do longer walks end up further away? Distances from one exhaustive ball."""
    root = ROOTS["int"]
    started = time.time()
    graph = TrajectoryGraph(functions=functions)
    result = graph.expand(graph.add_root(root), depth)
    depth_of = {
        graph._nodes[n].typed_list.search_key(): d
        for n, d in result.node_depths.items()
    }
    print(
        f"\n== Travel: depth-{depth} ball around the int root -- "
        f"{len(depth_of):,} states in {time.time() - started:.0f}s\n"
    )
    print(
        f"{'walk length':>12} {'inside the ball':>18} {'true distance if inside':>34}"
    )
    for length in (4, 6, 10, 20, 40, 80):
        rng = random.Random(7)
        inside: Counter = Counter()
        total = outside = 0
        for _ in range(walks):
            end = walk(root, length, functions, executor, rng)
            if end is None:
                continue
            total += 1
            found = depth_of.get(end.search_key())
            if found is None:
                outside += 1
            else:
                inside[found] += 1
        hist = " ".join(f"{d}:{c}" for d, c in sorted(inside.items()))
        share = f"{total - outside}/{total} ({100 * (total - outside) / max(total, 1):.0f}%)"
        print(f"{length:>12} {share:>18} {hist:>34}")


def main() -> None:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    def flag(name: str, default: int) -> int:
        return next(
            (int(a.split("=")[1]) for a in flags if a.startswith(f"--{name}=")), default
        )

    walks, length, depth = flag("walks", 300), flag("length", 40), flag("depth", 6)

    functions = load_basis_set("wl-core-v1").to_function_def_set()
    executor = Executor(functions)

    out_degree = Counter(fn.input_type for fn in functions.functions)
    print("ops applicable by state type:")
    print(
        "  "
        + ", ".join(
            f"{t.split('.')[-1]} {n}"
            for t, n in sorted(out_degree.items(), key=lambda kv: kv[1])
        )
    )

    collapse(functions, executor, walks, length)
    travel(functions, executor, depth, walks)


if __name__ == "__main__":
    main()
