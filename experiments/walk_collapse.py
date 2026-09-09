"""Does undirected expansion reach past the BFS horizon?

Everything measured on this substrate so far is scoped to distances 1-8, where
BFS gives a baseline. The far regime is the target, and the intuition offered
for it is *outer space with sparse stars*: most states useless and unvisited,
so getting the general direction right over long travel matters more than being
accurate over short travel. That intuition implies a cheap way to reach far
states -- walk a long way. This checks whether walking works, on both
substrates.

It does not, and the two substrates fail in opposite directions.

**`wl-core-v1` drains.** 300 uniform-random walks of length 40 from each of six
root types land on 43-56 *distinct* endpoint states, with an endpoint type
distribution near-identical whatever the walk started from. Against a depth-6
ball around one root, an 80-step walk lands within 6 of the root 99% of the
time, modal true distance 2 -- the same distribution a 4-step walk gives. The
mechanism is in the type flow: 37 of 118 ops output `int` and 27 output
`bool`, and the ops applicable to a state span 3 (`bytearray`) to 29 (`str`).
Undirected motion runs downhill into low-out-degree scalar types and stays
there. The volume is real -- 233,585 states at depth 6, still growing ~6x per
level -- it is just not reachable by drifting.

**The stack machine diffuses.** Under `executable-26`, 300 walks from each of
the four `stack_machine_bfs.py` roots give 300/300 distinct endpoints at length
40 and at 80; the share still inside the depth-5 ball falls 87% -> 17% -> 0%
at 6, 20 and 80 steps; endpoint out-degree settles at ~12.5, which is `b_eff`.
Literals are always legal, so the stack grows ~0.36 items per step (~15 deep
at 40 steps, ~29 at 80) and a walk buries its origin under junk instead of
forgetting it. Unguarded `mark-28` is worse: out-degree ~24 and over half the
walks end inside an open `[`. The #97 guard keeps the stack a third as deep by
forcing `]`, which folds the junk into nested lists -- endpoints stay 300/300
distinct and 0% inside the ball at 80. `reports/walk-collapse-stack-20260908/`.

**What this does and does not license.** The walk policy is uniform over
applicable ops, so this bounds what *undirected* expansion reaches, not what a
trained policy reaches: one substrate cannot leave home, the other cannot come
home. Whether a directed mechanism reaches further is a separate measurement
and has its own rows in `docs/STATUS.md`.

    uv run python experiments/walk_collapse.py [--walks=N] [--length=N] [--depth=N]
    uv run python experiments/walk_collapse.py --stack [--basis=NAME] [--guard=MODE] ...

`--stack` selects the stack machine. `--basis` is any name in
`stack_machine.BASIS_OPS` (default `DEFAULT_BASIS`); `--guard` applies
`quotation_guard.blocked` to successors as `mark-28` search does (`off`,
`close`, `body` or `both`; default `off`). `--cap` bounds the distance ball in
states; only complete layers count, so distances stay exact up to the depth
the cap allowed.
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter

from wandering_light.basis_set import load_basis_set
from wandering_light.executor import Executor
from wandering_light.proposer_pilot.graph import TrajectoryGraph
from wandering_light.quotation_guard import blocked
from wandering_light.stack_machine import (
    DEFAULT_BASIS,
    Bool,
    Config,
    Int,
    Lst,
    Mark,
    Sym,
    ops_for_basis,
    parse_program,
    run,
)
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

# The four roots of experiments/stack_machine_bfs.py, as programs from the
# empty stack, so the depth-5 ball here reproduces its pinned counts.
STACK_ROOTS = {
    "empty": "",
    "one list": "nil 1 cons 0 cons 1 cons",
    "two ints": "1 1",
    "list and int": "nil 1 cons 0 cons 1",
}

WALK_LENGTHS = (4, 6, 10, 20, 40, 80)


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
    for length in WALK_LENGTHS:
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


# --- stack machine -----------------------------------------------------------


def stack_root(program: str) -> tuple:
    config = Config()
    for reached in run(parse_program(program), config):
        config = reached
    return config.stack


def stack_successors(stack: tuple, ops, guard: str, remaining: int) -> list[tuple]:
    """Every legal, non-identity successor, as the guarded search would see it."""
    config = Config(stack)
    out = []
    for op in ops:
        if op.guard(config) is not None:
            continue
        new, _ = op.apply(config)
        if new == stack:
            continue
        if blocked(new, remaining, guard) is not None:
            continue
        out.append(new)
    return out


def stack_walk(root: tuple, steps: int, ops, guard: str, rng) -> tuple | None:
    """Uniformly random legal op at each step; None if the walk gets stuck."""
    stack = root
    for step in range(steps):
        successors = stack_successors(stack, ops, guard, steps - step - 1)
        if not successors:
            return None
        stack = rng.choice(successors)
    return stack


def stack_ball(root: tuple, ops, depth: int, cap: int) -> tuple[dict, int]:
    """Exact distances from the root, by complete layers only.

    Returns the distance map and the last depth fully expanded. A layer that
    would push the map past `cap` is discarded whole rather than half-built,
    so every recorded distance is exact and every miss is beyond the depth.
    """
    dist = {root: 0}
    frontier = [root]
    completed = 0
    for layer in range(1, depth + 1):
        nxt = []
        for stack in frontier:
            config = Config(stack)
            for op in ops:
                if op.guard(config) is not None:
                    continue
                new, _ = op.apply(config)
                if new not in dist:
                    dist[new] = layer
                    nxt.append(new)
        if len(dist) > cap:
            for stack in nxt:
                del dist[stack]
            break
        completed = layer
        frontier = nxt
        if not frontier:
            break
    return dist, completed


def top_type(stack: tuple) -> str:
    if not stack:
        return "empty"
    top = stack[-1]
    for cls, name in (
        (Int, "int"),
        (Bool, "bool"),
        (Lst, "list"),
        (Sym, "sym"),
        (Mark, "mark"),
    ):
        if isinstance(top, cls):
            return name
    return type(top).__name__


def stack_collapse(ops, guard: str, walks: int, length: int) -> None:
    """Distinct endpoints, and what the walk leaves on the stack."""
    print(f"\n== Collapse: {walks} walks of length {length}, per root\n")
    print(
        f"{'root':>13} {'distinct endpoints':>20} {'stack depth':>12} "
        f"{'out-degree':>11} {'top-of-stack types':>36}"
    )
    for name, program in STACK_ROOTS.items():
        root = stack_root(program)
        rng = random.Random(11)
        ends = [stack_walk(root, length, ops, guard, rng) for _ in range(walks)]
        ends = [e for e in ends if e is not None]
        distinct = len(set(ends))
        depth = sum(len(e) for e in ends) / max(len(ends), 1)
        degree = sum(len(stack_successors(e, ops, "off", length)) for e in ends) / max(
            len(ends), 1
        )
        types = Counter(top_type(e) for e in ends)
        top = " ".join(f"{k}:{v}" for k, v in types.most_common(4))
        print(
            f"{name:>13} {f'{distinct} of {len(ends)}':>20} {depth:>12.1f} "
            f"{degree:>11.1f} {top:>36}"
        )


def stack_travel(ops, guard: str, depth: int, cap: int, walks: int) -> None:
    """Do longer walks end up further away? Distances from one exhaustive ball."""
    for name, program in STACK_ROOTS.items():
        root = stack_root(program)
        started = time.time()
        depth_of, completed = stack_ball(root, ops, depth, cap)
        capped = (
            ""
            if completed == depth
            else f" (cap {cap:,} hit; depth {completed} is complete)"
        )
        print(
            f"\n== Travel: depth-{completed} ball around root '{name}' -- "
            f"{len(depth_of):,} states in {time.time() - started:.0f}s{capped}\n"
        )
        print(
            f"{'walk length':>12} {'inside the ball':>18} {'stack depth':>12} "
            f"{'true distance if inside':>30}"
        )
        for length in WALK_LENGTHS:
            rng = random.Random(7)
            inside: Counter = Counter()
            total = outside = items = 0
            for _ in range(walks):
                end = stack_walk(root, length, ops, guard, rng)
                if end is None:
                    continue
                total += 1
                items += len(end)
                found = depth_of.get(end)
                if found is None:
                    outside += 1
                else:
                    inside[found] += 1
            hist = " ".join(f"{d}:{c}" for d, c in sorted(inside.items()))
            share = f"{total - outside}/{total} ({100 * (total - outside) / max(total, 1):.0f}%)"
            print(f"{length:>12} {share:>18} {items / max(total, 1):>12.1f} {hist:>30}")


def main() -> None:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    def flag(name: str, default: int) -> int:
        return next(
            (int(a.split("=")[1]) for a in flags if a.startswith(f"--{name}=")), default
        )

    def text(name: str, default: str) -> str:
        return next(
            (a.split("=", 1)[1] for a in flags if a.startswith(f"--{name}=")), default
        )

    walks, length = flag("walks", 300), flag("length", 40)

    if "--stack" in flags:
        basis, guard = text("basis", DEFAULT_BASIS), text("guard", "off")
        depth, cap = flag("depth", 5), flag("cap", 2_000_000)
        ops = ops_for_basis(basis)
        print(f"stack machine, basis {basis} ({len(ops)} ops), guard {guard}")
        print("out-degree at each root:")
        for name, program in STACK_ROOTS.items():
            root = stack_root(program)
            print(f"  {name}: {len(stack_successors(root, ops, guard, length))}")
        stack_collapse(ops, guard, walks, length)
        stack_travel(ops, guard, depth, cap, walks)
        return

    depth = flag("depth", 6)
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
