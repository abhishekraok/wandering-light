"""What plain breadth-first search costs on the PBE corpus.

Before training anything, this is the number to beat. The corpus labels are
certified shortest, so BFS always finds *a* shortest program given enough
nodes; the question is how many nodes, and that is what a learned policy has to
improve on. A solver that "solves 90%" means nothing without this curve, since
search alone may already do that for free.

Search is the same behaviour-space expansion the corpus generator uses: carry
the tuple of stacks, one per example, and stop when it matches the target.

    uv run python experiments/stack_machine_baseline.py corpus.jsonl.gz
"""

import gzip
import json
import statistics
import sys
import time
from collections import defaultdict

from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, parse_stack, step_raw
from wandering_light.stack_machine_corpus import OPS, behaviour_key

BUDGETS = (100, 1_000, 10_000, 100_000, 1_000_000)


def solve(inputs: tuple, target: tuple, budget: int, max_depth: int = 12) -> dict:
    """BFS over behaviours until the target appears or the node budget runs out.

    `nodes` counts distinct behaviours expanded, which is the unit a learned
    policy would be reducing.
    """
    goal = behaviour_key(target)
    start = behaviour_key(inputs)
    if start == goal:
        return {"solved": True, "nodes": 0, "length": 0}
    seen = {start}
    frontier = [inputs]
    nodes = 0
    for depth in range(1, max_depth + 1):
        nxt = []
        for behaviour in frontier:
            nodes += 1
            if nodes > budget:
                return {"solved": False, "nodes": nodes, "length": None}
            for op in OPS:
                moved = []
                ok = True
                for stack in behaviour:
                    outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
                    if isinstance(outcome, str):
                        ok = False
                        break
                    moved.append(outcome[0])
                if not ok:
                    continue
                key = behaviour_key(tuple(moved))
                if key == goal:
                    return {"solved": True, "nodes": nodes, "length": depth}
                if key not in seen:
                    seen.add(key)
                    nxt.append(tuple(moved))
        frontier = nxt
        if not frontier:
            break
    return {"solved": False, "nodes": nodes, "length": None}


def main() -> None:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("wandering_light/evals/data/stack_pbe_v2.jsonl.gz")
    )
    split = sys.argv[2] if len(sys.argv) > 2 else "eval"
    with gzip.open(path, "rt") as handle:
        tasks = [json.loads(line) for line in handle]
    tasks = [t for t in tasks if t["split"] == split]
    print(f"{len(tasks)} {split} tasks from {path}", file=sys.stderr)

    largest = max(BUDGETS)
    results = []
    started = time.time()
    for n, task in enumerate(tasks):
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        target = tuple(parse_stack(s) for s in task["outputs"])
        outcome = solve(inputs, target, largest)
        results.append({**outcome, "label": task["length"], "family": task["family"]})
        if n % 50 == 0:
            print(f"  {n}/{len(tasks)}  {time.time() - started:.0f}s", file=sys.stderr)

    solved = [r for r in results if r["solved"]]
    nodes = sorted(r["nodes"] for r in solved)
    print(f"\nBFS baseline on {len(results)} {split} tasks\n")
    print("On THIS corpus the number to beat is NODES, not accuracy: it stops at")
    print("depth 6, where search still reaches everything, so solve rate saturates.")
    print("That is a fact about this corpus, not about forward generation -- see")
    print("deep-corpus-v1 (#37), where BFS fails 12.5% of tasks after an exhaustive")
    print("sweep. Generation pays b^d once per root; search pays it once per task.\n")
    if nodes:
        print(
            f"{'nodes to solution':>20}  median {nodes[len(nodes) // 2]:>9,}"
            f"   p90 {nodes[int(len(nodes) * 0.9)]:>9,}"
            f"   max {nodes[-1]:>9,}   total {sum(nodes):>12,}"
        )
    print()
    print(f"{'node budget':>12}  {'solved':>8}  {'rate':>7}")
    for budget in BUDGETS:
        hit = sum(1 for r in results if r["solved"] and r["nodes"] <= budget)
        print(f"{budget:>12,}  {hit:>8}  {hit / len(results) * 100:>6.1f}%")

    print(
        f"\n{'shortest':>9}  {'tasks':>6}  {'solved':>7}  {'median nodes':>13}  {'max nodes':>10}"
    )
    by_length = defaultdict(list)
    for r in results:
        by_length[r["label"]].append(r)
    for length in sorted(by_length):
        rows = by_length[length]
        got = [r["nodes"] for r in rows if r["solved"]]
        med = f"{statistics.median(got):,.0f}" if got else "-"
        mx = f"{max(got):,}" if got else "-"
        print(f"{length:>9}  {len(rows):>6}  {len(got):>7}  {med:>13}  {mx:>10}")

    wrong = [r for r in results if r["solved"] and r["length"] != r["label"]]
    print(f"\nBFS length disagreed with the certified label: {len(wrong)}")
    out = sys.argv[3] if len(sys.argv) > 3 else "baseline.json"
    with open(out, "w") as handle:
        json.dump({"split": split, "results": results}, handle)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
