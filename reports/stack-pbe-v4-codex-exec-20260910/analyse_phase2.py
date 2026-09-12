"""Phase 2 of #123: astra low on the full 1,800-task eval split, six shards merged.

Prints the split-level counts, the sample-versus-full delta, solves by
witness-length bucket, and the question phase 2 exists for: hand-written
(v3's 40) against `gpt-5.6-sol`-written behaviours at matched witness length,
for astra and for the BFS 10,000-node full-split row as the search-side
control. Then every behaviour, sorted by witness length.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
BASELINE = HERE.parent / "stack-pbe-v4-baseline-20260901"
BUCKETS = ((1, 10), (11, 20), (21, 40), (41, 80), (81, 1000))


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def merged(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(HERE.glob(pattern)):
        d = load_json(path)
        assert not d["partial"], path
        rows.extend(d["candidates"])
    rows.sort(key=lambda x: x["task_index"])
    return rows


def label(lo, hi):
    return f"{lo}-{hi}" if hi < 1000 else f"{lo}+"


def bucket_of(n):
    return next(label(lo, hi) for lo, hi in BUCKETS if lo <= n <= hi)


def rate(rows):
    return f"{sum(x['verdict'] == 'solved' for x in rows)}/{len(rows)}" if rows else "—"


def main() -> None:
    root = HERE.parent.parent
    sys.path.insert(0, str(root))
    from wandering_light.stack_pbe import load

    hand = {t.behaviour for t in load(root / "wandering_light/evals/data/stack_pbe_v3")}
    tasks = load(root / "wandering_light/evals/data/stack_pbe_v4")
    length = {t.behaviour: t.length for t in tasks}

    astra = merged("astra_low_v4_full_shard*.json")
    bfs = load_json(BASELINE / "bfs_v4_10k_full.json")["candidates"]
    assert len(astra) == 1800 and len(bfs) == 1800
    gen = [g for x in astra for g in x.get("generation", [])]
    tools = sum(g.get("tool_calls", 0) for g in gen)
    usage = [g.get("usage", {}) for g in gen]
    v = Counter(x["verdict"] for x in astra)
    shown_fit = v["solved"] + v["memorised"]
    print(
        f"astra low, full split: solved {v['solved']}/1800 ({100 * v['solved'] / 1800:.1f}%), "
        f"memorised {v['memorised']}, shown-only {100 * shown_fit / 1800:.1f}%, "
        f"tool attempts {tools}, input tok {sum(u.get('input_tokens', 0) for u in usage):,}, "
        f"output tok {sum(u.get('output_tokens', 0) for u in usage):,}, "
        f"reasoning tok {sum(u.get('reasoning_output_tokens', 0) for u in usage):,}, "
        f"wall {sum(g.get('seconds', 0) for g in gen):,.0f} s"
    )
    am = [
        x
        for x in astra
        if x["behaviour"] == "aligned_matches" and x["verdict"] == "solved"
    ]
    print(f"aligned_matches solves (sampler shortcut): {len(am)}")
    vb = Counter(x["verdict"] for x in bfs)
    print(
        f"BFS 10k, full split: solved {vb['solved']}/1800, memorised {vb['memorised']}"
    )

    labels = [label(lo, hi) for lo, hi in BUCKETS]
    print("\nSolves by witness length (solved/tasks):\n")
    print("| solver | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(labels))
    for name, rows in (("astra low", astra), ("BFS 10k", bfs)):
        print(
            f"| {name} | "
            + " | ".join(
                rate([x for x in rows if bucket_of(length[x["behaviour"]]) == b])
                for b in labels
            )
            + " |"
        )

    print(
        "\nAuthorship at matched witness length (solved/tasks; behaviours in parentheses):\n"
    )
    print("| solver | bucket | hand-written (v3) | sol-written |")
    print("|---|---|---:|---:|")
    for name, rows in (("astra low", astra), ("BFS 10k", bfs)):
        for b in labels:
            h = [
                x
                for x in rows
                if bucket_of(length[x["behaviour"]]) == b and x["behaviour"] in hand
            ]
            s = [
                x
                for x in rows
                if bucket_of(length[x["behaviour"]]) == b and x["behaviour"] not in hand
            ]
            hb = len({x["behaviour"] for x in h})
            sb = len({x["behaviour"] for x in s})
            print(f"| {name} | {b} | {rate(h)} ({hb}) | {rate(s)} ({sb}) |")
        h = [x for x in rows if x["behaviour"] in hand]
        s = [x for x in rows if x["behaviour"] not in hand]
        print(
            f"| {name} | all | {rate(h)} ({len({x['behaviour'] for x in h})}) | {rate(s)} ({len({x['behaviour'] for x in s})}) |"
        )

    print("\nEvery eval behaviour, astra low then BFS 10k (solved/40, memorised):\n")
    print("| behaviour | ops | author | astra solved | astra memorised | BFS solved |")
    print("|---|---:|---|---:|---:|---:|")
    by = {}
    for x in astra:
        by.setdefault(x["behaviour"], []).append(x)
    byb = {}
    for x in bfs:
        byb.setdefault(x["behaviour"], []).append(x)
    for name in sorted(by, key=lambda n: (length[n], n)):
        rows = by[name]
        print(
            f"| `{name}` | {length[name]} | {'v3' if name in hand else 'sol'} | "
            f"{sum(x['verdict'] == 'solved' for x in rows)} | {sum(x['verdict'] == 'memorised' for x in rows)} | "
            f"{sum(x['verdict'] == 'solved' for x in byb[name])} |"
        )

    print(
        "\nLongest behaviours astra solved at least once, with one solving program each:\n"
    )
    for name in sorted(by, key=lambda n: -length[n]):
        solved = [x for x in by[name] if x["verdict"] == "solved"]
        if solved and length[name] > 40:
            print(
                f"  {name} ({length[name]} ops, {len(solved)}/40): `{solved[0]['candidate']}`"
            )


if __name__ == "__main__":
    main()
