"""Summarise the codex-exec arms (#123) against the luna API row.

Per arm: solved / memorised, solves on `aligned_matches` (whose sampler admits a
head-of-first-list shortcut, see ../stack-pbe-v4-baseline-20260901), tool
attempts and tokens from the event stream, wall clock, solves by witness-length
bucket, and the split between v3's 40 hand-written behaviours and the 180
`gpt-5.6-sol` wrote. The luna arm is paired task-by-task with the API row: same
seed, same order.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
API_ROW = HERE.parent / "stack-pbe-v4-baseline-20260901" / "luna_v4_sample40.json"
BUCKETS = ((1, 10), (11, 20), (21, 40), (41, 1000))
ARMS = (
    ("luna_high_v4_sample40", "gpt-5.6-luna high"),
    ("terra_high_v4_sample40", "gpt-5.6-terra high"),
    ("astra_low_v4_sample40", "gpt-6-astra low"),
    ("astra_medium_v4_sample40", "gpt-6-astra medium"),
    ("astra_high_v4_sample40", "gpt-6-astra high"),
)


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def v3_names(root: Path) -> set[str]:
    sys.path.insert(0, str(root))
    from wandering_light.stack_pbe import load

    return {t.behaviour for t in load(root / "wandering_light/evals/data/stack_pbe_v3")}


def main() -> None:
    root = HERE.parent.parent
    hand_written = v3_names(root)
    api = load_json(API_ROW)
    lengths = {x["behaviour"]: len(x["witness"].split()) for x in api["candidates"]}
    for arm, _ in ARMS:
        path = HERE / f"{arm}.json"
        if path.exists():
            for name, row in load_json(path)["by_behaviour"].items():
                lengths[name] = row["witness_length"]

    def bucket_of(name):
        n = lengths[name]
        return next(
            f"{lo}-{hi if hi < 1000 else ''}" for lo, hi in BUCKETS if lo <= n <= hi
        )

    print(
        "| arm | tasks | solved | memorised | aligned_matches solves | tool attempts | in tok | out tok | reasoning tok | wall |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = {}
    for arm, label in ARMS:
        path = HERE / f"{arm}.json"
        if not path.exists():
            print(f"| {label} | (not run) |")
            continue
        d = load_json(path)
        rows[arm] = d
        c = d["candidates"]
        gen = [g for x in c for g in x.get("generation", [])]
        usage = [g.get("usage", {}) for g in gen]
        tools = sum(g.get("tool_calls", 0) for g in gen)
        wall = sum(g.get("seconds", 0) for g in gen)
        am = sum(
            1
            for x in c
            if x["behaviour"] == "aligned_matches" and x["verdict"] == "solved"
        )
        print(
            f"| {label} | {d['tasks']}{' PARTIAL' if d['partial'] else ''} | **{d['solved']} ({100 * d['solve_rate']:.1f}%)** | {d['memorised']} | {am} | {tools} | "
            f"{sum(u.get('input_tokens', 0) for u in usage):,} | {sum(u.get('output_tokens', 0) for u in usage):,} | "
            f"{sum(u.get('reasoning_output_tokens', 0) for u in usage):,} | {wall:,.0f} s |"
        )

    print("\nSolves by witness length (solved/tasks):\n")
    labels = [f"{lo}-{hi if hi < 1000 else ''}" for lo, hi in BUCKETS]
    print("| arm | " + " | ".join(labels) + " | hand-written (v3) | sol-written |")
    print("|---|" + "---:|" * (len(labels) + 2))
    for arm, label in ARMS:
        if arm not in rows:
            continue
        c = rows[arm]["candidates"]
        cells = []
        for b in labels:
            group = [x for x in c if bucket_of(x["behaviour"]) == b]
            cells.append(f"{sum(x['verdict'] == 'solved' for x in group)}/{len(group)}")
        hand = [x for x in c if x["behaviour"] in hand_written]
        sol = [x for x in c if x["behaviour"] not in hand_written]
        cells.append(f"{sum(x['verdict'] == 'solved' for x in hand)}/{len(hand)}")
        cells.append(f"{sum(x['verdict'] == 'solved' for x in sol)}/{len(sol)}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    api_c = api["candidates"]
    hand = [x for x in api_c if x["behaviour"] in hand_written]
    sol = [x for x in api_c if x["behaviour"] not in hand_written]
    print(
        "| luna high, API row | "
        + " | ".join(
            f"{sum(x['verdict'] == 'solved' for x in api_c if bucket_of(x['behaviour']) == b)}/{sum(bucket_of(x['behaviour']) == b for x in api_c)}"
            for b in labels
        )
        + f" | {sum(x['verdict'] == 'solved' for x in hand)}/{len(hand)} | {sum(x['verdict'] == 'solved' for x in sol)}/{len(sol)} |"
    )

    if "luna_high_v4_sample40" in rows:
        print("\nLuna paired, codex exec against the API row, same task order:\n")
        codex_c = rows["luna_high_v4_sample40"]["candidates"]
        agree = sum(
            (a["verdict"] == "solved") == (b["verdict"] == "solved")
            for a, b in zip(api_c, codex_c, strict=True)
        )
        both = sum(
            a["verdict"] == "solved" and b["verdict"] == "solved"
            for a, b in zip(api_c, codex_c, strict=True)
        )
        api_only = sum(
            a["verdict"] == "solved" and b["verdict"] != "solved"
            for a, b in zip(api_c, codex_c, strict=True)
        )
        codex_only = sum(
            a["verdict"] != "solved" and b["verdict"] == "solved"
            for a, b in zip(api_c, codex_c, strict=True)
        )
        print(
            f"agree on {agree}/{len(codex_c)}; both solve {both}; API only {api_only}; codex only {codex_only}"
        )
        for a, b in zip(api_c, codex_c, strict=True):
            if (a["verdict"] == "solved") != (b["verdict"] == "solved"):
                print(
                    f"  {a['behaviour']} ({lengths[a['behaviour']]} ops): API {a['verdict']}, codex {b['verdict']}"
                )

    print(
        "\nSolved programs on behaviours past 40 ops, and every aligned_matches solve:\n"
    )
    for arm, label in ARMS:
        if arm not in rows:
            continue
        for x in rows[arm]["candidates"]:
            if x["verdict"] == "solved" and (
                lengths[x["behaviour"]] > 40 or x["behaviour"] == "aligned_matches"
            ):
                print(
                    f"  {label}: {x['behaviour']} ({lengths[x['behaviour']]} ops): `{x['candidate']}`"
                )


if __name__ == "__main__":
    main()
