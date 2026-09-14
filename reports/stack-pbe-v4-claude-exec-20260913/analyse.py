"""Summarise the Opus effort arms (opus-effort-curve, #137) against astra's.

Per arm: solved / memorised / shown-only / illegal, the `aligned_matches`
sampler shortcut (see ../stack-pbe-v4-baseline-20260901), tool calls written
as text, thinking tokens per task (median, max), wall clock, solves by
witness-length bucket and by author (v3's 40 hand-written behaviours against
the 180 `gpt-5.6-sol` wrote). Each Opus arm is paired task-by-task with astra
low and astra high from ../stack-pbe-v4-codex-exec-20260910: same seed, same
order. Prints Markdown.
"""

import itertools
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
CODEX = HERE.parent / "stack-pbe-v4-codex-exec-20260910"
API_ROW = HERE.parent / "stack-pbe-v4-baseline-20260901" / "luna_v4_sample40.json"
BUCKETS = ((1, 10), (11, 20), (21, 40), (41, 1000))
ARMS = (
    ("opus_low_v4_sample40", "claude-opus-5 low"),
    ("opus_high_v4_sample40", "claude-opus-5 high"),
    ("opus_xhigh_v4_sample40", "claude-opus-5 xhigh"),
)
REFERENCE = (
    (CODEX / "astra_low_v4_sample40.json", "gpt-6-astra low"),
    (CODEX / "astra_high_v4_sample40.json", "gpt-6-astra high"),
)


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def v3_names(root: Path) -> set[str]:
    sys.path.insert(0, str(root))
    from wandering_light.stack_pbe import load

    return {t.behaviour for t in load(root / "wandering_light/evals/data/stack_pbe_v3")}


def solved(x) -> bool:
    return x["verdict"] == "solved"


def main() -> None:
    root = HERE.parent.parent
    hand_written = v3_names(root)
    lengths = {}
    rows: dict[str, dict] = {}
    for path, label in REFERENCE:
        rows[label] = load_json(path)
    for arm, label in ARMS:
        path = HERE / f"{arm}.json"
        if path.exists():
            rows[label] = load_json(path)
    for d in rows.values():
        for name, row in d["by_behaviour"].items():
            lengths[name] = row["witness_length"]
    if API_ROW.exists():
        for x in load_json(API_ROW)["candidates"]:
            lengths.setdefault(x["behaviour"], len(x["witness"].split()))

    def bucket_of(name):
        n = lengths[name]
        return next(
            f"{lo}-{hi if hi < 1000 else ''}" for lo, hi in BUCKETS if lo <= n <= hi
        )

    print(
        "| arm | tasks | solved | memorised | shown-only | `aligned_matches` shortcut | illegal | text tool calls | thinking tok/task, median (max) | in tok | out tok | wall | model |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label, d in rows.items():
        c = d["candidates"]
        gen = [g for x in c for g in x.get("generation", [])]
        usage = [g.get("usage", {}) for g in gen]
        reasoning = [u.get("reasoning_output_tokens", 0) for u in usage]
        text_calls = sum(1 for g in gen if g.get("text_tool_call"))
        tools = sum(g.get("tool_calls", 0) for g in gen)
        wall = sum(g.get("seconds", 0) for g in gen)
        models = sorted({m for g in gen for m in g.get("model", [])})
        am = sum(1 for x in c if x["behaviour"] == "aligned_matches" and solved(x))
        med = (
            f"{statistics.median(reasoning):,.0f} ({max(reasoning):,})"
            if reasoning
            else "—"
        )
        print(
            f"| {label} | {d['tasks']}{' PARTIAL' if d['partial'] else ''} | **{d['solved']} ({100 * d['solve_rate']:.1f}%)** | {d['memorised']} | {100 * d['shown_rate']:.1f}% | {am} → {d['solved'] - am} real | {d['illegal']} | {text_calls}{f' (+{tools} turns)' if tools else ''} | {med} | "
            f"{sum(u.get('input_tokens', 0) for u in usage):,} | {sum(u.get('output_tokens', 0) for u in usage):,} | {wall:,.0f} s | {', '.join(models) or '—'} |"
        )

    print("\nSolves by witness length (solved/tasks), and by author:\n")
    labels = [f"{lo}-{hi if hi < 1000 else ''}" for lo, hi in BUCKETS]
    print(
        "| arm | "
        + " | ".join(labels)
        + " | hand-written (v3's 40) | `gpt-5.6-sol`-written |"
    )
    print("|---|" + "---:|" * (len(labels) + 2))
    for label, d in rows.items():
        c = d["candidates"]
        cells = []
        for b in labels:
            group = [x for x in c if bucket_of(x["behaviour"]) == b]
            cells.append(f"{sum(map(solved, group))}/{len(group)}")
        hand = [x for x in c if x["behaviour"] in hand_written]
        sol = [x for x in c if x["behaviour"] not in hand_written]
        cells.append(f"{sum(map(solved, hand))}/{len(hand)}")
        cells.append(f"{sum(map(solved, sol))}/{len(sol)}")
        print(f"| {label} | " + " | ".join(cells) + " |")

    opus = [label for _, label in ARMS if label in rows]
    if len(opus) > 1:
        print(
            "\nThinking tokens per task, paired across Opus arms (same task order):\n"
        )
        for a, b in itertools.pairwise(opus):
            ra = [
                g.get("usage", {}).get("reasoning_output_tokens", 0)
                for x in rows[a]["candidates"]
                for g in x.get("generation", [])[:1]
            ]
            rb = [
                g.get("usage", {}).get("reasoning_output_tokens", 0)
                for x in rows[b]["candidates"]
                for g in x.get("generation", [])[:1]
            ]
            more = sum(y > x for x, y in zip(ra, rb, strict=True))
            print(
                f"  {b} thinks more than {a} on {more}/{len(ra)} tasks; median ratio {statistics.median(y / x if x else float('inf') for x, y in zip(ra, rb, strict=True)):.2f}"
            )

    print(
        "\nPaired verdicts, same task order (agree / both solve / left only / right only):\n"
    )
    pairs = [
        (a, b)
        for a in opus
        for b in [r for _, r in REFERENCE] + [o for o in opus if o != a]
    ]
    seen = set()
    for a, b in pairs:
        if (b, a) in seen or a == b:
            continue
        seen.add((a, b))
        ca, cb = rows[a]["candidates"], rows[b]["candidates"]
        agree = sum(solved(x) == solved(y) for x, y in zip(ca, cb, strict=True))
        both = sum(solved(x) and solved(y) for x, y in zip(ca, cb, strict=True))
        left = sum(solved(x) and not solved(y) for x, y in zip(ca, cb, strict=True))
        right = sum(solved(y) and not solved(x) for x, y in zip(ca, cb, strict=True))
        print(
            f"  {a} vs {b}: agree {agree}/{len(ca)}, both {both}, {a} only {left}, {b} only {right}"
        )
        for x, y in zip(ca, cb, strict=True):
            if solved(x) != solved(y):
                print(
                    f"    {x['behaviour']} ({lengths[x['behaviour']]} ops): {a} {x['verdict']}, {b} {y['verdict']}"
                )

    print("\nOpus answers flagged as tool calls written as text:\n")
    for label in opus:
        for x in rows[label]["candidates"]:
            if any(g.get("text_tool_call") for g in x.get("generation", [])):
                print(
                    f"  {label}: {x['behaviour']} ({lengths[x['behaviour']]} ops): `{x['candidate'][:160]}`"
                )

    print("\nSolved programs past 40 ops, and every aligned_matches solve:\n")
    for label in opus:
        for x in rows[label]["candidates"]:
            if solved(x) and (
                lengths[x["behaviour"]] > 40 or x["behaviour"] == "aligned_matches"
            ):
                print(
                    f"  {label}: {x['behaviour']} ({lengths[x['behaviour']]} ops): `{x['candidate']}`"
                )


if __name__ == "__main__":
    main()
