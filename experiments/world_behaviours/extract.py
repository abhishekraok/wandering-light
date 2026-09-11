"""Sample world-authored behaviours (MBPP, LeetCode-easy) inside the mark-28 domain.

training-data-sources (#118). A behaviour is eligible when every test argument
and result is an int or a flat list of ints (no bools, no nesting, no strings),
it takes at most three arguments, and it has enough examples to grade a program
on data it was never shown. MBPP ships three asserts per problem, so extra
examples are drawn by mutating the given arguments under the invariants they
share (sortedness, distinctness, sign, an index staying inside its list),
running the reference solution, and picking a set in which no output value
dominates; LeetCode ships ~100 tests and needs no generation.

Usage:
    uv run --with datasets python experiments/world_behaviours/extract.py \
        --out reports/world-behaviours-probe-20260909/behaviours.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

SEED = 17
PER_SOURCE = 50
MAX_ARGS = 3
SHOWN = 6
MIN_TESTS = 10
MAX_TESTS = 40
MBPP_EXTRA = 24  # generated examples to keep per behaviour
MBPP_CANDIDATES = 96  # draws to choose them from
OUTPUT_SHARE_CAP = 0.4  # no output value may exceed this share of the generated set


def in_domain(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) < 10**6
    if isinstance(value, (list, tuple)):
        return all(isinstance(x, int) and not isinstance(x, bool) for x in value) and (
            len(value) <= 12
        )
    return False


def as_plain(value):
    return list(value) if isinstance(value, tuple) else value


# ---------------------------------------------------------------------------
# Running reference solutions in a subprocess
# ---------------------------------------------------------------------------

RUNNER = textwrap.dedent(
    """
    import json, sys, signal
    src, entry, cases = json.load(sys.stdin)
    ns = {}
    exec(src, ns)
    fn = eval(entry, ns)
    out = []
    for args in cases:
        signal.alarm(2)
        try:
            r = fn(*args)
            if isinstance(r, tuple):
                r = list(r)
            out.append(["ok", r])
        except BaseException as e:  # noqa: BLE001
            out.append(["err", type(e).__name__])
        finally:
            signal.alarm(0)
    json.dump(out, sys.stdout)
    """
)


def run_solution(src: str, entry: str, cases: list[list]) -> list:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", RUNNER],
            input=json.dumps([src, entry, cases]),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return [["err", "crash"]] * len(cases)
        return json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return [["err", "timeout"]] * len(cases)


def infer_invariants(given: list[list]) -> list[dict]:
    """Per-argument constraints every given input satisfies.

    The reference solutions have preconditions their asserts respect (sorted
    input, distinct values, an index within the list). Shape mutation that
    ignores them makes the reference answer its failure value on most drawn
    inputs, which is faithful and uninformative. So each drawn argument keeps
    every property all the given ones share.
    """
    n_args = len(given[0])
    lists = [i for i in range(n_args) if isinstance(given[0][i], list)]
    out = []
    for i in range(n_args):
        column = [args[i] for args in given]
        if isinstance(column[0], list):
            flat = [x for xs in column for x in xs]
            out.append(
                {
                    "kind": "list",
                    "sorted": all(xs == sorted(xs) for xs in column)
                    and any(len(xs) >= 2 for xs in column),
                    "distinct": all(len(set(xs)) == len(xs) for xs in column)
                    and any(len(xs) >= 2 for xs in column),
                    "nonneg": all(x >= 0 for x in flat),
                    "positive": all(x > 0 for x in flat),
                    "min_len": min(len(xs) for xs in column),
                    "max_len": max(len(xs) for xs in column),
                    "hi": max([abs(x) for x in flat] + [5]),
                }
            )
        else:
            spec = {
                "kind": "int",
                "nonneg": all(x >= 0 for x in column),
                "positive": all(x > 0 for x in column),
                "lo": min(column),
                "hi": max(column),
                # an int that never exceeds a list's length is an index or count
                "bounded_by": next(
                    (
                        j
                        for j in lists
                        if all(0 <= args[i] <= len(args[j]) for args in given)
                    ),
                    None,
                ),
                # an int that always occurs in a list is a member lookup
                "member_of": next(
                    (j for j in lists if all(args[i] in args[j] for args in given)),
                    None,
                ),
            }
            out.append(spec)
    return out


def mutate(rng: random.Random, args: list, specs: list[dict]) -> list:
    """Draw fresh arguments with the shapes of ``args`` under ``specs``."""
    fresh: list = []
    for spec in specs:
        if spec["kind"] == "list":
            hi = spec["hi"]
            lo = 1 if spec["positive"] else 0 if spec["nonneg"] else -hi
            n = rng.randint(max(spec["min_len"], 0), spec["max_len"] + 3)
            if spec["distinct"]:
                hi = max(hi, lo + n)
                values = rng.sample(range(lo, hi + 1), n)
            else:
                values = [rng.randint(lo, hi) for _ in range(n)]
            fresh.append(sorted(values) if spec["sorted"] else values)
        else:
            fresh.append(None)  # ints may depend on lists; fill in below
    for i, spec in enumerate(specs):
        if spec["kind"] != "int":
            continue
        j = spec["member_of"]
        if j is not None and fresh[j] and rng.random() < 0.7:
            fresh[i] = rng.choice(fresh[j])
            continue
        j = spec["bounded_by"]
        if j is not None:
            fresh[i] = rng.randint(0, len(fresh[j]))
            continue
        hi = max(abs(spec["hi"]), abs(spec["lo"]), 5)
        lo = 1 if spec["positive"] else 0 if spec["nonneg"] else -hi
        fresh[i] = rng.randint(lo, hi)
    return fresh


def select_examples(
    rng: random.Random, pool: list[dict], want: int, share_cap: float, floor: int
) -> list[dict]:
    """Greedy pick under a cap on how often one output value may recur.

    A held-out set that is mostly one value (a reference answering -1 to every
    input outside its precondition) discriminates nothing; prefer outputs not
    yet represented, and never let one value exceed ``share_cap`` of the set.
    """
    chosen: list[dict] = []
    counts: dict[str, int] = {}
    remaining = list(pool)
    rng.shuffle(remaining)
    while remaining and len(chosen) < want:
        remaining.sort(key=lambda t: counts.get(json.dumps(t["out"]), 0))
        candidate = remaining[0]
        key = json.dumps(candidate["out"])
        if counts.get(key, 0) + 1 > max(2, share_cap * (len(chosen) + 1)):
            break  # every remaining candidate repeats an over-represented output
        remaining.pop(0)
        chosen.append(candidate)
        counts[key] = counts.get(key, 0) + 1
    # A reference with few distinct outputs (last digit of n!) cannot meet the
    # cap; keep it at the floor rather than dropping the behaviour.
    while remaining and len(chosen) < floor:
        chosen.append(remaining.pop(0))
    return chosen


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def parse_mbpp_assert(text: str):
    tree = ast.parse(text.strip())
    node = tree.body[0]
    if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
        return None
    call, (want,) = node.test.left, node.test.comparators
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    args = [ast.literal_eval(a) for a in call.args]
    return call.func.id, args, ast.literal_eval(want)


def mbpp_candidates(rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset(
        "google-research-datasets/mbpp", "full", split="train+test+validation+prompt"
    )
    out = []
    for row in rows:
        try:
            parsed = [parse_mbpp_assert(t) for t in row["test_list"]]
        except (SyntaxError, ValueError):
            continue
        if any(p is None for p in parsed):
            continue
        names = {p[0] for p in parsed}
        if len(names) != 1:
            continue
        (entry,) = names
        tests = [(list(map(as_plain, a)), as_plain(w)) for _, a, w in parsed]
        if not all(
            len(a) <= MAX_ARGS and all(in_domain(x) for x in a) and in_domain(w)
            for a, w in tests
        ):
            continue
        out.append(
            {
                "source": "mbpp",
                "id": f"mbpp-{row['task_id']}",
                "name": entry,
                "description": row["text"],
                "solution": row["code"],
                "entry": entry,
                "tests": [{"args": a, "out": w} for a, w in tests],
            }
        )
    return out


def parse_leetcode_input(text: str) -> list:
    call = ast.parse(f"f({text})", mode="eval").body
    if not isinstance(call, ast.Call):
        raise ValueError(text)
    return [ast.literal_eval(k.value) for k in call.keywords] + [
        ast.literal_eval(a) for a in call.args
    ]


def leetcode_candidates(rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("newfacade/LeetCodeDataset", split="train+test")
    out = []
    for row in rows:
        if row["difficulty"] != "Easy":
            continue
        tests = []
        try:
            for case in row["input_output"]:
                args = parse_leetcode_input(case["input"])
                want = ast.literal_eval(case["output"])
                tests.append((list(map(as_plain, args)), as_plain(want)))
        except (SyntaxError, ValueError, TypeError):
            continue
        if len(tests) < MIN_TESTS:
            continue
        if not all(
            len(a) <= MAX_ARGS and all(in_domain(x) for x in a) and in_domain(w)
            for a, w in tests
        ):
            continue
        out.append(
            {
                "source": "leetcode",
                "id": f"leetcode-{row['task_id']}",
                "name": row["task_id"],
                "description": row["problem_description"][:2000],
                "solution": row["prompt"] + "\n" + row["completion"],
                "entry": row["entry_point"],
                "tests": [{"args": a, "out": w} for a, w in tests[:MAX_TESTS]],
                "tags": row["tags"],
            }
        )
    return out


def extend_mbpp(seed: int, item: dict) -> dict | None:
    """Add generated examples to an MBPP behaviour, checking the given asserts too.

    Seeded per behaviour so a change to the sampler never changes which
    behaviours the source-level shuffle selects.
    """
    rng = random.Random(f"{seed}:{item['id']}")
    given = [t["args"] for t in item["tests"]]
    specs = infer_invariants(given)
    fresh = [mutate(rng, rng.choice(given), specs) for _ in range(MBPP_CANDIDATES)]
    results = run_solution(item["solution"], item["entry"], given + fresh)
    for t, r in zip(item["tests"], results[: len(given)], strict=True):
        if r[0] != "ok" or r[1] != t["out"]:
            return None  # the shipped solution fails its own assert; skip
    seen = {json.dumps(a) for a in given}
    pool = []
    for args, r in zip(fresh, results[len(given) :], strict=True):
        key = json.dumps(args)
        if r[0] != "ok" or not in_domain(r[1]) or key in seen:
            continue
        seen.add(key)
        pool.append({"args": args, "out": r[1], "generated": True})
    extra = select_examples(
        rng, pool, MBPP_EXTRA, OUTPUT_SHARE_CAP, MIN_TESTS - len(item["tests"])
    )
    if len(item["tests"]) + len(extra) < MIN_TESTS:
        return None
    item["tests"] = item["tests"] + extra
    return item


def finalize(item: dict) -> dict:
    tests = item["tests"]
    item["shown"] = list(range(min(SHOWN, len(tests))))
    item["held_out"] = list(range(len(item["shown"]), len(tests)))
    return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-source", type=int, default=PER_SOURCE)
    parser.add_argument(
        "--ids-from",
        type=Path,
        help="restrict to the behaviour ids in this behaviours.jsonl (regenerate a fixed set)",
    )
    args = parser.parse_args(argv)
    keep_ids = (
        {json.loads(line)["id"] for line in args.ids_from.open()}
        if args.ids_from
        else None
    )
    rng = random.Random(args.seed)

    pools = {"mbpp": mbpp_candidates(rng), "leetcode": leetcode_candidates(rng)}
    chosen: list[dict] = []
    stats = {}
    for source, candidates in pools.items():
        pool = sorted(candidates, key=lambda r: r["id"])
        rng.shuffle(pool)
        if keep_ids is not None:
            pool = [item for item in pool if item["id"] in keep_ids]
        kept, tried = [], 0
        for candidate in pool:
            if keep_ids is None and len(kept) >= args.per_source:
                break
            tried += 1
            item = extend_mbpp(args.seed, candidate) if source == "mbpp" else candidate
            if item is None:
                continue
            kept.append(finalize(item))
        stats[source] = {"eligible": len(pool), "tried": tried, "kept": len(kept)}
        chosen.extend(kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in chosen:
            fh.write(json.dumps(item) + "\n")
    print(json.dumps(stats, indent=1))
    print(f"wrote {len(chosen)} behaviours to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
