"""Sample world-authored behaviours (MBPP, LeetCode, CodeARC) inside the mark-28 domain.

training-data-sources (#118), world-authoring-500 (#127). A behaviour is
eligible when every test argument and result is an int or a flat list of ints
(no bools, no nesting, no strings), it takes at most three arguments, and it
has enough examples to grade a program on data it was never shown. Every
source ships a reference solution, so each behaviour gets extra examples
drawn by mutating the given arguments: most keep the properties the given
inputs share (sortedness, distinctness, sign, an index staying inside its
list), the rest deliberately break one, and the reference answers. MBPP
ships three asserts and CodeARC ten invocations per problem; LeetCode ships
~100 tests, of which the first 40 are kept before extension. (The 2026-09-09
probe extended MBPP only; its behaviours are frozen under ``reports/``.)

Usage:
    uv run --with 'datasets>=4.0' python experiments/world_behaviours/extract.py \
        --out reports/world-authoring-20260910/behaviours.jsonl \
        --take mbpp=36,codearc=34,leetcode=30 --leetcode-difficulty Medium,Hard \
        --exclude reports/world-behaviours-library-20260910/behaviours.jsonl
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
EXTRA = 24  # generated examples to keep per behaviour
CANDIDATES = 96  # draws to choose them from
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


RELAXABLE = (
    "sorted",
    "distinct",
    "nonneg",
    "positive",
    "min_len",
    "bounded_by",
    "member_of",
)
STRICT_SHARE = 0.6  # candidates drawn with every inferred property kept


def infer_invariants(given: list[list]) -> list[dict]:
    """Per-argument properties every given input shares.

    These are hypotheses, not preconditions: three asserts that happen to use
    distinct elements say nothing about duplicates. A reference with a real
    precondition (a sorted list, an index inside it) answers its failure value
    when the property is broken, and ``select_examples`` bounds how much of
    the set such answers may take; a reference without one just answers, and
    the example then discriminates a shortcut (``dedup sort`` for ``sort``)
    that inputs drawn only under the hypothesis would let through. So most
    candidates keep every property and the rest relax one (``mutate``).
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


def holding_properties(specs: list[dict]) -> list[tuple[int, str]]:
    """The (argument, property) pairs the given inputs happen to satisfy."""

    def holds(spec: dict, key: str) -> bool:
        if key not in spec:
            return False
        if key in ("bounded_by", "member_of"):
            return spec[key] is not None  # an argument index; 0 is a relation
        return bool(spec[key])

    return [
        (i, key)
        for i, spec in enumerate(specs)
        for key in RELAXABLE
        if holds(spec, key)
    ]


def relax(specs: list[dict], which: tuple[int, str]) -> list[dict]:
    """Copy ``specs`` with one inferred property switched off and marked to violate.

    Merely not enforcing a property leaves it to chance (a random list of
    seven values from nine is often distinct anyway); the hypothesis is only
    tested by an input that breaks it, so ``mutate`` forces the violation.
    """
    i, key = which
    relaxed = [dict(spec) for spec in specs]
    if key in ("bounded_by", "member_of"):
        relaxed[i][f"{key}_was"] = relaxed[i][key]
    relaxed[i][key] = (
        0 if key == "min_len" else None if key in ("bounded_by", "member_of") else False
    )
    relaxed[i]["violate"] = key
    return relaxed


def violate_list(
    rng: random.Random, values: list[int], key: str, spec: dict
) -> list[int]:
    hi = spec["hi"]
    if key == "distinct":
        if len(values) < 2:
            values = values + values[:1] or [rng.randint(0, hi)] * 2
        else:
            i, j = rng.sample(range(len(values)), 2)
            values[i] = values[j]
    elif key == "sorted":
        if len(values) < 2:
            values = [*values, rng.randint(0, hi)]
        if values == sorted(values):
            values = (
                list(reversed(values))
                if values[0] != values[-1]
                else [values[0] + 1, *values]
            )
    elif key == "nonneg":
        values = values or [0]
        values[rng.randrange(len(values))] = -rng.randint(1, hi)
    elif key == "positive":
        values = values or [1]
        values[rng.randrange(len(values))] = 0
    elif key == "min_len":
        values = values[: rng.randint(0, max(spec["min_len"] - 1, 0))]
    return values


def violate_int(
    rng: random.Random, value: int, key: str, spec: dict, fresh: list
) -> int:
    if key == "nonneg":
        return -rng.randint(1, max(abs(spec["hi"]), 1))
    if key == "positive":
        return 0
    if key == "bounded_by":
        n = len(fresh[spec["bounded_by_was"]])
        return rng.choice([-1, n + 1, n + rng.randint(2, 5)])
    if key == "member_of":
        members = set(fresh[spec["member_of_was"]])
        candidate = max(members, default=0) + 1
        while candidate in members:
            candidate += 1
        return candidate
    return value


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
            values = sorted(values) if spec["sorted"] else values
            if spec.get("violate"):
                values = violate_list(rng, values, spec["violate"], spec)
            fresh.append(values)
        else:
            fresh.append(None)  # ints may depend on lists; fill in below
    for i, spec in enumerate(specs):
        if spec["kind"] != "int":
            continue
        j = spec["member_of"]
        if j is not None and fresh[j] and rng.random() < 0.7:
            fresh[i] = rng.choice(fresh[j])
        elif spec["bounded_by"] is not None:
            fresh[i] = rng.randint(0, len(fresh[spec["bounded_by"]]))
        else:
            hi = max(abs(spec["hi"]), abs(spec["lo"]), 5)
            lo = 1 if spec["positive"] else 0 if spec["nonneg"] else -hi
            fresh[i] = rng.randint(lo, hi)
        if spec.get("violate"):
            fresh[i] = violate_int(rng, fresh[i], spec["violate"], spec, fresh)
    return fresh


def select_relaxed(
    rng: random.Random, pool: list[dict], want: int, share_cap: float
) -> list[dict]:
    """Round-robin over the relaxed property, each turn diversity-capped."""
    by_key: dict[str, list[dict]] = {}
    for t in pool:
        by_key.setdefault(t["relaxed"], []).append(t)
    chosen: list[dict] = []
    counts: dict[str, int] = {}
    keys = sorted(by_key)
    while keys and len(chosen) < want:
        for key in list(keys):
            if len(chosen) >= want:
                break
            candidates = by_key[key]
            candidates.sort(key=lambda t: counts.get(json.dumps(t["out"]), 0))
            picked = None
            for t in candidates:
                out_key = json.dumps(t["out"])
                if counts.get(out_key, 0) + 1 <= max(2, share_cap * (len(chosen) + 1)):
                    picked = t
                    break
            if picked is None:
                keys.remove(key)
                continue
            candidates.remove(picked)
            chosen.append(picked)
            counts[json.dumps(picked["out"])] = (
                counts.get(json.dumps(picked["out"]), 0) + 1
            )
            if not candidates:
                keys.remove(key)
    return chosen


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


def leetcode_candidates(
    rng: random.Random, difficulties: tuple[str, ...] = ("Easy",)
) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("newfacade/LeetCodeDataset", split="train+test")
    out = []
    for row in rows:
        if row["difficulty"] not in difficulties:
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
                "difficulty": row["difficulty"].lower(),
            }
        )
    return out


def parse_codearc_invocation(source: str, printed: str):
    """``print('Result 1: ' + str(f(1, 2)))`` printed ``Result 1: 3`` -> (f, [1, 2], 3)."""
    expr = ast.parse(source.strip()).body[0].value
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)):
        raise ValueError(source)
    (concat,) = expr.args
    call = concat.right.args[0]
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(source)
    prefix = ast.literal_eval(concat.left)
    if not printed.startswith(prefix):
        raise ValueError(printed)
    args = [ast.literal_eval(a) for a in call.args]
    return call.func.id, args, ast.literal_eval(printed[len(prefix) :])


def codearc_candidates(rng: random.Random) -> list[dict]:
    """CodeARC's 1,114 functions with their ten recorded invocations each.

    The ``annotated`` rows carry the docstring and the real function name; the
    ``anonymous`` twins (same ``problem_id``) are the same function renamed.
    """
    from datasets import load_dataset

    problems = {
        row["problem_id"]: row
        for row in load_dataset("anjiangwei/CodeARC-Problems", split="train")
        if row["type"] == "annotated"
    }
    invocations: dict[int, list] = {}
    for row in load_dataset("anjiangwei/CodeARC-Invocations", split="train"):
        if row["type"] != "annotated":
            continue
        try:
            parsed = (
                None
                if row["errored"]
                else parse_codearc_invocation(row["input"], row["output"])
            )
        except (SyntaxError, ValueError, AttributeError, IndexError):
            parsed = None  # strings, floats, nested structures: out of domain
        invocations.setdefault(row["problem_id"], []).append(parsed)
    out = []
    for pid, row in sorted(problems.items()):
        parsed = invocations.get(pid, [])
        if not parsed or any(p is None for p in parsed):
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
        functions = [
            node
            for node in ast.walk(ast.parse(row["code"]))
            if isinstance(node, ast.FunctionDef) and node.name == entry
        ]
        if not functions:
            continue
        out.append(
            {
                "source": "codearc",
                "id": f"codearc-{pid}",
                "name": entry,
                "description": ast.get_docstring(functions[0]) or entry,
                "solution": row["code"],
                "entry": entry,
                "tests": [{"args": a, "out": w} for a, w in tests],
            }
        )
    return out


def extend(seed: int, item: dict) -> dict | None:
    """Add generated examples to a behaviour, checking the given tests too.

    Seeded per behaviour so a change to the sampler never changes which
    behaviours the source-level shuffle selects.
    """
    rng = random.Random(f"{seed}:{item['id']}")
    given = [t["args"] for t in item["tests"]]
    specs = infer_invariants(given)
    strict = [
        mutate(rng, rng.choice(given), specs)
        for _ in range(int(CANDIDATES * STRICT_SHARE))
    ]
    # Relax each holding property in turn, so every hypothesis is tested
    # equally often rather than at the mercy of a random draw.
    holding = holding_properties(specs)
    relaxed_keys = [
        holding[n % len(holding)] if holding else None
        for n in range(CANDIDATES - len(strict))
    ]
    relaxed = [
        mutate(rng, rng.choice(given), relax(specs, which) if which else specs)
        for which in relaxed_keys
    ]
    fresh = strict + relaxed
    results = run_solution(item["solution"], item["entry"], given + fresh)
    for t, r in zip(item["tests"], results[: len(given)], strict=True):
        if r[0] != "ok" or r[1] != t["out"]:
            return None  # the shipped solution fails its own assert; skip
    seen = {json.dumps(a) for a in given}
    pools: dict[bool, list[dict]] = {True: [], False: []}
    for index, (args, r) in enumerate(zip(fresh, results[len(given) :], strict=True)):
        key = json.dumps(args)
        if r[0] != "ok" or not in_domain(r[1]) or key in seen:
            continue
        seen.add(key)
        is_relaxed = index >= len(strict)
        which = relaxed_keys[index - len(strict)] if is_relaxed else None
        pools[is_relaxed].append(
            {
                "args": args,
                "out": r[1],
                "generated": True,
                "relaxed": f"{which[0]}:{which[1]}" if which else None,
            }
        )
    # Relaxed candidates that the reference accepted carry the discriminating
    # cases, so they get their own quota, spread evenly over the properties.
    quota = int(EXTRA * (1 - STRICT_SHARE))
    extra = select_relaxed(rng, pools[True], quota, OUTPUT_SHARE_CAP)
    extra += select_examples(
        rng,
        pools[False],
        EXTRA - len(extra),
        OUTPUT_SHARE_CAP,
        MIN_TESTS - len(item["tests"]) - len(extra),
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
        "--take",
        default="mbpp,leetcode",
        help="sources to draw from, optionally with counts: mbpp=36,codearc=34,leetcode=30",
    )
    parser.add_argument(
        "--leetcode-difficulty",
        default="Easy",
        help="comma-separated: Easy,Medium,Hard",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="behaviours.jsonl whose ids are skipped (already authored); repeatable",
    )
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
    exclude = {json.loads(line)["id"] for path in args.exclude for line in path.open()}
    take: dict[str, int] = {}
    for spec in args.take.split(","):
        name, _, count = spec.partition("=")
        take[name.strip()] = int(count) if count else args.per_source
    rng = random.Random(args.seed)

    loaders = {
        "mbpp": lambda: mbpp_candidates(rng),
        "leetcode": lambda: leetcode_candidates(
            rng, tuple(args.leetcode_difficulty.split(","))
        ),
        "codearc": lambda: codearc_candidates(rng),
    }
    chosen: list[dict] = []
    stats = {}
    for source, want in take.items():
        pool = sorted(loaders[source](), key=lambda r: r["id"])
        rng.shuffle(pool)
        pool = [item for item in pool if item["id"] not in exclude]
        if keep_ids is not None:
            pool = [item for item in pool if item["id"] in keep_ids]
        kept, tried = [], 0
        for candidate in pool:
            if keep_ids is None and len(kept) >= want:
                break
            tried += 1
            item = extend(args.seed, candidate)
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
