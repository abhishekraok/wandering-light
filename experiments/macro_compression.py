"""macro-compression-check (#144): do the authoring macros compress v4 witnesses?

Static, no search, no training. For every distinct witness in ``stack_pbe_v4``
(train, eval, the #101 cohort) and ``stack_pbe_world_v3`` (train, eval):

- the fewest tokens over ops U macros that expand to the witness (dynamic
  programming over positions), and hops saved against the raw length;
- which macros hit, ranked by hops saved;
- effective branching at states along witness prefixes, primitives against
  primitives U macros, an action legal iff every shown example accepts it
  (plain ``step_raw``, no #70 guard), and the closed-form search-cost estimate
  highways (#96) step 3 uses, ``d · log2(b_eff)``, before and after.

    uv run python experiments/macro_compression.py --out reports/macro-compression-<date>
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments" / "world_behaviours"))

from verify import expand, load_macros  # noqa: E402

from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_machine import (  # noqa: E402
    DEFAULT_FUEL_LIMIT,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS, Task, load  # noqa: E402

MACRO_FILE = REPO / "experiments" / "world_behaviours" / "macros.txt"
MANIFESTS = {
    "v4": REPO / "wandering_light/evals/data/stack_pbe_v4/manifest.json",
    "world": REPO / "wandering_light/evals/data/stack_pbe_world_v3/manifest.json",
}
# compute_optimal_mark28.cohort, restated so this script does not import torch.
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))
TASKS_PER_BUCKET = 60
COHORT_SEED = 17
STATES_PER_WITNESS = 8
BRANCHING_SEED = 0


def bucket_of(length: int) -> str:
    for low, high in BUCKETS:
        if low <= length <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    raise ValueError(length)


def cohort(tasks: list[Task]) -> list[Task]:
    eval_tasks = [(i, t) for i, t in enumerate(tasks) if t.split == "eval"]
    random.Random(COHORT_SEED).shuffle(eval_tasks)
    taken: dict[str, list[Task]] = defaultdict(list)
    for _index, task in eval_tasks:
        bucket = bucket_of(task.length)
        if len(taken[bucket]) < TASKS_PER_BUCKET:
            taken[bucket].append(task)
    return [t for low, _high in BUCKETS for t in taken[bucket_of(low)]]


# ---- compression --------------------------------------------------------------


def primitive_macros(macros: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    """Every macro fully expanded to basis ops."""
    out = {}
    for name, body in macros.items():
        ops = tuple(expand(body, macros))
        unknown = sorted(set(ops) - set(OP_BY_NAME))
        if unknown:
            raise ValueError(f"macro {name!r} expands to non-basis tokens {unknown}")
        out[name] = ops
    return out


def compress(
    program: tuple[str, ...], by_first: dict[str, list[tuple[str, tuple[str, ...]]]]
):
    """Fewest tokens over ops U macros expanding to ``program``; returns (tokens, macros used)."""
    n = len(program)
    best = [math.inf] * (n + 1)
    choice: list[tuple[str, int] | None] = [None] * (n + 1)
    best[n] = 0
    for i in range(n - 1, -1, -1):
        best[i], choice[i] = best[i + 1] + 1, (program[i], 1)
        for name, ops in by_first.get(program[i], ()):
            k = len(ops)
            if i + k <= n and program[i : i + k] == ops and best[i + k] + 1 < best[i]:
                best[i], choice[i] = best[i + k] + 1, (name, k)
    tokens, i = [], 0
    while i < n:
        name, k = choice[i]
        tokens.append(name)
        i += k
    return tokens, [t for t in tokens if t not in OP_BY_NAME]


# ---- branching ----------------------------------------------------------------


def apply(product: tuple[tuple, ...], ops: tuple[str, ...]):
    """The product after ``ops`` on every example, or None if any example refuses."""
    for name in ops:
        op = OP_BY_NAME[name]
        moved = []
        for stack in product:
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                return None
            moved.append(outcome[0])
        product = tuple(moved)
    return product


def branching(task: Task, macros: dict[str, tuple[str, ...]], rng: random.Random):
    """(legal primitives, legal macros) at up to STATES_PER_WITNESS prefix states."""
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    prefixes = list(range(len(task.program)))
    if len(prefixes) > STATES_PER_WITNESS:
        prefixes = sorted(rng.sample(prefixes, STATES_PER_WITNESS))
    states, done = [], 0
    for k in prefixes:
        product = apply(product, task.program[done:k])
        done = k
        if product is None:
            break
        prim = sum(apply(product, (op.name,)) is not None for op in OPS)
        mac = sum(apply(product, ops) is not None for ops in macros.values())
        states.append((prim, mac))
    return states


_MACROS: dict[str, tuple[str, ...]] = {}


def _init(macros):
    _MACROS.update(macros)


def _branching(args):
    task, seed = args
    started = time.perf_counter()
    states = branching(task, _MACROS, random.Random(seed))
    return task.behaviour, states, time.perf_counter() - started


# ---- reporting ----------------------------------------------------------------


def summarise(name: str, tasks: list[Task], by_first, macros, rng) -> dict:
    programs = {t.program: t for t in tasks}  # one witness per distinct program
    rows, used = [], Counter()
    for program, task in programs.items():
        tokens, hits = compress(program, by_first)
        covered = sum(len(macros[m]) for m in hits)
        for m in hits:
            used[m] += len(macros[m]) - 1
        rows.append(
            {
                "behaviour": task.behaviour,
                "length": len(program),
                "compressed": len(tokens),
                "saved": len(program) - len(tokens),
                "covered": covered,
                "macros": hits,
                "authored": len(task.macro_program.split())
                if task.macro_program
                else None,
                "bucket": bucket_of(len(program)),
            }
        )
    jobs = [(task, rng.randrange(2**31)) for task in programs.values()]
    states, slow = [], []
    with multiprocessing.Pool(initializer=_init, initargs=(macros,)) as pool:
        for behaviour, found, seconds in pool.imap_unordered(_branching, jobs):
            states += found
            if seconds > 5:
                slow.append((round(seconds, 1), behaviour))
    if slow:
        print(
            f"  {name}: slow witnesses (s, behaviour): {sorted(slow, reverse=True)[:5]}",
            file=sys.stderr,
            flush=True,
        )
    b_raw = statistics.mean(p for p, _ in states)
    b_lib = statistics.mean(p + m for p, m in states)
    total = sum(r["length"] for r in rows)
    out = {
        "corpus": name,
        "tasks": len(tasks),
        "witnesses": len(rows),
        "ops": total,
        "compressed_ops": sum(r["compressed"] for r in rows),
        "saved": sum(r["saved"] for r in rows),
        "saved_share": sum(r["saved"] for r in rows) / total,
        "covered_share": sum(r["covered"] for r in rows) / total,
        "hit_share": sum(r["saved"] > 0 for r in rows) / len(rows),
        "median_length": statistics.median(r["length"] for r in rows),
        "median_compressed": statistics.median(r["compressed"] for r in rows),
        "macros_used": len(used),
        "top_macros": used.most_common(15),
        "states": len(states),
        "b_raw": b_raw,
        "b_lib": b_lib,
        # highways (#96) step 3: log2 of Σ b^d, per witness, mean over witnesses
        "log2_cost_raw": statistics.mean(r["length"] * math.log2(b_raw) for r in rows),
        "log2_cost_lib": statistics.mean(
            r["compressed"] * math.log2(b_lib) for r in rows
        ),
        "buckets": {},
        "rows": rows,
    }
    for bucket in sorted(
        {r["bucket"] for r in rows}, key=lambda b: int(b.split("-")[0].rstrip("+"))
    ):
        sub = [r for r in rows if r["bucket"] == bucket]
        out["buckets"][bucket] = {
            "witnesses": len(sub),
            "median_length": statistics.median(r["length"] for r in sub),
            "median_compressed": statistics.median(r["compressed"] for r in sub),
            "saved_share": sum(r["saved"] for r in sub) / sum(r["length"] for r in sub),
            "hit_share": sum(r["saved"] > 0 for r in sub) / len(sub),
            "log2_cost_raw": statistics.mean(
                r["length"] * math.log2(b_raw) for r in sub
            ),
            "log2_cost_lib": statistics.mean(
                r["compressed"] * math.log2(b_lib) for r in sub
            ),
        }
    authored = [(r["authored"], r["compressed"]) for r in rows if r["authored"]]
    if authored:
        out["authored_median"] = statistics.median(a for a, _ in authored)
        out["dp_shorter_than_authored"] = sum(c < a for a, c in authored)
        out["dp_longer_than_authored"] = sum(c > a for a, c in authored)
    return out


def tables(results: list[dict], macro_count: int) -> str:
    lines = [
        f"### Compression of distinct witnesses under the {macro_count}-macro library (fewest tokens over ops U macros)\n",
        "| corpus | tasks | witnesses | median length | median compressed | ops saved | share of ops saved | share of ops inside a macro | witnesses with a hit | macros used |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['corpus']} | {r['tasks']} | {r['witnesses']} | {r['median_length']:g} | {r['median_compressed']:g} | "
            f"{r['saved']} / {r['ops']} | {r['saved_share']:.3f} | {r['covered_share']:.3f} | {r['hit_share']:.3f} | {r['macros_used']} / {macro_count} |"
        )
    lines += [
        "",
        f"### Effective branching at witness-prefix states ({STATES_PER_WITNESS} per witness, every shown example must accept; no #70 guard) and the #96 step-3 estimate, mean over witnesses of d · log2 b\n",
        "| corpus | states | b_eff primitives | b_eff primitives U macros | log2 cost raw | log2 cost library |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['corpus']} | {r['states']} | {r['b_raw']:.1f} | {r['b_lib']:.1f} | {r['log2_cost_raw']:.1f} | {r['log2_cost_lib']:.1f} |"
        )
    lines += [
        "",
        "### By witness-length bucket\n",
        "| corpus | bucket | witnesses | median length | median compressed | share of ops saved | witnesses with a hit | log2 cost raw | log2 cost library |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        for bucket, b in r["buckets"].items():
            lines.append(
                f"| {r['corpus']} | {bucket} | {b['witnesses']} | {b['median_length']:g} | {b['median_compressed']:g} | {b['saved_share']:.3f} | {b['hit_share']:.3f} | {b['log2_cost_raw']:.1f} | {b['log2_cost_lib']:.1f} |"
            )
    lines += [
        "",
        "### Macros by ops saved (top 15)\n",
        "| corpus | macro: ops saved |",
        "|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['corpus']} | "
            + ", ".join(f"`{m}` {n}" for m, n in r["top_macros"])
            + " |"
        )
    authored = [r for r in results if "authored_median" in r]
    if authored:
        lines += [
            "",
            "### World witnesses: the DP parse against the authors' own macro form\n",
            "| corpus | median authored tokens | median DP tokens | DP shorter | DP longer |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in authored:
            lines.append(
                f"| {r['corpus']} | {r['authored_median']:g} | {r['median_compressed']:g} | {r['dp_shorter_than_authored']} | {r['dp_longer_than_authored']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    macros = primitive_macros(load_macros(MACRO_FILE))
    rng = random.Random(BRANCHING_SEED)
    v4 = load(fetch_corpus(MANIFESTS["v4"]))
    world = load(fetch_corpus(MANIFESTS["world"]))
    cells = [
        ("v4 train", [t for t in v4 if t.split == "train"]),
        ("v4 eval", [t for t in v4 if t.split == "eval"]),
        ("v4 cohort (#101)", cohort(v4)),
        ("world-v3 train", [t for t in world if t.split == "train"]),
        ("world-v3 eval", [t for t in world if t.split == "eval"]),
    ]

    def run(label: str, library: dict[str, tuple[str, ...]]) -> list[dict]:
        by_first: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
        for name, ops in library.items():
            by_first[ops[0]].append((name, ops))
        results = []
        for name, tasks in cells:
            results.append(summarise(name, tasks, by_first, library, rng))
            r = results[-1]
            print(
                f"[{label}] {name}: {r['witnesses']} witnesses, saved {r['saved']}/{r['ops']} ops, "
                f"hit {r['hit_share']:.3f}, b {r['b_raw']:.1f} -> {r['b_lib']:.1f}",
                flush=True,
            )
        return results

    full = run("full library", macros)
    # The crudest #96 step-3 selection: keep the macros that hit a v4 train witness.
    # Selected on train, so v4 eval and the cohort are held out from the choice.
    used = sorted({m for r in full[0]["rows"] for m in r["macros"]})
    subset = run("v4-train subset", {m: macros[m] for m in used})
    args.out.mkdir(parents=True, exist_ok=True)
    text = (
        "## Full library\n\n"
        + tables(full, len(macros))
        + f"\n## Subset: the {len(used)} macros that hit a v4 train witness (selected on train; v4 eval and the cohort are held out)\n\n"
        + ", ".join(f"`{m}`" for m in used)
        + "\n\n"
        + tables(subset, len(used))
    )
    (args.out / "tables.md").write_text(text)
    (args.out / "compression.json").write_text(
        json.dumps(
            {
                "macros": len(macros),
                "states_per_witness": STATES_PER_WITNESS,
                "full": full,
                "subset_macros": used,
                "subset": subset,
            },
            indent=1,
        )
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
