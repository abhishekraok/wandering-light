"""Build the candidate behaviour pool for distinct-programs (#174).

Four public pools, none used before, in the ``world_behaviours/extract.py``
record format (source, id, name, description, solution, entry, tests, shown,
held_out) plus ``needs_reference``:

- ``lf``: BIG-bench ``list_functions`` c001-c250. 32 string examples each,
  no reference (``solution`` null); the task's canary string is kept.
- ``dc``: the DreamCoder list domain, ``data/list_tasks.json`` from
  ellisk42/ec (217 hand-named tasks, 15 examples each, no reference).
  ``list_tasks2.json`` (174k tasks enumerated by ``makeListTasks``) is not a
  curated set and is not used; the few filter/sort tasks ``main.py`` adds at
  run time are drawn randomly per run and are not in any data file.
- ``lb``: LambdaBeam's 100 ``HANDWRITTEN_TASKS``. The upstream code is not
  on GitHub under google-research; the copy read is the AbstractBeam fork
  (jzenkner/AbstractBeam, ``crossbeam/data/deepcoder/deepcoder_tasks.py``).
  Each task ships a DSL solution (kept as ``dsl_solution``), which is made a
  Python reference by prefixing the DeepCoder operation semantics from
  ``crossbeam/dsl/deepcoder_operations.py``; tasks ship 3-5 examples, so
  they are extended by ``extract.extend`` like MBPP.
- ``mbpp`` / ``codearc``: every in-domain problem not already in a
  ``reports/world-*/behaviours.jsonl``, extended by ``extract.extend``.

Filter for all: every test argument and result in ``extract.in_domain`` (int
or flat int list, |x| < 10**6, lists at most 12 long), at most three
arguments, at least eight tests. Behaviours with identical test I/O are
deduplicated, keeping the first in source order (referenced sources first).
No v4 signature audit here; that needs references and is a later step.

Usage:
    uv run --with 'datasets>=4.0' python experiments/distinct_programs/pool.py \
        --out reports/distinct-programs-20260926/pool.jsonl \
        --raw reports/distinct-programs-20260926/raw \
        --exclude-glob 'reports/world-*/behaviours.jsonl'
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import random
import sys
import textwrap
import types
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "world_behaviours"))
import extract

MIN_TESTS = 8
LF_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/"
    "benchmark_tasks/list_functions/c{:03d}/task.json"
)
DC_URL = "https://raw.githubusercontent.com/ellisk42/ec/master/data/list_tasks.json"
LB_URL = (
    "https://raw.githubusercontent.com/jzenkner/AbstractBeam/master/"
    "crossbeam/data/deepcoder/deepcoder_tasks.py"
)


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


# ---------------------------------------------------------------------------
# Sources without a reference
# ---------------------------------------------------------------------------


def lf_candidates(raw: Path) -> list[dict]:
    out = []
    for n in range(1, 251):
        path = fetch(LF_URL.format(n), raw / "lf" / f"c{n:03d}.json")
        task = json.loads(path.read_text())
        tests = [
            {
                "args": [ast.literal_eval(e["input"])],
                "out": ast.literal_eval(e["target"]),
            }
            for e in task["examples"]
        ]
        desc = task["description"]
        target = (
            desc.split("The target function is", 1)[-1].strip().strip(".").strip('"')
        )
        out.append(
            {
                "source": "list_functions",
                "id": f"lf-c{n:03d}",
                "name": target,
                "description": desc,
                "solution": None,
                "entry": None,
                "tests": tests,
                "needs_reference": True,
                "canary": task["canary"],
            }
        )
    return out


def slug(name: str) -> str:
    s = name.lower().replace(" with ", "-").replace(" and ", "-").replace("=", "")
    return "-".join(
        p for p in "".join(c if c.isalnum() else "-" for c in s).split("-") if p
    )


def dc_candidates(raw: Path) -> list[dict]:
    tasks = json.loads(fetch(DC_URL, raw / "list_tasks.json").read_text())
    out = []
    for task in tasks:
        out.append(
            {
                "source": "dreamcoder",
                "id": f"dc-{slug(task['name'])}",
                "name": task["name"],
                "description": f"DreamCoder list task '{task['name']}' "
                f"({task['type']['input']} -> {task['type']['output']})",
                "solution": None,
                "entry": None,
                "tests": [{"args": [e["i"]], "out": e["o"]} for e in task["examples"]],
                "needs_reference": True,
            }
        )
    return out


# ---------------------------------------------------------------------------
# LambdaBeam: DSL solutions made executable
# ---------------------------------------------------------------------------

# DeepCoder operation semantics, from crossbeam/dsl/deepcoder_operations.py.
# Where the original returns None (a rejected value) this raises instead, so
# the example is dropped by ``extract.extend``.
LB_PRELUDE = textwrap.dedent(
    """
    def _int(*xs):
        if any(type(x) is not int for x in xs):
            raise TypeError
    def Add(a, b): _int(a, b); return a + b
    def Subtract(a, b): _int(a, b); return a - b
    def Multiply(a, b): _int(a, b); return a * b
    def IntDivide(a, b): _int(a, b); return a // b
    def Square(x): _int(x); return x ** 2
    def Min(a, b): _int(a, b); return min(a, b)
    def Max(a, b): _int(a, b); return max(a, b)
    def Greater(a, b): return a > b
    def Less(a, b): _int(a, b); return a < b
    def Equal(a, b): return a == b
    def IsEven(x): _int(x); return x % 2 == 0
    def IsOdd(x): _int(x); return x % 2 == 1
    def If(c, x, y):
        if type(c) is not bool:
            raise TypeError
        return x if c else y
    def Head(xs): return xs[0]
    def Last(xs): return xs[-1]
    def Take(n, xs): _int(n); return xs[:n]
    def Drop(n, xs): _int(n); return xs[n:]
    def Access(n, xs): _int(n); return xs[n]
    def Minimum(xs): return min(xs)
    def Maximum(xs): return max(xs)
    def Reverse(xs): return list(reversed(xs))
    def Sort(xs): return sorted(xs)
    def Sum(xs): return sum(xs)
    def Map(f, xs): return [f(x) for x in xs]
    def Filter(f, xs):
        cs = [f(x) for x in xs]
        if not all(isinstance(c, bool) for c in cs):
            raise TypeError
        return [x for x, c in zip(xs, cs) if c]
    def Count(f, xs):
        cs = [f(x) for x in xs]
        if not all(isinstance(c, bool) for c in cs):
            raise TypeError
        return sum(cs)
    def ZipWith(f, xs, ys): return [f(x, y) for x, y in zip(xs, ys)]
    def Scanl1(f, xs):
        ys = [xs[0]]
        for x in xs[1:]:
            y = f(ys[-1], x)
            if type(y) is not int or not -256 <= y <= 255:
                raise ValueError  # DeepCoder's small-value filter
            ys.append(y)
        return ys
    """
)


def lb_tasks(raw: Path) -> list[dict]:
    source = fetch(LB_URL, raw / "lb_deepcoder_tasks.py").read_text()
    source = source.split("\nSYNTHETIC_TASKS = [", 1)[0]  # only the handwritten set
    records: list[dict] = []
    stub = types.ModuleType("stub")
    stub.Task = lambda **kw: records.append(kw) or kw
    for name in (
        "crossbeam",
        "crossbeam.data",
        "crossbeam.data.deepcoder",
        "crossbeam.dsl",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["crossbeam.data.deepcoder"].solution_weight = None
    sys.modules["crossbeam.dsl"].task = stub
    sys.modules["crossbeam.dsl.task"] = stub
    ns: dict = {}
    exec(compile(source, "deepcoder_tasks.py", "exec"), ns)
    return ns["HANDWRITTEN_TASKS"]


def lb_candidates(raw: Path) -> list[dict]:
    out = []
    for n, task in enumerate(lb_tasks(raw)):
        names = list(task["inputs_dict"])
        columns = [task["inputs_dict"][k] for k in names]
        tests = [
            {"args": [col[i] for col in columns], "out": want}
            for i, want in enumerate(task["outputs"])
        ]
        entry = "f"
        solution = (
            LB_PRELUDE
            + f"\ndef f({', '.join(names)}):\n    return {task['solution']}\n"
        )
        out.append(
            {
                "source": "lambdabeam",
                "id": f"lb-{n:03d}",
                "name": task["name"],
                "description": f"LambdaBeam handwritten task {task['name']} "
                f"(inputs {', '.join(names)})",
                "solution": solution,
                "entry": entry,
                "tests": tests,
                "needs_reference": False,
                "dsl_solution": task["solution"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def in_domain_tests(item: dict) -> bool:
    return all(
        len(t["args"]) <= extract.MAX_ARGS
        and all(extract.in_domain(a) for a in t["args"])
        and extract.in_domain(t["out"])
        for t in item["tests"]
    )


def io_key(item: dict) -> str:
    return json.dumps(sorted(json.dumps([t["args"], t["out"]]) for t in item["tests"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="download cache")
    parser.add_argument("--seed", type=int, default=extract.SEED)
    parser.add_argument("--exclude-glob", default="reports/world-*/behaviours.jsonl")
    parser.add_argument(
        "--sources", default="mbpp,codearc,lambdabeam,list_functions,dreamcoder"
    )
    args = parser.parse_args(argv)
    exclude = {
        json.loads(line)["id"]
        for path in sorted(glob.glob(args.exclude_glob))
        for line in Path(path).read_text().splitlines()
    }
    rng = random.Random(args.seed)
    loaders = {
        "mbpp": lambda: extract.mbpp_candidates(rng),
        "codearc": lambda: extract.codearc_candidates(rng),
        "lambdabeam": lambda: lb_candidates(args.raw),
        "list_functions": lambda: lf_candidates(args.raw),
        "dreamcoder": lambda: dc_candidates(args.raw),
    }
    stats: dict[str, dict] = {}
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    pool: list[dict] = []
    for source in args.sources.split(","):
        items = sorted(loaders[source](), key=lambda r: r["id"])
        s = stats[source] = {"loaded": len(items)}
        # mbpp/codearc loaders already drop out-of-domain problems
        items = [r for r in items if r["id"] not in exclude]
        s["not_authored"] = len(items)
        items = [r for r in items if in_domain_tests(r)]
        s["in_domain"] = len(items)
        if source in ("mbpp", "codearc", "lambdabeam"):
            extended = []
            for r in items:
                r.setdefault("needs_reference", False)
                e = extract.extend(args.seed, r)
                if e is not None:
                    extended.append(e)
            items = extended
            s["reference_ok_extended"] = len(items)
        items = [r for r in items if len(r["tests"]) >= MIN_TESTS]
        s["min_tests"] = len(items)
        kept = 0
        for r in items:
            key = io_key(r)
            if key in seen:
                duplicates.append((r["id"], seen[key]))
                continue
            seen[key] = r["id"]
            pool.append(extract.finalize(r))
            kept += 1
        s["deduplicated"] = kept

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in pool:
            fh.write(json.dumps(item) + "\n")
    print(json.dumps(stats, indent=1))
    print(f"{len(duplicates)} identical-I/O duplicates dropped: {duplicates}")
    print(f"wrote {len(pool)} behaviours to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
