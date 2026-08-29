"""Does `mark-28` produce a more expressive corpus than `executable-26`?

solver-build-quotations (#60) landed `mark-28` and showed it reproduces every
branching and program-cost measurement claimed for it. Its own plan then gates
everything downstream on a separate question: *regenerate a corpus against it
and compare certified depth and the share of tasks that use a combinator.* "If
the corpus is not more expressive, the cheaper branching bought nothing."

This is that step. It generates both corpora at one depth and compares them.

**What it finds.** The corpus is not more expressive. At depth 5 the combinator
share moves 6.7% -> 7.0% and the share running code at all does not move at all.
Quotation construction does appear -- 0% -> 4.3%, which is what `mark-28` was
added for -- but it is degenerate: bodies are 1 or 2 ops, and **97% of the
programs that build a quotation never execute it**. They are using `[ ... ]` to
put a *symbol on the stack as data*, which was previously impossible, not to
construct a program body.

**Why, and it is not the basis's fault.** Tasks are certified *shortest*, and
the families hand quotations over in the input. So a built quotation can never
win against a supplied one: `[ 1 add ] map` is five ops where `(1 add) map` is
one. A richer basis cannot fix a corpus whose generator selects against using
it. The lever is the generator -- stop supplying quotations -- not the op set.

Cost, measured on one 16-core desktop: depth 5 takes 130 s under
`executable-26` and 1,364 s under `mark-28`, 10.5x. At `mark-28`'s 22.5x growth
per depth, depth 6 is roughly eight hours against fifty minutes, so it
effectively certifies one depth shallower for the same budget.

    uv run python experiments/mark28_corpus_probe.py --depth 5 [--dir DIR]
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

COMBINATOR = {"map", "filter", "fold", "ifte"}
RUNS_CODE = COMBINATOR | {"i", "dip"}


def generate(basis: str, depth: int, out: Path, per_depth: int) -> float:
    """Generate one corpus, returning wall-clock seconds. Skips if present."""
    if out.exists():
        return float("nan")
    started = time.time()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wandering_light.stack_machine_corpus",
            "--basis",
            basis,
            "--max-depth",
            str(depth),
            "--per-depth",
            str(per_depth),
            "--out",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.time() - started


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def bodies(program: list[str]) -> list[list[str]]:
    """Contents of each top-level `[ ... ]` -- the quotation actually built."""
    built: list[list[str]] = []
    depth = 0
    current: list[str] = []
    for token in program:
        if token == "[":
            depth += 1
            current = []
        elif token == "]":
            depth -= 1
            if depth == 0:
                built.append(current)
        elif depth:
            current.append(token)
    return built


def executes_what_it_built(program: list[str]) -> bool:
    if "]" not in program:
        return False
    return any(t in RUNS_CODE for t in program[program.index("]") + 1 :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--per-depth", type=int, default=40)
    parser.add_argument("--dir", type=Path, default=Path("corpora"))
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)

    corpora, seconds = {}, {}
    for basis in ("executable-26", "mark-28"):
        out = args.dir / f"{basis}_d{args.depth}.jsonl.gz"
        seconds[basis] = generate(basis, args.depth, out, args.per_depth)
        corpora[basis] = load(out)

    print(f"depth {args.depth}, per-depth {args.per_depth}\n")
    print(f"{'':<34}{'executable-26':>15}{'mark-28':>12}")

    def row(label, fn):
        cells = []
        for basis in ("executable-26", "mark-28"):
            tasks = corpora[basis]
            hit = sum(bool(fn(t)) for t in tasks)
            cells.append(f"{hit} ({hit / len(tasks):.1%})")
        print(f"{label:<34}{cells[0]:>15}{cells[1]:>12}")

    print(
        f"{'tasks':<34}{len(corpora['executable-26']):>15}{len(corpora['mark-28']):>12}"
    )
    print(
        f"{'generation seconds':<34}"
        f"{seconds['executable-26']:>15.0f}{seconds['mark-28']:>12.0f}"
    )
    row("uses a combinator", lambda t: any(x in COMBINATOR for x in t["program"]))
    row("runs code at all", lambda t: any(x in RUNS_CODE for x in t["program"]))
    row("builds a quotation", lambda t: "[" in t["program"])

    built = [t for t in corpora["mark-28"] if "[" in t["program"]]
    if built:
        ran = [t for t in built if executes_what_it_built(t["program"])]
        lengths = Counter(len(b) for t in built for b in bodies(t["program"]))
        print(f"\nof the {len(built)} mark-28 quotation builders:")
        print(f"  execute what they built   {len(ran)} ({len(ran) / len(built):.1%})")
        print(f"  body length histogram     {dict(sorted(lengths.items()))}")
        print(
            "\nBodies are op fragments used as data, not program bodies. The "
            "generator\ncertifies shortest programs and the families supply "
            "quotations, so building\na useful one is never shortest."
        )


if __name__ == "__main__":
    main()
