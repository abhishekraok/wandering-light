"""What does the stack corpus actually ask a solver to do?

`stack_pbe_v2.jsonl.gz` is the corpus behind the 75.2% BFS baseline and behind
the stack-side solver numbers. Those results read as program synthesis. This
measures what the tasks contain, so they can be quoted with the scope they have.

The finding, from stack-corpus-no-higher-order (#59) and reproduced here:
higher-order programming is not *hard* in this corpus, it is **absent**.
Nothing builds a quotation from op symbols, and three quarters of the tasks
handed a quotation never run it.

That follows from a decision recorded in `wandering_light/stack_machine_corpus.py`:
symbol pushes are excluded from `executable-26` because they are legal in every
state, which takes branching from 13.32 to 37.28 and costs two whole depths of
certification. The families hand quotations over so the combinators are
reachable at all. The decision was correct on its own terms; what was not
recorded is the consequence.

So "builds a quotation from symbols: 0" is **structural, not empirical** -- no
op in `executable-26` can push a symbol, so no program can. It is reported
because the corpus is often read as though the basis could.

    uv run python experiments/stack_corpus_scope.py [corpus.jsonl.gz]
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

DEFAULT = Path("wandering_light/evals/data/stack_pbe_v2.jsonl.gz")

# `executable-26` partitioned by what a program does with each op.
PLUMBING = {"dup", "drop", "swap", "over"}
STRUCTURE = {"cons", "uncons", "cat", "quote", "null?"}
ARITHMETIC = {"add", "sub", "mul", "div"}
PREDICATE = {"eq?", "lt?"}
COMBINATOR = {"map", "filter", "fold", "ifte"}
RUNS_CODE = COMBINATOR | {"i", "dip"}

# Names that may appear inside a quotation *value*. `true`, `false` and `nil`
# are deliberately absent: they print as ordinary values, so a group holding
# only those is data, not code. Including them counts 786 tasks instead of 680.
QUOTED_OPS = PLUMBING | STRUCTURE | ARITHMETIC | PREDICATE | RUNS_CODE

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9?]*")


def resolve(argv: list[str]) -> Path:
    candidates = [Path(argv[1])] if len(argv) > 1 else []
    candidates.append(DEFAULT)
    here = Path(__file__).resolve()
    candidates.extend(parent / DEFAULT for parent in here.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"corpus not found; looked for {DEFAULT}")


def parenthesised(text: str) -> list[str]:
    """Contents of every parenthesised group, innermost closing first."""
    found: list[str] = []
    open_at: list[int] = []
    for index, char in enumerate(text):
        if char == "(":
            open_at.append(index)
        elif char == ")" and open_at:
            found.append(text[open_at.pop() + 1 : index])
    return found


def carries_quotation(task: dict) -> bool:
    """Does any input hold a group whose contents name executable ops?

    A quotation arrives as a *value* placed by the corpus, so it is detected in
    the serialized input rather than in the program.
    """
    return any(
        any(token in QUOTED_OPS for token in TOKEN.findall(group))
        for text in task["inputs"]
        for group in parenthesised(text)
    )


def uses(task: dict, ops: set[str]) -> bool:
    return any(step in ops for step in task["program"])


def main() -> None:
    corpus = resolve(sys.argv)
    with gzip.open(corpus, "rt") as handle:
        tasks = [json.loads(line) for line in handle]
    total = len(tasks)

    quoted = [t for t in tasks if carries_quotation(t)]
    executed = [t for t in quoted if uses(t, RUNS_CODE)]
    rearrangement = sum(not uses(t, ARITHMETIC | PREDICATE | COMBINATOR) for t in tasks)

    rows = [
        (
            "programs that run code at all",
            sum(uses(t, RUNS_CODE) for t in tasks),
            total,
        ),
        (
            "using map / filter / fold / ifte",
            sum(uses(t, COMBINATOR) for t in tasks),
            total,
        ),
        ("no arithmetic, predicate or combinator", rearrangement, total),
        ("inputs carrying a quotation", len(quoted), total),
        ("  ...whose program runs it", len(executed), len(quoted)),
        ("  ...that never execute it", len(quoted) - len(executed), len(quoted)),
        ("programs that build a quotation from symbols", 0, total),
    ]

    print(f"{corpus}  --  {total} tasks\n")
    print(f"{'':<44}{'count':>8}{'share':>9}")
    for label, count, base in rows:
        print(f"{label:<44}{count:>8}{count / base:>9.1%}")
    print(
        "\nThe last row is structural: no op in `executable-26` pushes a symbol,\n"
        "so no program in this corpus can build a quotation out of them."
    )


if __name__ == "__main__":
    main()
