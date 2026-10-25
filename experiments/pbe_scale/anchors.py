"""Known simple witnesses for shortest-known representative selection.

This is a small anchor set, not a source of corpus-scale diversity. Existing
readouts/transforms have separately written Python expressions; wrappers merely
select an argument and do not execute any stack programs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.pbe_scale.generate import READOUTS, STAGES

SCALARS = (
    ("negate", "neg", "-n"),
    ("absolute", "abs", "abs(n)"),
    ("increment", "inc", "n+1"),
    ("decrement", "dec", "n-1"),
    ("square", "sq", "n*n"),
    ("negative", "0 lt? bool>int", "int(n<0)"),
    ("positive", "0 gt? bool>int", "int(n>0)"),
    ("zero", "0 eq? bool>int", "int(n==0)"),
    ("even", "even? bool>int", "int(n%2==0)"),
    ("odd", "odd? bool>int", "int(n%2!=0)"),
    ("half-magnitude", "abs 2 div", "abs(n)//2"),
    ("clipped-positive", "0 max2", "max(n,0)"),
    ("clipped-negative", "0 min2", "min(n,0)"),
)
PAIRS = (
    ("sum", "add", "a+b"),
    ("difference", "sub", "a-b"),
    ("product", "mul", "a*b"),
    ("maximum", "max2", "max(a,b)"),
    ("minimum", "min2", "min(a,b)"),
    ("distance", "sub abs", "abs(a-b)"),
    ("sum-magnitude", "add abs", "abs(a+b)"),
    ("less", "lt? bool>int", "int(a<b)"),
    ("greater", "gt? bool>int", "int(a>b)"),
    ("equal", "eq? bool>int", "int(a==b)"),
    ("safe-quotient", "[ dup 0 eq? ] [ drop drop 0 ] [ div ] ifte", "a//b if b else 0"),
    ("safe-remainder", "[ dup 0 eq? ] [ drop drop 0 ] [ mod ] ifte", "a%b if b else 0"),
)


def candidates() -> list[dict]:
    rows = []

    def add(name, kind, signature, program, expression, description):
        rows.append(
            {
                "id": f"anchor-{kind}-{name}",
                "family": "canonical-anchors",
                "template_id": f"anchor/{kind}/{name}",
                "input_kind": kind,
                "program": program.strip(),
                "reference": f"def f({signature}):\n    return {expression}\n",
                "description": description,
                "parameters": {"anchor": True},
            }
        )

    for kind, signature, selector, argument, label in (
        ("list", "xs", "", "xs", "list"),
        ("list_int", "xs,n", "drop", "xs", "list"),
        ("two_lists", "xs,ys", "drop", "xs", "first"),
        ("two_lists", "xs,ys", "swap drop", "ys", "second"),
    ):
        for operation in (*READOUTS[1:], *STAGES):
            add(
                f"{label}-{operation.name}",
                kind,
                signature,
                f"{selector} {operation.program}",
                operation.python(argument),
                f"On the {label} argument, {operation.description}; ignore other arguments.",
            )
    for name, program, expression in SCALARS:
        add(
            name,
            "int",
            "n",
            program,
            expression,
            f"Compute {expression} from the integer argument.",
        )
    for name, program, expression in PAIRS:
        add(
            name,
            "two_ints",
            "a,b",
            program,
            expression,
            f"Compute {expression} from the two integer arguments.",
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = candidates()
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"anchor_candidates": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
