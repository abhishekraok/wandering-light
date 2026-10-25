"""Explicit sequence-analysis algorithms and useful views of their outputs.

Algorithm families are declared individually with independent Python references.
Composition counts are not independent algorithm counts: starts, optional output
transforms, and readouts are recorded separately in every structural template.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.pbe_scale.generate import READOUTS, STAGES, STARTS
from experiments.world_behaviours.verify import expand, load_macros

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Algorithm:
    name: str
    family: str
    program: str
    python_body: str
    description: str


RUN_MAX = "nil [ [ over null? ] [ cons ] [ over head max2 cons ] ifte ] fold reverse"
RUN_MIN = "nil [ [ over null? ] [ cons ] [ over head min2 cons ] ifte ] fold reverse"


def _triple_program(body: str) -> str:
    return (
        "nil nil pack2 [ [ unpack2 ] dip [ pick2 len 2 lt? ] [ ] "
        f"[ dup pick3 unpack2 swap rot {body} [ swap ] dip cons swap ] ifte "
        "[ swap ] dip cons swap pack2 ] fold second reverse"
    )


def _prefix_count_program(body: str) -> str:
    return (
        "nil nil pack2 [ [ unpack2 ] dip "
        f"{body} [ swap ] dip cons [ cons ] dip pack2 ] fold second reverse"
    )


def _streak_program(comparison: str) -> str:
    return (
        "nil 0 nil pack3 [ [ unpack3 ] dip "
        "[ pick3 null? ] [ [ drop 1 ] dip2 ] "
        f"[ [ pick3 head over {comparison} ] [ [ inc ] dip2 ] [ [ drop 1 ] dip2 ] ifte ] ifte "
        "[ over cons ] dip [ drop ] dip3 quote -rot pack3 ] fold third reverse"
    )


def _streak_python(condition: str) -> str:
    return (
        "out = []\nstreak = 0\n"
        "for i, x in enumerate(values):\n"
        f"    streak = streak + 1 if i and ({condition}) else 1\n"
        "    out.append(streak)"
    )


ALGORITHMS = (
    Algorithm(
        "running-maximum",
        "running-extrema",
        RUN_MAX,
        "out = []\nfor x in values:\n    out.append(max(out[-1], x) if out else x)",
        "report the maximum observed at each prefix",
    ),
    Algorithm(
        "running-minimum",
        "running-extrema",
        RUN_MIN,
        "out = []\nfor x in values:\n    out.append(min(out[-1], x) if out else x)",
        "report the minimum observed at each prefix",
    ),
    Algorithm(
        "running-product",
        "running-aggregation",
        "1 nil pack2 [ [ unpack2 ] dip rot mul dup [ cons ] dip swap pack2 ] fold unpack2 swap drop reverse",
        "out = []\nproduct = 1\nfor x in values:\n    product *= x\n    out.append(product)",
        "report the product of every nonempty prefix",
    ),
    Algorithm(
        "prefix-nonzero-count",
        "running-aggregation",
        "[ 0 eq? not bool>int ] map prefix-sums",
        "out = [sum(x != 0 for x in values[:i+1]) for i in range(len(values))]",
        "count nonzero observations in each prefix",
    ),
    Algorithm(
        "prefix-positive-balance",
        "running-aggregation",
        "[ [ dup 0 lt? ] [ drop -1 ] [ [ dup 0 eq? ] [ drop 0 ] [ drop 1 ] ifte ] ifte ] map prefix-sums",
        "out = [sum(1 if x > 0 else -1 if x < 0 else 0 for x in values[:i+1]) for i in range(len(values))]",
        "report positive-minus-negative observation counts at each prefix",
    ),
    Algorithm(
        "record-highs",
        "record-sequences",
        "nil [ [ over null? ] [ cons ] [ [ over head over lt? ] [ cons ] [ drop ] ifte ] ifte ] fold reverse",
        "out = []\nfor x in values:\n    if not out or x > out[-1]:\n        out.append(x)",
        "retain the first observation and every subsequent strict record high",
    ),
    Algorithm(
        "record-lows",
        "record-sequences",
        "nil [ [ over null? ] [ cons ] [ [ over head over gt? ] [ cons ] [ drop ] ifte ] ifte ] fold reverse",
        "out = []\nfor x in values:\n    if not out or x < out[-1]:\n        out.append(x)",
        "retain the first observation and every subsequent strict record low",
    ),
    Algorithm(
        "run-representatives",
        "runs-and-streaks",
        "nil [ [ over null? ] [ cons ] [ [ over head over eq? ] [ drop ] [ cons ] ifte ] ifte ] fold reverse",
        "out = [x for i,x in enumerate(values) if i == 0 or x != values[i-1]]",
        "compress each consecutive run of equal values to one representative",
    ),
    Algorithm(
        "equal-streaks",
        "runs-and-streaks",
        _streak_program("eq?"),
        _streak_python("values[i-1] == x"),
        "report the current equal-value run length at each observation",
    ),
    Algorithm(
        "increasing-streaks",
        "runs-and-streaks",
        _streak_program("lt?"),
        _streak_python("values[i-1] < x"),
        "report the current strictly increasing run length at each observation",
    ),
    Algorithm(
        "nondecreasing-streaks",
        "runs-and-streaks",
        _streak_program("le?"),
        _streak_python("values[i-1] <= x"),
        "report the current nondecreasing run length at each observation",
    ),
    Algorithm(
        "decreasing-streaks",
        "runs-and-streaks",
        _streak_program("gt?"),
        _streak_python("values[i-1] > x"),
        "report the current strictly decreasing run length at each observation",
    ),
    Algorithm(
        "run-lengths",
        "runs-and-streaks",
        "nil nil pack2 [ [ unpack2 ] dip [ pick2 null? ] [ [ drop nil 1 cons ] dip ] "
        "[ [ pick2 head over eq? ] [ [ uncons inc cons ] dip ] [ [ 1 cons ] dip ] ifte ] ifte "
        "[ drop ] dip2 quote swap pack2 ] fold second reverse",
        "out = []\nfor i,x in enumerate(values):\n    if i == 0 or x != values[i-1]:\n        out.append(1)\n    else:\n        out[-1] += 1",
        "report the lengths of maximal consecutive equal-value runs",
    ),
    Algorithm(
        "longest-equal-run-so-far",
        "runs-and-streaks",
        _streak_program("eq?") + " " + RUN_MAX,
        _streak_python("values[i-1] == x")
        + "\nout = [max(out[:i+1]) for i in range(len(out))]",
        "report the longest equal-value run seen at each prefix",
    ),
    Algorithm(
        "longest-increasing-run-so-far",
        "runs-and-streaks",
        _streak_program("lt?") + " " + RUN_MAX,
        _streak_python("values[i-1] < x")
        + "\nout = [max(out[:i+1]) for i in range(len(out))]",
        "report the longest strictly increasing run seen at each prefix",
    ),
    Algorithm(
        "strict-sign-changes",
        "threshold-crossings",
        "adjacent [ unpack2 mul 0 lt? bool>int ] map",
        "out = [int(a*b < 0) for a,b in zip(values,values[1:])]",
        "flag adjacent observations with strictly opposite signs",
    ),
    Algorithm(
        "upward-zero-crossings",
        "threshold-crossings",
        "adjacent [ unpack2 0 gt? bool>int [ 0 le? bool>int ] dip mul ] map",
        "out = [int(a <= 0 < b) for a,b in zip(values,values[1:])]",
        "flag transitions from nonpositive to positive observations",
    ),
    Algorithm(
        "downward-zero-crossings",
        "threshold-crossings",
        "adjacent [ unpack2 0 lt? bool>int [ 0 ge? bool>int ] dip mul ] map",
        "out = [int(a >= 0 > b) for a,b in zip(values,values[1:])]",
        "flag transitions from nonnegative to negative observations",
    ),
    Algorithm(
        "neighbor-equality",
        "local-relations",
        "adjacent [ unpack2 eq? bool>int ] map",
        "out = [int(a == b) for a,b in zip(values,values[1:])]",
        "flag equal neighboring observations",
    ),
    Algorithm(
        "neighbor-increases",
        "local-relations",
        "adjacent [ unpack2 lt? bool>int ] map",
        "out = [int(a < b) for a,b in zip(values,values[1:])]",
        "flag strictly increasing neighboring observations",
    ),
    Algorithm(
        "local-peaks",
        "local-shape",
        _triple_program("over lt? bool>int [ lt? bool>int ] dip mul"),
        "out = [int(a < b and b > c) for a,b,c in zip(values,values[1:],values[2:])]",
        "flag strict local peaks among consecutive triples",
    ),
    Algorithm(
        "local-troughs",
        "local-shape",
        _triple_program("over gt? bool>int [ gt? bool>int ] dip mul"),
        "out = [int(a > b and b < c) for a,b,c in zip(values,values[1:],values[2:])]",
        "flag strict local troughs among consecutive triples",
    ),
    Algorithm(
        "local-flat-triples",
        "local-shape",
        _triple_program("over eq? bool>int [ eq? bool>int ] dip mul"),
        "out = [int(a == b == c) for a,b,c in zip(values,values[1:],values[2:])]",
        "flag triples of equal observations",
    ),
    Algorithm(
        "second-differences",
        "local-shape",
        _triple_program("[ 2 mul sub ] dip add"),
        "out = [a-2*b+c for a,b,c in zip(values,values[1:],values[2:])]",
        "compute discrete curvature from consecutive triples",
    ),
    Algorithm(
        "three-point-medians",
        "local-shape",
        _triple_program("pack3 sort second"),
        "out = [sorted([a,b,c])[1] for a,b,c in zip(values,values[1:],values[2:])]",
        "smooth each consecutive triple with its median",
    ),
    Algorithm(
        "three-point-ranges",
        "local-shape",
        _triple_program("pack3 dup max swap min sub"),
        "out = [max(a,b,c)-min(a,b,c) for a,b,c in zip(values,values[1:],values[2:])]",
        "measure the local range of consecutive triples",
    ),
    Algorithm(
        "occurrence-frequencies",
        "frequency-analysis",
        "dup [ swap count ] swap cons map",
        "out = [sum(y == x for y in values) for x in values]",
        "replace each observation by its total occurrence count",
    ),
    Algorithm(
        "lower-count-ranks",
        "rank-transforms",
        "dup [ swap [ lt? ] swap cons filter len ] swap cons map",
        "out = [sum(y < x for y in values) for x in values]",
        "rank observations by the number of strictly smaller samples",
    ),
    Algorithm(
        "higher-count-ranks",
        "rank-transforms",
        "dup [ swap [ gt? ] swap cons filter len ] swap cons map",
        "out = [sum(y > x for y in values) for x in values]",
        "rank observations by the number of strictly larger samples",
    ),
    Algorithm(
        "dense-lower-ranks",
        "rank-transforms",
        "dup dedup [ swap [ lt? ] swap cons filter len ] swap cons map",
        "out = [len(set(y for y in values if y < x)) for x in values]",
        "rank observations by the number of distinct smaller values",
    ),
    Algorithm(
        "total-distance",
        "all-pairs-relations",
        "dup [ swap [ sub abs ] swap cons map sum ] swap cons map",
        "out = [sum(abs(y-x) for y in values) for x in values]",
        "measure each observation's total absolute distance to all samples",
    ),
    Algorithm(
        "squared-distance",
        "all-pairs-relations",
        "dup [ swap [ sub sq ] swap cons map sum ] swap cons map",
        "out = [sum((y-x)**2 for y in values) for x in values]",
        "measure each observation's total squared distance to all samples",
    ),
    Algorithm(
        "sum-smaller",
        "all-pairs-relations",
        "dup [ swap [ lt? ] swap cons filter sum ] swap cons map",
        "out = [sum(y for y in values if y < x) for x in values]",
        "replace each observation by the sum of strictly smaller samples",
    ),
    Algorithm(
        "prior-equal-counts",
        "prefix-relations",
        _prefix_count_program("pick2 over [ eq? ] swap cons filter len"),
        "out = [sum(y == x for y in values[:i]) for i,x in enumerate(values)]",
        "count prior equal observations at each position",
    ),
    Algorithm(
        "prior-smaller-counts",
        "prefix-relations",
        _prefix_count_program("pick2 over [ lt? ] swap cons filter len"),
        "out = [sum(y < x for y in values[:i]) for i,x in enumerate(values)]",
        "count prior strictly smaller observations at each position",
    ),
    Algorithm(
        "prefix-distinct-counts",
        "prefix-relations",
        _prefix_count_program("pick2 over cons dedup len"),
        "out = [len(set(values[:i+1])) for i in range(len(values))]",
        "count distinct values in each nonempty prefix",
    ),
    Algorithm(
        "running-drawdown",
        "running-residuals",
        "dup " + RUN_MAX + " zip [ unpack2 swap sub ] map",
        "out = [max(values[:i+1])-x for i,x in enumerate(values)]",
        "measure each observation's distance below the running maximum",
    ),
    Algorithm(
        "running-rise",
        "running-residuals",
        "dup " + RUN_MIN + " zip [ unpack2 sub ] map",
        "out = [x-min(values[:i+1]) for i,x in enumerate(values)]",
        "measure each observation's distance above the running minimum",
    ),
    Algorithm(
        "running-mean-residuals",
        "running-residuals",
        "0 0 nil pack3 [ [ unpack3 ] dip [ rot ] dip dup [ add ] dip [ inc ] dip3 over pick4 div sub [ swap ] dip cons [ swap ] dip pack3 ] fold third reverse",
        "out = [x-sum(values[:i+1])//(i+1) for i,x in enumerate(values)]",
        "subtract the floored running mean from each observation",
    ),
)


def _reference(start, algorithm: Algorithm, stage, readout) -> str:
    signature = {"list": "xs", "list_int": "xs, k", "two_lists": "xs, ys"}[
        start.category
    ]
    lines = [f"def f({signature}):", f"    values = {start.expression}"]
    lines.extend("    " + line for line in algorithm.python_body.splitlines())
    result = "out"
    if stage is not None:
        lines.append(f"    transformed = {stage.python('out')}")
        result = "transformed"
    lines.append(f"    return {readout.python(result)}")
    return "\n".join(lines) + "\n"


def _legal(algorithm: Algorithm, stage, readout) -> bool:
    # Most algorithms emit one item per input, or per pair/triple. Counting
    # these outputs is only a length/clamped-length function, however long the
    # algorithm's witness is. Keep counts only for data-dependent collections.
    variable_length = algorithm.family == "record-sequences" or algorithm.name in {
        "run-representatives",
        "run-lengths",
    }
    if readout.name == "count" and stage is None and not variable_length:
        return False
    if stage is None:
        return True
    # These algorithms return nonnegative values: do not add identity absolute
    # maps or selectors that trivially remove every observation.
    nonnegative = (
        algorithm.family
        in {
            "runs-and-streaks",
            "rank-transforms",
            "frequency-analysis",
            "prefix-relations",
        }
        and algorithm.name != "run-representatives"
    )
    nonnegative |= algorithm.name in {
        "prefix-nonzero-count",
        "total-distance",
        "squared-distance",
        "running-drawdown",
        "running-rise",
        "three-point-ranges",
    }
    flags = algorithm.family in {
        "threshold-crossings",
        "local-relations",
    } or algorithm.name in {"local-peaks", "local-troughs", "local-flat-triples"}
    if (nonnegative or flags) and stage.name in {
        "absolute",
        "positive-part",
        "negative",
        "magnitude-residual",
    }:
        return False
    if flags and stage.name in {
        "squared",
        "signed-unit",
        "odd",
        "positive",
        "nonzero",
        "above-mean",
        "below-mean",
    }:
        return False
    if algorithm.name == "running-product" and stage.name in {
        "squared",
        "running-sum",
        "neighbor-sum",
    }:
        return False
    if algorithm.name == "running-product" and readout.name in {
        "energy",
        "weighted-total",
    }:
        return False
    if (
        readout.name == "count"
        and stage.category not in {"selection", "multiplicity"}
        and stage.name not in {"above-mean", "below-mean"}
    ):
        return False
    return not (
        stage.category == "ordering"
        and readout.name not in {"weighted-total", "values"}
    )


def iter_candidates(
    seed: int = 188, *, min_ops: int = 25, max_ops: int = 200
) -> Iterator[dict]:
    """Seeded finite enumeration of explicit algorithms and declared view choices."""
    if min_ops < 0 or max_ops < min_ops:
        raise ValueError("invalid primitive-op bounds")
    rng = random.Random(seed)
    macros = load_macros()
    choices = list(itertools.product(ALGORITHMS, STARTS, (None, *STAGES), READOUTS))
    rng.shuffle(choices)
    seen = set()
    for algorithm, start, stage, readout in choices:
        if not _legal(algorithm, stage, readout):
            continue
        program = " ".join(
            filter(
                None,
                [
                    start.program,
                    algorithm.program,
                    stage.program if stage else "",
                    readout.program,
                ],
            )
        )
        expanded = expand(program.split(), macros)
        if not min_ops <= len(expanded) <= max_ops:
            continue
        program_hash = hashlib.sha256(
            (start.category + "|" + " ".join(expanded)).encode()
        ).hexdigest()[:20]
        if program_hash in seen:
            continue
        seen.add(program_hash)
        names = [start.name, algorithm.name]
        description = [start.description, algorithm.description]
        if stage is not None:
            names.append(stage.name)
            description.append(stage.description)
        names.append(readout.name)
        description.append(readout.description)
        yield {
            "id": f"pbe-family-{program_hash}",
            "family": algorithm.family,
            "template_id": "/".join(names),
            "description": "; then ".join(description) + ".",
            "input_kind": start.category,
            "program": program,
            "reference": _reference(start, algorithm, stage, readout),
            "parameters": {
                "algorithm": algorithm.name,
                "start": start.name,
                "postprocess": stage.name if stage else None,
                "readout": readout.name,
            },
        }


def generate_candidates(
    count: int, seed: int = 188, *, min_ops: int = 25, max_ops: int = 200
) -> list[dict]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    rows = list(
        itertools.islice(iter_candidates(seed, min_ops=min_ops, max_ops=max_ops), count)
    )
    if len(rows) != count:
        raise ValueError(f"requested {count} candidates, only {len(rows)} fit")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--min-ops", type=int, default=25)
    parser.add_argument("--max-ops", type=int, default=200)
    args = parser.parse_args(argv)
    rows = generate_candidates(
        args.count, args.seed, min_ops=args.min_ops, max_ops=args.max_ops
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidates": len(rows),
                "declared_algorithms": len(ALGORITHMS),
                "emitted_algorithms": len(
                    {row["parameters"]["algorithm"] for row in rows}
                ),
                "families": dict(Counter(row["family"] for row in rows)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
