"""Deterministic, task-grounded compositions for the PBE scaling campaign.

Each stage pairs a stack program with a separately expressed Python operation.
Candidate generation never executes stack code; validate.py supplies the domain
and checks equivalence. No model, search, or parameter-grid expansion is used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.world_behaviours.verify import expand, load_macros

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Operation:
    name: str
    category: str
    program: str
    expression: str
    description: str

    def python(self, value: str) -> str:
        return self.expression.replace("VALUES", value)


STAGES = (
    Operation(
        "absolute",
        "pointwise",
        "[ abs ] map",
        "[abs(x) for x in VALUES]",
        "take absolute values",
    ),
    Operation(
        "squared",
        "pointwise",
        "[ sq ] map",
        "[x*x for x in VALUES]",
        "square each value",
    ),
    Operation(
        "positive-part",
        "pointwise",
        "[ 0 max2 ] map",
        "[max(x, 0) for x in VALUES]",
        "clip negative values to zero",
    ),
    Operation(
        "signed-unit",
        "pointwise",
        "[ [ dup 0 lt? ] [ drop -1 ] [ [ dup 0 eq? ] [ drop 0 ] [ drop 1 ] ifte ] ifte ] map",
        "[(1 if x > 0 else -1 if x < 0 else 0) for x in VALUES]",
        "encode signs",
    ),
    Operation(
        "magnitude-residual",
        "pointwise",
        "[ dup abs sub ] map",
        "[x-abs(x) for x in VALUES]",
        "retain twice the negative part",
    ),
    Operation(
        "nonzero",
        "selection",
        "[ 0 eq? not ] filter",
        "[x for x in VALUES if x != 0]",
        "remove zeros",
    ),
    Operation(
        "positive",
        "selection",
        "[ 0 gt? ] filter",
        "[x for x in VALUES if x > 0]",
        "retain positive values",
    ),
    Operation(
        "negative",
        "selection",
        "[ 0 lt? ] filter",
        "[x for x in VALUES if x < 0]",
        "retain negative values",
    ),
    Operation(
        "even",
        "selection",
        "[ even? ] filter",
        "[x for x in VALUES if x % 2 == 0]",
        "retain even values",
    ),
    Operation(
        "odd",
        "selection",
        "[ odd? ] filter",
        "[x for x in VALUES if x % 2 != 0]",
        "retain odd values",
    ),
    Operation(
        "unique",
        "multiplicity",
        "dedup",
        "[x for i, x in enumerate(VALUES) if x not in VALUES[:i]]",
        "keep first occurrences",
    ),
    Operation("sort", "ordering", "sort", "sorted(VALUES)", "sort ascending"),
    Operation(
        "descending",
        "ordering",
        "sort reverse",
        "sorted(VALUES, reverse=True)",
        "sort descending",
    ),
    Operation(
        "reverse", "ordering", "reverse", "list(reversed(VALUES))", "reverse order"
    ),
    Operation(
        "running-sum",
        "history",
        "prefix-sums",
        "[sum(VALUES[:i+1]) for i in range(len(VALUES))]",
        "compute running totals",
    ),
    Operation(
        "neighbor-difference",
        "neighborhood",
        "differences",
        "[VALUES[i]-VALUES[i+1] for i in range(len(VALUES)-1)]",
        "subtract each next observation",
    ),
    Operation(
        "neighbor-distance",
        "neighborhood",
        "differences [ abs ] map",
        "[abs(VALUES[i]-VALUES[i+1]) for i in range(len(VALUES)-1)]",
        "measure neighboring absolute changes",
    ),
    Operation(
        "neighbor-sum",
        "neighborhood",
        "adjacent [ unpack2 add ] map",
        "[VALUES[i]+VALUES[i+1] for i in range(len(VALUES)-1)]",
        "sum neighboring observations",
    ),
    Operation(
        "neighbor-min",
        "neighborhood",
        "adjacent [ unpack2 min2 ] map",
        "[min(VALUES[i], VALUES[i+1]) for i in range(len(VALUES)-1)]",
        "take neighboring minima",
    ),
    Operation(
        "neighbor-max",
        "neighborhood",
        "adjacent [ unpack2 max2 ] map",
        "[max(VALUES[i], VALUES[i+1]) for i in range(len(VALUES)-1)]",
        "take neighboring maxima",
    ),
    Operation(
        "center",
        "global",
        "dup sum over len 1 max2 div [ sub ] swap cons map",
        "[x-(sum(VALUES)//max(len(VALUES),1)) for x in VALUES]",
        "subtract the floored mean",
    ),
    Operation(
        "above-mean",
        "global",
        "dup sum over len 1 max2 div [ gt? ] swap cons filter",
        "[x for x in VALUES if x > sum(VALUES)//max(len(VALUES),1)]",
        "retain observations above the floored mean",
    ),
    Operation(
        "below-mean",
        "global",
        "dup sum over len 1 max2 div [ lt? ] swap cons filter",
        "[x for x in VALUES if x < sum(VALUES)//max(len(VALUES),1)]",
        "retain observations below the floored mean",
    ),
)

READOUTS = (
    Operation("values", "list", "", "VALUES", "return the observations"),
    Operation("total", "scalar", "sum", "sum(VALUES)", "sum the result"),
    Operation("count", "scalar", "len", "len(VALUES)", "count the result"),
    Operation(
        "energy",
        "scalar",
        "[ sq ] map sum",
        "sum(x*x for x in VALUES)",
        "sum squared values",
    ),
    Operation(
        "absolute-total",
        "scalar",
        "[ abs ] map sum",
        "sum(abs(x) for x in VALUES)",
        "sum magnitudes",
    ),
    Operation(
        "maximum",
        "scalar",
        "[ dup null? ] [ drop 0 ] [ max ] ifte",
        "max(VALUES, default=0)",
        "take the maximum, or zero when empty",
    ),
    Operation(
        "minimum",
        "scalar",
        "[ dup null? ] [ drop 0 ] [ min ] ifte",
        "min(VALUES, default=0)",
        "take the minimum, or zero when empty",
    ),
    Operation(
        "span",
        "scalar",
        "[ dup null? ] [ drop 0 ] [ dup max swap min sub ] ifte",
        "(max(VALUES)-min(VALUES) if VALUES else 0)",
        "take the range, or zero when empty",
    ),
    Operation(
        "mean",
        "scalar",
        "dup sum swap len 1 max2 div",
        "sum(VALUES)//max(len(VALUES),1)",
        "take the floored mean, or zero when empty",
    ),
    Operation(
        "weighted-total",
        "scalar",
        "enumerate [ unpack2 [ inc ] dip mul ] map sum",
        "sum((i+1)*x for i,x in enumerate(VALUES))",
        "sum values weighted by one-based position",
    ),
)

STARTS = (
    Operation("input", "list", "", "xs", "start with the input list"),
    Operation(
        "subtract-argument",
        "list_int",
        "[ sub ] swap cons map",
        "[x-k for x in xs]",
        "subtract the supplied baseline from each observation",
    ),
    Operation(
        "distance-to-argument",
        "list_int",
        "[ sub abs ] swap cons map",
        "[abs(x-k) for x in xs]",
        "measure distance from the supplied baseline",
    ),
    Operation(
        "above-argument",
        "list_int",
        "[ gt? ] swap cons filter",
        "[x for x in xs if x > k]",
        "retain observations above the supplied threshold",
    ),
    Operation(
        "below-argument",
        "list_int",
        "[ lt? ] swap cons filter",
        "[x for x in xs if x < k]",
        "retain observations below the supplied threshold",
    ),
    Operation(
        "pair-sum",
        "two_lists",
        "zip [ unpack2 add ] map",
        "[a+b for a,b in zip(xs,ys)]",
        "sum aligned observations, stopping at the shorter list",
    ),
    Operation(
        "pair-difference",
        "two_lists",
        "zip [ unpack2 sub ] map",
        "[a-b for a,b in zip(xs,ys)]",
        "subtract aligned observations, stopping at the shorter list",
    ),
    Operation(
        "pair-distance",
        "two_lists",
        "zip [ unpack2 sub abs ] map",
        "[abs(a-b) for a,b in zip(xs,ys)]",
        "measure aligned distances, stopping at the shorter list",
    ),
    Operation(
        "pair-product",
        "two_lists",
        "zip [ unpack2 mul ] map",
        "[a*b for a,b in zip(xs,ys)]",
        "multiply aligned observations, stopping at the shorter list",
    ),
    Operation(
        "pair-min",
        "two_lists",
        "zip [ unpack2 min2 ] map",
        "[min(a,b) for a,b in zip(xs,ys)]",
        "take aligned minima, stopping at the shorter list",
    ),
    Operation(
        "pair-max",
        "two_lists",
        "zip [ unpack2 max2 ] map",
        "[max(a,b) for a,b in zip(xs,ys)]",
        "take aligned maxima, stopping at the shorter list",
    ),
    Operation("concatenate", "two_lists", "cat", "xs+ys", "concatenate the two series"),
)


def _legal(stages: tuple[Operation, ...], readout: Operation) -> bool:
    """Reject obvious collapses before equivalence-based curation downstream."""
    categories = [stage.category for stage in stages]
    if len(set(categories)) != len(categories):
        return False
    names = [stage.name for stage in stages]
    if (
        readout.name
        in {
            "total",
            "count",
            "energy",
            "absolute-total",
            "maximum",
            "minimum",
            "span",
            "mean",
        }
        and stages[-1].category == "ordering"
    ):
        return False
    if (
        readout.name == "count"
        and stages[-1].category in {"pointwise", "history", "global"}
        and stages[-1].name not in {"above-mean", "below-mean"}
    ):
        return False
    # Sign/magnitude maps followed by incompatible selectors produce constants.
    for first, second in itertools.pairwise(stages):
        if first.name in {
            "absolute",
            "squared",
            "positive-part",
            "neighbor-distance",
        } and second.name in {"negative", "positive", "absolute"}:
            return False
        if first.name == "magnitude-residual" and second.name in {
            "positive",
            "even",
            "odd",
        }:
            return False
        if first.name in {"sort", "descending"} and second.name == "unique":
            return False  # dedup before sorting is the less redundant ordering
        if (
            first.name in {"positive", "negative", "nonzero"}
            and second.name == "signed-unit"
        ):
            return False
    return not (
        "center" in names
        and readout.name in {"total", "mean"}
        and names[-1] == "center"
    )


def _reference(
    start: Operation, stages: tuple[Operation, ...], readout: Operation
) -> str:
    signature = {"list": "xs", "list_int": "xs, k", "two_lists": "xs, ys"}[
        start.category
    ]
    lines = [f"def f({signature}):", f"    v0 = {start.expression}"]
    for index, stage in enumerate(stages, 1):
        lines.append(f"    v{index} = {stage.python(f'v{index - 1}')}")
    lines.append(f"    return {readout.python(f'v{len(stages)}')}")
    return "\n".join(lines) + "\n"


def _row(
    family: str,
    template: str,
    description: str,
    kind: str,
    program: str,
    reference: str,
) -> dict:
    identifier = hashlib.sha256((kind + "|" + program).encode()).hexdigest()[:20]
    return {
        "id": f"pbe-scale-{identifier}",
        "family": family,
        "template_id": template,
        "description": description,
        "input_kind": kind,
        "program": program,
        "reference": reference,
    }


def _pipeline(
    start: Operation, stages: tuple[Operation, ...], readout: Operation
) -> dict:
    family = {
        "list": "series-pipeline",
        "list_int": "baseline-pipeline",
        "two_lists": "paired-series-pipeline",
    }[start.category]
    template = "/".join([start.name, *(stage.name for stage in stages), readout.name])
    description = (
        "; then ".join(
            [
                start.description,
                *(stage.description for stage in stages),
                readout.description,
            ]
        )
        + "."
    )
    program = " ".join(
        filter(
            None, [start.program, *(stage.program for stage in stages), readout.program]
        )
    )
    return _row(
        family,
        template,
        description,
        start.category,
        program,
        _reference(start, stages, readout),
    )


def _branched(stages: tuple[Operation, ...], readout: Operation, mode: int) -> dict:
    body = " ".join(stage.program for stage in stages)
    lines = _reference(STARTS[0], stages, readout).splitlines()
    result = lines[-1].removeprefix("    return ")
    lines = lines[:-1]
    if mode == 0:
        program = f"dup {body} {readout.program} swap sum sub"
        lines.append(f"    return ({result}) - sum(xs)")
        name, detail = (
            "total-difference",
            "subtract the original total from that summary",
        )
    elif mode == 1:
        program = f"dup {body} {readout.program} swap len mul"
        lines.append(f"    return ({result}) * len(xs)")
        name, detail = (
            "sample-weighted",
            "multiply that summary by the original sample count",
        )
    elif mode == 2:
        program = f"[ dup sum 0 lt? ] [ {body} {readout.program} ] [ reverse {body} {readout.program} ] ifte"
        # Evaluate precisely the same pipeline on a branch-selected source.
        lines[1] = "    v0 = xs if sum(xs) < 0 else list(reversed(xs))"
        lines.append(f"    return {result}")
        name, detail = (
            "direction-by-total",
            "use input order for negative-total series and reversed order otherwise",
        )
    else:
        # Two distinct views of the same source; keep all original observations
        # on one side and analyze local changes on the other.
        program = f"dup {body} {readout.program} swap differences {body} {readout.program} sub"
        lines.append(f"    left = {result}")
        lines.append("    w0 = [xs[i]-xs[i+1] for i in range(len(xs)-1)]")
        for index, stage in enumerate(stages, 1):
            lines.append(f"    w{index} = {stage.python(f'w{index - 1}')}")
        lines.append(f"    return left - ({readout.python(f'w{len(stages)}')})")
        name, detail = (
            "level-versus-change",
            "subtract the same summary of neighboring changes",
        )
    template = "/".join([name, *(stage.name for stage in stages), readout.name])
    description = (
        "; then ".join(
            [*(stage.description for stage in stages), readout.description, detail]
        )
        + "."
    )
    return _row(
        "conditional-summary" if mode == 2 else "original-comparison",
        template,
        description,
        "list",
        program,
        "\n".join(lines) + "\n",
    )


def _permuted_stages(
    operations: list[Operation], depth: int, rng: random.Random
) -> Iterator[tuple[Operation, ...]]:
    """Visit every ordered stage choice once, without lexicographic prefix bias."""
    capacity = math.perm(len(operations), depth)
    offset = rng.randrange(capacity)
    stride = rng.randrange(1, capacity)
    while math.gcd(stride, capacity) != 1:
        stride = rng.randrange(1, capacity)
    for index in range(capacity):
        rank = (offset + index * stride) % capacity
        remaining = list(operations)
        selected = []
        for position in range(depth):
            block = math.perm(len(remaining) - 1, depth - position - 1)
            choice, rank = divmod(rank, block)
            selected.append(remaining.pop(choice))
        yield tuple(selected)


def iter_candidates(
    seed: int = 0, *, min_ops: int = 50, max_ops: int = 200
) -> Iterator[dict]:
    """Yield a reproducible diverse stream, with structural templates as IDs.

    This is a finite seeded enumeration, not rejection sampling. Every accepted
    candidate has a distinct macro program and 2-4 declared transformation stages.
    Some stages can be algebraically redundant; verification does not certify
    shortest witnesses. Parameter values do not multiply the candidate count.
    """
    if min_ops < 0 or max_ops < min_ops:
        raise ValueError("invalid primitive-op bounds")
    rng = random.Random(seed)
    macros = load_macros()
    lengths = {
        op.program: len(expand(op.program.split(), macros))
        for op in (*STAGES, *READOUTS, *STARTS)
    }
    stages_order = list(STAGES)
    rng.shuffle(stages_order)
    starts_order = list(STARTS)
    rng.shuffle(starts_order)
    readouts_order = list(READOUTS)
    rng.shuffle(readouts_order)
    # Interleave depths so short prefixes are not dominated by one complexity.
    depth_orders = {}
    for depth in (2, 3, 4):
        depth_orders[depth] = _permuted_stages(stages_order, depth, rng)
    active = list(depth_orders)
    seen: set[str] = set()
    while active:
        for depth in active[:]:
            try:
                stages = next(depth_orders[depth])
            except StopIteration:
                active.remove(depth)
                continue
            readout_offset = rng.randrange(len(readouts_order))
            for offset in range(len(readouts_order)):
                readout = readouts_order[
                    (readout_offset + offset) % len(readouts_order)
                ]
                if not _legal(stages, readout):
                    continue
                base_length = (
                    sum(lengths[stage.program] for stage in stages)
                    + lengths[readout.program]
                )
                # Rotate starts across structures rather than exhausting a large
                # parameter family before the next semantic pipeline.
                start = starts_order[rng.randrange(len(starts_order))]
                if min_ops <= base_length + lengths[start.program] <= max_ops:
                    row = _pipeline(start, stages, readout)
                    if row["id"] not in seen:
                        seen.add(row["id"])
                        yield row
                if readout.category == "scalar":
                    mode = rng.randrange(4)
                    if (
                        mode == 2
                        and not any(
                            stage.category in {"neighborhood", "history"}
                            for stage in stages
                        )
                        and readout.name != "weighted-total"
                    ):
                        continue
                    if mode == 3 and any(
                        stage.category == "neighborhood" for stage in stages
                    ):
                        continue
                    row = _branched(stages, readout, mode)
                    size = len(expand(row["program"].split(), macros))
                    if min_ops <= size <= max_ops and row["id"] not in seen:
                        seen.add(row["id"])
                        yield row


def generate_candidates(
    count: int, seed: int = 0, *, min_ops: int = 50, max_ops: int = 200
) -> list[dict]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    rows = list(
        itertools.islice(iter_candidates(seed, min_ops=min_ops, max_ops=max_ops), count)
    )
    if len(rows) != count:
        raise ValueError(
            f"requested {count} candidates but only {len(rows)} satisfy the bounds"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--min-ops", type=int, default=50)
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
        json.dumps({"output": str(args.output), "count": len(rows), "seed": args.seed})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
