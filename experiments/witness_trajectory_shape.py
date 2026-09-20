"""witness-trajectory-shape (#150): cycles of grow-and-run, or write once and run once?

Replays every distinct witness of a stack corpus on its first shown input and
labels each step:

  W  write    taken under an open mark (or the ``[`` that opens one): the op is
              captured as a symbol, so the step adds program text and the data
              below the mark does not move
  D  direct   a top-level op: one visible state transition
  X  execute  a top-level combinator (``i dip map filter fold ifte``): a whole
              quotation runs inside one step

The labels depend on the program alone. Fuel and state sizes depend on the
input too, and are read off the first shown example.

    uv run python experiments/witness_trajectory_shape.py \
        --out reports/witness-trajectory-shape-20260919
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.corpus_hub import fetch_corpus
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    Sym,
    Value,
    ops_for_basis,
    parse_stack,
    quotation_open,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, Task, load

if TYPE_CHECKING:
    from collections.abc import Sequence

DATA = Path(__file__).resolve().parents[1] / "wandering_light" / "evals" / "data"
DEFAULT_MANIFESTS = (
    DATA / "stack_pbe_v4" / "manifest.json",
    DATA / "stack_pbe_world_v3" / "manifest.json",
)
OP_BY_NAME = {op.name: op for op in ops_for_basis(BASIS)}
# The witness-length buckets of compute-optimal-mark-28 (#101).
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))
QUANTILES = (10, 25, 50, 75, 90)
QUOTATIONS = {"ifte": 3}


def atoms(value: Value) -> int:
    """Size of a value: one per scalar, one per list plus its contents."""
    if isinstance(value, Lst):
        return 1 + sum(atoms(item) for item in value.items)
    return 1


def holds_code(value: Value) -> bool:
    if isinstance(value, Lst):
        return any(holds_code(item) for item in value.items)
    return isinstance(value, Sym)


def replay(program: Sequence[str], stack: tuple[Value, ...]) -> list[dict[str, Any]]:
    """One record per step: its kind, fuel, and the state it leaves."""
    steps = []
    for name in program:
        op = OP_BY_NAME[name]
        body = 0
        if name == "[" or quotation_open(stack):
            kind = "W"
        elif op._run is not None:
            kind = "X"
            # `ifte` consumes three quotations (condition, then, else); the rest one.
            body = sum(atoms(q) - 1 for q in stack[-QUOTATIONS.get(name, 1) :])
        else:
            kind = "D"
        outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(outcome, str):
            raise ValueError(f"witness refused at {name}: {outcome}")
        stack, fuel = outcome
        steps.append(
            {
                "kind": kind,
                "fuel": fuel,
                "body": body,
                "stack": stack,
                # A direct step that leaves code on top is quotation assembly
                # (`]` then `cons` / `cat`), not progress on the data.
                "assembly": kind == "D" and bool(stack) and holds_code(stack[-1]),
            }
        )
    return steps


def run_lengths(kinds: str) -> str:
    out, start = [], 0
    for index in range(1, len(kinds) + 1):
        if index == len(kinds) or kinds[index] != kinds[start]:
            out.append(f"{kinds[start]}{index - start}")
            start = index
    return " ".join(out)


def describe(task: Task) -> dict[str, Any]:
    root = parse_stack(task.inputs[0], basis=BASIS)
    steps = replay(task.program, root)
    n = len(steps)
    states = [step["stack"] for step in steps]
    kinds = "".join(step["kind"] for step in steps)
    executions = [index for index, kind in enumerate(kinds) if kind == "X"]
    total_fuel = sum(step["fuel"] for step in steps)
    hidden = sum(step["fuel"] - 1 for step in steps if step["kind"] == "X")
    touched = next(
        (i for i, step in enumerate(steps) if step["stack"][: len(root)] != root), n
    )
    sizes = [sum(atoms(value) for value in stack) for stack in (root, *states)]
    ends = max(1, sizes[0], sizes[-1])
    body = max((step["body"] for step in steps), default=0)
    return {
        "behaviour": task.behaviour,
        "split": task.split,
        "length": n,
        "kinds": run_lengths(kinds),
        "write_share": kinds.count("W") / n,
        "direct_share": kinds.count("D") / n,
        "assembly_share": sum(step["assembly"] for step in steps) / n,
        "executions": len(executions),
        "execution_positions": [index / max(1, n - 1) for index in executions],
        # States a relabelled sub-goal could name: a quotation has just run and
        # no mark is open. The final state is one of them when the program ends
        # on an execution.
        "post_execution_states": sum(
            not quotation_open(steps[index]["stack"]) for index in executions
        ),
        "largest_body_share": body / n,
        "hidden_fuel_share": hidden / total_fuel,
        "first_touch": touched / n,
        "peak_size_ratio": max(sizes) / ends,
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        f"p{q}": ordered[min(len(ordered) - 1, len(ordered) * q // 100)]
        for q in QUANTILES
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "write_share",
        "direct_share",
        "assembly_share",
        "executions",
        "post_execution_states",
        "largest_body_share",
        "hidden_fuel_share",
        "first_touch",
        "peak_size_ratio",
    )
    deciles = Counter(
        min(9, int(position * 10))
        for row in rows
        for position in row["execution_positions"]
    )
    total = sum(deciles.values())
    return {
        "programs": len(rows),
        "median_length": statistics.median(row["length"] for row in rows),
        **{name: quantiles([row[name] for row in rows]) for name in fields},
        "no_execution": sum(row["executions"] == 0 for row in rows),
        "one_execution": sum(row["executions"] == 1 for row in rows),
        "write_then_run": sum(
            row["write_share"] >= 0.8 and row["executions"] <= 2 for row in rows
        ),
        "execution_deciles": [
            round(deciles[i] / total, 3) if total else 0.0 for i in range(10)
        ],
    }


def bucket_label(low: int, high: int) -> str:
    return f"{low}+" if high >= 10**9 else f"{low}-{high}"


def analyse(manifest: Path) -> dict[str, Any]:
    tasks = load(fetch_corpus(manifest))
    seen: set[tuple[str, ...]] = set()
    rows = []
    for task in tasks:
        if task.program not in seen:
            seen.add(task.program)
            rows.append(describe(task))
    cells = {"all": summarize(rows)}
    for low, high in BUCKETS:
        chosen = [row for row in rows if low <= row["length"] <= high]
        if chosen:
            cells[bucket_label(low, high)] = summarize(chosen)
    return {
        "corpus": json.loads(manifest.read_text())["corpus"],
        "tasks": len(tasks),
        "cells": cells,
        "rows": rows,
    }


def tables(results: Sequence[dict[str, Any]]) -> str:
    lines = []
    for result in results:
        lines += [
            f"## {result['corpus']}",
            "",
            f"{result['tasks']} tasks, {result['cells']['all']['programs']} distinct witnesses. "
            "Cells are p10 / p50 / p90 over witnesses.",
            "",
            "| length | n | W share | D share | X | largest body / length "
            "| hidden fuel | first touch | peak size | one X | write-then-run |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for label, cell in result["cells"].items():

            def trio(name: str, cell: dict[str, Any] = cell) -> str:
                q = cell[name]
                return f"{q['p10']:.2f} / {q['p50']:.2f} / {q['p90']:.2f}"

            lines.append(
                f"| {label} | {cell['programs']} | {trio('write_share')} | {trio('direct_share')} "
                f"| {trio('executions')} | {trio('largest_body_share')} "
                f"| {trio('hidden_fuel_share')} | {trio('first_touch')} | {trio('peak_size_ratio')} "
                f"| {cell['one_execution']} | {cell['write_then_run']} |"
            )
        deciles = " ".join(
            f"{v:.2f}" for v in result["cells"]["all"]["execution_deciles"]
        )
        lines += ["", f"Executions by decile of the program: `{deciles}`", ""]
        ordered = sorted(result["rows"], key=lambda row: row["length"])
        lines += ["| witness | ops | X | shape |", "|---|---|---|---|"]
        for row in ordered[:: max(1, len(ordered) // 12)]:
            shape = (
                row["kinds"] if len(row["kinds"]) <= 120 else row["kinds"][:117] + "..."
            )
            lines.append(
                f"| `{row['behaviour']}` | {row['length']} | {row['executions']} | `{shape}` |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--corpus", type=Path, nargs="+", default=list(DEFAULT_MANIFESTS)
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = [analyse(manifest) for manifest in args.corpus]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shape.json").write_text(json.dumps(results, indent=1) + "\n")
    (args.out / "tables.md").write_text(tables(results))
    print(tables(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
