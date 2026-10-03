"""Model-independent primitives for capped-policy-glue-loss (#178), D1 + A.

This module does not select real tasks, open checkpoints, or run a diagnostic.
Ownership is lexical emission ownership, not attribution of runtime work.
Spans end at top-level combinators and need not be genuinely closed states.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from wandering_light.stack_beam import step_all_explained
from wandering_light.stack_machine import Mark, parse_stack
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.stack_beam import Product
    from wandering_light.stack_pbe import Task

STRUCTURAL = frozenset(("[", "]", "map", "filter", "fold", "ifte", "dip", "i"))
RUNNERS = STRUCTURAL - {"[", "]"}
ASSEMBLY = frozenset(("quote", "cons", "cat"))
RowKey = tuple[str, int]


class InvalidDiagnosticError(ValueError):
    """A construction/replay error, never evidence against the prediction."""


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise InvalidDiagnosticError("cooperative deadline exceeded")


@dataclass(frozen=True)
class Owner:
    depth: int
    opener: int | None


def lexical_owners(program: Sequence[str]) -> tuple[Owner, ...]:
    """Opener belongs to parent; closer to the body it closes, as `levels`."""
    opened: list[int] = []
    owners = []
    for step, name in enumerate(program):
        if name not in OP_BY_NAME:
            raise InvalidDiagnosticError(f"unknown mark-28 op at {step}: {name}")
        if name == "]" and not opened:
            raise InvalidDiagnosticError(f"unmatched closer at {step}")
        owners.append(Owner(len(opened), opened[-1] if opened else None))
        if name == "[":
            opened.append(step)
        elif name == "]":
            opened.pop()
    if opened:
        raise InvalidDiagnosticError(f"unmatched openers: {opened}")
    return tuple(owners)


def check_mark_depth(product: Product, owner: Owner, *, context: str) -> None:
    """Top-level Marks only: a Mark stored inside a list is not an open quote."""
    for example, stack in enumerate(product):
        actual = sum(isinstance(value, Mark) for value in stack)
        if actual != owner.depth:
            raise InvalidDiagnosticError(
                f"{context}, example {example}: lexical depth {owner.depth} "
                f"!= interpreter Mark count {actual}"
            )


@dataclass(frozen=True)
class ShownReplay:
    states: tuple[Product, ...]
    owners: tuple[Owner, ...]
    kinds: tuple[str, ...]
    fuel: tuple[int, ...]
    executions: tuple[int, ...]


def replay_shown(task: Task, *, deadline: float | None = None) -> ShownReplay:
    """`replay_states` semantics, with reasons and between-call deadline checks.

    `step_all_explained` uses DEFAULT_FUEL_LIMIT per example/op, depth zero,
    and records only the first refusing example. Hidden examples are not read.
    A single interpreter call can overrun the cooperative deadline; an external
    resource bound is still required by the eventual runner.
    """
    if task.split != "train" or task.basis != BASIS:
        raise InvalidDiagnosticError("expected a train task in the mark-28 basis")
    if not task.program or not task.inputs or len(task.inputs) != len(task.outputs):
        raise InvalidDiagnosticError("empty program/examples or unequal shown pairs")
    owners = lexical_owners(task.program)
    states = [tuple(parse_stack(text, basis=BASIS) for text in task.inputs)]
    target = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    kinds, fuel, executions = [], [], []
    for step, (name, owner) in enumerate(zip(task.program, owners, strict=True)):
        check_deadline(deadline)
        # Checking all emissions is stronger than the registered content check.
        # For '[' and ']', ownership is exactly the pre-op Mark count as well.
        check_mark_depth(states[-1], owner, context=f"{task.behaviour}, step {step}")
        kind = (
            "open"
            if name == "["
            else "close"
            if name == "]"
            else "body"
            if owner.depth
            else "run"
            if name in RUNNERS
            else "direct"
        )
        moved, spent, stepped, reason = step_all_explained(states[-1], OP_BY_NAME[name])
        check_deadline(deadline)
        if moved is None:
            raise InvalidDiagnosticError(
                f"{task.behaviour}, witness step {step} {name}: {reason}"
            )
        states.append(moved)
        kinds.append(kind)
        fuel.append(spent)
        executions.append(stepped)
    if states[-1] != target:
        raise InvalidDiagnosticError(f"{task.behaviour}: witness misses shown target")
    check_mark_depth(states[-1], Owner(0, None), context=f"{task.behaviour}, final")
    return ShownReplay(
        tuple(states), owners, tuple(kinds), tuple(fuel), tuple(executions)
    )


@dataclass(frozen=True)
class Occurrence:
    behaviour: str
    group: str
    step: int
    op: str
    depth: int
    opener: int | None
    kind: str
    span: int
    index_in_span: int
    span_length: int
    position: str
    before_opener: bool
    between_closer_and_combinator: bool

    @property
    def key(self) -> RowKey:
        return self.behaviour, self.step

    @property
    def side(self) -> str:
        return "body" if self.depth else "top"

    @property
    def structural(self) -> bool:
        return self.op in STRUCTURAL


def occurrences(task: Task, replay: ShownReplay) -> tuple[Occurrence, ...]:
    """Full-span positions; contexts are descriptive and may overlap.

    For top content, before_opener means a later top-owned '[' in this span;
    between_closer_and_combinator means an earlier depth-one ']' and a 'run'
    at the span end. Neither context changes eligibility or matching.
    """
    if len(task.program) != len(replay.kinds) or len(replay.owners) != len(
        task.program
    ):
        raise InvalidDiagnosticError("program/replay length mismatch")
    result = []
    start = 0
    span = 0
    for end, kind in enumerate(replay.kinds):
        if kind != "run" and end != len(task.program) - 1:
            continue
        length = end - start + 1
        top_openers = [
            i
            for i in range(start, end + 1)
            if task.program[i] == "[" and replay.owners[i].depth == 0
        ]
        top_closers = [
            i
            for i in range(start, end + 1)
            if task.program[i] == "]" and replay.owners[i].depth == 1
        ]
        for step in range(start, end + 1):
            name, owner = task.program[step], replay.owners[step]
            index = step - start
            position = (
                "only"
                if length == 1
                else "first"
                if index == 0
                else "last"
                if index == length - 1
                else "middle"
            )
            top_content = owner.depth == 0 and name not in STRUCTURAL
            result.append(
                Occurrence(
                    behaviour=task.behaviour,
                    group=task.skeleton or task.behaviour,
                    step=step,
                    op=name,
                    depth=owner.depth,
                    opener=owner.opener,
                    kind=replay.kinds[step],
                    span=span,
                    index_in_span=index,
                    span_length=length,
                    position=position,
                    before_opener=top_content and any(i > step for i in top_openers),
                    between_closer_and_combinator=(
                        top_content
                        and kind == "run"
                        and any(i < step for i in top_closers)
                    ),
                )
            )
        start = end + 1
        span += 1
    return tuple(result)


@dataclass(frozen=True)
class MatchedCell:
    behaviour: str
    group: str
    op: str
    position: str
    top: tuple[RowKey, ...]
    body: tuple[RowKey, ...]

    @property
    def assembly(self) -> bool:
        return self.op in ASSEMBLY


def matched_cells(rows: Sequence[Occurrence]) -> tuple[MatchedCell, ...]:
    """One cell per behaviour/exact-op/position, using every occurrence once."""
    groups = {}
    seen = set()
    buckets = defaultdict(lambda: {"top": [], "body": []})
    for row in rows:
        if row.key in seen:
            raise InvalidDiagnosticError(f"duplicate occurrence identity: {row.key}")
        seen.add(row.key)
        group = groups.setdefault(row.behaviour, row.group)
        if group != row.group:
            raise InvalidDiagnosticError("behaviour has inconsistent original groups")
        if not row.structural:
            buckets[row.behaviour, row.op, row.position][row.side].append(row.key)
    result = []
    for (behaviour, op, position), sides in sorted(buckets.items()):
        if not sides["top"] or not sides["body"]:
            continue
        if position != "middle":
            raise InvalidDiagnosticError("body common support outside middle position")
        result.append(
            MatchedCell(
                behaviour,
                groups[behaviour],
                op,
                position,
                tuple(sides["top"]),
                tuple(sides["body"]),
            )
        )
    return tuple(result)


def support(cells: Sequence[MatchedCell]) -> dict:
    groups = sorted({cell.group for cell in cells})
    behaviours = sorted({cell.behaviour for cell in cells})
    return {
        "groups": groups,
        "behaviours": behaviours,
        "cells": len(cells),
        "passes": len(groups) >= 20 and len(cells) >= 40,
    }


def aggregate_cells(
    cells: Sequence[MatchedCell], losses: Mapping[RowKey, float]
) -> dict:
    """Equal side occurrence means, equal cells, then equal behaviours.

    This is a scalar arithmetic primitive for fixture losses, not an inference
    entry point. The future runner must gate support/seal Stage A first.
    """
    if not cells:
        raise InvalidDiagnosticError("no common-support cells to aggregate")
    by_behaviour = defaultdict(list)
    details = []
    for cell in cells:
        sides = {}
        for side in ("top", "body"):
            values = []
            for key in getattr(cell, side):
                if key not in losses or not math.isfinite(losses[key]):
                    raise InvalidDiagnosticError(f"missing/nonfinite scalar at {key}")
                values.append(losses[key])
            if not values:
                raise InvalidDiagnosticError("cell lacks a matched side")
            sides[side] = fmean(values)
        delta = sides["top"] - sides["body"]
        by_behaviour[cell.behaviour].append(delta)
        details.append(
            {
                "behaviour": cell.behaviour,
                "group": cell.group,
                "op": cell.op,
                "position": cell.position,
                "assembly": cell.assembly,
                "top_count": len(cell.top),
                "body_count": len(cell.body),
                "top_mean": sides["top"],
                "body_mean": sides["body"],
                "delta": delta,
            }
        )
    means = {name: fmean(values) for name, values in sorted(by_behaviour.items())}
    return {
        "cells": details,
        "behaviours": means,
        "delta": fmean(means.values()),
        "positive_behaviour_fraction": fmean(value > 0 for value in means.values()),
    }


def refusal_class(reason: str) -> str:
    """Classify the final raw reason, preserving example/op wrapper prefixes."""
    if reason.endswith("out of fuel"):
        return "fuel"
    if reason.endswith("quotation nested too deep"):
        return "depth"
    return "ordinary"


def legal_alternatives(
    product: Product,
    witness: str,
    *,
    deadline: float | None = None,
) -> dict:
    """Bounded-search legality, reasons only from the first refusing example.

    The eventual Stage A orchestrator calls this only after support passes.
    Resource refusal is illegal in this mask, not proof of semantic impossibility.
    """
    if witness not in OP_BY_NAME:
        raise InvalidDiagnosticError(f"unknown witness op: {witness}")
    mask, refusals = [], []
    for op in OPS:
        check_deadline(deadline)
        moved, _fuel, _executions, reason = step_all_explained(product, op)
        check_deadline(deadline)
        legal = moved is not None
        mask.append(legal)
        if not legal:
            if reason is None:
                raise InvalidDiagnosticError("refused alternative lacks reason")
            if op.name == witness:
                raise InvalidDiagnosticError(f"witness {witness} refused: {reason}")
            refusals.append(
                {"op": op.name, "reason": reason, "class": refusal_class(reason)}
            )
    counts = {
        name: sum(row["class"] == name for row in refusals)
        for name in ("ordinary", "fuel", "depth")
    }
    return {
        "mask": mask,
        "refusals": refusals,
        "first_refusal_counts": counts,
        "resource_affected": bool(counts["fuel"] or counts["depth"]),
    }
