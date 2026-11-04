"""Grade a candidate program against both halves of a stack-PBE task.

Every task in the corpus carries example pairs the solver is shown and eight
it never sees (``Task.held_out_inputs`` / ``held_out_outputs``). Until this
module nothing read the hidden half, so every reported number was a
shown-example fit -- which a program that special-cases the examples passes.

``grade`` scores the two halves independently. A candidate that fits what it
was shown and fails what it was not is *memorised*, and ``Report`` states how
far a shown-only metric would overstate the solve rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    MachineError,
    Op,
    Value,
    ops_for_basis,
    parse_stack,
    step_raw,
    value_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from wandering_light.stack_pbe import Task

# Refusals the machine reports as strings are the normal case; these are the
# exceptions a hostile program could still provoke from Python itself.
_RUN_ERRORS = (
    MachineError,
    RecursionError,
    ValueError,
    ZeroDivisionError,
    OverflowError,
    MemoryError,
)


@dataclass(frozen=True)
class Score:
    """How one program fares on one task.

    ``parsed``: every token is an op of the task's basis. ``legal``: it ran on
    every input, shown and held out. ``shown`` / ``held_out``: it reproduces
    every output of that half.
    """

    program: tuple[str, ...]
    parsed: bool = False
    legal: bool = False
    shown: bool = False
    held_out: bool = False

    @property
    def solved(self) -> bool:
        return self.shown and self.held_out

    @property
    def memorised(self) -> bool:
        return self.shown and not self.held_out


def basis_ops(task: Task) -> dict[str, Op]:
    return {op.name: op for op in ops_for_basis(task.basis)}


def execute(
    program: Sequence[Op], stack: tuple[Value, ...]
) -> tuple[Value, ...] | None:
    """Run ``program`` under the corpus fuel limit; None if the machine refuses."""
    fuel = 0
    try:
        for op in program:
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - fuel, 0)
            if isinstance(outcome, str):
                return None
            stack, spent = outcome
            fuel += spent
    except _RUN_ERRORS:
        return None
    return stack


def stack_key(stack: tuple[Value, ...]) -> tuple:
    return tuple(value_key(value) for value in stack)


def read(text: str, basis: str) -> tuple[Value, ...]:
    return parse_stack(text, basis=basis)


def _half(
    program: Sequence[Op], inputs: Sequence[str], outputs: Sequence[str], basis: str
) -> tuple[bool, bool]:
    """(ran on every input, matched every output) for one half of a task."""
    legal, fits = True, True
    for given, wanted in zip(inputs, outputs, strict=True):
        got = execute(program, read(given, basis))
        if got is None:
            legal = fits = False
            continue
        if stack_key(got) != stack_key(read(wanted, basis)):
            fits = False
    return legal, fits


def grade(candidate: Sequence[str] | str, task: Task) -> Score:
    """Score ``candidate`` on the shown and the held-out half separately.

    A string is split on whitespace. An empty or unparseable program scores
    all False; this never raises on candidate text.
    """
    tokens = tuple(candidate.split() if isinstance(candidate, str) else candidate)
    by_name = basis_ops(task)
    if not tokens or not all(
        isinstance(token, str) and token in by_name for token in tokens
    ):
        return Score(tokens)
    program = tuple(by_name[token] for token in tokens)
    shown_legal, shown = _half(program, task.inputs, task.outputs, task.basis)
    hidden_legal, held_out = _half(
        program, task.held_out_inputs, task.held_out_outputs, task.basis
    )
    return Score(
        program=tokens,
        parsed=True,
        legal=shown_legal and hidden_legal,
        shown=shown,
        held_out=held_out,
    )


def _rank(score: Score) -> tuple[bool, bool, bool, bool]:
    return (score.solved, score.shown, score.legal, score.parsed)


def best(candidates: Iterable[Sequence[str] | str], task: Task) -> Score:
    """The best of several attempts. A solve beats a memorisation in any order."""
    return max(
        (grade(candidate, task) for candidate in candidates),
        key=_rank,
        default=Score(()),
    )


@dataclass(frozen=True)
class Report:
    tasks: int
    solved: int
    memorised: int
    shown_fit: int
    illegal: int

    @property
    def solve_rate(self) -> float:
        return self.solved / self.tasks if self.tasks else 0.0

    @property
    def shown_rate(self) -> float:
        return self.shown_fit / self.tasks if self.tasks else 0.0

    @property
    def overstatement(self) -> float:
        """How much a shown-only grader would inflate the solve rate."""
        return self.shown_rate - self.solve_rate

    def __str__(self) -> str:
        return (
            f"{self.solved}/{self.tasks} solved ({self.solve_rate:.1%}); "
            f"shown-only would say {self.shown_rate:.1%} "
            f"(+{self.overstatement:.1%}, {self.memorised} memorised)"
        )


def summarise(scores: Iterable[Score]) -> Report:
    scores = list(scores)
    return Report(
        tasks=len(scores),
        solved=sum(s.solved for s in scores),
        memorised=sum(s.memorised for s in scores),
        shown_fit=sum(s.shown for s in scores),
        illegal=sum(not s.legal for s in scores),
    )
