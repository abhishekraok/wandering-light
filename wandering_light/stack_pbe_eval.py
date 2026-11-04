"""Grade a candidate program against a stack PBE task, held-out half included.

Every task in ``stack_pbe_v3`` carries eight hidden example pairs that no
solver sees, and until now nothing read them. This module is the reader. A
candidate is scored on the shown examples *and* on the held-out ones, and the
two verdicts are kept apart, because their difference is the number the
corpus exists to measure: a program that reproduces what it was shown and
fails what it was not has memorised the examples, not solved the task. A
shown-only grader would call that a solve.

* ``Score`` is one candidate's verdict on one task. ``solved`` means both
  halves pass; ``memorised`` means the shown half passes and the hidden half
  does not.
* ``grade`` never raises. An empty or unparseable program scores all-False,
  and a run the machine refuses makes the candidate ``legal=False`` and fails
  the half it refused on. The halves are scored independently, so a program
  can be legal and right on the shown inputs and refuse on a hidden one.
* ``best`` picks the strongest of several candidates by
  ``(solved, shown, legal, parsed)``, so a solve beats a memorisation whatever
  order the candidates came in.
* ``Report`` aggregates scores and prints the solve rate next to what a
  shown-only grader would have claimed, with the gap named as overstatement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wandering_light.stack_machine import parse_stack, value_key
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, run_program

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from wandering_light.stack_machine import Value


@dataclass(frozen=True)
class Score:
    """One candidate program's verdict on one task.

    ``parsed``: every token names an op of the corpus basis.
    ``legal``: the machine ran it to completion on every input, shown and held
    out. ``shown`` / ``held_out``: it reproduced every output of that half.
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
        """Fits what it was shown, fails what it was not: the overfit case."""
        return self.shown and not self.held_out

    @property
    def rank(self) -> tuple[bool, bool, bool, bool]:
        """What ``best`` maximises. A solve beats everything; then shown fit."""
        return (self.solved, self.shown, self.legal, self.parsed)


def normalise(candidate: Sequence[str] | str) -> tuple[str, ...]:
    """A program as a tuple of tokens. Strings are split on whitespace."""
    if isinstance(candidate, str):
        return tuple(candidate.split())
    return tuple(candidate)


def _stacks(texts: Iterable[str]) -> tuple[tuple[Value, ...], ...]:
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def _stack_key(stack: tuple[Value, ...]) -> tuple:
    return tuple(value_key(value) for value in stack)


def _run_half(
    program: tuple[str, ...], inputs: Iterable[str], outputs: Iterable[str]
) -> tuple[bool, bool]:
    """``(legal, fits)`` for one half: ran everywhere, and matched everywhere.

    A refused run is both illegal and a miss. Nothing short-circuits on the
    first miss, so ``legal`` reports every input, not just those before it.
    """
    legal = fits = True
    for stack, want in zip(_stacks(inputs), _stacks(outputs), strict=True):
        outcome = run_program(program, stack)
        if isinstance(outcome, str):
            legal = fits = False
            continue
        got, _fuel = outcome
        if _stack_key(got) != _stack_key(want):
            fits = False
    return legal, fits


def grade(candidate: Sequence[str] | str, task) -> Score:
    """Score one candidate on one task. Never raises.

    ``task`` is anything with ``inputs``, ``outputs``, ``held_out_inputs`` and
    ``held_out_outputs`` as printed stacks, which is what ``stack_pbe.Task``
    and the corpus rows are.
    """
    program = normalise(candidate)
    if not program or any(token not in OP_BY_NAME for token in program):
        return Score(program)
    shown_legal, shown = _run_half(program, task.inputs, task.outputs)
    hidden_legal, held_out = _run_half(
        program, task.held_out_inputs, task.held_out_outputs
    )
    return Score(
        program,
        parsed=True,
        legal=shown_legal and hidden_legal,
        shown=shown,
        held_out=held_out,
    )


def best(candidates: Iterable[Sequence[str] | str], task) -> Score:
    """The strongest candidate's score, by ``(solved, shown, legal, parsed)``.

    No candidates gives the unparsed empty program, so a solver that returned
    nothing is scored exactly like one that returned garbage.
    """
    return max(
        (grade(candidate, task) for candidate in candidates),
        key=lambda score: score.rank,
        default=Score(()),
    )


@dataclass(frozen=True)
class Report:
    """Counts over a set of scores, and the two rates whose gap is the point."""

    tasks: int = 0
    solved: int = 0
    memorised: int = 0
    shown_fit: int = 0
    illegal: int = 0

    @property
    def solve_rate(self) -> float:
        return self.solved / self.tasks if self.tasks else 0.0

    @property
    def shown_rate(self) -> float:
        """What a grader that never read the held-out examples would report."""
        return self.shown_fit / self.tasks if self.tasks else 0.0

    @property
    def overstatement(self) -> float:
        return self.shown_rate - self.solve_rate

    def __str__(self) -> str:
        return (
            f"{self.solved}/{self.tasks} solved ({self.solve_rate:.1%}); "
            f"shown-only would say {self.shown_rate:.1%} "
            f"(+{self.overstatement:.1%}, {self.memorised} memorised)"
        )


def summarise(scores: Iterable[Score]) -> Report:
    tasks = solved = memorised = shown_fit = illegal = 0
    for score in scores:
        tasks += 1
        solved += score.solved
        memorised += score.memorised
        shown_fit += score.shown
        illegal += not score.legal
    return Report(tasks, solved, memorised, shown_fit, illegal)
