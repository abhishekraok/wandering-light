"""Scoring candidate programs against a behaviour-first PBE task.

One thing here is different from every other evaluator in this repo, and it is
the reason the corpus carries hidden examples at all:

    a candidate is scored twice -- on the examples it was shown, and on the
    ones it was not -- and the gap between those two numbers is reported.

A program that fits the shown examples and fails the held-out ones has not
solved the task; it has memorised the prompt. Reporting only the second number
hides how often that happens, and reporting only the first counts it as a win.
`stack_pbe_v2` could not measure this at all, because its data came from a
seventeen-row table and there was nothing left over to hold out.

Nothing here knows how a candidate was produced. Search, a language model and a
hand-written guess are scored by the same function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wandering_light.stack_machine import parse_stack, value_key
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, run_program

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from wandering_light.stack_pbe import Task


@dataclass(frozen=True, slots=True)
class Score:
    """How one candidate program did on one task."""

    program: tuple[str, ...]
    parsed: bool  # every token names an op in the basis
    legal: bool  # ran to completion on every input, shown and held out
    shown: bool  # reproduces every shown output
    held_out: bool  # reproduces every held-out output

    @property
    def solved(self) -> bool:
        """The only number worth quoting: right on data it was never given."""
        return self.shown and self.held_out

    @property
    def memorised(self) -> bool:
        """Fits what it saw, fails what it did not. The failure mode."""
        return self.shown and not self.held_out


def _matches(program: Sequence[str], given: str, wanted: str) -> bool | None:
    """True/False if the program ran; None if the machine refused it."""
    outcome = run_program(program, parse_stack(given, basis=BASIS))
    if isinstance(outcome, str):
        return None
    stack, _fuel = outcome
    want = parse_stack(wanted, basis=BASIS)
    return tuple(map(value_key, stack)) == tuple(map(value_key, want))


def grade(candidate: Sequence[str] | str, task: Task) -> Score:
    """Score one candidate. An illegal or unparseable program simply loses."""
    program = tuple(candidate.split() if isinstance(candidate, str) else candidate)
    if not program or any(name not in OP_BY_NAME for name in program):
        return Score(program, parsed=False, legal=False, shown=False, held_out=False)

    legal = True
    results: dict[str, bool] = {}
    for label, ins, outs in (
        ("shown", task.inputs, task.outputs),
        ("held_out", task.held_out_inputs, task.held_out_outputs),
    ):
        ok = True
        for given, wanted in zip(ins, outs, strict=True):
            matched = _matches(program, given, wanted)
            if matched is None:
                legal = ok = False
                break
            ok = ok and matched
        results[label] = ok
    return Score(
        program,
        parsed=True,
        legal=legal,
        shown=results["shown"],
        held_out=results["held_out"],
    )


def best(candidates: Iterable[Sequence[str] | str], task: Task) -> Score:
    """The best of k candidates, preferring a solve over a memorisation.

    Ordering matters: a run that scored `any(shown)` would count a memorised
    candidate as a hit whenever one appears alongside a real solution.
    """
    scores = [grade(candidate, task) for candidate in candidates]
    if not scores:
        return Score((), parsed=False, legal=False, shown=False, held_out=False)
    return max(scores, key=lambda s: (s.solved, s.shown, s.legal, s.parsed))


@dataclass(frozen=True)
class Report:
    """Aggregate over a set of tasks."""

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
        """What a grader without held-out examples would have reported."""
        return self.shown_fit / self.tasks if self.tasks else 0.0

    @property
    def overstatement(self) -> float:
        """How much the shown-only number flatters the solver, in points."""
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
