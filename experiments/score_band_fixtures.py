"""Fixed hand-written software-QA fixtures for score-band-retention (#179).

FIXTURES.md A.1 was committed before this module or any fixture execution.
No corpus, held-out data, model creation, or checkpoint reads are allowed here.
The catalog is shared by development tests and a later separately reviewed gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from experiments.score_band_search import BandObserver
from wandering_light.stack_pbe import BASIS, OPS, Task

if TYPE_CHECKING:
    from experiments.score_band_search import BandLayer

OP_INDEX = {op.name: i for i, op in enumerate(OPS)}
TASK_TABLE = {
    "T-pair": (("2 3", "4 5"), ("12", "40"), "mul dup add"),
    "T-list": (("(1 2)", "(3 4)"), ("(2 1)", "(4 3)"), "nil [ cons ] fold"),
    "T-typed": (
        ("(true 1) false", "(false 2) true"),
        ("(1 true) false", "(2 false) true"),
        "swap nil [ cons ] fold swap",
    ),
    "T-single": (("2 3",), ("12",), "mul dup add"),
    "T-same": (("2 2",), ("4",), "add"),
    "T-probe": (("2 3",), ("9999",), ""),
    "T-equal-probe": (("2 2",), ("9999",), ""),
    "T-empty": (("[]",), ("9999",), ""),
    "T-open": (("MARK 0",), ("9999",), ""),
    "T-long": ((" ".join(["0"] * 163),), ("9999",), ""),
}


def manual_task(name: str) -> Task:
    inputs, outputs, program = TASK_TABLE[name]
    witness = tuple(program.split())
    return Task(
        behaviour=name,
        summary="Manually typed score-band QA fixture",
        split="train",
        task_hash_split="train",
        inputs=inputs,
        outputs=outputs,
        held_out_inputs=(),
        held_out_outputs=(),
        program=witness,
        length=len(witness),
        examples=len(inputs),
        held_out=0,
        fuel=0,
        tags=("synthetic",),
        basis=BASIS,
        witness_source="synthetic-manual" if witness else "synthetic-no-witness",
    )


@dataclass(frozen=True)
class Script:
    rules: tuple[tuple[int, int, str, float], ...] = ()
    default: float = -100.0
    illegal: float = -np.inf
    seed: int | None = None
    raises: bool = False


class ScriptedScorer:
    def __init__(self, script: Script):
        self.script = script
        self.calls = 0

    def log_probs(self, blocks, legal):
        if self.script.raises:
            raise RuntimeError("fixture scorer must not have been called")
        call = self.calls
        self.calls += 1
        raw = np.full(legal.shape, self.script.default, dtype=np.float64)
        if self.script.seed is not None:
            for row in range(len(blocks)):
                for index in range(len(OPS)):
                    raw[row, index] = -(
                        (7 * index + 11 * row + 13 * call + 17 * self.script.seed) % 29
                    )
        for at_call, row, name, value in self.script.rules:
            if at_call == call and row < len(blocks):
                raw[row, OP_INDEX[name]] = value
        return np.where(legal, raw, self.script.illegal)


class Capture(BandObserver):
    def __init__(self):
        self.events = []
        self.layers: list[BandLayer] = []

    def layer_started(self, depth, beam):
        self.events.append(("started", depth, tuple(beam)))

    def expansion_attempted(
        self, depth, node, op_index, successor, reason, fuel, stepped
    ):
        self.events.append(
            ("attempt", depth, node, op_index, successor, reason, fuel, stepped)
        )

    def layer_scored(self, depth, log_probs, legal):
        # Illegal entries are outside the search contract. NaN-aware legal-only trace.
        values = tuple("nan" if np.isnan(v) else float(v) for v in log_probs[legal])
        self.events.append(("scored", depth, legal.shape, legal.tobytes(), values))

    def layer_retained(self, depth, beam, verdicts, parents):
        self.events.append(
            ("retained", depth, tuple(beam), dict(verdicts), tuple(parents))
        )

    def search_terminated(self, termination, depth, executions, fuel, expansions):
        self.events.append(
            ("terminated", termination, depth, executions, fuel, expansions)
        )

    def band_retained(self, layer):
        self.layers.append(layer)


@dataclass(frozen=True)
class Case:
    id: str
    task: str = "T-probe"
    script: Script = field(default_factory=Script)
    width: int = 4
    horizon: int = 1
    guard: str = "off"
    cap: int | None = None
    budget: int | None = None
    unit: str = "executions"
    deadline: float | None = None
    band: float | None = 0.0
    expectation: str = "retention-oracle"

    def settings(self) -> dict:
        return dict(
            beam_width=self.width,
            max_depth=self.horizon,
            guard=self.guard,
            max_state_atoms=self.cap,
            budget=self.budget,
            budget_unit=self.unit,
            deadline=self.deadline,
        )


def scores(root: dict[str, float], later=()) -> Script:
    return Script(
        tuple((0, 0, name, value) for name, value in root.items()) + tuple(later)
    )


def disabled_cases() -> tuple[Case, ...]:
    return tuple(
        Case(
            f"disabled/{task}/s{s}/w{w}/h{h}/{g}",
            task,
            Script(seed=s),
            w,
            h,
            g,
            band=None,
            expectation="reference-equivalence",
        )
        for task in ("T-pair", "T-list", "T-typed")
        for s in (0, 1, 2)
        for w in (1, 4, 16)
        for h in (1, 4, 8)
        for g in ("off", "body", "close", "both")
    )


def semantic_cases() -> tuple[Case, ...]:
    cases = [
        Case(
            "A-safe",
            "T-single",
            scores(dict(mul=0, drop=-2, quote=-3, dup=-4), [(1, 0, "dup", 0)]),
            horizon=3,
            band=1,
            expectation="safe-6-vs-3",
        ),
        Case(
            "B-only-solution-cut",
            "T-single",
            scores(
                dict(drop=0, mul=-2, quote=-3, dup=-4),
                [(1, 0, "dup", -10), (1, 1, "dup", 0)],
            ),
            horizon=3,
            band=1,
            expectation="reference-fit-finite-failure",
        ),
        Case(
            "C-later-first-fit",
            "T-single",
            scores(
                dict(mul=0, dup=-2, drop=-3, quote=-4),
                [(1, 0, "dup", -10), (1, 1, "add", 0)],
            ),
            horizon=3,
            band=1,
            expectation="changed-first-fit",
        ),
        Case(
            "fit/enumeration",
            "T-same",
            Script(raises=True),
            expectation="add-no-scorer",
        ),
        Case(
            "fit/before-size",
            "T-same",
            Script(raises=True),
            cap=0,
            expectation="add-no-scorer",
        ),
        Case("anchor/seen", "T-equal-probe", scores(dict(swap=0, add=-2, mul=-3))),
        Case("anchor/close", "T-empty", scores({"[": 0, "0": -2}), guard="close"),
        Case("anchor/type", "T-open", scores({"null?": 0, "]": -2}), guard="body"),
        Case("anchor/size", script=scores(dict(dup=0, mul=-2)), cap=2),
        Case("duplicate/same-layer", "T-equal-probe", scores(dict(add=0, mul=0))),
        Case(
            "seen/reentry",
            script=scores(dict(swap=0, mul=-2, drop=-3, quote=-4), [(1, 0, "mul", 0)]),
            horizon=2,
            band=1,
        ),
        Case("identity/encoding", "T-long", scores(dict(dup=0, drop=0)), cap=200),
        Case(
            "boundary/included", script=scores(dict(mul=0, drop=-1, quote=-2)), band=1
        ),
        Case("boundary/top-ties", script=scores(dict(mul=0, drop=0, quote=-1))),
        Case(
            "boundary/top-ties-w1",
            script=scores(dict(mul=0, drop=0, quote=-1)),
            width=1,
        ),
        Case(
            "ordering/cumulative",
            "T-single",
            scores(
                dict(mul=0, dup=-2, drop=-3, quote=-4),
                [(1, 0, "dup", -10), (1, 1, "add", 0)],
            ),
            horizon=3,
            band=1,
        ),
        Case(
            "band/inactive",
            script=Script(default=0),
            horizon=2,
            band=2,
            expectation="reference-equivalence",
        ),
        Case("score/all-inf", script=Script(default=-np.inf), expectation="all-inf"),
        Case(
            "score/illegal-nan",
            script=Script(default=0, illegal=np.nan),
            horizon=2,
            expectation="reference-equivalence",
        ),
        Case("empty/eligible", "T-empty", cap=0, expectation="frontier-exhausted"),
        Case(
            "stop/width-before-unread-guard",
            "T-empty",
            Script(default=0),
            width=5,
            guard="close",
            cap=1,
            band=2,
            expectation="width",
        ),
    ]
    for band in (None, 1):
        cases.append(
            Case(
                f"stop/last-ranked-width/{band}",
                "T-empty",
                Script(default=0),
                width=6,
                band=band,
                expectation="exhausted-at-width",
            )
        )
    for invalid, value in (("nan", np.nan), ("posinf", np.inf)):
        for scope, task, op, cap in (
            ("late", "T-probe", "quote", None),
            ("size", "T-probe", "dup", 2),
            ("seen", "T-equal-probe", "swap", None),
        ):
            cases.append(
                Case(
                    f"score/executable-{invalid}/{scope}",
                    task,
                    scores({op: value}),
                    cap=cap,
                    expectation="ValueError",
                )
            )
    return tuple(cases)


def resource_cases() -> tuple[Case, ...]:
    cases = []
    for unit in ("executions", "fuel", "executions+fuel"):
        for cap in (None, 64):
            for budget in (0, 1, 7):
                for band in (None, 1):
                    cases.append(
                        Case(
                            f"resource/{unit}/{cap}/{budget}/{band}",
                            "T-pair",
                            width=4,
                            horizon=3,
                            guard="both",
                            cap=cap,
                            budget=budget,
                            unit=unit,
                            band=band,
                            expectation="budget-before-score",
                        )
                    )
            body = Script(
                tuple(
                    (i, 0, name, 0)
                    for i, name in enumerate(("nil", "[", "cons", "]", "fold"))
                )
            )
            cases.append(
                Case(
                    f"resource/body/{unit}/{cap}",
                    "T-list",
                    body,
                    horizon=6,
                    cap=cap,
                    budget=100,
                    unit=unit,
                    band=None,
                    expectation="reference-equivalence",
                )
            )
    cases.append(
        Case(
            "resource/past-deadline",
            "T-same",
            Script(raises=True),
            deadline=0,
            expectation="past-deadline",
        )
    )
    return tuple(cases)
