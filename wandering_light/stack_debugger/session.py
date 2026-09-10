"""Pick a v4 eval task, run the production beam on it, and keep what it did.

A session is what the page navigates and what a deep link reproduces: it
records the corpus revision, the checkpoint, the search settings and how the
task was chosen. The witness is read from the task *after* the search has
finished -- ``run_search`` never sees it, so there is no path by which the
authored answer could reach the beam.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wandering_light.stack_debugger.trace import OP_NAMES, TraceRecorder
from wandering_light.stack_machine import Config, format_stack, parse_stack
from wandering_light.stack_pbe import BASIS
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import (
    DIGIT_BASE,
    INT_BASE,
    SMALL_INT,
    STATE_CAP,
    SYM_BASE,
    TARGET_CAP,
    VOCAB,
    encode_stack,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_beam import BeamResult
    from wandering_light.stack_pbe import Task
    from wandering_light.stack_policy import PolicyScorer

DEFAULT_BEAM_WIDTH = 64
DEFAULT_MAX_DEPTH = 48
DEFAULT_GUARD = "both"
# compute-optimal-mark-28 (#101): the eval tasks every timed cell ran on.
COHORT_SEED = 17
COHORT_BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))
COHORT_PER_BUCKET = 60


@dataclass(frozen=True)
class SearchSettings:
    beam_width: int = DEFAULT_BEAM_WIDTH
    max_depth: int = DEFAULT_MAX_DEPTH
    guard: str = DEFAULT_GUARD
    budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "guard": self.guard,
            "budget": self.budget,
        }


def bucket_of(length: int) -> str:
    for low, high in COHORT_BUCKETS:
        if low <= length <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    raise ValueError(length)


def cohort_indices(tasks: Sequence[Task]) -> list[int]:
    """The #101 cohort: seed-17 shuffle of the eval split, first 60 per bucket.

    Restated from ``experiments/compute_optimal_mark28.cohort`` rather than
    imported, so serving the page does not import a campaign driver;
    ``tests/test_stack_debugger_session.py`` checks the two agree.
    """
    pool = [(index, task) for index, task in enumerate(tasks) if task.split == "eval"]
    random.Random(COHORT_SEED).shuffle(pool)
    taken: dict[str, list[int]] = {}
    for index, task in pool:
        bucket = bucket_of(task.length)
        if len(taken.setdefault(bucket, [])) < COHORT_PER_BUCKET:
            taken[bucket].append(index)
    return [
        index
        for low, _high in COHORT_BUCKETS
        for index in taken.get(bucket_of(low), [])
    ]


def select_task(
    tasks: Sequence[Task],
    *,
    task: int | None = None,
    behaviour: str | None = None,
    bucket: str | None = None,
    cohort: bool = False,
    seed: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """One eval task by index, or one sampled from a filtered pool with a seed."""
    if task is not None:
        if not 0 <= task < len(tasks):
            raise KeyError(f"No eval task {task}; the split has {len(tasks)}")
        return task, {"mode": "task", "task": task}

    pool = list(range(len(tasks)))
    if cohort:
        members = set(cohort_indices(tasks))
        pool = [index for index in pool if index in members]
    if behaviour is not None:
        pool = [index for index in pool if tasks[index].behaviour == behaviour]
    if bucket is not None:
        pool = [index for index in pool if bucket_of(tasks[index].length) == bucket]
    if not pool:
        raise ValueError(
            f"No eval task matches: behaviour={behaviour}, bucket={bucket}, cohort={cohort}"
        )
    chosen_seed = random.SystemRandom().randrange(2**32) if seed is None else seed
    index = random.Random(chosen_seed).choice(pool)
    return index, {
        "mode": "sampled",
        "seed": chosen_seed,
        "behaviour": behaviour,
        "bucket": bucket,
        "cohort": cohort,
        "pool_size": len(pool),
        "task": index,
    }


def run_search(
    scorer: PolicyScorer, task: Task, settings: SearchSettings
) -> tuple[BeamResult, TraceRecorder]:
    from wandering_light.stack_beam import beam_solve

    recorder = TraceRecorder(scorer=scorer)
    result = beam_solve(
        task,
        scorer,
        beam_width=settings.beam_width,
        max_depth=settings.max_depth,
        guard=settings.guard,
        budget=settings.budget,
        observer=recorder,
    )
    return result, recorder


def witness_programs(task: Task) -> list[list[str]]:
    """The authored witness first, then its verified alternatives."""
    programs = [list(task.program)]
    for alternative in task.alternative_programs:
        if list(alternative) not in programs:
            programs.append(list(alternative))
    return programs


def witness_overlay(
    layers: Sequence[dict[str, Any]], programs: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    """Per layer and per witness: is its prefix in the beam, and where did the
    policy rank its next op from the node that is on it.

    Diagnosis only, computed from the finished trace. A layer at depth d
    expands nodes whose programs are d ops long; the witness node there is the
    one whose program is the witness's first d ops, and the ground truth from
    it is the witness's op d+1.
    """
    overlay = []
    for layer in layers:
        depth = layer["depth"]
        entries = []
        for which, program in enumerate(programs):
            prefix = list(program[:depth])
            node = next(
                (i for i, n in enumerate(layer["nodes"]) if n["program"] == prefix),
                None,
            )
            entry: dict[str, Any] = {
                "witness": which,
                "node": node,
                "survives": any(
                    r["program"] == list(program[: depth + 1])
                    for r in layer["retained"]
                ),
                "next": program[depth] if depth < len(program) else None,
                "rank": None,
                "legal": None,
                "probability": None,
            }
            if node is not None and entry["next"] is not None and layer["scores"]:
                row = layer["scores"][node]
                legal = [
                    (name, value)
                    for name, value in zip(OP_NAMES, row, strict=True)
                    if value is not None
                ]
                legal.sort(key=lambda item: (-item[1], item[0]))
                names = [name for name, _ in legal]
                if entry["next"] in names:
                    entry["rank"] = names.index(entry["next"]) + 1
                    entry["probability"] = round(
                        math.exp(row[OP_NAMES.index(entry["next"])]), 6
                    )
                entry["legal"] = len(names)
            entries.append(entry)
        overlay.append({"depth": depth, "witnesses": entries})
    return overlay


@dataclass
class DebugSession:
    index: int
    task: Task
    settings: SearchSettings
    result: BeamResult
    recorder: TraceRecorder
    provenance: dict[str, Any]
    basis: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        task = self.task
        trace = self.recorder.to_dict()
        programs = witness_programs(task)
        score = grade(self.result.program, task) if self.result.program else None
        return {
            "task": {
                "index": self.index,
                "behaviour": task.behaviour,
                "summary": task.summary,
                "bucket": bucket_of(task.length),
                "length": task.length,
                "examples": task.examples,
                "held_out": task.held_out,
                "inputs": list(task.inputs),
                "outputs": list(task.outputs),
                "tags": list(task.tags),
                "skeleton": task.skeleton,
            },
            "witnesses": programs,
            "settings": self.settings.to_dict(),
            "basis": self.basis,
            "result": {
                "termination": self.result.termination,
                "program": None
                if self.result.program is None
                else list(self.result.program),
                "seconds": round(self.result.seconds, 3),
                "executions": self.result.executions,
                "fuel": self.result.fuel,
                "expansions": self.result.expansions,
                "depth": self.result.depth,
                "shown": None if score is None else score.shown,
                "held_out": None if score is None else score.held_out,
                "solved": None if score is None else score.solved,
            },
            "witness_overlay": witness_overlay(trace["layers"], programs),
            "provenance": self.provenance,
            "trace": trace,
        }


def build_session(
    scorer: PolicyScorer,
    tasks: Sequence[Task],
    index: int,
    *,
    settings: SearchSettings,
    provenance: dict[str, Any],
) -> DebugSession:
    from wandering_light.stack_pbe import OPS

    task = tasks[index]
    result, recorder = run_search(scorer, task, settings)
    return DebugSession(
        index=index,
        task=task,
        settings=settings,
        result=result,
        recorder=recorder,
        provenance=provenance,
        basis=[{"name": op.name, "group": op.group, "effect": op.effect} for op in OPS],
    )


# ---------------------------------------------------------------------------
# What the model sees
# ---------------------------------------------------------------------------


def token_names() -> list[str]:
    names = ["PAD", "SEP", "(", ")", "MARK", "true", "false", "NEG", "BIG"]
    assert len(names) == SYM_BASE
    names += list(OP_NAMES)
    assert len(names) == INT_BASE
    names += [str(value) for value in range(-SMALL_INT, SMALL_INT + 1)]
    assert len(names) == DIGIT_BASE
    names += [f"·{digit}" for digit in range(10)]
    assert len(names) == VOCAB
    return names


_NAMES = token_names()


def encoding_view(product: Sequence[tuple], outputs: Sequence[tuple]) -> dict[str, Any]:
    """The exact token streams the policy is fed for one product state.

    Calls the policy's own ``encode_stack``, so the cap and the keep-the-top
    rule are the real ones. ``overflow`` says how many tokens the cap dropped.
    """
    items = []
    for stack, target in zip(product, outputs, strict=True):
        full = encode_stack(stack, 10**9)
        kept = encode_stack(stack, STATE_CAP)
        target_full = encode_stack(target, 10**9)
        target_kept = encode_stack(target, TARGET_CAP)
        items.append(
            {
                "state_tokens": [_NAMES[t] for t in kept],
                "state_overflow": len(full) - len(kept),
                "target_tokens": [_NAMES[t] for t in target_kept],
                "target_overflow": len(target_full) - len(target_kept),
            }
        )
    return {"state_cap": STATE_CAP, "target_cap": TARGET_CAP, "items": items}


def parse_product(texts: Sequence[str]) -> tuple[tuple, ...]:
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def stack_display(stack: tuple) -> str:
    return format_stack(Config(stack))
