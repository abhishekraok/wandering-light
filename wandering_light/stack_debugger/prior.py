"""What the training witnesses do next, as a baseline for the policy's ranking.

The policy was cloned from `stack_pbe_v4`'s train witnesses, 175 programs on
fresh inputs. A ranking it produces can be read against the plain frequency
of the next op in those witnesses: where the two agree the policy may be
repeating the corpus, where they differ it has read something off the state
or the target. The condition is the previous op and whether the state is
inside an open quotation -- the smallest condition that tells "after `]` at
top level" (mostly `fold`) from "after `]` inside a body" (mostly `[`).

Counted over train *tasks*, which is what the trainer saw: every task replays
its behaviour's witness, so a behaviour with more tasks weighs more.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

START = "start"
Key = tuple[str, bool]  # (previous op or START, inside an open quotation)


def _condition(previous: str | None, inside: bool) -> str:
    where = "inside an open quotation" if inside else "at top level"
    return f"after {previous} {where}" if previous else "at the start"


class TrainPrior:
    def __init__(self, tasks: Sequence[Task]) -> None:
        counts: dict[Key, Counter[str]] = defaultdict(Counter)
        for task in tasks:
            if task.split != "train":
                continue
            depth = 0
            previous: str | None = None
            for op in task.program:
                counts[previous or START, depth > 0][op] += 1
                if op == "[":
                    depth += 1
                elif op == "]":
                    depth = max(0, depth - 1)
                previous = op
        self.counts = dict(counts)
        self.tasks = sum(1 for task in tasks if task.split == "train")

    def after(self, path: Sequence[str]) -> dict[str, Any]:
        """The next-op distribution the train witnesses show from this path's shape."""
        depth = 0
        for op in path:
            if op == "[":
                depth += 1
            elif op == "]":
                depth = max(0, depth - 1)
        previous = path[-1] if path else None
        row = self.counts.get((previous or START, depth > 0), Counter())
        total = sum(row.values())
        return {
            "condition": _condition(previous, depth > 0),
            "n": total,
            "ops": {op: round(count / total, 4) for op, count in row.most_common()}
            if total
            else {},
        }
