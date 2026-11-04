"""An optional policy adviser: which op a ``stack_policy`` checkpoint would try next.

Torch and the policy code are imported only when a checkpoint is loaded or
ranked, so the workshop runs without them; ``status()`` says what is missing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from wandering_light.stack_machine import step_raw
from wandering_light.stack_workshop.engine import (
    FUEL,
    MACHINE_LOCK,
    OPS,
    compile_program,
    snapshot,
    stack_from,
)

CHECKPOINT_ENV = "STACK_POLICY_CHECKPOINT"
MAX_SUGGESTIONS = 6


class Adviser:
    def __init__(self) -> None:
        self.scorer: Any = None
        self.checkpoint: str | None = None
        self.error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.scorer is not None,
            "checkpoint": self.checkpoint,
            "error": self.error,
        }

    def load(
        self, checkpoint: str | Path | None = None, download: bool = False
    ) -> dict[str, Any]:
        """Load a ``stack_policy`` checkpoint; a failure is recorded, not raised."""
        try:
            if checkpoint is None:
                checkpoint = os.environ.get(CHECKPOINT_ENV)
            if checkpoint is None and download:
                from wandering_light.stack_debugger.context import resolve_checkpoint

                checkpoint, _provenance = resolve_checkpoint(None)
            if checkpoint is None:
                raise ValueError(
                    f"No checkpoint: pass one, set {CHECKPOINT_ENV}, or allow download"
                )
            from wandering_light.stack_policy import PolicyScorer, load_policy

            self.scorer = PolicyScorer(load_policy(Path(checkpoint)))
            self.checkpoint = str(checkpoint)
            self.error = None
        except Exception as error:
            self.scorer = None
            self.checkpoint = None if checkpoint is None else str(checkpoint)
            self.error = f"{type(error).__name__}: {error}"
        return self.status()

    def rank(self, program: str, examples: Any) -> dict[str, Any]:
        """Run the program, then rank the ops legal on every example."""
        tokens = compile_program(program)
        if not isinstance(examples, list) or not 1 <= len(examples) <= 16:
            raise ValueError("Give between 1 and 16 examples")
        states, targets, spent = [], [], []
        with MACHINE_LOCK:
            for example in examples:
                stack = stack_from(example["input"])
                fuel = 0
                for token in tokens:
                    outcome = step_raw(OPS[token["op"]], stack, FUEL - fuel, 0)
                    if isinstance(outcome, str):
                        raise ValueError(f"{token['op']}: {outcome}")
                    stack, cost = outcome
                    fuel += cost
                states.append(stack)
                targets.append(stack_from(example["output"], expected=True))
                spent.append(fuel)
            scorer = self.scorer
            if scorer is None:
                return {**self.status(), "suggestions": []}
            successors: dict[str, list] = {}
            for name, op in OPS.items():
                moved = []
                for stack, fuel in zip(states, spent, strict=True):
                    outcome = step_raw(op, stack, FUEL - fuel, 0)
                    if isinstance(outcome, str):
                        break
                    moved.append(outcome[0])
                else:
                    successors[name] = moved
        if not successors:
            return {**self.status(), "suggestions": []}

        import numpy as np

        from wandering_light.stack_policy import (
            OP_NAMES,
            encode_item,
            encode_targets,
            item_array,
        )

        encoded = encode_targets(targets)
        block = item_array(
            [encode_item(s, t) for s, t in zip(states, encoded, strict=True)]
        )
        legal = np.array([[name in successors for name in OP_NAMES]])
        log_probs = scorer.log_probs([block], legal)[0]
        ranked = sorted(
            (name for name in OP_NAMES if name in successors),
            key=lambda name: -log_probs[OP_NAMES.index(name)],
        )
        suggestions = [
            {
                "op": name,
                "probability": math.exp(float(log_probs[OP_NAMES.index(name)])),
                "previews": [snapshot(stack) for stack in successors[name]],
            }
            for name in ranked[:MAX_SUGGESTIONS]
        ]
        return {**self.status(), "suggestions": suggestions}


ADVISER = Adviser()
