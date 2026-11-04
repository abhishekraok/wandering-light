"""An optional adviser: a `stack_policy` checkpoint ranking the next op.

The workshop runs without it. ``Adviser.load`` imports torch lazily and
records any failure in its status instead of raising, so a machine without
the policy's dependencies still serves every lesson. ``rank`` always computes
which ops the machine would accept on every example; the policy only orders
them.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from wandering_light.stack_machine import MARK_OPS, step_raw
from wandering_light.stack_workshop.engine import (
    FUEL,
    MACHINE_LOCK,
    OPS,
    _examples,
    _tokens,
    snapshot,
)

CHECKPOINT_ENV = "STACK_POLICY_CHECKPOINT"
# ``stack_policy.MANIFEST_FILE``, spelled here so a missing torch cannot stop
# the path check that precedes it.
MANIFEST_FILE = "training_manifest.json"
MAX_SUGGESTIONS = 6


class Adviser:
    def __init__(self) -> None:
        self.scorer: Any = None
        self.provenance: dict[str, Any] | None = None
        self.error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.scorer is not None,
            "provenance": self.provenance,
            "error": self.error,
        }

    def load(
        self, checkpoint: str | Path | None = None, download: bool = False
    ) -> dict:
        """Load a checkpoint; a failure is recorded, not raised."""
        try:
            path, provenance = self._resolve(checkpoint, download=download)
            from wandering_light.stack_policy import PolicyScorer, load_policy

            self.scorer = PolicyScorer(load_policy(path, device="cpu"), "cpu")
            self.provenance = provenance
            self.error = None
        except Exception as error:  # the status carries it
            self.scorer = None
            self.provenance = None
            self.error = f"{type(error).__name__}: {error}"
        return self.status()

    @staticmethod
    def _resolve(checkpoint: str | Path | None, *, download: bool) -> tuple[Path, dict]:
        if checkpoint is None and os.environ.get(CHECKPOINT_ENV):
            checkpoint = os.environ[CHECKPOINT_ENV]
        if checkpoint is not None:
            path = Path(checkpoint)
            if not (path / MANIFEST_FILE).exists():
                raise FileNotFoundError(f"{path} is not a stack policy checkpoint")
            return path, {"source": "local", "path": str(path.resolve())}
        if not download:
            raise FileNotFoundError("no checkpoint given; pass one or allow a download")
        from wandering_light.stack_debugger.context import resolve_checkpoint

        return resolve_checkpoint(None)

    def rank(self, program: str | list[dict], examples: list[dict]) -> dict[str, Any]:
        """Ops legal on every example, most probable first when a policy is loaded."""
        tokens = _tokens(program)
        pairs = _examples(examples)
        stacks = []
        with MACHINE_LOCK:
            for given, _ in pairs:
                stack, spent = given, 0
                for token in tokens:
                    outcome = step_raw(OPS[token["op"]], stack, FUEL - spent, 0)
                    if isinstance(outcome, str):
                        raise ValueError(f"{token['op']}: {outcome}")
                    stack, cost = outcome
                    spent += cost
                stacks.append((stack, spent))
            previews: dict[str, list] = {}
            for op in MARK_OPS:
                after = []
                for stack, spent in stacks:
                    outcome = step_raw(op, stack, FUEL - spent, 0)
                    if isinstance(outcome, str):
                        break
                    after.append(snapshot(outcome[0]))
                else:
                    previews[op.name] = after
        suggestions: list[dict[str, Any]] = []
        if self.scorer is not None and previews:
            probabilities = self._probabilities(
                [stack for stack, _ in stacks],
                [output for _, output in pairs],
                previews,
            )
            ranked = sorted(previews, key=lambda name: -probabilities[name])
            suggestions = [
                {
                    "op": name,
                    "probability": probabilities[name],
                    "previews": previews[name],
                }
                for name in ranked[:MAX_SUGGESTIONS]
            ]
        return {**self.status(), "suggestions": suggestions}

    def _probabilities(
        self, stacks: list[tuple], outputs: list[tuple], legal: dict[str, list]
    ) -> dict[str, float]:
        import numpy as np

        from wandering_light.stack_policy import (
            OP_NAMES,
            encode_item,
            encode_targets,
            item_array,
        )

        targets = encode_targets(outputs)
        block = item_array(
            [
                encode_item(stack, target)
                for stack, target in zip(stacks, targets, strict=True)
            ]
        )
        mask = np.array([[name in legal for name in OP_NAMES]], dtype=bool)
        log_probs = self.scorer.log_probs([block], mask)[0]
        return {
            name: round(math.exp(float(log_probs[index])), 6)
            for index, name in enumerate(OP_NAMES)
            if name in legal
        }


ADVISER = Adviser()
