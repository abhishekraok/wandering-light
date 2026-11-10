"""The optional policy adviser: what a trained ``stack_policy`` would try next.

Nothing here is needed to run the workshop. torch and the policy modules are
imported only when a checkpoint is loaded, and every failure is recorded in
the status instead of raised, so a workshop without weights still starts.

A checkpoint is found, in order: the ``checkpoint`` argument, the
``STACK_POLICY_CHECKPOINT`` environment variable, the #101 campaign's local
weights, the stack debugger's artifact cache. ``download=True`` lets the
stack debugger's resolver fetch the archived W&B artifact when none of those
exist.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

from wandering_light.stack_workshop.engine import (
    OPS,
    final_states,
    snapshot,
    try_step,
)

SUGGESTIONS = 6
CHECKPOINT_ENV = "STACK_POLICY_CHECKPOINT"


class Adviser:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scorer: Any = None
        self.checkpoint: str | None = None
        self.provenance: dict[str, Any] | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.scorer is not None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checkpoint": self.checkpoint,
            "provenance": self.provenance,
            "error": self.error,
        }

    def _resolve(
        self, checkpoint: str | Path | None, download: bool
    ) -> tuple[Path, dict[str, Any]]:
        from wandering_light.stack_policy import MANIFEST_FILE

        if checkpoint is None and os.environ.get(CHECKPOINT_ENV):
            checkpoint = os.environ[CHECKPOINT_ENV]
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            if not (path / MANIFEST_FILE).exists():
                raise FileNotFoundError(f"{path} is not a stack policy checkpoint")
            return path, {"source": "local", "path": str(path.resolve())}

        from wandering_light.stack_debugger import context

        candidates = [
            context._main_checkout() / context.LOCAL_CHECKPOINTS / context.DEFAULT_ARM,
            context.DEFAULT_CACHE
            / context.DEFAULT_ARTIFACT.replace("/", "__").replace(":", "__"),
        ]
        for path in candidates:
            if (path / MANIFEST_FILE).exists():
                return path, {"source": "local", "path": str(path.resolve())}
        if not download:
            raise FileNotFoundError(
                "No local stack policy checkpoint; pass a checkpoint path, set "
                f"{CHECKPOINT_ENV}, or load with download=True."
            )
        return context.resolve_checkpoint(None)

    def load(
        self, checkpoint: str | Path | None = None, download: bool = False
    ) -> dict[str, Any]:
        """Load a ``stack_policy`` checkpoint; the status records any failure."""
        with self._lock:
            try:
                from wandering_light.stack_policy import PolicyScorer, load_policy

                path, provenance = self._resolve(checkpoint, download)
                scorer = PolicyScorer(load_policy(path))
            except Exception as error:  # recorded, not raised: the adviser is optional
                self.error = f"{type(error).__name__}: {error}"
            else:
                self.scorer = scorer
                self.checkpoint = str(path)
                self.provenance = provenance
                self.error = None
            return self.status()

    def rank(self, program: Any, examples: Any) -> dict[str, Any]:
        """The policy's most probable next ops after ``program``.

        Raises ``ValueError`` when a step of the program is refused. Only ops
        legal on every example are ranked; each suggestion previews the stack
        it would leave on every example.
        """
        states = final_states(program, examples)
        scorer = self.scorer
        if scorer is None:
            return {**self.status(), "suggestions": []}

        import numpy as np

        from wandering_light.stack_policy import (
            OP_NAMES,
            encode_item,
            encode_targets,
            item_array,
        )

        previews: dict[str, list] = {}
        for name in OP_NAMES:
            after = []
            for stack, spent, _ in states:
                outcome = try_step(name, stack, spent)
                if isinstance(outcome, str):
                    break
                after.append(outcome[0])
            else:
                previews[name] = after
        if not previews:
            return {**self.status(), "suggestions": []}

        legal = np.array([[name in previews for name in OP_NAMES]], dtype=bool)
        targets = encode_targets([wanted for _, _, wanted in states])
        block = item_array(
            [
                encode_item(stack, target)
                for (stack, _, _), target in zip(states, targets, strict=True)
            ]
        )
        with self._lock:
            log_probs = scorer.log_probs([block], legal)[0]
        ranked = sorted(previews, key=lambda name: -log_probs[OP_NAMES.index(name)])
        suggestions = [
            {
                "op": name,
                "probability": float(math.exp(log_probs[OP_NAMES.index(name)])),
                "previews": [snapshot(stack) for stack in previews[name]],
            }
            for name in ranked[:SUGGESTIONS]
            if name in OPS
        ]
        return {**self.status(), "suggestions": suggestions}


ADVISER = Adviser()
