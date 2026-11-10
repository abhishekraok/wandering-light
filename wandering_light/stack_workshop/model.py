"""The optional adviser: a trained `mark-28` policy's view of the next op.

Importing this module costs nothing. torch and the policy code are imported
only when a checkpoint is loaded, so the workshop runs without them and says
so in ``status()`` instead of failing.

The adviser is shown the same thing the beam search shows the policy: one
``(current stack, expected output)`` item per example, and a mask of the ops
the machine accepts on every example.
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

from wandering_light.stack_workshop.engine import (
    FUEL,
    LOCK,
    OPS,
    compile_program,
    final_stacks,
    machine_step,
    read_examples,
    snapshot,
)

SUGGESTIONS = 6


def find_checkpoint(checkpoint: str | Path | None, download: bool) -> Path:
    """A stack-policy checkpoint directory, without the network unless asked.

    In order: the directory given, ``$STACK_POLICY_CHECKPOINT``, the archived
    2m-s1 artifact if the stack debugger already cached it. With ``download``
    the stack debugger's own resolution takes over, which may fetch that
    artifact from W&B.
    """
    if checkpoint is not None and not Path(checkpoint).expanduser().is_dir():
        # Said before anything heavy is imported, so the reason is the path
        # and not a missing torch.
        raise FileNotFoundError(f"{checkpoint} is not a stack policy checkpoint")

    from wandering_light.stack_debugger.context import (
        CHECKPOINT_ENV,
        DEFAULT_ARTIFACT,
        DEFAULT_CACHE,
        resolve_checkpoint,
    )
    from wandering_light.stack_policy import CONFIG_FILE, MODEL_FILE

    def holds_policy(path: Path) -> bool:
        return (path / CONFIG_FILE).is_file() and (path / MODEL_FILE).is_file()

    given = checkpoint if checkpoint is not None else os.environ.get(CHECKPOINT_ENV)
    if given:
        path = Path(given).expanduser()
        if not holds_policy(path):
            raise FileNotFoundError(f"{path} is not a stack policy checkpoint")
        return path
    cached = DEFAULT_CACHE / DEFAULT_ARTIFACT.replace("/", "__").replace(":", "__")
    if holds_policy(cached):
        return cached
    if download:
        return resolve_checkpoint(None)[0]
    raise FileNotFoundError(
        "No stack policy checkpoint on this machine. Give a checkpoint "
        f"directory, set {CHECKPOINT_ENV}, or allow a download."
    )


class Adviser:
    """Holds at most one loaded policy and ranks next ops with it."""

    def __init__(self) -> None:
        self._loading = threading.Lock()
        self._scoring = threading.Lock()
        self._scorer: Any = None
        self._loaded: dict[str, Any] = {}
        self._error: str | None = None

    def status(self) -> dict[str, Any]:
        """What is loaded, and the error from the last attempt to load."""
        return {
            "ready": self._scorer is not None,
            "checkpoint": self._loaded.get("checkpoint"),
            "parameters": self._loaded.get("parameters"),
            "arm": self._loaded.get("arm"),
            "wandb_url": self._loaded.get("wandb_url"),
            "error": self._error,
        }

    def load(
        self, checkpoint: str | Path | None = None, download: bool = False
    ) -> dict[str, Any]:
        """Load a ``stack_policy`` checkpoint; never raises, the status says why.

        A failed load leaves whatever was loaded before in place.
        """
        with self._loading:
            try:
                path = find_checkpoint(checkpoint, download)
                from wandering_light.stack_policy import (
                    MANIFEST_FILE,
                    PolicyScorer,
                    load_policy,
                )

                scorer = PolicyScorer(load_policy(path, device="cpu"), "cpu")
                manifest_file = path / MANIFEST_FILE
                manifest = (
                    json.loads(manifest_file.read_text())
                    if manifest_file.is_file()
                    else {}
                )
            except Exception as error:
                # Anything at all: torch missing, a wrong directory, a
                # checkpoint for another basis, the network. The workshop
                # carries on without an adviser either way.
                self._error = f"{type(error).__name__}: {error}"
            else:
                self._scorer = scorer
                self._loaded = {
                    "checkpoint": str(path.resolve()),
                    "parameters": scorer.model.parameter_count,
                    "arm": manifest.get("arm"),
                    "wandb_url": manifest.get("wandb_url"),
                }
                self._error = None
            return self.status()

    def rank(self, program: str, examples: Any) -> dict[str, Any]:
        """The policy's most probable next ops after ``program``.

        The program is run on every example first, and a refused step is a
        ``ValueError``: there is no state to advise from. Only ops the machine
        accepts on every example are candidates, and their probabilities are
        normalised over that set, as in the beam. ``previews`` holds the stack
        each example would have after the op. With no policy loaded the
        suggestions are empty.
        """
        tokens = compile_program(program)
        pairs = read_examples(examples)
        finals = final_stacks(tokens, [given for given, _ in pairs])
        scorer, status = self._scorer, self.status()
        if scorer is None:
            return {**status, "suggestions": []}

        moves: dict[str, list[tuple]] = {}
        with LOCK:
            for name, op in OPS.items():
                after = []
                for stack, spent in finals:
                    outcome = machine_step(op, stack, FUEL - spent)
                    if isinstance(outcome, str):
                        break
                    after.append(outcome[0])
                else:
                    moves[name] = after
        if not moves:
            return {**status, "suggestions": []}

        import numpy as np

        from wandering_light.stack_policy import (
            OP_NAMES,
            encode_item,
            encode_targets,
            item_array,
        )

        try:
            targets = encode_targets([wanted for _, wanted in pairs])
            block = item_array(
                [
                    encode_item(stack, target)
                    for (stack, _), target in zip(finals, targets, strict=True)
                ]
            )
        except (KeyError, ValueError, RecursionError) as error:
            raise ValueError(f"The adviser cannot read these stacks: {error}") from None
        legal = np.array([[name in moves for name in OP_NAMES]])
        with self._scoring:
            log_probs = scorer.log_probs([block], legal)[0]
        ranked = sorted(
            (
                (float(log_probs[index]), index, name)
                for index, name in enumerate(OP_NAMES)
                if name in moves
            ),
            key=lambda row: (-row[0], row[1]),
        )
        return {
            **status,
            "suggestions": [
                {
                    "op": name,
                    "probability": round(math.exp(log_prob), 6),
                    "previews": [snapshot(stack) for stack in moves[name]],
                }
                for log_prob, _, name in ranked[:SUGGESTIONS]
            ],
        }


ADVISER = Adviser()
