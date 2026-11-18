"""Optional mark-28 policy advice for the current set of examples."""

from __future__ import annotations

import math
import os
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from wandering_light import stack_machine as machine
from wandering_light.stack_workshop.engine import (
    FUEL,
    MACHINE_LOCK,
    OPS,
    _example_stacks,
    _run_tokens,
    compile_program,
    snapshot,
)

if TYPE_CHECKING:
    from wandering_light.stack_policy import PolicyScorer


class Adviser:
    def __init__(self) -> None:
        self._lock = RLock()
        self._scorer: PolicyScorer | None = None
        self._source: dict[str, str] | None = None
        self._error: str | None = None

    def status(self) -> dict:
        with self._lock:
            return {
                "ready": self._scorer is not None,
                "source": self._source.copy() if self._source else None,
                "error": self._error,
            }

    def load(
        self, checkpoint: str | Path | None = None, download: bool = False
    ) -> dict:
        """Load a local policy, or explicitly opt in to the published artifact."""
        with self._lock:
            self._scorer = None
            self._source = None
            self._error = None
            try:
                if checkpoint is None:
                    checkpoint = os.environ.get("STACK_POLICY_CHECKPOINT")
                if checkpoint is not None:
                    path = Path(checkpoint)
                    source = {"source": "local", "path": str(path.resolve())}
                elif download:
                    from wandering_light.stack_debugger.context import (
                        resolve_checkpoint,
                    )

                    path, source = resolve_checkpoint(None)
                else:
                    raise ValueError("Provide a checkpoint path or enable download")

                from wandering_light.rollout_policy import _read_wandb_url
                from wandering_light.stack_policy import (
                    WANDB_LINK_FILE,
                    PolicyScorer,
                    load_policy,
                )

                scorer = PolicyScorer(load_policy(path))
                with suppress(OSError, ValueError):
                    source["wandb_run_url"] = _read_wandb_url(path / WANDB_LINK_FILE)
                self._scorer = scorer
                self._source = source
            except Exception as error:
                self._error = str(error)
            return self.status()

    def rank(self, program: str, examples: list[dict]) -> dict:
        """Score the next ops over the full set of example states and targets."""
        tokens = compile_program(program)
        pairs = _example_stacks(examples)
        with self._lock:
            payload = {**self.status(), "suggestions": []}
            with MACHINE_LOCK:
                runs = [_run_tokens(tokens, initial) for initial, _ in pairs]
                for run in runs:
                    if run.error:
                        raise ValueError(run.error)
                if self._scorer is None:
                    return payload
                successors = {}
                for name, op in OPS.items():
                    advanced = []
                    for run in runs:
                        outcome = machine.step_raw(op, run.stack, FUEL - run.fuel, 0)
                        if isinstance(outcome, str):
                            break
                        advanced.append(outcome[0])
                    else:
                        successors[name] = advanced

            if not successors:
                return payload

            import numpy as np

            from wandering_light.stack_policy import (
                OP_NAMES,
                encode_item,
                encode_targets,
                item_array,
            )

            targets = encode_targets([output for _, output in pairs])
            block = item_array(
                [
                    encode_item(run.stack, target)
                    for run, target in zip(runs, targets, strict=True)
                ]
            )
            mask = np.array([[name in successors for name in OP_NAMES]], dtype=bool)
            scores = self._scorer.log_probs([block], mask)[0]
            ranked = sorted(
                (index for index, name in enumerate(OP_NAMES) if name in successors),
                key=lambda index: -scores[index],
            )[:6]
            payload["suggestions"] = [
                {
                    "op": OP_NAMES[index],
                    "probability": math.exp(float(scores[index])),
                    "previews": [
                        snapshot(stack) for stack in successors[OP_NAMES[index]]
                    ],
                }
                for index in ranked
            ]
            return payload


ADVISER = Adviser()
