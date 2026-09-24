"""Optional, lazy access to the existing trained policy; never blocks startup."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .engine import (
    FUEL,
    OPS,
    ROOT,
    compile_program,
    machine_operation,
    stack_from,
    step_raw,
    text_stack,
)

DEFAULT_ARTIFACT = (
    "abhishekraok-na/wandering-light-rollout-policy/mark28-policy-2m-s1:v0"
)
CACHE = ROOT / ".workshop-cache" / "policy"


class Adviser:
    def __init__(self):
        self.scorer = None
        self.lock = threading.Lock()
        self.error = None
        self.path = None
        self.loading = False

    def status(self):
        return {
            "ready": self.scorer is not None,
            "loading": self.loading,
            "error": self.error,
            "checkpoint": str(self.path) if self.path else None,
            "artifact": DEFAULT_ARTIFACT,
            "note": "Ranked next instructions, not a probability of solving the task.",
        }

    def load(self, checkpoint=None, download=False):
        with self.lock:
            self.loading = True
            self.error = None
            try:
                import torch

                from wandering_light.stack_policy import PolicyScorer, load_policy

                torch.set_num_threads(2)
                path = Path(
                    checkpoint or os.environ.get("STACK_POLICY_CHECKPOINT", CACHE)
                )
                if not (path / "policy_config.json").exists():
                    if not download:
                        raise ValueError(
                            "No checkpoint loaded. Connect a local checkpoint or download the archived 2m-s1 policy."
                        )
                    import wandb

                    path.mkdir(parents=True, exist_ok=True)
                    artifact = wandb.Api(timeout=30).artifact(
                        DEFAULT_ARTIFACT, type="model"
                    )
                    artifact.download(root=str(path))
                self.scorer = PolicyScorer(load_policy(path), "cpu")
                if self.scorer.encoding != "machine":
                    raise ValueError(
                        f"workshop: machine blocks only, not {self.scorer.encoding}"
                    )
                self.path = path
            except Exception as exc:
                # A failed replacement must not leave the previous checkpoint
                # reported as the connected one.
                self.scorer, self.path = None, None
                self.error = str(exc)
            finally:
                self.loading = False
            return self.status()

    @machine_operation
    def rank(self, program, examples):
        if self.scorer is None:
            return {**self.status(), "suggestions": []}
        import numpy as np

        from wandering_light.stack_policy import encode_item, encode_targets, item_array

        product = [stack_from(e["input"]) for e in examples]
        outputs = [stack_from(e["output"], expected=True) for e in examples]
        fuel = [0] * len(product)
        for token in compile_program(program):
            for i, stack in enumerate(product):
                result = step_raw(OPS[token["op"]], stack, FUEL - fuel[i], 0)
                if isinstance(result, str):
                    raise ValueError(
                        f"Example {i + 1}: {result}. Move the timeline before this error to ask for guidance."
                    )
                product[i], spent = result
                fuel[i] += spent
        legal, previews = [], []
        for op in OPS.values():
            results = [
                step_raw(op, s, FUEL - f, 0) for s, f in zip(product, fuel, strict=True)
            ]
            valid = all(not isinstance(r, str) for r in results)
            legal.append(valid)
            previews.append([text_stack(r[0]) for r in results] if valid else [])
        if not any(legal):
            return {
                **self.status(),
                "suggestions": [],
                "message": "No instruction is legal on every example.",
            }
        block = item_array(
            [
                encode_item(s, t)
                for s, t in zip(product, encode_targets(outputs), strict=True)
            ]
        )
        with self.lock:
            probabilities = np.exp(self.scorer.log_probs([block], np.array([legal]))[0])
        order = np.argsort(-probabilities)
        return {
            **self.status(),
            "suggestions": [
                {
                    "op": list(OPS)[i],
                    "probability": float(probabilities[i]),
                    "previews": previews[i],
                }
                for i in order
                if legal[i]
            ][:6],
        }


ADVISER = Adviser()
