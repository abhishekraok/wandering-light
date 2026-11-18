"""Optional one-step policy advice over all of a learner's examples."""

import math
import os
from pathlib import Path

from wandering_light import stack_machine as machine
from wandering_light.stack_workshop import engine


class Adviser:
    def __init__(self):
        self.scorer = None
        self.checkpoint = None
        self.error = None

    def status(self):
        with engine.MACHINE_LOCK:
            return {
                "ready": self.scorer is not None,
                "checkpoint": self.checkpoint,
                "error": self.error,
            }

    def load(self, checkpoint=None, download=False):
        with engine.MACHINE_LOCK:
            try:
                from wandering_light import stack_policy
                from wandering_light.stack_debugger import context

                path = checkpoint or os.environ.get(context.CHECKPOINT_ENV)
                if path is None:
                    if download:
                        path, _ = context.resolve_checkpoint(
                            None, artifact=context.DEFAULT_ARTIFACT
                        )
                    else:
                        artifact = context.DEFAULT_ARTIFACT.replace("/", "__").replace(
                            ":", "__"
                        )
                        path = context.DEFAULT_CACHE / artifact
                        if not path.exists():
                            raise ValueError(
                                "Supply a local checkpoint or enable download."
                            )
                self.scorer = stack_policy.PolicyScorer(
                    stack_policy.load_policy(Path(path))
                )
                self.checkpoint = str(Path(path).resolve())
                self.error = None
            except Exception as exc:
                # Missing dependencies, weights, or artifact access are all optional.
                self.error = str(exc)
            return self.status()

    def rank(self, program, examples):
        tokens = engine.compile_program(program)
        pairs = engine.example_stacks(examples)
        with engine.MACHINE_LOCK:
            current = []
            for initial, _ in pairs:
                _, stack, fuel, error = engine.replay_final(tokens, initial)
                if error:
                    raise ValueError(error)
                current.append((stack, fuel))
            result = {**self.status(), "suggestions": []}
            if self.scorer is None:
                return result

            import numpy as np

            from wandering_light import stack_policy as policy

            legal = np.zeros((1, len(policy.OP_NAMES)), dtype=bool)
            previews = {}
            for index, name in enumerate(policy.OP_NAMES):
                after = []
                for stack, fuel in current:
                    outcome = machine.step_raw(
                        engine.OPS[name], stack, engine.FUEL - fuel, 0
                    )
                    if isinstance(outcome, str):
                        break
                    after.append(outcome[0])
                else:
                    previews[index] = after
                    legal[0, index] = True
            if previews:
                targets = policy.encode_targets([target for _, target in pairs])
                block = policy.item_array(
                    [
                        policy.encode_item(stack, target)
                        for (stack, _), target in zip(current, targets, strict=True)
                    ]
                )
                scores = self.scorer.log_probs([block], legal)[0]
                for i in sorted(previews, key=scores.__getitem__, reverse=True)[:6]:
                    result["suggestions"].append(
                        {
                            "op": policy.OP_NAMES[i],
                            "probability": math.exp(scores[i]),
                            "previews": [engine.snapshot(s) for s in previews[i]],
                        }
                    )
            return result


ADVISER = Adviser()
