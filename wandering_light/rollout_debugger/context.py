"""Load the corpus and the policy once, and hold them warm.

Loading is the slow part: verifying a split and reading a checkpoint takes
seconds, while scoring one state takes milliseconds. A debugging session is
interactive, so this is loaded once per process and reused for every task,
every drive-mode keystroke and every re-run at a different beam width.

The basis a session searches over is the corpus's own, and the policy's head
is pinned to an ordered basis. ``DebugContext.load`` checks they agree before
anything is searched, so a mismatch is a startup error rather than a wrong
answer displayed convincingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.policy_hub import DEFAULT_POLICY_REPO_ID, ensure_policy_checkpoint
from wandering_light.rollout_debugger.session import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_SPLIT,
)

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.basis_set import BasisSet
    from wandering_light.function_def import FunctionDefSet
    from wandering_light.rollout_policy import RolloutPolicy

DEFAULT_POLICY_CACHE = Path.home() / ".cache" / "wandering-light" / "policies"


@dataclass
class DebugContext:
    """A loaded corpus split and a loaded policy, ready to search with."""

    records: list[BasisTaskRecord]
    basis: BasisSet
    available_functions: FunctionDefSet
    policy: RolloutPolicy
    provenance: dict[str, Any]

    @classmethod
    def load(
        cls,
        *,
        corpus_dir: Path = DEFAULT_CORPUS_DIR,
        split: str = DEFAULT_SPLIT,
        checkpoint: Path | None = None,
        repo_id: str = DEFAULT_POLICY_REPO_ID,
        revision: str | None = None,
        device: str = "cpu",
        cache_dir: Path = DEFAULT_POLICY_CACHE,
    ) -> DebugContext:
        from experiments.evaluate_deep_corpus_sft import verify_and_load_split
        from wandering_light.rollout_policy import RolloutPolicy

        manifest, basis, records, _ = verify_and_load_split(Path(corpus_dir), split)
        path, policy_provenance = ensure_policy_checkpoint(
            checkpoint, cache_dir=cache_dir, repo_id=repo_id, revision=revision
        )
        policy = RolloutPolicy.load(path, device=device)
        available_functions = basis.as_function_set()

        if (policy.basis_set_id, policy.basis_set_digest) != (
            basis.basis_set_id,
            basis.digest,
        ):
            raise ValueError(
                "Policy checkpoint basis differs from the corpus basis: "
                f"policy {policy.basis_set_id}@{policy.basis_set_digest[:12]}, "
                f"corpus {basis.basis_set_id}@{basis.digest[:12]}"
            )

        return cls(
            records=list(records),
            basis=basis,
            available_functions=available_functions,
            policy=policy,
            provenance={
                "corpus": {
                    "dir": str(Path(corpus_dir)),
                    "split": split,
                    "basis_set_id": manifest["basis_set_id"],
                    "basis_set_digest": manifest["basis_set_digest"],
                    "hub": manifest.get("hub", {}),
                    "tasks": len(records),
                },
                "policy": policy_provenance,
                "device": device,
            },
        )
