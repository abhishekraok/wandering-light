"""certified-coverage (#170): V_emp from sampled rollouts of a goal-conditioned policy.

A rollout starts at ``s`` with ``t`` as the target block, samples one legal op
at a time from the policy at ``temperature``, and hits when the product equals
``t`` on every shown example within ``budget`` ops. It dies when no op is
legal or the product exceeds ``max_state_atoms`` (`BEST`'s #157 cap). No guard:
legality is only what the machine refuses, as in the beams' mask. V is defined
relative to this procedure, so every result row carries `Procedure.row`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from wandering_light.stack_beam import atoms_exceed, step_all_quiet
from wandering_light.stack_pbe import OPS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from experiments.certified_coverage.pairs import Pair


@dataclass(frozen=True, slots=True)
class Procedure:
    policy: str  # a checkpoint id, e.g. "staged-s0-e3@e5a0b139"
    budget: int = 16
    temperature: float = 1.0
    n: int = 8
    max_state_atoms: int = 16_384

    def row(self) -> dict:
        return {"rollout": asdict(self)}


def _successors(product) -> tuple[np.ndarray, list]:
    legal = np.zeros(len(OPS), dtype=bool)
    moved = [None] * len(OPS)
    for index, op in enumerate(OPS):
        after, _fuel, _stepped = step_all_quiet(product, op)
        if after is not None:
            legal[index] = True
            moved[index] = after
    return legal, moved


def rollout_hits(
    scorer,
    pairs: Sequence[Pair],
    procedure: Procedure,
    seed: int,
    *,
    batch: int = 256,
) -> np.ndarray:
    """Hits out of ``procedure.n`` for each pair; walkers run ``batch`` at a time.

    ``scorer`` needs `PolicyScorer`'s ``encoder(target)`` and
    ``log_probs(blocks, legal)``. Walkers are ordered pair-major, so a seed
    gives the same draws whatever ``batch`` is only when ``batch`` is fixed.
    """
    rng = np.random.default_rng(seed)
    encoders = [scorer.encoder(pair.t) for pair in pairs]
    walkers = [index for index in range(len(pairs)) for _ in range(procedure.n)]
    hits = np.zeros(len(pairs), dtype=np.int64)
    for start in range(0, len(walkers), batch):
        owners = walkers[start : start + batch]
        products = {w: pairs[owner].s for w, owner in enumerate(owners)}
        for _step in range(procedure.budget):
            if not products:
                break
            live, masks, options = [], [], []
            for w, product in products.items():
                legal, moved = _successors(product)
                if legal.any():
                    live.append(w)
                    masks.append(legal)
                    options.append(moved)
            if not live:
                break
            blocks = [encoders[owners[w]](products[w]) for w in live]
            log_probs = scorer.log_probs(blocks, np.stack(masks))
            tempered = np.where(
                np.stack(masks), log_probs / procedure.temperature, -np.inf
            )
            tempered -= tempered.max(axis=1, keepdims=True)
            weights = np.exp(tempered)
            weights /= weights.sum(axis=1, keepdims=True)
            draws = rng.random(len(live))
            picks = (weights.cumsum(axis=1) < draws[:, None]).sum(axis=1)
            following = {}
            for w, row, drawn, moved in zip(live, weights, picks, options, strict=True):
                # A draw on the float edge can step past the last legal op.
                after = moved[min(int(drawn), int(np.flatnonzero(row > 0)[-1]))]
                if after == pairs[owners[w]].t:
                    hits[owners[w]] += 1
                elif not atoms_exceed(after, procedure.max_state_atoms):
                    following[w] = after
            products = following
    return hits
