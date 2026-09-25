"""novelty-reduction (#171): ridge logistic probes on frozen features, shared by M2b and M3.

A probe is ``evaluate.fit_logistic`` on standardised features with lambda
chosen by lineage-grouped CV inside the fitting set, on the mean-BCE scale
(mean BCE + lambda ||w||^2 / 2, intercept free; ``fit_logistic``'s ridge acts
on the summed loss, so it is passed lambda * rows).

The E operating point (``set_threshold``) is the rule settled on #171 after
M2b: the largest e0 among the gate partition's own E_hat values (and 1.0)
whose open set ``E_hat < e0`` holds at least ``min_open`` pairs with
Wilson-hit precision at least ``precision``; closed (None) if none does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.certified_coverage import score as scoring
from experiments.novelty_reduction import evaluate as ev

if TYPE_CHECKING:
    from collections.abc import Sequence

    from experiments.novelty_reduction.archive import LoopArchive

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def folds_of(groups, k: int, seed: int) -> np.ndarray:
    order = list(dict.fromkeys(groups))
    np.random.default_rng(seed).shuffle(order)
    fold = {g: i % k for i, g in enumerate(order)}
    return np.array([fold[g] for g in groups])


def ridge(
    x,
    hits,
    n: int,
    groups,
    seed: int,
    lambdas: Sequence[float] = LAMBDAS,
    folds: int = 5,
) -> dict:
    """Standardised ridge logistic, lambda by lineage CV inside ``x`` (mean-BCE scale)."""
    x, hits = np.asarray(x, dtype=float), np.asarray(hits)
    fold = folds_of(groups, folds, seed)
    table = {}
    for lam in lambdas:
        losses = []
        for f in range(folds):
            train, held = fold != f, fold == f
            if not held.any() or not train.any():
                continue
            model = ev.fit_logistic(x[train], hits[train], n, ridge=lam * train.sum())
            losses.append(
                ev.log_loss(ev.predict_logistic(model, x[held]), hits[held], n)
            )
        table[lam] = float(np.concatenate(losses).mean()) if losses else np.inf
    best = min(table, key=lambda lam: (table[lam], lam))
    model = ev.fit_logistic(x, hits, n, ridge=best * len(x))
    model["lambda"] = best
    model["cv"] = {str(k): v for k, v in table.items()}
    return model


class Probe:
    """A fitted logistic with ``predict``, so it stands where an MLP head stood."""

    def __init__(self, model: dict):
        self.model = model

    def predict(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0)
        return ev.predict_logistic(self.model, features)


def plain_logistic(model: dict, names: Sequence[str]) -> dict:
    """Scaler, weights (intercept first), lambda and feature names, as plain JSON."""
    return {
        "features": list(names),
        "mean": np.asarray(model["mean"]).tolist(),
        "scale": np.asarray(model["scale"]).tolist(),
        "theta": np.asarray(model["theta"]).tolist(),
        "lambda": model.get("lambda"),
        "cv": model.get("cv"),
    }


def logistic_from(plain: dict) -> dict:
    """A plain logistic model back into ``evaluate.predict_logistic``'s form."""
    return {k: np.asarray(plain[k]) for k in ("mean", "scale", "theta")}


def trunk_names(width: int) -> list[str]:
    return [f"pooled_mean_{i}" for i in range(width // 2)] + [
        f"pooled_max_{i}" for i in range(width - width // 2)
    ]


def set_threshold(
    e_hat: np.ndarray,
    p_hat: np.ndarray,
    hits: np.ndarray,
    n: int,
    min_open: int = 10,
    precision: float = 0.8,
) -> dict:
    """The largest qualifying e0 on a gate partition, with what it opens there."""
    e_hat = np.asarray(e_hat, dtype=float)
    hit = scoring.hit_wilson(p_hat, hits, n) if len(e_hat) else np.zeros(0, bool)
    best = None
    for e0 in sorted({*np.round(e_hat, 6).tolist(), 1.0}):
        open_ = e_hat < e0
        if open_.sum() >= min_open and hit[open_].mean() >= precision:
            best = e0
    facts = {
        "e0": best,
        "pairs": len(e_hat),
        "min_open": min_open,
        "precision_target": precision,
        "e_hat_quantiles": {
            str(q): float(np.quantile(e_hat, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        if len(e_hat)
        else {},
    }
    if best is not None:
        open_ = e_hat < best
        facts.update(
            open_pairs=int(open_.sum()),
            open_share=float(open_.mean()),
            precision=float(hit[open_].mean()),
        )
    return facts


def count_novelty(archive: LoopArchive, s_states, t_states) -> np.ndarray:
    """max over s, t of 1 / sqrt(1 + v); a state the archive never met has v = 0."""

    def one(state):
        node = archive.index.get(state)
        return 1.0 if node is None else archive.novelty(node)

    return np.array(
        [max(one(s), one(t)) for s, t in zip(s_states, t_states, strict=True)]
    )
