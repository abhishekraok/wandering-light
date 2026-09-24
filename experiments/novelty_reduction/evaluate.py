"""novelty-reduction (#171): arms, controls and proper scores on shared rollouts.

Every arm is scored on the same evaluation rollouts (F3). Per arm:

- **proper scores** of its P over every pair, gate or not: per-rollout Brier
  and binomial log loss, with a bootstrap interval clustered by lineage (F1);
- **certified coverage and false certification** under each of its gates
  (#170's registered readout).

The Wilson hit rule can reward a biased P (F1), so the proper scores and the
``wilson_constant`` control (the constant that maximises Wilson hits on the
fitting partition) sit beside it; ``zero`` (P = 0) is there because a weak
policy makes "everything fails" easy to certify (F2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.certified_coverage import score as scoring

if TYPE_CHECKING:
    from collections.abc import Sequence

CLIP = 1e-4


def brier(p_hat: np.ndarray, hits: np.ndarray, n: int) -> np.ndarray:
    """Per pair, the mean over its n rollouts of (P_hat - outcome)^2."""
    v = np.asarray(p_hat, dtype=float)
    h = np.asarray(hits, dtype=float)
    return (h * (1 - v) ** 2 + (n - h) * v**2) / n


def log_loss(p_hat: np.ndarray, hits: np.ndarray, n: int) -> np.ndarray:
    """Per pair, the mean binomial log loss per rollout, P_hat clipped to [1e-4, 1 - 1e-4]."""
    v = np.clip(np.asarray(p_hat, dtype=float), CLIP, 1 - CLIP)
    h = np.asarray(hits, dtype=float)
    return -(h * np.log(v) + (n - h) * np.log(1 - v)) / n


def clustered_interval(
    values: np.ndarray, clusters: Sequence, seed: int = 0, reps: int = 1000
) -> dict:
    """The mean and a 90% bootstrap interval, resampling whole clusters."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": None, "low": None, "high": None, "clusters": 0}
    labels = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    ids = np.array([labels[c] for c in clusters])
    sums = np.bincount(ids, weights=values, minlength=len(labels))
    counts = np.bincount(ids, minlength=len(labels))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(reps, len(labels)))
    means = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.05)),
        "high": float(np.quantile(means, 0.95)),
        "clusters": len(labels),
    }


def fit_logistic(
    features: np.ndarray,
    hits: np.ndarray,
    n: int,
    ridge: float = 1e-3,
    steps: int = 100,
) -> dict:
    """Binomial logistic regression by Newton's method on standardised features."""
    x = np.asarray(features, dtype=float)
    if len(x) == 0:  # nothing to fit on: P = 0.5 everywhere
        width = x.shape[1] if x.ndim == 2 else 0
        return {
            "mean": np.zeros(width),
            "scale": np.ones(width),
            "theta": np.zeros(width + 1),
        }
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.hstack([np.ones((len(x), 1)), (x - mean) / scale])
    y = np.asarray(hits, dtype=float) / n
    theta = np.zeros(design.shape[1])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(steps):
        prob = 1 / (1 + np.exp(-(design @ theta)))
        gradient = design.T @ (y - prob) - penalty @ theta
        curvature = (
            design.T @ (design * (prob * (1 - prob))[:, None])
            + penalty
            + 1e-9 * np.eye(len(theta))
        )
        update = np.linalg.solve(curvature, gradient)
        theta += update
        if np.abs(update).max() < 1e-9:
            break
    return {"mean": mean, "scale": scale, "theta": theta}


def predict_logistic(model: dict, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=float) - model["mean"]) / model["scale"]
    return 1 / (1 + np.exp(-(model["theta"][0] + x @ model["theta"][1:])))


def wilson_constant(hits: np.ndarray, n: int, grid: int = 101) -> float:
    """The constant P that passes the most Wilson checks on ``hits`` (the gaming control)."""
    candidates = np.linspace(0, 1, grid)
    passes = [
        scoring.hit_wilson(np.full(len(hits), c), hits, n).sum() for c in candidates
    ]
    return float(candidates[int(np.argmax(passes))])


def arm_report(
    stratifications: dict[str, Sequence],
    p_hat: np.ndarray,
    gates: dict[str, np.ndarray],
    hits: np.ndarray,
    n: int,
    clusters: Sequence,
    confidence: np.ndarray | None = None,
) -> dict:
    """One arm on one pool: proper scores and gated shares, under each stratification.

    ``stratifications`` maps a name (d_ops, body, truncated, ...) to a label
    per pair. Proper scores get a lineage-clustered interval overall and a
    mean per stratum; ``confidence``, when given, adds false certification at
    matched coverage (the most confident share of pairs, #171 F12).
    """
    b = brier(p_hat, hits, n)
    ll = log_loss(p_hat, hits, n)
    rule = scoring.Rule()
    out = {
        "brier": clustered_interval(b, clusters),
        "log_loss": clustered_interval(ll, clusters),
        "mean_p": float(np.mean(p_hat)) if len(p_hat) else None,
        "by": {},
    }
    for name, given in stratifications.items():
        labels = np.asarray(given)
        proper = {}
        for stratum in sorted(set(labels.tolist())):
            at = labels == stratum
            proper[str(stratum)] = {
                "pairs": int(at.sum()),
                "brier": float(b[at].mean()),
                "log_loss": float(ll[at].mean()),
            }
        out["by"][name] = {
            "proper": proper,
            "gates": {
                gate: scoring.report(labels, p_hat, mask, hits, n, rule)
                for gate, mask in gates.items()
            },
        }
    if confidence is not None:
        out["matched_coverage"] = risk_at_coverage(confidence, p_hat, hits, n)
    return out


def risk_at_coverage(
    confidence: np.ndarray,
    p_hat: np.ndarray,
    hits: np.ndarray,
    n: int,
    coverages: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> dict:
    """False certification among the most confident ``q`` of pairs, per coverage ``q``."""
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    hit = scoring.hit_wilson(p_hat, hits, n)
    out = {}
    for q in coverages:
        take = order[: round(q * len(order))]
        out[str(q)] = float((~hit[take]).mean()) if len(take) else None
    return out


def paired_gain(
    p_arm: np.ndarray, p_control: np.ndarray, hits: np.ndarray, n: int, clusters
) -> dict:
    """Control's log loss minus the arm's, per pair, clustered: > 0 means the arm is better."""
    return clustered_interval(
        log_loss(p_control, hits, n) - log_loss(p_arm, hits, n), clusters
    )
