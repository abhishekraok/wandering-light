"""certified-coverage (#170): the selective-prediction score and its two baselines.

Per judged pair, with ``hits`` of ``n`` rollouts::

    hit   = V_hat inside the 90% Wilson interval of (hits, n)   # primary
    hit   = |V_hat - hits / n| <= tau                           # secondary column
    score = 0 if not gate; +r if gate and hit; -p if gate and miss

Two shares are reported per stratum, never summed: certified coverage (gate
open and hit) and false certification (gate open and miss).

Baselines have no N or E head. Each is fitted on archive pairs:

- ``constant``: V_hat = the base rate of V_emp in the pair's ``d_cuts`` stratum;
- ``distance``: V_hat = a logistic fit on ``d_ops`` alone.

A baseline's gate opens on a ``d_ops`` value whose archive pairs (at least
``min_bin`` of them) hit at a rate of at least ``p / (r + p)``, the break-even
precision; it is also reported with the gate always open.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

Z90 = 1.6448536269514722  # two-sided 90%


@dataclass(frozen=True, slots=True)
class Rule:
    r: float = 1.0
    p: float = 4.0
    tau: float = 0.15
    z: float = Z90
    min_bin: int = 10

    @property
    def precision(self) -> float:
        """The hit rate above which opening the gate pays in expectation."""
        return self.p / (self.r + self.p)


def wilson(hits: np.ndarray, n: int, z: float = Z90) -> tuple[np.ndarray, np.ndarray]:
    hits = np.asarray(hits, dtype=float)
    rate = hits / n
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def hit_wilson(
    v_hat: np.ndarray, hits: np.ndarray, n: int, z: float = Z90
) -> np.ndarray:
    low, high = wilson(hits, n, z)
    v_hat = np.asarray(v_hat, dtype=float)
    # A hair of slack so a V_hat of exactly 0 or 1 sits inside [0, x] or [x, 1].
    return (low - 1e-12 <= v_hat) & (v_hat <= high + 1e-12)


def hit_tau(v_hat: np.ndarray, hits: np.ndarray, n: int, tau: float) -> np.ndarray:
    return np.abs(np.asarray(v_hat, dtype=float) - np.asarray(hits) / n) <= tau + 1e-12


def shares(gate: np.ndarray, hit: np.ndarray) -> dict:
    """Certified coverage and false certification, as counts and shares of all pairs."""
    pairs = len(gate)
    certified = int((gate & hit).sum())
    false = int((gate & ~hit).sum())
    return {
        "pairs": pairs,
        "gate_open": int(gate.sum()),
        "certified": certified,
        "false_certified": false,
        "certified_coverage": certified / pairs if pairs else None,
        "false_certification": false / pairs if pairs else None,
    }


def report(
    strata: Sequence,
    v_hat: np.ndarray,
    gate: np.ndarray,
    hits: np.ndarray,
    n: int,
    rule: Rule,
) -> dict:
    """Both hit rules' shares per stratum, and each stratum's V_emp histogram."""
    strata = np.asarray(strata)
    v_hat = np.asarray(v_hat, dtype=float)
    gate = np.asarray(gate, dtype=bool)
    hits = np.asarray(hits)
    wil = hit_wilson(v_hat, hits, n, rule.z)
    tau = hit_tau(v_hat, hits, n, rule.tau)
    out = {}
    for stratum in sorted(set(strata.tolist())):
        at = strata == stratum
        out[str(stratum)] = {
            "wilson": shares(gate[at], wil[at]),
            "tau": shares(gate[at], tau[at]),
            "v_emp_histogram": np.bincount(hits[at], minlength=n + 1).tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def fit_constant(d_cuts: np.ndarray, hits: np.ndarray, n: int) -> dict:
    """Base rate of V_emp per ``d_cuts`` stratum, with the overall rate for unseen strata."""
    rates = defaultdict(list)
    for stratum, h in zip(
        np.asarray(d_cuts).tolist(), np.asarray(hits).tolist(), strict=True
    ):
        rates[stratum].append(h / n)
    overall = float(np.mean(np.asarray(hits) / n)) if len(hits) else 0.0
    return {
        "by_stratum": {s: float(np.mean(v)) for s, v in rates.items()},
        "overall": overall,
    }


def predict_constant(model: dict, d_cuts: np.ndarray) -> np.ndarray:
    return np.array(
        [
            model["by_stratum"].get(s, model["overall"])
            for s in np.asarray(d_cuts).tolist()
        ],
        dtype=float,
    )


def fit_distance(d_ops: np.ndarray, hits: np.ndarray, n: int, steps: int = 50) -> dict:
    """Logistic V_hat = sigmoid(a + b * d_ops) by Newton's method on the binomial likelihood."""
    x = np.asarray(d_ops, dtype=float)
    y = np.asarray(hits, dtype=float) / n
    design = np.stack([np.ones_like(x), x], axis=1)
    theta = np.zeros(2)
    for _ in range(steps):
        prob = 1 / (1 + np.exp(-(design @ theta)))
        gradient = design.T @ (y - prob)
        curvature = design.T @ (design * (prob * (1 - prob))[:, None]) + 1e-6 * np.eye(
            2
        )
        update = np.linalg.solve(curvature, gradient)
        theta += update
        if np.abs(update).max() < 1e-10:
            break
    return {"intercept": float(theta[0]), "slope": float(theta[1])}


def predict_distance(model: dict, d_ops: np.ndarray) -> np.ndarray:
    z = model["intercept"] + model["slope"] * np.asarray(d_ops, dtype=float)
    return 1 / (1 + np.exp(-z))


def fit_gate(
    d_ops: np.ndarray, v_hat: np.ndarray, hits: np.ndarray, n: int, rule: Rule
) -> list[int]:
    """The ``d_ops`` values whose archive pairs hit (Wilson rule) at the break-even precision."""
    d_ops = np.asarray(d_ops)
    hit = hit_wilson(v_hat, hits, n, rule.z)
    return sorted(
        int(d)
        for d in set(d_ops.tolist())
        if (d_ops == d).sum() >= rule.min_bin
        and hit[d_ops == d].mean() >= rule.precision
    )


def apply_gate(open_bins: Sequence[int], d_ops: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(d_ops), np.asarray(list(open_bins), dtype=int))
