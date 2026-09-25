"""novelty-reduction (#171) M3: does certified coverage on a fixed panel grow as the policy learns?

Registered on novelty-reduction (#171), issuecomment-5824696752. Everything
starts from one frozen M2 run's policy and archive (m2-s0 cycle 3), grammar
(4, 3, none), B = 16, T = 1, n = 8.

**Baseline** (``--branch baseline``, run once): draw the test panel, fit
heads, roll the panel. No policy phase, no attempts.

- **Panel**: 1,000 fresh, unseen, untruncated proposal-generator pairs whose
  input lineage the archive never met; immutable from here on. Its canonical
  endpoints and input lineages are quarantined: no label pool may use them and
  no proposal or executed walk touching them is archived, so they never reach
  training either.
- **Heads**: 2,000 fresh label pairs a cycle (M2b's population B, also kept
  off the panel's endpoints and lineages), lineage-split 1,000 P / 500 E /
  500 gate, never archived; M2b's ridge probes and controls
  (``m2b.fit_population``).
- **Gate** (as amended on #171 after the first baseline, whose E ladder
  closed at a best bound of 0.204): two ladders, one on the E partition's Ê
  deciles and one on its P-confidence u = 1 - max(P̂, 1 - P̂) deciles. The
  certification gate is the widest rung (most gate-partition pairs open)
  across both whose one-sided 90% Clopper-Pearson upper bound on the
  conditional Wilson-miss rate on the gate partition, Bonferroni over all 2
  x 10 rungs, is at most 0.20; closed if none. It is read everywhere
  certification is (attempt certification and the uncertified filter, the
  panel's coverage, false certification and conditional miss, the
  submarine monitor). The E-only and u-only ladders (Bonferroni over their
  own 10) and M2b's point rule (``probes.set_threshold``) are logged beside.

**Branches** (``adaptive``, ``control``, ``solved``), each cycle:

1. attempt phase with the previous measurement's heads and policy: 1,000
   archive pairs at n = 8, 10% uniform re-attempts, the rest by the branch's
   ranking over up to ``e_candidates`` uncertified eligible pairs (adaptive:
   descending rank-sum of Ê and count novelty; control: the same with Ê
   replaced by u). A pair is certified when it is open under the
   certification gate and a Wilson hit on its own rollouts. Walks are archived as M2 archived
   calibration walks;
2. proposals as M2 (1,000, half from archived nodes);
3. policy phase: M2's hindsight imitation (``imitate="all"``), or only paths
   a rollout wrote (``imitate="solved"``, the solved branch);
4. freeze; fresh labels; heads and gate; panel.

Hard stops as M2 (an exception; label hits nearly all zero; no new proposal
pair; panel success below half the previous measurement's) plus a wall stop
per branch. ``--compare`` reads the falsifier off saved panel rows.

Run::

    uv run python experiments/novelty_reduction/m3.py --branch baseline \\
        --run-dir <m3-baseline> --source <m2-s0>
    uv run python experiments/novelty_reduction/m3.py --branch adaptive \\
        --run-dir <m3-adaptive> --source <m2-s0> --baseline <m3-baseline>
    uv run python experiments/novelty_reduction/m3.py --compare \\
        --baseline <m3-baseline> --adaptive <dir> --control <dir> [--solved <dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.certified_coverage import score as scoring  # noqa: E402
from experiments.certified_coverage.rollout import Procedure  # noqa: E402
from experiments.novelty_reduction import (  # noqa: E402
    canon,
    heads,
    m2b,
    probes,
    programs,
)
from experiments.novelty_reduction import cycle as loop  # noqa: E402
from experiments.novelty_reduction.archive import LoopArchive  # noqa: E402
from wandering_light.rollout_policy import PolicyArchitecture  # noqa: E402
from wandering_light.stack_policy import load_policy, save_policy  # noqa: E402

BRANCHES = {
    "baseline": ("", "all"),
    "adaptive": ("adaptive", "all"),
    "control": ("control", "all"),
    "solved": ("adaptive", "solved"),
}
REATTEMPT = "reattempt"
RANKED = "ranked"


@dataclass(frozen=True, slots=True)
class Config:
    source: str = ""  # the frozen M2 run
    source_cycle: int = 3
    baseline: str = ""  # the baseline run dir (branches only)
    branch: str = "baseline"
    ranking: str = ""  # "adaptive" | "control"; "" for the baseline
    imitate: str = "all"  # "all" | "solved"
    cycles: int = 3
    seed: int = 0
    labels: tuple[int, int, int] = (1000, 500, 500)
    panel: int = 1000
    attempts: int = 1000
    reattempt: float = 0.10
    e_candidates: int = 20_000
    lambdas: tuple[float, ...] = probes.LAMBDAS
    folds: int = 5
    ladder: int = 10
    alpha: float = 0.10  # one-sided, Bonferroni over the ladder
    miss_bound: float = 0.20
    point_min_open: int = 10
    point_precision: float = 0.8
    q: float = 0.2
    one_sided: float = 0.10
    reps: int = 2000
    proposals: int = 1000
    start_archived_share: float = 0.5
    epochs: int = 0  # 0: the source run's
    budget: int = 16
    temperature: float = 1.0
    n: int = 8
    max_cuts: int = 1
    grammar_level: str = "4,3,none"
    train_device: str = "cuda"
    threads: int = 1
    wall_seconds: float = 6 * 3600.0
    collapse_ratio: float = 0.5
    uninformative_zero_share: float = 0.98
    behaviours: tuple[str, ...] | None = None


def stream(config: Config, cycle: int, role: str) -> random.Random:
    """Branch-free: every branch draws the same labels and rollout seeds per cycle."""
    return random.Random(f"m3:{config.seed}:{cycle}:{role}")


def fit_config(config: Config) -> m2b.Config:
    return m2b.Config(
        seed=config.seed,
        split=config.labels,
        lambdas=config.lambdas,
        folds=config.folds,
        q=config.q,
        one_sided=config.one_sided,
        reps=config.reps,
        budget=config.budget,
        temperature=config.temperature,
        n=config.n,
        max_cuts=config.max_cuts,
        grammar_level=config.grammar_level,
    )


# ---------------------------------------------------------------------------
# Panel quarantine and fresh draws
# ---------------------------------------------------------------------------


@dataclass
class Quarantine:
    """The panel's canonical endpoints and input lineages, kept out of everything else."""

    endpoints: set
    lineages: set

    def admit(self, states, lineage) -> bool:
        if lineage in self.lineages:
            return False
        return not any(canon.state_key(x) in self.endpoints for x in states)


def archive_lineages(archive: LoopArchive) -> set:
    return {o[0] for o in archive.origins if o is not None}


def draw_fresh(
    count: int,
    rng,
    behaviours,
    canonical: loop.Canonical,
    lineages: set,
    quarantine: Quarantine | None,
    config: Config,
) -> tuple[loop.Pool, dict]:
    """M2b's population B: unseen and untruncated, plus lineage and quarantine checks."""
    grammar = programs.Grammar.parse(config.grammar_level)
    kept, seen, drawn, rejected = [], set(), 0, Counter()
    while len(kept) < count and drawn < 100 * count:
        batch = programs.fresh_pairs(
            200, rng, behaviours, max_cuts=config.max_cuts, grammar=grammar
        )
        if not batch:
            break
        for pair in batch:
            drawn += 1
            label = canonical.classify(pair.s, pair.t)
            key = canon.pair_key(pair.s, pair.t)
            if key in seen:
                rejected["duplicate"] += 1
            elif label != "unseen":
                rejected[label] += 1
            elif canon.truncated(pair.s, pair.t):
                rejected["truncated"] += 1
            elif pair.inputs in lineages:
                rejected["archive_lineage"] += 1
            elif quarantine is not None and not quarantine.admit(
                (pair.s, pair.t), pair.inputs
            ):
                rejected["quarantine"] += 1
            else:
                seen.add(key)
                kept.append(pair)
                if len(kept) == count:
                    break
    return loop.fresh_pool(kept, "F3"), {
        "kept": len(kept),
        "drawn": drawn,
        "rejected": dict(rejected),
    }


# ---------------------------------------------------------------------------
# Heads and the gate
# ---------------------------------------------------------------------------


def binomial_cdf(x: int, m: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if x >= m else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(m + 1)
        - math.lgamma(k + 1)
        - math.lgamma(m - k + 1)
        + k * log_p
        + (m - k) * log_q
        for k in range(x + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def binomial_upper(x: int, m: int, level: float) -> float:
    """One-sided Clopper-Pearson upper bound: the largest p with P(X <= x | m, p) >= level."""
    if m == 0 or x >= m:
        return 1.0
    lo, hi = x / m, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binomial_cdf(x, m, mid) >= level:
            lo = mid
        else:
            hi = mid
    return hi


def uncertainty(p_hat: np.ndarray) -> np.ndarray:
    """u = 1 - max(P, 1 - P): 0 when P is sure either way, 0.5 at P = 0.5."""
    return 1 - np.maximum(p_hat, 1 - p_hat)


def ladder(fit_scores, gate_scores, gate_hit, config: Config, ladders: int = 1) -> dict:
    """Rungs at the fitting partition's deciles; open is score <= rung; the widest safe rung.

    Bonferroni over ``ladders`` x ``config.ladder`` rungs.
    """
    fit_scores = np.asarray(fit_scores, dtype=float)
    level = config.alpha / (config.ladder * ladders)
    rungs, chosen = [], None
    quantiles = [(i + 1) / config.ladder for i in range(config.ladder)]
    for q, rung in zip(quantiles, np.quantile(fit_scores, quantiles), strict=True):
        open_ = gate_scores <= rung
        m = int(open_.sum())
        misses = int((~gate_hit[open_]).sum())
        upper = binomial_upper(misses, m, level)
        ok = m > 0 and upper <= config.miss_bound
        rungs.append(
            {
                "quantile": q,
                "threshold": float(rung),
                "open": m,
                "misses": misses,
                "upper": upper,
                "qualifies": ok,
            }
        )
        if ok:
            chosen = float(rung)
    return {
        "threshold": chosen,
        "level_per_rung": level,
        "miss_bound": config.miss_bound,
        "rungs": rungs,
    }


def certification_gate(e_fit, u_fit, e_gate, u_gate, gate_hit, config: Config) -> dict:
    """The amended gate: the widest qualifying rung across the E and u ladders,
    Bonferroni over both (2 x ``config.ladder`` rungs); ties go to E."""
    ladders = {
        "E": ladder(e_fit, e_gate, gate_hit, config, ladders=2),
        "u": ladder(u_fit, u_gate, gate_hit, config, ladders=2),
    }
    best = None
    for name in ("E", "u"):
        for rung in ladders[name]["rungs"]:
            if rung["qualifies"] and (best is None or rung["open"] > best[1]["open"]):
                best = (name, rung)
    out = {"ladder": None, "threshold": None, "quantile": None}
    if best is not None:
        out = {
            "ladder": best[0],
            "threshold": best[1]["threshold"],
            "quantile": best[1]["quantile"],
            "open_on_gate_partition": best[1]["open"],
            "upper": best[1]["upper"],
        }
    return {**out, "ladders": ladders}


@dataclass
class HeadSet:
    """P and E probes and the thresholds a measurement set; what an attempt phase reads."""

    p: dict
    e: dict
    t_e: float | None
    t_u: float | None
    e0_point: float | None
    gate: str | None = None  # the certification gate's ladder, "E" or "u"
    t_gate: float | None = None

    def predict(self, features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_hat = probes.Probe(self.p).predict(features)
        e_hat = probes.Probe(self.e).predict(features)
        return p_hat, e_hat, uncertainty(p_hat)

    def open_e(self, e_hat) -> np.ndarray:
        if self.t_e is None:
            return np.zeros(len(e_hat), dtype=bool)
        return e_hat <= self.t_e

    def open_u(self, u) -> np.ndarray:
        if self.t_u is None:
            return np.zeros(len(u), dtype=bool)
        return u <= self.t_u

    def open_cert(self, e_hat, u) -> np.ndarray:
        """Open under the certification gate, whichever ladder won."""
        if self.gate is None:
            return np.zeros(len(e_hat), dtype=bool)
        return (e_hat if self.gate == "E" else u) <= self.t_gate

    def plain(self, width: int) -> dict:
        names = probes.trunk_names(width)
        return {
            "p": probes.plain_logistic(self.p, names),
            "e": probes.plain_logistic(self.e, names),
            "t_e": self.t_e,
            "t_u": self.t_u,
            "e0_point": self.e0_point,
            "gate": self.gate,
            "t_gate": self.t_gate,
        }

    @staticmethod
    def load(path: Path) -> HeadSet:
        plain = json.loads(path.read_text())
        return HeadSet(
            probes.logistic_from(plain["p"]),
            probes.logistic_from(plain["e"]),
            plain["t_e"],
            plain["t_u"],
            plain["e0_point"],
            plain.get("gate"),
            plain.get("t_gate"),
        )


def fit_heads(pool: loop.Pool, features, hits, config: Config, seed: int):
    """M2b's probes and controls on fresh labels, then both ladders and the point rule."""
    parts = m2b.split(pool.lineage, seed, config.labels)
    fitted = m2b.fit_population(pool, features, hits, parts, fit_config(config), seed)
    second, third = parts == 1, parts == 2
    p_second = ev_predict(fitted.p, features[second])
    e_second = ev_predict(fitted.e, features[second])
    p_third = ev_predict(fitted.p, features[third])
    e_third = ev_predict(fitted.e, features[third])
    hit_third = scoring.hit_wilson(p_third, hits[third], config.n)
    gate_e = ladder(e_second, e_third, hit_third, config)
    gate_u = ladder(uncertainty(p_second), uncertainty(p_third), hit_third, config)
    gate = certification_gate(
        e_second,
        uncertainty(p_second),
        e_third,
        uncertainty(p_third),
        hit_third,
        config,
    )
    point = probes.set_threshold(
        e_third,
        p_third,
        hits[third],
        config.n,
        config.point_min_open,
        config.point_precision,
    )
    head_set = HeadSet(
        fitted.p,
        fitted.e,
        gate_e["threshold"],
        gate_u["threshold"],
        point["e0"],
        gate["ladder"],
        gate["threshold"],
    )
    facts = {
        **fitted.facts,
        "certification_gate": gate,
        "gate_e": gate_e,
        "gate_u": gate_u,
        "point_rule": point,
    }
    return head_set, fitted, facts


def ev_predict(model: dict, features) -> np.ndarray:
    return probes.Probe(model).predict(features)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def body_stats(atoms) -> dict:
    atoms = np.asarray(list(atoms), dtype=float)
    if not len(atoms):
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": len(atoms),
        "median": float(np.median(atoms)),
        "p90": float(np.quantile(atoms, 0.9)),
        "max": int(atoms.max()),
    }


def strata_counts(paths) -> dict:
    return dict(
        sorted(
            Counter(
                programs.body_stratum(programs.body_atoms(p)) for p in paths
            ).items()
        )
    )


def gate_metrics(open_, hit) -> dict:
    total = len(hit)
    m = int(open_.sum())
    return {
        "open": m,
        "certified_coverage": float((open_ & hit).sum() / total) if total else None,
        "false_certification": float((open_ & ~hit).sum() / total) if total else None,
        "conditional_miss": float((~hit[open_]).mean()) if m else None,
    }


def measure_panel(
    head_set: HeadSet,
    fitted,
    scorer,
    panel: loop.Pool,
    procedure,
    config: Config,
    seed: int,
    deadline: float,
):
    """Roll the panel, gate it both ways, score the probes against M2b's controls."""
    hits, _, ends, walks, done = loop.roll(scorer, panel, procedure, seed, deadline)
    if done < len(panel):
        return None, None, None
    features = heads.pair_features(scorer, panel.jobs())
    p_hat, e_hat, u = head_set.predict(features)
    hit = scoring.hit_wilson(p_hat, hits, config.n)
    open_e, open_u = head_set.open_e(e_hat), head_set.open_u(u)
    open_cert = head_set.open_cert(e_hat, u)
    open_point = (
        e_hat < head_set.e0_point
        if head_set.e0_point is not None
        else np.zeros(len(panel), dtype=bool)
    )
    strata = {
        "all": ["all"] * len(panel),
        "d_ops": [str(d) for d in panel.d_ops],
        "body": [programs.body_stratum(programs.body_atoms(p)) for p in panel.path],
    }
    scores = m2b.score(
        m2b.arms_on(fitted, panel, features),
        hits,
        config.n,
        panel.lineage,
        strata,
        fit_config(config),
        seed,
    )
    solved = [programs.body_atoms(w.ops) for w in walks if w.hit]
    report = {
        **loop.success(hits, config.n),
        "ends": ends,
        "gates": {
            "certification": gate_metrics(open_cert, hit),
            "E": gate_metrics(open_e, hit),
            "P_confidence": gate_metrics(open_u, hit),
            "E_point_rule": gate_metrics(open_point, hit),
        },
        "thresholds": {
            "gate": head_set.gate,
            "t_gate": head_set.t_gate,
            "t_e": head_set.t_e,
            "t_u": head_set.t_u,
            "e0_point": head_set.e0_point,
        },
        "by_body": loop.body_record(panel, hits, config.n),
        "solved_bodies": body_stats(solved),
        "log_loss_gain": scores["log_loss_gain"],
        "matched_coverage_gain": scores["matched_coverage_gain"],
        "log_loss": {
            arm: v["log_loss"]
            for arm, v in scores["arms"].items()
            if arm in ("ridge_full", "distance", "cheap", "constant")
        },
    }
    rows = [
        {
            "pair": i,
            "lineage": str(panel.lineage[i]),
            "d_ops": int(panel.d_ops[i]),
            "body": strata["body"][i],
            "hits": int(hits[i]),
            "p_hat": float(p_hat[i]),
            "e_hat": float(e_hat[i]),
            "u": float(u[i]),
            "wilson_hit": bool(hit[i]),
            "open_cert": bool(open_cert[i]),
            "open_e": bool(open_e[i]),
            "open_u": bool(open_u[i]),
            "open_point": bool(open_point[i]),
            "certified": bool(open_cert[i] and hit[i]),
            "certified_e_only": bool(open_e[i] and hit[i]),
            "certified_u_only": bool(open_u[i] and hit[i]),
        }
        for i in range(len(panel))
    ]
    return report, rows, open_cert & hit


def attempt_phase(
    archive: LoopArchive,
    head_set: HeadSet,
    scorer,
    procedure,
    config: Config,
    cycle: int,
    certified: set,
    quarantine: Quarantine,
    seed: int,
    deadline: float,
):
    """1,000 archive pairs at n: re-attempts, then the branch's rank-sum order."""
    rng = stream(config, cycle, "attempt")
    eligible = sorted(
        k for k, r in archive.pairs.items() if r.witness_d_cuts <= config.max_cuts
    )
    attempted = [k for k in eligible if archive.pairs[k].attempts > 0]
    count = min(round(config.attempts * config.reattempt), len(attempted))
    again = rng.sample(attempted, count)
    taken = set(again)
    open_pool = [k for k in eligible if k not in certified and k not in taken]
    sample = sorted(rng.sample(open_pool, min(len(open_pool), config.e_candidates)))
    features = heads.pair_features(
        scorer, [(archive.states[s], archive.states[t]) for s, t in sample]
    )
    _p_hat, e_hat, u = head_set.predict(features)
    novelty = probes.count_novelty(
        archive,
        [archive.states[s] for s, _ in sample],
        [archive.states[t] for _, t in sample],
    )
    signal = e_hat if config.ranking == "adaptive" else u
    rank_sum = m2b.ranks(signal) + m2b.ranks(novelty) if sample else np.zeros(0)
    order = sorted(range(len(sample)), key=lambda i: (-rank_sum[i], sample[i]))
    ranked = [sample[i] for i in order[: config.attempts - len(again)]]
    at = {k: i for i, k in enumerate(sample)}
    keys = again + ranked
    rules = [REATTEMPT] * len(again) + [RANKED] * len(ranked)
    pool = loop.archive_pool(archive, keys, "attempt")
    hits, _, ends, walks, done = loop.roll(scorer, pool, procedure, seed, deadline)
    if done < len(pool):
        return None, None
    pool_features = heads.pair_features(scorer, pool.jobs())
    p_pool, e_pool, u_pool = head_set.predict(pool_features)
    hit = scoring.hit_wilson(p_pool, hits, config.n)
    open_cert = head_set.open_cert(e_pool, u_pool)
    now = open_cert & hit
    before = len(certified)
    certified |= {k for k, c in zip(keys, now, strict=True) if c}
    made, _touched = loop.archive_walks(
        archive, keys, walks, config.source_cycle + cycle, admit=quarantine.admit
    )
    solved = [programs.body_atoms(w.ops) for w in walks if w.hit]
    rows = []
    for i, (k, rule) in enumerate(zip(keys, rules, strict=True)):
        j = at.get(k)
        rows.append(
            {
                "s": k[0],
                "t": k[1],
                "rule": rule,
                "probability": count / len(attempted)
                if rule == REATTEMPT
                else len(sample) / max(len(open_pool), 1),
                "rank_sum": float(rank_sum[j]) if j is not None else None,
                "count_novelty": float(novelty[j]) if j is not None else None,
                "e_hat": float(e_pool[i]),
                "u": float(u_pool[i]),
                "p_hat": float(p_pool[i]),
                "hits": int(hits[i]),
                "wilson_hit": bool(hit[i]),
                "open_cert": bool(open_cert[i]),
                "certified": bool(now[i]),
                "origin": archive.pairs[k].origin,
                "body": programs.body_stratum(
                    programs.body_atoms(archive.pairs[k].witness_path)
                ),
            }
        )
    report = {
        "pairs": len(keys),
        "rules": dict(Counter(rules)),
        "ranking": config.ranking,
        "eligible": len(eligible),
        "uncertified_scored": len(sample),
        "uncertified_available": len(open_pool),
        **loop.success(hits, config.n),
        "ends": ends,
        "archived": made,
        "gate": {"ladder": head_set.gate, "threshold": head_set.t_gate},
        "certified_this_phase": int(now.sum()),
        "certified_new": len(certified) - before,
        "certified_total": len(certified),
        "attempted_by_body": strata_counts(pool.path),
        "solved_by_body": strata_counts(
            [p for p, h in zip(pool.path, hits, strict=True) if h > 0]
        ),
        "solved_bodies": body_stats(solved),
    }
    return report, rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def train_policy(model, optimizer, rows, loop_config, epochs: int, cycle: int) -> dict:
    """M2's recipe; the shuffle is seeded by the cycle alone, so every branch
    trains identical rows in the identical order."""
    return loop.train_epochs(
        model, optimizer, rows, loop_config, epochs or loop_config.epochs, f"m3:{cycle}"
    )


def write_rows(path: Path, rows) -> None:
    with gzip.open(path, "wt") as handle:
        handle.writelines(json.dumps(row, default=str) + "\n" for row in rows)


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def run(config: Config, run_dir: Path, log=print) -> list[dict]:
    if config.branch not in BRANCHES:
        raise ValueError(f"branch {config.branch!r}")
    if config.branch != "baseline" and config.ranking not in ("adaptive", "control"):
        raise ValueError(f"ranking {config.ranking!r}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    start = time.perf_counter()
    deadline = start + config.wall_seconds
    source = Path(config.source) / f"cycle-{config.source_cycle}"
    source_config = json.loads(
        (Path(config.source) / "cycle-0" / "report.json").read_text()
    )["config"]
    loop_config = loop.Config(
        **{
            **source_config,
            "architecture": tuple(source_config["architecture"]),
            "behaviours": tuple(source_config["behaviours"])
            if source_config.get("behaviours")
            else None,
            **{
                k: tuple(source_config[k])
                for k in ("ridge_lambdas", "widen_levels")
                if k in source_config
            },
            "train_device": config.train_device,
            "grammar_level": config.grammar_level,
            "max_cuts": config.max_cuts,
        }
    )
    architecture = PolicyArchitecture(*loop_config.architecture)
    behaviours = list(
        config.behaviours or loop_config.behaviours or programs.train_behaviours()
    )
    archive = LoopArchive.load(source / "archive.pkl.gz")
    model = load_policy(source / "policy")
    scorer = loop.frozen_scorer(model, architecture)
    procedure = Procedure(
        f"m3 {config.branch} s{config.seed}",
        config.budget,
        config.temperature,
        config.n,
    )
    reports: list[dict] = []

    def stop(report: dict, name: str, why: str) -> list[dict]:
        report["stop"] = why
        report["finished"] = loop.now_pdt()
        (run_dir / name).mkdir(exist_ok=True)
        (run_dir / name / "report.json").write_text(
            json.dumps(report, indent=1, default=str)
        )
        reports.append(report)
        log(f"{name} stop: {why}")
        return reports

    def labels_and_heads(report: dict, cycle: int, quarantine: Quarantine):
        canonical = loop.Canonical()
        canonical.update(archive)
        pool, report["label_draws"] = draw_fresh(
            sum(config.labels),
            stream(config, cycle, "labels"),
            behaviours,
            canonical,
            archive_lineages(archive),
            quarantine,
            config,
        )
        hits, _, ends, _, done = loop.roll(
            scorer, pool, procedure, config.seed * 1000 + 100 * cycle + 1, deadline
        )
        if done < len(pool):
            return None
        report["labels"] = {**loop.success(hits, config.n), "ends": ends}
        if len(hits) and (hits == 0).mean() >= config.uninformative_zero_share:
            report["uninformative"] = True
        features = heads.pair_features(scorer, pool.jobs())
        head_set, fitted, report["heads"] = fit_heads(
            pool, features, hits, config, config.seed + cycle
        )
        return head_set, fitted, features.shape[1]

    def save_heads(name: str, head_set: HeadSet, fitted, width: int) -> None:
        (run_dir / name / "heads.json").write_text(
            json.dumps(head_set.plain(width), indent=1)
        )
        (run_dir / name / "models.json").write_text(
            json.dumps(m2b.plain_models(fitted, width), indent=1, default=str)
        )

    # ---- baseline: the panel, heads on the frozen source policy
    if config.branch == "baseline":
        report: dict = {
            "cycle": 0,
            "config": dataclasses.asdict(config),
            "started": loop.now_pdt(),
            "procedure": procedure.row()["rollout"],
            "source_archive": str((source / "archive.pkl.gz").resolve()),
            "source_policy": str((source / "policy").resolve()),
        }
        canonical = loop.Canonical()
        canonical.update(archive)
        panel, report["panel_draws"] = draw_fresh(
            config.panel,
            stream(config, 0, "panel"),
            behaviours,
            canonical,
            archive_lineages(archive),
            None,
            config,
        )
        quarantine = Quarantine(
            {canon.state_key(x) for x in (*panel.s, *panel.t)}, set(panel.lineage)
        )
        (run_dir / "cycle-0").mkdir(exist_ok=True)
        with gzip.open(run_dir / "panel.pkl.gz", "wb") as handle:
            pickle.dump({"panel": panel, "quarantine": quarantine}, handle)
        fitted_all = labels_and_heads(report, 0, quarantine)
        if fitted_all is None:
            return stop(report, "cycle-0", "wall stop during labels")
        head_set, fitted, width = fitted_all
        if report.get("uninformative"):
            # before the panel and heads.json exist, so no branch can start
            return stop(report, "cycle-0", "uninformative: label hits almost all zero")
        panel_report, rows, _ = measure_panel(
            head_set,
            fitted,
            scorer,
            panel,
            procedure,
            config,
            config.seed * 1000 + 2,
            deadline,
        )
        if panel_report is None:
            return stop(report, "cycle-0", "wall stop during the panel")
        report["panel"] = panel_report
        report["finished"] = loop.now_pdt()
        save_heads("cycle-0", head_set, fitted, width)
        write_rows(run_dir / "cycle-0" / "panel_rows.jsonl.gz", rows)
        (run_dir / "cycle-0" / "report.json").write_text(
            json.dumps(report, indent=1, default=str)
        )
        log(
            f"baseline: panel success {panel_report['walker_success']:.3f}, "
            f"certified {panel_report['gates']['certification']['certified_coverage']}"
            f" via {head_set.gate}"
        )
        return [report]

    # ---- branches
    base = Path(config.baseline)
    base_report = json.loads((base / "cycle-0" / "report.json").read_text())
    if base_report.get("stop"):
        raise ValueError(f"baseline stopped: {base_report['stop']}")
    with gzip.open(base / "panel.pkl.gz", "rb") as handle:
        saved = pickle.load(handle)
    panel, quarantine = saved["panel"], saved["quarantine"]
    head_set = HeadSet.load(base / "cycle-0" / "heads.json")
    previous_success = base_report["panel"]["walker_success"]
    previous_certified = np.array(
        [r["certified"] for r in read_rows(base / "cycle-0" / "panel_rows.jsonl.gz")]
    )
    optimizer = torch.optim.Adam(
        model.to(config.train_device).parameters(), lr=loop_config.learning_rate
    )
    certified: set = set()
    previous_paths: dict = {}
    for cycle in range(1, config.cycles + 1):
        name = f"cycle-{cycle}"
        clock = loop.Clock()
        report = {
            "cycle": cycle,
            "config": dataclasses.asdict(config),
            "started": loop.now_pdt(),
            "procedure": procedure.row()["rollout"],
        }
        rollouts = 0

        # 1. attempt phase: the previous measurement's heads and policy
        attempt, attempt_rows = attempt_phase(
            archive,
            head_set,
            scorer,
            procedure,
            config,
            cycle,
            certified,
            quarantine,
            config.seed * 1000 + 100 * cycle + 3,
            deadline,
        )
        if attempt is None:
            return stop(report, name, "wall stop during the attempt phase")
        report["attempt"] = attempt
        rollouts += attempt["pairs"] * config.n
        clock.mark("attempt")

        # 2. proposals
        report["propose"], proposal_log = loop.propose_phase(
            archive,
            loop_config,
            config.source_cycle + cycle,
            behaviours,
            config.proposals,
            archived_share=config.start_archived_share,
            level=config.grammar_level,
            admit=quarantine.admit,
        )
        report["propose"]["by_body"] = dict(
            sorted(
                Counter(
                    programs.body_stratum(p["body_atoms"])
                    for p in proposal_log
                    if not p.get("quarantined") and p["cuts"]
                ).items()
            )
        )
        clock.mark("propose")
        if report["propose"].get("pairs", 0) == 0:
            return stop(report, name, "no new proposal pairs")

        # 3. policy phase
        rows, report["rows"], previous_paths = loop.policy_rows(
            archive,
            loop_config,
            config.source_cycle + cycle,
            previous_paths,
            imitate=config.imitate,
        )
        report["rows"]["trained_by_body"] = strata_counts(previous_paths.values())
        model.to(config.train_device)
        report["policy"] = {
            **train_policy(model, optimizer, rows, loop_config, config.epochs, cycle),
            "branch": config.branch,
        }
        scorer = loop.frozen_scorer(model, architecture)
        clock.mark("policy")

        # 4. freeze; labels, heads and gate; the panel
        fitted_all = labels_and_heads(report, cycle, quarantine)
        if fitted_all is None:
            return stop(report, name, "wall stop during labels")
        head_set, fitted, width = fitted_all
        rollouts += sum(config.labels) * config.n
        clock.mark("labels_and_heads")
        panel_report, panel_rows, now = measure_panel(
            head_set,
            fitted,
            scorer,
            panel,
            procedure,
            config,
            config.seed * 1000 + 100 * cycle + 2,
            deadline,
        )
        if panel_report is None:
            return stop(report, name, "wall stop during the panel")
        rollouts += len(panel) * config.n
        report["panel"] = panel_report
        newly = int((now & ~previous_certified).sum())
        report["executions"] = {
            "rollouts_this_cycle": rollouts,
            "attempt_rollouts": attempt["pairs"] * config.n,
            "newly_certified_panel_pairs": newly,
            "rollouts_per_newly_certified": rollouts / newly if newly else None,
            "attempt_rollouts_per_newly_certified": attempt["pairs"] * config.n / newly
            if newly
            else None,
        }
        report["archive"] = archive.summary()
        report["timings"] = clock.phases
        report["finished"] = loop.now_pdt()
        (run_dir / name).mkdir(exist_ok=True)
        archive.save(run_dir / name / "archive.pkl.gz")
        save_policy(
            run_dir / name / "policy",
            scorer.model,
            trainer_state={"cycle": cycle, **report["policy"]},
            manifest={
                "issue": "novelty-reduction (#171) M3",
                "config": dataclasses.asdict(config),
            },
            wandb_url=None,
        )
        save_heads(name, head_set, fitted, width)
        write_rows(run_dir / name / "panel_rows.jsonl.gz", panel_rows)
        write_rows(run_dir / name / "attempt_rows.jsonl.gz", attempt_rows)
        write_rows(run_dir / name / "proposals.jsonl.gz", proposal_log)
        why = None
        if report.get("uninformative"):
            why = "uninformative: label hits almost all zero"
        elif (
            previous_success
            and panel_report["walker_success"]
            < config.collapse_ratio * previous_success
        ):
            why = "panel success collapsed"
        if why:
            return stop(report, name, why)
        (run_dir / name / "report.json").write_text(
            json.dumps(report, indent=1, default=str)
        )
        reports.append(report)
        log(
            f"{name}: attempt success {attempt['walker_success']:.3f}, panel success "
            f"{panel_report['walker_success']:.3f}, certified "
            f"{panel_report['gates']['certification']['certified_coverage']} via "
            f"{head_set.gate}, new {newly}, "
            f"timings {clock.phases}"
        )
        previous_success = panel_report["walker_success"]
        previous_certified = now
    return reports


# ---------------------------------------------------------------------------
# The falsifier, read off saved panel rows
# ---------------------------------------------------------------------------


def final_rows(run_dir: Path) -> tuple[int, list[dict]]:
    cycles = sorted(
        int(p.name.split("-")[1])
        for p in run_dir.glob("cycle-*")
        if (p / "panel_rows.jsonl.gz").exists()
    )
    return cycles[-1], read_rows(
        run_dir / f"cycle-{cycles[-1]}" / "panel_rows.jsonl.gz"
    )


def cycle_record(baseline: Path, run_dir: Path) -> list[dict]:
    """Per cycle: the panel line, the gate record and the submarine monitor.

    ``newly_cumulative`` counts panel pairs certified this cycle and never at
    baseline or in an earlier cycle (the executions line reads it);
    ``newly_vs_previous`` is the runner's logged per-cycle-previous count.
    """
    ever = np.array(
        [
            r["certified"]
            for r in read_rows(baseline / "cycle-0" / "panel_rows.jsonl.gz")
        ]
    )
    out = []
    cycles = sorted(
        int(p.name.split("-")[1])
        for p in run_dir.glob("cycle-*")
        if (p / "panel_rows.jsonl.gz").exists()
    )
    for cycle in cycles:
        folder = run_dir / f"cycle-{cycle}"
        report = json.loads((folder / "report.json").read_text())
        now = np.array(
            [r["certified"] for r in read_rows(folder / "panel_rows.jsonl.gz")]
        )
        cumulative = int((now & ~ever).sum())
        ever |= now
        gate = report["heads"]["certification_gate"]
        rollouts = report["executions"]["rollouts_this_cycle"]
        panel = report["panel"]
        out.append(
            {
                "cycle": cycle,
                "walker_success": panel["walker_success"],
                **{
                    k: panel["gates"]["certification"][k]
                    for k in (
                        "open",
                        "certified_coverage",
                        "false_certification",
                        "conditional_miss",
                    )
                },
                "gate": {
                    "ladder": gate["ladder"],
                    "quantile": gate["quantile"],
                    "open_on_gate_partition": gate.get("open_on_gate_partition"),
                    "best_bound": {
                        name: min(r["upper"] for r in ladder["rungs"])
                        for name, ladder in gate["ladders"].items()
                    },
                },
                "newly_cumulative": cumulative,
                "newly_vs_previous": report["executions"][
                    "newly_certified_panel_pairs"
                ],
                "ever_certified": int(ever.sum()),
                "rollouts": rollouts,
                "rollouts_per_newly_cumulative": rollouts / cumulative
                if cumulative
                else None,
            }
        )
    return out


def compare(
    baseline: Path,
    adaptive: Path,
    control: Path,
    solved: Path | None = None,
    reps: int = 2000,
    seed: int = 0,
    n: int = 8,
) -> dict:
    """Paired one-sided 90% bounds on the panel, per the M3 registration."""
    base = read_rows(baseline / "cycle-0" / "panel_rows.jsonl.gz")
    lineage = [r["lineage"] for r in base]

    def paired(a, b, field):
        fa = np.array([float(r[field]) for r in a])
        fb = np.array([float(r[field]) for r in b])
        return m2b.clustered(fa - fb, lineage, reps, seed, 0.10)

    def success(rows):
        return [{"s": r["hits"] / n} for r in rows]

    cycle_a, final_a = final_rows(adaptive)
    cycle_c, final_c = final_rows(control)
    miss = sum(r["open_cert"] and not r["wilson_hit"] for r in final_a)
    opened = sum(r["open_cert"] for r in final_a)
    out = {
        "cycles": {"adaptive": cycle_a, "control": cycle_c},
        "adaptive_minus_baseline_certified": paired(final_a, base, "certified"),
        "adaptive_minus_control_certified": paired(final_a, final_c, "certified"),
        "adaptive_minus_baseline_success": paired(success(final_a), success(base), "s"),
        "control_minus_baseline_certified": paired(final_c, base, "certified"),
        "adaptive_final_conditional_miss": {
            "open": opened,
            "misses": miss,
            "upper_90": binomial_upper(miss, opened, 0.10),
        },
    }
    met = (
        out["adaptive_minus_baseline_certified"]["one_sided_lower"] > 0
        and out["adaptive_minus_control_certified"]["one_sided_lower"] > 0
        and out["adaptive_minus_baseline_success"]["one_sided_lower"] > 0
        and out["adaptive_final_conditional_miss"]["upper_90"] <= 0.20
    )
    out["met"] = met
    out["by_cycle"] = {
        name: cycle_record(baseline, path)
        for name, path in (
            ("adaptive", adaptive),
            ("control", control),
            ("solved", solved),
        )
        if path is not None
    }
    if solved is not None:
        cycle_s, final_s = final_rows(solved)
        report_s = json.loads((solved / f"cycle-{cycle_s}" / "report.json").read_text())
        report_a = json.loads(
            (adaptive / f"cycle-{cycle_a}" / "report.json").read_text()
        )
        out["solved"] = {
            "cycle": cycle_s,
            "solved_body_p90": report_s["attempt"]["solved_bodies"]["p90"],
            "adaptive_solved_body_p90": report_a["attempt"]["solved_bodies"]["p90"],
            "solved_minus_adaptive_success": paired(
                success(final_s), success(final_a), "s"
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--branch", choices=sorted(BRANCHES), default="baseline")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wall-seconds", type=float, default=6 * 3600.0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--adaptive", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--solved", type=Path)
    args = parser.parse_args(argv)
    if args.compare:
        print(
            json.dumps(
                compare(Path(args.baseline), args.adaptive, args.control, args.solved),
                indent=1,
            )
        )
        return 0
    ranking, imitate = BRANCHES[args.branch]
    config = Config(
        source=args.source,
        baseline=args.baseline,
        branch=args.branch,
        ranking=ranking,
        imitate=imitate,
        cycles=args.cycles,
        seed=args.seed,
        train_device=args.train_device,
        threads=args.threads,
        wall_seconds=args.wall_seconds,
    )
    run(config, args.run_dir, log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
