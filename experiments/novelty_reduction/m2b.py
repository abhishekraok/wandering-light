"""novelty-reduction (#171) M2b: regularised frozen-feature P and E on a frozen policy.

Registered in the M2b comment on novelty-reduction (#171). The M2 heads
overfit 167 pairs; this asks whether ridge probes on the same frozen trunk
beat the predictive controls, fitted on two populations:

- **A, archive**: 2,000 pairs from the frozen run's final archive by the M2
  calibration sampler (re-attempts and novel counts logged), by origin;
- **B, fresh**: 2,000 new unseen, untruncated F3 pairs.

Each population is split by lineage into 1,000 P / 500 E / 500 gate pairs.
P and E are standardised ridge logistic probes on the pooled trunk features,
lambda by inner lineage CV inside their own partition (mean BCE + lambda
||w||^2 / 2, intercept free). E's target is the Wilson miss of the fitted P.
Every predictive control (constant, distance on witness d_ops of every
origin, cheap) is fit on the same P partition; their d_ops-bin gates are fit
on the gate partition, which is also every predictor's held-out score.

Test: 1,000 unseen, untruncated F3 pairs disjoint from both populations,
rollouts shared by every arm. RND (from the replay) and count novelty are
logged per pair, not gating; risk-coverage curves rank the ridge P's misses
by E, RND, count and E+RND.

Falsifier, per population, both on paired clustered one-sided 90% bounds on
the test pool: the ridge P's log loss is lower than every predictive
control's, and its Wilson-miss share at matched coverage q = 0.20 (accepted
x miss / q per pair, random ties) is lower than distance's.

Run::

    uv run python experiments/novelty_reduction/m2b.py --run-dir <dir> \\
        --source <m2-s0 run dir> --rnd-dump <replay>/cycle-3/dump.pkl.gz
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
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
from experiments.novelty_reduction import canon, heads, programs  # noqa: E402
from experiments.novelty_reduction import cycle as loop  # noqa: E402
from experiments.novelty_reduction import evaluate as ev  # noqa: E402
from experiments.novelty_reduction import select as selector  # noqa: E402
from experiments.novelty_reduction.archive import LoopArchive  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402

E0S = (0.1, 0.2, 0.3)
COVERAGES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
CONTROLS = ("distance", "cheap", "constant")


@dataclass(frozen=True, slots=True)
class Config:
    source: str = ""  # the frozen run's directory
    cycle: int = 3  # its policy and archive
    rnd_dump: str = ""  # replay dump carrying that cycle's RND
    seed: int = 0
    population: int = 2000
    split: tuple[int, int, int] = (1000, 500, 500)
    test: int = 1000
    lambdas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    folds: int = 5
    q: float = 0.2
    one_sided: float = 0.10  # the lower bound's quantile
    reps: int = 2000
    budget: int = 16
    temperature: float = 1.0
    n: int = 8
    max_cuts: int = 1
    grammar_level: str = "4,3,none"
    hard_stop_seconds: float = 7200.0
    threads: int = 1
    behaviours: tuple[str, ...] | None = None


def stream(config: Config, role: str) -> random.Random:
    return random.Random(f"m2b:{config.seed}:{role}")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def split(groups, seed: int, targets) -> np.ndarray:
    """A part per item, whole groups at a time, each to the part furthest below its target share."""
    order = list(dict.fromkeys(groups))
    np.random.default_rng(seed).shuffle(order)
    sizes = Counter(groups)
    load = [0] * len(targets)
    assign = {}
    for g in order:
        part = int(np.argmin([load[i] / targets[i] for i in range(len(targets))]))
        assign[g] = part
        load[part] += sizes[g]
    return np.array([assign[g] for g in groups], dtype=int)


def folds_of(groups, k: int, seed: int) -> np.ndarray:
    order = list(dict.fromkeys(groups))
    np.random.default_rng(seed).shuffle(order)
    fold = {g: i % k for i, g in enumerate(order)}
    return np.array([fold[g] for g in groups])


def ridge(x, hits, n: int, groups, config: Config, seed: int) -> dict:
    """Standardised ridge logistic, lambda by lineage CV inside ``x`` (mean-BCE scale)."""
    x, hits = np.asarray(x, dtype=float), np.asarray(hits)
    folds = folds_of(groups, config.folds, seed)
    table = {}
    for lam in config.lambdas:
        losses = []
        for f in range(config.folds):
            train, held = folds != f, folds == f
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


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


def draw_unseen(
    count: int, rng, behaviours, canonical, exclude: set, config: Config
) -> tuple[loop.Pool, dict]:
    """Fresh F3 pairs, unseen against the archive and untruncated, not in ``exclude``."""
    grammar = programs.Grammar.parse(config.grammar_level)
    kept, drawn, rejected = [], 0, Counter()
    while len(kept) < count and drawn < 100 * count:
        for pair in programs.fresh_pairs(
            200, rng, behaviours, max_cuts=config.max_cuts, grammar=grammar
        ):
            drawn += 1
            key = canon.pair_key(pair.s, pair.t)
            label = canonical.classify(pair.s, pair.t)
            if key in exclude:
                rejected["duplicate"] += 1
            elif label != "unseen":
                rejected[label] += 1
            elif canon.truncated(pair.s, pair.t):
                rejected["truncated"] += 1
            else:
                exclude.add(key)
                kept.append(pair)
                if len(kept) == count:
                    break
    return loop.fresh_pool(kept, "F3"), {
        "kept": len(kept),
        "drawn": drawn,
        "rejected": dict(rejected),
    }


def witness_ops(pool: loop.Pool) -> np.ndarray:
    return np.array([len(p) for p in pool.path])


def cheap_matrix(pool: loop.Pool) -> np.ndarray:
    return np.array(
        [
            canon.cheap_features(s, t, p)
            for s, t, p in zip(pool.s, pool.t, pool.path, strict=True)
        ]
    ).reshape(len(pool), -1)


def count_novelty(archive: LoopArchive, pool: loop.Pool) -> np.ndarray:
    """max over s, t of 1 / sqrt(1 + v); a state the archive never met has v = 0."""

    def one(state):
        node = archive.index.get(state)
        return 1.0 if node is None else archive.novelty(node)

    return np.array([max(one(s), one(t)) for s, t in zip(pool.s, pool.t, strict=True)])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def clustered(values, clusters, reps: int, seed: int, low: float) -> dict:
    """Mean with a two-sided 90% interval and a one-sided lower bound, resampling clusters."""
    values = np.asarray(values, dtype=float)
    labels = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    ids = np.array([labels[c] for c in clusters])
    sums = np.bincount(ids, weights=values, minlength=len(labels))
    counts = np.bincount(ids, minlength=len(labels))
    draws = np.random.default_rng(seed).integers(0, len(labels), (reps, len(labels)))
    means = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.05)),
        "high": float(np.quantile(means, 0.95)),
        "one_sided_lower": float(np.quantile(means, low)),
        "clusters": len(labels),
    }


def accepted(confidence, q: float, seed: int) -> np.ndarray:
    """The top ``q`` share by confidence, ties broken by a label-independent random draw."""
    confidence = np.asarray(confidence, dtype=float)
    ties = np.random.default_rng(seed).random(len(confidence))
    order = np.lexsort((ties, -confidence))
    out = np.zeros(len(confidence), dtype=bool)
    out[order[: round(q * len(confidence))]] = True
    return out


def risk_coverage(miss, confidence, seed: int) -> dict:
    return {
        str(q): float(miss[accepted(confidence, q, seed)].mean())
        for q in COVERAGES
        if round(q * len(miss))
    }


def ranks(values) -> np.ndarray:
    """Average ranks scaled to [0, 1]: equal values share a rank, so a constant signal adds nothing."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    out = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and ordered[end + 1] == ordered[start]:
            end += 1
        out[order[start : end + 1]] = (start + end) / 2
        start = end + 1
    return out / max(len(values) - 1, 1)


@dataclass
class Fitted:
    p: dict
    e: dict
    constant: float
    distance: dict
    cheap: dict
    cheap_e: dict  # E from cheap features against distance's misses (secondary)
    wilson_constant: float
    bins: dict
    bin_rates: dict
    facts: dict


def fit_population(pool, features, hits, parts, config: Config, seed: int) -> Fitted:
    n = config.n
    first, second, third = parts == 0, parts == 1, parts == 2
    groups = [str(g) for g in pool.lineage]
    at = lambda mask: [g for g, m in zip(groups, mask, strict=True) if m]  # noqa: E731
    p = ridge(features[first], hits[first], n, at(first), config, seed)
    p_second = ev.predict_logistic(p, features[second])
    miss = ~scoring.hit_wilson(p_second, hits[second], n)
    e = ridge(features[second], miss.astype(int), 1, at(second), config, seed + 1)
    ops = witness_ops(pool)
    cheap_x = cheap_matrix(pool)
    distance = scoring.fit_distance(ops[first], hits[first], n)
    cheap = ev.fit_logistic(cheap_x[first], hits[first], n)
    d_miss = ~scoring.hit_wilson(
        scoring.predict_distance(distance, ops[second]), hits[second], n
    )
    cheap_e = ridge(
        cheap_x[second], d_miss.astype(int), 1, at(second), config, seed + 2
    )
    constant = float((hits[first] / n).mean())
    predictions = {
        "constant": np.full(len(pool), constant),
        "distance": scoring.predict_distance(distance, ops),
        "cheap": ev.predict_logistic(cheap, cheap_x),
    }
    rule = scoring.Rule()
    bins, rates = {}, {}
    for arm, p_hat in predictions.items():
        bins[arm] = scoring.fit_gate(ops[third], p_hat[third], hits[third], n, rule)
        hit = scoring.hit_wilson(p_hat[third], hits[third], n)
        rates[arm] = {
            int(d): float(hit[ops[third] == d].mean()) for d in set(ops[third].tolist())
        }
    facts = {
        "pairs_by_part": np.bincount(parts, minlength=3).tolist(),
        "lineages_by_part": [len(set(at(parts == i))) for i in range(3)],
        "p_lambda": p["lambda"],
        "p_cv": p["cv"],
        "e_lambda": e["lambda"],
        "e_cv": e["cv"],
        "e_target_rate": float(miss.mean()),
        "cheap_e_lambda": cheap_e["lambda"],
        "cheap_e_target_rate": float(d_miss.mean()),
        "gate_bins": bins,
        "bin_rates": {
            arm: {str(k): v for k, v in r.items()} for arm, r in rates.items()
        },
    }
    return Fitted(
        p,
        e,
        constant,
        distance,
        cheap,
        cheap_e,
        ev.wilson_constant(hits[first], n),
        bins,
        rates,
        facts,
    )


CHEAP_FEATURES = (
    "d_ops",
    "k",
    "atoms_s",
    "atoms_t",
    "atoms_delta",
    "depth_s",
    "depth_t",
    *(f"top_s_{kind}" for kind in canon.TYPES),
    *(f"top_t_{kind}" for kind in canon.TYPES),
    *(f"last_{name}" for name in canon.COMBINATORS),
)


def plain_models(fitted: Fitted, trunk_width: int) -> dict:
    """The fitted models as plain JSON: scaler, weights (intercept first), lambda, feature names."""
    trunk = [f"pooled_mean_{i}" for i in range(trunk_width // 2)] + [
        f"pooled_max_{i}" for i in range(trunk_width - trunk_width // 2)
    ]

    def logistic(model: dict, names) -> dict:
        return {
            "features": list(names),
            "mean": np.asarray(model["mean"]).tolist(),
            "scale": np.asarray(model["scale"]).tolist(),
            "theta": np.asarray(model["theta"]).tolist(),
            "lambda": model.get("lambda"),
            "cv": model.get("cv"),
        }

    return {
        "p": logistic(fitted.p, trunk),
        "e": logistic(fitted.e, trunk),
        "cheap": logistic(fitted.cheap, CHEAP_FEATURES),
        "cheap_e": logistic(fitted.cheap_e, CHEAP_FEATURES),
        "distance": fitted.distance,
        "constant": fitted.constant,
        "wilson_constant": fitted.wilson_constant,
        "bins": fitted.bins,
        "bin_rates": {
            a: {str(d): r for d, r in v.items()} for a, v in fitted.bin_rates.items()
        },
    }


def logistic_from(plain: dict) -> dict:
    """A plain logistic model back into ``evaluate.predict_logistic``'s form."""
    return {k: np.asarray(plain[k]) for k in ("mean", "scale", "theta")}


def arms_on(fitted: Fitted, pool, features) -> dict:
    """Each arm's P_hat, its gates and its matched-coverage confidence on ``pool``."""
    ops = witness_ops(pool)
    cheap_x = cheap_matrix(pool)
    p_ridge = ev.predict_logistic(fitted.p, features)
    e_ridge = ev.predict_logistic(fitted.e, features)
    e_cheap = ev.predict_logistic(fitted.cheap_e, cheap_x)
    p_distance = scoring.predict_distance(fitted.distance, ops)
    p_cheap = ev.predict_logistic(fitted.cheap, cheap_x)
    ones = np.ones(len(pool), dtype=bool)

    def e_gate(e_hat):
        return {f"e0_{e0}": e_hat < e0 for e0 in E0S}

    def binned(arm):
        return {"bins": scoring.apply_gate(fitted.bins[arm], ops), "open": ones}

    def bin_rate(arm):
        return np.array([fitted.bin_rates[arm].get(int(d), 0.0) for d in ops])

    constant = np.full(len(pool), fitted.constant)
    return {
        "ridge_full": (p_ridge, e_gate(e_ridge), 1 - e_ridge),
        "ridge_p_open": (p_ridge, {"open": ones}, None),
        "distance": (p_distance, binned("distance"), bin_rate("distance")),
        "distance_E": (p_distance, e_gate(e_ridge), 1 - e_ridge),
        "distance_cheapE": (p_distance, e_gate(e_cheap), 1 - e_cheap),
        "cheap": (p_cheap, binned("cheap"), bin_rate("cheap")),
        "cheap_E": (p_cheap, e_gate(e_ridge), 1 - e_ridge),
        "constant": (constant, binned("constant"), bin_rate("constant")),
        "zero": (np.zeros(len(pool)), {"open": ones}, None),
        "wilson_constant": (
            np.full(len(pool), fitted.wilson_constant),
            {"open": ones},
            None,
        ),
        "_e_ridge": e_ridge,
    }


def score(arms: dict, hits, n: int, lineage, strata: dict, config: Config, seed: int):
    """Proper scores, gated shares and matched coverage per arm; the falsifier's paired bounds."""
    out: dict = {"arms": {}}
    miss_of, accept_of = {}, {}
    for name, value in arms.items():
        if name.startswith("_"):
            continue
        p_hat, gates, confidence = value
        report = ev.arm_report(strata, p_hat, gates, hits, n, lineage)
        miss = ~scoring.hit_wilson(p_hat, hits, n)
        miss_of[name] = miss
        if confidence is not None:
            take = accepted(confidence, config.q, seed)
            accept_of[name] = take
            report["matched_coverage"] = {
                "q": config.q,
                "accepted": int(take.sum()),
                "miss_rate_among_accepted": float(miss[take].mean()),
                "miss_share_of_population": float((take & miss).sum() / len(miss)),
            }
        out["arms"][name] = report
    p_ridge = arms["ridge_full"][0]
    ll_ridge = ev.log_loss(p_ridge, hits, n)
    out["log_loss_gain"] = {
        control: clustered(
            ev.log_loss(arms[control][0], hits, n) - ll_ridge,
            lineage,
            config.reps,
            seed,
            config.one_sided,
        )
        for control in (*CONTROLS, "wilson_constant", "zero")
    }
    q = config.q

    def per_pair(name):
        return (accept_of[name] & miss_of[name]) / q

    out["matched_coverage_gain"] = {
        control: clustered(
            per_pair(control) - per_pair("ridge_full"),
            lineage,
            config.reps,
            seed,
            config.one_sided,
        )
        for control in ("distance", "distance_cheapE", "cheap", "constant")
    }
    log_ok = all(out["log_loss_gain"][c]["one_sided_lower"] > 0 for c in CONTROLS)
    cover_ok = out["matched_coverage_gain"]["distance"]["one_sided_lower"] > 0
    out["falsifier"] = {
        "log_loss_below_every_control": log_ok,
        "matched_coverage_below_distance": cover_ok,
        "met": log_ok and cover_ok,
    }
    return out


def by_origin(arms: dict, pool, hits, n: int) -> dict:
    """Log loss and Brier per arm on each origin of a pool (the gate partition's held-out read)."""
    origins = np.array(pool.origin)
    out = {}
    for origin in sorted(set(origins.tolist())):
        at = origins == origin
        out[origin] = {
            "pairs": int(at.sum()),
            **{
                name: {
                    "log_loss": float(ev.log_loss(v[0][at], hits[at], n).mean()),
                    "brier": float(ev.brier(v[0][at], hits[at], n).mean()),
                }
                for name, v in arms.items()
                if not name.startswith("_")
            },
        }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(config: Config, run_dir: Path, log=print) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    start = time.perf_counter()
    deadline = start + config.hard_stop_seconds
    clock = loop.Clock()
    source = Path(config.source) / f"cycle-{config.cycle}"
    behaviours = list(config.behaviours or programs.train_behaviours())
    archive = LoopArchive.load(source / "archive.pkl.gz")
    scorer = PolicyScorer(load_policy(source / "policy"), "cpu")
    rnd = None
    if config.rnd_dump:
        with gzip.open(config.rnd_dump, "rb") as handle:
            dump = pickle.load(handle)
        rnd = dump["rnd"]
    source_report = json.loads((source / "report.json").read_text())
    n0 = source_report.get("rnd", {}).get("n0")
    procedure = Procedure(
        f"m2b s{config.seed} frozen {config.source} cycle {config.cycle}",
        config.budget,
        config.temperature,
        config.n,
    )
    canonical = loop.Canonical()
    canonical.update(archive)
    report: dict = {
        "config": dataclasses.asdict(config),
        "started": loop.now_pdt(),
        "procedure": procedure.row()["rollout"],
        "source_archive": str((source / "archive.pkl.gz").resolve()),
        "source_policy": str((source / "policy").resolve()),
        "rnd_n0": n0,
    }

    # pools: test first, so neither population can take its pairs
    taken: set = set()
    test, report["test_draws"] = draw_unseen(
        config.test, stream(config, "test"), behaviours, canonical, taken, config
    )
    fresh, report["fresh_draws"] = draw_unseen(
        config.population, stream(config, "fresh"), behaviours, canonical, taken, config
    )
    picks = selector.select(
        archive,
        config.cycle + 1,
        stream(config, "select"),
        selector.Selector(size=config.population, max_cuts=config.max_cuts),
    )
    keys = [k for k, _ in picks]
    report["archive_rules"] = dict(Counter(rule for _, rule in picks))
    held = loop.archive_pool(archive, keys, "archive")
    report["archive_origins"] = dict(Counter(held.origin))
    clock.mark("pools")

    labels = {}
    for name, pool, offset in (
        ("test", test, 1),
        ("fresh", fresh, 2),
        ("archive", held, 3),
    ):
        hits, _, ends, _, done = loop.roll(
            scorer, pool, procedure, config.seed * 1000 + 900 + offset, deadline
        )
        if done < len(pool):
            report["stop"] = f"hard stop during {name} rollouts ({done}/{len(pool)})"
            report["finished"] = loop.now_pdt()
            (run_dir / "report.json").write_text(
                json.dumps(report, indent=1, default=str)
            )
            log(report["stop"])
            return report
        labels[name] = hits
        report[f"{name}_rollouts"] = {**loop.success(hits, config.n), "ends": ends}
        log(f"{name}: {len(pool)} pairs, walker success {hits.mean() / config.n:.3f}")
    clock.mark("rollouts")

    features = {
        "test": heads.pair_features(scorer, test.jobs()),
        "fresh": heads.pair_features(scorer, fresh.jobs()),
        "archive": heads.pair_features(scorer, held.jobs()),
    }
    test_hits = labels["test"]
    test_strata = {
        "all": ["all"] * len(test),
        "d_ops": [str(d) for d in witness_ops(test)],
        "body": [programs.body_stratum(programs.body_atoms(p)) for p in test.path],
    }
    novelty = {
        "count": count_novelty(archive, test),
        "rnd": np.maximum(rnd.novelty(test.s), rnd.novelty(test.t))
        if rnd is not None
        else None,
    }
    rows = []
    report["populations"] = {}
    for offset, (name, pool) in enumerate((("archive", held), ("fresh", fresh))):
        hits = labels[name]
        parts = split(pool.lineage, config.seed + offset, config.split)
        fitted = fit_population(
            pool, features[name], hits, parts, config, config.seed + offset
        )
        # held-out read on the gate partition, by origin
        third = np.where(parts == 2)[0]
        gate_pool = pool.subset(third)
        gate_arms = arms_on(fitted, gate_pool, features[name][third])
        test_arms = arms_on(fitted, test, features["test"])
        e_ridge = test_arms["_e_ridge"]
        miss = ~scoring.hit_wilson(test_arms["ridge_full"][0], test_hits, config.n)
        curves = {
            "E": risk_coverage(miss, 1 - e_ridge, config.seed),
            "count": risk_coverage(miss, -novelty["count"], config.seed),
        }
        if novelty["rnd"] is not None:
            curves["RND"] = risk_coverage(miss, -novelty["rnd"], config.seed)
            curves["E+RND"] = risk_coverage(
                miss, ranks(1 - e_ridge) + ranks(-novelty["rnd"]), config.seed
            )
        report["populations"][name] = {
            "fit": fitted.facts,
            "origins_by_part": {
                str(i): dict(Counter(np.array(pool.origin)[parts == i].tolist()))
                for i in range(3)
            },
            "gate_partition": by_origin(gate_arms, gate_pool, hits[third], config.n),
            "test": score(
                test_arms,
                test_hits,
                config.n,
                test.lineage,
                test_strata,
                config,
                config.seed + offset,
            ),
            "risk_coverage_ridge_misses": curves,
        }
        for i in range(len(test)):
            rows.append(
                {
                    "population": name,
                    "pair": i,
                    "hits": int(test_hits[i]),
                    "d_ops": int(witness_ops(test)[i]),
                    **{
                        arm: float(v[0][i])
                        for arm, v in test_arms.items()
                        if not arm.startswith("_")
                    },
                    "e_ridge": float(e_ridge[i]),
                }
            )
        (run_dir / f"models-{name}.json").write_text(
            json.dumps(plain_models(fitted, features[name].shape[1]), indent=1)
        )
        log(f"{name}: falsifier {report['populations'][name]['test']['falsifier']}")
    report["test_novelty"] = {
        "count_mean": float(novelty["count"].mean()),
        "rnd_pass_rate": float((novelty["rnd"] <= n0).mean())
        if novelty["rnd"] is not None and n0 is not None
        else None,
    }
    clock.mark("fit_and_score")
    report["timings"] = clock.phases
    report["finished"] = loop.now_pdt()
    with gzip.open(run_dir / "test_rows.jsonl.gz", "wt") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    with gzip.open(run_dir / "pools.pkl.gz", "wb") as handle:
        pickle.dump(
            {
                "test": test,
                "fresh": fresh,
                "archive": held,
                "labels": labels,
                "features": features,
                "novelty": novelty,
            },
            handle,
        )
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--rnd-dump", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cycle", type=int, default=3)
    args = parser.parse_args(argv)
    config = Config(
        source=args.source, rnd_dump=args.rnd_dump, seed=args.seed, cycle=args.cycle
    )
    run(config, args.run_dir, log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
