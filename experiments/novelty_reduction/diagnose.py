"""novelty-reduction (#171): offline diagnostics on a replayed run's dumps (M2 seed 0).

Diagnostic, not registered. Reads ``<replay>/cycle-<c>/dump.pkl.gz`` (from
``replay.py``), the cycle reports and archives, and prints one JSON object:

1. body growth: walker success by d_ops, body stratum and last combinator on
   F3 and calibration rows; the body-atom distribution of the rows trained on;
2. why P loses: refits scored on partition 2 and on F3, and a reliability table;
3. why E and N carry no signal: E's AUROC, the Wilson-miss noise floor of a
   perfect predictor, and RND pass rates by coarse-key frequency.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.certified_coverage import score as scoring
from experiments.novelty_reduction import canon, heads, programs
from experiments.novelty_reduction import evaluate as ev
from experiments.novelty_reduction.archive import LoopArchive
from experiments.novelty_reduction.cycle import last_combinator
from wandering_light.stack_machine import Bool, Int, Lst

N = 8


def load(path: Path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def by(labels, hits) -> dict:
    groups = defaultdict(list)
    for label, h in zip(labels, hits, strict=True):
        groups[label].append(h)
    return {
        str(k): {"pairs": len(v), "walker_success": round(float(np.mean(v)) / N, 3)}
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def breakdown(paths, hits) -> dict:
    return {
        "pairs": len(paths),
        "walker_success": round(float(np.mean(hits)) / N, 3),
        "d_ops": by([len(p) for p in paths], hits),
        "body": by(
            [programs.body_stratum(programs.body_atoms(p)) for p in paths], hits
        ),
        "last_combinator": by([last_combinator(p) for p in paths], hits),
    }


def scores(p_hat, hits, clusters=None) -> dict:
    out = {
        "pairs": len(hits),
        "log_loss": round(float(ev.log_loss(p_hat, hits, N).mean()), 4),
        "brier": round(float(ev.brier(p_hat, hits, N).mean()), 4),
    }
    if clusters is not None:
        interval = ev.clustered_interval(ev.log_loss(p_hat, hits, N), clusters)
        out["log_loss_90"] = [round(interval["low"], 4), round(interval["high"], 4)]
    return out


def grouped_folds(groups, k: int, seed: int) -> np.ndarray:
    order = list(dict.fromkeys(groups))
    np.random.default_rng(seed).shuffle(order)
    fold = {g: i % k for i, g in enumerate(order)}
    return np.array([fold[g] for g in groups])


def ridge_cv(x, hits, groups, lambdas, k=5, seed=0) -> tuple[float, dict]:
    folds = grouped_folds(groups, k, seed)
    table = {}
    for lam in lambdas:
        losses = []
        for f in range(k):
            train, held = folds != f, folds == f
            if not held.any() or not train.any():
                continue
            model = ev.fit_logistic(x[train], hits[train], N, ridge=lam)
            losses.append(
                ev.log_loss(ev.predict_logistic(model, x[held]), hits[held], N)
            )
        table[lam] = float(np.concatenate(losses).mean())
    best = min(table, key=table.get)
    return best, {str(k): round(v, 4) for k, v in table.items()}


def cheap(dump_pool, states=None) -> np.ndarray:
    return np.array(
        [
            canon.cheap_features(s, t, p)
            for s, t, p in zip(
                dump_pool["s"], dump_pool["t"], dump_pool["paths"], strict=True
            )
        ]
    )


def auroc(score, positive) -> float | None:
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return None
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1.0
    # average ties
    for value in np.unique(score):
        tied = score == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    pos = positive.sum()
    return float(
        (ranks[positive].sum() - pos * (pos + 1) / 2) / (pos * (~positive).sum())
    )


def miss_by_true_p() -> dict:
    """P(a perfect P = p fails a fresh n-rollout Wilson check), exact, for p = k / n."""
    from math import comb

    out = {}
    for k in range(N + 1):
        p = k / N
        total = 0.0
        for h in range(N + 1):
            miss = not scoring.hit_wilson(np.array([p]), np.array([h]), N)[0]
            total += comb(N, h) * p**h * (1 - p) ** (N - h) * miss
        out[f"{k}/8"] = round(total, 4)
    return out


def miss_given_observed(histogram) -> dict:
    """P(miss | fresh hits = h) for a perfect P, prior on true p = the P_emp histogram (prior-dependent)."""
    from math import comb

    weights = np.asarray(histogram, float) / sum(histogram)
    out = {}
    for h in range(N + 1):
        joint = miss_joint = 0.0
        for k, w in enumerate(weights):
            p = k / N
            like = w * comb(N, h) * p**h * (1 - p) ** (N - h)
            joint += like
            miss_joint += like * (
                not scoring.hit_wilson(np.array([p]), np.array([h]), N)[0]
            )
        out[str(h)] = round(miss_joint / joint, 4) if joint else None
    return out


def value_type(value) -> str:
    if isinstance(value, Int):
        return "i"
    if isinstance(value, Bool):
        return "b"
    if isinstance(value, Lst):
        return "l"
    return "?"


def coarse_key(state) -> tuple:
    """Per example, the stack's top-level type signature (so its depth too); examples sorted."""
    return tuple(sorted("".join(value_type(v) for v in stack) for stack in state))


def count_bin(c: int) -> str:
    if c == 0:
        return "0"
    if c < 10:
        return "1-9"
    if c < 100:
        return "10-99"
    if c < 1000:
        return "100-999"
    return "1000+"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    out: dict = {"body_growth": {}, "p": {}, "e": {}, "rnd": {}}
    dumps = {
        c: load(args.replay / f"cycle-{c}" / "dump.pkl.gz")
        for c in range(args.cycles + 1)
    }
    reports = {
        c: json.loads((args.replay / f"cycle-{c}" / "report.json").read_text())
        for c in range(args.cycles + 1)
    }

    # 1. body growth
    boot = reports[0]["blocks"][-1]
    out["body_growth"]["cycle-0 bootstrap check"] = {
        "walker_success": boot["walker_success"],
        "d_ops": boot["by_d_ops"],
        "body": boot["by_body"],
        "last_combinator": boot["by_combinator"],
    }
    for c in range(args.cycles + 1):
        d = dumps[c]
        entry = {}
        if "F3" in d:
            entry["F3"] = breakdown(d["F3"]["paths"], d["F3"]["hits"])
        if "calibration" in d:
            entry["calibration"] = breakdown(
                d["calibration"]["paths"], d["calibration"]["hits"]
            )
        trained = {}
        for origin in ("proposal", "rollout"):
            paths = [path for o, _, path in d["rows"] if o == origin]
            strata = Counter(
                programs.body_stratum(programs.body_atoms(p)) for p in paths
            )
            rows = Counter()
            for p in paths:
                rows[programs.body_stratum(programs.body_atoms(p))] += len(p)
            trained[origin] = {
                "pairs": len(paths),
                "pairs_by_body": dict(sorted(strata.items())),
                "rows_by_body": dict(sorted(rows.items())),
                "mean_len": round(float(np.mean([len(p) for p in paths])), 2)
                if paths
                else None,
            }
        entry["trained_rows"] = trained
        out["body_growth"][f"cycle-{c}"] = entry

    # 2. why P loses
    for c in range(1, args.cycles + 1):
        cal, f3 = dumps[c]["calibration"], dumps[c]["F3"]
        x, hits, parts = cal["features"], cal["hits"], cal["parts"]
        models = cal["models"]
        third = parts == 2
        rate = hits / N
        entry = {
            "pairs_by_part": np.bincount(parts, minlength=3).tolist(),
            "width": int(x.shape[1]),
        }
        seed = cal["seed"]
        fits = {"a_as_fit": models.p_head}
        fits["b_parts_0_1"] = heads.fit_head(
            x[parts != 2], rate[parts != 2], heads.HEAD_FIT, seed
        )
        for name, head in fits.items():
            entry[name] = {
                "partition_2": scores(head.predict(x[third]), hits[third]),
                "F3": scores(head.predict(f3["features"]), f3["hits"], f3["lineage"]),
            }
        # d: ridge logistic on cheap features (+) trunk embedding, lambda by lineage-grouped CV
        cal_pool = {"s": [], "t": [], "paths": cal["paths"]}
        archive = LoopArchive.load(args.replay / f"cycle-{c}" / "archive.pkl.gz")
        cal_pool["s"] = [archive.states[k[0]] for k in cal["keys"]]
        cal_pool["t"] = [archive.states[k[1]] for k in cal["keys"]]
        cx, fx = cheap(cal_pool), cheap(f3)
        train = parts != 2
        groups = [str(g) for g, keep in zip(cal["lineage"], train, strict=True) if keep]
        lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        for name, xa, xf in (
            ("d_cheap_plus_trunk", np.hstack([cx, x]), np.hstack([fx, f3["features"]])),
            ("d_trunk_only", x, f3["features"]),
            ("d_cheap_only", cx, fx),
        ):
            lam, table = ridge_cv(xa[train], hits[train], groups, lambdas)
            model = ev.fit_logistic(xa[train], hits[train], N, ridge=lam)
            entry[name] = {
                "lambda": lam,
                "cv_log_loss": table,
                "partition_2": scores(
                    ev.predict_logistic(model, xa[third]), hits[third]
                ),
                "F3": scores(ev.predict_logistic(model, xf), f3["hits"], f3["lineage"]),
            }
        entry["distance_as_fit"] = {
            "partition_2": scores(
                scoring.predict_distance(
                    models.distance, np.array([len(p) for p in cal["paths"]])[third]
                ),
                hits[third],
            ),
            "F3": scores(
                scoring.predict_distance(
                    models.distance, np.array([len(p) for p in f3["paths"]])
                ),
                f3["hits"],
                f3["lineage"],
            ),
        }
        out["p"][f"cycle-{c}"] = entry

    # e: reliability of P_hat on the last cycle's F3
    p_hat, emp = f3["p_hat"], f3["hits"] / N
    edges = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p_hat, edges) - 1, 0, 9)
    out["p"]["reliability_F3"] = [
        {
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "pairs": int((which == b).sum()),
            "mean_p_hat": round(float(p_hat[which == b].mean()), 3)
            if (which == b).any()
            else None,
            "mean_p_emp": round(float(emp[which == b].mean()), 3)
            if (which == b).any()
            else None,
        }
        for b in range(10)
    ]

    # 3a. E: AUROC for the Wilson miss, and the noise floor
    out["e"]["perfect_p_miss_by_true_p"] = miss_by_true_p()
    for c in range(1, args.cycles + 1):
        cal, f3 = dumps[c]["calibration"], dumps[c]["F3"]
        models, third = cal["models"], cal["parts"] == 2
        p2 = models.p_head.predict(cal["features"][third])
        e2 = models.e_head.predict(cal["features"][third])
        miss2 = ~scoring.hit_wilson(p2, cal["hits"][third], N)
        miss3 = ~scoring.hit_wilson(f3["p_hat"], f3["hits"], N)
        hist = reports[c]["calibration"]["histogram"]
        out["e"][f"cycle-{c}"] = {
            "auroc_partition_2": auroc(e2, miss2),
            "miss_rate_partition_2": round(float(miss2.mean()), 3),
            "auroc_F3": auroc(f3["e_hat"], miss3),
            "miss_rate_F3": round(float(miss3.mean()), 3),
            "observed_e_target_by_hits": {
                k: round(v["rate"], 3)
                for k, v in reports[c]["calibration"]["heads"][
                    "e_target_rate_by_hits"
                ].items()
            },
            "observed_e_target_rate": round(
                reports[c]["calibration"]["heads"]["e_target_rate"], 3
            ),
            "floor_given_observed_prior_dependent": miss_given_observed(hist),
        }

    # 3b. RND: F3 pass rate by coarse-key frequency in the archive
    for c in range(1, args.cycles + 1):
        f3 = dumps[c]["F3"]
        archive = LoopArchive.load(args.replay / f"cycle-{c}" / "archive.pkl.gz")
        keys = Counter(coarse_key(state) for state in archive.states)
        n0 = f3["n0"]
        nodes = defaultdict(list)
        exact = defaultdict(list)
        for states, novelty in ((f3["s"], f3["n_s"]), (f3["t"], f3["n_t"])):
            for state, value in zip(states, novelty, strict=True):
                nodes[count_bin(keys.get(coarse_key(state), 0))].append(value <= n0)
                exact[
                    "in_archive" if state in archive.index else "not_in_archive"
                ].append(value <= n0)
        pair_open = np.maximum(f3["n_s"], f3["n_t"]) <= n0
        pair_bin = [
            count_bin(min(keys.get(coarse_key(s), 0), keys.get(coarse_key(t), 0)))
            for s, t in zip(f3["s"], f3["t"], strict=True)
        ]
        pairs = defaultdict(list)
        for label, ok in zip(pair_bin, pair_open, strict=True):
            pairs[label].append(ok)
        order = ["0", "1-9", "10-99", "100-999", "1000+"]
        out["rnd"][f"cycle-{c}"] = {
            "n0": n0,
            "distinct_coarse_keys_in_archive": len(keys),
            "archive_nodes": len(archive.states),
            "node_pass_by_coarse_count": {
                b: {"nodes": len(nodes[b]), "pass": round(float(np.mean(nodes[b])), 3)}
                for b in order
                if nodes[b]
            },
            "node_pass_by_exact_membership": {
                b: {"nodes": len(v), "pass": round(float(np.mean(v)), 3)}
                for b, v in exact.items()
            },
            "pair_pass_by_min_coarse_count": {
                b: {"pairs": len(pairs[b]), "pass": round(float(np.mean(pairs[b])), 3)}
                for b in order
                if pairs[b]
            },
        }
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
