"""novelty-reduction (#171): replay a finished run with its saved policies, dumping per-pair rows.

The loop's report keeps aggregates only. This re-runs ``cycle.run`` with the
same config and seed, but every policy phase loads that cycle's saved weights
instead of training. Every other phase is seeded, so proposals, rollouts,
RND, heads and test pools should replay exactly; ``--check`` compares each
replayed report with the original, and the dumps are only read if it passes.

Dumps, per cycle, into ``<out>/cycle-<c>/dump.pkl.gz``:

- ``calibration``: pooled features, hits, partitions, witness paths, and the
  fitted models (P, E heads and the baselines);
- ``F3`` and ``archive``: per pair s, t, witness path, lineage, strata labels,
  hits, features, P_hat, E_hat, RND novelty, n0;
- ``rows``: (origin, path) of every pair the policy phase trained on;
- ``rnd``: the RND module after the cycle, for later probes.

Diagnostic only: the numbers from it are not registered results.

Known limitation: every bootstrap block loads the final cycle-0 weights, so a
run whose floor was met after block 1 does not replay (its per-block rollout
seeds differ, and ``--check`` rejects it). m2-s0 met the floor at block 1.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from experiments.novelty_reduction import cycle as loop
from experiments.novelty_reduction import heads


def replay(source: Path, out: Path) -> None:
    report0 = json.loads((source / "cycle-0" / "report.json").read_text())
    config = loop.Config(
        **{
            **report0["config"],
            "architecture": tuple(report0["config"]["architecture"]),
            "train_device": "cpu",
        }
    )
    state = {"cycle": 0, "dump": {}}
    dumps: dict[int, dict] = {}

    def train_epochs(model, optimizer, rows, config, epochs, tag):
        cycle = int(tag.split(":")[0])
        state["cycle"] = cycle
        weights = load_file(
            str(source / f"cycle-{cycle}" / "policy" / "model.safetensors")
        )
        model.load_state_dict(weights)
        model.eval()
        return {
            "train_cross_entropy": [],
            "epochs": epochs,
            "steps": 0,
            "replayed": True,
        }

    original_rows = loop.policy_rows

    def policy_rows(archive, config, cycle, previous_paths):
        rows, facts, paths = original_rows(archive, config, cycle, previous_paths)
        dumps.setdefault(cycle, {})["rows"] = [
            (archive.pairs[k].origin, archive.pairs[k].witness_path, path)
            for k, path in paths.items()
        ]
        return rows, facts, paths

    original_fit = loop.fit_models

    def fit_models(scorer, pool, hits, n, parts, seed, n0):
        models, features = original_fit(scorer, pool, hits, n, parts, seed, n0)
        dumps.setdefault(state["cycle"], {})["calibration"] = {
            "features": features,
            "hits": np.asarray(hits),
            "parts": np.asarray(parts),
            "paths": list(pool.path),
            "lineage": list(pool.lineage),
            "keys": list(pool.keys),
            "origin": list(pool.origin),
            "models": copy.deepcopy(models),
            "seed": seed,
        }
        return models, features

    original_score = loop.score_pool

    def score_pool(models, scorer, rnd, pool, hits, n, stratifications, counted=None):
        result = original_score(
            models, scorer, rnd, pool, hits, n, stratifications, counted
        )
        if len(pool):
            features = heads.pair_features(scorer, pool.jobs())
            entry = dumps.setdefault(state["cycle"], {})
            entry[pool.source] = {
                "s": list(pool.s),
                "t": list(pool.t),
                "paths": list(pool.path),
                "lineage": list(pool.lineage),
                "keys": list(pool.keys),
                "origin": list(pool.origin),
                "labels": {k: list(v) for k, v in stratifications.items()},
                "hits": np.asarray(hits),
                "features": features,
                "p_hat": models.p_head.predict(features),
                "e_hat": models.e_head.predict(features),
                "n_s": rnd.novelty(pool.s),
                "n_t": rnd.novelty(pool.t),
                "n0": models.n0,
            }
            entry["rnd"] = copy.deepcopy(rnd)
        return result

    loop.train_epochs = train_epochs
    loop.policy_rows = policy_rows
    loop.fit_models = fit_models
    loop.score_pool = score_pool

    def log(line: str) -> None:
        print(line, flush=True)
        # a cycle's report is saved just before its log line: flush its dump
        for cycle in sorted(dumps):
            path = out / f"cycle-{cycle}" / "dump.pkl.gz"
            if (out / f"cycle-{cycle}" / "report.json").exists() and not path.exists():
                with gzip.open(path, "wb") as handle:
                    pickle.dump(dumps[cycle], handle, protocol=pickle.HIGHEST_PROTOCOL)

    loop.run(config, out, log=log)


def check(source: Path, out: Path) -> bool:
    """Replayed reports must match the originals on every rollout-derived number."""
    same = True
    for cycle_dir in sorted(source.glob("cycle-*")):
        a = json.loads((cycle_dir / "report.json").read_text())
        b = json.loads((out / cycle_dir.name / "report.json").read_text())
        if cycle_dir.name == "cycle-0":
            pairs = [
                (
                    "blocks.histogram",
                    a["blocks"][-1]["histogram"],
                    b["blocks"][-1]["histogram"],
                )
            ]
        else:
            pairs = [
                ("paired_check", a["paired_check"], b["paired_check"]),
                (
                    "calibration.histogram",
                    a["calibration"]["histogram"],
                    b["calibration"]["histogram"],
                ),
                (
                    "calibration.heads",
                    a["calibration"]["heads"],
                    b["calibration"]["heads"],
                ),
                ("rnd.n0", a["rnd"]["n0"], b["rnd"]["n0"]),
                (
                    "F3.histogram",
                    a["test"]["F3"]["histogram"],
                    b["test"]["F3"]["histogram"],
                ),
                (
                    "archive.histogram",
                    a["test"]["archive"]["histogram"],
                    b["test"]["archive"]["histogram"],
                ),
                (
                    "F3.full.log_loss",
                    a["test"]["F3"]["arms"]["full"]["log_loss"],
                    b["test"]["F3"]["arms"]["full"]["log_loss"],
                ),
                ("archive", a["archive"], b["archive"]),
            ]
        for name, x, y in pairs:
            ok = x == y
            same &= ok
            print(f"{cycle_dir.name} {name}: {'same' if ok else 'DIFFERENT'}")
    return same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        replay(args.source, args.out)
    return 0 if check(args.source, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
