"""certified-coverage (#170): pairs, rollouts and the two baselines, one seed per call.

Snapshot the live train-task archive first (never read it while it is being
written), then run a seed on the snapshot::

    uv run python experiments/certified_coverage/run.py snapshot \\
        --archive checkpoints/short-segment-search/short-segment-difference-20260923/archive/train-capped-s0-e3-segment-k16-d8.jsonl.gz
    uv run python experiments/certified_coverage/run.py run \\
        --archive checkpoints/certified-coverage/archive-<stamp>.jsonl.gz --seed 0

``run`` builds deduplicated pairs from the archive (memory) and from the F1
and F2 witness lineages (generalisation), draws ``--pairs`` per source with
the seed, rolls each out ``n`` times with BEST, fits both baselines on the
archive pairs and scores them on every source. ``--pairs-only`` stops after
the pair counts. Outputs, in ``--out-dir``: ``pairs-s<seed>.jsonl.gz`` (one row
per judged pair with its hits and the rollout procedure) and
``report-s<seed>.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from experiments.certified_coverage import pairs as sampler  # noqa: E402
from experiments.certified_coverage import score as scoring  # noqa: E402
from experiments.certified_coverage.rollout import Procedure, rollout_hits  # noqa: E402

DEFAULT_OUT = REPO / "checkpoints" / "certified-coverage"
CAPPED = REPO / "reports" / "short-segment-20260923" / "train_capped.jsonl"
PDT = ZoneInfo("America/Los_Angeles")


def snapshot(live: Path, out_dir: Path) -> tuple[Path, int]:
    """Copy the complete records of a gzip archive that may still be appended to."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(PDT).strftime("%Y%m%d-%H%M%S-PDT")
    target = out_dir / f"archive-{stamp}.jsonl.gz"
    records = []
    try:
        with gzip.open(live, "rt") as lines:
            for line in lines:
                if line.endswith("\n"):
                    json.loads(line)
                    records.append(line)
    except (EOFError, gzip.BadGzipFile, zlib.error, json.JSONDecodeError):
        pass  # a member mid-write: keep what came before it
    with gzip.open(target, "wt") as handle:
        handle.writelines(records)
    return target, len(records)


def count_tasks(path: Path) -> int:
    return sum(1 for _ in sampler.archive_records(path))


def summary(pairs: list[sampler.Pair]) -> dict:
    multiplicity = np.array([p.multiplicity for p in pairs]) if pairs else np.zeros(0)
    return {
        "pairs": len(pairs),
        "lineage_pairs": int(multiplicity.sum()),
        "max_multiplicity": int(multiplicity.max()) if len(pairs) else 0,
        "behaviours": len({p.behaviour for p in pairs}),
        "by_d_cuts": {
            str(d): sum(p.d_cuts == d for p in pairs)
            for d in sorted({p.d_cuts for p in pairs})
        },
    }


def load_scorer(checkpoint: str, device: str, threads: int):
    import torch

    from wandering_light.solve import BEST
    from wandering_light.stack_policy import PolicyScorer, load_policy
    from wandering_light.stack_policy_hub import fetch_stack_policy

    torch.set_num_threads(threads)
    if checkpoint == "best":
        path, name = (
            fetch_stack_policy(BEST.repo_id, BEST.revision),
            (f"{BEST.name} {BEST.repo_id}@{BEST.revision}"),
        )
    else:
        path, name = Path(checkpoint), str(checkpoint)
    return PolicyScorer(load_policy(path, device=device), device), name


def build_pairs(archive: Path, max_cuts: int | None) -> dict[str, list[sampler.Pair]]:
    from experiments.staged_witnesses import check as staged

    capped = {
        row["behaviour"]: row["program"]
        for row in map(json.loads, CAPPED.read_text().splitlines())
    }
    fresh = sampler.fresh_tasks(staged.v4_tasks(), capped)
    out = {sampler.ARCHIVE: sampler.archive_pairs(archive, max_cuts=max_cuts)}
    for source, tasks in fresh.items():
        out[source] = sampler.witness_pairs(tasks, source=source, max_cuts=max_cuts)
    return out


def score_baselines(
    judged: dict[str, list[sampler.Pair]],
    hits: dict[str, np.ndarray],
    n: int,
    rule: scoring.Rule,
) -> dict:
    def column(source, field):
        return np.array([getattr(p, field) for p in judged[source]])

    a_cuts, a_ops, a_hits = (
        column(sampler.ARCHIVE, "d_cuts"),
        column(sampler.ARCHIVE, "d_ops"),
        hits[sampler.ARCHIVE],
    )
    constant = scoring.fit_constant(a_cuts, a_hits, n)
    distance = scoring.fit_distance(a_ops, a_hits, n)
    predictors = {
        "constant": (
            constant,
            lambda cuts, ops: scoring.predict_constant(constant, cuts),
        ),
        "distance": (
            distance,
            lambda cuts, ops: scoring.predict_distance(distance, ops),
        ),
    }
    out = {}
    for name, (model, predict) in predictors.items():
        open_bins = scoring.fit_gate(a_ops, predict(a_cuts, a_ops), a_hits, n, rule)
        scored = {}
        for source in judged:
            cuts, ops = column(source, "d_cuts"), column(source, "d_ops")
            v_hat = predict(cuts, ops)
            scored[source] = {
                "gated": scoring.report(
                    cuts,
                    v_hat,
                    scoring.apply_gate(open_bins, ops),
                    hits[source],
                    n,
                    rule,
                ),
                "always_open": scoring.report(
                    cuts, v_hat, np.ones(len(cuts), dtype=bool), hits[source], n, rule
                ),
            }
        out[name] = {
            "model": {str(k): v for k, v in model.items()}
            if name == "distance"
            else {
                "by_stratum": {str(k): v for k, v in model["by_stratum"].items()},
                "overall": model["overall"],
            },
            "open_d_ops": open_bins,
            "scores": scored,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--archive", type=Path, required=True)
    snap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    run = commands.add_parser("run")
    run.add_argument(
        "--archive", type=Path, required=True, help="a snapshot, not the live file"
    )
    run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--max-cuts", type=int, default=1, help="stratum: d_cuts at most this"
    )
    run.add_argument("--pairs", type=int, default=1000, help="pairs drawn per source")
    run.add_argument("--checkpoint", default="best")
    run.add_argument("--n", type=int, default=8)
    run.add_argument("--budget", type=int, default=16)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--pairs-only", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        target, tasks = snapshot(args.archive, args.out_dir)
        print(f"{tasks} tasks -> {target}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    everything = build_pairs(args.archive, args.max_cuts)
    report = {
        "issue": "certified-coverage (#170)",
        "archive_snapshot": str(args.archive.resolve()),
        "archive_tasks": count_tasks(args.archive),
        "seed": args.seed,
        "max_cuts": args.max_cuts,
        "pairs_per_source": args.pairs,
        "sources": {source: summary(pairs) for source, pairs in everything.items()},
    }
    judged = {
        source: sampler.draw(pairs, args.pairs, args.seed)
        for source, pairs in everything.items()
    }
    report["judged"] = {source: summary(pairs) for source, pairs in judged.items()}
    print(json.dumps(report["judged"]), flush=True)
    if args.pairs_only:
        (args.out_dir / f"pairs-only-s{args.seed}.json").write_text(
            json.dumps(report, indent=1)
        )
        return 0

    scorer, policy = load_scorer(args.checkpoint, args.device, args.threads)
    procedure = Procedure(policy, args.budget, args.temperature, args.n)
    rule = scoring.Rule()
    hits = {}
    with gzip.open(args.out_dir / f"pairs-s{args.seed}.jsonl.gz", "wt") as rows:
        for offset, (source, pairs) in enumerate(judged.items()):
            clock = time.perf_counter()
            hits[source] = rollout_hits(
                scorer, pairs, procedure, args.seed * 1000 + offset
            )
            for pair, h in zip(pairs, hits[source].tolist(), strict=True):
                rows.write(
                    json.dumps({**pair.row(), "hits": h, **procedure.row()}) + "\n"
                )
            print(
                f"{source}: {len(pairs)} pairs, mean V_emp {hits[source].mean() / args.n:.3f}, "
                f"{time.perf_counter() - clock:.0f}s",
                flush=True,
            )
    report.update(
        procedure.row(),
        rule={
            "r": rule.r,
            "p": rule.p,
            "tau": rule.tau,
            "z": rule.z,
            "min_bin": rule.min_bin,
        },
        baselines=score_baselines(judged, hits, args.n, rule),
        seconds=time.perf_counter() - started,
        finished=datetime.now(PDT).isoformat(timespec="seconds"),
    )
    (args.out_dir / f"report-s{args.seed}.json").write_text(
        json.dumps(report, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
