"""novelty-reduction (#171): the phase runner.

One cycle:

1. **propose** ``P`` goals: random capped programs from fresh inputs or
   archived nodes; every closed state and pair on each path is archived;
2. **policy phase** (``--train-device``): imitate the shortest known path of
   every archived pair within the curriculum, rollout-made rows capped at the
   proposal-made count, shortest ``d_ops`` first;
3. **calibration** (CPU): the selector's pairs, ``n`` rollouts each on the
   frozen policy, every closed state reached archived; V and E cross-fit, the
   RND predictor trained and its gate threshold recalibrated;
4. **test**: fresh F3 pairs (never archived) and archive pairs, scored by the
   heads against both certified-coverage (#170) baselines at each e0.

The policy starts from scratch and keeps training across cycles. Each cycle
writes ``cycle-<c>/``: the archive snapshot, the policy (weights only),
``report.json`` and the judged rows. Run::

    uv run python experiments/novelty_reduction/cycle.py --run-dir <dir> --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from experiments.certified_coverage import pairs as sampler  # noqa: E402
from experiments.certified_coverage import score as scoring  # noqa: E402
from experiments.certified_coverage.rollout import Procedure, walk  # noqa: E402
from experiments.novelty_reduction import heads as H  # noqa: E402, N812
from experiments.novelty_reduction import programs  # noqa: E402
from experiments.novelty_reduction import select as selector  # noqa: E402
from experiments.novelty_reduction.archive import (  # noqa: E402
    PROPOSAL,
    ROLLOUT,
    LoopArchive,
)
from wandering_light.rollout_policy import PolicyArchitecture  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    DIFFERENCE,
    OP_INDEX,
    PolicyScorer,
    Row,
    StackPairPolicy,
    collate,
    node_encoder,
    save_policy,
    seed_everything,
)

PDT = ZoneInfo("America/Los_Angeles")
E0S = (0.1, 0.2, 0.3)
ARCHITECTURE_1M = (128, 320, 640)  # compute_optimal_mark28.ARMS["1m"]


@dataclass(frozen=True, slots=True)
class Config:
    seed: int = 0
    cycles: int = 3
    proposals: int = 1000
    start_archived_share: float = 0.5  # proposals from archived nodes, cycle 2 on
    calibration: int = 1000
    test_fresh: int = 1000
    test_fresh_final: int = 0  # the last cycle's F3 pool when > 0
    test_archive: int = 1000
    tracked_new: int = 5000
    max_cuts: int = 1
    epochs: int = 1
    row_cap: int = 300_000
    learning_rate: float = 1.25e-4
    batch: int = 256
    architecture: tuple[int, int, int] = ARCHITECTURE_1M
    rnd_replay: int = 20_000
    rnd_epochs: int = 1
    budget: int = 16
    temperature: float = 1.0
    n: int = 8
    train_device: str = "cuda"
    threads: int = 1
    behaviours: tuple[str, ...] | None = None  # None: the v4 train behaviours


def stream(config: Config, cycle: int, role: str) -> random.Random:
    """One rng per (seed, cycle, role): F3 never shares a draw with the loop."""
    return random.Random(f"{config.seed}:{cycle}:{role}")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def propose_phase(
    archive: LoopArchive, config: Config, cycle: int, behaviours
) -> tuple[dict, list[dict]]:
    """Counts, and one log row per proposal with its start kind."""
    rng = stream(config, cycle, "propose")
    grammar = programs.Grammar()
    made = Counter()
    log = []
    for _ in range(config.proposals):
        if cycle > 1 and archive.states and rng.random() < config.start_archived_share:
            start = rng.randrange(len(archive.states))
            product, route = archive.states[start], archive.origins[start]
            kind, behaviour = "archive", None
        else:
            behaviour, inputs = programs.draw_inputs(rng, behaviours)
            product, route = sampler.parse_inputs(inputs), (inputs, ())
            kind, start = "inputs", None
        made[f"from_{kind}"] += 1
        proposal = programs.propose(product, rng, grammar)
        entry = {
            "start_kind": kind,
            "start_node": start,
            "behaviour": behaviour,
            "k": len(product),
            "program": " ".join(proposal.ops),
            "cuts": proposal.cuts,
        }
        log.append(entry)
        if not proposal.cuts:
            made["empty"] += 1
            continue
        added, relaxed = archive.add_path(
            proposal.states, proposal.ops, PROPOSAL, cycle, start_origin=route
        )
        entry["pairs_added"], entry["relaxations"] = added, relaxed
        made["pairs"] += added
        made["relaxations"] += relaxed
        made["segments"] += len(proposal.cuts)
    return dict(made), log


def policy_rows(archive: LoopArchive, config: Config) -> tuple[list[Row], dict]:
    """Rows along shortest known paths; rollout-made rows at most the proposal-made count."""
    within = [r for r in archive.pairs.values() if r.d_cuts <= config.max_cuts]
    made = [r for r in within if r.origin == PROPOSAL]
    walked = sorted(
        (r for r in within if r.origin == ROLLOUT), key=lambda r: (r.d_ops, r.s, r.t)
    )
    budget = sum(r.d_ops for r in made)
    kept = []
    for record in walked:
        if budget - record.d_ops < 0:
            break
        budget -= record.d_ops
        kept.append(record)
    rows = []
    for index, record in enumerate([*made, *kept]):
        encode = node_encoder(archive.states[record.t], DIFFERENCE)
        states = sampler.replay(archive.states[record.s], record.path)
        for step, name in enumerate(record.path):
            rows.append(
                Row(
                    encode(states[step]),
                    OP_INDEX[name],
                    index,
                    step,
                    record.d_ops - step,
                )
            )
    proposal_rows = sum(r.d_ops for r in made)
    if len(rows) > config.row_cap:
        rows = random.Random(config.seed).sample(rows, config.row_cap)
    return rows, {
        "proposal_rows": proposal_rows,
        "rollout_rows": sum(r.d_ops for r in kept),
        "rollout_pairs_available": len(walked),
        "rows_trained": len(rows),
    }


def train_epochs(model, optimizer, rows: list[Row], config: Config, cycle: int) -> dict:
    device = next(model.parameters()).device
    rng = random.Random(f"{config.seed}:{cycle}:train")
    model.train()
    history = []
    steps = 0
    for _ in range(config.epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), config.batch):
            chunk = [rows[i] for i in order[start : start + config.batch]]
            inputs = collate([row.items for row in chunk], device)
            labels = torch.tensor([row.label for row in chunk], device=device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(*inputs), labels, reduction="sum")
            (loss / len(chunk)).backward()
            optimizer.step()
            steps += 1
            total += loss.item()
        history.append(total / max(len(rows), 1))
    model.eval()
    return {"train_cross_entropy": history, "epochs": config.epochs, "steps": steps}


def attempt(archive, scorer, keys, procedure, seed, cycle, *, record: bool):
    """Rollouts on archived pairs; with ``record``, archive their paths and count visits."""
    jobs = [(archive.states[s], archive.states[t]) for s, t in keys]
    walks = walk(scorer, jobs, procedure, seed)
    hits = np.zeros(len(keys), dtype=np.int64)
    steps = np.zeros(len(keys))
    made = Counter()
    touched: set[int] = set()
    for walker in walks:
        made[f"end_{walker.end}"] += 1
        hits[walker.owner] += walker.hit
        steps[walker.owner] += len(walker.ops) if walker.hit else procedure.budget
        if record:
            added, relaxed = archive.add_path(
                walker.states,
                walker.ops,
                ROLLOUT,
                cycle,
                start_origin=archive.origins[keys[walker.owner][0]],
            )
            made["pairs"] += added
            made["relaxations"] += relaxed
            # "through": past the start, which every walker of the pair shares
            through = {
                archive.index[s] for s in walker.states[1:] if s in archive.index
            }
            for node in through:
                archive.visits[node] += 1
            touched |= through
    if record:
        for s, t in keys:
            archive.pairs[(s, t)].attempts += 1
            archive.visits[s] += 1
            archive.visits[t] += 1
            touched |= {s, t}
    return hits, steps / procedure.n, dict(made), touched


def score_block(strata, v_hat, gates: dict, hits, n) -> dict:
    rule = scoring.Rule()
    return {
        name: scoring.report(strata, v_hat, gate, hits, n, rule)
        for name, gate in gates.items()
    }


def baselines(cal_ops, cal_hits, n, tests: dict) -> dict:
    """#170's constant and distance baselines, fit on this cycle's calibration pairs."""
    rule = scoring.Rule()
    ones = np.ones(len(cal_ops), dtype=int)
    constant = scoring.fit_constant(ones, cal_hits, n)
    distance = scoring.fit_distance(cal_ops, cal_hits, n)
    out = {}
    for name, predict in (
        (
            "constant",
            lambda ops: scoring.predict_constant(
                constant, np.ones(len(ops), dtype=int)
            ),
        ),
        ("distance", lambda ops: scoring.predict_distance(distance, ops)),
    ):
        open_bins = scoring.fit_gate(cal_ops, predict(cal_ops), cal_hits, n, rule)
        out[name] = {"open_d_ops": open_bins, "scores": {}}
        for source, (strata, ops, hits) in tests.items():
            v_hat = predict(ops)
            out[name]["scores"][source] = score_block(
                strata,
                v_hat,
                {
                    "gated": scoring.apply_gate(open_bins, ops),
                    "always_open": np.ones(len(ops), dtype=bool),
                },
                hits,
                n,
            )
    out["distance"]["model"] = distance
    out["constant"]["model"] = {"overall": constant["overall"]}
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(config: Config, run_dir: Path, log=print) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    behaviours = list(config.behaviours or programs.train_behaviours())
    architecture = PolicyArchitecture(*config.architecture)
    model = StackPairPolicy(architecture, encoding=DIFFERENCE).to(config.train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rnd = H.RND(architecture, config.seed)
    procedure = Procedure(
        f"novelty-reduction s{config.seed} (cycle policy)",
        config.budget,
        config.temperature,
        config.n,
    )
    archive = LoopArchive()
    e_history: dict[int, dict] = {}
    last_error: dict[tuple[int, int], float] = {}
    rnd_seen: list = []
    reports = []
    for cycle in range(1, config.cycles + 1):
        clock = time.perf_counter()
        timings: dict[str, float] = {}
        last = [clock]

        def mark(phase: str, timings=timings, last=last) -> None:
            now = time.perf_counter()
            timings[phase] = round(now - last[0], 2)
            last[0] = now

        report: dict = {"cycle": cycle, "config": dataclasses.asdict(config)}
        report["propose"], proposal_log = propose_phase(
            archive, config, cycle, behaviours
        )
        mark("propose")
        rows, report["rows"] = policy_rows(archive, config)
        mark("rows")
        model.to(config.train_device)
        report["policy"] = train_epochs(model, optimizer, rows, config, cycle)
        mark("policy")
        frozen = StackPairPolicy(architecture, encoding=DIFFERENCE)
        frozen.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        scorer = PolicyScorer(frozen.eval(), "cpu")
        cycle_procedure = dataclasses.replace(
            procedure, policy=f"novelty-reduction s{config.seed} cycle {cycle}"
        )

        # calibration
        picks = selector.select(
            archive,
            cycle,
            e_history,
            stream(config, cycle, "select"),
            selector.Selector(size=config.calibration, max_cuts=config.max_cuts),
        )
        keys = [key for key, _ in picks]
        before = {key: archive.pairs[key].attempts for key in keys}
        visited_before = len(archive.states)
        hits, steps, made, touched = attempt(
            archive,
            scorer,
            keys,
            cycle_procedure,
            config.seed * 1000 + cycle,
            cycle,
            record=True,
        )
        mark("calibration_rollouts")
        report["calibration"] = {
            "v_emp_histogram": np.bincount(hits, minlength=config.n + 1).tolist(),
            "pairs": len(keys),
            "rules": dict(Counter(rule for _, rule in picks)),
            "unique_pairs": len(set(keys)),
            "share_on_pairs_attempted_3_plus": float(
                np.mean([before[k] + 1 >= 3 for k in keys]) if keys else 0.0
            ),
            "mean_v_emp": float(hits.mean() / config.n) if keys else None,
            "mean_steps": float(steps.mean()) if keys else None,
            "rollout_made": made,
            "nodes_added": len(archive.states) - visited_before,
        }
        features = H.pair_features(
            scorer, [(archive.states[s], archive.states[t]) for s, t in keys]
        )
        v_head, e_head, facts, v_oof = H.fit_heads(
            features, hits, config.n, config.seed + cycle
        )
        report["calibration"]["heads"] = facts
        # Measured progress: |V_hat - V_emp| (out-of-fold) on re-attempted pairs,
        # against the same pair's error when it was last attempted.
        error_now = np.abs(v_oof - hits / config.n)
        again = [
            (key, error_now[i])
            for i, (key, rule) in enumerate(picks)
            if rule == selector.REATTEMPT and key in last_error
        ]
        measured = {
            "reattempted": len(again),
            "measured_error_drop_mean": float(
                np.mean([last_error[key] - err for key, err in again])
            )
            if again
            else None,
        }
        last_error.update(zip(keys, error_now.tolist(), strict=True))
        mark("heads")
        # RND: this phase's visited nodes plus a replay of earlier ones
        visited_states = [archive.states[i] for i in sorted(touched)]
        replay = random.Random(f"{config.seed}:{cycle}:rnd").sample(
            rnd_seen, min(len(rnd_seen), config.rnd_replay)
        )
        report["rnd"] = {
            "train_loss": rnd.train(
                visited_states + replay, config.rnd_epochs, seed=cycle
            )
        }
        rnd_seen.extend(visited_states)
        reference = random.Random(f"{config.seed}:{cycle}:rndref").sample(
            archive.states, min(len(archive.states), 5000)
        )
        rnd.normalise(reference)
        well = [
            s for s, v in zip(archive.states, archive.visits, strict=True) if v >= 3
        ]
        never = [
            s for s, v in zip(archive.states, archive.visits, strict=True) if v == 0
        ]
        sample = random.Random(f"{config.seed}:{cycle}:rndcal")
        well = sample.sample(well, min(len(well), 5000))
        never = sample.sample(never, min(len(never), 5000))
        threshold = H.calibrate_threshold(rnd.novelty(well), rnd.novelty(never))
        report["rnd"].update(threshold)
        mark("rnd")
        # E on the tracked set, for predicted progress
        tracked_rng = stream(config, cycle, "tracked")
        tracked = {key for key, record in archive.pairs.items() if record.attempts > 0}
        fresh_keys = [
            k
            for k, r in archive.pairs.items()
            if r.d_cuts <= config.max_cuts and k not in tracked
        ]
        tracked |= set(
            tracked_rng.sample(fresh_keys, min(len(fresh_keys), config.tracked_new))
        )
        for old in e_history.values():
            tracked |= set(old)
        tracked_list = sorted(tracked)
        e_now = e_head.predict(
            H.pair_features(
                scorer,
                [(archive.states[s], archive.states[t]) for s, t in tracked_list],
            )
        )
        e_history[cycle] = dict(zip(tracked_list, e_now.tolist(), strict=True))
        if cycle - 1 in e_history:
            previous = e_history[cycle - 1]
            shared = [k for k in e_history[cycle] if k in previous]
            measured["predicted_e_drop_mean"] = (
                float(  # predicted progress
                    np.mean([previous[k] - e_history[cycle][k] for k in shared])
                )
                if shared
                else None
            )
        report["progress"] = measured

        mark("tracked_e")
        # test: never archived
        before_test = (len(archive.states), len(archive.pairs), sum(archive.visits))
        final = cycle == config.cycles and config.test_fresh_final > 0
        fresh = programs.fresh_pairs(
            config.test_fresh_final if final else config.test_fresh,
            stream(config, cycle, "F3"),
            behaviours,
            max_cuts=config.max_cuts,
        )
        test_rng = stream(config, cycle, "test")
        pool = sorted(
            k for k, r in archive.pairs.items() if r.d_cuts <= config.max_cuts
        )
        archive_keys = test_rng.sample(pool, min(len(pool), config.test_archive))
        f3_jobs = [(p.s, p.t) for p in fresh]
        a_jobs = [(archive.states[s], archive.states[t]) for s, t in archive_keys]
        f3_hits = np.zeros(len(fresh), dtype=np.int64)
        f3_ends = Counter()
        for walker in walk(
            scorer, f3_jobs, cycle_procedure, config.seed * 1000 + 500 + cycle
        ):
            f3_hits[walker.owner] += walker.hit
            f3_ends[walker.end] += 1
        a_hits, _, a_made, _ = attempt(
            archive,
            scorer,
            archive_keys,
            cycle_procedure,
            config.seed * 1000 + 700 + cycle,
            cycle,
            record=False,
        )
        tests = {}
        blocks = {}
        for source, jobs, hits_now, strata, ops in (
            (
                "F3",
                f3_jobs,
                f3_hits,
                [f"F3:{p.d_cuts}" for p in fresh],
                [p.d_ops for p in fresh],
            ),
            (
                "archive",
                a_jobs,
                a_hits,
                [
                    f"{archive.pairs[k].origin}:{archive.pairs[k].d_cuts}"
                    for k in archive_keys
                ],
                [archive.pairs[k].d_ops for k in archive_keys],
            ),
        ):
            feats = H.pair_features(scorer, jobs)
            v_hat, e_hat = v_head.predict(feats), e_head.predict(feats)
            n_pair = (
                np.maximum(
                    rnd.novelty([s for s, _ in jobs]), rnd.novelty([t for _, t in jobs])
                )
                if jobs
                else np.zeros(0)
            )
            n_open = (
                n_pair <= threshold["n0"]
                if threshold["n0"] is not None
                else np.zeros(len(jobs), dtype=bool)
            )
            gates = {f"rnd_e0_{e0}": n_open & (e_hat < e0) for e0 in E0S}
            if source == "archive":
                counted = np.array(
                    [
                        archive.visits[s] >= 3 and archive.visits[t] >= 3
                        for s, t in archive_keys
                    ],
                    dtype=bool,
                )
                gates |= {f"count_e0_{e0}": counted & (e_hat < e0) for e0 in E0S}
            blocks[source] = score_block(strata, v_hat, gates, hits_now, config.n)
            tests[source] = (strata, np.array(ops), hits_now)
        report["test"] = {
            "heads": blocks,
            "baselines": baselines(
                np.array([archive.pairs[k].d_ops for k in keys]), hits, config.n, tests
            ),
            "fresh_rollout_success": float(f3_hits.mean() / config.n)
            if len(fresh)
            else None,
            "archive_rollout_success": float(a_hits.mean() / config.n)
            if len(archive_keys)
            else None,
            "fresh_pairs": len(fresh),
            "fresh_walker_ends": dict(f3_ends),
            "archive_walker_ends": {
                k: v for k, v in a_made.items() if k.startswith("end_")
            },
            "fresh_v_emp_histogram": np.bincount(
                f3_hits, minlength=config.n + 1
            ).tolist(),
            "archive_v_emp_histogram": np.bincount(
                a_hits, minlength=config.n + 1
            ).tolist(),
            "archive_pairs": len(archive_keys),
        }
        mark("test")
        report["timings"] = timings
        report["archive"] = archive.summary()
        report["test_left_archive_unchanged"] = before_test == (
            len(archive.states),
            len(archive.pairs),
            sum(archive.visits),
        )
        report["relaxations_this_cycle"] = sum(
            r["cycle"] == cycle for r in archive.relaxations
        )
        report["procedure"] = cycle_procedure.row()["rollout"]
        report["seconds"] = time.perf_counter() - clock
        report["finished"] = datetime.now(PDT).isoformat(timespec="seconds")

        out = run_dir / f"cycle-{cycle}"
        out.mkdir(exist_ok=True)
        archive.save(out / "archive.pkl.gz")
        report["archive_snapshot"] = str((out / "archive.pkl.gz").resolve())
        save_policy(
            out / "policy",
            frozen,
            trainer_state={"cycle": cycle, **report["policy"], **report["rows"]},
            manifest={
                "issue": "novelty-reduction (#171)",
                "config": dataclasses.asdict(config),
            },
            wandb_url=None,
        )
        with gzip.open(out / "proposals.jsonl.gz", "wt") as handle:
            handle.writelines(json.dumps(entry) + "\n" for entry in proposal_log)
        with gzip.open(out / "calibration.jsonl.gz", "wt") as handle:
            for (key, rule), h, st in zip(
                picks, hits.tolist(), steps.tolist(), strict=True
            ):
                record = archive.pairs[key]
                handle.write(
                    json.dumps(
                        {
                            "s": key[0],
                            "t": key[1],
                            "rule": rule,
                            "hits": h,
                            "mean_steps": st,
                            "d_ops": record.d_ops,
                            "d_cuts": record.d_cuts,
                            "origin": record.origin,
                            "attempts": record.attempts,
                        }
                    )
                    + "\n"
                )
        (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
        log(
            f"cycle {cycle}: archive {report['archive']}, rows {report['rows']}, "
            f"cal V_emp {report['calibration']['mean_v_emp']}, "
            f"F3 success {report['test']['fresh_rollout_success']}, {report['seconds']:.0f}s"
        )
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    for field in dataclasses.fields(Config):
        if field.name in ("architecture", "behaviours"):
            continue
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=type(field.default),
            default=field.default,
        )
    args = parser.parse_args(argv)
    values = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(Config)
        if f.name not in ("architecture", "behaviours")
    }
    run(Config(**values), args.run_dir, log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
