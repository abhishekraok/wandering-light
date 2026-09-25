"""novelty-reduction (#171): the phase runner, as registered for M2.

Heads are P (P(reach t within B)), E (P(the frozen P fails a fresh n-rollout
Wilson check)) and N (RND; the count table drives only the selector).

**Cycle 0, bootstrap** (F9): proposals from fresh inputs, then imitation in
blocks until fresh-proposal success clears a floor or the budget runs out.
Below the floor the run writes its bootstrap diagnosis and stops.

**Cycles 1..C:**

1. propose ``P`` goals (random programs at the grammar level) from fresh
   inputs or archived nodes; every closed state and pair is archived;
2. policy phase: imitate the shortest known path of every eligible proposal
   pair, plus rollout-made pairs up to the same row count, sampled across
   (distance, combinator) strata a lineage at a time (F10);
3. paired check: the previous and the current policy on the same fresh pairs,
   never archived (F12 (d));
4. calibration: the selector's pairs (novelty-uniform + 10% re-attempts,
   F11), ``n`` rollouts each with distances frozen before the attempt (F5);
   three lineage partitions: P on the first, E on the second, gates on the
   third, nothing refit (F6); RND trained and its threshold set;
5. test: fresh F3 pairs and archive pairs, never archived, every arm and
   control scored on the same rollouts (F1-F4, F8), under a wall-clock cap.

**M3 switches** (each default reproduces M2; replay needs the defaults):

- ``heads="ridge"``: M2b's ridge logistic probes replace the MLP P and E
  (P on partition 0, E on partition 1 against the ridge P's Wilson misses,
  lambda by inner lineage CV); distance is fit on every origin; the models
  are saved per cycle as ``models.json``.
- ``gate="e_set"``: E alone gates, at e0 set on partition 2 each cycle
  (``probes.set_threshold``: the largest gate-partition E_hat with at least
  ``gate_min_open`` open pairs at Wilson precision ``gate_precision``,
  closed if none), beside the {0.1, 0.2, 0.3} sweep. RND and count novelty
  go per F3 pair into ``test_rows.jsonl.gz``, never gating. Certified fresh
  pairs, and so executions per certified pair, are read at that e0.
- ``selector="e_ranked"``: from cycle 2, the previous cycle's E (on its own
  trunk) ranks up to ``e_candidates`` uncertified archive pairs; calibration
  keeps the 10% re-attempts and takes the rest highest E_hat first, and the
  rollout-made training rows take ranked pairs first. An archive pair is
  certified once a calibration phase finds it gate-open and a Wilson hit.
- ``widen_levels``: proposals, the paired check and (unless ``test_level``)
  F3 use the current level; after the paired check the next cycle moves one
  level on when fresh success on the level's widest body stratum clears
  ``widen_threshold`` over at least ``widen_min_pairs`` pairs. Levels past
  (4, 3, none) are refused.

Hard stops (F12): an exception; after bootstrap, calibration labels nearly all
zero ("uninformative"); a cycle that archives no new proposal pair; current
fresh success below ``collapse_ratio`` of the previous policy's on the same
pairs. Everything else is a diagnostic, read only with ``min_stratum`` pairs.

Run::

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
from collections import Counter, defaultdict
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
from experiments.novelty_reduction import (  # noqa: E402
    canon,
    probes,
    programs,
)
from experiments.novelty_reduction import evaluate as ev  # noqa: E402
from experiments.novelty_reduction import heads as H  # noqa: E402, N812
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
HEADLINE_E0 = 0.2
ARCHITECTURE_1M = (128, 320, 640)  # compute_optimal_mark28.ARMS["1m"]


@dataclass(frozen=True, slots=True)
class Config:
    seed: int = 0
    cycles: int = 3
    proposals: int = 1000
    start_archived_share: float = 0.5  # proposals from archived nodes, cycle 2 on
    grammar_level: str = "4,3,none"  # max body atoms, max segments, nesting
    calibration: int = 500
    test_fresh: int = 250
    test_fresh_final: int = 1000  # the last cycle's F3 pool
    test_archive: int = 250
    paired_check: int = 250
    tracked_new: int = 5000
    max_cuts: int = 1
    epochs: int = 5
    row_cap: int = 300_000
    learning_rate: float = 1.25e-4
    batch: int = 256
    architecture: tuple[int, int, int] = ARCHITECTURE_1M
    rnd_replay: int = 20_000
    rnd_epochs: int = 1
    rnd_cohort: int = 500
    budget: int = 16
    temperature: float = 1.0
    n: int = 8
    train_device: str = "cuda"
    threads: int = 1
    behaviours: tuple[str, ...] | None = None  # None: the v4 train behaviours
    bootstrap_proposals: int = 3000
    bootstrap_block_epochs: int = 5
    bootstrap_max_epochs: int = 30
    bootstrap_check: int = 250
    floor_success: float = 0.05  # walker success on fresh proposal pairs
    floor_pairs: float = 0.20  # share of those pairs with any hit
    uninformative_zero_share: float = 0.98
    collapse_ratio: float = 0.5
    min_stratum: int = 100
    final_eval_seconds: float = 3600.0
    # M3 switches; each default reproduces M2
    heads: str = "mlp"  # "ridge": M2b's ridge probes for P and E (probes.py)
    ridge_lambdas: tuple[float, ...] = probes.LAMBDAS
    ridge_folds: int = 5
    gate: str = "rnd_e"  # "e_set": E only, e0 set on partition 2 each cycle
    gate_min_open: int = 10
    gate_precision: float = 0.8
    selector: str = "uniform"  # "e_ranked": highest previous-cycle E_hat first
    e_candidates: int = 20_000  # uncertified archive pairs scored for the ranking
    widen_levels: tuple[str, ...] = ()  # () keeps grammar_level fixed
    widen_threshold: float = 0.5  # paired-check success on the widest body stratum
    widen_min_pairs: int = 20
    test_level: str = ""  # F3's grammar level; "" follows the current level


GATES = ("rnd_e", "e_set")
HEADS = ("mlp", "ridge")
SELECTORS = ("uniform", "e_ranked")
CEILING = programs.Grammar.parse("4,3,none")


def check_config(config: Config) -> None:
    """Unknown switches and levels past the M2 ceiling fail before any work."""
    if config.heads not in HEADS or config.gate not in GATES:
        raise ValueError(f"heads {config.heads!r} / gate {config.gate!r}")
    if config.selector not in SELECTORS:
        raise ValueError(f"selector {config.selector!r}")
    for level in (*config.widen_levels, config.test_level or config.grammar_level):
        grammar = programs.Grammar.parse(level)
        if grammar.max_segments > CEILING.max_segments:
            raise ValueError(f"level {level} is past the ceiling {CEILING.level}")


def stream(config: Config, cycle: int, role: str) -> random.Random:
    """One rng per (seed, cycle, role): fresh pools never share a draw with the loop."""
    return random.Random(f"{config.seed}:{cycle}:{role}")


def now_pdt() -> str:
    return datetime.now(PDT).isoformat(timespec="seconds")


class Clock:
    def __init__(self) -> None:
        self.last = time.perf_counter()
        self.phases: dict[str, float] = {}

    def mark(self, phase: str) -> None:
        now = time.perf_counter()
        self.phases[phase] = round(self.phases.get(phase, 0.0) + now - self.last, 2)
        self.last = now


# ---------------------------------------------------------------------------
# Proposals and the policy phase
# ---------------------------------------------------------------------------


def propose_phase(
    archive: LoopArchive,
    config: Config,
    cycle: int,
    behaviours,
    count: int,
    *,
    archived_share: float,
    level: str | None = None,
    admit=None,
) -> tuple[dict, list[dict]]:
    """Counts, and one log row per proposal (start kind, body atoms, segments).

    ``admit(states, lineage)``, when given, can refuse a proposal before it is
    archived (M3's test-panel quarantine); refusals are counted.
    """
    rng = stream(config, cycle, "propose")
    grammar = programs.Grammar.parse(level or config.grammar_level)
    made = Counter()
    log = []
    for _ in range(count):
        if archive.states and rng.random() < archived_share:
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
            "body_atoms": programs.body_atoms(proposal.ops),
            "segments": len(proposal.cuts),
        }
        log.append(entry)
        if not proposal.cuts:
            made["empty"] += 1
            continue
        if admit is not None and not admit(proposal.states, route[0]):
            made["quarantined"] += 1
            entry["quarantined"] = True
            continue
        added, relaxed = archive.add_path(
            proposal.states, proposal.ops, PROPOSAL, cycle, start_origin=route
        )
        entry["pairs_added"], entry["relaxations"] = added, relaxed
        made["pairs"] += added
        made["relaxations"] += relaxed
        made["segments"] += len(proposal.cuts)
    return dict(made), log


def last_combinator(path) -> str:
    return next((op for op in reversed(path) if op in programs.RUNS), "none")


def policy_rows(
    archive: LoopArchive,
    config: Config,
    cycle: int,
    previous_paths: dict,
    ranked: dict | None = None,
    imitate: str = "all",
) -> tuple[list[Row], dict, dict]:
    """Imitation rows on shortest known paths (F10).

    Every eligible proposal pair is in. Rollout-made pairs fill at most the
    same number of rows, drawn round-robin over (d_ops, last combinator)
    strata and, within a stratum, a lineage at a time, ties broken by key.
    With ``ranked`` (E_hat per uncertified pair, the M3 E-ranked selector),
    rollout-made pairs in it go first, highest E_hat first, and the
    round-robin fills what budget is left. With ``imitate="solved"`` (M3's
    adaptive-solved branch) no proposal-made path is imitated: the pairs are
    those whose shortest known path a rollout wrote (``solver_found``), and
    they fill twice the proposal-row budget, the most the default can train.
    Also returns the paths used, for the next cycle's turnover.
    """
    eligible = sorted(
        (r for r in archive.pairs.values() if r.witness_d_cuts <= config.max_cuts),
        key=lambda r: (r.s, r.t),
    )
    made = [r for r in eligible if r.origin == PROPOSAL]
    walked = [r for r in eligible if r.origin == ROLLOUT]
    budget = sum(r.d_ops for r in made)
    if imitate == "solved":
        solver = solver_found(archive)
        walked = [r for r in eligible if (r.s, r.t) in solver]
        made, budget = [], 2 * budget
    elif imitate != "all":
        raise ValueError(f"imitate {imitate!r}")
    rng = stream(config, cycle, "rows")
    available = len(walked)
    kept = []
    if ranked:
        first = sorted(
            (r for r in walked if (r.s, r.t) in ranked),
            key=lambda r: (-ranked[(r.s, r.t)], r.s, r.t),
        )
        for record in first:
            if record.d_ops <= budget:
                kept.append(record)
                budget -= record.d_ops
        walked = [r for r in walked if (r.s, r.t) not in ranked]
    e_ranked_pairs = len(kept)
    strata: dict[tuple, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for record in walked:
        key = (record.d_ops, last_combinator(record.path))
        strata[key][archive.lineage(record.s)].append(record)
    order = sorted(strata)
    rng.shuffle(order)
    queues = {}
    for key in order:
        lineages = sorted(strata[key], key=str)
        rng.shuffle(lineages)
        queues[key] = [list(strata[key][lineage]) for lineage in lineages]
    while budget > 0 and any(queues.values()):
        for key in order:
            lanes = queues[key]
            if not lanes or budget <= 0:
                continue
            lane = lanes.pop(0)
            record = lane.pop(0)
            if lane:
                lanes.append(lane)
            if record.d_ops <= budget:
                kept.append(record)
                budget -= record.d_ops
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
    if len(rows) > config.row_cap:
        rows = rng.sample(rows, config.row_cap)
    paths = {(r.s, r.t): r.path for r in [*made, *kept]}
    shared = [k for k in paths if k in previous_paths]
    turnover = (
        sum(paths[k] != previous_paths[k] for k in shared) / len(shared)
        if shared
        else None
    )
    facts = {
        "proposal_rows": sum(r.d_ops for r in made),
        "rollout_rows": sum(r.d_ops for r in kept),
        "proposal_pairs": len(made),
        "rollout_pairs": len(kept),
        "rollout_pairs_available": available,
        "rollout_pairs_e_ranked": e_ranked_pairs,
        "rows_trained": len(rows),
        "path_turnover": turnover,
    }
    return rows, facts, paths


def solver_found(archive: LoopArchive) -> set:
    """Pairs whose current shortest path a rollout wrote: first archived by one
    and not since relaxed by a proposal, or last relaxed by one."""
    last = {(r["s"], r["t"]): r["by"] for r in archive.relaxations}
    return {
        key
        for key, record in archive.pairs.items()
        if last.get(key, record.origin) == ROLLOUT
    }


def train_epochs(
    model, optimizer, rows: list[Row], config: Config, epochs: int, tag: str
) -> dict:
    device = next(model.parameters()).device
    rng = random.Random(f"{config.seed}:{tag}:train")
    model.train()
    history = []
    steps = 0
    for _ in range(epochs):
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
    return {"train_cross_entropy": history, "epochs": epochs, "steps": steps}


def frozen_scorer(model, architecture) -> PolicyScorer:
    frozen = StackPairPolicy(architecture, encoding=DIFFERENCE)
    frozen.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    return PolicyScorer(frozen.eval(), "cpu")


# ---------------------------------------------------------------------------
# Pools: the pairs a phase judges
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Pool:
    """Pairs with their frozen witness distances, lineages and strata labels."""

    source: str
    s: list
    t: list
    path: list  # witness path, frozen when the pool was drawn
    lineage: list
    keys: list  # archive keys, or None for fresh pairs
    origin: list

    def __len__(self) -> int:
        return len(self.s)

    @property
    def d_ops(self) -> np.ndarray:
        return np.array([len(p) for p in self.path])

    def jobs(self) -> list[tuple]:
        return list(zip(self.s, self.t, strict=True))

    def subset(self, index) -> Pool:
        return Pool(
            self.source,
            *(
                [getattr(self, f)[i] for i in index]
                for f in ("s", "t", "path", "lineage", "keys", "origin")
            ),
        )


def fresh_pool(pairs: list[sampler.Pair], source: str) -> Pool:
    return Pool(
        source,
        [p.s for p in pairs],
        [p.t for p in pairs],
        [p.t_program[len(p.s_program) :] for p in pairs],
        [p.inputs for p in pairs],
        [None] * len(pairs),
        [source] * len(pairs),
    )


def archive_pool(archive: LoopArchive, keys, source: str) -> Pool:
    records = [archive.pairs[k] for k in keys]
    return Pool(
        source,
        [archive.states[r.s] for r in records],
        [archive.states[r.t] for r in records],
        [r.witness_path for r in records],
        [archive.lineage(r.s) for r in records],
        list(keys),
        [r.origin for r in records],
    )


def roll(
    scorer, pool: Pool, procedure: Procedure, seed: int, deadline: float | None = None
):
    """Hits per pair, steps-to-go, walker ends and walks; pairs past ``deadline`` unrolled."""
    hits = np.zeros(len(pool), dtype=np.int64)
    steps = np.zeros(len(pool))
    ends = Counter()
    walks_all = []
    done = 0
    chunk = 100
    for start in range(0, len(pool), chunk):
        if deadline is not None and time.perf_counter() > deadline:
            break
        jobs = pool.jobs()[start : start + chunk]
        for walker in walk(scorer, jobs, procedure, seed + start):
            walker.owner += start
            ends[walker.end] += 1
            hits[walker.owner] += walker.hit
            steps[walker.owner] += len(walker.ops) if walker.hit else procedure.budget
            walks_all.append(walker)
        done = start + len(jobs)
    return hits[:done], steps[:done] / procedure.n, dict(ends), walks_all, done


def archive_walks(
    archive: LoopArchive, keys, walks, cycle: int, admit=None
) -> tuple[dict, set]:
    """Archive calibration walks (hindsight), count visits; the start is not "through".

    ``admit(states, lineage)`` as in ``propose_phase``: a refused walk is not
    archived and its visits are not counted.
    """
    made = Counter()
    touched: set[int] = set()
    for walker in walks:
        s, _t = keys[walker.owner]
        if admit is not None and not admit(walker.states, archive.lineage(s)):
            made["quarantined"] += 1
            continue
        added, relaxed = archive.add_path(
            walker.states, walker.ops, ROLLOUT, cycle, start_origin=archive.origins[s]
        )
        made["pairs"] += added
        made["relaxations"] += relaxed
        through = {archive.index[x] for x in walker.states[1:] if x in archive.index}
        for node in through:
            archive.visits[node] += 1
        touched |= through
    for s, t in keys:
        archive.pairs[(s, t)].attempts += 1
        archive.visits[s] += 1
        archive.visits[t] += 1
        touched |= {s, t}
    return dict(made), touched


def success(hits: np.ndarray, n: int) -> dict:
    return {
        "pairs": len(hits),
        "walker_success": float(hits.mean() / n) if len(hits) else None,
        "pairs_with_hit": float((hits > 0).mean()) if len(hits) else None,
        "histogram": np.bincount(hits, minlength=n + 1).tolist(),
    }


def by_label(values: np.ndarray, labels, n: int) -> dict:
    out = {}
    labels = np.asarray(labels)
    for label in sorted(set(labels.tolist())):
        at = labels == label
        out[str(label)] = {
            "pairs": int(at.sum()),
            "walker_success": float(values[at].mean() / n),
        }
    return out


# ---------------------------------------------------------------------------
# Models fitted on one calibration phase
# ---------------------------------------------------------------------------


@dataclass
class Models:
    p_head: H.Head | probes.Probe
    e_head: H.Head | probes.Probe
    n0: float | None
    constant: float
    distance: dict
    cheap: dict
    wilson_constant: float
    bins: dict  # arm -> open d_ops values, fitted on the third partition
    bin_rates: dict  # arm -> {d_ops: hit rate on the third partition}
    facts: dict
    e0_set: float | None = None  # M3: E's threshold, set on the third partition
    plain: dict | None = None  # M3: the ridge models as plain JSON


def fit_ridge_heads(features, hits, n, parts, lineage, seed, config: Config):
    """M2b's probes: ridge P on part 0, ridge E on part 1 against P's Wilson misses."""
    first, second = parts == 0, parts == 1
    groups = [str(g) for g in lineage]
    at = lambda mask: [g for g, m in zip(groups, mask, strict=True) if m]  # noqa: E731
    fit = {"lambdas": config.ridge_lambdas, "folds": config.ridge_folds}
    p = probes.ridge(features[first], hits[first], n, at(first), seed, **fit)
    miss = ~scoring.hit_wilson(
        ev.predict_logistic(p, features[second]), hits[second], n
    )
    e = probes.ridge(features[second], miss.astype(int), 1, at(second), seed + 1, **fit)
    facts = {
        "pairs_by_part": np.bincount(parts, minlength=3).tolist(),
        "e_target_rate": float(miss.mean()) if len(miss) else None,
        "e_target_rate_by_hits": {
            str(h): {
                "pairs": int((hits[second] == h).sum()),
                "rate": float(miss[hits[second] == h].mean()),
            }
            for h in sorted(set(hits[second].tolist()))
        },
        "p_lambda": p["lambda"],
        "p_cv": p["cv"],
        "e_lambda": e["lambda"],
        "e_cv": e["cv"],
    }
    return probes.Probe(p), probes.Probe(e), facts


def fit_models(
    scorer, pool: Pool, hits, n, parts, seed, n0, config: Config | None = None
) -> tuple[Models, np.ndarray]:
    """M2's heads and controls; with ``config`` its M3 switches (heads, gate) apply."""
    features = H.pair_features(scorer, pool.jobs())
    ridge = config is not None and config.heads == "ridge"
    if ridge:
        p_head, e_head, facts = fit_ridge_heads(
            features, hits, n, parts, pool.lineage, seed, config
        )
    else:
        p_head, e_head, facts = H.fit_partitioned(features, hits, n, parts, seed)
    first, third = parts == 0, parts == 2
    rate = hits / n
    constant = float(rate[first].mean()) if first.any() else 0.0
    # M2 fit distance on proposal pairs only; M3 (as M2b) on every origin
    proposal_first = first & (
        np.ones(len(pool), dtype=bool) if ridge else np.array(pool.origin) == PROPOSAL
    )
    ops = pool.d_ops
    distance = scoring.fit_distance(ops[proposal_first], hits[proposal_first], n)
    cheap_x = np.array(
        [
            canon.cheap_features(s, t, p)
            for s, t, p in zip(pool.s, pool.t, pool.path, strict=True)
        ]
    )
    cheap = ev.fit_logistic(cheap_x[first], hits[first], n)
    rule = scoring.Rule()
    predictions = {
        "constant": np.full(len(pool), constant),
        "distance": scoring.predict_distance(distance, ops),
        "cheap": ev.predict_logistic(cheap, cheap_x),
    }
    bins, rates = {}, {}
    for arm, p_hat in predictions.items():
        bins[arm] = scoring.fit_gate(ops[third], p_hat[third], hits[third], n, rule)
        hit = scoring.hit_wilson(p_hat[third], hits[third], n)
        rates[arm] = {
            int(d): float(hit[ops[third] == d].mean()) for d in set(ops[third].tolist())
        }
    models = Models(
        p_head,
        e_head,
        n0,
        constant,
        distance,
        cheap,
        ev.wilson_constant(hits[first], n),
        bins,
        rates,
        facts,
    )
    if config is not None and config.gate == "e_set":
        threshold = probes.set_threshold(
            e_head.predict(features[third]),
            p_head.predict(features[third]),
            hits[third],
            n,
            config.gate_min_open,
            config.gate_precision,
        )
        models.e0_set = threshold["e0"]
        facts["e_set"] = threshold
    if ridge:
        models.plain = {
            "p": probes.plain_logistic(
                p_head.model, probes.trunk_names(features.shape[1])
            ),
            "e": probes.plain_logistic(
                e_head.model, probes.trunk_names(features.shape[1])
            ),
            "cheap": probes.plain_logistic(cheap, canon.CHEAP_FEATURES),
            "distance": distance,
            "constant": constant,
            "wilson_constant": models.wilson_constant,
            "bins": bins,
            "bin_rates": {
                a: {str(d): r for d, r in v.items()} for a, v in rates.items()
            },
            "e0_set": models.e0_set,
        }
    return models, features


def score_pool(
    models: Models,
    scorer,
    rnd: H.RND,
    pool: Pool,
    hits: np.ndarray,
    n: int,
    stratifications: dict,
    counted: np.ndarray | None = None,
    *,
    gate: str = "rnd_e",
    count: np.ndarray | None = None,
    rows_out: list | None = None,
) -> dict:
    """Every arm and control on the same rollouts of one pool (F1-F3).

    ``gate="rnd_e"`` is M2's (RND and E gate the learned arms). ``"e_set"``
    is M3's: E alone gates, at the e0 sweep and at the partition-set
    ``models.e0_set`` (closed when None); RND and count novelty are logged
    per pair into ``rows_out``, never gating.
    """
    if len(pool) == 0:  # e.g. the final evaluation's deadline passed first
        return {}, 0
    features = H.pair_features(scorer, pool.jobs())
    p_hat = models.p_head.predict(features)
    e_hat = models.e_head.predict(features)
    if len(pool) and models.n0 is not None:
        n_pair = np.maximum(rnd.novelty(pool.s), rnd.novelty(pool.t))
        rnd_open = n_pair <= models.n0
    else:
        n_pair = np.zeros(len(pool))
        rnd_open = np.zeros(len(pool), dtype=bool)
    ops = pool.d_ops
    cheap_x = np.array(
        [
            canon.cheap_features(s, t, p)
            for s, t, p in zip(pool.s, pool.t, pool.path, strict=True)
        ]
    ).reshape(len(pool), -1)
    ones = np.ones(len(pool), dtype=bool)
    p_distance = scoring.predict_distance(models.distance, ops)
    p_cheap = ev.predict_logistic(models.cheap, cheap_x) if len(pool) else np.zeros(0)

    e_only = gate == "e_set"

    def learned(e_mask_from):
        if e_only:
            set_open = (
                e_mask_from < models.e0_set
                if models.e0_set is not None
                else np.zeros(len(pool), dtype=bool)
            )
            return {"e_set": set_open, **{f"e0_{e0}": e_mask_from < e0 for e0 in E0S}}
        return {f"e0_{e0}": rnd_open & (e_mask_from < e0) for e0 in E0S}

    learned_confidence = 1 - e_hat if e_only else np.where(rnd_open, 1 - e_hat, -1.0)

    def binned(arm):
        return {"bins": scoring.apply_gate(models.bins[arm], ops), "open": ones}

    def bin_confidence(arm):
        return np.array([models.bin_rates[arm].get(int(d), 0.0) for d in ops])

    arms = {
        "full": (p_hat, learned(e_hat), learned_confidence),
        "p_open": (p_hat, {"open": ones}, None),
        "p_e": (p_hat, {f"e0_{e0}": e_hat < e0 for e0 in E0S}, 1 - e_hat),
        "distance_learned_gate": (p_distance, learned(e_hat), learned_confidence),
        "cheap": (p_cheap, binned("cheap"), bin_confidence("cheap")),
        "constant": (
            np.full(len(pool), models.constant),
            binned("constant"),
            bin_confidence("constant"),
        ),
        "distance": (p_distance, binned("distance"), bin_confidence("distance")),
        "zero": (np.zeros(len(pool)), {"open": ones}, None),
        "wilson_constant": (
            np.full(len(pool), models.wilson_constant),
            {"open": ones},
            None,
        ),
    }
    if counted is not None and not e_only:
        arms["full_count_gate"] = (
            p_hat,
            {f"e0_{e0}": counted & (e_hat < e0) for e0 in E0S},
            np.where(counted, 1 - e_hat, -1.0),
        )
    out = {}
    for name, (predicted, gates, confidence) in arms.items():
        out[name] = ev.arm_report(
            stratifications, predicted, gates, hits, n, pool.lineage, confidence
        )
    out["log_loss_gain_over"] = {
        control: ev.paired_gain(p_hat, arms[control][0], hits, n, pool.lineage)
        for control in ("distance", "constant", "cheap", "zero", "wilson_constant")
    }
    out["rnd_pass_rate"] = float(rnd_open.mean()) if len(pool) else None
    out["n_pair_mean"] = float(n_pair.mean()) if len(pool) else None
    hit = scoring.hit_wilson(p_hat, hits, n)
    if e_only:
        headline = arms["full"][1]["e_set"] & hit
        out["e0_set"] = models.e0_set
    else:
        headline = rnd_open & (e_hat < HEADLINE_E0) & hit
    if rows_out is not None:
        for i in range(len(pool)):
            rows_out.append(
                {
                    "lineage": str(pool.lineage[i]),
                    "origin": pool.origin[i],
                    "d_ops": int(ops[i]),
                    **{k: v[i] for k, v in stratifications.items() if k != "all"},
                    "hits": int(hits[i]),
                    "p_hat": float(p_hat[i]),
                    "e_hat": float(e_hat[i]),
                    "wilson_hit": bool(hit[i]),
                    "certified": bool(headline[i]),
                    "rnd": float(n_pair[i]),
                    "rnd_open": bool(rnd_open[i]),
                    "count_novelty": float(count[i]) if count is not None else None,
                    "counted": bool(counted[i]) if counted is not None else None,
                }
            )
    return out, int(headline.sum())


def diagnostics(report: dict, config: Config) -> dict:
    """The F12 monitors, each read only with ``min_stratum`` pairs behind it."""

    def headline(source: str, arm: str) -> float | None:
        if arm not in report["test"][source]["arms"]:
            return None  # an empty pool
        gates = report["test"][source]["arms"][arm]["by"]["all"]["gates"]
        gate = next((g for g in ("e_set", f"e0_{HEADLINE_E0}") if g in gates), "bins")
        return gates[gate]["all"]["wilson"]["certified_coverage"]

    fresh = report["test"]["F3"]
    held = report["test"]["archive"]
    paired = report["paired_check"]
    matched = {
        arm: fresh["arms"].get(arm, {}).get("matched_coverage")
        for arm in ("full", "p_e", "distance", "cheap", "constant")
    }
    return {
        "fresh_success": {
            "previous": paired["previous"]["walker_success"],
            "current": paired["current"]["walker_success"],
            "pairs": paired["current"]["pairs"],
            "sufficient": paired["current"]["pairs"] >= config.min_stratum,
        },
        "selector": {
            "unique_pairs": report["calibration"]["unique_pairs"],
            "share_on_pairs_attempted_3_plus": report["calibration"][
                "share_on_pairs_attempted_3_plus"
            ],
            "sufficient": report["calibration"]["pairs"] >= config.min_stratum,
        },
        "relaxations": report.get("relaxations_this_cycle"),
        "false_certification_at_matched_coverage_F3": matched,
        "headline_coverage": {
            "F3_full": headline("F3", "full"),
            "archive_full": headline("archive", "full"),
            "F3_distance_bins": headline("F3", "distance"),
            "sufficient": fresh["pairs"] >= config.min_stratum
            and held["pairs"] >= config.min_stratum,
        },
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Canonical:
    """Permutation-canonical keys of archived states and pairs, extended as it grows (F4)."""

    def __init__(self) -> None:
        self.stacks: list[tuple] = []
        self.states: set = set()
        self.pairs: set = set()
        self.seen_pairs = 0

    def update(self, archive: LoopArchive) -> None:
        for node in range(len(self.stacks), len(archive.states)):
            keys = tuple(canon.stack_key(stack) for stack in archive.states[node])
            self.stacks.append(keys)
            self.states.add(tuple(sorted(keys)))
        pairs = list(archive.pairs)
        for s, t in pairs[self.seen_pairs :]:
            self.pairs.add(
                tuple(sorted(zip(self.stacks[s], self.stacks[t], strict=True)))
            )
        self.seen_pairs = len(pairs)

    def classify(self, s, t) -> str:
        if canon.pair_key(s, t) in self.pairs:
            return "collision"
        if canon.state_key(s) in self.states or canon.state_key(t) in self.states:
            return "shared_end"
        return "unseen"


def stratify(pool: Pool, scorer, extra: dict | None = None) -> tuple[dict, dict]:
    """Stratum labels per pair, and the pool's truncation and encoded-collision facts (F8)."""
    trunc = np.array([canon.truncated(s, t) for s, t in pool.jobs()], dtype=bool)
    blocks = [scorer.encode(s, t) for s, t in pool.jobs()]
    collided = canon.encoded_collisions(
        blocks, [canon.pair_key(s, t) for s, t in pool.jobs()]
    )
    body = [programs.body_stratum(programs.body_atoms(p)) for p in pool.path]
    labels = {
        "all": ["all"] * len(pool),
        "truncated": ["truncated" if x else "untruncated" for x in trunc],
        "d_ops": [str(d) for d in pool.d_ops],
        "body": body,
        "segments": [str(programs.path_segments(p)) for p in pool.path],
        **(extra or {}),
    }
    facts = {
        "pairs": len(pool),
        "truncated_rate": float(trunc.mean()) if len(pool) else None,
        "encoded_collision_rate": float(collided.mean()) if len(pool) else None,
    }
    return labels, facts


def rank_by_e(
    archive: LoopArchive,
    models: Models,
    scorer,
    config: Config,
    cycle: int,
    certified: set,
) -> tuple[dict, dict]:
    """E_hat on up to ``e_candidates`` uncertified eligible pairs, for the E-ranked arm.

    ``models`` and ``scorer`` are the previous cycle's, so E reads the trunk it
    was fit on.
    """
    eligible = sorted(
        k
        for k, r in archive.pairs.items()
        if r.witness_d_cuts <= config.max_cuts and k not in certified
    )
    sample = stream(config, cycle, "e_rank").sample(
        eligible, min(len(eligible), config.e_candidates)
    )
    sample.sort()
    e_hat = models.e_head.predict(
        H.pair_features(
            scorer, [(archive.states[s], archive.states[t]) for s, t in sample]
        )
    )
    ranked = dict(zip(sample, e_hat.tolist(), strict=True))
    facts = {
        "eligible_uncertified": len(eligible),
        "scored": len(sample),
        "certified_archive_pairs": len(certified),
        "e_hat_quantiles": {
            str(q): float(np.quantile(e_hat, q)) for q in (0.05, 0.5, 0.95)
        }
        if len(e_hat)
        else {},
    }
    return ranked, facts


def body_record(pool: Pool, hits: np.ndarray, n: int) -> dict:
    """Per body stratum (diagnose.py's): pairs, pairs with a hit, walker success."""
    strata = np.array(
        [programs.body_stratum(programs.body_atoms(p)) for p in pool.path]
    )
    return {
        str(b): {
            "pairs": int((strata == b).sum()),
            "solved": int((hits[strata == b] > 0).sum()),
            "walker_success": float(hits[strata == b].mean() / n),
        }
        for b in sorted(set(strata.tolist()))
    }


def widen(level: str, levels: tuple, pool: Pool, hits, config: Config) -> dict:
    """Whether the paired check clears ``widen_threshold`` on the level's widest body stratum."""
    widest = programs.body_stratum(programs.Grammar.parse(level).max_body)
    record = body_record(pool, hits, config.n).get(
        widest, {"pairs": 0, "solved": 0, "walker_success": None}
    )
    at = levels.index(level)
    clears = (
        record["pairs"] > 0
        and record["pairs"] >= config.widen_min_pairs
        and record["walker_success"] >= config.widen_threshold
    )
    following = levels[at + 1] if clears and at + 1 < len(levels) else level
    return {
        "level": level,
        "widest_stratum": widest,
        **record,
        "threshold": config.widen_threshold,
        "min_pairs": config.widen_min_pairs,
        "cleared": bool(clears),
        "next_level": following,
    }


def run(config: Config, run_dir: Path, log=print) -> list[dict]:
    check_config(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    behaviours = list(config.behaviours or programs.train_behaviours())
    architecture = PolicyArchitecture(*config.architecture)
    model = StackPairPolicy(architecture, encoding=DIFFERENCE).to(config.train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rnd = H.RND(architecture, config.seed)
    base = Procedure(
        f"novelty-reduction s{config.seed}", config.budget, config.temperature, config.n
    )
    archive = LoopArchive()
    canonical = Canonical()
    reports: list[dict] = []
    rollouts_total = 0
    certified_total = 0
    e_history: dict[int, dict] = {}
    last_error: dict = {}
    rnd_seen: list = []
    previous_paths: dict = {}
    # M3 state: the grammar level, E-ranked selection, archive certification
    levels = tuple(config.widen_levels) or (config.grammar_level,)
    level = levels[0]
    previous_models: Models | None = None
    certified_keys: set = set()
    cohort_rng = stream(config, -1, "rnd_cohort")
    cohort = [
        p.t for p in programs.fresh_pairs(config.rnd_cohort, cohort_rng, behaviours)
    ]

    def save(report: dict, name: str, extra=None) -> None:
        out = run_dir / name
        out.mkdir(exist_ok=True)
        archive.save(out / "archive.pkl.gz")
        report["archive_snapshot"] = str((out / "archive.pkl.gz").resolve())
        report["finished"] = now_pdt()
        if extra:
            for filename, rows in extra.items():
                with gzip.open(out / filename, "wt") as handle:
                    handle.writelines(
                        json.dumps(row, default=str) + "\n" for row in rows
                    )
        (out / "report.json").write_text(json.dumps(report, indent=1, default=str))

    # ---- cycle 0: bootstrap (F9)
    clock = Clock()
    boot: dict = {
        "cycle": 0,
        "config": dataclasses.asdict(config),
        "started": now_pdt(),
    }
    boot["propose"], boot_log = propose_phase(
        archive,
        config,
        0,
        behaviours,
        config.bootstrap_proposals,
        archived_share=0.0,
        level=level,
    )
    rows, boot["rows"], previous_paths = policy_rows(archive, config, 0, {})
    check = programs.fresh_pairs(
        config.bootstrap_check,
        stream(config, 0, "bootcheck"),
        behaviours,
        grammar=programs.Grammar.parse(level),
    )
    check_pool = fresh_pool(check, "bootcheck")
    blocks = []
    floor_block = None
    for block in range(
        1, config.bootstrap_max_epochs // config.bootstrap_block_epochs + 1
    ):
        model.to(config.train_device)
        trained = train_epochs(
            model, optimizer, rows, config, config.bootstrap_block_epochs, f"0:{block}"
        )
        scorer = frozen_scorer(model, architecture)
        hits, _, ends, _, _ = roll(scorer, check_pool, base, config.seed * 1000 + block)
        rollouts_total += len(check_pool) * config.n
        entry = {
            "block": block,
            "epochs": block * config.bootstrap_block_epochs,
            **trained,
            **success(hits, config.n),
            "ends": ends,
            "by_d_ops": by_label(hits, check_pool.d_ops, config.n),
            "by_combinator": by_label(
                hits, [last_combinator(p) for p in check_pool.path], config.n
            ),
            "by_body": by_label(
                hits,
                [
                    programs.body_stratum(programs.body_atoms(p))
                    for p in check_pool.path
                ],
                config.n,
            ),
        }
        blocks.append(entry)
        log(
            f"bootstrap block {block}: success {entry['walker_success']:.3f}, "
            f"pairs with hit {entry['pairs_with_hit']:.3f}"
        )
        if (
            entry["walker_success"] >= config.floor_success
            and entry["pairs_with_hit"] >= config.floor_pairs
        ):
            floor_block = block
            break
    clock.mark("bootstrap")
    boot.update(blocks=blocks, floor_met_at_block=floor_block, timings=clock.phases)
    save(boot, "cycle-0", {"proposals.jsonl.gz": boot_log})
    save_policy(
        run_dir / "cycle-0" / "policy",
        frozen_scorer(model, architecture).model,
        trainer_state={"cycle": 0, "blocks": blocks},
        manifest={
            "issue": "novelty-reduction (#171)",
            "config": dataclasses.asdict(config),
        },
        wandb_url=None,
    )
    reports.append(boot)
    if floor_block is None:
        boot["stop"] = "bootstrap floor not met"
        save(boot, "cycle-0")
        log("stop: bootstrap floor not met")
        return reports
    previous_scorer = frozen_scorer(model, architecture)

    for cycle in range(1, config.cycles + 1):
        clock = Clock()
        report: dict = {
            "cycle": cycle,
            "config": dataclasses.asdict(config),
            "started": now_pdt(),
        }
        rollouts = 0
        procedure = dataclasses.replace(
            base, policy=f"novelty-reduction s{config.seed} cycle {cycle}"
        )
        report["procedure"] = procedure.row()["rollout"]

        # 1. propose
        report["propose"], proposal_log = propose_phase(
            archive,
            config,
            cycle,
            behaviours,
            config.proposals,
            archived_share=config.start_archived_share,
            level=level,
        )
        report["grammar_level"] = level
        clock.mark("propose")
        if report["propose"].get("pairs", 0) == 0:
            report["stop"] = "no new proposal pairs"
            save(report, f"cycle-{cycle}", {"proposals.jsonl.gz": proposal_log})
            reports.append(report)
            log(f"cycle {cycle} stop: {report['stop']}")
            break

        # 2. policy phase; the E-ranked arm ranks by the previous cycle's E
        ranked = None
        if config.selector == "e_ranked" and previous_models is not None:
            ranked, report["e_ranking"] = rank_by_e(
                archive, previous_models, previous_scorer, config, cycle, certified_keys
            )
            clock.mark("e_ranking")
        rows, report["rows"], previous_paths = policy_rows(
            archive,
            config,
            cycle,
            previous_paths,
            **({"ranked": ranked} if ranked else {}),
        )
        clock.mark("rows")
        model.to(config.train_device)
        report["policy"] = train_epochs(
            model, optimizer, rows, config, config.epochs, str(cycle)
        )
        scorer = frozen_scorer(model, architecture)
        clock.mark("policy")

        # 3. paired check: previous vs current policy on the same fresh pairs
        paired = fresh_pool(
            programs.fresh_pairs(
                config.paired_check,
                stream(config, cycle, "paired"),
                behaviours,
                max_cuts=config.max_cuts,
                grammar=programs.Grammar.parse(level),
            ),
            "paired",
        )
        seed = config.seed * 1000 + 300 + cycle
        before, _, _, _, _ = roll(previous_scorer, paired, procedure, seed)
        after, _, _, _, _ = roll(scorer, paired, procedure, seed)
        rollouts += 2 * len(paired) * config.n
        report["paired_check"] = {
            "previous": success(before, config.n),
            "current": success(after, config.n),
        }
        if config.widen_levels:
            report["widening"] = widen(level, levels, paired, after, config)
            report["paired_check"]["by_body"] = body_record(paired, after, config.n)
        clock.mark("paired_check")
        prior = report["paired_check"]["previous"]["walker_success"] or 0.0
        current = report["paired_check"]["current"]["walker_success"] or 0.0
        previous_scorer = scorer

        # 4. calibration
        rule = selector.Selector(size=config.calibration, max_cuts=config.max_cuts)
        if ranked:
            picks = selector.select_ranked(
                archive, cycle, stream(config, cycle, "select"), ranked, rule
            )
        else:
            picks = selector.select(
                archive, cycle, stream(config, cycle, "select"), rule
            )
        keys = [key for key, _ in picks]
        frozen_d_ops = [
            archive.pairs[k].d_ops for k in keys
        ]  # before this attempt (F5)
        attempts_before = {k: archive.pairs[k].attempts for k in keys}
        cal = archive_pool(archive, keys, "calibration")
        hits, steps, ends, walks, _ = roll(
            scorer, cal, procedure, config.seed * 1000 + cycle
        )
        rollouts += len(cal) * config.n
        nodes_before = len(archive.states)
        made, touched = archive_walks(archive, keys, walks, cycle)
        clock.mark("calibration_rollouts")
        report["calibration"] = {
            "pairs": len(keys),
            "rules": dict(Counter(rule for _, rule in picks)),
            "unique_pairs": len(set(keys)),
            "share_on_pairs_attempted_3_plus": float(
                np.mean([attempts_before[k] + 1 >= 3 for k in keys]) if keys else 0.0
            ),
            **success(hits, config.n),
            "steps_to_go_mean": float(steps.mean()) if keys else None,
            "ends": ends,
            "rollout_made": made,
            "nodes_added": len(archive.states) - nodes_before,
            "predicted_progress_top": selector.predicted_progress(e_history, cycle, 20),
        }
        if keys and (hits == 0).mean() >= config.uninformative_zero_share:
            report["stop"] = "uninformative: calibration labels almost all zero"
        parts = H.partition(cal.lineage, config.seed + cycle)
        # RND first, so n0 exists when the models are fitted
        visited_states = [archive.states[i] for i in sorted(touched)]
        replay = stream(config, cycle, "rnd").sample(
            rnd_seen, min(len(rnd_seen), config.rnd_replay)
        )
        report["rnd"] = {
            "train_loss": rnd.train(
                visited_states + replay, config.rnd_epochs, seed=cycle
            )
        }
        rnd_seen.extend(visited_states)
        reference = stream(config, cycle, "rndref").sample(
            archive.states, min(len(archive.states), 5000)
        )
        rnd.normalise(reference)
        pick = stream(config, cycle, "rndcal")
        well = [
            s for s, v in zip(archive.states, archive.visits, strict=True) if v >= 3
        ]
        never = [
            s for s, v in zip(archive.states, archive.visits, strict=True) if v == 0
        ]
        well = pick.sample(well, min(len(well), 5000))
        never = pick.sample(never, min(len(never), 5000))
        threshold = H.calibrate_threshold(rnd.novelty(well), rnd.novelty(never))
        cohort_novelty = rnd.novelty(cohort)
        report["rnd"].update(
            threshold,
            cohort_pass=float((cohort_novelty <= threshold["n0"]).mean())
            if threshold["n0"] is not None
            else None,
            cohort_raw_error_mean=float(rnd.raw_errors(cohort).mean()),
            scale=rnd.scale,
        )
        clock.mark("rnd")
        models, features = fit_models(
            scorer,
            cal,
            hits,
            config.n,
            parts,
            config.seed + cycle,
            threshold["n0"],
            config=config,
        )
        report["calibration"]["heads"] = models.facts
        report["calibration"]["gate_bins"] = models.bins
        report["calibration"]["by_body"] = body_record(cal, hits, config.n)
        if config.gate == "e_set":
            # archive pairs certified on this calibration's own rollouts
            open_cal = (
                models.e_head.predict(features) < models.e0_set
                if models.e0_set is not None
                else np.zeros(len(keys), dtype=bool)
            )
            now = open_cal & scoring.hit_wilson(
                models.p_head.predict(features), hits, config.n
            )
            before_certified = len(certified_keys)
            certified_keys |= {k for k, c in zip(keys, now, strict=True) if c}
            report["calibration"]["certified"] = {
                "this_phase": int(now.sum()),
                "new_archive_pairs": len(certified_keys) - before_certified,
                "archive_pairs_total": len(certified_keys),
            }
        # Measured progress: |P_hat - P_emp| on re-attempted pairs, P_hat from a
        # head that did not see them (partitions 1-2 were not P's).
        p_cal = models.p_head.predict(features)
        error_now = np.abs(p_cal - hits / config.n)
        again = [
            (k, error_now[i])
            for i, (k, rule) in enumerate(picks)
            if rule == selector.REATTEMPT and k in last_error and parts[i] != 0
        ]
        report["progress"] = {
            "reattempted_scored": len(again),
            "measured_error_drop_mean": float(
                np.mean([last_error[k] - e for k, e in again])
            )
            if again
            else None,
        }
        last_error.update(
            {k: error_now[i] for i, k in enumerate(keys) if parts[i] != 0}
        )
        # E on the tracked set, logged for the predicted-progress ranking only
        tracked = {k for k, r in archive.pairs.items() if r.attempts > 0}
        fresh_keys = [
            k
            for k, r in archive.pairs.items()
            if r.witness_d_cuts <= config.max_cuts and k not in tracked
        ]
        tracked |= set(
            stream(config, cycle, "tracked").sample(
                fresh_keys, min(len(fresh_keys), config.tracked_new)
            )
        )
        tracked_list = sorted(tracked)
        e_now = models.e_head.predict(
            H.pair_features(
                scorer,
                [(archive.states[s], archive.states[t]) for s, t in tracked_list],
            )
        )
        e_history[cycle] = dict(zip(tracked_list, e_now.tolist(), strict=True))
        clock.mark("fit")

        # 5. test: never archived
        before_test = (len(archive.states), len(archive.pairs), sum(archive.visits))
        canonical.update(archive)
        final = cycle == config.cycles
        fresh_pairs = programs.fresh_pairs(
            config.test_fresh_final if final else config.test_fresh,
            stream(config, cycle, "F3"),
            behaviours,
            max_cuts=config.max_cuts,
            grammar=programs.Grammar.parse(config.test_level or level),
        )
        fresh = fresh_pool(fresh_pairs, "F3")
        deadline = time.perf_counter() + config.final_eval_seconds if final else None
        f_hits, _, f_ends, _, f_done = roll(
            scorer, fresh, procedure, config.seed * 1000 + 500 + cycle, deadline
        )
        fresh = fresh.subset(range(f_done))
        rollouts += f_done * config.n
        pool_keys = sorted(
            k for k, r in archive.pairs.items() if r.witness_d_cuts <= config.max_cuts
        )
        archive_keys = stream(config, cycle, "test").sample(
            pool_keys, min(len(pool_keys), config.test_archive)
        )
        held = archive_pool(archive, archive_keys, "archive")
        a_hits, _, a_ends, _, _ = roll(
            scorer, held, procedure, config.seed * 1000 + 700 + cycle
        )
        rollouts += len(held) * config.n
        clock.mark("test_rollouts")
        f_labels, f_facts = stratify(
            fresh,
            scorer,
            {"novelty_class": [canonical.classify(s, t) for s, t in fresh.jobs()]},
        )
        a_labels, a_facts = stratify(held, scorer, {"origin": held.origin})
        counted = np.array(
            [
                archive.visits[s] >= 3 and archive.visits[t] >= 3
                for s, t in archive_keys
            ],
            dtype=bool,
        )
        test_rows: list[dict] = []
        per_pair = (
            {
                "gate": config.gate,
                "count": probes.count_novelty(archive, fresh.s, fresh.t),
                "rows_out": test_rows,
            }
            if config.gate == "e_set"
            else {}
        )
        fresh_scores, certified = score_pool(
            models, scorer, rnd, fresh, f_hits, config.n, f_labels, **per_pair
        )
        archive_scores, _ = score_pool(
            models,
            scorer,
            rnd,
            held,
            a_hits,
            config.n,
            a_labels,
            counted,
            **({"gate": config.gate} if config.gate == "e_set" else {}),
        )
        rollouts_total += rollouts
        certified_total += certified
        report["test"] = {
            "F3": {
                "facts": {
                    **f_facts,
                    "evaluated": f_done,
                    "drawn": len(fresh_pairs),
                    "ends": f_ends,
                    "novelty_class": dict(Counter(f_labels["novelty_class"])),
                },
                **success(f_hits, config.n),
                "by_body": body_record(fresh, f_hits, config.n),
                "arms": fresh_scores,
            },
            "archive": {
                "facts": {**a_facts, "ends": a_ends},
                **success(a_hits, config.n),
                "arms": archive_scores,
            },
        }
        report["executions"] = {
            "selector": config.selector,
            "certified_gate": "e_set"
            if config.gate == "e_set"
            else f"e0_{HEADLINE_E0}",
            "rollouts_this_cycle": rollouts,
            "certified_fresh_this_cycle": certified,
            "rollouts_per_certified": rollouts / certified if certified else None,
            "rollouts_cumulative": rollouts_total,
            "certified_fresh_cumulative": certified_total,
            "rollouts_per_certified_cumulative": rollouts_total / certified_total
            if certified_total
            else None,
        }
        report["relaxations_this_cycle"] = sum(
            r["cycle"] == cycle for r in archive.relaxations
        )
        report["diagnostics"] = diagnostics(report, config)
        clock.mark("test_scoring")
        if prior > 0 and current < config.collapse_ratio * prior:
            report["stop"] = (
                report.get("stop") or "fresh success collapsed after the policy phase"
            )
        report["archive"] = archive.summary()
        report["test_left_archive_unchanged"] = before_test == (
            len(archive.states),
            len(archive.pairs),
            sum(archive.visits),
        )
        report["timings"] = clock.phases
        calibration_rows = [
            {
                "s": k[0],
                "t": k[1],
                "rule": rule,
                "part": int(parts[i]),
                "hits": int(hits[i]),
                "p_hat": float(p_cal[i]),
                "steps_to_go": float(steps[i]),
                "frozen_d_ops": frozen_d_ops[i],
                "witness_d_ops": archive.pairs[k].witness_d_ops,
                "shortest_d_ops": archive.pairs[k].d_ops,
                "d_cuts": archive.pairs[k].witness_d_cuts,
                "origin": archive.pairs[k].origin,
                "attempts": archive.pairs[k].attempts,
                "body_atoms": programs.body_atoms(archive.pairs[k].witness_path),
                "segments": programs.path_segments(archive.pairs[k].witness_path),
            }
            for i, (k, rule) in enumerate(picks)
        ]
        extra = {
            "proposals.jsonl.gz": proposal_log,
            "calibration.jsonl.gz": calibration_rows,
        }
        if test_rows:
            extra["test_rows.jsonl.gz"] = test_rows
        save(report, f"cycle-{cycle}", extra)
        if models.plain is not None:
            (run_dir / f"cycle-{cycle}" / "models.json").write_text(
                json.dumps(models.plain, indent=1, default=str)
            )
        save_policy(
            run_dir / f"cycle-{cycle}" / "policy",
            scorer.model,
            trainer_state={"cycle": cycle, **report["policy"], **report["rows"]},
            manifest={
                "issue": "novelty-reduction (#171)",
                "config": dataclasses.asdict(config),
            },
            wandb_url=None,
        )
        reports.append(report)
        previous_models = models
        if config.widen_levels:
            level = report["widening"]["next_level"]
        log(
            f"cycle {cycle}: archive {report['archive']}, cal success "
            f"{report['calibration']['walker_success']}, F3 success "
            f"{report['test']['F3']['walker_success']}, timings {clock.phases}"
        )
        if report.get("stop"):
            log(f"cycle {cycle} stop: {report['stop']}")
            break
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    skip = ("architecture", "behaviours", "ridge_lambdas", "widen_levels")
    for field in dataclasses.fields(Config):
        if field.name in skip:
            continue
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=type(field.default),
            default=field.default,
        )
    # levels contain commas, so they are space-separated: --widen-levels 2,1,none 4,3,none
    parser.add_argument("--widen-levels", nargs="*", default=[])
    parser.add_argument(
        "--ridge-lambdas", nargs="*", type=float, default=list(probes.LAMBDAS)
    )
    args = parser.parse_args(argv)
    values = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(Config)
        if f.name not in skip
    }
    values["widen_levels"] = tuple(args.widen_levels)
    values["ridge_lambdas"] = tuple(args.ridge_lambdas)
    run(Config(**values), args.run_dir, log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
