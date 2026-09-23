"""inner-frames (#160) E2's first gate: does a policy shown the frame pay less inside bodies?

The segment composite (`reports/segment-composite-20260922`) found that quotation
bodies carry 79-92% of the per-step deficit, and that even a budget per body
leaves world-v3 at 2 of 44 on the world-arm checkpoint `1m-s0-e5`. A search that
commits to a quotation's consumer can run the body on its frame. This driver
measures what that is worth, paired op for op against `1m-s0-e5`, on one
witness per eval behaviour of v4 and world-v3:

- **frame-legal mask** (the search side, no training): at a body step whose
  frame E1 defines, an op that refuses on the frame is not a candidate. A
  witness op the frame refuses kills the witness, since a real search would
  have pruned it too.
- **frame-conditioned policies** (the training side): clear-world-v3 (#161)'s
  recipe (`experiments/world_arm.py`) with body steps re-encoded by
  `experiments/frame_rows.py`, as ``frames`` (tag and frame) and as the
  ``tags`` control (tag, machine items).

Three things keep a deficit comparison honest across differently trained
policies, and every stage applies all three:

- **Temperature.** A deficit is a logit gap, so a policy that is merely less
  confident has smaller deficits without ranking any better. Each checkpoint
  gets one temperature, fitted to minimise cross-entropy on its own validation
  rows (the v4 holdout the trainer already scores, in its own encoding), and
  deficits are divided by it. The witness op's rank among the legal ops is
  reported beside it, and needs no temperature at all.
- **Commitment.** Tags and frames are the witness's own, in hindsight. A search
  must choose, at each `[`, the consumer, the `ifte` role and how many values
  the closure idiom will capture; a `[` is charged ``-log p(that choice | the
  shape of the stack under the mark)``, with ``p`` counted on the **train**
  split and applied to eval witnesses, so it is out of sample.
- **Which ops are controls.** A body op is *defined* when E1 defines its
  innermost frame on every example, *re-encoded* when it is not but some level
  or some example is still rewritten, and *untouched* when every example shows
  the machine's state. Only top-level and untouched ops see identical inputs in
  every arm.

Stages, each writing under ``--run-dir`` (default `checkpoints/frame-arm-20260922`):
``predict`` (the world arm alone), ``probe`` (one-epoch learning rates),
``train``, ``measure``, ``truncation`` (which end the state cap cuts) and
``report`` (the registered verdict).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.hindsight_subtargets as hs  # noqa: E402
import experiments.world_arm as wa  # noqa: E402
from experiments.frame_rows import (  # noqa: E402
    META,
    example_walks,
    frame_state,
    frame_witness_rows,
    level_exits,
)
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from experiments.inner_frames import (  # noqa: E402
    BUDGET,
    COMBINATORS,
    PREVIEW_DEPTH,
    attribute,
    replay,
    walk_one_example,
    witnesses,
)
from experiments.search_proxy import TOLERANCE  # noqa: E402
from experiments.segment_composite import levels, segments  # noqa: E402
from experiments.staged_transfer import step_kinds  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_beam import step_all_quiet  # noqa: E402
from wandering_light.stack_machine import Mark, Sym, parse_stack, run_body  # noqa: E402
from wandering_light.stack_pbe import BASIS, OPS, load  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    STATE_CAP,
    PolicyScorer,
    _encode_value,
    encode_item,
    encode_targets,
    item_array,
    load_policy,
    train_policy,
    witness_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Task

DATA = REPO / "wandering_light" / "evals" / "data"
RUN_DIR = REPO / "checkpoints" / "frame-arm-20260922"
# The world arm's cell every measurement pairs against, fixed by the segment
# composite. It is the lowest-deficit of 18 and was selected on world-v3 eval.
BASELINE = REPO / "checkpoints" / "world-arm-20260922" / "policies" / "1m-s0-e5"
CORPORA = ("stack_pbe_v4", "stack_pbe_world_v3")
STRUCTURAL = ("[", "]")
SCALES = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
NO_COMMITMENT = "none"


def eval_witnesses(corpus: str) -> list[Task]:
    """One task per eval behaviour, as `witness_deficit.py` and the composite use."""
    seen, out = set(), []
    for t in load(fetch_corpus(DATA / corpus / "manifest.json")):
        if t.split == "eval" and t.behaviour not in seen:
            seen.add(t.behaviour)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------


def commitments(program: Sequence[str], first_input: str) -> dict[int, tuple]:
    """``{step of a [: (shape under the mark, (consumer, role, captures))}``.

    Read on the first shown example, as E1 reads shapes. A quotation with no
    attributable consumer commits to ``none``.
    """
    states = replay(program, parse_stack(first_input, basis=BASIS))
    if states is None:
        return {}
    tags = attribute(program)
    walk = walk_one_example(program, states, tags, capture_aware=True)
    out = {}
    for step, name in enumerate(program):
        if name != "[":
            continue
        kind, role, captures, _consumer = tags.get(step, ("never_closed", 0, 0, None))
        choice = (
            (kind, role or 0, captures or 0)
            if kind in COMBINATORS
            else (NO_COMMITMENT, 0, 0)
        )
        out[step] = (walk["shape"].get(step, NO_COMMITMENT), choice)
    return out


class CommitmentModel:
    """``-log p(choice | shape)`` from train-split counts, add-one smoothed.

    Smoothing is over a closed vocabulary, so the probabilities of every
    possible choice under a shape sum to one: no consumer, each non-`ifte`
    consumer with 0..K captures, and `ifte` at each of its three roles with
    0..K captures, where K is one more than the most captures seen in train.
    """

    def __init__(self, counts: Counter):
        self.joint = counts
        self.by_shape = Counter()
        for (shape, _choice), n in counts.items():
            self.by_shape[shape] += n
        most = max((choice[2] for _shape, choice in counts), default=0)
        self.max_captures = most + 1
        self.vocabulary = 1 + (len(COMBINATORS) - 1 + 3) * (self.max_captures + 1)

    @classmethod
    def fit(cls, corpus: str) -> CommitmentModel:
        tasks = load(fetch_corpus(DATA / corpus / "manifest.json"))
        counts = Counter()
        for _behaviour, program, inputs in witnesses(tasks, "train"):
            for shape, choice in commitments(program, inputs[0]).values():
                counts[shape, choice] += 1
        return cls(counts)

    def cost(self, shape: str, choice: tuple) -> float:
        if choice[2] > self.max_captures:
            raise ValueError(f"{choice} is outside the commitment vocabulary")
        n = self.joint.get((shape, choice), 0)
        return -math.log((n + 1) / (self.by_shape.get(shape, 0) + self.vocabulary))

    def per_step(self, task: Task) -> list[float]:
        """The commitment charge at each witness step: nonzero only at a `[`."""
        charged = commitments(task.program, task.inputs[0])
        return [
            self.cost(*charged[step]) if step in charged else 0.0
            for step in range(len(task.program))
        ]


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def step_classes(task: Task, walks) -> tuple[list[str], list[str | None]]:
    """Per step: ``top``, ``defined``, ``reencoded`` or ``untouched``; and the tag."""
    classes, tags = [], []
    for step in range(len(task.program)):
        per = [views.get(step, ()) for _states, views in walks]
        if not per[0]:
            classes.append("top")
            tags.append(None)
            continue
        inner = [v[-1] for v in per]
        kinds = {t for t, _role, _captures, _virtual in inner}
        if (
            len(kinds) == 1
            and next(iter(kinds)) in COMBINATORS
            and all(virtual is not None for _t, _r, _c, virtual in inner)
        ):
            classes.append("defined")
        elif all(
            frame_state(states[step], views.get(step, ())) == states[step]
            for states, views in walks
        ):
            classes.append("untouched")
        else:
            classes.append("reencoded")
        tags.append(inner[0][0])
    return classes, tags


def frame_legal(frames: Sequence[tuple], name: str) -> bool:
    """Does op ``name`` run on every example's frame, as E1 advances a body?"""
    return all(
        not isinstance(run_body((Sym(name),), frame, BUDGET, PREVIEW_DEPTH), str)
        for frame in frames
    )


def walk(
    task: Task, scorer: PolicyScorer, *, encoding: str | None, mask: bool
) -> dict[str, Any]:
    """Per-step raw deficit and rank of the witness op.

    ``encoding`` is what the policy is shown at body steps: ``None`` for the
    machine state, or a `frame_rows` encoding. ``mask`` removes, at defined body
    steps, every non-structural op that refuses on the frame; a witness op it
    removes is recorded as a kill (deficit ``None``), not restored.
    """
    states = replay_states(task)
    targets = encode_targets(states[-1])
    walks = example_walks(task)
    classes, tags = step_classes(task, walks)
    exits = [
        level_exits(task.program, views) if encoding == "exit_inner" else {}
        for _states, views in walks
    ]
    deficits: list[float | None] = []
    ranks: list[int | None] = []
    for step, name in enumerate(task.program):
        product = states[step]
        legal = np.zeros((1, len(OPS)), dtype=bool)
        for index, op in enumerate(OPS):
            if step_all_quiet(product, op)[0] is not None:
                legal[0, index] = True
        want = OP_INDEX[name]
        if mask and classes[step] == "defined":
            inner = [views[step][-1][3] for _states, views in walks]
            for index, op in enumerate(OPS):
                if legal[0, index] and op.name not in STRUCTURAL:
                    legal[0, index] = frame_legal(inner, op.name)
            if not legal[0, want]:
                deficits.append(None)
                ranks.append(None)
                continue
        shown = (
            tuple(
                frame_state(s[step], v.get(step, ()), encoding, ex.get(step, ()))
                for (s, v), ex in zip(walks, exits, strict=True)
            )
            if encoding
            else product
        )
        block = item_array(
            [encode_item(s, t) for s, t in zip(shown, targets, strict=True)]
        )
        row = scorer.log_probs([block], legal)[0]
        mine = float(row[want])
        deficits.append(float(np.max(row[legal[0]])) - mine)
        ranks.append(int((row[legal[0]] > mine).sum()) + 1)
    return {
        "behaviour": task.behaviour,
        "deficits": [None if d is None else round(d, 5) for d in deficits],
        "ranks": ranks,
        "classes": classes,
        "tags": tags,
        "kinds": step_kinds(task),
        "program": list(task.program),
        "killed": sum(d is None for d in deficits),
    }


def body_summary(rows: Sequence[dict], temperature: float) -> dict[str, Any]:
    """Per class and per defined consumer: ops, deficit (raw and calibrated), ranks."""
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        for d, rank, cls, tag in zip(
            r["deficits"], r["ranks"], r["classes"], r["tags"], strict=True
        ):
            if d is None:
                continue
            groups[cls].append((d, rank))
            if cls != "top":
                groups["body"].append((d, rank))
            if cls == "defined":
                groups[f"defined:{tag}"].append((d, rank))
    out = {}
    for key, values in sorted(groups.items()):
        deficits = [d for d, _ in values]
        out[key] = {
            "ops": len(values),
            "mean": round(statistics.fmean(deficits), 4),
            "calibrated_mean": round(statistics.fmean(deficits) / temperature, 4),
            "total": round(sum(deficits), 3),
            "calibrated_total": round(sum(deficits) / temperature, 3),
            "top1": round(sum(rank == 1 for _, rank in values) / len(values), 4),
            "mean_rank": round(statistics.fmean(rank for _, rank in values), 4),
        }
    return out


def composite(
    rows: Sequence[dict],
    *,
    temperature: float,
    commit: dict[str, list[float]] | None = None,
    scale: float = 1.0,
) -> dict[str, int]:
    """Clears under the four decompositions, on calibrated deficits.

    ``commit`` maps a behaviour to its per-step commitment charges; ``scale``
    multiplies the deficit of defined body ops. A killed witness clears nothing.
    """
    out = defaultdict(int)
    for r in rows:
        if r["killed"]:
            continue
        charges = commit[r["behaviour"]] if commit else [0.0] * len(r["deficits"])
        deficits = [
            (d / temperature) * (scale if cls == "defined" else 1.0) + c
            for d, cls, c in zip(r["deficits"], r["classes"], charges, strict=True)
        ]
        segs = segments(deficits, r["kinds"])
        top, bodies = levels(r["program"], deficits, r["kinds"])
        worst_body = max(bodies, default=0.0)
        out["flat"] += sum(deficits) <= TOLERANCE
        out["segmented"] += max(segs) <= TOLERANCE
        out["quotations"] += max(sum(top), worst_body) <= TOLERANCE
        out["both"] += max(*top, worst_body) <= TOLERANCE
    out["witnesses"] = len(rows)
    return dict(out)


def exchange_rate(
    rows: Sequence[dict], temperature: float, commit: dict[str, list[float]]
) -> list[dict[str, Any]]:
    """Composite clears as calibrated defined-body deficit is scaled."""
    return [
        {
            "scale": scale,
            "free_commitment": composite(rows, temperature=temperature, scale=scale),
            "with_commitment": composite(
                rows, temperature=temperature, commit=commit, scale=scale
            ),
        }
        for scale in SCALES
    ]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

_VALIDATION: dict[str | None, list] = {}
_LEGAL: list[np.ndarray] = []
# log T is searched over [1/64, 64]; an optimum at either edge is reported, and
# the gate refuses a verdict on it, rather than being silently clipped.
T_BOUNDS = (1 / 64, 64.0)


def validation_rows(encoding: str | None) -> list:
    """The world arm's validation rows (v4 holdout), in ``encoding``."""
    if encoding not in _VALIDATION:
        _fit, held = co.split_tasks(wa.configure())
        _VALIDATION[encoding] = (
            witness_rows(held)
            if encoding is None
            else frame_witness_rows(held, encoding)
        )
    return _VALIDATION[encoding]


def validation_legal() -> np.ndarray:
    """The machine's legal mask at every validation row, in `witness_rows` order.

    A deficit is taken over the legal ops only, so the temperature is fitted
    over them too: an illegal op's logit must not move it.
    """
    if not _LEGAL:
        _fit, held = co.split_tasks(wa.configure())
        for task in held:
            states = replay_states(task)
            for step in range(len(task.program)):
                legal = np.zeros(len(OPS), dtype=bool)
                for index, op in enumerate(OPS):
                    if step_all_quiet(states[step], op)[0] is not None:
                        legal[index] = True
                _LEGAL.append(legal)
    return np.stack(_LEGAL)


def fit_temperature(
    scorer: PolicyScorer, rows: Sequence, legal: np.ndarray | None = None
) -> dict[str, float]:
    """The temperature minimising masked validation cross-entropy, by golden section on log T."""
    logits = np.concatenate(
        [
            scorer.logits([row.items for row in rows[i : i + 512]])
            for i in range(0, len(rows), 512)
        ]
    ).astype(np.float64)
    labels = np.array([row.label for row in rows])
    if legal is not None:
        if (
            legal.shape != logits.shape
            or not legal[np.arange(len(labels)), labels].all()
        ):
            raise ValueError("the legal mask must cover every row and its label")
        logits = np.where(legal, logits, -np.inf)

    def nll(log_t: float) -> float:
        z = logits / math.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(np.mean(lse - z[np.arange(len(labels)), labels]))

    lo, hi = math.log(T_BOUNDS[0]), math.log(T_BOUNDS[1])
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(b)
    best = (lo + hi) / 2
    edge = min(best - math.log(T_BOUNDS[0]), math.log(T_BOUNDS[1]) - best)
    return {
        "temperature": round(math.exp(best), 5),
        "validation_ce": round(nll(0.0), 5),
        "calibrated_validation_ce": round(nll(best), 5),
        "at_boundary": bool(edge < 1e-3),
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def scorer_for(path: Path) -> PolicyScorer:
    return PolicyScorer(load_policy(path, device="cpu"), "cpu")


_COMMIT: dict[str, dict[str, list[float]]] = {}


def commit_charges(corpus: str, tasks: Sequence[Task]) -> dict[str, list[float]]:
    if corpus not in _COMMIT:
        model = CommitmentModel.fit(corpus)
        _COMMIT[corpus] = {t.behaviour: model.per_step(t) for t in tasks}
    return _COMMIT[corpus]


def measure_cells(scorer: PolicyScorer, *, encoding: str | None, log) -> dict[str, Any]:
    calibration = fit_temperature(scorer, validation_rows(encoding), validation_legal())
    t = calibration["temperature"]
    log(f"calibration {calibration}")
    cells: dict[str, Any] = {"calibration": calibration}
    for corpus in CORPORA:
        tasks = eval_witnesses(corpus)
        commit = commit_charges(corpus, tasks)
        variants = {}
        for mask in (False, True):
            rows = [walk(task, scorer, encoding=encoding, mask=mask) for task in tasks]
            key = "frame_mask" if mask else "machine_mask"
            variants[key] = {
                "body": body_summary(rows, t),
                "composite": composite(rows, temperature=t),
                "composite_with_commitment": composite(
                    rows, temperature=t, commit=commit
                ),
                "exchange_rate": exchange_rate(rows, t, commit),
                "killed_steps": sum(r["killed"] for r in rows),
                "killed_witnesses": sum(bool(r["killed"]) for r in rows),
                "rows": rows,
            }
            b = variants[key]["body"]["defined"]
            log(
                f"{corpus} {key}: defined body calibrated {b['calibrated_mean']} "
                f"top1 {b['top1']} over {b['ops']} ops | killed "
                f"{variants[key]['killed_witnesses']} | "
                f"{variants[key]['composite_with_commitment']}"
            )
        cells[corpus] = variants
    cells["commitment"] = {
        corpus: {
            "charged": sum(1 for c in _COMMIT[corpus].values() for x in c if x),
            "mean": round(
                statistics.fmean(x for c in _COMMIT[corpus].values() for x in c if x), 4
            ),
        }
        for corpus in CORPORA
    }
    return cells


def predict(run_dir: Path, log) -> Path:
    """The registration's inputs, from the world arm alone: no training."""
    out = run_dir / "prediction.json"
    report = {
        "policy": str(BASELINE),
        "budget_nats": TOLERANCE,
        "cells": measure_cells(scorer_for(BASELINE), encoding=None, log=log),
    }
    co.write_json(out, report)
    log(f"wrote {out}")
    return out


# Each arm trains at the world arm's rate unless the one-epoch probe found a rate
# more than 2% better in validation CE (the rule registered on #160). None did
# for the three-item label, so every arm is at 1.25e-4; a frames arm at another
# rate would be measured as `MATCHED`'s counterpart.
ARM_LR = {
    "world": wa.LEARNING_RATE,
    "tags": wa.LEARNING_RATE,
    "frames": wa.LEARNING_RATE,
    # The second campaign (#160): innermost-only labels and frames, and the
    # exit-frame oracle, at the rate every stacked arm trained at.
    "tags_inner": wa.LEARNING_RATE,
    "frames_inner": wa.LEARNING_RATE,
    "exit_inner": wa.LEARNING_RATE,
}
MATCHED = "framesmatched"


def arm_dir(run_dir: Path, arm: str, learning_rate: float = wa.LEARNING_RATE) -> Path:
    """Where an arm's snapshots live; a non-default learning rate gets its own."""
    if arm == "world":
        return BASELINE.parents[1]
    suffix = "" if learning_rate == wa.LEARNING_RATE else f"-lr{learning_rate:g}"
    return run_dir / f"{arm}{suffix}"


def rows_for(encoding: str):
    return lambda tasks: frame_witness_rows(tasks, encoding)


def train(
    run_dir: Path,
    device: str,
    seeds: Sequence[int],
    log,
    *,
    encoding: str = "frames",
    learning_rate: float = wa.LEARNING_RATE,
) -> None:
    """`world_arm.train` with the rows re-encoded; nothing else changes."""
    tasks = wa.configure()
    fit, held = co.training_rows(tasks, rows=rows_for(encoding))
    log(f"RESULT rows ({encoding}): fit {len(fit)} validation {len(held)}")
    out_dir = arm_dir(run_dir, encoding, learning_rate)
    for seed in seeds:
        if hs.policy_dir(out_dir, wa.ARM, seed, wa.EPOCHS).exists():
            log(f"skip train {encoding} s{seed} (snapshots exist)")
            continue
        manifest = {
            "arm": wa.ARM,
            "encoding": f"{encoding} (experiments/frame_rows.py)",
            "corpora": list(CORPORA),
            "epochs": wa.EPOCHS,
            "learning_rate": learning_rate,
            "seed": seed,
            "dropped_behaviours": sorted(wa.CONTAMINATED),
            "provenance": co.provenance(),
        }
        started = time.perf_counter()
        _model, state = train_policy(
            fit,
            held,
            co.architecture(wa.ARM),
            epochs=wa.EPOCHS,
            learning_rate=learning_rate,
            batch_size=wa.BATCH,
            validation_batch_size=wa.VAL_BATCH,
            seed=seed,
            device=device,
            epoch_end=hs.snapshotter(out_dir, wa.ARM, seed, manifest),
        )
        co.write_json(out_dir / "training" / f"s{seed}.json", state)
        curve = " ".join(
            f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}"
            for r in state["history"]
        )
        log(
            f"RESULT train {encoding} s{seed}: [{curve}] "
            f"{time.perf_counter() - started:.0f}s"
        )


PROBE_RATES = (6.25e-5, 1.25e-4, 2.5e-4, 5e-4)


def probe(
    run_dir: Path,
    device: str,
    log,
    *,
    encodings: Sequence[str] = ("flat", "frames", "tags"),
    rates: Sequence[float] = PROBE_RATES,
) -> dict[str, dict[str, float]]:
    """One epoch at each learning rate, seed 0, per encoding: validation CE.

    The frame rows change 77% of the inputs and add tag tokens, so the world
    arm's rate is not known to suit them; a gate that fails at an untuned rate
    has not failed frames. ``flat`` is the world arm's own rows, so its cell at
    the world arm's rate must reproduce that arm's first epoch exactly, and
    every other rate is compared with flat at the same rate.
    """
    path = run_dir / "probe.json"
    out: dict[str, dict[str, float]] = (
        json.loads(path.read_text()) if path.exists() else {}
    )
    tasks = wa.configure()
    for encoding in encodings:
        todo = [r for r in rates if f"{r:g}" not in out.get(encoding, {})]
        if not todo:
            continue
        rows = witness_rows if encoding == "flat" else rows_for(encoding)
        fit, held = co.training_rows(tasks, rows=rows)
        for rate in todo:
            _model, state = train_policy(
                fit,
                held,
                co.architecture(wa.ARM),
                epochs=1,
                learning_rate=rate,
                batch_size=wa.BATCH,
                validation_batch_size=wa.VAL_BATCH,
                seed=0,
                device=device,
            )
            ce = state["history"][-1]["validation_cross_entropy"]
            out.setdefault(encoding, {})[f"{rate:g}"] = ce
            co.write_json(path, out)
            log(f"RESULT probe {encoding} lr {rate:g}: validation CE {ce:.4f}")
        del fit, held
    return out


def measure(
    run_dir: Path,
    arm: str,
    cells: Sequence[tuple[int, int]],
    log,
    learning_rate: float | None = None,
) -> list[Path]:
    """Per-step deficits of one arm's snapshots, each shown what it was trained on.

    An arm is measured at its registered rate (`ARM_LR`); the frames arm at the
    world arm's rate is labelled `MATCHED`.
    """
    rate = ARM_LR[arm] if learning_rate is None else learning_rate
    if rate == ARM_LR[arm]:
        label = arm
    elif arm == "frames" and rate == wa.LEARNING_RATE:
        label = MATCHED
    else:
        raise ValueError(f"no registered cell for {arm} at {rate:g}")
    source = arm_dir(run_dir, arm, rate)
    paths = []
    for seed, epoch in cells:
        out = run_dir / "measure" / f"{label}-s{seed}-e{epoch}.json"
        if out.exists():
            log(f"skip {out.name}")
            paths.append(out)
            continue
        path = hs.policy_dir(source, wa.ARM, seed, epoch)
        log(f"measure {label} s{seed}-e{epoch}: {path}")
        report_ = {
            "policy": str(path),
            "arm": arm,
            "paired_against": str(BASELINE),
            "budget_nats": TOLERANCE,
            "cells": measure_cells(
                scorer_for(path), encoding=None if arm == "world" else arm, log=log
            ),
        }
        co.write_json(out, report_)
        paths.append(out)
    return paths


# The registered gate (#160). Clears are v4 eval, both decompositions with the
# commitment charged, on calibrated deficits, at the primary cell; ratios are
# of calibrated defined-body deficit totals, medians over matched cells. The
# clear thresholds are the world arm's own exchange rate at `PRIMARY`: 12 at
# its measured deficit, 18 with defined-body deficit halved.
PRIMARY = (0, 5)
PASS_CLEARS, FAIL_CLEARS = 18, 12
PASS_RATIO, FAIL_RATIO = 0.6, 0.85
# The frame must add this much over the tag alone for a PASS to be the frame's.
FRAME_RATIO = 0.85
# What each verdict licenses, as registered. With commitment charged, even free
# bodies clear 23 of 45 v4 witnesses on the world arm, so cheap bodies alone
# cannot carry a committing search.
READING = {
    "PASS": (
        "Executing bodies made them cheap enough and ranked better. This licenses "
        "the committing search (tagged opens, bodies executed on frames, a budget "
        "per body) only with a consumer ranker in the same build: with commitment "
        "charged, even free bodies clear 23 of 45 v4 witnesses, so frames alone "
        "do not suffice."
    ),
    "TAG": "The hindsight label did the work, not executing the body.",
    "FAIL": (
        "Frames do not make bodies cheaper. The next lever is shortening them: "
        "macros under a mark (#162, via #149)."
    ),
    "FAIL_AT_TUNED_RATE_ONLY": (
        "FAIL at the tuned rate; the matched-rate frames arm is not measured."
    ),
    "WEAK": "Neither PASS nor FAIL: reported, and the owner's call.",
    "INCOMPLETE": "Not every registered cell is measured: no verdict.",
    "CALIBRATION_AT_BOUNDARY": (
        "A temperature fit sits at the edge of its search: no verdict."
    ),
}


def _defined(cell, corpus):
    return cell[corpus]["machine_mask"]["body"]["defined"]


REQUIRED = tuple((s, e) for s in wa.SEEDS for e in range(1, wa.EPOCHS + 1))


def report(
    run_dir: Path, log, required: Sequence[tuple[int, int]] = REQUIRED
) -> dict[str, Any]:
    """Pair every measured cell of the arms and read the registered gate off them.

    No verdict is given until every ``required`` cell of frames, tags and world
    is measured (and of the matched frames arm, for a FAIL), or while any of
    them was calibrated at the edge of the temperature search.
    """
    cells: dict[str, dict[tuple[int, int], dict]] = {
        "frames": {},
        MATCHED: {},
        "tags": {},
        "world": {},
    }
    for path in sorted((run_dir / "measure").glob("*.json")):
        arm, s, e = path.stem.split("-")
        cells[arm][int(s[1:]), int(e[1:])] = json.loads(path.read_text())["cells"]
    required = set(required)
    missing = {
        arm: sorted(required - set(cells[arm])) for arm in ("frames", "tags", "world")
    }
    boundary = sorted(
        f"{arm}-s{key[0]}-e{key[1]}"
        for arm, measured in cells.items()
        for key, cell in measured.items()
        if key in required and cell["calibration"].get("at_boundary")
    )
    pairs = sorted(required & set(cells["frames"]) & set(cells["world"]))
    table = []
    for key in pairs:
        for corpus in CORPORA:
            f = _defined(cells["frames"][key], corpus)
            w = _defined(cells["world"][key], corpus)
            # Full precision: a ratio rounded to 0.8500 before the threshold
            # would turn a true 0.84996 into a FAIL.
            row = {
                "seed": key[0],
                "epoch": key[1],
                "corpus": corpus,
                "defined_ratio": f["calibrated_total"] / w["calibrated_total"],
                "raw_defined_ratio": f["total"] / w["total"],
                "top1_gain": f["top1"] - w["top1"],
                "frames": cells["frames"][key][corpus]["machine_mask"][
                    "composite_with_commitment"
                ],
                "world": cells["world"][key][corpus]["machine_mask"][
                    "composite_with_commitment"
                ],
            }
            if key in cells[MATCHED]:
                m = _defined(cells[MATCHED][key], corpus)
                row["matched_defined_ratio"] = (
                    m["calibrated_total"] / w["calibrated_total"]
                )
                row["matched"] = cells[MATCHED][key][corpus]["machine_mask"][
                    "composite_with_commitment"
                ]
            if key in cells["tags"]:
                g = _defined(cells["tags"][key], corpus)
                row["frame_over_tag_ratio"] = (
                    f["calibrated_total"] / g["calibrated_total"]
                )
                row["tags"] = cells["tags"][key][corpus]["machine_mask"][
                    "composite_with_commitment"
                ]
            table.append(row)

    def median_of(field, corpus):
        values = [r[field] for r in table if r["corpus"] == corpus and field in r]
        return statistics.median(values) if values else None

    v4 = [r for r in table if r["corpus"] == "stack_pbe_v4"]
    primary = next((r for r in v4 if (r["seed"], r["epoch"]) == PRIMARY), None)
    full = median_of("defined_ratio", "stack_pbe_v4")
    gain = median_of("top1_gain", "stack_pbe_v4")
    frame = median_of("frame_over_tag_ratio", "stack_pbe_v4")
    verdict = None
    if any(missing.values()):
        verdict = "INCOMPLETE"
    elif boundary:
        verdict = "CALIBRATION_AT_BOUNDARY"
    elif primary is not None:
        clears = primary["frames"]["both"]
        if clears >= PASS_CLEARS and full < PASS_RATIO and gain > 0:
            verdict = "PASS" if frame < FRAME_RATIO else "TAG"
        elif clears <= FAIL_CLEARS and full >= FAIL_RATIO:
            # FAIL must hold at both frames rates, or it is a failure of the rate;
            # when frames already trains at the world arm's rate there is one.
            matched = median_of("matched_defined_ratio", "stack_pbe_v4")
            if ARM_LR["frames"] == wa.LEARNING_RATE:
                verdict = "FAIL"
            elif required - set(cells[MATCHED]) or "matched" not in primary:
                verdict = "FAIL_AT_TUNED_RATE_ONLY"
            elif primary["matched"]["both"] <= FAIL_CLEARS and matched >= FAIL_RATIO:
                verdict = "FAIL"
            else:
                verdict = "WEAK"
        else:
            verdict = "WEAK"
    out = {
        "pairs": len(pairs),
        "missing": missing,
        "calibration_at_boundary": boundary,
        "primary": primary,
        "median": {
            corpus: {
                field: median_of(field, corpus)
                for field in (
                    "defined_ratio",
                    "matched_defined_ratio",
                    "raw_defined_ratio",
                    "top1_gain",
                    "frame_over_tag_ratio",
                )
            }
            for corpus in CORPORA
        },
        "verdict": verdict,
        "reading": READING.get(verdict),
        "controls": controls(cells, pairs),
        "truncation": (
            json.loads((run_dir / "truncation.json").read_text())
            if (run_dir / "truncation.json").exists()
            else None
        ),
        "rows": table,
    }
    co.write_json(run_dir / "report.json", out)

    def shown(x):
        return None if x is None else f"{x:.4f}"

    log(
        f"RESULT gate {verdict}: {len(pairs)} pairs; v4 median calibrated ratio "
        f"{shown(full)}, top-1 gain {shown(gain)}, frame/tag {shown(frame)}; "
        f"primary clears {primary and primary['frames']['both']} vs world "
        f"{primary and primary['world']['both']}"
    )
    return out


EX_FOLD = "defined:ex-fold"


def controls(
    cells: dict[str, dict[tuple[int, int], dict]], pairs: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """Every op class's calibrated-total ratios, medians over the matched cells.

    Top-level and untouched ops see the same input in every arm, so their
    ratio is what training on the rewritten rows did to the policy as a whole;
    ``defined_over_top`` divides it out of the defined-body ratio, cell by cell.
    Not part of the verdict; reported beside it.
    """
    if not pairs:
        return {}
    arms = [a for a in ("frames", "tags") if all(k in cells[a] for k in pairs)]
    out: dict[str, Any] = {}
    for corpus in CORPORA:

        def body(arm, key, cls, corpus=corpus):
            summary = cells[arm][key][corpus]["machine_mask"]["body"]
            if cls != EX_FOLD:
                return summary.get(cls)
            # One v4 witness's fold body moves the label arms' defined total.
            fold = summary.get("defined:fold", {"ops": 0, "calibrated_total": 0.0})
            return {
                "ops": summary["defined"]["ops"] - fold["ops"],
                "calibrated_total": summary["defined"]["calibrated_total"]
                - fold["calibrated_total"],
            }

        classes = sorted(cells["world"][pairs[0]][corpus]["machine_mask"]["body"])
        classes.append(EX_FOLD)
        per_class = {}
        for cls in classes:
            entry = {"ops": body("world", pairs[0], cls)["ops"]}
            for num, den in (
                ("frames", "world"),
                ("tags", "world"),
                ("frames", "tags"),
            ):
                if num not in arms or (den != "world" and den not in arms):
                    continue
                ratios = [
                    body(num, k, cls)["calibrated_total"]
                    / body(den, k, cls)["calibrated_total"]
                    for k in pairs
                    if body(den, k, cls)["calibrated_total"] > 0
                ]
                if ratios:
                    entry[f"{num}/{den}"] = statistics.median(ratios)
                    entry[f"{num}/{den} range"] = [min(ratios), max(ratios)]
            per_class[cls] = entry
        clears = {
            arm: [
                cells[arm][k][corpus]["machine_mask"]["composite_with_commitment"][
                    "both"
                ]
                for k in pairs
            ]
            for arm in ("world", *arms)
        }
        corpus_out = {
            "classes": per_class,
            "clears": {
                arm: {"median": statistics.median(v), "range": [min(v), max(v)]}
                for arm, v in clears.items()
            },
        }
        if "frames" in arms:
            ratios = [
                (
                    body("frames", k, "defined")["calibrated_total"]
                    / body("world", k, "defined")["calibrated_total"]
                )
                / (
                    body("frames", k, "top")["calibrated_total"]
                    / body("world", k, "top")["calibrated_total"]
                )
                for k in pairs
            ]
            corpus_out["frames_defined_over_top"] = {
                "median": statistics.median(ratios),
                "range": [min(ratios), max(ratios)],
            }
        out[corpus] = corpus_out
    return out


def _token_starts(state: Sequence) -> tuple[list[int], int]:
    out: list[int] = []
    starts = []
    for value in state:
        starts.append(len(out))
        _encode_value(value, out)
    return starts, len(out)


def truncation(
    corpus: str, encodings: Sequence[str] = ("flat", "tags", "frames")
) -> dict[str, int]:
    """Which end the state cap cuts, per encoding, on eval states under a mark.

    The encoder keeps a state's top ``STATE_CAP`` tokens and each frame sits
    above its level's written items, so an overflow cuts the deep stack first.
    A label is counted as cut when its separator falls below the cap.
    """
    counts: Counter[str] = Counter()
    for task in eval_witnesses(corpus):
        for states, views in example_walks(task):
            exits = level_exits(task.program, views)
            for step in range(len(task.program)):
                state = states[step]
                if not any(isinstance(v, Mark) for v in state):
                    continue
                counts["under_mark"] += 1
                for encoding in encodings:
                    seen = (
                        state
                        if encoding == "flat"
                        else frame_state(
                            state, views.get(step, ()), encoding, exits.get(step, ())
                        )
                    )
                    starts, width = _token_starts(seen)
                    if width <= STATE_CAP:
                        continue
                    counts[f"{encoding}_clipped"] += 1
                    cut = width - STATE_CAP
                    labels = [i for i, v in enumerate(seen) if v == META]
                    if labels and starts[labels[-1]] < cut:
                        counts[f"{encoding}_innermost_label_cut"] += 1
                    if any(starts[i] < cut for i in labels[:-1]):
                        counts[f"{encoding}_outer_label_cut"] += 1
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "stage",
        choices=("predict", "probe", "train", "measure", "truncation", "report"),
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default=",".join(str(s) for s in wa.SEEDS))
    parser.add_argument("--arm", choices=tuple(ARM_LR), default="frames")
    parser.add_argument(
        "--lr", type=float, default=None, help="default: the arm's registered rate"
    )
    parser.add_argument("--encodings", default="flat,frames,tags")
    parser.add_argument("--rates", default=",".join(f"{r:g}" for r in PROBE_RATES))
    parser.add_argument(
        "--cells", default="0:5", help="seed:epoch pairs, comma-separated, or 'all'"
    )
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with open(args.run_dir / "log.txt", "a") as handle:
            handle.write(stamped + "\n")

    torch.set_num_threads(8)
    if args.stage == "predict":
        predict(args.run_dir, log)
    elif args.stage == "probe":
        probe(
            args.run_dir,
            args.device,
            log,
            encodings=args.encodings.split(","),
            rates=[float(r) for r in args.rates.split(",")],
        )
    elif args.stage == "train":
        if args.arm == "world":
            parser.error("the world arm is the fixed baseline; train frames or tags")
        seeds = [int(s) for s in args.seeds.split(",")]
        rate = ARM_LR[args.arm] if args.lr is None else args.lr
        train(
            args.run_dir, args.device, seeds, log, encoding=args.arm, learning_rate=rate
        )
    elif args.stage == "truncation":
        encodings = tuple(dict.fromkeys(("flat", *args.encodings.split(","))))
        out = {corpus: truncation(corpus, encodings) for corpus in CORPORA}
        co.write_json(args.run_dir / "truncation.json", out)
        log(f"RESULT truncation {out}")
    elif args.stage == "report":
        report(args.run_dir, log)
    else:
        cells = (
            [(s, e) for s in wa.SEEDS for e in range(1, wa.EPOCHS + 1)]
            if args.cells == "all"
            else [tuple(int(x) for x in c.split(":")) for c in args.cells.split(",")]
        )
        measure(args.run_dir, args.arm, cells, log, args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
