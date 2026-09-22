"""inner-frames (#160) E2's first gate: does a policy shown the frame pay less inside bodies?

The segment composite (`reports/segment-composite-20260922`) found that quotation
bodies carry 79-92% of the per-step deficit, and that even a budget per body
leaves world-v3 at 2 of 44 on the world-arm checkpoint `1m-s0-e5`. A search that
commits to a quotation's consumer can run the body on its frame, so two things
could lower what a body costs. This driver measures both, paired op for op
against `1m-s0-e5`, on one witness per eval behaviour of v4 and world-v3:

- **frame-legal mask** (the search side, no training): at a body step whose
  frame E1 defines, an op the machine would capture but that refuses on the
  frame is not a candidate. Measured on `1m-s0-e5` directly.
- **frame-conditioned policy** (the training side): `1m-s0-e5`'s recipe
  (clear-world-v3 #161, `experiments/world_arm.py`) on the same tasks, split
  and seeds, with every body step re-encoded by `experiments/frame_rows.py`.

Stages, each writing under ``--run-dir`` (default `checkpoints/frame-arm-20260922`):

- ``predict``: before any training. The world arm's per-step deficit on body
  ops, split by whether the frame is defined, under the machine's mask and the
  frame-legal one; and the exchange rate from a reduction in defined-body
  deficit to composite clears, which is what the registration prices.
- ``train``: the frame arm, three seeds, six epochs, every epoch snapshotted.
- ``measure``: the frame arm's per-step deficit on the same ops, and the composite.

Deficits are nats behind the best legal op, as `staged_transfer.score` walks
them; the legal mask is always the machine's except where the frame-legal
variant says otherwise. Tags and frames are hindsight, so what committing to a
consumer costs is added as a separate term (E1's in-sample H(tag | shape)).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
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
    example_walks,
    frame_state,
    frame_witness_rows,
)
from experiments.hindsight_subtargets import replay_states  # noqa: E402
from experiments.inner_frames import (  # noqa: E402
    BUDGET,
    COMBINATORS,
    PREVIEW_DEPTH,
)
from experiments.search_proxy import TOLERANCE  # noqa: E402
from experiments.segment_composite import levels, segments  # noqa: E402
from experiments.staged_transfer import step_kinds  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_beam import step_all_quiet  # noqa: E402
from wandering_light.stack_machine import Sym, run_body  # noqa: E402
from wandering_light.stack_pbe import OPS, load  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    PolicyScorer,
    encode_item,
    encode_targets,
    item_array,
    load_policy,
    train_policy,
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
# E1's capture-aware in-sample H(tag | shape) per eval split, in nats: what a
# search pays to commit to a consumer at each `[` if it gets the shape free.
COMMIT_NATS = {"stack_pbe_v4": 0.377, "stack_pbe_world_v3": 0.462}
SCALES = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)


def eval_witnesses(corpus: str) -> list[Task]:
    """One task per eval behaviour, as `witness_deficit.py` and the composite use."""
    seen, out = set(), []
    for t in load(fetch_corpus(DATA / corpus / "manifest.json")):
        if t.split == "eval" and t.behaviour not in seen:
            seen.add(t.behaviour)
            out.append(t)
    return out


def body_steps(task: Task) -> tuple[list[bool], list[bool], list[str | None]]:
    """Per step: is it under a mark, is the innermost frame defined on every example, its tag."""
    walks = example_walks(task)
    body, defined, tag = [], [], []
    for step in range(len(task.program)):
        per = [views.get(step, ()) for _states, views in walks]
        inside = bool(per[0])
        inner = [v[-1] for v in per] if inside else []
        tags = {t for t, _role, _virtual in inner}
        body.append(inside)
        defined.append(
            inside
            and len(tags) == 1
            and next(iter(tags)) in COMBINATORS
            and all(virtual is not None for _t, _r, virtual in inner)
        )
        tag.append(inner[0][0] if inside else None)
    return body, defined, tag


def frame_legal(frames: Sequence[tuple], name: str) -> bool:
    """Does op ``name`` run on every example's frame, as E1 advances a body?"""
    return all(
        not isinstance(run_body((Sym(name),), frame, BUDGET, PREVIEW_DEPTH), str)
        for frame in frames
    )


def walk(
    task: Task, scorer: PolicyScorer, *, frames: bool, mask: bool
) -> dict[str, Any]:
    """Per-step deficits along the witness.

    ``frames`` shows the policy the frame-conditioned state at body steps;
    ``mask`` removes, at defined body steps, every non-structural op that
    refuses on the frame. A witness op the frame refuses keeps the machine's
    mask for that step and is counted.
    """
    states = replay_states(task)
    targets = encode_targets(states[-1])
    walks = example_walks(task)
    body, defined, tag = body_steps(task)
    deficits, fallback = [], 0
    for step, name in enumerate(task.program):
        product = states[step]
        legal = np.zeros((1, len(OPS)), dtype=bool)
        for index, op in enumerate(OPS):
            if step_all_quiet(product, op)[0] is not None:
                legal[0, index] = True
        if mask and defined[step]:
            inner = [views[step][-1][2] for _states, views in walks]
            masked = legal.copy()
            for index, op in enumerate(OPS):
                if masked[0, index] and op.name not in STRUCTURAL:
                    masked[0, index] = frame_legal(inner, op.name)
            if masked[0, OP_INDEX[name]]:
                legal = masked
            else:
                fallback += 1
        shown = (
            tuple(frame_state(s[step], v.get(step, ())) for s, v in walks)
            if frames
            else product
        )
        block = item_array(
            [encode_item(s, t) for s, t in zip(shown, targets, strict=True)]
        )
        row = scorer.log_probs([block], legal)[0]
        deficits.append(float(np.max(row[legal[0]])) - float(row[OP_INDEX[name]]))
    return {
        "behaviour": task.behaviour,
        "deficits": [round(d, 5) for d in deficits],
        "body": body,
        "defined": defined,
        "tag": tag,
        "kinds": step_kinds(task),
        "program": list(task.program),
        "mask_fallback": fallback,
    }


def body_summary(rows: Sequence[dict]) -> dict[str, Any]:
    """Per-op deficit on body steps, all and split by whether the frame is defined."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for d, b, f, t in zip(
            r["deficits"], r["body"], r["defined"], r["tag"], strict=True
        ):
            if not b:
                groups["top_level"].append(d)
                continue
            groups["body"].append(d)
            groups["body_defined" if f else "body_undefined"].append(d)
            if f:
                groups[f"defined:{t}"].append(d)
    out = {}
    for key, values in sorted(groups.items()):
        out[key] = {
            "ops": len(values),
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
            "total": round(sum(values), 2),
        }
    return out


def composite(
    rows: Sequence[dict], *, scale: float = 1.0, commit: float = 0.0
) -> dict[str, int]:
    """Clears under the four decompositions, with defined-body deficits scaled.

    ``commit`` nats are added to every `[`: the price of choosing its consumer.
    """
    out = defaultdict(int)
    for r in rows:
        deficits = [
            d * scale if f else d
            for d, f in zip(r["deficits"], r["defined"], strict=True)
        ]
        if commit:
            deficits = [
                d + commit if name == "[" else d
                for d, name in zip(deficits, r["program"], strict=True)
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


def exchange_rate(rows: Sequence[dict], corpus: str) -> list[dict[str, Any]]:
    """Composite clears as defined-body deficit is scaled, with and without commitment."""
    table = []
    for scale in SCALES:
        table.append(
            {
                "scale": scale,
                "free_commitment": composite(rows, scale=scale),
                "with_commitment": composite(
                    rows, scale=scale, commit=COMMIT_NATS[corpus]
                ),
            }
        )
    return table


def scorer_for(path: Path) -> PolicyScorer:
    return PolicyScorer(load_policy(path, device="cpu"), "cpu")


def measure_cells(scorer: PolicyScorer, *, frames: bool, log) -> dict[str, Any]:
    cells = {}
    for corpus in CORPORA:
        tasks = eval_witnesses(corpus)
        variants = {}
        for mask in (False, True):
            rows = [walk(t, scorer, frames=frames, mask=mask) for t in tasks]
            key = "frame_mask" if mask else "machine_mask"
            variants[key] = {
                "body": body_summary(rows),
                "composite": composite(rows),
                "composite_with_commitment": composite(
                    rows, commit=COMMIT_NATS[corpus]
                ),
                "exchange_rate": exchange_rate(rows, corpus),
                "mask_fallbacks": sum(r["mask_fallback"] for r in rows),
                "rows": rows,
            }
            b = variants[key]["body"]
            log(
                f"{corpus} {key}: body mean {b['body']['mean']} "
                f"(defined {b['body_defined']['mean']} over {b['body_defined']['ops']} ops, "
                f"undefined {b['body_undefined']['mean']} over {b['body_undefined']['ops']}) "
                f"| composite {variants[key]['composite']}"
            )
        cells[corpus] = variants
    return cells


def predict(run_dir: Path, log) -> Path:
    """The registration's inputs, from the world arm alone: no training."""
    out = run_dir / "prediction.json"
    report = {
        "policy": str(BASELINE),
        "budget_nats": TOLERANCE,
        "commit_nats": COMMIT_NATS,
        "cells": measure_cells(scorer_for(BASELINE), frames=False, log=log),
    }
    co.write_json(out, report)
    log(f"wrote {out}")
    return out


def train(run_dir: Path, device: str, seeds: Sequence[int], log) -> None:
    """`world_arm.train` with the rows re-encoded on frames; nothing else changes."""
    tasks = wa.configure()
    fit, held = co.training_rows(tasks, rows=frame_witness_rows)
    log(f"RESULT rows: fit {len(fit)} validation {len(held)}")
    for seed in seeds:
        if hs.policy_dir(run_dir, wa.ARM, seed, wa.EPOCHS).exists():
            log(f"skip train s{seed} (snapshots exist)")
            continue
        manifest = {
            "arm": wa.ARM,
            "encoding": "frames (experiments/frame_rows.py)",
            "corpora": list(CORPORA),
            "epochs": wa.EPOCHS,
            "learning_rate": wa.LEARNING_RATE,
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
            learning_rate=wa.LEARNING_RATE,
            batch_size=wa.BATCH,
            validation_batch_size=wa.VAL_BATCH,
            seed=seed,
            device=device,
            epoch_end=hs.snapshotter(run_dir, wa.ARM, seed, manifest),
        )
        co.write_json(run_dir / "training" / f"s{seed}.json", state)
        curve = " ".join(
            f"e{r['epoch']}:{r['validation_cross_entropy']:.3f}"
            for r in state["history"]
        )
        log(f"RESULT train s{seed}: [{curve}] {time.perf_counter() - started:.0f}s")


def measure(
    run_dir: Path, arm: str, cells: Sequence[tuple[int, int]], log
) -> list[Path]:
    """Per-step deficits of the frame arm, or of the world arm it pairs against.

    ``arm`` is ``frames`` (this run's snapshots, shown frames) or ``world``
    (the world arm's snapshots, shown the machine state), so the 18-cell spread
    of both is measured the same way.
    """
    source = run_dir if arm == "frames" else BASELINE.parents[1]
    paths = []
    for seed, epoch in cells:
        out = run_dir / "measure" / f"{arm}-s{seed}-e{epoch}.json"
        if out.exists():
            log(f"skip {out.name}")
            paths.append(out)
            continue
        path = hs.policy_dir(source, wa.ARM, seed, epoch)
        log(f"measure {arm} s{seed}-e{epoch}: {path}")
        report = {
            "policy": str(path),
            "arm": arm,
            "paired_against": str(BASELINE),
            "budget_nats": TOLERANCE,
            "commit_nats": COMMIT_NATS,
            "cells": measure_cells(scorer_for(path), frames=arm == "frames", log=log),
        }
        co.write_json(out, report)
        paths.append(out)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("predict", "train", "measure"))
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default=",".join(str(s) for s in wa.SEEDS))
    parser.add_argument("--arm", choices=("frames", "world"), default="frames")
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
    elif args.stage == "train":
        train(args.run_dir, args.device, [int(s) for s in args.seeds.split(",")], log)
    else:
        cells = (
            [(s, e) for s in wa.SEEDS for e in range(1, wa.EPOCHS + 1)]
            if args.cells == "all"
            else [tuple(int(x) for x in c.split(":")) for c in args.cells.split(",")]
        )
        measure(args.run_dir, args.arm, cells, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
