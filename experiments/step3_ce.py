"""Next-op cross-entropy of a #101-recipe checkpoint on named row sets (#134).

The training run records only the cross-entropy of its own model-selection
holdout. The step-3 comparison also needs each arm scored on row sets it did
not select on — world's train holdout, and world's eval split — because
`world`'s witnesses are far longer than the search's depth cap, so solve rate
on world eval is near zero for both arms by construction and CE is the only
read with signal there.

Row sets, all built with ``witness_rows`` exactly as training does:

    v4-holdout          the 1-in-10 v4 train skeletons both arms select on
    v4-fit              the v4 rows both arms fit (a memorisation reference)
    world-holdout       world's own 1-in-10 train skeletons, in neither arm's fit
    world-eval-clean    world eval behaviours no arm has seen a program or
                        skeleton of, in v4 (either split) or in world train
    world-eval-dirty    the rest of world eval, reported separately
    world-eval-all      both of the above

Usage:
    uv run python experiments/step3_ce.py \
        --checkpoint checkpoints/step3-v4only-20260912/policies/1m-s0 \
        --checkpoint checkpoints/step3-mixed-20260912/policies/1m-s0 \
        --out reports/world-plus-v4-policy-20260912/ce.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_pbe import Task, load, skeleton  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    _mean_cross_entropy,
    load_policy,
    validation_skeleton,
    witness_rows,
)

V4 = REPO / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
WORLD = REPO / "wandering_light/evals/data/stack_pbe_world_v3/manifest.json"
BATCH = 512


def seen_identities(v4: list[Task], world: list[Task]) -> tuple[set[str], set[str]]:
    """Programs and skeletons either arm could have trained on.

    Arm A fits v4 train; arm B fits v4 train plus world train. A world eval
    behaviour whose program or skeleton appears in v4 (either split, since v4
    eval overlap is what the drop list removes from arm B only) or in world
    train is not a clean generalisation read for both arms, so it is scored
    separately rather than dropped silently.
    """
    programs, skeletons = set(), set()
    for task in v4 + [t for t in world if t.split == "train"]:
        programs.add(" ".join(task.program))
        skeletons.add(task.skeleton or skeleton(task.program))
    return programs, skeletons


def row_sets(v4: list[Task], world: list[Task]) -> dict[str, list[Task]]:
    programs, skeletons = seen_identities(v4, world)

    def clean(task: Task) -> bool:
        return (
            " ".join(task.program) not in programs
            and (task.skeleton or skeleton(task.program)) not in skeletons
        )

    v4_train = [t for t in v4 if t.split == "train"]
    world_train = [t for t in world if t.split == "train"]
    world_eval = [t for t in world if t.split == "eval"]
    return {
        "v4-holdout": [t for t in v4_train if validation_skeleton(t)],
        "v4-fit": [t for t in v4_train if not validation_skeleton(t)],
        "world-holdout": [t for t in world_train if validation_skeleton(t)],
        "world-eval-clean": [t for t in world_eval if clean(t)],
        "world-eval-dirty": [t for t in world_eval if not clean(t)],
        "world-eval-all": world_eval,
    }


def leakage(v4: list[Task], world: list[Task], dropped: set[str]) -> dict[str, Any]:
    """What the mixed arm could have memorised of the sets it is scored on.

    The model-selection holdout is drawn from v4's *train* split, so removing
    world's overlap with v4 *eval* does not by itself make the headline
    comparison clean: a world behaviour identical to a v4 holdout behaviour
    would let the mixed arm train on its own validation answers. This records
    both directions so the artifact carries the audit.
    """

    def identities(tasks: list[Task]) -> tuple[set[str], set[str]]:
        return (
            {" ".join(t.program) for t in tasks},
            {t.skeleton or skeleton(t.program) for t in tasks},
        )

    world_train = [
        t for t in world if t.split == "train" and t.behaviour not in dropped
    ]
    w_prog, w_skel = identities(world_train)
    out: dict[str, Any] = {"dropped_from_world_train": sorted(dropped)}
    for name, pool in (
        (
            "v4_holdout",
            [t for t in v4 if t.split == "train" and validation_skeleton(t)],
        ),
        (
            "v4_fit",
            [t for t in v4 if t.split == "train" and not validation_skeleton(t)],
        ),
        ("v4_eval", [t for t in v4 if t.split == "eval"]),
    ):
        behaviours = {}
        for task in pool:
            behaviours.setdefault(task.behaviour, task)
        out[name] = {
            "behaviours": len(behaviours),
            "world_train_shares_program": sorted(
                b for b, t in behaviours.items() if " ".join(t.program) in w_prog
            ),
            "world_train_shares_skeleton": sorted(
                b
                for b, t in behaviours.items()
                if (t.skeleton or skeleton(t.program)) in w_skel
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument(
        "--dropped",
        action="append",
        help="behaviour excluded from the mixed arm's training, for the audit",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    v4 = load(fetch_corpus(V4))
    world = load(fetch_corpus(WORLD))
    sets = row_sets(v4, world)
    rows = {name: witness_rows(tasks) for name, tasks in sets.items()}
    shape = {name: {"tasks": len(sets[name]), "rows": len(rows[name])} for name in sets}
    print(json.dumps(shape, indent=1))

    audit = leakage(v4, world, set(args.dropped or ()))
    for key in ("v4_holdout", "v4_eval"):
        shared = audit[key]["world_train_shares_skeleton"]
        print(f"leakage into {key}: {shared or 'none'}")
    out: dict[str, Any] = {"row_sets": shape, "leakage": audit, "checkpoints": {}}
    for checkpoint in args.checkpoint:
        model = load_policy(checkpoint, device=args.device)
        model.eval()
        # An empty set scores None rather than dividing by zero: after the
        # contaminated world eval behaviours are removed, `world-eval-dirty`
        # is empty by construction, and that emptiness is itself the check.
        scores = {
            name: _mean_cross_entropy(
                model, rows[name], args.batch, torch.device(args.device)
            )
            if rows[name]
            else None
            for name in sets
        }
        state_path = checkpoint / "trainer_state.json"
        recorded = json.loads(state_path.read_text()) if state_path.exists() else {}
        out["checkpoints"][str(checkpoint)] = {
            "cross_entropy": scores,
            "recorded_best_validation_cross_entropy": recorded.get(
                "best_validation_cross_entropy"
            ),
            "recorded_best_epoch": recorded.get("best_epoch"),
            "device": args.device,
        }
        print(
            f"{checkpoint}: "
            + " ".join(
                f"{name}={scores[name]:.4f}"
                if scores[name] is not None
                else f"{name}=empty"
                for name in sorted(scores)
            )
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
