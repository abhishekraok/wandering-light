"""Small train-only baseline/retention screen for #183; see DESIGN + Amendment A.

Parent: select once, run isolated timed cells, then grade/verify. Cell children
read only shown examples and the fixed checkpoint. No oracle work runs between
primary timing cells. Neither this module nor its tests fetch data on import.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/abhishekrao/repos/wandering-light")
TRAIN_SHA = "13450717aadfce2ec16e890f3e9509514cf4c82bdff3d1199006854fe80c683b"
POLICY_REPO = "abhishekraok/wandering-light-stack-policy"
POLICY_REV = "e5a0b1391c5268c3ae04a49e6862f19a30779bb2"
SEEDS = (17, 29, 43)
ORDER = (
    ("baseline", 17),
    ("candidate", 17),
    ("candidate", 29),
    ("baseline", 29),
    ("baseline", 43),
    ("candidate", 43),
)
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    # The complete cell/phase file exists only on normal completion. An
    # interrupted exclusive write is still incomplete/unparseable, never valid.
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class ShownTask:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def select_panel(tasks, count=24):
    """First original source row per skeleton, then outcome-independent hash."""
    groups = {}
    for index, task in enumerate(tasks):
        if (
            task.split != "train"
            or not isinstance(task.skeleton, str)
            or not task.skeleton
        ):
            raise ValueError("missing train/skeleton identity")
        if not task.inputs or len(task.inputs) != len(task.outputs):
            raise ValueError("malformed shown pairs")
        groups.setdefault(task.skeleton, index)
    if len(groups) < count:
        raise ValueError("insufficient skeleton population; no refill")
    ordered = sorted(
        groups,
        key=lambda s: (
            hashlib.sha256(b"gumbel-retention-v1\0" + s.encode()).hexdigest(),
            s,
        ),
    )
    identities, shown = [], []
    for skeleton in ordered[:count]:
        index = groups[skeleton]
        task = tasks[index]
        pairs = {"inputs": list(task.inputs), "outputs": list(task.outputs)}
        identities.append(
            {
                "source_index": index,
                "skeleton": skeleton,
                "behaviour": task.behaviour,
                "shown_sha256": hashlib.sha256(
                    json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        shown.append({"source_index": index, **pairs})
    return identities, shown


def baseline_gate(counts):
    if len(counts) != 3 or any(
        not isinstance(c, int) or not 0 <= c <= 24 for c in counts
    ):
        raise ValueError("three complete baseline counts required")
    if max(counts) > 22:
        return "NO_HEADROOM"
    if max(counts) - min(counts) >= 2:
        return "BASELINE_UNSTABLE"
    return "READY_FOR_REGISTRATION"


def comparison_gate(cells):
    """Counts are exact integers; ratios use exact saved binary-float values."""
    deltas, ratios, pairs = [], [], []
    for seed in SEEDS:
        b, g = cells[f"baseline-{seed}"], cells[f"candidate-{seed}"]
        delta = g["verified_count"] - b["verified_count"]
        bt, gt = b["solve_seconds"], g["solve_seconds"]
        if not math.isfinite(bt) or not math.isfinite(gt) or bt <= 0 or gt < 0:
            raise ValueError("invalid solve clock")
        ratio = Fraction.from_float(float(gt)) / Fraction.from_float(float(bt))
        deltas.append(delta)
        ratios.append(ratio)
        pairs.append(
            {
                "seed": seed,
                "net_gain": delta,
                "ratio": float(ratio),
                "new": sorted(set(g["verified"]) - set(b["verified"])),
                "lost": sorted(set(b["verified"]) - set(g["verified"])),
            }
        )
    passed = (
        statistics.median(deltas) >= 2
        and min(deltas) >= 0
        and statistics.median(ratios) <= Fraction(11, 10)
    )
    return {
        "reading": "POSITIVE_TRAIN_SCREEN" if passed else "NEGATIVE_TRAIN_SCREEN",
        "pairs": pairs,
        "median_gain": statistics.median(deltas),
        "median_clock_ratio": float(statistics.median(ratios)),
    }


def provenance(checkpoint):
    files = [
        ROOT / "experiments/gumbel_retention.py",
        Path(__file__),
        ROOT / "wandering_light/stack_beam.py",
        ROOT / "wandering_light/stack_machine.py",
        ROOT / "wandering_light/quotation_guard.py",
        ROOT / "wandering_light/stack_pbe_eval.py",
        ROOT / "wandering_light/corpus_hub.py",
        ROOT / "wandering_light/stack_pbe.py",
        ROOT / "wandering_light/stack_policy.py",
        ROOT / "experiments/step3_verify_solves.py",
    ]
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "host": platform.node(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "platform": platform.platform(),
        "source": {str(p): sha(p) for p in files},
        "checkpoint": {
            str(p): sha(p) for p in sorted(Path(checkpoint).iterdir()) if p.is_file()
        },
        "policy_repo": POLICY_REPO,
        "policy_revision": POLICY_REV,
        "python": sys.version,
    }


def run_cell(shown_path, checkpoint, out, arm, seed):
    # Parent launches a fresh process with these variables set before numpy/torch
    # import. Refuse an accidentally different thread setup.
    import torch

    from experiments.gumbel_retention import jitter_beam_solve
    from wandering_light.stack_beam import beam_solve
    from wandering_light.stack_machine import run_body
    from wandering_light.stack_policy import PolicyScorer, load_policy

    started = time.perf_counter()
    if any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("child thread environment must be one")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scorer = PolicyScorer(load_policy(Path(checkpoint), device="cpu"), "cpu")
    shown = read(shown_path)
    if len(shown) != 24:
        raise ValueError("cell requires the fixed 24-row shown panel")
    run_body.cache_clear()
    gc.freeze()
    rows = []
    load_at_start = os.getloadavg()
    for entry in shown:
        if set(entry) != {"source_index", "inputs", "outputs"}:
            raise ValueError("solver input must be shown-only")
        task = ShownTask(tuple(entry["inputs"]), tuple(entry["outputs"]))
        t0 = time.perf_counter()
        kwargs = {
            "beam_width": 64,
            "max_depth": 48,
            "guard": "both",
            "max_state_atoms": 16384,
            "deadline": t0 + 5.0,
        }
        if arm == "baseline":
            result = beam_solve(task, scorer, **kwargs)
        else:
            rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([seed, entry["source_index"]]))
            )
            result = jitter_beam_solve(task, scorer, rng=rng, **kwargs)
        elapsed = time.perf_counter() - t0
        row = {
            "source_index": entry["source_index"],
            **asdict(result),
            "solve_seconds": elapsed,
        }
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    save(
        out,
        {
            "complete": True,
            "arm": arm,
            "seed": seed,
            "rows": rows,
            "process_seconds": time.perf_counter() - started,
            "load_average_start": load_at_start,
            "load_average_end": os.getloadavg(),
            "threads": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
            "host": platform.node(),
            "module_origins": {
                name: str(sys.modules[name].__file__)
                for name in (
                    "experiments.gumbel_retention",
                    "wandering_light.stack_beam",
                    "wandering_light.stack_policy",
                    "wandering_light.stack_machine",
                    "wandering_light.quotation_guard",
                )
            },
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    )


def summarize(cells, identities, originals, verify_fn=None):
    """After *all* timing cells, replay/grade/check distinct returned programs."""
    from experiments.step3_verify_solves import check
    from wandering_light.stack_beam import replay_program
    from wandering_light.stack_machine import parse_stack
    from wandering_light.stack_pbe import BASIS, CATALOGUE
    from wandering_light.stack_pbe_eval import grade

    checker = check if verify_fn is None else verify_fn
    catalogue = {b.name: b for b in CATALOGUE}
    selected = [d["source_index"] for d in identities]
    verdicts, summaries = {}, {}
    hosts = {cell["host"] for cell in cells.values()}
    if len(hosts) != 1:
        raise ValueError("mixed-host timing cells")
    for name, cell in cells.items():
        if (
            not cell["complete"]
            or [r["source_index"] for r in cell["rows"]] != selected
        ):
            raise ValueError("incomplete/reordered timing cell")
        good, graded, kinds = (
            set(),
            set(),
            {
                k: set()
                for k in (
                    "genuine",
                    "equals_catalogue_witness",
                    "shortcut",
                    "inconclusive",
                )
            },
        )
        for ident, row in zip(identities, cell["rows"], strict=True):
            if row["program"] is None:
                continue
            task = originals[ident["source_index"]]
            program = tuple(row["program"])
            inputs = tuple(parse_stack(x, basis=BASIS) for x in task.inputs)
            outputs = tuple(parse_stack(x, basis=BASIS) for x in task.outputs)
            if not replay_program(program, inputs, outputs)[0]:
                raise ValueError("returned program fails independent shown replay")
            scored = grade(program, task)
            if scored.solved:
                graded.add(task.behaviour)
            key = (task.behaviour, program)
            if key not in verdicts:
                verdicts[key] = checker(
                    program,
                    catalogue[task.behaviour],
                    10000,
                    0,
                    dense=True,
                    min_tested=500,
                )
            kind = verdicts[key]["verdict"]
            kinds[kind].add(task.behaviour)
            if scored.solved and kind in ("genuine", "equals_catalogue_witness"):
                good.add(task.behaviour)
        summaries[name] = {
            "verified_count": len(good),
            "verified": sorted(good),
            "graded": sorted(graded),
            "verified_draw_tested": sorted(good & kinds["genuine"]),
            "verified_witness_equal": sorted(good & kinds["equals_catalogue_witness"]),
            "kinds": {k: sorted(v) for k, v in kinds.items()},
            "solve_seconds": math.fsum(r["solve_seconds"] for r in cell["rows"]),
            "executions": sum(r["executions"] for r in cell["rows"]),
            "fuel": sum(r["fuel"] for r in cell["rows"]),
            "terminations": {
                reason: sum(r["termination"] == reason for r in cell["rows"])
                for reason in sorted({r["termination"] for r in cell["rows"]})
            },
            "program_lengths": {
                str(r["source_index"]): len(r["program"])
                for r in cell["rows"]
                if r["program"] is not None
            },
        }
    return summaries, [
        {"behaviour": b, "program": list(p), **v} for (b, p), v in verdicts.items()
    ]


def host_snapshot():
    """Record external load outside all timing cells, without process arguments."""
    result = {"load_average": os.getloadavg(), "host": platform.node()}
    commands = {
        "processes": ["ps", "-eo", "pid,comm,pcpu,pmem"],
        "gpu": ["/usr/lib/wsl/lib/nvidia-smi"],
        "units": ["systemctl", "--user", "list-units", "wl-*", "--no-pager"],
    }
    for name, command in commands.items():
        try:
            child = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False
            )
            result[name] = {
                "returncode": child.returncode,
                "stdout": child.stdout,
                "stderr": child.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


def run_phase(args):
    from wandering_light.corpus_hub import fetch_corpus
    from wandering_light.stack_pbe import load
    from wandering_light.stack_policy_hub import fetch_stack_policy

    out = args.out.resolve()
    if not out.is_relative_to((MAIN / "checkpoints").resolve()):
        raise ValueError("run output belongs in main checkpoints")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        manifest = ROOT / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
        data = fetch_corpus(
            manifest,
            destination=out / "inputs",
            splits=("train",),
            include_details=False,
        )
        payload = data / "stack_pbe_v4_train.jsonl.gz"
        if sha(payload) != TRAIN_SHA:
            raise ValueError("wrong original train payload")
        originals = load(payload)
        identities, shown = select_panel(originals)
        save(out / "selection.json", identities)
        save(out / "shown.json", shown)
        if args.phase == "comparison":
            if args.baseline is None or not args.registration_url:
                raise ValueError(
                    "baseline and operative issue registration URL required"
                )
            previous = read(args.baseline / "summary.json")
            if not previous.get("complete") or previous.get("phase") != "baseline":
                raise ValueError("incomplete baseline feasibility")
            counts = [
                previous["cells"][f"baseline-{seed}"]["verified_count"]
                for seed in SEEDS
            ]
            if baseline_gate(counts) != "READY_FOR_REGISTRATION":
                raise ValueError("baseline counts fail feasibility")
            if previous["reading"] != "READY_FOR_REGISTRATION":
                raise ValueError("baseline feasibility did not pass")
            if identities != read(args.baseline / "selection.json") or shown != read(
                args.baseline / "shown.json"
            ):
                raise ValueError("comparison must retain the feasibility panel")
        checkpoint = fetch_stack_policy(POLICY_REPO, POLICY_REV)
        pins = provenance(checkpoint)
        if args.phase == "comparison":
            prior_pins = read(args.baseline / "provenance.json")
            if pins["checkpoint"] != prior_pins["checkpoint"]:
                raise ValueError("comparison checkpoint differs from baseline")
        save(out / "provenance.json", pins)
        env = {**os.environ, **dict.fromkeys(THREAD_ENV, "1"), "PYTHONPATH": str(ROOT)}
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        order = (
            tuple(("baseline", s) for s in SEEDS) if args.phase == "baseline" else ORDER
        )
        save(out / "host-before.json", host_snapshot())
        cells = {}
        for arm, seed in order:
            name = f"{arm}-{seed}"
            cmd = [
                sys.executable,
                "-m",
                "experiments.gumbel_retention_run",
                "cell",
                "--shown",
                str(out / "shown.json"),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(out / f"{name}.json"),
                "--arm",
                arm,
                "--seed",
                str(seed),
            ]
            with (out / f"{name}.log").open("x") as log:
                cell_started = time.perf_counter()
                subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                elapsed = time.perf_counter() - cell_started
            cells[name] = read(out / f"{name}.json")
            cells[name]["subprocess_wall_seconds"] = elapsed
        save(out / "host-after.json", host_snapshot())
        # Grading imports/calls start after the entire timing phase.
        summaries, verdicts = summarize(cells, identities, originals)
        if args.phase == "baseline":
            counts = [summaries[f"baseline-{s}"]["verified_count"] for s in SEEDS]
            reading = {
                "reading": baseline_gate(counts),
                "baseline_counts": counts,
                "baseline_count_range": max(counts) - min(counts),
            }
        else:
            reading = comparison_gate(summaries)
        if provenance(checkpoint) != pins or sha(payload) != TRAIN_SHA:
            raise ValueError("inputs/source/checkpoint changed during phase")
        save(out / "verification.json", verdicts)
        save(
            out / "summary.json",
            {
                "complete": True,
                "phase": args.phase,
                **reading,
                "cells": summaries,
                "registration_url": args.registration_url,
                "phase_seconds": time.perf_counter() - started,
                "subprocess_seconds": {
                    k: c["subprocess_wall_seconds"] for k, c in cells.items()
                },
            },
        )
        print(json.dumps(read(out / "summary.json"), sort_keys=True), flush=True)
    except Exception as error:
        save(
            out / "INCOMPLETE.json",
            {
                "complete": False,
                "error": repr(error),
                "elapsed": time.perf_counter() - started,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    phase = sub.add_parser("phase")
    phase.add_argument("--phase", choices=("baseline", "comparison"), required=True)
    phase.add_argument("--out", type=Path, required=True)
    phase.add_argument("--baseline", type=Path)
    phase.add_argument("--registration-url")
    cell = sub.add_parser("cell")
    cell.add_argument("--shown", type=Path, required=True)
    cell.add_argument("--checkpoint", type=Path, required=True)
    cell.add_argument("--out", type=Path, required=True)
    cell.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    cell.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args(argv)
    if args.mode == "cell":
        run_cell(args.shown, args.checkpoint, args.out, args.arm, args.seed)
    else:
        run_phase(args)


if __name__ == "__main__":
    main()
