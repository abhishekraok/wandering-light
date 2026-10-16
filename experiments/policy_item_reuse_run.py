"""Registered TRAIN-only encoded-item reuse profile (#184); no cached inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from experiments.gumbel_retention_run import (
    MAIN,
    POLICY_REPO,
    POLICY_REV,
    THREAD_ENV,
    TRAIN_SHA,
    ShownTask,
    host_snapshot,
    read,
    save,
    sha,
)
from experiments.policy_item_reuse import (
    ItemRecorder,
    RecordingScorer,
    eligibility,
    empty_totals,
    replay_counts,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = (17, 29, 43)
MODULES = (
    "experiments.policy_item_reuse",
    "experiments.gumbel_retention_run",
    "wandering_light.stack_beam",
    "wandering_light.stack_machine",
    "wandering_light.quotation_guard",
    "wandering_light.stack_policy",
    "wandering_light.rollout_policy",
    "wandering_light.stack_pbe",
    "wandering_light.corpus_hub",
    "wandering_light.stack_policy_hub",
)


def origins():
    paths = {
        name: str(Path(importlib.import_module(name).__file__).resolve())
        for name in MODULES
    }
    if any(not Path(p).is_relative_to(ROOT) for p in paths.values()):
        raise ValueError("primitive module resolves outside frozen source")
    return paths


def provenance(checkpoint):
    files = {Path(__file__).resolve(), *(Path(p) for p in origins().values())}
    files.add(ROOT / "wandering_light/evals/data/stack_pbe_v4/manifest.json")
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "host": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "policy_repo": POLICY_REPO,
        "policy_revision": POLICY_REV,
        "source": {str(p): sha(p) for p in sorted(files)},
        "checkpoint": {
            str(p): sha(p) for p in sorted(Path(checkpoint).iterdir()) if p.is_file()
        },
    }


def select_panel(tasks, count=24):
    first = {}
    for index, task in enumerate(tasks):
        if (
            task.split != "train"
            or not isinstance(task.skeleton, str)
            or not task.skeleton
            or not task.inputs
            or len(task.inputs) != len(task.outputs)
        ):
            raise ValueError("invalid original train metadata/shown pairs")
        first.setdefault(task.skeleton, index)
    if len(first) < count:
        raise ValueError("insufficient skeleton groups; no refill")
    ordered = sorted(
        first,
        key=lambda s: (
            hashlib.sha256(b"policy-item-reuse-v1\0" + s.encode()).hexdigest(),
            s,
        ),
    )[:count]
    identities, shown = [], []
    for skeleton in ordered:
        index = first[skeleton]
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


def run_cell(shown_path, checkpoint, out, label):
    import torch

    from wandering_light.stack_beam import beam_solve
    from wandering_light.stack_machine import run_body
    from wandering_light.stack_policy import load_policy

    started = time.perf_counter()
    if any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("numerical library threads must be one")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scorer = RecordingScorer(load_policy(Path(checkpoint), device="cpu"), "cpu")
    input_pin = sha(shown_path)
    shown = read(shown_path)
    if len(shown) != 24:
        raise ValueError("24 original shown-only tasks required")
    module_origins = origins()
    run_body.cache_clear()
    gc.freeze()
    rows = []
    load_start = os.getloadavg()
    for entry in shown:
        if set(entry) != {"source_index", "inputs", "outputs"}:
            raise ValueError("task exposes non-shown metadata")
        task = ShownTask(tuple(entry["inputs"]), tuple(entry["outputs"]))
        t0 = time.perf_counter()
        scorer.recorder = ItemRecorder()
        result = beam_solve(
            task,
            scorer,
            beam_width=64,
            max_depth=48,
            guard="both",
            max_state_atoms=16384,
            deadline=t0 + 5.0,
        )
        elapsed = time.perf_counter() - t0
        recorder = scorer.recorder
        row = {
            "source_index": entry["source_index"],
            **asdict(result),
            "solve_seconds": elapsed,
            "observed": dict(recorder.totals),
            "calls": recorder.calls,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "calls"}), flush=True)
    if sha(shown_path) != input_pin:
        raise ValueError("shown input changed during cell")
    save(
        out,
        {
            "complete": True,
            "label": label,
            "shown_sha256": input_pin,
            "rows": rows,
            "host": platform.node(),
            "process_seconds": time.perf_counter() - started,
            "load_average_start": load_start,
            "load_average_end": os.getloadavg(),
            "threads": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
            "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
            "cgroup": Path("/proc/self/cgroup").read_text(),
            "module_origins": module_origins,
            "torch": torch.__version__,
            "device": str(scorer.device),
            "model_parameter_dtypes": sorted(
                {str(p.dtype) for p in scorer.model.parameters()}
            ),
        },
    )


def summarize(cells, shown):
    """After ALL timing cells: counter replay and typed shown replay only."""
    from wandering_light.stack_beam import replay_program
    from wandering_light.stack_machine import parse_stack
    from wandering_light.stack_pbe import BASIS

    selected = [entry["source_index"] for entry in shown]
    if (
        len(selected) != 24
        or len(set(selected)) != 24
        or set(cells) != {str(x) for x in LABELS}
    ):
        raise ValueError("complete fixed 24-task/three-repeat population required")
    summaries = {}
    for label in LABELS:
        cell = cells[str(label)]
        if (
            not cell["complete"]
            or cell["label"] != label
            or [r["source_index"] for r in cell["rows"]] != selected
            or cell["threads"] != 1
            or cell["interop"] != 1
            or cell["host"] != platform.node()
        ):
            raise ValueError("incomplete/reordered cell or wrong threads")
        totals = empty_totals()
        fits = 0
        for entry, row in zip(shown, cell["rows"], strict=True):
            if (row["program"] is not None) != (row["termination"] == "fit"):
                raise ValueError("program/termination mismatch")
            computed = replay_counts(row["calls"])
            if computed != row["observed"]:
                raise ValueError("saved counters disagree with ordered keys")
            for key, value in computed.items():
                totals[key] += value
            if row["program"] is not None:
                inputs = tuple(parse_stack(x, basis=BASIS) for x in entry["inputs"])
                outputs = tuple(parse_stack(x, basis=BASIS) for x in entry["outputs"])
                if not replay_program(tuple(row["program"]), inputs, outputs)[0]:
                    raise ValueError("returned program fails typed shown replay")
                fits += 1
            if not math.isfinite(row["solve_seconds"]) or row["solve_seconds"] < 0:
                raise ValueError("invalid solve clock")
        summaries[str(label)] = {
            **totals,
            "shown_fits": fits,
            "solve_seconds": math.fsum(r["solve_seconds"] for r in cell["rows"]),
            "executions": sum(r["executions"] for r in cell["rows"]),
            "fuel": sum(r["fuel"] for r in cell["rows"]),
        }
    return {
        **eligibility([summaries[str(label)] for label in LABELS]),
        "cells": summaries,
    }


def prepare_inputs(out):
    from wandering_light.corpus_hub import fetch_corpus
    from wandering_light.stack_pbe import load
    from wandering_light.stack_policy_hub import fetch_stack_policy

    manifest = ROOT / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
    data = fetch_corpus(
        manifest, destination=out / "inputs", splits=("train",), include_details=False
    )
    payload = data / "stack_pbe_v4_train.jsonl.gz"
    if sha(payload) != TRAIN_SHA:
        raise ValueError("wrong original train payload")
    identities, shown = select_panel(load(payload))
    checkpoint = fetch_stack_policy(POLICY_REPO, POLICY_REV)
    return payload, checkpoint, identities, shown


def run_phase(out):
    out = out.resolve()
    if not out.is_relative_to((MAIN / "checkpoints").resolve()):
        raise ValueError("outputs belong under main checkpoints")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        payload, checkpoint, identities, shown = prepare_inputs(out)
        if len(identities) != 24 or len(shown) != 24:
            raise ValueError("fixed 24-task panel required")
        save(out / "selection.json", identities)
        save(out / "shown.json", shown)
        raw_pins = {
            str(out / name): sha(out / name)
            for name in ("selection.json", "shown.json")
        }
        pins = provenance(checkpoint)
        save(out / "provenance.json", pins)
        save(out / "host-before.json", host_snapshot())
        env = {**os.environ, **dict.fromkeys(THREAD_ENV, "1"), "PYTHONPATH": str(ROOT)}
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        cells, subprocess_seconds = {}, {}
        for label in LABELS:
            name = str(label)
            cmd = [
                sys.executable,
                "-m",
                "experiments.policy_item_reuse_run",
                "cell",
                "--shown",
                str(out / "shown.json"),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(out / f"cell-{name}.json"),
                "--label",
                name,
            ]
            with (out / f"cell-{name}.log").open("x") as log:
                t0 = time.perf_counter()
                subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                subprocess_seconds[name] = time.perf_counter() - t0
            cells[name] = read(out / f"cell-{name}.json")
            if cells[name]["shown_sha256"] != raw_pins[str(out / "shown.json")]:
                raise ValueError("cell did not consume the pinned shown file")
            raw_pins[str(out / f"cell-{name}.json")] = sha(out / f"cell-{name}.json")
        save(out / "host-after.json", host_snapshot())
        summary = summarize(cells, shown)
        if provenance(checkpoint) != pins or sha(payload) != TRAIN_SHA:
            raise ValueError("input/source/checkpoint changed during profile")
        if any(sha(path) != digest for path, digest in raw_pins.items()):
            raise ValueError("saved raw inputs/cells changed during checks")
        save(
            out / "summary.json",
            {
                "complete": True,
                "raw_sha256": raw_pins,
                **summary,
                "subprocess_seconds": subprocess_seconds,
                "phase_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(read(out / "summary.json"), sort_keys=True), flush=True)
    except Exception as error:
        save(out / "INCOMPLETE.json", {"complete": False, "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    phase = sub.add_parser("phase")
    phase.add_argument("--out", type=Path, required=True)
    cell = sub.add_parser("cell")
    cell.add_argument("--shown", type=Path, required=True)
    cell.add_argument("--checkpoint", type=Path, required=True)
    cell.add_argument("--out", type=Path, required=True)
    cell.add_argument("--label", type=int, choices=LABELS, required=True)
    args = parser.parse_args()
    if args.mode == "phase":
        run_phase(args.out)
    else:
        run_cell(args.shown, args.checkpoint, args.out, args.label)


if __name__ == "__main__":
    main()
