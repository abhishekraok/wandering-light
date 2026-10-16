"""Registered #185 two-source hash compatibility/process-clock pilot; see DESIGN."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from fractions import Fraction
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

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "ce7662de0b632ce2e97f4f2d1bfb37cf08d8bff1"
ORDER = (
    ("control", 17),
    ("candidate", 17),
    ("candidate", 29),
    ("control", 29),
    ("control", 43),
    ("candidate", 43),
)
MODULES = (
    "experiments.list_hash_memo_run",
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
FUNCTIONS = {
    "beam_solve": ("wandering_light.stack_beam", "beam_solve"),
    "step_raw": ("wandering_light.stack_machine", "step_raw"),
    "run_body": ("wandering_light.stack_machine", "run_body"),
    "parse_stack": ("wandering_light.stack_machine", "parse_stack"),
    "ops_for_basis": ("wandering_light.stack_machine", "ops_for_basis"),
    "load_policy": ("wandering_light.stack_policy", "load_policy"),
    "node_encoder": ("wandering_light.stack_policy", "node_encoder"),
    "logits": ("wandering_light.stack_policy", "PolicyScorer.logits"),
    "log_probs": ("wandering_light.stack_policy", "PolicyScorer.log_probs"),
    "pooled_features": (
        "wandering_light.stack_policy",
        "StackPairPolicy.pooled_features",
    ),
}

CHECKPOINT_SHA = {
    "model.safetensors": "2113f001e8c8edfb58382545375b881c5dc10cae033aa8f0da56881b8312fd2d",
    "policy_config.json": "d0f27ec19dd900ae5e6d78f617df11ddaf9d8b7003cbb3ec305a63b6eb391f18",
    "trainer_state.json": "a291e267474b5efa8952fe1f4be17a1fed4b95e9d58d08bb48c59cfc6fece249",
    "training_manifest.json": "0f2d2e4175d7972c5ccfb537c9b2a908e1f0a76ff6367cd59c29597bf2fb537c",
}


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_info(root):
    root = Path(root).resolve()
    files = [root / (name.replace(".", "/") + ".py") for name in MODULES]
    files.append(root / "wandering_light/evals/data/stack_pbe_v4/manifest.json")
    if git(root, "status", "--porcelain"):
        raise ValueError("source tree is dirty")
    return {
        "root": str(root),
        "head": git(root, "rev-parse", "HEAD"),
        "files": {str(p): sha(p) for p in files},
    }


def origins():
    paths = {
        name: str(Path(importlib.import_module(name).__file__).resolve())
        for name in MODULES
    }
    if any(not Path(p).is_relative_to(ROOT) for p in paths.values()):
        raise ValueError("actual primitive origin outside intended source")
    return paths


def callable_origins():
    paths = {}
    for name, (module, attributes) in FUNCTIONS.items():
        function = importlib.import_module(module)
        for attribute in attributes.split("."):
            function = getattr(function, attribute)
        path = Path(inspect.getsourcefile(inspect.unwrap(function))).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("actual primitive callable outside intended source")
        paths[name] = str(path)
    return paths


def runtime_limits():
    cgroup = Path("/proc/self/cgroup").read_text()
    rows = [line[3:] for line in cgroup.splitlines() if line.startswith("0::")]
    if len(rows) != 1 or not rows[0].endswith(".service"):
        raise ValueError("persistent unified-cgroup service required")
    group = Path("/sys/fs/cgroup") / rows[0].lstrip("/")
    memory = (group / "memory.max").read_text().strip()
    swap = (group / "memory.swap.max").read_text().strip()
    unit = Path(rows[0]).name
    properties = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "RuntimeMaxUSec",
            "-p",
            "MemoryMax",
            "-p",
            "MemorySwapMax",
        ],
        text=True,
    )
    fields = dict(line.split("=", 1) for line in properties.splitlines())
    if (
        memory != str(12 * 1024**3)
        or swap != "0"
        or fields
        != {"RuntimeMaxUSec": "20min", "MemoryMax": memory, "MemorySwapMax": "0"}
    ):
        raise ValueError("actual 1200s/12GiB/no-swap controls required")
    return {"cgroup": cgroup, "memory_max": memory, "swap_max": swap, "systemd": fields}


def validate_sources(control, candidate, control_head, candidate_head):
    control, candidate = Path(control).resolve(), Path(candidate).resolve()
    if control == candidate or candidate != ROOT:
        raise ValueError("two genuine source roots; parent must run from candidate")
    pins = {"control": source_info(control), "candidate": source_info(candidate)}
    if (
        pins["control"]["head"] != control_head
        or pins["candidate"]["head"] != candidate_head
    ):
        raise ValueError("source head differs from frozen reviewed revision")
    if git(control, "diff", BASELINE, control_head, "--", "wandering_light"):
        raise ValueError("control production differs from fixed baseline")
    changes = git(
        candidate,
        "diff",
        "--name-only",
        control_head,
        candidate_head,
        "--",
        "wandering_light",
    ).splitlines()
    if changes != ["wandering_light/stack_machine.py"]:
        raise ValueError("candidate must change only the machine's Lst mechanism")
    if (control / "experiments/list_hash_memo_run.py").read_bytes() != Path(
        __file__
    ).read_bytes():
        raise ValueError("different experiment driver between arms")
    return pins


def select_panel(tasks, count=24):
    first = {}
    for i, task in enumerate(tasks):
        if (
            task.split != "train"
            or not isinstance(task.skeleton, str)
            or not task.skeleton
            or not task.inputs
            or len(task.inputs) != len(task.outputs)
        ):
            raise ValueError("invalid original train metadata/shown pairs")
        first.setdefault(task.skeleton, i)
    if len(first) < count:
        raise ValueError("insufficient skeletons; no refill")
    ordered = sorted(
        first,
        key=lambda s: (
            hashlib.sha256(b"list-hash-memo-v1\0" + s.encode()).hexdigest(),
            s,
        ),
    )[:count]
    identities, shown = [], []
    for s in ordered:
        i = first[s]
        pairs = {"inputs": list(tasks[i].inputs), "outputs": list(tasks[i].outputs)}
        identities.append(
            {
                "source_index": i,
                "skeleton": s,
                "behaviour": tasks[i].behaviour,
                "shown_sha256": hashlib.sha256(
                    json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        shown.append({"source_index": i, **pairs})
    return identities, shown


def checkpoint_pins(checkpoint):
    checkpoint = Path(checkpoint)
    pins = {p.name: sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
    if any(pins.get(name) != digest for name, digest in CHECKPOINT_SHA.items()):
        raise ValueError("fixed checkpoint bytes do not match accepted pins")
    return pins


def run_cell(shown_path, checkpoint, out, arm, label):
    import torch

    from wandering_light.stack_beam import beam_solve
    from wandering_light.stack_machine import run_body
    from wandering_light.stack_policy import PolicyScorer, load_policy

    limits = runtime_limits()
    if (
        any(os.environ.get(k) != "1" for k in THREAD_ENV)
        or os.environ.get("PYTHONHASHSEED") != "0"
    ):
        raise ValueError("one thread and PYTHONHASHSEED0 required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before, model_pin = source_info(ROOT), checkpoint_pins(checkpoint)
    input_pin, module_origins = sha(shown_path), origins()
    function_pins = callable_origins()
    shown = read(shown_path)
    if len(shown) != 24:
        raise ValueError("24 fixed shown-only tasks required")
    scorer = PolicyScorer(load_policy(Path(checkpoint), device="cpu"), "cpu")
    run_body.cache_clear()
    gc.freeze()
    rows = []
    for entry in shown:
        if set(entry) != {"source_index", "inputs", "outputs"}:
            raise ValueError("child exposes non-shown metadata")
        task = ShownTask(tuple(entry["inputs"]), tuple(entry["outputs"]))
        t0 = time.perf_counter()
        result = beam_solve(
            task,
            scorer,
            beam_width=64,
            max_depth=48,
            guard="both",
            max_state_atoms=16384,
            budget=100000,
            budget_unit="executions",
            deadline=None,
        )
        row = {
            "source_index": entry["source_index"],
            **asdict(result),
            "solve_seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if (
        source_info(ROOT) != before
        or checkpoint_pins(checkpoint) != model_pin
        or sha(shown_path) != input_pin
        or callable_origins() != function_pins
        or runtime_limits() != limits
    ):
        raise ValueError("source/model/input/runtime changed during cell")
    save(
        out,
        {
            "complete": True,
            "arm": arm,
            "label": label,
            "rows": rows,
            "source": before,
            "checkpoint": model_pin,
            "shown_sha256": input_pin,
            "module_origins": module_origins,
            "callable_origins": function_pins,
            "list_hash_code": importlib.import_module(
                "wandering_light.stack_machine"
            ).Lst.__hash__.__code__.co_filename,
            "runtime": limits,
            "host": platform.node(),
            "threads": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
            "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "torch": torch.__version__,
            "device": str(scorer.device),
            "model_parameter_dtypes": sorted(
                {str(p.dtype) for p in scorer.model.parameters()}
            ),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def paired_summary(cells, clocks, shown, pins, model_pin, limits):
    """All timing cells first; then exact paired outcomes and shown integrity."""
    from wandering_light.stack_beam import replay_program
    from wandering_light.stack_machine import parse_stack

    expected = {f"{arm}-{label}" for arm, label in ORDER}
    indices = [r["source_index"] for r in shown]
    if (
        set(cells) != expected
        or set(clocks) != expected
        or len(indices) != 24
        or len(set(indices)) != 24
    ):
        raise ValueError("complete fixed six-cell/24-task population required")
    for arm, label in ORDER:
        name = f"{arm}-{label}"
        cell = cells[name]
        if (
            cell["complete"] is not True
            or cell["arm"] != arm
            or cell["label"] != label
            or cell["source"] != pins[arm]
            or cell["checkpoint"] != model_pin
            or cell["runtime"] != limits
            or cell["host"] != platform.node()
            or cell["threads"] != 1
            or cell["interop"] != 1
            or cell["hash_seed"] != "0"
            or cell["device"] != "cpu"
            or cell["model_parameter_dtypes"] != ["torch.float32"]
            or cell["thread_env"] != dict.fromkeys(THREAD_ENV, "1")
            or [r["source_index"] for r in cell["rows"]] != indices
        ):
            raise ValueError("cell identity/source/runtime/process/population mismatch")
        expected_origins = {
            m: str(Path(pins[arm]["root"]) / (m.replace(".", "/") + ".py"))
            for m in MODULES
        }
        if cell["module_origins"] != expected_origins:
            raise ValueError("actual cell origins differ from frozen source")
        if (
            set(cell["callable_origins"]) != set(FUNCTIONS)
            or any(
                path not in pins[arm]["files"]
                for path in cell["callable_origins"].values()
            )
            or cell["list_hash_code"]
            != (
                "<string>"
                if arm == "control"
                else str(Path(pins[arm]["root"]) / "wandering_light/stack_machine.py")
            )
        ):
            raise ValueError("wrong primitive callable or hash implementation origin")
        if not math.isfinite(clocks[name]) or clocks[name] <= 0:
            raise ValueError("invalid full subprocess clock")
        for entry, row in zip(shown, cell["rows"], strict=True):
            if row["termination"] not in {
                "fit",
                "budget",
                "max_depth",
                "frontier_exhausted",
            } or (row["program"] is not None) != (row["termination"] == "fit"):
                raise ValueError("invalid/unexpected terminal")
            if (
                set(row)
                != {
                    "source_index",
                    "program",
                    "termination",
                    "seconds",
                    "executions",
                    "fuel",
                    "expansions",
                    "depth",
                    "pruned",
                    "solve_seconds",
                }
                or any(
                    type(row[k]) is not int or row[k] < 0
                    for k in (
                        "source_index",
                        "executions",
                        "fuel",
                        "expansions",
                        "depth",
                    )
                )
                or type(row["pruned"]) is not bool
                or (
                    row["program"] is not None
                    and (
                        not isinstance(row["program"], list)
                        or len(row["program"]) != row["depth"]
                    )
                )
            ):
                raise ValueError("invalid substantive result fields")
            for field in ("seconds", "solve_seconds"):
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError("invalid solve clock")
            if row["program"] is not None:
                ins = tuple(parse_stack(x, basis="mark-28") for x in entry["inputs"])
                outs = tuple(parse_stack(x, basis="mark-28") for x in entry["outputs"])
                if not replay_program(tuple(row["program"]), ins, outs)[0]:
                    raise ValueError("returned program fails shown replay")
    pairs = []
    for label in (17, 29, 43):
        a, b = cells[f"control-{label}"], cells[f"candidate-{label}"]

        def clean(row):
            return {
                k: v for k, v in row.items() if k not in {"seconds", "solve_seconds"}
            }

        equal = all(
            clean(x) == clean(y) for x, y in zip(a["rows"], b["rows"], strict=True)
        )
        ratio = Fraction.from_float(clocks[f"candidate-{label}"]) / Fraction.from_float(
            clocks[f"control-{label}"]
        )
        pairs.append(
            {
                "label": label,
                "results_equal": equal,
                "process_ratio": float(ratio),
                "process_ratio_exact": [ratio.numerator, ratio.denominator],
                "clock_pass": ratio <= Fraction(9, 10),
            }
        )
    compatible = all(p["results_equal"] for p in pairs)
    faster = all(p["clock_pass"] for p in pairs)
    return {
        "pairs": pairs,
        "exact_results_equal": compatible,
        "every_process_ratio_at_most_nine_tenths": faster,
        "reading": "QUALIFY_RELEASE_COHORT"
        if compatible and faster
        else "PARK_FIXED_CONSTRUCTION",
    }


def prepare_inputs(out):
    from wandering_light.corpus_hub import fetch_corpus
    from wandering_light.stack_pbe import load
    from wandering_light.stack_policy_hub import fetch_stack_policy

    data = fetch_corpus(
        ROOT / "wandering_light/evals/data/stack_pbe_v4/manifest.json",
        destination=out / "inputs",
        splits=("train",),
        include_details=False,
    )
    payload = data / "stack_pbe_v4_train.jsonl.gz"
    if sha(payload) != TRAIN_SHA:
        raise ValueError("wrong original train payload")
    identities, shown = select_panel(load(payload))
    checkpoint = fetch_stack_policy(POLICY_REPO, POLICY_REV)
    return payload, checkpoint, identities, shown


def run_phase(out, control, candidate, control_head, candidate_head):
    out = out.resolve()
    if not out.is_relative_to((MAIN / "checkpoints").resolve()):
        raise ValueError("outputs belong under main checkpoints")
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        limits = runtime_limits()
        pins = validate_sources(control, candidate, control_head, candidate_head)
        payload, checkpoint, identities, shown = prepare_inputs(out)
        if len(identities) != 24 or len(shown) != 24:
            raise ValueError("fixed 24-task panel required")
        model_pin = checkpoint_pins(checkpoint)
        save(out / "selection.json", identities)
        save(out / "shown.json", shown)
        raw_pins = {
            str(out / n): sha(out / n) for n in ("selection.json", "shown.json")
        }
        save(
            out / "provenance.json",
            {
                "sources": pins,
                "checkpoint": str(checkpoint),
                "model": model_pin,
                "runtime": limits,
                "order": ORDER,
                "policy_revision": POLICY_REV,
            },
        )
        save(out / "host-before.json", host_snapshot())
        for path in (out / "provenance.json", out / "host-before.json", payload):
            raw_pins[str(path)] = sha(path)
        cells, clocks = {}, {}
        for arm, label in ORDER:
            name, root = f"{arm}-{label}", Path(pins[arm]["root"])
            env = {
                **os.environ,
                **dict.fromkeys(THREAD_ENV, "1"),
                "PYTHONPATH": str(root),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)
            cmd = [
                sys.executable,
                "-m",
                "experiments.list_hash_memo_run",
                "cell",
                "--shown",
                str(out / "shown.json"),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(out / f"{name}.json"),
                "--arm",
                arm,
                "--label",
                str(label),
            ]
            with (out / f"{name}.log").open("x") as log:
                t0 = time.perf_counter()
                subprocess.run(
                    cmd,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                clocks[name] = time.perf_counter() - t0
            cells[name] = read(out / f"{name}.json")
            if cells[name]["shown_sha256"] != raw_pins[str(out / "shown.json")]:
                raise ValueError("cell consumed different shown file")
            raw_pins[str(out / f"{name}.json")] = sha(out / f"{name}.json")
        save(out / "host-after.json", host_snapshot())
        raw_pins[str(out / "host-after.json")] = sha(out / "host-after.json")
        summary = paired_summary(cells, clocks, shown, pins, model_pin, limits)
        if (
            source_info(control) != pins["control"]
            or source_info(candidate) != pins["candidate"]
            or checkpoint_pins(checkpoint) != model_pin
            or sha(payload) != TRAIN_SHA
            or runtime_limits() != limits
            or any(sha(p) != digest for p, digest in raw_pins.items())
        ):
            raise ValueError("source/model/input/raw/runtime changed after replay")
        save(
            out / "summary.json",
            {
                "complete": True,
                **summary,
                "process_seconds": clocks,
                "raw_sha256": raw_pins,
                "phase_seconds": time.perf_counter() - start,
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
    for name in ("out", "control", "candidate"):
        phase.add_argument("--" + name, type=Path, required=True)
    phase.add_argument("--control-head", required=True)
    phase.add_argument("--candidate-head", required=True)
    cell = sub.add_parser("cell")
    for name in ("shown", "checkpoint", "out"):
        cell.add_argument("--" + name, type=Path, required=True)
    cell.add_argument("--arm", choices=("control", "candidate"), required=True)
    cell.add_argument("--label", type=int, choices=(17, 29, 43), required=True)
    a = parser.parse_args()
    if a.mode == "phase":
        run_phase(a.out, a.control, a.candidate, a.control_head, a.candidate_head)
    else:
        run_cell(a.shown, a.checkpoint, a.out, a.arm, a.label)


if __name__ == "__main__":
    main()
