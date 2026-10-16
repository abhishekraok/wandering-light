"""One registered #186 TRAIN-only pilot; see reports/root-map-first-20261015/DESIGN.md."""

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
from experiments.list_hash_memo_run import CHECKPOINT_SHA

ROOT = Path(__file__).resolve().parents[1]
ORDER = (
    ("control", 17),
    ("candidate", 17),
    ("candidate", 29),
    ("control", 29),
    ("control", 43),
    ("candidate", 43),
)
MODULES = (
    "experiments.root_map_first_run",
    "experiments.root_map_first",
    "experiments.gumbel_retention_run",
    "experiments.list_hash_memo_run",
    "wandering_light.stack_beam",
    "wandering_light.stack_machine",
    "wandering_light.quotation_guard",
    "wandering_light.stack_policy",
    "wandering_light.rollout_policy",
    "wandering_light.stack_pbe",
    "wandering_light.stack_pbe_eval",
    "wandering_light.corpus_hub",
    "wandering_light.stack_policy_hub",
)
FUNCTIONS = {
    "solve_task": ("experiments.root_map_first", "solve_task"),
    "verify_wrapper": ("experiments.root_map_first", "verify_wrapper"),
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
    "grade": ("wandering_light.stack_pbe_eval", "grade"),
    "run_program": ("wandering_light.stack_pbe", "run_program"),
}


def source_info():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("dirty source tree")
    files = [ROOT / (m.replace(".", "/") + ".py") for m in MODULES]
    files += [
        ROOT / "wandering_light/evals/data/stack_pbe_v4/manifest.json",
        ROOT / "reports/root-map-first-20261015/DESIGN.md",
    ]
    return {
        "root": str(ROOT),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {str(p): sha(p) for p in files},
    }


def origins():
    modules = {
        m: str(Path(importlib.import_module(m).__file__).resolve()) for m in MODULES
    }
    functions = {}
    for name, (module, attrs) in FUNCTIONS.items():
        fun = importlib.import_module(module)
        for attr in attrs.split("."):
            fun = getattr(fun, attr)
        functions[name] = str(
            Path(inspect.getsourcefile(inspect.unwrap(fun))).resolve()
        )
    if any(
        not Path(p).is_relative_to(ROOT)
        for p in [*modules.values(), *functions.values()]
    ):
        raise ValueError("actual primitive/scorer/grader outside intended frozen tree")
    return {"modules": modules, "functions": functions}


def runtime_limits():
    cgroup = Path("/proc/self/cgroup").read_text()
    rows = [l[3:] for l in cgroup.splitlines() if l.startswith("0::")]
    if len(rows) != 1 or not rows[0].endswith(".service"):
        raise ValueError("persistent unified-cgroup service required")
    group = Path("/sys/fs/cgroup") / rows[0].lstrip("/")
    memory, swap = (
        (group / n).read_text().strip() for n in ("memory.max", "memory.swap.max")
    )
    fields = dict(
        l.split("=", 1)
        for l in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                group.name,
                "-p",
                "RuntimeMaxUSec",
                "-p",
                "MemoryMax",
                "-p",
                "MemorySwapMax",
            ],
            text=True,
        ).splitlines()
    )
    if (
        memory != "12884901888"
        or swap != "0"
        or fields
        != {"RuntimeMaxUSec": "30min", "MemoryMax": memory, "MemorySwapMax": swap}
    ):
        raise ValueError("actual 1800s/12GiB/no-swap controls required")
    return {"cgroup": cgroup, "memory_max": memory, "swap_max": swap, "systemd": fields}


def checkpoint_pins(checkpoint):
    pins = {p.name: sha(p) for p in sorted(Path(checkpoint).iterdir()) if p.is_file()}
    if any(pins.get(n) != d for n, d in CHECKPOINT_SHA.items()):
        raise ValueError("wrong fixed checkpoint bytes")
    return pins


def review_pin(path, head):
    lines = path.read_text().splitlines()
    if not lines or lines[-1] != "exit 0":
        raise ValueError("incomplete review")
    result = json.loads("\n".join(lines[:-1]))
    verdict = result.get("result", "").splitlines()
    if (
        result.get("subtype") != "success"
        or result.get("is_error") is not False
        or not verdict
        or verdict[0] != "CODE REVIEW PASS"
        or head not in result["result"]
    ):
        raise ValueError("explicit exact-source CODE REVIEW PASS required")
    return {str(path): sha(path)}


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
            raise ValueError("malformed original TRAIN population")
        first.setdefault(task.skeleton, i)
    if len(first) < count:
        raise ValueError("insufficient fixed population; no refill")
    skeletons = sorted(
        first,
        key=lambda s: (
            hashlib.sha256(b"root-map-first-v1\0" + s.encode()).hexdigest(),
            s,
        ),
    )[:count]
    identities, shown = [], []
    for s in skeletons:
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
        raise ValueError("wrong original TRAIN")
    tasks = load(payload)
    identities, shown = select_panel(tasks)
    return payload, fetch_stack_policy(POLICY_REPO, POLICY_REV), identities, shown


def run_cell(shown_path, checkpoint, out, arm, label):
    import torch

    from experiments.root_map_first import solve_task
    from wandering_light.stack_machine import run_body
    from wandering_light.stack_policy import PolicyScorer, load_policy

    if (
        any(os.environ.get(k) != "1" for k in THREAD_ENV)
        or os.environ.get("PYTHONHASHSEED") != "0"
    ):
        raise ValueError("wrong fresh-process thread/hash controls")
    limits, pins, imported = runtime_limits(), source_info(), origins()
    model_pin, shown_pin = checkpoint_pins(checkpoint), sha(shown_path)
    shown = read(shown_path)
    if len(shown) != 24 or any(
        set(e) != {"source_index", "inputs", "outputs"} for e in shown
    ):
        raise ValueError("fixed shown-only 24-task input required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scorer = PolicyScorer(load_policy(checkpoint, device="cpu"), "cpu")
    dtypes = sorted({str(p.dtype) for p in scorer.model.parameters()})
    if (
        dtypes != ["torch.float32"]
        or torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
    ):
        raise ValueError("wrong actual model/threads")
    run_body.cache_clear()
    gc.freeze()
    rows = []
    for e in shown:
        row = {
            "source_index": e["source_index"],
            **solve_task(
                ShownTask(tuple(e["inputs"]), tuple(e["outputs"])), scorer, arm
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    if (
        source_info() != pins
        or origins() != imported
        or runtime_limits() != limits
        or checkpoint_pins(checkpoint) != model_pin
        or sha(shown_path) != shown_pin
    ):
        raise ValueError("cell provenance changed")
    save(
        out,
        {
            "complete": True,
            "arm": arm,
            "label": label,
            "rows": rows,
            "source": pins,
            "origins": imported,
            "checkpoint": model_pin,
            "runtime": limits,
            "shown_sha256": shown_pin,
            "host": platform.node(),
            "threads": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
            "thread_env": {k: os.environ[k] for k in THREAD_ENV},
            "hash_seed": "0",
            "device": "cpu",
            "model_parameter_dtypes": dtypes,
            "torch": torch.__version__,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def validate_row(row, entry, arm):
    from experiments.root_map_first import BODY_SECONDS, TASK_SECONDS, derive

    if (
        row["source_index"] != entry["source_index"]
        or any(
            not math.isfinite(row[k]) or row[k] < 0
            for k in ("seconds", "task_started", "task_deadline", "body_deadline")
        )
        or row["task_deadline"] != row["task_started"] + TASK_SECONDS
        or row["body_deadline"] != row["task_started"] + BODY_SECONDS
    ):
        raise ValueError("wrong task identity/clock/deadlines")
    program = row["program"]
    if (
        row["termination"]
        not in {"fit", "wall_clock_cap", "max_depth", "frontier_exhausted"}
        or (program is not None) != (row["termination"] == "fit")
        or (
            program is not None
            and (
                not isinstance(program, list)
                or not 0 < len(program) <= 48
                or any(not isinstance(p, str) for p in program)
            )
        )
    ):
        raise ValueError("malformed result terminal/program")
    stages = row["stages"]
    names = [s["stage"] for s in stages]
    if arm == "control":
        if (
            names != ["control"]
            or row["source"] != "control"
            or row["eligibility"] is not None
        ):
            raise ValueError("wrong control stage population")
    else:
        template = derive(ShownTask(tuple(entry["inputs"]), tuple(entry["outputs"])))
        if (
            not names
            or names[0] != "classify"
            or row["eligibility"] != template.reason
            or row["invocations"] != template.invocations
            or stages[0]["reason"] != template.reason
        ):
            raise ValueError("wrong candidate classification")
        allowed = (
            ["classify"]
            if template.reason != "eligible"
            else ["classify", "empty_body", "verify"]
            if template.empty_body
            else None
        )
        if allowed is not None:
            expected = allowed + (["fallback"] if row["source"] == "fallback" else [])
            if names != expected:
                raise ValueError("wrong empty/ineligible stage sequence")
        elif names[1:] not in (
            ["body"],
            ["body_skipped"],
            ["body", "verify"],
            ["body", "fallback"],
            ["body_skipped", "fallback"],
            ["body", "verify", "fallback"],
        ):
            raise ValueError("wrong body/verify/fallback order")
        if row["source"] == "root_map":
            if (
                names[-1] != "verify"
                or stages[-1]["fit"] is not True
                or stages[-1]["program"] != program
            ):
                raise ValueError("unverified root proposal")
        elif row["source"] == "none":
            if program is not None or row["termination"] != "wall_clock_cap":
                raise ValueError("wrong exhausted result")
        elif row["source"] != "fallback":
            raise ValueError("unknown proposal source")
    for stage in stages:
        if not math.isfinite(stage["elapsed_seconds"]) or stage["elapsed_seconds"] < 0:
            raise ValueError("invalid stage clock")
        if stage["stage"] in {"body", "fallback", "control"}:
            body = stage["stage"] == "body"
            expected_deadline = row["body_deadline"] if body else row["task_deadline"]
            if stage["deadline"] != expected_deadline:
                raise ValueError("wrong stage deadline")
            if stage["depth_limit"] != (45 if body else 48):
                raise ValueError("wrong search depth")
            if any(
                type(stage[k]) is not int or stage[k] < 0
                for k in ("executions", "fuel", "expansions", "depth")
            ):
                raise ValueError("invalid search charges")
    if row["source"] == "fallback" and names[-1] != "fallback":
        raise ValueError("fallback source without fallback stage")
    if "verify" in names:
        verification = next(s for s in stages if s["stage"] == "verify")
        if (
            type(verification["fit"]) is not bool
            or verification["fit"] != (verification["reason"] == "fit")
            or verification["reason"]
            not in {"fit", "shown_mismatch", "wall_clock_cap", "refusal"}
            or any(
                type(verification[k]) is not int or verification[k] < 0
                for k in ("executions", "fuel", "attempts", "completed_ops")
            )
        ):
            raise ValueError("invalid verification terminal/charges")

        body_stage = next(s for s in stages if s["stage"] in {"body", "empty_body"})
        if body_stage["program"] is None or verification["program"] != [
            "[",
            *body_stage["program"],
            "]",
            "map",
        ]:
            raise ValueError("verified program differs from one proposed body")
    if row["source"] in {"control", "fallback"} and (
        stages[-1]["program"] != program
        or stages[-1]["termination"] != row["termination"]
    ):
        raise ValueError("returned program differs from final search")


def paired_summary(
    cells, clocks, shown, originals, identities, pins, imported, model, limits
):
    from experiments.root_map_first import verify_wrapper
    from wandering_light.stack_machine import parse_stack
    from wandering_light.stack_pbe_eval import grade

    expected = {f"{a}-{n}" for a, n in ORDER}
    if (
        set(cells) != expected
        or set(clocks) != expected
        or len(shown) != 24
        or len(identities) != 24
    ):
        raise ValueError("complete six-cell/24 population required")
    indices = [e["source_index"] for e in shown]
    if len(set(indices)) != 24 or indices != [e["source_index"] for e in identities]:
        raise ValueError("wrong selected population")
    verified, grades = {}, {}
    for arm, label in ORDER:
        name = f"{arm}-{label}"
        cell = cells[name]
        if (
            cell["complete"] is not True
            or cell["arm"] != arm
            or cell["label"] != label
            or cell["source"] != pins
            or cell["origins"] != imported
            or cell["checkpoint"] != model
            or cell["runtime"] != limits
            or cell["threads"] != 1
            or cell["interop"] != 1
            or cell["thread_env"] != dict.fromkeys(THREAD_ENV, "1")
            or cell["hash_seed"] != "0"
            or cell["device"] != "cpu"
            or cell["model_parameter_dtypes"] != ["torch.float32"]
            or cell["host"] != platform.node()
            or not math.isfinite(clocks[name])
            or clocks[name] <= 0
            or len(cell["rows"]) != 24
        ):
            raise ValueError("incomplete/wrong-source/runtime cell")
        verified[name], grades[name] = [], []
        for row, entry in zip(cell["rows"], shown, strict=True):
            validate_row(row, entry, arm)
            program, score = row["program"], None
            if program is not None:
                ins = tuple(parse_stack(s, basis="mark-28") for s in entry["inputs"])
                outs = tuple(parse_stack(s, basis="mark-28") for s in entry["outputs"])
                if not verify_wrapper(program, ins, outs, float("inf"))["fit"]:
                    raise ValueError("returned path fails shown replay")
                score = grade(program, originals[entry["source_index"]])
                if score.solved:
                    verified[name].append(entry["source_index"])
            grades[name].append(
                {
                    "source_index": entry["source_index"],
                    "score": asdict(score) if score else None,
                    "verified": bool(score and score.solved),
                }
            )
    pairs = []
    for label in (17, 29, 43):
        a, b = set(verified[f"control-{label}"]), set(verified[f"candidate-{label}"])
        ratio = Fraction(clocks[f"candidate-{label}"]) / Fraction(
            clocks[f"control-{label}"]
        )
        pairs.append(
            {
                "label": label,
                "control_verified": sorted(a),
                "candidate_verified": sorted(b),
                "gained": sorted(b - a),
                "lost": sorted(a - b),
                "proper_superset": a < b,
                "process_ratio": float(ratio),
                "process_ratio_exact": [ratio.numerator, ratio.denominator],
            }
        )
    descriptive = {}
    for name, cell in cells.items():
        stages = [s["stage"] for row in cell["rows"] for s in row["stages"]]
        eligibility = [r["eligibility"] or "not_applicable" for r in cell["rows"]]
        descriptive[name] = {
            "tasks": 24,
            "verified_count": len(verified[name]),
            "returned_search_fits": sum(r["program"] is not None for r in cell["rows"]),
            "graded_shown_count": sum(
                bool(g["score"] and g["score"]["shown"]) for g in grades[name]
            ),
            "task_seconds": sum(r["seconds"] for r in cell["rows"]),
            "process_seconds": clocks[name],
            "max_rss_kib": cell["max_rss_kib"],
            "eligibility_counts": {
                k: eligibility.count(k) for k in sorted(set(eligibility))
            },
            "eligibility_denominator": 24,
            "stage_counts": {k: stages.count(k) for k in sorted(set(stages))},
        }
    return {
        "pairs": pairs,
        "grades": grades,
        "verified": verified,
        "descriptive": descriptive,
        "reading": "QUALIFY_RELEASE_COHORT"
        if all(p["proper_superset"] for p in pairs)
        else "PARK_FIXED_CONSTRUCTION",
    }


def run_phase(out, review_log, review_head):
    from wandering_light.stack_pbe import load

    out = out.resolve()
    if not out.is_relative_to((MAIN / "checkpoints").resolve()):
        raise ValueError("output belongs under main checkpoints")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        limits, pins, imported = runtime_limits(), source_info(), origins()
        if pins["head"] != review_head:
            raise ValueError("execution HEAD differs from accepted review")
        raw = review_pin(review_log.resolve(), review_head)
        payload, checkpoint, identities, shown = prepare_inputs(out)
        model = checkpoint_pins(checkpoint)
        save(out / "selection.json", identities)
        save(out / "shown.json", shown)
        save(
            out / "provenance.json",
            {
                "source": pins,
                "origins": imported,
                "checkpoint": str(checkpoint),
                "model": model,
                "runtime": limits,
                "order": ORDER,
                "policy_revision": POLICY_REV,
                "review_log": str(review_log.resolve()),
                "review_sha256": sha(review_log),
            },
        )
        save(out / "host-before.json", host_snapshot())
        for path in [
            payload,
            *(
                out / n
                for n in (
                    "selection.json",
                    "shown.json",
                    "provenance.json",
                    "host-before.json",
                )
            ),
        ]:
            raw[str(path)] = sha(path)
        cells, clocks = {}, {}
        for arm, label in ORDER:
            name = f"{arm}-{label}"
            env = {
                **os.environ,
                **dict.fromkeys(THREAD_ENV, "1"),
                "PYTHONPATH": str(ROOT),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)
            cmd = [
                sys.executable,
                "-m",
                "experiments.root_map_first_run",
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
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                clocks[name] = time.perf_counter() - t0
            cells[name] = read(out / f"{name}.json")
            if cells[name]["shown_sha256"] != raw[str(out / "shown.json")]:
                raise ValueError("cell consumed different shown bytes")
            for path in (out / f"{name}.json", out / f"{name}.log"):
                raw[str(path)] = sha(path)
        save(out / "host-after.json", host_snapshot())
        raw[str(out / "host-after.json")] = sha(out / "host-after.json")
        # First original-data replay/grading/analysis happens only after ALL six timing cells.
        summary = paired_summary(
            cells,
            clocks,
            shown,
            load(payload),
            identities,
            pins,
            imported,
            model,
            limits,
        )
        if (
            source_info() != pins
            or origins() != imported
            or checkpoint_pins(checkpoint) != model
            or runtime_limits() != limits
            or sha(payload) != TRAIN_SHA
            or any(sha(p) != d for p, d in raw.items())
        ):
            raise ValueError(
                "source/runtime/model/input/raw/review evidence changed after replay"
            )
        save(
            out / "summary.json",
            {
                "complete": True,
                **summary,
                "process_seconds": clocks,
                "phase_seconds": time.perf_counter() - started,
                "raw_sha256": raw,
            },
        )
        print(
            json.dumps(read(out / "summary.json"), sort_keys=True, allow_nan=False),
            flush=True,
        )
    except Exception as error:
        save(out / "INCOMPLETE.json", {"complete": False, "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("phase")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--review-log", type=Path, required=True)
    p.add_argument("--review-head", required=True)
    p = sub.add_parser("cell")
    for name in ("shown", "checkpoint", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--arm", choices=("control", "candidate"), required=True)
    p.add_argument("--label", type=int, choices=(17, 29, 43), required=True)
    a = parser.parse_args()
    if a.mode == "phase":
        run_phase(a.out, a.review_log, a.review_head)
    else:
        run_cell(a.shown, a.checkpoint, a.out, a.arm, a.label)


if __name__ == "__main__":
    main()
