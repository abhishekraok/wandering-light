"""Exact-source/review and process provenance for #180; no task/model reads."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.score_gap_inputs import hash_file
from experiments.score_gap_storage import DiagnosticProvenanceError, canonical_json
from wandering_light import stack_beam, stack_machine, stack_policy
from wandering_light.stack_pbe import BASIS, OPS

THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
LAUNCH_THREADS = {name: os.environ.get(name) for name in THREAD_KEYS}
REPORT_ROOT = "reports/score-gap-feasibility-20261003"
REQUIRED_FILES = tuple(
    f"experiments/score_gap_{name}.py"
    for name in (
        "profiles",
        "lineage",
        "records",
        "storage",
        "recorder",
        "read",
        "analysis",
        "inputs",
        "provenance",
        "run",
        "verify",
    )
) + tuple(
    f"{REPORT_ROOT}/{name}"
    for name in (
        "DESIGN.md",
        "AMENDMENT_A.md",
        "AMENDMENT_B.md",
        "IMPLEMENTATION_NOTES.md",
        "INPUT_STAGE.md",
        "RUNNER.md",
        "inputs.json",
    )
)


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _fail(message):
    raise DiagnosticProvenanceError(message)


def module_origins(repo, modules, budget):
    """Actual origin must be the frozen source path, even if bytes match elsewhere."""
    origins = {}
    for name, module in sorted(modules.items()):
        budget.check()
        if not (
            name == "wandering_light"
            or name.startswith(("wandering_light.", "experiments.score_gap_"))
        ):
            continue
        actual = getattr(module, "__file__", None)
        if actual is None:
            _fail("loaded local module has no file origin")
        actual = Path(actual).resolve()
        relative = name.replace(".", "/")
        expected = repo / relative
        expected = (
            expected / "__init__.py"
            if actual.name == "__init__.py"
            else expected.with_suffix(".py")
        )
        if actual != expected or not actual.is_relative_to(repo):
            _fail(f"loaded module outside intended frozen source: {name}")
        origins[name] = {"path": str(actual), **hash_file(actual, budget)}
    return origins


def capture_source(repo, budget):
    repo = Path(repo).resolve()
    budget.check()
    revision = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        _fail("diagnostic source must be clean and frozen")
    tracked = set(_git(repo, "ls-files").splitlines())
    if any(relative not in tracked for relative in REQUIRED_FILES):
        _fail("complete tracked runner/verifier/design source is required")
    origins = module_origins(repo, dict(sys.modules), budget)
    files = set(REQUIRED_FILES)
    files.update(str(Path(pin["path"]).relative_to(repo)) for pin in origins.values())
    files.update(
        name
        for name in tracked
        if name.startswith("tests/test_score_gap_") and name.endswith(".py")
    )
    if not files <= tracked:
        _fail("untracked executed local source")
    checked = {}
    for relative in sorted(files):
        path = repo / relative
        if path.resolve() != path or not path.is_file():
            _fail("source path alias or missing source")
        checked[relative] = hash_file(path, budget)
    functions = {}
    for name, function in (
        ("beam_solve", stack_beam.beam_solve),
        ("step_raw", stack_machine.step_raw),
        ("parse_stack", stack_machine.parse_stack),
        ("PolicyScorer", stack_policy.PolicyScorer),
        ("load_policy", stack_policy.load_policy),
        ("node_encoder", stack_policy.node_encoder),
    ):
        path = Path(inspect.getsourcefile(function)).resolve()
        expected = repo / function.__module__.replace(".", "/")
        if (
            path != expected.with_suffix(".py")
            or str(path.relative_to(repo)) not in checked
        ):
            _fail("primitive callable origin differs from frozen source")
        functions[name] = {"module": function.__module__, "path": str(path)}
    budget.check()
    return {
        "schema": "SG-SOURCE1",
        "repo": str(repo),
        "revision": revision,
        "files": checked,
        "modules": origins,
        "functions": functions,
    }


def recheck_source(source, budget):
    repo = Path(source["repo"])
    if _git(repo, "rev-parse", "HEAD") != source["revision"] or _git(
        repo, "status", "--porcelain", "--untracked-files=all"
    ):
        _fail("source revision/tree changed during diagnostic")
    for relative, expected in source["files"].items():
        if hash_file(repo / relative, budget) != expected:
            _fail("frozen source bytes changed")
    current = module_origins(repo, dict(sys.modules), budget)
    if current != source["modules"]:
        _fail("loaded local module population/origins changed")
    budget.check()


def require_review(receipt_path, source, budget):
    """Only exact COMPLETE code PASS plus actual exit0/nonerror log grants entry."""
    receipt_path = Path(receipt_path).resolve()
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    expected = {
        "schema",
        "issue",
        "verdict",
        "source_revision",
        "source_files",
        "reviewer_family",
        "effort",
        "complete_code",
        "run_log",
        "report",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected
        or receipt["schema"] != "SG-CODE-REVIEW1"
        or type(receipt["issue"]) is not int
        or receipt["issue"] != 180
        or receipt["verdict"] != "COMPLETE CODE PASS"
        or receipt["source_revision"] != source["revision"]
        or canonical_json(receipt["source_files"]) != canonical_json(source["files"])
        or receipt["reviewer_family"] != "claude"
        or receipt["effort"] != "high"
        or receipt["complete_code"] is not True
    ):
        _fail("exact-source high-effort COMPLETE code PASS required")
    checked = {}
    for role in ("run_log", "report"):
        pin = receipt[role]
        if type(pin) is not dict or set(pin) != {"path", "sha256", "bytes"}:
            _fail("invalid complete-code review evidence pin")
        path = Path(pin["path"]).resolve()
        if path.is_relative_to(Path(source["repo"])):
            _fail("review artifacts must remain outside frozen run source")
        got = hash_file(path, budget)
        if type(pin["bytes"]) is not int or got != {
            "bytes": pin["bytes"],
            "sha256": pin["sha256"],
        }:
            _fail("complete-code review evidence changed")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != got["sha256"]:
            _fail("complete-code review evidence changed on read")
        checked[role] = payload
    log = checked["run_log"]
    if not log.endswith(b"\nexit 0\n"):
        _fail("complete-code reviewer did not exit0")
    cli = json.loads(log[: -len(b"\nexit 0\n")])
    result = cli.get("result", "")
    if (
        cli.get("is_error") is not False
        or type(result) is not str
        or not result.strip()
        or result.strip().splitlines()[0] != "VERDICT: COMPLETE CODE PASS"
    ):
        _fail("review did not explicitly grant COMPLETE code PASS")
    if checked["report"] != (result.strip() + "\n").encode("utf-8"):
        _fail("complete-code report differs from actual reviewer output")
    if receipt_path.read_bytes() != raw:
        _fail("complete-code receipt changed on read")
    budget.check()
    return {
        "path": str(receipt_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "receipt": receipt,
    }


def configure_cpu():
    job = job_limits()
    if any(
        LAUNCH_THREADS[name] != "1" or os.environ.get(name) != "1"
        for name in THREAD_KEYS
    ):
        _fail("OMP/MKL/OpenBLAS must be one thread before process imports")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        _fail("torch CPU thread configuration failed")
    return {
        "schema": "SG-RUNTIME1",
        "job": job,
        "device": "cpu",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "safetensors": version("safetensors"),
        "thread_env": dict(LAUNCH_THREADS),
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "torch_parallel_info": torch.__config__.parallel_info(),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "basis": BASIS,
        "ops": [op.name for op in OPS],
        "fuel_limit": stack_machine.DEFAULT_FUEL_LIMIT,
        "quotation_depth_limit": stack_machine.MAX_QUOTATION_DEPTH,
        "timing_claim": False,
    }


def check_runtime(runtime):
    if (
        any(os.environ.get(name) != "1" for name in THREAD_KEYS)
        or torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or job_limits() != runtime["job"]
    ):
        _fail("CPU/thread/systemd resource provenance changed during diagnostic")


def validate_job_limits(unit, cgroup, memory_max, properties):
    expected = "wl-score-gap-feasibility-20261003.service"
    if (
        unit != expected
        or not cgroup.endswith("/" + expected)
        or memory_max != str(4 * 1024**3)
        or properties.get("MemoryMax") != memory_max
        or properties.get("RuntimeMaxUSec") != "1h 30min"
        or properties.get("ControlGroup") != cgroup
        or properties.get("ActiveState") != "active"
    ):
        _fail("registered systemd unit must enforce exact MemoryMax4G/RuntimeMax5400")
    return {
        "unit": unit,
        "cgroup": cgroup,
        "memory_max": 4 * 1024**3,
        "runtime_max_seconds": 5400,
        "systemd_properties": properties,
    }


def job_limits():
    # This process must really be in its registered service; declaring limits
    # in a final JSON is not resource enforcement. No other service is changed.
    rows = Path("/proc/self/cgroup").read_text().splitlines()
    cgroup = next((row[3:] for row in rows if row.startswith("0::")), "")
    unit = cgroup.rsplit("/", 1)[-1]
    if unit != "wl-score-gap-feasibility-20261003.service":
        _fail("real driver must run in its registered persistent systemd job")
    memory_max = (
        (Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.max").read_text().strip()
    )
    raw = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=MemoryMax",
            "--property=RuntimeMaxUSec",
            "--property=ControlGroup",
            "--property=ActiveState",
        ],
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    properties = dict(row.split("=", 1) for row in raw.splitlines())
    return validate_job_limits(unit, cgroup, memory_max, properties)
