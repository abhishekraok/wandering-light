"""Observe Claude subprocess output while executing the baseline unchanged.

Records original commands, stdout, stderr and inference-time provenance. Does
not change the CLI command, prompt, sample, parser, grader or generation count.
"""

import hashlib
import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
RAW = OUT.parent / "raw" / OUT.stem
RAW.mkdir(parents=True, exist_ok=False)
OriginalPopen = subprocess.Popen
call_numbers = count(1)


def run_metadata():
    def output(*command):
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()

    memory_scope = {}
    try:
        group = next(
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        memory_scope["cgroup"] = group
        directory = Path("/sys/fs/cgroup") / group.lstrip("/")
        for name in ("memory.max", "memory.swap.max"):
            memory_scope[name] = (directory / name).read_text().strip()
    except (OSError, StopIteration) as error:
        memory_scope["unavailable"] = str(error)

    return {
        "capture_method": "recorded at inference time",
        "source_commit": output("git", "rev-parse", "HEAD"),
        "source_dirty": bool(output("git", "status", "--porcelain")),
        "cli_version_before": output("claude", "--version"),
        "started_at": datetime.now(UTC).isoformat(),
        "baseline_arguments": sys.argv[1:],
        "memory_scope": memory_scope,
        "inference_source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "experiments/stack_pbe_baseline.py",
                "wandering_light/solver.py",
                "wandering_light/stack_pbe_eval.py",
                "wandering_light/stack_machine.py",
                "wandering_light/evals/data/stack_pbe_v4/manifest.json",
                "reports/stack-pbe-v4-fable52-20261110/capture.py",
                "uv.lock",
            )
        },
    }


class RecordingPopen(OriginalPopen):
    def __init__(self, args, *positional, **kwargs):
        self.audit_path = None
        if isinstance(args, list) and args[:2] == ["claude", "-p"]:
            self.audit_path = RAW / f"call-{next(call_numbers):03d}"
            self.audit_path.with_suffix(".command.json").write_text(
                json.dumps(args, indent=2) + "\n"
            )
        super().__init__(args, *positional, **kwargs)

    def communicate(self, *args, **kwargs):
        stdout, stderr = super().communicate(*args, **kwargs)
        if self.audit_path is not None:
            self.audit_path.with_suffix(".stdout.json").write_text(stdout or "")
            self.audit_path.with_suffix(".stderr.txt").write_text(stderr or "")
        return stdout, stderr


if __name__ == "__main__":
    provenance = run_metadata()
    provenance_path = OUT.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.Popen = RecordingPopen
    sys.argv[0] = str(ROOT / "experiments/stack_pbe_baseline.py")
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        provenance["finished_at"] = datetime.now(UTC).isoformat()
        provenance["cli_version_after"] = subprocess.check_output(
            ["claude", "--version"], text=True
        ).strip()
        if OUT.exists():
            provenance["artifact_sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
