"""Keep raw CLI audit evidence while executing the baseline unchanged.

This observes completed Codex subprocess output only. It does not change the
CLI command, prompt, selected tasks, output parser, grader, or generation count.
"""

import json
import runpy
import subprocess
import sys
from itertools import count
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
RAW = OUT.parent / "raw" / OUT.stem
RAW.mkdir(parents=True, exist_ok=True)
OriginalPopen = subprocess.Popen
call_numbers = count(1)


class RecordingPopen(OriginalPopen):
    def __init__(self, args, *positional, **kwargs):
        self.audit_path = None
        if isinstance(args, list) and args[:2] == ["codex", "exec"]:
            self.audit_path = RAW / f"call-{next(call_numbers):03d}"
            self.audit_path.with_suffix(".command.json").write_text(
                json.dumps(args, indent=2) + "\n"
            )
        super().__init__(args, *positional, **kwargs)

    def communicate(self, *args, **kwargs):
        stdout, stderr = super().communicate(*args, **kwargs)
        if self.audit_path is not None:
            self.audit_path.with_suffix(".stdout.jsonl").write_text(stdout or "")
            self.audit_path.with_suffix(".stderr.txt").write_text(stderr or "")
        return stdout, stderr


if __name__ == "__main__":
    subprocess.Popen = RecordingPopen
    sys.argv[0] = str(ROOT / "experiments/stack_pbe_baseline.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
