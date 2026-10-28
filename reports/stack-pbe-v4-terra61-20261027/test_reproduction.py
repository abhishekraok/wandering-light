"""Offline regression checks; no model calls or generated programs execute."""

import hashlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_analysis():
    spec = importlib.util.spec_from_file_location("terra_analysis", HERE / "analyse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fails", [False, True])
def test_capture_records_current_run_even_on_failure(tmp_path, fails):
    out = tmp_path / "terra61_high_v4_sample40.json"
    real_check_output = subprocess.check_output
    real_run_path = runpy.run_path
    versions = iter(("codex-cli fixture-before", "codex-cli fixture-after"))

    def check_output(command, **kwargs):
        if list(command) == ["codex", "--version"]:
            return next(versions)
        return real_check_output(command, **kwargs)

    def baseline(path, **kwargs):
        assert path == str(ROOT / "experiments/stack_pbe_baseline.py")
        sidecar = json.loads(out.with_suffix(".provenance.json").read_text())
        assert sidecar["cli_version_before"] == "codex-cli fixture-before"
        if fails:
            raise RuntimeError("fixture failure")
        out.write_text('{"fixture": true}\n')

    expected_commit = real_check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    expected_dirty = bool(
        real_check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    with (
        patch.object(sys, "argv", [str(HERE / "capture.py"), "--out", str(out)]),
        patch("subprocess.check_output", side_effect=check_output),
        patch("subprocess.Popen", subprocess.Popen),
        patch("runpy.run_path", side_effect=baseline),
        patch.dict(os.environ, {"RUST_LOG": "fixture-trace-targets"}),
    ):
        if fails:
            with pytest.raises(RuntimeError, match="fixture failure"):
                real_run_path(str(HERE / "capture.py"), run_name="__main__")
        else:
            real_run_path(str(HERE / "capture.py"), run_name="__main__")

    record = json.loads(out.with_suffix(".provenance.json").read_text())
    assert record["source_commit"] == expected_commit
    assert record["source_dirty"] == expected_dirty
    assert record["cli_version_after"] == "codex-cli fixture-after"
    assert record["baseline_arguments"] == ["--out", str(out)]
    assert record["identity_trace_logging"] == "fixture-trace-targets"
    assert "memory_scope" in record
    assert record["finished_at"] >= record["started_at"]
    if not fails:
        assert record["artifact_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        loaded, _ = load_analysis().inference_provenance(out)
        assert loaded == record


def test_analysis_reports_absent_or_mismatched_provenance(tmp_path):
    path = tmp_path / "arm.json"
    path.write_text("{}")
    analysis = load_analysis()
    provenance, caveats = analysis.inference_provenance(path)
    assert provenance is None and caveats
    record = {"source_commit": "different-revision", "source_dirty": True}
    path.with_suffix(".provenance.json").write_text(json.dumps(record))
    provenance, caveats = analysis.inference_provenance(path)
    assert provenance == record
    assert any("digest" in caveat for caveat in caveats)
    assert any("dirty" in caveat for caveat in caveats)


def test_runner_requires_memory_scope_and_propagates_failure(tmp_path):
    for name, contents in {
        "codex": "#!/bin/sh\nprintf 'fixture-cli\\n'\n",
        "systemd-run": '#!/bin/sh\nprintf "%s\\n" "$@" > "$SCOPE_ARGS"\nprintf "%s" "$RUST_LOG" > "$TRACE_ENV"\nexit 42\n',
    }.items():
        command = tmp_path / name
        command.write_text(contents)
        command.chmod(0o755)
    args_path = tmp_path / "scope-args"
    result = subprocess.run(
        ["bash", str(HERE / "run.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "OUT": str(tmp_path / "output"),
            "EFFORTS": "low medium",
            "SCOPE_ARGS": str(args_path),
            "TRACE_ENV": str(tmp_path / "trace-env"),
            "RUST_LOG": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 42
    assert args_path.read_text().splitlines()[:7] == [
        "--user",
        "--scope",
        "-p",
        "MemoryMax=12G",
        "-p",
        "MemorySwapMax=0",
        "--",
    ]
    assert "INCOMPLETE: low arm failed" in result.stderr
    assert (tmp_path / "trace-env").read_text() == (
        "codex_api::sse::responses=trace,tungstenite::protocol=trace"
    )
    assert not (tmp_path / "output/terra61_medium_v4_sample40.log").exists()


@pytest.mark.parametrize("served", ["gpt-6.1-terra", "wrong-model"])
def test_preflight_requires_incoming_identity_and_preserves_evidence(tmp_path, served):
    from wandering_light.solver import CodexExecTokenGenerator

    out = tmp_path / "preflight.json"
    stdout = json.dumps({"type": "turn.completed", "usage": {"output_tokens": 5}})
    stderr = "Received message " + json.dumps(
        {"type": "response.completed", "response": {"model": served}}
    )
    generator = SimpleNamespace(
        command=lambda: ["fixture-codex"],
        env={},
        parse_events=CodexExecTokenGenerator.parse_events,
    )
    with (
        patch.object(sys, "argv", [str(HERE / "preflight.py"), "--out", str(out)]),
        patch("wandering_light.solver.CodexExecTokenGenerator", return_value=generator),
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout, stderr),
        ),
        patch("subprocess.check_output", return_value="fixture-version"),
        pytest.raises(SystemExit) as stopped,
    ):
        runpy.run_path(str(HERE / "preflight.py"), run_name="__main__")
    passed = served == "gpt-6.1-terra"
    assert stopped.value.code == (0 if passed else 1)
    audit = json.loads(out.read_text())
    assert audit["passed"] is passed
    assert audit["served_models"] == [served]
    assert audit["requested_model"] == "gpt-6.1-terra"
    assert audit["effort"] == "low"
    assert audit["parsed_metadata"]["tool_calls"] == 0
    assert (tmp_path / "raw/preflight/stderr.txt").read_text() == stderr
