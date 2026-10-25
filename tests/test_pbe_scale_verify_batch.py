"""Frozen sharding and process supervision without launching a validator."""

import hashlib
import json

import pytest

from experiments.pbe_scale import verify_batch


def test_sharding_preserves_rows_deduplicates_and_freezes(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    raw = b'{ "id": "a", "program": "sum" }'
    a.write_bytes(raw + b'\n{"id":"b"}\n')
    b.write_bytes(b'{"program":"sum","id":"a"}\n{"id":"c"}\n')
    first = verify_batch.prepare([b, a], tmp_path / "first", 8)
    second = verify_batch.prepare([a, b], tmp_path / "second", 8)
    assert first["unique_candidates"] == 3
    assert [s["sha256"] for s in first["shards"]] == [
        s["sha256"] for s in second["shards"]
    ]
    payload = b"".join(
        (tmp_path / "first" / f"input-{i:02d}.jsonl").read_bytes() for i in range(8)
    )
    assert raw in payload
    assert first["sources"][str(a)] == hashlib.sha256(a.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        verify_batch.prepare([a], tmp_path / "first", 8)


@pytest.mark.parametrize(("code", "expected"), [(0, "complete"), (3, "incomplete")])
def test_supervisor_records_exact_handles_and_failures(
    tmp_path, monkeypatch, code, expected
):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"one"}\n')
    output = tmp_path / "round"
    manifest = verify_batch.prepare([source], output, 2)
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 4000 + len(launched)
            self.returncode = code
            self.command = command
            launched.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(verify_batch.subprocess, "Popen", Process)
    result = verify_batch.supervise(
        manifest, output, tmp_path / "validated", poll_seconds=0
    )
    assert result["status"] == expected
    assert len(launched) == 2
    assert [child["pid"] for child in result["children"]] == [4000, 4001]
    assert "--input" in launched[0].command
    assert "20261024" in launched[0].command
    assert json.loads((output / "result.json").read_text())["status"] == expected


def test_sigterm_cleans_up_tracked_children_and_records_incomplete(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"one"}\n')
    output = tmp_path / "round"
    manifest = verify_batch.prepare([source], output, 1)
    handlers, children = [], []
    previous = object()

    def install(signum, handler):
        handlers.append(handler)
        return previous

    class Process:
        def __init__(self, *args, **kwargs):
            self.pid, self.returncode = 4010, None
            children.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(verify_batch.signal, "signal", install)
    monkeypatch.setattr(verify_batch.subprocess, "Popen", Process)
    monkeypatch.setattr(
        verify_batch.time, "sleep", lambda seconds: handlers[0](15, None)
    )
    with pytest.raises(SystemExit):
        verify_batch.supervise(manifest, output, tmp_path / "validated")
    assert children[0].returncode == -15
    assert handlers[-1] is previous
    assert json.loads((output / "result.json").read_text())["status"] == "incomplete"
