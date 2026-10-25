"""Pipeline order, waiting and failure behavior without running subprocesses."""

import json

import pytest

from experiments.pbe_scale import finish


def ready(tmp_path):
    (tmp_path / "v3-round-000").mkdir()
    (tmp_path / "v3-round-000/result.json").write_text('{"status":"complete"}')
    (tmp_path / "campaign_status.json").write_text('{"status":"stopped"}')
    (tmp_path / "candidates").mkdir()
    (tmp_path / "candidates/batch-000000-a0.jsonl").write_text("{}\n")


@pytest.mark.parametrize("fail_at", [None, 3])
def test_subcommands_order_and_failure(tmp_path, monkeypatch, fail_at):
    ready(tmp_path)
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 9000 + len(commands)
            self.returncode = 9 if len(commands) == fail_at else 0
            assert kwargs["start_new_session"]
            assert kwargs["stdin"] == finish.subprocess.DEVNULL

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(finish.subprocess, "Popen", Process)
    monkeypatch.setattr(finish.os, "killpg", lambda *args: None)
    assert finish.run_pipeline(tmp_path, poll=0) == (1 if fail_at else 0)
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert [record["name"] for record in status["stages"]] == [
        "final-authors",
        "final-anchors",
        "merge",
        "audit",
        "augment",
        "report",
        "prepare-new",
        "train-old",
        "train-mixed",
        "freeze-eval",
        "eval-old",
        "eval-mixed",
        "compare-eval",
        "verify-eval",
    ][:fail_at]
    assert status["status"] == ("incomplete" if fail_at else "complete")
    assert all(record["pid"] and record["log"] for record in status["stages"])
    if fail_at:
        assert len(commands) == fail_at
        assert status["stages"][-1]["returncode"] == 9
    else:
        assert [command[2] for command in commands] == [
            "experiments.pbe_scale.verify_batch",
            "experiments.pbe_scale.validate",
            "experiments.pbe_scale.merge",
            "experiments.pbe_scale.audit_eval",
            "experiments.pbe_scale.augment",
            "experiments.pbe_scale.report",
            "experiments.pbe_scale.train",
            "experiments.pbe_scale.train",
            "experiments.pbe_scale.train",
            "experiments.pbe_scale.eval_compare",
            "wandering_light.solve",
            "wandering_light.solve",
            "experiments.pbe_scale.eval_compare",
            "experiments.pbe_scale.verify_eval",
        ]
        assert "--mix-data" in commands[8]
        assert all(
            command[command.index("--device") + 1] == "cuda"
            for command in commands[7:9]
        )
        assert all(
            command[command.index("--device") + 1] == "cpu"
            for command in commands[10:12]
        )
        assert not any(
            "claude" in word.lower() or word == "codex"
            for command in commands
            for word in command
        )
    with pytest.raises(FileExistsError):
        finish.run_pipeline(tmp_path)


def test_waiting_requires_both_prerequisites(tmp_path, monkeypatch):
    (tmp_path / "v3-round-000").mkdir()
    status, sleeps = {}, []

    def advance(_):
        sleeps.append(1)
        (tmp_path / "v3-round-000/result.json").write_text('{"status":"complete"}')
        if len(sleeps) == 2:
            (tmp_path / "campaign_status.json").write_text('{"status":"stopped"}')

    monkeypatch.setattr(finish.time, "sleep", advance)
    finish.wait_ready(tmp_path, tmp_path / "pipeline_status.json", status, 0)
    assert len(sleeps) == 2
    assert status["status"] == "waiting"


def test_incomplete_prerequisite_stops_immediately(tmp_path):
    (tmp_path / "v3-round-000").mkdir()
    (tmp_path / "v3-round-000/result.json").write_text('{"status":"incomplete"}')
    with pytest.raises(RuntimeError, match="incomplete"):
        finish.wait_ready(tmp_path, tmp_path / "pipeline_status.json", {}, 0)


def test_copied_finisher_uses_installed_source_checkout(tmp_path):
    import runpy
    from pathlib import Path

    copied = tmp_path / "config" / "finish-reviewed.py"
    copied.parent.mkdir()
    copied.write_text(Path(finish.__file__).read_text())
    loaded = runpy.run_path(str(copied))
    assert loaded["ROOT"] == finish.ROOT
    assert (loaded["ROOT"] / "experiments/pbe_scale/train.py").is_file()
