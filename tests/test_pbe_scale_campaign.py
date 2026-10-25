"""Quota and lifecycle checks without making any model requests."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from experiments.pbe_scale import campaign


def receipt(used, when, window=10080):
    return {
        "timestamp": when,
        "payload": {
            "rate_limits": {
                "primary": {"window_minutes": 300, "used_percent": 99},
                "secondary": {"window_minutes": window, "used_percent": used},
            }
        },
    }


@pytest.mark.parametrize("remaining", [0, 4.99, 95.01, 100])
def test_quota_outside_authorization_stops(remaining):
    assert campaign.quota_problem(
        {"remaining_percent": remaining, "observed_at": 1000}, 1000, 180
    )


@pytest.mark.parametrize("remaining", [5, 52, 95])
def test_quota_boundaries_are_authorized(remaining):
    assert (
        campaign.quota_problem(
            {"remaining_percent": remaining, "observed_at": 1000}, 1000, 180
        )
        is None
    )


def test_missing_and_stale_receipts_fail_closed():
    assert campaign.quota_problem(None, 1000, 180)
    assert campaign.quota_problem(
        {"remaining_percent": 52, "observed_at": 800}, 1000, 180
    )


def test_meter_uses_newest_weekly_event_not_file_mtime(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(receipt(48, 999)) + "\n")
    second.write_text(json.dumps(receipt(20, 998)) + "\n" + "truncated junk\n")
    reading = campaign.latest_quota(tmp_path, now=1000)
    assert reading["remaining_percent"] == 52
    assert reading["observed_at"] == 999


def test_invalid_and_nonweekly_receipts_are_ignored(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        "\n".join(
            json.dumps(receipt(used, 999, window))
            for used, window in [(48, 300), (float("nan"), 10080)]
        )
    )
    assert campaign.latest_quota(tmp_path, now=1000) is None


def test_worker_is_plan_billed_read_only_and_persistent(monkeypatch, tmp_path):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.setenv(key, "must-not-reach-worker")
    assert not any(
        key.endswith("API_KEY")
        for key in campaign.worker_environment()
        if key
        in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"}
    )
    command = campaign.worker_command(tmp_path, tmp_path / "schema.json")
    assert command[command.index("-s") + 1] == "read-only"
    assert command[command.index("-m") + 1] == "gpt-6.1-sol"
    assert "model_reasoning_effort=low" in command
    assert "--ephemeral" not in command
    assert "--ignore-user-config" in command
    assert "project_doc_max_bytes=0" in command


def arguments(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path,
        deadline=datetime.now(campaign.PACIFIC) + timedelta(minutes=1),
        dry_run=False,
        quota_max_age=180,
        quota_poll=15,
        quota_grace=0,
        sessions_dir=tmp_path / "sessions",
        max_batches=1,
        max_retries=1,
        max_consecutive_failures=8,
        concurrency=8,
        rows=24,
    )


@pytest.mark.parametrize("remaining", [4, 96, None])
def test_out_of_range_or_missing_quota_never_launches(monkeypatch, tmp_path, remaining):
    reading = (
        None
        if remaining is None
        else {"remaining_percent": remaining, "observed_at": datetime.now().timestamp()}
    )
    monkeypatch.setattr(campaign, "latest_quota", lambda *args: reading)

    def forbidden(*args, **kwargs):
        pytest.fail("must not launch a worker")

    monkeypatch.setattr(campaign.subprocess, "Popen", forbidden)
    assert campaign.run(arguments(tmp_path)) == 0
    assert (
        json.loads((tmp_path / "campaign_status.json").read_text())["status"]
        == "stopped"
    )


def test_deadline_stops_without_launch(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    args.deadline = datetime.now(campaign.PACIFIC) - timedelta(seconds=1)
    assert campaign.run(args) == 0
    assert (
        "deadline"
        in json.loads((tmp_path / "campaign_status.json").read_text())["stop_reason"]
    )


def test_collect_output_preserves_batch_usage(tmp_path):
    row = dict.fromkeys(campaign.REQUIRED, "test")
    rows = [{**row, "id": str(i)} for i in range(24)]
    (tmp_path / "output.json").write_text(json.dumps({"programs": rows}))
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 123}})
    )
    destination = tmp_path / "candidates.jsonl"
    assert (
        campaign.collect_output(tmp_path, destination, {"prompt_sha256": "digest"})
        == 24
    )
    provenance = json.loads(destination.read_text().splitlines()[0])["_provenance"]
    assert provenance["token_usage"] == {"input_tokens": 123}
    assert provenance["prompt_sha256"] == "digest"


def test_deadline_defaults_to_owner_timezone():
    assert (
        str(campaign.deadline_argument("2026-10-25T00:00:00").tzinfo)
        == "America/Los_Angeles"
    )


def test_clean_resume_preserves_prior_batches_and_rows(tmp_path):
    prior = {"batch-000000-a0": {"status": "complete", "candidate_rows": 24}}
    (tmp_path / "campaign_status.json").write_text(
        json.dumps({"status": "stopped", "batches": prior})
    )
    args = arguments(tmp_path)
    args.deadline = datetime.now(campaign.PACIFIC) - timedelta(seconds=1)
    assert campaign.run(args) == 0
    state = json.loads((tmp_path / "campaign_status.json").read_text())
    assert state["batches"] == prior
    assert state["candidate_rows"] == 24


def test_cleanup_escalation_exit_race_still_closes_all_workers(monkeypatch):
    import signal
    import subprocess
    from unittest.mock import Mock

    first = Mock(pid=101, returncode=0)
    second = Mock(pid=102, returncode=0)
    first.poll.return_value = second.poll.return_value = None
    first.wait.side_effect = [subprocess.TimeoutExpired("worker", 0), 0]
    second.wait.return_value = 0
    handles = [Mock(), Mock()]
    active = {"first": (first, [handles[0]]), "second": (second, [handles[1]])}
    records = {"first": {}, "second": {}}

    def kill(pid, sig):
        if pid == 101 and sig == signal.SIGKILL:
            raise ProcessLookupError("worker exited during escalation")

    monkeypatch.setattr(campaign.os, "killpg", kill)
    campaign.stop_workers(active, records, "deadline")
    assert not active
    assert all(row["status"] == "stopped" for row in records.values())
    for handle in handles:
        handle.close.assert_called_once()


def test_author_notes_are_reread_between_batches(tmp_path):
    notes = tmp_path / "author-notes.txt"
    initial = campaign.make_prompt(1, 24, notes)
    assert "Do not use Claude" in initial
    notes.write_text("Avoid decimal digit families.")
    assert "Avoid decimal digit families." in campaign.make_prompt(1, 24, notes)
    notes.write_text("Prefer interval statistics.")
    revised = campaign.make_prompt(2, 24, notes)
    assert "Prefer interval statistics." in revised
    assert "Avoid decimal digit families." not in revised
