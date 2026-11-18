"""The stack solver's replay, timing, grading, and checkpoint boundary."""

import gzip
import hashlib
import io
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError

from experiments import solve_with_cache
from wandering_light import solve
from wandering_light import stack_policy_hub as hub
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import BeamResult, replay_program
from wandering_light.stack_machine import Int
from wandering_light.stack_pbe import Task, load
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import StackPairPolicy, save_policy


@pytest.fixture
def task():
    return Task(
        behaviour="add_pair",
        summary="",
        split="eval",
        task_hash_split="eval",
        inputs=("[ 3 4 ]", "[ 6 -2 ]"),
        outputs=("[ 7 ]", "[ 4 ]"),
        held_out_inputs=("[ 8 2 ]",),
        held_out_outputs=("[ 10 ]",),
        program=("add",),
        length=1,
        examples=2,
        held_out=1,
        fuel=1,
        tags=(),
    )


@pytest.fixture
def checkpoint(tmp_path):
    return save_policy(
        tmp_path / "policy",
        StackPairPolicy(PolicyArchitecture(8, 16, 32)),
        trainer_state={},
        manifest={},
        wandb_url=None,
    )


def test_replay_ends_on_first_refusal_and_charges_previous_stacks():
    assert solve_with_cache.replay_program is replay_program
    pair, one = (Int(4), Int(2)), (Int(6),)
    assert replay_program(("add",), (pair, pair), (one, one)) == (True, 2, 2)
    assert replay_program(("add", "dup"), (pair, one, pair), ()) == (False, 1, 1)


def test_library_order_costs_and_deferred_grading(task, monkeypatch):
    double = replace(
        task,
        behaviour="double",
        inputs=("[ 2 ]",),
        outputs=("[ 4 ]",),
        held_out_inputs=("[ 3 ]",),
        held_out_outputs=("[ 6 ]",),
        examples=1,
    )
    dup = replace(
        double, behaviour="dup", outputs=("[ 2 2 ]",), held_out_outputs=("[ 3 3 ]",)
    )
    shortcut = replace(dup, behaviour="shortcut", held_out_outputs=("[ 3 4 ]",))
    miss = replace(double, behaviour="miss", outputs=("[ 5 ]",))
    tasks = [double, dup, shortcut, dup, miss]
    system = replace(solve.BEST, beam_width=2, max_depth=3, guard="close")
    events = []
    answers = iter([("dup", "add"), ("dup",), None])
    now = 0

    def search(task, scorer, *, beam_width, max_depth, guard):
        nonlocal now
        assert (beam_width, max_depth, guard) == (2, 3, "close")
        events.append(("search", task.behaviour))
        now += 5
        program = next(answers)
        return BeamResult(program, "fit" if program else "max_depth", 0, 10, 20, 2, 2)

    def replay(*args):
        nonlocal now
        now += 1
        return replay_program(*args)

    def progress(row):
        nonlocal now
        events.append(("progress", row.index))
        now += 10

    def grading(program, task):
        events.append(("grade", task.behaviour))
        return grade(program, task)

    monkeypatch.setattr(solve, "beam_solve", search)
    monkeypatch.setattr(solve, "replay_program", replay)
    monkeypatch.setattr(solve, "grade", grading)
    monkeypatch.setattr(solve, "time", SimpleNamespace(perf_counter=lambda: now))
    rows = solve.solve_tasks(tasks, None, system, progress=progress)

    assert [r.task for r in rows] == tasks
    assert events == [
        ("search", "double"),
        ("progress", 0),
        ("search", "dup"),
        ("progress", 1),
        ("progress", 2),
        ("progress", 3),
        ("search", "miss"),
        ("progress", 4),
        ("grade", "double"),
        ("grade", "dup"),
        ("grade", "shortcut"),
        ("grade", "dup"),
    ]
    assert [r.verdict for r in rows] == [
        "solved",
        "solved",
        "memorised",
        "solved",
        "unsolved",
    ]
    assert [r.executions for r in rows] == [10, 12, 1, 1, 13]
    assert [r.fuel for r in rows] == [20, 22, 1, 1, 23]
    assert [r.seconds for r in rows] == [5, 6, 1, 1, 7]
    assert rows[2].termination == "fit"
    assert rows[2].expansions == rows[2].depth == 0
    assert rows[4].record()["program"] is None
    summary = solve.summarise(rows)
    assert summary == solve.Summary(5, 3, 1, 1, 20, 37)
    assert summary.line() == (
        "solved 3/5 (coverage 0.600); wall clock 20.0 s over every task attempted; "
        "executions 37; 1 solved from the library; 1 memorised"
    )


def test_no_library_searches_each_time(task):
    system = replace(solve.BEST, library=False)
    rows = solve.solve_tasks([task, task], None, system)
    assert system.describe().endswith("library off")
    assert all(row.solved and row.source == "search" for row in rows)
    assert rows[0].executions == rows[1].executions


@pytest.mark.parametrize("compressed", [True, False])
def test_load_sniffs_payload_content(task, tmp_path, compressed):
    data = (json.dumps(asdict(task)) + "\n").encode()
    path = tmp_path / ("tasks.jsonl" if compressed else "tasks.jsonl.gz")
    path.write_bytes(gzip.compress(data) if compressed else data)
    assert load(path) == [task]


def test_local_checkpoint_cli_table_and_json(tmp_path, task, checkpoint, capsys):
    path = tmp_path / "tasks.jsonl"
    path.write_text((json.dumps(asdict(task)) + "\n") * 2)
    report_path = tmp_path / "run.json"
    assert (
        solve.main(
            [str(path), "--checkpoint", str(checkpoint), "--json", str(report_path)]
        )
        == 0
    )
    stdout, stderr = capsys.readouterr()
    lines = stdout.splitlines()
    assert lines[:2] == [
        "system  staged-s0-e3 = abhishekraok/wandering-light-stack-policy@e5a0b1391c52; beam 64 x depth 48, guard both, library on",
        f"policy  {checkpoint}  (local, not BEST)",
    ]
    assert lines[2].startswith("host    ")
    assert lines[3] == f"tasks   2 from {path}"
    assert lines[4].split() == ["#", "behaviour", "verdict", "seconds", "source"]
    assert len(stderr.splitlines()) == 2
    report = json.loads(report_path.read_text())
    assert report["system"] == asdict(solve.BEST)
    assert report["tasks_file"] == str(path)
    assert report["checkpoint"] == str(checkpoint)
    assert report["checkpoint_is_best"] is False
    assert {"host", "commit", "recorded_at_utc"} <= report.keys()
    assert report["summary"]["coverage"] == 1.0
    assert "solved 2/2 (coverage 1.000)" in lines[-1]
    for i, (record, source) in enumerate(
        zip(report["rows"], ("search", "library"), strict=True)
    ):
        assert record["program"] == ["add"]
        assert record["index"] == i and record["behaviour"] == "add_pair"
        assert record["examples"] == 2 and record["length"] == 1
        assert (
            record["verdict"] == "solved" and record["solved"] and record["shown_fit"]
        )
        assert (
            lines[5 + i]
            == f"{i:>5}  {'add_pair':<9}  {'solved':<9}  {record['seconds']:>8.3f}  {source}"
        )


def test_default_fetch_and_empty_corpus(tmp_path, checkpoint, monkeypatch):
    def fetch(repo_id, revision):
        assert (repo_id, revision) == (hub.DEFAULT_REPO_ID, solve.BEST.revision)
        return checkpoint

    monkeypatch.setattr(solve, "fetch_stack_policy", fetch)
    path = tmp_path / "empty"
    path.mkdir()
    (path / "eval.jsonl").touch()
    split = {
        "path": "eval.jsonl",
        "sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps({"splits": {"eval": split}}))
    destination = tmp_path / "report.json"
    out, log = io.StringIO(), io.StringIO()
    summary = solve.solve_file(path, out=out, log=log, json_path=destination)
    assert summary == solve.Summary(0, 0, 0, 0, 0.0, 0)
    assert summary.coverage == 0.0
    assert "(local" not in out.getvalue()
    assert log.getvalue() == ""
    assert json.loads(destination.read_text())["checkpoint_is_best"] is True


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "g" * 40, "a" * 41])
def test_fetch_requires_pinned_revision(revision, tmp_path):
    with pytest.raises(ValueError) as err:
        hub.fetch_stack_policy("user/model", revision, cache_dir=tmp_path)
    assert str(err.value) == f"revision must be a 40-hex commit id, not {revision!r}"


def test_fetch_retries_then_reuses_complete_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    revision = "d" * 40
    cache = tmp_path / "wandering-light" / "stack-policy"
    target = cache / "acct__model" / revision
    assert hub.default_cache_dir() == cache
    assert hub.checkpoint_dir("acct/model", revision) == target
    calls = []
    missing = {hub.WANDB_LINK_FILE, hub.TRAINER_STATE_FILE}

    def download(repo_id, name, *, revision):
        assert repo_id == "acct/model" and revision == "d" * 40
        calls.append(name)
        if name in missing:
            raise EntryNotFoundError("missing")
        artifact = tmp_path / name
        artifact.write_text(name)
        return artifact

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(EntryNotFoundError):
        hub.fetch_stack_policy("acct/model", revision)
    assert not hub.is_checkpoint(target)
    calls.clear()
    missing.remove(hub.TRAINER_STATE_FILE)
    assert hub.fetch_stack_policy("acct/model", revision) == target
    assert hub.is_checkpoint(target)
    assert not (target / hub.WANDB_LINK_FILE).exists()
    for name in hub.REQUIRED_FILES:
        assert (target / name).read_text() == name
    assert hub.fetch_stack_policy("acct/model", revision) == target
    assert calls == list(hub.REQUIRED_FILES + hub.OPTIONAL_FILES)


def test_publish_one_weights_only_commit(checkpoint, monkeypatch):
    (checkpoint / hub.WANDB_LINK_FILE).write_text("lineage")
    (checkpoint / "optimizer.pt").touch()
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            assert kwargs == {
                "repo_id": hub.DEFAULT_REPO_ID,
                "repo_type": "model",
                "private": True,
                "exist_ok": True,
            }

        def create_commit(self, *, repo_id, repo_type, operations, commit_message):
            assert (repo_id, repo_type, commit_message) == (
                hub.DEFAULT_REPO_ID,
                "model",
                "publish",
            )
            calls.append({op.path_in_repo: op.path_or_fileobj for op in operations})
            return SimpleNamespace(oid=solve.BEST.revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    revision = hub.publish_stack_policy(
        checkpoint, card="model card", commit_message="publish"
    )
    assert revision == solve.BEST.revision
    assert len(calls) == 1
    assert set(calls[0]) == {*hub.REQUIRED_FILES, *hub.OPTIONAL_FILES, "README.md"}
    assert calls[0]["README.md"] == b"model card"
