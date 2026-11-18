import gzip
import json
from dataclasses import asdict, replace
from datetime import datetime
from io import StringIO
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError

from wandering_light import solve, stack_pbe, stack_policy_hub
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import BeamResult, replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_policy import StackPairPolicy, save_policy

ADD_TASK = stack_pbe.Task(
    behaviour="add_pair",
    summary="Sum two integers",
    split="eval",
    task_hash_split="eval",
    inputs=("1 2", "4 7"),
    outputs=("3", "11"),
    held_out_inputs=("8 9",),
    held_out_outputs=("17",),
    program=("add",),
    length=1,
    examples=2,
    held_out=1,
    fuel=2,
    tags=(),
)
SHORTCUT_TASK = replace(
    ADD_TASK, behaviour="shown_only_shortcut", held_out_outputs=("0",)
)


@pytest.mark.parametrize(
    ("compressed", "suffix"),
    [(False, "jsonl"), (True, "jsonl.gz"), (True, "jsonl"), (False, "jsonl.gz")],
)
def test_payload_compression_is_detected_by_content(tmp_path, compressed, suffix):
    tasks = [replace(ADD_TASK, summary="Somme de deux entiers 🧮"), SHORTCUT_TASK]
    payload = "".join(
        json.dumps(asdict(task), ensure_ascii=False) + "\n" for task in tasks
    )
    path = tmp_path / f"tasks.{suffix}"
    data = payload.encode("utf-8")
    path.write_bytes(gzip.compress(data) if compressed else data)

    assert stack_pbe.load(path) == tasks


def test_replay_charges_executed_ops_and_stops_at_first_refusal():
    from experiments.solve_with_cache import replay_program as experiment_replay

    assert experiment_replay is replay_program
    inputs = tuple(parse_stack(text, basis=stack_pbe.BASIS) for text in ("1 2", "4 7"))
    outputs = tuple(
        parse_stack(text, basis=stack_pbe.BASIS) for text in ("3 3", "11 11")
    )
    assert replay_program(("add", "dup"), inputs, outputs) == (True, 4, 4)
    inputs = (inputs[0], parse_stack("4", basis=stack_pbe.BASIS), inputs[1])
    assert replay_program(("add", "dup"), inputs, outputs) == (False, 1, 1)


def test_tasks_time_replay_and_search_before_grading(monkeypatch):
    miss = replace(ADD_TASK, behaviour="miss", outputs=("2 1", "7 4"))
    tasks = [ADD_TASK, SHORTCUT_TASK, ADD_TASK, miss]
    system = replace(solve.BEST, beam_width=7, max_depth=3, guard="close")
    scorer = object()
    events = []
    now = 0.0

    def spend(event, seconds):
        nonlocal now
        now += seconds
        events.append(event)

    def search(task, model, **options):
        assert model is scorer
        assert options == {"beam_width": 7, "max_depth": 3, "guard": "close"}
        spend(("search", task.behaviour), 3)
        program = ("add",) if task is ADD_TASK else None
        return BeamResult(program, "fit" if program else "max_depth", 99, 10, 11, 12, 1)

    def replay(program, inputs, outputs):
        spend("replay", 2)
        return replay_program(program, inputs, outputs)

    grade = solve.stack_pbe_eval.grade

    def tracked_grade(program, task):
        spend(("grade", task.behaviour), 100)
        return grade(program, task)

    def progress(row):
        spend(("progress", row.index), 100)
        assert not row.solved and not row.shown_fit

    monkeypatch.setattr(solve, "time", SimpleNamespace(perf_counter=lambda: now))
    monkeypatch.setattr(solve, "beam_solve", search)
    monkeypatch.setattr(solve, "replay_program", replay)
    monkeypatch.setattr(solve.stack_pbe_eval, "grade", tracked_grade)

    rows = solve.solve_tasks(tasks, scorer, system, progress=progress)

    assert events == [
        ("search", "add_pair"),
        ("progress", 0),
        "replay",
        ("progress", 1),
        "replay",
        ("progress", 2),
        "replay",
        ("search", "miss"),
        ("progress", 3),
        ("grade", "add_pair"),
        ("grade", "shown_only_shortcut"),
        ("grade", "add_pair"),
    ]
    assert [row.task for row in rows] == tasks
    assert [row.verdict for row in rows] == [
        "solved",
        "memorised",
        "solved",
        "unsolved",
    ]
    assert [row.source for row in rows] == ["search", "library", "library", "search"]
    assert [row.termination for row in rows] == ["fit", "fit", "fit", "max_depth"]
    assert [row.seconds for row in rows] == [3, 2, 2, 5]
    assert [row.executions for row in rows] == [10, 2, 2, 12]
    assert [row.fuel for row in rows] == [11, 2, 2, 13]
    assert [row.expansions for row in rows] == [12, 0, 0, 12]
    assert [row.depth for row in rows] == [1, 0, 0, 1]
    assert solve.summarise(rows) == solve.Summary(4, 2, 1, 1, 12, 26)
    assert solve.summarise([]).coverage == 0.0


@pytest.mark.parametrize("enabled", [True, False])
def test_library_replays_shortest_first_without_duplicates(monkeypatch, enabled):
    long, short = ("dup", "add"), ("swap",)
    attempts = []
    programs = iter([long, short, short, None])

    def search(*_args, **_kwargs):
        return BeamResult(next(programs), "fit", 0.0, 0, 0, 0, 0)

    def replay(program, _inputs, _outputs):
        attempts.append(program)
        return False, 0, 0

    monkeypatch.setattr(solve, "beam_solve", search)
    monkeypatch.setattr(solve, "replay_program", replay)
    solve.solve_tasks([ADD_TASK] * 4, None, replace(solve.BEST, library=enabled))

    assert attempts == ([long, short, long, short, long] if enabled else [])


def test_cli_with_a_real_local_checkpoint(tmp_path, capsys, monkeypatch):
    checkpoint = save_policy(
        tmp_path / "policy",
        StackPairPolicy(PolicyArchitecture(8, 16, 32)),
        trainer_state={},
        manifest={},
        wandb_url=None,
    )
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(task)) + "\n" for task in (ADD_TASK, SHORTCUT_TASK))
    )
    report_path = tmp_path / "reports" / "run.json"
    fetches = []

    def fetch(repo_id, revision):
        fetches.append((repo_id, revision))
        return checkpoint

    monkeypatch.setattr(solve, "fetch_stack_policy", fetch)
    assert (
        solve.main(
            [str(path), "--checkpoint", str(checkpoint), "--json", str(report_path)]
        )
        == 0
    )
    assert fetches == []
    out, log = capsys.readouterr()
    lines = out.splitlines()
    assert lines[:2] == [
        f"system  {solve.BEST.describe()}",
        f"policy  {checkpoint}  (local, not BEST)",
    ]
    assert lines[2].startswith("host    ")
    assert lines[3] == f"tasks   2 from {path}"
    assert lines[4].split() == ["#", "behaviour", "verdict", "seconds", "source"]
    assert lines[-1].startswith("solved 1/2 (coverage 0.500); wall clock ")
    assert len(log.splitlines()) == 2

    report = json.loads(report_path.read_text())
    assert report["system"] == asdict(solve.BEST)
    assert report["checkpoint"] == str(checkpoint)
    assert report["checkpoint_is_best"] is False
    assert report["host"]["device"] == "cpu"
    assert "commit" in report
    assert report["tasks_file"] == str(path)
    assert (
        datetime.fromisoformat(report["recorded_at_utc"]).utcoffset().total_seconds()
        == 0
    )
    assert report["summary"]["coverage"] == 0.5
    assert report["summary"]["memorised"] == 1
    assert report["summary"]["from_library"] == 0
    width = len(SHORTCUT_TASK.behaviour)
    for index, (record, task, verdict, source) in enumerate(
        zip(
            report["rows"],
            (ADD_TASK, SHORTCUT_TASK),
            ("solved", "memorised"),
            ("search", "library"),
            strict=True,
        )
    ):
        assert record["index"] == index
        assert record["behaviour"] == task.behaviour
        assert record["program"] == ["add"]
        assert record["examples"] == task.examples
        assert record["length"] == task.length
        assert record["verdict"] == verdict
        assert lines[5 + index] == (
            f"{index:>5}  {task.behaviour:<{width}}  {verdict:<9}  "
            f"{record['seconds']:>8.3f}  {source}"
        )

    corpus = tmp_path / "corpus"
    stack_pbe.write_corpus(
        [replace(ADD_TASK, split="train"), SHORTCUT_TASK], corpus, config={}
    )
    out = StringIO()
    summary = solve.solve_file(corpus, out=out, log=StringIO(), json_path=report_path)
    assert summary.tasks == 2
    assert summary.from_library == 1
    assert fetches == [(solve.BEST.repo_id, solve.BEST.revision)]
    assert out.getvalue().splitlines()[1] == f"policy  {checkpoint}"
    report = json.loads(report_path.read_text())
    assert report["checkpoint_is_best"] is True
    assert report["rows"][0]["behaviour"] == SHORTCUT_TASK.behaviour


@pytest.mark.parametrize("optional_exists", [True, False])
def test_fetch_pinned_checkpoint_is_cached(tmp_path, monkeypatch, optional_exists):
    repo, revision = solve.BEST.repo_id, solve.BEST.revision
    files = stack_policy_hub.REQUIRED_FILES + stack_policy_hub.OPTIONAL_FILES
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name in files:
        (downloads / name).write_text(name)
    calls = []

    def download(repo_id, name, *, revision):
        calls.append((repo_id, name, revision))
        if name in stack_policy_hub.OPTIONAL_FILES and not optional_exists:
            raise EntryNotFoundError(name)
        return str(downloads / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    cache = tmp_path / "xdg" / "wandering-light" / "stack-policy"
    expected = cache / repo.replace("/", "__") / revision
    for bad in ("main", "f" * 39, "F" * 40, f"{revision}\n"):
        with pytest.raises(ValueError, match="revision must be a 40-hex commit id"):
            stack_policy_hub.fetch_stack_policy(repo, bad)
    assert calls == []

    assert stack_policy_hub.default_cache_dir() == cache
    assert stack_policy_hub.checkpoint_dir(repo, revision) == expected
    assert stack_policy_hub.fetch_stack_policy(repo, revision) == expected
    assert stack_policy_hub.is_checkpoint(expected)
    assert {file.name: file.read_text() for file in expected.iterdir()} == {
        name: name
        for name in (files if optional_exists else stack_policy_hub.REQUIRED_FILES)
    }
    assert stack_policy_hub.fetch_stack_policy(repo, revision) == expected
    assert calls == [(repo, name, revision) for name in files]


def test_missing_required_file_does_not_leave_a_checkpoint(tmp_path, monkeypatch):
    repo, revision = solve.BEST.repo_id, solve.BEST.revision

    def download(_repo, name, *, revision):
        if name == stack_policy_hub.MANIFEST_FILE:
            raise EntryNotFoundError(name)
        path = tmp_path / name
        path.write_text(name)
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(EntryNotFoundError, match=stack_policy_hub.MANIFEST_FILE):
        stack_policy_hub.fetch_stack_policy(repo, revision, cache_dir=tmp_path)
    assert not stack_policy_hub.is_checkpoint(
        stack_policy_hub.checkpoint_dir(repo, revision, cache_dir=tmp_path)
    )


def test_publish_sends_weights_and_card_in_one_commit(tmp_path, monkeypatch):
    repo, revision = solve.BEST.repo_id, solve.BEST.revision
    files = stack_policy_hub.REQUIRED_FILES + stack_policy_hub.OPTIONAL_FILES
    for name in (*files, "optimizer.pt", "README.md"):
        (tmp_path / name).write_text(name)
    calls = []

    class Hub:
        def create_repo(self, **kwargs):
            calls.append(kwargs)

        def create_commit(self, *, operations, **kwargs):
            calls.append(kwargs)
            assert [operation.path_in_repo for operation in operations] == [
                *files,
                "README.md",
            ]
            assert operations[-1].path_or_fileobj == b"# Stack policy"
            return SimpleNamespace(oid=revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    assert (
        stack_policy_hub.publish_stack_policy(
            tmp_path, card="# Stack policy", commit_message="Publish policy"
        )
        == revision
    )
    assert calls == [
        {"repo_id": repo, "private": True, "exist_ok": True},
        {"repo_id": repo, "commit_message": "Publish policy"},
    ]
