"""`python -m wandering_light.solve`: the best system on a task file."""

import gzip
import io
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest

import wandering_light.stack_policy_hub as hub
from experiments import solve_with_cache
from wandering_light import solve
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task, load
from wandering_light.stack_policy import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
    PolicyScorer,
    StackPairPolicy,
    save_policy,
)

TINY = PolicyArchitecture(8, 16, 32)
SHA = "e5a0b1391c5268c3ae04a49e6862f19a30779bb2"


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def write_payload(path: Path, tasks, *, gzipped: bool) -> Path:
    text = "".join(json.dumps(asdict(task)) + "\n" for task in tasks)
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


def tiny_checkpoint(path: Path, *, wandb_url: str | None = None) -> Path:
    return save_policy(
        path,
        StackPairPolicy(TINY),
        trainer_state={"epoch": 0},
        manifest={"arm": "tiny"},
        wandb_url=wandb_url,
    )


def parsed(task):
    inputs = tuple(parse_stack(t, basis=BASIS) for t in task.inputs)
    outputs = tuple(parse_stack(t, basis=BASIS) for t in task.outputs)
    return inputs, outputs


# --- replay_program ---------------------------------------------------------


def test_replay_program_moved_and_reexported():
    assert solve_with_cache.replay_program is replay_program


def test_replay_charges_one_execution_per_op_per_example():
    task = task_for("add_pair", 6)
    inputs, outputs = parsed(task)
    fits, executions, fuel = replay_program(("add",), inputs, outputs)
    assert fits
    assert executions == task.examples
    assert fuel > 0
    fits, executions, _ = replay_program(("add", "add"), inputs, outputs)
    assert not fits


def test_replay_stops_at_a_refused_op():
    task = task_for("add_pair", 6)
    inputs, outputs = parsed(task)
    # Every example is two ints; `add` leaves one, a second `add` is refused by
    # the first example, so the attempt costs examples + 0 and the rest is skipped.
    fits, executions, _ = replay_program(("add", "add", "add"), inputs, outputs)
    assert not fits
    assert executions == task.examples


# --- payload reading ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "gzipped"),
    [("t.jsonl", False), ("t.jsonl.gz", True), ("misnamed.jsonl", True)],
)
def test_load_reads_plain_and_gzipped_payloads(tmp_path, name, gzipped):
    tasks = [task_for("add_pair", 6), task_for("sum", 2)]
    path = write_payload(tmp_path / name, tasks, gzipped=gzipped)
    assert load(path) == tasks


# --- stack_policy_hub --------------------------------------------------------


def test_hub_constants():
    assert hub.REQUIRED_FILES == (
        MODEL_FILE,
        CONFIG_FILE,
        TRAINER_STATE_FILE,
        MANIFEST_FILE,
    )
    assert hub.OPTIONAL_FILES == (WANDB_LINK_FILE,)
    assert hub.MODEL_FILE == MODEL_FILE
    assert hub.DEFAULT_REPO_ID == "abhishekraok/wandering-light-stack-policy"


def test_cache_dir_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert hub.default_cache_dir() == tmp_path / "wandering-light" / "stack-policy"
    assert hub.checkpoint_dir("a/b", SHA) == hub.default_cache_dir() / "a__b" / SHA
    assert hub.checkpoint_dir("a/b", SHA, cache_dir=tmp_path / "c") == (
        tmp_path / "c" / "a__b" / SHA
    )
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert hub.default_cache_dir() == (
        tmp_path / "home" / ".cache" / "wandering-light" / "stack-policy"
    )


@pytest.mark.parametrize("revision", ["main", SHA[:12], SHA.upper(), SHA + "0"])
def test_fetch_requires_a_commit_id(tmp_path, revision):
    with pytest.raises(ValueError, match="40-hex commit id"):
        hub.fetch_stack_policy("a/b", revision, cache_dir=tmp_path)


def test_fetch_returns_a_cached_checkpoint_without_the_network(tmp_path, monkeypatch):
    import huggingface_hub

    tiny_checkpoint(hub.checkpoint_dir("a/b", SHA, cache_dir=tmp_path))

    def refuse(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    path = hub.fetch_stack_policy("a/b", SHA, cache_dir=tmp_path)
    assert hub.is_checkpoint(path)


def test_fetch_downloads_required_and_skips_missing_optional(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    source = tiny_checkpoint(tmp_path / "source")
    calls = []

    def download(repo_id, name, *, revision):
        calls.append((repo_id, name, revision))
        if not (source / name).exists():
            raise EntryNotFoundError(name)
        return str(source / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    path = hub.fetch_stack_policy("a/b", SHA, cache_dir=tmp_path / "cache")
    assert path == hub.checkpoint_dir("a/b", SHA, cache_dir=tmp_path / "cache")
    assert [c[1] for c in calls] == [*hub.REQUIRED_FILES, *hub.OPTIONAL_FILES]
    assert all(c[0] == "a/b" and c[2] == SHA for c in calls)
    assert hub.is_checkpoint(path)
    assert not (path / WANDB_LINK_FILE).exists()


def test_fetch_raises_on_a_missing_required_file(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    def download(repo_id, name, *, revision):
        raise EntryNotFoundError(name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(EntryNotFoundError):
        hub.fetch_stack_policy("a/b", SHA, cache_dir=tmp_path)


def test_publish_is_one_commit_with_the_card(tmp_path, monkeypatch):
    import huggingface_hub

    checkpoint = tiny_checkpoint(tmp_path / "ckpt", wandb_url="https://wandb.ai/x")
    seen = {}

    class Commit:
        oid = SHA

    class FakeApi:
        def create_repo(self, **kwargs):
            seen["repo"] = kwargs

        def create_commit(self, **kwargs):
            seen.setdefault("commits", []).append(kwargs)
            return Commit()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    revision = hub.publish_stack_policy(
        checkpoint, card="# card\n", commit_message="publish"
    )
    assert revision == SHA
    assert seen["repo"]["private"] is True
    assert seen["repo"]["repo_id"] == hub.DEFAULT_REPO_ID
    (commit,) = seen["commits"]
    assert commit["commit_message"] == "publish"
    names = [op.path_in_repo for op in commit["operations"]]
    assert names == [*hub.REQUIRED_FILES, *hub.OPTIONAL_FILES, "README.md"]


# --- solve ---------------------------------------------------------------------


def test_best_describes_itself():
    assert solve.BEST.describe() == (
        "staged-s0-e3 = abhishekraok/wandering-light-stack-policy@e5a0b1391c52; "
        "beam 64 x depth 48, guard both, library on"
    )
    assert solve.BEST.revision == SHA
    assert solve.BEST.measured
    off = solve.System("x", "a/b", SHA, 8, 4, "off", False, "")
    assert off.describe().endswith("guard off, library off")


def test_summary_line_and_empty_coverage():
    assert solve.summarise([]).coverage == 0.0
    line = solve.Summary(10, 3, 1, 2, 12.34, 12345).line()
    assert line == (
        "solved 3/10 (coverage 0.300); wall clock 12.3 s over every task "
        "attempted; executions 12,345; 2 solved from the library; 1 memorised"
    )


def test_row_verdicts_and_record():
    task = task_for("add_pair", 6)
    row = solve.Row(3, task, ("add",), "search", "fit", 0.5, 4, 9, 1, 1)
    assert row.verdict == "unsolved"
    row.shown_fit = True
    assert row.verdict == "memorised"
    row.solved = True
    assert row.verdict == "solved"
    record = row.record()
    json.dumps(record)
    assert record["index"] == 3
    assert record["behaviour"] == "add_pair"
    assert record["length"] == task.length
    assert record["examples"] == task.examples
    assert record["program"] == ["add"]
    for name in ("source", "termination", "seconds", "executions", "fuel"):
        assert name in record
    assert record["expansions"] == 1 and record["depth"] == 1
    assert record["solved"] and record["shown_fit"]
    none = solve.Row(0, task, None, "search", "max_depth", 0.1, 0, 0, 0, 0)
    assert none.record()["program"] is None


SMALL = solve.System("tiny", "a/b", SHA, 64, 3, "both", True, "test")


def test_solve_tasks_replays_the_library_and_grades_last(monkeypatch):
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7), task_for("sum", 2)]
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    events = []
    real_grade = solve.grade

    def grade(program, task):
        events.append("grade")
        return real_grade(program, task)

    monkeypatch.setattr(solve, "grade", grade)
    rows = solve.solve_tasks(
        tasks, scorer, SMALL, progress=lambda row: events.append(row.index)
    )
    # Every clock stops before any held-out example is read.
    assert events[:3] == [0, 1, 2]
    assert set(events[3:]) == {"grade"}
    assert [r.index for r in rows] == [0, 1, 2]
    first, second = rows[0], rows[1]
    assert first.source == "search" and first.termination == "fit"
    assert first.program == ("add",)
    assert second.source == "library" and second.termination == "fit"
    assert second.program == ("add",)
    assert second.expansions == 0
    assert second.executions == tasks[1].examples  # charged for the replay
    assert first.solved and second.solved
    # The third task was charged for replaying `add` before its search.
    assert rows[2].source == "search"
    assert rows[2].executions >= tasks[2].examples
    summary = solve.summarise(rows)
    assert summary.tasks == 3
    assert summary.from_library == 1
    assert summary.executions == sum(r.executions for r in rows)


def test_solve_tasks_without_library_searches_every_task():
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7)]
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    off = solve.System("tiny", "a/b", SHA, 64, 3, "both", False, "test")
    rows = solve.solve_tasks(tasks, scorer, off)
    assert [r.source for r in rows] == ["search", "search"]
    assert all(r.expansions >= 1 for r in rows)


def test_solve_file_with_a_local_checkpoint(tmp_path):
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7)]
    payload = write_payload(tmp_path / "tasks.jsonl", tasks, gzipped=False)
    checkpoint = tiny_checkpoint(tmp_path / "ckpt")
    out, log = io.StringIO(), io.StringIO()
    report = tmp_path / "out" / "report.json"
    summary = solve.solve_file(
        payload,
        system=SMALL,
        checkpoint=checkpoint,
        json_path=report,
        out=out,
        log=log,
    )
    assert summary.tasks == 2 and summary.solved == 2 and summary.from_library == 1
    lines = out.getvalue().splitlines()
    assert lines[0] == f"system  {SMALL.describe()}"
    assert lines[1] == f"policy  {checkpoint}  (local, not BEST)"
    assert lines[2].startswith("host    ")
    assert lines[3] == f"tasks   2 from {payload}"
    assert lines[4].split() == ["#", "behaviour", "verdict", "seconds", "source"]
    assert lines[5].startswith(f"{0:>5}  {'add_pair':<9}  {'solved':<9}  ")
    assert lines[5].endswith("  search")
    assert lines[6].endswith("  library")
    assert lines[-1] == summary.line()
    assert len(log.getvalue().splitlines()) == 2
    data = json.loads(report.read_text())
    assert data["checkpoint_is_best"] is False
    assert data["checkpoint"] == str(checkpoint)
    assert data["tasks_file"] == str(payload)
    assert data["system"]["name"] == "tiny"
    assert data["summary"]["solved"] == 2
    for key in ("host", "commit", "recorded_at_utc"):
        assert key in data
    assert [r["source"] for r in data["rows"]] == ["search", "library"]


def test_solve_file_fetches_best_without_a_checkpoint(tmp_path, monkeypatch):
    checkpoint = tiny_checkpoint(tmp_path / "cached")
    asked = []

    def fetch(repo_id, revision):
        asked.append((repo_id, revision))
        return checkpoint

    monkeypatch.setattr(solve, "fetch_stack_policy", fetch)
    payload = write_payload(
        tmp_path / "t.jsonl.gz", [task_for("add_pair", 6)], gzipped=True
    )
    out = io.StringIO()
    report = tmp_path / "r.json"
    solve.solve_file(
        payload, system=SMALL, json_path=report, out=out, log=io.StringIO()
    )
    assert asked == [("a/b", SHA)]
    assert out.getvalue().splitlines()[1] == f"policy  {checkpoint}"
    assert json.loads(report.read_text())["checkpoint_is_best"] is True


def test_main_runs_best(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake(path, **kwargs):
        seen.update(kwargs, path=path)
        return solve.Summary(0, 0, 0, 0, 0.0, 0)

    monkeypatch.setattr(solve, "solve_file", fake)
    assert solve.main(["t.jsonl", "--checkpoint", "c", "--json", "o.json"]) == 0
    assert seen == {
        "path": "t.jsonl",
        "system": solve.BEST,
        "checkpoint": "c",
        "device": "cpu",
        "json_path": "o.json",
    }
