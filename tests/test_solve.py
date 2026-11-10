"""The one command that runs the best system on a task file, and what it rests on.

`wandering_light.solve` (the command), `wandering_light.stack_policy_hub` (the
pinned checkpoint it fetches), `stack_beam.replay_program` (the library replay)
and `stack_pbe.load` on a plain ``.jsonl``. Nothing here touches the network: the
Hub is stood in for by monkeypatching ``huggingface_hub`` at the names the
module looks up at call time.
"""

from __future__ import annotations

import gzip
import io
import json
import random
import re
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError

from wandering_light import solve, stack_pbe, stack_policy_hub
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_policy import (
    PolicyScorer,
    StackPairPolicy,
    save_policy,
)
from wandering_light.stack_policy_hub import (
    DEFAULT_REPO_ID,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    checkpoint_dir,
    default_cache_dir,
    fetch_stack_policy,
    is_checkpoint,
    publish_stack_policy,
)

if TYPE_CHECKING:
    from pathlib import Path

TINY = PolicyArchitecture(8, 16, 32)
REVISION = "0123456789abcdef0123456789abcdef01234567"
# Small enough to stay fast on an untrained policy; every task below has a
# one-op witness, which depth 1 at beam 64 finds exhaustively.
SMALL = replace(solve.BEST, beam_width=64, max_depth=2)


def task_for(name: str, seed: int):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def parsed(task):
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    return inputs, outputs


@pytest.fixture
def scorer():
    return PolicyScorer(StackPairPolicy(TINY), "cpu")


@pytest.fixture
def checkpoint(tmp_path):
    return save_policy(
        tmp_path / "policy",
        StackPairPolicy(TINY),
        trainer_state={"best_epoch": 0},
        manifest={"arm": "tiny"},
        wandb_url=None,
    )


def write_jsonl(path: Path, tasks, *, gzipped: bool = False) -> Path:
    text = "".join(json.dumps(asdict(task)) + "\n" for task in tasks)
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# replay_program
# ---------------------------------------------------------------------------


def test_replay_fits_the_witness_and_charges_one_execution_per_op_per_stack():
    task = task_for("tail", 1)
    inputs, outputs = parsed(task)
    fits, executions, fuel = replay_program(("uncons", "drop"), inputs, outputs)
    assert fits
    assert executions == 2 * task.examples
    assert fuel > 0


def test_replay_of_a_wrong_program_runs_and_does_not_fit():
    task = task_for("tail", 1)
    inputs, outputs = parsed(task)
    fits, executions, _ = replay_program(("uncons",), inputs, outputs)
    assert not fits
    assert executions == task.examples


def test_a_refused_op_ends_the_replay():
    task = task_for("add_pair", 2)
    inputs, outputs = parsed(task)
    # `add` consumes the pair; the second `add` has one int and is refused at the
    # first example, so it is charged nothing and nothing after it runs.
    fits, executions, _ = replay_program(("add", "add", "add"), inputs, outputs)
    assert not fits
    assert executions == task.examples


def test_the_cache_experiment_reexports_replay_program():
    from experiments import solve_with_cache

    assert solve_with_cache.replay_program is replay_program


# ---------------------------------------------------------------------------
# stack_pbe.load on plain and gzipped payloads
# ---------------------------------------------------------------------------


def test_load_reads_plain_and_gzipped_payloads_alike(tmp_path):
    tasks = [task_for("add_pair", 1), task_for("tail", 2)]
    plain = write_jsonl(tmp_path / "tasks.jsonl", tasks)
    zipped = write_jsonl(tmp_path / "tasks.jsonl.gz", tasks, gzipped=True)
    assert stack_pbe.load(plain) == tasks
    assert stack_pbe.load(zipped) == tasks


def test_load_tells_gzip_by_its_bytes_not_its_suffix(tmp_path):
    tasks = [task_for("append", 3)]
    misnamed = write_jsonl(tmp_path / "tasks.jsonl", tasks, gzipped=True)
    assert stack_pbe.load(misnamed) == tasks


# ---------------------------------------------------------------------------
# stack_policy_hub
# ---------------------------------------------------------------------------


def test_file_lists_are_the_stack_policy_constants():
    assert REQUIRED_FILES == (
        "model.safetensors",
        "policy_config.json",
        "trainer_state.json",
        "training_manifest.json",
    )
    assert OPTIONAL_FILES == ("wandb_run.url",)
    assert REQUIRED_FILES[0] == stack_policy_hub.MODEL_FILE
    assert DEFAULT_REPO_ID == "abhishekraok/wandering-light-stack-policy"


def test_cache_dir_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "wandering-light" / "stack-policy"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_cache_dir() == (
        tmp_path / "home" / ".cache" / "wandering-light" / "stack-policy"
    )


def test_checkpoint_dir_names_repo_and_revision(monkeypatch, tmp_path):
    assert checkpoint_dir("a/b", REVISION, cache_dir=tmp_path) == (
        tmp_path / "a__b" / REVISION
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert checkpoint_dir("a/b", REVISION) == (
        tmp_path / "xdg" / "wandering-light" / "stack-policy" / "a__b" / REVISION
    )


def test_is_checkpoint_needs_every_required_file(tmp_path, checkpoint):
    assert is_checkpoint(checkpoint)
    assert not (checkpoint / "wandb_run.url").exists()  # optional
    (checkpoint / "trainer_state.json").unlink()
    assert not is_checkpoint(checkpoint)
    assert not is_checkpoint(tmp_path / "absent")


@pytest.mark.parametrize(
    "revision", ["main", REVISION.upper(), REVISION[:39], REVISION + "0", "v1.0"]
)
def test_fetch_refuses_anything_but_a_commit_id(revision, tmp_path):
    with pytest.raises(ValueError, match="40-hex commit id") as raised:
        fetch_stack_policy("a/b", revision, cache_dir=tmp_path)
    assert str(raised.value) == (
        f"revision must be a 40-hex commit id, not {revision!r}"
    )


class FakeHub:
    """Serves files from a directory, as ``hf_hub_download`` would from a cache."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo_id, filename, *, revision):
        self.calls.append((repo_id, filename, revision))
        path = self.root / filename
        if not path.exists():
            raise EntryNotFoundError(f"{filename} not in {repo_id}@{revision}")
        return str(path)


def test_fetch_downloads_once_and_skips_a_missing_optional_file(
    monkeypatch, tmp_path, checkpoint
):
    hub = FakeHub(checkpoint)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    cache = tmp_path / "cache"
    path = fetch_stack_policy("a/b", REVISION, cache_dir=cache)
    assert path == checkpoint_dir("a/b", REVISION, cache_dir=cache)
    assert is_checkpoint(path)
    assert sorted(p.name for p in path.iterdir()) == sorted(REQUIRED_FILES)
    assert [name for _, name, _ in hub.calls] == [*REQUIRED_FILES, *OPTIONAL_FILES]
    assert {(repo, rev) for repo, _, rev in hub.calls} == {("a/b", REVISION)}
    for name in REQUIRED_FILES:
        assert (path / name).read_bytes() == (checkpoint / name).read_bytes()

    hub.calls.clear()
    assert fetch_stack_policy("a/b", REVISION, cache_dir=cache) == path
    assert hub.calls == []
    assert [p.name for p in path.parent.iterdir()] == [REVISION]  # no staging left


def test_fetch_copies_the_optional_file_when_published(
    monkeypatch, tmp_path, checkpoint
):
    (checkpoint / "wandb_run.url").write_text("[InternetShortcut]\nURL=x\n")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", FakeHub(checkpoint))
    path = fetch_stack_policy("a/b", REVISION, cache_dir=tmp_path / "cache")
    assert (path / "wandb_run.url").read_text().endswith("URL=x\n")


def test_fetch_raises_on_a_missing_required_file(monkeypatch, tmp_path, checkpoint):
    (checkpoint / "training_manifest.json").unlink()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", FakeHub(checkpoint))
    cache = tmp_path / "cache"
    with pytest.raises(EntryNotFoundError):
        fetch_stack_policy("a/b", REVISION, cache_dir=cache)
    target = checkpoint_dir("a/b", REVISION, cache_dir=cache)
    assert not is_checkpoint(target)
    assert list(target.parent.iterdir()) == []


def test_publish_is_one_commit_of_the_files_and_the_card(monkeypatch, checkpoint):
    (checkpoint / "wandb_run.url").write_text("link")
    seen = {}

    class FakeApi:
        def create_repo(self, **kwargs):
            seen["repo"] = kwargs

        def create_commit(self, **kwargs):
            seen.setdefault("commits", []).append(kwargs)
            return type("Commit", (), {"oid": REVISION})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    revision = publish_stack_policy(
        checkpoint, card="# card\n", commit_message="publish tiny"
    )
    assert revision == REVISION
    assert seen["repo"]["repo_id"] == DEFAULT_REPO_ID
    assert seen["repo"]["private"] is True
    (commit,) = seen["commits"]
    assert commit["commit_message"] == "publish tiny"
    added = {op.path_in_repo: op for op in commit["operations"]}
    assert set(added) == {*REQUIRED_FILES, *OPTIONAL_FILES, "README.md"}


# ---------------------------------------------------------------------------
# System, Row, Summary
# ---------------------------------------------------------------------------


def test_best_is_pinned_and_described():
    best = solve.BEST
    assert best.name == "staged-s0-e3"
    assert re.fullmatch(r"[0-9a-f]{40}", best.revision)
    assert best.describe() == (
        "staged-s0-e3 = abhishekraok/wandering-light-stack-policy@e5a0b1391c52; "
        "beam 64 x depth 48, guard both, library on"
    )
    assert replace(best, library=False).describe().endswith("library off")
    assert best.measured


def row(task, **kwargs) -> solve.Row:
    defaults = {
        "index": 0,
        "task": task,
        "program": ("add",),
        "source": "search",
        "termination": "fit",
        "seconds": 0.5,
        "executions": 10,
        "fuel": 20,
        "expansions": 1,
        "depth": 1,
    }
    return solve.Row(**{**defaults, **kwargs})


def test_verdicts():
    task = task_for("add_pair", 1)
    assert row(task, solved=True, shown_fit=True).verdict == "solved"
    assert row(task, shown_fit=True).verdict == "memorised"
    assert row(task).verdict == "unsolved"


def test_record_is_json_and_carries_every_field():
    task = task_for("add_pair", 1)
    record = row(task, index=3, solved=True, shown_fit=True).record()
    assert json.loads(json.dumps(record)) == record
    assert record["index"] == 3
    assert record["behaviour"] == "add_pair"
    assert record["length"] == task.length
    assert record["examples"] == task.examples
    assert record["program"] == ["add"]
    for name in (
        "source",
        "termination",
        "seconds",
        "executions",
        "fuel",
        "expansions",
        "depth",
        "solved",
        "shown_fit",
    ):
        assert name in record
    assert "task" not in record
    assert row(task, program=None).record()["program"] is None


def test_summary_sums_rows_and_counts_library_solves():
    task = task_for("add_pair", 1)
    rows = [
        row(task, solved=True, shown_fit=True, seconds=1.0, executions=1000),
        row(task, source="library", solved=True, shown_fit=True, seconds=0.25),
        row(task, source="library", shown_fit=True, seconds=0.25),
        row(task, program=None, termination="max_depth", seconds=2.0),
    ]
    summary = solve.summarise(rows)
    assert summary == solve.Summary(
        tasks=4,
        solved=2,
        memorised=1,
        from_library=1,
        seconds=3.5,
        executions=1030,
    )
    assert summary.coverage == 0.5
    assert summary.line() == (
        "solved 2/4 (coverage 0.500); wall clock 3.5 s over every task attempted; "
        "executions 1,030; 1 solved from the library; 1 memorised"
    )
    assert solve.summarise([]).coverage == 0.0


# ---------------------------------------------------------------------------
# solve_tasks
# ---------------------------------------------------------------------------


def test_a_program_found_once_is_replayed_on_the_next_task(scorer):
    tasks = [task_for("add_pair", 1), task_for("add_pair", 2), task_for("tail", 3)]
    seen = []
    rows = solve.solve_tasks(tasks, scorer, SMALL, progress=seen.append)
    assert seen == rows
    assert [r.index for r in rows] == [0, 1, 2]
    first, second, third = rows
    assert (first.source, first.program) == ("search", ("add",))
    assert (second.source, second.termination) == ("library", "fit")
    assert second.program == ("add",)
    assert second.executions == tasks[1].examples  # one op on each shown stack
    assert second.expansions == second.depth == 0
    assert third.source == "search"
    assert third.executions > 0  # the failed replay of `add` is charged
    assert all(r.solved and r.verdict == "solved" for r in rows)
    assert all(r.seconds > 0 for r in rows)


def test_without_the_library_every_task_is_searched(scorer):
    tasks = [task_for("add_pair", 1), task_for("add_pair", 2)]
    rows = solve.solve_tasks(tasks, scorer, replace(SMALL, library=False))
    assert [r.source for r in rows] == ["search", "search"]
    assert all(r.solved for r in rows)


def test_grading_waits_for_the_last_clock(monkeypatch, scorer):
    events = []
    real_grade = solve.grade

    def grade(program, task):
        events.append("grade")
        return real_grade(program, task)

    monkeypatch.setattr(solve, "grade", grade)
    tasks = [task_for("add_pair", 1), task_for("append", 2)]
    solve.solve_tasks(
        tasks, scorer, SMALL, progress=lambda r: events.append(("row", r.solved))
    )
    assert events == [("row", False), ("row", False), "grade", "grade"]


def test_held_out_examples_are_never_read_while_solving(scorer):
    """Corrupt every held-out output: the search and library are unchanged."""
    tasks = [task_for("add_pair", 1), task_for("add_pair", 2)]
    blinded = [
        replace(t, held_out_outputs=tuple("999" for _ in t.held_out_outputs))
        for t in tasks
    ]
    clean = solve.solve_tasks(tasks, scorer, SMALL)
    dark = solve.solve_tasks(blinded, scorer, SMALL)
    assert [(r.program, r.source) for r in clean] == [
        (r.program, r.source) for r in dark
    ]
    assert all(r.solved for r in clean)
    assert [r.verdict for r in dark] == ["memorised", "memorised"]


def test_library_keeps_programs_shortest_first_without_duplicates(scorer):
    # `tail` is two ops and found first; `add` is one op and found second. A
    # third `add_pair` task must be served by `add`, which sits before `tail`.
    tasks = [task_for("tail", 1), task_for("add_pair", 2), task_for("add_pair", 3)]
    rows = solve.solve_tasks(tasks, scorer, SMALL)
    assert [r.source for r in rows] == ["search", "search", "library"]
    assert rows[2].executions == tasks[2].examples  # `add` was replayed first


# ---------------------------------------------------------------------------
# solve_file and main
# ---------------------------------------------------------------------------


def test_solve_file_prints_header_table_and_summary(tmp_path, checkpoint):
    tasks = [task_for("add_pair", 1), task_for("add_pair", 2), task_for("tail", 3)]
    path = write_jsonl(tmp_path / "tasks.jsonl", tasks)
    out, log = io.StringIO(), io.StringIO()
    report = tmp_path / "out" / "run.json"
    summary = solve.solve_file(
        path,
        system=SMALL,
        checkpoint=checkpoint,
        json_path=report,
        out=out,
        log=log,
    )
    assert (summary.tasks, summary.solved, summary.from_library) == (3, 3, 1)
    lines = out.getvalue().splitlines()
    assert lines[0] == f"system  {SMALL.describe()}"
    assert lines[1] == f"policy  {checkpoint}  (local, not BEST)"
    assert lines[2].startswith("host    ")
    assert lines[3] == f"tasks   3 from {path}"
    assert lines[4] == "    #  behaviour  verdict     seconds  source"
    assert re.fullmatch(r"    0  add_pair   solved    +\d+\.\d{3}  search", lines[5])
    assert re.fullmatch(r"    1  add_pair   solved    +\d+\.\d{3}  library", lines[6])
    assert re.fullmatch(r"    2  tail       solved    +\d+\.\d{3}  search", lines[7])
    assert lines[8] == summary.line()
    assert len(lines) == 9
    assert len(log.getvalue().splitlines()) == 3

    written = json.loads(report.read_text())
    assert written["checkpoint"] == str(checkpoint)
    assert written["checkpoint_is_best"] is False
    assert written["tasks_file"] == str(path)
    assert written["system"]["name"] == SMALL.name
    assert written["summary"]["solved"] == 3
    assert written["summary"]["coverage"] == 1.0
    assert [r["source"] for r in written["rows"]] == ["search", "library", "search"]
    for key in ("host", "commit", "recorded_at_utc"):
        assert key in written


def test_behaviour_column_widens_to_the_longest_name(tmp_path, checkpoint):
    task = task_for("push_front", 1)
    path = write_jsonl(tmp_path / "tasks.jsonl", [task])
    out = io.StringIO()
    solve.solve_file(
        path, system=SMALL, checkpoint=checkpoint, out=out, log=io.StringIO()
    )
    lines = out.getvalue().splitlines()
    assert lines[4] == "    #  behaviour   verdict     seconds  source"
    assert lines[5].startswith("    0  push_front  solved    ")


def test_main_runs_best_from_the_pinned_checkpoint(
    monkeypatch, tmp_path, checkpoint, capsys
):
    fetched = []

    def fetch(repo_id, revision):
        fetched.append((repo_id, revision))
        return checkpoint

    monkeypatch.setattr(solve, "fetch_stack_policy", fetch)
    # BEST's own depth; one-op tasks keep it fast.
    tasks = [task_for("add_pair", 1), task_for("add_pair", 2)]
    path = write_jsonl(tmp_path / "tasks.jsonl.gz", tasks, gzipped=True)
    report = tmp_path / "run.json"
    assert solve.main([str(path), "--json", str(report)]) == 0
    assert fetched == [(solve.BEST.repo_id, solve.BEST.revision)]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"system  {solve.BEST.describe()}"
    assert out[1] == f"policy  {checkpoint}"
    assert out[-1].startswith("solved 2/2 (coverage 1.000)")
    written = json.loads(report.read_text())
    assert written["checkpoint_is_best"] is True
    assert written["system"]["revision"] == solve.BEST.revision
