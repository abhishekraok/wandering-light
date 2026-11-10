"""`python -m wandering_light.solve`: the best system, replayed on a task file."""

import dataclasses
import gzip
import io
import json
import random
from dataclasses import asdict
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError

from wandering_light import solve, stack_beam, stack_policy, stack_policy_hub
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.solve import BEST, Row, Summary, System
from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task, load
from wandering_light.stack_policy import PolicyScorer, StackPairPolicy, save_policy

TINY = PolicyArchitecture(8, 16, 32)
REVISION = "0123456789abcdef0123456789abcdef01234567"
# Depth 1 is exhaustive at beam 64, and so is depth 2 wherever the guard lets a
# first op through, so what these searches find does not depend on the weights.
SHALLOW = System("tiny", "acct/policy", REVISION, 64, 2, "both", True, "nowhere")


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def shown(task):
    return (
        tuple(parse_stack(text, basis=BASIS) for text in task.inputs),
        tuple(parse_stack(text, basis=BASIS) for text in task.outputs),
    )


def wrong_when_held_out(task):
    """The task with held-out answers its own witness does not give."""
    return dataclasses.replace(task, held_out_outputs=task.held_out_outputs[::-1])


@pytest.fixture
def scorer():
    return PolicyScorer(StackPairPolicy(TINY), "cpu")


@pytest.fixture
def checkpoint(tmp_path):
    return save_policy(
        tmp_path / "policy",
        StackPairPolicy(TINY),
        trainer_state={"best_epoch": 1},
        manifest={"arm": "tiny"},
        wandb_url="https://wandb.example/run",
    )


def write_jsonl(path: Path, tasks) -> Path:
    path.write_text("".join(json.dumps(asdict(task)) + "\n" for task in tasks))
    return path


# ---------------------------------------------------------------------------
# replay_program, moved beside the beam
# ---------------------------------------------------------------------------


def test_replay_charges_one_execution_per_op_per_example():
    task = task_for("tail")
    inputs, outputs = shown(task)
    fits, executions, fuel = replay_program(task.program, inputs, outputs)
    assert fits
    assert executions == len(task.program) * task.examples
    assert fuel > 0


def test_replay_that_runs_and_misses_is_charged_in_full():
    task = task_for("tail")
    inputs, outputs = shown(task)
    fits, executions, _fuel = replay_program(("uncons",), inputs, outputs)
    assert not fits
    assert executions == task.examples


def test_a_refused_op_ends_the_replay():
    task = task_for("add_pair", 6)
    inputs, outputs = shown(task)
    # `add` leaves one integer, so the second `add` is refused on the first
    # stack it meets: nothing after it runs and the refusal itself is free.
    fits, executions, _fuel = replay_program(("add", "add", "add"), inputs, outputs)
    assert not fits
    assert executions == task.examples


def test_the_experiment_reexports_replay_program():
    from experiments import cache_order_sim, self_shorten, solve_with_cache

    assert solve_with_cache.replay_program is stack_beam.replay_program
    assert self_shorten.replay_program is stack_beam.replay_program
    assert cache_order_sim.replay_program is stack_beam.replay_program


# ---------------------------------------------------------------------------
# stack_pbe.load on a plain payload
# ---------------------------------------------------------------------------


def test_load_reads_plain_and_gzipped_payloads_alike(tmp_path):
    tasks = [task_for("add_pair", 6), task_for("tail")]
    plain = write_jsonl(tmp_path / "tasks.jsonl", tasks)
    zipped = tmp_path / "tasks.jsonl.gz"
    zipped.write_bytes(gzip.compress(plain.read_bytes()))
    assert load(plain) == tasks
    assert load(zipped) == tasks
    assert isinstance(load(plain)[0].inputs, tuple)


def test_gzip_is_detected_by_content_not_by_name(tmp_path):
    tasks = [task_for("add_pair", 6)]
    plain = write_jsonl(tmp_path / "misnamed.jsonl.gz", tasks)
    zipped = tmp_path / "misnamed.jsonl"
    zipped.write_bytes(gzip.compress(plain.read_bytes()))
    assert load(plain) == tasks
    assert load(zipped) == tasks


def test_a_plain_payload_may_end_in_blank_lines(tmp_path):
    tasks = [task_for("add_pair", 6)]
    path = write_jsonl(tmp_path / "tasks.jsonl", tasks)
    path.write_text(path.read_text() + "\n\n")
    assert load(path) == tasks


# ---------------------------------------------------------------------------
# stack_policy_hub
# ---------------------------------------------------------------------------


def test_hub_names_are_the_stack_policy_constants():
    assert stack_policy_hub.REQUIRED_FILES == (
        stack_policy.MODEL_FILE,
        stack_policy.CONFIG_FILE,
        stack_policy.TRAINER_STATE_FILE,
        stack_policy.MANIFEST_FILE,
    )
    assert stack_policy_hub.OPTIONAL_FILES == (stack_policy.WANDB_LINK_FILE,)
    assert stack_policy_hub.MODEL_FILE == "model.safetensors"
    assert stack_policy_hub.DEFAULT_REPO_ID == (
        "abhishekraok/wandering-light-stack-policy"
    )


def test_cache_dir_follows_xdg_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert stack_policy_hub.default_cache_dir() == (
        tmp_path / "xdg" / "wandering-light" / "stack-policy"
    )
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert stack_policy_hub.default_cache_dir() == (
        tmp_path / "home" / ".cache" / "wandering-light" / "stack-policy"
    )
    assert stack_policy_hub.checkpoint_dir("acct/policy", REVISION) == (
        tmp_path / "home/.cache/wandering-light/stack-policy/acct__policy" / REVISION
    )


def test_checkpoint_dir_is_one_directory_per_commit(tmp_path):
    assert stack_policy_hub.checkpoint_dir(
        "acct/policy", REVISION, cache_dir=tmp_path
    ) == (tmp_path / "acct__policy" / REVISION)


def test_is_checkpoint_needs_every_required_file(checkpoint):
    assert stack_policy_hub.is_checkpoint(checkpoint)
    (checkpoint / stack_policy.WANDB_LINK_FILE).unlink()
    assert stack_policy_hub.is_checkpoint(checkpoint)
    (checkpoint / stack_policy.TRAINER_STATE_FILE).unlink()
    assert not stack_policy_hub.is_checkpoint(checkpoint)
    assert not stack_policy_hub.is_checkpoint(checkpoint / "absent")


def serve(monkeypatch, source: Path) -> list[tuple[str, str, str]]:
    """Stand ``source`` in for the Hub; return the list the downloads are logged to."""
    calls = []

    def fake_download(repo_id, filename, *, revision):
        calls.append((repo_id, filename, revision))
        if not (source / filename).is_file():
            raise EntryNotFoundError(f"{filename} is not in {repo_id}")
        return str(source / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


def test_fetch_downloads_once_then_reads_the_cache(tmp_path, monkeypatch, checkpoint):
    calls = serve(monkeypatch, checkpoint)
    cache = tmp_path / "cache"

    fetched = stack_policy_hub.fetch_stack_policy(
        "acct/policy", REVISION, cache_dir=cache
    )

    assert fetched == cache / "acct__policy" / REVISION
    assert calls == [
        ("acct/policy", name, REVISION)
        for name in (
            *stack_policy_hub.REQUIRED_FILES,
            *stack_policy_hub.OPTIONAL_FILES,
        )
    ]
    assert sorted(p.name for p in fetched.iterdir()) == sorted(
        p.name for p in checkpoint.iterdir()
    )
    for name in stack_policy_hub.REQUIRED_FILES:
        assert (fetched / name).read_bytes() == (checkpoint / name).read_bytes()
    stack_policy.load_policy(fetched)

    again = stack_policy_hub.fetch_stack_policy(
        "acct/policy", REVISION, cache_dir=cache
    )
    assert again == fetched
    assert len(calls) == 5  # the second fetch asked the Hub nothing


def test_fetch_skips_a_missing_optional_file(tmp_path, monkeypatch, checkpoint):
    (checkpoint / stack_policy.WANDB_LINK_FILE).unlink()
    calls = serve(monkeypatch, checkpoint)
    fetched = stack_policy_hub.fetch_stack_policy(
        "acct/policy", REVISION, cache_dir=tmp_path / "cache"
    )
    assert calls[-1][1] == stack_policy.WANDB_LINK_FILE
    assert stack_policy_hub.is_checkpoint(fetched)
    assert not (fetched / stack_policy.WANDB_LINK_FILE).exists()


def test_fetch_raises_on_a_missing_required_file(tmp_path, monkeypatch, checkpoint):
    (checkpoint / stack_policy.TRAINER_STATE_FILE).unlink()
    serve(monkeypatch, checkpoint)
    cache = tmp_path / "cache"
    with pytest.raises(EntryNotFoundError):
        stack_policy_hub.fetch_stack_policy("acct/policy", REVISION, cache_dir=cache)
    # What did arrive is not mistaken for a checkpoint by the next call.
    assert not stack_policy_hub.is_checkpoint(cache / "acct__policy" / REVISION)


@pytest.mark.parametrize(
    "revision", ["main", "e5a0b1391c52", REVISION.upper(), REVISION + "0", "", None]
)
def test_fetch_refuses_anything_but_a_commit_id(tmp_path, monkeypatch, revision):
    def no_download(*args, **kwargs):
        pytest.fail("a bad revision reached the Hub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    with pytest.raises(ValueError) as raised:
        stack_policy_hub.fetch_stack_policy("acct/policy", revision, cache_dir=tmp_path)
    assert str(raised.value) == (
        f"revision must be a 40-hex commit id, not {revision!r}"
    )


def test_publish_is_one_commit_of_the_checkpoint_and_its_card(monkeypatch, checkpoint):
    observed = {}

    class FakeOperation:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeApi:
        def create_repo(self, **kwargs):
            observed["repo"] = kwargs

        def create_commit(self, **kwargs):
            observed.setdefault("commits", []).append(kwargs)
            return type("Commit", (), {"oid": REVISION})()

    monkeypatch.setattr(huggingface_hub, "CommitOperationAdd", FakeOperation)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    (checkpoint / "optimizer.pt").write_bytes(b"not part of a checkpoint")

    revision = stack_policy_hub.publish_stack_policy(
        checkpoint, card="# tiny\n", commit_message="Publish tiny"
    )

    assert revision == REVISION
    assert observed["repo"] == {
        "repo_id": stack_policy_hub.DEFAULT_REPO_ID,
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    (commit,) = observed["commits"]
    assert commit["repo_id"] == stack_policy_hub.DEFAULT_REPO_ID
    assert commit["commit_message"] == "Publish tiny"
    uploaded = {op.path_in_repo: op.path_or_fileobj for op in commit["operations"]}
    assert set(uploaded) == {
        *stack_policy_hub.REQUIRED_FILES,
        *stack_policy_hub.OPTIONAL_FILES,
        "README.md",
    }
    assert uploaded["README.md"] == b"# tiny\n"
    assert uploaded[stack_policy.MODEL_FILE] == str(
        checkpoint / stack_policy.MODEL_FILE
    )


def test_publish_refuses_a_directory_that_is_not_a_checkpoint(tmp_path, monkeypatch):
    def no_api():
        pytest.fail("an incomplete checkpoint reached the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", no_api)
    with pytest.raises(FileNotFoundError):
        stack_policy_hub.publish_stack_policy(
            tmp_path, repo_id="acct/policy", card="", commit_message="no"
        )


# ---------------------------------------------------------------------------
# System, Row, Summary
# ---------------------------------------------------------------------------


def test_best_is_pinned_to_a_commit_and_says_so():
    expected = System(
        "staged-s0-e3",
        "abhishekraok/wandering-light-stack-policy",
        "e5a0b1391c5268c3ae04a49e6862f19a30779bb2",
        64,
        48,
        "both",
        True,
        BEST.measured,
    )
    assert expected == BEST
    assert BEST.measured
    assert BEST.repo_id == stack_policy_hub.DEFAULT_REPO_ID
    assert BEST.describe() == (
        "staged-s0-e3 = abhishekraok/wandering-light-stack-policy@e5a0b1391c52; "
        "beam 64 x depth 48, guard both, library on"
    )
    assert dataclasses.replace(BEST, library=False).describe().endswith("library off")
    with pytest.raises(dataclasses.FrozenInstanceError):
        BEST.beam_width = 1


def row(index=0, *, task=None, program=("add",), source="search", **fields):
    values = {
        "termination": "fit",
        "seconds": 0.25,
        "executions": 10,
        "fuel": 20,
        "expansions": 1,
        "depth": 1,
        **fields,
    }
    return Row(index, task or task_for("add_pair", 6), program, source, **values)


def test_row_verdict_is_decided_by_held_out_grading():
    assert row().verdict == "unsolved"
    assert row(shown_fit=True).verdict == "memorised"
    assert row(solved=True, shown_fit=True).verdict == "solved"
    assert (row().solved, row().shown_fit) == (False, False)


def test_row_record_is_json_and_names_the_task_without_its_examples():
    task = task_for("add_pair", 6)
    found = row(3, task=task, solved=True, shown_fit=True)
    assert found.record() == {
        "index": 3,
        "behaviour": "add_pair",
        "length": task.length,
        "examples": task.examples,
        "program": ["add"],
        "source": "search",
        "termination": "fit",
        "seconds": 0.25,
        "executions": 10,
        "fuel": 20,
        "expansions": 1,
        "depth": 1,
        "solved": True,
        "shown_fit": True,
    }
    missed = row(program=None, termination="max_depth").record()
    assert missed["program"] is None
    assert json.loads(json.dumps(missed)) == missed


def test_summary_counts_every_task_attempted():
    rows = [
        row(0, solved=True, shown_fit=True),
        row(1, source="library", solved=True, shown_fit=True, seconds=0.5),
        row(2, source="library", shown_fit=True, executions=5),
        row(3, program=None, termination="max_depth", seconds=2.0, executions=1200),
    ]
    summary = solve.summarise(rows)
    assert summary == Summary(
        tasks=4, solved=2, memorised=1, from_library=1, seconds=3.0, executions=1225
    )
    assert summary.coverage == 0.5
    assert summary.line() == (
        "solved 2/4 (coverage 0.500); wall clock 3.0 s over every task attempted; "
        "executions 1,225; 1 solved from the library; 1 memorised"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.solved = 3


def test_summary_of_nothing_has_zero_coverage():
    summary = solve.summarise([])
    assert summary == Summary(0, 0, 0, 0, 0, 0)
    assert summary.coverage == 0.0
    assert summary.line().startswith("solved 0/0 (coverage 0.000); ")


# ---------------------------------------------------------------------------
# solve_tasks
# ---------------------------------------------------------------------------


def test_a_program_found_earlier_is_replayed_instead_of_searched(scorer):
    first, second = task_for("add_pair", 6), task_for("add_pair", 7)
    rows = solve.solve_tasks([first, second], scorer, SHALLOW)

    assert [r.index for r in rows] == [0, 1]
    assert [r.task for r in rows] == [first, second]
    assert [r.source for r in rows] == ["search", "library"]
    assert [r.program for r in rows] == [("add",), ("add",)]
    assert [r.verdict for r in rows] == ["solved", "solved"]
    assert [r.termination for r in rows] == ["fit", "fit"]
    searched = beam_solve(first, scorer, beam_width=64, max_depth=2, guard="both")
    assert (rows[0].executions, rows[0].fuel, rows[0].expansions, rows[0].depth) == (
        searched.executions,
        searched.fuel,
        1,
        1,
    )
    _fits, executions, fuel = replay_program(("add",), *shown(second))
    assert (rows[1].executions, rows[1].fuel, rows[1].expansions, rows[1].depth) == (
        executions,
        fuel,
        0,
        0,
    )
    assert all(r.seconds > 0 for r in rows)


def test_a_library_miss_is_charged_on_top_of_the_search(scorer):
    first, second = task_for("tail"), task_for("add_pair", 6)
    rows = solve.solve_tasks([first, second], scorer, SHALLOW)

    assert [r.source for r in rows] == ["search", "search"]
    assert [r.verdict for r in rows] == ["solved", "solved"]
    _fits, replayed, burnt = replay_program(rows[0].program, *shown(second))
    searched = beam_solve(second, scorer, beam_width=64, max_depth=2, guard="both")
    assert rows[1].executions == replayed + searched.executions
    assert rows[1].fuel == burnt + searched.fuel


def test_the_library_is_tried_shortest_first_and_holds_no_duplicates(
    scorer, monkeypatch
):
    replayed = []

    def recording(program, inputs, outputs):
        replayed.append(program)
        return replay_program(program, inputs, outputs)

    monkeypatch.setattr(solve, "replay_program", recording)
    tasks = [
        task_for("tail"),  # two ops, found first
        task_for("add_pair", 6),  # one op, found second
        task_for("add_pair", 7),  # replays `add`, which now sorts ahead
        task_for("append"),  # misses both, in length order
        dataclasses.replace(  # nothing fits, so everything is tried
            task_for("append", 5), outputs=task_for("append", 6).outputs
        ),
    ]
    rows = solve.solve_tasks(tasks, scorer, SHALLOW)

    two_ops = rows[0].program
    assert len(two_ops) == 2
    assert rows[4].program is None
    sources = [r.source for r in rows]
    assert sources == ["search", "search", "library", "search", "search"]
    assert replayed == [
        # the first task meets an empty library
        two_ops,  # second: a miss, then a search that finds `add`
        ("add",),  # third: a hit on the first candidate
        *[("add",), two_ops],  # fourth: two misses, then a search that finds `cat`
        *[("add",), ("cat",), two_ops],  # fifth: all three miss
    ]


def test_library_off_searches_every_task(scorer, monkeypatch):
    def never(*args):
        pytest.fail("the library was replayed with the library off")

    monkeypatch.setattr(solve, "replay_program", never)
    system = dataclasses.replace(SHALLOW, library=False)
    rows = solve.solve_tasks(
        [task_for("add_pair", 6), task_for("add_pair", 7)], scorer, system
    )
    assert [r.source for r in rows] == ["search", "search"]
    assert [r.verdict for r in rows] == ["solved", "solved"]


def test_the_search_is_the_systems_width_depth_and_guard(scorer, monkeypatch):
    seen = []

    def recording(task, given, **settings):
        seen.append((given, settings))
        return beam_solve(task, given, **settings)

    monkeypatch.setattr(solve, "beam_solve", recording)
    system = dataclasses.replace(SHALLOW, beam_width=3, max_depth=1, guard="off")
    (only,) = solve.solve_tasks([task_for("tail")], scorer, system)
    assert seen == [(scorer, {"beam_width": 3, "max_depth": 1, "guard": "off"})]
    assert (only.program, only.termination, only.verdict) == (
        None,
        "max_depth",
        "unsolved",
    )


def test_a_shown_fit_that_fails_held_out_is_memorised_not_solved(scorer):
    honest = task_for("add_pair", 6)
    rows = solve.solve_tasks(
        [wrong_when_held_out(honest), wrong_when_held_out(task_for("add_pair", 7))],
        scorer,
        SHALLOW,
    )
    assert [r.program for r in rows] == [("add",), ("add",)]
    assert [r.source for r in rows] == ["search", "library"]
    assert [r.verdict for r in rows] == ["memorised", "memorised"]
    summary = solve.summarise(rows)
    assert (summary.solved, summary.memorised, summary.from_library) == (0, 2, 0)


def test_nothing_is_graded_until_the_last_clock_has_stopped(scorer, monkeypatch):
    events = []
    real_grade = solve.grade

    def searching(task, given, **settings):
        events.append("search")
        return beam_solve(task, given, **settings)

    def grading(program, task):
        events.append("grade")
        return real_grade(program, task)

    def progress(seen):
        events.append("progress")
        # The row is handed over when its clock stops, before it has a verdict.
        assert (seen.solved, seen.shown_fit, seen.verdict) == (
            False,
            False,
            "unsolved",
        )
        assert seen.seconds > 0

    monkeypatch.setattr(solve, "beam_solve", searching)
    monkeypatch.setattr(solve, "grade", grading)
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7), task_for("append")]
    rows = solve.solve_tasks(tasks, scorer, SHALLOW, progress=progress)

    assert events == [
        *["search", "progress"],
        *["progress"],  # the library answered; no search
        *["search", "progress"],
        *["grade", "grade", "grade"],
    ]
    assert [r.verdict for r in rows] == ["solved", "solved", "solved"]


# ---------------------------------------------------------------------------
# solve_file and the command
# ---------------------------------------------------------------------------


def test_solve_file_reports_every_task_and_the_totals(tmp_path, checkpoint):
    tasks = [
        task_for("add_pair", 6),
        task_for("add_pair", 7),
        wrong_when_held_out(task_for("append")),
    ]
    path = write_jsonl(tmp_path / "tasks.jsonl", tasks)
    out, log = io.StringIO(), io.StringIO()

    summary = solve.solve_file(
        path, system=SHALLOW, checkpoint=checkpoint, out=out, log=log
    )

    assert (summary.tasks, summary.solved, summary.memorised) == (3, 2, 1)
    assert summary.from_library == 1
    lines = out.getvalue().splitlines()
    assert lines[0] == f"system  {SHALLOW.describe()}"
    assert lines[1] == f"policy  {checkpoint}  (local, not BEST)"
    assert lines[2].startswith("host    ")
    assert lines[3] == f"tasks   3 from {path}"
    assert lines[4].split() == ["#", "behaviour", "verdict", "seconds", "source"]
    assert lines[4].index("behaviour") == lines[5].index("add_pair")
    assert lines[4].index("verdict") == lines[5].index("solved")
    table = [line.split() for line in lines[5:8]]
    assert [(r[0], r[1], r[2], r[4]) for r in table] == [
        ("0", "add_pair", "solved", "search"),
        ("1", "add_pair", "solved", "library"),
        ("2", "append", "memorised", "search"),
    ]
    assert lines[5].startswith("    0  add_pair   solved     ")
    assert lines[5].endswith("  search")
    assert len(lines[5].split("  ")[-2].strip()) == len("0.000")
    assert lines[8] == summary.line()
    assert len(lines) == 9
    assert len(log.getvalue().splitlines()) == 3
    assert "add_pair" in log.getvalue().splitlines()[0]


def test_the_behaviour_column_is_as_wide_as_its_longest_name(tmp_path, checkpoint):
    task = dataclasses.replace(task_for("add_pair", 6), behaviour="a_much_longer_name")
    path = write_jsonl(tmp_path / "tasks.jsonl", [task, task_for("add_pair", 7)])
    out = io.StringIO()
    solve.solve_file(
        path, system=SHALLOW, checkpoint=checkpoint, out=out, log=io.StringIO()
    )
    lines = out.getvalue().splitlines()
    assert lines[5].startswith("    0  a_much_longer_name  solved     ")
    assert lines[6].startswith("    1  add_pair            solved     ")
    assert lines[4].startswith("    #  behaviour           verdict    ")


def test_solve_file_fetches_the_systems_pinned_checkpoint(
    tmp_path, monkeypatch, checkpoint
):
    asked = []

    def fake_fetch(repo_id, revision):
        asked.append((repo_id, revision))
        return checkpoint

    monkeypatch.setattr(solve, "fetch_stack_policy", fake_fetch)
    path = tmp_path / "tasks.jsonl.gz"
    path.write_bytes(
        gzip.compress(json.dumps(asdict(task_for("add_pair", 6))).encode() + b"\n")
    )
    out, target = io.StringIO(), tmp_path / "runs" / "run.json"

    summary = solve.solve_file(
        path, system=SHALLOW, json_path=target, out=out, log=io.StringIO()
    )

    assert asked == [("acct/policy", REVISION)]
    assert out.getvalue().splitlines()[1] == f"policy  {checkpoint}"
    payload = json.loads(target.read_text())
    assert set(payload) == {
        "system",
        "checkpoint",
        "checkpoint_is_best",
        "host",
        "commit",
        "tasks_file",
        "recorded_at_utc",
        "summary",
        "rows",
    }
    assert payload["system"] == asdict(SHALLOW)
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["checkpoint_is_best"] is True
    assert payload["tasks_file"] == str(path)
    assert payload["summary"] == {**asdict(summary), "coverage": 1.0}
    (only,) = payload["rows"]
    assert only["program"] == ["add"]
    assert (only["behaviour"], only["solved"], only["source"]) == (
        "add_pair",
        True,
        "search",
    )


def test_a_local_checkpoint_is_recorded_as_not_the_best(
    tmp_path, monkeypatch, checkpoint
):
    def no_fetch(*args, **kwargs):
        pytest.fail("a local checkpoint was given and the Hub was asked anyway")

    monkeypatch.setattr(solve, "fetch_stack_policy", no_fetch)
    path = write_jsonl(tmp_path / "tasks.jsonl", [task_for("add_pair", 6)])
    target = tmp_path / "run.json"
    solve.solve_file(
        path,
        system=SHALLOW,
        checkpoint=checkpoint,
        json_path=target,
        out=io.StringIO(),
        log=io.StringIO(),
    )
    assert json.loads(target.read_text())["checkpoint_is_best"] is False


def test_solve_file_writes_to_stdout_and_stderr_by_default(
    tmp_path, checkpoint, capsys
):
    path = write_jsonl(tmp_path / "tasks.jsonl", [task_for("add_pair", 6)])
    summary = solve.solve_file(path, system=SHALLOW, checkpoint=checkpoint)
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == summary.line()
    assert len(captured.err.splitlines()) == 1


def test_an_empty_task_file_is_a_run_of_nothing(tmp_path, checkpoint):
    path = tmp_path / "tasks.jsonl"
    path.write_text("")
    out = io.StringIO()
    summary = solve.solve_file(
        path, system=SHALLOW, checkpoint=checkpoint, out=out, log=io.StringIO()
    )
    assert summary.tasks == 0
    assert out.getvalue().splitlines()[-1] == summary.line()


def test_main_runs_the_best_system_on_the_file(monkeypatch):
    calls = []

    def fake_solve_file(path, **kwargs):
        calls.append((path, kwargs))
        return Summary(0, 0, 0, 0, 0.0, 0)

    monkeypatch.setattr(solve, "solve_file", fake_solve_file)

    assert solve.main(["tasks.jsonl"]) == 0
    assert (
        solve.main(
            ["corpus", "--checkpoint", "local", "--device", "cuda", "--json", "r.json"]
        )
        == 0
    )

    (plain_path, plain), (full_path, full) = calls
    assert str(plain_path) == "tasks.jsonl"
    assert plain["system"] is BEST
    assert (plain["checkpoint"], plain["device"], plain["json_path"]) == (
        None,
        "cpu",
        None,
    )
    assert str(full_path) == "corpus"
    assert full["system"] is BEST
    assert (str(full["checkpoint"]), full["device"], str(full["json_path"])) == (
        "local",
        "cuda",
        "r.json",
    )
