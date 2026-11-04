"""One command runs the best system on a task file: ``python -m wandering_light.solve``.

Three things are promised and checked here. The clock covers the library
replay and the search and nothing else; grading happens after the last clock
stops. A program solved earlier in the run is replayed before the next search,
so the second task of a behaviour comes from the library. And the checkpoint is
pinned to a commit id, fetched once, and read from the cache after that.
"""

from __future__ import annotations

import gzip
import io
import json
import random
from dataclasses import asdict
from pathlib import Path  # noqa: TC003

import pytest

from wandering_light import stack_beam, stack_pbe, stack_policy_hub
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_policy import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
)

torch = pytest.importorskip("torch")

from wandering_light.rollout_policy import PolicyArchitecture  # noqa: E402
from wandering_light.solve import (  # noqa: E402
    BEST,
    Row,
    Summary,
    System,
    main,
    solve_file,
    solve_tasks,
    summarise,
)
from wandering_light.stack_policy import (  # noqa: E402
    PolicyScorer,
    StackPairPolicy,
    save_policy,
)

TINY = PolicyArchitecture(8, 16, 32)
SMALL = System("tiny", "acct/repo", "0" * 40, 64, 4, "both", True, "nowhere")
COMMIT = "e5a0b1391c5268c3ae04a49e6862f19a30779bb2"


def task_for(name: str, seed: int):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def product_of(texts):
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


@pytest.fixture
def scorer():
    torch.manual_seed(0)
    return PolicyScorer(StackPairPolicy(TINY), "cpu")


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    return save_policy(
        tmp_path / "ckpt",
        StackPairPolicy(TINY),
        trainer_state={"epoch": 3},
        manifest={"arm": "tiny"},
        wandb_url="https://wandb.ai/x/y/runs/z",
    )


# ---------------------------------------------------------------------------
# replay_program lives in stack_beam and is charged like the beam
# ---------------------------------------------------------------------------


def test_replay_program_fits_and_charges_one_execution_per_op_per_example():
    task = task_for("add_pair", 6)
    inputs, outputs = product_of(task.inputs), product_of(task.outputs)

    fits, executions, fuel = stack_beam.replay_program(("add",), inputs, outputs)

    assert fits
    assert executions == task.examples
    assert fuel > 0


def test_replay_program_stops_at_the_first_refused_op():
    task = task_for("add_pair", 6)
    inputs, outputs = product_of(task.inputs), product_of(task.outputs)

    # ``i`` needs a quotation on top; an integer pair refuses it on the first
    # example, so nothing after the refusal is charged.
    fits, executions, _fuel = stack_beam.replay_program(("i", "add"), inputs, outputs)

    assert not fits
    assert executions == 0
    fits, executions, _fuel = stack_beam.replay_program(("add", "i"), inputs, outputs)
    assert not fits
    assert executions == task.examples


def test_replay_program_rejects_a_program_that_runs_but_misses():
    task = task_for("add_pair", 6)
    inputs, outputs = product_of(task.inputs), product_of(task.outputs)

    fits, executions, _fuel = stack_beam.replay_program(("sub",), inputs, outputs)

    assert not fits
    assert executions == task.examples


def test_experiment_re_exports_replay_program():
    experiment = pytest.importorskip("experiments.solve_with_cache")
    assert experiment.replay_program is stack_beam.replay_program


# ---------------------------------------------------------------------------
# A plain .jsonl payload loads like a gzipped one
# ---------------------------------------------------------------------------


def test_load_payload_reads_plain_and_gzipped_jsonl(tmp_path):
    tasks = [task_for("add_pair", 1), task_for("sum", 2)]
    lines = "".join(json.dumps(asdict(t), sort_keys=True) + "\n" for t in tasks)
    plain = tmp_path / "tasks.jsonl"
    plain.write_text(lines)
    zipped = tmp_path / "tasks.jsonl.gz"
    zipped.write_bytes(gzip.compress(lines.encode()))
    # The suffix lies: gzip bytes under a .jsonl name, text under a .gz name.
    lying_gz = tmp_path / "lying.jsonl"
    lying_gz.write_bytes(gzip.compress(lines.encode()))
    lying_plain = tmp_path / "lying.jsonl.gz"
    lying_plain.write_text(lines)

    for path in (plain, zipped, lying_gz, lying_plain):
        assert stack_pbe.load(path) == tasks


# ---------------------------------------------------------------------------
# stack_policy_hub
# ---------------------------------------------------------------------------


def test_hub_exposes_the_checkpoint_file_names():
    assert stack_policy_hub.REQUIRED_FILES == (
        MODEL_FILE,
        CONFIG_FILE,
        TRAINER_STATE_FILE,
        MANIFEST_FILE,
    )
    assert stack_policy_hub.OPTIONAL_FILES == (WANDB_LINK_FILE,)
    assert stack_policy_hub.MODEL_FILE == MODEL_FILE
    assert stack_policy_hub.WANDB_LINK_FILE == WANDB_LINK_FILE
    assert (
        stack_policy_hub.DEFAULT_REPO_ID == "abhishekraok/wandering-light-stack-policy"
    )


def test_default_cache_dir_follows_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert stack_policy_hub.default_cache_dir() == (
        tmp_path / "xdg" / "wandering-light" / "stack-policy"
    )
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert stack_policy_hub.default_cache_dir() == (
        tmp_path / "home" / ".cache" / "wandering-light" / "stack-policy"
    )


def test_checkpoint_dir_is_keyed_by_repo_and_revision(tmp_path):
    assert stack_policy_hub.checkpoint_dir("a/b", COMMIT, cache_dir=tmp_path) == (
        tmp_path / "a__b" / COMMIT
    )


def test_is_checkpoint_needs_every_required_file(checkpoint, tmp_path):
    assert stack_policy_hub.is_checkpoint(checkpoint)
    assert not stack_policy_hub.is_checkpoint(tmp_path / "missing")
    (checkpoint / WANDB_LINK_FILE).unlink()
    assert stack_policy_hub.is_checkpoint(checkpoint)
    (checkpoint / MANIFEST_FILE).unlink()
    assert not stack_policy_hub.is_checkpoint(checkpoint)


@pytest.mark.parametrize("revision", ["main", COMMIT[:12], COMMIT.upper(), ""])
def test_fetch_refuses_anything_but_a_commit_id(revision, tmp_path):
    with pytest.raises(ValueError, match="40-hex commit id"):
        stack_policy_hub.fetch_stack_policy("a/b", revision, cache_dir=tmp_path)


def test_fetch_returns_a_cached_checkpoint_without_the_hub(
    checkpoint, tmp_path, monkeypatch
):
    import huggingface_hub

    def no_network(*args, **kwargs):
        pytest.fail("the Hub was asked for a checkpoint already in the cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    target = stack_policy_hub.checkpoint_dir("a/b", COMMIT, cache_dir=tmp_path)
    target.parent.mkdir(parents=True)
    checkpoint.rename(target)

    assert stack_policy_hub.fetch_stack_policy("a/b", COMMIT, cache_dir=tmp_path) == (
        target
    )


def test_fetch_downloads_each_file_pinned_to_the_revision(
    checkpoint, tmp_path, monkeypatch
):
    import huggingface_hub

    calls = []

    def fake_download(repo_id, filename, *, revision):
        calls.append((repo_id, filename, revision))
        return str(checkpoint / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cache = tmp_path / "cache"

    fetched = stack_policy_hub.fetch_stack_policy("a/b", COMMIT, cache_dir=cache)

    assert fetched == cache / "a__b" / COMMIT
    assert stack_policy_hub.is_checkpoint(fetched)
    assert (fetched / WANDB_LINK_FILE).read_bytes() == (
        checkpoint / WANDB_LINK_FILE
    ).read_bytes()
    assert [c[1] for c in calls] == [
        *stack_policy_hub.REQUIRED_FILES,
        *stack_policy_hub.OPTIONAL_FILES,
    ]
    assert {(c[0], c[2]) for c in calls} == {("a/b", COMMIT)}
    # Second call: the cache answers and the Hub is not asked again.
    calls.clear()
    assert stack_policy_hub.fetch_stack_policy("a/b", COMMIT, cache_dir=cache) == (
        fetched
    )
    assert calls == []


def test_fetch_skips_a_missing_optional_file_and_raises_on_a_required_one(
    checkpoint, tmp_path, monkeypatch
):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    missing = {WANDB_LINK_FILE}

    def fake_download(repo_id, filename, *, revision):
        if filename in missing:
            raise EntryNotFoundError(f"no {filename}")
        return str(checkpoint / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    fetched = stack_policy_hub.fetch_stack_policy(
        "a/b", COMMIT, cache_dir=tmp_path / "one"
    )
    assert stack_policy_hub.is_checkpoint(fetched)
    assert not (fetched / WANDB_LINK_FILE).exists()

    missing.add(MANIFEST_FILE)
    with pytest.raises(EntryNotFoundError):
        stack_policy_hub.fetch_stack_policy("a/b", COMMIT, cache_dir=tmp_path / "two")


def test_publish_makes_one_commit_of_the_files_and_the_card(checkpoint, monkeypatch):
    import huggingface_hub

    created = {}
    committed = {}

    class FakeApi:
        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            created.update(repo_id=repo_id, repo_type=repo_type, private=private)
            assert exist_ok

        def create_commit(self, *, repo_id, repo_type, operations, commit_message):
            committed.update(
                repo_id=repo_id,
                repo_type=repo_type,
                message=commit_message,
                paths=[op.path_in_repo for op in operations],
                card=next(
                    op.path_or_fileobj
                    for op in operations
                    if op.path_in_repo == "README.md"
                ),
            )
            return type("Commit", (), {"oid": COMMIT})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    revision = stack_policy_hub.publish_stack_policy(
        checkpoint, card="# card\n", commit_message="Publish tiny"
    )

    assert revision == COMMIT
    assert created == {
        "repo_id": stack_policy_hub.DEFAULT_REPO_ID,
        "repo_type": "model",
        "private": True,
    }
    assert committed["message"] == "Publish tiny"
    assert committed["paths"] == [
        *stack_policy_hub.REQUIRED_FILES,
        *stack_policy_hub.OPTIONAL_FILES,
        "README.md",
    ]
    assert committed["card"] == b"# card\n"


def test_publish_refuses_a_directory_that_is_not_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        stack_policy_hub.publish_stack_policy(
            tmp_path, card="", commit_message="nothing"
        )


# ---------------------------------------------------------------------------
# System, Row, Summary
# ---------------------------------------------------------------------------


def test_best_is_pinned_to_a_commit_and_describes_itself():
    assert BEST.name == "staged-s0-e3"
    assert BEST.repo_id == stack_policy_hub.DEFAULT_REPO_ID
    assert BEST.revision == COMMIT
    assert (BEST.beam_width, BEST.max_depth, BEST.guard, BEST.library) == (
        64,
        48,
        "both",
        True,
    )
    assert BEST.measured
    assert BEST.describe() == (
        "staged-s0-e3 = abhishekraok/wandering-light-stack-policy@e5a0b1391c52; "
        "beam 64 x depth 48, guard both, library on"
    )
    assert SMALL.describe().endswith("library on")
    with pytest.raises(AttributeError):
        BEST.beam_width = 1  # frozen


def test_row_verdict_and_record():
    task = task_for("add_pair", 6)
    row = Row(3, task, ("add",), "library", "fit", 0.5, 4, 8, 0, 0)
    assert row.verdict == "unsolved"
    row.shown_fit = True
    assert row.verdict == "memorised"
    row.solved = True
    assert row.verdict == "solved"

    record = row.record()
    assert record == {
        "index": 3,
        "behaviour": "add_pair",
        "length": task.length,
        "examples": task.examples,
        "program": ["add"],
        "source": "library",
        "termination": "fit",
        "seconds": 0.5,
        "executions": 4,
        "fuel": 8,
        "expansions": 0,
        "depth": 0,
        "solved": True,
        "shown_fit": True,
        "verdict": "solved",
    }
    json.dumps(record)
    assert (
        Row(0, task, None, "search", "max_depth", 1.0, 1, 1, 1, 1).record()["program"]
        is None
    )


def test_summary_line_and_coverage():
    assert Summary(0, 0, 0, 0, 0.0, 0).coverage == 0.0
    summary = Summary(4, 3, 1, 2, 12.345, 123456)
    assert summary.coverage == 0.75
    assert summary.line() == (
        "solved 3/4 (coverage 0.750); wall clock 12.3 s over every task attempted; "
        "executions 123,456; 2 solved from the library; 1 memorised"
    )


def test_summarise_sums_rows_and_counts_solved_library_rows():
    task = task_for("add_pair", 6)
    rows = [
        Row(0, task, ("add",), "search", "fit", 1.0, 10, 1, 1, 1, solved=True),
        Row(1, task, ("add",), "library", "fit", 0.5, 4, 1, 0, 0, solved=True),
        Row(2, task, ("sub",), "library", "fit", 0.5, 4, 1, 0, 0, shown_fit=True),
        Row(3, task, None, "search", "max_depth", 2.0, 100, 1, 9, 4),
    ]
    assert summarise(rows) == Summary(4, 2, 1, 1, 4.0, 118)


# ---------------------------------------------------------------------------
# solve_tasks
# ---------------------------------------------------------------------------


def test_second_task_of_a_behaviour_is_solved_from_the_library(scorer):
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7)]
    seen = []

    rows = solve_tasks(tasks, scorer, SMALL, progress=seen.append)

    assert [r.index for r in rows] == [0, 1]
    assert seen == rows
    first, second = rows
    assert (first.source, first.termination) == ("search", "fit")
    assert first.program == ("add",)
    assert first.expansions == 1 and first.depth == 1
    assert (second.source, second.termination) == ("library", "fit")
    assert second.program == ("add",)
    assert second.expansions == 0
    assert second.executions == tasks[1].examples  # one replay, one op
    assert second.executions < first.executions
    assert all(r.solved for r in rows)
    assert all(r.seconds > 0 for r in rows)
    assert summarise(rows) == Summary(
        2, 2, 0, 1, first.seconds + second.seconds, first.executions + second.executions
    )


def test_progress_sees_rows_before_they_are_graded(scorer):
    graded_when_seen = []

    def progress(row):
        graded_when_seen.append((row.solved, row.shown_fit))

    rows = solve_tasks([task_for("add_pair", 6)], scorer, SMALL, progress=progress)

    assert graded_when_seen == [(False, False)]
    assert rows[0].solved


def test_library_off_searches_every_task(scorer):
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7)]
    off = System("off", "a/b", "0" * 40, 64, 4, "both", False, "")

    rows = solve_tasks(tasks, scorer, off)

    assert [r.source for r in rows] == ["search", "search"]
    assert all(r.solved for r in rows)


def test_library_replay_is_charged_and_a_miss_falls_through_to_search(scorer):
    tasks = [task_for("add_pair", 6), task_for("push_front", 8)]

    rows = solve_tasks(tasks, scorer, SMALL)

    assert rows[1].source == "search"
    assert rows[1].program == ("cons",)
    assert rows[1].solved
    # The replay of ``add`` on the second task was paid for inside its clock:
    # the row's cost is the search's plus what the replay charged, and the
    # same search with the library off costs exactly the search.
    _fits, replayed, replayed_fuel = stack_beam.replay_program(
        ("add",), product_of(tasks[1].inputs), product_of(tasks[1].outputs)
    )
    off = System("off", "a/b", "0" * 40, 64, 4, "both", False, "")
    alone = solve_tasks([tasks[1]], scorer, off)[0]
    assert rows[1].executions == alone.executions + replayed
    assert rows[1].fuel == alone.fuel + replayed_fuel


def test_an_unsolved_task_is_a_row_with_no_program(scorer):
    task = task_for("add_pair", 6)
    hopeless = System("none", "a/b", "0" * 40, 1, 0, "both", True, "")

    rows = solve_tasks([task], scorer, hopeless)

    assert rows[0].program is None
    assert rows[0].termination == "max_depth"
    assert rows[0].verdict == "unsolved"
    assert summarise(rows).coverage == 0.0


# ---------------------------------------------------------------------------
# solve_file and main
# ---------------------------------------------------------------------------


def write_tasks(path: Path, tasks) -> Path:
    path.write_text(
        "".join(json.dumps(asdict(t), sort_keys=True) + "\n" for t in tasks)
    )
    return path


def test_solve_file_prints_header_table_and_summary(checkpoint, tmp_path):
    tasks = [task_for("add_pair", 6), task_for("add_pair", 7)]
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", tasks)
    out, log = io.StringIO(), io.StringIO()

    summary = solve_file(
        tasks_path,
        system=SMALL,
        checkpoint=checkpoint,
        json_path=tmp_path / "out" / "run.json",
        out=out,
        log=log,
    )

    assert summary.solved == 2 and summary.from_library == 1
    lines = out.getvalue().splitlines()
    assert lines[0] == f"system  {SMALL.describe()}"
    assert lines[1] == f"policy  {checkpoint}  (local, not BEST)"
    assert lines[2].startswith("host    ")
    assert lines[3] == f"tasks   2 from {tasks_path}"
    assert lines[4].split() == ["#", "behaviour", "verdict", "seconds", "source"]
    assert lines[5].split()[:3] == ["0", "add_pair", "solved"]
    assert lines[5].endswith("search")
    assert lines[6].split()[:3] == ["1", "add_pair", "solved"]
    assert lines[6].endswith("library")
    assert lines[7] == summary.line()
    assert len(lines) == 8
    assert len(log.getvalue().splitlines()) == 2
    assert "add_pair" in log.getvalue()

    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["system"] == asdict(SMALL)
    assert record["checkpoint"] == str(checkpoint)
    assert record["checkpoint_is_best"] is False
    assert record["tasks_file"] == str(tasks_path)
    assert record["recorded_at_utc"]
    assert set(record["host"]) == {
        "hostname",
        "platform",
        "threads",
        "torch",
        "cuda_device",
    }
    assert "commit" in record
    assert record["summary"] == {**asdict(summary), "coverage": 1.0}
    assert [r["verdict"] for r in record["rows"]] == ["solved", "solved"]
    assert [r["source"] for r in record["rows"]] == ["search", "library"]


def test_solve_file_fetches_best_when_no_checkpoint_is_given(
    checkpoint, tmp_path, monkeypatch
):
    from wandering_light import solve

    asked = []

    def fake_fetch(repo_id, revision):
        asked.append((repo_id, revision))
        return checkpoint

    monkeypatch.setattr(solve, "fetch_stack_policy", fake_fetch)
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [task_for("add_pair", 6)])
    out = io.StringIO()

    solve_file(
        tasks_path,
        system=SMALL,
        json_path=tmp_path / "run.json",
        out=out,
        log=io.StringIO(),
    )

    assert asked == [(SMALL.repo_id, SMALL.revision)]
    assert f"policy  {checkpoint}\n" in out.getvalue()
    assert "(local, not BEST)" not in out.getvalue()
    assert json.loads((tmp_path / "run.json").read_text())["checkpoint_is_best"]


def test_table_pads_the_behaviour_column_to_the_longest_name(checkpoint, tmp_path):
    tasks = [task_for("add_pair", 6), task_for("push_front", 8)]
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", tasks)
    out = io.StringIO()

    solve_file(
        tasks_path, system=SMALL, checkpoint=checkpoint, out=out, log=io.StringIO()
    )

    header, first, second = out.getvalue().splitlines()[4:7]
    assert header.index("verdict") == first.index("solved") == second.index("solved")


def test_main_runs_best_on_the_file(checkpoint, tmp_path, monkeypatch, capsys):
    from wandering_light import solve

    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [task_for("add_pair", 6)])
    seen = {}

    def fake_solve_file(path, *, system, checkpoint, device, json_path):
        seen.update(
            path=path,
            system=system,
            checkpoint=checkpoint,
            device=device,
            json_path=json_path,
        )
        return Summary(1, 1, 0, 0, 0.1, 10)

    monkeypatch.setattr(solve, "solve_file", fake_solve_file)

    assert main([str(tasks_path)]) == 0
    assert seen == {
        "path": tasks_path,
        "system": BEST,
        "checkpoint": None,
        "device": "cpu",
        "json_path": None,
    }
    assert (
        main(
            [
                str(tasks_path),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--json",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    assert seen["checkpoint"] == checkpoint
    assert seen["json_path"] == tmp_path / "r.json"


def test_main_end_to_end_with_a_local_checkpoint(checkpoint, tmp_path, capsys):
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [task_for("add_pair", 6)])

    assert main([str(tasks_path), "--checkpoint", str(checkpoint)]) == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("system  ")
    assert "solved 1/1 (coverage 1.000)" in captured.out
    assert "add_pair" in captured.err
