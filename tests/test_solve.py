"""The one-command harness of best-system-command (#158) and its Hub fetch."""

import gzip
import json
import random
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from wandering_light import solve, stack_policy_hub
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task, load
from wandering_light.stack_policy import save_policy, train_policy, witness_rows

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def write_payload(path, tasks, *, gzipped=True):
    text = "".join(json.dumps(asdict(t), sort_keys=True) + "\n" for t in tasks)
    if gzipped:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A policy fit to `sum`'s own witness prefixes, as the beam tests do."""
    rows = witness_rows([task_for("sum", 4), task_for("sum", 5)])
    model, state = train_policy(
        rows,
        rows,
        TINY,
        epochs=40,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=8,
        seed=0,
        device="cpu",
    )
    return save_policy(
        tmp_path_factory.mktemp("policy") / "tiny",
        model,
        trainer_state=state,
        manifest={"arm": "tiny"},
        wandb_url=None,
    )


def small_system(depth: int) -> solve.System:
    return solve.System(
        name="tiny",
        repo_id="nobody/nothing",
        revision="0" * 40,
        beam_width=4,
        max_depth=depth,
        guard="both",
        library=True,
        measured="a test",
    )


def test_solves_grades_on_held_out_and_replays_the_library(tmp_path, checkpoint):
    first, second, long = (
        task_for("sum", 4),
        task_for("sum", 6),
        task_for("median_floor", 8),
    )
    assert long.length > first.length
    payload = write_payload(tmp_path / "tasks.jsonl.gz", [first, second, long])
    out_path = tmp_path / "run.json"
    with (
        open(tmp_path / "stdout.txt", "w") as out,
        open(tmp_path / "log.txt", "w") as log,
    ):
        summary = solve.solve_file(
            payload,
            system=small_system(first.length),
            checkpoint=checkpoint,
            json_path=out_path,
            out=out,
            log=log,
        )
    assert (summary.tasks, summary.solved) == (3, 2)
    assert summary.from_library == 1  # the second `sum` task fit the first's program
    record = json.loads(out_path.read_text())
    rows = record["rows"]
    assert [r["source"] for r in rows] == ["search", "library", "search"]
    assert [r["solved"] for r in rows] == [True, True, False]
    assert rows[1]["program"] == rows[0]["program"]
    assert rows[2]["termination"] in ("max_depth", "frontier_exhausted")
    assert record["checkpoint_is_best"] is False
    assert summary.seconds == pytest.approx(sum(r["seconds"] for r in rows))
    text = (tmp_path / "stdout.txt").read_text()
    assert text.startswith("system  tiny = nobody/nothing@000000000000; beam 4 x depth")
    assert "library on, state cap none" in text.splitlines()[0]
    assert text.rstrip().splitlines()[-1] == summary.line()
    assert "median_floor  unsolved" in text
    assert (tmp_path / "log.txt").read_text().count("\n") == 3


def test_an_empty_program_is_the_identity_and_none_is_no_solution(monkeypatch):
    base = task_for("sum", 4)
    identity = replace(base, outputs=base.inputs, held_out_outputs=base.held_out_inputs)
    found = iter([(), None])

    def fake_beam_solve(task, scorer, **_):
        return SimpleNamespace(
            program=next(found),
            termination="solved",
            executions=0,
            fuel=0,
            expansions=0,
            depth=0,
        )

    monkeypatch.setattr(solve, "beam_solve", fake_beam_solve)
    system = replace(small_system(2), library=False)
    rows = solve.solve_tasks([identity, identity], None, system)
    assert [r.program for r in rows] == [(), None]
    assert [r.record()["program"] for r in rows] == [[], None]
    assert [r.solved for r in rows] == [True, False]
    assert [r.verdict for r in rows] == ["solved", "unsolved"]


def test_a_plain_jsonl_payload_loads_like_a_gzipped_one(tmp_path):
    tasks = [task_for("sum", 4), task_for("sum", 6)]
    plain = write_payload(tmp_path / "tasks.jsonl", tasks, gzipped=False)
    zipped = write_payload(tmp_path / "tasks.jsonl.gz", tasks)
    assert load(plain) == load(zipped) == tasks


def test_main_runs_best_search_settings_on_a_local_checkpoint(
    tmp_path, checkpoint, capsys
):
    payload = write_payload(tmp_path / "one.jsonl.gz", [task_for("sum", 4)])
    assert solve.main([str(payload), "--checkpoint", str(checkpoint)]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"system  {solve.BEST.describe()}\n")
    assert "(local, not BEST)" in text
    assert (
        text.rstrip()
        .splitlines()[-1]
        .startswith("solved 1/1 (coverage 1.000); wall clock ")
    )


def test_fetch_replaces_an_incomplete_cache_directory(tmp_path, monkeypatch):
    """A non-empty directory that is not a checkpoint must not wedge every later fetch."""
    import huggingface_hub

    def fake_download(repo_id, filename, revision=None):
        if filename == stack_policy_hub.WANDB_LINK_FILE:
            raise huggingface_hub.errors.EntryNotFoundError("no run")
        source = tmp_path / "hub" / filename
        source.parent.mkdir(exist_ok=True)
        source.write_text(filename)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    revision = "cd" * 20
    target = stack_policy_hub.checkpoint_dir(
        "x/y", revision, cache_dir=tmp_path / "cache"
    )
    target.mkdir(parents=True)
    (target / stack_policy_hub.MODEL_FILE).write_text("half a download")
    assert not stack_policy_hub.is_checkpoint(target)

    got = stack_policy_hub.fetch_stack_policy(
        "x/y", revision, cache_dir=tmp_path / "cache"
    )
    assert got == target and stack_policy_hub.is_checkpoint(got)
    assert (
        got / stack_policy_hub.MODEL_FILE
    ).read_text() == stack_policy_hub.MODEL_FILE
    # Nothing is left beside it but the install lock: no staging directory, no stale copy.
    assert sorted(p.name for p in target.parent.iterdir()) == sorted(
        [revision, f".{revision}.lock"]
    )


def test_a_repairing_fetch_never_discards_a_checkpoint_another_fetch_returned(
    tmp_path, monkeypatch
):
    """Two fetches repairing the same incomplete directory must not undo each other.

    Fetch A finds the directory incomplete and is about to discard it. If fetch
    B can install a complete copy and return it in that window, A's discard
    deletes the checkpoint B's caller is about to load.
    """
    import threading

    import huggingface_hub

    def fake_download(repo_id, filename, revision=None):
        if filename == stack_policy_hub.WANDB_LINK_FILE:
            raise huggingface_hub.errors.EntryNotFoundError("no run")
        source = tmp_path / "hub" / filename
        source.parent.mkdir(exist_ok=True)
        source.write_text(filename)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    revision = "ef" * 20
    cache = tmp_path / "cache"
    target = stack_policy_hub.checkpoint_dir("x/y", revision, cache_dir=cache)
    target.mkdir(parents=True)
    (target / stack_policy_hub.MODEL_FILE).write_text("half a download")

    real_discard = stack_policy_hub._discard
    b_returned = threading.Event()
    discarded_after_b_returned = []

    def fetch_b():
        stack_policy_hub.fetch_stack_policy("x/y", revision, cache_dir=cache)
        b_returned.set()

    b = threading.Thread(target=fetch_b)

    def discard(path):
        if not b.is_alive() and not b_returned.is_set():
            # A is about to discard: give B the chance to finish first. With the
            # install serialised B cannot, and this wait simply times out.
            b.start()
            b_returned.wait(timeout=0.5)
        if b_returned.is_set() and path == target:
            discarded_after_b_returned.append(path)
        real_discard(path)

    monkeypatch.setattr(stack_policy_hub, "_discard", discard)
    got = stack_policy_hub.fetch_stack_policy("x/y", revision, cache_dir=cache)
    b.join(timeout=10)
    assert b_returned.is_set()
    assert not discarded_after_b_returned
    assert got == target and stack_policy_hub.is_checkpoint(target)


def test_fetch_pins_a_commit_and_downloads_once(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_download(repo_id, filename, revision=None):
        calls.append(filename)
        if filename == stack_policy_hub.WANDB_LINK_FILE:
            raise huggingface_hub.errors.EntryNotFoundError("no run")
        source = tmp_path / "hub" / filename
        source.parent.mkdir(exist_ok=True)
        source.write_text(filename)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    revision = "ab" * 20
    got = stack_policy_hub.fetch_stack_policy(
        "x/y", revision, cache_dir=tmp_path / "cache"
    )
    assert got == tmp_path / "cache" / "x__y" / revision
    assert stack_policy_hub.is_checkpoint(got)
    assert not (got / stack_policy_hub.WANDB_LINK_FILE).exists()
    assert sorted(calls) == sorted(
        stack_policy_hub.REQUIRED_FILES + stack_policy_hub.OPTIONAL_FILES
    )
    again = stack_policy_hub.fetch_stack_policy(
        "x/y", revision, cache_dir=tmp_path / "cache"
    )
    assert again == got and len(calls) == 5
    with pytest.raises(ValueError, match="commit id"):
        stack_policy_hub.fetch_stack_policy("x/y", "main", cache_dir=tmp_path / "cache")
