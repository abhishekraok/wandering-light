"""The one-command harness of best-system-command (#158) and its Hub fetch."""

import gzip
import json
import random
from dataclasses import asdict

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
