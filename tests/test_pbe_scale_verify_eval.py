"""Exercise aggregation with mocked verdicts; never execute generated programs."""

import gzip
import json
from types import SimpleNamespace

import pytest

from experiments.pbe_scale import eval_compare as compare
from experiments.pbe_scale import verify_eval as verify


@pytest.fixture
def case(tmp_path, monkeypatch):
    frozen = tmp_path / "eval-v4"
    frozen.mkdir()
    payload = frozen / "tasks.jsonl.gz"
    with gzip.open(payload, "wt") as stream:
        stream.writelines(
            json.dumps({"behaviour": name}) + "\n" for name in ("a", "a", "b")
        )
    compare.write(
        frozen / "input_manifest.json",
        {"tasks": 3, "behaviors": 2, "sha256": compare.sha(payload)},
    )
    for label, programs in (
        ("old", ["good", None, "bad"]),
        ("mixed", ["witness", "good", "unclear"]),
    ):
        checkpoint = tmp_path / label
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(label.encode())
        compare.write(checkpoint / "trainer_state.json", {"seed": 0, "epoch": 3})
        compare.write(checkpoint / "policy_config.json", {"encoding": "machine"})
        compare.write(
            checkpoint / "training_manifest.json",
            {"wandb_run.url": f"https://wandb.test/{label}"},
        )
        compare.write(
            frozen / f"{label}.json",
            {
                "checkpoint": str(checkpoint),
                "system": {"beam_width": 64},
                "host": {"threads": 8},
                "commit": "test",
                "rows": [
                    {
                        "index": i,
                        "behaviour": "a" if i < 2 else "b",
                        "solved": program is not None,
                        "program": [program] if program else None,
                    }
                    for i, program in enumerate(programs)
                ],
            },
        )
    monkeypatch.setattr(
        verify.cells, "CATALOGUE", [SimpleNamespace(name=name) for name in ("a", "b")]
    )
    calls = []

    def fake_verdicts(output, records, log):
        calls.append(records)
        kinds = {
            "good": "genuine",
            "witness": "equals_catalogue_witness",
            "bad": "shortcut",
            "unclear": "inconclusive",
        }
        verdicts = {
            verify.pair_key(row): {"verdict": kinds[row["program"][0]]}
            for record in records
            for row in record["rows"]
            if row["solved"]
        }
        compare.write(output / "verified.json", verdicts)
        log("private program contents must not appear in progress")
        return verdicts

    monkeypatch.setattr(verify.cells, "verified_programs", fake_verdicts)
    return (
        tmp_path,
        frozen / "old.json",
        frozen / "mixed.json",
        tmp_path / "verified",
        calls,
    )


def test_existing_verifier_aggregation_and_provenance(case, capsys):
    run, old, mixed, output, calls = case
    result = verify.verify(run, old, mixed, output)
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert result["distinct_program_pairs"] == 4
    assert (
        result["task_gain"],
        result["task_loss"],
        result["task_delta"],
        result["behavior_delta"],
    ) == (1, 0, 1, 0)
    assert result["counts"]["old"]["graded_tasks"] == 2
    assert result["counts"]["mixed"]["graded_tasks"] == 3
    assert result["counts"]["mixed"]["accepted_tasks"] == 2
    assert result["counts"]["mixed"]["accepted_behaviors"] == 1
    assert (
        result["counts"]["mixed"]["by_verdict"]["equals_catalogue_witness"]["tasks"]
        == 1
    )
    assert result["counts"]["old"]["by_verdict"]["shortcut"]["tasks"] == 1
    origin = result["provenance"]
    assert origin["inputs"]["old"]["sha256"] == compare.sha(old)
    assert origin["protocol"] == {
        "draws_per_mode": 10_000,
        "seed": 134,
        "modes": ["sampler", "perturbed", "dense"],
        "min_tested_per_mode": 500,
        "catalogue_equal_new_draws": 0,
    }
    assert origin["checkpoints"]["old"]["wandb_run.url"] == "https://wandb.test/old"
    assert compare.read(output / "summary.json")["status"] == "complete"
    assert "private program" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "malformed",
    ["missing-row", "wrong-behavior", "duplicate", "grade-string", "no-program"],
)
def test_bad_inputs_never_start_verification(case, malformed):
    run, old, mixed, output, calls = case
    record = compare.read(mixed)
    if malformed == "missing-row":
        record["rows"].pop()
    elif malformed == "wrong-behavior":
        record["rows"][0]["behaviour"] = "world-task"
    elif malformed == "duplicate":
        record["rows"][1] = record["rows"][0]
    elif malformed == "grade-string":
        record["rows"][0]["solved"] = "false"
    else:
        record["rows"][0]["program"] = None
    mixed.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"incomplete|boolean|missing"):
        verify.verify(run, old, mixed, output)
    assert not calls
    assert not output.exists()


def test_no_graded_solutions_is_a_complete_zero(case, monkeypatch):
    run, old, mixed, output, _ = case
    for path in (old, mixed):
        record = compare.read(path)
        for row in record["rows"]:
            row.update(solved=False, program=None)
        path.write_text(json.dumps(record))
    monkeypatch.setattr(verify.cells, "verified_programs", lambda *args: {})
    result = verify.verify(run, old, mixed, output)
    assert result["status"] == "complete"
    assert result["counts"]["mixed"]["accepted_tasks"] == 0
    assert result["distinct_program_pairs"] == 0
    assert compare.read(output / "verified.json") == {}


def test_unresolved_frozen_reference_is_not_a_failure_score(case, monkeypatch):
    run, old, mixed, output, calls = case
    monkeypatch.setattr(verify.cells, "CATALOGUE", [SimpleNamespace(name="a")])
    with pytest.raises(ValueError, match="unresolved frozen v4 references"):
        verify.verify(run, old, mixed, output)
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("failure", ["interrupted", "missing-pair", "unknown-verdict"])
def test_incomplete_verification_never_publishes_summary(case, monkeypatch, failure):
    run, old, mixed, output, _ = case

    def fail(*args):
        if failure == "interrupted":
            raise RuntimeError("memory-capped verification interrupted")
        return (
            {}
            if failure == "missing-pair"
            else {"a :: good": {"verdict": "no_sampler"}}
        )

    monkeypatch.setattr(verify.cells, "verified_programs", fail)
    with pytest.raises((RuntimeError, KeyError, ValueError)):
        verify.verify(run, old, mixed, output)
    assert (output / "input_manifest.json").exists()
    assert not (output / "summary.json").exists()
