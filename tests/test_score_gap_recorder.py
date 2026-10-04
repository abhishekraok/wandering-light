"""Synthetic production callback/file integration; not independent verification."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.score_gap_lineage import same_node
from experiments.score_gap_recorder import CellRecorder
from experiments.score_gap_records import decode_attempts
from experiments.score_gap_storage import (
    DiagnosticProvenanceError,
    DiagnosticResourceError,
    PayloadBudget,
    decode_array,
    decode_beam,
)
from tests.test_score_gap_lineage import ScriptedScorer, task
from wandering_light.stack_beam import beam_solve


def ledger(**kwargs):
    return PayloadBudget(deadline=10.0, clock=lambda: 0.0, **kwargs)


def events(directory):
    return [
        json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize(("max_depth", "output"), [(1, "999"), (3, "12"), (3, "6")])
def test_complete_callback_frames_bit_exact_sources_and_final_layer(
    tmp_path, max_depth, output
):
    directory = tmp_path / "cell"
    budget = ledger()
    with CellRecorder(directory, budget, {"synthetic": True}) as recorder:
        result = beam_solve(
            task(outputs=(output,)),
            ScriptedScorer(),
            beam_width=8,
            max_depth=max_depth,
            guard="off",
            observer=recorder,
        )
        seal = recorder.finish(result)
    rows = events(directory)
    assert [row["kind"] for row in rows][-2:] == ["terminal", "result"]
    assert json.loads((directory / "cell-seal.json").read_text()) == json.loads(
        json.dumps(seal)
    )
    assert budget.bytes_written == sum(p.stat().st_size for p in directory.iterdir())
    for name, record in seal["files"].items():
        payload = (directory / name).read_bytes()
        assert record == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    attempts = list(decode_attempts((directory / "attempts.bin").read_bytes()))
    assert len(attempts) == budget.attempts_written == seal["attempts"]
    assert sum(row.stepped for row in attempts) == result.executions
    assert sum(row.fuel for row in attempts) == result.fuel
    beams, scores = (
        (directory / "beams.bin").read_bytes(),
        (directory / "scores.bin").read_bytes(),
    )
    stored_states = []
    for row in rows:
        if row["kind"] == "started":
            assert row["attempt_offset"] == next(
                (
                    i * 19
                    for i, attempt in enumerate(attempts)
                    if attempt.depth == row["depth"]
                ),
                len(attempts) * 19,
            )
            if row["depth"]:
                previous = next(
                    x
                    for x in rows
                    if x["kind"] == "retained" and x["depth"] == row["depth"] - 1
                )
                assert row["state"] == previous["state"]
                assert row["source_retention"] == row["depth"] - 1
            else:
                assert row["source_retention"] is None
                stored_states.append(row["state"])
        if row["kind"] == "retained":
            stored_states.append(row["state"])
        if row["kind"] in ("started", "retained"):
            record = row["state"]
            payload = beams[record["offset"] : record["offset"] + record["bytes"]]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
            restored = decode_beam(payload)
            expected = (
                recorder.beam
                if row["kind"] == "started" and row["depth"] == recorder.depth
                else None
            )
            if expected is not None:
                assert all(
                    same_node(a, b) for a, b in zip(restored, expected, strict=True)
                )
        if row["kind"] == "scored":
            for name in ("scores", "legal"):
                record = row[name]
                payload = scores[record["offset"] : record["offset"] + record["bytes"]]
                assert hashlib.sha256(payload).hexdigest() == record["sha256"]
                decode_array(payload)
    assert sum(state["bytes"] for state in stored_states) == len(beams)
    if output == "6":
        assert result.program == ("mul",)
        assert not any(row["kind"] in ("scored", "retained") for row in rows)
    if max_depth == 1:
        assert [row["kind"] for row in rows][-3] == "retained"


@pytest.mark.parametrize("stop", ["budget", "wall_clock_cap"])
def test_zero_attempt_parent_entry_charged_and_partial_layer_sealed(tmp_path, stop):
    kwargs = {"budget": 0} if stop == "budget" else {"deadline": 0.0}
    directory = tmp_path / "cell"
    with CellRecorder(directory, ledger(), {}) as recorder:
        result = beam_solve(task(), ScriptedScorer(), observer=recorder, **kwargs)
        seal = recorder.finish(result)
    assert result.termination == stop
    assert (result.expansions, result.executions, result.fuel) == (1, 0, 0)
    assert seal["attempts"] == 0
    terminal = next(row for row in events(directory) if row["kind"] == "terminal")
    assert terminal["partial_expansion"] and terminal["unscored_layer"]
    assert terminal["terminal"][4] == 1
    assert not (directory / "scores.bin").read_bytes()


def test_mid_layer_budget_exact_attempt_prefix_and_actual_expansion_counter(tmp_path):
    directory = tmp_path / "cell"
    with CellRecorder(directory, ledger(), {}) as recorder:
        result = beam_solve(
            task(outputs=("999",)),
            ScriptedScorer(),
            beam_width=8,
            max_depth=3,
            guard="off",
            observer=recorder,
            budget=35,
        )
        recorder.finish(result)
    assert result.termination == "budget"
    rows = events(directory)
    assert rows[-3]["kind"] == "started"
    started = [row for row in rows if row["kind"] == "started"]
    attempts = list(decode_attempts((directory / "attempts.bin").read_bytes()))
    assert len(started) >= 2
    for start in started:
        depth = start["depth"]
        depth_rows = [row for row in attempts if row.depth == depth]
        assert [(row.node, row.op) for row in depth_rows] == [
            (i // 28, i % 28) for i in range(len(depth_rows))
        ]
    assert rows[-2]["terminal"][4] == result.expansions
    assert rows[-2]["partial_expansion"]


def test_global_attempt_exhaustion_is_unsealed_and_has_no_normal_terminal(tmp_path):
    directory = tmp_path / "cell"
    with CellRecorder(directory, ledger(max_attempts=1), {}) as recorder:
        with pytest.raises(DiagnosticResourceError, match="global attempts"):
            beam_solve(task(), ScriptedScorer(), observer=recorder)
        assert recorder.terminal is None
    assert not (directory / "cell-seal.json").exists()
    assert len(list(decode_attempts((directory / "attempts.bin").read_bytes()))) == 1
    assert not any(row["kind"] == "terminal" for row in events(directory))


def test_invalid_executable_scores_invalidates_run_before_scored_event(tmp_path):
    class BadScorer(ScriptedScorer):
        def log_probs(self, blocks, legal):
            scores = super().log_probs(blocks, legal)
            scores[legal] = np.inf
            return scores

    directory = tmp_path / "cell"
    with (
        CellRecorder(directory, ledger(), {}) as recorder,
        pytest.raises(DiagnosticProvenanceError, match="executable cumulative"),
    ):
        beam_solve(task(outputs=("999",)), BadScorer(), observer=recorder)
    assert not (directory / "cell-seal.json").exists()
    assert not any(row["kind"] in ("scored", "terminal") for row in events(directory))


def test_terminal_result_mismatch_and_missing_last_callback_never_seal(tmp_path):
    directory = tmp_path / "cell"
    with CellRecorder(directory, ledger(), {}) as recorder:
        result = beam_solve(task(), ScriptedScorer(), max_depth=1, observer=recorder)
        with pytest.raises(DiagnosticProvenanceError, match="differs from terminal"):
            recorder.finish(replace(result, expansions=result.expansions + 1))
    assert not (directory / "cell-seal.json").exists()
    with (
        CellRecorder(tmp_path / "missing", ledger(), {}) as recorder,
        pytest.raises(DiagnosticProvenanceError, match="phase/order"),
    ):
        recorder.finish(result)


def test_duplicate_cell_directory_and_callback_order_rejected(tmp_path):
    directory = tmp_path / "cell"
    with (
        CellRecorder(directory, ledger(), {}) as recorder,
        pytest.raises(DiagnosticProvenanceError, match="phase/order"),
    ):
        recorder.layer_scored(0, np.zeros((1, 28)), np.ones((1, 28), dtype=bool))
    original = (directory / "events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        CellRecorder(directory, ledger(), {})
    assert (directory / "events.jsonl").read_bytes() == original
