"""Synthetic sealed-file bridge units; full independent verification still owed."""

import hashlib
import json

import pytest

from experiments.score_gap_read import load_cell
from experiments.score_gap_recorder import CellRecorder
from experiments.score_gap_storage import (
    DiagnosticProvenanceError,
    PayloadBudget,
    canonical_json,
)
from tests.test_score_gap_lineage import ScriptedScorer, task
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS

IDENTITY = {"group": "synthetic-group", "half": "selection", "seed": 0}


def budget():
    return PayloadBudget(deadline=10.0, clock=lambda: 0.0)


def recorded(tmp_path, *, max_depth=3, output="12", beam_width=8, **kwargs):
    manual = task(outputs=(output,))
    directory = tmp_path / "cell"
    with CellRecorder(directory, budget(), IDENTITY) as recorder:
        result = beam_solve(
            manual,
            ScriptedScorer(),
            beam_width=beam_width,
            max_depth=max_depth,
            guard="off",
            observer=recorder,
            **kwargs,
        )
        recorder.finish(result)
    inputs = tuple(parse_stack(text, basis=BASIS) for text in manual.inputs)
    targets = tuple(parse_stack(text, basis=BASIS) for text in manual.outputs)
    return directory, inputs, targets, result


def mutate_events(directory, mutation):
    path = directory / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    payload = b"".join(canonical_json(row) for row in rows)
    path.write_bytes(payload)
    seal = json.loads((directory / "cell-seal.json").read_text())
    seal["files"]["events.jsonl"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (directory / "cell-seal.json").write_bytes(canonical_json(seal))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_depth": 1},
        {"output": "6"},
        {"budget": 0},
        {"deadline": 0.0},
        {"budget": 35, "output": "999"},
    ],
)
def test_actual_production_sealed_read_lineage_and_partial_stops(tmp_path, options):
    directory, inputs, targets, result = recorded(tmp_path, **options)
    loaded = load_cell(directory, budget(), IDENTITY, inputs, targets)
    assert loaded.result == result
    assert loaded.cell.termination == result.termination
    assert (
        sum(loaded.refusal_counts.values())
        == (directory / "attempts.bin").stat().st_size // 19
    )
    if result.program == ("mul", "dup", "add"):
        assert [layer.ancestor_rank for layer in loaded.cell.layers] == [1, 1]
    if result.termination == "max_depth":
        assert len(loaded.cell.layers) == result.depth
    if result.executions == 0 or result.program == ("mul",):
        assert loaded.cell.layers == ()


@pytest.mark.parametrize(
    "name", ["events.jsonl", "attempts.bin", "beams.bin", "scores.bin"]
)
def test_stream_hashes_checked_before_array_decode(tmp_path, name, monkeypatch):
    directory, inputs, targets, _ = recorded(tmp_path)
    with (directory / name).open("ab") as stream:
        stream.write(b"corrupt")

    def forbidden_decode(_):
        raise AssertionError("decoded before verifying every stream")

    monkeypatch.setattr("experiments.score_gap_read.decode_array", forbidden_decode)
    with pytest.raises(DiagnosticProvenanceError, match="size mismatch"):
        load_cell(directory, budget(), IDENTITY, inputs, targets)


@pytest.mark.parametrize(
    "change",
    [
        "drop_last_retention",
        "offset",
        "identity_bool",
        "terminal_bool",
        "refusal_counts",
        "extra_callback",
        "shared_source",
    ],
)
def test_resealed_structural_forgery_rejected(tmp_path, change):
    directory, inputs, targets, _ = recorded(tmp_path)

    def mutate(rows):
        starts = [row for row in rows if row["kind"] == "started"]
        if change == "drop_last_retention":
            index = max(i for i, row in enumerate(rows) if row["kind"] == "retained")
            rows.pop(index)
        elif change == "offset":
            starts[-1]["attempt_offset"] += 19
        elif change == "identity_bool":
            rows[0]["identity"]["seed"] = False
        elif change == "terminal_bool":
            next(row for row in rows if row["kind"] == "terminal")["terminal"][2] = True
        elif change == "refusal_counts":
            next(row for row in rows if row["kind"] == "terminal")["refusal_counts"][
                "fuel"
            ] += 1
        elif change == "shared_source":
            starts[-1]["state"]["offset"] = 0
        else:
            rows.append(rows[-1])

    mutate_events(directory, mutate)
    with pytest.raises((DiagnosticProvenanceError, ValueError)):
        load_cell(directory, budget(), IDENTITY, inputs, targets)


def test_pinned_typed_input_and_returned_targets_replayed(tmp_path):
    directory, inputs, targets, _ = recorded(tmp_path)
    with pytest.raises(DiagnosticProvenanceError, match="typed root"):
        load_cell(directory, budget(), IDENTITY, targets, targets)
    with pytest.raises(ValueError, match="targets"):
        load_cell(directory, budget(), IDENTITY, inputs, inputs)


def test_unsuccessful_seal_rejected_before_any_payload_decode(tmp_path, monkeypatch):
    directory, inputs, targets, _ = recorded(tmp_path)
    path = directory / "cell-seal.json"
    seal = json.loads(path.read_text())
    seal["complete"] = False
    path.write_bytes(canonical_json(seal))

    def forbidden(_):
        raise AssertionError("decoded failed input")

    monkeypatch.setattr("experiments.score_gap_read.decode_beam", forbidden)
    with pytest.raises(DiagnosticProvenanceError, match="incomplete cell"):
        load_cell(directory, budget(), IDENTITY, inputs, targets)


def test_width64_actual_positive_bridge_into_no_headroom_phase_control(tmp_path):
    from dataclasses import replace

    from experiments.score_gap_analysis import analyze
    from experiments.score_gap_storage import AttemptReceipts

    directory, inputs, targets, _ = recorded(tmp_path, beam_width=64)
    loaded = load_cell(directory, budget(), IDENTITY, inputs, targets)
    assert loaded.cell.informative
    # Explicit synthetic repetition of one manual trace; this is not a real
    # metadata-selected panel, trained scorer, or a generalization measurement.
    groups = tuple(f"replicated-manual-{i}" for i in range(48))
    cells = tuple(
        replace(
            loaded.cell,
            group=group,
            half="selection" if i % 2 == 0 else "confirmation",
            seed=seed,
        )
        for i, group in enumerate(groups)
        for seed in range(3)
    )
    receipts = AttemptReceipts(tmp_path / "synthetic-analysis")
    receipts.reserve({"synthetic": True})
    report = analyze(cells, groups, receipts, budget(), tmp_path / "analysis.jsonl")
    assert report["terminal"] == "NO_HEADROOM_OVER_PREFIX32"
    assert report["tau"] is None


@pytest.mark.parametrize("kind", ["fuel", "depth"])
def test_recorded_refusal_taxonomy_from_production_step_all_at_default_limits(
    tmp_path, kind
):
    from wandering_light.stack_beam import BeamResult, step_all_explained
    from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, Int, Lst, Sym
    from wandering_light.stack_pbe import OPS

    body = Lst((Sym("swap"),) * DEFAULT_FUEL_LIMIT)
    if kind == "depth":
        body = Lst()
        for _ in range(40):
            body = Lst((body, Sym("i")))
    inputs = ((Int(1), Int(2), Lst()), (Int(1), Int(2), body))
    directory = tmp_path / "direct-primitive-callbacks"
    # Synthetic direct observer callbacks from the production low-level helper;
    # this is a taxonomy/charge test, not a whole-search compatibility claim.
    with CellRecorder(directory, budget(), IDENTITY) as recorder:
        recorder.layer_started(0, ((inputs, (), 0.0),))
        for index, op in enumerate(OPS):
            moved, fuel, stepped, reason = step_all_explained(inputs, op)
            recorder.expansion_attempted(0, 0, index, moved, reason, fuel, stepped)
            if op.name == "i":
                assert moved is None and stepped == 1 and fuel == 1
                break
        result = BeamResult(
            None, "budget", 0.0, recorder.executions, recorder.fuel, 1, 1
        )
        recorder.search_terminated("budget", 1, result.executions, result.fuel, 1)
        recorder.finish(result)
    loaded = load_cell(directory, budget(), IDENTITY, inputs, ((), ()))
    assert loaded.refusal_counts[kind] >= 1


def test_resealed_nonobject_event_is_provenance_failure(tmp_path):
    directory, inputs, targets, _ = recorded(tmp_path)
    mutate_events(directory, lambda rows: rows.__setitem__(0, []))
    with pytest.raises(DiagnosticProvenanceError):
        load_cell(directory, budget(), IDENTITY, inputs, targets)
