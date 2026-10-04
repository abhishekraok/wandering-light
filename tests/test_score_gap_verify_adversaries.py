"""Resealed synthetic adversaries for the independent verifier's acceptance.

These deliberately pass the STRUCTURAL reader, whose scope excludes raw charge
and full search proof. They are not results or successful independent checks.
"""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from experiments.score_gap_inputs import InputSeal, load_inputs
from experiments.score_gap_read import load_cell
from experiments.score_gap_recorder import CellRecorder
from experiments.score_gap_storage import PayloadBudget, canonical_json
from tests.test_score_gap_inputs import stage
from tests.test_score_gap_lineage import ScriptedScorer, task
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, parse_stack, step_raw
from wandering_light.stack_pbe import BASIS, OPS

ATTEMPT = struct.Struct("<BBBQQ")
IDENTITY = {"group": "manual-charge", "half": "selection", "seed": 0}


def budget():
    return PayloadBudget(deadline=10.0, clock=lambda: 0.0)


def cell_fixture(directory):
    manual = task(inputs=("2 3", "3 4"), outputs=("6", "12"))
    with CellRecorder(directory, budget(), IDENTITY) as observer:
        result = beam_solve(
            manual,
            ScriptedScorer(),
            beam_width=64,
            max_depth=48,
            guard="both",
            max_state_atoms=16384,
            observer=observer,
        )
        observer.finish(result)
    assert result.program == ("mul",)
    initial = tuple(parse_stack(text, basis=BASIS) for text in manual.inputs)
    target = tuple(parse_stack(text, basis=BASIS) for text in manual.outputs)
    return initial, target


def raw_root_summary(directory, initial):
    # Literal low-level root-only oracle. No diagnostic decoder or reason/charge
    # helper participates in this independent expected summary.
    raw = (directory / "attempts.bin").read_bytes()
    fuel = executions = 0
    histogram = dict.fromkeys(("success", "ordinary", "fuel", "depth"), 0)
    for depth, node, op, saved_fuel, saved_stepped in ATTEMPT.iter_unpack(raw):
        assert depth == node == 0
        spent = 0
        reason = None
        stepped = len(initial)
        for example_index, stack in enumerate(initial):
            outcome = step_raw(OPS[op], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                reason = outcome
                stepped = example_index
                break
            spent += outcome[1]
        assert (saved_fuel, saved_stepped) == (spent, stepped)
        fuel += spent
        executions += stepped
        category = (
            "success"
            if reason is None
            else (
                "fuel"
                if reason == "out of fuel" or reason.endswith(": out of fuel")
                else "depth"
                if reason == "quotation nested too deep"
                or reason.endswith(": quotation nested too deep")
                else "ordinary"
            )
        )
        histogram[category] += 1
    return {
        "fuel": fuel,
        "executions": executions,
        "expansions": 1,
        "refusal_counts": histogram,
    }


def reseal_cell(directory, mutation):
    rows = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_bytes().splitlines()
    ]
    terminal = next(row for row in rows if row["kind"] == "terminal")
    result = next(row for row in rows if row["kind"] == "result")["result"]
    seal = json.loads((directory / "cell-seal.json").read_bytes())
    if mutation == "fuel":
        raw = bytearray((directory / "attempts.bin").read_bytes())
        depth, node, op, fuel, stepped = ATTEMPT.unpack_from(raw)
        ATTEMPT.pack_into(raw, 0, depth, node, op, fuel + 1, stepped)
        (directory / "attempts.bin").write_bytes(raw)
        terminal["terminal"][3] += 1
        result["fuel"] += 1
    elif mutation == "expansions":
        terminal["terminal"][4] += 1
        result["expansions"] += 1
    elif mutation == "refusal_counts":
        assert terminal["refusal_counts"]["ordinary"] > 0
        terminal["refusal_counts"]["ordinary"] -= 1
        terminal["refusal_counts"]["fuel"] += 1
    else:
        raise AssertionError(mutation)
    (directory / "events.jsonl").write_bytes(
        b"".join(canonical_json(row) for row in rows)
    )
    seal["result"] = result
    for name in seal["files"]:
        payload = (directory / name).read_bytes()
        seal["files"][name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (directory / "cell-seal.json").write_bytes(canonical_json(seal))


@pytest.mark.parametrize("mutation", ["fuel", "expansions", "refusal_counts"])
def test_resealed_semantic_adversary_needs_full_low_level_proof(tmp_path, mutation):
    directory = tmp_path / "fresh-synthetic-cell"
    initial, target = cell_fixture(directory)
    expected = raw_root_summary(directory, initial)
    original = load_cell(directory, budget(), IDENTITY, initial, target)
    assert original.result.fuel == expected["fuel"]
    assert original.result.expansions == expected["expansions"]
    assert original.refusal_counts == expected["refusal_counts"]
    reseal_cell(directory, mutation)
    # Passing structural checks after every hash/counter copy is updated is
    # intentional here. The complete independent verifier MUST reject all three.
    loaded = load_cell(directory, budget(), IDENTITY, initial, target)
    actual = (
        loaded.refusal_counts
        if mutation == "refusal_counts"
        else getattr(loaded.result, mutation)
    )
    assert actual != expected[mutation]


def reseal_input(seal, manifest):
    path = seal.directory / "inputs.json"
    path.write_bytes(canonical_json(manifest))
    control_path = seal.directory / "input-seal.json"
    control = json.loads(control_path.read_bytes())
    payload = path.read_bytes()
    control["files"]["inputs.json"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    control_path.write_bytes(canonical_json(control))
    return InputSeal(
        seal.directory, hashlib.sha256(control_path.read_bytes()).hexdigest(), seal.kind
    )


@pytest.mark.parametrize("mutation", ["train_rows", "source_index"])
def test_resealed_input_claim_needs_independent_original_population(tmp_path, mutation):
    seal, spec, _, _, _, _ = stage(tmp_path, compressed=False)
    original = [
        json.loads(line)
        for line in Path(spec["inputs"]["train"]["path"]).read_bytes().splitlines()
    ]
    manifest, _ = load_inputs(seal, budget())
    assert len(original) == manifest["population"]["train_rows"] == 60
    if mutation == "train_rows":
        manifest["population"]["train_rows"] += 1
    else:
        occupied = {row["shown"]["source_index"] for row in manifest["selected"]}
        unused = next(index for index in range(len(original)) if index not in occupied)
        manifest["selected"][0]["shown"]["source_index"] = unused
    changed = reseal_input(seal, manifest)
    reread, _ = load_inputs(changed, budget())
    if mutation == "train_rows":
        assert reread["population"]["train_rows"] != len(original)
    else:
        row = reread["selected"][0]
        assert (
            row["shown"]["behaviour"]
            != original[row["shown"]["source_index"]]["behaviour"]
        )
