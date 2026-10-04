"""Only dummy corpus/checkpoint bytes; no real selection, weights or model."""

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments import score_gap_inputs as inputs
from experiments.score_gap_storage import (
    AttemptReceipts,
    DiagnosticProvenanceError,
    DiagnosticResourceError,
    PayloadBudget,
    canonical_json,
)
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import Bool, Int, Lst, Sym, parse_stack
from wandering_light.stack_pbe import BASIS


def budget(**kwargs):
    return PayloadBudget(deadline=10.0, clock=lambda: 0.0, **kwargs)


def reserved(tmp_path):
    receipts = AttemptReceipts(tmp_path / "attempt")
    receipts.reserve({"issue": 180, "test": "synthetic-only"})
    return receipts


def fixture(tmp_path, *, size=60, compressed=True):
    root = tmp_path / "dummy-inputs"
    root.mkdir()
    names = [
        f"manual-group-{index}"
        for index in range(1000)
        if int(hashlib.sha256(f"manual-group-{index}".encode()).hexdigest(), 16) % 10
        != 0
    ][:size]
    rows = [
        {
            "behaviour": name,
            "skeleton": name,
            "split": "train",
            "inputs": ["2 3", "true ( 1 false dup )"],
            "outputs": ["6", "false ( 7 true swap )"],
            "program": "POISON-NOT-A-WITNESS",
            "held_out_inputs": {"POISON-HIDDEN": [9001]},
            "held_out_outputs": {"POISON-HIDDEN": [9002]},
            "length": "POISON-LENGTH",
        }
        for name in names
    ]
    train = root / ("train.jsonl.gz" if compressed else "train.jsonl")
    payload = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    if compressed:
        train.write_bytes(gzip.compress(payload))
    else:
        train.write_bytes(payload)
    side = root / "side.jsonl"
    side.write_bytes(
        b"".join(
            canonical_json({"behaviour": name, "program": {"POISON-SIDE": 1}})
            for name in names
        )
    )
    spec = {
        "inputs": {
            role: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for role, path in (("train", train), ("side", side))
        },
        "checkpoints": [],
    }
    for seed in range(3):
        directory = root / f"dummy-s{seed}"
        directory.mkdir()
        hashes = {}
        for name in sorted(inputs.CHECKPOINT_FILES):
            payload = f"synthetic-not-model:{seed}:{name}".encode()
            (directory / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        spec["checkpoints"].append(
            {
                "path": str(directory),
                "seed": seed,
                "encoding": "difference",
                "hashes": hashes,
            }
        )
    return spec, names, rows


def stage(tmp_path, **kwargs):
    spec, names, rows = fixture(tmp_path, **kwargs)
    receipts, ledger = reserved(tmp_path), budget()
    seal = inputs.seal_inputs(
        spec, tmp_path, receipts, ledger, receipts.root / "inputs"
    )
    return seal, spec, names, rows, receipts, ledger


@pytest.mark.parametrize("compressed", [False, True])
def test_dummy_files_pins_parity_original_shown_and_full_typed_seal(
    tmp_path, compressed
):
    seal, spec, names, rows, receipts, ledger = stage(tmp_path, compressed=compressed)
    assert seal.kind == "selected"
    manifest, panel = inputs.load_inputs(seal, ledger)
    expected = sorted(
        names,
        key=lambda g: (
            hashlib.sha256(("score-gap-D1-group:" + g).encode()).hexdigest(),
            g,
        ),
    )[:48]
    assert [item.group for item in panel] == expected
    assert [item.half for item in panel] == ["selection", "confirmation"] * 24
    assert [item.source_index for item in panel] == [names.index(g) for g in expected]
    assert manifest["pins"] == spec
    assert manifest["population"] == {
        "train_rows": 60,
        "train_behaviours": 60,
        "original_groups": 60,
        "side_rows": 60,
        "side_behaviours": 60,
    }
    assert set(manifest["checked_files"]) == {
        spec["inputs"]["train"]["path"],
        spec["inputs"]["side"]["path"],
        *(
            str(Path(row["path"]) / name)
            for row in spec["checkpoints"]
            for name in inputs.CHECKPOINT_FILES
        ),
    }
    assert all(
        item.solver.inputs == tuple(rows[0]["inputs"])
        and item.solver.outputs == tuple(rows[0]["outputs"])
        for item in panel
    )
    assert set(vars(panel[0].solver)) == {"inputs", "outputs"}
    assert not any(
        word in (seal.directory / "inputs.json").read_text()
        for word in ("POISON", "held_out", '"program"', '"length"')
    )
    assert panel[0].product[1] == (Bool(True), Lst((Int(1), Bool(False), Sym("dup"))))
    assert panel[0].target[1] == (Bool(False), Lst((Int(7), Bool(True), Sym("swap"))))
    assert ledger.bytes_written == sum(
        path.stat().st_size for path in seal.directory.iterdir()
    )
    assert ledger.attempts_written == 0
    assert receipts.reserved and not receipts.selected and not receipts.finished
    assert (
        len(json.loads((seal.directory / "input-seal.json").read_text())["files"]) == 2
    )


class PoisonSide(dict):
    def __getitem__(self, key):
        if key != "behaviour":
            raise AssertionError("side witness accessed")
        return super().__getitem__(key)


def test_side_whitelist_ignores_program_and_duplicate_names():
    assert inputs.side_eligibility(
        [
            PoisonSide(behaviour="a", program=object()),
            PoisonSide(behaviour="a", program=object()),
        ]
    ) == (frozenset({"a"}), 2)


def test_no_input_access_without_real_reservation(tmp_path, monkeypatch):
    receipts = AttemptReceipts(tmp_path)
    monkeypatch.setattr(
        inputs,
        "validate_spec",
        lambda *args: pytest.fail("pins inspected before reservation"),
    )
    with pytest.raises(DiagnosticProvenanceError, match="attempt not active"):
        inputs.seal_inputs({}, tmp_path, receipts, budget(), tmp_path / "out")
    receipts.reserved = True
    with pytest.raises(FileNotFoundError):
        inputs.seal_inputs({}, tmp_path, receipts, budget(), tmp_path / "out")


@pytest.mark.parametrize(
    "mutation",
    ["encoding", "seed_bool", "order", "missing_file", "uppercase_hash", "alias"],
)
def test_entire_bad_spec_rejected_before_opening_pins(tmp_path, monkeypatch, mutation):
    spec, _, _ = fixture(tmp_path)
    if mutation == "encoding":
        spec["checkpoints"][2]["encoding"] = "machine"
    elif mutation == "seed_bool":
        spec["checkpoints"][1]["seed"] = True
    elif mutation == "order":
        spec["checkpoints"].reverse()
    elif mutation == "missing_file":
        spec["checkpoints"][2]["hashes"].pop("model.safetensors")
    elif mutation == "uppercase_hash":
        spec["inputs"]["train"]["sha256"] = "A" * 64
    else:
        spec["inputs"]["side"] = dict(spec["inputs"]["train"])
    original = inputs.hash_file

    def only_attempt(path, ledger):
        assert path.name == "attempt.json", "pinned file opened with invalid spec"
        return original(path, ledger)

    monkeypatch.setattr(inputs, "hash_file", only_attempt)
    with pytest.raises(DiagnosticProvenanceError):
        inputs.seal_inputs(
            spec, tmp_path, reserved(tmp_path), budget(), tmp_path / "out"
        )


@pytest.mark.parametrize("role", ["train", "side", "model"])
def test_changed_byte_pin_rejected_before_payload_projection(
    tmp_path, monkeypatch, role
):
    spec, _, _ = fixture(tmp_path)
    if role == "model":
        path = Path(spec["checkpoints"][2]["path"]) / "model.safetensors"
    else:
        path = Path(spec["inputs"][role]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(
        inputs,
        "_records",
        lambda *args: pytest.fail("payload opened before all pins passed"),
    )
    with pytest.raises(DiagnosticProvenanceError, match="byte pin mismatch"):
        inputs.seal_inputs(
            spec, tmp_path, reserved(tmp_path), budget(), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_streaming_input_change_is_invalid_even_on_insufficient_selection(
    tmp_path, monkeypatch
):
    spec, _, _ = fixture(tmp_path, size=2, compressed=False)
    original = inputs._records

    def change_after_read(path, ledger):
        yield from original(path, ledger)
        if path.name == "train.jsonl":
            path.write_bytes(path.read_bytes() + b"\n")

    monkeypatch.setattr(inputs, "_records", change_after_read)
    with pytest.raises(DiagnosticProvenanceError, match="byte pin mismatch"):
        inputs.seal_inputs(
            spec, tmp_path, reserved(tmp_path), budget(), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_insufficient_selection_seals_only_population_and_no_models(tmp_path):
    seal, _, _, _, receipts, ledger = stage(tmp_path, size=2)
    assert seal.kind == "INSUFFICIENT_SELECTION"
    manifest, selected = inputs.load_inputs(seal, ledger)
    assert manifest["selected"] == [] and selected == ()
    assert manifest["population"]["train_rows"] == 2
    assert (seal.directory / "shown.bin").read_bytes() == b""
    assert not receipts.selected


def test_global_resource_stop_has_no_input_seal_or_false_cell_terminal(tmp_path):
    spec, _, _ = fixture(tmp_path)
    receipts = reserved(tmp_path)
    with pytest.raises(DiagnosticResourceError, match="trace bytes"):
        inputs.seal_inputs(
            spec, tmp_path, receipts, budget(max_bytes=1), tmp_path / "out"
        )
    assert not (tmp_path / "out" / "input-seal.json").exists()
    assert not (receipts.root / "terminal.json").exists()


@pytest.mark.parametrize("name", ["inputs.json", "shown.bin"])
def test_changed_stream_rejected_before_any_typed_decode(tmp_path, monkeypatch, name):
    seal, _, _, _, _, ledger = stage(tmp_path)
    path = seal.directory / name
    path.write_bytes(path.read_bytes() + b"x")
    monkeypatch.setattr(
        inputs, "decode_beam", lambda *args: pytest.fail("decode before hashes")
    )
    with pytest.raises(DiagnosticProvenanceError, match="stream hash mismatch"):
        inputs.load_inputs(seal, ledger)


def reseal(seal, mutation):
    path = seal.directory / "inputs.json"
    value = json.loads(path.read_bytes())
    mutation(value)
    path.write_bytes(canonical_json(value))
    control_path = seal.directory / "input-seal.json"
    control = json.loads(control_path.read_bytes())
    control["files"]["inputs.json"] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    payload = canonical_json(control)
    control_path.write_bytes(payload)
    return replace(seal, sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize(
    "mutation", ["half", "shown", "hidden", "position_bool", "offset", "order"]
)
def test_resealed_inconsistent_projection_or_framing_rejected(tmp_path, mutation):
    seal, _, _, _, _, ledger = stage(tmp_path)

    def alter(manifest):
        row = manifest["selected"][0]
        if mutation == "half":
            row["half"] = "confirmation"
        elif mutation == "shown":
            row["shown"]["outputs"][0] = "7"
        elif mutation == "hidden":
            row["shown"]["held_out_inputs"] = ["99"]
        elif mutation == "position_bool":
            row["position"] = False
        elif mutation == "offset":
            row["typed"]["offset"] = True
        else:
            manifest["selected"].reverse()

    seal = reseal(seal, alter)
    with pytest.raises(DiagnosticProvenanceError):
        inputs.load_inputs(seal, ledger)


def test_minimal_solver_object_matches_imported_production_first_fit():
    class NeverCalledScorer:
        def log_probs(self, blocks, legal):
            pytest.fail("one-op first fit should precede scoring")

    result = beam_solve(
        inputs.SolverShown(("2 3",), ("6",)),
        NeverCalledScorer(),
        beam_width=64,
        max_depth=48,
        guard="both",
        max_state_atoms=16384,
        budget=200000,
    )
    assert result.program == ("mul",) and result.termination == "fit"
    assert parse_stack("6", basis=BASIS) == (Int(6),)


def test_first_original_representative_pairs_survive_duplicate_side_and_source(
    tmp_path,
):
    spec, names, rows = fixture(tmp_path, compressed=False)
    group = min(
        names,
        key=lambda g: hashlib.sha256(("score-gap-D1-group:" + g).encode()).hexdigest(),
    )
    original_hash = hashlib.sha256(
        ("score-gap-D1-behaviour:" + group).encode()
    ).hexdigest()
    behaviour = next(
        f"manual-candidate-{index}"
        for index in range(1000)
        if hashlib.sha256(
            ("score-gap-D1-behaviour:" + f"manual-candidate-{index}").encode()
        ).hexdigest()
        < original_hash
    )
    first = {
        **rows[0],
        "behaviour": behaviour,
        "skeleton": group,
        "inputs": ["4 2"],
        "outputs": ["8"],
    }
    last = {**first, "inputs": ["5 2"], "outputs": ["10"]}
    train, side = (Path(spec["inputs"][role]["path"]) for role in ("train", "side"))
    train.write_bytes(train.read_bytes() + canonical_json(first) + canonical_json(last))
    side.write_bytes(side.read_bytes() + canonical_json({"behaviour": behaviour}) * 2)
    for role, path in (("train", train), ("side", side)):
        spec["inputs"][role]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipts, ledger = reserved(tmp_path), budget()
    seal = inputs.seal_inputs(
        spec, tmp_path, receipts, ledger, receipts.root / "inputs"
    )
    manifest, selected = inputs.load_inputs(seal, ledger)
    assert selected[0].behaviour == behaviour
    assert selected[0].source_index == 60
    assert selected[0].solver == inputs.SolverShown(("4 2",), ("8",))
    assert selected[0].product == ((Int(4), Int(2)),)
    assert selected[0].target == ((Int(8),),)
    assert manifest["population"]["side_rows"] == 62
    assert manifest["population"]["side_behaviours"] == 61


def test_existing_input_stage_refuses_before_any_second_pinned_read(
    tmp_path, monkeypatch
):
    seal, spec, _, _, receipts, ledger = stage(tmp_path)
    monkeypatch.setattr(
        inputs,
        "hash_file",
        lambda *args: pytest.fail("reopened pins before duplicate-stage rejection"),
    )
    with pytest.raises(FileExistsError):
        inputs.seal_inputs(spec, tmp_path, receipts, ledger, seal.directory)


@pytest.mark.parametrize("field", ["population", "checked_files", "attempt"])
def test_resealed_provenance_type_and_population_inconsistency_rejected(
    tmp_path, field
):
    seal, _, _, _, _, ledger = stage(tmp_path)

    def alter(manifest):
        if field == "population":
            manifest[field]["train_rows"] = True
        elif field == "checked_files":
            manifest[field].pop(next(iter(manifest[field])))
        else:
            manifest[field]["bytes"] = False

    with pytest.raises(DiagnosticProvenanceError):
        inputs.load_inputs(reseal(seal, alter), ledger)
