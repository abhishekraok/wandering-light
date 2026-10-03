"""Synthetic train/side bytes only; never select the registered real payload."""

import gzip
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import capped_glue_loss as core
from experiments import capped_glue_stage_a as stage_a
from wandering_light.stack_machine import Int, Lst, Mark, Sym, parse_stack
from wandering_light.stack_pbe import Task
from wandering_light.stack_policy import validation_skeleton

PROGRAM = "dup dup [ dup ] i drop drop [ drop ] i"


def synthetic(
    behaviour="fixture", skeleton="original", *, inputs=("2",), outputs=("2",)
):
    return Task(
        behaviour=behaviour,
        summary="synthetic",
        split="train",
        task_hash_split="train",
        inputs=inputs,
        outputs=outputs,
        held_out_inputs=("(",),
        held_out_outputs=(")",),
        program=("dup", "drop"),
        length=2,
        examples=len(inputs),
        held_out=1,
        fuel=0,
        tags=("synthetic",),
        skeleton=skeleton,
    )


def population(size):
    tasks = []
    index = 0
    while len(tasks) < size:
        candidate = synthetic(f"behaviour-{index}", f"original-group-{index}")
        if not validation_skeleton(candidate):
            tasks.append(candidate)
        index += 1
    return tasks, {task.behaviour: PROGRAM for task in tasks}


def build(size=20):
    tasks, programs = population(size)
    return stage_a.build_stage_a(
        tasks, programs, {}, source_revision="synthetic-source"
    )


def test_selection_matches_independent_hash_reference_and_caps_without_refill():
    tasks, programs = population(70)
    panel, summary = stage_a.select_panel(tasks, programs)

    def digest(salt, name):
        return hashlib.sha256((salt + name).encode("utf-8")).hexdigest(), name

    expected = sorted(
        tasks, key=lambda task: digest("capped-glue-D1-group:", task.skeleton)
    )[:64]
    assert [item.task.behaviour for item in panel] == [
        task.behaviour for task in expected
    ]
    assert len(panel) == 64 and summary["eligible_groups"] == 70
    smaller, _ = stage_a.select_panel(tasks[:3], programs)
    assert len(smaller) == 3
    # Shuffling the pool changes file indices, but not hash-selected identities.
    shuffled, _ = stage_a.select_panel(list(reversed(tasks)), programs)
    assert [item.task.behaviour for item in shuffled] == [
        item.task.behaviour for item in panel
    ]


def test_behaviour_representative_uses_separate_salt_and_first_original_row():
    tasks, _ = population(1)
    original = tasks[0]
    alias = replace(original, behaviour="another-alias")
    later = replace(original, inputs=("3",), outputs=("3",))
    programs = {original.behaviour: PROGRAM, alias.behaviour: PROGRAM}
    panel, _ = stage_a.select_panel([original, later, alias], programs)
    winner = min(
        programs,
        key=lambda name: (
            hashlib.sha256(("capped-glue-D1-behaviour:" + name).encode()).hexdigest(),
            name,
        ),
    )
    assert len(panel) == 1 and panel[0].task.behaviour == winner
    if winner == original.behaviour:
        assert panel[0].original_index == 0 and panel[0].task.inputs == ("2",)
    else:
        assert panel[0].original_index == 2
    assert panel[0].original_program == ("dup", "drop")
    assert panel[0].task.length == 2 and panel[0].task.program == tuple(PROGRAM.split())
    assert panel[0].task.skeleton == original.skeleton


def test_original_validation_holdout_and_untreated_fallback_are_excluded():
    tasks, programs = population(2)
    held = next(
        synthetic("held", f"holdout-{i}")
        for i in range(100)
        if validation_skeleton(synthetic("held", f"holdout-{i}"))
    )
    untreated = replace(tasks[0], behaviour="untreated")
    pool = [*tasks, held, untreated]
    programs[held.behaviour] = PROGRAM
    panel, _ = stage_a.select_panel(pool, programs)
    assert {item.task.behaviour for item in panel} == {task.behaviour for task in tasks}


def test_group_fallback_and_inconsistent_original_skeletons():
    original = next(
        synthetic(f"fallback-{i}", "")
        for i in range(100)
        if not validation_skeleton(synthetic(f"fallback-{i}", ""))
    )
    panel, _ = stage_a.select_panel([original], {original.behaviour: PROGRAM})
    assert stage_a.selected_metadata(panel[0])["group"] == original.behaviour
    with pytest.raises(core.InvalidDiagnosticError, match="inconsistent ORIGINAL"):
        stage_a.select_panel(
            [original, replace(original, skeleton="different")],
            {original.behaviour: PROGRAM},
        )
    with pytest.raises(core.InvalidDiagnosticError, match="non-train"):
        stage_a.select_panel([replace(original, split="eval")], {})


def test_salt_hash_ties_break_lexically(monkeypatch):
    class EqualDigest:
        def hexdigest(self):
            return "00" * 32

    monkeypatch.setattr(stage_a.hashlib, "sha256", lambda value: EqualDigest())
    assert stage_a.hash_order("salt:", "aaa") < stage_a.hash_order("salt:", "bbb")


def test_hidden_fields_are_not_accessed_or_serialized():
    class Guarded(dict):
        def __getitem__(self, key):
            if key in stage_a.HIDDEN_FIELDS:
                pytest.fail("hidden example field accessed")
            return super().__getitem__(key)

    record = Guarded(asdict(synthetic()))
    loaded = stage_a.shown_task_from_record(record)
    assert loaded.held_out_inputs == () and loaded.held_out_outputs == ()
    assert loaded.inputs == ("2",) and loaded.held_out == 1
    selected = stage_a.Selected(0, loaded.program, loaded)
    meta = stage_a.selected_metadata(selected)
    assert not stage_a.HIDDEN_FIELDS & set(meta["original_metadata"])
    assert meta["hidden_pairs_retained"] is False


@pytest.mark.parametrize("compressed", [False, True])
def test_single_payload_loader_only_reads_synthetic_train_shown_pairs(
    tmp_path, compressed
):
    path = tmp_path / ("train.jsonl.gz" if compressed else "train.jsonl")
    payload = json.dumps(asdict(synthetic())) + "\n"
    if compressed:
        path.write_bytes(gzip.compress(payload.encode()))
    else:
        path.write_text(payload)
    (loaded,) = stage_a.read_train(path)
    assert loaded.inputs == ("2",) and loaded.held_out_inputs == ()
    record = asdict(synthetic())
    record["split"] = "eval"
    with pytest.raises(core.InvalidDiagnosticError, match="non-train"):
        stage_a.shown_task_from_record(record)


@pytest.mark.parametrize(
    "records",
    [
        [{"behaviour": "a", "program": "dup"}, {"behaviour": "a", "program": "drop"}],
        [{"behaviour": "a", "program": " "}],
        [{"program": "dup"}],
        [{"behaviour": "a", "program": ["dup"]}],
    ],
)
def test_side_duplicates_and_malformed_rows_are_invalid(records):
    with pytest.raises(core.InvalidDiagnosticError):
        stage_a.side_programs(records)


def test_duplicate_capped_text_is_reported_without_deduplicating_groups():
    tasks, programs = population(3)
    panel, _ = stage_a.select_panel(tasks, programs)
    (duplicates,) = stage_a.duplicate_programs(panel)
    assert duplicates["program"] == PROGRAM
    assert duplicates["behaviours"] == sorted(programs)
    assert len(panel) == 3


def test_support_failure_never_enumerates_alternatives(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported panel enumerated alternatives")

    monkeypatch.setattr(core, "legal_alternatives", forbidden)
    monkeypatch.setattr(stage_a, "encoder_diagnostics", forbidden)
    result = build(19)
    assert result["kind"] == "support_only" and result["complete"]
    assert result["support"]["cells"] == 38 and not result["support"]["passes"]
    assert result["models_loaded"] == result["logit_rows"] == 0
    assert all("legality" not in row for row in result["rows"])
    stage_a.validate_stage_a(result)


def test_all_tasks_replay_before_alternative_masks_and_errors_do_not_refill(
    monkeypatch,
):
    tasks, programs = population(20)
    panel, _ = stage_a.select_panel(tasks, programs)
    bad = panel[0].task.behaviour
    programs[bad] = "drop"  # syntactically valid, misses the original target

    def forbidden(*args, **kwargs):
        pytest.fail("legality started before all shown replays completed")

    monkeypatch.setattr(core, "legal_alternatives", forbidden)
    with pytest.raises(core.InvalidDiagnosticError, match="misses shown target"):
        stage_a.build_stage_a(tasks, programs, {}, source_revision="synthetic")
    assert bad in {
        item.task.behaviour for item in stage_a.select_panel(tasks, programs)[0]
    }


def test_supported_stage_contains_all_rows_cells_and_reason_masks():
    result = build()
    assert result["kind"] == "supported" and result["support"]["passes"]
    assert result["support"]["cells"] == 40 and len(result["support"]["groups"]) == 20
    assert result["counts"]["all_rows"] == 240
    assert result["counts"]["matched_rows"] == 80
    assert (
        result["counts"]["matched_top_rows"]
        == result["counts"]["matched_body_rows"]
        == 40
    )
    assert result["counts"]["omitted_content_rows"] == 40
    assert sum(row["count"] for row in result["counts"]["by_label"]) == 240
    assert all(len(row["legality"]["mask"]) == 28 for row in result["rows"])
    assert all(row["witness_executions"] == 1 for row in result["rows"])
    assert result["models_loaded"] == result["logit_rows"] == 0
    stage_a.validate_stage_a(result)


def test_serialized_states_round_trip_open_marks_and_typed_quotations(tmp_path):
    result = build(1)
    seal = stage_a.seal_stage_a(result, tmp_path / "stage-a.json")
    loaded = stage_a.read_sealed_stage_a(seal)
    states = [parse_stack(row["product"][0], basis="mark-28") for row in loaded["rows"]]
    assert states[3] == (Int(2), Int(2), Int(2), Mark())
    assert states[4] == (Int(2), Int(2), Int(2), Mark(), Sym("dup"))
    assert states[5] == (Int(2), Int(2), Int(2), Lst((Sym("dup"),)))
    assert states[6] == (Int(2), Int(2), Int(2), Int(2))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_row",
        "duplicate_row",
        "count",
        "mask",
        "model",
        "gate",
        "cell",
        "label",
        "target",
        "hidden",
        "op",
        "encoder",
    ],
)
def test_seal_validation_rejects_incomplete_or_inconsistent_stage(mutation):
    result = build()
    if mutation == "missing_row":
        result["rows"].pop()
    elif mutation == "duplicate_row":
        result["rows"][-1] = result["rows"][0]
    elif mutation == "count":
        result["counts"]["all_rows"] += 1
    elif mutation == "mask":
        result["rows"][0]["legality"]["mask"].pop()
    elif mutation == "model":
        result["models_loaded"] = 1
    elif mutation == "gate":
        result["support"]["passes"] = False
    elif mutation == "cell":
        result["cells"].pop()
    elif mutation == "label":
        result["rows"][0]["side"] = "body"
    elif mutation == "target":
        result["rows"][0]["target"] = ("3",)
    elif mutation == "hidden":
        result["panel"][0]["original_metadata"]["held_out_inputs"] = ("3",)
    elif mutation == "encoder":
        result["rows"][0]["encoder_diagnostics"]["examples"][0][
            "state_pre_cap_width"
        ] += 1
    else:
        result["rows"][0]["op"] = "1"
    with pytest.raises(core.InvalidDiagnosticError):
        stage_a.validate_stage_a(result)


def test_support_only_seal_rejects_any_legal_enumeration_artifact():
    result = build(1)
    result["rows"][0]["legality"] = {"mask": [True] * 28}
    with pytest.raises(core.InvalidDiagnosticError, match="unsupported"):
        stage_a.validate_stage_a(result)


def test_seal_is_deterministic_atomic_and_never_overwrites(tmp_path):
    result = build(1)
    path = tmp_path / "stage-a.json"
    seal = stage_a.seal_stage_a(result, path)
    assert seal.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert stage_a.canonical_bytes(
        stage_a.read_sealed_stage_a(seal)
    ) == stage_a.canonical_bytes(result)
    assert seal.rows == 12 and seal.kind == "support_only"
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        stage_a.seal_stage_a(result, path)
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]
    path.write_bytes(original + b" ")
    with pytest.raises(core.InvalidDiagnosticError, match="hash mismatch"):
        stage_a.read_sealed_stage_a(seal)


def test_invalid_stage_does_not_publish_a_seal(tmp_path):
    result = build(1)
    result["complete"] = False
    path = tmp_path / "stage-a.json"
    with pytest.raises(core.InvalidDiagnosticError, match="incomplete"):
        stage_a.seal_stage_a(result, path)
    assert not path.exists()


def toy_spec(tmp_path):
    spec = {"inputs": {}, "checkpoints": []}
    for role in ("train", "side"):
        path = tmp_path / f"{role}.jsonl"
        path.write_bytes(role.encode())
        spec["inputs"][role] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for encoding, seed in stage_a.CHECKPOINT_ORDER:
        path = tmp_path / f"{encoding}-{seed}"
        path.mkdir()
        hashes = {}
        for name in stage_a.CHECKPOINT_FILES:
            data = f"synthetic {encoding} {seed} {name}".encode()
            (path / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        spec["checkpoints"].append(
            {"encoding": encoding, "seed": seed, "path": path.name, "hashes": hashes}
        )
    return spec


def test_provenance_hashes_all_pinned_files_without_deserializing_weights(tmp_path):
    spec = toy_spec(tmp_path)
    checked = stage_a.verify_provenance(spec, tmp_path)
    assert checked["inputs"]["train"]["sha256"] == spec["inputs"]["train"]["sha256"]
    assert len(checked["checkpoints"]) == 6
    assert checked["checkpoints"][0]["hashes"] == spec["checkpoints"][0]["hashes"]
    # Absolute pins are resolved as absolute, not relocated beneath the repo.
    spec["inputs"]["train"]["path"] = str(tmp_path / "train.jsonl")
    stage_a.verify_provenance(spec, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ["train_hash", "weight_hash", "missing_pin", "order", "missing_file", "eval_role"],
)
def test_provenance_fails_closed(tmp_path, mutation):
    spec = toy_spec(tmp_path)
    if mutation == "train_hash":
        spec["inputs"]["train"]["sha256"] = "00" * 32
    elif mutation == "weight_hash":
        spec["checkpoints"][0]["hashes"]["model.safetensors"] = "00" * 32
    elif mutation == "missing_pin":
        del spec["checkpoints"][0]["hashes"]["policy_config.json"]
    elif mutation == "order":
        spec["checkpoints"].reverse()
    elif mutation == "missing_file":
        (tmp_path / "train.jsonl").unlink()
    else:
        spec["inputs"]["eval"] = deepcopy(spec["inputs"]["train"])
    with pytest.raises((core.InvalidDiagnosticError, FileNotFoundError)):
        stage_a.verify_provenance(spec, tmp_path)
