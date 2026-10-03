"""Independent verifier challenged with synthetic producer artifacts only."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import capped_glue_stage_a as producer
from wandering_light.stack_machine import Int, Lst, parse_stack
from wandering_light.stack_pbe import Task
from wandering_light.stack_policy import validation_skeleton

VERIFY_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports/capped-glue-loss-20261003/verify_stage.py"
)
verifier = SimpleNamespace(**runpy.run_path(str(VERIFY_PATH)))
PROGRAM = "dup dup [ [ dup ] i ] i drop drop [ drop ] i"


def population(size):
    tasks = []
    for index in range(200):
        task = Task(
            behaviour=f"independent{index}",
            summary="synthetic",
            split="train",
            task_hash_split="train",
            inputs=("2", "7"),
            outputs=("2", "7"),
            held_out_inputs=("invalid poison hidden pair",),
            held_out_outputs=("invalid poison hidden pair",),
            program=("dup", "drop"),
            length=2,
            examples=2,
            held_out=1,
            fuel=0,
            tags=(),
            skeleton=f"original-independent-{index}",
        )
        if not validation_skeleton(task):
            tasks.append(task)
        if len(tasks) == size:
            break
    return tasks, {task.behaviour: PROGRAM for task in tasks}


@pytest.mark.parametrize("size", [0, 19, 20])
def test_separately_reconstructed_stage_matches_synthetic_producer(size):
    tasks, programs = population(size) if size else ([], {})
    stage = producer.build_stage_a(tasks, programs, {}, source_revision="synthetic")
    result = verifier.verify_stage(stage, tasks, programs)
    assert result["support_passes"] == (size >= 20)
    assert result["rows"] == size * len(PROGRAM.split())
    assert result["cells"] == 2 * size
    assert result["independent_shown_replay"]


def test_support_failure_does_no_alternative_or_encoder_work(monkeypatch):
    tasks, programs = population(19)
    stage = producer.build_stage_a(tasks, programs, {}, source_revision="synthetic")

    def forbidden(*args):
        raise AssertionError("unsupported verifier performed downstream work")

    monkeypatch.setitem(verifier.verify_stage.__globals__, "legality", forbidden)
    monkeypatch.setitem(verifier.verify_stage.__globals__, "exposure", forbidden)
    assert not verifier.verify_stage(stage, tasks, programs)["support_passes"]


@pytest.mark.parametrize(
    "tamper",
    [
        "selection",
        "ownership",
        "product",
        "cell",
        "mask",
        "width",
        "order",
        "counts",
        "refusals",
        "original",
        "limits",
    ],
)
def test_detects_corrupted_artifacts_even_when_stage_self_labels_are_consistent(tamper):
    tasks, programs = population(20)
    stage = json.loads(
        json.dumps(
            producer.build_stage_a(tasks, programs, {}, source_revision="synthetic")
        )
    )
    if tamper == "selection":
        stage["panel"][0]["original_index"] += 1
    elif tamper == "ownership":
        row = next(r for r in stage["rows"] if r["depth"] == 2)
        row["depth"] = 1
    elif tamper == "product":
        stage["rows"][0]["product"][1] = "8"
    elif tamper == "cell":
        stage["cells"][0]["top"].append(stage["cells"][0]["top"][0])
    elif tamper == "mask":
        stage["rows"][0]["legality"]["mask"][0] = not stage["rows"][0]["legality"][
            "mask"
        ][0]
    elif tamper == "width":
        stage["rows"][0]["encoder_diagnostics"]["examples"][1][
            "state_pre_cap_width"
        ] += 1
    elif tamper == "order":
        stage["rows"][:2] = reversed(stage["rows"][:2])
    elif tamper == "counts":
        stage["counts"]["matched_rows"] += 1
    elif tamper == "refusals":
        stage["alternative_first_refusal_counts"]["ordinary"] += 1
    elif tamper == "original":
        stage["panel"][0]["original_program"] = ["drop"]
    else:
        stage["limits"]["call_fuel"] += 1
    with pytest.raises(ValueError):
        verifier.verify_stage(stage, tasks, programs)


def test_deadline_rejects_before_any_replay():
    tasks, programs = population(19)
    stage = producer.build_stage_a(tasks, programs, {}, source_revision="synthetic")
    with pytest.raises(ValueError, match="deadline"):
        verifier.verify_stage(stage, tasks, programs, deadline=0)


def test_refusal_prefixes_preserve_first_example_and_resource_category(monkeypatch):
    original = verifier.move.__globals__["step_raw"]

    def controlled(op, stack, budget, depth):
        if op.name in {"map", "fold"}:
            if stack == (Int(2),):
                return stack, 1
            return (
                "i: out of fuel"
                if op.name == "map"
                else "dip: quotation nested too deep"
            )
        return original(op, stack, budget, depth)

    monkeypatch.setitem(verifier.move.__globals__, "step_raw", controlled)
    result = verifier.legality((parse_stack("2"), parse_stack("7")), "dup", None)
    failures = {row["op"]: row for row in result["refusals"]}
    assert failures["map"] == {
        "op": "map",
        "reason": "example 1: i: out of fuel",
        "class": "fuel",
    }
    assert failures["fold"]["class"] == "depth"
    assert failures["fold"]["reason"].startswith("example 1: dip:")
    assert result["resource_affected"]


def test_typed_widths_are_measured_before_caps():
    state = (Lst(tuple(Int(i) for i in range(200))),)
    target = (Lst(tuple(Int(i) for i in range(190))),)
    result = verifier.exposure((state,), (target,))
    row = result["examples"][0]
    assert row["state_pre_cap_width"] > 160
    assert row["target_pre_cap_width"] > 64
    assert row["state_retained_width"] == 160
    assert row["target_retained_width"] == 64
    assert row["difference_flags"] == 1
    assert result["any_state_truncated"] and result["any_target_truncated"]
