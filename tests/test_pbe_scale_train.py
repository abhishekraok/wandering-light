"""Packed rows match the shared encoder and fixed epochs save loadable weights."""

import dataclasses
import gzip
import json

import numpy as np
import pytest

from experiments.pbe_scale import train as driver
from experiments.world_behaviours.verify import expand, load_macros
from wandering_light.stack_pbe import Task
from wandering_light.stack_policy import load_policy, witness_rows


def task():
    program = tuple(expand(["sum"], load_macros()))
    return Task(
        behaviour="sum",
        summary="sum",
        split="train",
        task_hash_split="train",
        inputs=("( 1 2 )", "( 3 -1 )"),
        outputs=("3", "2"),
        held_out_inputs=(),
        held_out_outputs=(),
        program=program,
        length=len(program),
        examples=2,
        held_out=0,
        fuel=0,
        tags=("sum",),
    )


def source(tmp_path, rows=None):
    path = tmp_path / "tasks.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        for row in rows or [task()]:
            stream.write(json.dumps(dataclasses.asdict(row)) + "\n")
    return path


@pytest.mark.parametrize("encoding", ["machine", "difference"])
@pytest.mark.parametrize("prefixes", [0, 3])
def test_packed_rows_match_existing_training_rows(tmp_path, encoding, prefixes):
    frozen = task()
    payload = source(tmp_path, [frozen, dataclasses.replace(frozen, split="eval")])
    manifest = driver.prepare(
        [payload], tmp_path / "packed", prefixes=prefixes, encoding=encoding
    )
    packed = driver.PackedRows(tmp_path / "packed")
    selected = sorted(driver.prefix_steps(frozen, prefixes, 0))
    expected = witness_rows([frozen], encoding)
    assert manifest["tasks"] == 1
    assert manifest["rows"] == len(selected)
    assert manifest["full_prefixes"] == frozen.length
    assert packed.tokens.dtype == np.dtype("uint16")
    for row, step in zip(packed[:], selected, strict=True):
        np.testing.assert_array_equal(row.items, expected[step].items)
        assert (row.label, row.step, row.remaining) == (
            expected[step].label,
            step,
            frozen.length - step,
        )


def test_preparation_deterministic_and_partial_files_not_ready(tmp_path):
    payload = source(tmp_path)
    first = driver.prepare([payload], tmp_path / "first", prefixes=3)
    second = driver.prepare([payload], tmp_path / "second", prefixes=3)
    assert first["files"] == second["files"]
    assert first["data_sha256"] == second["data_sha256"]
    invalid = dataclasses.replace(task(), program=("drop", "drop"), length=2)
    payload = source(tmp_path, [invalid])
    with pytest.raises(ValueError, match="refused"):
        driver.prepare([payload], tmp_path / "partial")
    assert not (tmp_path / "partial/manifest.json").exists()
    assert not (tmp_path / "partial/tokens.bin").exists()


def test_prepare_reports_long_quotation_and_target_clipping(tmp_path):
    # A quotation is a stack value: one structural pair plus 180 op tokens.
    quotation = "( " + " ".join(["dup"] * 180) + " )"
    long_target = "( " + " ".join(["1"] * 70) + " )"
    frozen = dataclasses.replace(
        task(),
        inputs=(quotation,),
        outputs=(long_target,),
        program=("drop", "nil"),
        length=2,
        examples=1,
    )
    result = driver.prepare([source(tmp_path, [frozen])], tmp_path / "packed")
    stats = result["encoding_stats"]
    assert stats["max_state_tokens"] == 182
    assert stats["state_overflow_fraction"] == 0.5
    assert stats["max_target_tokens"] == 72
    assert stats["target_overflow_fraction"] == 1.0


def test_mixed_rows_preserve_labels_and_deterministic_old_mass(tmp_path):
    payload = source(tmp_path)
    driver.prepare([payload], tmp_path / "new", prefixes=0)
    driver.prepare([payload], tmp_path / "old", prefixes=3)
    new, old = driver.PackedRows(tmp_path / "new"), driver.PackedRows(tmp_path / "old")
    first = driver.MixedRows(new, old, 0.2, 17)
    second = driver.MixedRows(new, old, 0.2, 17)
    assert first.old_count == 2
    assert len(first) == len(new) + 2
    assert first.manifest["mixture"]["actual_old_share"] == 2 / len(first)
    assert [first.old_index(i) for i in range(9)] == [
        second.old_index(i) for i in range(9)
    ]
    assert len({first.old_index(i) for i in range(len(old))}) == len(old)
    for i in range(len(first)):
        expected = new[i] if i < len(new) else old[first.old_index(i - len(new))]
        assert first[i].label == expected.label
        np.testing.assert_array_equal(first[i].items, expected.items)
    assert first.manifest["stored_tokens"] == sum(row.items.size for row in first)
    with pytest.raises(IndexError):
        first[len(first)]


@pytest.mark.parametrize("mixture", [False, True])
def test_fixed_epoch_cpu_training_and_weights_only_checkpoint(
    monkeypatch, tmp_path, mixture
):
    payload = source(tmp_path)
    driver.prepare([payload], tmp_path / "packed", prefixes=0)
    monkeypatch.setitem(driver.ARMS, "1m", {"dims": (4, 8, 12)})
    result = driver.train(
        tmp_path / "packed",
        tmp_path / "policy",
        epochs=2,
        batch_size=2,
        wandb_mode="disabled",
        threads=1,
        device="cpu",
        mix_data=tmp_path / "packed" if mixture else None,
        mix_share=0.2,
    )
    assert result["optimizer_steps"] == 2 * (4 if mixture else 3)
    assert result["final_checkpoint"].endswith("epoch-002")
    for epoch in (1, 2):
        checkpoint = tmp_path / f"policy/epoch-{epoch:03d}"
        assert load_policy(checkpoint).parameter_count > 0
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["epoch"] == epoch
        assert "validation_cross_entropy" not in state
        assert "train_probe_cross_entropy" in state
        assert sorted(p.name for p in checkpoint.iterdir()) == [
            "model.safetensors",
            "policy_config.json",
            "trainer_state.json",
            "training_manifest.json",
        ]
