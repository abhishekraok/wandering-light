import gzip
import json

import pytest

from wandering_light.basis_dataset import (
    BasisTaskRecord,
    iter_basis_task_records,
    read_basis_task_records,
    task_id_for,
    typed_list_from_builtin_str,
    write_basis_task_records,
)
from wandering_light.typed_list import TypedList


def make_record() -> BasisTaskRecord:
    return BasisTaskRecord.create(
        split="discovery",
        input_value=TypedList([1, 2]),
        output_value=TypedList([2, 3]),
        witness_function_ids=["inc@1"],
        witness_function_names=["inc"],
        basis_set_id="basic-current-v1",
        basis_set_digest="a" * 64,
        generator="balanced-random-walk-v1",
        seed=17,
        source_index=3,
        metadata={"requested_length": 1},
    )


def test_record_round_trip_is_deterministic(tmp_path):
    record = make_record()
    first = write_basis_task_records([record], tmp_path / "first.jsonl.gz")
    second = write_basis_task_records([record], tmp_path / "second.jsonl.gz")

    assert first.read_bytes() == second.read_bytes()
    assert read_basis_task_records(first) == [record]


def test_task_id_ignores_json_whitespace():
    original = TypedList([1, 2]).to_string()
    spaced = json.dumps(json.loads(original), indent=2)
    output = TypedList([2, 3]).to_string()

    assert task_id_for(original, output) == task_id_for(spaced, output)


def test_record_rejects_tampered_task_id():
    data = make_record().to_dict()
    data["task_id"] = "0" * 64

    with pytest.raises(ValueError, match="task_id mismatch"):
        BasisTaskRecord.from_dict(data)


def test_reader_rejects_basis_mismatch(tmp_path):
    path = write_basis_task_records([make_record()], tmp_path / "tasks.jsonl.gz")

    with pytest.raises(ValueError, match="basis set ID mismatch"):
        list(iter_basis_task_records(path, expected_basis_set_id="different"))
    with pytest.raises(ValueError, match="basis set digest mismatch"):
        list(iter_basis_task_records(path, expected_basis_set_digest="b" * 64))


def test_reader_reports_invalid_line(tmp_path):
    path = tmp_path / "invalid.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("{}\n")

    with pytest.raises(ValueError, match="invalid record at line 1"):
        list(iter_basis_task_records(path))


def test_builtin_decoder_never_imports_record_controlled_module(monkeypatch):
    def fail_import(*args, **kwargs):
        raise AssertionError("dataset decoder attempted a dynamic import")

    monkeypatch.setattr("importlib.import_module", fail_import)
    serialized = json.dumps({"type": "attacker.payload", "items": []})

    with pytest.raises(ValueError, match="unsupported basis-task item type"):
        typed_list_from_builtin_str(serialized)
