"""Basis-bound, non-executable task datasets for library experiments."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.typed_list import TypedList, _deserialize_value

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


SCHEMA_VERSION = 2

_SAFE_BUILTIN_TYPES: dict[str, type[Any]] = {
    "builtins.int": int,
    "builtins.float": float,
    "builtins.str": str,
    "builtins.bool": bool,
    "builtins.list": list,
    "builtins.tuple": tuple,
    "builtins.set": set,
    "builtins.dict": dict,
    "builtins.bytes": bytes,
    "builtins.bytearray": bytearray,
    "builtins.complex": complex,
    "builtins.range": range,
}


def typed_list_from_builtin_str(serialized: str) -> TypedList:
    """Decode corpus values without importing a record-controlled module.

    ``TypedList.from_str`` intentionally supports application-defined classes and
    therefore imports the serialized type. Basis-task artifacts are portable data,
    so their schema is deliberately narrower and accepts only the built-in types
    exercised by the registered DSL.
    """
    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid TypedList JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"type", "items"}:
        raise ValueError("TypedList data must contain exactly 'type' and 'items'")
    type_name = payload["type"]
    item_type = _SAFE_BUILTIN_TYPES.get(type_name)
    if item_type is None:
        raise ValueError(f"unsupported basis-task item type: {type_name!r}")
    if not isinstance(payload["items"], list):
        raise ValueError("TypedList 'items' must be a JSON array")
    try:
        items = [_deserialize_value(item) for item in payload["items"]]
        return TypedList(items, item_type=item_type)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid items for {type_name}: {error}") from error


def _canonical_typed_list(serialized: str) -> str:
    """Return the canonical JSON representation of a serialized ``TypedList``."""
    value = typed_list_from_builtin_str(serialized)
    return json.dumps(
        json.loads(value.to_string()), sort_keys=True, separators=(",", ":")
    )


def task_id_for(input_value: str, output_value: str) -> str:
    """Build an identity for an input/output task independent of its witness."""
    payload = {
        "input": json.loads(_canonical_typed_list(input_value)),
        "output": json.loads(_canonical_typed_list(output_value)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BasisTaskRecord:
    """One frozen input/output task and its generation provenance.

    The target output is stored directly. Candidate basis sets must solve that frozen
    task rather than recomputing the target through a potentially changed witness.
    """

    task_id: str
    split: str
    input: str
    output: str
    witness_function_ids: tuple[str, ...]
    witness_function_names: tuple[str, ...]
    basis_set_id: str
    basis_set_digest: str
    generator: str
    seed: int
    source_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported basis task schema version: {self.schema_version}"
            )
        if not self.split:
            raise ValueError("split must not be empty")
        if not self.basis_set_id or not self.basis_set_digest:
            raise ValueError("basis_set_id and basis_set_digest are required")
        if len(self.witness_function_ids) != len(self.witness_function_names):
            raise ValueError("witness function IDs and names must have equal length")
        expected_task_id = task_id_for(self.input, self.output)
        if self.task_id != expected_task_id:
            raise ValueError(
                f"task_id mismatch: expected {expected_task_id}, got {self.task_id}"
            )

    @property
    def witness_length(self) -> int:
        return len(self.witness_function_ids)

    @property
    def input_value(self) -> TypedList:
        return typed_list_from_builtin_str(self.input)

    @property
    def output_value(self) -> TypedList:
        return typed_list_from_builtin_str(self.output)

    @classmethod
    def create(
        cls,
        *,
        split: str,
        input_value: TypedList,
        output_value: TypedList,
        witness_function_ids: Iterable[str],
        witness_function_names: Iterable[str],
        basis_set_id: str,
        basis_set_digest: str,
        generator: str,
        seed: int,
        source_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> BasisTaskRecord:
        serialized_input = input_value.to_string()
        serialized_output = output_value.to_string()
        return cls(
            task_id=task_id_for(serialized_input, serialized_output),
            split=split,
            input=serialized_input,
            output=serialized_output,
            witness_function_ids=tuple(witness_function_ids),
            witness_function_names=tuple(witness_function_names),
            basis_set_id=basis_set_id,
            basis_set_digest=basis_set_digest,
            generator=generator,
            seed=seed,
            source_index=source_index,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["witness_function_ids"] = list(self.witness_function_ids)
        data["witness_function_names"] = list(self.witness_function_names)
        data["witness_length"] = self.witness_length
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BasisTaskRecord:
        values = dict(data)
        values.pop("witness_length", None)
        values["witness_function_ids"] = tuple(values["witness_function_ids"])
        values["witness_function_names"] = tuple(values["witness_function_names"])
        return cls(**values)


def write_basis_task_records(
    records: Iterable[BasisTaskRecord], path: str | Path
) -> Path:
    """Write deterministic gzip JSONL without executing dataset-provided code."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("wb") as raw_file,
        gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text_file,
    ):
        for record in records:
            text_file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return output_path


def iter_basis_task_records(
    path: str | Path,
    *,
    expected_basis_set_id: str | None = None,
    expected_basis_set_digest: str | None = None,
) -> Iterator[BasisTaskRecord]:
    """Stream and validate basis-task records from deterministic gzip JSONL."""
    with gzip.open(path, "rt", encoding="utf-8") as text_file:
        for line_number, line in enumerate(text_file, start=1):
            try:
                record = BasisTaskRecord.from_dict(json.loads(line))
            except Exception as error:
                raise ValueError(
                    f"invalid record at line {line_number}: {error}"
                ) from error
            if (
                expected_basis_set_id is not None
                and record.basis_set_id != expected_basis_set_id
            ):
                raise ValueError(
                    "basis set ID mismatch at line "
                    f"{line_number}: expected {expected_basis_set_id}, "
                    f"got {record.basis_set_id}"
                )
            if (
                expected_basis_set_digest is not None
                and record.basis_set_digest != expected_basis_set_digest
            ):
                raise ValueError(
                    "basis set digest mismatch at line "
                    f"{line_number}: expected {expected_basis_set_digest}, "
                    f"got {record.basis_set_digest}"
                )
            yield record


def read_basis_task_records(
    path: str | Path,
    *,
    expected_basis_set_id: str | None = None,
    expected_basis_set_digest: str | None = None,
) -> list[BasisTaskRecord]:
    """Read all records, retaining strict optional basis provenance checks."""
    return list(
        iter_basis_task_records(
            path,
            expected_basis_set_id=expected_basis_set_id,
            expected_basis_set_digest=expected_basis_set_digest,
        )
    )
