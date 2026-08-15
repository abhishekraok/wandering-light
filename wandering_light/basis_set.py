"""Immutable, content-addressed basis-function palettes.

Basis sets are stored as ordered JSON manifests under ``wandering_light/basis_sets``.
Aliases such as ``default`` are only lookup conveniences: a loaded :class:`BasisSet`
always exposes the immutable manifest ID and verified SHA-256 digest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import keyword
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from wandering_light.function_def import FunctionDef, FunctionDefSet

MANIFEST_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.json"
DEFAULT_BASIS_SET_ALIAS = "default"
RUNTIME_METADATA_KEYS = frozenset(
    {
        "basis_function_id",
        "basis_function_fingerprint",
        "basis_set_id",
        "basis_set_digest",
    }
)

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_INDEX_KEYS = frozenset({"schema_version", "aliases", "basis_sets"})
_INDEX_BASIS_SET_KEYS = frozenset({"resource", "digest"})
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "basis_set_id",
        "description",
        "parent_basis_set_id",
        "provenance",
        "functions",
        "digest",
    }
)
_FUNCTION_KEYS = frozenset(
    {
        "function_id",
        "fingerprint",
        "name",
        "input_type",
        "output_type",
        "code",
        "metadata",
    }
)


class BasisSetError(ValueError):
    """Base error for invalid or unavailable basis-set manifests."""


class BasisSetNotFoundError(BasisSetError):
    """Raised when an immutable ID or alias is not registered."""


class BasisSetValidationError(BasisSetError):
    """Raised when a basis-set resource does not satisfy the strict schema."""


class BasisSetDigestMismatchError(BasisSetValidationError):
    """Raised when content does not match its recorded SHA-256 digest."""


def _runtime_hash_probe() -> tuple[int, int]:
    return (
        hash("wandering-light-hash-seed-probe"),
        hash(frozenset(("wandering", "light", 1729))),
    )


@lru_cache(maxsize=16)
def _declared_hash_seed_probe(hash_seed: str) -> tuple[int, int]:
    probe_code = (
        "import json; "
        "print(json.dumps([hash('wandering-light-hash-seed-probe'), "
        "hash(frozenset(('wandering', 'light', 1729)))]))"
    )
    child_environment = os.environ.copy()
    child_environment["PYTHONHASHSEED"] = hash_seed
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=True,
            capture_output=True,
            text=True,
            env=child_environment,
        )
        raw = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not validate declared PYTHONHASHSEED={hash_seed!r}"
        ) from exc
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(not isinstance(value, int) for value in raw)
    ):
        raise RuntimeError("Python hash-seed probe returned invalid output")
    return raw[0], raw[1]


def require_reproducible_basis_runtime(basis_set: BasisSet) -> str | None:
    """Reject randomized-hash palettes unless Python started with a fixed seed."""
    if any("hash(" in function.code for function in basis_set.functions):
        hash_seed = os.environ.get("PYTHONHASHSEED")
        if hash_seed is None or hash_seed.lower() == "random":
            raise RuntimeError(
                f"Basis {basis_set.basis_set_id!r} uses process-randomized Python "
                "hashing. Relaunch Python with a fixed seed, for example "
                "PYTHONHASHSEED=0."
            )
        if _runtime_hash_probe() != _declared_hash_seed_probe(hash_seed):
            raise RuntimeError(
                f"Declared PYTHONHASHSEED={hash_seed!r} does not match the running "
                "interpreter. Relaunch Python with that seed before evaluation."
            )
        return hash_seed
    return None


def _canonical_json(value: Any) -> str:
    def validate_json_tree(item: Any, path: str) -> None:
        if item is None or isinstance(item, str | int | float | bool):
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate_json_tree(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise BasisSetValidationError(
                        f"Basis-set JSON object key at {path} must be a string: {key!r}"
                    )
                validate_json_tree(child, f"{path}.{key}")
            return
        raise BasisSetValidationError(
            f"Basis-set value at {path} is not canonical JSON data: {type(item)!r}"
        )

    validate_json_tree(value, "$")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BasisSetValidationError(
            f"Basis-set content must be canonical JSON data: {exc}"
        ) from exc


def _sha256(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _definition_payload(
    *,
    name: str,
    input_type: str,
    output_type: str,
    code: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "input_type": input_type,
        "output_type": output_type,
        "code": code,
        "metadata": copy.deepcopy(dict(metadata)),
    }


def basis_function_fingerprint(
    *,
    name: str,
    input_type: str,
    output_type: str,
    code: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return the exact-definition SHA-256 shared across basis-set versions."""
    return _sha256(
        _definition_payload(
            name=name,
            input_type=input_type,
            output_type=output_type,
            code=code,
            metadata=metadata or {},
        )
    )


def basis_function_id(name: str, fingerprint: str) -> str:
    """Return a readable, stable ID derived from a function definition."""
    if not _SHA256_RE.fullmatch(fingerprint):
        raise BasisSetValidationError(
            f"Invalid basis-function fingerprint {fingerprint!r}"
        )
    return f"bf:{name}:{fingerprint.removeprefix('sha256:')[:16]}"


def basis_set_digest(manifest: Mapping[str, Any]) -> str:
    """Compute a manifest digest, excluding its self-referential ``digest`` key."""
    payload = copy.deepcopy(dict(manifest))
    payload.pop("digest", None)
    return _sha256(payload)


def _without_runtime_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key not in RUNTIME_METADATA_KEYS
    }


def function_manifest_record(function_def: FunctionDef) -> dict[str, Any]:
    """Serialize the immutable part of a runtime ``FunctionDef``."""
    metadata = _without_runtime_metadata(function_def.metadata)
    fingerprint = basis_function_fingerprint(
        name=function_def.name,
        input_type=function_def.input_type,
        output_type=function_def.output_type,
        code=function_def.code,
        metadata=metadata,
    )
    return {
        "function_id": basis_function_id(function_def.name, fingerprint),
        "fingerprint": fingerprint,
        "name": function_def.name,
        "input_type": function_def.input_type,
        "output_type": function_def.output_type,
        "code": function_def.code,
        "metadata": metadata,
    }


def create_basis_set_manifest(
    *,
    basis_set_id: str,
    description: str,
    function_defs: Sequence[FunctionDef],
    parent_basis_set_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete manifest dict with function and palette digests."""
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "basis_set_id": basis_set_id,
        "description": description,
        "parent_basis_set_id": parent_basis_set_id,
        "provenance": copy.deepcopy(dict(provenance or {})),
        "functions": [function_manifest_record(fn) for fn in function_defs],
    }
    manifest["digest"] = basis_set_digest(manifest)
    # Validate generated content through the same path used by the loader.
    _parse_manifest(manifest, expected_basis_set_id=basis_set_id)
    return manifest


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class BasisFunction:
    """One immutable function record from a verified manifest."""

    function_id: str
    fingerprint: str
    name: str
    input_type: str
    output_type: str
    code: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BasisSet:
    """An ordered, immutable, content-addressed function palette."""

    schema_version: int
    basis_set_id: str
    digest: str
    description: str
    parent_basis_set_id: str | None
    provenance: Mapping[str, Any]
    functions: tuple[BasisFunction, ...]

    def __len__(self) -> int:
        return len(self.functions)

    def __iter__(self):
        return iter(self.functions)

    def identity_dict(self) -> dict[str, str]:
        """Return the immutable identity fields to persist with an artifact."""
        return {
            "basis_set_id": self.basis_set_id,
            "basis_set_digest": self.digest,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Return a mutable, canonical-JSON representation of this basis set."""
        manifest = {
            "schema_version": self.schema_version,
            "basis_set_id": self.basis_set_id,
            "description": self.description,
            "parent_basis_set_id": self.parent_basis_set_id,
            "provenance": _deep_thaw(self.provenance),
            "functions": [
                {
                    "function_id": function.function_id,
                    "fingerprint": function.fingerprint,
                    "name": function.name,
                    "input_type": function.input_type,
                    "output_type": function.output_type,
                    "code": function.code,
                    "metadata": _deep_thaw(function.metadata),
                }
                for function in self.functions
            ],
            "digest": self.digest,
        }
        if basis_set_digest(manifest) != self.digest:
            raise BasisSetDigestMismatchError(
                f"Could not reproduce digest for loaded basis {self.basis_set_id!r}"
            )
        return manifest

    def as_function_set(self) -> FunctionDefSet:
        """Create a fresh executable palette with provenance in every function."""
        definitions = []
        for function in self.functions:
            metadata = _deep_thaw(function.metadata)
            metadata.update(
                {
                    "basis_function_id": function.function_id,
                    "basis_function_fingerprint": function.fingerprint,
                    "basis_set_id": self.basis_set_id,
                    "basis_set_digest": self.digest,
                }
            )
            definitions.append(
                FunctionDef(
                    name=function.name,
                    input_type=function.input_type,
                    output_type=function.output_type,
                    code=function.code,
                    usage_count=0,
                    metadata=metadata,
                )
            )
        return FunctionDefSet(definitions)

    # A descriptive synonym for callers that prefer the concrete return type.
    to_function_def_set = as_function_set


def _validate_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BasisSetValidationError(
            f"{context} has invalid keys (missing={missing}, extra={extra})"
        )


def _validate_identifier(identifier: Any, context: str) -> str:
    if not isinstance(identifier, str) or not _IDENTIFIER_RE.fullmatch(identifier):
        raise BasisSetValidationError(f"Invalid {context}: {identifier!r}")
    return identifier


def _validate_string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise BasisSetValidationError(f"{context} must be a non-empty string")
    return value


def _parse_manifest(raw: Any, *, expected_basis_set_id: str | None = None) -> BasisSet:
    if not isinstance(raw, dict):
        raise BasisSetValidationError("Basis-set manifest must be a JSON object")
    _validate_exact_keys(raw, _MANIFEST_KEYS, "Basis-set manifest")

    if (
        not isinstance(raw["schema_version"], int)
        or isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise BasisSetValidationError(
            f"Unsupported basis-set schema version {raw['schema_version']!r}"
        )
    basis_set_id_value = _validate_identifier(raw["basis_set_id"], "basis_set_id")
    if (
        expected_basis_set_id is not None
        and basis_set_id_value != expected_basis_set_id
    ):
        raise BasisSetValidationError(
            f"Manifest ID {basis_set_id_value!r} does not match registered ID "
            f"{expected_basis_set_id!r}"
        )
    description = _validate_string(raw["description"], "description")
    parent = raw["parent_basis_set_id"]
    if parent is not None:
        parent = _validate_identifier(parent, "parent_basis_set_id")
    if not isinstance(raw["provenance"], dict):
        raise BasisSetValidationError("provenance must be a JSON object")
    if not isinstance(raw["functions"], list):
        raise BasisSetValidationError("functions must be an ordered JSON array")
    if not _SHA256_RE.fullmatch(str(raw["digest"])):
        raise BasisSetValidationError(
            "digest must be a sha256:<64 lowercase hex> value"
        )

    computed_digest = basis_set_digest(raw)
    if raw["digest"] != computed_digest:
        raise BasisSetDigestMismatchError(
            f"Basis set {basis_set_id_value!r} digest mismatch: "
            f"recorded={raw['digest']}, computed={computed_digest}"
        )

    functions: list[BasisFunction] = []
    names: set[str] = set()
    function_ids: set[str] = set()
    for index, record in enumerate(raw["functions"]):
        context = f"functions[{index}]"
        if not isinstance(record, dict):
            raise BasisSetValidationError(f"{context} must be a JSON object")
        _validate_exact_keys(record, _FUNCTION_KEYS, context)
        name = _validate_string(record["name"], f"{context}.name")
        if not name.isidentifier() or keyword.iskeyword(name):
            raise BasisSetValidationError(
                f"{context}.name must be a valid Python function identifier"
            )
        input_type = _validate_string(record["input_type"], f"{context}.input_type")
        output_type = _validate_string(record["output_type"], f"{context}.output_type")
        code = _validate_string(record["code"], f"{context}.code")
        if not isinstance(record["metadata"], dict):
            raise BasisSetValidationError(f"{context}.metadata must be a JSON object")
        reserved = RUNTIME_METADATA_KEYS.intersection(record["metadata"])
        if reserved:
            raise BasisSetValidationError(
                f"{context}.metadata contains runtime-reserved keys: {sorted(reserved)}"
            )

        fingerprint = basis_function_fingerprint(
            name=name,
            input_type=input_type,
            output_type=output_type,
            code=code,
            metadata=record["metadata"],
        )
        if record["fingerprint"] != fingerprint:
            raise BasisSetValidationError(
                f"{context} fingerprint mismatch: recorded={record['fingerprint']!r}, "
                f"computed={fingerprint!r}"
            )
        expected_function_id = basis_function_id(name, fingerprint)
        if record["function_id"] != expected_function_id:
            raise BasisSetValidationError(
                f"{context} function_id mismatch: recorded={record['function_id']!r}, "
                f"computed={expected_function_id!r}"
            )
        if name in names:
            raise BasisSetValidationError(
                f"Basis set {basis_set_id_value!r} contains duplicate name {name!r}"
            )
        if expected_function_id in function_ids:
            raise BasisSetValidationError(
                f"Basis set {basis_set_id_value!r} contains duplicate function ID "
                f"{expected_function_id!r}"
            )
        names.add(name)
        function_ids.add(expected_function_id)
        functions.append(
            BasisFunction(
                function_id=expected_function_id,
                fingerprint=fingerprint,
                name=name,
                input_type=input_type,
                output_type=output_type,
                code=code,
                metadata=_deep_freeze(copy.deepcopy(record["metadata"])),
            )
        )

    return BasisSet(
        schema_version=MANIFEST_SCHEMA_VERSION,
        basis_set_id=basis_set_id_value,
        digest=computed_digest,
        description=description,
        parent_basis_set_id=parent,
        provenance=_deep_freeze(copy.deepcopy(raw["provenance"])),
        functions=tuple(functions),
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BasisSetValidationError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(resource_root: Any, filename: str) -> Any:
    try:
        text = resource_root.joinpath(filename).read_text(encoding="utf-8")
        return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except BasisSetError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise BasisSetValidationError(
            f"Could not read basis-set resource {filename!r}: {exc}"
        ) from exc


def _basis_sets_root(manifest_dir: str | Path | None) -> Any:
    if manifest_dir is not None:
        return Path(manifest_dir)
    return resources.files("wandering_light").joinpath("basis_sets")


def _load_index(resource_root: Any) -> dict[str, Any]:
    raw = _read_json(resource_root, INDEX_FILENAME)
    if not isinstance(raw, dict):
        raise BasisSetValidationError("Basis-set index must be a JSON object")
    _validate_exact_keys(raw, _INDEX_KEYS, "Basis-set index")
    if (
        not isinstance(raw["schema_version"], int)
        or isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise BasisSetValidationError(
            f"Unsupported basis-set index schema version {raw['schema_version']!r}"
        )
    aliases = raw["aliases"]
    basis_sets = raw["basis_sets"]
    if not isinstance(aliases, dict) or not isinstance(basis_sets, dict):
        raise BasisSetValidationError("aliases and basis_sets must be JSON objects")
    namespace_collisions = set(aliases).intersection(basis_sets)
    if namespace_collisions:
        raise BasisSetValidationError(
            "Basis-set aliases and immutable IDs share one namespace; collisions: "
            f"{sorted(namespace_collisions)}"
        )

    for basis_id, registration in basis_sets.items():
        _validate_identifier(basis_id, "registered basis_set_id")
        if not isinstance(registration, dict):
            raise BasisSetValidationError(
                f"Registration for {basis_id!r} must be a JSON object"
            )
        _validate_exact_keys(
            registration,
            _INDEX_BASIS_SET_KEYS,
            f"Registration for {basis_id!r}",
        )
        filename = registration["resource"]
        if (
            not isinstance(filename, str)
            or not filename.endswith(".json")
            or filename == INDEX_FILENAME
            or Path(filename).name != filename
        ):
            raise BasisSetValidationError(
                f"Unsafe manifest filename for {basis_id!r}: {filename!r}"
            )
        if not isinstance(registration["digest"], str) or not _SHA256_RE.fullmatch(
            registration["digest"]
        ):
            raise BasisSetValidationError(
                f"Registration for {basis_id!r} has an invalid pinned digest"
            )
    for alias, target in aliases.items():
        _validate_identifier(alias, "basis-set alias")
        if target not in basis_sets:
            raise BasisSetValidationError(
                f"Alias {alias!r} targets unregistered basis set {target!r}"
            )
    return raw


def resolve_basis_set_id(
    basis_set_id_or_alias: str = DEFAULT_BASIS_SET_ALIAS,
    *,
    manifest_dir: str | Path | None = None,
) -> str:
    """Resolve an alias to an immutable registered basis-set ID."""
    _validate_identifier(basis_set_id_or_alias, "basis-set ID or alias")
    root = _basis_sets_root(manifest_dir)
    index = _load_index(root)
    if basis_set_id_or_alias in index["basis_sets"]:
        return basis_set_id_or_alias
    try:
        return index["aliases"][basis_set_id_or_alias]
    except KeyError as exc:
        raise BasisSetNotFoundError(
            f"Unknown basis-set ID or alias {basis_set_id_or_alias!r}"
        ) from exc


def load_basis_set(
    basis_set_id_or_alias: str = DEFAULT_BASIS_SET_ALIAS,
    *,
    manifest_dir: str | Path | None = None,
) -> BasisSet:
    """Load and fully verify a registered immutable basis-set manifest."""
    _validate_identifier(basis_set_id_or_alias, "basis-set ID or alias")
    root = _basis_sets_root(manifest_dir)
    index = _load_index(root)
    resolved_id = resolve_basis_set_id(basis_set_id_or_alias, manifest_dir=manifest_dir)
    if resolved_id not in index["basis_sets"]:
        raise BasisSetNotFoundError(
            f"Unknown basis-set ID or alias {basis_set_id_or_alias!r}"
        )
    registration = index["basis_sets"][resolved_id]
    raw = _read_json(root, registration["resource"])
    basis_set = _parse_manifest(raw, expected_basis_set_id=resolved_id)
    if basis_set.digest != registration["digest"]:
        raise BasisSetDigestMismatchError(
            f"Basis set {resolved_id!r} does not match its index-pinned digest: "
            f"registered={registration['digest']}, manifest={basis_set.digest}"
        )
    return basis_set


def available_basis_sets(*, manifest_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return registered immutable IDs in index order."""
    index = _load_index(_basis_sets_root(manifest_dir))
    return tuple(index["basis_sets"])


def available_basis_set_aliases(
    *, manifest_dir: str | Path | None = None
) -> Mapping[str, str]:
    """Return an immutable alias-to-ID snapshot."""
    index = _load_index(_basis_sets_root(manifest_dir))
    return MappingProxyType(dict(index["aliases"]))
