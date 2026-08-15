import copy
import json
import re
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import wandering_light.basis_set as basis_set_module
from wandering_light.basis_set import (
    BasisSetDigestMismatchError,
    BasisSetNotFoundError,
    BasisSetValidationError,
    available_basis_set_aliases,
    available_basis_sets,
    basis_function_fingerprint,
    basis_function_id,
    basis_set_digest,
    create_basis_set_manifest,
    load_basis_set,
    resolve_basis_set_id,
)
from wandering_light.common_functions import BASIC_BASIS_SET, basic_fns
from wandering_light.function_def import FunctionDef, FunctionDefSet


def _copy_builtin_manifests(destination: Path) -> Path:
    source = Path(basis_set_module.__file__).with_name("basis_sets")
    destination.mkdir()
    for path in source.glob("*.json"):
        shutil.copyfile(path, destination / path.name)
    return destination


def _rewrite_manifest(directory: Path, basis_set_id: str, raw: dict) -> None:
    path = directory / f"{basis_set_id}.json"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def test_default_alias_resolves_to_immutable_current_basis():
    basis = load_basis_set("default")

    assert basis.basis_set_id == "wl-core-v1"
    assert basis.digest == (
        "sha256:ffc602fb24db249df7eb4f6b0d5ba38d5a5070b2c68c6c5f94c7f387de682494"
    )
    assert basis.parent_basis_set_id == "wl-core-pyhash-v1"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", basis.digest)
    assert basis.identity_dict() == {
        "basis_set_id": "wl-core-v1",
        "basis_set_digest": basis.digest,
    }
    assert resolve_basis_set_id("default") == basis.basis_set_id
    assert resolve_basis_set_id(basis.basis_set_id) == basis.basis_set_id


def test_loaded_basis_can_round_trip_to_canonical_json_manifest():
    basis = load_basis_set("pilot-compressed")

    manifest = basis.to_manifest()

    assert isinstance(manifest["provenance"]["additions"], list)
    assert basis_set_digest(manifest) == basis.digest
    assert json.loads(json.dumps(manifest)) == manifest


def test_registry_lists_immutable_ids_and_alias_snapshots():
    assert available_basis_sets() == (
        "wl-core-v1",
        "wl-core-pyhash-v1",
        "wl-pilot-compressed-v1",
    )
    aliases = available_basis_set_aliases()
    assert aliases["default"] == "wl-core-v1"
    assert aliases["checkpoint-rl-6k-with-lp"] == "wl-core-pyhash-v1"
    assert aliases["pilot-compressed"] == "wl-pilot-compressed-v1"
    with pytest.raises(TypeError):
        aliases["default"] = "another-basis"


def test_current_manifest_preserves_basic_fns_order_and_compatibility():
    loaded = load_basis_set("wl-core-v1")
    runtime = loaded.as_function_set()

    assert loaded == BASIC_BASIS_SET
    assert len(runtime) == len(basic_fns) == 118
    assert [fn.name for fn in runtime] == [fn.name for fn in basic_fns]
    assert runtime[0].name == "inc"
    assert runtime[-1].name == "set_hash"
    assert list(runtime) == list(basic_fns)


def test_as_function_set_returns_fresh_objects_with_exact_provenance_keys():
    basis = load_basis_set()
    first = basis.as_function_set()
    second = basis.as_function_set()

    assert first is not second
    assert first[0] is not second[0]
    assert first[0].usage_count == second[0].usage_count == 0
    assert first[0].metadata == {
        "basis_function_id": basis.functions[0].function_id,
        "basis_function_fingerprint": basis.functions[0].fingerprint,
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
    }
    first[0].increment_usage()
    first[0].metadata["local"] = True
    assert second[0].usage_count == 0
    assert "local" not in second[0].metadata


def test_loaded_basis_and_nested_manifest_data_are_immutable():
    basis = load_basis_set()

    with pytest.raises(FrozenInstanceError):
        basis.basis_set_id = "changed"
    with pytest.raises(TypeError):
        basis.provenance["kind"] = "changed"
    with pytest.raises(TypeError):
        basis.functions[0].metadata["key"] = "value"


def test_function_ids_are_stable_across_bases_for_exact_definitions():
    current = {fn.name: fn for fn in load_basis_set("default")}
    checkpoint_basis = load_basis_set("checkpoint-rl-6k-with-lp")
    checkpoint = {fn.name: fn for fn in checkpoint_basis}

    assert len(checkpoint_basis) == 118
    assert checkpoint_basis.parent_basis_set_id is None
    assert checkpoint_basis.provenance["source_git_commit"] == (
        "4e9ac78cf2f1dd8c4ea9229d4d3e69c5ff36b7c7"
    )
    assert checkpoint_basis.provenance["hf_revision"] == (
        "0c4ea07bfa618321b8dc5ce956ce5a86560d99a7"
    )
    assert checkpoint_basis.provenance["pythonhashseed"] == "unknown"
    assert current["inc"].function_id == checkpoint["inc"].function_id
    assert current["inc"].fingerprint == checkpoint["inc"].fingerprint
    changed = {
        name
        for name in current
        if current[name].fingerprint != checkpoint[name].fingerprint
    }
    assert changed == {"set_to_list", "str_hash", "set_hash"}
    assert checkpoint["str_hash"].code == "return hash(x)"
    assert "PYTHONHASHSEED is unknown" in checkpoint_basis.description


def test_pilot_candidate_has_explicit_deprecation_lineage_and_added_macros():
    parent = load_basis_set("wl-core-v1")
    candidate = load_basis_set("pilot-compressed")
    parent_by_name = {function.name: function for function in parent}
    candidate_by_name = {function.name: function for function in candidate}

    assert candidate.basis_set_id == "wl-pilot-compressed-v1"
    assert candidate.parent_basis_set_id == parent.basis_set_id
    assert candidate.provenance["parent_basis_set_digest"] == parent.digest
    assert len(candidate) == 116
    assert set(parent_by_name) - set(candidate_by_name) == {
        "identity_int",
        "bool_identity",
        "duplicate",
    }
    assert set(candidate_by_name) - set(parent_by_name) == {"bytearray_is_empty"}
    assert candidate_by_name["bytearray_is_empty"].code == "return len(x) == 0"

    unchanged = set(parent_by_name).intersection(candidate_by_name)
    assert all(
        parent_by_name[name].function_id == candidate_by_name[name].function_id
        for name in unchanged
    )
    tombstones = {
        item["function_name"]: item for item in candidate.provenance["deprecations"]
    }
    assert set(tombstones) == {"identity_int", "bool_identity", "duplicate"}
    assert tombstones["duplicate"]["replacement"] == "repeat"
    assert tombstones["identity_int"]["function_id"] == (
        parent_by_name["identity_int"].function_id
    )


def test_basis_digest_is_canonical_but_function_order_is_significant():
    functions = list(basic_fns)[:2]
    forward = create_basis_set_manifest(
        basis_set_id="test-v1",
        description="Forward order.",
        function_defs=functions,
    )
    reordered_keys = dict(reversed(list(forward.items())))
    reverse = create_basis_set_manifest(
        basis_set_id="test-v1",
        description="Forward order.",
        function_defs=list(reversed(functions)),
    )

    assert basis_set_digest(reordered_keys) == forward["digest"]
    assert reverse["digest"] != forward["digest"]


def test_fingerprint_rejects_non_string_metadata_keys():
    with pytest.raises(BasisSetValidationError, match="must be a string"):
        basis_function_fingerprint(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
            metadata={1: "ambiguous after JSON round trip"},
        )


def test_hash_runtime_guard_rejects_environment_changed_after_startup(monkeypatch):
    basis = load_basis_set("wl-core-pyhash-v1")
    runtime_probe = basis_set_module._runtime_hash_probe()
    declared_seed = (
        "1" if runtime_probe == basis_set_module._declared_hash_seed_probe("0") else "0"
    )
    monkeypatch.setenv("PYTHONHASHSEED", declared_seed)

    with pytest.raises(RuntimeError, match="does not match the running interpreter"):
        basis_set_module.require_reproducible_basis_runtime(basis)


def test_loader_rejects_tampered_manifest_digest(tmp_path):
    manifests = _copy_builtin_manifests(tmp_path / "basis_sets")
    raw = json.loads((manifests / "wl-core-v1.json").read_text(encoding="utf-8"))
    raw["functions"][0]["code"] = "return x + 2"
    _rewrite_manifest(manifests, "wl-core-v1", raw)

    with pytest.raises(BasisSetDigestMismatchError, match="digest mismatch"):
        load_basis_set("default", manifest_dir=manifests)


def test_loader_rejects_stale_function_fingerprint_even_with_new_basis_digest(
    tmp_path,
):
    manifests = _copy_builtin_manifests(tmp_path / "basis_sets")
    raw = json.loads((manifests / "wl-core-v1.json").read_text(encoding="utf-8"))
    raw["functions"][0]["code"] = "return x + 2"
    raw["digest"] = basis_set_digest(raw)
    _rewrite_manifest(manifests, "wl-core-v1", raw)

    with pytest.raises(BasisSetValidationError, match="fingerprint mismatch"):
        load_basis_set("wl-core-v1", manifest_dir=manifests)


def test_index_digest_pins_immutable_id_after_valid_manifest_rehash(tmp_path):
    manifests = _copy_builtin_manifests(tmp_path / "basis_sets")
    raw = json.loads((manifests / "wl-core-v1.json").read_text(encoding="utf-8"))
    function = raw["functions"][0]
    function["code"] = "return x + 2"
    function["fingerprint"] = basis_function_fingerprint(
        name=function["name"],
        input_type=function["input_type"],
        output_type=function["output_type"],
        code=function["code"],
        metadata=function["metadata"],
    )
    function["function_id"] = basis_function_id(
        function["name"], function["fingerprint"]
    )
    raw["digest"] = basis_set_digest(raw)
    _rewrite_manifest(manifests, "wl-core-v1", raw)

    with pytest.raises(BasisSetDigestMismatchError, match="index-pinned digest"):
        load_basis_set("wl-core-v1", manifest_dir=manifests)


def test_loader_rejects_duplicate_function_names(tmp_path):
    manifests = _copy_builtin_manifests(tmp_path / "basis_sets")
    raw = json.loads((manifests / "wl-core-v1.json").read_text(encoding="utf-8"))
    raw["functions"].append(copy.deepcopy(raw["functions"][0]))
    raw["digest"] = basis_set_digest(raw)
    _rewrite_manifest(manifests, "wl-core-v1", raw)

    with pytest.raises(BasisSetValidationError, match="duplicate name 'inc'"):
        load_basis_set("wl-core-v1", manifest_dir=manifests)


def test_function_def_set_rejects_same_name_with_different_definition():
    existing = FunctionDef(
        name="same_name",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    conflicting = FunctionDef(
        name="same_name",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 2",
    )

    with pytest.raises(ValueError, match=r"Conflicting definitions.*same_name"):
        FunctionDefSet([existing, conflicting])

    deduplicated = FunctionDefSet([existing, existing.model_copy(deep=True)])
    assert len(deduplicated) == 1


def test_function_def_set_rejects_same_code_from_different_bases():
    first = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
        metadata={
            "basis_function_id": "inc-v1",
            "basis_function_fingerprint": "definition-v1",
            "basis_set_id": "basis-a",
            "basis_set_digest": "digest-a",
        },
    )
    second = first.model_copy(deep=True)
    second.metadata.update({"basis_set_id": "basis-b", "basis_set_digest": "digest-b"})

    with pytest.raises(ValueError, match=r"Conflicting definitions.*provenance"):
        FunctionDefSet([first, second])


def test_loader_only_accepts_registered_safe_identifiers():
    with pytest.raises(BasisSetNotFoundError):
        load_basis_set("unknown-v1")
    with pytest.raises(BasisSetValidationError, match="Invalid basis-set ID"):
        load_basis_set("../wl-core-v1")


def test_loader_rejects_alias_id_namespace_collision(tmp_path):
    manifests = _copy_builtin_manifests(tmp_path / "basis_sets")
    index_path = manifests / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["aliases"]["wl-core-v1"] = "wl-core-pyhash-v1"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BasisSetValidationError, match="share one namespace"):
        load_basis_set("wl-core-v1", manifest_dir=manifests)
