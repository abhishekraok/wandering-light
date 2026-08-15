"""Regenerate the built-in basis-set manifests from durable sources.

Run from the repository root with::

    uv run python -m wandering_light.basis_sets.generate_builtin_manifests

The parent snapshot is the exact palette associated with the ``rl-6k-with-lp``
checkpoint. The child snapshot is the deterministic-hash palette on current main.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from wandering_light.basis_set import create_basis_set_manifest
from wandering_light.function_def import FunctionDef

CURRENT_BASIS_SET_ID = "wl-core-v1"
CHECKPOINT_BASIS_SET_ID = "wl-core-pyhash-v1"
PILOT_BASIS_SET_ID = "wl-pilot-compressed-v1"
CHECKPOINT_SOURCE_COMMIT = "4e9ac78cf2f1dd8c4ea9229d4d3e69c5ff36b7c7"
CURRENT_SOURCE_COMMIT = "deddaf19243b4d8ab395fcc66653e820abf936b9"
HF_REVISION = "0c4ea07bfa618321b8dc5ce956ce5a86560d99a7"
WANDB_RUN_URL = (
    "https://wandb.ai/abhishekraok-na/wandering-light-rl_induction/runs/dp8ylg8y"
)
PILOT_BASELINE_WANDB_RUN_URL = (
    "https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/3fefq3zb"
)
PILOT_CORPUS_MANIFEST_DIGEST = (
    "sha256:40fa81a3376fb45b41e93dc95dcc23e60a666c024aca5f40b184da891e15025c"
)
PILOT_BASELINE_AGGREGATE_DIGEST = (
    "sha256:b21f0457d1ed7eb952005a11dc15dbfe15258292e4e0ef665ba86081ff7cfd10"
)

PILOT_DEPRECATED_FUNCTIONS = {
    "identity_int": {
        "function_id": "bf:identity_int:d4850619def23cdf",
        "total_occurrences": 306,
        "task_coverage": 194,
        "reason": "Semantic no-op; omitting it strictly shortens every use.",
        "replacement": "zero-step identity",
    },
    "bool_identity": {
        "function_id": "bf:bool_identity:0215e6b5f83c85af",
        "total_occurrences": 25,
        "task_coverage": 21,
        "reason": "Semantic no-op; omitting it strictly shortens every use.",
        "replacement": "zero-step identity",
    },
    "duplicate": {
        "function_id": "bf:duplicate:b6fcd3e3946b815e",
        "total_occurrences": 231,
        "task_coverage": 224,
        "reason": "Semantically equivalent to repeat, which had 328 occurrences.",
        "replacement": "repeat",
    },
}

PILOT_ADDED_FUNCTIONS = (
    FunctionDef(
        name="bytearray_is_empty",
        input_type="builtins.bytearray",
        output_type="builtins.bool",
        code="return len(x) == 0",
    ),
)


def _function_defs_at_commit(commit: str) -> list[FunctionDef]:
    source_path = "wandering_light/common_functions.py"
    result = subprocess.run(
        ["git", "show", f"{commit}:{source_path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    namespace: dict[str, object] = {
        "__name__": "basis_set_historical_source",
        "__file__": f"git:{commit}:{source_path}",
    }
    exec(compile(result.stdout, namespace["__file__"], "exec"), namespace)
    function_defs = namespace.get("_basic_fns_list")
    if not isinstance(function_defs, list) or not all(
        isinstance(function, FunctionDef) for function in function_defs
    ):
        raise RuntimeError(f"Could not extract _basic_fns_list at {commit}")
    return function_defs


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    output_dir = Path(__file__).parent
    checkpoint = create_basis_set_manifest(
        basis_set_id=CHECKPOINT_BASIS_SET_ID,
        description=(
            "Exact ordered 118-function palette associated with rl-6k-with-lp. "
            "Its str_hash and set_hash use process-randomized Python hashing; "
            "the training PYTHONHASHSEED is unknown."
        ),
        function_defs=_function_defs_at_commit(CHECKPOINT_SOURCE_COMMIT),
        provenance={
            "kind": "git_and_checkpoint_snapshot",
            "source_git_commit": CHECKPOINT_SOURCE_COMMIT,
            "source_path": "wandering_light/common_functions.py",
            "checkpoint_alias": "rl-6k-with-lp",
            "hf_revision": HF_REVISION,
            "wandb_run_url": WANDB_RUN_URL,
            "pythonhashseed": "unknown",
        },
    )
    current = create_basis_set_manifest(
        basis_set_id=CURRENT_BASIS_SET_ID,
        description=(
            "Current ordered 118-function core palette with deterministic set/list "
            "ordering and SHA-256-backed str_hash and set_hash."
        ),
        function_defs=_function_defs_at_commit(CURRENT_SOURCE_COMMIT),
        parent_basis_set_id=CHECKPOINT_BASIS_SET_ID,
        provenance={
            "kind": "git_snapshot",
            "source_git_commit": CURRENT_SOURCE_COMMIT,
            "source_path": "wandering_light/common_functions.py",
            "changes_from_parent": ["set_to_list", "str_hash", "set_hash"],
        },
    )
    current_function_defs = _function_defs_at_commit(CURRENT_SOURCE_COMMIT)
    pilot_function_defs = [
        function
        for function in current_function_defs
        if function.name not in PILOT_DEPRECATED_FUNCTIONS
    ]
    pilot_function_defs.extend(PILOT_ADDED_FUNCTIONS)
    pilot = create_basis_set_manifest(
        basis_set_id=PILOT_BASIS_SET_ID,
        description=(
            "First measured basis-library candidate: removes two no-op identities "
            "and the lower-use duplicate string function, then adds a direct "
            "bytearray emptiness predicate mined from verified solver traces."
        ),
        function_defs=pilot_function_defs,
        parent_basis_set_id=CURRENT_BASIS_SET_ID,
        provenance={
            "kind": "basis_library_pilot_candidate",
            "parent_basis_set_digest": current["digest"],
            "selection_metric": "total_occurrences_in_verified_solutions",
            "corpus_manifest_digest": PILOT_CORPUS_MANIFEST_DIGEST,
            "baseline_aggregate_digest": PILOT_BASELINE_AGGREGATE_DIGEST,
            "baseline_wandb_run_url": PILOT_BASELINE_WANDB_RUN_URL,
            "deprecations": [
                {"function_name": name, **evidence}
                for name, evidence in PILOT_DEPRECATED_FUNCTIONS.items()
            ],
            "additions": [
                {
                    "function_name": "bytearray_is_empty",
                    "source_sequence": ["bytearray_to_bytes", "bytes_is_empty"],
                    "total_occurrences": 357,
                    "task_coverage": 357,
                },
            ],
        },
    )
    _write_json(output_dir / f"{CHECKPOINT_BASIS_SET_ID}.json", checkpoint)
    _write_json(output_dir / f"{CURRENT_BASIS_SET_ID}.json", current)
    _write_json(output_dir / f"{PILOT_BASIS_SET_ID}.json", pilot)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "aliases": {
                "default": CURRENT_BASIS_SET_ID,
                "checkpoint-rl-6k-with-lp": CHECKPOINT_BASIS_SET_ID,
                "pilot-compressed": PILOT_BASIS_SET_ID,
            },
            "basis_sets": {
                CURRENT_BASIS_SET_ID: {
                    "resource": f"{CURRENT_BASIS_SET_ID}.json",
                    "digest": current["digest"],
                },
                CHECKPOINT_BASIS_SET_ID: {
                    "resource": f"{CHECKPOINT_BASIS_SET_ID}.json",
                    "digest": checkpoint["digest"],
                },
                PILOT_BASIS_SET_ID: {
                    "resource": f"{PILOT_BASIS_SET_ID}.json",
                    "digest": pilot["digest"],
                },
            },
        },
    )


if __name__ == "__main__":
    main()
