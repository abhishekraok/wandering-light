"""Offline, reproducible analysis for a basis-library pilot.

The analyzer deliberately does not load a model or contact an external service.  It
validates the frozen corpus and evaluation artifacts, verifies provenance-directed
counterfactual rewrites by executing them, and (when two real evaluations are
available) computes paired statistical and cost comparisons.

Example::

    python -m experiments.analyze_basis_pilot \
      --corpus-dir results/basis-library-20260815/baseline/corpus \
      --champion-eval-dir results/basis-library-20260815/baseline/test \
      --challenger-eval-dir results/basis-library-20260815/challenger/test \
      --source-basis-set-id wl-core-v1 \
      --target-basis-set-id wl-pilot-compressed-v1 \
      --output reports/basis-library-20260815/analysis.json
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.basis_library_pilot import load_corpus
from wandering_light.basis_set import BasisFunction, BasisSet, load_basis_set
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.function_def import FunctionDef


ANALYSIS_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
EXPLICIT_COST_SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_815
EXECUTION_ENVIRONMENT_SCHEMA_VERSION = 1

_RESULT_FILENAMES = ("aggregate.json", "results.jsonl.gz")
_REQUIRED_RESULT_KEYS = frozenset(
    {
        "task_id",
        "split",
        "input_type",
        "witness_length",
        "success",
        "solution_length",
        "solution_function_ids",
        "solution_function_names",
    }
)
_STATIC_COST_WEIGHTS = {
    "dispatch": 1.0,
    "call": 1.0,
    "binary_operation": 1.0,
    "boolean_link": 1.0,
    "comparison": 1.0,
    "unary_operation": 1.0,
    "subscript": 1.0,
    "branch": 1.0,
    "comprehension_generator": 2.0,
    "attribute_lookup": 0.25,
}
_DECODING_PROTOCOL_FIELDS = (
    "budget",
    "deterministic",
    "seed",
    "requested_batch_size",
    "device",
    "max_new_tokens",
    "temperature",
)
_EXECUTION_ENVIRONMENT_KEYS = frozenset(
    {
        "schema_version",
        "requested_device",
        "resolved_device",
        "hardware_fingerprint",
        "hardware",
        "batch_protocol",
    }
)
_BATCH_PROTOCOL_KEYS = frozenset(
    {
        "requested_batch_size",
        "solver_inference_batch_size",
        "candidates_per_task",
        "observed_batch_count",
        "observed_batch_size_histogram",
        "effective_inference_batch_count",
        "effective_inference_batch_size_histogram",
    }
)


def _canonical_json(value: Any) -> str:
    """Encode strict, byte-stable JSON (and reject NaN/infinity)."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_tree(value: Any) -> Any:
    """Thaw immutable manifest containers into ordinary JSON containers."""
    if isinstance(value, Mapping):
        return {key: _json_tree(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_tree(item) for item in value]
    return value


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_canonical_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write one canonical JSON value followed by a newline."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_canonical_json(dict(report)) + "\n", encoding="utf-8")
    return output_path


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _model_metadata(artifact: EvaluationArtifact, label: str) -> dict[str, Any]:
    model = _require_mapping(artifact.aggregate.get("model"), f"{label} model")
    if not model:
        raise ValueError(f"{label} model metadata must not be empty")
    return _json_tree(model)


def _decoding_protocol(artifact: EvaluationArtifact, label: str) -> dict[str, Any]:
    decoding = _require_mapping(artifact.aggregate.get("decoding"), f"{label} decoding")
    missing = set(_DECODING_PROTOCOL_FIELDS) - set(decoding)
    if missing:
        raise ValueError(f"{label} decoding metadata is missing {sorted(missing)}")
    protocol = _json_tree(decoding)
    if (
        not isinstance(protocol["budget"], int)
        or isinstance(protocol["budget"], bool)
        or protocol["budget"] <= 0
    ):
        raise ValueError(f"{label} decoding budget must be a positive integer")
    if not isinstance(protocol["deterministic"], bool):
        raise ValueError(f"{label} deterministic flag must be boolean")
    if not isinstance(protocol["seed"], int) or isinstance(protocol["seed"], bool):
        raise ValueError(f"{label} decoding seed must be an integer")
    for key in ("requested_batch_size", "max_new_tokens"):
        if (
            not isinstance(protocol[key], int)
            or isinstance(protocol[key], bool)
            or protocol[key] <= 0
        ):
            raise ValueError(f"{label} decoding {key} must be a positive integer")
    if not isinstance(protocol["device"], str) or not protocol["device"]:
        raise ValueError(f"{label} decoding device must be a non-empty string")
    temperature = protocol["temperature"]
    if temperature is not None and (
        not isinstance(temperature, int | float)
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(f"{label} decoding temperature is invalid")
    if protocol["deterministic"] and temperature is not None:
        raise ValueError(f"{label} deterministic decoding must have null temperature")
    if not protocol["deterministic"] and temperature is None:
        raise ValueError(f"{label} sampled decoding requires a temperature")
    return protocol


def _execution_environment(
    artifact: EvaluationArtifact, label: str
) -> dict[str, Any] | None:
    raw = artifact.aggregate.get("execution_environment")
    if raw is None:
        return None
    environment = _require_mapping(raw, f"{label} execution_environment")
    if set(environment) != _EXECUTION_ENVIRONMENT_KEYS:
        raise ValueError(
            f"{label} execution_environment has invalid keys: "
            f"{sorted(set(environment) ^ _EXECUTION_ENVIRONMENT_KEYS)}"
        )
    if environment["schema_version"] != EXECUTION_ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError(f"{label} execution_environment schema is unsupported")
    hardware = _require_mapping(environment["hardware"], f"{label} hardware")
    if not hardware:
        raise ValueError(f"{label} hardware evidence must not be empty")
    fingerprint = environment["hardware_fingerprint"]
    expected_fingerprint = _sha256_bytes(
        _canonical_json(_json_tree(hardware)).encode("utf-8")
    )
    if fingerprint != expected_fingerprint:
        raise ValueError(f"{label} hardware fingerprint does not match its details")
    batch_protocol = _require_mapping(
        environment["batch_protocol"], f"{label} batch_protocol"
    )
    if set(batch_protocol) != _BATCH_PROTOCOL_KEYS:
        raise ValueError(f"{label} batch protocol has invalid keys")
    decoding = _decoding_protocol(artifact, label)
    if environment["requested_device"] != decoding["device"]:
        raise ValueError(f"{label} requested device metadata is inconsistent")
    if batch_protocol.get("requested_batch_size") != decoding["requested_batch_size"]:
        raise ValueError(f"{label} requested batch metadata is inconsistent")
    if (
        batch_protocol.get("solver_inference_batch_size")
        != decoding["requested_batch_size"]
    ):
        raise ValueError(f"{label} effective inference batch size is inconsistent")
    if batch_protocol.get("candidates_per_task") != decoding["budget"]:
        raise ValueError(f"{label} candidate budget metadata is inconsistent")
    for count_key, histogram_key in (
        ("observed_batch_count", "observed_batch_size_histogram"),
        (
            "effective_inference_batch_count",
            "effective_inference_batch_size_histogram",
        ),
    ):
        count = batch_protocol[count_key]
        histogram = _require_mapping(
            batch_protocol[histogram_key], f"{label} {histogram_key}"
        )
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not histogram
            or any(
                not isinstance(key, str)
                or not key.isdigit()
                or int(key) <= 0
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for key, value in histogram.items()
            )
            or sum(histogram.values()) != count
        ):
            raise ValueError(f"{label} {histogram_key} is inconsistent")
        if any(int(size) > decoding["requested_batch_size"] for size in histogram):
            raise ValueError(f"{label} {histogram_key} exceeds requested batch size")
    observed_histogram = batch_protocol["observed_batch_size_histogram"]
    effective_histogram = batch_protocol["effective_inference_batch_size_histogram"]
    if sum(int(size) * count for size, count in observed_histogram.items()) != len(
        artifact.rows
    ):
        raise ValueError(f"{label} observed batches do not cover its result rows")
    if (
        sum(int(size) * count for size, count in effective_histogram.items())
        != len(artifact.rows) * decoding["budget"]
    ):
        raise ValueError(f"{label} inference batches do not cover candidate prompts")
    latency = _require_mapping(artifact.aggregate.get("latency"), f"{label} latency")
    if latency.get("batch_count") != batch_protocol["observed_batch_count"]:
        raise ValueError(f"{label} latency/effective batch counts are inconsistent")
    if (
        not isinstance(environment["resolved_device"], str)
        or not environment["resolved_device"]
    ):
        raise ValueError(f"{label} resolved device must be a non-empty string")
    return _json_tree(environment)


def validate_paired_protocol(
    champion: EvaluationArtifact, challenger: EvaluationArtifact
) -> dict[str, Any]:
    """Require identical evaluation settings except model and basis identity."""
    champion_model = _model_metadata(champion, "champion")
    challenger_model = _model_metadata(challenger, "challenger")
    champion_decoding = _decoding_protocol(champion, "champion")
    challenger_decoding = _decoding_protocol(challenger, "challenger")
    if champion_decoding != challenger_decoding:
        differing = {
            key: {
                "champion": champion_decoding.get(key),
                "challenger": challenger_decoding.get(key),
            }
            for key in sorted(set(champion_decoding) | set(challenger_decoding))
            if champion_decoding.get(key) != challenger_decoding.get(key)
        }
        raise ValueError(f"evaluation protocol mismatch: {differing}")

    champion_environment = _execution_environment(champion, "champion")
    challenger_environment = _execution_environment(challenger, "challenger")
    if champion_environment is not None and challenger_environment is not None:
        if (
            champion_environment["resolved_device"]
            != challenger_environment["resolved_device"]
        ):
            raise ValueError(
                "evaluation protocol mismatch: resolved execution devices differ"
            )
        if (
            champion_environment["batch_protocol"]
            != challenger_environment["batch_protocol"]
        ):
            raise ValueError(
                "evaluation protocol mismatch: effective batch protocols differ"
            )
    return {
        "intended_differences": ["model", "evaluation_basis_set"],
        "models_identical": champion_model == challenger_model,
        "champion_model": champion_model,
        "challenger_model": challenger_model,
        "decoding": champion_decoding,
        "champion_execution_environment": champion_environment,
        "challenger_execution_environment": challenger_environment,
    }


@dataclass(frozen=True)
class EvaluationArtifact:
    """A validated aggregate/result pair from ``basis_library_pilot``."""

    directory: Path
    aggregate: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    file_digests: Mapping[str, str]


def load_evaluation_artifact(path: str | Path) -> EvaluationArtifact:
    """Load an evaluation directory and reject malformed or duplicate rows."""
    directory = Path(path)
    aggregate_path = directory / _RESULT_FILENAMES[0]
    results_path = directory / _RESULT_FILENAMES[1]
    aggregate = _require_mapping(
        json.loads(aggregate_path.read_text(encoding="utf-8")), "aggregate"
    )
    if aggregate.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported evaluation schema version: "
            f"{aggregate.get('schema_version')!r}"
        )

    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    with gzip.open(results_path, "rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            row = _require_mapping(json.loads(line), f"result row {line_number}")
            missing = _REQUIRED_RESULT_KEYS - set(row)
            if missing:
                raise ValueError(
                    f"result row {line_number} is missing keys: {sorted(missing)}"
                )
            task_id = row["task_id"]
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"result row {line_number} has an invalid task_id")
            if task_id in seen:
                raise ValueError(f"duplicate result task_id: {task_id}")
            seen.add(task_id)
            success = row["success"]
            ids = row["solution_function_ids"]
            names = row["solution_function_names"]
            if (
                not isinstance(success, bool)
                or not isinstance(row["split"], str)
                or not row["split"]
                or not isinstance(row["input_type"], str)
                or not row["input_type"]
                or not isinstance(row["witness_length"], int)
                or isinstance(row["witness_length"], bool)
                or row["witness_length"] < 0
                or not isinstance(ids, list)
                or not isinstance(names, list)
                or len(ids) != len(names)
                or any(not isinstance(item, str) or not item for item in ids + names)
            ):
                raise ValueError(f"result row {line_number} has invalid solution data")
            expected_length = len(ids) if success else None
            actual_length = row["solution_length"]
            if (
                actual_length != expected_length
                or isinstance(actual_length, bool)
                or (success and not isinstance(actual_length, int))
                or (not success and ids)
            ):
                raise ValueError(
                    f"result row {line_number} has inconsistent solution_length"
                )
            rows.append(row)

    overall = _require_mapping(aggregate.get("overall"), "aggregate.overall")
    if overall.get("tasks") != len(rows):
        raise ValueError(
            "aggregate/result task count mismatch: "
            f"{overall.get('tasks')!r} != {len(rows)}"
        )
    successes = sum(bool(row["success"]) for row in rows)
    if overall.get("successes") != successes:
        raise ValueError("aggregate/result success count mismatch")
    return EvaluationArtifact(
        directory=directory.resolve(),
        aggregate=aggregate,
        rows=tuple(rows),
        file_digests={
            "aggregate.json": _sha256_file(aggregate_path),
            "results.jsonl.gz": _sha256_file(results_path),
        },
    )


def _basis_function_maps(
    basis: BasisSet,
) -> tuple[dict[str, BasisFunction], dict[str, BasisFunction]]:
    by_id = {function.function_id: function for function in basis.functions}
    by_name = {function.name: function for function in basis.functions}
    if len(by_id) != len(basis.functions) or len(by_name) != len(basis.functions):
        raise ValueError(f"basis {basis.basis_set_id!r} has duplicate IDs or names")
    return by_id, by_name


@dataclass(frozen=True)
class RewriteRule:
    """One exact rewrite justified by target-manifest provenance."""

    rule_id: str
    kind: str
    source_function_ids: tuple[str, ...]
    source_function_names: tuple[str, ...]
    target_function_id: str | None
    target_function_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "source_function_ids": list(self.source_function_ids),
            "source_function_names": list(self.source_function_names),
            "target_function_id": self.target_function_id,
            "target_function_name": self.target_function_name,
        }


def build_rewrite_rules(source: BasisSet, target: BasisSet) -> tuple[RewriteRule, ...]:
    """Compile and validate direct-child deprecations/additions from provenance."""
    if target.parent_basis_set_id != source.basis_set_id:
        raise ValueError(
            f"target basis {target.basis_set_id!r} is not a direct child of "
            f"{source.basis_set_id!r}"
        )
    provenance = target.provenance
    if provenance.get("parent_basis_set_digest") != source.digest:
        raise ValueError("target provenance has the wrong parent basis digest")
    source_by_id, source_by_name = _basis_function_maps(source)
    target_by_id, target_by_name = _basis_function_maps(target)

    raw_deprecations = provenance.get("deprecations")
    raw_additions = provenance.get("additions")
    if not isinstance(raw_deprecations, tuple | list) or not isinstance(
        raw_additions, tuple | list
    ):
        raise ValueError("target provenance must list deprecations and additions")

    rules: list[RewriteRule] = []
    covered_removed: set[str] = set()
    for index, raw in enumerate(raw_deprecations):
        item = _require_mapping(raw, f"deprecation {index}")
        function_id = item.get("function_id")
        function_name = item.get("function_name")
        source_function = source_by_id.get(function_id)
        if source_function is None or source_function.name != function_name:
            raise ValueError(f"deprecation {index} does not identify a source function")
        if function_id in target_by_id:
            raise ValueError(f"deprecated function {function_id!r} remains in target")
        if function_id in covered_removed:
            raise ValueError(f"duplicate deprecation for {function_id!r}")
        covered_removed.add(function_id)
        replacement = item.get("replacement")
        if replacement == "zero-step identity":
            kind = "delete_identity"
            target_function = None
        elif isinstance(replacement, str) and replacement:
            kind = "replace"
            target_function = target_by_name.get(replacement)
            if target_function is None:
                raise ValueError(
                    f"deprecation replacement {replacement!r} is not in target"
                )
        else:
            raise ValueError(f"deprecation {index} has an invalid replacement")
        rules.append(
            RewriteRule(
                rule_id=f"deprecation:{function_id}",
                kind=kind,
                source_function_ids=(function_id,),
                source_function_names=(source_function.name,),
                target_function_id=(
                    target_function.function_id if target_function is not None else None
                ),
                target_function_name=(
                    target_function.name if target_function is not None else None
                ),
            )
        )

    removed_ids = set(source_by_id) - set(target_by_id)
    if covered_removed != removed_ids:
        raise ValueError(
            "target provenance deprecations do not exactly cover removed functions: "
            f"missing={sorted(removed_ids - covered_removed)}, "
            f"extra={sorted(covered_removed - removed_ids)}"
        )

    covered_added: set[str] = set()
    for index, raw in enumerate(raw_additions):
        item = _require_mapping(raw, f"addition {index}")
        function_name = item.get("function_name")
        target_function = target_by_name.get(function_name)
        if target_function is None or target_function.function_id in source_by_id:
            raise ValueError(
                f"addition {index} does not identify a new target function"
            )
        sequence = item.get("source_sequence")
        if (
            not isinstance(sequence, tuple | list)
            or len(sequence) < 2
            or any(not isinstance(name, str) or not name for name in sequence)
        ):
            raise ValueError(f"addition {index} has an invalid source_sequence")
        source_functions: list[BasisFunction] = []
        for name in sequence:
            function = source_by_name.get(name)
            if function is None:
                raise ValueError(
                    f"addition {function_name!r} references unknown source {name!r}"
                )
            source_functions.append(function)
        if target_function.function_id in covered_added:
            raise ValueError(f"duplicate addition for {target_function.function_id!r}")
        covered_added.add(target_function.function_id)
        rules.append(
            RewriteRule(
                rule_id=f"addition:{target_function.function_id}",
                kind="collapse_macro",
                source_function_ids=tuple(
                    function.function_id for function in source_functions
                ),
                source_function_names=tuple(
                    function.name for function in source_functions
                ),
                target_function_id=target_function.function_id,
                target_function_name=target_function.name,
            )
        )

    added_ids = set(target_by_id) - set(source_by_id)
    if covered_added != added_ids:
        raise ValueError(
            "target provenance additions do not exactly cover new functions: "
            f"missing={sorted(added_ids - covered_added)}, "
            f"extra={sorted(covered_added - added_ids)}"
        )
    # Longest macros win; exact rule IDs make equal-length ties deterministic.
    return tuple(
        sorted(
            rules,
            key=lambda rule: (
                0 if rule.kind == "collapse_macro" else 1,
                -len(rule.source_function_ids),
                rule.rule_id,
            ),
        )
    )


def rewrite_solution(
    function_ids: Sequence[str],
    function_names: Sequence[str],
    *,
    source: BasisSet,
    target: BasisSet,
    rules: Sequence[RewriteRule],
) -> tuple[tuple[str, ...], tuple[str, ...], Counter[str]]:
    """Rewrite one source path using longest-match macros then deprecations."""
    if len(function_ids) != len(function_names):
        raise ValueError("source solution IDs and names must align")
    source_by_id, _ = _basis_function_maps(source)
    target_by_id, _ = _basis_function_maps(target)
    for index, (function_id, function_name) in enumerate(
        zip(function_ids, function_names, strict=True)
    ):
        function = source_by_id.get(function_id)
        if function is None or function.name != function_name:
            raise ValueError(
                f"source solution step {index} is not in the source basis: "
                f"{function_id!r}/{function_name!r}"
            )

    macros = [rule for rule in rules if rule.kind == "collapse_macro"]
    deprecations = {
        rule.source_function_ids[0]: rule
        for rule in rules
        if rule.kind != "collapse_macro"
    }
    rewritten_ids: list[str] = []
    rewritten_names: list[str] = []
    counts: Counter[str] = Counter()
    index = 0
    while index < len(function_ids):
        matched = next(
            (
                rule
                for rule in macros
                if tuple(function_ids[index : index + len(rule.source_function_ids)])
                == rule.source_function_ids
            ),
            None,
        )
        if matched is not None:
            assert matched.target_function_id is not None
            assert matched.target_function_name is not None
            rewritten_ids.append(matched.target_function_id)
            rewritten_names.append(matched.target_function_name)
            counts[matched.rule_id] += 1
            index += len(matched.source_function_ids)
            continue

        function_id = function_ids[index]
        rule = deprecations.get(function_id)
        if rule is not None:
            counts[rule.rule_id] += 1
            if rule.target_function_id is not None:
                rewritten_ids.append(rule.target_function_id)
                rewritten_names.append(rule.target_function_name or "")
        else:
            if function_id not in target_by_id:
                raise ValueError(
                    f"source function {function_id!r} has no target or rewrite rule"
                )
            rewritten_ids.append(function_id)
            rewritten_names.append(function_names[index])
        index += 1
    return tuple(rewritten_ids), tuple(rewritten_names), counts


def _runtime_functions_by_id(basis: BasisSet) -> dict[str, FunctionDef]:
    functions = basis.as_function_set().functions
    return {
        str(function.metadata["basis_function_id"]): function for function in functions
    }


def _execute_ids(
    record: BasisTaskRecord,
    function_ids: Sequence[str],
    functions_by_id: Mapping[str, FunctionDef],
    executor: Executor,
):
    functions = []
    for function_id in function_ids:
        function = functions_by_id.get(function_id)
        if function is None:
            raise ValueError(
                f"trajectory references unknown function ID {function_id!r}"
            )
        functions.append(function)
    return executor.execute_trajectory(
        TrajectorySpec(record.input_value, FunctionDefList(functions))
    )


def _empty_rewrite_stats() -> dict[str, int]:
    return {
        "attempted_tasks": 0,
        "verified_tasks": 0,
        "failed_tasks": 0,
        "changed_tasks": 0,
        "source_steps": 0,
        "rewritten_steps": 0,
        "net_step_savings": 0,
    }


def _finish_rewrite_stats(stats: Mapping[str, int]) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = dict(stats)
    attempted = stats["attempted_tasks"]
    result["mean_source_steps"] = (
        stats["source_steps"] / attempted if attempted else None
    )
    result["mean_rewritten_steps"] = (
        stats["rewritten_steps"] / attempted if attempted else None
    )
    result["mean_step_savings"] = (
        stats["net_step_savings"] / attempted if attempted else None
    )
    result["verification_rate"] = (
        stats["verified_tasks"] / attempted if attempted else None
    )
    return result


def analyze_counterfactual(
    *,
    records: Sequence[BasisTaskRecord],
    champion_rows: Sequence[Mapping[str, Any]],
    source: BasisSet,
    target: BasisSet,
) -> dict[str, Any]:
    """Rewrite every champion success and verify source and target execution."""
    rules = build_rewrite_rules(source, target)
    records_by_id = {record.task_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("corpus records contain duplicate task IDs")
    _validate_rows_against_records(champion_rows, records_by_id)

    source_functions = _runtime_functions_by_id(source)
    target_functions = _runtime_functions_by_id(target)
    source_executor = Executor(list(source_functions.values()))
    target_executor = Executor(list(target_functions.values()))
    overall = _empty_rewrite_stats()
    per_split: defaultdict[str, dict[str, int]] = defaultdict(_empty_rewrite_stats)
    per_type: defaultdict[str, dict[str, int]] = defaultdict(_empty_rewrite_stats)
    rule_occurrences: Counter[str] = Counter()
    rule_task_coverage: Counter[str] = Counter()
    failed_task_ids: list[str] = []
    failure_reasons: Counter[str] = Counter()

    for row in champion_rows:
        if not row["success"]:
            continue
        record = records_by_id[str(row["task_id"])]
        source_ids = tuple(row["solution_function_ids"])
        source_names = tuple(row["solution_function_names"])
        source_result = _execute_ids(
            record, source_ids, source_functions, source_executor
        )
        if (
            not source_result.success
            or source_result.trajectory is None
            or source_result.trajectory.output != record.output_value
        ):
            raise ValueError(
                f"champion success {record.task_id} does not reproduce its frozen output"
            )
        rewritten_ids, _, applied = rewrite_solution(
            source_ids,
            source_names,
            source=source,
            target=target,
            rules=rules,
        )
        target_result = _execute_ids(
            record, rewritten_ids, target_functions, target_executor
        )
        verified = (
            target_result.success
            and target_result.trajectory is not None
            and target_result.trajectory.output == record.output_value
        )
        reason = None
        if not target_result.success:
            reason = "execution_failure"
        elif target_result.trajectory is None:
            reason = "missing_trajectory"
        elif target_result.trajectory.output != record.output_value:
            reason = "output_mismatch"

        changed = source_ids != rewritten_ids
        source_steps = len(source_ids)
        rewritten_steps = len(rewritten_ids)
        for stats in (
            overall,
            per_split[record.split],
            per_type[str(record.metadata["input_type"])],
        ):
            stats["attempted_tasks"] += 1
            stats["verified_tasks" if verified else "failed_tasks"] += 1
            stats["changed_tasks"] += int(changed)
            stats["source_steps"] += source_steps
            stats["rewritten_steps"] += rewritten_steps
            stats["net_step_savings"] += source_steps - rewritten_steps
        rule_occurrences.update(applied)
        for rule_id in applied:
            rule_task_coverage[rule_id] += 1
        if not verified:
            failed_task_ids.append(record.task_id)
            failure_reasons[str(reason)] += 1

    rule_rows = []
    for rule in sorted(rules, key=lambda item: item.rule_id):
        rule_rows.append(
            {
                **rule.to_dict(),
                "total_occurrences": rule_occurrences[rule.rule_id],
                "task_coverage": rule_task_coverage[rule.rule_id],
            }
        )
    return {
        "method": (
            "longest provenance-declared macro collapse, then declared "
            "deprecation replacement/deletion; both source and rewritten paths "
            "are executed against frozen outputs"
        ),
        "source_basis_set_id": source.basis_set_id,
        "source_basis_set_digest": source.digest,
        "target_basis_set_id": target.basis_set_id,
        "target_basis_set_digest": target.digest,
        "rules": rule_rows,
        "overall": _finish_rewrite_stats(overall),
        "by_split": {
            key: _finish_rewrite_stats(value)
            for key, value in sorted(per_split.items())
        },
        "by_input_type": {
            key: _finish_rewrite_stats(value) for key, value in sorted(per_type.items())
        },
        "failed_task_ids": sorted(failed_task_ids),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
    }


def _validate_rows_against_records(
    rows: Sequence[Mapping[str, Any]], records_by_id: Mapping[str, BasisTaskRecord]
) -> None:
    row_ids: set[str] = set()
    for row in rows:
        task_id = str(row["task_id"])
        record = records_by_id.get(task_id)
        if record is None:
            raise ValueError(f"evaluation references task outside corpus: {task_id}")
        if task_id in row_ids:
            raise ValueError(f"evaluation contains duplicate task ID: {task_id}")
        row_ids.add(task_id)
        if row["split"] != record.split:
            raise ValueError(f"evaluation/corpus split mismatch for {task_id}")
        if row["input_type"] != record.metadata["input_type"]:
            raise ValueError(f"evaluation/corpus input_type mismatch for {task_id}")
        if row["witness_length"] != record.witness_length:
            raise ValueError(f"evaluation/corpus witness_length mismatch for {task_id}")


def _validate_solution_basis(
    rows: Sequence[Mapping[str, Any]], basis: BasisSet, label: str
) -> None:
    by_id, _ = _basis_function_maps(basis)
    for row in rows:
        for index, (function_id, function_name) in enumerate(
            zip(
                row["solution_function_ids"],
                row["solution_function_names"],
                strict=True,
            )
        ):
            function = by_id.get(function_id)
            if function is None or function.name != function_name:
                raise ValueError(
                    f"{label} solution {row['task_id']} step {index} is not in "
                    f"basis {basis.basis_set_id}: {function_id!r}/{function_name!r}"
                )


def _verify_successful_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[BasisTaskRecord],
    basis: BasisSet,
    label: str,
) -> int:
    """Independently execute each recorded success against its frozen output."""
    records_by_id = {record.task_id: record for record in records}
    functions = _runtime_functions_by_id(basis)
    executor = Executor(list(functions.values()))
    verified = 0
    for row in rows:
        if not row["success"]:
            continue
        record = records_by_id[str(row["task_id"])]
        result = _execute_ids(
            record, tuple(row["solution_function_ids"]), functions, executor
        )
        if (
            not result.success
            or result.trajectory is None
            or result.trajectory.output != record.output_value
        ):
            raise ValueError(
                f"{label} success {record.task_id} does not reproduce frozen output"
            )
        verified += 1
    return verified


class _CostVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter({"dispatch": 1})

    def visit_Call(self, node: ast.Call) -> None:
        self.counts["call"] += 1
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.counts["binary_operation"] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.counts["boolean_link"] += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.counts["comparison"] += len(node.ops)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.counts["unary_operation"] += 1
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.counts["subscript"] += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.counts["branch"] += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.counts["branch"] += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.counts["comprehension_generator"] += 1
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.counts["attribute_lookup"] += 1
        self.generic_visit(node)


def static_function_cost(function: BasisFunction) -> tuple[float, dict[str, int]]:
    """Return a transparent static operation-count proxy for one invocation."""
    wrapped = "def _basis_cost_probe(x):\n" + "\n".join(
        f"    {line}" for line in function.code.splitlines()
    )
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as error:
        raise ValueError(
            f"cannot parse code for cost proxy: {function.name}"
        ) from error
    visitor = _CostVisitor()
    visitor.visit(tree)
    counts = {key: visitor.counts[key] for key in _STATIC_COST_WEIGHTS}
    cost = sum(counts[key] * weight for key, weight in _STATIC_COST_WEIGHTS.items())
    return cost, counts


def load_explicit_costs(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = _require_mapping(
        json.loads(Path(path).read_text(encoding="utf-8")), "cost table"
    )
    if set(payload) != {"schema_version", "costs"}:
        raise ValueError("cost table must contain exactly schema_version and costs")
    if payload["schema_version"] != EXPLICIT_COST_SCHEMA_VERSION:
        raise ValueError("unsupported explicit cost-table schema version")
    raw_costs = _require_mapping(payload["costs"], "cost table costs")
    result: dict[str, float] = {}
    for function_id, raw_cost in raw_costs.items():
        if (
            not isinstance(function_id, str)
            or not function_id
            or not isinstance(raw_cost, int | float)
            or isinstance(raw_cost, bool)
            or not math.isfinite(raw_cost)
            or raw_cost <= 0
        ):
            raise ValueError(f"invalid explicit cost: {function_id!r}={raw_cost!r}")
        result[function_id] = float(raw_cost)
    return result


def build_execution_cost_table(
    source: BasisSet,
    target: BasisSet,
    explicit_costs: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build costs for the union, with optional per-ID measured overrides."""
    explicit = dict(explicit_costs or {})
    union = {function.function_id: function for function in source.functions}
    union.update({function.function_id: function for function in target.functions})
    unknown = set(explicit) - set(union)
    if unknown:
        raise ValueError(
            f"explicit costs reference unknown function IDs: {sorted(unknown)}"
        )
    functions = []
    for function_id, function in sorted(union.items()):
        derived, counts = static_function_cost(function)
        override = explicit.get(function_id)
        functions.append(
            {
                "function_id": function_id,
                "function_name": function.name,
                "cost": override if override is not None else derived,
                "source": "explicit" if override is not None else "static_proxy",
                "static_proxy_cost": derived,
                "static_operation_counts": counts,
            }
        )
    return {
        "method": (
            "per-element static AST proxy; path analysis multiplies it by the "
            "frozen TypedList item count; container-value complexity is not "
            "modeled; explicit per-function overrides supersede the proxy"
        ),
        "path_cost_formula": (
            "input_item_count * sum(function_cost for function in path)"
        ),
        "weights": dict(_STATIC_COST_WEIGHTS),
        "explicit_override_count": len(explicit),
        "functions": functions,
    }


def _basis_mdl_summary(basis: BasisSet) -> dict[str, Any]:
    function_payloads = [
        {
            "function_id": function.function_id,
            "fingerprint": function.fingerprint,
            "name": function.name,
            "input_type": function.input_type,
            "output_type": function.output_type,
            "code": function.code,
            "metadata": _json_tree(function.metadata),
        }
        for function in basis.functions
    ]
    manifest_payload = {
        "schema_version": basis.schema_version,
        "basis_set_id": basis.basis_set_id,
        "digest": basis.digest,
        "description": basis.description,
        "parent_basis_set_id": basis.parent_basis_set_id,
        "provenance": _json_tree(basis.provenance),
        "functions": function_payloads,
    }
    return {
        "basis_set_id": basis.basis_set_id,
        "function_count": len(basis.functions),
        "function_code_utf8_bytes": sum(
            len(function.code.encode("utf-8")) for function in basis.functions
        ),
        "canonical_function_definitions_utf8_bytes": len(
            _canonical_json(function_payloads).encode("utf-8")
        ),
        "canonical_manifest_utf8_bytes": len(
            _canonical_json(manifest_payload).encode("utf-8")
        ),
    }


def library_mdl_comparison(source: BasisSet, target: BasisSet) -> dict[str, Any]:
    """Report library description size separately from path execution cost."""
    source_row = _basis_mdl_summary(source)
    target_row = _basis_mdl_summary(target)
    return {
        "method": (
            "descriptive library-size telemetry only; these values are not added "
            "to per-task execution cost"
        ),
        "source": source_row,
        "target": target_row,
        "target_minus_source": {
            key: target_row[key] - source_row[key]
            for key in (
                "function_count",
                "function_code_utf8_bytes",
                "canonical_function_definitions_utf8_bytes",
                "canonical_manifest_utf8_bytes",
            )
        },
    }


def mcnemar_exact(champion_only: int, challenger_only: int) -> dict[str, Any]:
    """Two-sided exact McNemar/binomial test over discordant pairs."""
    if champion_only < 0 or challenger_only < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = champion_only + challenger_only
    if discordant == 0:
        return {
            "method": "two-sided exact binomial McNemar test",
            "discordant_pairs": 0,
            "p_value": 1.0,
            "log10_p_value": 0.0,
        }
    tail = min(champion_only, challenger_only)
    coefficient = 1
    cumulative = 1
    for index in range(1, tail + 1):
        coefficient = coefficient * (discordant - index + 1) // index
        cumulative += coefficient
    numerator = min(2 * cumulative, 1 << discordant)
    denominator = 1 << discordant
    p_value = numerator / denominator
    log10_p = (
        0.0
        if numerator == denominator
        else (math.log(numerator) - discordant * math.log(2)) / math.log(10)
    )
    return {
        "method": "two-sided exact binomial McNemar test",
        "discordant_pairs": discordant,
        "p_value": p_value,
        "log10_p_value": log10_p,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_mean_ci(
    paired_differences: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any] | None:
    """Percentile CI from deterministic paired task resampling."""
    if not paired_differences:
        return None
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap samples/confidence are invalid")
    import numpy as np

    values = np.asarray(paired_differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means: list[float] = []
    remaining = samples
    while remaining:
        count = min(64, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.extend(np.mean(values[indices], axis=1).tolist())
        remaining -= count
    alpha = (1 - confidence) / 2
    return {
        "point_estimate": statistics.fmean(
            float(value) for value in paired_differences
        ),
        "confidence": confidence,
        "lower": _percentile(means, alpha),
        "upper": _percentile(means, 1 - alpha),
        "samples": samples,
        "seed": seed,
        "resampling_unit": "paired task",
        "rng": "numpy.default_rng(PCG64)",
    }


def _sequence_cost(
    row: Mapping[str, Any], costs: Mapping[str, float], *, item_count: int
) -> float:
    total = 0.0
    for function_id in row["solution_function_ids"]:
        try:
            total += costs[function_id]
        except KeyError as error:
            raise ValueError(
                f"solution references function with no cost: {function_id!r}"
            ) from error
    return total * item_count


def _mean_comparison(
    champion: Sequence[float], challenger: Sequence[float]
) -> dict[str, Any]:
    if not champion:
        return {
            "tasks": 0,
            "champion_mean": None,
            "challenger_mean": None,
            "challenger_minus_champion": None,
            "challenger_to_champion_ratio": None,
        }
    champion_mean = statistics.fmean(champion)
    challenger_mean = statistics.fmean(challenger)
    return {
        "tasks": len(champion),
        "champion_mean": champion_mean,
        "challenger_mean": challenger_mean,
        "challenger_minus_champion": challenger_mean - champion_mean,
        "challenger_to_champion_ratio": (
            challenger_mean / champion_mean if champion_mean else None
        ),
    }


def _paired_outcomes(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, int | float]:
    both_success = sum(left["success"] and right["success"] for left, right in pairs)
    champion_only = sum(
        left["success"] and not right["success"] for left, right in pairs
    )
    challenger_only = sum(
        not left["success"] and right["success"] for left, right in pairs
    )
    both_fail = len(pairs) - both_success - champion_only - challenger_only
    champion_successes = both_success + champion_only
    challenger_successes = both_success + challenger_only
    return {
        "tasks": len(pairs),
        "both_success": both_success,
        "champion_only": champion_only,
        "challenger_only": challenger_only,
        "both_fail": both_fail,
        "champion_successes": champion_successes,
        "challenger_successes": challenger_successes,
        "champion_solve_rate": champion_successes / len(pairs) if pairs else 0.0,
        "challenger_solve_rate": challenger_successes / len(pairs) if pairs else 0.0,
        "solve_rate_delta_challenger_minus_champion": (
            (challenger_successes - champion_successes) / len(pairs) if pairs else 0.0
        ),
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if (
        not isinstance(numerator, int | float)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int | float)
        or isinstance(denominator, bool)
        or denominator == 0
    ):
        return None
    return numerator / denominator


def _latency_metadata(artifact: EvaluationArtifact, label: str) -> dict[str, Any]:
    latency = _require_mapping(
        artifact.aggregate.get("latency"), f"{label} aggregate latency"
    )
    for key in ("wall_seconds", "tasks_per_second", "mean_ms_per_task_by_batch"):
        value = latency.get(key)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{label} latency {key} must be finite and positive")
    return _json_tree(latency)


def _latency_comparability(protocol: Mapping[str, Any]) -> dict[str, Any]:
    champion = protocol["champion_execution_environment"]
    challenger = protocol["challenger_execution_environment"]
    if champion is None or challenger is None:
        missing = []
        if champion is None:
            missing.append("champion")
        if challenger is None:
            missing.append("challenger")
        return {
            "comparable": False,
            "reason": "missing_execution_environment",
            "missing": missing,
        }
    if champion["hardware_fingerprint"] != challenger["hardware_fingerprint"]:
        return {
            "comparable": False,
            "reason": "hardware_fingerprint_mismatch",
            "champion_hardware_fingerprint": champion["hardware_fingerprint"],
            "challenger_hardware_fingerprint": challenger["hardware_fingerprint"],
        }
    return {
        "comparable": True,
        "reason": None,
        "hardware_fingerprint": champion["hardware_fingerprint"],
        "resolved_device": champion["resolved_device"],
        "batch_protocol": champion["batch_protocol"],
    }


def analyze_paired_evaluations(
    *,
    records: Sequence[BasisTaskRecord],
    champion: EvaluationArtifact,
    challenger: EvaluationArtifact,
    cost_table: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Join exact task IDs and compute paired outcome, cost, and compute metrics."""
    protocol = validate_paired_protocol(champion, challenger)
    records_by_id = {record.task_id: record for record in records}
    _validate_rows_against_records(champion.rows, records_by_id)
    _validate_rows_against_records(challenger.rows, records_by_id)
    champion_by_id = {str(row["task_id"]): row for row in champion.rows}
    challenger_by_id = {str(row["task_id"]): row for row in challenger.rows}
    if set(champion_by_id) != set(challenger_by_id):
        raise ValueError(
            "paired evaluations must contain exactly the same task IDs: "
            f"champion_only={len(set(champion_by_id) - set(challenger_by_id))}, "
            f"challenger_only={len(set(challenger_by_id) - set(champion_by_id))}"
        )
    ordered_ids = sorted(champion_by_id)
    pairs = [(champion_by_id[key], challenger_by_id[key]) for key in ordered_ids]
    outcomes = _paired_outcomes(pairs)
    by_split: defaultdict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    by_type: defaultdict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for task_id, pair in zip(ordered_ids, pairs, strict=True):
        record = records_by_id[task_id]
        by_split[record.split].append(pair)
        by_type[str(record.metadata["input_type"])].append(pair)

    costs = {
        str(row["function_id"]): float(row["cost"]) for row in cost_table["functions"]
    }
    common = [
        (task_id, left, right)
        for task_id, (left, right) in zip(ordered_ids, pairs, strict=True)
        if left["success"] and right["success"]
    ]
    champion_raw = [float(row[1]["solution_length"]) for row in common]
    challenger_raw = [float(row[2]["solution_length"]) for row in common]
    champion_weighted = [
        _sequence_cost(row[1], costs, item_count=len(records_by_id[row[0]].input_value))
        for row in common
    ]
    challenger_weighted = [
        _sequence_cost(row[2], costs, item_count=len(records_by_id[row[0]].input_value))
        for row in common
    ]

    raw_differences = [
        right - left for left, right in zip(champion_raw, challenger_raw, strict=True)
    ]
    weighted_differences = [
        right - left
        for left, right in zip(champion_weighted, challenger_weighted, strict=True)
    ]
    solve_differences = [
        float(right["success"]) - float(left["success"]) for left, right in pairs
    ]
    champion_latency = _latency_metadata(champion, "champion")
    challenger_latency = _latency_metadata(challenger, "challenger")
    latency_comparability = _latency_comparability(protocol)
    ratios = (
        {
            "challenger_to_champion_wall_seconds": _safe_ratio(
                challenger_latency.get("wall_seconds"),
                champion_latency.get("wall_seconds"),
            ),
            "challenger_to_champion_mean_ms_per_task_by_batch": _safe_ratio(
                challenger_latency.get("mean_ms_per_task_by_batch"),
                champion_latency.get("mean_ms_per_task_by_batch"),
            ),
            "challenger_to_champion_throughput": _safe_ratio(
                challenger_latency.get("tasks_per_second"),
                champion_latency.get("tasks_per_second"),
            ),
        }
        if latency_comparability["comparable"]
        else None
    )
    return {
        "join_key": "exact task_id",
        "evaluation_protocol": protocol,
        "outcomes": outcomes,
        "by_split": {
            key: _paired_outcomes(value) for key, value in sorted(by_split.items())
        },
        "by_input_type": {
            key: _paired_outcomes(value) for key, value in sorted(by_type.items())
        },
        "mcnemar_exact": mcnemar_exact(
            int(outcomes["champion_only"]), int(outcomes["challenger_only"])
        ),
        "common_success_cost": {
            "population": "tasks solved by both actual evaluations",
            "raw_path_length": _mean_comparison(champion_raw, challenger_raw),
            "execution_weighted_path_length": _mean_comparison(
                champion_weighted, challenger_weighted
            ),
        },
        "actual_compute": {
            "champion": dict(champion_latency),
            "challenger": dict(challenger_latency),
            "comparability": latency_comparability,
            "ratios": ratios,
            "note": (
                "wall-clock/throughput are measured evaluation aggregates; they "
                "are not mixed into the static execution-weighted path proxy"
            ),
        },
        "paired_bootstrap": {
            "solve_rate_delta": paired_bootstrap_mean_ci(
                solve_differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "common_success_raw_path_delta": paired_bootstrap_mean_ci(
                raw_differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 1,
            ),
            "common_success_execution_weighted_path_delta": (
                paired_bootstrap_mean_ci(
                    weighted_differences,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 2,
                )
            ),
        },
    }


def _validate_evaluation_provenance(
    *,
    artifact: EvaluationArtifact,
    corpus_manifest: Mapping[str, Any],
    source: BasisSet,
    evaluation: BasisSet,
    label: str,
) -> None:
    aggregate = artifact.aggregate
    expected = {
        "task_source_basis_set_id": source.basis_set_id,
        "task_source_basis_set_digest": source.digest,
        "evaluation_basis_set_id": evaluation.basis_set_id,
        "evaluation_basis_set_digest": evaluation.digest,
        "corpus_manifest_digest": corpus_manifest["manifest_digest"],
    }
    mismatches = {
        key: {"expected": value, "actual": aggregate.get(key)}
        for key, value in expected.items()
        if aggregate.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} evaluation provenance mismatch: {mismatches}")
    _validate_solution_basis(artifact.rows, evaluation, label)


def analyze_pilot(
    *,
    corpus_dir: str | Path,
    champion_eval_dir: str | Path,
    source_basis_set_id: str,
    target_basis_set_id: str,
    challenger_eval_dir: str | Path | None = None,
    explicit_cost_path: str | Path | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Create the complete deterministic offline analysis artifact."""
    champion = load_evaluation_artifact(champion_eval_dir)
    challenger = (
        load_evaluation_artifact(challenger_eval_dir)
        if challenger_eval_dir is not None
        else None
    )
    source = load_basis_set(source_basis_set_id)
    target = load_basis_set(target_basis_set_id)
    splits = champion.aggregate.get("evaluated_splits")
    if (
        not isinstance(splits, list)
        or not splits
        or any(not isinstance(split, str) or not split for split in splits)
    ):
        raise ValueError("champion aggregate has invalid evaluated_splits")
    if (
        challenger is not None
        and challenger.aggregate.get("evaluated_splits") != splits
    ):
        raise ValueError("champion/challenger evaluated_splits differ")
    corpus_manifest, records = load_corpus(corpus_dir, splits=tuple(splits))
    if corpus_manifest["basis_set_id"] != source.basis_set_id:
        raise ValueError("requested source basis does not match corpus")
    _validate_evaluation_provenance(
        artifact=champion,
        corpus_manifest=corpus_manifest,
        source=source,
        evaluation=source,
        label="champion",
    )
    if challenger is not None:
        _validate_evaluation_provenance(
            artifact=challenger,
            corpus_manifest=corpus_manifest,
            source=source,
            evaluation=target,
            label="challenger",
        )
    records_by_id = {record.task_id: record for record in records}
    _validate_rows_against_records(champion.rows, records_by_id)
    if set(records_by_id) != {str(row["task_id"]) for row in champion.rows}:
        raise ValueError(
            "champion evaluation does not cover every selected corpus task"
        )

    explicit_costs = load_explicit_costs(explicit_cost_path)
    cost_table = build_execution_cost_table(source, target, explicit_costs)
    corpus_manifest_path = Path(corpus_dir) / "manifest.json"
    champion_model = _model_metadata(champion, "champion")
    champion_decoding = _decoding_protocol(champion, "champion")
    champion_environment = _execution_environment(champion, "champion")
    challenger_model = (
        _model_metadata(challenger, "challenger") if challenger is not None else None
    )
    challenger_decoding = (
        _decoding_protocol(challenger, "challenger") if challenger is not None else None
    )
    challenger_environment = (
        _execution_environment(challenger, "challenger")
        if challenger is not None
        else None
    )
    inputs: dict[str, Any] = {
        "corpus": {
            "directory": str(Path(corpus_dir).resolve()),
            "manifest_digest": corpus_manifest["manifest_digest"],
            "manifest_file_sha256": _sha256_file(corpus_manifest_path),
            "evaluated_splits": list(splits),
            "tasks": len(records),
            "split_files": {
                split: {
                    "path": str(
                        (
                            Path(corpus_dir) / corpus_manifest["splits"][split]["path"]
                        ).resolve()
                    ),
                    "sha256": corpus_manifest["splits"][split]["sha256"],
                }
                for split in splits
            },
        },
        "champion_evaluation": {
            "directory": str(champion.directory),
            "files": dict(champion.file_digests),
            "model": champion_model,
            "decoding": champion_decoding,
            "execution_environment": champion_environment,
        },
        "challenger_evaluation": (
            {
                "directory": str(challenger.directory),
                "files": dict(challenger.file_digests),
                "model": challenger_model,
                "decoding": challenger_decoding,
                "execution_environment": challenger_environment,
            }
            if challenger is not None
            else None
        ),
        "explicit_cost_table": (
            {
                "path": str(Path(explicit_cost_path).resolve()),
                "sha256": _sha256_file(Path(explicit_cost_path)),
            }
            if explicit_cost_path is not None
            else None
        ),
    }
    counterfactual = analyze_counterfactual(
        records=records,
        champion_rows=champion.rows,
        source=source,
        target=target,
    )
    verified_challenger_successes = (
        _verify_successful_rows(challenger.rows, records, target, "challenger")
        if challenger is not None
        else None
    )
    report: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "inputs": inputs,
        "basis_sets": {
            "source": source.identity_dict(),
            "target": target.identity_dict(),
        },
        "counterfactual_rewrite": counterfactual,
        "execution_cost_table": cost_table,
        "library_mdl": library_mdl_comparison(source, target),
        "paired_actual_evaluations": (
            {
                "independently_verified_challenger_successes": (
                    verified_challenger_successes
                ),
                **analyze_paired_evaluations(
                    records=records,
                    champion=champion,
                    challenger=challenger,
                    cost_table=cost_table,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
            if challenger is not None
            else None
        ),
    }
    report["analysis_digest"] = _sha256_bytes(_canonical_json(report).encode("utf-8"))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--champion-eval-dir", required=True)
    parser.add_argument("--challenger-eval-dir")
    parser.add_argument("--source-basis-set-id", required=True)
    parser.add_argument("--target-basis-set-id", required=True)
    parser.add_argument("--explicit-cost-table")
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_pilot(
        corpus_dir=args.corpus_dir,
        champion_eval_dir=args.champion_eval_dir,
        challenger_eval_dir=args.challenger_eval_dir,
        source_basis_set_id=args.source_basis_set_id,
        target_basis_set_id=args.target_basis_set_id,
        explicit_cost_path=args.explicit_cost_table,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_path = write_canonical_report(report, args.output)
    print(f"wrote {output_path} ({report['analysis_digest']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
