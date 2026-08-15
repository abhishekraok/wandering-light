import gzip
import hashlib
import json

import pytest

from experiments.analyze_basis_pilot import (
    EvaluationArtifact,
    analyze_counterfactual,
    analyze_paired_evaluations,
    build_execution_cost_table,
    build_rewrite_rules,
    library_mdl_comparison,
    load_evaluation_artifact,
    mcnemar_exact,
    paired_bootstrap_mean_ci,
    rewrite_solution,
    validate_paired_protocol,
    write_canonical_report,
)
from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.basis_set import load_basis_set
from wandering_light.typed_list import TypedList


@pytest.fixture
def source_basis():
    return load_basis_set("wl-core-v1")


@pytest.fixture
def target_basis():
    return load_basis_set("wl-pilot-compressed-v1")


def _basis_function(basis, name):
    return next(function for function in basis.functions if function.name == name)


def _record(*, source_basis, split, input_value, output_value, functions):
    return BasisTaskRecord.create(
        split=split,
        input_value=input_value,
        output_value=output_value,
        witness_function_ids=[function.function_id for function in functions],
        witness_function_names=[function.name for function in functions],
        basis_set_id=source_basis.basis_set_id,
        basis_set_digest=source_basis.digest,
        generator="test",
        seed=1,
        source_index=0,
        metadata={
            "input_type": (
                f"{input_value.item_type.__module__}."
                f"{input_value.item_type.__qualname__}"
            ),
            "requested_witness_length": len(functions),
        },
    )


def _result_row(record, functions, *, success=True):
    return {
        "task_id": record.task_id,
        "split": record.split,
        "input_type": record.metadata["input_type"],
        "witness_length": record.witness_length,
        "success": success,
        "solution_length": len(functions) if success else None,
        "solution_function_ids": (
            [function.function_id for function in functions] if success else []
        ),
        "solution_function_names": (
            [function.name for function in functions] if success else []
        ),
    }


def test_provenance_rewrites_are_executed_and_verified(source_basis, target_basis):
    bytearray_to_bytes = _basis_function(source_basis, "bytearray_to_bytes")
    bytes_is_empty = _basis_function(source_basis, "bytes_is_empty")
    duplicate = _basis_function(source_basis, "duplicate")
    identity_int = _basis_function(source_basis, "identity_int")

    bytearrays = _record(
        source_basis=source_basis,
        split="discovery",
        input_value=TypedList([bytearray(), bytearray(b"a")], item_type=bytearray),
        output_value=TypedList([True, False], item_type=bool),
        functions=[bytearray_to_bytes, bytes_is_empty],
    )
    strings = _record(
        source_basis=source_basis,
        split="validation",
        input_value=TypedList(["a", "bc"], item_type=str),
        output_value=TypedList(["aa", "bcbc"], item_type=str),
        functions=[duplicate],
    )
    integers = _record(
        source_basis=source_basis,
        split="validation",
        input_value=TypedList([1, -2], item_type=int),
        output_value=TypedList([1, -2], item_type=int),
        functions=[identity_int],
    )
    records = [bytearrays, strings, integers]
    rows = [
        _result_row(bytearrays, [bytearray_to_bytes, bytes_is_empty]),
        _result_row(strings, [duplicate]),
        _result_row(integers, [identity_int]),
    ]

    result = analyze_counterfactual(
        records=records,
        champion_rows=rows,
        source=source_basis,
        target=target_basis,
    )

    assert result["overall"] == {
        "attempted_tasks": 3,
        "verified_tasks": 3,
        "failed_tasks": 0,
        "changed_tasks": 3,
        "source_steps": 4,
        "rewritten_steps": 2,
        "net_step_savings": 2,
        "mean_source_steps": 4 / 3,
        "mean_rewritten_steps": 2 / 3,
        "mean_step_savings": 2 / 3,
        "verification_rate": 1.0,
    }
    assert result["by_split"]["validation"]["net_step_savings"] == 1
    assert result["by_input_type"]["builtins.bytearray"]["verified_tasks"] == 1
    used_rules = {
        row["kind"]: row["total_occurrences"]
        for row in result["rules"]
        if row["total_occurrences"]
    }
    assert used_rules == {"collapse_macro": 1, "replace": 1, "delete_identity": 1}
    assert result["failed_task_ids"] == []


def test_rewrite_rejects_solution_name_id_mismatch(source_basis, target_basis):
    duplicate = _basis_function(source_basis, "duplicate")
    with pytest.raises(ValueError, match="not in the source basis"):
        rewrite_solution(
            [duplicate.function_id],
            ["repeat"],
            source=source_basis,
            target=target_basis,
            rules=build_rewrite_rules(source_basis, target_basis),
        )


def test_cost_proxy_override_and_library_size_stay_separate(source_basis, target_basis):
    repeat = _basis_function(source_basis, "repeat")
    table = build_execution_cost_table(
        source_basis, target_basis, {repeat.function_id: 9.5}
    )
    repeat_row = next(
        row for row in table["functions"] if row["function_id"] == repeat.function_id
    )
    assert repeat_row["source"] == "explicit"
    assert repeat_row["cost"] == 9.5
    assert repeat_row["static_operation_counts"]["dispatch"] == 1
    assert repeat_row["static_operation_counts"]["binary_operation"] == 1

    mdl = library_mdl_comparison(source_basis, target_basis)
    assert mdl["source"]["function_count"] == 118
    assert mdl["target"]["function_count"] == 116
    assert mdl["target_minus_source"]["function_count"] == -2
    assert "not added" in mdl["method"]


def _paired_records_and_rows(source_basis, target_basis):
    inc_source = _basis_function(source_basis, "inc")
    inc_target = _basis_function(target_basis, "inc")
    records = []
    for index in range(4):
        records.append(
            _record(
                source_basis=source_basis,
                split="test",
                input_value=TypedList([index], item_type=int),
                output_value=TypedList([index], item_type=int),
                functions=[],
            )
        )
    champion_rows = [
        _result_row(records[0], [inc_source]),
        _result_row(records[1], [inc_source]),
        _result_row(records[2], [], success=False),
        _result_row(records[3], [], success=False),
    ]
    challenger_rows = [
        _result_row(records[0], [inc_target, inc_target]),
        _result_row(records[1], [], success=False),
        _result_row(records[2], [inc_target]),
        _result_row(records[3], [], success=False),
    ]
    return records, champion_rows, challenger_rows


def _test_execution_environment(
    *, hardware_name="test-hardware", decoding=None, corrupt_fingerprint=False
):
    decoding = decoding or {
        "budget": 1,
        "deterministic": True,
        "seed": 7,
        "requested_batch_size": 4,
        "device": "cpu",
        "max_new_tokens": 32,
        "temperature": None,
    }
    hardware = {"test_hardware": hardware_name}
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                hardware,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    if corrupt_fingerprint:
        fingerprint = "sha256:" + "0" * 64
    return {
        "schema_version": 1,
        "requested_device": decoding["device"],
        "resolved_device": decoding["device"],
        "hardware_fingerprint": fingerprint,
        "hardware": hardware,
        "batch_protocol": {
            "requested_batch_size": decoding["requested_batch_size"],
            "solver_inference_batch_size": decoding["requested_batch_size"],
            "candidates_per_task": decoding["budget"],
            "observed_batch_count": 1,
            "observed_batch_size_histogram": {"4": 1},
            "effective_inference_batch_count": 1,
            "effective_inference_batch_size_histogram": {"4": 1},
        },
    }


def _artifact(
    tmp_path,
    name,
    rows,
    *,
    wall,
    throughput,
    decoding=None,
    include_environment=True,
    hardware_name="test-hardware",
    corrupt_fingerprint=False,
):
    decoding = decoding or {
        "budget": 1,
        "deterministic": True,
        "seed": 7,
        "requested_batch_size": 4,
        "device": "cpu",
        "max_new_tokens": 32,
        "temperature": None,
    }
    return EvaluationArtifact(
        directory=tmp_path / name,
        aggregate={
            "model": {"resolved": name, "digest": f"sha256:{name}"},
            "decoding": decoding,
            "execution_environment": (
                _test_execution_environment(
                    hardware_name=hardware_name,
                    decoding=decoding,
                    corrupt_fingerprint=corrupt_fingerprint,
                )
                if include_environment
                else None
            ),
            "latency": {
                "wall_seconds": wall,
                "tasks_per_second": throughput,
                "mean_ms_per_task_by_batch": 4 * wall / len(rows),
                "batch_count": 1,
            },
        },
        rows=tuple(rows),
        file_digests={},
    )


def test_paired_actual_statistics_cost_and_bootstrap(
    tmp_path, source_basis, target_basis
):
    records, champion_rows, challenger_rows = _paired_records_and_rows(
        source_basis, target_basis
    )
    cost_table = build_execution_cost_table(source_basis, target_basis)
    result = analyze_paired_evaluations(
        records=records,
        champion=_artifact(
            tmp_path, "champion", champion_rows, wall=2.0, throughput=2.0
        ),
        challenger=_artifact(
            tmp_path, "challenger", challenger_rows, wall=4.0, throughput=1.0
        ),
        cost_table=cost_table,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert result["outcomes"] == {
        "tasks": 4,
        "both_success": 1,
        "champion_only": 1,
        "challenger_only": 1,
        "both_fail": 1,
        "champion_successes": 2,
        "challenger_successes": 2,
        "champion_solve_rate": 0.5,
        "challenger_solve_rate": 0.5,
        "solve_rate_delta_challenger_minus_champion": 0.0,
    }
    assert result["mcnemar_exact"]["p_value"] == 1.0
    assert result["evaluation_protocol"]["models_identical"] is False
    assert result["evaluation_protocol"]["decoding"]["budget"] == 1
    assert (
        result["common_success_cost"]["raw_path_length"]["challenger_minus_champion"]
        == 1.0
    )
    assert (
        result["common_success_cost"]["execution_weighted_path_length"][
            "challenger_minus_champion"
        ]
        == 2.0
    )
    assert (
        result["actual_compute"]["ratios"]["challenger_to_champion_wall_seconds"] == 2.0
    )
    raw_ci = result["paired_bootstrap"]["common_success_raw_path_delta"]
    assert raw_ci["point_estimate"] == raw_ci["lower"] == raw_ci["upper"] == 1.0


@pytest.mark.parametrize(
    ("champion_overrides", "challenger_overrides"),
    [
        ({}, {"budget": 2}),
        ({}, {"seed": 8}),
        ({}, {"requested_batch_size": 2}),
        ({}, {"device": "cuda:0"}),
        ({}, {"max_new_tokens": 64}),
        (
            {"deterministic": False, "temperature": 0.5},
            {"deterministic": False, "temperature": 0.7},
        ),
        ({}, {"deterministic": False, "temperature": 0.5}),
    ],
)
def test_paired_protocol_rejects_every_mismatch(
    tmp_path,
    source_basis,
    target_basis,
    champion_overrides,
    challenger_overrides,
):
    _, champion_rows, challenger_rows = _paired_records_and_rows(
        source_basis, target_basis
    )
    base = {
        "budget": 1,
        "deterministic": True,
        "seed": 7,
        "requested_batch_size": 4,
        "device": "cpu",
        "max_new_tokens": 32,
        "temperature": None,
    }
    champion_decoding = {**base, **champion_overrides}
    challenger_decoding = {**base, **challenger_overrides}
    champion = _artifact(
        tmp_path,
        "champion",
        champion_rows,
        wall=2.0,
        throughput=2.0,
        decoding=champion_decoding,
    )
    challenger = _artifact(
        tmp_path,
        "challenger",
        challenger_rows,
        wall=2.0,
        throughput=2.0,
        decoding=challenger_decoding,
    )
    with pytest.raises(ValueError, match="evaluation protocol mismatch"):
        validate_paired_protocol(champion, challenger)


def test_compute_ratios_require_matching_hardware_evidence(
    tmp_path, source_basis, target_basis
):
    records, champion_rows, challenger_rows = _paired_records_and_rows(
        source_basis, target_basis
    )
    cost_table = build_execution_cost_table(source_basis, target_basis)
    champion = _artifact(
        tmp_path,
        "champion",
        champion_rows,
        wall=2.0,
        throughput=2.0,
        include_environment=False,
    )
    challenger = _artifact(
        tmp_path,
        "challenger",
        challenger_rows,
        wall=4.0,
        throughput=1.0,
    )
    result = analyze_paired_evaluations(
        records=records,
        champion=champion,
        challenger=challenger,
        cost_table=cost_table,
        bootstrap_samples=10,
        bootstrap_seed=1,
    )
    assert result["actual_compute"]["ratios"] is None
    assert result["actual_compute"]["comparability"] == {
        "comparable": False,
        "reason": "missing_execution_environment",
        "missing": ["champion"],
    }

    mismatched_hardware = _artifact(
        tmp_path,
        "challenger-other-hardware",
        challenger_rows,
        wall=4.0,
        throughput=1.0,
        hardware_name="other-hardware",
    )
    result = analyze_paired_evaluations(
        records=records,
        champion=_artifact(
            tmp_path, "champion", champion_rows, wall=2.0, throughput=2.0
        ),
        challenger=mismatched_hardware,
        cost_table=cost_table,
        bootstrap_samples=10,
        bootstrap_seed=1,
    )
    assert result["actual_compute"]["ratios"] is None
    assert (
        result["actual_compute"]["comparability"]["reason"]
        == "hardware_fingerprint_mismatch"
    )


def test_corrupt_hardware_evidence_is_rejected(tmp_path, source_basis, target_basis):
    _, champion_rows, challenger_rows = _paired_records_and_rows(
        source_basis, target_basis
    )
    champion = _artifact(tmp_path, "champion", champion_rows, wall=2.0, throughput=2.0)
    challenger = _artifact(
        tmp_path,
        "challenger",
        challenger_rows,
        wall=2.0,
        throughput=2.0,
        corrupt_fingerprint=True,
    )
    with pytest.raises(ValueError, match="hardware fingerprint"):
        validate_paired_protocol(champion, challenger)


def test_resolved_device_and_effective_batch_mismatches_are_rejected(
    tmp_path, source_basis, target_basis
):
    _, champion_rows, challenger_rows = _paired_records_and_rows(
        source_basis, target_basis
    )
    champion = _artifact(tmp_path, "champion", champion_rows, wall=2.0, throughput=2.0)
    resolved_mismatch = _artifact(
        tmp_path, "challenger", challenger_rows, wall=2.0, throughput=2.0
    )
    resolved_mismatch.aggregate["execution_environment"]["resolved_device"] = "cpu:1"
    with pytest.raises(ValueError, match="resolved execution devices"):
        validate_paired_protocol(champion, resolved_mismatch)

    batch_mismatch = _artifact(
        tmp_path, "challenger", challenger_rows, wall=2.0, throughput=2.0
    )
    batch = batch_mismatch.aggregate["execution_environment"]["batch_protocol"]
    batch["observed_batch_count"] = 2
    batch["observed_batch_size_histogram"] = {"2": 2}
    batch["effective_inference_batch_count"] = 2
    batch["effective_inference_batch_size_histogram"] = {"2": 2}
    batch_mismatch.aggregate["latency"]["batch_count"] = 2
    with pytest.raises(ValueError, match="effective batch protocols"):
        validate_paired_protocol(champion, batch_mismatch)


def test_mcnemar_and_bootstrap_are_deterministic():
    assert mcnemar_exact(2, 0)["p_value"] == 0.5
    first = paired_bootstrap_mean_ci([1.0, -1.0, 0.0], samples=50, seed=9)
    second = paired_bootstrap_mean_ci([1.0, -1.0, 0.0], samples=50, seed=9)
    assert first == second


def test_evaluation_loader_and_canonical_writer(tmp_path):
    evaluation_dir = tmp_path / "eval"
    evaluation_dir.mkdir()
    rows = [
        {
            "task_id": "task-1",
            "split": "test",
            "input_type": "builtins.int",
            "witness_length": 1,
            "success": False,
            "solution_length": None,
            "solution_function_ids": [],
            "solution_function_names": [],
        }
    ]
    (evaluation_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "overall": {"tasks": 1, "successes": 0},
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(evaluation_dir / "results.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(rows[0]) + "\n")
    artifact = load_evaluation_artifact(evaluation_dir)
    assert artifact.rows == tuple(rows)
    assert set(artifact.file_digests) == {"aggregate.json", "results.jsonl.gz"}

    output = write_canonical_report({"z": 1, "a": [2]}, tmp_path / "report.json")
    assert output.read_text(encoding="utf-8") == '{"a":[2],"z":1}\n'
