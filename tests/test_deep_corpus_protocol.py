from __future__ import annotations

from copy import deepcopy

import pytest

from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.evals.deep_corpus_protocol import (
    ALLOWED_ARM_IDS,
    ARM_IDS,
    CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    COHORT_SIZE,
    SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    aggregate_candidate_funnels,
    begin_test_opening,
    build_bfs_settings,
    build_chained_source_amendment,
    build_evaluator_source_identity,
    build_frozen_cohorts,
    build_generation_config,
    build_protocol_lock,
    build_selection_lock,
    build_source_amendment,
    build_tokenizer_identity,
    canonical_json,
    classify_parse_failure,
    complete_test_opening,
    create_test_opening_ledger_file,
    freeze_cohort,
    interrupt_test_opening,
    load_test_opening_ledger,
    new_test_opening_ledger,
    protocol_training_evaluator_sources,
    record_test_arm_result,
    replace_test_opening_ledger_file,
    resume_test_opening,
    sha256_json,
    validate_active_arm_ids,
    validate_frozen_cohorts,
    validate_protocol_lock,
    validate_selection_locks,
    validate_test_opening_ledger,
)
from wandering_light.typed_list import TypedList


def _record(*, split: str, distance: int, index: int) -> BasisTaskRecord:
    unique = {"validation": 10_000, "test": 20_000}.get(split, 30_000) + index
    return BasisTaskRecord.create(
        split=split,
        input_value=TypedList([unique], int),
        output_value=TypedList([unique + distance], int),
        witness_function_ids=[f"bf:inc:{step}" for step in range(distance)],
        witness_function_names=["inc"] * distance,
        basis_set_id="wl-core-v1",
        basis_set_digest="sha256:" + "1" * 64,
        generator="unit-test",
        seed=7,
        source_index=index,
        metadata={
            "certified_distance": distance,
            "root_digest": f"sha256:{unique:064x}",
        },
    )


@pytest.fixture
def corpus_records() -> list[BasisTaskRecord]:
    records = []
    for split in ("validation", "test"):
        for distance in range(1, 9):
            for offset in range(13):
                records.append(
                    _record(
                        split=split,
                        distance=distance,
                        index=distance * 100 + offset,
                    )
                )
    return records


@pytest.fixture
def protocol_components(tmp_path, corpus_records):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text('{"vocab":{"inc":1}}', encoding="utf-8")
    evaluator_file = tmp_path / "evaluate.py"
    evaluator_file.write_text("def evaluate(): return 1\n", encoding="utf-8")
    cohorts = build_frozen_cohorts(corpus_records, selection_seed=20260825)
    generation = build_generation_config(
        sampling_seed=20260825,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=64,
        batch_size=32,
        max_sequence_length=2048,
        extra={"padding_side": "left", "eos_token_id": 2},
    )
    bfs = build_bfs_settings(budget=20_000_000)
    tokenizer = build_tokenizer_identity(
        [tokenizer_file],
        root=tmp_path,
        metadata={"vocab_size": 2, "basis_names_atomic": True},
    )
    evaluator_sources = build_evaluator_source_identity(
        [evaluator_file], root=tmp_path, metadata={"git_tree": "abc123"}
    )
    return {
        "root": tmp_path,
        "tokenizer_file": tokenizer_file,
        "evaluator_file": evaluator_file,
        "cohorts": cohorts,
        "generation": generation,
        "bfs": bfs,
        "tokenizer": tokenizer,
        "evaluator_sources": evaluator_sources,
    }


@pytest.fixture
def protocol_lock(protocol_components):
    return build_protocol_lock(
        corpus_identity={
            "manifest_digest": "sha256:" + "2" * 64,
            "basis_set_id": "wl-core-v1",
            "basis_set_digest": "sha256:" + "1" * 64,
            "hub_revision": "a" * 40,
        },
        cohorts=protocol_components["cohorts"],
        generation=protocol_components["generation"],
        bfs=protocol_components["bfs"],
        tokenizer=protocol_components["tokenizer"],
        evaluator_sources=protocol_components["evaluator_sources"],
    )


@pytest.fixture
def selection_locks(protocol_lock):
    return [
        build_selection_lock(
            arm_id=arm_id,
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            checkpoint_identity={
                "weights_sha256": "sha256:" + f"{index + 3:x}" * 64,
                "parameter_count": (index + 1) * 10_000_000,
            },
            validation_selection={
                "summary_sha256": "sha256:" + f"{index + 6:x}" * 64,
                "learning_rate": (1e-4, 3e-4, 1e-3)[index],
            },
        )
        for index, arm_id in enumerate(ARM_IDS)
    ]


def test_canonical_hash_is_order_independent_and_rejects_nan():
    left = {"b": [2, 3], "a": {"y": False, "x": 1}}
    right = {"a": {"x": 1, "y": False}, "b": [2, 3]}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)
    assert sha256_json(left).startswith("sha256:")
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json({"bad": float("nan")})


def test_cohorts_are_exact_balanced_deterministic_and_order_independent(
    corpus_records,
):
    first = build_frozen_cohorts(corpus_records, selection_seed=19)
    second = build_frozen_cohorts(list(reversed(corpus_records)), selection_seed=19)

    assert first == second
    assert first["cohorts_sha256"] == sha256_json(
        {key: value for key, value in first.items() if key != "cohorts_sha256"}
    )
    validation = first["validation"]
    test = first["test"]
    assert validation["size"] == test["size"] == COHORT_SIZE
    assert (
        validation["distance_histogram"]
        == test["distance_histogram"]
        == {str(distance): 12 for distance in range(1, 9)}
    )
    assert not set(validation["task_ids"]).intersection(test["task_ids"])
    assert validation["task_ids_sha256"] == sha256_json(validation["task_ids"])
    assert test["task_ids_sha256"] == sha256_json(test["task_ids"])
    assert validate_frozen_cohorts(first, records=corpus_records) == first


def test_cohort_validation_rejects_mutation_wrong_membership_and_short_bucket(
    corpus_records,
):
    cohorts = build_frozen_cohorts(corpus_records, selection_seed=19)
    mutated = deepcopy(cohorts)
    mutated["validation"]["tasks"][0]["task_id"] = "changed"
    with pytest.raises(ValueError, match="changed after it was frozen"):
        validate_frozen_cohorts(mutated)

    source_without_selected = [
        record
        for record in corpus_records
        if record.task_id != cohorts["validation"]["task_ids"][0]
    ]
    with pytest.raises(ValueError, match="deterministic source membership"):
        validate_frozen_cohorts(cohorts, records=source_without_selected)

    short = [
        record
        for record in corpus_records
        if not (
            record.split == "validation"
            and record.metadata["certified_distance"] == 8
            and record.source_index % 100 >= 11
        )
    ]
    with pytest.raises(ValueError, match="12 are required"):
        freeze_cohort(short, split="validation", selection_seed=19)


def test_cohort_builder_rejects_task_overlap_across_splits(corpus_records):
    duplicate = corpus_records[0]
    cross_split = BasisTaskRecord(
        **{
            **duplicate.__dict__,
            "split": "test",
            "metadata": {
                **duplicate.metadata,
                "root_digest": "sha256:" + "9" * 64,
            },
        }
    )

    with pytest.raises(ValueError, match="globally disjoint"):
        build_frozen_cohorts([*corpus_records, cross_split], selection_seed=19)


def test_protocol_lock_freezes_all_identities_and_rejects_any_mutation(
    protocol_lock, protocol_components, corpus_records
):
    assert set(protocol_lock["allowed_arm_ids"]) == ALLOWED_ARM_IDS
    assert (
        validate_protocol_lock(
            protocol_lock,
            expected_sha256=protocol_lock["protocol_lock_sha256"],
            records=corpus_records,
            tokenizer_root=protocol_components["root"],
            evaluator_root=protocol_components["root"],
        )
        == protocol_lock
    )

    mutated = deepcopy(protocol_lock)
    mutated["generation"]["temperature"] = 0.7
    with pytest.raises(ValueError, match="content changed"):
        validate_protocol_lock(mutated)

    protocol_components["evaluator_file"].write_text(
        "def evaluate(): return 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after freezing"):
        validate_protocol_lock(
            protocol_lock,
            tokenizer_root=protocol_components["root"],
            evaluator_root=protocol_components["root"],
        )


def test_protocol_source_amendment_preserves_training_sources_and_legacy_fallback(
    protocol_components, corpus_records
):
    values = protocol_components
    metadata = {"git_commit": "a" * 40, "roots": ["evaluate.py"]}
    training_sources = build_evaluator_source_identity(
        [values["evaluator_file"]], root=values["root"], metadata=metadata
    )
    predecessor = build_protocol_lock(
        corpus_identity={
            "manifest_digest": "sha256:" + "2" * 64,
            "generator_source": {
                "protocol_git_commit": metadata["git_commit"],
                "evaluator_source_sha256": training_sources["source_sha256"],
            },
        },
        cohorts=values["cohorts"],
        generation=values["generation"],
        bfs=values["bfs"],
        tokenizer=values["tokenizer"],
        evaluator_sources=training_sources,
    )
    values["evaluator_file"].write_text("def evaluate(): return 2\n", encoding="utf-8")
    successor_sources = build_evaluator_source_identity(
        [values["evaluator_file"]],
        root=values["root"],
        metadata={**metadata, "git_commit": "b" * 40},
    )
    amendment = build_source_amendment(
        predecessor_protocol_lock_sha256=predecessor["protocol_lock_sha256"],
        training_evaluator_sources=training_sources,
        successor_evaluator_sources=successor_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=["evaluate.py"],
        failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    )
    successor = build_protocol_lock(
        corpus_identity=predecessor["corpus"],
        cohorts=predecessor["cohorts"],
        generation=predecessor["generation"],
        bfs=predecessor["bfs"],
        tokenizer=predecessor["tokenizer"],
        evaluator_sources=successor_sources,
        source_amendment=amendment,
    )

    assert (
        validate_protocol_lock(
            successor,
            records=corpus_records,
            tokenizer_root=values["root"],
            evaluator_root=values["root"],
        )
        == successor
    )
    assert protocol_training_evaluator_sources(predecessor) == training_sources
    assert protocol_training_evaluator_sources(successor) == training_sources
    assert successor["evaluator_sources"] == successor_sources
    for field in ("corpus", "cohorts", "generation", "bfs", "tokenizer"):
        assert successor[field] == predecessor[field]

    for invalid_generator_source in (None, "not-a-mapping"):
        invalid_corpus = deepcopy(successor["corpus"])
        if invalid_generator_source is None:
            invalid_corpus.pop("generator_source")
        else:
            invalid_corpus["generator_source"] = invalid_generator_source
        invalid_predecessor = build_protocol_lock(
            corpus_identity=invalid_corpus,
            cohorts=successor["cohorts"],
            generation=successor["generation"],
            bfs=successor["bfs"],
            tokenizer=successor["tokenizer"],
            evaluator_sources=training_sources,
        )
        invalid_amendment = build_source_amendment(
            predecessor_protocol_lock_sha256=invalid_predecessor[
                "protocol_lock_sha256"
            ],
            training_evaluator_sources=training_sources,
            successor_evaluator_sources=successor_sources,
            reason="Suppress tokenizer token-type IDs before retrying validation.",
            changed_source_allowlist=["evaluate.py"],
            failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
        )
        invalid = build_protocol_lock(
            corpus_identity=invalid_corpus,
            cohorts=successor["cohorts"],
            generation=successor["generation"],
            bfs=successor["bfs"],
            tokenizer=successor["tokenizer"],
            evaluator_sources=successor_sources,
            source_amendment=invalid_amendment,
        )
        with pytest.raises(ValueError, match="no preserved generator-source binding"):
            validate_protocol_lock(invalid)

    wrong_digest_amendment = build_source_amendment(
        predecessor_protocol_lock_sha256="sha256:" + "f" * 64,
        training_evaluator_sources=training_sources,
        successor_evaluator_sources=successor_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=["evaluate.py"],
        failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    )
    wrong_digest_lock = build_protocol_lock(
        corpus_identity=successor["corpus"],
        cohorts=successor["cohorts"],
        generation=successor["generation"],
        bfs=successor["bfs"],
        tokenizer=successor["tokenizer"],
        evaluator_sources=successor_sources,
        source_amendment=wrong_digest_amendment,
    )
    with pytest.raises(ValueError, match="does not reconstruct its predecessor"):
        validate_protocol_lock(wrong_digest_lock)

    changed_corpus = deepcopy(successor["corpus"])
    changed_corpus["manifest_digest"] = "sha256:" + "3" * 64
    changed_generation_body = deepcopy(successor["generation"])
    changed_generation_body.pop("generation_sha256")
    changed_generation_body["temperature"] = 0.7
    changed_generation = {
        **changed_generation_body,
        "generation_sha256": sha256_json(changed_generation_body),
    }
    changed_cohorts = build_frozen_cohorts(corpus_records, selection_seed=20260826)
    for overrides in (
        {"corpus_identity": changed_corpus},
        {"generation": changed_generation},
        {"cohorts": changed_cohorts},
    ):
        changed_lock = build_protocol_lock(
            corpus_identity=overrides.get("corpus_identity", successor["corpus"]),
            cohorts=overrides.get("cohorts", successor["cohorts"]),
            generation=overrides.get("generation", successor["generation"]),
            bfs=successor["bfs"],
            tokenizer=successor["tokenizer"],
            evaluator_sources=successor_sources,
            source_amendment=amendment,
        )
        with pytest.raises(ValueError, match="does not reconstruct its predecessor"):
            validate_protocol_lock(changed_lock)


def test_v2_source_amendment_pins_immediate_predecessor_and_frozen_inputs(
    protocol_components, corpus_records
):
    values = protocol_components
    metadata = {"git_commit": "a" * 40, "roots": ["evaluate.py"]}
    training_sources = build_evaluator_source_identity(
        [values["evaluator_file"]], root=values["root"], metadata=metadata
    )
    original = build_protocol_lock(
        corpus_identity={
            "manifest_digest": "sha256:" + "2" * 64,
            "generator_source": {
                "protocol_git_commit": metadata["git_commit"],
                "evaluator_source_sha256": training_sources["source_sha256"],
            },
        },
        cohorts=values["cohorts"],
        generation=values["generation"],
        bfs=values["bfs"],
        tokenizer=values["tokenizer"],
        evaluator_sources=training_sources,
    )
    values["evaluator_file"].write_text("def evaluate(): return 2\n", encoding="utf-8")
    first_sources = build_evaluator_source_identity(
        [values["evaluator_file"]],
        root=values["root"],
        metadata={**metadata, "git_commit": "b" * 40},
    )
    first_amendment = build_source_amendment(
        predecessor_protocol_lock_sha256=original["protocol_lock_sha256"],
        training_evaluator_sources=training_sources,
        successor_evaluator_sources=first_sources,
        reason="Suppress tokenizer token-type IDs before retrying validation.",
        changed_source_allowlist=["evaluate.py"],
        failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    )
    first = build_protocol_lock(
        corpus_identity=original["corpus"],
        cohorts=original["cohorts"],
        generation=original["generation"],
        bfs=original["bfs"],
        tokenizer=original["tokenizer"],
        evaluator_sources=first_sources,
        source_amendment=first_amendment,
    )

    values["evaluator_file"].write_text("def evaluate(): return 3\n", encoding="utf-8")
    second_sources = build_evaluator_source_identity(
        [values["evaluator_file"]],
        root=values["root"],
        metadata={**metadata, "git_commit": "c" * 40},
    )
    second_amendment = build_chained_source_amendment(
        predecessor_protocol_lock_sha256=first["protocol_lock_sha256"],
        predecessor_evaluator_sources=first_sources,
        predecessor_source_amendment=first_amendment,
        successor_evaluator_sources=second_sources,
        reason="Reuse the completed validation BFS without changing its semantics.",
        changed_source_allowlist=["evaluate.py"],
        failure_evidence=CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE,
    )
    second = build_protocol_lock(
        corpus_identity=first["corpus"],
        cohorts=first["cohorts"],
        generation=first["generation"],
        bfs=first["bfs"],
        tokenizer=first["tokenizer"],
        evaluator_sources=second_sources,
        source_amendment=second_amendment,
    )

    assert (
        validate_protocol_lock(
            second,
            records=corpus_records,
            tokenizer_root=values["root"],
            evaluator_root=values["root"],
        )
        == second
    )
    assert second_amendment["schema_version"] == 2
    assert (
        second_amendment["predecessor_protocol_lock_sha256"]
        == first["protocol_lock_sha256"]
    )
    assert second_amendment["predecessor_evaluator_sources"] == first_sources
    assert second_amendment["predecessor_source_amendment"] == first_amendment
    assert protocol_training_evaluator_sources(second) == training_sources
    for field in ("corpus", "cohorts", "generation", "bfs", "tokenizer"):
        assert second[field] == first[field]

    changed_bfs = build_bfs_settings(budget=19_000_000)
    semantic_change = build_protocol_lock(
        corpus_identity=second["corpus"],
        cohorts=second["cohorts"],
        generation=second["generation"],
        bfs=changed_bfs,
        tokenizer=second["tokenizer"],
        evaluator_sources=second_sources,
        source_amendment=second_amendment,
    )
    with pytest.raises(ValueError, match="does not reconstruct its predecessor"):
        validate_protocol_lock(semantic_change)

    bad_evidence = {
        **CHAINED_SOURCE_AMENDMENT_FAILURE_EVIDENCE,
        "generated_candidates": 6_143,
    }
    with pytest.raises(ValueError, match="exact completed-work"):
        build_chained_source_amendment(
            predecessor_protocol_lock_sha256=first["protocol_lock_sha256"],
            predecessor_evaluator_sources=first_sources,
            predecessor_source_amendment=first_amendment,
            successor_evaluator_sources=second_sources,
            reason="Reuse the completed validation BFS without changing its semantics.",
            changed_source_allowlist=["evaluate.py"],
            failure_evidence=bad_evidence,
        )


def test_protocol_source_amendment_rejects_output_evidence_and_unlisted_change(
    protocol_components,
):
    values = protocol_components
    metadata = {"git_commit": "a" * 40, "roots": ["evaluate.py"]}
    training_sources = build_evaluator_source_identity(
        [values["evaluator_file"]], root=values["root"], metadata=metadata
    )
    values["evaluator_file"].write_text("CHANGED = True\n", encoding="utf-8")
    successor_sources = build_evaluator_source_identity(
        [values["evaluator_file"]],
        root=values["root"],
        metadata={**metadata, "git_commit": "b" * 40},
    )
    common = {
        "predecessor_protocol_lock_sha256": "sha256:" + "f" * 64,
        "training_evaluator_sources": training_sources,
        "successor_evaluator_sources": successor_sources,
        "reason": "Repair pre-forward evaluator failure.",
    }
    bad_evidence = {**SOURCE_AMENDMENT_FAILURE_EVIDENCE, "scored_tasks": 1}
    with pytest.raises(ValueError, match="zero-output"):
        build_source_amendment(
            **common,
            changed_source_allowlist=["evaluate.py"],
            failure_evidence=bad_evidence,
        )
    with pytest.raises(ValueError, match="absent from the amendment allowlist"):
        build_source_amendment(
            **common,
            changed_source_allowlist=["other.py"],
            failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        build_source_amendment(
            **common,
            changed_source_allowlist=["other.py", "evaluate.py"],
            failure_evidence=SOURCE_AMENDMENT_FAILURE_EVIDENCE,
        )


@pytest.mark.parametrize(
    ("row", "category", "names"),
    [
        (
            {"parse_valid": False, "raw_response": "", "parse_error": "empty"},
            "empty",
            (),
        ),
        (
            {
                "parse_valid": False,
                "raw_response": "inc,f_sqrt",
                "parse_error": "unknown function names: ['f_sqrt']",
            },
            "unknown_function",
            ("f_sqrt",),
        ),
        (
            {
                "parse_valid": False,
                "raw_response": "inc,,double",
                "parse_error": "empty function name in comma-separated response",
            },
            "malformed_separator",
            (),
        ),
        (
            {
                "parse_valid": False,
                "raw_response": "inc,double",
                "finish_reason": "length",
            },
            "truncation",
            (),
        ),
        (
            {
                "parse_valid": False,
                "raw_response": "???",
                "parse_error": "unrecognized output",
            },
            "other",
            (),
        ),
    ],
)
def test_parse_failure_classification_is_structured(row, category, names):
    failure = classify_parse_failure(row)

    assert failure is not None
    assert failure.category == category
    assert failure.unknown_function_names == names


def _candidate_rows() -> list[dict]:
    rows = []
    for arm_id in ARM_IDS:
        for sample_index in range(64):
            row = {
                "arm_id": arm_id,
                "task_id": "task-1",
                "sample_index": sample_index,
                "raw_response": "inc",
                "parse_valid": True,
                "executor_success": True,
                "type_mismatch": False,
                "execution_correct": False,
                "exact_match": False,
            }
            if sample_index == 0:
                row.update(
                    {
                        "raw_response": "",
                        "parse_valid": False,
                        "executor_success": False,
                        "parse_failure": {
                            "category": "empty",
                            "unknown_function_names": [],
                            "detail": "empty response",
                        },
                    }
                )
            elif sample_index in {1, 2}:
                name = "invented" if sample_index == 1 else "also_invented"
                row.update(
                    {
                        "raw_response": name,
                        "parse_valid": False,
                        "executor_success": False,
                        "parse_failure": {
                            "category": "unknown_function",
                            "unknown_function_names": [name],
                            "detail": None,
                        },
                    }
                )
            elif sample_index == 3:
                row.update(
                    {
                        "raw_response": "inc,,double",
                        "parse_valid": False,
                        "executor_success": False,
                        "parse_failure": {
                            "category": "malformed_separator",
                            "unknown_function_names": [],
                            "detail": None,
                        },
                    }
                )
            elif sample_index == 4:
                row.update(
                    {
                        "raw_response": "inc,double",
                        "parse_valid": False,
                        "executor_success": False,
                        "parse_failure": {
                            "category": "truncation",
                            "unknown_function_names": [],
                            "detail": None,
                        },
                    }
                )
            elif sample_index == 5:
                row.update(
                    {
                        "executor_success": False,
                        "type_mismatch": True,
                        "execution_error": "Type mismatch at function 'inc'",
                    }
                )
            elif sample_index == 6:
                row.update(
                    {
                        "executor_success": False,
                        "execution_error": "Function 'inc' failed",
                    }
                )
            elif sample_index == 8:
                row["execution_correct"] = True
            elif sample_index == 9:
                row["execution_correct"] = True
                row["exact_match"] = True
            rows.append(row)
    return rows


def test_candidate_funnels_report_conditional_rates_and_top_reasons():
    summary = aggregate_candidate_funnels(
        _candidate_rows(), expected_task_ids=["task-1"]
    )

    assert summary["sample_prefixes"] == [1, 8, 64]
    for arm_id in ARM_IDS:
        at_one = summary["arms"][arm_id]["by_k"]["1"]
        assert at_one["candidates"] == 1
        assert at_one["unparseable_rate"] == 1.0
        assert at_one["correct_given_parseable_rate"] == 0.0

        at_eight = summary["arms"][arm_id]["by_k"]["8"]
        assert at_eight["candidates"] == 8
        assert at_eight["parseable"] == 3
        assert at_eight["unparseable"] == 5
        assert at_eight["executable"] == 1
        assert at_eight["type_mismatch"] == 1
        assert at_eight["execution_failure_other"] == 1
        assert at_eight["parse_failure_counts"] == {
            "empty": 1,
            "unknown_function": 2,
            "malformed_separator": 1,
            "truncation": 1,
            "other": 0,
        }
        assert at_eight["top_parse_failure_reasons"][0]["category"] == "empty"
        assert at_eight["top_unknown_function_names"] == [
            {"name": "also_invented", "count": 1},
            {"name": "invented", "count": 1},
        ]

        at_64 = summary["arms"][arm_id]["by_k"]["64"]
        assert at_64["parseable"] == 59
        assert at_64["executable"] == 57
        assert at_64["execution_correct"] == 2
        assert at_64["exact_match"] == 1
        assert at_64["correct_given_parseable_rate"] == pytest.approx(2 / 59)
        assert at_64["correct_given_executable_rate"] == pytest.approx(2 / 57)
        assert at_64["solve_rate"] == 1.0
        assert at_64["exact_solve_rate"] == 1.0

    active = ARM_IDS[:2]
    partial = aggregate_candidate_funnels(
        [row for row in _candidate_rows() if row["arm_id"] in active],
        expected_task_ids=["task-1"],
    )
    assert partial["arm_ids"] == list(active)
    assert set(partial["arms"]) == set(active)


def test_candidate_funnels_reject_missing_arms_bad_order_and_bad_implications():
    rows = _candidate_rows()
    with pytest.raises(ValueError, match="exactly the active arms"):
        aggregate_candidate_funnels(
            [row for row in rows if row["arm_id"] != ARM_IDS[-1]],
            active_arm_ids=ARM_IDS,
        )

    disordered = deepcopy(rows)
    first_arm = [
        index for index, row in enumerate(disordered) if row["arm_id"] == ARM_IDS[0]
    ]
    disordered[first_arm[0]], disordered[first_arm[1]] = (
        disordered[first_arm[1]],
        disordered[first_arm[0]],
    )
    with pytest.raises(ValueError, match="ordered complete prefix"):
        aggregate_candidate_funnels(disordered)

    invalid = deepcopy(rows)
    invalid[0]["execution_correct"] = True
    with pytest.raises(ValueError, match="cannot execute or match"):
        aggregate_candidate_funnels(invalid)


def test_active_arms_and_selection_locks_accept_only_ordered_frozen_subsets(
    protocol_lock, selection_locks
):
    by_arm = validate_selection_locks(
        selection_locks,
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
    )
    assert tuple(by_arm) == ARM_IDS

    active = ARM_IDS[:2]
    assert validate_active_arm_ids(active) == active
    assert (
        tuple(
            validate_selection_locks(
                selection_locks[:2],
                protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            )
        )
        == active
    )
    for invalid in ((), (ARM_IDS[1], ARM_IDS[0]), (ARM_IDS[0], ARM_IDS[0])):
        with pytest.raises(ValueError):
            validate_active_arm_ids(invalid)
    with pytest.raises(ValueError, match="active arms"):
        validate_selection_locks(
            selection_locks[:2],
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            active_arm_ids=(ARM_IDS[0], ARM_IDS[2]),
        )
    duplicate = [selection_locks[0], selection_locks[0], selection_locks[2]]
    with pytest.raises(ValueError, match="duplicate selection"):
        validate_selection_locks(
            duplicate,
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        )
    mutated = deepcopy(selection_locks)
    mutated[0]["checkpoint"]["parameter_count"] += 1
    with pytest.raises(ValueError, match="changed after it was frozen"):
        validate_selection_locks(
            mutated,
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        )


def test_test_opening_ledger_is_once_only_and_resumes_same_interrupted_opening(
    protocol_lock, selection_locks
):
    ledger = new_test_opening_ledger(
        opening_id="issue-54-test-opening-1",
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        selection_locks=selection_locks,
    )
    assert ledger["status"] == "prepared"
    assert ledger["test_access_count"] == 0

    opened = begin_test_opening(ledger)
    assert opened["status"] == "in_progress"
    assert opened["test_access_count"] == 1
    with pytest.raises(ValueError, match="already been opened"):
        begin_test_opening(opened)

    interrupted = interrupt_test_opening(opened, reason="worker preempted")
    resumed = resume_test_opening(interrupted)
    assert resumed["status"] == "in_progress"
    assert resumed["test_access_count"] == 1
    assert resume_test_opening(resumed) == resumed

    current = resumed
    for index, arm_id in enumerate(ARM_IDS):
        result = {"candidate_sha256": "sha256:" + f"{index + 9:x}" * 64}
        current = record_test_arm_result(current, arm_id=arm_id, result_identity=result)
        assert (
            record_test_arm_result(current, arm_id=arm_id, result_identity=result)
            == current
        )
    complete = complete_test_opening(
        current, shared_bfs_identity={"rows_sha256": "sha256:" + "c" * 64}
    )
    assert complete["status"] == "complete"
    assert complete["test_access_count"] == 1
    assert (
        validate_test_opening_ledger(
            complete,
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            selection_locks=selection_locks,
        )
        == complete
    )
    with pytest.raises(ValueError, match="same interrupted"):
        resume_test_opening(complete)


def test_opening_cannot_complete_early_or_rescore_an_arm(
    protocol_lock, selection_locks
):
    ledger = begin_test_opening(
        new_test_opening_ledger(
            opening_id="opening",
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            selection_locks=selection_locks,
        )
    )
    with pytest.raises(ValueError, match="all active"):
        complete_test_opening(ledger, shared_bfs_identity={"sha256": "x"})

    first = record_test_arm_result(
        ledger, arm_id=ARM_IDS[0], result_identity={"sha256": "first"}
    )
    with pytest.raises(ValueError, match="already scored differently"):
        record_test_arm_result(
            first, arm_id=ARM_IDS[0], result_identity={"sha256": "different"}
        )


def test_ledger_file_helpers_use_exclusive_create_and_optimistic_atomic_replace(
    tmp_path, protocol_lock, selection_locks
):
    ledger = new_test_opening_ledger(
        opening_id="opening",
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        selection_locks=selection_locks,
    )
    path = tmp_path / "locks" / "test-opening.json"
    create_test_opening_ledger_file(path, ledger)
    assert load_test_opening_ledger(path) == ledger
    with pytest.raises(FileExistsError):
        create_test_opening_ledger_file(path, ledger)

    opened = begin_test_opening(ledger)
    replace_test_opening_ledger_file(
        path,
        opened,
        expected_previous_sha256=ledger["ledger_sha256"],
    )
    assert load_test_opening_ledger(path) == opened
    with pytest.raises(ValueError, match="changed concurrently"):
        replace_test_opening_ledger_file(
            path,
            interrupt_test_opening(opened, reason="stop"),
            expected_previous_sha256=ledger["ledger_sha256"],
        )


def test_ledger_mutation_is_rejected(protocol_lock, selection_locks):
    ledger = new_test_opening_ledger(
        opening_id="opening",
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        selection_locks=selection_locks,
    )
    mutated = deepcopy(ledger)
    mutated["test_access_count"] = 1
    with pytest.raises(ValueError, match="changed after it was frozen"):
        validate_test_opening_ledger(mutated)


def test_two_active_arm_ledger_seals_deferred_arm_and_feasibility(
    protocol_lock, selection_locks
):
    active = ARM_IDS[:2]
    deferred = ARM_IDS[2:]
    feasibility_digest = "sha256:" + "f" * 64
    ledger = new_test_opening_ledger(
        opening_id="two-active-arms",
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        selection_locks=selection_locks[:2],
        active_arm_ids=active,
        feasibility_artifact_digest=feasibility_digest,
    )
    assert ledger["active_arm_ids"] == list(active)
    assert ledger["deferred_arm_ids"] == list(deferred)
    assert set(ledger["arms"]) == set(active)

    opened = begin_test_opening(ledger)
    with pytest.raises(ValueError, match="not active"):
        record_test_arm_result(
            opened,
            arm_id=deferred[0],
            result_identity={"sha256": "deferred"},
        )
    for index, arm_id in enumerate(active):
        opened = record_test_arm_result(
            opened,
            arm_id=arm_id,
            result_identity={"sha256": f"active-{index}"},
        )
    complete = complete_test_opening(opened, shared_bfs_identity={"sha256": "shared"})
    assert complete["status"] == "complete"
    assert (
        validate_test_opening_ledger(
            complete,
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            selection_locks=selection_locks[:2],
            active_arm_ids=active,
            deferred_arm_ids=deferred,
            feasibility_artifact_digest=feasibility_digest,
        )
        == complete
    )

    tampered = deepcopy(complete)
    tampered["active_arm_ids"].append(deferred[0])
    tampered["deferred_arm_ids"] = []
    tampered["ledger_sha256"] = sha256_json(
        {key: value for key, value in tampered.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match=r"active arms|active selections"):
        validate_test_opening_ledger(
            tampered,
            active_arm_ids=active,
            deferred_arm_ids=deferred,
            feasibility_artifact_digest=feasibility_digest,
        )


def test_partial_ledger_requires_feasibility_evidence(protocol_lock, selection_locks):
    with pytest.raises(ValueError, match="require a feasibility artifact digest"):
        new_test_opening_ledger(
            opening_id="partial-without-feasibility",
            protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
            selection_locks=selection_locks[:2],
            active_arm_ids=ARM_IDS[:2],
        )

    ledger = new_test_opening_ledger(
        opening_id="partial-with-feasibility",
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        selection_locks=selection_locks[:2],
        active_arm_ids=ARM_IDS[:2],
        feasibility_artifact_digest="sha256:" + "f" * 64,
    )
    ledger["feasibility_artifact_digest"] = None
    ledger["ledger_sha256"] = sha256_json(
        {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match="has no feasibility artifact digest"):
        validate_test_opening_ledger(ledger)


def test_atomic_ledger_replace_cannot_add_a_deferred_arm(
    tmp_path, protocol_lock, selection_locks
):
    active = ARM_IDS[:2]
    ledger = new_test_opening_ledger(
        opening_id="immutable-active-set",
        protocol_lock_sha256=protocol_lock["protocol_lock_sha256"],
        selection_locks=selection_locks[:2],
        active_arm_ids=active,
        feasibility_artifact_digest="sha256:" + "f" * 64,
    )
    path = tmp_path / "ledger.json"
    create_test_opening_ledger_file(path, ledger)
    changed = deepcopy(ledger)
    changed["active_arm_ids"] = list(ARM_IDS)
    changed["deferred_arm_ids"] = []
    changed["selection_lock_sha256_by_arm"][ARM_IDS[2]] = selection_locks[2][
        "selection_lock_sha256"
    ]
    changed["arms"][ARM_IDS[2]] = {"status": "pending", "result_identity": None}
    changed["revision"] = 1
    changed["history"].append({"revision": 1, "event": "add-deferred"})
    changed["ledger_sha256"] = sha256_json(
        {key: value for key, value in changed.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match="identity cannot change"):
        replace_test_opening_ledger_file(
            path,
            changed,
            expected_previous_sha256=ledger["ledger_sha256"],
        )
