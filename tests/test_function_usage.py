import json

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.function_usage import FunctionUsageTracker


def make_function(
    name: str,
    *,
    basis_function_id: str | None = None,
    basis_function_fingerprint: str | None = None,
) -> FunctionDef:
    metadata = {}
    if basis_function_id is not None:
        metadata["basis_function_id"] = basis_function_id
    if basis_function_fingerprint is not None:
        metadata["basis_function_fingerprint"] = basis_function_fingerprint
    return FunctionDef(
        name=name,
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
        metadata=metadata,
    )


@pytest.fixture
def functions() -> FunctionDefSet:
    return FunctionDefSet(
        [
            make_function("increment"),
            make_function("double"),
            make_function("unused"),
        ]
    )


def test_tracks_solution_and_total_invocation_counts(functions):
    tracker = FunctionUsageTracker()
    increment, double, _ = functions

    tracker.record_solution([increment, increment, double])
    tracker.record_solution([double])

    assert tracker.successful_solve_count == 2
    assert tracker.get(increment).solution_count == 1
    assert tracker.get(increment).invocation_count == 2
    assert tracker.get(increment).last_used_solve == 1
    assert tracker.get(double).solution_count == 2
    assert tracker.get(double).invocation_count == 2
    assert tracker.get(double).last_used_solve == 2
    assert tracker.usage_rate(increment) == 0.5
    assert tracker.usage_rate(double) == 1.0


def test_empty_success_counts_in_usage_rate_denominator(functions):
    tracker = FunctionUsageTracker()
    increment, _, _ = functions

    tracker.record_solution([])
    tracker.record_solution([increment])

    assert tracker.successful_solve_count == 2
    assert tracker.usage_rate(increment) == 0.5


def test_uses_id_then_fingerprint_then_name_and_supports_name_lookup():
    identified = make_function(
        "increment",
        basis_function_id="fn:increment:v1",
        basis_function_fingerprint="ignored-fingerprint",
    )
    fingerprinted = make_function("double", basis_function_fingerprint="sha256:double")
    legacy = make_function("legacy")
    tracker = FunctionUsageTracker()

    tracker.record_solution([identified, fingerprinted, legacy])

    assert set(tracker.snapshot()) == {
        "fn:increment:v1",
        "sha256:double",
        "legacy",
    }
    assert tracker.get("increment").invocation_count == 1
    assert tracker.get("double").invocation_count == 1
    assert tracker.get("legacy").invocation_count == 1


def test_rejects_malformed_stable_identifier_instead_of_falling_back_to_name():
    function = make_function("increment")
    function.metadata["basis_function_id"] = 123

    with pytest.raises(ValueError, match="basis_function_id"):
        FunctionUsageTracker().record_solution([function])


def test_rejects_ambiguous_name_lookup():
    first = make_function("same_name", basis_function_id="fn:first")
    second = make_function("same_name", basis_function_id="fn:second")
    tracker = FunctionUsageTracker()
    tracker.record_solution([first])
    tracker.record_solution([second])

    with pytest.raises(ValueError, match="Ambiguous function name"):
        tracker.get("same_name")


def test_identifies_and_ranks_unused_functions(functions):
    tracker = FunctionUsageTracker()
    increment, double, unused = functions
    tracker.record_solution([increment, increment])
    tracker.record_solution([double])

    assert tracker.unused_functions(functions) == [unused]
    assert tracker.least_used_functions(functions) == [unused, double, increment]


def test_least_used_ranking_uses_total_occurrences_as_primary_measure():
    repeated = make_function("repeated")
    broad = make_function("broad")
    tracker = FunctionUsageTracker()
    tracker.record_solution([repeated, repeated, repeated])
    tracker.record_solution([broad])
    tracker.record_solution([broad])

    assert tracker.get(repeated).solution_count == 1
    assert tracker.get(repeated).invocation_count == 3
    assert tracker.get(broad).solution_count == 2
    assert tracker.get(broad).invocation_count == 2
    assert tracker.least_used_functions([repeated, broad]) == [broad, repeated]


def test_rejects_negative_rank_limit(functions):
    tracker = FunctionUsageTracker()

    with pytest.raises(ValueError, match="limit must be non-negative"):
        tracker.least_used_functions(functions, limit=-1)


def test_round_trips_version_two_with_basis_provenance(tmp_path):
    tracker = FunctionUsageTracker("basis-v1", "sha256:basis-v1")
    increment = make_function("increment", basis_function_id="fn:increment:v1")
    tracker.record_solution([increment, increment])

    output_path = tracker.save(tmp_path / FunctionUsageTracker.FILE_NAME)
    restored = FunctionUsageTracker.load(
        output_path,
        basis_set_id="basis-v1",
        basis_digest="sha256:basis-v1",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["basis_set_id"] == "basis-v1"
    assert payload["basis_digest"] == "sha256:basis-v1"
    assert payload["functions"]["fn:increment:v1"]["function_name"] == "increment"
    assert restored.to_dict() == tracker.to_dict()


def test_loads_legacy_version_one_and_reserializes_as_version_two():
    legacy = {
        "version": 1,
        "successful_solve_count": 2,
        "functions": {
            "increment": {
                "solution_count": 1,
                "invocation_count": 2,
                "last_used_solve": 2,
            }
        },
    }

    tracker = FunctionUsageTracker.from_dict(legacy)

    assert tracker.basis_set_id is None
    assert tracker.basis_digest is None
    assert tracker.get("increment").invocation_count == 2
    assert tracker.to_dict()["version"] == 2


def test_rejects_loading_incompatible_or_unknown_provenance():
    payload = FunctionUsageTracker("basis-v1", "digest-v1").to_dict()

    with pytest.raises(ValueError, match="Incompatible basis_set_id"):
        FunctionUsageTracker.from_dict(
            payload, basis_set_id="basis-v2", basis_digest="digest-v1"
        )
    with pytest.raises(ValueError, match="Incompatible basis_digest"):
        FunctionUsageTracker.from_dict(
            payload, basis_set_id="basis-v1", basis_digest="digest-v2"
        )
    with pytest.raises(ValueError, match="must be provided together"):
        FunctionUsageTracker.from_dict(payload, basis_set_id="basis-v1")

    legacy_payload = {
        "version": 1,
        "successful_solve_count": 0,
        "functions": {},
    }
    with pytest.raises(ValueError, match="Incompatible basis_set_id"):
        FunctionUsageTracker.from_dict(
            legacy_payload, basis_set_id="basis-v1", basis_digest="digest-v1"
        )


def test_merges_compatible_trackers_and_offsets_last_use_sequence():
    increment = make_function("increment", basis_function_id="fn:increment")
    double = make_function("double", basis_function_id="fn:double")
    first = FunctionUsageTracker("basis-v1", "digest-v1")
    second = FunctionUsageTracker("basis-v1", "digest-v1")
    first.record_solution([increment])
    second.record_solution([])
    second.record_solution([increment, double])

    returned = first.merge(second)

    assert returned is first
    assert first.successful_solve_count == 3
    assert first.get(increment).solution_count == 2
    assert first.get(increment).invocation_count == 2
    assert first.get(increment).last_used_solve == 3
    assert first.get(double).last_used_solve == 3


def test_rejects_merging_incompatible_basis_provenance():
    first = FunctionUsageTracker("basis-v1", "digest-v1")
    second = FunctionUsageTracker("basis-v2", "digest-v2")

    with pytest.raises(ValueError, match="incompatible basis provenance"):
        first.merge(second)


def test_record_solution_can_bind_empty_tracker_but_not_mix_bases():
    tracker = FunctionUsageTracker()
    tracker.record_solution([], basis_set_id="basis-v1", basis_digest="digest-v1")

    assert tracker.basis_set_id == "basis-v1"
    assert tracker.basis_digest == "digest-v1"
    with pytest.raises(ValueError, match="Incompatible basis provenance"):
        tracker.record_solution([], basis_set_id="basis-v2", basis_digest="digest-v2")


def test_record_solution_infers_registry_provenance_from_metadata():
    function = make_function("increment", basis_function_id="fn:increment")
    function.metadata.update(
        {"basis_set_id": "basis-v1", "basis_set_digest": "digest-v1"}
    )
    tracker = FunctionUsageTracker()

    tracker.record_solution([function])

    assert tracker.basis_set_id == "basis-v1"
    assert tracker.basis_digest == "digest-v1"


def test_rejects_functions_with_mixed_basis_provenance():
    first = make_function("first", basis_function_id="fn:first")
    second = make_function("second", basis_function_id="fn:second")
    first.metadata.update({"basis_set_id": "basis-v1", "basis_set_digest": "digest-v1"})
    second.metadata.update(
        {"basis_set_id": "basis-v2", "basis_set_digest": "digest-v2"}
    )

    with pytest.raises(ValueError, match="mixed or incomplete basis provenance"):
        FunctionUsageTracker().record_solution([first, second])


def test_rejects_partial_basis_provenance():
    with pytest.raises(ValueError, match="must be provided together"):
        FunctionUsageTracker("basis-v1", None)
    with pytest.raises(ValueError, match="must be provided together"):
        FunctionUsageTracker(None, "digest-v1")

    partial = make_function("increment", basis_function_id="fn:increment")
    partial.metadata["basis_set_id"] = "basis-v1"
    with pytest.raises(ValueError, match="must be provided together"):
        FunctionUsageTracker.basis_provenance([partial])
