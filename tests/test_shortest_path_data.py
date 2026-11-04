from pathlib import Path

import pytest

from wandering_light.common_functions import basic_fns
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.shortest_path_data import (
    ShortestPathRecertification,
    apply_solver_candidate,
    bounded_relabel,
    certified_specs,
    read_jsonl_gz,
    recertify_shortest_path,
    recertify_shortest_v1_record,
    write_jsonl_gz,
)
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList

inc = FunctionDef(
    name="inc",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 1",
)
add_two = FunctionDef(
    name="add_two",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 2",
)
identity = FunctionDef(
    name="identity",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x",
)
neg_zero = FunctionDef(
    name="neg_zero",
    input_type="builtins.float",
    output_type="builtins.float",
    code="return -(x * 0.0)",
)
negate = FunctionDef(
    name="negate",
    input_type="builtins.float",
    output_type="builtins.float",
    code="return -x",
)

SHORTEST_V1_PATH = (
    Path(__file__).resolve().parents[1]
    / "wandering_light"
    / "evals"
    / "data"
    / "random_inputs_500_shortest_v1.jsonl.gz"
)


def _spec(functions):
    return TrajectorySpec(TypedList([0]), FunctionDefList(functions))


def _float_spec(functions):
    return TrajectorySpec(TypedList([1.0]), FunctionDefList(functions))


def test_bfs_replaces_random_walk_with_global_shortcut():
    record = bounded_relabel(
        _spec([inc, inc]),
        0,
        split="test",
        available_functions=FunctionDefSet([inc, add_two]),
    )

    assert record["certified"]
    assert record["method"] == "bfs"
    assert record["relabeled_functions"] == ["add_two"]
    assert record["lower_bound"] == record["upper_bound"] == 1


def test_original_upper_bound_certifies_length_four_after_depth_three():
    record = bounded_relabel(
        _spec([inc] * 4),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )

    assert record["certified"]
    assert record["method"] == "original_after_bfs"
    assert record["lower_bound"] == record["upper_bound"] == 4


def test_length_four_subsequence_certifies_length_five_proposal():
    record = bounded_relabel(
        _spec([inc, inc, identity, inc, inc]),
        0,
        split="test",
        available_functions=FunctionDefSet([inc, identity]),
    )

    assert record["certified"]
    assert record["method"] == "subsequence_after_bfs"
    assert record["relabeled_functions"] == ["inc"] * 4


def test_length_five_without_length_four_witness_remains_bracketed():
    record = bounded_relabel(
        _spec([inc] * 5),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )

    assert not record["certified"]
    assert (record["lower_bound"], record["upper_bound"]) == (4, 5)


def test_valid_solver_candidate_tightens_bound_and_certifies():
    record = bounded_relabel(
        _spec([inc] * 5),
        0,
        split="test",
        available_functions=FunctionDefSet([inc]),
    )
    record["output"] = TypedList([4]).to_string()

    updated = apply_solver_candidate(
        record,
        FunctionDefList([inc] * 4),
        available_functions=FunctionDefSet([inc]),
        budget=8,
    )

    assert updated["certified"]
    assert updated["method"] == "rl_after_bfs"
    assert updated["rl_budget"] == 8


def test_identity_is_certified_but_excluded_from_dataset():
    record = bounded_relabel(
        _spec([identity]),
        0,
        split="test",
        available_functions=FunctionDefSet([identity]),
    )

    assert record["certified"] and record["relabeled_length"] == 0
    assert (
        len(certified_specs([record], available_functions=FunctionDefSet([identity])))
        == 0
    )


def test_compressed_records_are_deterministic_and_round_trip(tmp_path):
    records = [{"certified": True, "source_index": 3}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    write_jsonl_gz(records, first)
    write_jsonl_gz(records, second)

    assert first.read_bytes() == second.read_bytes()
    assert read_jsonl_gz(first) == records


def test_bfs_destination_is_judged_by_answer_equality():
    # [1.0] -neg_zero-> [-0.0] -negate-> [0.0]. Search keeps the two zeros
    # apart, but a solver answering [-0.0] is graded correct for [0.0], so the
    # recorded distance must be 1, not 2.
    record = bounded_relabel(
        _float_spec([neg_zero, negate]),
        0,
        split="test",
        available_functions=FunctionDefSet([neg_zero, negate]),
    )

    assert record["output"] == TypedList([0.0]).to_string()
    assert record["certified"]
    assert record["method"] == "bfs"
    assert record["relabeled_functions"] == ["neg_zero"]
    assert record["relabeled_length"] == 1


class TestRecertifyShortestPath:
    def test_complete_expansion_without_shortcut_certifies(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([4]),
            4,
            available_functions=FunctionDefSet([inc]),
        )

        assert isinstance(result, ShortestPathRecertification)
        assert result.ok
        assert result.outcome == "certified"
        assert result.recorded_distance == 4
        assert result.shorter_distance is None
        assert result.search_depth == 3
        assert result.certified_search_depth == 3
        assert result.complete_expansion
        assert result.states_searched == 4
        assert result.transitions_attempted == 3
        assert result.stop_reason is None

    def test_shorter_witness_is_inflated(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([2]),
            2,
            available_functions=FunctionDefSet([inc, add_two]),
        )

        assert not result.ok
        assert result.outcome == "inflated"
        assert result.shorter_distance == 1
        assert result.search_depth == 1

    def test_identity_recorded_as_positive_distance_is_inflated(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([0]),
            1,
            available_functions=FunctionDefSet([inc]),
        )

        assert result.outcome == "inflated"
        assert result.shorter_distance == 0
        assert result.search_depth == 0

    def test_inflation_is_detected_from_a_partial_expansion(self):
        # One transition is enough to reach [2] through add_two, which the
        # budget then cuts off; a reached state is reachable regardless.
        functions = FunctionDefSet([add_two, inc])
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([2]),
            3,
            available_functions=functions,
            max_transitions=1,
        )

        assert result.outcome == "inflated"
        assert result.shorter_distance == 1
        assert not result.complete_expansion
        assert result.stop_reason == "max_transitions"

    def test_budget_exhaustion_without_shortcut_is_inconclusive(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([4]),
            4,
            available_functions=FunctionDefSet([inc]),
            max_transitions=2,
        )

        assert not result.ok
        assert result.outcome == "inconclusive"
        assert result.shorter_distance is None
        assert result.stop_reason == "max_transitions"
        assert not result.complete_expansion
        assert result.certified_search_depth == 2
        assert result.transitions_attempted == 2

    def test_state_budget_is_reported(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([9]),
            3,
            available_functions=FunctionDefSet([inc, add_two]),
            max_states=2,
        )

        assert result.outcome == "inconclusive"
        assert result.stop_reason == "max_states"
        assert result.states_searched == 2

    def test_distance_zero_identity_is_certified_without_search(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([0]),
            0,
            available_functions=FunctionDefSet([inc]),
        )

        assert result.ok
        assert result.search_depth == 0
        assert result.transitions_attempted == 0
        assert result.states_searched == 1

    def test_distance_one_needs_no_transitions(self):
        result = recertify_shortest_path(
            TypedList([0]),
            TypedList([1]),
            1,
            available_functions=FunctionDefSet([inc]),
        )

        assert result.ok
        assert result.search_depth == 0
        assert result.transitions_attempted == 0

    def test_signed_zero_shortcut_is_judged_in_answer_space(self):
        # [0.0] is only reached at depth 2 as a search state, but [-0.0] at
        # depth 1 is the same answer, so a recorded distance of 2 is inflated.
        result = recertify_shortest_path(
            TypedList([1.0]),
            TypedList([0.0]),
            2,
            available_functions=FunctionDefSet([neg_zero, negate]),
        )

        assert result.outcome == "inflated"
        assert result.shorter_distance == 1

    @pytest.mark.parametrize("distance", [-1, 1.0, "2", True, None])
    def test_rejects_invalid_distances(self, distance):
        with pytest.raises(ValueError):
            recertify_shortest_path(
                TypedList([0]),
                TypedList([1]),
                distance,
                available_functions=FunctionDefSet([inc]),
            )

    def test_rejects_distance_zero_for_distinct_target(self):
        with pytest.raises(ValueError, match="target == input"):
            recertify_shortest_path(
                TypedList([0]),
                TypedList([1]),
                0,
                available_functions=FunctionDefSet([inc]),
            )


class TestRecertifyShortestV1Record:
    functions = FunctionDefSet([inc, add_two])

    def _record(self, spec_functions):
        return bounded_relabel(
            _spec(spec_functions),
            0,
            split="test",
            available_functions=self.functions,
        )

    def test_certified_record_recertifies(self):
        record = self._record([inc, inc])

        result = recertify_shortest_v1_record(
            record, available_functions=self.functions
        )

        assert result.ok
        assert result.recorded_distance == 1

    def test_tampered_length_is_inflated(self):
        record = self._record([inc, inc])
        record.update(relabeled_functions=["inc", "inc"], relabeled_length=2)

        result = recertify_shortest_v1_record(
            record, available_functions=self.functions
        )

        assert result.outcome == "inflated"
        assert result.shorter_distance == 1

    def test_budgets_are_forwarded(self):
        record = self._record([inc, inc, inc])
        record.update(
            relabeled_functions=["inc", "inc", "inc"],
            relabeled_length=3,
            output=TypedList([3]).to_string(),
        )

        result = recertify_shortest_v1_record(
            record, available_functions=self.functions, max_transitions=0
        )

        assert result.outcome == "inconclusive"
        assert result.stop_reason == "max_transitions"

    def test_witness_that_misses_the_output_is_rejected(self):
        record = self._record([inc, inc])
        record["relabeled_functions"] = ["inc"]

        with pytest.raises(
            ValueError, match="stored witness does not reproduce the recorded output"
        ):
            recertify_shortest_v1_record(record, available_functions=self.functions)

    def test_uncertified_record_is_rejected(self):
        record = self._record([inc, inc])
        record["certified"] = False

        with pytest.raises(ValueError, match="certified"):
            recertify_shortest_v1_record(record, available_functions=self.functions)

    def test_length_mismatch_is_rejected(self):
        record = self._record([inc, inc])
        record["relabeled_length"] = 2

        with pytest.raises(ValueError, match="relabeled_length"):
            recertify_shortest_v1_record(record, available_functions=self.functions)

    def test_missing_output_is_rejected(self):
        record = self._record([inc, inc])
        record["output"] = None

        with pytest.raises(ValueError, match="output"):
            recertify_shortest_v1_record(record, available_functions=self.functions)

    def test_unknown_witness_function_is_rejected(self):
        record = self._record([inc, inc])

        with pytest.raises(ValueError, match="unknown"):
            recertify_shortest_v1_record(
                record, available_functions=FunctionDefSet([inc])
            )

    def test_identity_record_is_certified(self):
        record = bounded_relabel(
            _spec([identity]),
            0,
            split="test",
            available_functions=FunctionDefSet([identity]),
        )

        result = recertify_shortest_v1_record(
            record, available_functions=FunctionDefSet([identity])
        )

        assert result.ok
        assert result.recorded_distance == 0


def test_committed_shortest_v1_dataset_recertifies():
    records = read_jsonl_gz(SHORTEST_V1_PATH)
    assert records

    failures = {}
    for record in records:
        result = recertify_shortest_v1_record(record, available_functions=basic_fns)
        if not result.ok:
            failures[record["source_index"]] = result

    assert failures == {}
