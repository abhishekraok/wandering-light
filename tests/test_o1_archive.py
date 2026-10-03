"""The #164 audit must not turn path metadata into false distance labels."""

import pytest

from experiments.branch_check.o1_archive import (
    audit_row,
    candidate_indices,
    replay_path,
)
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS


def product(*texts):
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def row(**updates):
    return {
        "behaviour": "add_pair",
        "example_set": 0,
        "split": "train",
        "inputs": ["[ 1 2 ]", "[ 4 5 ]"],
        "outputs": ["[ 3 ]", "[ 9 ]"],
        "witness": "add",
        "witness_cuts": [],
        "program": None,
        "segments": [],
        "termination": "wall_clock_cap",
    } | updates


def test_direct_ops_and_initial_state_are_available_even_without_saved_cuts():
    result = audit_row(row())
    assert result["all_closed_occurrences"] == 2
    assert result["candidate_indices"] == [0]
    assert result["recorded_suffix_ops"] == [1]
    assert result["candidate_pairs"] == 0


def test_equal_states_use_latest_occurrence_and_goal_is_excluded():
    states = replay_path(product("[ 1 2 ]"), ("swap", "swap", "1", "add"))
    assert candidate_indices(states, states[-1]) == [1, 2, 3]


def test_closed_requires_every_example_and_code_inside_lists_is_not_closed():
    closed = product("[ 1 ]", "[ 2 ]")
    code = product("[ 1 ]", "[ [ add ] ]")
    goal = product("[ 3 ]", "[ 4 ]")
    assert candidate_indices([closed, code, goal], goal) == [0]


def test_timeout_does_not_supply_negatives():
    result = audit_row(row())
    assert result["candidate_pairs"] == 0
    assert result["shown_fit_ops"] is None
    assert result["termination"] == "wall_clock_cap"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"split": "eval"}, "training-only"),
        ({"outputs": ["[ 0 ]", "[ 0 ]"]}, "shown target"),
        ({"witness": "add add"}, "refused"),
        ({"witness_cuts": [1]}, "cuts"),
        ({"program": "swap"}, "returned fit"),
    ],
)
def test_bad_provenance_or_continuations_are_rejected(updates, message):
    with pytest.raises(ValueError, match=message):
        audit_row(row(**updates))


def test_missing_or_false_witness_flags_do_not_remove_replayed_positives():
    result = audit_row(
        row(
            segments=[
                {"states": [{"program": "swap", "on_path": False, "on_witness": False}]}
            ]
        )
    )
    assert result["candidate_indices"] == [0]
    assert result["kept_records"] == 1


def test_combinator_cuts_can_leave_code_parked_and_are_not_training_states():
    result = audit_row(
        row(
            witness="[ add ] [ ] dip i",
            witness_cuts=[6, 7],
            segments=[
                {
                    "states": [
                        {
                            "program": "[ add ] [ ] dip",
                            "on_path": False,
                            "on_witness": True,
                        }
                    ]
                }
            ],
        )
    )
    assert result["saved_cuts_not_closed"] == 1
    assert result["candidate_indices"] == [0]
    assert result["on_witness_records"] == 1
    assert result["on_witness_closed_records"] == 0


def test_flagged_prefix_must_reach_a_saved_witness_cut_state():
    with pytest.raises(ValueError, match="on_witness"):
        audit_row(
            row(
                segments=[
                    {
                        "states": [
                            {"program": "swap", "on_path": False, "on_witness": True}
                        ]
                    }
                ]
            )
        )
