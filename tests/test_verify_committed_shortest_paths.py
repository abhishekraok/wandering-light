from experiments import verify_committed_shortest_paths as verify
from wandering_light.shortest_path_data import read_jsonl_gz
from wandering_light.typed_list import TypedList


def _record(index: int, distance: int, *, zero_target: bool):
    return {
        "source_index": index,
        "relabeled_length": distance,
        "input": TypedList([float(index + 1)]).to_string(),
        "output": TypedList([0.0 if zero_target else 1.0]).to_string(),
    }


def test_induction_sample_balances_distance_and_zero_float_risk():
    records = [
        _record(index, distance, zero_target=zero)
        for index, (distance, zero) in enumerate(
            (distance, zero) for distance in range(1, 5) for zero in (False, True)
        )
    ]

    sample = verify._stratified_induction_sample(records, sample_size=8)

    assert len(sample) == 8
    assert {verify._induction_stratum(record) for record in sample} == {
        (distance, zero) for distance in range(1, 5) for zero in (False, True)
    }


def test_committed_induction_sample_recertifies():
    records = read_jsonl_gz(verify.INDUCTION_DATA)
    sample = verify._stratified_induction_sample(records)
    summary = verify._verify_shortest_v1_records(sample)
    strata = verify._stratum_counts(sample)

    assert len(records) == 118_730
    assert summary["outcomes"] == {
        "certified": verify.PR_INDUCTION_SAMPLE,
        "inflated": 0,
        "inconclusive": 0,
    }
    assert len(strata) == 8
    assert set(strata.values()) == {12}


def test_deep_summary_does_not_treat_incomplete_search_as_a_pass():
    row = {"complete_expansion": False, "outcome": "inconclusive"}
    summary = verify._deep_summary(
        {
            "records": 1,
            "witness_failures": [],
            "roots_leaking_across_splits": [],
            "recertification_failures": [],
            "recertification_inconclusive": [row],
            "recertified": [row],
            "ok": True,
        }
    )

    assert summary["inconclusive"] == 1
    assert not summary["ok"]
