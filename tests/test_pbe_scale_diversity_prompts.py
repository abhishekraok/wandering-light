import json

from experiments.pbe_scale.diversity_prompts import summarize


def test_only_shown_examples_count_and_excluded_rows_are_ignored(tmp_path):
    row = {
        "candidate_digest": "keep",
        "candidate": {"input_kind": "list_int", "reference": "must never execute"},
        "task": {
            "inputs": ["[ (1 2 3 4) 0 ]", "[ (1) 2 ]"],
            "outputs": ["[ 4 ]", "[ 1 ]"],
            "held_out_inputs": ["[ nil 0 ]"] * 8,
            "held_out_outputs": ["[ 0 ]"] * 8,
        },
    }
    accepted, exclusions = tmp_path / "accepted.jsonl", tmp_path / "exclusions.json"
    accepted.write_text(
        json.dumps(row)
        + "\n"
        + json.dumps({**row, "candidate_digest": "exclude"})
        + "\n"
    )
    exclusions.write_text(json.dumps({"excluded_hashes": ["exclude"]}))
    group = summarize(accepted, exclusions)["groups"]["list_int"]
    assert group["rows"] == 1 and group["shown_pairs"] == 2
    assert group["shown_long_case_counts_per_task"] == {1: 1}
    assert group["distinct_shown_outputs"] == {2: 1}
