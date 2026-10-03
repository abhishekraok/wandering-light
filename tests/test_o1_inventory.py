"""Synthetic checks for O1-B: never load an archive or held-out examples."""

from dataclasses import replace

import pytest

from experiments.branch_check.o1_inventory import (
    Budget,
    Limits,
    ProvenanceError,
    analyze_task,
    catalogue,
    digest,
    gate,
    graph_paths,
    group_rows,
    ordered_pairs,
    replay,
    required_path,
    selected_records,
    selected_rows,
)
from wandering_light.stack_beam import key_of
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS


def product(*texts):
    return tuple(parse_stack(text, basis=BASIS) for text in texts)


def record(program, **updates):
    return {"program": program, "on_witness": False, "rank_cumulative": 0} | updates


def segment(index, programs=(), roots=("",), stop=None):
    return {
        "segment": index,
        "stop": stop,
        "roots": [{"program": program} for program in roots],
        "states": [record(program) for program in programs],
    }


def row(**updates):
    return {
        "behaviour": "add_pair",
        "example_set": 0,
        "split": "train",
        "inputs": ["[ 1 2 ]", "[ 4 5 ]"],
        "outputs": ["[ 3 ]", "[ 9 ]"],
        "witness": "[ add ] i",
        "witness_cuts": [4],
        "segments": [segment(0, ("[ swap ] i", "[ drop ] i"))],
    } | updates


def candidate(text, label, segment=0, length=4, on_witness=False, digest="a"):
    return {
        "product": key_of(product(text)),
        "label": label,
        "segment": segment,
        "prefix_ops": length,
        "on_witness": on_witness,
        "digest": digest,
    }


def test_loop_splicing_tightens_earlier_state_and_global_lexicographic_tie():
    budget = Budget()
    start = product("[ 1 2 ]")
    ops = ("swap", "swap", "1", "add")
    states = required_path(start, ops, budget)
    assert graph_paths(states, ops, budget)[start] == ("1", "add")
    # Two same-length routes to [2], with the lexically smaller route later.
    start = product("[ 1 ]")
    ops = ("dup", "add", "1", "sub", "1", "add")
    states = required_path(start, ops, budget)
    paths = graph_paths(states, ops, budget)
    assert paths[start] == ("1", "add")
    assert states[-1] not in paths  # goal is absorbing, even mid-witness
    assert () not in catalogue(states, ops, budget)["programs"]


def test_future_root_does_not_exclude_candidate_but_current_and_prior_roots_do():
    data = row(
        segments=[
            segment(0, ("[ swap ] i", "[ ] i")),
            segment(1, ("[ swap ] i",), roots=("[ swap ] i",)),
            segment(2, ("[ swap ] i",), roots=("",)),
        ]
    )
    result = analyze_task(data, Budget())
    assert result["status"] == "complete"
    records = {(c["segment"], c["program"]): c for c in result["candidates"]}
    assert not records[0, "[ swap ] i"]["exclusions"]
    assert records[0, "[ swap ] i"]["label"] == 4
    assert "expanded" in records[0, "[ ] i"]["exclusions"]
    assert "expanded" in records[1, "[ swap ] i"]["exclusions"]
    assert "expanded" in records[2, "[ swap ] i"]["exclusions"]


def test_closure_checks_every_example_and_detects_code_inside_a_list():
    data = row(
        inputs=["[ 1 ]", "[ [ add ] ]"],
        outputs=["[ 1 ]", "[ [ add ] ]"],
        witness="[ ] i",
        witness_cuts=[3],
        segments=[segment(0, ("[ ] i",))],
    )
    result = analyze_task(data, Budget())
    assert result["status"] == "complete"
    assert "code_bearing" in result["candidates"][0]["exclusions"]


@pytest.mark.parametrize(
    "changes",
    [
        {"segment": 1},
        {"prefix_ops": 5},
        {"label": 4},
        {"label": None},
        {"on_witness": True},
    ],
)
def test_pairs_reject_unmatched_tied_unlabelled_or_both_witness_members(changes):
    a = candidate("[ 1 ]", 4, on_witness=True)
    b = candidate("[ 2 ]", 8, digest="b") | changes
    assert ordered_pairs([a, b]) == ([], 0)


def test_product_pair_counts_once_per_task_across_segments():
    a, b = candidate("[ 1 ]", 4), candidate("[ 2 ]", 8, digest="b")
    pairs, repeats = ordered_pairs([a, b, a | {"segment": 1}, b | {"segment": 1}])
    assert len(pairs) == 1 and repeats == 1
    assert pairs[0]["both_off_witness"]


def test_unlabelled_candidate_and_membership_ignore_saved_flags():
    data = row(segments=[segment(0, ("[ swap ] i", "[ drop ] i"))])
    for entry in data["segments"][0]["states"]:
        entry["on_witness"] = True  # deliberately untrusted
    result = analyze_task(data, Budget())
    candidates = {c["program"]: c for c in result["candidates"]}
    assert not candidates["[ swap ] i"]["on_witness"]
    assert candidates["[ swap ] i"]["label"] == 4
    assert candidates["[ drop ] i"]["label"] is None
    assert result["pairs"] == []


def test_guard_refusal_combinator_refusal_and_repeated_prefix_charges():
    budget = Budget()
    assert replay(product("[ ]"), ("add",), budget)["status"] == "refused"
    assert budget.counters()["charged_fuel"] == 1
    assert replay(product("[ ]"), ("i",), budget)["status"] == "refused"
    assert budget.counters()["charged_fuel"] == 2001
    start = product("[ 1 2 ]", "[ 4 5 ]")
    assert replay(start, ("swap",), budget)["status"] == "complete"
    assert replay(start, ("swap",), budget)["status"] == "complete"
    assert budget.calls == 6 and budget.fuel == 2005  # no shared-prefix memoization


def test_full_per_call_fuel_exhaustion_is_censored_and_cannot_inflate_label():
    budget = Budget(replace(Limits(), call_fuel=2))
    outcome = replay(product("[ 1 2 ]"), ("[", "swap", "swap", "]", "i"), budget)
    assert outcome["status"] == "censored"
    assert outcome["reason"].endswith("out of fuel")
    # A matrix resource hit clears whole-task primary pairs.
    result = analyze_task(row(), Budget(replace(Limits(), task_calls=25)))
    assert result["status"] == "censored" and result["pairs"] == []
    assert result["phase"] == "matrix"
    assert result["candidates"][0]["trials"][-1]["status"] == "censored"


def test_atom_censoring_of_root_and_candidate_prefix_excludes_whole_task():
    # The witness reaches 8 atoms across two examples; these prefixes exceed it.
    limits = replace(Limits(), atoms=8)
    for data, phase in (
        (row(segments=[segment(0, (), roots=("dup dup dup",))]), "roots"),
        (row(segments=[segment(0, ("[ dup dup dup ] i",))]), "candidates"),
    ):
        result = analyze_task(data, Budget(limits))
        assert result["status"] == "censored"
        assert result["phase"] == phase and result["pairs"] == []


def test_remaining_allocation_and_run_scope_stop_are_exact():
    budget = Budget(replace(Limits(), run_fuel=1))
    result = replay(product("[ 1 ]"), ("dup", "dup"), budget)
    assert result["status"] == "censored" and result["scope"] == "run"
    assert budget.calls == 1 and budget.fuel == 1


def test_sampling_digest_and_no_refill_after_exclusion():
    assert (
        digest("o1b-v1|add_pair")
        == "1857440e7d38e6bf0e118fff76d772dac6e54d7bc4eb1e5b6987515b7afbe1c4"
    )
    rows = [row(behaviour=f"fake-{i}") for i in range(70)]
    selected = selected_rows([*rows, row(example_set=1)])
    assert len(selected) == 64 and all(r["example_set"] == 0 for r in selected)
    programs = [
        " ".join(["1"] * i + ["[", "add", "]", "[", "]", "dip"]) for i in range(40)
    ]
    data = row(segments=[segment(0, programs)])
    records, totals = selected_records(data)
    assert len(records) == 32 and totals["unique_completed_records"] == 40
    result = analyze_task(data, Budget())
    assert result["status"] == "complete"
    assert len(result["selected_records"]) == 32
    assert result["counts"].get("closed_eligible", 0) == 0
    assert len(result["candidates"]) == 32


def test_inconsistent_duplicates_are_errors_and_incomplete_segments_are_excluded():
    data = row(segments=[segment(0, ("[ swap ] i", "[ swap ] i"))])
    data["segments"][0]["states"][1]["on_witness"] = True
    with pytest.raises(ProvenanceError, match="inconsistent duplicate"):
        selected_records(data)
    data = row(segments=[segment(0, ("[ swap ] i",), stop="fit")])
    assert selected_records(data)[0] == []


def test_grouping_transitively_uses_unselected_behaviour_bridge(monkeypatch):
    from types import SimpleNamespace

    import experiments.branch_check.o1_inventory as module

    monkeypatch.setattr(
        module,
        "BY_NAME",
        {
            "a": SimpleNamespace(skeleton="s1"),
            "bridge": SimpleNamespace(skeleton="s1"),
            "b": SimpleNamespace(skeleton="s2"),
        },
    )
    groups = group_rows(
        [
            {"behaviour": "a", "witness": "add"},
            {"behaviour": "bridge", "witness": "sub"},
            {"behaviour": "b", "witness": "sub"},
        ]
    )
    assert groups == {"a": "a", "b": "a", "bridge": "a"}


def test_one_provenance_error_falsifies_gate_even_with_sufficient_good_tasks():
    tasks = [
        {
            "behaviour": f"b{i}",
            "status": "complete",
            "pairs": [{"both_off_witness": True} for _ in range(5)],
        }
        for i in range(20)
    ]
    groups = {task["behaviour"]: task["behaviour"] for task in tasks}
    assert gate(tasks, groups, [])["passed"]
    assert not gate(tasks, groups, ["bad witness"])["passed"]
    tasks[0]["status"] = "censored"
    assert gate(tasks, groups, [])["pairs"] == 95


def test_two_distinct_paths_to_equal_product_violate_archive_merge_invariant():
    result = analyze_task(
        row(segments=[segment(0, ("[ ] i", "[ swap swap ] i"))]), Budget()
    )
    assert result["status"] == "error"
    assert "equal products retained twice" in result["reason"]


def test_cli_preserves_partial_matrix_and_unprocessed_manifest(tmp_path, monkeypatch):
    import gzip
    import hashlib
    import json

    import experiments.branch_check.o1_inventory as module

    source = tmp_path / "synthetic.jsonl.gz"
    with gzip.open(source, "wt") as stream:
        for data in (row(), row(behaviour="abs_diff_pair")):
            stream.write(json.dumps(data) + "\n")
    monkeypatch.setattr(
        module, "SOURCE_SHA", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        module, "Budget", lambda: Budget(replace(Limits(), run_calls=25))
    )
    output = tmp_path / "inventory.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["inventory", "--archive", str(source), "--out", str(output)],
    )
    assert module.main() == 0
    saved = json.loads(output.read_text())
    assert len(saved["manifest"]) == 2 and len(saved["unprocessed"]) == 1
    assert saved["stop"] == "run_calls" and not saved["complete"]
    assert saved["tasks"][0]["status"] == "censored"
    assert saved["tasks"][0]["candidates"][0]["trials"][-1]["status"] == "censored"
    assert saved["gate"]["pairs"] == 0 and saved["counters"]["calls"] == 25
    # A second CLI invocation may not overwrite a result or silently rerun.
    with pytest.raises(SystemExit):
        module.main()
