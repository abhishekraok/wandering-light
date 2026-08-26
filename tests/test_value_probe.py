"""The probe's conclusion rests on its labels and its scorer. Pin both."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from value_probe import (
    evaluate,
    evaluate_successors,
    keys_of,
    make_constant,
    make_lookup,
    median,
    rank_successors,
)


def _example(remaining, distance=None, **fields):
    base = {
        "state": "s",
        "target": "t",
        "state_type": "int",
        "target_type": "int",
        "state_size": 4,
        "target_size": 4,
        "remaining": remaining,
        "certified_distance": distance if distance is not None else remaining,
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def test_the_l1_optimal_constant_is_the_median_not_the_mean():
    """Scoring absolute error against a mean baseline would flatter the model.

    The whole probe is `learned` against `dict`, so the dicts have to be the
    strongest trivial predictor available. Under L1 that is the median. Here
    the mean is 3.0 and scores 2.4; the median is 1 and scores 2.0.
    """
    train = [_example(r) for r in (1, 1, 1, 5, 7)]
    assert median([e["remaining"] for e in train]) == 1

    constant = make_constant(train)
    assert evaluate(constant, train, "median")["mae"] == pytest.approx(2.0)
    assert evaluate(lambda e: 3.0, train, "mean")["mae"] == pytest.approx(2.4)


def test_median_is_the_lower_one_on_an_even_count():
    assert median([1, 2, 3, 4]) == 2
    assert median([5]) == 5


def test_bias_keeps_its_sign():
    """Over- and under-estimation are not interchangeable.

    An `h` that overestimates is inadmissible, so A* stops guaranteeing
    shortest paths; one that underestimates stays admissible but weak. The
    report has to be able to tell them apart, so `bias` must not be folded
    into the absolute error.
    """
    tasks = [_example(4), _example(4)]
    over = evaluate(lambda e: 6.0, tasks, "over")
    under = evaluate(lambda e: 2.0, tasks, "under")
    assert over["bias"] == pytest.approx(2.0)
    assert under["bias"] == pytest.approx(-2.0)
    assert over["mae"] == under["mae"] == pytest.approx(2.0)


def test_breakdown_is_by_remaining_distance_not_by_task_distance():
    """These are different axes and only one of them is the point.

    A state four ops into a distance-6 witness has two steps left, and it is
    the two that decides whether best-first can use it. Bucketing that state
    under d6 would blur the exact regime the bar is stated for.
    """
    tasks = [_example(2, distance=6), _example(6, distance=6)]
    row = evaluate(lambda e: 2.0, tasks, "x")
    assert row["by_remaining"] == {2: pytest.approx(0.0), 6: pytest.approx(4.0)}
    assert row["by_distance"] == {6: pytest.approx(2.0)}


def test_within1_counts_an_exact_one_step_miss():
    tasks = [_example(3), _example(3), _example(3)]
    row = evaluate(lambda e: 4.0, tasks, "off by one")
    assert row["within1"] == 1.0
    assert evaluate(lambda e: 4.5, tasks, "off by 1.5")["within1"] == 0.0


# ---------------------------------------------------------------------------
# The lookup ladder
# ---------------------------------------------------------------------------


def test_lookup_backs_off_field_by_field():
    train = [
        _example(2, state_type="int", target_type="str"),
        _example(2, state_type="int", target_type="str"),
        _example(6, state_type="int", target_type="bool"),
    ]
    predict = make_lookup(train, ("state_type", "target_type"))
    assert predict(_example(0, state_type="int", target_type="str")) == 2
    assert predict(_example(0, state_type="int", target_type="bool")) == 6
    # unseen target type falls back to the state-type table over all three
    assert predict(_example(0, state_type="int", target_type="range")) == 2
    # unseen state type falls back to the global median
    assert predict(_example(0, state_type="bytes", target_type="range")) == 2


def test_size_fields_are_bucketed_but_type_fields_are_not():
    """Raw item counts would shatter the table into single-example cells."""
    assert keys_of(_example(1, state_size=40), ("state_size",)) == (8,)
    assert keys_of(_example(1, state_size=3), ("state_size",)) == (3,)
    assert keys_of(_example(1, state_type="int"), ("state_type",)) == ("int",)


# ---------------------------------------------------------------------------
# Successor discrimination
# ---------------------------------------------------------------------------


def _group(*successors, distance=3):
    return {
        "successors": [
            _example(distance - 1, distance=distance, state=name, optimal=optimal)
            for name, optimal in successors
        ],
        "certified_distance": distance,
    }


def test_successors_rank_by_ascending_h():
    """`h` is a distance, so smaller is expanded first."""
    group = _group(("a", False), ("b", True), ("c", False))
    scores = {"a": 5.0, "b": 1.0, "c": 9.0}
    assert rank_successors(lambda s: scores[s["state"]], group) == 1
    flipped = {"a": 1.0, "b": 5.0, "c": 9.0}
    assert rank_successors(lambda s: flipped[s["state"]], group) == 2


def test_a_flat_h_ranks_worst_not_best():
    """The same pessimistic tie-break `policy_probe` had backwards once.

    A constant predictor carries no information about which successor to
    expand, so it must land at the bottom -- otherwise the constant baseline
    would appear to solve the very problem this section exists to measure.
    """
    group = _group(("a", False), ("b", True), ("c", False), ("d", False))
    assert rank_successors(lambda s: 2.0, group) == 4


def test_hitting_any_optimal_successor_counts():
    group = _group(("a", False), ("b", True), ("c", True))
    scores = {"a": 0.0, "b": 1.0, "c": 2.0}
    assert rank_successors(lambda s: scores[s["state"]], group) == 2


def test_evaluate_successors_reports_top1_and_distance_breakdown():
    good = _group(("a", True), ("b", False), distance=2)
    bad = _group(("a", False), ("b", True), distance=5)
    scores = {"a": 0.0, "b": 1.0}
    row = evaluate_successors(lambda s: scores[s["state"]], [good, bad], "x")
    assert row["top1"] == pytest.approx(0.5)
    assert row["mean_rank"] == pytest.approx(1.5)
    assert row["by_distance"] == {2: pytest.approx(1.0), 5: pytest.approx(2.0)}


# ---------------------------------------------------------------------------
# The labels
# ---------------------------------------------------------------------------


def _synthetic_corpus(tmp_path, witnesses):
    """A corpus shaped like `deep-corpus-v1`, built by running the real basis.

    The payload is not committed (#40), so the label test builds its own rather
    than skipping when the Hub download is absent.
    """
    import gzip
    import json

    from value_probe import basis_pieces

    from wandering_light.typed_list import TypedList

    _, by_name, executor = basis_pieces()
    with gzip.open(tmp_path / "discovery.jsonl.gz", "wt") as handle:
        for names in witnesses:
            state = TypedList([2, 3], item_type=int)
            source = state.to_string()
            for name in names:
                state = executor.execute(by_name[name], state)
            handle.write(
                json.dumps(
                    {
                        "input": source,
                        "output": state.to_string(),
                        "witness_function_names": names,
                        "metadata": {
                            "certified_distance": len(names),
                            "input_type": "builtins.int",
                            "output_type": "builtins.int",
                        },
                    }
                )
                + "\n"
            )
    return tmp_path


def test_a_prefix_label_is_the_exact_remaining_distance(tmp_path):
    """Prefix `j` of a witness of length `d` is labelled `d - j`, and that is
    exact -- if anything reached the target from there in fewer steps, the
    witness would not have been shortest.

    The last prefix (`j = d`) is excluded: its state *is* the target, so search
    settles it with a goal test and never asks `h`. Including it would credit
    every predictor for a free zero.
    """
    from value_probe import basis_pieces, build_examples

    _, by_name, executor = basis_pieces()
    corpus = _synthetic_corpus(tmp_path, [["inc", "double", "neg"], ["square"]])
    examples = build_examples(corpus, "discovery", by_name, executor)

    assert [e["remaining"] for e in examples] == [3, 2, 1, 1]
    assert [e["certified_distance"] for e in examples] == [3, 3, 3, 1]
    assert all(e["remaining"] > 0 for e in examples)


def test_replayed_prefixes_advance_and_land_on_the_target(tmp_path):
    """Replay has to actually move, and it has to arrive.

    A silent no-op -- reusing the task input at every step -- would still emit
    the right labels while training `h` on the wrong states, and nothing in the
    MAE table would show it.
    """
    from value_probe import basis_pieces, build_examples

    _, by_name, executor = basis_pieces()
    corpus = _synthetic_corpus(tmp_path, [["inc", "double", "neg"]])
    examples = build_examples(corpus, "discovery", by_name, executor)

    from wandering_light.typed_list import TypedList

    states = [TypedList.from_str(e["state"]) for e in examples]
    assert states[0] == TypedList([2, 3], item_type=int)
    assert states[1] == TypedList([3, 4], item_type=int)
    assert states[2] == TypedList([6, 8], item_type=int)
    target = TypedList.from_str(examples[0]["target"])
    assert executor.execute(by_name["neg"], states[2]) == target
    assert target == TypedList([-6, -8], item_type=int)
