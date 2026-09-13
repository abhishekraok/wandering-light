"""Witness containment for the step-3 solve split (world-plus-v4-policy, #134)."""

from experiments.step3_containment import contained, containment, span_in


def test_spans_match_whole_ops_only():
    assert span_in("cons filter", "[ lt? ] swap cons filter")
    assert not span_in("cons", "uncons swap drop")
    assert not span_in("[ lt? ]", "[ 0 lt? ] filter")
    assert span_in("[ lt? ] swap cons filter", "[ lt? ] swap cons filter")


def test_contained_needs_world_fit_and_no_v4_fit():
    witnesses = {
        "only_world": ["[ lt? ] swap cons filter"],
        "both": ["reverse"],
        "neither": ["dup add"],
        "second_witness_in_v4": ["1 add", "dup mul"],
    }
    v4_fit = {"a": "reverse drop", "b": "swap dup mul"}
    world_fit = {
        "w1": "dup [ lt? ] swap cons filter size",
        "w2": "reverse",
        "w3": "3 1 add",
    }
    table = containment(witnesses, v4_fit, world_fit)
    assert table["only_world"] == {"witness_ops": [6], "world_fit": 1, "v4_fit": 0}
    assert contained(table) == {"only_world"}
