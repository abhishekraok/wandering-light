"""How the segment composite cuts a witness and assigns ops to quotation bodies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.segment_composite import levels, segments

# `[ 1 add ] map dup [ 0 lt? ] filter`, labelled as `step_kinds` labels it.
PROGRAM = ["[", "1", "add", "]", "map", "dup", "[", "0", "lt?", "]", "filter"]
KINDS = [
    "open", "body", "body", "close", "run",
    "direct", "open", "body", "body", "close", "run",
]  # fmt: skip
DEFICITS = [float(i) for i in range(1, 12)]


def test_a_segment_carries_its_bodies_and_ends_at_a_top_level_combinator():
    assert segments(DEFICITS, KINDS) == [1 + 2 + 3 + 4 + 5, 6 + 7 + 8 + 9 + 10 + 11]


def test_trailing_glue_after_the_last_combinator_is_its_own_segment():
    assert segments([1.0, 2.0, 3.0], ["open", "run", "direct"]) == [3.0, 3.0]


def test_an_op_belongs_to_the_innermost_open_quotation():
    top, bodies = levels(PROGRAM, DEFICITS, KINDS)
    # The `[` is the parent's decision and the `]` the body's own.
    assert bodies == [2 + 3 + 4, 8 + 9 + 10]
    assert top == [1 + 5, 6 + 7 + 11]


def test_a_nested_quotation_is_charged_to_itself_not_to_its_parent():
    program = ["[", "[", "1", "add", "]", "map", "]", "map"]
    kinds = ["open", "open", "body", "body", "body", "body", "close", "run"]
    top, bodies = levels(program, [1.0] * 8, kinds)
    # Inner body: 1 add ]; outer body: its own [ , the inner map and its ].
    assert bodies == [3.0, 3.0]
    assert top == [2.0]
