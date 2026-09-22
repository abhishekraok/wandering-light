"""The frame-conditioned re-encoding of witness rows (inner-frames #160 E2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from experiments.frame_rows import META, frame_product, frame_state, frame_witness_rows
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import witness_rows


def task(name, seed=0):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def test_a_top_level_state_is_untouched():
    state = (Lst((Int(1), Int(2))), Int(3))
    assert frame_state(state, ()) is state


def test_a_defined_level_keeps_its_items_and_adds_its_tag_and_frame():
    state = (Lst((Int(-9),)), Mark(), Int(1), Sym("add"))
    got = frame_state(state, (("map", 0, (Int(-8),)),))
    assert got == (
        Lst((Int(-9),)),
        Mark(),
        Int(1),
        Sym("add"),
        META,
        Sym("map"),
        Int(-8),
    )
    # The first design replaced the written body; it is kept only for its probe.
    replaced = frame_state(state, (("map", 0, (Int(-8),)),), "replace")
    assert replaced == (Lst((Int(-9),)), Mark(), Sym("map"), Int(-8))


def test_an_ifte_body_carries_its_role():
    state = (Int(4), Lst((Sym("dup"),)), Mark(), Sym("drop"))
    got = frame_state(state, (("ifte", 1, (Int(4),)),))
    assert got == (
        Int(4), Lst((Sym("dup"),)), Mark(), Sym("drop"), META, Sym("ifte"), Int(1), Int(4)
    )  # fmt: skip


def test_an_undefined_level_keeps_the_machines_items():
    state = (Int(1), Mark(), Sym("lt?"), Mark(), Int(0))
    views = (("filter", 0, None), ("map", 0, (Bool(True),)))
    got = frame_state(state, views)
    assert got == (
        Int(1),
        Mark(),
        Sym("lt?"),
        Mark(),
        Int(0),
        META,
        Sym("map"),
        Bool(True),
    )


def test_marks_and_levels_must_agree():
    with pytest.raises(ValueError, match="marks"):
        frame_state((Int(1), Mark()), ())


def test_a_witness_without_quotations_gives_the_flat_rows():
    tasks = [task("add_pair"), task("rotate_left", 1)]
    assert all("[" not in t.program for t in tasks)
    a, b = witness_rows(tasks), frame_witness_rows(tasks)
    assert len(a) == len(b)
    assert all(
        np.array_equal(x.items, y.items) and (x.label, x.step) == (y.label, y.step)
        for x, y in zip(a, b, strict=True)
    )


def test_a_map_body_runs_on_the_first_item():
    t = task("increment_each")
    assert t.program == ("[", "1", "add", "]", "map")
    # Before `add` each example shows the written `1`, then `META, map` and the
    # frame (first item, 1); before `]` the frame has run `add`: first item + 1.
    for before_add, before_close, text in zip(
        frame_product(t)[2], frame_product(t)[3], t.inputs, strict=True
    ):
        mark = next(i for i, v in enumerate(before_close) if isinstance(v, Mark))
        assert before_add[mark + 1 : mark + 4] == (Int(1), META, Sym("map"))
        first, one = before_add[mark + 4 :]
        assert one == Int(1)
        assert before_close[mark + 1 : mark + 5] == (
            Int(1),
            Sym("add"),
            META,
            Sym("map"),
        )
        assert before_close[mark + 5 :] == (Int(first.value + 1),), text
    # The rows differ from the flat ones exactly on the three body steps.
    flat, framed = witness_rows([t]), frame_witness_rows([t])
    differs = [
        not np.array_equal(x.items, y.items) for x, y in zip(flat, framed, strict=True)
    ]
    assert differs == [False, True, True, True, False]


def test_the_tag_control_shows_the_tag_but_keeps_the_machines_items():
    state = (Lst((Int(-9),)), Mark(), Int(1), Sym("add"))
    got = frame_state(state, (("map", 0, (Int(-8),)),), "tags")
    assert got == (Lst((Int(-9),)), Mark(), Int(1), Sym("add"), META, Sym("map"))
    # An undefined level shows no tag in either encoding.
    undefined = (("map", 0, None),)
    assert frame_state(state, undefined, "tags") == frame_state(state, undefined)


def test_the_two_encodings_rewrite_exactly_the_same_rows():
    t = task("increment_each")
    flat = witness_rows([t])
    frames = frame_witness_rows([t], "frames")
    tags = frame_witness_rows([t], "tags")
    changed = [
        [not np.array_equal(x.items, y.items) for x, y in zip(flat, arm, strict=True)]
        for arm in (frames, tags)
    ]
    assert changed[0] == changed[1] == [False, True, True, True, False]


def test_the_separator_never_occurs_in_a_machine_state():
    """`]` collapses a quotation as it is written, so META cannot be a captured item."""
    from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, parse_stack, step_raw
    from wandering_light.stack_pbe import BASIS, OP_BY_NAME

    stack = parse_stack("1", basis=BASIS)
    for name in ("[", "[", "]", "1", "]"):
        stack = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)[0]
        assert META not in stack
        assert all(META not in v.items for v in stack if isinstance(v, Lst))
