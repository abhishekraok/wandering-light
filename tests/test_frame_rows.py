"""The frame-conditioned re-encoding of witness rows (inner-frames #160 E2)."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.frame_rows import (
    EXIT,
    META,
    example_walks,
    frame_product,
    frame_state,
    frame_witness_rows,
    level_exits,
)
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import witness_rows


def task(name, seed=0):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def label(tag, role=0, captures=0):
    return (META, Sym(tag), Int(role), Int(captures))


def test_a_top_level_state_is_untouched():
    state = (Lst((Int(1), Int(2))), Int(3))
    assert frame_state(state, ()) is state


def test_a_defined_level_keeps_its_items_and_adds_its_label_and_frame():
    state = (Lst((Int(-9),)), Mark(), Int(1), Sym("add"))
    view = (("map", 0, 0, (Int(-8),)),)
    assert frame_state(state, view) == (
        Lst((Int(-9),)), Mark(), Int(1), Sym("add"), *label("map"), Int(-8)
    )  # fmt: skip
    # The first design replaced the written body; it is kept only for its probe.
    assert frame_state(state, view, "replace") == (
        Lst((Int(-9),)), Mark(), *label("map")[1:], Int(-8)
    )  # fmt: skip


def test_an_ifte_body_carries_its_role():
    state = (Int(4), Lst((Sym("dup"),)), Mark(), Sym("drop"))
    got = frame_state(state, (("ifte", 1, 0, (Int(4),)),))
    assert got == (
        Int(4), Lst((Sym("dup"),)), Mark(), Sym("drop"), *label("ifte", 1), Int(4)
    )  # fmt: skip


def test_the_label_carries_the_closure_capture_count():
    """`x [ lt? ] swap cons filter`: the frame shows x, so the label says one capture."""
    state = (Lst((Int(5),)), Int(4), Mark(), Sym("lt?"))
    view = (("filter", 0, 1, (Bool(False),)),)
    assert frame_state(state, view, "tags") == (
        Lst((Int(5),)), Int(4), Mark(), Sym("lt?"), *label("filter", 0, 1)
    )  # fmt: skip


def test_an_undefined_level_keeps_the_machines_items():
    state = (Int(1), Mark(), Sym("lt?"), Mark(), Int(0))
    views = (("filter", 0, 0, None), ("map", 0, 0, (Bool(True),)))
    assert frame_state(state, views) == (
        Int(1), Mark(), Sym("lt?"), Mark(), Int(0), *label("map"), Bool(True)
    )  # fmt: skip


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
    # Before `add` each example shows the written `1`, the label, and the frame
    # (first item, 1); before `]` the frame has run `add`: the first item + 1.
    product = frame_product(t)
    for before_add, before_close, text in zip(
        product[2], product[3], t.inputs, strict=True
    ):
        mark = next(i for i, v in enumerate(before_close) if isinstance(v, Mark))
        assert before_add[mark + 1 : mark + 6] == (Int(1), *label("map"))
        first, one = before_add[mark + 6 :]
        assert one == Int(1)
        assert before_close[mark + 1 : mark + 7] == (Int(1), Sym("add"), *label("map"))
        assert before_close[mark + 7 :] == (Int(first.value + 1),), text
    # The rows differ from the flat ones exactly on the three body steps.
    flat, framed = witness_rows([t]), frame_witness_rows([t])
    differs = [
        not np.array_equal(x.items, y.items) for x, y in zip(flat, framed, strict=True)
    ]
    assert differs == [False, True, True, True, False]


def test_the_tag_control_shows_the_label_but_not_the_frame():
    state = (Lst((Int(-9),)), Mark(), Int(1), Sym("add"))
    got = frame_state(state, (("map", 0, 0, (Int(-8),)),), "tags")
    assert got == (Lst((Int(-9),)), Mark(), Int(1), Sym("add"), *label("map"))
    # An undefined level shows no label in either encoding.
    undefined = (("map", 0, 0, None),)
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


def test_the_inner_encodings_annotate_only_the_innermost_defined_level():
    state = (Int(1), Mark(), Sym("dup"), Mark(), Int(0))
    views = (("dip", 0, 0, (Int(7),)), ("map", 0, 0, (Int(2),)))
    assert frame_state(state, views, "frames_inner") == (
        Int(1), Mark(), Sym("dup"), Mark(), Int(0), *label("map"), Int(2)
    )  # fmt: skip
    assert frame_state(state, views, "tags_inner") == (
        Int(1), Mark(), Sym("dup"), Mark(), Int(0), *label("map")
    )  # fmt: skip
    # The stacked encoding labels both levels.
    assert frame_state(state, views, "tags").count(META) == 2


def test_an_undefined_innermost_level_passes_the_label_to_its_defined_ancestor():
    state = (Int(1), Mark(), Sym("dup"), Mark(), Int(0))
    views = (("dip", 0, 0, (Int(7),)), ("map", 0, 0, None))
    assert frame_state(state, views, "frames_inner") == (
        Int(1), Mark(), Sym("dup"), *label("dip"), Int(7), Mark(), Int(0)
    )  # fmt: skip


def test_the_oracle_adds_the_exit_frame_after_its_own_separator():
    state = (Lst((Int(-9),)), Mark(), Int(1))
    view = (("map", 0, 0, (Int(-9), Int(1))),)
    assert frame_state(state, view, "exit_inner", ((Int(-8),),)) == (
        Lst((Int(-9),)), Mark(), Int(1), *label("map"), Int(-9), Int(1), EXIT, Int(-8)
    )  # fmt: skip
    # Without an exit frame the oracle shows what frames_inner shows.
    assert frame_state(state, view, "exit_inner", (None,)) == frame_state(
        state, view, "frames_inner"
    )


def test_a_levels_exit_is_its_frame_when_its_bracket_closes():
    t = task("increment_each")
    assert t.program == ("[", "1", "add", "]", "map")
    for states, views in example_walks(t):
        exits = level_exits(t.program, views)
        first = states[1][0].items[0]
        # Every body step of the one level sees the same exit: first item + 1.
        assert exits[1] == exits[2] == exits[3] == ((Int(first.value + 1),),)
        assert exits[0] == ()


def test_one_level_inner_encodings_match_the_stacked_ones():
    t = task("increment_each")
    for inner, stacked in (("frames_inner", "frames"), ("tags_inner", "tags")):
        assert frame_product(t, inner) == frame_product(t, stacked)
    rows = frame_witness_rows([t], "exit_inner")
    flat = frame_witness_rows([t], "frames_inner")
    changed = [
        not np.array_equal(x.items, y.items) for x, y in zip(flat, rows, strict=True)
    ]
    assert changed == [False, True, True, True, False]


def test_the_exit_separator_never_occurs_in_a_machine_state():
    """`[` pushes a Mark, so the oracle's separator cannot be a captured item."""
    from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, parse_stack, step_raw
    from wandering_light.stack_pbe import BASIS, OP_BY_NAME

    stack = parse_stack("1", basis=BASIS)
    for name in ("[", "[", "]", "1", "]", "[", "dup", "]"):
        stack = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)[0]
        assert EXIT not in stack
        assert all(EXIT not in v.items for v in stack if isinstance(v, Lst))
