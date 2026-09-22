"""The frame-conditioned re-encoding of witness rows (inner-frames #160 E2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from experiments.frame_rows import frame_product, frame_state, frame_witness_rows
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import witness_rows


def task(name, seed=0):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def test_a_top_level_state_is_untouched():
    state = (Lst((Int(1), Int(2))), Int(3))
    assert frame_state(state, ()) is state


def test_a_defined_level_shows_its_tag_and_frame_in_place_of_its_items():
    state = (Lst((Int(-9),)), Mark(), Int(1), Sym("add"))
    got = frame_state(state, (("map", 0, (Int(-8),)),))
    assert got == (Lst((Int(-9),)), Mark(), Sym("map"), Int(-8))


def test_an_ifte_body_carries_its_role():
    state = (Int(4), Lst((Sym("dup"),)), Mark(), Sym("drop"))
    got = frame_state(state, (("ifte", 1, (Int(4),)),))
    assert got == (Int(4), Lst((Sym("dup"),)), Mark(), Sym("ifte"), Int(1), Int(4))


def test_an_undefined_level_keeps_the_machines_items():
    state = (Int(1), Mark(), Sym("lt?"), Mark(), Int(0))
    views = (("filter", 0, None), ("map", 0, (Bool(True),)))
    got = frame_state(state, views)
    assert got == (Int(1), Mark(), Sym("lt?"), Mark(), Sym("map"), Bool(True))


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
    # Before `]` every example shows `Mark, map, first item + 1`, and nothing else
    # above the mark; before the first body op it shows the first item itself.
    for before_add, before_close, text in zip(
        frame_product(t)[2], frame_product(t)[3], t.inputs, strict=True
    ):
        mark = next(i for i, v in enumerate(before_close) if isinstance(v, Mark))
        assert before_close[mark + 1] == Sym("map")
        assert len(before_close) == mark + 3
        first = before_add[mark + 2]
        assert before_close[mark + 2] == Int(first.value + 1), text
    # The rows differ from the flat ones exactly on the three body steps.
    flat, framed = witness_rows([t]), frame_witness_rows([t])
    differs = [
        not np.array_equal(x.items, y.items) for x, y in zip(flat, framed, strict=True)
    ]
    assert differs == [False, True, True, True, False]
