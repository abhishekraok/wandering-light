"""Pins for the substrate constants ``docs/STATUS.md`` quotes.

Each test re-derives one published number from the stack machine and asserts
it at the precision STATUS states it. They are regression tests for the
machine, not reproductions of a study: if one fails, either the machine
changed or the row is wrong, and the answer is a Superseded entry, never an
edit here (reproduce-status-rows (#75)).

Numbers with a corpus behind them (the v3 audit, the guard's state ratios, the
v4 baselines) are not pinned: they need a Hub payload and minutes to hours,
and they reproduce by re-measurement when a programme needs them.
"""

from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

import pytest

from experiments.stack_machine_bfs import (
    ROOTS,
    basis_summary,
    measure_frontiers,
    root_stack,
)
from experiments.stack_pbe_basis_floor import roots as data_only_roots
from experiments.stack_pbe_basis_floor import sweep
from experiments.stack_pbe_guard_probe import measure_lengths
from wandering_light.quotation_guard import FIXED_SUPPLY
from wandering_light.stack_machine import ops_for_basis

# STATUS "Branching, stack machine": the bounded published protocol.
# (b_eff, bits/step, growth per depth) — b_eff pools depths 1–4 over four
# roots; growth is the mean of the last two frontier ratios on the one-list
# root, to depth 5 for the two live bases.
BRANCHING = {
    "executable-26": (13.38, 3.742, 7.29),
    "mark-28": (23.11, 4.530, 22.47),
}


@pytest.fixture(scope="module")
def summaries() -> dict[str, dict]:
    return {basis: basis_summary(basis, ops_for_basis(basis)) for basis in BRANCHING}


@pytest.mark.parametrize("basis", sorted(BRANCHING))
def test_branching_of_the_live_bases(summaries, basis):
    b_eff, bits, growth = BRANCHING[basis]
    summary = summaries[basis]
    assert summary["b_eff"] == pytest.approx(b_eff, abs=0.005)
    assert summary["bits_per_step"] == pytest.approx(bits, abs=0.0005)
    assert summary["bits_per_step"] == pytest.approx(math.log2(summary["b_eff"]))
    assert summary["growth_per_depth"] == pytest.approx(growth, abs=0.005)


def test_full_52_branching_through_depth_three():
    # STATUS quotes full-52 at 37.33 pooled through depth 4, which costs 17 s
    # and 1.4M states on the one-list root alone; that figure lives in
    # reports/mark-28-20260827. This pins the same walk to depth 3, exactly.
    rows = measure_frontiers(root_stack(ROOTS["one list"]), 3, ops_for_basis("full-52"))
    assert [(r["expanded"], r["edges"], r["new"]) for r in rows] == [
        (1, 37, 37),
        (37, 1408, 1291),
        (1291, 48373, 41911),
    ]


def test_executable_26_empty_root_ball_is_9503_states():
    # The depth-5 ball #41 pinned and the stack walk-collapse report re-derives.
    rows = measure_frontiers(root_stack(""), 5, ops_for_basis("executable-26"))
    assert rows[-1]["cumulative"] == 9503


def test_mark_28_quote_closed_frontier(summaries):
    # STATUS "mark-28 reproduction": one-list root, quote-closed / all states.
    summary = summaries["mark-28"]
    split = {
        row["depth"]: (row["quote_closed"], row["all_states"])
        for row in summary["state_space_split"]
    }
    assert split == {3: (721, 1842), 4: (6298, 37312)}


def _floor(basis: str, depth: int) -> dict:
    totals: Counter[str] = Counter()
    legal: Counter[str] = Counter()
    ran: Counter[str] = Counter()
    for stack in data_only_roots(4, 17):
        result = sweep(stack, basis, depth)
        totals.update(
            {k: result[k] for k in ("states", "quotations", "executable_quotations")}
        )
        legal.update(result["legal"])
        ran.update(result["ran"])
    return {**totals, "legal": dict(legal), "ran": dict(ran)}


@pytest.mark.slow
def test_executable_26_reaches_no_executable_quotation_from_data():
    # STATUS "mark-28 is forced once inputs carry no quotations": the depth-5
    # ball around the four data-only roots, executable-26 half.
    floor = _floor("executable-26", 5)
    assert floor["states"] == 307707
    assert floor["quotations"] == 6058
    assert floor["executable_quotations"] == 0
    assert sum(floor["legal"].values()) == 33468
    assert floor["ran"] == {}


def test_mark_28_executes_what_it_builds_from_data():
    # The mark-28 half of the same row is 47 s and 1.5 GB at depth 5; STATUS's
    # 78,731 / 2,289 stay in reports/stack-pbe-v3-20260828. Depth 4 is the
    # same structural contrast at 2 s, pinned exactly.
    floor = _floor("mark-28", 4)
    assert (floor["states"], floor["quotations"], floor["executable_quotations"]) == (
        186510,
        4208,
        2856,
    )
    assert sum(floor["legal"].values()) == 3972
    assert floor["ran"] == {"i": 36, "dip": 15, "fold": 9, "map": 2}


@pytest.mark.slow
def test_guard_prunes_long_bodies(capsys):
    # STATUS "Guarding the open-quotation states", body arm: 99.5% of
    # length-16 bodies unrunnable, surviving branching ~18.8 against 26.
    args = SimpleNamespace(
        nested=False, seed=17, supply=FIXED_SUPPLY, samples=50_000, max_length=16
    )
    rows = measure_lengths(args)["rows"]
    capsys.readouterr()
    by_length = {row["length"]: row for row in rows}
    assert by_length[1]["dead"] == 0.0
    assert by_length[1]["b_eff"] == 26.0
    assert by_length[16]["dead"] == pytest.approx(0.995, abs=0.0005)
    assert by_length[16]["b_eff"] == pytest.approx(18.78, abs=0.01)
