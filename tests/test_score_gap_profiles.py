"""Synthetic #180 units only; no dataset, checkpoint, scorer or trial access."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from experiments.score_gap_profiles import (
    Cell,
    Choice,
    SeedStats,
    ShownTask,
    Snapshot,
    band_size,
    by_seed,
    choose_selection,
    confirm,
    gate_components,
    profile_sizes,
    project_shown,
    select_panel,
    slot_fraction,
    summarize,
)


class HiddenTrap(dict):
    def __getitem__(self, key):
        if key.startswith("held_out") or key == "program":
            raise AssertionError("solver projection accessed forbidden metadata")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key.startswith("held_out") or key == "program":
            raise AssertionError("solver projection accessed forbidden metadata")
        return super().get(key, default)


def test_solver_projection_never_reads_hidden_or_witness():
    raw = HiddenTrap(
        behaviour="a",
        skeleton="alpha",
        split="train",
        inputs=["true"],
        outputs=["1"],
        held_out_inputs=object(),
        held_out_outputs=object(),
        program=object(),
    )
    shown = project_shown(raw, 7)
    assert shown == ShownTask(7, "a", "alpha", ("true",), ("1",))
    assert not hasattr(shown, "program")
    assert not hasattr(shown, "held_out_inputs")


@pytest.mark.parametrize(
    "change",
    [
        {"split": "eval"},
        {"behaviour": ""},
        {"skeleton": None},
        {"inputs": []},
        {"outputs": ["1", "2"]},
        {"inputs": [True]},
    ],
)
def test_projection_rejects_malformed_shown_fields(change):
    raw = {
        "behaviour": "a",
        "skeleton": "alpha",
        "split": "train",
        "inputs": ["0"],
        "outputs": ["1"],
    }
    raw.update(change)
    with pytest.raises(ValueError):
        project_shown(raw, 0)


def test_frozen_hash_order_original_group_and_first_source_identity():
    # Literal independently computed hash ordering:epsilon,zeta,beta,theta,...
    groups = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")
    tasks = [
        ShownTask(i, group, group, ("0",), ("1",)) for i, group in enumerate(groups)
    ]
    tasks.extend(
        [
            ShownTask(20, "a", "epsilon", ("20",), ("21",)),
            ShownTask(21, "b", "epsilon", ("21",), ("22",)),
            ShownTask(22, "a", "epsilon", ("22",), ("23",)),
        ]
    )
    selected = select_panel(
        reversed(tasks), frozenset(t.behaviour for t in tasks), size=4
    )
    assert [row.task.group for row in selected] == ["epsilon", "zeta", "beta", "theta"]
    assert [row.half for row in selected] == [
        "selection",
        "confirmation",
        "selection",
        "confirmation",
    ]
    # Test representative digest and minimum source identity independently.
    selected_a = select_panel(
        tasks, frozenset({"a", "b", "zeta", "beta", "theta"}), size=4
    )
    assert selected_a[0].task.behaviour == "a"
    assert selected_a[0].task.source_index == 20
    assert selected_a[0].task.inputs == ("20",)


def test_original_validation_is_not_side_witness_shape():
    # Independent literal: SHA256("validation-4") % 10 ==0; alpha is8.
    tasks = [
        ShownTask(i, f"b{i}", group, ("0",), ("1",))
        for i, group in enumerate(("alpha", "beta", "validation-4"))
    ]
    with pytest.raises(ValueError, match="INSUFFICIENT_SELECTION"):
        select_panel(tasks, frozenset({"b0", "b2"}), size=2)
    selected = select_panel(tasks, frozenset({"b0", "b1", "b2"}), size=2)
    assert {row.task.group for row in selected} == {"alpha", "beta"}
    broken = [tasks[0], replace(tasks[0], source_index=9, skeleton="other")]
    with pytest.raises(ValueError, match="inconsistent original"):
        select_panel(broken, frozenset({"b0"}), size=2)


def test_empty_original_skeleton_falls_back_to_behaviour():
    tasks = [
        ShownTask(0, "alpha", "", ("0",), ("1",)),
        ShownTask(1, "beta", "", ("0",), ("1",)),
    ]
    selected = select_panel(tasks, frozenset({"alpha", "beta"}), size=2)
    assert [row.task.group for row in selected] == ["beta", "alpha"]


@pytest.mark.parametrize("size", [0, -2, 3, True])
def test_panel_count_is_not_silently_refilled(size):
    with pytest.raises(ValueError, match="panel size"):
        select_panel([], frozenset(), size=size)


def test_float64_band_test_does_not_round_down_float32_boundary():
    layer = Snapshot((np.float32(0.1), np.float32(-0.15)))
    assert isinstance(layer.scores[0], np.float32)
    # float32 subtraction rounds the threshold down to exactly the second score.
    assert layer.scores[1] >= np.float32(layer.scores[0] - np.float32(0.25))
    assert band_size(layer, 0.25) == 1


def test_inclusive_ties_minus_infinity_and_all_inf_fallback():
    assert band_size(Snapshot((0.0, -0.25, -0.25, -float("inf"))), 0.25) == 3
    assert band_size(Snapshot((-float("inf"), -float("inf"))), 0.25) == 2
    assert band_size(Snapshot(()), 1) == 0


@pytest.mark.parametrize("scores", [(float("nan"),), (float("inf"),), (-1.0, 0.0)])
def test_bad_recorded_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="score"):
        Snapshot(scores)


@pytest.mark.parametrize("rank", [0, 2, True, -1])
def test_invalid_recorded_ancestor_is_not_ignored(rank):
    with pytest.raises(ValueError, match="ancestor rank"):
        Snapshot((0.0,), rank)


def test_empty_retained_denominator_and_unscored_cells_both_count():
    empty = Cell("empty", "selection", 0, "frontier_exhausted", (Snapshot(()),))
    unscored = Cell("root-fit", "selection", 0, "fit", ())
    for cell in (empty, unscored):
        assert slot_fraction(cell, profile_sizes(cell, 0.25)) == 1
        assert slot_fraction(cell, profile_sizes(cell, None)) == 1
        assert not cell.informative


def test_equal_group_weighting_includes_failures_and_final_layers():
    big = Cell("big", "selection", 0, "budget", (Snapshot((0.0,) + (-20.0,) * 63),))
    small = Cell("small", "selection", 0, "max_depth", (Snapshot((0.0,)),) * 10)
    stats = summarize([big, small], 0.25)
    assert stats.band_slots == Fraction(65, 128)
    assert stats.prefix_slots == Fraction(3, 4)
    assert stats.band_slots != Fraction(11, 74)  # raw occurrence-weighted slots
    assert stats.informative == 0
    assert stats.band_protection is None


def test_ancestor_and_terminal_consistency():
    with pytest.raises(ValueError, match="ancestry"):
        Cell("x", "selection", 0, "fit", (Snapshot((0.0,)),))
    with pytest.raises(ValueError, match="ancestry"):
        Cell("x", "selection", 0, "budget", (Snapshot((0.0,), 1),))


@pytest.mark.parametrize("n", range(12, 25))
def test_all_protection_and_gain_boundaries_are_exact(n):
    for band in range(n + 1):
        for prefix in range(n + 1):
            stats = SeedStats(24, n, band, prefix, Fraction(1, 4), Fraction(1, 2))
            checks = gate_components(stats)
            assert checks["protection"] == (20 * band >= 19 * n)
            assert checks["gain"] == (10 * (band - prefix) >= n)
    # Includes20/20-18/20 and19/20-17/20, both exactly1/10.


def synthetic_panel(half, *, lose_seed=None, misses=4):
    result = []
    for seed in range(3):
        for group in range(24):
            hard = group < misses
            gap = 1.0 if seed == lose_seed else 0.5
            scores = (
                (0.0,) * 39 + (-gap,) + (-20.0,) * 24
                if hard
                else (0.0,) + (-20.0,) * 63
            )
            layer = Snapshot(scores, 40 if hard else 1)
            result.append(Cell(f"{half}-{group}", half, seed, "fit", (layer, layer)))
    return result


def test_smallest_selection_choice_then_one_frozen_confirmation():
    choice = choose_selection(synthetic_panel("selection"))
    assert choice == Choice("SELECTION_PROFILE_LOCKED", 0.5)
    terminal, checks = confirm(synthetic_panel("confirmation"), choice)
    assert terminal == "PASS_PROFILE_FEASIBILITY"
    assert all(all(row.values()) for row in checks.values())


def test_confirmation_failure_cannot_reselect_band_or_pool_seeds():
    choice = choose_selection(synthetic_panel("selection"))
    terminal, checks = confirm(synthetic_panel("confirmation", lose_seed=2), choice)
    assert terminal == "NO_CONFIRMATION_PROFILE"
    assert not checks[2]["gain"] and not checks[2]["protection"]
    assert all(checks[0].values()) and all(checks[1].values())
    assert choice.tau == 0.5


def test_no_headroom_is_distinct_from_no_slot_savings():
    cells = synthetic_panel("selection", misses=2)
    assert choose_selection(cells) == Choice("NO_HEADROOM_OVER_PREFIX32", None)
    stats = summarize([cell for cell in cells if cell.seed == 0], 0.5)
    assert stats.band_slots < stats.prefix_slots
    with pytest.raises(ValueError, match="no locked"):
        confirm(
            synthetic_panel("confirmation"), Choice("NO_HEADROOM_OVER_PREFIX32", None)
        )


def test_incomplete_population_and_cross_seed_group_substitution_rejected():
    cells = synthetic_panel("selection")
    with pytest.raises(ValueError, match="incomplete"):
        choose_selection(cells[:-1])
    changed = cells.copy()
    changed[-1] = replace(changed[-1], group="substituted-group")
    with pytest.raises(ValueError, match="population differs"):
        choose_selection(changed)
    with pytest.raises(ValueError, match="wrong-half"):
        by_seed(synthetic_panel("confirmation"), "selection")


def test_support_failure_does_not_become_zero_average_loss():
    cells = synthetic_panel("selection")
    cells = [
        replace(
            cell,
            termination="wall_clock_cap",
            layers=(Snapshot(cell.layers[0].scores),),
        )
        if int(cell.group.rsplit("-", 1)[1]) >= 11
        else cell
        for cell in cells
    ]
    assert choose_selection(cells) == Choice("INSUFFICIENT_LINEAGE_SUPPORT", None)


def test_malformed_summary_cannot_fake_threshold_success():
    with pytest.raises(ValueError, match="denominators"):
        SeedStats(24, 20, 21, 17, Fraction(1, 2), Fraction(1, 2))
    with pytest.raises(ValueError, match="exact bounded"):
        SeedStats(24, 20, 20, 17, 0.5, 0.5)
