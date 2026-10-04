"""Sealed-cell phase control and exact #180 reports, synthetic-tested only.

No input/model reader, trial entrypoint or independent result verifier. Actual
runner must supply cells ONLY from sealed load_cell outputs after all144 seals.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

from experiments.score_gap_profiles import (
    GRID,
    HALVES,
    SEEDS,
    Choice,
    by_seed,
    gate_components,
    profile_sizes,
    protected,
    slot_fraction,
    summarize,
)
from experiments.score_gap_storage import DiagnosticProvenanceError, canonical_json


def ratio(value: Fraction | None):
    return (
        None
        if value is None
        else {"numerator": value.numerator, "denominator": value.denominator}
    )


def _stats(cells, tau):
    stats = summarize(cells, tau)
    sizes = [profile_sizes(cell, tau) for cell in cells]
    total = sum(len(layer.scores) for cell in cells for layer in cell.layers)
    used = sum(sum(row) for row in sizes)
    prefix = sum(sum(profile_sizes(cell, None)) for cell in cells)
    depths = []
    for depth in range(max((len(cell.layers) for cell in cells), default=0)):
        present = [
            (cell, row)
            for cell, row in zip(cells, sizes, strict=True)
            if depth < len(cell.layers)
        ]
        reference = sum(len(cell.layers[depth].scores) for cell, _ in present)
        band = sum(row[depth] for _, row in present)
        fixed = sum(min(32, len(cell.layers[depth].scores)) for cell, _ in present)
        depths.append(
            {
                "source_depth": depth,
                "present_cells": len(present),
                "reference_slots": reference,
                "band_slots": band,
                "prefix_slots": fixed,
                "band_occurrence_weighted": ratio(
                    Fraction(band, reference) if reference else None
                ),
                "prefix_occurrence_weighted": ratio(
                    Fraction(fixed, reference) if reference else None
                ),
            }
        )
    return stats, {
        "groups": stats.groups,
        "informative": stats.informative,
        "band_protected": stats.band_protected,
        "prefix_protected": stats.prefix_protected,
        "band_protection": ratio(stats.band_protection),
        "prefix_protection": ratio(stats.prefix_protection),
        "gain": ratio(
            None
            if stats.band_protection is None
            else stats.band_protection - stats.prefix_protection
        ),
        "band_slots_equal_group": ratio(stats.band_slots),
        "prefix_slots_equal_group": ratio(stats.prefix_slots),
        "raw_band_slots": used,
        "raw_prefix_slots": prefix,
        "raw_reference_slots": total,
        "band_slots_occurrence_weighted": ratio(
            Fraction(used, total) if total else None
        ),
        "prefix_slots_occurrence_weighted": ratio(
            Fraction(prefix, total) if total else None
        ),
        "gate_components": gate_components(stats),
        "per_depth": depths,
    }


def _prefix(cells, width):
    informative = [cell for cell in cells if cell.informative]
    sizes = [profile_sizes(cell, None, width=width) for cell in informative]
    n = len(informative)
    protected_count = sum(
        protected(cell, row) for cell, row in zip(informative, sizes, strict=True)
    )
    return {
        "population": "informative_only",
        "informative": n,
        "protected": protected_count,
        "protection": ratio(Fraction(protected_count, n) if n else None),
        "slot_fraction_equal_group": ratio(
            sum(
                (
                    slot_fraction(cell, row)
                    for cell, row in zip(informative, sizes, strict=True)
                ),
                Fraction(),
            )
            / n
            if n
            else None
        ),
        "raw_profile_slots": sum(sum(row) for row in sizes),
        "raw_reference_slots": sum(
            len(layer.scores) for cell in informative for layer in cell.layers
        ),
    }


def _gap(layer, score):
    anchor, saved = float(layer.scores[0]), float(score)
    if anchor == -math.inf:
        return {"kind": "all_minus_inf_fallback"}
    if saved == -math.inf:
        return {"kind": "plus_inf"}
    return {"kind": "finite", "value": anchor - saved}


def analyze(cells, group_order, receipts, budget, output: Path):
    """Both-half support first; lock once; no unchosen confirmation profile.

    group_order is the48-group order from the pre-model selection seal. This
    generic helper cannot establish that seal's source provenance itself.
    """
    budget.check()
    receipts._active()
    expected = [
        (group, HALVES[position % 2], seed)
        for position, group in enumerate(group_order)
        for seed in SEEDS
    ]
    if (
        len(group_order) != 48
        or len(set(group_order)) != 48
        or [(cell.group, cell.half, cell.seed) for cell in cells] != expected
    ):
        raise DiagnosticProvenanceError("cell order differs from sealed48-group panel")
    grouped = {
        half: by_seed(tuple(cell for cell in cells if cell.half == half), half)
        for half in HALVES
    }
    support = {
        half: {
            str(seed): sum(cell.informative for cell in rows)
            for seed, rows in grouped[half].items()
        }
        for half in HALVES
    }
    support_ok = all(n >= 12 for row in support.values() for n in row.values())
    digest, size = hashlib.sha256(), 0
    selected_stats = {}
    with Path(output).open("xb") as stream:

        def emit(row):
            nonlocal size
            payload = canonical_json(row)
            budget.write(stream, payload)
            digest.update(payload)
            size += len(payload)

        emit({"kind": "support", "support": support, "both_halves_pass": support_ok})
        # Descriptive selection profiles remain even on support/headroom failure.
        for tau in GRID:
            selected_stats[tau] = {}
            for seed in SEEDS:
                budget.check()
                stats, report = _stats(grouped["selection"][seed], tau)
                selected_stats[tau][seed] = stats
                emit(
                    {
                        "kind": "selection_profile",
                        "half": "selection",
                        "seed": seed,
                        "tau": tau,
                        **report,
                    }
                )
        if not support_ok:
            choice = Choice("INSUFFICIENT_LINEAGE_SUPPORT", None)
        elif any(
            stats.prefix_protection > Fraction(9, 10)
            for stats in selected_stats[GRID[0]].values()
        ):
            choice = Choice("NO_HEADROOM_OVER_PREFIX32", None)
        else:
            tau = next(
                (
                    tau
                    for tau in GRID
                    if all(
                        all(gate_components(stats).values())
                        for stats in selected_stats[tau].values()
                    )
                ),
                None,
            )
            choice = Choice(
                "NO_SELECTION_PROFILE" if tau is None else "SELECTION_PROFILE_LOCKED",
                tau,
            )
        stream.flush()
        os.fsync(stream.fileno())
        budget.check()
        selection = {
            "terminal": choice.terminal,
            "tau": choice.tau,
            "support": support,
            "selection_prefix_bytes": size,
            "selection_prefix_sha256": digest.hexdigest(),
        }
        receipts.selection(selection)
        budget.check()
        emit({"kind": "selection_lock", **selection})
        terminal = choice.terminal
        checks = {}
        if choice.tau is not None:
            for seed in SEEDS:
                budget.check()
                stats, report = _stats(grouped["confirmation"][seed], choice.tau)
                checks[str(seed)] = gate_components(stats)
                emit(
                    {
                        "kind": "confirmation_profile",
                        "half": "confirmation",
                        "seed": seed,
                        "tau": choice.tau,
                        "prefix_has_gain_headroom": stats.prefix_protection
                        <= Fraction(9, 10),
                        **report,
                    }
                )
            terminal = (
                "PASS_PROFILE_FEASIBILITY"
                if all(all(row.values()) for row in checks.values())
                else "NO_CONFIRMATION_PROFILE"
            )
        # Threshold-independent confirmation descriptives open only AFTER lock.
        for half in HALVES:
            for seed, rows in grouped[half].items():
                budget.check()
                counts = Counter(cell.termination for cell in rows)
                emit(
                    {
                        "kind": "population",
                        "half": half,
                        "seed": seed,
                        "groups": len(rows),
                        "terminals": dict(counts),
                        "informative": sum(cell.informative for cell in rows),
                        "uninformative_fits": sum(
                            cell.termination == "fit" and not cell.informative
                            for cell in rows
                        ),
                        "no_scored_layers": sum(not cell.layers for cell in rows),
                        "only_empty_beams": sum(
                            bool(cell.layers)
                            and not any(layer.scores for layer in cell.layers)
                            for cell in rows
                        ),
                    }
                )
                for width in range(1, 65):
                    budget.check()
                    emit(
                        {
                            "kind": "prefix_curve",
                            "half": half,
                            "seed": seed,
                            "width": width,
                            **_prefix(rows, width),
                        }
                    )
                for cell in rows:
                    for source_depth, layer in enumerate(cell.layers):
                        budget.check()
                        emit(
                            {
                                "kind": "retained_layer",
                                "half": half,
                                "seed": seed,
                                "group": cell.group,
                                "source_depth": source_depth,
                                "retained_depth": source_depth + 1,
                                "C": len(layer.scores),
                                "ancestor_rank": layer.ancestor_rank,
                                "informative_cell": cell.informative,
                                "rank_gap": [
                                    {
                                        "rank": rank,
                                        "known_ancestor": rank == layer.ancestor_rank,
                                        "gap": _gap(layer, score),
                                    }
                                    for rank, score in enumerate(layer.scores, 1)
                                ],
                            }
                        )
        emit(
            {
                "kind": "terminal",
                "terminal": terminal,
                "tau": choice.tau,
                "confirmation_checks": checks,
                "confirmation_profile_opened": choice.tau is not None,
            }
        )
        stream.flush()
        os.fsync(stream.fileno())
        budget.check()
    return {
        "terminal": terminal,
        "tau": choice.tau,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "confirmation_checks": checks,
    }
