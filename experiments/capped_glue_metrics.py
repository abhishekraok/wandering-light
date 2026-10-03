"""Encoder exposure and raw-logit diagnostics for #178, D1 + Amendment A.

Pure diagnostics: no checkpoint loading, training, calibration or solver call.
Lexical emission labels do not attribute runtime work or imply causal effects.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from statistics import fmean, median
from typing import TYPE_CHECKING

import numpy as np

from experiments import capped_glue_loss as core
from wandering_light.stack_policy import (
    DIFFERENCE,
    ENCODINGS,
    OP_INDEX,
    OP_NAMES,
    STATE_CAP,
    TARGET_CAP,
    encode_stack,
    flags,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.stack_beam import Product


def encoder_diagnostics(product: Product, target: Product) -> dict:
    """Full token widths before caps; difference flags compare full stacks."""
    if not product or len(product) != len(target):
        raise core.InvalidDiagnosticError(
            "encoder requires equal nonempty shown products"
        )
    examples = []
    for state, wanted in zip(product, target, strict=True):
        state_width = len(encode_stack(state, sys.maxsize))
        target_width = len(encode_stack(wanted, sys.maxsize))
        examples.append(
            {
                "state_pre_cap_width": state_width,
                "target_pre_cap_width": target_width,
                "state_retained_width": min(state_width, STATE_CAP),
                "target_retained_width": min(target_width, TARGET_CAP),
                "state_truncated": state_width > STATE_CAP,
                "target_truncated": target_width > TARGET_CAP,
                "difference_flags": len(flags(state, wanted)),
            }
        )
    layouts = {}
    for encoding in ENCODINGS:
        widths = [
            row["state_retained_width"]
            + 1
            + row["target_retained_width"]
            + (1 + row["difference_flags"] if encoding == DIFFERENCE else 0)
            for row in examples
        ]
        layouts[encoding] = {"item_widths": widths, "block_width": max(widths)}
    return {
        "state_cap": STATE_CAP,
        "target_cap": TARGET_CAP,
        "examples": examples,
        "layouts": layouts,
        "any_state_truncated": any(row["state_truncated"] for row in examples),
        "any_target_truncated": any(row["target_truncated"] for row in examples),
        "training_row_multiplicity": None,
        "training_row_multiplicity_status": "unavailable_not_reconstructed",
    }


def loss_metrics(logits: Sequence[float], legal: Sequence[bool], witness: str) -> dict:
    """Unmasked training CE primary; bounded legal-mask secondaries.

    Rank counts strictly greater legal logits; ties share the best rank.
    Independent peak shifts avoid overflow and illegal-peak underflow.
    """
    values = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(legal)
    if values.shape != (len(OP_NAMES),) or not np.isfinite(values).all():
        raise core.InvalidDiagnosticError("expected a finite raw 28-op logit vector")
    if mask.shape != values.shape or mask.dtype != np.dtype(bool):
        raise core.InvalidDiagnosticError("expected a Boolean 28-op legal vector")
    if witness not in OP_INDEX or not mask[OP_INDEX[witness]]:
        raise core.InvalidDiagnosticError(
            "witness is unknown or illegal in diagnostic mask"
        )
    want = float(values[OP_INDEX[witness]])
    all_peak = float(values.max())
    legal_peak = float(values[mask].max())
    all_shift_norm = float(np.log(np.exp(values - all_peak).sum()))
    legal_shift_norm = float(np.log(np.exp(values[mask] - legal_peak).sum()))
    result = {
        "unmasked_nll": all_shift_norm + (all_peak - want),
        "legal_nll": legal_shift_norm + (legal_peak - want),
        "negative_log_legal_mass": all_shift_norm
        - legal_shift_norm
        + (all_peak - legal_peak),
        "best_legal_deficit": legal_peak - want,
        "legal_rank": 1 + int(np.count_nonzero(values[mask] > want)),
        "legal_size": int(mask.sum()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise core.InvalidDiagnosticError("nonfinite derived diagnostic")
    return result


def _cells(stage: Mapping) -> tuple[core.MatchedCell, ...]:
    return tuple(
        core.MatchedCell(
            row["behaviour"],
            row["group"],
            row["op"],
            row["position"],
            tuple(tuple(key) for key in row["top"]),
            tuple(tuple(key) for key in row["body"]),
        )
        for row in stage["cells"]
    )


def _descriptive(rows: Sequence[Mapping], metrics: Mapping) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["side"],
            "structural" if row["structural"] else "content",
            row["position"],
            row["op"],
            row["depth"],
        )
        grouped[key].append(row)
    labels = []
    for (side, role, position, op, depth), group in sorted(grouped.items()):
        cell = {
            "side": side,
            "class": role,
            "position": position,
            "op": op,
            "depth": depth,
            "rows": len(group),
        }
        for metric in (
            "unmasked_nll",
            "legal_nll",
            "negative_log_legal_mass",
            "best_legal_deficit",
            "legal_rank",
            "legal_size",
        ):
            values = [metrics[row["behaviour"], row["step"]][metric] for row in group]
            cell[metric] = {"sum": math.fsum(values), "mean": fmean(values)}
        cell.update(
            resource_affected_rows=sum(
                row["legality"]["resource_affected"] for row in group
            ),
            state_truncated_rows=sum(
                row["encoder_diagnostics"]["any_state_truncated"] for row in group
            ),
            target_truncated_rows=sum(
                row["encoder_diagnostics"]["any_target_truncated"] for row in group
            ),
        )
        labels.append(cell)
    return {"rows": len(rows), "by_label": labels}


def _top_metric_share(
    rows: Sequence[Mapping], metrics: Mapping, metric: str
) -> float | None:
    denominator = math.fsum(
        metrics[row["behaviour"], row["step"]][metric] for row in rows
    )
    numerator = math.fsum(
        metrics[row["behaviour"], row["step"]][metric]
        for row in rows
        if row["side"] == "top"
    )
    return numerator / denominator if denominator else None


def summarize_checkpoint(
    stage: Mapping, scored: Sequence[Mapping], *, encoding: str, seed: int
) -> dict:
    """Recompute metrics from complete raw logits; equal cells/behaviours primary."""
    if encoding not in ENCODINGS or seed not in (0, 1, 2):
        raise core.InvalidDiagnosticError("checkpoint descriptor is outside D1")
    if (
        not stage["support"]["passes"]
        or stage["kind"] != "supported"
        or stage["complete"] is not True
    ):
        raise core.InvalidDiagnosticError(
            "no supported sealed construction to summarize"
        )
    expected = {(row["behaviour"], row["step"]) for row in stage["rows"]}
    raw = {}
    for row in scored:
        key = row["behaviour"], row["step"]
        if key in raw:
            raise core.InvalidDiagnosticError("duplicate checkpoint logit identity")
        raw[key] = row["logits"]
    if set(raw) != expected or len(expected) != len(stage["rows"]):
        raise core.InvalidDiagnosticError(
            "checkpoint logits do not cover exact Stage A rows"
        )
    metrics = {
        (row["behaviour"], row["step"]): loss_metrics(
            raw[row["behaviour"], row["step"]],
            row["legality"]["mask"],
            row["op"],
        )
        for row in stage["rows"]
    }
    losses = {key: value["unmasked_nll"] for key, value in metrics.items()}
    cells = _cells(stage)
    primary = core.aggregate_cells(cells, losses)
    ordinary_cells = tuple(cell for cell in cells if not cell.assembly)
    without_assembly = (
        core.aggregate_cells(ordinary_cells, losses) if ordinary_cells else None
    )
    shares = []
    behaviours = defaultdict(list)
    for row in stage["rows"]:
        behaviours[row["behaviour"]].append(row)
    for behaviour, rows in sorted(behaviours.items()):
        top = [row for row in rows if row["side"] == "top"]
        shares.append(
            {
                "behaviour": behaviour,
                "rows": len(rows),
                "top_rows": len(top),
                "top_token_share": len(top) / len(rows),
                "top_unmasked_loss_share": _top_metric_share(
                    rows, metrics, "unmasked_nll"
                ),
                "top_legal_deficit_share": _top_metric_share(
                    rows, metrics, "best_legal_deficit"
                ),
                "top_legal_nll_share": _top_metric_share(rows, metrics, "legal_nll"),
                "resource_affected_rows": sum(
                    row["legality"]["resource_affected"] for row in rows
                ),
            }
        )
    return {
        "encoding": encoding,
        "seed": seed,
        "primary": primary,
        "without_assembly_descriptive": without_assembly,
        "all_rows": _descriptive(stage["rows"], metrics),
        "common_support_rows": _descriptive(
            [row for row in stage["rows"] if row["matched"]], metrics
        ),
        "behaviour_loss_and_exposure": shares,
        "per_row_metrics": [
            {
                "behaviour": row["behaviour"],
                "step": row["step"],
                **metrics[row["behaviour"], row["step"]],
                "resource_affected": row["legality"]["resource_affected"],
            }
            for row in stage["rows"]
        ],
        "rank_ties": "strictly_greater_legal_logits",
        "interpretation": "descriptive_midspan_exact_op_emissions_only",
    }


def six_checkpoint_reading(summaries: Sequence[Mapping]) -> dict:
    """Frozen both-encoding gate; seeds are replicates, not extra behaviours."""
    expected = [(encoding, seed) for encoding in ENCODINGS for seed in range(3)]
    if [(row["encoding"], row["seed"]) for row in summaries] != expected:
        raise core.InvalidDiagnosticError(
            "missing/duplicate/reordered checkpoint summaries"
        )
    encodings = {}
    for encoding in ENCODINGS:
        group = [row for row in summaries if row["encoding"] == encoding]
        deltas = [row["primary"]["delta"] for row in group]
        fractions = [row["primary"]["positive_behaviour_fraction"] for row in group]
        if not all(math.isfinite(value) for value in (*deltas, *fractions)) or not all(
            0 <= value <= 1 for value in fractions
        ):
            raise core.InvalidDiagnosticError("nonfinite/invalid checkpoint primary")
        encodings[encoding] = {
            "seed_deltas": deltas,
            "seed_positive_behaviour_fractions": fractions,
            "median_delta": median(deltas),
            "positive_seeds": sum(value > 0 for value in deltas),
            "median_positive_behaviour_fraction": median(fractions),
            "passes": median(deltas) >= 0.25
            and sum(value > 0 for value in deltas) >= 2
            and median(fractions) >= 0.60,
        }
    return {
        "encodings": encodings,
        "reading": "TOP_LOSS_ROUTE"
        if all(row["passes"] for row in encodings.values())
        else "NO_MATCHED_MIDSPAN_TOP_EXCESS",
    }
