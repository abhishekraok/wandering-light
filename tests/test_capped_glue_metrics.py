"""Synthetic states, analytic logits and newly instantiated fixture policies."""

import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import capped_glue_loss as core
from experiments import capped_glue_metrics as metrics
from experiments import capped_glue_stage_a as stage_a
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym
from wandering_light.stack_pbe import Task
from wandering_light.stack_policy import (
    ENCODINGS,
    OP_INDEX,
    OP_NAMES,
    STATE_CAP,
    TARGET_CAP,
    PolicyScorer,
    StackPairPolicy,
    encode_node,
    encode_stack,
    validation_skeleton,
)


def test_full_token_width_counts_typed_values_and_flags_not_printed_characters():
    product = ((Int(3), Lst((Int(-100), Int(1))), Mark(), Sym("add"), Int(7)), ())
    target = ((Int(3),), ())
    report = metrics.encoder_diagnostics(product, target)
    assert report["examples"][0]["state_pre_cap_width"] == 12
    assert report["examples"][0]["target_pre_cap_width"] == 1
    assert report["examples"][0]["difference_flags"] == 5
    assert report["examples"][1]["state_pre_cap_width"] == 0
    assert report["layouts"]["machine"]["item_widths"] == [14, 1]
    assert report["layouts"]["difference"]["item_widths"] == [20, 2]
    assert report["training_row_multiplicity"] is None
    for encoding in ENCODINGS:
        assert (
            encode_node(product, target, encoding).shape[1]
            == report["layouts"][encoding]["block_width"]
        )


def test_truncation_keeps_existing_top_caps_and_flags_compare_full_stacks():
    product = (tuple(Int(1) for _ in range(STATE_CAP + 3)),)
    target = (tuple(Int(2) for _ in range(TARGET_CAP + 5)),)
    report = metrics.encoder_diagnostics(product, target)
    (example,) = report["examples"]
    assert (
        example["state_pre_cap_width"] == 163 and example["state_retained_width"] == 160
    )
    assert (
        example["target_pre_cap_width"] == 69 and example["target_retained_width"] == 64
    )
    assert report["any_state_truncated"] and report["any_target_truncated"]
    assert example["difference_flags"] == 8
    assert report["layouts"]["machine"]["block_width"] == 225
    assert report["layouts"]["difference"]["block_width"] == 234
    assert encode_stack(product[0], STATE_CAP) == encode_stack(
        product[0][3:], STATE_CAP
    )
    for encoding in ENCODINGS:
        assert (
            encode_node(product, target, encoding).shape[1]
            == report["layouts"][encoding]["block_width"]
        )


def test_exact_caps_are_not_truncation():
    report = metrics.encoder_diagnostics(
        ((Int(0),) * STATE_CAP,), ((Int(0),) * TARGET_CAP,)
    )
    assert not report["any_state_truncated"] and not report["any_target_truncated"]


def test_unmasked_loss_matches_training_cross_entropy_with_full_op_support():
    logits = np.linspace(-2, 3, 28)
    mask = [True] * 28
    out = metrics.loss_metrics(logits, mask, "dup")
    reference = torch.nn.functional.cross_entropy(
        torch.tensor(logits)[None, :], torch.tensor([OP_INDEX["dup"]])
    ).item()
    assert out["unmasked_nll"] == pytest.approx(reference, abs=1e-12)
    assert out["legal_nll"] == out["unmasked_nll"]
    assert out["negative_log_legal_mass"] == 0


def test_uniform_logits_have_analytic_legal_mass_decomposition():
    logits = [0.0] * 28
    legal = [name in ("dup", "drop") for name in OP_NAMES]
    out = metrics.loss_metrics(logits, legal, "dup")
    assert out["unmasked_nll"] == pytest.approx(math.log(28))
    assert out["legal_nll"] == pytest.approx(math.log(2))
    assert out["negative_log_legal_mass"] == pytest.approx(math.log(14))
    assert out["unmasked_nll"] == pytest.approx(
        out["legal_nll"] + out["negative_log_legal_mass"]
    )
    assert out["best_legal_deficit"] == 0 and out["legal_rank"] == 1
    assert out["legal_size"] == 2


def test_illegal_peak_does_not_underflow_the_legal_normalizer_and_offsets_cancel():
    logits = np.zeros(28)
    logits[OP_INDEX["dup"]] = -2
    logits[OP_INDEX["drop"]] = -1
    logits[OP_INDEX["i"]] = 1000
    legal = [name in ("dup", "drop") for name in OP_NAMES]
    out = metrics.loss_metrics(logits, legal, "dup")
    assert out["unmasked_nll"] == pytest.approx(1002)
    assert out["legal_nll"] == pytest.approx(math.log1p(math.e))
    assert out["best_legal_deficit"] == 1 and out["legal_rank"] == 2
    assert out["unmasked_nll"] == pytest.approx(
        out["legal_nll"] + out["negative_log_legal_mass"]
    )
    shifted = metrics.loss_metrics(logits + 10000, legal, "dup")
    assert shifted == pytest.approx(out, abs=1e-10)


@pytest.mark.parametrize(
    "bad", [[0.0] * 27, [0.0] * 29, [float("nan")] * 28, [float("inf")] * 28]
)
def test_logit_errors_are_invalid(bad):
    with pytest.raises(core.InvalidDiagnosticError, match="finite raw"):
        metrics.loss_metrics(bad, [True] * 28, "dup")


def test_nonboolean_mask_and_refused_witness_are_invalid():
    with pytest.raises(core.InvalidDiagnosticError, match="Boolean"):
        metrics.loss_metrics([0.0] * 28, [1] * 28, "dup")
    mask = [True] * 28
    mask[OP_INDEX["dup"]] = False
    with pytest.raises(core.InvalidDiagnosticError, match="illegal"):
        metrics.loss_metrics([0.0] * 28, mask, "dup")


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_fixture_policy_batch_one_and_32_agree(encoding):
    # Random fixture model only: no training or existing checkpoint loaded.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(178)
            scorer = PolicyScorer(
                StackPairPolicy(PolicyArchitecture(8, 16, 32), encoding=encoding), "cpu"
            )
        blocks = []
        for row in range(32):
            examples = 1 + row % 3
            state = ((Int(row),) * (170 if row == 31 else 1 + row % 5),) * examples
            target = ((Lst((Sym("dup"), Int(1))), Bool(True)),) * examples
            blocks.append(encode_node(state, target, encoding))
        singles = np.stack([scorer.logits([block])[0] for block in blocks])
        batched = scorer.logits(blocks)
        assert singles.shape == batched.shape == (32, 28)
        np.testing.assert_allclose(singles, batched, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def supported_stage():
    tasks = []
    index = 0
    while len(tasks) < 20:
        task = Task(
            behaviour=f"b{index}",
            summary="synthetic",
            split="train",
            task_hash_split="train",
            inputs=("2",),
            outputs=("2",),
            held_out_inputs=(),
            held_out_outputs=(),
            program=("dup", "drop"),
            length=2,
            examples=1,
            held_out=0,
            fuel=0,
            tags=(),
            skeleton=f"g{index}",
        )
        if not validation_skeleton(task):
            tasks.append(task)
        index += 1
    programs = {
        task.behaviour: "dup dup [ dup ] i drop drop [ drop ] i" for task in tasks
    }
    return stage_a.build_stage_a(tasks, programs, {}, source_revision="fixture")


def raw_rows(stage, top_penalty=0):
    out = []
    for row in stage["rows"]:
        logits = [0.0] * 28
        if row["side"] == "top":
            logits[OP_INDEX[row["op"]]] = -top_penalty
        out.append(
            {"behaviour": row["behaviour"], "step": row["step"], "logits": logits}
        )
    return out


def test_summary_uses_complete_raw_logits_and_keeps_population_and_exposure_separate(
    supported_stage,
):
    stage = supported_stage
    out = metrics.summarize_checkpoint(
        stage, raw_rows(stage), encoding="machine", seed=0
    )
    assert (
        len(out["primary"]["cells"]) == 40 and len(out["primary"]["behaviours"]) == 20
    )
    assert (
        out["primary"]["delta"] == 0
        and out["primary"]["positive_behaviour_fraction"] == 0
    )
    assert out["all_rows"]["rows"] == 240 and out["common_support_rows"]["rows"] == 80
    assert len(out["per_row_metrics"]) == 240
    assert all(
        row["top_token_share"] == 2 / 3 for row in out["behaviour_loss_and_exposure"]
    )
    assert all(
        row["top_legal_deficit_share"] is None
        for row in out["behaviour_loss_and_exposure"]
    )
    assert out["without_assembly_descriptive"]["delta"] == out["primary"]["delta"]
    body = [row for row in stage["rows"] if row["side"] == "body"]
    lookup = {(row["behaviour"], row["step"]): row for row in out["per_row_metrics"]}
    assert all(
        lookup[row["behaviour"], row["step"]]["negative_log_legal_mass"] == 0
        for row in body
    )


def test_secondary_resource_flags_do_not_change_primary(supported_stage):
    stage = supported_stage
    raw = raw_rows(stage, top_penalty=1)
    ordinary = metrics.summarize_checkpoint(stage, raw, encoding="machine", seed=0)
    modified = deepcopy(stage)
    modified["rows"][0]["legality"]["resource_affected"] = True
    flagged = metrics.summarize_checkpoint(modified, raw, encoding="machine", seed=0)
    assert flagged["primary"] == ordinary["primary"]
    assert flagged["primary"]["delta"] > 0.25
    assert sum(row["resource_affected"] for row in flagged["per_row_metrics"]) == 1


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "nonfinite", "wrong_identity"]
)
def test_partial_bad_or_duplicate_logits_cannot_produce_summary(
    supported_stage, change
):
    raw = raw_rows(supported_stage)
    if change == "missing":
        raw.pop()
    elif change == "duplicate":
        raw[-1] = raw[0]
    elif change == "nonfinite":
        raw[-1]["logits"][0] = float("nan")
    else:
        raw[-1]["behaviour"] = "outside"
    with pytest.raises(core.InvalidDiagnosticError):
        metrics.summarize_checkpoint(supported_stage, raw, encoding="machine", seed=0)


def summaries(delta=0.25, fraction=0.60):
    return [
        {
            "encoding": encoding,
            "seed": seed,
            "primary": {"delta": delta, "positive_behaviour_fraction": fraction},
        }
        for encoding in ENCODINGS
        for seed in range(3)
    ]


def test_frozen_gate_needs_each_encoding_at_full_precision_and_keeps_disagreement():
    positive = metrics.six_checkpoint_reading(summaries())
    assert positive["reading"] == "TOP_LOSS_ROUTE"
    assert all(row["median_delta"] == 0.25 for row in positive["encodings"].values())
    changed = summaries()
    changed[3]["primary"]["delta"] = changed[4]["primary"]["delta"] = 0.249999
    negative = metrics.six_checkpoint_reading(changed)
    assert negative["reading"] == "NO_MATCHED_MIDSPAN_TOP_EXCESS"
    assert negative["encodings"]["machine"]["passes"]
    assert not negative["encodings"]["difference"]["passes"]
    changed = summaries(fraction=0.599999)
    assert (
        metrics.six_checkpoint_reading(changed)["reading"]
        == "NO_MATCHED_MIDSPAN_TOP_EXCESS"
    )


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "order", "nonfinite", "fraction"]
)
def test_checkpoint_gate_rejects_incompleteness_and_bad_primaries(change):
    rows = summaries()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "order":
        rows.reverse()
    elif change == "nonfinite":
        rows[0]["primary"]["delta"] = float("nan")
    else:
        rows[0]["primary"]["positive_behaviour_fraction"] = 1.1
    with pytest.raises(core.InvalidDiagnosticError):
        metrics.six_checkpoint_reading(rows)
