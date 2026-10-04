"""Manually typed production-search units, never a declared gate or real panel."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.score_gap_lineage import (
    SyntheticCapture,
    reconstruct_lineage,
    replay_states,
)
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, parse_stack
from wandering_light.stack_pbe import BASIS, OPS, Task


def task(inputs=("2 3",), outputs=("12",)):
    return Task(
        behaviour="synthetic-lineage",
        summary="manual unit",
        split="train",
        task_hash_split="train",
        inputs=inputs,
        outputs=outputs,
        held_out_inputs=(),
        held_out_outputs=(),
        program=(),
        length=0,
        examples=len(inputs),
        held_out=0,
        fuel=0,
        tags=(),
    )


class ScriptedScorer:
    def __init__(self):
        self.calls = 0

    def log_probs(self, blocks, legal):
        scores = np.full(legal.shape, -20.0, dtype=np.float32)
        op = "mul" if self.calls == 0 else "dup"
        scores[0, next(i for i, item in enumerate(OPS) if item.name == op)] = 0.0
        self.calls += 1
        return np.where(legal, scores, np.nan)


def observed(*, manual=None, max_depth=3, budget=None):
    manual = manual or task()
    capture = SyntheticCapture()
    result = beam_solve(
        manual,
        ScriptedScorer(),
        beam_width=8,
        max_depth=max_depth,
        guard="off",
        observer=capture,
        budget=budget,
    )
    initial = tuple(parse_stack(text, basis=BASIS) for text in manual.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in manual.outputs)
    return capture, initial, outputs, result


def test_exact_actual_parent_chain_and_float32_scores():
    capture, initial, outputs, result = observed()
    assert result.program == ("mul", "dup", "add")
    layers = reconstruct_lineage(capture, initial, outputs, result)
    assert len(layers) == 2
    assert [layer.ancestor_rank for layer in layers] == [1, 1]
    assert all(
        isinstance(score, np.float32) for layer in layers for score in layer.scores
    )


def test_unscored_first_fit_is_not_zero_loss_evidence():
    capture, initial, outputs, result = observed(manual=task(outputs=("6",)))
    assert result.program == ("mul",)
    assert not capture.scored
    assert reconstruct_lineage(capture, initial, outputs, result) == ()


def test_missing_last_unexpanded_retention_rejected():
    capture, initial, outputs, result = observed(max_depth=2)
    assert result.termination == "max_depth"
    assert len(reconstruct_lineage(capture, initial, outputs, result)) == 2
    capture.retained.pop()
    with pytest.raises(ValueError, match="exact scored/retained"):
        reconstruct_lineage(capture, initial, outputs, result)


def test_parent_index_mutation_rejected():
    capture, initial, outputs, result = observed()
    layer = capture.retained[-1]
    parents = list(layer.parents)
    parents[0] = (99, parents[0][1])
    capture.retained[-1] = replace(layer, parents=tuple(parents))
    with pytest.raises(ValueError, match="invalid recorded parent"):
        reconstruct_lineage(capture, initial, outputs, result)


def test_returned_prefix_needs_actual_typed_parent_not_equal_display():
    capture, initial, outputs, result = observed()
    terminal = list(capture.starts[-1][1])
    original = terminal[0]
    terminal[0] = (((Bool(True),),), original[1], original[2])
    capture.starts[-1] = (capture.starts[-1][0], tuple(terminal))
    with pytest.raises(ValueError, match="next started beam"):
        reconstruct_lineage(capture, initial, outputs, result)


def test_wrong_target_bool_int_and_failed_continuation_rejected():
    capture, initial, _, result = observed()
    with pytest.raises(ValueError, match="misses shown"):
        reconstruct_lineage(capture, initial, ((Bool(True),),), result)
    with pytest.raises(ValueError, match="refused"):
        replay_states(("mul",), ((Bool(True), Int(1)),))


def test_exact_score_dtype_bits_and_parent_legal_row_checked():
    capture, initial, outputs, result = observed()
    scores, legal = capture.scored[0]
    mul = next(i for i, op in enumerate(OPS) if op.name == "mul")
    scores[0, mul] = -1.0
    with pytest.raises(ValueError, match="cumulative score"):
        reconstruct_lineage(capture, initial, outputs, result)
    scores[0, mul] = 0.0
    legal[0, mul] = False
    with pytest.raises(ValueError, match="score/legal"):
        reconstruct_lineage(capture, initial, outputs, result)


def test_open_quotation_state_preserves_code_as_typed_value():
    initial = ((Int(2),),)
    states = replay_states(("[", "dup", "]", "i"), initial)
    assert states[1] == ((Int(2), Mark()),)
    assert states[2] == ((Int(2), Mark(), Sym("dup")),)
    assert states[3] == ((Int(2), Lst((Sym("dup"),))),)
    assert states[-1] == ((Int(2), Int(2)),)


def test_budget_zero_charges_parent_before_op0_without_scores():
    capture, initial, outputs, result = observed(budget=0)
    assert result.termination == "budget"
    assert result.expansions == 1 and result.executions == 0
    assert len(capture.starts) == 1 and not capture.scored
    assert reconstruct_lineage(capture, initial, outputs, result) == ()
