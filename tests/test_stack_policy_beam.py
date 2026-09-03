"""The stack-token policy and the beam search it guides (compute-optimal-mark-28, #101)."""

import random

import numpy as np
import pytest
import torch

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import Int, Lst, Mark, Sym, parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_pbe_eval import grade
from wandering_light.stack_policy import (
    OP_NAMES,
    PAD,
    SEP,
    STATE_CAP,
    VOCAB,
    PolicyScorer,
    StackPairPolicy,
    encode_item,
    encode_stack,
    load_policy,
    save_policy,
    train_policy,
    witness_rows,
)

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def test_tokens_are_one_per_symbol_and_stay_in_vocab():
    stack = (Int(3), Lst((Int(-100), Int(1))), Mark(), Sym("add"), Int(7))
    tokens = encode_stack(stack, STATE_CAP)
    # 3 · ( BIG NEG 1 0 0 1 ) · MARK · add · 7
    assert len(tokens) == 1 + 8 + 1 + 1 + 1
    assert all(0 < t < VOCAB for t in tokens)
    assert PAD not in tokens
    assert len({encode_stack((Int(v),), 8)[0] for v in range(-64, 65)}) == 129


def test_overflow_keeps_the_top_of_the_stack():
    stack = tuple(Int(i % 50) for i in range(STATE_CAP + 40))
    tokens = encode_stack(stack, STATE_CAP)
    assert len(tokens) == STATE_CAP
    assert tokens == encode_stack(stack[40:], STATE_CAP)


def test_item_is_state_sep_target():
    item = encode_item((Int(1),), [5, 6])
    assert item == [encode_stack((Int(1),), STATE_CAP)[0], SEP, 5, 6]


def test_witness_rows_label_the_next_op():
    task = task_for("sum")
    rows = witness_rows([task])
    assert [OP_NAMES[r.label] for r in rows] == list(task.program)
    assert [r.remaining for r in rows] == list(range(task.length, 0, -1))
    assert all(r.items.shape[0] == task.examples for r in rows)


def test_beam_finds_a_one_op_witness_under_any_policy():
    """Depth 1 is exhaustive at beam 64, so the hit does not depend on scores."""
    task = task_for("add_pair", 6)
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    result = beam_solve(task, scorer, beam_width=64, max_depth=4, guard="both")
    assert result.termination == "fit"
    assert result.program == ("add",)
    assert grade(result.program, task).solved
    assert result.op_applications == result.expansions * len(OP_NAMES) == len(OP_NAMES)
    assert result.depth == 1


def test_trained_policy_guides_a_narrow_beam_to_the_witness():
    """A policy fit to a task's own prefixes should carry beam 4 down its witness."""
    task = task_for("sum")
    rows = witness_rows([task, task_for("sum", 5)])
    model, _ = train_policy(
        rows,
        rows,
        TINY,
        epochs=40,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=8,
        seed=0,
        device="cpu",
    )
    result = beam_solve(
        task, PolicyScorer(model, "cpu"), beam_width=4, max_depth=task.length
    )
    assert result.termination == "fit", result
    assert grade(result.program, task).solved
    assert result.depth == len(result.program) <= task.length


def test_beam_reports_max_depth_when_the_witness_is_out_of_reach():
    task = task_for("sum")
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    result = beam_solve(task, scorer, beam_width=4, max_depth=1)
    assert result.program is None
    assert result.termination in ("max_depth", "frontier_exhausted")


def test_train_then_round_trip_checkpoint(tmp_path):
    rows = witness_rows([task_for("sum"), task_for("sum", 5)])
    model, state = train_policy(
        rows,
        rows,
        TINY,
        epochs=2,
        learning_rate=1e-2,
        batch_size=4,
        validation_batch_size=8,
        seed=0,
        device="cpu",
    )
    assert state["best_epoch"] in (1, 2)
    path = save_policy(
        tmp_path / "policy",
        model,
        trainer_state=state,
        manifest={"arm": "tiny"},
        wandb_url=None,
    )
    loaded = load_policy(path)
    block = np.array(
        [encode_item(parse_stack("[ (1 2) ]", basis=BASIS), [1])], dtype=np.int16
    )
    logits = PolicyScorer(loaded).log_probs([block], np.ones((1, len(OP_NAMES)), bool))
    assert logits.shape == (1, len(OP_NAMES))
    assert np.isclose(np.exp(logits).sum(), 1.0)
    with pytest.raises(FileExistsError):
        save_policy(path, model, trainer_state=state, manifest={}, wandb_url=None)


def test_scorer_masks_illegal_ops():
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    legal = np.zeros((1, len(OP_NAMES)), bool)
    legal[0, :2] = True
    block = np.array([[SEP, 3]], dtype=np.int16)
    out = scorer.log_probs([block], legal)
    assert np.isneginf(out[0, 2:]).all()
    assert np.isclose(np.exp(out[0, :2]).sum(), 1.0)
    assert torch.isfinite(torch.tensor(out[0, :2])).all()
