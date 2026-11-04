"""The stack-token policy and the beam search it guides (compute-optimal-mark-28, #101)."""

import random
import sys

import numpy as np
import pytest
import torch

from wandering_light.quotation_guard import blocked
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import BeamObserver, atoms_exceed, beam_solve, step_all
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, OPS, build_task, draw_task
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
    assert result.expansions == 1
    assert 0 < result.executions <= len(OP_NAMES) * task.examples
    assert result.depth == 1


def test_budget_is_checked_before_every_attempt():
    """Codex review: a zero budget must not return a program."""
    task = task_for("add_pair", 6)
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    none = beam_solve(task, scorer, beam_width=64, max_depth=4, budget=0)
    assert none.program is None
    assert none.termination == "budget"
    assert none.executions == 0
    # One attempt at a time: the budget trips as soon as the running count
    # reaches it, after however many stacks the first legal op stepped.
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    expected = 0
    for op in OPS:
        expected += step_all(product, op)[2]
        if expected >= 1:
            break
    one = beam_solve(task, scorer, beam_width=64, max_depth=4, budget=1)
    assert one.termination == "budget"
    assert one.executions == expected


def test_executions_and_fuel_charge_executed_work_not_the_retained_beam():
    """Codex review: successors the beam discards were still executed."""
    task = task_for("sum")
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    expected_executions = expected_fuel = 0
    for op in OPS:
        _moved, fuel, stepped = step_all(product, op)
        expected_executions += stepped
        expected_fuel += fuel
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    result = beam_solve(task, scorer, beam_width=1, max_depth=1)
    assert result.program is None
    assert result.expansions == 1
    assert result.executions == expected_executions
    assert result.fuel == expected_fuel


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


def test_atoms_count_every_value_and_list_cell_across_examples():
    """A list of n ints is n + 1 atoms, and every example's stack adds up."""
    one = ((Int(1), Bool(True), Sym("add"), Mark()),)
    assert not atoms_exceed(one, 4)
    assert atoms_exceed(one, 3)
    listed = ((Lst((Int(1), Int(2), Int(3))),),)  # the cell and three ints
    assert not atoms_exceed(listed, 4)
    assert atoms_exceed(listed, 3)
    nested = ((Int(0), Lst((Int(1), Lst((Int(2), Int(3)))))),)  # 0 ( 1 ( 2 3 ) )
    assert not atoms_exceed(nested, 6)
    assert atoms_exceed(nested, 5)
    assert not atoms_exceed((), 0)
    assert not atoms_exceed(((), ()), 0)
    product = ((Int(1), Int(2)), (Lst((Int(3),)),), ())  # 2 + 2 + 0
    assert not atoms_exceed(product, 4)
    assert atoms_exceed(product, 3)


def test_atoms_walk_is_iterative_and_stops_at_the_cap():
    """A value nested past the recursion limit must neither overflow nor be walked."""
    depth = sys.getrecursionlimit() * 4
    value = Lst(())
    for _ in range(depth):
        value = Lst((value,))
    product = ((value,), (value,))
    assert atoms_exceed(product, 10)
    assert atoms_exceed(product, 2 * (depth + 1) - 1)
    assert not atoms_exceed(product, 2 * (depth + 1))
    # A wide list past the cap is answered from its length, not its items.
    wide = ((Lst(tuple(Int(i) for i in range(1000))),),)
    assert atoms_exceed(wide, 1000)
    assert not atoms_exceed(wide, 1001)


class _Watcher(BeamObserver):
    def __init__(self):
        self.successors = {}  # (depth, node, op) -> product
        self.verdicts = {}  # (depth, node, op) -> verdict
        self.retained = []  # (depth, product)
        self.remaining = {}

    def expansion_attempted(self, depth, node, op_index, successor, *_):
        if successor is not None:
            self.successors[depth, node, op_index] = successor

    def layer_retained(self, depth, beam, verdicts, parents):
        for (node, op), verdict in verdicts.items():
            self.verdicts[depth, node, op] = verdict
        self.retained.extend((depth, product) for product, _, _ in beam)


def test_state_cap_refuses_oversized_successors_after_the_guard_and_before_seen():
    task = task_for("sum")
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    cap = next(n for n in range(10**6) if not atoms_exceed(inputs, n))
    watch = _Watcher()
    max_depth = 3
    result = beam_solve(
        task,
        scorer,
        beam_width=8,
        max_depth=max_depth,
        guard="both",
        max_state_atoms=cap,
        observer=watch,
    )
    assert result.program is None
    assert "guard:size" in set(watch.verdicts.values())
    assert "retained" in set(watch.verdicts.values())
    for (depth, node, op), verdict in watch.verdicts.items():
        product = watch.successors[depth, node, op]
        why = blocked(product[0], max_depth - depth - 1, "both")
        oversized = atoms_exceed(product, cap)
        if why is not None:
            assert verdict == f"guard:{why}"
        elif oversized:
            assert verdict == "guard:size"
        else:
            assert verdict in ("retained", "duplicate")
    for _depth, product in watch.retained:
        assert not atoms_exceed(product, cap)


def test_no_state_cap_is_the_unbounded_search():
    task = task_for("sum")
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    settings = {"beam_width": 8, "max_depth": 3}
    watched = _Watcher()
    unbounded = beam_solve(
        task, scorer, max_state_atoms=None, observer=watched, **settings
    )
    assert "guard:size" not in set(watched.verdicts.values())
    plain = beam_solve(task, scorer, **settings)
    loose = beam_solve(task, scorer, max_state_atoms=10**9, **settings)
    for other in (plain, loose):
        assert (other.program, other.termination, other.executions, other.fuel) == (
            unbounded.program,
            unbounded.termination,
            unbounded.executions,
            unbounded.fuel,
        )
        assert (other.expansions, other.depth) == (
            unbounded.expansions,
            unbounded.depth,
        )
    # A cap below every successor empties the frontier at the first layer.
    starved = beam_solve(task, scorer, max_state_atoms=0, **settings)
    assert starved.program is None
    assert starved.termination == "frontier_exhausted"
    assert starved.depth == 1


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
