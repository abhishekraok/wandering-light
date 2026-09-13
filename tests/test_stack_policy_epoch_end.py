"""train_policy exposes every epoch's weights (world-plus-v4-policy, #134).

Solve counts on the v4 cohort track the epoch the best-validation checkpoint
was taken at, so an epoch-resolved comparison needs each epoch's weights. The
hook must see every epoch, and the weights it sees at the best epoch must be
exactly the weights train_policy returns.
"""

from __future__ import annotations

import torch

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_machine import Config, format_stack
from wandering_light.stack_pbe import Task, run_program, skeleton, to_stack
from wandering_light.stack_policy import train_policy, witness_rows


def _task(index: int) -> Task:
    program = ("dup",) * (index + 1) + ("add",) * (index + 1)
    outcome = run_program(program, to_stack((1,)))
    assert not isinstance(outcome, str), outcome
    result = (format_stack(Config(outcome[0])),)
    return Task(
        behaviour=f"double-{index}",
        summary="",
        split="train",
        task_hash_split="train",
        inputs=("[ 1 ]",),
        outputs=result,
        held_out_inputs=("[ 1 ]",),
        held_out_outputs=result,
        program=program,
        length=len(program),
        examples=1,
        held_out=1,
        fuel=len(program),
        tags=(),
        skeleton=skeleton(program),
    )


def test_epoch_end_sees_every_epoch_and_the_best_weights_come_back() -> None:
    rows = witness_rows([_task(i) for i in range(6)])
    fit, validation = rows[:30], rows[30:]
    seen: dict[int, dict[str, torch.Tensor]] = {}

    def keep(epoch: int, model, row) -> None:
        assert row["epoch"] == epoch
        assert not model.training
        seen[epoch] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model, state = train_policy(
        fit,
        validation,
        PolicyArchitecture(8, 8, 8),
        epochs=3,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=8,
        seed=0,
        device="cpu",
        epoch_end=keep,
    )
    assert sorted(seen) == [1, 2, 3]
    best = seen[state["best_epoch"]]
    returned = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert returned.keys() == best.keys()
    assert all(torch.equal(returned[k], best[k]) for k in returned)
    assert not all(torch.equal(seen[1][k], seen[3][k]) for k in seen[1]), (
        "weights should change across epochs"
    )
