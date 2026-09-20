"""The pure helpers of `experiments.staged_cycles`: no corpus, no training."""

import pytest

from experiments.hindsight_subtargets import replay_states
from experiments.staged_cycles import (
    closed_indices,
    closed_product,
    cycle_bucket,
    cycle_task,
    cycles,
    next_closed,
    replays_to,
    whole_witness_share,
)
from experiments.witness_trajectory_shape import BASIS
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import Task

# Two cycles over one list: square each, then double each. Closed at the root,
# after the first `map` and after the second; the eight steps in between are
# either under an open mark or hold a finished-but-unexecuted quotation.
STAGED = "[ dup mul ] map [ dup add ] map"
FUSED = "[ dup mul dup add ] map"


def task(program: str, *inputs: str) -> Task:
    """A witness on given inputs. Only the fields the helpers read are real."""
    ops = tuple(program.split())
    return Task(
        behaviour="square_then_double",
        summary="square each element, then double it",
        split="train",
        task_hash_split="train",
        inputs=tuple(inputs),
        outputs=tuple(inputs),  # replaced by `cycle_task`; never read here
        held_out_inputs=("[ (7) ]",),
        held_out_outputs=("[ (98) ]",),
        program=ops,
        length=len(ops),
        examples=len(inputs),
        held_out=1,
        fuel=0,
        tags=(),
    )


def states(program: str, *inputs: str):
    return replay_states(task(program, *inputs))


def test_a_product_is_closed_only_when_every_example_is():
    open_quote = parse_stack("[ MARK dup ]", basis=BASIS)
    data = parse_stack("[ (1 2) ]", basis=BASIS)
    assert closed_product((data, data))
    assert not closed_product((data, open_quote))


def test_a_parked_quotation_is_not_a_closed_state():
    # After `]` no mark is open, but the body has not run: still program text.
    assert closed_indices(states(STAGED, "[ (1 2 3) ]")) == [0, 5, 10]


def test_the_fused_witness_of_the_same_behaviour_closes_only_at_its_ends():
    assert closed_indices(states(FUSED, "[ (1 2 3) ]")) == [0, 7]


def test_two_examples_close_at_the_same_steps():
    assert closed_indices(states(STAGED, "[ (1 2 3) ]", "[ (4 5) ]")) == [0, 5, 10]


def test_next_closed_is_the_nearest_one_strictly_ahead():
    assert next_closed(states(STAGED, "[ (1 2 3) ]")) == [5] * 5 + [10] * 5


def test_next_closed_of_the_fused_witness_is_the_final_output_everywhere():
    assert next_closed(states(FUSED, "[ (1 2 3) ]")) == [7] * 7


def test_next_closed_refuses_a_witness_that_ends_mid_quotation():
    with pytest.raises(ValueError, match="final state is not closed"):
        next_closed(states("[ dup mul", "[ (1 2 3) ]"))


def test_cycles_partition_the_witness_between_consecutive_closed_states():
    got = cycles(task(STAGED, "[ (1 2 3) ]"))
    assert [c.ops for c in got] == [
        ("[", "dup", "mul", "]", "map"),
        ("[", "dup", "add", "]", "map"),
    ]
    assert [c.index for c in got] == [0, 1]
    assert [c.kinds for c in got] == ["W4 X1", "W4 X1"]
    assert got[0].start == states(STAGED, "[ (1 2 3) ]")[0]
    assert got[0].target == got[1].start


def test_the_fused_witness_is_one_cycle_of_the_whole_program():
    got = cycles(task(FUSED, "[ (1 2 3) ]"))
    assert len(got) == 1
    assert got[0].ops == tuple(FUSED.split())


def test_a_direct_op_between_two_cycles_is_a_cycle_of_its_own():
    got = cycles(task("0 [ add ] fold 1 add", "[ (1 2 3) ]"))
    assert [c.ops for c in got] == [("0",), ("[", "add", "]", "fold"), ("1",), ("add",)]
    assert [c.kinds for c in got] == ["D1", "W3 X1", "D1", "D1"]


def test_cycle_task_asks_for_the_transition_and_carries_no_held_out_pair():
    original = task(STAGED, "[ (1 2 3) ]", "[ (4 5) ]")
    synthetic = cycle_task(original, cycles(original)[0])
    assert synthetic.inputs == ("[ (1 2 3) ]", "[ (4 5) ]")
    assert synthetic.outputs == ("[ (1 4 9) ]", "[ (16 25) ]")
    assert (
        synthetic.held_out_inputs,
        synthetic.held_out_outputs,
        synthetic.held_out,
    ) == (
        (),
        (),
        0,
    )
    assert synthetic.program == ("[", "dup", "mul", "]", "map")
    assert synthetic.length == 5


def test_cycle_task_targets_round_trip_through_the_printed_form():
    original = task(STAGED, "[ (1 2 3) ]")
    cycle = cycles(original)[0]
    synthetic = cycle_task(original, cycle)
    parsed = tuple(parse_stack(text, basis=BASIS) for text in synthetic.outputs)
    assert parsed == cycle.target


def test_replays_to_accepts_any_program_that_reaches_the_target():
    cycle = cycles(task(STAGED, "[ (1 2 3) ]"))[0]
    assert replays_to(cycle.ops, cycle.start, cycle.target)
    # A longer body with the same effect: `dup dup mul swap drop` leaves x*x.
    assert replays_to(
        ("[", "dup", "dup", "mul", "swap", "drop", "]", "map"),
        cycle.start,
        cycle.target,
    )


def test_replays_to_rejects_a_program_that_stops_short_or_refuses():
    cycle = cycles(task(STAGED, "[ (1 2 3) ]"))[0]
    assert not replays_to(("[", "dup", "mul", "]"), cycle.start, cycle.target)
    assert not replays_to(("add",), cycle.start, cycle.target)  # a list is not an int
    assert not replays_to(("nonesuch",), cycle.start, cycle.target)


def test_cycle_buckets():
    assert [cycle_bucket(n) for n in (1, 2, 5, 6, 12, 13, 400)] == [
        "1",
        "2-5",
        "2-5",
        "6-12",
        "6-12",
        "13+",
        "13+",
    ]


def test_whole_witness_share_counts_witnesses_not_transitions():
    rows = [
        {"behaviour": "a", "task_index": 0, "solved": True},
        {"behaviour": "a", "task_index": 0, "solved": True},
        {"behaviour": "a", "task_index": 1, "solved": True},
        {"behaviour": "a", "task_index": 1, "solved": False},
    ]
    assert whole_witness_share(rows) == 0.5
    assert whole_witness_share([]) is None
