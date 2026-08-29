"""The behaviour-first corpus claims three things. Check all three.

1. Every behaviour has two independent statements -- a Python reference and a
   stack-machine witness -- and they agree on data neither of them chose.
2. No task input contains a quotation. That is the defect the generator exists
   to remove, so it is checked structurally rather than trusted.
3. A candidate can be graded on examples it was never shown, which means the
   held-out set is really disjoint and really solved by the witness.
"""

import gzip
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    Sym,
    Value,
    ops_for_basis,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import (
    BASIS,
    BY_NAME,
    CATALOGUE,
    Task,
    _informative,
    assign_split,
    build_task,
    check,
    draw_task,
    encode_task,
    generate,
    load,
    run_program,
    to_stack,
    to_value,
    verify,
)

CORPUS = (
    Path(__file__).resolve().parent.parent
    / "wandering_light"
    / "evals"
    / "data"
    / "stack_pbe_v3.jsonl.gz"
)


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return load(str(CORPUS))


def contains_quotation(value: Value) -> bool:
    if isinstance(value, Sym):
        return True
    if isinstance(value, Lst):
        return any(contains_quotation(item) for item in value.items)
    return False


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


def test_behaviour_names_are_unique():
    names = [b.name for b in CATALOGUE]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_witness_agrees_with_reference_on_fresh_data(behaviour):
    """The whole guarantee: two independent statements, checked against each other.

    Neither is privileged. A wrong label would need the same mistake made twice,
    in two languages, on inputs the sampler chose after both were written.
    """
    assert verify(behaviour, trials=120, seed=1234) is None


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_witness_leaves_exactly_one_value(behaviour):
    """A task's output is a value, not a stack with the answer somewhere in it."""
    rng = random.Random(7)
    for _ in range(20):
        outcome = run_program(behaviour.program, to_stack(behaviour.sample(rng)))
        assert not isinstance(outcome, str), outcome
        assert len(outcome[0]) == 1


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_sampler_never_produces_a_quotation(behaviour):
    """Inputs are numbers and lists of numbers. Nothing else, ever."""
    rng = random.Random(11)
    for _ in range(20):
        for value in to_stack(behaviour.sample(rng)):
            assert not contains_quotation(value)


def test_a_wrong_witness_is_caught():
    """The agreement check has to be able to fail, or it proves nothing."""
    from dataclasses import replace

    broken = replace(BY_NAME["sum"], witness="1 [ add ] fold")
    assert verify(broken, trials=50, seed=3) is not None


# --------------------------------------------------------------------------
# Task drawing
# --------------------------------------------------------------------------


def test_held_out_inputs_are_disjoint_from_shown():
    rng = random.Random(5)
    for behaviour in CATALOGUE:
        drawn = draw_task(behaviour, rng, examples=4, held_out=6)
        shown = {json.dumps(a) for a in drawn.shown}
        hidden = {json.dumps(a) for a in drawn.held_out}
        assert len(shown) == 4
        assert len(hidden) == 6
        assert not (shown & hidden)


def test_constant_example_sets_are_rejected():
    """Every output the same means a constant program fits."""
    args = [([1, 2],), ([3, 4],)]
    stacks = [to_stack(a) for a in args]
    assert not _informative(args, [7, 7], stacks)


def test_projection_example_sets_are_rejected():
    """Output equal to a fixed input position means a projection fits."""
    args = [([1, 2], 5), ([3, 4], 6)]
    stacks = [to_stack(a) for a in args]
    assert not _informative(args, [5, 6], stacks)
    assert _informative(args, [3, 7], stacks)


def test_build_task_rejects_an_output_its_witness_does_not_produce():
    """Labels are made by execution, so a wrong one cannot be written out."""
    from dataclasses import replace

    from wandering_light.stack_machine import MachineError

    rng = random.Random(2)
    drawn = draw_task(BY_NAME["sum"], rng, examples=3, held_out=2)
    corrupted = replace(
        drawn, shown_results=[r + 1 for r in drawn.shown_results]
    )
    with pytest.raises(MachineError, match="disagree"):
        build_task(BY_NAME["sum"], corrupted)

    corrupted_hidden = replace(
        drawn, held_out_results=[r + 1 for r in drawn.held_out_results]
    )
    with pytest.raises(MachineError, match="disagree"):
        build_task(BY_NAME["sum"], corrupted_hidden)


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


def test_split_is_on_the_behaviour_not_the_task():
    """No behaviour may appear in both halves, whatever its data."""
    for behaviour in CATALOGUE:
        assert assign_split(behaviour.name) in {"train", "eval"}
    both = {b.name: assign_split(b.name) for b in CATALOGUE}
    assert set(both.values()) == {"train", "eval"}


@pytest.mark.parametrize("seed", ["0", "1", "31337"])
def test_split_does_not_depend_on_pythonhashseed(seed):
    """SHA-256, not `hash`, so regenerating cannot move a behaviour across."""
    script = """
import json
from wandering_light.stack_pbe import CATALOGUE, assign_split
print(json.dumps({b.name: assign_split(b.name) for b in CATALOGUE}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    expected = {b.name: assign_split(b.name) for b in CATALOGUE}
    assert json.loads(result.stdout) == expected


# --------------------------------------------------------------------------
# The committed corpus
# --------------------------------------------------------------------------


def test_corpus_is_present_and_sized(tasks):
    assert len(tasks) == 40 * 40
    assert {task.basis for task in tasks} == {BASIS}
    assert {task.witness_source for task in tasks} == {"authored"}


def test_no_task_input_contains_a_quotation(tasks):
    """The defect the generator exists to remove, checked on the artefact."""
    for task in tasks:
        for text in (*task.inputs, *task.held_out_inputs):
            for value in parse_stack(text, basis=BASIS):
                assert not contains_quotation(value), text


def test_every_witness_reproduces_its_shown_outputs(tasks):
    for task in tasks:
        for given, wanted in zip(task.inputs, task.outputs, strict=True):
            outcome = run_program(task.program, parse_stack(given, basis=BASIS))
            assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
            stack, _fuel = outcome
            want = parse_stack(wanted, basis=BASIS)
            assert tuple(map(value_key, stack)) == tuple(map(value_key, want))


def test_every_witness_reproduces_its_held_out_outputs(tasks):
    """Grading on hidden data only means something if the label is right."""
    for task in tasks:
        pairs = zip(task.held_out_inputs, task.held_out_outputs, strict=True)
        for given, wanted in pairs:
            outcome = run_program(task.program, parse_stack(given, basis=BASIS))
            assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
            stack, _fuel = outcome
            want = parse_stack(wanted, basis=BASIS)
            assert tuple(map(value_key, stack)) == tuple(map(value_key, want))


def test_held_out_examples_are_disjoint_from_shown_in_the_corpus(tasks):
    for task in tasks:
        assert not (set(task.inputs) & set(task.held_out_inputs))
        assert task.held_out >= 1


def test_corpus_data_is_varied(tasks):
    """The defect being fixed was 17 input sets across 1,857 tasks."""
    inputs = {i for task in tasks for i in (*task.inputs, *task.held_out_inputs)}
    assert len(inputs) > 10_000
    assert len({tuple(task.inputs) for task in tasks}) == len(tasks)


def test_no_behaviour_spans_both_splits(tasks):
    by_behaviour: dict[str, set[str]] = {}
    for task in tasks:
        by_behaviour.setdefault(task.behaviour, set()).add(task.split)
    assert all(len(splits) == 1 for splits in by_behaviour.values())


def test_corpus_round_trips_through_load(tmp_path):
    made, _stats = generate(per_behaviour=1, seed=3)
    path = tmp_path / "corpus.jsonl.gz"
    from dataclasses import asdict

    with gzip.open(path, "wt") as handle:
        for task in made:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    assert load(str(path)) == made


def test_encode_task_shows_examples_and_hides_the_held_out_set(tasks):
    encoded = encode_task(tasks[0])
    assert encoded["prompt"].endswith("=>")
    assert encoded["completion"].strip() == " ".join(tasks[0].program)
    for hidden in tasks[0].held_out_inputs:
        assert hidden not in encoded["prompt"]


# --------------------------------------------------------------------------
# Why the basis is `mark-28` and not `executable-26`
# --------------------------------------------------------------------------


def test_executable_26_reaches_no_executable_quotation_from_data():
    """The reason this corpus cannot run on the default basis.

    No op in `executable-26` puts an op name on the stack, so from a data-only
    start every reachable quotation is a list of data -- which `map` can only
    apply as the identity, and which `filter` and `fold` reject. `mark-28`
    reaches executable quotations from the same start.
    """
    start = to_stack(([1, 2, 3],))

    def reachable_quotations(basis: str, depth: int) -> set:
        ops = ops_for_basis(basis)
        seen = {tuple(map(value_key, start))}
        frontier = [start]
        found = set()
        for _ in range(depth):
            nxt = []
            for stack in frontier:
                for op in ops:
                    outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
                    if isinstance(outcome, str):
                        continue
                    moved = outcome[0]
                    for value in moved:
                        if contains_quotation(value):
                            found.add(value_key(value))
                    key = tuple(map(value_key, moved))
                    if key not in seen:
                        seen.add(key)
                        nxt.append(moved)
            frontier = nxt
        return found

    assert reachable_quotations("executable-26", 4) == set()
    assert reachable_quotations("mark-28", 4)


def test_check_reports_the_input_it_disagreed_on():
    from dataclasses import replace

    broken = replace(BY_NAME["reverse"], witness="nil [ swap cons ] fold")
    bad = check(broken, ([1, 2, 3],))
    assert bad is not None
    assert bad.args == ([1, 2, 3],)


def test_to_value_refuses_bools():
    from wandering_light.stack_machine import MachineError

    with pytest.raises(MachineError):
        to_value(True)
