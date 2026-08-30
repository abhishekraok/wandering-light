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
    skeleton,
    to_stack,
    to_value,
    verify,
)

DATA = Path(__file__).resolve().parent.parent / "wandering_light" / "evals" / "data"
CORPUS = DATA / "stack_pbe_v4.jsonl.gz"
CORPUS_V3 = DATA / "stack_pbe_v3.jsonl.gz"


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
    assert len(behaviour.witnesses) == len(set(behaviour.witnesses))


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

    broken = replace(BY_NAME["sum"], witnesses=("1 [ add ] fold",))
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
    corrupted = replace(drawn, shown_results=[r + 1 for r in drawn.shown_results])
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


def test_split_is_on_the_skeleton_not_the_behaviour_or_the_task():
    """Behaviours sharing a program shape must land on the same side."""
    by_skeleton: dict[str, set[str]] = {}
    for behaviour in CATALOGUE:
        assert assign_split(behaviour.skeleton) in {"train", "eval"}
        by_skeleton.setdefault(behaviour.skeleton, set()).add(
            assign_split(behaviour.skeleton)
        )
    assert all(len(sides) == 1 for sides in by_skeleton.values())
    assert set().union(*by_skeleton.values()) == {"train", "eval"}


def test_skeleton_blanks_literals_and_interchangeable_ops():
    """`keep_zeros` and `keep_negative` are two functions of one shape."""
    assert BY_NAME["keep_zeros"].skeleton == BY_NAME["keep_negative"].skeleton
    assert skeleton(("[", "0", "eq?", "]", "filter")) == "[ _ _ ] filter"
    assert BY_NAME["sum"].skeleton != BY_NAME["reverse"].skeleton
    assert assign_split(BY_NAME["keep_zeros"].skeleton) == assign_split(
        BY_NAME["keep_negative"].skeleton
    )


def test_every_witness_of_a_behaviour_is_checked():
    """An alternative that is subtly wrong must not ride along on the first."""
    from dataclasses import replace

    behaviour = BY_NAME["sum"]
    assert len(behaviour.witnesses) > 1
    broken = replace(behaviour, witnesses=(behaviour.witness, "1 [ add ] fold"))
    assert verify(broken, trials=80, seed=1) is not None


def test_a_bare_string_of_witnesses_is_refused():
    """`witnesses="add"` would iterate characters. Fail loudly instead."""
    from dataclasses import replace

    from wandering_light.stack_machine import MachineError

    with pytest.raises(MachineError, match="tuple of programs"):
        replace(BY_NAME["sum"], witnesses="0 [ add ] fold")
    with pytest.raises(MachineError, match="at least one"):
        replace(BY_NAME["sum"], witnesses=())


@pytest.mark.parametrize("seed", ["0", "1", "31337"])
def test_split_does_not_depend_on_pythonhashseed(seed):
    """SHA-256, not `hash`, so regenerating cannot move a behaviour across."""
    script = """
import json
from wandering_light.stack_pbe import CATALOGUE, assign_split
print(json.dumps({b.name: assign_split(b.skeleton) for b in CATALOGUE}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    expected = {b.name: assign_split(b.skeleton) for b in CATALOGUE}
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


def test_no_behaviour_or_skeleton_spans_both_splits(tasks):
    for key in ("behaviour", "skeleton"):
        grouped: dict[str, set[str]] = {}
        for task in tasks:
            grouped.setdefault(getattr(task, key), set()).add(task.split)
        assert all(len(splits) == 1 for splits in grouped.values()), key


def test_shown_examples_never_hold_list_length_fixed(tasks):
    """The hole that produced v3's one repeated memorisation, closed."""
    for task in tasks:
        shapes = {
            tuple(
                len(v.items)
                for v in parse_stack(text, basis=BASIS)
                if isinstance(v, Lst)
            )
            for text in task.inputs
        }
        if next(iter(shapes)):
            assert len(shapes) > 1, task.behaviour


def test_alternative_programs_also_solve_their_task(tasks):
    """They are labels too, so they get executed like the canonical one."""
    checked = 0
    for task in tasks:
        for program in task.alternative_programs:
            for given, wanted in zip(task.inputs, task.outputs, strict=True):
                outcome = run_program(program, parse_stack(given, basis=BASIS))
                assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
                want = parse_stack(wanted, basis=BASIS)
                assert tuple(map(value_key, outcome[0])) == tuple(map(value_key, want))
            checked += 1
    assert checked > 0


def test_a_schema_1_corpus_still_loads():
    """v3 is quoted by STATUS and by two reports; it must stay readable."""
    old = load(str(CORPUS_V3))
    assert old and {t.schema_version for t in old} == {1}
    assert all(t.skeleton == "" and t.alternative_programs == () for t in old)


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

    broken = replace(BY_NAME["reverse"], witnesses=("nil [ swap cons ] fold",))
    bad = check(broken, ([1, 2, 3],))
    assert bad is not None
    assert bad.args == ([1, 2, 3],)


def test_to_value_refuses_bools():
    from wandering_light.stack_machine import MachineError

    with pytest.raises(MachineError):
        to_value(True)
