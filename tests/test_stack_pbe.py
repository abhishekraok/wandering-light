"""The behaviour-first corpus claims four things. Check all four.

1. Every behaviour has independent statements -- a Python reference and one or
   more stack-machine witnesses -- and they agree on data none of them chose.
2. No task input contains a quotation. That is the defect the generator exists
   to remove, so it is checked structurally rather than trusted.
3. A candidate can be graded on examples it was never shown, which means the
   held-out set is really disjoint and really solved by every witness.
4. The split is on the program's *shape*, so two behaviours of one shape --
   `keep_zeros` and `keep_negative` are both `[ _ _ ] filter` -- cannot sit on
   opposite sides of it, and no shown set is drawn at a single list length.
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
    MachineError,
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
    INTERCHANGEABLE,
    SCHEMA_VERSION,
    Behaviour,
    Task,
    _informative,
    _varied_shapes,
    apply_witness,
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
V3_CORPUS = DATA / "stack_pbe_v3.jsonl.gz"


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
def test_every_witness_leaves_exactly_one_value(behaviour):
    """A task's output is a value, not a stack with the answer somewhere in it."""
    rng = random.Random(7)
    for _ in range(20):
        stack = to_stack(behaviour.sample(rng))
        for program in behaviour.programs:
            outcome = run_program(program, stack)
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


def test_a_wrong_second_witness_is_caught():
    """`verify` checks every witness, not just the canonical one."""
    from dataclasses import replace

    broken = replace(BY_NAME["sum"], witnesses=("0 [ add ] fold", "1 [ add ] fold"))
    bad = verify(broken, trials=50, seed=3)
    assert bad is not None
    assert bad.program == ("1", "[", "add", "]", "fold")
    assert check(broken, ([1, 2],)) is None
    assert check(broken, ([1, 2],), program=("1", "[", "add", "]", "fold")) is not None


# --------------------------------------------------------------------------
# Witnesses: one canonical, the rest alternatives
# --------------------------------------------------------------------------


def _behaviour(witnesses) -> Behaviour:
    return Behaviour(
        name="probe",
        summary="",
        witnesses=witnesses,
        sample=lambda rng: ([1, 2],),
        reference=sum,
    )


def test_witnesses_must_be_a_tuple():
    with pytest.raises(MachineError, match="probe: witnesses must be a tuple"):
        _behaviour("0 [ add ] fold")
    with pytest.raises(MachineError, match="probe: witnesses must be a tuple"):
        _behaviour(["0 [ add ] fold"])


def test_at_least_one_witness_is_required():
    with pytest.raises(MachineError, match="probe: needs at least one witness"):
        _behaviour(())


def test_the_first_witness_is_canonical():
    b = _behaviour(("0 [ add ] fold", "uncons [ add ] fold"))
    assert b.witness == "0 [ add ] fold"
    assert b.program == ("0", "[", "add", "]", "fold")
    assert b.programs == (
        ("0", "[", "add", "]", "fold"),
        ("uncons", "[", "add", "]", "fold"),
    )
    assert b.skeleton == "_ [ _ ] fold"
    assert b.length == 5


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_witnesses_are_distinct_programs(behaviour):
    """An alternative that is the canonical program again says nothing."""
    assert len(set(behaviour.programs)) == len(behaviour.programs)


def test_the_catalogue_carries_the_second_witnesses():
    """The shapes a solver might reasonably produce instead of the canonical."""
    expected = {
        "double_each": "[ 1 1 add mul ] map",
        "increment_each": "[ 1 swap add ] map",
        "negate_each": "[ 0 1 sub mul ] map",
        "sum": "uncons [ add ] fold",
        "product": "uncons [ mul ] fold",
        "length": "[ drop 1 ] map 0 [ add ] fold",
        "last": "nil [ cons ] fold uncons swap drop",
        "sum_both": "0 [ add ] fold swap 0 [ add ] fold add",
        "sum_of_squares": "[ dup mul ] map 0 [ add ] fold",
    }
    for name, alternative in expected.items():
        assert BY_NAME[name].witnesses[1:] == (alternative,), name
    assert BY_NAME["sum_of_squares"].witness == "0 [ dup mul add ] fold"
    for b in CATALOGUE:
        if b.name not in expected:
            assert len(b.witnesses) == 1, b.name


def test_apply_witness_runs_the_program_it_is_given():
    b = BY_NAME["sum"]
    canonical = apply_witness(b, ([2, 3],))
    alternative = apply_witness(b, ([2, 3],), program=b.programs[1])
    assert not isinstance(canonical, str) and not isinstance(alternative, str)
    assert (
        value_key(canonical[0]) == value_key(alternative[0]) == value_key(to_value(5))
    )
    assert apply_witness(b, ([2, 3],), program=("dup",)) == "left 2 values, want 1"


# --------------------------------------------------------------------------
# Skeletons
# --------------------------------------------------------------------------


def test_skeleton_blanks_the_interchangeable_ops():
    assert skeleton(("[", "0", "eq?", "]", "filter")) == "[ _ _ ] filter"
    assert skeleton(("[", "0", "lt?", "]", "filter")) == "[ _ _ ] filter"
    assert skeleton(("nil", "[", "cons", "]", "fold")) == "nil [ cons ] fold"
    assert skeleton(("add",)) == "_"
    assert skeleton(()) == ""


def test_interchangeable_ops_are_the_literals_arithmetic_and_predicates():
    expected = {"0", "1", "add", "sub", "mul", "div", "eq?", "lt?", "null?"}
    assert expected == INTERCHANGEABLE


def test_two_filters_of_one_shape_share_a_skeleton():
    """The leak the split exists to close."""
    assert BY_NAME["keep_zeros"].skeleton == BY_NAME["keep_negative"].skeleton
    assert BY_NAME["sum"].skeleton == BY_NAME["product"].skeleton
    assert BY_NAME["scale_by"].skeleton == BY_NAME["add_to_each"].skeleton


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


def _list_lengths(args) -> tuple[int, ...]:
    return tuple(len(a) for a in args if isinstance(a, list))


def test_a_shown_set_at_one_list_length_is_rejected():
    """`uncons drop i` fit `last` on three two-element lists. Never again."""
    assert not _varied_shapes([([1, 2],), ([3, 4],), ([5, 6],)])
    assert _varied_shapes([([1, 2],), ([3, 4, 5],), ([5, 6],)])
    assert not _varied_shapes([([1, 2], [3]), ([4, 5], [6])])
    assert _varied_shapes([([1, 2], [3]), ([4], [5, 6])])
    # Stacks without lists have no shape to vary and are exempt.
    assert _varied_shapes([(1, 2), (3, 4)])


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_drawn_shown_sets_vary_their_list_lengths(behaviour):
    rng = random.Random(9)
    for _ in range(10):
        drawn = draw_task(behaviour, rng, examples=3, held_out=2)
        shapes = {_list_lengths(a) for a in drawn.shown}
        if shapes != {()}:
            assert len(shapes) > 1, drawn.shown


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


def test_split_is_on_the_skeleton_not_the_task():
    """No shape may appear in both halves, whatever its data or its name."""
    for behaviour in CATALOGUE:
        assert assign_split(behaviour.skeleton) in {"train", "eval"}
    both = {b.skeleton: assign_split(b.skeleton) for b in CATALOGUE}
    assert set(both.values()) == {"train", "eval"}


def test_behaviours_of_one_shape_fall_on_one_side():
    """A behaviour-name split put `keep_zeros` and `keep_negative` apart."""
    by_shape: dict[str, set[str]] = {}
    for b in CATALOGUE:
        by_shape.setdefault(b.skeleton, set()).add(assign_split(b.skeleton))
    assert all(len(sides) == 1 for sides in by_shape.values())
    assert assign_split(BY_NAME["keep_zeros"].skeleton) == assign_split(
        BY_NAME["keep_negative"].skeleton
    )


@pytest.mark.parametrize("seed", ["0", "1", "31337"])
def test_split_does_not_depend_on_pythonhashseed(seed):
    """SHA-256, not `hash`, so regenerating cannot move a shape across."""
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
    assert SCHEMA_VERSION == 2
    assert {task.schema_version for task in tasks} == {2}


def test_corpus_tasks_carry_their_skeleton_and_alternatives(tasks):
    for task in tasks:
        behaviour = BY_NAME[task.behaviour]
        assert task.program == behaviour.program
        assert task.skeleton == skeleton(task.program) == behaviour.skeleton
        assert task.alternative_programs == behaviour.programs[1:]
        assert task.split == assign_split(task.skeleton)
    with_alternatives = {t.behaviour for t in tasks if t.alternative_programs}
    assert with_alternatives == {
        "double_each",
        "increment_each",
        "negate_each",
        "sum",
        "product",
        "length",
        "last",
        "sum_both",
        "sum_of_squares",
    }


def test_every_alternative_program_reproduces_every_example(tasks):
    """An alternative is only a label if it solves the task, hidden half too."""
    for task in tasks:
        pairs = [
            *zip(task.inputs, task.outputs, strict=True),
            *zip(task.held_out_inputs, task.held_out_outputs, strict=True),
        ]
        for program in task.alternative_programs:
            for given, wanted in pairs:
                outcome = run_program(program, parse_stack(given, basis=BASIS))
                assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
                stack, _fuel = outcome
                want = parse_stack(wanted, basis=BASIS)
                assert tuple(map(value_key, stack)) == tuple(map(value_key, want))


def test_no_corpus_task_shows_a_single_list_length(tasks):
    """The `last` memorisation: three two-element lists, `uncons drop i`."""
    for task in tasks:
        shapes = set()
        for text in task.inputs:
            stack = parse_stack(text, basis=BASIS)
            shapes.add(tuple(len(v.items) for v in stack if isinstance(v, Lst)))
        if shapes != {()}:
            assert len(shapes) > 1, (task.behaviour, task.inputs)


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


def test_no_skeleton_spans_both_splits(tasks):
    by_skeleton: dict[str, set[str]] = {}
    for task in tasks:
        by_skeleton.setdefault(task.skeleton, set()).add(task.split)
    assert all(len(splits) == 1 for splits in by_skeleton.values())
    assert {task.split for task in tasks} == {"train", "eval"}
    filters = {t.split for t in tasks if t.behaviour in {"keep_zeros", "keep_negative"}}
    assert len(filters) == 1


def test_corpus_round_trips_through_load(tmp_path):
    made, _stats = generate(per_behaviour=1, seed=3)
    path = tmp_path / "corpus.jsonl.gz"
    from dataclasses import asdict

    with gzip.open(path, "wt") as handle:
        for task in made:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    assert load(str(path)) == made


def test_load_restores_alternative_programs_as_tuples(tmp_path):
    made, _stats = generate(per_behaviour=1, seed=3, catalogue=[BY_NAME["sum"]])
    path = tmp_path / "corpus.jsonl.gz"
    from dataclasses import asdict

    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(asdict(made[0]), sort_keys=True) + "\n")
    (task,) = load(str(path))
    assert task == made[0]
    assert task.alternative_programs == (("uncons", "[", "add", "]", "fold"),)
    assert all(isinstance(p, tuple) for p in task.alternative_programs)


def test_load_still_reads_the_schema_one_corpus():
    """`stack_pbe_v3` stays in place and stays loadable; its reports cite it."""
    tasks = load(str(V3_CORPUS))
    assert len(tasks) == 40 * 40
    assert {task.schema_version for task in tasks} == {1}
    assert {task.skeleton for task in tasks} == {""}
    assert {task.alternative_programs for task in tasks} == {()}
    assert {task.split for task in tasks} == {"train", "eval"}


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
