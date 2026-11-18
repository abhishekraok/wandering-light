"""The corpus claims two things. Check both.

Every task carries a program and a length labelled `certified`. The program
must reproduce the outputs from the inputs, and the length must really be the
shortest -- which means exhaustively confirming nothing shorter works.
"""

import gzip
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pytest

from wandering_light import stack_machine_corpus
from wandering_light.stack_machine import (
    OP_BY_NAME,
    Config,
    Op,
    format_stack,
    parse_stack,
)
from wandering_light.stack_machine_corpus import (
    BASIS,
    FAMILIES,
    OPS,
    CostedPath,
    Task,
    assign_split,
    behaviour_key,
    encode_task,
    expand,
    expand_cost,
    generate,
    is_useful,
    replay,
    replay_costed,
    step_all,
    step_all_costed,
)

CORPUS = (
    Path(__file__).resolve().parent.parent
    / "wandering_light"
    / "evals"
    / "data"
    / "stack_pbe_v2.jsonl.gz"
)


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    with gzip.open(CORPUS, "rt") as handle:
        return [json.loads(line) for line in handle]


def test_parse_stack_round_trips_the_display_form():
    """The corpus stores stacks as display text, so the parser is load-bearing."""
    for text in (
        "[]",
        "[ nil 1 ]",
        "[ (1 0 1) ]",
        "[ ((1) nil true) -4 ]",
        "[ (true 0 0 1) ]",
        "[ false ]",
        "[ (1 0 1) (1 add) ]",
        "[ (1 2 3) 0 (add) ]",
        "[ ((dup mul) map) ]",
    ):
        assert format_stack(Config(parse_stack(text))) == text


def test_corpus_is_not_empty(tasks):
    assert len(tasks) > 100
    assert {t["family"] for t in tasks} <= set(FAMILIES)


def test_every_task_records_the_basis_it_was_certified_against(tasks):
    """Labels are basis-relative; a corpus that does not say which is useless."""
    assert {t["basis"] for t in tasks} == {BASIS}


def test_the_split_is_deterministic_and_covers_both_sides(tasks):
    counts = Counter(t["split"] for t in tasks)
    assert set(counts) == {"train", "eval"}
    share = counts["eval"] / len(tasks)
    assert 0.10 < share < 0.30, counts
    # and recomputing it from the task itself must agree
    for task in tasks:
        assert assign_split(tuple(task["inputs"]), tuple(task["outputs"])) == task["split"]


def test_no_task_appears_in_both_splits(tasks):
    """The split is keyed on the task, so an identical task cannot straddle."""
    seen: dict[tuple, str] = {}
    for task in tasks:
        key = (tuple(task["inputs"]), tuple(task["outputs"]))
        if key in seen:
            assert seen[key] == task["split"], key
        seen[key] = task["split"]


def test_the_combinators_are_actually_exercised(tasks):
    """The point of this basis is loops; the corpus must contain some."""
    used = Counter(op for t in tasks for op in t["program"])
    for name in ("map", "fold", "filter", "i"):
        assert used[name] > 0, name
    variable_length = [
        t for t in tasks if "map" in t["program"] or "fold" in t["program"]
    ]
    assert variable_length, "no task needs a loop"


def test_encoding_round_trips_into_the_program(tasks):
    """The training encoding must recover exactly the certified program."""
    for task in tasks[:200]:
        record = Task(**task)
        encoded = encode_task(record)
        assert encoded["completion"].split() == list(task["program"])
        for given in task["inputs"]:
            assert given in encoded["prompt"]
        for wanted in task["outputs"]:
            assert wanted in encoded["prompt"]


def test_every_program_reproduces_its_outputs(tasks):
    """The witness must actually solve the task, on every example."""
    for task in tasks:
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        produced = replay(inputs, tuple(task["program"]))
        expected = tuple(parse_stack(s) for s in task["outputs"])
        assert behaviour_key(produced) == behaviour_key(expected), task


def test_program_length_matches_the_label(tasks):
    for task in tasks:
        assert len(task["program"]) == task["length"], task


def stratified(tasks, per_cell: int, max_length: int = 99, min_length: int = 2):
    """A sample covering every (family, length) cell, not just the first rows.

    Re-certifying a task costs a full exhaustive search one op shorter than its
    label, so cost grows with the label. Callers bound the length to keep the
    default suite fast; the slow test covers the rest.
    """
    from collections import defaultdict

    cells = defaultdict(list)
    for task in tasks:
        if min_length <= task["length"] <= max_length:
            cells[(task["family"], task["length"])].append(task)
    out = []
    for key in sorted(cells):
        out.extend(cells[key][:per_cell])
    return out


def test_labels_are_certified_shortest(tasks):
    """Exhaustively confirm no shorter program has the same behaviour.

    Search every program strictly shorter than the label and require the task's
    behaviour to be absent. This is the claim the corpus exists to make, so it
    is checked directly rather than trusted from the generator.
    """
    sample = stratified(tasks, per_cell=3, max_length=4)
    families = {t["family"] for t in sample}
    assert families == set(FAMILIES), families
    assert {t["length"] for t in sample} >= {2, 3, 4}
    for task in sample:
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        target = behaviour_key(tuple(parse_stack(s) for s in task["outputs"]))
        found, complete = expand(inputs, max_depth=task["length"] - 1)
        assert complete, task
        assert target not in found, f"shorter program exists for {task}"


def test_the_behaviour_is_reachable_at_exactly_that_length(tasks):
    for task in stratified(tasks, per_cell=2, max_length=4):
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        target = behaviour_key(tuple(parse_stack(s) for s in task["outputs"]))
        found, _ = expand(inputs, max_depth=task["length"])
        assert found[target][0] == task["length"], task


def test_tasks_are_not_degenerate(tasks):
    """Every task leaves one answer, and that answer depends on the input."""
    for task in tasks:
        outputs = tuple(parse_stack(s) for s in task["outputs"])
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        assert all(len(stack) == 1 for stack in outputs), task
        assert is_useful(inputs, outputs), task


def test_witness_count_is_at_least_one(tasks):
    for task in tasks:
        assert task["witnesses"] >= 1, task


def test_multi_example_guard_is_the_conjunction():
    """An op is legal on a task exactly when it is legal on every example."""
    inputs = FAMILIES["int_pair"][0]
    for op in OPS:
        each = [op.guard(Config(stack)) is None for stack in inputs]
        assert (step_all(inputs, op) is not None) == all(each), op.name


def test_replay_costed_counts_loop_work_on_each_example():
    inputs = (parse_stack("(1 2 3) (1 add)"), parse_stack("(5) (1 add)"))
    program = ("map", "dup")
    behaviour, path = replay_costed(inputs, program)

    assert behaviour == replay(inputs, program)
    assert path == CostedPath(program, (8, 4))
    assert path.length == 2
    assert path.fuel == 12
    assert path.score == 48
    assert path.kt == pytest.approx(5.5849625)

    root = CostedPath((), (0, 0))
    assert replay_costed(inputs, ()) == (inputs, root)
    assert root.score == 0
    assert root.kt == 0.0


def test_costed_steps_enforce_each_examples_remaining_budget(monkeypatch):
    monkeypatch.setattr(stack_machine_corpus, "DEFAULT_FUEL_LIMIT", 9)
    inputs = (parse_stack("(1 2 3) (1 add)"), parse_stack("(5) (1 add)"))
    op = OP_BY_NAME["map"]

    assert step_all_costed(inputs, (1, 5), op) == (step_all(inputs, op), (8, 8))
    assert step_all_costed(inputs, (2, 5), op) is None
    assert step_all_costed(inputs, (1, 6), op) is None
    with pytest.raises(ValueError, match="one fuel count per example"):
        step_all_costed(inputs, (9,), op)
    with pytest.raises(ValueError, match="dup is illegal"):
        replay_costed(inputs, ("map", "dup", "dup", "dup"))
    with pytest.raises(ValueError, match="unknown is illegal"):
        replay_costed(inputs, ("unknown",))


@pytest.mark.parametrize(
    ("noop_pairs", "intermediate_path"),
    [
        # One pair gives equal scores, so the shorter program wins the tie.
        (1, CostedPath(("i",), (4,))),
        (3, CostedPath(("drop", "drop"), (2,))),
    ],
)
def test_expand_cost_keeps_shorter_arrivals_for_an_expensive_suffix(
    noop_pairs, intermediate_path
):
    suffix = "0 drop " * 8 + "7"
    prefix = "dup drop " * noop_pairs + "drop"
    inputs = (parse_stack(f"({suffix}) 42 ({prefix})"),)
    ops = (OP_BY_NAME["drop"], OP_BY_NAME["i"])

    found, complete = expand_cost(inputs, max_depth=3, ops=ops)

    assert complete
    intermediate = behaviour_key((parse_stack(f"({suffix})"),))
    assert found[intermediate] == intermediate_path
    # In both cases the expensive suffix makes the shorter arrival cheaper.
    assert found[behaviour_key((parse_stack("7"),))] == CostedPath(
        ("i", "i"), (20 + 2 * noop_pairs,)
    )


def test_expand_cost_keeps_arrivals_with_incomparable_example_fuels(monkeypatch):
    monkeypatch.setattr(stack_machine_corpus, "DEFAULT_FUEL_LIMIT", 8)
    first, second = parse_stack("1"), parse_stack("2")
    middle_first, middle_second = parse_stack("3"), parse_stack("4")
    end = parse_stack("5")

    def make_op(name, transitions):
        def run(stack, budget, _depth):
            moved, spent = transitions.get(stack, (stack, budget + 1))
            return (moved, spent) if spent <= budget else "refused"

        return Op(name, "test", "", lambda _: None, lambda _: None, run)

    cheap = make_op("cheap", {first: (middle_first, 4), second: (middle_second, 1)})
    balanced = make_op(
        "balanced", {first: (middle_first, 1), second: (middle_second, 5)}
    )
    finish = make_op("finish", {middle_first: (end, 5), middle_second: (end, 1)})

    found, complete = expand_cost((first, second), 2, ops=(cheap, balanced, finish))

    assert complete
    assert found[behaviour_key((middle_first, middle_second))] == CostedPath(
        ("cheap",), (4, 1)
    )
    assert found[behaviour_key((end, end))] == CostedPath(
        ("balanced", "finish"), (6, 6)
    )


def test_expand_cost_matches_exhaustive_replay():
    inputs = FAMILIES["with_map"][0]
    ops = tuple(reversed(OPS))
    expected = {}
    for depth in range(4):
        for program in product((op.name for op in ops), repeat=depth):
            try:
                behaviour, path = replay_costed(inputs, program)
            except ValueError:
                continue
            key = behaviour_key(behaviour)
            best = expected.get(key)
            if best is None or (path.score, path.length, path.program) < (
                best.score,
                best.length,
                best.program,
            ):
                expected[key] = path

    assert expand_cost(inputs, 3, max_states=None, ops=ops) == (expected, True)


def test_expand_cost_applies_inclusive_limits_in_cost_order():
    inputs = (parse_stack("(1 2 3) (1 add)"),)
    root = {behaviour_key(inputs): CostedPath((), (0,))}
    dropped = {behaviour_key((parse_stack("(1 2 3)"),)): CostedPath(("drop",), (1,))}
    ops = (OP_BY_NAME["map"], OP_BY_NAME["drop"])

    assert expand_cost(inputs, 0, max_states=1, ops=ops) == (root, True)
    assert expand_cost(inputs, 1, max_score=0, ops=ops) == (root, True)
    assert expand_cost(inputs, 1, max_states=1, ops=ops) == (root, False)
    assert expand_cost(inputs, 1, max_states=2, ops=ops) == (root | dropped, False)
    assert expand_cost(inputs, 2, max_states=2, max_score=2, ops=ops) == (
        root | dropped,
        True,
    )


def test_expand_cost_breaks_equal_cost_and_length_ties_by_program():
    inputs = (parse_stack("nil"),)
    ops = (OP_BY_NAME["i"], OP_BY_NAME["drop"])

    found, complete = expand_cost(inputs, 1, max_states=2, ops=ops)

    assert complete
    assert found[behaviour_key(((),))] == CostedPath(("drop",), (1,))


@pytest.mark.parametrize(
    "limits",
    [
        {"max_depth": -1},
        {"max_states": 0},
        {"max_score": -1},
    ],
)
def test_expand_cost_rejects_invalid_limits(limits):
    with pytest.raises(ValueError):
        expand_cost(((),), **({"max_depth": 0} | limits))


def test_generator_is_deterministic():
    a, _ = generate(max_depth=3, per_depth=5, seed=1)
    b, _ = generate(max_depth=3, per_depth=5, seed=1)
    assert [t.program for t in a] == [t.program for t in b]


@pytest.mark.slow
def test_long_labels_are_certified_shortest(tasks):
    """The same check at length 5, where each search is much larger.

    Length 6 is deliberately excluded: re-certifying one costs a full expansion
    to depth 5, minutes per task. It is covered instead, and more broadly, by
    `experiments/stack_machine_baseline.py` -- that runs a separate BFS
    implementation over the whole eval split and reports whether the length it
    finds matches the label. That is the same minimality claim over all 355
    tasks rather than a sample of them.
    """
    sample = stratified(tasks, per_cell=2, min_length=5, max_length=5)
    assert sample, "expected long tasks to check"
    for task in sample:
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        target = behaviour_key(tuple(parse_stack(s) for s in task["outputs"]))
        found, complete = expand(inputs, max_depth=task["length"] - 1)
        assert complete, task
        assert target not in found, f"shorter program exists for {task}"
