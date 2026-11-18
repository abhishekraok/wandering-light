"""The corpus claims two things. Check both.

Every task carries a program and a length labelled `certified`. The program
must reproduce the outputs from the inputs, and the length must really be the
shortest -- which means exhaustively confirming nothing shorter works.
"""

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from wandering_light import stack_machine_corpus
from wandering_light.stack_machine import OP_BY_NAME, Config, format_stack, parse_stack
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
    from wandering_light.stack_machine import Config
    from wandering_light.stack_machine_corpus import step_all

    inputs = FAMILIES["int_pair"][0]
    for op in OPS:
        each = [op.guard(Config(stack)) is None for stack in inputs]
        assert (step_all(inputs, op) is not None) == all(each), op.name


def test_generator_is_deterministic():
    a, _ = generate(max_depth=3, per_depth=5, seed=1)
    b, _ = generate(max_depth=3, per_depth=5, seed=1)
    assert [t.program for t in a] == [t.program for t in b]


def test_costed_replay_counts_work_on_every_example():
    inputs = (parse_stack("(2 3 4) (dup mul)"), parse_stack("(5) (dup mul)"))
    behaviour, root = replay_costed(inputs, ())
    assert behaviour == inputs
    assert root == CostedPath((), (0, 0))
    assert (root.length, root.fuel, root.score, root.kt) == (0, 0, 0, 0.0)

    behaviour, path = replay_costed(inputs, ("map", "dup"))
    assert behaviour == replay(inputs, path.program)
    assert path == CostedPath(("map", "dup"), (8, 4))
    assert (path.length, path.fuel, path.score) == (2, 12, 48)
    assert path.kt == pytest.approx(math.log2(48))


def test_costed_steps_use_the_remaining_budget_on_each_example(monkeypatch):
    monkeypatch.setattr(stack_machine_corpus, "DEFAULT_FUEL_LIMIT", 4)
    inputs = (parse_stack("(1 2)"), parse_stack("(3 4)"))
    op = OP_BY_NAME["i"]
    assert step_all_costed(inputs, (1, 0), op) == (
        (parse_stack("1 2"), parse_stack("3 4")),
        (4, 3),
    )
    assert step_all_costed(inputs, (0, 2), op) is None
    with pytest.raises(ValueError, match="one fuel count"):
        step_all_costed(inputs, (4,), op)
    with pytest.raises(ValueError, match="illegal"):
        replay_costed(inputs, ("i", "dup", "drop"))
    with pytest.raises(ValueError, match="unknown"):
        replay_costed(inputs, ("missing",))


def test_shorter_arrival_wins_after_expensive_suffix():
    inputs = (parse_stack("(7 dup drop dup drop) 12 (dup drop dup drop drop)"),)
    ops = (OP_BY_NAME["i"], OP_BY_NAME["drop"])
    found, complete = expand_cost(inputs, 3, ops=ops)
    assert complete
    # Both prefixes expose the bottom quotation. Two drops cost less Kt
    # here, but the one-op prefix pays a smaller length factor on the suffix.
    middle, short = replay_costed(inputs, ("i",))
    assert short.score == 12
    assert found[behaviour_key(middle)] == CostedPath(("drop", "drop"), (2,))
    target, longer = replay_costed(inputs, ("drop", "drop", "i"))
    assert longer.score == 64
    assert found[behaviour_key(target)] == CostedPath(("i", "i"), (12,))
    # Even a truncated search settles the cheaper two-drop arrival first.
    capped, complete = expand_cost(inputs, 3, max_states=3, ops=ops)
    assert not complete
    assert capped[behaviour_key(middle)] == found[behaviour_key(middle)]


def test_incomparable_example_fuels_allow_different_suffixes(monkeypatch):
    monkeypatch.setattr(stack_machine_corpus, "DEFAULT_FUEL_LIMIT", 14)
    # Test for nil on top. dip hides the top value, so the examples take
    # opposite branches under dip and i, each leaving its stack intact.
    body = "((nil eq?) (dup drop dup drop) () ifte)"
    inputs = (parse_stack(f"1 () {body}"), parse_stack(f"() (1 1 add) {body}"))
    ops = (OP_BY_NAME["i"], OP_BY_NAME["dip"])
    found, complete = expand_cost(inputs, 2, ops=ops)
    assert complete
    middle, path = replay_costed(inputs, ("i",))
    assert path.example_fuels == (11, 7)
    assert found[behaviour_key(middle)] == CostedPath(("dip",), (7, 11))
    # The lexicographic winner lacks the four fuel needed on example two.
    target, path = replay_costed(inputs, ("i", "i"))
    assert found[behaviour_key(target)] == path


def test_cost_search_breaks_ties_by_length_then_program():
    inputs = (parse_stack("42 (dup drop drop)"),)
    ops = (OP_BY_NAME["drop"], OP_BY_NAME["i"])
    found, complete = expand_cost(inputs, 2, ops=ops)
    assert complete
    target, two_drops = replay_costed(inputs, ("drop", "drop"))
    winner = found[behaviour_key(target)]
    assert winner.score == two_drops.score
    assert winner.program == ("i",)

    inputs = (parse_stack("1"),)
    target = behaviour_key((parse_stack("1 1 1"),))
    for basis in (("dup", "1"), ("1", "dup")):
        found, complete = expand_cost(
            inputs, 2, ops=(OP_BY_NAME[name] for name in basis)
        )
        assert complete
        assert found[target].program == ("1", "1")


def test_cost_search_bounds_are_inclusive():
    inputs = (parse_stack("5"),)
    ops = (OP_BY_NAME["dup"],)
    root = {behaviour_key(inputs): CostedPath((), (0,))}
    assert expand_cost(inputs, 0) == (root, True)
    assert expand_cost(inputs, 2, max_score=0) == (root, True)
    assert expand_cost(inputs, 2, max_states=1, ops=ops) == (root, False)
    expected = {
        **root,
        behaviour_key((parse_stack("5 5"),)): CostedPath(("dup",), (1,)),
    }
    assert expand_cost(inputs, 2, None, max_score=2, ops=ops) == (expected, True)
    assert expand_cost(inputs, 1, 2, ops=ops) == (expected, True)
    assert expand_cost(inputs, 2, 2, ops=ops) == (expected, False)


@pytest.mark.parametrize(
    "kwargs", [{"max_depth": -1}, {"max_states": 0}, {"max_score": -1}]
)
def test_cost_search_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        expand_cost((), **({"max_depth": 1} | kwargs))


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
