"""The corpus claims two things. Check both.

Every task carries a program and a length labelled `certified`. The program
must reproduce the outputs from the inputs, and the length must really be the
shortest -- which means exhaustively confirming nothing shorter works.
"""

import gzip
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Config,
    Int,
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
        assert (
            assign_split(tuple(task["inputs"]), tuple(task["outputs"])) == task["split"]
        )


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


# --------------------------------------------------------------------------
# Cost-ordered search under Levin's Kt
# --------------------------------------------------------------------------

BY_NAME = {op.name: op for op in OPS}


def test_costed_path_properties():
    empty = CostedPath((), (0, 0))
    assert (empty.length, empty.fuel, empty.score, empty.kt) == (0, 0, 0, 0.0)
    path = CostedPath(("dup", "map"), (3, 9))
    assert path.length == 2
    assert path.fuel == 12
    assert path.score == 12 << 2 == 48
    assert path.kt == pytest.approx(2 + math.log2(12))
    with pytest.raises(AttributeError):
        path.program = ()  # frozen
    assert not hasattr(path, "__dict__")  # slots


def test_step_all_costed_charges_each_example_separately():
    inputs = (parse_stack("(1 2 3) (dup mul)"), parse_stack("(4) (dup mul)"))
    moved, fuels = step_all_costed(inputs, (5, 0), BY_NAME["map"])
    assert moved == step_all(inputs, BY_NAME["map"])
    # map costs 1, plus 2 per element for the body
    assert fuels == (5 + 1 + 3 * 2, 0 + 1 + 1 * 2)
    assert step_all_costed(inputs, (0, 0), BY_NAME["add"]) is None
    with pytest.raises(ValueError):
        step_all_costed(inputs, (0,), BY_NAME["dup"])


def test_step_all_costed_respects_each_examples_remaining_budget():
    inputs = (parse_stack("1"), parse_stack("1"))
    assert step_all_costed(inputs, (0, DEFAULT_FUEL_LIMIT - 1), BY_NAME["dup"])
    assert step_all_costed(inputs, (0, DEFAULT_FUEL_LIMIT), BY_NAME["dup"]) is None


def test_replay_costed_matches_replay():
    inputs = FAMILIES["with_fold"][0]
    behaviour, path = replay_costed(inputs, ("fold",))
    assert behaviour == replay(inputs, ("fold",))
    assert path.program == ("fold",)
    assert path.example_fuels == (1 + 3, 1 + 2, 1 + 1)
    with pytest.raises(ValueError):
        replay_costed(inputs, ("fold", "fold"))


def test_expand_cost_validates_arguments():
    inputs = FAMILIES["int_pair"][0]
    for kwargs in (
        {"max_depth": -1},
        {"max_depth": 1, "max_states": 0},
        {"max_depth": 1, "max_score": -1},
    ):
        with pytest.raises(ValueError):
            expand_cost(inputs, **kwargs)
    found, complete = expand_cost(inputs, 1, None, max_score=0)
    assert complete
    assert found == {behaviour_key(inputs): CostedPath((), (0, 0, 0))}


def test_expand_cost_reaches_what_bfs_reaches_and_labels_replay():
    for inputs in (FAMILIES["with_map"][0], FAMILIES["bool_int"][0]):
        bfs, _ = expand(inputs, 3)
        found, complete = expand_cost(inputs, 3)
        assert complete
        assert set(found) == set(bfs)
        for key, path in found.items():
            assert path.length >= bfs[key][0]
            behaviour, replayed = replay_costed(inputs, path.program)
            assert behaviour_key(behaviour) == key
            assert replayed == path


def exhaustive_cheapest(inputs, ops, max_depth):
    """Every program up to max_depth, keeping the min (score, length, program)."""
    best: dict = {}
    for length in range(max_depth + 1):
        for program in itertools.product(ops, repeat=length):
            behaviour, fuels = inputs, (0,) * len(inputs)
            for op in program:
                stepped = step_all_costed(behaviour, fuels, op)
                if stepped is None:
                    break
                behaviour, fuels = stepped
            else:
                path = CostedPath(tuple(op.name for op in program), fuels)
                key = behaviour_key(behaviour)
                rank = (path.score, path.length, path.program)
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, path)
    return {key: path for key, (_rank, path) in best.items()}


def test_expand_cost_agrees_with_exhaustive_enumeration():
    ops = tuple(
        BY_NAME[n]
        for n in ("dup", "swap", "drop", "over", "cons", "i", "map", "dip", "quote")
    )
    inputs = (parse_stack("(1 0) (dup add)"), parse_stack("(2 2 2) (0 add)"))
    found, complete = expand_cost(inputs, 4, None, ops=ops)
    assert complete
    assert found == exhaustive_cheapest(inputs, ops, 4)


def test_cheapest_program_can_be_longer_than_shortest():
    """A handed-over quotation can be short to call and expensive to run."""
    body = "drop 1" + " dup drop" * 10
    inputs = (parse_stack(f"0 ({body})"), parse_stack(f"5 ({body})"))
    bfs, _ = expand(inputs, 3)
    found, complete = expand_cost(inputs, 3)
    assert complete
    key = behaviour_key((parse_stack("1"), parse_stack("1")))
    assert bfs[key][:2] == (1, ("i",))
    cheapest = found[key]
    assert cheapest.length == 3
    assert cheapest.score < replay_costed(inputs, ("i",))[1].score
    assert cheapest.example_fuels == (3, 3)


def _toy_op(name: str, source: int, target: int, cost: int) -> Op:
    def guard(config):
        top = config.stack[-1] if config.stack else None
        return None if top == Int(source) else f"top is not {source}"

    return Op(
        name=name,
        group="toy",
        effect=f"( {source} -- {target} )",
        _guard=guard,
        _apply=lambda config: ((Int(target),), cost),
    )


def test_pareto_frontier_keeps_the_dominated_now_but_cheaper_later_arrival():
    """Kt is not additive, so the first arrival at a state is not always the
    one to extend.

    State 2 is settled by `s t u` (fuel 3, length 3, score 24). Only after that
    does `a big` arrive (fuel 14, length 2, score 56): worse now, so a search
    that stopped extending settled states would drop it. But after the
    expensive suffix `x`, `a big x` scores 114 << 3 = 912 against `s t u x` at
    103 << 4 = 1648.
    """
    ops = (
        _toy_op("a", 0, 4, 13),
        _toy_op("big", 4, 2, 1),
        _toy_op("s", 0, 1, 1),
        _toy_op("t", 1, 5, 1),
        _toy_op("u", 5, 2, 1),
        _toy_op("x", 2, 3, 100),
    )
    inputs = ((Int(0),),)
    found, complete = expand_cost(inputs, 4, ops=ops)
    assert complete
    assert found[behaviour_key(((Int(2),),))] == CostedPath(("s", "t", "u"), (3,))
    assert found[behaviour_key(((Int(3),),))] == CostedPath(("a", "big", "x"), (114,))
    assert found == exhaustive_cheapest(inputs, ops, 4)


def test_expand_cost_max_score_and_max_states():
    inputs = FAMILIES["with_map"][0]
    full, _ = expand_cost(inputs, 3)
    capped, complete = expand_cost(inputs, 3, max_score=40)
    assert complete
    assert capped == {k: p for k, p in full.items() if p.score <= 40}

    for limit in (1, 50):
        found, complete = expand_cost(inputs, 3, max_states=limit)
        assert not complete
        assert len(found) <= limit
        assert all(full[k] == p for k, p in found.items())
    found, complete = expand_cost(inputs, 3, max_states=len(full))
    assert complete
    assert found == full
