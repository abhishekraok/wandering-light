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

import wandering_light.stack_machine_corpus as corpus
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    OP_BY_NAME,
    Config,
    format_stack,
    parse_stack,
    step_raw,
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
# Cost-ordered search: cheapest under Kt rather than fewest ops
# --------------------------------------------------------------------------


def pick(*names: str) -> tuple:
    return tuple(OP_BY_NAME[name] for name in names)


def rank(path: CostedPath) -> tuple:
    return (path.score, path.length, path.program)


def cheapest_by_enumeration(inputs, max_depth, ops=OPS, max_score=None) -> dict:
    """Every program up to `max_depth`, one at a time, keeping the best per key.

    Deliberately shares nothing with `expand_cost` but the machine: no queue,
    no pruning, no frontier. It reads the fuel limit when called, so a test
    that lowers the limit lowers it here too.
    """
    limit = corpus.DEFAULT_FUEL_LIMIT
    best: dict = {}

    def walk(behaviour, fuels, program):
        path = CostedPath(program, fuels)
        if max_score is None or path.score <= max_score:
            key = behaviour_key(behaviour)
            if key not in best or rank(path) < rank(best[key]):
                best[key] = path
        if len(program) == max_depth:
            return
        for op in ops:
            moved, spent = [], []
            for stack, fuel in zip(behaviour, fuels, strict=True):
                outcome = step_raw(op, stack, limit - fuel, 0)
                if isinstance(outcome, str):
                    break
                moved.append(outcome[0])
                spent.append(fuel + outcome[1])
            else:
                walk(tuple(moved), tuple(spent), (*program, op.name))

    walk(inputs, (0,) * len(inputs), ())
    return best


# A quotation that does `swap` the slow way: nine steps, so `i` costs ten.
SLOW_SWAP = "(dup drop dup drop dup drop dup drop swap)"
NON_ADDITIVE = (parse_stack(f"(1 add) (1 2 3 4 5 6 7 8) {SLOW_SWAP}"),)

# Two quotations that both push `(1 add)`. Which is the slow one differs by
# example, so "run the lower" and "run the upper" each win on one example.
PER_EXAMPLE = (
    parse_stack("(1) ((1 add)) (0 drop 0 drop 0 drop 0 drop (1 add))"),
    parse_stack("(1 2 3 4 5 6) (0 drop 0 drop 0 drop (1 add)) ((1 add))"),
)


def test_costed_path_scores_two_to_the_kt():
    path = CostedPath(("i", "map"), (20, 7))
    assert path.length == 2
    assert path.fuel == 27
    assert path.score == 27 << 2 == 108
    assert path.kt == math.log2(108)
    assert math.isclose(path.kt, path.length + math.log2(path.fuel))
    assert math.isclose(2**path.kt, path.score)


def test_costed_path_is_an_immutable_value():
    path = CostedPath(("dup",), (1, 1))
    assert path == CostedPath(("dup",), (1, 1))
    assert len({path, CostedPath(("dup",), (1, 1))}) == 1
    assert not hasattr(path, "__dict__")
    with pytest.raises(AttributeError):
        path.program = ()


def test_the_empty_path_costs_nothing():
    empty = CostedPath((), (0, 0, 0))
    assert (empty.length, empty.fuel, empty.score) == (0, 0, 0)
    assert empty.kt == 0.0
    assert isinstance(empty.kt, float)


def test_score_stays_exact_where_a_float_would_not():
    """The point of `fuel << length`: ordering never goes through a float."""
    program = ("dup",) * 80
    low, high = CostedPath(program, (3,)), CostedPath(program, (3, 1))
    assert low.score == 3 << 80
    assert high.score - low.score == 1 << 80
    assert math.isclose(low.kt, 80 + math.log2(3))


def test_step_all_costed_charges_each_example_what_it_spent():
    inputs = FAMILIES["with_fold"][0]  # lists of three, two and one
    fold = OP_BY_NAME["fold"]
    moved, fuels = step_all_costed(inputs, (0, 0, 0), fold)
    assert fuels == (4, 3, 2)  # one for fold, one per `add`
    assert behaviour_key(moved) == behaviour_key(step_all(inputs, fold))
    # earlier spending is carried forward per example, not pooled
    again, fuels = step_all_costed(inputs, (10, 20, 30), fold)
    assert fuels == (14, 23, 32)
    assert behaviour_key(again) == behaviour_key(moved)
    assert isinstance(moved, tuple) and isinstance(fuels, tuple)


def test_step_all_costed_gives_each_example_its_own_remaining_budget():
    inputs = FAMILIES["with_fold"][0]
    fold, dup = OP_BY_NAME["fold"], OP_BY_NAME["dup"]
    spent = (0, 0, DEFAULT_FUEL_LIMIT - 1)  # the third example has one left
    assert step_all_costed(inputs, spent, dup)[1] == (1, 1, DEFAULT_FUEL_LIMIT)
    assert step_all_costed(inputs, spent, fold) is None  # fold needs two there
    assert step_all_costed(inputs, (0, 0, DEFAULT_FUEL_LIMIT), dup) is None
    # the other examples' slack does not help the one that ran dry
    assert step_all_costed(inputs, (DEFAULT_FUEL_LIMIT, 0, 0), dup) is None


def test_step_all_costed_refuses_when_any_example_refuses():
    inputs = (parse_stack("2 1"), parse_stack("2 0"))
    assert step_all_costed(inputs, (0, 0), OP_BY_NAME["div"]) is None
    assert step_all_costed(inputs, (0, 0), OP_BY_NAME["add"]) is not None


def test_step_all_costed_rejects_mismatched_fuels():
    inputs = FAMILIES["int_pair"][0]
    for fuels in ((), (0, 0), (0, 0, 0, 0)):
        with pytest.raises(ValueError):
            step_all_costed(inputs, fuels, OP_BY_NAME["add"])
    with pytest.raises(ValueError):
        step_all_costed((), (0,), OP_BY_NAME["add"])


def test_step_all_is_step_all_costed_from_zero_fuel():
    for input_sets in FAMILIES.values():
        inputs = input_sets[0]
        zero = (0,) * len(inputs)
        for op in OPS:
            costed = step_all_costed(inputs, zero, op)
            plain = step_all(inputs, op)
            if costed is None:
                assert plain is None, op.name
            else:
                assert behaviour_key(plain) == behaviour_key(costed[0]), op.name


def test_replay_costed_agrees_with_replay_and_meters_fuel(tasks):
    for task in stratified(tasks, per_cell=3):
        inputs = tuple(parse_stack(s) for s in task["inputs"])
        program = tuple(task["program"])
        behaviour, path = replay_costed(inputs, program)
        assert behaviour_key(behaviour) == behaviour_key(replay(inputs, program))
        assert path.program == program
        assert len(path.example_fuels) == len(inputs)
        # every op costs at least one fuel on every example
        assert all(fuel >= len(program) for fuel in path.example_fuels), task
        if not any(OP_BY_NAME[name].runs_code for name in program):
            assert path.fuel == len(program) * len(inputs), task


def test_replay_costed_of_nothing_is_the_empty_path():
    inputs = FAMILIES["int_pair"][0]
    behaviour, path = replay_costed(inputs, ())
    assert behaviour_key(behaviour) == behaviour_key(inputs)
    assert path == CostedPath((), (0, 0, 0))


def test_replay_costed_raises_on_an_illegal_step():
    inputs = FAMILIES["int_pair"][0]
    with pytest.raises(ValueError):
        replay_costed(inputs, ("add", "add"))  # one integer left, add needs two
    with pytest.raises(ValueError):
        replay_costed(inputs, ("no-such-op",))


def test_expand_cost_matches_exhaustive_enumeration():
    """The claim itself: every entry is the cheapest program, ties included."""
    for family, input_sets in FAMILIES.items():
        for inputs in input_sets:
            found, complete = expand_cost(inputs, max_depth=3)
            assert complete, family
            assert found == cheapest_by_enumeration(inputs, 3), family


def test_expand_cost_reaches_what_bfs_reaches_and_never_pays_more():
    for family, input_sets in FAMILIES.items():
        inputs = input_sets[0]
        shortest, _ = expand(inputs, max_depth=3)
        cheapest, complete = expand_cost(inputs, max_depth=3)
        assert complete
        assert set(cheapest) == set(shortest), family
        for key, path in cheapest.items():
            depth, program, _witnesses = shortest[key]
            assert path.length >= depth
            assert path.score <= replay_costed(inputs, program)[1].score
            behaviour, replayed = replay_costed(inputs, path.program)
            assert behaviour_key(behaviour) == key
            assert replayed == path


def test_the_root_maps_to_the_empty_path():
    inputs = FAMILIES["with_map"][0]
    root = behaviour_key(inputs)
    found, complete = expand_cost(inputs, max_depth=0)
    assert complete
    assert found == {root: CostedPath((), (0, 0, 0))}
    # and nothing that returns to the root later displaces it
    found, _ = expand_cost(inputs, max_depth=2)
    assert found[root] == CostedPath((), (0, 0, 0))


def test_cheapest_is_not_fewest_ops():
    """One `i` runs nine steps; two plain ops do the same work for two."""
    target = behaviour_key((parse_stack("(1 2 3 4 5 6 7 8) (1 add)"),))
    shortest, _ = expand(NON_ADDITIVE, max_depth=2)
    cheapest, _ = expand_cost(NON_ADDITIVE, max_depth=2)
    assert shortest[target][:2] == (1, ("i",))
    assert cheapest[target] == CostedPath(("drop", "swap"), (2,))
    assert cheapest[target].score == 8 < replay_costed(NON_ADDITIVE, ("i",))[1].score


def test_kt_is_not_additive_so_one_label_per_behaviour_is_not_enough():
    """The cheapest way to a behaviour is not the cheapest way through it.

    `drop swap` (score 8) beats `i` (score 20) to the midpoint. Then `map`
    spends 17, and that is charged once per op already written: `i map` scores
    27 << 2 = 108 against 19 << 3 = 152 for `drop swap map`. A search keeping
    only the cheapest arrival at the midpoint would certify 152.
    """
    middle = behaviour_key((parse_stack("(1 2 3 4 5 6 7 8) (1 add)"),))
    end = behaviour_key((parse_stack("(2 3 4 5 6 7 8 9)"),))
    for ops in (OPS, pick("drop", "swap", "i", "map")):
        found, complete = expand_cost(NON_ADDITIVE, max_depth=3, ops=ops)
        assert complete
        assert found[middle] == CostedPath(("drop", "swap"), (2,))
        assert found[end] == CostedPath(("i", "map"), (27,))
        assert found[end].score == 108
        assert found[end].program[:1] != found[middle].program[:1]
    _, through_cheapest_middle = replay_costed(NON_ADDITIVE, ("drop", "swap", "map"))
    assert through_cheapest_middle.score == 152


def test_fuel_is_compared_per_example_because_the_budget_is(monkeypatch):
    """An arrival that spent less in total can still be the one that runs dry.

    `drop i` arrives having spent (3, 9); `i swap drop` arrives having spent
    (12, 4): one more op and more fuel in total, but less on the second
    example. With 20 fuel per example the `map` that follows costs 13 there,
    and only the second arrival can afford it. Pruning on total fuel would
    lose the behaviour altogether.
    """
    ops = pick("drop", "swap", "i", "map")
    middle = behaviour_key(
        (parse_stack("(1) (1 add)"), parse_stack("(1 2 3 4 5 6) (1 add)"))
    )
    end = behaviour_key((parse_stack("(2)"), parse_stack("(2 3 4 5 6 7)")))

    found, _ = expand_cost(PER_EXAMPLE, max_depth=4, ops=ops)
    assert found[middle] == CostedPath(("drop", "i"), (3, 9))
    assert found[end] == CostedPath(("drop", "i", "map"), (6, 22))

    monkeypatch.setattr(corpus, "DEFAULT_FUEL_LIMIT", 20)
    with pytest.raises(ValueError):
        replay_costed(PER_EXAMPLE, ("drop", "i", "map"))
    found, complete = expand_cost(PER_EXAMPLE, max_depth=4, ops=ops)
    assert complete
    assert found[middle] == CostedPath(("drop", "i"), (3, 9))
    assert found[end] == CostedPath(("i", "swap", "drop", "map"), (15, 17))
    assert found == cheapest_by_enumeration(PER_EXAMPLE, 4, ops)
    assert all(fuel <= 20 for path in found.values() for fuel in path.example_fuels)


def test_ties_go_to_the_shorter_then_the_lesser_program():
    inputs = (parse_stack("2 2"),)
    four = behaviour_key((parse_stack("4"),))
    for ops in (pick("add", "mul"), pick("mul", "add")):  # not the order tried
        found, _ = expand_cost(inputs, max_depth=1, ops=ops)
        assert found[four] == CostedPath(("add",), (1,))
    # equal score, different lengths: fuel 4 in one op against fuel 2 in two
    inputs = (parse_stack("7 5 (0 drop swap)"),)
    swapped = behaviour_key((parse_stack("5 7"),))
    assert replay_costed(inputs, ("drop", "swap"))[1].score == 8
    for ops in (pick("i", "drop", "swap"), pick("swap", "drop", "i")):
        found, _ = expand_cost(inputs, max_depth=2, ops=ops)
        assert found[swapped] == CostedPath(("i",), (4,))


def test_max_score_keeps_exactly_the_paths_that_cheap():
    inputs = FAMILIES["with_fold"][0]
    full, _ = expand_cost(inputs, max_depth=3)
    scores = sorted({path.score for path in full.values()})
    for max_score in (0, 5, 6, scores[len(scores) // 2], scores[-1] - 1, scores[-1]):
        found, complete = expand_cost(inputs, max_depth=3, max_score=max_score)
        assert complete
        expected = {k: v for k, v in full.items() if v.score <= max_score}
        assert found == expected, max_score
    assert len(expand_cost(inputs, max_depth=3, max_score=0)[0]) == 1
    assert expand_cost(inputs, max_depth=3, max_score=scores[-1])[0] == full
    # the bound is on score, so it can be the only bound that bites
    found, complete = expand_cost(inputs, max_depth=50, max_score=40)
    assert complete
    assert found == cheapest_by_enumeration(inputs, 4, max_score=40)
    assert max(path.score for path in found.values()) <= 40


def test_max_states_stops_at_the_cheapest_behaviours():
    """An early stop still certifies what it returns: the cheapest so far."""
    inputs = FAMILIES["list_int"][0]
    full, complete = expand_cost(inputs, max_depth=3, max_states=None)
    assert complete
    cheapest_first = sorted(full.values(), key=rank)
    for max_states in (1, 2, 17, len(full) - 1):
        found, complete = expand_cost(inputs, max_depth=3, max_states=max_states)
        assert not complete
        assert len(found) == max_states
        assert all(full[key] == path for key, path in found.items())
        assert sorted(found.values(), key=rank) == cheapest_first[:max_states]
    found, complete = expand_cost(inputs, max_depth=3, max_states=len(full))
    assert complete
    assert found == full
    assert expand_cost(inputs, max_depth=0, max_states=1) == (
        {behaviour_key(inputs): CostedPath((), (0, 0))},
        True,
    )


def test_expand_cost_rejects_bounds_that_mean_nothing():
    inputs = FAMILIES["int_pair"][0]
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=-1)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=2, max_states=0)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=2, max_states=-3)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=2, max_score=-1)
    for found, complete in (
        expand_cost(inputs, max_depth=0),
        expand_cost(inputs, 1, None),
        expand_cost(inputs, 1, 1, max_score=0),
    ):
        assert behaviour_key(inputs) in found
        assert isinstance(complete, bool)


def test_a_task_with_no_examples_has_one_behaviour_and_free_programs():
    found, complete = expand_cost((), max_depth=3)
    assert complete
    assert found == {(): CostedPath((), ())}
    _, path = replay_costed((), ("dup", "add"))
    assert (path.fuel, path.score) == (0, 0)
    assert path.kt == -math.inf


@pytest.mark.slow
def test_expand_cost_matches_exhaustive_enumeration_one_op_deeper():
    for family in ("int_pair", "with_fold", "with_apply"):
        inputs = FAMILIES[family][0]
        found, complete = expand_cost(inputs, max_depth=4)
        assert complete, family
        assert found == cheapest_by_enumeration(inputs, 4), family
