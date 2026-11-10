"""The corpus claims two things. Check both.

Every task carries a program and a length labelled `certified`. The program
must reproduce the outputs from the inputs, and the length must really be the
shortest -- which means exhaustively confirming nothing shorter works.
"""

import dataclasses
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    OP_BY_NAME,
    Config,
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

# `i` runs eleven ops of padding plus `1 add`; `drop 1 add` gets the same
# result directly. Two examples, so fuel is summed across them.
PADDED = (
    parse_stack("3 (dup drop dup drop dup drop dup drop dup drop 1 add)"),
    parse_stack("5 (dup drop dup drop dup drop dup drop dup drop 1 add)"),
)
SMALL_OPS = tuple(OP_BY_NAME[name] for name in ("1", "dup", "drop", "swap", "add", "i"))


def tie_key(path: CostedPath) -> tuple:
    return (path.score, path.length, path.program)


def brute_force_cost(inputs, max_depth, ops=OPS, max_score=None) -> dict:
    """Enumerate every legal program and keep the cheapest per behaviour."""
    root = CostedPath((), (0,) * len(inputs))
    best = {behaviour_key(inputs): root}
    layer = [(root, inputs)]
    for _ in range(max_depth):
        nxt = []
        for path, behaviour in layer:
            for op in ops:
                outcome = step_all_costed(behaviour, path.example_fuels, op)
                if outcome is None:
                    continue
                moved, fuels = outcome
                extended = CostedPath((*path.program, op.name), fuels)
                if max_score is not None and extended.score > max_score:
                    continue
                key = behaviour_key(moved)
                if key not in best or tie_key(extended) < tie_key(best[key]):
                    best[key] = extended
                nxt.append((extended, moved))
        layer = nxt
    return best


def test_costed_path_scores_are_exact_powers_of_kt():
    path = CostedPath(("map", "dup"), (3, 5))
    assert path.length == 2
    assert path.fuel == 8
    assert path.score == 32  # 8 * 2**2
    assert path.kt == 5.0  # 2 + log2(8)
    assert path.kt == pytest.approx(path.length + math.log2(path.fuel))

    empty = CostedPath((), (0, 0, 0))
    assert (empty.length, empty.fuel, empty.score, empty.kt) == (0, 0, 0, 0.0)


def test_costed_path_is_a_frozen_slots_dataclass():
    path = CostedPath(("dup",), (1,))
    assert not hasattr(path, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        path.program = ()
    assert path == CostedPath(("dup",), (1,))
    assert hash(path) == hash(CostedPath(("dup",), (1,)))


def test_step_all_costed_charges_each_example_its_own_fuel():
    inputs = (parse_stack("(1 2 3) (dup mul)"), parse_stack("(4 5) (dup mul)"))
    moved, fuels = step_all_costed(inputs, (10, 20), OP_BY_NAME["map"])
    assert behaviour_key(moved) == behaviour_key(
        (parse_stack("(1 4 9)"), parse_stack("(16 25)"))
    )
    # map costs 1, plus `dup mul` (2 fuel) once per element
    assert fuels == (10 + 1 + 3 * 2, 20 + 1 + 2 * 2)


def test_step_all_costed_runs_each_example_under_its_remaining_budget():
    inputs = (parse_stack("(1 2 3) (dup mul)"), parse_stack("(4 5) (dup mul)"))
    mapper = OP_BY_NAME["map"]
    assert step_all_costed(inputs, (0, 0), mapper) is not None
    # 5 fuel left is too little for a map costing 7, on that example alone
    assert step_all_costed(inputs, (DEFAULT_FUEL_LIMIT - 5, 0), mapper) is None
    # the second example's map costs 5; the machine wants budget left over
    assert step_all_costed(inputs, (0, DEFAULT_FUEL_LIMIT - 5), mapper) is None
    _moved, fuels = step_all_costed(inputs, (0, DEFAULT_FUEL_LIMIT - 6), mapper)
    assert fuels == (7, DEFAULT_FUEL_LIMIT - 1)
    # an exhausted example refuses even a one-fuel op
    dup = OP_BY_NAME["dup"]
    assert step_all_costed(inputs, (DEFAULT_FUEL_LIMIT, 0), dup) is None
    _moved, fuels = step_all_costed(inputs, (DEFAULT_FUEL_LIMIT - 1, 0), dup)
    assert fuels == (DEFAULT_FUEL_LIMIT, 1)


def test_step_all_costed_rejects_mismatched_fuels():
    with pytest.raises(ValueError):
        step_all_costed(PADDED, (0,), OP_BY_NAME["dup"])
    with pytest.raises(ValueError):
        step_all_costed(PADDED, (0, 0, 0), OP_BY_NAME["dup"])


def test_step_all_is_step_all_costed_from_zero():
    for inputs in (*FAMILIES["with_fold"], *FAMILIES["int_pair"], PADDED):
        for op in OPS:
            costed = step_all_costed(inputs, (0,) * len(inputs), op)
            plain = step_all(inputs, op)
            if costed is None:
                assert plain is None, op.name
            else:
                assert behaviour_key(plain) == behaviour_key(costed[0]), op.name


def test_cheapest_program_can_be_longer_than_shortest():
    """`i` reaches the result in one op; `drop 1 add` does in three, for less."""
    found, complete = expand_cost(PADDED, max_depth=3, ops=SMALL_OPS)
    assert complete
    target = behaviour_key((parse_stack("4"), parse_stack("6")))
    assert found[target] == CostedPath(("drop", "1", "add"), (3, 3))
    assert found[target].score == 6 << 3 < 26 << 1
    shortest, _ = expand(PADDED, max_depth=3)
    assert shortest[target][0] == 1


def test_a_non_cheapest_prefix_can_win_after_a_suffix():
    """Why the search keeps a Pareto frontier and not one label per behaviour.

    At `[4]` the long low-fuel arrival is cheaper (48 < 52), but after `dup`
    the short one wins: (26+2) << 2 = 112 against (6+2) << 4 = 128. A search
    keeping only the cheapest arrival at `[4]` would report `drop 1 add dup`.
    """
    found, complete = expand_cost(PADDED, max_depth=4, ops=SMALL_OPS)
    assert complete
    middle = behaviour_key((parse_stack("4"), parse_stack("6")))
    after = behaviour_key((parse_stack("4 4"), parse_stack("6 6")))
    assert found[middle].program == ("drop", "1", "add")
    assert found[after] == CostedPath(("i", "dup"), (14, 14))
    assert found == brute_force_cost(PADDED, 4, ops=SMALL_OPS)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_expand_cost_matches_brute_force(family):
    for inputs in FAMILIES[family]:
        found, complete = expand_cost(inputs, max_depth=2)
        assert complete
        assert found == brute_force_cost(inputs, 2), family


def test_expand_cost_matches_brute_force_at_depth_three():
    for inputs in (FAMILIES["with_map"][0], FAMILIES["with_fold"][0]):
        found, complete = expand_cost(inputs, max_depth=3)
        assert complete
        assert found == brute_force_cost(inputs, 3)


def test_expand_cost_reaches_what_breadth_first_search_reaches():
    """Same depth bound, same legal programs: the behaviour sets must agree."""
    inputs = FAMILIES["with_filter"][0]
    found, _ = expand_cost(inputs, max_depth=3)
    shortest, _ = expand(inputs, max_depth=3)
    assert set(found) == set(shortest)
    for key, path in found.items():
        assert path.length >= shortest[key][0]


def test_expand_cost_root_is_the_empty_path():
    inputs = FAMILIES["int_pair"][0]
    start = behaviour_key(inputs)
    for depth in (0, 2):
        found, complete = expand_cost(inputs, max_depth=depth)
        assert complete
        assert found[start] == CostedPath((), (0, 0, 0))
    assert expand_cost(inputs, max_depth=0) == (
        {start: CostedPath((), (0, 0, 0))},
        True,
    )
    assert expand_cost(inputs, max_depth=3, ops=()) == (
        {start: CostedPath((), (0, 0, 0))},
        True,
    )


def test_expand_cost_respects_max_score():
    inputs = FAMILIES["with_map"][0]
    full, _ = expand_cost(inputs, max_depth=3)
    for cap in (0, 6, 24, 100):
        capped, complete = expand_cost(inputs, max_depth=3, max_score=cap)
        assert complete
        assert capped == {k: v for k, v in full.items() if v.score <= cap}
        assert capped == brute_force_cost(inputs, 3, max_score=cap)
    assert list(expand_cost(inputs, max_depth=3, max_score=0)[0].values()) == [
        CostedPath((), (0, 0, 0))
    ]


def test_expand_cost_stops_at_max_states_with_exact_labels():
    """A truncated run returns the cheapest behaviours, labelled correctly."""
    inputs = FAMILIES["with_fold"][0]
    full, complete = expand_cost(inputs, max_depth=3)
    assert complete
    ranked = sorted(full, key=lambda k: tie_key(full[k]))
    for cap in (1, 2, 50):
        partial, complete = expand_cost(inputs, max_depth=3, max_states=cap)
        assert not complete
        assert partial == {k: full[k] for k in ranked[:cap]}
    exact, complete = expand_cost(inputs, max_depth=3, max_states=len(full))
    assert complete
    assert exact == full
    unbounded, complete = expand_cost(inputs, max_depth=3, max_states=None)
    assert complete
    assert unbounded == full


def test_expand_cost_validates_arguments():
    inputs = FAMILIES["int_pair"][0]
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=-1)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=2, max_states=0)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=2, max_score=-1)


def test_replay_costed_reproduces_expand_cost_labels():
    inputs = FAMILIES["with_fold"][0]
    found, _ = expand_cost(inputs, max_depth=3)
    for key, path in found.items():
        behaviour, replayed = replay_costed(inputs, path.program)
        assert behaviour_key(behaviour) == key
        assert replayed == path
        assert behaviour_key(replay(inputs, path.program)) == key


def test_replay_costed_charges_fuel_per_example():
    behaviour, path = replay_costed(PADDED, ("i",))
    assert behaviour_key(behaviour) == behaviour_key(
        (parse_stack("4"), parse_stack("6"))
    )
    assert path == CostedPath(("i",), (13, 13))
    assert path.score == 26 << 1


def test_replay_costed_rejects_illegal_steps():
    inputs = FAMILIES["bitlist3"][0]
    with pytest.raises(ValueError):
        replay_costed(inputs, ("add",))
    with pytest.raises(ValueError):
        replay_costed(inputs, ("dup", "no-such-op"))
    behaviour, path = replay_costed(inputs, ())
    assert behaviour == inputs
    assert path == CostedPath((), (0, 0, 0))
