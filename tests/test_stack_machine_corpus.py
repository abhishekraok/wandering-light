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
# Cost-ordered search under Levin's Kt
# --------------------------------------------------------------------------


def test_costed_path_scores_are_exact_powers_of_kt():
    empty = CostedPath((), (0, 0))
    assert (empty.length, empty.fuel, empty.score, empty.kt) == (0, 0, 0, 0.0)

    path = CostedPath(("dup", "add"), (3, 5))
    assert path.length == 2
    assert path.fuel == 8
    assert path.score == 8 << 2 == 32
    assert path.kt == 5.0
    assert 2**path.kt == path.score

    odd = CostedPath(("map",), (7, 6))
    assert odd.score == 26
    assert odd.kt == pytest.approx(1 + math.log2(13))


def test_costed_path_is_frozen_and_slotted():
    path = CostedPath(("add",), (1,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        path.program = ()  # type: ignore[misc]
    assert not hasattr(path, "__dict__")
    assert CostedPath(("add",), (1,)) == path
    assert {path: 1}[CostedPath(("add",), (1,))] == 1


def test_step_all_costed_charges_each_example_what_it_spent():
    inputs = FAMILIES["int_pair"][0]
    moved = step_all_costed(inputs, (0, 0, 0), OP_BY_NAME["add"])
    assert moved is not None
    behaviour, fuels = moved
    assert behaviour_key(behaviour) == behaviour_key(((Int(8),), (Int(9),), (Int(13),)))
    assert fuels == (1, 1, 1)
    # fuel accumulates on top of what was already spent
    assert step_all_costed(inputs, (4, 0, 9), OP_BY_NAME["add"])[1] == (5, 1, 10)


def test_step_all_costed_charges_a_combinator_for_the_work_it_runs():
    """`map` costs more than one; the per-example charge must match `Op.apply`."""
    inputs = FAMILIES["with_map"][0]
    op = OP_BY_NAME["map"]
    moved = step_all_costed(inputs, (0, 0, 0), op)
    assert moved is not None
    _behaviour, fuels = moved
    assert all(f > 1 for f in fuels)
    assert fuels == tuple(op.apply(Config(stack))[1] for stack in inputs)


def test_step_all_costed_refuses_when_any_example_refuses():
    inputs = FAMILIES["int_pair"][0]
    assert step_all_costed(inputs, (0, 0, 0), OP_BY_NAME["map"]) is None
    # one example out of budget is enough to refuse the whole task
    nearly = (0, DEFAULT_FUEL_LIMIT, 0)
    assert step_all_costed(inputs, nearly, OP_BY_NAME["add"]) is None
    # and the budget is what is left, not the whole limit
    just = (0, DEFAULT_FUEL_LIMIT - 1, 0)
    assert step_all_costed(inputs, just, OP_BY_NAME["add"]) is not None


def test_step_all_costed_rejects_mismatched_fuels():
    inputs = FAMILIES["int_pair"][0]
    with pytest.raises(ValueError):
        step_all_costed(inputs, (0, 0), OP_BY_NAME["add"])
    with pytest.raises(ValueError):
        step_all_costed(inputs, (0, 0, 0, 0), OP_BY_NAME["add"])


def test_step_all_is_the_costed_step_from_a_full_budget():
    for inputs in (FAMILIES["int_pair"][0], FAMILIES["with_map"][0]):
        for op in OPS:
            moved = step_all_costed(inputs, (0,) * len(inputs), op)
            plain = step_all(inputs, op)
            if moved is None:
                assert plain is None, op.name
            else:
                assert plain is not None, op.name
                assert behaviour_key(plain) == behaviour_key(moved[0]), op.name


def test_replay_costed_matches_replay_and_the_search():
    inputs = FAMILIES["with_fold"][0]
    program = ("fold",)
    behaviour, path = replay_costed(inputs, program)
    assert behaviour_key(behaviour) == behaviour_key(replay(inputs, program))
    assert path.program == program
    assert path.length == 1
    assert path.fuel > len(inputs), "fold ran its quotation, so it costs more than one"
    found, _ = expand_cost(inputs, max_depth=1)
    assert found[behaviour_key(behaviour)] == path


def test_replay_costed_raises_on_an_illegal_step():
    inputs = FAMILIES["int_pair"][0]
    with pytest.raises(ValueError):
        replay_costed(inputs, ("add", "add"))
    with pytest.raises(ValueError):
        replay_costed(inputs, ("no-such-op",))


def test_expand_cost_validates_its_bounds():
    inputs = FAMILIES["int_pair"][0]
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=-1)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=1, max_states=0)
    with pytest.raises(ValueError):
        expand_cost(inputs, max_depth=1, max_score=-1)
    # the permitted edges
    found, complete = expand_cost(inputs, max_depth=0, max_states=1, max_score=0)
    assert complete
    assert found == {behaviour_key(inputs): CostedPath((), (0, 0, 0))}


def test_expand_cost_root_is_the_empty_path_and_depth_zero_stops_there():
    inputs = FAMILIES["int_pair"][0]
    found, complete = expand_cost(inputs, max_depth=0)
    assert complete
    assert list(found) == [behaviour_key(inputs)]
    assert found[behaviour_key(inputs)] == CostedPath((), (0, 0, 0))
    found, complete = expand_cost(inputs, max_depth=3, ops=())
    assert complete
    assert list(found) == [behaviour_key(inputs)]


def test_expand_cost_respects_max_states():
    inputs = FAMILIES["int_pair"][0]
    found, complete = expand_cost(inputs, max_depth=2, max_states=1)
    assert not complete
    assert list(found) == [behaviour_key(inputs)]
    found, complete = expand_cost(inputs, max_depth=2, max_states=5)
    assert not complete
    assert len(found) == 5
    full, complete = expand_cost(inputs, max_depth=2, max_states=None)
    assert complete
    assert len(full) > 5
    # what a truncated search has found is still the certified-cheapest
    assert all(full[key] == path for key, path in found.items())


def test_expand_cost_every_path_replays_to_its_behaviour():
    inputs = FAMILIES["with_map"][1]
    found, complete = expand_cost(inputs, max_depth=3)
    assert complete
    for key, path in found.items():
        behaviour, replayed = replay_costed(inputs, path.program)
        assert behaviour_key(behaviour) == key
        assert replayed == path


def test_expand_cost_reaches_what_bfs_reaches_but_never_shorter():
    """Same depth bound, same behaviours; the cheapest is at least the shortest."""
    for inputs in (FAMILIES["int_pair"][0], FAMILIES["with_map"][0]):
        by_depth, complete_depth = expand(inputs, max_depth=3)
        by_cost, complete_cost = expand_cost(inputs, max_depth=3)
        assert complete_depth and complete_cost
        assert set(by_depth) == set(by_cost)
        for key, (depth, _program, _witnesses) in by_depth.items():
            assert by_cost[key].length >= depth


def enumerate_cheapest(inputs, max_depth: int, ops=OPS) -> dict:
    """Every legal program up to `max_depth`, keeping the best per behaviour.

    The reference the search is checked against. Depth-first over programs,
    with no pruning at all, so it cannot share a mistake with the search.
    """
    best: dict = {}

    def visit(behaviour, path: CostedPath) -> None:
        key = behaviour_key(behaviour)
        rank = (path.score, path.length, path.program)
        if key not in best or rank < (
            best[key].score,
            best[key].length,
            best[key].program,
        ):
            best[key] = path
        if path.length == max_depth:
            return
        for op in ops:
            moved = step_all_costed(behaviour, path.example_fuels, op)
            if moved is None:
                continue
            visit(moved[0], CostedPath((*path.program, op.name), moved[1]))

    visit(inputs, CostedPath((), (0,) * len(inputs)))
    return best


@pytest.mark.parametrize(
    ("family", "index"),
    [("int_pair", 0), ("with_map", 0), ("with_fold", 1), ("with_apply", 1)],
)
def test_expand_cost_is_certified_cheapest_against_brute_force(family, index):
    """Exhaustive: the search must agree with enumerating every program."""
    inputs = FAMILIES[family][index]
    expected = enumerate_cheapest(inputs, max_depth=3)
    found, complete = expand_cost(inputs, max_depth=3)
    assert complete
    assert found == expected
    # on a family with a quotation to run, some cheapest program spent more
    # than one fuel per op per example, so Kt really did see the loop run
    if family != "int_pair":
        assert any(p.fuel > p.length * len(inputs) for p in found.values()), family
    else:
        assert all(p.kt >= p.length for p in found.values())


def test_expand_cost_max_score_is_a_filter_on_the_full_search():
    inputs = FAMILIES["with_map"][0]
    full, _ = expand_cost(inputs, max_depth=3)
    scores = sorted({p.score for p in full.values()})
    cutoff = scores[len(scores) // 2]
    found, complete = expand_cost(inputs, max_depth=3, max_score=cutoff)
    assert complete
    assert found == {k: p for k, p in full.items() if p.score <= cutoff}
    assert max(p.score for p in found.values()) == cutoff


def test_expand_cost_is_deterministic():
    inputs = FAMILIES["with_filter"][0]
    a, _ = expand_cost(inputs, max_depth=3)
    b, _ = expand_cost(inputs, max_depth=3)
    assert list(a.items()) == list(b.items())


# A synthetic basis over a single integer, built to make Kt's non-additivity
# bite. Every op is legal on exactly one value and costs what it says.
#
#     0 --big(100)--> 1 --suffix(1000)--> 2
#     0 --s1(1)--> 5 --s2(1)--> 1
#     0 --t1(1)--> 7 --t2(1)--> 1
#
# Reaching 1: (big) is 1 op and 100 fuel, score 200; (s1 s2) is 2 ops and 2
# fuel, score 8. So the cheapest arrival at 1 is the longer one. Reaching 2
# through it: (s1 s2 suffix) is 3 ops and 1002 fuel, score 8016, while
# (big suffix) is 2 ops and 1100 fuel, score 4400. The shorter, dearer arrival
# wins after the suffix, so a search that kept only the cheapest arrival per
# behaviour would label 2 wrongly. (t1 t2) ties (s1 s2) exactly, to pin the
# tie-break.


def _toy_op(name: str, source: int, target: int, cost: int) -> Op:
    def run(stack, budget, depth):
        if len(stack) != 1 or stack[-1] != Int(source):
            return f"{name} wants {source}"
        if cost > budget:
            return "out of fuel"
        return (Int(target),), cost

    return Op(
        name=name,
        group="toy",
        effect=f"( {source} -- {target} )",
        _guard=lambda config: None,
        _apply=lambda config: (config.stack, 1),
        _run=run,
    )


TOY_OPS = (
    _toy_op("big", 0, 1, 100),
    _toy_op("suffix", 1, 2, 1000),
    _toy_op("s1", 0, 5, 1),
    _toy_op("s2", 5, 1, 1),
    _toy_op("t1", 0, 7, 1),
    _toy_op("t2", 7, 1, 1),
)


def _toy_key(value: int) -> tuple:
    return behaviour_key(((Int(value),),))


def test_expand_cost_keeps_a_pareto_frontier_because_kt_is_not_additive():
    inputs = ((Int(0),),)
    found, complete = expand_cost(inputs, max_depth=3, ops=TOY_OPS)
    assert complete
    assert found == enumerate_cheapest(inputs, max_depth=3, ops=TOY_OPS)
    assert found[_toy_key(1)] == CostedPath(("s1", "s2"), (2,))
    assert found[_toy_key(2)] == CostedPath(("big", "suffix"), (1100,))
    assert found[_toy_key(1)].score == 8
    assert found[_toy_key(2)].score == 4400
    # the dominated-but-expensive route is still exactly what the frontier
    # held onto: a shorter arrival with more fuel
    cheapest_now = CostedPath(("s1", "s2", "suffix"), (1002,))
    assert cheapest_now.score == 8016 > found[_toy_key(2)].score
    assert cheapest_now.kt > found[_toy_key(2)].kt


def test_expand_cost_prunes_dominated_arrivals():
    """`max_depth` only reaches 2 by way of `big`, so only one arrival is kept.

    With the long route cut off, `suffix` is reachable solely from `(big)`.
    And when both arrivals are kept at depth 3 the result is unchanged for
    behaviour 2, which is what makes the frontier correct rather than merely
    bigger.
    """
    inputs = ((Int(0),),)
    found, _ = expand_cost(inputs, max_depth=2, ops=TOY_OPS)
    assert found[_toy_key(2)] == CostedPath(("big", "suffix"), (1100,))
    assert found[_toy_key(1)] == CostedPath(("s1", "s2"), (2,))


def test_expand_cost_breaks_ties_on_program():
    """`(s1 s2)` and `(t1 t2)` have identical scores; the smaller program wins."""
    inputs = ((Int(0),),)
    found, _ = expand_cost(inputs, max_depth=2, ops=TOY_OPS)
    assert found[_toy_key(1)].program == ("s1", "s2")
    reversed_ops = tuple(reversed(TOY_OPS))
    found, _ = expand_cost(inputs, max_depth=2, ops=reversed_ops)
    assert found[_toy_key(1)].program == ("s1", "s2"), "op order must not matter"


def test_expand_cost_max_score_cuts_the_dear_route():
    inputs = ((Int(0),),)
    found, complete = expand_cost(inputs, max_depth=3, ops=TOY_OPS, max_score=8)
    assert complete
    assert set(found) == {_toy_key(0), _toy_key(5), _toy_key(7), _toy_key(1)}
    assert found[_toy_key(1)] == CostedPath(("s1", "s2"), (2,))


def test_expand_cost_fuel_limit_bounds_the_search():
    """An op whose cost exceeds the remaining budget is refused, not charged."""
    inputs = ((Int(0),),)
    dear = (_toy_op("dear", 0, 1, DEFAULT_FUEL_LIMIT + 1),)
    found, complete = expand_cost(inputs, max_depth=1, ops=dear)
    assert complete
    assert list(found) == [_toy_key(0)]
    exact = (_toy_op("exact", 0, 1, DEFAULT_FUEL_LIMIT),)
    found, _ = expand_cost(inputs, max_depth=1, ops=exact)
    assert found[_toy_key(1)] == CostedPath(("exact",), (DEFAULT_FUEL_LIMIT,))
