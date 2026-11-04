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
from wandering_light.stack_machine import (
    CORE_OPS,
    DEFAULT_BASIS,
    MARK,
    MARK_OPS,
    Config,
    Int,
    Lst,
    MachineError,
    Op,
    Sym,
    format_stack,
    ops_for_basis,
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
    main,
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


def test_cost_recertification_scope_is_the_101_loop_witnesses(tasks):
    loop_ops = {"map", "fold", "filter"}
    selected = [task for task in tasks if not loop_ops.isdisjoint(task["program"])]

    assert len(selected) == 101
    assert Counter(task["family"] for task in selected) == {
        "with_filter": 40,
        "with_fold": 37,
        "with_map": 18,
        "bool_int": 3,
        "lm_order2": 2,
        "int_pair": 1,
    }


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


def test_cost_expansion_trades_program_bits_for_log_fuel():
    desired = " ".join(["1"] * 16)
    source = " ".join(["0"] * 16)
    inputs = (parse_stack(f"[ ({desired}) ({source}) (1 add) ]"),)
    target = behaviour_key((parse_stack(f"[ ({desired}) ({desired}) ]"),))

    breadth_first, _ = expand(inputs, max_depth=3)
    assert breadth_first[target][1] == ("map",)

    costed, complete = expand_cost(inputs, max_depth=3)
    assert complete
    assert costed[target].program == ("drop", "drop", "dup")
    assert costed[target].fuel == 3
    assert costed[target].kt == pytest.approx(3 + math.log2(3))


def test_cost_expansion_retains_nondominated_arrivals_for_later_suffixes():
    def transition(name: str, source: int, target: int, fuel: int) -> Op:
        def guard(config):
            return None if config.stack == (Int(source),) else "wrong state"

        def apply(_config):
            return (Int(target),), fuel

        return Op(name, "test", "", guard, apply)

    ops = (
        transition("direct", 0, 3, 16),
        transition("cheap-1", 0, 1, 1),
        transition("cheap-2", 1, 2, 1),
        transition("cheap-3", 2, 3, 1),
        transition("finish", 3, 4, 2),
    )
    found, complete = expand_cost(((Int(0),),), max_depth=4, ops=ops)

    assert complete
    merge = found[behaviour_key(((Int(3),),))]
    assert merge.program == ("cheap-1", "cheap-2", "cheap-3")
    target = found[behaviour_key(((Int(4),),))]
    assert target.program == ("direct", "finish")
    assert target.score == 18 << 2


def test_costed_steps_sum_work_and_preserve_per_example_fuel():
    fold = next(op for op in OPS if op.name == "fold")
    inputs = FAMILIES["with_fold"][0]
    outcome = step_all_costed(inputs, (0, 0, 0), fold)

    assert outcome is not None
    _moved, example_fuels = outcome
    path = CostedPath(("fold",), example_fuels)
    assert path.example_fuels == (4, 3, 2)
    assert path.fuel == 9
    assert path.score == 9 << 1


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


# ---------------------------------------------------------------------------
# Bases: the corpus can be certified against mark-28 as well
# ---------------------------------------------------------------------------

# `[ quote ] map` on the bitlist3 family wraps every element: four ops, one
# answer, and the answer depends on the input. Without brackets the executable
# basis has no way to put `(quote)` on the stack in four ops, so this is a
# behaviour the mark basis reaches and the default one does not. (Summing with
# `[ add ] fold` is not such a behaviour: a list of ints is itself a runnable
# quotation, so `swap dip add add` sums list_int in four executable ops.)
BRACKET_PROGRAM = ("[", "quote", "]", "map")
BRACKET_INPUTS = FAMILIES["bitlist3"][0]
BRACKET_OUTPUTS = ["[ ((1) (0) (1)) ]", "[ ((0) (1) (1)) ]", "[ ((1) (1) (0)) ]"]


def test_default_basis_is_the_executable_core():
    assert BASIS == DEFAULT_BASIS == "executable-26"
    assert OPS is CORE_OPS
    assert OPS is ops_for_basis(BASIS)
    assert Task.__dataclass_fields__["basis"].default == BASIS


def test_replay_takes_the_basis():
    inputs = BRACKET_INPUTS
    outputs = replay(inputs, BRACKET_PROGRAM, ops=MARK_OPS)
    assert [format_stack(Config(s)) for s in outputs] == BRACKET_OUTPUTS
    assert is_useful(inputs, outputs)
    with pytest.raises(KeyError):
        replay(inputs, BRACKET_PROGRAM)  # `[` is not in the default basis


def test_replay_costed_charges_one_per_quoted_op():
    inputs = BRACKET_INPUTS
    outputs, path = replay_costed(inputs, BRACKET_PROGRAM, ops=MARK_OPS)
    assert [format_stack(Config(s)) for s in outputs] == BRACKET_OUTPUTS
    # 3 to build (quote), then map: 1 plus one quote per element
    assert path.example_fuels == (3 + 1 + 3,) * 3
    assert path.length == 4
    assert path.fuel == 21


def test_an_unfinished_quotation_is_a_behaviour_with_the_mark_in_it():
    """The marker is state, so search sees half-built quotations as states."""
    inputs = FAMILIES["int_pair"][0]
    outputs = replay(inputs, ("[", "add"), ops=MARK_OPS)
    assert outputs[0] == (Int(3), Int(5), MARK, Sym("add"))
    assert format_stack(Config(outputs[0])) == "[ 3 5 MARK add ]"
    assert parse_stack("[ 3 5 MARK add ]", basis="mark-28") == outputs[0]
    assert not is_useful(inputs, outputs)


def test_multi_example_guard_quotes_per_example():
    """Quoting is decided per example, so a mark on one example alone can
    make an op legal there and illegal elsewhere -- the conjunction decides."""
    add = next(op for op in MARK_OPS if op.name == "add")
    assert step_all(((MARK,), (Int(1),)), add) is None
    assert step_all(((MARK,), (Int(1), Int(1))), add) == (
        (MARK, Sym("add")),
        (Int(2),),
    )
    for op in MARK_OPS:
        each = [op.guard(Config(stack)) is None for stack in ((MARK,), (Lst(()),))]
        assert (step_all(((MARK,), (Lst(()),)), op) is not None) == all(each), op.name


def test_expand_takes_the_basis():
    inputs = BRACKET_INPUTS
    target = behaviour_key(replay(inputs, BRACKET_PROGRAM, ops=MARK_OPS))

    marked, complete = expand(inputs, max_depth=4, ops=MARK_OPS)
    assert complete
    depth, program, witnesses = marked[target]
    assert depth == 4
    assert program == BRACKET_PROGRAM
    assert witnesses == 1

    plain, complete = expand(inputs, max_depth=4)
    assert complete
    assert target not in plain
    assert not any("[" in program for _d, program, _w in plain.values())
    # with no mark on any path the default-basis search is unchanged
    assert plain == expand(inputs, max_depth=4, ops=CORE_OPS)[0]


def test_expand_cost_takes_the_basis():
    inputs = BRACKET_INPUTS
    target = behaviour_key(replay(inputs, BRACKET_PROGRAM, ops=MARK_OPS))
    found, complete = expand_cost(inputs, max_depth=4, ops=MARK_OPS)
    assert complete
    assert found[target].program == BRACKET_PROGRAM
    assert found[target].fuel == 21


def test_generate_records_the_basis_on_every_task():
    with pytest.raises(MachineError, match="unknown basis"):
        generate(max_depth=1, per_depth=1, seed=1, basis="mark-29")

    plain, _ = generate(max_depth=2, per_depth=3, seed=1)
    assert plain and {t.basis for t in plain} == {"executable-26"}
    marked, _ = generate(max_depth=2, per_depth=3, seed=1, basis="mark-28")
    assert marked and {t.basis for t in marked} == {"mark-28"}
    # nothing two ops long can close a bracket and still leave one value
    assert not any("[" in t.program for t in marked)
    assert [t.program for t in marked] == [t.program for t in plain]


def test_generate_under_the_mark_basis_keeps_bracket_programs(monkeypatch):
    monkeypatch.setattr(
        stack_machine_corpus, "FAMILIES", {"bitlist3": FAMILIES["bitlist3"]}
    )
    tasks, stats = generate(max_depth=4, per_depth=400, seed=1, basis="mark-28")
    assert stats["bitlist3"]["by_depth"][4] > 0
    assert {t.basis for t in tasks} == {"mark-28"}
    bracketed = [t for t in tasks if "[" in t.program]
    assert any(t.program == BRACKET_PROGRAM for t in bracketed)
    for task in bracketed:
        assert task.program.count("[") == task.program.count("]")
        inputs = tuple(parse_stack(s, basis=task.basis) for s in task.inputs)
        produced = replay(inputs, task.program, ops=ops_for_basis(task.basis))
        expected = tuple(parse_stack(s, basis=task.basis) for s in task.outputs)
        assert behaviour_key(produced) == behaviour_key(expected), task
        assert task.certified
    # the encoding carries brackets as ordinary tokens
    for task in bracketed:
        assert encode_task(task)["completion"].split() == list(task.program)


def test_main_takes_the_basis(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        stack_machine_corpus, "FAMILIES", {"int_pair": FAMILIES["int_pair"]}
    )
    out = tmp_path / "mark.jsonl.gz"
    assert (
        main(
            [
                "--out",
                str(out),
                "--max-depth",
                "2",
                "--per-depth",
                "2",
                "--basis",
                "mark-28",
            ]
        )
        == 0
    )
    with gzip.open(out, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows and {row["basis"] for row in rows} == {"mark-28"}
    assert "wrote" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--out", str(out), "--basis", "mark-29"])
