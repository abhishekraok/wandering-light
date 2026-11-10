"""Signal House: the workshop observes the canonical machine and grades honestly.

What could drift silently: that a step shown to a learner is the machine's own
step, that what a combinator did inside is read from the machine rather than
restated, that a lesson's expected outputs are independent of its solution,
and that the page is never handed an answer it did not ask for.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from wandering_light import stack_machine
from wandering_light.stack_machine import (
    MARK,
    MARK_OPS,
    Bool,
    Int,
    Lst,
    Sym,
    format_value,
    run_body,
    step_raw,
    value_key,
)
from wandering_light.stack_workshop import curriculum, engine, model, server
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS, REFERENCES
from wandering_light.stack_workshop.engine import (
    FUEL,
    MACROS,
    OPS,
    compile_program,
    execute,
    inspect_op,
    snapshot,
    stack_from,
)

LESSON_FIELDS = {
    "id",
    "chapter",
    "title",
    "concept",
    "story",
    "goal",
    "starter",
    "solution",
    "examples",
    "heldout",
    "hints",
    "discovery",
    "review",
    "prediction",
    "macros",
}


def ops_of(text: str) -> list[str]:
    return [token["op"] for token in compile_program(text)]


def texts(display: list[dict]) -> list[str]:
    return [value["text"] for value in display]


def one(given, wanted) -> list[dict]:
    return [{"input": given, "output": wanted}]


# ---------------------------------------------------------------------------
# engine: ops, macros, compiling
# ---------------------------------------------------------------------------


def test_ops_are_the_mark_basis_itself():
    assert list(OPS) == [op.name for op in MARK_OPS]
    assert all(OPS[op.name] is op for op in MARK_OPS)
    assert len(OPS) == 28
    assert FUEL == 100_000


def test_macros_come_from_the_file_with_their_groups():
    assert MACROS["inc"] == {"body": "1 add", "group": "scalars"}
    assert MACROS["2"] == {"body": "1 1 add", "group": "constants"}
    assert MACROS["rot"]["group"] == "stack shuffles"
    assert MACROS["sum"]["group"] == "lists"
    # A heading may carry a note after a colon; the group is the name before it.
    assert MACROS["range"]["group"] == "ranges"
    assert not set(MACROS) & set(OPS)

    definitions = [
        line
        for line in engine.MACRO_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(MACROS) == len(definitions)
    assert list(MACROS) == [line.split(":", 1)[0] for line in definitions]


def test_every_macro_expands_to_primitives():
    for name in MACROS:
        tokens = compile_program(name)
        assert tokens, name
        assert all(token["op"] in OPS for token in tokens), name
        assert {token["macro"] for token in tokens} == {name}


def test_compile_keeps_the_span_of_the_word_as_written():
    text = "dup  [inc] # ignored add\n  swap"
    tokens = compile_program(text)
    assert tokens == [
        {"op": "dup", "start": 0, "end": 3, "macro": None},
        {"op": "[", "start": 5, "end": 6, "macro": None},
        {"op": "1", "start": 6, "end": 9, "macro": "inc"},
        {"op": "add", "start": 6, "end": 9, "macro": "inc"},
        {"op": "]", "start": 9, "end": 10, "macro": None},
        {"op": "swap", "start": 27, "end": 31, "macro": None},
    ]
    for token in tokens:
        written = text[token["start"] : token["end"]]
        assert written == (token["macro"] or token["op"])


def test_nested_macros_report_the_outermost_name():
    assert MACROS["even?"]["body"] == "2 mod 0 eq?"
    tokens = compile_program("0 even?")
    assert [token["macro"] for token in tokens] == [None] + ["even?"] * (
        len(tokens) - 1
    )
    assert [token["op"] for token in tokens[1:]] == ops_of(
        "1 1 add over over div mul sub 0 eq?"
    )
    assert {(token["start"], token["end"]) for token in tokens[1:]} == {(2, 7)}


def test_compile_handles_empty_and_comment_only_programs():
    assert compile_program("") == []
    assert compile_program("  # nothing here\n\t# nor here") == []
    assert ops_of("dup#add") == ["dup"]


def test_unknown_word_is_named():
    with pytest.raises(ValueError, match=r"^Unknown word 'frobnicate'\. "):
        compile_program("dup frobnicate")
    with pytest.raises(ValueError, match=r"^Unknown word '42'\. .*Only 0 and 1"):
        compile_program("42")
    with pytest.raises(ValueError, match="Did you mean 'swap'"):
        compile_program("swp")


def test_a_macro_reaching_itself_is_refused(monkeypatch):
    monkeypatch.setitem(MACROS, "loop", {"body": "dup loop", "group": "test"})
    with pytest.raises(ValueError, match=r"^Recursive macro: loop$"):
        compile_program("loop")

    monkeypatch.setitem(MACROS, "ping", {"body": "pong", "group": "test"})
    monkeypatch.setitem(MACROS, "pong", {"body": "1 ping", "group": "test"})
    with pytest.raises(ValueError, match=r"^Recursive macro: ping$"):
        compile_program("ping")
    with pytest.raises(ValueError, match=r"^Recursive macro: pong$"):
        compile_program("pong")
    # Using a macro twice is not recursion.
    monkeypatch.setitem(MACROS, "twice", {"body": "inc inc", "group": "test"})
    assert ops_of("twice") == ["1", "add", "1", "add"]


def test_a_program_that_could_never_finish_is_refused_at_compile(monkeypatch):
    assert engine.MAX_PROGRAM_OPS == FUEL
    monkeypatch.setattr(engine, "MAX_PROGRAM_OPS", 8)
    assert len(compile_program("dup " * 8)) == 8
    with pytest.raises(ValueError, match="more than 8 ops"):
        compile_program("dup " * 9)
    with pytest.raises(ValueError, match="more than 8 ops"):
        compile_program("zeros8")


# ---------------------------------------------------------------------------
# engine: stacks in
# ---------------------------------------------------------------------------


def test_stack_from_reads_arrays_and_notation_alike():
    expected = (Int(3), Lst((Int(1), Lst(()), Bool(False))), Bool(True), Int(-7))
    assert stack_from([3, [1, [], False], True, -7]) == expected
    assert stack_from("3 (1 nil false) true -7") == expected
    assert stack_from("[ 3 (1 () false) true -7 ]") == expected
    assert stack_from([]) == ()
    assert stack_from("") == ()
    # True is not 1: the machine keeps them apart and so must the reader.
    assert stack_from([True, 1]) == (Bool(True), Int(1))
    assert value_key(stack_from([True])[0]) != value_key(stack_from([1])[0])


@pytest.mark.parametrize(
    "bad",
    [
        [1.0],
        [1.5],
        [float("nan")],
        ["dup"],
        [[1, "2"]],
        [None],
        [{"a": 1}],
        "dup",
        "1 (2 add)",
        "MARK",
        "(1 MARK)",
        "(1 2",
        "1 )",
        "1.5",
        None,
        7,
        True,
        {"input": [1]},
    ],
)
def test_stack_from_refuses_what_is_not_a_stack(bad):
    with pytest.raises(ValueError):
        stack_from(bad)
    with pytest.raises(ValueError):
        stack_from(bad, expected=True)


def nested(depth: int) -> list:
    value: list = []
    for _ in range(depth - 1):
        value = [value]
    return value


def test_stack_from_limits_sit_exactly_where_stated():
    limit = 10**12
    assert stack_from([limit, -limit]) == (Int(limit), Int(-limit))
    assert stack_from(f"{limit}") == (Int(limit),)
    for too_big in (limit + 1, -limit - 1):
        with pytest.raises(ValueError):
            stack_from([too_big])
        with pytest.raises(ValueError):
            stack_from(str(too_big))
        with pytest.raises(ValueError):
            stack_from([[too_big]])

    # An expected output may be a long number: 512 digits, not 513.
    longest = 10**512 - 1
    assert stack_from([longest, -longest], expected=True)[0] == Int(longest)
    assert stack_from(str(longest), expected=True) == (Int(longest),)
    for too_long in (10**512, -(10**512)):
        with pytest.raises(ValueError):
            stack_from([too_long], expected=True)
        with pytest.raises(ValueError):
            stack_from(str(too_long), expected=True)

    assert len(stack_from([0] * 16)) == 16
    assert len(stack_from("0 " * 16)) == 16
    with pytest.raises(ValueError):
        stack_from([0] * 17)
    with pytest.raises(ValueError):
        stack_from("0 " * 17)

    assert len(stack_from([[1] * 64])[0].items) == 64
    assert len(stack_from("(" + "1 " * 64 + ")")[0].items) == 64
    with pytest.raises(ValueError):
        stack_from([[1] * 65])
    with pytest.raises(ValueError):
        stack_from("(" + "1 " * 65 + ")")
    with pytest.raises(ValueError):
        stack_from([[[1] * 65]], expected=True)

    assert stack_from([nested(12)]) == stack_from("(" * 12 + ")" * 12)
    with pytest.raises(ValueError):
        stack_from([nested(13)])
    with pytest.raises(ValueError):
        stack_from("(" * 13 + ")" * 13)
    # Far past the limit must still be a ValueError, not a crashed parser.
    with pytest.raises(ValueError):
        stack_from("(" * 50_000)
    with pytest.raises(ValueError):
        stack_from(nested(5_000))
    with pytest.raises(ValueError):
        stack_from("9" * 5_000)


# ---------------------------------------------------------------------------
# engine: stacks out
# ---------------------------------------------------------------------------


def test_snapshot_names_every_kind_of_value():
    stack = (
        Int(-4),
        Bool(True),
        Sym("add"),
        MARK,
        Lst((Int(1), Lst(()), Sym("dup"))),
    )
    shown = snapshot(stack)
    assert [value["type"] for value in shown] == [
        "int",
        "bool",
        "symbol",
        "mark",
        "list",
    ]
    assert texts(shown) == [format_value(value) for value in stack]
    assert shown[0] == {"type": "int", "text": "-4"}
    assert shown[4]["items"] == [
        {"type": "int", "text": "1"},
        {"type": "list", "text": "nil", "items": []},
        {"type": "symbol", "text": "dup"},
    ]
    assert all("items" not in value for value in shown[:4])
    assert snapshot(()) == []
    json.dumps(shown)


def test_snapshot_elides_the_middle_of_a_deep_stack():
    exact = tuple(Int(n) for n in range(48))
    assert texts(snapshot(exact)) == [str(n) for n in range(48)]

    deep = tuple(Int(n) for n in range(100))
    shown = snapshot(deep)
    assert len(shown) == 49
    assert shown[0] == {"type": "int", "text": "0"}
    assert shown[1]["type"] == "elision"
    assert texts(shown[2:]) == [str(n) for n in range(53, 100)]

    just_over = snapshot(tuple(Int(n) for n in range(49)))
    assert [value["type"] for value in just_over[:3]] == ["int", "elision", "int"]
    assert texts(just_over[2:]) == [str(n) for n in range(2, 49)]


def test_snapshot_survives_what_the_machine_can_build():
    # Everything a learner can type is shown whole...
    typed = stack_from([[list(range(64))], nested(12), 10**12])
    assert texts(snapshot(typed)) == [format_value(value) for value in typed]
    assert len(snapshot(typed)[0]["items"][0]["items"]) == 64

    # ...and what only the machine can build is bounded, not fatal.
    long_list = Lst(tuple(Int(n) for n in range(500)))
    shown = snapshot((long_list,))[0]
    assert len(shown["items"]) == 64
    assert shown["items"][-1]["type"] == "elision"
    assert texts(shown["items"][:63]) == [str(n) for n in range(63)]

    tower = Lst(())
    for _ in range(5_000):
        tower = Lst((tower,))
    assert snapshot((tower,))[0]["type"] == "list"

    huge = Int(7 ** (10**5))
    shown = snapshot((huge, Int(-(7 ** (10**5)))))
    assert shown[0]["type"] == "int"
    assert len(shown[0]["text"]) < 100
    assert shown[1]["text"].startswith("-")
    json.dumps(shown)


# ---------------------------------------------------------------------------
# engine: execute
# ---------------------------------------------------------------------------


def test_execute_reports_every_step_of_every_example():
    examples = [
        {"input": [[1, 2, 3]], "output": [6]},
        {"input": "nil", "output": "0"},
        {"input": [[5]], "output": [4]},
    ]
    result = execute("0 [ add ] fold", examples)
    assert set(result) == {
        "tokens",
        "examples",
        "matches",
        "count",
        "primitive_count",
        "fuel",
        "max_depth",
    }
    assert result["tokens"] == compile_program("0 [ add ] fold")
    assert result["count"] == 3
    assert result["matches"] == 2
    assert result["primitive_count"] == 5
    assert [row["matched"] for row in result["examples"]] == [True, True, False]

    row = result["examples"][0]
    assert set(row) == {"frames", "matched", "error", "fuel", "max_depth", "output"}
    frames = row["frames"]
    assert len(frames) == 6
    assert all(
        set(frame) == {"stack", "fuel", "op", "error", "quoting"} for frame in frames
    )
    assert frames[0] == {
        "stack": snapshot(stack_from([[1, 2, 3]])),
        "fuel": 0,
        "op": None,
        "error": None,
        "quoting": False,
    }
    assert [frame["op"] for frame in frames] == [None, "0", "[", "add", "]", "fold"]
    assert [texts(frame["stack"]) for frame in frames] == [
        ["(1 2 3)"],
        ["(1 2 3)", "0"],
        ["(1 2 3)", "0", "MARK"],
        ["(1 2 3)", "0", "MARK", "add"],
        ["(1 2 3)", "0", "(add)"],
        ["6"],
    ]
    assert [frame["quoting"] for frame in frames] == [False, False, True, True] + [
        False
    ] * 2
    # fold costs one for itself and one per item it adds.
    assert [frame["fuel"] for frame in frames] == [0, 1, 2, 3, 4, 8]
    assert row["fuel"] == 8
    assert row["max_depth"] == 4
    assert row["error"] is None
    assert row["output"] == snapshot((Int(6),))

    # Across examples fuel is the total and depth is the deepest.
    assert [row["fuel"] for row in result["examples"]] == [8, 5, 6]
    assert result["fuel"] == 19
    assert [row["max_depth"] for row in result["examples"]] == [4, 4, 4]
    assert result["max_depth"] == 4
    deeper = execute("dup dup drop", one([1], [1, 1]) + one([1, 2, 3], [1, 2, 3, 3]))
    assert [row["max_depth"] for row in deeper["examples"]] == [3, 5]
    assert deeper["max_depth"] == 5
    json.dumps(result)


def test_execute_is_the_machine_stepping(monkeypatch):
    """Each token is one top-level `step_raw` on the remaining budget."""
    calls = []

    def recording(op, stack, budget, depth):
        if depth == 0:
            calls.append((op.name, stack, budget))
        return step_raw(op, stack, budget, depth)

    monkeypatch.setattr(engine, "step_raw", recording)
    program = "dup [ dup mul ] i add"
    result = execute(program, one([3], [12]))
    assert result["matches"] == 1

    stack, spent, expected = (Int(3),), 0, []
    for name in ops_of(program):
        expected.append((name, stack, FUEL - spent))
        stack, cost = step_raw(OPS[name], stack, FUEL - spent, 0)
        spent += cost
    assert calls == expected
    assert result["examples"][0]["fuel"] == spent


def test_a_refusal_halts_that_example_only():
    examples = [
        {"input": [2, 0], "output": [3]},
        {"input": [6, 3], "output": [3]},
    ]
    result = execute("div 1 add", examples)
    refused, fine = result["examples"]

    assert refused["error"] == "div: division by zero"
    assert refused["matched"] is False
    assert [frame["op"] for frame in refused["frames"]] == [None, "div"]
    assert refused["frames"][1]["stack"] == refused["frames"][0]["stack"]
    assert refused["frames"][1]["error"] == "div: division by zero"
    assert refused["frames"][1]["fuel"] == 0
    assert refused["fuel"] == 0

    assert fine["error"] is None
    assert fine["matched"] is True
    assert len(fine["frames"]) == 4
    assert result["matches"] == 1

    # A refusal inside a quotation names the op that refused.
    inside = execute("[ add ] i", one([1], [1]))["examples"][0]
    assert inside["error"] == "i: add: needs 2 value(s), stack has 1"
    # A stack that happens to equal the output is still not a match.
    assert execute("drop drop", one([1], []))["examples"][0]["matched"] is False


def test_running_out_of_fuel_is_a_refusal(monkeypatch):
    monkeypatch.setattr(engine, "FUEL", 3)
    row = execute("dup dup dup dup", one([1], [1, 1, 1, 1, 1]))["examples"][0]
    assert row["error"] == "dup: out of fuel"
    assert row["fuel"] == 3
    assert len(row["frames"]) == 5
    assert texts(row["frames"][-1]["stack"]) == ["1"] * 4

    # The budget a combinator gets is what the earlier tokens left.
    monkeypatch.setattr(engine, "FUEL", 8)
    examples = one([[1, 2, 3]], [[2, 4, 6]])
    row = execute("[ dup add ] map", examples)["examples"][0]
    assert row["error"] == "map: out of fuel"
    assert row["fuel"] == 4
    # Four tokens, then `map` costs one and two per item, and the machine
    # wants fuel left over when a combinator finishes.
    monkeypatch.setattr(engine, "FUEL", 11)
    assert execute("[ dup add ] map", examples)["examples"][0]["error"]
    monkeypatch.setattr(engine, "FUEL", 12)
    row = execute("[ dup add ] map", examples)["examples"][0]
    assert (row["matched"], row["fuel"]) == (True, 11)


def test_the_default_fuel_bounds_a_program_that_never_stops():
    # 133 fuel a round: four tokens to write the quotation, then `map` costs
    # one and two per item. 800 rounds would need more than there is.
    long_night = "[ dup sub ] map " * 800
    row = execute(long_night, one([[1] * 64], [[0] * 64]), capture=False)["examples"][0]
    assert row["error"] == "map: out of fuel"
    assert FUEL - 133 < row["fuel"] <= FUEL
    assert execute("[ dup sub ] map " * 700, one([[1] * 64], [[0] * 64]))["matches"]


def test_matching_is_by_value_key():
    assert execute("", one([1], [True]))["matches"] == 0
    assert execute("", one([1], [1]))["matches"] == 1
    assert execute("0 eq?", one([0], [True]))["matches"] == 1
    assert execute("0 eq?", one([0], [1]))["matches"] == 0
    assert execute("", one([[]], "nil"))["matches"] == 1
    # The whole stack is the output, not its top.
    assert execute("dup", one([4], [4]))["matches"] == 0


def test_capture_false_keeps_the_first_and_final_frames():
    examples = one([[1, 2, 3]], [6]) + one([[1]], [0]) + one([7], [7])
    full = execute("0 [ add ] fold", examples)
    brief = execute("0 [ add ] fold", examples, capture=False)
    for whole, short in zip(full["examples"], brief["examples"], strict=True):
        assert short["frames"] == [whole["frames"][0], whole["frames"][-1]]
        assert {k: v for k, v in short.items() if k != "frames"} == {
            k: v for k, v in whole.items() if k != "frames"
        }
    assert brief["examples"][2]["error"] == "fold: third value is not a list"
    assert {k: v for k, v in brief.items() if k != "examples"} == {
        k: v for k, v in full.items() if k != "examples"
    }
    # No tokens: the input is both the first frame and the final one.
    assert len(execute("", one([1], [1]), capture=False)["examples"][0]["frames"]) == 1


@pytest.mark.parametrize(
    "examples",
    [
        [],
        [{"input": [1], "output": [1]}] * 17,
        None,
        "1 -> 1",
        [{"input": [1]}],
        [{"output": [1]}],
        [[1], [1]],
        [{"input": [1.5], "output": [1]}],
        [{"input": [1], "output": ["dup"]}],
        [{"input": [10**12 + 1], "output": [1]}],
    ],
)
def test_execute_refuses_bad_examples(examples):
    with pytest.raises(ValueError):
        execute("dup", examples)


def test_execute_takes_one_to_sixteen_examples_and_long_outputs():
    assert execute("", one([1], [1]) * 16)["count"] == 16
    assert execute("", one([1], [1]))["count"] == 1
    # An output may be longer than any input: 10^12 squared three times.
    program = "dup mul dup mul dup mul"
    assert execute(program, one([10**12], [10**96]))["matches"] == 1


def test_execute_survives_values_the_machine_itself_chokes_on():
    row = execute("quote " * 3000 + "[ ] i", one([1], [1]))["examples"][0]
    assert row["error"] == "i: a value is nested too deep for the machine"
    assert row["matched"] is False
    assert len(row["frames"]) == 3004


# ---------------------------------------------------------------------------
# engine: inspect
# ---------------------------------------------------------------------------


def test_inspect_records_the_nested_steps_in_return_order():
    examples = one([[1, 2, 3]], [[1, 4, 9]])
    inside = inspect_op("[ dup mul ] map", examples, 4)
    assert set(inside) == {"frames", "truncated", "error"}
    assert inside["truncated"] is False
    assert inside["error"] is None
    assert all(
        set(frame) == {"op", "depth", "before", "after", "error"}
        for frame in inside["frames"]
    )
    assert [
        (frame["op"], texts(frame["before"]), texts(frame["after"]))
        for frame in inside["frames"]
    ] == [
        ("dup", ["1"], ["1", "1"]),
        ("mul", ["1", "1"], ["1"]),
        ("dup", ["2"], ["2", "2"]),
        ("mul", ["2", "2"], ["4"]),
        ("dup", ["3"], ["3", "3"]),
        ("mul", ["3", "3"], ["9"]),
    ]
    assert all(frame["error"] is None for frame in inside["frames"])

    # An inner call returns before the call that made it, and sits deeper.
    nest = inspect_op("[ [ dup ] i mul ] i", one([3], [9]), 7)["frames"]
    assert [frame["op"] for frame in nest] == ["dup", "i", "mul"]
    dup, inner_i, mul = (frame["depth"] for frame in nest)
    assert dup > inner_i == mul > 0

    # `ifte` shows the test and only the branch it took.
    branch = inspect_op(
        "[ 0 lt? ] [ 0 swap sub ] [ ] ifte", one([-5], [5]), 11, example=0
    )
    assert [frame["op"] for frame in branch["frames"]] == ["lt?", "swap", "sub"]


def test_inspect_agrees_with_the_machine_about_what_ran():
    """The record is the machine's calls, checked against an independent count."""
    program = "0 [ [ dup 0 lt? ] [ drop ] [ add ] ifte ] fold"
    examples = one([[3, -1, 4]], [7])
    step = len(compile_program(program)) - 1

    seen = []
    original = stack_machine.step_raw

    def counting(op, stack, budget, depth):
        outcome = original(op, stack, budget, depth)
        if depth > 0:
            seen.append((op.name, depth))
        return outcome

    run_body.cache_clear()
    stack_machine.step_raw = counting
    try:
        stack, spent = (Lst((Int(3), Int(-1), Int(4))),), 0
        for name in ops_of(program):
            stack, cost = original(OPS[name], stack, FUEL - spent, 0)
            spent += cost
    finally:
        stack_machine.step_raw = original
    assert stack == (Int(7),)

    frames = inspect_op(program, examples, step)["frames"]
    assert [(frame["op"], frame["depth"]) for frame in frames] == seen
    # Per item: the `ifte`, the two ops of its test (the 0 is pushed, not
    # stepped), and the one op of the branch it took.
    assert len(frames) == 3 * (1 + 2 + 1)


def test_a_warm_cache_never_hides_nested_execution():
    program = "[ dup mul ] map"
    examples = one([[1, 2, 3]], [[1, 4, 9]])
    execute(program, examples)
    execute(program, examples)
    assert run_body.cache_info().currsize > 0

    first = inspect_op(program, examples, 4)
    again = inspect_op(program, examples, 4)
    assert len(first["frames"]) == 6
    assert again == first

    # Without the clear, the second look would have been empty: a cached
    # quotation run returns its answer without stepping anything.
    inspect_op(program, examples, 4)
    counted = []

    def watch(frame, event, arg):
        if event == "call" and frame.f_code is step_raw.__code__:
            counted.append(frame.f_locals["depth"])

    stack = (*stack_from([[1, 2, 3]]), Lst((Sym("dup"), Sym("mul"))))
    sys.setprofile(watch)
    try:
        step_raw(OPS["map"], stack, FUEL - 4, 0)
    finally:
        sys.setprofile(None)
    assert counted == [0]


def test_inspect_stops_recording_at_the_limit():
    wide = one([list(range(64))], [[8 * n for n in range(64)]])
    program = "[ dup add dup add dup add ] map"
    assert execute(program, wide)["matches"] == 1
    inside = inspect_op(program, wide, 8)
    assert len(inside["frames"]) == 350
    assert inside["truncated"] is True
    assert inside["error"] is None
    assert [frame["op"] for frame in inside["frames"][:6]] == ["dup", "add"] * 3

    # 58 items of 6 ops is 348 nested calls: under the limit, nothing dropped.
    under = inspect_op(program, one([list(range(58))], [[]]), 8)
    assert len(under["frames"]) == 348
    assert under["truncated"] is False


def test_inspect_reports_a_refusal_and_where_it_happened():
    inside = inspect_op("[ 1 div ] map", one([[4]], [[4]]), 4)
    assert inside["error"] is None

    inside = inspect_op("[ dup uncons ] map", one([[4, 5]], [[]]), 4)
    assert inside["error"] == "map: uncons: top is not a list"
    assert [frame["op"] for frame in inside["frames"]] == ["dup", "uncons"]
    refused = inside["frames"][1]
    assert refused["error"] == "top is not a list"
    assert refused["after"] == refused["before"]
    assert texts(refused["before"]) == ["4", "4"]
    assert inside["frames"][0]["error"] is None

    # An op that runs no quotation makes no nested calls.
    assert inspect_op("dup add", one([2], [4]), 1) == {
        "frames": [],
        "truncated": False,
        "error": None,
    }
    # Under an open bracket a combinator is only written down.
    assert inspect_op("[ [ dup ] map ]", one([[1]], [[1]]), 4)["frames"] == []


def test_inspect_looks_at_the_example_asked_for():
    examples = one([[1]], [[2]]) + one([[1, 2, 3, 4]], [[2, 4, 6, 8]])
    program = "[ dup add ] map"
    assert len(inspect_op(program, examples, 4)["frames"]) == 2
    assert len(inspect_op(program, examples, 4, 1)["frames"]) == 8
    assert len(inspect_op(program, examples, 4, example=1)["frames"]) == 8


@pytest.mark.parametrize(
    ("step", "example"),
    [(5, 0), (-1, 0), (4, 1), (4, -1), (None, 0), ("4", 0), (True, 0), (4.0, 0)],
)
def test_inspect_refuses_a_step_or_example_that_does_not_exist(step, example):
    with pytest.raises(ValueError):
        inspect_op("[ dup add ] map", one([[1]], [[2]]), step, example)


def test_inspect_refuses_a_step_the_machine_never_reached():
    with pytest.raises(ValueError, match="never reached"):
        inspect_op("add [ dup ] i", one([1], [1]), 4)
    # The refusing step itself can be looked at.
    assert (
        inspect_op("add [ dup ] i", one([1], [1]), 0)["error"]
        == "add: needs 2 value(s), stack has 1"
    )


def test_inspect_leaves_the_profile_hook_as_it_found_it():
    def mine(frame, event, arg):
        return None

    sys.setprofile(mine)
    try:
        inspect_op("[ dup mul ] map", one([[1, 2]], [[1, 4]]), 4)
        assert sys.getprofile() is mine
        with pytest.raises(ValueError):
            inspect_op("[ dup mul ] map", one([[1, 2]], [[1, 4]]), 99)
        assert sys.getprofile() is mine
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize(
    "call",
    [
        lambda: execute("dup add", one([1], [2])),
        lambda: inspect_op("[ dup ] i", one([1], [1, 1]), 3),
        lambda: model.Adviser().rank("dup", one([1], [1, 1])),
    ],
)
def test_machine_operations_share_one_lock(call):
    done = threading.Event()

    def work():
        call()
        done.set()

    thread = threading.Thread(target=work, daemon=True)
    with engine.LOCK:
        thread.start()
        assert not done.wait(0.3)
    assert done.wait(10)
    thread.join()


# ---------------------------------------------------------------------------
# curriculum
# ---------------------------------------------------------------------------


def test_there_are_fourteen_lessons_with_every_field():
    assert len(LESSONS) == 14
    assert [lesson["id"] for lesson in LESSONS][:2] == ["first-light", "echo"]
    assert len(BY_ID) == 14
    assert all(BY_ID[lesson["id"]] is lesson for lesson in LESSONS)
    for lesson in LESSONS:
        assert set(lesson) == LESSON_FIELDS, lesson["id"]
        assert len(lesson["examples"]) == 3
        assert len(lesson["heldout"]) == 3
        assert lesson["hints"]
        for field in ("chapter", "title", "concept", "story", "goal", "discovery"):
            assert isinstance(lesson[field], str) and lesson[field].strip()
        assert isinstance(lesson["starter"], str)
    json.dumps(LESSONS)
    # Chapters are contiguous: a lesson never returns to an earlier chapter.
    chapters = [lesson["chapter"] for lesson in LESSONS]
    assert len(set(chapters)) == sum(
        1 for a, b in zip([None, *chapters], chapters, strict=False) if a != b
    )


def test_the_first_two_lessons_are_the_ones_named():
    first, echo = BY_ID["first-light"], BY_ID["echo"]
    assert first["solution"] == "1 add"
    assert echo["solution"] == "dup add"
    for example in first["examples"] + first["heldout"]:
        (number,) = example["input"]
        assert example["output"] == [number + 1]
    for example in echo["examples"] + echo["heldout"]:
        (number,) = example["input"]
        assert example["output"] == [2 * number]


def test_the_lessons_cover_what_they_say_they_cover():
    solutions = {lesson["id"]: ops_of(lesson["solution"]) for lesson in LESSONS}
    order = [lesson["id"] for lesson in LESSONS]

    def first_use(op: str) -> int:
        return next(i for i, name in enumerate(order) if op in solutions[name])

    introduced = [first_use(op) for op in ("swap", "i", "map", "filter", "fold")]
    introduced += [first_use("dip"), first_use("ifte")]
    assert introduced == sorted(introduced)
    assert introduced[0] > 1
    assert "over" in {op for ops in solutions.values() for op in ops}
    assert "[" in solutions["sealed-orders"]

    # Macros come last, after the lesson about evidence.
    with_macros = [lesson for lesson in LESSONS if lesson["macros"]]
    assert with_macros == [LESSONS[-1]]
    assert set(LESSONS[-1]["macros"]) <= set(MACROS)
    written = {token["macro"] for token in compile_program(LESSONS[-1]["solution"])}
    assert written - {None} == set(LESSONS[-1]["macros"])
    for lesson in LESSONS[:-1]:
        for text in (lesson["solution"], lesson["starter"]):
            assert all(token["macro"] is None for token in compile_program(text))


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson["id"])
def test_every_solution_passes_all_six_examples(lesson):
    result = execute(lesson["solution"], lesson["examples"] + lesson["heldout"])
    assert result["count"] == 6
    assert result["matches"] == 6, [row["error"] for row in result["examples"]]


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson["id"])
def test_expected_outputs_come_from_the_reference_not_the_solution(lesson):
    reference = REFERENCES[lesson["id"]]
    examples = lesson["examples"] + lesson["heldout"]
    for example in examples:
        assert example["output"] == reference(*example["input"])
    # Six different inputs, so the held-out tests are new evidence.
    assert len({json.dumps(example["input"]) for example in examples}) == 6
    # The starter is never already the answer.
    assert execute(lesson["starter"], examples)["matches"] < 6
    # And the answer is not "do nothing".
    assert execute("", examples)["matches"] < 6


def test_references_are_python_and_never_touch_the_engine(monkeypatch):
    """Recompute every output with the machine made unusable."""

    def unusable(*args, **kwargs):
        raise AssertionError("a reference ran the machine")

    monkeypatch.setattr(engine, "step_raw", unusable)
    monkeypatch.setattr(engine, "compile_program", unusable)
    for lesson in LESSONS:
        for example in lesson["examples"] + lesson["heldout"]:
            assert REFERENCES[lesson["id"]](*example["input"]) == example["output"]


def test_a_fit_is_not_an_explanation():
    lesson = BY_ID["three-nights"]
    starter = lesson["starter"]
    assert starter and starter != lesson["solution"]
    assert execute(starter, lesson["examples"])["matches"] == 3
    assert execute(starter, lesson["heldout"])["matches"] < 3


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson["id"])
def test_every_prediction_is_what_the_machine_does(lesson):
    prediction = lesson["prediction"]
    assert prediction["answer"] in prediction["options"]
    assert len(set(prediction["options"])) == len(prediction["options"]) >= 3
    result = execute(
        prediction["program"], one(prediction["input"], prediction["answer"])
    )
    assert result["matches"] == 1
    # Exactly one option is right.
    right = [
        option
        for option in prediction["options"]
        if execute(prediction["program"], one(prediction["input"], option))["matches"]
    ]
    assert right == [prediction["answer"]]
    assert set(lesson["review"]) == {"question", "answer"}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_the_core_does_not_import_torch():
    code = (
        "import sys\n"
        "from wandering_light.stack_workshop import server\n"
        "server.api('/api/evaluate', {'lesson': 'echo', 'program': 'dup add'})\n"
        "server.api('/api/suggest', {'lesson': 'echo', 'program': 'dup'})\n"
        "heavy = {'torch', 'numpy', 'fastapi', 'transformers'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_an_adviser_with_nothing_loaded_says_so(tmp_path):
    adviser = model.Adviser()
    assert adviser.status()["ready"] is False
    assert adviser.status()["error"] is None

    ranked = adviser.rank("dup", one([1], [2]))
    assert ranked["ready"] is False
    assert ranked["suggestions"] == []

    # A refused step is an error whether or not a policy is loaded.
    with pytest.raises(ValueError, match="add: needs 2"):
        adviser.rank("add", one([1], [2]))
    with pytest.raises(ValueError, match="Unknown word"):
        adviser.rank("nope", one([1], [2]))
    with pytest.raises(ValueError):
        adviser.rank("dup", [])

    status = adviser.load(tmp_path / "missing")
    assert status == adviser.status()
    assert status["ready"] is False
    assert "missing" in status["error"]
    (tmp_path / "empty").mkdir()
    assert adviser.load(tmp_path / "empty")["ready"] is False

    assert isinstance(model.ADVISER, model.Adviser)


def test_loading_never_downloads_unless_asked(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from wandering_light.stack_debugger import context

    asked = []

    def resolve(checkpoint, **kwargs):
        asked.append(checkpoint)
        raise FileNotFoundError("offline")

    monkeypatch.delenv(context.CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(context, "DEFAULT_CACHE", tmp_path / "cache")
    monkeypatch.setattr(context, "resolve_checkpoint", resolve)

    adviser = model.Adviser()
    status = adviser.load()
    assert status["ready"] is False
    assert "No stack policy checkpoint" in status["error"]
    assert asked == []

    status = adviser.load(download=True)
    assert status["ready"] is False
    assert "offline" in status["error"]
    assert asked == [None]


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from wandering_light.rollout_policy import PolicyArchitecture
    from wandering_light.stack_policy import StackPairPolicy, save_policy

    torch.manual_seed(0)
    return save_policy(
        tmp_path / "tiny",
        StackPairPolicy(PolicyArchitecture(8, 16, 32)),
        trainer_state={},
        manifest={"arm": "tiny", "seed": 0},
        wandb_url="https://wandb.example/run",
    )


def test_a_loaded_adviser_ranks_only_legal_ops(tiny_checkpoint, monkeypatch):
    adviser = model.Adviser()
    status = adviser.load(tiny_checkpoint)
    assert status["ready"] is True
    assert status["error"] is None
    assert status["checkpoint"] == str(tiny_checkpoint.resolve())
    assert status["arm"] == "tiny"
    assert status["wandb_url"] == "https://wandb.example/run"
    assert status["parameters"] > 0

    examples = one([[1, 2]], [3]) + one([[]], [0])
    ranked = adviser.rank("0", examples)
    assert ranked["ready"] is True
    suggestions = ranked["suggestions"]
    assert 0 < len(suggestions) <= 6
    assert all(set(s) == {"op", "probability", "previews"} for s in suggestions)
    probabilities = [s["probability"] for s in suggestions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= p <= 1 for p in probabilities)
    assert sum(probabilities) <= 1 + 1e-6

    stacks = [stack_from([[1, 2], 0]), stack_from([[], 0])]
    legal = {
        name
        for name, op in OPS.items()
        if all(not isinstance(step_raw(op, stack, FUEL, 0), str) for stack in stacks)
    }
    # `uncons` and `div` are refused on these stacks; `cons` is not.
    assert "cons" in legal and "add" not in legal and "div" not in legal
    for suggestion in suggestions:
        assert suggestion["op"] in legal
        assert suggestion["previews"] == [
            snapshot(step_raw(OPS[suggestion["op"]], stack, FUEL, 0)[0])
            for stack in stacks
        ]
    json.dumps(ranked)

    # The ranking is the policy's own, normalised over the legal ops.
    import numpy as np

    from wandering_light.stack_policy import (
        OP_NAMES,
        PolicyScorer,
        encode_item,
        encode_targets,
        item_array,
        load_policy,
    )

    scorer = PolicyScorer(load_policy(tiny_checkpoint))
    targets = encode_targets([stack_from([3]), stack_from([0])])
    block = item_array(
        [encode_item(s, t) for s, t in zip(stacks, targets, strict=True)]
    )
    mask = np.array([[name in legal for name in OP_NAMES]])
    log_probs = scorer.log_probs([block], mask)[0]
    best = sorted(legal, key=lambda name: -log_probs[OP_NAMES.index(name)])[:6]
    assert [s["op"] for s in suggestions] == best
    assert suggestions[0]["probability"] == pytest.approx(
        float(np.exp(log_probs[OP_NAMES.index(best[0])])), abs=1e-5
    )

    # An op legal on one example and refused on another is not offered.
    mixed = adviser.rank("", one([[1]], [1]) + one([[]], [0]))
    assert "uncons" not in {s["op"] for s in mixed["suggestions"]}
    alone = adviser.rank("", one([[1]], [1]))
    assert len(alone["suggestions"]) == 6

    # A failed reload keeps the working policy and records the failure.
    status = adviser.load(tiny_checkpoint / "nope")
    assert status["ready"] is True
    assert status["error"]
    with pytest.raises(ValueError):
        adviser.rank("add", examples)


# ---------------------------------------------------------------------------
# server: api
# ---------------------------------------------------------------------------


def test_context_never_hands_over_an_answer():
    body = server.api("/api/context", {})
    assert set(body) == {"lessons", "ops", "macros", "model", "world"}
    assert [lesson["id"] for lesson in body["lessons"]] == [
        lesson["id"] for lesson in LESSONS
    ]
    for shown, lesson in zip(body["lessons"], LESSONS, strict=True):
        assert set(shown) == (LESSON_FIELDS - {"solution", "heldout", "hints"}) | {
            "extra_tests",
            "hint_count",
        }
        assert shown["extra_tests"] == 3
        assert shown["hint_count"] == len(lesson["hints"])
        assert shown["examples"] == lesson["examples"]
        text = json.dumps(shown)
        assert all(json.dumps(hint) not in text for hint in lesson["hints"])
        # Nor does the prose around a lesson spell its solution out.
        prose = [shown[key] for key in ("concept", "story", "goal", "discovery")]
        prose += [shown["starter"], *shown["review"].values()]
        assert all(lesson["solution"] not in passage for passage in prose)
        assert shown["prediction"]["program"] != lesson["solution"]
        held = [json.dumps(example) for example in lesson["heldout"]]
        assert all(example not in text for example in held)
    # The page got copies: the curriculum still has its answers.
    assert all("solution" in lesson and "heldout" in lesson for lesson in LESSONS)

    assert [op["name"] for op in body["ops"]] == list(OPS)
    assert body["ops"][0] == {"name": "0", "group": "literals", "effect": "( -- 0 )"}

    assert [macro["name"] for macro in body["macros"]] == list(MACROS)
    by_name = {macro["name"]: macro for macro in body["macros"]}
    assert by_name["inc"] == {
        "name": "inc",
        "body": "1 add",
        "group": "scalars",
        "expansion": "1 add",
    }
    assert by_name["even?"]["expansion"] == "1 1 add over over div mul sub 0 eq?"
    assert all(
        macro["expansion"].split() == ops_of(macro["name"]) for macro in body["macros"]
    )

    assert body["model"] == model.ADVISER.status()
    assert "ready" in body["model"]
    json.dumps(body)


def test_the_world_shelf_never_fails_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORLD_MANIFEST", tmp_path / "absent" / "manifest.json")
    world = server.api("/api/context", {})["world"]
    assert world["available"] is False
    assert world["tasks"] == []
    assert world["note"]

    # A manifest whose payload is not on disk, and no Hub to ask.
    manifest = {
        "corpus": "tiny_world",
        "hub": {"repo_id": "nobody/nothing", "revision": "0" * 40},
        "splits": {"train": {"path": "tiny.jsonl.gz", "sha256": "sha256:0"}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(server, "WORLD_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    world = server.world_shelf()
    assert world["available"] is False
    assert world["corpus"] == "tiny_world"
    assert world["tasks"] == []


def test_the_world_shelf_lists_what_is_on_disk(tmp_path, monkeypatch):
    import gzip
    import hashlib

    rows = [
        {
            "behaviour": "long_one",
            "summary": "second",
            "length": 9,
            "inputs": ["[ (1 2) ]"],
            "outputs": ["[ 3 ]"],
            "program": ["0", "[", "add", "]", "fold"],
        },
        {
            "behaviour": "double",
            "summary": "first",
            "length": 2,
            "inputs": ["[ 1 ]", "[ 4 ]"],
            "outputs": ["[ 2 ]", "[ 8 ]"],
            "program": ["dup", "add"],
        },
        {
            "behaviour": "too_big_to_type",
            "summary": "left off",
            "length": 1,
            "inputs": [f"[ {10**13} ]"],
            "outputs": ["[ 0 ]"],
            "program": ["drop"],
        },
    ]
    payload = tmp_path / "tiny.jsonl.gz"
    with gzip.open(payload, "wt") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    digest = "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "corpus": "tiny_world",
        "splits": {"train": {"path": "tiny.jsonl.gz", "sha256": digest}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(server, "WORLD_MANIFEST", tmp_path / "manifest.json")

    world = server.world_shelf()
    assert world["available"] is True
    assert [task["id"] for task in world["tasks"]] == ["double", "long_one"]
    task = world["tasks"][0]
    assert set(task) == {"id", "summary", "length", "examples"}
    # A shelf task runs in the sandbox as it stands, and carries no answer.
    verdict = server.api(
        "/api/verify",
        {"lesson": "sandbox", "program": "dup add", "examples": task["examples"]},
    )
    assert verdict["passed"] is True

    # A payload that is not the one the manifest pins is not read.
    manifest["splits"]["train"]["sha256"] = "sha256:" + "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert server.world_shelf()["available"] is False


def test_evaluate_is_execute_on_the_shown_examples():
    lesson = BY_ID["tally"]
    program = "0 [ add ] fold"
    body = server.api("/api/evaluate", {"lesson": "tally", "program": program})
    assert body == execute(program, lesson["examples"])
    assert body["count"] == 3
    assert len(body["examples"][0]["frames"]) == 6

    examples = one("3", "6") + one([5], [10])
    sandbox = server.api(
        "/api/evaluate",
        {"lesson": "sandbox", "program": "dup add", "examples": examples},
    )
    assert sandbox == execute("dup add", examples)
    assert sandbox["matches"] == 2


def test_verify_says_what_the_verdict_rests_on():
    for lesson in LESSONS:
        verdict = server.api(
            "/api/verify", {"lesson": lesson["id"], "program": lesson["solution"]}
        )
        assert verdict == {
            "passed": True,
            "shown": 3,
            "shown_count": 3,
            "extra": 3,
            "extra_count": 3,
            "failures": [],
            "evidence": "shown and extra tests",
        }, lesson["id"]


def test_verify_catches_a_program_that_only_fits_what_was_shown():
    lesson = BY_ID["three-nights"]
    verdict = server.api(
        "/api/verify", {"lesson": "three-nights", "program": lesson["starter"]}
    )
    assert verdict["passed"] is False
    assert (verdict["shown"], verdict["shown_count"]) == (3, 3)
    assert (verdict["extra"], verdict["extra_count"]) == (1, 3)
    assert verdict["evidence"] == "shown and extra tests"
    assert len(verdict["failures"]) == 2
    for failure in verdict["failures"]:
        assert failure["source"] == "extra"
        assert failure["matched"] is False
        # First and final frame: the counterexample and what the program left.
        assert len(failure["frames"]) == 2
    first = verdict["failures"][0]
    assert first["index"] == 0
    assert texts(first["frames"][0]["stack"]) == ["7", "3"]
    assert texts(first["frames"][-1]["stack"]) == ["3"]
    assert texts(first["output"]) == ["7"]

    # The same program on the shown examples alone looks finished.
    shown = server.api(
        "/api/evaluate", {"lesson": "three-nights", "program": lesson["starter"]}
    )
    assert shown["matches"] == shown["count"] == 3


def test_verify_returns_the_first_three_failing_rows():
    verdict = server.api("/api/verify", {"lesson": "tally", "program": "drop 1 1 add"})
    assert verdict["passed"] is False
    assert (verdict["shown"], verdict["extra"]) == (0, 0)
    assert [(f["source"], f["index"]) for f in verdict["failures"]] == [
        ("shown", 0),
        ("shown", 1),
        ("shown", 2),
    ]

    refused = server.api("/api/verify", {"lesson": "tally", "program": "add"})
    assert refused["passed"] is False
    assert refused["failures"][0]["error"] == "add: needs 2 value(s), stack has 1"


def test_the_sandbox_has_no_extra_tests():
    examples = one([1, 5], [5]) + one([2, 2], [2])
    verdict = server.api(
        "/api/verify",
        {"lesson": "sandbox", "program": "swap drop", "examples": examples},
    )
    assert verdict == {
        "passed": True,
        "shown": 2,
        "shown_count": 2,
        "extra": 0,
        "extra_count": 0,
        "failures": [],
        "evidence": "examples",
    }
    failing = server.api(
        "/api/verify",
        {"lesson": "sandbox", "program": "drop drop", "examples": examples},
    )
    assert failing["passed"] is False
    assert failing["evidence"] == "examples"
    assert [f["source"] for f in failing["failures"]] == ["shown", "shown"]

    for route in ("/api/evaluate", "/api/verify", "/api/suggest"):
        with pytest.raises(ValueError):
            server.api(route, {"lesson": "sandbox", "program": "dup"})
        with pytest.raises(ValueError):
            server.api(
                route,
                {"lesson": "sandbox", "program": "dup", "examples": one([1], [1]) * 17},
            )


def test_the_remaining_routes():
    inside = server.api(
        "/api/inspect",
        {"lesson": "every-lamp", "program": "[ dup add ] map", "step": 4},
    )
    assert inside == inspect_op("[ dup add ] map", BY_ID["every-lamp"]["examples"], 4)
    assert len(inside["frames"]) == 6
    third = server.api(
        "/api/inspect",
        {
            "lesson": "every-lamp",
            "program": "[ dup add ] map",
            "step": 4,
            "example": 2,
        },
    )
    assert len(third["frames"]) == 4

    hints = BY_ID["echo"]["hints"]
    assert server.api("/api/hint", {"lesson": "echo"}) == {
        "hint": hints[0],
        "index": 0,
        "count": len(hints),
    }
    last = server.api("/api/hint", {"lesson": "echo", "index": len(hints) - 1})
    assert last["hint"] == hints[-1]
    with pytest.raises(ValueError):
        server.api("/api/hint", {"lesson": "echo", "index": len(hints)})
    with pytest.raises(ValueError):
        server.api("/api/hint", {"lesson": "echo", "index": -1})

    assert server.api("/api/reveal", {"lesson": "echo"}) == {"solution": "dup add"}
    for route in ("/api/hint", "/api/reveal"):
        with pytest.raises(ValueError):
            server.api(route, {"lesson": "sandbox", "examples": one([1], [1])})


def test_suggest_and_model_go_through_the_adviser(tmp_path, monkeypatch):
    adviser = model.Adviser()
    monkeypatch.setattr(server, "ADVISER", adviser)

    body = server.api("/api/suggest", {"lesson": "echo", "program": "dup"})
    assert body == {**adviser.status(), "suggestions": []}
    with pytest.raises(ValueError, match="add: needs 2"):
        server.api("/api/suggest", {"lesson": "echo", "program": "add"})

    status = server.api("/api/model", {"checkpoint": str(tmp_path / "missing")})
    assert status == adviser.status()
    assert status["ready"] is False
    assert status["error"]
    assert server.api("/api/context", {})["model"] == status
    with pytest.raises(ValueError):
        server.api("/api/model", {"checkpoint": 7})


def test_model_route_loads_a_checkpoint(tiny_checkpoint, monkeypatch):
    adviser = model.Adviser()
    monkeypatch.setattr(server, "ADVISER", adviser)
    status = server.api("/api/model", {"checkpoint": str(tiny_checkpoint)})
    assert status["ready"] is True
    body = server.api("/api/suggest", {"lesson": "tally", "program": "0 [ add ]"})
    assert body["ready"] is True
    assert 0 < len(body["suggestions"]) <= 6
    assert all(len(s["previews"]) == 3 for s in body["suggestions"])


@pytest.mark.parametrize(
    ("route", "data"),
    [
        ("/api/nope", {}),
        ("/", {}),
        ("/api/evaluate", {}),
        ("/api/evaluate", {"lesson": "no-such-lesson", "program": "dup"}),
        ("/api/evaluate", {"lesson": ["echo"], "program": "dup"}),
        ("/api/evaluate", {"lesson": "echo"}),
        ("/api/evaluate", {"lesson": "echo", "program": ["dup"]}),
        ("/api/evaluate", {"lesson": "echo", "program": "frobnicate"}),
        ("/api/verify", {"lesson": "echo", "program": None}),
        ("/api/inspect", {"lesson": "echo", "program": "dup add"}),
        ("/api/inspect", {"lesson": "echo", "program": "dup add", "step": 9}),
        ("/api/hint", {}),
        ("/api/hint", {"lesson": "echo", "index": "0"}),
        ("/api/reveal", {"lesson": "nope"}),
        ("/api/evaluate", []),
        ("/api/context", None),
    ],
)
def test_a_request_that_cannot_be_answered_is_a_value_error(route, data):
    with pytest.raises(ValueError):
        server.api(route, data)


# ---------------------------------------------------------------------------
# server: HTTP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def post():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def send(path, body=None, *, raw=None, headers=None, method="POST"):
        data = raw if raw is not None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}{path}",
            data=None if method == "GET" else data,
            headers={"Content-Type": "application/json", **(headers or {})},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    send.port = httpd.server_port
    yield send
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_http_carries_json_to_the_api_and_back(post):
    status, body = post("/api/evaluate", {"lesson": "echo", "program": "dup add"})
    assert status == 200
    assert body == server.api("/api/evaluate", {"lesson": "echo", "program": "dup add"})

    status, body = post("/api/context", {})
    assert status == 200
    assert len(body["lessons"]) == 14

    # No body at all is an empty object; a query string is not part of the route.
    status, body = post("/api/context?cache=1", raw=b"")
    assert status == 200
    assert len(body["ops"]) == 28


def test_http_errors_are_400_with_a_reason(post):
    status, body = post("/api/evaluate", {"lesson": "echo", "program": "frobnicate"})
    assert status == 400
    assert set(body) == {"error"}
    assert body["error"].startswith("Unknown word 'frobnicate'. ")

    for path, payload in [
        ("/api/nope", {}),
        ("/api/evaluate", {"lesson": "nope", "program": "dup"}),
        ("/api/evaluate", {"lesson": "sandbox", "program": "dup", "examples": []}),
        ("/api/inspect", {"lesson": "echo", "program": "dup", "step": 3}),
        ("/api/evaluate", [1, 2]),
        ("/api/evaluate", "dup"),
    ]:
        status, body = post(path, payload)
        assert status == 400, (path, payload)
        assert set(body) == {"error"} and body["error"]

    for raw in (b"{not json", b"[" * 200_000, b"\xff\xfe", b'{"lesson": 1e999999}x'):
        status, body = post("/api/evaluate", raw=raw)
        assert status == 400, raw[:20]
        assert set(body) == {"error"}

    # A body over the limit is refused on its declared length, unread.
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", post.port, timeout=10)
    connection.putrequest("POST", "/api/evaluate")
    connection.putheader("Content-Length", str(server.MAX_BODY + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 400
    assert set(json.loads(response.read())) == {"error"}
    connection.close()

    # A machine refusal is a result, not an error.
    status, body = post("/api/evaluate", {"lesson": "echo", "program": "add"})
    assert status == 200
    assert body["matches"] == 0

    status, body = post("/api/context", method="GET")
    assert status == 405
    assert set(body) == {"error"}


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        (None, 200),
        ("http://127.0.0.1:8018", 200),
        ("http://localhost:5173", 200),
        ("http://LOCALHOST", 200),
        ("https://localhost", 200),
        ("http://evil.example", 403),
        ("http://127.0.0.1.evil.example", 403),
        ("http://localhost.evil.example:8018", 403),
        ("http://evil.example/?http://localhost", 403),
        ("http://localhost@evil.example", 403),
        ("http://127.0.0.2", 403),
        ("http://[::1]:8018", 403),
        ("http://[bad", 403),
        ("null", 403),
        ("", 403),
        ("localhost", 403),
    ],
)
def test_only_this_machine_may_ask(post, origin, status):
    headers = {} if origin is None else {"Origin": origin}
    got, body = post(
        "/api/evaluate", {"lesson": "echo", "program": "dup add"}, headers=headers
    )
    assert got == status
    if status == 403:
        assert set(body) == {"error"}
    else:
        assert body["matches"] == 3


def test_main_serves_on_loopback_port_8018_by_default(monkeypatch, capsys):
    bound = []

    class Fake:
        def __init__(self, address, handler):
            bound.append((address, handler))
            self.server_port = address[1]
            self.closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            bound.append("closed")

    monkeypatch.setattr(server, "ThreadingHTTPServer", Fake)
    assert server.main([]) == 0
    assert server.main(["--port", "9001"]) == 0
    assert bound == [
        (("127.0.0.1", 8018), server.Handler),
        "closed",
        (("127.0.0.1", 9001), server.Handler),
        "closed",
    ]
    assert "127.0.0.1:8018" in capsys.readouterr().out


def test_the_handler_is_a_plain_http_server_handler():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    assert issubclass(server.Handler, BaseHTTPRequestHandler)
    assert server.ThreadingHTTPServer is ThreadingHTTPServer
    assert curriculum.LESSONS is LESSONS
