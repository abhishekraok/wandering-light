"""The search-side predicate over partially built ``mark-28`` quotations.

The machine is unchanged by this module, and these tests hold it to that: the
guard only ever *narrows* what the machine already allows, and it must never
narrow away a program a person has written. The decisive test is the last
group, which replays every catalogue witness and checks that no prefix of any
of them is blocked under any mode.
"""

import random

import pytest

from wandering_light.quotation_guard import (
    FIXED_SUPPLY,
    MODES,
    blocked,
    body_effect,
    open_body,
)
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    MARK,
    MARK_OPS,
    NIL,
    Bool,
    Int,
    Lst,
    Sym,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import CATALOGUE, to_stack

OP_BY_NAME = {op.name: op for op in MARK_OPS}


def body(text: str) -> tuple:
    """A body from its printed form; op names become symbols, literals values."""
    return parse_stack(text, basis="mark-28")


def stack(text: str) -> tuple:
    return parse_stack(text, basis="mark-28")


def run_prefixes(program: tuple[str, ...], start: tuple):
    """Yield ``(stack, remaining)`` before each op and after the last."""
    current = start
    for index, name in enumerate(program):
        yield current, len(program) - index
        outcome = step_raw(OP_BY_NAME[name], current, DEFAULT_FUEL_LIMIT, 0)
        assert not isinstance(outcome, str), f"{' '.join(program)}: {name}: {outcome}"
        current = outcome[0]
    yield current, 0


# ---------------------------------------------------------------------------
# body_effect
# ---------------------------------------------------------------------------


def test_empty_body_is_runnable_and_reaches_nothing():
    assert body_effect(()) == (True, 0)


def test_reach_counts_pops_beyond_pushes():
    assert body_effect(body("add add add")) == (True, 4)
    assert body_effect(body("add")) == (True, 2)
    assert body_effect(body("1 add")) == (True, 1)
    assert body_effect(body("1 1 add")) == (True, 0)
    assert body_effect(body("dup")) == (True, 1)
    assert body_effect(body("swap drop drop")) == (True, 2)


def test_reach_is_the_high_water_mark_not_the_net():
    # ``over`` pops two and pushes three, so the body ends a value *deeper*
    # than it started, yet it reached two consumer values to get there.
    assert body_effect(body("over")) == (True, 2)
    assert body_effect(body("over over")) == (True, 2)


@pytest.mark.parametrize(
    "text",
    [
        "0 null?",  # list op on an int
        "nil add",  # arithmetic on a list
        "true uncons",  # list op on a bool
        "1 nil cons",  # cons wants the list second, not on top
        "nil 0 cat",  # cat wants two lists
        "0 i",  # i runs a list
        "1 0 lt? add",  # a bool the body made, then arithmetic on it
        "nil nil cons null? dip",  # dip runs a list, sees a bool
        "0 nil map",  # map wants the list below the quotation
        "nil 0 map",  # ... and a quotation on top
        "nil 0 0 fold",  # fold wants a quotation on top
        "0 nil nil fold",  # ... and a list third
        "nil nil 0 ifte",  # ifte wants three quotations
        "1 1 add dup mul null?",  # an int through several ops, then a list op
    ],
)
def test_bodies_that_must_fail_are_unrunnable(text):
    runnable, _reach = body_effect(body(text))
    assert runnable is False


@pytest.mark.parametrize(
    "text",
    [
        "add",  # consumer supplies two ints
        "null?",  # consumer supplies a list
        "nil 1 cons",  # the body built a list and consed onto it
        "nil 1 cons uncons add",  # uncons leaves two unknowns
        "nil 1 cons uncons null?",
        "dup 0 lt?",  # unknown dup'd, then compared with the body's 0
        "1 1 add 1 lt?",
        "nil 1 cons nil cat",
        "0 quote null?",  # quote makes a list from anything
        "0 1 eq?",  # eq? accepts anything
        "true false eq?",
        "nil ( add ) map",
        "nil 0 ( add ) fold",
        "( dup ) ( drop ) ( swap ) ifte",
        "( add ) i",
        "0 ( add ) dip",
        "( add ) i null?",  # after i, nothing is known
        "0 ( add ) dip add",  # dip keeps x but knows nothing below
        "nil ( add ) map null?",  # map always leaves a list
        "nil 0 ( add ) fold null?",  # fold leaves an unknown
        "nil 0 ( add ) fold add",
    ],
)
def test_bodies_that_can_run_are_runnable(text):
    runnable, _reach = body_effect(body(text))
    assert runnable is True


def test_nested_closed_quotation_is_a_value_not_judged():
    assert body_effect(body("( 0 null? ) drop")) == (True, 0)
    assert body_effect(body("( 0 null? ) i")) == (True, 0)
    # ... but it *is* a list, and arithmetic on a list cannot run.
    assert body_effect(body("( 0 null? ) 1 add"))[0] is False


def test_reach_stops_counting_once_code_runs():
    # ``i`` pops the quotation from the consumer: that one pop counts.
    assert body_effect(body("i")) == (True, 1)
    assert body_effect(body("i add add add")) == (True, 1)
    assert body_effect(body("dip")) == (True, 2)
    assert body_effect(body("dip drop drop")) == (True, 2)
    assert body_effect(body("fold")) == (True, 3)
    assert body_effect(body("ifte add")) == (True, 3)
    assert body_effect(body("drop i")) == (True, 2)
    assert body_effect(body("add i"))[0] is False  # i would run the int add made


def test_unknown_symbols_cannot_run():
    # ``[`` and ``]`` are never captured, and ``run_body`` refuses an unknown
    # name on every stack.
    assert body_effect((Sym("["),))[0] is False
    assert body_effect((Sym("]"),))[0] is False
    assert body_effect((Sym("not-an-op"),))[0] is False


def test_full_basis_symbol_ops_push_a_symbol():
    # Not reachable in ``mark-28``, but ``run_body`` knows them.
    assert body_effect((Sym("'add"),)) == (True, 0)
    assert body_effect((Sym("'add"), Sym("add")))[0] is False


def _random_body(rng: random.Random, length: int) -> tuple:
    names = [op.name for op in MARK_OPS if op.name not in ("[", "]")]
    items = []
    for _ in range(length):
        op = OP_BY_NAME[rng.choice(names)]
        if op.group == "literals":
            items.append(step_raw(op, (), 10, 0)[0][0])
        elif rng.random() < 0.15:
            items.append(Lst((Sym("add"),)))
        else:
            items.append(Sym(op.name))
    return tuple(items)


def test_effect_is_monotone_under_extension():
    rng = random.Random(7)
    for _ in range(500):
        items = _random_body(rng, rng.randint(1, 10))
        runnable, reach = True, 0
        for end in range(len(items) + 1):
            next_runnable, next_reach = body_effect(items[:end])
            assert next_runnable <= runnable  # True may become False, never back
            assert next_reach >= reach
            runnable, reach = next_runnable, next_reach


def test_unrunnable_means_the_machine_refuses_on_every_stack():
    """An ``i`` of an unrunnable body refuses on each of several stacks."""
    rng = random.Random(11)
    stacks = [
        (),
        (Int(3),),
        (Int(3), Int(4)),
        (NIL,),
        (Lst((Int(1), Int(2))), Int(0)),
        (Bool(True), Lst((Int(1),)), Int(2)),
        (Int(1), Int(2), Int(3), Lst((Int(4),))),
    ]
    checked = 0
    for _ in range(2000):
        items = _random_body(rng, rng.randint(1, 6))
        if body_effect(items)[0]:
            continue
        checked += 1
        for base in stacks:
            outcome = step_raw(OP_BY_NAME["i"], (*base, Lst(items)), 1000, 0)
            assert isinstance(outcome, str), (items, base, outcome)
    assert checked > 50


# ---------------------------------------------------------------------------
# open_body
# ---------------------------------------------------------------------------


def test_open_body_without_a_mark_is_none():
    assert open_body(()) is None
    assert open_body(stack("1 2 ( add )")) is None


def test_open_body_reads_after_the_innermost_mark():
    assert open_body((MARK,)) == ((), 0)
    assert open_body((Int(1), MARK, Sym("add"))) == ((Sym("add"),), 1)
    nested = (Int(1), Int(2), MARK, Sym("add"), MARK, Int(0), Sym("dup"))
    assert open_body(nested) == ((Int(0), Sym("dup")), 4)


def test_open_body_counts_values_below_including_outer_marks():
    assert open_body((MARK, MARK, MARK))[1] == 2


# ---------------------------------------------------------------------------
# blocked
# ---------------------------------------------------------------------------


def test_modes_are_the_documented_four():
    assert MODES == ("off", "body", "close", "both")


@pytest.mark.parametrize("mode", ["on", "", "BOTH", "type"])
def test_unknown_mode_raises(mode):
    with pytest.raises(ValueError, match="guard mode"):
        blocked((MARK,), 1, mode)


@pytest.mark.parametrize("mode", MODES)
def test_mark_free_stack_is_never_blocked(mode):
    assert blocked((), 0, mode) is None
    assert blocked(stack("1 ( 0 null? )"), 0, mode) is None


def test_off_blocks_nothing():
    assert blocked((MARK,), 0, "off") is None
    assert blocked((MARK, Int(0), Sym("null?")), 0, "off") is None


def test_close_needs_an_op_per_open_mark():
    assert blocked((MARK,), 1, "close") is None
    assert blocked((MARK,), 0, "close") == "close"
    assert blocked((MARK, Sym("add"), MARK), 2, "close") is None
    assert blocked((MARK, Sym("add"), MARK), 1, "close") == "close"
    assert blocked((MARK, Sym("add"), MARK), 1, "both") == "close"
    # ``body`` mode does not count marks.
    assert blocked((MARK,), 0, "body") is None


def test_body_blocks_an_unrunnable_innermost_body():
    bad = (MARK, Int(0), Sym("null?"))
    assert blocked(bad, 5, "body") == "type"
    assert blocked(bad, 5, "both") == "type"
    assert blocked(bad, 5, "close") is None
    good = (MARK, Int(0), Sym("add"))
    assert blocked(good, 5, "body") is None
    assert blocked(good, 5, "both") is None


def test_close_is_reported_before_type():
    bad = (MARK, Int(0), Sym("null?"))
    assert blocked(bad, 0, "both") == "close"


def test_only_the_innermost_body_is_judged():
    # The outer body is unrunnable, the inner is fine: search is not told to
    # stop, because the inner quotation is what it is building now.
    outer_bad = (MARK, Int(0), Sym("null?"), MARK, Sym("add"))
    assert blocked(outer_bad, 5, "both") is None
    inner_bad = (MARK, Sym("add"), MARK, Int(0), Sym("null?"))
    assert blocked(inner_bad, 5, "both") == "type"


def test_reach_is_reported_never_pruned_on():
    deep = (MARK, *([Sym("add")] * 20))
    assert body_effect(deep[1:]) == (True, 21)
    assert body_effect(deep[1:])[1] > FIXED_SUPPLY
    for mode in MODES:
        assert blocked(deep, 1, mode) is None


def test_fixed_supply_is_what_fold_hands_a_body():
    assert FIXED_SUPPLY == 2


# ---------------------------------------------------------------------------
# No catalogue witness is ever blocked
# ---------------------------------------------------------------------------


def _witness_cases():
    for behaviour in CATALOGUE:
        for witness in behaviour.witnesses:
            yield pytest.param(
                behaviour, tuple(witness.split()), id=f"{behaviour.name}:{witness[:40]}"
            )


@pytest.mark.parametrize(("behaviour", "program"), list(_witness_cases()))
def test_no_catalogue_witness_prefix_is_blocked(behaviour, program):
    rng = random.Random(behaviour.name)
    for _ in range(3):
        start = to_stack(behaviour.sample(rng))
        for current, remaining in run_prefixes(program, start):
            for mode in MODES:
                reason = blocked(current, remaining, mode)
                assert reason is None, (behaviour.name, mode, remaining, current)
            found = open_body(current)
            if found is not None:
                assert body_effect(found[0])[0] is True


def test_catalogue_exercises_every_mode_and_a_nested_quotation():
    """The sweep above is only evidence if the witnesses open quotations."""
    opened = nested = 0
    for behaviour in CATALOGUE:
        rng = random.Random(behaviour.name)
        start = to_stack(behaviour.sample(rng))
        for current, _remaining in run_prefixes(behaviour.program, start):
            marks = sum(value is MARK for value in current)
            opened += marks > 0
            nested += marks > 1
    assert opened > 100
    assert nested > 10
