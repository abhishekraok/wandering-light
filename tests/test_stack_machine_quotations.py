"""Quotations, combinators, and the fuel guard that replaces totality.

The first-order machine was total by construction: every op consumed and
produced in bounded time. Combinators break that -- a quotation can call
itself -- so totality now rests on the guard refusing what it cannot finish.
These tests are mostly about that.
"""

import json
import os
import subprocess
import sys

import pytest

from wandering_light.stack_machine import (
    BASIS_OPS,
    CORE_OPS,
    FIRST_ORDER_OPS,
    MARK,
    MARK_OPS,
    NIL,
    OP_BY_NAME,
    OPS,
    Bool,
    Config,
    IllegalOpError,
    Int,
    Lst,
    MachineError,
    Mark,
    Sym,
    format_stack,
    format_value,
    legal_ops,
    ops_for_basis,
    parse_program,
    parse_stack,
    run,
    step,
    value_key,
)


def execute(source: str, config: Config | None = None, *, ops=OPS) -> Config:
    final = config if config is not None else Config()
    for reached in run(parse_program(source, ops=ops), final):
        final = reached
    return final


def execute_mark(source: str, config: Config | None = None) -> Config:
    return execute(source, config, ops=MARK_OPS)


# ---------------------------------------------------------------------------
# Symbols are values
# ---------------------------------------------------------------------------


def test_symbol_is_structurally_distinct_from_everything_else():
    assert value_key(Sym("1")) != value_key(Int(1))
    assert value_key(Sym("true")) != value_key(Bool(True))
    assert value_key(Sym("nil")) != value_key(NIL)
    assert Sym("add") != Int(1)
    assert value_key(Sym("add")) == ("sym", "add")


def test_symbol_rejects_a_non_string():
    with pytest.raises(MachineError):
        Sym(1)


def test_every_core_op_has_exactly_one_symbol():
    symbols = {op.name for op in OPS if op.group == "symbols"}
    assert symbols == {f"'{op.name}" for op in CORE_OPS}
    assert len(OPS) == 2 * len(CORE_OPS)


def test_first_order_basis_is_unchanged_and_is_a_prefix():
    """The 18-op basis must survive intact, in order, so earlier work holds."""
    assert len(FIRST_ORDER_OPS) == 18
    assert OPS[:18] == FIRST_ORDER_OPS
    assert all(op.group != "symbols" for op in FIRST_ORDER_OPS)


def test_a_quotation_is_an_ordinary_list():
    """No separate program representation: nil and cons build programs."""
    final = execute("nil 'add cons '1 cons")
    assert final.stack == (Lst((Sym("1"), Sym("add"))),)
    assert format_value(final.stack[0]) == "(1 add)"
    # and it comes apart with the same ops
    assert execute("uncons", final).stack == (Lst((Sym("add"),)), Sym("1"))


# ---------------------------------------------------------------------------
# MARK quotations
# ---------------------------------------------------------------------------


def test_mark_is_a_structural_tagged_value():
    assert Mark() == MARK
    assert value_key(MARK) == ("mark",)
    assert value_key(MARK) != value_key(Sym("MARK"))
    assert Config((MARK,)).key == (("mark",),)
    assert all(type(part) is str for part in value_key(MARK))
    encoded = format_stack(Config((MARK,)))
    assert parse_stack(encoded) == (Sym("MARK"),)
    assert parse_stack(encoded, basis="mark-28") == (MARK,)


def test_named_bases_are_stable_and_mark_28_has_28_ops():
    assert BASIS_OPS["executable-26"] is CORE_OPS
    assert BASIS_OPS["full-52"] is OPS
    assert BASIS_OPS["mark-28"] is MARK_OPS
    assert ops_for_basis("mark-28") is MARK_OPS
    assert len(MARK_OPS) == 28
    assert [op.name for op in MARK_OPS[-2:]] == ["[", "]"]


def test_open_mark_is_always_structurally_legal():
    open_mark = MARK_OPS[-2]
    for stack in ((), (Int(7),), (NIL, Bool(False)), (MARK, Int(1))):
        config = Config(stack)
        assert open_mark.guard(config) is None
        assert open_mark.apply(config) == ((*stack, MARK), 1)


def test_close_mark_is_guarded_on_mark_presence():
    close_mark = MARK_OPS[-1]
    for stack in ((), (Int(1),), (Lst((MARK,)),)):
        assert close_mark.guard(Config(stack)) == "no open quotation"
    assert close_mark.guard(Config((Int(1), MARK, Int(0)))) is None


def test_core_ops_capture_their_names_under_any_mark_and_run_without_one():
    marked = Config((Int(2), MARK, Int(7)))
    for op in MARK_OPS[:-2]:
        if op.group == "literals":
            continue
        assert op.guard(marked) is None, op.name
        assert op.apply(marked) == ((*marked.stack, Sym(op.name)), 1), op.name

    add = next(op for op in MARK_OPS if op.name == "add")
    assert step(Config((Int(2), Int(7))), add).stack == (Int(9),)


def test_literals_remain_values_under_a_mark():
    expected = {
        "0": Int(0),
        "1": Int(1),
        "nil": NIL,
        "true": Bool(True),
        "false": Bool(False),
    }
    by_name = {op.name: op for op in MARK_OPS}
    for name, value in expected.items():
        assert step(Config((MARK,)), by_name[name]).stack == (MARK, value)


def test_nested_close_uses_the_nearest_mark():
    inner_closed = execute_mark("[ 1 [ 0 add ]")
    assert inner_closed.stack == (
        MARK,
        Int(1),
        Lst((Int(0), Sym("add"))),
    )

    final = execute_mark("true ]", inner_closed)
    assert final.stack == (Lst((Int(1), Lst((Int(0), Sym("add"))), Bool(True))),)
    assert execute_mark("[ ]").stack == (NIL,)


def test_quote_open_config_key_is_visible_and_sufficient():
    reached = execute_mark("[ add")
    restored = Config((Mark(), Sym("add")), fuel=99)
    different_body = execute_mark("[ sub")
    closed = Config((Lst((Sym("add"),)),))

    assert reached.key == restored.key
    assert reached.key != different_body.key
    assert reached.key != closed.key
    assert legal_ops(reached, ops=MARK_OPS) == legal_ops(restored, ops=MARK_OPS)
    for op in legal_ops(reached, ops=MARK_OPS):
        assert op.apply(reached) == op.apply(restored), op.name


def test_published_mark_28_program_costs_and_outputs():
    full_predicate = (
        "nil 'eq? cons 0 cons 'sub cons 'mul cons 'add cons 1 cons 1 cons "
        "'div cons 'add cons 1 cons 1 cons 'dup cons"
    )
    mark_predicate = "[ dup 1 1 add div 1 1 add mul sub 0 eq? ]"
    full_even_squares = (
        f"{full_predicate} filter nil 'mul cons 'dup cons map 0 nil 'add cons fold"
    )
    mark_even_squares = f"{mark_predicate} filter [ dup mul ] map 0 [ add ] fold"
    full_recursive_sum = (
        "nil 'ifte cons nil 'i cons 'dup cons 'swap cons 'cons cons 'swap cons "
        "'dip cons nil 'add cons cons 'swap cons 'uncons cons 'swap cons "
        "'uncons cons 'swap cons cons nil 'drop cons 'swap cons 'uncons cons "
        "'drop cons cons nil 'null? cons 'drop cons 'uncons cons 'over cons "
        "cons dup i"
    )
    mark_recursive_sum = (
        "[ [ over uncons drop null? ] [ drop uncons swap drop ] "
        "[ swap uncons swap uncons swap [ add ] dip swap cons swap dup i ] "
        "ifte ] dup i"
    )
    predicate = "[ (dup 1 1 add div 1 1 add mul sub 0 eq?) ]"
    even_input = (Lst(tuple(Int(value) for value in (3, 4, 1, 2, 7, 6))),)
    sum_input = (Lst(tuple(Int(value) for value in (0, 1, 2, 3, 4))),)
    cases = (
        (full_predicate, OPS, (), predicate, 25, 25),
        (mark_predicate, MARK_OPS, (), predicate, 14, 14),
        (full_even_squares, OPS, even_input, "[ 56 ]", 37, 118),
        (mark_even_squares, MARK_OPS, even_input, "[ 56 ]", 25, 106),
        (full_recursive_sum, OPS, sum_input, "[ 10 ]", 53, 149),
        (mark_recursive_sum, MARK_OPS, sum_input, "[ 10 ]", 33, 129),
    )

    for source, ops, initial, expected, length, fuel in cases:
        program = parse_program(source, ops)
        final = execute(source, Config(initial), ops=ops)
        assert len(program) == length
        assert final.fuel == fuel
        assert format_stack(final) == expected


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


def test_i_runs_a_quotation():
    assert execute("1 1 nil 'add cons i").stack == (Int(2),)


def test_quote_and_cat_compose_quotations():
    assert execute("1 quote").stack == (Lst((Int(1),)),)
    assert execute("nil '1 cons nil 'add cons cat").stack == (
        Lst((Sym("1"), Sym("add"))),
    )


def test_quote_equals_nil_swap_cons():
    """quote is a convenience; it must agree with the long way round."""
    for source in ("1", "nil", "true", "nil 1 cons"):
        assert (
            execute(f"{source} quote").stack == execute(f"{source} nil swap cons").stack
        )


def test_dip_runs_under_the_top_value():
    assert execute("1 1 0 nil 'add cons dip").stack == (Int(2), Int(0))


def test_map_applies_the_body_elementwise():
    assert execute("nil 1 cons 0 cons 1 cons nil 'add cons '1 cons map").stack == (
        Lst((Int(2), Int(1), Int(2))),
    )


def test_filter_keeps_what_the_body_calls_true():
    assert execute("nil 1 cons 0 cons 1 cons nil 'eq? cons '0 cons filter").stack == (
        Lst((Int(0),)),
    )


def test_fold_accumulates():
    assert execute("nil 1 cons 1 cons 1 cons 0 nil 'add cons fold").stack == (Int(3),)


def test_ifte_picks_a_branch():
    assert execute("0 1 nil 'lt? cons nil '1 cons nil '0 cons ifte").stack == (
        Int(0),
        Int(1),
        Int(1),
    )
    assert execute("1 0 nil 'lt? cons nil '1 cons nil '0 cons ifte").stack == (
        Int(1),
        Int(0),
        Int(0),
    )


def test_combinators_nest():
    source = (
        "nil  nil 1 cons 0 cons cons  nil 1 cons 1 cons cons  "
        "nil 'add cons '1 cons quote nil 'map cons cat map"
    )
    assert execute(source).stack == (
        Lst((Lst((Int(2), Int(2))), Lst((Int(1), Int(2))))),
    )


def test_map_body_must_leave_exactly_one_value():
    config = execute("nil 1 cons nil 'dup cons")
    assert "exactly one value" in OP_BY_NAME["map"].guard(config)


def test_filter_body_must_leave_a_boolean():
    # body (drop 1) leaves exactly one value, but an integer rather than a bool
    config = execute("nil 1 cons nil '1 cons 'drop cons")
    assert "boolean" in OP_BY_NAME["filter"].guard(config)


# ---------------------------------------------------------------------------
# Fuel: the guard is now what makes the machine total
# ---------------------------------------------------------------------------


def test_a_divergent_program_is_refused_not_hung():
    """`[dup i] dup i` is the classic non-terminating concatenative program."""
    config = execute("nil 'i cons 'dup cons dup")
    reason = OP_BY_NAME["i"].guard(config)
    assert reason is not None and "deep" in reason
    assert OP_BY_NAME["i"] not in legal_ops(config)
    with pytest.raises(IllegalOpError):
        step(config, OP_BY_NAME["i"])


def test_self_application_that_terminates_still_works():
    assert execute("nil 'drop cons dup i").stack == ()


def test_fuel_limit_refuses_a_long_but_finite_computation():
    config = execute("nil 1 cons 1 cons 1 cons nil 'add cons '1 cons")
    generous = Config(config.stack, fuel=0, fuel_limit=100)
    assert OP_BY_NAME["map"].guard(generous) is None
    tight = Config(config.stack, fuel=0, fuel_limit=3)
    assert OP_BY_NAME["map"].guard(tight) == "out of fuel"


# Each program ends in its combinator and costs exactly the fuel given, so a
# limit equal to the cost is the case `map`, `filter` and `fold` used to refuse
# after spending on the last item (#145). The last is the issue's repro.
FUEL_BOUNDARY = (
    ("nil 1 cons 1 cons [ 1 add ] map", 14, MARK_OPS),
    ("nil 0 cons 1 cons [ 0 eq? ] filter", 14, MARK_OPS),
    ("nil 1 cons 1 cons 0 [ add ] fold", 12, MARK_OPS),
    ("1 quote 0 'add quote fold", 7, OPS),
)


@pytest.mark.parametrize(("source", "cost", "ops"), FUEL_BOUNDARY)
def test_a_combinator_finishes_on_its_last_unit_of_fuel(source, cost, ops):
    """A computation needing N fuel runs under a limit of N, and reports N."""
    assert execute(source, Config(fuel_limit=cost), ops=ops).fuel == cost
    with pytest.raises(IllegalOpError, match="out of fuel"):
        execute(source, Config(fuel_limit=cost - 1), ops=ops)


@pytest.mark.parametrize(("source", "cost", "ops"), FUEL_BOUNDARY)
def test_a_combinator_never_spends_more_than_its_limit(source, cost, ops):
    """Every limit either refuses or finishes, and never overspends."""
    for limit in range(1, cost + 2):
        try:
            final = execute(source, Config(fuel_limit=limit), ops=ops)
        except IllegalOpError:
            continue
        assert final.fuel == cost <= limit


def test_combinator_cost_is_one_plus_the_body():
    """Fuel stops equalling program length once an op can run code."""
    before = execute("nil 1 cons 0 cons 1 cons nil 'add cons '1 cons")
    after = step(before, OP_BY_NAME["map"])
    assert after.fuel - before.fuel == 1 + 3 * 2  # map, then push+add per element


def test_first_order_ops_still_cost_exactly_one():
    probe = Config((Lst((Int(1), Int(0))), Int(1), Int(1)))
    for op in FIRST_ORDER_OPS:
        if op.guard(probe) is None:
            assert op.apply(probe)[1] == 1, op.name


def test_guard_and_apply_agree_for_every_op():
    """If the guard says legal, apply must succeed, and vice versa."""
    configs = [
        execute(s)
        for s in (
            "",
            "1 1",
            "nil 1 cons nil 'add cons '1 cons",
            "nil 'i cons 'dup cons dup",
            "0 1 nil 'lt? cons nil '1 cons nil '0 cons",
        )
    ]
    for config in configs:
        for op in OPS:
            if op.guard(config) is None:
                op.apply(config)  # must not raise
            else:
                with pytest.raises(IllegalOpError):
                    step(config, op)


def test_caching_does_not_change_results():
    """run_body is cached; asking twice must give the same answer."""
    config = execute("nil 1 cons 0 cons nil 'add cons '1 cons")
    first = OP_BY_NAME["map"].apply(config)
    second = OP_BY_NAME["map"].apply(config)
    assert first == second
    assert OP_BY_NAME["map"].guard(config) is None


# ---------------------------------------------------------------------------
# Identity still holds with programs on the stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["0", "1", "31337"])
def test_keys_are_identical_across_pythonhashseed_with_quotations(seed):
    script = """
import json, sys
from wandering_light.stack_machine import Config, parse_program, run
config = Config()
for config in run(parse_program(sys.argv[1]), config):
    pass
print(json.dumps(config.key))
"""
    program = "nil 'add cons '1 cons dup nil 1 cons 0 cons swap map swap quote cat"
    result = subprocess.run(
        [sys.executable, "-c", script, program],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    assert result.stdout.strip() == json.dumps(execute(program).key)


@pytest.mark.parametrize("seed", ["0", "1", "31337"])
def test_open_quote_keys_are_identical_across_pythonhashseed(seed):
    script = """
import json, sys
from wandering_light.stack_machine import Config, MARK_OPS, parse_program, run
config = Config()
for config in run(parse_program(sys.argv[1], ops=MARK_OPS), config):
    pass
print(json.dumps(config.key))
"""
    program = "[ add 1 [ mul"
    result = subprocess.run(
        [sys.executable, "-c", script, program],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    assert result.stdout.strip() == json.dumps(execute_mark(program).key)
