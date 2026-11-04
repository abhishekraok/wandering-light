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
    CLOSE_MARK,
    CORE_OPS,
    DEFAULT_BASIS,
    FIRST_ORDER_OPS,
    MARK,
    MARK_OPS,
    NIL,
    OP_BY_NAME,
    OPEN_MARK,
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
    legal_mask,
    legal_ops,
    main,
    ops_for_basis,
    parse_program,
    parse_stack,
    run,
    step,
    step_raw,
    value_key,
)


def execute(source: str, config: Config | None = None) -> Config:
    final = config if config is not None else Config()
    for reached in run(parse_program(source), final):
        final = reached
    return final


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


# ---------------------------------------------------------------------------
# mark-28: structural quotation with [ and ]
# ---------------------------------------------------------------------------


def execute_mark(source: str, config: Config | None = None) -> Config:
    final = config if config is not None else Config()
    for reached in run(parse_program(source, MARK_OPS), final):
        final = reached
    return final


def test_mark_is_a_fieldless_structural_value():
    assert Mark() == MARK
    assert value_key(MARK) == ("mark",)
    assert value_key(MARK) != value_key(Sym("MARK"))
    assert format_value(MARK) == "MARK"
    with pytest.raises(AttributeError):
        MARK.anything = 1  # frozen and slotted


def test_bases_are_named_and_hold_the_canonical_tuples():
    assert DEFAULT_BASIS == "executable-26"
    assert BASIS_OPS["executable-26"] is CORE_OPS
    assert BASIS_OPS["full-52"] is OPS
    assert BASIS_OPS["mark-28"] is MARK_OPS
    assert (*CORE_OPS, OPEN_MARK, CLOSE_MARK) == MARK_OPS
    assert len(MARK_OPS) == 28 and len(OPS) == 52
    assert ops_for_basis() is CORE_OPS
    for name, ops in BASIS_OPS.items():
        assert ops_for_basis(name) is ops
    with pytest.raises(MachineError, match="unknown basis 'nope'; choose one of"):
        ops_for_basis("nope")


def test_brackets_build_a_quotation_one_op_per_quoted_op():
    final = execute_mark("[ dup mul ]")
    assert final.stack == (Lst((Sym("dup"), Sym("mul"))),)
    assert final.fuel == 4
    assert final.stack == execute("nil 'mul cons 'dup cons").stack


def test_empty_brackets_give_nil():
    assert execute_mark("[ ]").stack == (NIL,)


def test_literals_still_push_values_inside_a_quotation():
    final = execute_mark("[ 1 add ]")
    assert final.stack == (Lst((Int(1), Sym("add"))),)


def test_a_built_quotation_runs():
    assert execute_mark("1 1 [ add ] i").stack == (Int(2),)
    assert execute_mark("nil 1 cons 0 cons [ 1 add ] map").stack == (
        Lst((Int(1), Int(2))),
    )


def test_combinators_inside_marks_are_named_not_run():
    """`i` with nothing to run would be illegal; quoted it is just a name."""
    final = execute_mark("[ i map ifte dip ]")
    assert final.stack == (Lst(tuple(Sym(n) for n in ("i", "map", "ifte", "dip"))),)


def test_marks_nest_and_close_innermost_first():
    final = execute_mark("[ dup [ mul ] ]")
    assert final.stack == (Lst((Sym("dup"), Lst((Sym("mul"),)))),)


def test_quote_construction_is_visible_in_the_state():
    config = execute_mark("1 [ dup")
    assert config.stack == (Int(1), MARK, Sym("dup"))
    assert format_stack(config) == "[ 1 MARK dup ]"


def test_close_is_refused_without_an_open_mark():
    assert CLOSE_MARK.guard(Config()) == "no open quotation"
    assert CLOSE_MARK.guard(Config((Int(1), NIL))) == "no open quotation"
    with pytest.raises(IllegalOpError):
        step(Config(), CLOSE_MARK)
    assert step_raw(CLOSE_MARK, (), 10, 0) == "no open quotation"


def test_close_only_collects_above_the_topmost_mark():
    config = Config((Int(1), MARK, Int(0), MARK, Sym("add")))
    assert CLOSE_MARK.apply(config) == (
        (Int(1), MARK, Int(0), Lst((Sym("add"),))),
        1,
    )


def test_every_quoted_op_is_legal_and_costs_one_under_a_mark():
    config = Config((MARK,))
    for op in OPS:
        assert op.guard(config) is None, op.name
        stack, cost = op.apply(config)
        assert cost == 1
        assert step_raw(op, config.stack, 100, 0) == (stack, cost)
        if op.group in ("literals", "symbols"):
            # already data: they run as usual
            assert (stack, cost) == op._apply(config), op.name
        else:
            assert stack == (MARK, Sym(op.name)), op.name
    assert all(legal_mask(config, MARK_OPS))


def test_a_mark_anywhere_on_the_stack_quotes():
    """The marker need not be on top: it is any top-level stack item."""
    config = Config((MARK, Int(1), Int(2)))
    assert OP_BY_NAME["add"].apply(config) == ((MARK, Int(1), Int(2), Sym("add")), 1)


def test_without_a_mark_nothing_changes():
    config = Config((Int(1), Int(2)))
    assert OP_BY_NAME["add"].apply(config) == ((Int(3),), 1)
    assert OP_BY_NAME["i"].guard(Config()) is not None
    assert legal_mask(config) == legal_mask(config, OPS)


def test_fuel_refusal_comes_before_quoting():
    spent = Config((MARK,), fuel=5, fuel_limit=5)
    for op in MARK_OPS:
        assert op.guard(spent) == "out of fuel", op.name
    assert step_raw(OP_BY_NAME["add"], (MARK,), 0, 0) == "out of fuel"


def test_legal_ops_and_mask_take_the_basis():
    empty = Config()
    assert OPEN_MARK in legal_ops(empty, MARK_OPS)
    assert CLOSE_MARK not in legal_ops(empty, MARK_OPS)
    assert len(legal_mask(empty, MARK_OPS)) == 28
    assert len(legal_mask(empty)) == 52


def test_parse_program_takes_the_basis():
    assert parse_program("[ ]", MARK_OPS) == (OPEN_MARK, CLOSE_MARK)
    with pytest.raises(MachineError):
        parse_program("[ ]")
    with pytest.raises(MachineError):
        parse_program("'add", MARK_OPS)


def test_parse_stack_reads_mark_only_under_mark_28():
    assert parse_stack("[ 1 MARK dup ]", basis="mark-28") == (Int(1), MARK, Sym("dup"))
    assert parse_stack("[ 1 MARK ]") == (Int(1), Sym("MARK"))
    assert parse_stack("[ MARK ]", basis="full-52") == (Sym("MARK"),)
    with pytest.raises(MachineError, match="unknown basis"):
        parse_stack("[ ]", basis="nope")
    config = execute_mark("0 [ 1 dup")
    assert parse_stack(format_stack(config), basis="mark-28") == config.stack


def test_cli_takes_a_basis(capsys):
    assert main(["--basis", "mark-28", "1 1 [ add ] i"]) == 0
    out = capsys.readouterr().out
    assert "/28" in out and "[ 2 ]" in out
    assert main(["[ ]"]) == 2  # default full-52 has no brackets
