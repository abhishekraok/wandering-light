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
    UNQUOTED_GROUPS,
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
    has_open_mark,
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


def execute(source: str, config: Config | None = None, ops=OPS) -> Config:
    final = config if config is not None else Config()
    for reached in run(parse_program(source, ops), final):
        final = reached
    return final


def execute_marked(source: str, config: Config | None = None) -> Config:
    """`execute` under the mark-28 basis, where `[` and `]` are ops."""
    return execute(source, config, MARK_OPS)


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
# Structural quotation: the mark-28 basis
# ---------------------------------------------------------------------------


def test_mark_is_a_value_with_its_own_identity():
    assert Mark() == MARK
    assert value_key(MARK) == ("mark",)
    assert format_value(MARK) == "MARK"
    # the marker is not the symbol that prints the same way
    assert value_key(MARK) != value_key(Sym("MARK"))
    assert value_key(MARK) != value_key(NIL)
    assert format_stack(Config((Int(1), MARK))) == "[ 1 MARK ]"


def test_mark_has_no_fields():
    with pytest.raises(TypeError):
        Mark(1)


def test_the_bases_are_the_same_tuples_as_the_constants():
    assert len(MARK_OPS) == 28
    assert MARK_OPS[: len(CORE_OPS)] == CORE_OPS
    assert MARK_OPS[-2:] == (OPEN_MARK, CLOSE_MARK)
    assert (OPEN_MARK.name, CLOSE_MARK.name) == ("[", "]")
    assert {OPEN_MARK.group, CLOSE_MARK.group} == {"brackets"}
    assert len(OPS) == 52  # the full basis is untouched
    assert BASIS_OPS["executable-26"] is CORE_OPS
    assert BASIS_OPS["full-52"] is OPS
    assert BASIS_OPS["mark-28"] is MARK_OPS
    assert DEFAULT_BASIS == "executable-26"
    assert ops_for_basis() is CORE_OPS
    for name, ops in BASIS_OPS.items():
        assert ops_for_basis(name) is ops
    # the brackets are not in the full basis: no symbol names them either
    assert "[" not in OP_BY_NAME and "]" not in OP_BY_NAME


def test_unknown_basis_is_refused_with_the_choices():
    with pytest.raises(MachineError) as caught:
        ops_for_basis("executable-27")
    message = str(caught.value)
    assert message.startswith("unknown basis 'executable-27'; choose one of: ")
    for name in BASIS_OPS:
        assert name in message


def test_open_mark_is_always_legal_and_pushes_the_marker():
    for config in (Config(), Config((Int(1),)), Config((MARK,)), Config((NIL, MARK))):
        assert OPEN_MARK.guard(config) is None
        assert OPEN_MARK.apply(config) == ((*config.stack, MARK), 1)
    assert step(Config(), OPEN_MARK).stack == (MARK,)


def test_close_mark_needs_an_open_quotation():
    for config in (Config(), Config((Int(1), Sym("MARK"))), Config((Lst((MARK,)),))):
        assert CLOSE_MARK.guard(config) == "no open quotation"
        with pytest.raises(IllegalOpError, match="no open quotation"):
            step(config, CLOSE_MARK)
    assert CLOSE_MARK not in legal_ops(Config(), MARK_OPS)
    assert step_raw(CLOSE_MARK, (), 10, 0) == "no open quotation"


def test_empty_brackets_give_nil():
    assert execute_marked("[ ]").stack == (NIL,)
    assert execute_marked("[ ]").fuel == 2


def test_close_mark_collects_the_topmost_mark_only():
    config = Config((Int(1), MARK, Int(0), MARK, Sym("add"), Int(1)))
    after = step(config, CLOSE_MARK)
    assert after.stack == (Int(1), MARK, Int(0), Lst((Sym("add"), Int(1))))
    assert after.fuel == 1
    # and the next `]` collects the outer one
    assert step(after, CLOSE_MARK).stack == (
        Int(1),
        Lst((Int(0), Lst((Sym("add"), Int(1))))),
    )


def test_brackets_build_a_quotation_structurally():
    """Each quoted op costs one op, and the marker stays visible throughout."""
    trace = [format_stack(c) for c in run(parse_program("[ 1 add ]", MARK_OPS))]
    assert trace == ["[ MARK ]", "[ MARK 1 ]", "[ MARK 1 add ]", "[ (1 add) ]"]
    final = execute_marked("[ 1 add ]")
    assert final.stack == (Lst((Int(1), Sym("add"))),)
    assert final.fuel == 4
    # the symbol route needs a 'name cons pair per quoted op
    assert execute("nil 'add cons '1 cons").fuel == 5


def test_literals_are_values_inside_brackets_but_run_the_same():
    """`1` inside brackets pushes Int(1), not Sym("1"); both run to the same."""
    structural = execute_marked("[ 1 add ]").stack[0]
    symbolic = execute("nil 'add cons '1 cons").stack[0]
    assert format_value(structural) == format_value(symbolic) == "(1 add)"
    assert value_key(structural) != value_key(symbolic)
    assert execute_marked("1 [ 1 add ] i").stack == (Int(2),)
    assert execute("1 nil 'add cons '1 cons i").stack == (Int(2),)


def test_structural_quotations_run_under_every_combinator():
    assert execute_marked("nil 1 cons 0 cons 1 cons [ 1 add ] map").stack == (
        Lst((Int(2), Int(1), Int(2))),
    )
    assert execute_marked("nil 1 cons 0 cons 1 cons [ 0 eq? ] filter").stack == (
        Lst((Int(0),)),
    )
    assert execute_marked("nil 1 cons 1 cons 1 cons 0 [ add ] fold").stack == (Int(3),)
    assert execute_marked("1 1 0 [ add ] dip").stack == (Int(2), Int(0))
    assert execute_marked("0 1 [ lt? ] [ 1 ] [ 0 ] ifte").stack == (
        Int(0),
        Int(1),
        Int(1),
    )


def test_brackets_nest():
    assert execute_marked("[ [ add ] i ]").stack == (
        Lst((Lst((Sym("add"),)), Sym("i"))),
    )
    assert format_value(execute_marked("[ [ add ] i ]").stack[0]) == "((add) i)"
    assert execute_marked("1 1 [ [ add ] i ] i").stack == (Int(2),)
    assert execute_marked("[ [ ] [ [ ] ] ]").stack == (Lst((NIL, Lst((NIL,)))),)


def test_while_a_mark_is_open_executable_ops_are_quoted():
    """Every op but the literals, the symbols and the brackets pushes its name."""
    for config in (
        Config((MARK,)),
        Config((MARK, Int(1))),
        Config((Int(1), MARK, NIL)),
    ):
        assert has_open_mark(config.stack)
        for op in (*MARK_OPS, *OPS):
            assert op.guard(config) is None, op.name
            stack, cost = op.apply(config)
            assert cost == 1, op.name
            assert step_raw(op, config.stack, 10, 0) == (stack, cost), op.name
            if op.group in UNQUOTED_GROUPS:
                continue
            assert op.quoted_by(config.stack)
            assert stack == (*config.stack, Sym(op.name)), op.name
    assert {"literals", "symbols", "brackets"} == UNQUOTED_GROUPS


def test_quoting_covers_the_ops_that_run_code():
    """`i`, `dip` and the combinators cannot run with a mark open: they quote."""
    config = Config((MARK, Lst((Sym("add"),))))
    for name in ("i", "dip", "map", "filter", "fold", "ifte"):
        op = OP_BY_NAME[name]
        assert op.runs_code
        assert op.guard(config) is None
        assert op.apply(config) == ((*config.stack, Sym(name)), 1)


def test_literals_and_symbols_keep_running_inside_brackets():
    config = Config((MARK,))
    assert step(config, OP_BY_NAME["nil"]).stack == (MARK, NIL)
    assert step(config, OP_BY_NAME["true"]).stack == (MARK, Bool(True))
    assert step(config, OP_BY_NAME["0"]).stack == (MARK, Int(0))
    assert step(config, OP_BY_NAME["'add"]).stack == (MARK, Sym("add"))
    assert step(config, OPEN_MARK).stack == (MARK, MARK)


def test_fuel_refusal_comes_before_quoting():
    spent = Config((MARK,), fuel=3, fuel_limit=3)
    for op in MARK_OPS:
        assert op.guard(spent) == "out of fuel", op.name
    assert legal_ops(spent, MARK_OPS) == ()
    assert step_raw(OP_BY_NAME["add"], (MARK,), 0, 0) == "out of fuel"
    assert step_raw(OP_BY_NAME["add"], (MARK,), 1, 0) == ((MARK, Sym("add")), 1)


def test_without_an_open_mark_nothing_changes():
    """A `Sym("MARK")`, or a marker inside a list, does not open a quotation."""
    for stack in ((), (Int(1), Int(1)), (Sym("MARK"),), (Lst((MARK,)),)):
        config = Config(stack)
        assert not has_open_mark(stack)
        for op in OPS:
            assert not op.quoted_by(stack)
            if op.guard(config) is None:
                assert op.apply(config)[1] == 1 or op.runs_code
    assert step(Config((Int(1), Int(1))), OP_BY_NAME["add"]).stack == (Int(2),)
    assert OP_BY_NAME["add"].guard(Config((Sym("MARK"),))) is not None


def test_quoting_applies_inside_a_running_body_too():
    """step_raw is what run_body uses, so a mark seen there quotes as well."""
    stack = (Int(1), MARK)
    assert step_raw(OP_BY_NAME["add"], stack, 10, 0) == ((*stack, Sym("add")), 1)
    assert step_raw(OP_BY_NAME["dup"], stack, 10, 0) == ((*stack, Sym("dup")), 1)
    # `map` over a list whose element is a mark: `drop` is quoted, not run,
    # so the body leaves two values instead of none
    config = Config((Lst((MARK,)), Lst((Sym("drop"),))))
    reason = OP_BY_NAME["map"].guard(config)
    assert reason == "body must leave exactly one value, left 2"
    # `i` on a quotation sitting above a mark cannot happen: `i` is quoted
    assert OP_BY_NAME["i"].apply(Config((MARK, Lst((Sym("add"),))))) == (
        (MARK, Lst((Sym("add"),)), Sym("i")),
        1,
    )


def test_guard_and_apply_agree_with_marks_on_the_stack():
    configs = [
        Config((MARK,)),
        Config((Int(1), MARK, Sym("dup"))),
        Config((Lst((Sym("add"),)), MARK)),
        execute_marked("[ 1 add ]"),
        execute_marked("1 [ [ ] ]"),
    ]
    for config in configs:
        for op in (*MARK_OPS, *OPS):
            if op.guard(config) is None:
                op.apply(config)
                assert step(config, op).fuel == config.fuel + op.apply(config)[1]
            else:
                with pytest.raises(IllegalOpError):
                    step(config, op)


def test_legal_mask_and_legal_ops_take_the_basis():
    empty = Config()
    assert len(legal_mask(empty, MARK_OPS)) == 28
    assert len(legal_mask(empty)) == len(OPS)
    assert [op.name for op in legal_ops(empty, MARK_OPS)] == [
        "0",
        "1",
        "nil",
        "true",
        "false",
        "[",
    ]
    marked = Config((MARK,))
    assert all(legal_mask(marked, MARK_OPS))
    assert legal_ops(marked, MARK_OPS) == MARK_OPS
    assert legal_ops(marked, CORE_OPS) == CORE_OPS
    masked = [
        op.name
        for op, ok in zip(MARK_OPS, legal_mask(marked, MARK_OPS), strict=True)
        if ok
    ]
    assert masked == [op.name for op in legal_ops(marked, MARK_OPS)]


def test_parse_program_takes_the_basis():
    assert parse_program("[ add ]", MARK_OPS) == (
        OPEN_MARK,
        OP_BY_NAME["add"],
        CLOSE_MARK,
    )
    with pytest.raises(MachineError, match="unknown op '\\['"):
        parse_program("[ add ]")
    with pytest.raises(MachineError, match='unknown op "\'add"'):
        parse_program("'add", MARK_OPS)
    assert parse_program("1 add", CORE_OPS) == parse_program("1 add")


def test_parse_stack_reads_mark_only_under_the_mark_basis():
    assert parse_stack("[ MARK ]", basis="mark-28") == (MARK,)
    assert parse_stack("[ 1 MARK (add) ]", basis="mark-28") == (
        Int(1),
        MARK,
        Lst((Sym("add"),)),
    )
    assert parse_stack("[ (MARK) ]", basis="mark-28") == (Lst((MARK,)),)
    assert parse_stack("[ MARK ]") == (Sym("MARK"),)
    assert parse_stack("[ MARK ]", basis="executable-26") == (Sym("MARK"),)
    assert parse_stack("[ MARK ]", basis="full-52") == (Sym("MARK"),)
    with pytest.raises(MachineError, match="unknown basis"):
        parse_stack("[ 1 ]", basis="mark-29")
    # a given basis does not change anything else
    assert parse_stack("[ (1 0) 4 (add) ]", basis="mark-28") == parse_stack(
        "[ (1 0) 4 (add) ]"
    )


def test_a_marked_stack_round_trips_through_its_display_form():
    for source in ("[ 1 add", "[ [ ]", "1 [ dup ] [ [ add", "[ ]"):
        config = execute_marked(source)
        text = format_stack(config)
        assert parse_stack(text, basis="mark-28") == config.stack, text
        assert format_stack(Config(parse_stack(text, basis="mark-28"))) == text


def test_cli_takes_a_basis(capsys):
    assert main(["--basis", "mark-28", "1 [ 1 add ] i"]) == 0
    out = capsys.readouterr().out
    assert "[ 1 MARK 1 add ]" in out
    assert "[ 1 (1 add) ]" in out
    assert "/28" in out
    # 5 to build and run up to `i`, then 1 for `i` plus 2 for its body
    assert out.rstrip().endswith("[ 2 ]".ljust(34) + "  fuel   8  legal  9/28")
    # the default basis is the full one, where `[` is not an op
    assert main(["[ ]"]) == 2
    err = capsys.readouterr().err
    assert "unknown op '['" in err
    assert "known ops:" in err and "'add" in err
    assert main(["--basis", "mark-28", "]"]) == 1
    assert "ILLEGAL: no open quotation" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--basis", "mark-29", "[ ]"])
