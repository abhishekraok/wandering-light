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
    CORE_OPS,
    FIRST_ORDER_OPS,
    NIL,
    OP_BY_NAME,
    OPS,
    Bool,
    Config,
    IllegalOpError,
    Int,
    Lst,
    MachineError,
    Sym,
    format_value,
    legal_ops,
    parse_program,
    run,
    step,
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
