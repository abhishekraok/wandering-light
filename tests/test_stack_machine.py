import json
import os
import subprocess
import sys

import pytest

from wandering_light.stack_machine import (
    NIL,
    OP_BY_NAME,
    OPS,
    Bool,
    Config,
    IllegalOpError,
    Int,
    Lst,
    MachineError,
    format_value,
    legal_mask,
    legal_ops,
    parse_program,
    run,
    same,
    step,
    value_key,
)

EXPECTED_OPS = {
    "literals": ["0", "1", "nil", "true", "false"],
    "plumbing": ["dup", "swap", "drop", "over"],
    "structure": ["cons", "uncons", "null?"],
    "arithmetic": ["add", "sub", "mul", "div"],
    "predicates": ["eq?", "lt?"],
}


def execute(source: str, config: Config | None = None) -> Config:
    """Run a program to completion and return the final config."""
    final = config if config is not None else Config()
    for step_config in run(parse_program(source), final):
        final = step_config
    return final


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------


def test_basis_is_exactly_the_named_ops():
    by_group: dict[str, list[str]] = {}
    for op in OPS:
        by_group.setdefault(op.group, []).append(op.name)
    assert by_group == EXPECTED_OPS
    assert len(OPS) == 18


def test_op_names_are_unique_and_order_is_stable():
    names = [op.name for op in OPS]
    assert len(set(names)) == len(names)
    assert list(OP_BY_NAME) == names


# ---------------------------------------------------------------------------
# Structural identity — the reason #42 exists
# ---------------------------------------------------------------------------


def test_bool_is_not_an_int():
    assert Int(1) != Bool(True)
    assert value_key(Int(1)) != value_key(Bool(True))
    assert not same(Int(1), Bool(True))
    assert value_key(Int(0)) != value_key(Bool(False))


def test_bool_cannot_be_smuggled_into_an_int():
    with pytest.raises(MachineError):
        Int(True)
    with pytest.raises(MachineError):
        Bool(1)


def test_nil_is_the_empty_list():
    assert same(NIL, Lst(()))
    assert value_key(NIL) == ("list", ())


def test_keys_are_structural_not_referential():
    left = Lst((Int(1), Lst((Bool(True), NIL))))
    right = Lst((Int(1), Lst((Bool(True), Lst(())))))
    assert left is not right
    assert value_key(left) == value_key(right)


def test_key_holds_only_plain_immutable_scalars():
    key = value_key(Lst((Int(7), Bool(False), NIL)))

    def check(node):
        assert isinstance(node, tuple), node
        for part in node:
            if isinstance(part, tuple):
                check(part)
            else:
                assert type(part) in (str, int, bool), part

    check(key)


def test_dataclass_equality_agrees_with_the_key():
    """There must be exactly one notion of sameness, not two that drift."""
    values = [
        Int(0),
        Int(1),
        Int(-1),
        Bool(True),
        Bool(False),
        NIL,
        Lst((Int(1),)),
        Lst((Bool(True),)),
        Lst((Lst((Int(1),)),)),
    ]
    for left in values:
        for right in values:
            assert (left == right) == same(left, right)


def test_config_key_ignores_fuel_but_not_the_stack():
    stack = (Int(1), NIL)
    assert Config(stack, fuel=0).key == Config(stack, fuel=99).key
    assert Config(stack).key != Config((NIL, Int(1))).key


def test_big_integers_are_exact():
    final = execute("1 1 add dup mul dup mul dup mul")
    assert final.stack == (Int(((2**2) ** 2) ** 2),)
    assert value_key(Int(2**200)) == ("int", 2**200)


@pytest.mark.parametrize("seed", ["0", "1", "12345"])
def test_keys_are_identical_across_pythonhashseed(seed):
    """State identity must not depend on hash randomization.

    Every value-identity bug in this repo (#26, #31, #35, #37) came from an
    identity that varied with the process. Compute a key in a fresh interpreter
    under a different seed and require the serialized bytes to match exactly.
    """
    script = """
import json, sys
from wandering_light.stack_machine import Config, parse_program, run
config = Config()
for config in run(parse_program(sys.argv[1]), config):
    pass
print(json.dumps(config.key))
"""
    program = "nil 1 cons true cons nil cons dup 0 1 add eq?"
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    result = subprocess.run(
        [sys.executable, "-c", script, program],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    in_process = json.dumps(execute(program).key)
    assert result.stdout.strip() == in_process


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_only_literals_are_legal_on_an_empty_stack():
    assert [op.name for op in legal_ops(Config())] == EXPECTED_OPS["literals"]


def test_every_non_literal_op_refuses_an_empty_stack_with_a_reason():
    for op in OPS:
        reason = op.guard(Config())
        if op.group == "literals":
            assert reason is None
        else:
            assert isinstance(reason, str) and reason


def test_legal_mask_lines_up_with_ops():
    config = Config((NIL, Int(1)))
    mask = legal_mask(config)
    assert len(mask) == len(OPS)
    masked = [op.name for op, ok in zip(OPS, mask, strict=True) if ok]
    assert masked == [op.name for op in legal_ops(config)]


def test_guard_refuses_division_by_zero():
    config = Config((Int(1), Int(0)))
    assert OP_BY_NAME["div"].guard(config) == "division by zero"
    assert OP_BY_NAME["div"].guard(Config((Int(1), Int(2)))) is None


def test_guard_refuses_uncons_on_an_empty_list():
    assert OP_BY_NAME["uncons"].guard(Config((NIL,))) == "list is empty"
    assert OP_BY_NAME["uncons"].guard(Config((Lst((Int(1),)),))) is None


def test_guard_refuses_type_confusion():
    assert OP_BY_NAME["uncons"].guard(Config((Int(1),))) == "top is not a list"
    assert OP_BY_NAME["null?"].guard(Config((Bool(True),))) == "top is not a list"
    assert OP_BY_NAME["cons"].guard(Config((Int(1), Int(2)))) == (
        "second value is not a list"
    )
    reason = "top two values are not both integers"
    assert OP_BY_NAME["add"].guard(Config((NIL, Int(1)))) == reason
    assert OP_BY_NAME["lt?"].guard(Config((Bool(True), Int(1)))) == reason


def test_booleans_are_not_arithmetic_operands():
    """A guard, not a coercion: `true 1 add` must be an absent transition."""
    assert OP_BY_NAME["add"].guard(Config((Bool(True), Int(1)))) is not None


def test_eq_accepts_any_two_values():
    assert OP_BY_NAME["eq?"].guard(Config((Bool(True), NIL))) is None
    assert execute("true nil eq?").stack == (Bool(False),)
    assert execute("nil nil eq?").stack == (Bool(True),)
    assert execute("1 1 eq?").stack == (Bool(True),)
    assert execute("1 true eq?").stack == (Bool(False),)


def test_applying_a_guarded_op_raises_rather_than_returning_junk():
    with pytest.raises(IllegalOpError, match="division by zero"):
        step(Config((Int(1), Int(0))), OP_BY_NAME["div"])


def test_fuel_limit_travels_through_the_guard_channel():
    config = Config(fuel_limit=2)
    config = step(config, OP_BY_NAME["1"])
    config = step(config, OP_BY_NAME["1"])
    assert config.fuel == 2
    assert legal_ops(config) == ()
    assert OP_BY_NAME["add"].guard(config) == "out of fuel"


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


def test_literals_push():
    assert execute("0 1 nil true false").stack == (
        Int(0),
        Int(1),
        NIL,
        Bool(True),
        Bool(False),
    )


def test_plumbing():
    assert execute("0 1 dup").stack == (Int(0), Int(1), Int(1))
    assert execute("0 1 swap").stack == (Int(1), Int(0))
    assert execute("0 1 drop").stack == (Int(0),)
    assert execute("0 1 over").stack == (Int(0), Int(1), Int(0))


def test_cons_takes_the_element_on_top_and_the_list_second():
    assert execute("nil 1 cons").stack == (Lst((Int(1),)),)
    assert execute("nil 1 cons 0 cons").stack == (Lst((Int(0), Int(1))),)
    assert execute("nil nil cons").stack == (Lst((NIL,)),)


def test_uncons_puts_head_on_top_and_tail_below():
    final = execute("nil 1 cons 0 cons uncons")
    assert final.stack == (Lst((Int(1),)), Int(0))


@pytest.mark.parametrize(
    "build", ["nil", "nil 1 cons", "nil 1 cons 0 cons", "nil nil cons true cons"]
)
def test_cons_and_uncons_are_inverses(build):
    built = execute(build)
    if not built.stack[-1].items:
        pytest.skip("uncons is guarded on the empty list")
    assert execute("uncons cons", built).stack == built.stack


def test_null_predicate():
    assert execute("nil null?").stack == (Bool(True),)
    assert execute("nil 1 cons null?").stack == (Bool(False),)


def test_arithmetic():
    assert execute("1 1 add").stack == (Int(2),)
    assert execute("1 1 add 1 sub").stack == (Int(1),)
    assert execute("1 1 add dup mul").stack == (Int(4),)
    assert execute("1 1 add 1 1 add add 1 1 add div").stack == (Int(2),)


def test_operand_order():
    """`a b sub` is `a - b`; `a b lt?` is `a < b`."""
    assert execute("1 1 add 1 sub").stack == (Int(1),)
    assert execute("0 1 lt?").stack == (Bool(True),)
    assert execute("1 0 lt?").stack == (Bool(False),)


def test_div_floors():
    """Python floor semantics, chosen so `div` matches the host language."""
    seven = Config((Int(7), Int(2)))
    assert step(seven, OP_BY_NAME["div"]).stack == (Int(3),)
    negative = Config((Int(-7), Int(2)))
    assert step(negative, OP_BY_NAME["div"]).stack == (Int(-4),)


def test_fuel_counts_one_per_op():
    assert execute("nil 1 cons uncons").fuel == 4
    assert execute("").fuel == 0


def test_nested_lists():
    final = execute("nil nil 1 cons cons")
    assert final.stack == (Lst((Lst((Int(1),)),)),)
    assert format_value(final.stack[0]) == "((1))"


def test_format_is_display_only_and_unambiguous():
    assert format_value(Int(1)) == "1"
    assert format_value(Bool(True)) == "true"
    assert format_value(NIL) == "nil"
    assert format_value(Lst((Int(1), Bool(False), NIL))) == "(1 false nil)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "wandering_light.stack_machine", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_traces_each_op():
    result = cli("nil 1 cons uncons")
    assert result.returncode == 0
    lines = result.stdout.strip().splitlines()
    body = [line for line in lines if f"/{len(OPS)}" in line]
    assert len(body) == 5  # (start) plus one line per op
    assert "fuel   4" in body[-1]
    assert f"/{len(OPS)}" in body[-1]
    assert "[ nil 1 ]" in body[-1]


def test_cli_reports_a_refusal_and_exits_nonzero():
    result = cli("1 0 div")
    assert result.returncode == 1
    assert "ILLEGAL: division by zero" in result.stdout


def test_cli_rejects_an_unknown_op():
    result = cli("nil frobnicate")
    assert result.returncode == 2
    assert "unknown op 'frobnicate'" in result.stderr
