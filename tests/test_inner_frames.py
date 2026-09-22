"""The frame machinery of inner-frames (#160).

The driver's own numbers are checked against the machine at run time, but the
cases that matter most are rare in the corpora -- a literal captured inside a
nested quotation appears in none of v4, world-v3 or the staged witnesses -- so
they are pinned here instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.inner_frames as inf
from wandering_light.stack_machine import Int, Lst, Sym, parse_stack
from wandering_light.stack_pbe import BASIS


def walk(program: str, stack: str):
    ops = program.split()
    root = parse_stack(stack, basis=BASIS)
    states = inf.replay(ops, root)
    assert states is not None, f"witness refused: {program}"
    return ops, states, inf.walk_one_example(ops, states, inf.attribute(ops))


def test_a_literal_under_a_mark_is_captured_as_its_value_not_its_name():
    """`1` under a mark pushes `Int(1)`; rebuilding it as `Sym("1")` runs the same
    way but stores the wrong thing, and a body that `uncons`es the quotation then
    meets a symbol where the machine put an integer."""
    _ops, states, out = walk("[ [ 1 ] uncons 1 add ] i", "")
    # The machine's own capture, for reference: the closed inner quotation holds
    # the VALUE `Int(1)`, not the symbol `1`.
    inner = states[4][-1]
    assert inner == Lst((Int(1),)), inner
    # The inner quotation is `uncons`ed rather than run, so under the strict
    # reading it has no consumer and its own ops get no frame (steps 1-3).
    assert [out["defined"][i] for i in (1, 2, 3)] == [False, False, False]
    assert out["reason"][1] == "other_op_first"
    # What matters is the OUTER body, whose virtual execution `uncons`es that
    # reconstructed quotation and then adds to what came out. Rebuilt as
    # `Sym("1")` the addition refuses; rebuilt as the machine's `Int(1)` it does
    # not, and steps 4-7 (`uncons`, `1`, `add`, `]`) all keep their frame.
    assert [out["defined"][i] for i in (0, 4, 5, 6, 7)] == [True] * 5, out["reason"]


def test_every_frame_equals_the_stack_the_machine_hands_the_body():
    """The non-circular check: record what `run_body` is actually called with."""
    for program, stack in (
        ("[ 1 add ] map", "( 1 2 3 )"),
        ("[ 0 lt? ] filter", "( 1 -2 3 )"),
        ("0 [ add ] fold", "( 1 2 3 )"),
        ("[ dup ] dip", "1 2"),
        ("[ dup add ] i", "5"),
        ("[ 0 lt? ] [ 0 swap sub ] [ ] ifte", "-4"),
    ):
        ops = program.split()
        root = parse_stack(stack, basis=BASIS)
        states = inf.replay(ops, root)
        assert states is not None, program
        out = inf.walk_one_example(ops, states, inf.attribute(ops))
        observed = inf.observed_frames(ops, root)
        checks = inf.check_against_machine(out["computed"], observed)
        assert checks["compared"] >= 1, (program, checks)
        assert checks["mismatch"] == 0, (program, checks)


def test_captures_push_in_stack_order_not_reversed():
    """`x [ lt? ] swap cons filter` runs the body on `(item, x)`.

    With two captures the machine builds `[ x y body ]`, so the body sees `x`
    then `y`. Reversing is a no-op at one capture, which is why only the
    two-capture case can tell the orders apart.
    """
    below = (Lst((Int(1), Int(2))), Int(7))
    assert inf.frame_with_captures("filter", below, 0, 1) == (Int(1), Int(7))
    deeper = (Lst((Int(1), Int(2))), Int(7), Int(9))
    assert inf.frame_with_captures("filter", deeper, 0, 2) == (Int(1), Int(7), Int(9))
    # And the machine agrees, on a witness that really uses the idiom.
    ops = ["[", "lt?", "]", "swap", "cons", "filter"]
    root = parse_stack("( 1 5 9 ) 4", basis=BASIS)
    states = inf.replay(ops, root)
    assert states is not None
    out = inf.walk_one_example(ops, states, inf.attribute(ops), capture_aware=True)
    checks = inf.check_against_machine(out["computed"], inf.observed_frames(ops, root))
    assert (checks["compared"], checks["mismatch"], checks["captures_1"]) == (1, 0, 1)


def test_the_closure_idiom_has_no_tag_under_the_strict_reading():
    """#160 as written says the consumer is the next token, so a closure has none."""
    ops = ["[", "lt?", "]", "swap", "cons", "filter"]
    root = parse_stack("( 1 5 9 ) 4", basis=BASIS)
    states = inf.replay(ops, root)
    strict = inf.walk_one_example(ops, states, inf.attribute(ops))
    assert strict["reason"][0] == "other_op_first"
    assert not strict["defined"][0]
    aware = inf.walk_one_example(ops, states, inf.attribute(ops), capture_aware=True)
    assert aware["defined"][0]


def test_a_quotation_with_no_consumer_is_not_labelled_a_combinator():
    """Ops inside an unconsumed quotation must not be attributed to `i`."""
    _ops, _states, out = walk("[ dup add ] drop", "5")
    assert out["tag"][0] == "other_op_first"
    assert out["tag"][1] == inf.NO_TAG
    assert inf.NO_TAG not in inf.COMBINATORS


def test_attribution_reaches_nested_quotations():
    """Value-tracking cannot attribute a nested quotation; bracket matching can."""
    ops = ["[", "[", "1", "add", "]", "map", "]", "map"]
    tags = inf.attribute(ops)
    assert tags[0][0] == "map"  # outer, consumed by the trailing `map`
    assert tags[1][0] == "map"  # inner, consumed inside the outer body
    root = parse_stack("( ( 1 2 ) ( 3 ) )", basis=BASIS)
    states = inf.replay(ops, root)
    assert states is not None
    out = inf.walk_one_example(ops, states, tags)
    assert all(out["defined"].values()), out["reason"]


def test_ifte_roles_take_the_stack_below_all_three_quotations():
    below = (Int(4), Lst((Sym("a"),)), Lst((Sym("b"),)))
    assert inf.frame_of("ifte", below, 0) == below
    assert inf.frame_of("ifte", below, 1) == below[:-1]
    assert inf.frame_of("ifte", below, 2) == below[:-2]


def test_an_empty_list_has_a_legal_tag_but_no_data():
    assert inf.frame_of("map", (Lst(()),), 0) == "empty"
    assert "map" in inf.legal_tags((Lst(()),))
    assert inf.frame_of("map", (Int(1),), 0).startswith("map needs a list")
