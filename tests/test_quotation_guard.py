"""The guard must never cost a witness, and must still catch real nonsense."""

from __future__ import annotations

import pytest

from wandering_light.quotation_guard import (
    FIXED_SUPPLY,
    blocked,
    body_effect,
    open_body,
)
from wandering_light.stack_machine import Bool, Int, Lst, Sym, Value, parse_stack
from wandering_light.stack_pbe import CATALOGUE

LITERALS: dict[str, Value] = {
    "0": Int(0),
    "1": Int(1),
    "nil": Lst(()),
    "true": Bool(True),
    "false": Bool(False),
}


def bodies(program: str) -> list[tuple[Value, ...]]:
    """Every innermost open body the machine passes through running `program`.

    Capture under a ``MARK`` does not consult the stack -- a literal pushes its
    value, anything else pushes its own name -- so the sequence of open bodies is
    a function of the program text alone, and no input data is needed to walk it.
    """
    frames: list[list[Value]] = []
    seen: list[tuple[Value, ...]] = []
    for token in program.split():
        if token == "[":
            frames.append([])
        elif token == "]":
            closed = frames.pop()
            if frames:
                frames[-1].append(Lst(tuple(closed)))
        elif frames:
            frames[-1].append(LITERALS.get(token) or Sym(token))
        if frames:
            seen.append(tuple(frames[-1]))
    return seen


def body_of(text: str) -> tuple[Value, ...]:
    found = open_body(parse_stack(text, basis="mark-28"))
    assert found is not None
    return found[0]


def test_the_empty_body_survives():
    # `abs_each` uses `[ ]` as its ifte else-branch, so "non-trivial" is wrong.
    assert body_effect(()) == (True, 0)
    assert blocked(parse_stack("[ MARK ]", basis="mark-28"), remaining=3) is None


def test_abs_each_witness_is_never_guarded():
    witness = "[ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] map"
    for body in bodies(witness):
        runnable, reach = body_effect(body)
        assert runnable, body
        assert reach <= FIXED_SUPPLY, body


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_no_catalogue_witness_body_has_a_type_error(behaviour):
    """The gate the issue pre-registers: no authored witness is ever unreachable.

    Only the type half is checkable from program text -- ``reach`` is judged
    against the stack under the mark, which the text alone does not fix.
    `test_a_cons_widened_closure_survives` covers the case that half nearly got
    wrong.
    """
    for program in behaviour.witnesses:
        for body in bodies(program):
            assert body_effect(body)[0], f"{behaviour.name}: {program}"


def test_a_cons_widened_closure_survives():
    """85 of v4's 220 witnesses build a closure, and reach must allow for it.

    `replace_value` is `[ [ drop eq? ] ... ] swap cons swap cons map`: the body
    reaches below what `map` hands it, and the shortfall is supplied by the
    `cons`es that
    come *after* the quotation closes, not by `map`. Bounding reach by what the
    consumer hands over would have pruned the witness.
    """
    body = body_of("[ MARK add add add ]")
    assert body_effect(body) == (True, 4)
    stack = parse_stack("[ MARK add add add ]", basis="mark-28")
    assert blocked(stack, remaining=1) == "reach"  # no budget left to widen it
    assert blocked(stack, remaining=2) is None  # one cons away from usable


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("[ MARK 0 null? ]", "type"),  # null? wants a list, sees an int
        ("[ MARK nil add ]", "type"),  # add wants two ints, sees a list
        ("[ MARK 0 i ]", "type"),  # i wants a quotation, sees an int
        ("[ MARK true 1 lt? ]", "type"),  # lt? wants ints, sees a bool
        ("[ MARK 1 1 cat ]", "type"),  # cat wants two quotations
        ("[ MARK add add add add add add ]", "reach"),  # deeper than reachable
    ],
)
def test_nonsense_bodies_are_caught(text, reason):
    assert blocked(parse_stack(text, basis="mark-28"), remaining=3) == reason


def test_a_body_every_combinator_rejects():
    """`0 null?` cannot run under map, filter, fold, i, dip or ifte."""
    body = body_of("[ MARK 0 null? ]")
    assert body_effect(body)[0] is False


def test_the_checks_are_monotone_under_extension():
    """A dead body stays dead however it is extended -- what makes this a prune."""
    dead = body_of("[ MARK 0 null? ]")
    for extra in ("drop", "add", "dup", "i", "map", "0", "nil"):
        runnable, _ = body_effect((*dead, LITERALS.get(extra) or Sym(extra)))
        assert runnable is False


def test_reach_only_grows():
    reaches = [body_effect(body)[1] for body in bodies("[ add add add add ]")]
    assert reaches == sorted(reaches)


def test_a_closed_inner_quotation_is_not_judged():
    """`[ [ 0 null? ] drop ]` builds an unrunnable body and throws it away."""
    assert body_effect(bodies("[ [ 0 null? ] drop ]")[-1])[0] is True


def test_off_mode_never_blocks():
    assert blocked(parse_stack("[ MARK 0 null? ]", basis="mark-28"), 0, "off") is None


def test_close_mode_needs_a_bracket_it_cannot_afford():
    stack = parse_stack("[ MARK 0 ]", basis="mark-28")
    assert blocked(stack, remaining=0, mode="close") == "close"
    assert blocked(stack, remaining=1, mode="close") is None


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="guard mode"):
        blocked(parse_stack("[ MARK ]", basis="mark-28"), 1, "sideways")
