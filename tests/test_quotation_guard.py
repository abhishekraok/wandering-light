"""The search-side quotation guard: sound on witnesses, monotone, and exact on
the cases the spec names."""

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
    FALSE,
    MARK,
    MARK_OPS,
    NIL,
    ONE,
    TRUE,
    ZERO,
    Int,
    Lst,
    Sym,
    run_body,
    step_raw,
)
from wandering_light.stack_pbe import CATALOGUE, OP_BY_NAME, run_program, to_stack

LITERALS = {"0": ZERO, "1": ONE, "nil": NIL, "true": TRUE, "false": FALSE}


def body(text: str) -> tuple:
    """A body as the machine accumulates it: literals as values, ops as Syms."""
    return tuple(LITERALS.get(token) or Sym(token) for token in text.split())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", (True, 0)),
        ("add add add", (True, 4)),
        ("0 null?", (False, 0)),
        ("true 1 add", (False, 0)),
        ("nil 0 cons uncons add", (True, 0)),  # uncons yields two unknowns
        ("0 1 cat", (False, 0)),
        ("0 i", (False, 0)),
        ("i 0 null?", (True, 1)),  # nothing is judged after code runs
        ("map add add add", (True, 2)),  # reach stops when code runs
        ("dup add", (True, 1)),
        ("0 nil swap cons", (True, 0)),
        ("nil 0 swap cons", (False, 0)),
        ("0 nil over cons drop null?", (False, 0)),  # over copies the 0
        ("1 eq? 0 lt?", (False, 1)),
        ("1 quote 0 swap cons", (False, 0)),
        ("nil nil nil 0 fold", (False, 0)),
        ("nil 0 nil fold", (True, 0)),
        ("nil nil 0 ifte", (False, 0)),
        ("drop drop 0", (True, 2)),
    ],
)
def test_body_effect_cases(text, expected):
    assert body_effect(body(text)) == expected


def test_closed_nested_quotation_is_a_value():
    stack = (MARK, MARK, ZERO, Sym("null?"))
    assert blocked(stack, 10) == "type"
    closed = (MARK, Lst((ZERO, Sym("null?"))), Sym("drop"))
    assert blocked(closed, 10) is None
    assert body_effect(closed[1:]) == (True, 0)


def test_open_body():
    assert open_body(()) is None
    assert open_body((ZERO, NIL)) is None
    assert open_body((ZERO, MARK)) == ((), 1)
    assert open_body((ZERO, MARK, ONE, MARK, Sym("add"))) == ((Sym("add"),), 3)


def test_blocked_modes():
    unrunnable = (ZERO, MARK, ZERO, Sym("null?"))
    deep = (MARK, MARK, MARK)
    assert MODES == ("off", "body", "close", "both")
    assert FIXED_SUPPLY == 2
    for mode in MODES:
        assert blocked((ZERO, NIL), 0, mode) is None
    assert blocked(unrunnable, 5, "off") is None
    assert blocked(unrunnable, 5, "close") is None
    assert blocked(unrunnable, 5, "body") == "type"
    assert blocked(unrunnable, 5, "both") == "type"
    assert blocked(deep, 2, "close") == "close"
    assert blocked(deep, 2, "both") == "close"
    assert blocked(deep, 2, "body") is None
    assert blocked(deep, 3, "both") is None
    # close is checked first under both
    assert blocked((MARK, ZERO, Sym("null?")), 0, "both") == "close"
    # reach is never pruned on
    assert blocked((MARK, *body("add add add add add")), 1) is None
    with pytest.raises(ValueError, match="guard mode"):
        blocked((), 0, "sideways")
    with pytest.raises(ValueError, match="guard mode"):
        blocked((MARK,), 0, "")


def _random_body(rng: random.Random, length: int) -> tuple:
    names = [op.name for op in MARK_OPS if op.name not in ("[", "]")]
    out = []
    for _ in range(length):
        name = rng.choice(names)
        out.append(LITERALS.get(name) or Sym(name))
    return tuple(out)


def _random_value(rng: random.Random, depth: int = 0):
    kind = rng.randrange(4 if depth < 2 else 3)
    if kind == 0:
        return Int(rng.randrange(-3, 4))
    if kind == 1:
        return rng.choice((TRUE, FALSE))
    if kind == 2:
        return Lst(tuple(Int(rng.randrange(-3, 4)) for _ in range(rng.randrange(3))))
    return Lst(tuple(_random_value(rng, depth + 1) for _ in range(rng.randrange(3))))


def test_monotone_under_extension():
    rng = random.Random(0)
    for _ in range(2000):
        items = _random_body(rng, rng.randrange(1, 12))
        previous = body_effect(())
        for cut in range(1, len(items) + 1):
            current = body_effect(items[:cut])
            assert current[1] >= previous[1]
            assert previous[0] or not current[0]
            previous = current


def test_unrunnable_bodies_fail_on_every_stack_tried():
    """Soundness, sampled: an unrunnable body never runs successfully."""
    rng = random.Random(1)
    checked = 0
    for _ in range(3000):
        items = _random_body(rng, rng.randrange(1, 8))
        if body_effect(items)[0]:
            continue
        checked += 1
        for _ in range(20):
            stack = tuple(_random_value(rng) for _ in range(rng.randrange(6)))
            assert isinstance(run_body(items, stack, 1000), str), (items, stack)
    assert checked > 100


def _states(program: list[str], stack: tuple):
    """Each intermediate stack of a witness, with the ops left after it."""
    yield stack, len(program)
    for index, name in enumerate(program):
        outcome = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
        assert not isinstance(outcome, str), (program, index, outcome)
        stack = outcome[0]
        yield stack, len(program) - index - 1


def test_no_catalogue_witness_body_is_ever_blocked():
    rng = random.Random(2)
    witnesses = 0
    opened = 0
    for behaviour in CATALOGUE:
        stack = to_stack(behaviour.sample(rng))
        for witness in behaviour.programs:
            program = list(witness)
            assert not isinstance(run_program(program, stack), str)
            witnesses += 1
            for state, remaining in _states(program, stack):
                if open_body(state) is not None:
                    opened += 1
                assert blocked(state, remaining) is None, (
                    behaviour.name,
                    " ".join(program),
                    state,
                )
    assert witnesses >= len(CATALOGUE)
    assert opened > 1000
