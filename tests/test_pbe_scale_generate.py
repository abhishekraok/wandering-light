"""Semantic checks for task-grounded PBE generation."""

from __future__ import annotations

import itertools
import random

import pytest

from experiments.pbe_scale.generate import (
    READOUTS,
    STAGES,
    STARTS,
    _branched,
    _pipeline,
    generate_candidates,
)
from experiments.world_behaviours.verify import expand, load_macros
from wandering_light.stack_machine import value_key
from wandering_light.stack_pbe import run_program, to_stack, to_value

MACROS = load_macros()
BY_NAME = {operation.name: operation for operation in STAGES}
EDGE_LISTS = [
    [],
    [0],
    [-9],
    [9],
    [0] * 12,
    [-9] * 12,
    [9] * 12,
    list(range(-6, 6)),
    [9, -9] * 6,
    [3, 1, 3, -2, 0, -2],
]


def _check(row: dict, args: tuple) -> None:
    namespace: dict = {}
    exec(row["reference"], namespace)
    expected = namespace["f"](*args)
    assert type(expected) in (int, list)
    assert not isinstance(expected, list) or all(
        type(value) is int for value in expected
    )
    result = run_program(expand(row["program"].split(), MACROS), to_stack(args))
    assert not isinstance(result, str), (row["template_id"], args, result)
    assert tuple(map(value_key, result[0])) == (value_key(to_value(expected)),), (
        row["template_id"],
        args,
        expected,
        result,
    )


@pytest.mark.parametrize(
    "operation", [*STAGES, *READOUTS], ids=lambda operation: operation.name
)
def test_semantic_operations(operation):
    row = _pipeline(STARTS[0], (operation,), READOUTS[0])
    for xs in EDGE_LISTS:
        _check(row, (xs,))


@pytest.mark.parametrize("start", STARTS, ids=lambda start: start.name)
def test_argument_combinations(start):
    row = _pipeline(
        start, (BY_NAME["running-sum"], BY_NAME["neighbor-difference"]), READOUTS[9]
    )
    for xs in EDGE_LISTS:
        args = (
            (xs,)
            if start.category == "list"
            else (xs, -4)
            if start.category == "list_int"
            else (xs, [1, -4, 3])
        )
        _check(row, args)


@pytest.mark.parametrize("mode", range(4))
def test_original_input_interactions(mode):
    row = _branched((BY_NAME["running-sum"], BY_NAME["nonzero"]), READOUTS[9], mode)
    for xs in EDGE_LISTS:
        _check(row, (xs,))


def test_deterministic_structural_diversity():
    rows = generate_candidates(500, seed=188)
    assert rows == generate_candidates(500, seed=188)
    assert rows != generate_candidates(500, seed=189)
    assert len({row["id"] for row in rows}) == 500
    assert len({row["template_id"] for row in rows}) == 500
    assert {row["input_kind"] for row in rows} == {"list", "list_int", "two_lists"}
    assert len({row["family"] for row in rows}) >= 4
    assert len({row["template_id"].split("/")[1] for row in rows}) >= 18
    for row in rows:
        assert 50 <= len(expand(row["program"].split(), MACROS)) <= 200
        assert "parameters" not in row


def test_composed_cross_section():
    rng = random.Random(919)
    rows = generate_candidates(300, seed=382)
    for row in itertools.islice(rows, 0, None, 3):
        for xs in [
            *EDGE_LISTS[:5],
            [rng.randint(-9, 9) for _ in range(rng.randrange(13))],
        ]:
            args = (
                (xs,)
                if row["input_kind"] == "list"
                else (xs, rng.randint(-10, 10))
                if row["input_kind"] == "list_int"
                else (xs, [-9, 0, 4, 9])
            )
            _check(row, args)


def test_invalid_generation_arguments():
    with pytest.raises(ValueError):
        generate_candidates(-1)
    with pytest.raises(ValueError):
        generate_candidates(1, min_ops=80, max_ops=50)
