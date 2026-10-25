"""Independent-reference checks of explicit sequence algorithm families."""

import random

import pytest

from experiments.pbe_scale.generate import READOUTS, STARTS
from experiments.pbe_scale.generate_families import (
    ALGORITHMS,
    _legal,
    _reference,
    generate_candidates,
)
from experiments.world_behaviours.verify import expand, load_macros
from wandering_light.stack_machine import value_key
from wandering_light.stack_pbe import run_program, to_stack, to_value

MACROS = load_macros()
EDGES = [
    [],
    [0],
    [-10],
    [10],
    [-10] * 12,
    [10] * 12,
    [1, 1, 2, 2, 1, 1, 1, 0],
    [10, -10] * 6,
    list(range(-6, 6)),
    [4, 4, 0, 3, -1, 3, 5, 5, 5, 4, 4, 0],
]


def _check(program, reference, args):
    namespace = {}
    exec(reference, namespace)
    want = namespace["f"](*args)
    result = run_program(expand(program.split(), MACROS), to_stack(args))
    assert not isinstance(result, str), (program, args, result)
    assert tuple(map(value_key, result[0])) == (value_key(to_value(want)),), (
        program,
        args,
        want,
        result,
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda algorithm: algorithm.name)
def test_explicit_algorithm(algorithm):
    assert len(expand(algorithm.program.split(), MACROS)) <= 200
    reference = _reference(STARTS[0], algorithm, None, READOUTS[0])
    for xs in EDGES:
        _check(algorithm.program, reference, (xs,))


def test_composed_views():
    rng = random.Random(392)
    rows = generate_candidates(400, seed=188)
    assert rows == generate_candidates(400, seed=188)
    assert len({row["id"] for row in rows}) == 400
    assert len({row["parameters"]["algorithm"] for row in rows}) >= 25
    for row in rows[::3]:
        for xs in [
            [],
            [10, -10] * 6,
            [rng.randint(-10, 10) for _ in range(rng.randrange(13))],
        ]:
            kind = row["input_kind"]
            args = (
                (xs,)
                if kind == "list"
                else (xs, rng.randint(-10, 10))
                if kind == "list_int"
                else (xs, [10, 1, -2, -10])
            )
            _check(row["program"], row["reference"], args)


@pytest.mark.parametrize(
    "name",
    [
        "running-maximum",
        "longest-increasing-run-so-far",
        "local-peaks",
        "prior-equal-counts",
    ],
)
def test_count_rejects_algorithms_with_data_independent_output_lengths(name):
    algorithm = next(algorithm for algorithm in ALGORITHMS if algorithm.name == name)
    count = next(readout for readout in READOUTS if readout.name == "count")
    assert not _legal(algorithm, None, count)


def test_count_keeps_data_dependent_run_and_record_collections():
    count = next(readout for readout in READOUTS if readout.name == "count")
    for algorithm in ALGORITHMS:
        if algorithm.name in {
            "run-lengths",
            "run-representatives",
            "record-highs",
            "record-lows",
        }:
            assert _legal(algorithm, None, count)
