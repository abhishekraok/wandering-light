"""Every macro in the canonical library has at least one executed case (#124)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from wandering_light.stack_machine import Bool, Int, Lst, value_key
from wandering_light.stack_pbe import run_program, to_stack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "world_behaviours"))
from verify import expand, load_macros  # noqa: E402

MACROS = load_macros(ROOT / "experiments" / "world_behaviours" / "macros.txt")

T, F = True, False
Z8 = [0] * 8

# (program in macro form, arguments deepest first, expected stack deepest first)
CASES = [
    # constants
    ("2", (), [2]),
    ("3", (), [3]),
    ("4", (), [4]),
    ("5", (), [5]),
    ("6", (), [6]),
    ("7", (), [7]),
    ("8", (), [8]),
    ("9", (), [9]),
    ("10", (), [10]),
    ("16", (), [16]),
    ("32", (), [32]),
    ("100", (), [100]),
    ("1000", (), [1000]),
    ("-1", (), [-1]),
    # scalars
    ("neg", (5,), [-5]),
    ("inc", (5,), [6]),
    ("dec", (5,), [4]),
    ("abs", (-3,), [3]),
    ("abs", (3,), [3]),
    ("sq", (4,), [16]),
    ("max2", (3, 9), [9]),
    ("max2", (9, 3), [9]),
    ("min2", (3, 9), [3]),
    ("mod", (17, 5), [2]),
    ("mod", (-7, 3), [2]),
    ("true not", (), [F]),
    ("false not", (), [T]),
    ("true bool>int", (), [1]),
    ("false bool>int", (), [0]),
    ("gt?", (3, 2), [T]),
    ("le?", (2, 2), [T]),
    ("le?", (3, 2), [F]),
    ("ge?", (1, 2), [F]),
    ("even?", (4,), [T]),
    ("odd?", (4,), [F]),
    ("divides?", (6, 3), [T]),
    ("gcd", (832040, 514229), [1]),
    ("gcd", (-4, -6), [2]),
    ("isqrt", (0,), [0]),
    ("isqrt", (-9,), [0]),
    ("isqrt", (999999,), [999]),
    ("pack5 unpack5", (1, 2, 3, 4, 5), [1, 2, 3, 4, 5]),
    ("pack5 second", (1, 2, 3, 4, 5), [2]),
    ("pack5 third", (1, 2, 3, 4, 5), [3]),
    ("pack5 fourth", (1, 2, 3, 4, 5), [4]),
    ("pack5 fifth", (1, 2, 3, 4, 5), [5]),
    ("flatten", ([[1, 2], [], [3]],), [[1, 2, 3]]),
    ("inversions", ([3, 1, 2],), [2]),
    ("inversions", ([],), [0]),
    ("decimal-digits", (0,), [[0]]),
    ("decimal-digits", (120,), [[1, 2, 0]]),
    ("from-digits", ([1, 2, 0],), [120]),
    ("palindrome?", (12321,), [T]),
    ("palindrome?", (-121,), [F]),
    ("bitlen", (0,), [0]),
    ("bitlen", (5,), [3]),
    ("log2", (8,), [3]),
    ("pow2", (10,), [1024]),
    ("divides?", (7, 3), [F]),
    ("between", (5, 1, 10), [T]),
    ("between", (11, 1, 10), [F]),
    ("divides-both", (3, 6, 9), [T]),
    ("divides-both", (3, 6, 10), [F]),
    ("gcd", (48, 18), [6]),
    ("pow", (2, 10), [1024]),
    ("pow", (3, 0), [1]),
    ("sum-multiples", (10, 3), [18]),
    ("leap?", (2000,), [T]),
    ("leap?", (1900,), [F]),
    ("leap?", (2024,), [T]),
    # stack shuffles
    ("rot", (1, 2, 3), [2, 3, 1]),
    ("-rot", (1, 2, 3), [3, 1, 2]),
    ("[ inc ] dip2", (1, 2, 3), [2, 2, 3]),
    ("[ inc ] dip3", (1, 2, 3, 4), [2, 2, 3, 4]),
    ("pick2", (1, 2, 3), [1, 2, 3, 1]),
    ("pick3", (1, 2, 3, 4), [1, 2, 3, 4, 1]),
    ("pick4", (1, 2, 3, 4, 5), [1, 2, 3, 4, 5, 1]),
    # packed accumulators
    ("pack2", (1, 2), [[1, 2]]),
    ("unpack2", ([1, 2],), [1, 2]),
    ("pack3", (1, 2, 3), [[1, 2, 3]]),
    ("unpack3", ([1, 2, 3],), [1, 2, 3]),
    ("pack4", (1, 2, 3, 4), [[1, 2, 3, 4]]),
    ("unpack4", ([1, 2, 3, 4],), [1, 2, 3, 4]),
    # lists
    ("len", ([1, 2, 3],), [3]),
    ("sum", ([1, 2, 3],), [6]),
    ("product", ([2, 3, 4],), [24]),
    ("reverse", ([1, 2, 3],), [[3, 2, 1]]),
    ("head", ([4, 5],), [4]),
    ("tail", ([4, 5, 6],), [[5, 6]]),
    ("last", ([4, 5, 6],), [6]),
    ("max", ([3, 9, 2],), [9]),
    ("min", ([3, 9, 2],), [2]),
    ("take", ([1, 2, 3, 4], 2), [[1, 2]]),
    ("take", ([1, 2, 3, 4], 0), [[]]),
    ("drop-n", ([1, 2, 3, 4], 1), [[2, 3, 4]]),
    ("nth", ([5, 6, 7], 2), [7]),
    ("at", ([10, 20, 30], -1), [30]),
    ("at", ([10, 20, 30], 1), [20]),
    ("contains", ([5, 6, 7], 6), [1]),
    ("contains", ([5, 6, 7], 9), [0]),
    ("count", ([1, 2, 1, 3], 1), [2]),
    ("dedup", ([1, 2, 1, 3, 2],), [[1, 2, 3]]),
    ("remove1", ([5, 1, 9, 1], 1), [[5, 9, 1]]),
    ("zip", ([1, 2, 3], [4, 5, 6, 7]), [[[1, 4], [2, 5], [3, 6]]]),
    ("seconds", ([[1, 2], [3, 4]],), [[2, 4]]),
    ("dot", ([1, 2, 3], [4, 5, 6]), [32]),
    ("prefix-sums", ([1, 2, 3, 4],), [[1, 3, 6, 10]]),
    ("insert", ([1, 3, 5], 4), [[1, 3, 4, 5]]),
    ("sort", ([3, 1, 2, 1],), [[1, 1, 2, 3]]),
    ("nlargest", ([5, 1, 9, 3], 2), [[9, 5]]),
    ("nsmallest", ([5, 1, 9, 3], 2), [[1, 3]]),
    ("lis-best", ([[1, 10], [2, 5]], 3), [[[1, 10], [2, 5]], 3, 10]),
    ("max-subarray-sum", ([1, -2, 3, 4],), [7]),
    # ranges
    ("double", ([0, 1],), [[0, 1, 2, 3]]),
    ("double-by", ([0, 1], 2), [[0, 1, 2, 3]]),
    ("zeros8", (), [Z8]),
    ("zeros16", (), [Z8 * 2]),
    ("zeros128", (), [Z8 * 16]),
    ("zeros1024", (), [Z8 * 128]),
    ("range", (0,), [[]]),
    ("range", (1,), [[0]]),
    ("range", (5,), [[0, 1, 2, 3, 4]]),
    ("range", (1000,), [list(range(1000))]),
    ("range1", (3,), [[1, 2, 3]]),
    ("range-between", (3, 6), [[3, 4, 5, 6]]),
    # digits
    ("digits", (1234,), [[1, 2, 3, 4]]),
    ("digits", (0,), [[]]),
    ("digits", (10**15 + 7,), [[1] + [0] * 14 + [7]]),
    ("digit-sum", (1234,), [10]),
    ("digit-product", (234,), [24]),
    ("powers", (10,), [[10**i for i in range(19, -1, -1)]]),
    ("powers-step", (10, [1]), [10, [10, 1]]),
    ("base-digit-sum", (1234, 10), [10]),
    ("base-digit-sum", (64, 2), [1]),
    ("base-digit-sum", (128, 2), [1]),
    ("base-digit-sum", (2**20 - 1, 2), [20]),
    ("base-digit-sum", (10**6 - 1, 3), [13]),
    # bits
    ("pows2", (), [[2**i for i in range(32)]]),
    ("pows2-step", ([1, 2],), [[1, 2, 4, 8]]),
    ("band", (12, 10), [8]),
    ("band", (65536, 65536), [65536]),
    ("band", (2**31 + 5, 2**31 + 3), [2**31 + 1]),
    ("bxor", (12, 10), [6]),
    ("bxor", (65536, 65536), [0]),
    ("bxor", (2**31, 1), [2**31 + 1]),
]


def as_value(x):
    if isinstance(x, bool):
        return Bool(x)
    if isinstance(x, int):
        return Int(x)
    return Lst(tuple(as_value(i) for i in x))


@pytest.mark.parametrize(
    ("program", "args", "expected"), CASES, ids=[c[0] for c in CASES]
)
def test_macro_case(program: str, args: tuple, expected: list) -> None:
    outcome = run_program(expand(program.split(), MACROS), to_stack(args))
    assert not isinstance(outcome, str), outcome
    stack, _fuel = outcome
    assert [value_key(v) for v in stack] == [value_key(as_value(e)) for e in expected]


def test_every_macro_has_a_case() -> None:
    covered = {tok for program, _, _ in CASES for tok in program.split()}
    missing = sorted(set(MACROS) - covered)
    assert not missing, f"macros without a test case: {missing}"


def test_isqrt_matches_python() -> None:
    import math

    for n in [*range(200), 4095, 4096, 65535, 10**6 - 1, 2**40]:
        outcome = run_program(expand(["isqrt"], MACROS), to_stack((n,)))
        assert not isinstance(outcome, str), (n, outcome)
        assert outcome[0][0] == Int(math.isqrt(n)), n


def test_range_is_cheap_and_needs_no_recursion() -> None:
    outcome = run_program(expand(["range"], MACROS), to_stack((1000,)))
    assert not isinstance(outcome, str), outcome
    _stack, fuel = outcome
    assert fuel < 6000
