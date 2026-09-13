"""The step-3 solve check: verdict rules, dense draws, reproducible streams (#134)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from experiments import step3_verify_solves as verify
from wandering_light.stack_pbe import CATALOGUE

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "reports" / "world-plus-v4-policy-20260912" / "verified_solves.json"
BY_NAME = {b.name: b for b in CATALOGUE}


def _tally(tested: int, failed: int) -> dict:
    return {"tested": tested, "failed": failed, "counterexample": None}


def _recorded(behaviour: str, verdict: str) -> dict:
    programs = json.loads(RECORDED.read_text())["programs"]
    return next(
        p
        for p in programs
        if p["behaviour"] == behaviour and p["verdict"] == verdict and "draws" in p
    )


def test_a_mode_below_the_floor_is_inconclusive_not_genuine() -> None:
    assert (
        verify.verdict_of({"sampler": _tally(600, 0), "perturbed": _tally(700, 0)}, 500)
        == "genuine"
    )
    assert (
        verify.verdict_of({"sampler": _tally(600, 0), "perturbed": _tally(499, 0)}, 500)
        == "inconclusive"
    )
    assert (
        verify.verdict_of({"sampler": _tally(600, 1), "perturbed": _tally(10, 0)}, 500)
        == "shortcut"
    )


def test_a_catalogue_witness_needs_no_draws() -> None:
    behaviour = BY_NAME["keep_below"]
    result = verify.check(behaviour.program, behaviour, draws=10, seed=0, dense=True)
    assert result == {"verdict": "genuine", "why": "equals a catalogue witness"}


def test_the_recorded_shortcut_is_still_caught_in_every_mode() -> None:
    entry = _recorded("all_positive", "shortcut")
    result = verify.check(
        tuple(entry["program"].split()),
        BY_NAME["all_positive"],
        draws=300,
        seed=134,
        dense=True,
        min_tested=50,
    )
    assert result["verdict"] == "shortcut"
    assert set(result["draws"]) == {"sampler", "perturbed", "dense"}


def test_dense_draws_use_the_small_pool_and_keep_booleans() -> None:
    rng = random.Random(0)
    lengths = set()
    for _ in range(300):
        values, number, flag = verify.densify(rng, ([5, 9], 7, True))
        assert all(-2 <= v <= 2 for v in values)
        assert -2 <= number <= 2
        assert flag is True
        lengths.add(len(values))
    assert min(lengths) == 0 and max(lengths) == 10


def test_adding_dense_moves_neither_recorded_stream() -> None:
    """sampler and perturbed tallies must reproduce with and without --dense."""
    entry = _recorded("drop_last", "genuine")
    program, behaviour = tuple(entry["program"].split()), BY_NAME["drop_last"]
    plain = verify.check(program, behaviour, draws=200, seed=134)
    with_dense = verify.check(program, behaviour, draws=200, seed=134, dense=True)
    for mode in ("sampler", "perturbed"):
        assert plain["draws"][mode] == with_dense["draws"][mode]
