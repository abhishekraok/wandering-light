"""Development checks only: this is NOT the declared complete Gate A run."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.score_band_checks import (
    check_semantic_expectation,
    result_key,
    verify_attempts,
    verify_layers,
)
from experiments.score_band_fixtures import (
    OP_INDEX,
    TASK_TABLE,
    Capture,
    Script,
    ScriptedScorer,
    disabled_cases,
    manual_task,
    resource_cases,
    semantic_cases,
)
from experiments.score_band_search import _retain, band_beam_solve
from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import Bool, Int, Lst, Mark, parse_stack
from wandering_light.stack_pbe import BASIS, OPS
from wandering_light.stack_policy import PolicyScorer


@pytest.mark.parametrize("case", disabled_cases(), ids=lambda c: c.id)
def test_disabled_matches_imported_production(case):
    task = manual_task(case.task)
    ref, own = Capture(), Capture()
    reference = beam_solve(
        task, ScriptedScorer(case.script), observer=ref, **case.settings()
    )
    candidate = band_beam_solve(
        task,
        ScriptedScorer(case.script),
        observer=own,
        score_band_nats=None,
        **case.settings(),
    )
    assert result_key(candidate.beam_result) == result_key(reference)
    assert own.events == ref.events and not candidate.band_pruned
    verify_attempts(own, candidate.beam_result)
    verify_layers(own, case)


@pytest.mark.parametrize("case", semantic_cases(), ids=lambda c: c.id)
def test_fixed_semantics_and_observer_transparency(case):
    task = manual_task(case.task)
    own = Capture()
    scorer = ScriptedScorer(case.script)
    if case.expectation == "ValueError":
        for observer in (own, None):
            with pytest.raises(ValueError, match="cumulative"):
                band_beam_solve(
                    task,
                    ScriptedScorer(case.script),
                    observer=observer,
                    score_band_nats=case.band,
                    **case.settings(),
                )
        assert any(e[0] == "scored" for e in own.events)
        assert not any(e[0] == "retained" for e in own.events)
        verify_attempts(own)
        return
    ref = Capture()
    reference = beam_solve(
        task, ScriptedScorer(case.script), observer=ref, **case.settings()
    )
    candidate = band_beam_solve(
        task, scorer, observer=own, score_band_nats=case.band, **case.settings()
    )
    check_semantic_expectation(case, reference, candidate, own, scorer, ref)
    verify_attempts(own, candidate.beam_result)
    verify_layers(own, case)
    quiet = band_beam_solve(
        task, ScriptedScorer(case.script), score_band_nats=case.band, **case.settings()
    )
    assert result_key(quiet.beam_result) == result_key(candidate.beam_result)
    assert quiet.band_pruned == candidate.band_pruned


@pytest.mark.parametrize("case", resource_cases(), ids=lambda c: c.id)
def test_resource_reference_and_low_level_charging(case):
    task = manual_task(case.task)
    ref, own = Capture(), Capture()
    reference = beam_solve(
        task, ScriptedScorer(case.script), observer=ref, **case.settings()
    )
    scorer = ScriptedScorer(case.script)
    candidate = band_beam_solve(
        task, scorer, observer=own, score_band_nats=case.band, **case.settings()
    )
    assert (
        result_key(candidate.beam_result) == result_key(reference)
        and own.events == ref.events
    )
    verify_attempts(own, candidate.beam_result)
    verify_layers(own, case)
    result = candidate.beam_result
    if case.expectation == "budget-before-score":
        expected = (
            0
            if case.budget == 0
            else 2
            if case.budget == 1
            else 4
            if case.unit == "executions+fuel"
            else 8
        )
        assert result.termination == "budget"
        assert result.executions == result.fuel == expected and result.expansions == 1
        assert scorer.calls == 0
    elif case.expectation == "past-deadline":
        assert result.termination == "wall_clock_cap" and scorer.calls == 0
        assert result.executions == result.fuel == 0 and result.expansions == 1
    else:
        root_i = next(
            e
            for e in own.events
            if e[0] == "attempt" and e[1] == 0 and e[3] == OP_INDEX["i"]
        )
        assert (
            root_i[6] > root_i[7] > 0
        )  # quotation-body fuel, not just outer executions


@pytest.mark.parametrize("name", TASK_TABLE)
def test_manual_typed_states_and_witnesses(name):
    task = manual_task(name)
    inputs = tuple(parse_stack(x, basis=BASIS) for x in task.inputs)
    outputs = tuple(parse_stack(x, basis=BASIS) for x in task.outputs)
    if task.program:
        assert replay_program(task.program, inputs, outputs)[0]
    if name == "T-empty":
        assert inputs == ((),)
    elif name == "T-open":
        assert inputs == ((Mark(), Int(0)),)
    elif name == "T-typed":
        assert inputs[0] == (Lst((Bool(True), Int(1))), Bool(False))
    elif name == "T-long":
        assert inputs == ((Int(0),) * 163,)


@pytest.mark.parametrize("tau", [-1, np.nan, np.inf, -np.inf, True, "1"])
@pytest.mark.parametrize("collecting", [False, True])
def test_invalid_band_precedes_task_access_and_restores_gc(tau, collecting):
    before = gc.isenabled()
    (gc.enable if collecting else gc.disable)()
    try:
        with pytest.raises(ValueError, match="score_band_nats"):
            band_beam_solve(object(), object(), score_band_nats=tau)
        assert gc.isenabled() == collecting
    finally:
        (gc.enable if before else gc.disable)()


@pytest.mark.parametrize("collecting", [False, True])
def test_scorer_exception_restores_gc(collecting):
    before = gc.isenabled()
    (gc.enable if collecting else gc.disable)()
    try:
        with pytest.raises(RuntimeError, match="scorer"):
            band_beam_solve(
                manual_task("T-probe"),
                ScriptedScorer(Script(raises=True)),
                score_band_nats=1,
                guard="off",
            )
        assert gc.isenabled() == collecting
    finally:
        (gc.enable if before else gc.disable)()


@pytest.mark.parametrize(
    "kind",
    [
        "last-ranked-width",
        "last-ranked-width-all-inf",
        "width-before-band",
        "last-row-band",
    ],
)
def test_arithmetic_units_have_no_solver_outcome_claim(kind):
    scores = [0.0, -2.0] if kind == "last-row-band" else [0.0, 0.0]
    if kind == "last-ranked-width-all-inf":
        scores = [-np.inf, -np.inf]
    width = 1 if kind == "width-before-band" else 4 if kind == "last-row-band" else 2
    root = ((Int(9),),)
    products = [((Int(0),),), ((Int(1),),)]
    beam = [(root, (), 0.0)]
    rows = [(0, i, p, (OPS[i].name,)) for i, p in enumerate(products)]
    layer = _retain(
        0,
        rows,
        beam,
        np.array([scores]),
        {root},
        beam_width=width,
        remaining=0,
        guard="off",
        max_state_atoms=None,
        score_band_nats=1,
    )
    stop = (
        "width"
        if kind == "width-before-band"
        else "score_band"
        if kind == "last-row-band"
        else "exhausted"
    )
    assert layer.stop == stop
    assert layer.width_pruned == (stop == "width")
    assert layer.band_pruned == (stop == "score_band")
    assert len(layer.verdicts) == (1 if stop == "width" else 2)


def test_scorer_api_dead_and_live_rows_without_model():
    stub = SimpleNamespace(logits=lambda blocks: np.zeros((2, len(OPS))))
    mask = np.zeros((2, len(OPS)), dtype=bool)
    mask[1, OP_INDEX["0"]] = True
    with pytest.warns(RuntimeWarning) as warnings:
        values = PolicyScorer.log_probs(stub, [None, None], mask)
    assert {str(w.message) for w in warnings} == {
        "divide by zero encountered in log",
        "invalid value encountered in subtract",
    }
    assert np.isnan(values[0]).all() and values[1, OP_INDEX["0"]] == 0
    assert np.isneginf(values[1, ~mask[1]]).all()


def test_frozen_production_reference_hash():
    import hashlib

    from wandering_light import stack_beam

    assert (
        hashlib.sha256(Path(stack_beam.__file__).read_bytes()).hexdigest()
        == "dcf4558a6cfa11d8b8dde9aaa02ff21c9837ec16279a692f9b7e723c94ec616b"
    )


@pytest.mark.parametrize("budget", [193, 194, 339, 340, 395, 396])
def test_independent_expansion_charging_at_node_boundaries(budget):
    from dataclasses import replace

    case = replace(
        next(c for c in resource_cases() if c.id == "resource/body/executions/None"),
        budget=budget,
    )
    task = manual_task(case.task)
    reference_observer, candidate_observer = Capture(), Capture()
    reference = beam_solve(
        task,
        ScriptedScorer(case.script),
        observer=reference_observer,
        **case.settings(),
    )
    candidate = band_beam_solve(
        task,
        ScriptedScorer(case.script),
        observer=candidate_observer,
        **case.settings(),
    )
    assert result_key(candidate.beam_result) == result_key(reference)
    verify_attempts(reference_observer, reference)
    verify_attempts(candidate_observer, candidate.beam_result)
    verify_layers(candidate_observer, case)


@pytest.mark.parametrize("tamper", ["initial-seen", "next-beam", "retained-event"])
def test_independent_layer_check_rejects_inconsistent_observation(tamper):
    from dataclasses import replace

    case = next(c for c in semantic_cases() if c.id == "A-safe")
    observer = Capture()
    band_beam_solve(
        manual_task(case.task),
        ScriptedScorer(case.script),
        observer=observer,
        score_band_nats=case.band,
        **case.settings(),
    )
    if tamper == "initial-seen":
        observer.layers[0] = replace(observer.layers[0], old_seen=frozenset())
    else:
        kind, depth = ("started", 1) if tamper == "next-beam" else ("retained", 0)
        at = next(
            i for i, e in enumerate(observer.events) if e[0] == kind and e[1] == depth
        )
        event = observer.events[at]
        observer.events[at] = (*event[:2], (), *event[3:])
    with pytest.raises(AssertionError):
        verify_layers(observer, case)
