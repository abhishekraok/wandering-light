"""#183 synthetic-only search and runner checks; no corpus/checkpoint access."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gumbel_retention import jitter_beam_solve
from experiments.gumbel_retention_run import (
    SEEDS,
    ShownTask,
    baseline_gate,
    comparison_gate,
    select_panel,
)
from wandering_light.stack_beam import BeamObserver, beam_solve
from wandering_light.stack_pbe import OPS


class Scorer:
    def log_probs(self, blocks, legal):
        logits = np.broadcast_to(
            -np.arange(len(OPS), dtype=np.float32) / 20, legal.shape
        ).copy()
        logits[~legal] = -np.inf
        peak = logits.max(axis=1, keepdims=True)
        norm = np.log(np.exp(logits - peak).sum(axis=1, keepdims=True))
        return logits - peak - norm


def node_key(nodes):
    return [
        (p, program, np.asarray(score).dtype.str, np.asarray(score).tobytes())
        for p, program, score in nodes
    ]


class Trace(BeamObserver):
    def __init__(self):
        self.events = []
        self.starts = {}
        self.scores = {}
        self.retained = {}
        self.attempts = []

    def layer_started(self, depth, beam):
        self.starts[depth] = list(beam)
        self.events.append(("start", depth, node_key(beam)))

    def expansion_attempted(
        self, depth, node, op_index, successor, reason, fuel, stepped
    ):
        event = (depth, node, op_index, successor, reason, fuel, stepped)
        self.attempts.append(event)
        self.events.append(("attempt", event))

    def layer_scored(self, depth, log_probs, legal):
        self.scores[depth] = log_probs.copy()
        self.events.append(("scored", depth, log_probs.tobytes(), legal.tobytes()))

    def layer_retained(self, depth, beam, verdicts, parents):
        self.retained[depth] = (list(beam), list(parents), dict(verdicts))
        self.events.append(
            ("retained", depth, node_key(beam), dict(verdicts), list(parents))
        )

    def search_terminated(self, *args):
        self.events.append(("terminated", args))


def substantive(result):
    return {k: v for k, v in asdict(result).items() if k != "seconds"}


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        (("2 3", "3 4"), ("12", "24")),
        (("2 3", "3 4"), ("2 3", "3 4")),
        (("(1 2)", "(2 3)"), ("999", "999")),
        (("",), ("1",)),
    ],
)
@pytest.mark.parametrize(
    "settings",
    [
        {"beam_width": 8, "max_depth": 3, "guard": "both", "max_state_atoms": 16},
        {"beam_width": 3, "max_depth": 4, "guard": "off", "max_state_atoms": 4},
        {"beam_width": 8, "max_depth": 3, "guard": "close", "budget": 11},
        {"beam_width": 8, "max_depth": 3, "deadline": 0.0},
    ],
)
def test_disabled_matches_imported_production_on_real_typed_tasks(
    inputs, outputs, settings
):
    a, b = Trace(), Trace()
    task = ShownTask(inputs, outputs)
    expected = beam_solve(task, Scorer(), observer=a, **settings)
    actual = jitter_beam_solve(task, Scorer(), observer=b, **settings)
    assert substantive(actual) == substantive(expected)
    assert b.events == a.events


def test_nonempty_identity_fit_still_uses_production_first_fit():
    task = ShownTask(("2 3", "3 4"), ("2 3", "3 4"))
    result = jitter_beam_solve(
        task, Scorer(), rng=np.random.default_rng(17), beam_width=8, max_depth=3
    )
    assert result.program and result.termination == "fit"
    assert result.depth > 0


def test_noise_is_fresh_for_every_successor_and_not_stored_in_path_scores():
    class CountingNoise:
        def __init__(self):
            self.calls = 0

        def gumbel(self):
            self.calls += 1
            return 1000.0 if self.calls % 3 else -1000.0

    trace, noise = Trace(), CountingNoise()
    result = jitter_beam_solve(
        ShownTask(("2",), ("999",)),
        Scorer(),
        rng=noise,
        observer=trace,
        beam_width=5,
        max_depth=3,
        guard="both",
        max_state_atoms=8,
    )
    assert result.termination == "max_depth"
    assert noise.calls == sum(e[3] is not None for e in trace.attempts)
    assert set(trace.retained) == set(trace.scores) == {0, 1, 2}
    for depth, (nodes, parents, _verdicts) in trace.retained.items():
        for (_, _, score), (parent, op) in zip(nodes, parents, strict=True):
            wanted = trace.starts[depth][parent][2] + trace.scores[depth][parent, op]
            assert np.asarray(score).dtype == np.asarray(wanted).dtype
            assert np.asarray(score).tobytes() == np.asarray(wanted).tobytes()
            assert score <= 0  # injected +/-1000 is not retained


def test_seeded_candidate_is_reproducible_and_actually_changes_retention():
    task = ShownTask(("2",), ("999",))
    traces = []
    for seed in (17, 17, 29):
        trace = Trace()
        jitter_beam_solve(
            task,
            Scorer(),
            rng=np.random.default_rng(seed),
            observer=trace,
            beam_width=3,
            max_depth=3,
        )
        traces.append(trace.events)
    assert traces[0] == traces[1]
    assert traces[0] != traces[2]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_candidate_rejects_nonfinite_executable_scores_and_restores_gc(bad):
    class BadScorer(Scorer):
        def log_probs(self, blocks, legal):
            out = super().log_probs(blocks, legal)
            out[legal] = bad
            return out

    collecting = gc.isenabled()
    with pytest.raises(ValueError, match="nonfinite"):
        jitter_beam_solve(
            ShownTask(("2",), ("999",)),
            BadScorer(),
            rng=np.random.default_rng(17),
            max_depth=2,
        )
    assert gc.isenabled() is collecting


def fake_task(skeleton, behaviour, *, split="train"):
    return SimpleNamespace(
        skeleton=skeleton,
        behaviour=behaviour,
        split=split,
        inputs=("2",),
        outputs=("3",),
    )


def test_selection_uses_first_source_per_skeleton_and_ignores_witness_metadata():
    tasks = [
        fake_task("alpha", "a-first"),
        fake_task("alpha", "a-later"),
        fake_task("beta", "b"),
        fake_task("gamma", "c"),
    ]
    ids, shown = select_panel(tasks, count=3)
    assert {x["source_index"] for x in ids} == {0, 2, 3}
    assert next(x for x in ids if x["skeleton"] == "alpha")["behaviour"] == "a-first"
    assert all(set(x) == {"source_index", "inputs", "outputs"} for x in shown)
    # Fields absent from these fixtures include held-out pairs and program.
    tasks[0].program = ("arbitrary",)
    tasks[0].held_out_inputs = ("changed",)
    assert select_panel(tasks, count=3) == (ids, shown)


@pytest.mark.parametrize(
    "tasks",
    [
        [fake_task("", "x")],
        [fake_task("s", "x", split="eval")],
        [fake_task("s", "x"), fake_task("s", "y")],
    ],
)
def test_selection_refuses_missing_or_insufficient_population_without_refill(tasks):
    with pytest.raises(ValueError):
        select_panel(tasks, count=2)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([22, 22, 22], "READY_FOR_REGISTRATION"),
        ([21, 22, 21], "READY_FOR_REGISTRATION"),
        ([22, 23, 22], "NO_HEADROOM"),
        ([20, 22, 21], "BASELINE_UNSTABLE"),
    ],
)
def test_baseline_gate_exact_boundaries(counts, expected):
    assert baseline_gate(counts) == expected


def summary(count, seconds):
    return {
        "verified_count": count,
        "solve_seconds": seconds,
        "verified": [str(i) for i in range(count)],
    }


def test_comparison_gate_requires_all_three_parts_and_keeps_exact_ratio_boundary():
    cells = {}
    for seed in SEEDS:
        cells[f"baseline-{seed}"] = summary(5, 10.0)
        cells[f"candidate-{seed}"] = summary(7, 11.0)
    assert comparison_gate(cells)["reading"] == "POSITIVE_TRAIN_SCREEN"
    cells["candidate-17"] = summary(4, 11.0)
    assert comparison_gate(cells)["reading"] == "NEGATIVE_TRAIN_SCREEN"
    cells["candidate-17"] = summary(7, 11.0)
    cells["candidate-29"] = summary(7, 11.01)
    cells["candidate-43"] = summary(7, 11.01)
    assert comparison_gate(cells)["reading"] == "NEGATIVE_TRAIN_SCREEN"


def test_cell_integration_uses_production_on_shown_only_fake_checkpoint(
    tmp_path, monkeypatch
):
    # Explicitly mock model loading and thread APIs. This tests runner plumbing,
    # not a real checkpoint, runtime resource enforcement or source provenance.
    import torch

    from experiments import gumbel_retention_run as runner
    from wandering_light import stack_policy

    for key in runner.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    for name in ("set_num_threads", "set_num_interop_threads"):
        monkeypatch.setattr(torch, name, lambda _: None)
    for name in ("get_num_threads", "get_num_interop_threads"):
        monkeypatch.setattr(torch, name, lambda: 1)
    monkeypatch.setattr(stack_policy, "load_policy", lambda *a, **k: object())
    monkeypatch.setattr(stack_policy, "PolicyScorer", lambda *a, **k: Scorer())
    panel = tmp_path / "shown.json"
    panel.write_text(
        json.dumps(
            [{"source_index": i, "inputs": [""], "outputs": ["1"]} for i in range(24)]
        )
    )
    out = tmp_path / "cell.json"
    runner.run_cell(panel, tmp_path / "FAKE_CHECKPOINT", out, "baseline", 17)
    result = json.loads(out.read_text())
    assert result["complete"] and len(result["rows"]) == 24
    assert all(row["termination"] == "fit" for row in result["rows"])
    assert all(row["program"] == ["1"] for row in result["rows"])
    assert all(row["solve_seconds"] >= row["seconds"] for row in result["rows"])


@pytest.mark.parametrize("phase", ["baseline", "comparison"])
def test_phase_does_all_timing_before_any_verification(tmp_path, monkeypatch, phase):
    # Only orchestration is real here: data fetch/source/model/runtime and
    # subprocess outputs are synthetic mocks, never real eligibility evidence.
    from experiments import gumbel_retention_run as runner
    from wandering_light import corpus_hub, stack_pbe, stack_policy_hub

    monkeypatch.setattr(runner, "MAIN", tmp_path)
    tasks = [fake_task(f"s{i}", f"b{i}") for i in range(24)]
    monkeypatch.setattr(corpus_hub, "fetch_corpus", lambda *a, **k: tmp_path)
    monkeypatch.setattr(stack_pbe, "load", lambda *a: tasks)
    monkeypatch.setattr(stack_policy_hub, "fetch_stack_policy", lambda *a: tmp_path)
    monkeypatch.setattr(runner, "sha", lambda p: runner.TRAIN_SHA)
    monkeypatch.setattr(
        runner, "provenance", lambda p: {"checkpoint": {"MOCK": "fake"}}
    )
    events = []

    def fake_run(cmd, **kwargs):
        arm, seed = cmd[cmd.index("--arm") + 1], int(cmd[cmd.index("--seed") + 1])
        shown = runner.read(cmd[cmd.index("--shown") + 1])
        events.append(("timing", arm, seed))
        runner.save(
            cmd[cmd.index("--out") + 1],
            {
                "complete": True,
                "host": "MOCK",
                "rows": [{"source_index": d["source_index"]} for d in shown],
            },
        )

    def fake_summarize(cells, identities, originals):
        assert len(events) == (3 if phase == "baseline" else 6)
        assert all(e[0] == "timing" for e in events)
        assert len(identities) == len(originals) == 24
        events.append(("verification",))
        return {name: summary(10, 1.0) for name in cells}, []

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "summarize", fake_summarize)
    out = tmp_path / "checkpoints" / phase
    baseline = tmp_path / "feasibility"
    if phase == "comparison":
        baseline.mkdir()
        runner.save(
            baseline / "summary.json",
            {
                "complete": True,
                "phase": "baseline",
                "reading": "READY_FOR_REGISTRATION",
                "cells": {f"baseline-{seed}": summary(10, 1.0) for seed in SEEDS},
            },
        )
        identities, shown = runner.select_panel(tasks)
        runner.save(baseline / "selection.json", identities)
        runner.save(baseline / "shown.json", shown)
        runner.save(baseline / "provenance.json", {"checkpoint": {"MOCK": "fake"}})
    runner.run_phase(
        SimpleNamespace(
            out=out,
            phase=phase,
            baseline=baseline if phase == "comparison" else None,
            registration_url="SYNTHETIC_TEST_ONLY",
        )
    )
    result = runner.read(out / "summary.json")
    assert result["reading"] == (
        "READY_FOR_REGISTRATION" if phase == "baseline" else "NEGATIVE_TRAIN_SCREEN"
    )
    assert events[-1] == ("verification",)
    expected = (
        tuple(("baseline", s) for s in SEEDS) if phase == "baseline" else runner.ORDER
    )
    assert [(e[1], e[2]) for e in events[:-1]] == list(expected)


def test_real_grade_cannot_be_overridden_by_mock_genuine_verdict(tmp_path):
    """Real shown replay and hidden grading; only fresh-reference checking mocked."""
    from experiments.gumbel_retention_run import summarize

    task = SimpleNamespace(
        behaviour="sum",
        inputs=("2",),
        outputs=("3",),
        held_out_inputs=("9",),
        held_out_outputs=("99",),
    )
    row = {
        "source_index": 0,
        "program": ["1", "add"],
        "solve_seconds": 1.0,
        "executions": 2,
        "fuel": 2,
        "termination": "fit",
    }
    cells = {
        name: {"complete": True, "host": "SYNTHETIC", "rows": [dict(row)]}
        for name in ("baseline-17", "candidate-17")
    }
    calls = []

    def fake_check(*args, **kwargs):
        calls.append((args, kwargs))
        return {"verdict": "genuine", "draws": {"MOCK": True}}

    result, _verdicts = summarize(cells, [{"source_index": 0}], [task], fake_check)
    assert len(calls) == 1  # identical behaviour/program checked once after timing
    assert all(cell["verified_count"] == 0 for cell in result.values())
    assert all(not cell["graded"] for cell in result.values())
    assert all(cell["kinds"]["genuine"] == ["sum"] for cell in result.values())


def test_comparison_checks_baseline_counts_before_loading_checkpoint(
    tmp_path, monkeypatch
):
    """Forging READY in a synthetic baseline summary must not open the contrast."""
    from experiments import gumbel_retention_run as runner
    from wandering_light import corpus_hub, stack_pbe, stack_policy_hub

    monkeypatch.setattr(runner, "MAIN", tmp_path)
    tasks = [fake_task(f"s{i}", f"b{i}") for i in range(24)]
    monkeypatch.setattr(corpus_hub, "fetch_corpus", lambda *a, **k: tmp_path)
    monkeypatch.setattr(stack_pbe, "load", lambda *a: tasks)
    monkeypatch.setattr(runner, "sha", lambda p: runner.TRAIN_SHA)

    def forbidden_fetch(*args):
        pytest.fail("model bytes must not be accessed for a closed baseline gate")

    monkeypatch.setattr(stack_policy_hub, "fetch_stack_policy", forbidden_fetch)
    baseline = tmp_path / "prior"
    baseline.mkdir()
    runner.save(
        baseline / "summary.json",
        {
            "complete": True,
            "phase": "baseline",
            "reading": "READY_FOR_REGISTRATION",
            "cells": {f"baseline-{s}": {"verified_count": 23} for s in SEEDS},
        },
    )
    out = tmp_path / "checkpoints" / "comparison"
    with pytest.raises(ValueError, match="baseline counts"):
        runner.run_phase(
            SimpleNamespace(
                out=out,
                phase="comparison",
                baseline=baseline,
                registration_url="SYNTHETIC_TEST_ONLY",
            )
        )
    assert not (out / "summary.json").exists()
    assert runner.read(out / "INCOMPLETE.json")["complete"] is False
