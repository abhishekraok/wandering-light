"""Synthetic contract checks only; no real payload or pretrained checkpoint."""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import root_map_first as route
from experiments import root_map_first_run as runner
from experiments.gumbel_retention_run import ShownTask
from wandering_light import stack_machine as m
from wandering_light.stack_beam import BeamResult, beam_solve


def task(ins, outs):
    return ShownTask(tuple(ins), tuple(outs))


class Uniform:
    def log_probs(self, blocks, legal):
        result = np.zeros(legal.shape, dtype=np.float32)
        return np.where(legal, result, -np.inf)


@dataclass
class FakeTask:
    skeleton: str
    inputs: tuple = ("[ 0 ]",)
    outputs: tuple = ("[ 1 ]",)
    split: str = "train"
    behaviour: str = "synthetic"


@pytest.mark.parametrize(
    ("ins", "outs", "reason"),
    [
        (["[ 1 ]"], ["[ 2 ]"], "not_list_tops"),
        (["[ 0 (1) ]"], ["[ 1 (2) ]"], "outer_rest_changed"),
        (["[ (1 2) ]"], ["[ (2) ]"], "length_changed"),
        (["[ (1 1) ]"], ["[ (2 3) ]"], "contradictory_element"),
        (["[ (1) ]", "[ (1) ]"], ["[ (2) ]", "[ (3) ]"], "contradictory_element"),
        (["[ (0) ]"], ["[ (MARK) ]"], "unsupported_mark_goal"),
    ],
)
def test_rejection(ins, outs, reason):
    assert route.derive(task(ins, outs)).reason == reason


def test_occurrences_types_context_and_empty():
    t = route.derive(task(["[ 7 (0 false 0) ]"], ["[ 7 (1 true 1) ]"]))
    assert t.reason == "eligible" and t.invocations == 3
    assert t.body.inputs == ("[ 0 ]", "[ false ]", "[ 0 ]")
    assert not t.empty_body
    assert route.derive(task(["[ () ]"], ["[ () ]"])).empty_body
    assert route.derive(task(["[ (0 (1) true) ]"], ["[ (0 (1) true) ]"])).empty_body
    # A capture-varying map could solve this; this ROOT uncaptured template cannot.
    assert (
        route.derive(
            task(["[ 1 (0) ]", "[ 2 (0) ]"], ["[ 1 (1) ]", "[ 2 (2) ]"])
        ).reason
        == "contradictory_element"
    )


def test_roundtrip_rejection_is_real_typed_value(monkeypatch):
    before = (m.Lst((m.Sym("["),)),)
    after = (m.Lst((m.Int(1),)),)
    real = route.parse_stack

    def parse(text, **kwargs):
        if text == "given":
            return before
        if text == "wanted":
            return after
        return real(text, **kwargs)

    monkeypatch.setattr(route, "parse_stack", parse)
    assert (
        route.derive(task(["given"], ["wanted"])).reason == "unrepresentable_singleton"
    )


def test_malformed_is_not_silent_miss():
    with pytest.raises(ValueError):
        route.derive(task([], []))


def test_real_assembled_increment_and_nested_quotation():
    t = route.derive(task(["[ (0 1 2) ]"], ["[ (1 2 3) ]"]))
    q = ("[", "1", "add", "]", "map")
    v = route.verify_wrapper(q, t.inputs, t.outputs, math.inf)
    assert v["fit"] and v["executions"] == 5 and v["attempts"] == 5
    assert v["fuel"] == 11  # quotation construction4 + map(1 + 2*3)
    t = route.derive(task(["[ (0 1) ]"], ["[ (() ()) ]"]))
    assert route.verify_wrapper(
        ("[", "drop", "[", "]", "]", "map"), t.inputs, t.outputs, math.inf
    )["fit"]


def test_default_shared_fuel_is_not_individual_body_fuel():
    # Actual production allocation, source and values, no parser bracket ambiguity.
    body = tuple(v for _ in range(10) for v in (m.Int(1), m.Sym("add")))
    assert m.run_body(body, (m.Int(0),), m.DEFAULT_FUEL_LIMIT, 1) == ((m.Int(10),), 20)
    source = m.Lst((m.Int(0),) * 5000)
    q = m.Lst(body)
    outcome = m.step_raw(route.OPS["map"], (source, q), m.DEFAULT_FUEL_LIMIT, 0)
    assert isinstance(outcome, str) and "out of fuel" in outcome
    # The complete reconstructed primitive path must reject it too.
    v = route.verify_wrapper(
        ("[", *(x for _ in range(10) for x in ("1", "add")), "]", "map"),
        ((source,),),
        ((m.Lst((m.Int(10),) * 5000),),),
        math.inf,
    )
    assert not v["fit"] and v["reason"] == "refusal" and "out of fuel" in v["refusal"]


def test_wrapper_preserves_depth_failure():
    # Real run_body depth boundary remains observable, never bypassed by fitting.
    result = m.run_body((m.Sym("1"),), (m.Int(0),), 100, m.MAX_QUOTATION_DEPTH + 1)
    assert isinstance(result, str) and "deep" in result


def test_prospective_deadline_and_failed_call_charges(monkeypatch):
    ticks = iter([0.0, 6.0])
    monkeypatch.setattr(route.time, "perf_counter", lambda: next(ticks))
    v = route.verify_wrapper(("1", "add"), ((m.Int(0),),), ((m.Int(1),),), 5.0)
    assert (
        v["reason"] == "wall_clock_cap"
        and v["completed_ops"] == 1
        and v["executions"] == 1
    )
    monkeypatch.setattr(route.time, "perf_counter", lambda: 0.0)
    v = route.verify_wrapper(
        ("add",), ((m.Int(1), m.Int(2)), (m.Int(0),)), ((m.Int(3),), ()), 5.0
    )
    assert v["reason"] == "refusal" and (
        v["executions"],
        v["attempts"],
        v["example"],
    ) == (1, 2, 1)


def test_final_authorized_op_may_finish_after_deadline(monkeypatch):
    clock = [0.0]
    original = route.step_raw

    def step(*a):
        result = original(*a)
        clock[0] = 99.0
        return result

    monkeypatch.setattr(route.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(route, "step_raw", step)
    assert route.verify_wrapper(("1",), ((),), ((m.Int(1),),), 5.0)["fit"]


def test_actual_production_control_compatibility():
    for t in [
        task(["[ 0 ]", "[ 4 ]"], ["[ 1 ]", "[ 5 ]"]),
        task(["[ 2 3 ]"], ["[ 5 ]"]),
        task(["[ () ]"], ["[ () ]"]),
    ]:
        original = beam_solve(
            t,
            Uniform(),
            deadline=time.perf_counter() + 30,
            max_depth=48,
            **route.SETTINGS,
        )
        wrapped = route.solve_task(t, Uniform(), "control")
        assert {
            k: v
            for k, v in wrapped["stages"][0].items()
            if k in asdict(original) and k != "seconds"
        } == {k: v for k, v in asdict(original).items() if k != "seconds"}
        assert (
            wrapped["program"] == list(original.program)
            if original.program is not None
            else wrapped["program"] is None
        )


def test_real_candidate_empty_identity_and_increment():
    for t in [
        task(["[ () ]"], ["[ () ]"]),
        task(["[ (0 1) ]"], ["[ (0 1) ]"]),
        task(["[ (0 1) ]"], ["[ (1 2) ]"]),
    ]:
        row = route.solve_task(t, Uniform(), "candidate")
        assert row["source"] == "root_map" and row["termination"] == "fit"
        assert row["task_deadline"] == row["task_started"] + 5.0
        runner.validate_row(
            {"source_index": 0, **row},
            {"source_index": 0, "inputs": t.inputs, "outputs": t.outputs},
            "candidate",
        )


def test_classifier_spends_body_slice_and_fallback_has_original_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route.time, "perf_counter", lambda: clock[0])
    original = route.derive

    def derive(t):
        result = original(t)
        clock[0] = 2.0
        return result

    monkeypatch.setattr(route, "derive", derive)
    calls = []

    def solve(t, scorer, **kw):
        calls.append(kw)
        return BeamResult(None, "max_depth", 0.0, 0, 0, 0, 48, False)

    monkeypatch.setattr(route, "beam_solve", solve)
    row = route.solve_task(task(["[ (0) ]"], ["[ (1) ]"]), Uniform(), "candidate")
    assert [s["stage"] for s in row["stages"]] == [
        "classify",
        "body_skipped",
        "fallback",
    ]
    assert (
        len(calls) == 1 and calls[0]["deadline"] == 5.0 and calls[0]["max_depth"] == 48
    )


def test_one_body_only_failed_assembly_falls_back(monkeypatch):
    calls = []

    def solve(t, scorer, **kw):
        calls.append(kw)
        if kw["max_depth"] == 45:
            # Deliberate software mock: a dishonest body fit cannot return as a whole fit.
            return BeamResult(("drop",), "fit", 0.0, 1, 1, 1, 1, False)
        return BeamResult(None, "max_depth", 0.0, 0, 0, 0, 48, False)

    monkeypatch.setattr(route, "beam_solve", solve)
    row = route.solve_task(task(["[ (0) ]"], ["[ (1) ]"]), Uniform(), "candidate")
    assert len(calls) == 2 and [c["max_depth"] for c in calls] == [45, 48]
    assert calls[1]["deadline"] == row["task_deadline"]
    assert row["program"] is None and row["source"] == "fallback"
    assert (
        next(s for s in row["stages"] if s["stage"] == "verify")["reason"] == "refusal"
    )


def test_no_work_after_classification_uses_task_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(route.time, "perf_counter", lambda: clock[0])
    original = route.derive

    def derive(t):
        result = original(t)
        clock[0] = 6.0
        return result

    monkeypatch.setattr(route, "derive", derive)
    monkeypatch.setattr(
        route, "beam_solve", lambda *a, **k: pytest.fail("no remaining task clock")
    )
    row = route.solve_task(task(["[ 0 ]"], ["[ 1 ]"]), Uniform(), "candidate")
    assert row["source"] == "none" and row["seconds"] == 6.0


def test_origins_actual_frozen_tree():
    root = Path(__file__).resolve().parents[1]
    assert (
        Path(inspect.getsourcefile(inspect.unwrap(m.step_raw))).resolve()
        == root / "wandering_light/stack_machine.py"
    )
    assert runner.origins()["functions"]["solve_task"] == str(
        root / "experiments/root_map_first.py"
    )


def test_selection_new_salt_first_row_and_no_filter():
    tasks = [FakeTask(str(i)) for i in range(30)] + [
        FakeTask("0", ("[ () ]",), ("[ () ]",))
    ]
    identities, shown = runner.select_panel(tasks)
    assert len(identities) == len(shown) == 24
    assert all(e["source_index"] < 30 for e in identities)
    assert all(set(e) == {"source_index", "inputs", "outputs"} for e in shown)
    with pytest.raises(ValueError):
        runner.select_panel(tasks[:10])


def test_explicit_review_verdict(tmp_path):
    p = tmp_path / "review.log"
    for text in [
        "CHANGES_REQUIRED\nquoted CODE REVIEW PASS",
        "",
        "CODE REVIEW PASS\nwrong head",
    ]:
        p.write_text(
            json.dumps({"subtype": "success", "is_error": False, "result": text})
            + "\nexit 0\n"
        )
        with pytest.raises(ValueError):
            runner.review_pin(p, "frozen-head")
    p.write_text(
        json.dumps(
            {
                "subtype": "success",
                "is_error": False,
                "result": "CODE REVIEW PASS\nfrozen-head",
            }
        )
        + "\nexit 0\n"
    )
    assert str(p) in runner.review_pin(p, "frozen-head")


def test_malformed_result_population_rejected():
    t = task(["[ () ]"], ["[ () ]"])
    e = {"source_index": 0, "inputs": t.inputs, "outputs": t.outputs}
    row = {"source_index": 0, **route.solve_task(t, Uniform(), "candidate")}
    for field, value in [
        ("seconds", math.nan),
        ("program", ["1"]),
        ("task_deadline", 0),
        ("eligibility", "wrong"),
    ]:
        bad = copy.deepcopy(row)
        bad[field] = value
        with pytest.raises(ValueError):
            runner.validate_row(bad, e, "candidate")


def synthetic_cells():
    shown = [
        {"source_index": i, "inputs": ["[ () ]"], "outputs": ["[ () ]"]}
        for i in range(24)
    ]
    originals = [
        SimpleNamespace(
            inputs=e["inputs"],
            outputs=e["outputs"],
            held_out_inputs=["[ () ]"],
            held_out_outputs=["[ () ]"],
        )
        for e in shown
    ]
    rows = {
        a: route.solve_task(task(shown[0]["inputs"], shown[0]["outputs"]), Uniform(), a)
        for a in ("control", "candidate")
    }
    # Control synthetic fixture uses a universally valid identity program from production.
    assert rows["control"]["program"] is not None
    pins, imported, model, limits = (
        {"head": "fixture"},
        {"MOCK_ORIGINS": True},
        {"MOCK_MODEL": True},
        {"MOCK_RUNTIME": True},
    )
    cells = {}
    for a, n in runner.ORDER:
        rr = []
        for e in shown:
            r = copy.deepcopy(rows[a])
            r["source_index"] = e["source_index"]
            if a == "control" and e["source_index"] == 0:
                # This is a saved-result/gate fixture, not a claim of measured production search.
                r["program"] = None
                r["termination"] = "max_depth"
                r["stages"][-1].update(program=None, termination="max_depth", depth=48)
            rr.append(r)
        cells[f"{a}-{n}"] = {
            "complete": True,
            "arm": a,
            "label": n,
            "rows": rr,
            "source": pins,
            "origins": imported,
            "checkpoint": model,
            "runtime": limits,
            "threads": 1,
            "interop": 1,
            "thread_env": dict.fromkeys(runner.THREAD_ENV, "1"),
            "hash_seed": "0",
            "device": "cpu",
            "model_parameter_dtypes": ["torch.float32"],
            "host": runner.platform.node(),
            "max_rss_kib": 1,
        }
    return (
        shown,
        originals,
        json.loads(json.dumps(cells)),
        pins,
        imported,
        model,
        limits,
    )


def test_exact_verified_set_gate_real_original_grader():
    shown, originals, cells, pins, imported, model, limits = synthetic_cells()
    clocks = dict.fromkeys(cells, 1.0)
    identities = [{"source_index": e["source_index"]} for e in shown]
    s = runner.paired_summary(
        cells, clocks, shown, originals, identities, pins, imported, model, limits
    )
    assert s["reading"] == "QUALIFY_RELEASE_COHORT" and all(
        p["gained"] == [0] and not p["lost"] for p in s["pairs"]
    )
    # Averaging cannot rescue a failed pair; a bad hidden fit is not a verified positive.
    cells["candidate-43"]["rows"][0]["program"] = None
    cells["candidate-43"]["rows"][0]["source"] = "none"
    cells["candidate-43"]["rows"][0]["termination"] = "wall_clock_cap"
    cells["candidate-43"]["rows"][0]["stages"][-1]["fit"] = False
    cells["candidate-43"]["rows"][0]["stages"][-1]["reason"] = "wall_clock_cap"
    # Keep an ordinary verification timeout structure: this fixture intentionally has no returned fit.
    s = runner.paired_summary(
        cells, clocks, shown, originals, identities, pins, imported, model, limits
    )
    assert s["reading"] == "PARK_FIXED_CONSTRUCTION"
    with pytest.raises(ValueError):
        runner.paired_summary(
            {}, clocks, shown, originals, identities, pins, imported, model, limits
        )


def test_phase_real_helpers_mocked_provider_source_runtime_children(
    tmp_path, monkeypatch
):
    from wandering_light import stack_pbe, stack_pbe_eval

    shown, originals, cells, pins, imported, model, limits = synthetic_cells()
    events = []
    monkeypatch.setattr(runner, "MAIN", tmp_path)
    monkeypatch.setattr(runner, "source_info", lambda: pins)
    monkeypatch.setattr(runner, "origins", lambda: imported)
    monkeypatch.setattr(runner, "runtime_limits", lambda: limits)
    monkeypatch.setattr(runner, "checkpoint_pins", lambda _: model)
    monkeypatch.setattr(runner, "host_snapshot", lambda: {"MOCK_HOST": True})
    monkeypatch.setattr(stack_pbe, "load", lambda _: originals)
    real_grade = stack_pbe_eval.grade

    def grade(*args):
        assert len(events) == 6, "grading before all timing cells"
        return real_grade(*args)

    monkeypatch.setattr(stack_pbe_eval, "grade", grade)

    def prepare(out):
        payload = out / "fixture-payload.txt"
        payload.write_text("synthetic only")
        monkeypatch.setattr(runner, "TRAIN_SHA", runner.sha(payload))
        return (
            payload,
            tmp_path / "mock-checkpoint",
            [{"source_index": e["source_index"]} for e in shown],
            shown,
        )

    monkeypatch.setattr(runner, "prepare_inputs", prepare)

    def child(cmd, **kwargs):
        arm = cmd[cmd.index("--arm") + 1]
        label = int(cmd[cmd.index("--label") + 1])
        name = f"{arm}-{label}"
        events.append((arm, label))
        c = copy.deepcopy(cells[name])
        c["shown_sha256"] = runner.sha(Path(cmd[cmd.index("--shown") + 1]))
        runner.save(Path(cmd[cmd.index("--out") + 1]), c)
        kwargs["stdout"].write("synthetic child (explicit process mock)\n")

    monkeypatch.setattr(runner.subprocess, "run", child)
    review = tmp_path / "review.log"
    review.write_text(
        json.dumps(
            {
                "subtype": "success",
                "is_error": False,
                "result": "CODE REVIEW PASS\nfixture",
            }
        )
        + "\nexit 0\n"
    )
    out = tmp_path / "checkpoints" / "mock-phase"
    runner.run_phase(out, review, "fixture")
    assert events == list(runner.ORDER)
    s = runner.read(out / "summary.json")
    assert s["complete"] and s["reading"] == "QUALIFY_RELEASE_COHORT"
    # Failed child cannot publish a complete summary or perform hidden grading.
    events.clear()

    def failing(cmd, **kwargs):
        if len(events) == 2:
            raise RuntimeError("synthetic child failure")
        return child(cmd, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", failing)
    out = tmp_path / "checkpoints" / "mock-incomplete"
    with pytest.raises(RuntimeError, match="child failure"):
        runner.run_phase(out, review, "fixture")
    assert (out / "INCOMPLETE.json").exists() and not (out / "summary.json").exists()
    # Replaying successfully does not authorize changed raw streams or review evidence.
    events.clear()
    monkeypatch.setattr(runner.subprocess, "run", child)
    real_summary = runner.paired_summary
    out = tmp_path / "checkpoints" / "mock-tampered"

    def tamper(*args):
        result = real_summary(*args)
        with (out / "control-17.log").open("a") as handle:
            handle.write("changed after replay")
        return result

    monkeypatch.setattr(runner, "paired_summary", tamper)
    with pytest.raises(ValueError, match="changed after replay"):
        runner.run_phase(out, review, "fixture")
    assert not (out / "summary.json").exists()


def test_actual_cell_handler_with_tiny_untrained_model_explicit_provenance_mocks(
    tmp_path, monkeypatch
):
    import torch

    from wandering_light import stack_policy
    from wandering_light.rollout_policy import PolicyArchitecture

    model = stack_policy.StackPairPolicy(
        PolicyArchitecture(byte_embedding=4, conv_channels=8, hidden=8)
    )
    for f in ("set_num_threads", "set_num_interop_threads"):
        monkeypatch.setattr(torch, f, lambda _: None)
    for f in ("get_num_threads", "get_num_interop_threads"):
        monkeypatch.setattr(torch, f, lambda: 1)
    for key in runner.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(runner, "source_info", lambda: {"MOCK_SOURCE": True})
    monkeypatch.setattr(runner, "runtime_limits", lambda: {"MOCK_RUNTIME": True})
    monkeypatch.setattr(runner, "checkpoint_pins", lambda _: {"MOCK_MODEL": True})
    monkeypatch.setattr(stack_policy, "load_policy", lambda *a, **k: model)
    monkeypatch.setattr(runner.gc, "freeze", lambda: None)
    # Real handler, real imported beam/scorer/encoding/interpreter; ONLY source/runtime/checkpoint acquisition mocked.
    shown = [
        {"source_index": i, "inputs": ["[ (0 1) ]"], "outputs": ["[ (1 2) ]"]}
        for i in range(24)
    ]
    p = tmp_path / "shown.json"
    runner.save(p, shown)
    out = tmp_path / "cell.json"
    runner.run_cell(p, tmp_path / "mock-checkpoint", out, "candidate", 17)
    c = runner.read(out)
    assert c["complete"] and len(c["rows"]) == 24
    assert all(
        r["source"] == "root_map" and r["termination"] == "fit" for r in c["rows"]
    )
    assert c["origins"]["functions"]["beam_solve"] == str(
        Path(__file__).resolve().parents[1] / "wandering_light/stack_beam.py"
    )


def test_original_grader_cumulative_fuel_can_reject_per_op_search_fit():
    from wandering_light.stack_pbe_eval import grade

    source = m.Lst((m.Int(0),) * 4999)
    target = m.Lst((m.Int(10),) * 4999)
    program = ("[", *(x for _ in range(10) for x in ("1", "add")), "]", "map")
    assert route.verify_wrapper(program, ((source,),), ((target,),), math.inf)["fit"]
    ins = m.format_stack(m.Config(stack=(source,)))
    outs = m.format_stack(m.Config(stack=(target,)))
    original = SimpleNamespace(
        inputs=[ins], outputs=[outs], held_out_inputs=[ins], held_out_outputs=[outs]
    )
    score = grade(program, original)
    assert score.parsed and not score.legal and not score.shown and not score.solved


def test_original_grader_failure_is_unverified_not_incomplete(monkeypatch):
    from wandering_light import stack_pbe_eval

    shown, originals, cells, pins, imported, model, limits = synthetic_cells()

    def grade(program, _task):
        return stack_pbe_eval.Score(
            tuple(program), parsed=True, legal=False, shown=False, held_out=False
        )

    monkeypatch.setattr(stack_pbe_eval, "grade", grade)
    summary = runner.paired_summary(
        cells,
        dict.fromkeys(cells, 1.0),
        shown,
        originals,
        [{"source_index": e["source_index"]} for e in shown],
        pins,
        imported,
        model,
        limits,
    )
    assert summary["reading"] == "PARK_FIXED_CONSTRUCTION"
    assert not any(summary["verified"].values())


def test_real_short_body_fits_but_wrapped_map_crosses_default_depth():
    code = m.Lst((m.Int(1),))
    for _ in range(15):
        code = m.Lst((code, m.Sym("i")))
    direct = m.step_raw(route.OPS["i"], (code,), m.DEFAULT_FUEL_LIMIT, 0)
    assert not isinstance(direct, str) and direct[0] == (m.Int(1),)
    source, target = m.Lst((code,)), m.Lst((m.Int(1),))
    wrapped = route.verify_wrapper(
        ("[", "i", "]", "map"), ((source,),), ((target,),), math.inf
    )
    assert (
        not wrapped["fit"]
        and wrapped["reason"] == "refusal"
        and "deep" in wrapped["refusal"]
    )
