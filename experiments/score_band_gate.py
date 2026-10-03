"""Single bounded, complete synthetic Gate A; NOT a real allocation campaign.

Execution requires separate pre-gate high-effort code PASS. Development tests
are distinct; this CLI writes an exclusive attempt before any fixture work and
an atomic complete manifest only after all fixed IDs finish. Never loads data,
models or checkpoints. No primary/comparative clock claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
from wandering_light import stack_beam
from wandering_light.stack_beam import beam_solve, replay_program
from wandering_light.stack_machine import Bool, Int, Lst, Mark, parse_stack
from wandering_light.stack_pbe import BASIS, OPS
from wandering_light.stack_policy import PolicyScorer

REFERENCE_SHA256 = "dcf4558a6cfa11d8b8dde9aaa02ff21c9837ec16279a692f9b7e723c94ec616b"
PARAMETERS = {
    "negative": -1,
    "nan": np.nan,
    "posinf": np.inf,
    "neginf": -np.inf,
    "bool": True,
    "string": "1",
}
ARITHMETIC = (
    "last-ranked-width",
    "last-ranked-width-all-inf",
    "width-before-band",
    "last-row-band",
)
SOURCE_PATHS = (
    "experiments/score_band_search.py",
    "experiments/score_band_fixtures.py",
    "experiments/score_band_checks.py",
    "experiments/score_band_gate.py",
    "tests/test_score_band_search.py",
    "tests/test_score_band_gate.py",
    "reports/score-band-contract-20261003/DESIGN.md",
    "reports/score-band-contract-20261003/AMENDMENT_A.md",
    "reports/score-band-contract-20261003/FIXTURES.md",
    "wandering_light/stack_beam.py",
    "wandering_light/stack_machine.py",
    "wandering_light/stack_pbe.py",
    "wandering_light/stack_policy.py",
    "wandering_light/quotation_guard.py",
)


def json_value(value):
    """Strict JSON preserving typed dataclasses and nonfinite support values."""
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            **{f.name: json_value(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (json_value(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True)
        )
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+inf" if value > 0 else "-inf"
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(json_value(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def catalog_ids():
    semantic = semantic_cases()
    ids = [c.id for c in (*disabled_cases(), *semantic, *resource_cases())]
    ids += [f"observation/{c.id}" for c in semantic]
    ids += [f"input/{name}" for name in TASK_TABLE]
    ids += [
        f"parameter/{name}/gc{int(state)}"
        for name in PARAMETERS
        for state in (False, True)
    ]
    ids += [f"exception/scorer/gc{int(state)}" for state in (False, True)]
    ids += [f"unit/{name}" for name in ARITHMETIC]
    ids += ["unit/scorer-dead-and-live", "source/production-reference"]
    assert len(ids) == len(set(ids)) == 455
    return tuple(ids)


def capture_summary(capture):
    return {
        "events": len(capture.events),
        "event_sha256": digest(capture.events),
        "attempts": sum(e[0] == "attempt" for e in capture.events),
        "layers": [
            {
                "depth": layer.depth,
                "retained": layer.retained,
                "parents": layer.parents,
                "verdicts_read_only": layer.verdicts,
                "ranked_count": len(layer.ranked),
                "ranked_sha256": digest(layer.ranked),
                "old_seen_sha256": digest(layer.old_seen),
                "anchor": layer.anchor,
                "tau": layer.tau,
                "stop": layer.stop,
                "first_band_rejected": layer.first_band_rejected,
                "all_inf_fallback": layer.all_inf_fallback,
                "width_pruned": layer.width_pruned,
                "band_pruned": layer.band_pruned,
            }
            for layer in capture.layers
        ],
    }


def run_search_case(case, category):
    task = manual_task(case.task)
    own = Capture()
    scorer = ScriptedScorer(case.script)
    common = {
        "id": case.id,
        "scope": "real-machine-software-QA",
        "definition": asdict(case),
        "task": asdict(task),
        "expected": case.expectation,
        "check": "PASS",
    }
    if case.expectation == "ValueError":
        try:
            band_beam_solve(
                task, scorer, observer=own, score_band_nats=case.band, **case.settings()
            )
        except ValueError as error:
            if "cumulative" not in str(error):
                raise AssertionError("unexpected ValueError") from error
        else:
            raise AssertionError("invalid executable score was accepted")
        assert any(e[0] == "scored" for e in own.events)
        assert not any(e[0] == "retained" for e in own.events)
        verify_attempts(own)
        try:
            band_beam_solve(
                task,
                ScriptedScorer(case.script),
                score_band_nats=case.band,
                **case.settings(),
            )
        except ValueError as error:
            if "cumulative" not in str(error):
                raise AssertionError("unexpected ValueError") from error
        else:
            raise AssertionError("unobserved invalid score was accepted")
        common.update(
            actual="ValueError after scored/before retained",
            candidate_trace=capture_summary(own),
        )
        observation = {
            "id": f"observation/{case.id}",
            "expected": "same ValueError",
            "actual": "ValueError",
            "check": "PASS",
        }
        return common, observation
    ref = Capture()
    reference = beam_solve(
        task, ScriptedScorer(case.script), observer=ref, **case.settings()
    )
    candidate = band_beam_solve(
        task, scorer, observer=own, score_band_nats=case.band, **case.settings()
    )
    result = candidate.beam_result
    if category == "semantic":
        check_semantic_expectation(case, reference, candidate, own, scorer, ref)
    else:
        assert result_key(result) == result_key(reference) and own.events == ref.events
        assert not candidate.band_pruned
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
            assert (
                result.termination == "budget"
                and result.executions == result.fuel == expected
            )
            assert result.expansions == 1 and scorer.calls == 0
        elif case.expectation == "past-deadline":
            assert result.termination == "wall_clock_cap" and scorer.calls == 0
            assert result.executions == result.fuel == 0 and result.expansions == 1
        elif case.id.startswith("resource/body/"):
            root_i = next(
                e
                for e in own.events
                if e[0] == "attempt" and e[1] == 0 and e[3] == OP_INDEX["i"]
            )
            assert root_i[6] > root_i[7] > 0
    verify_attempts(ref, reference)
    verify_attempts(own, result)
    verify_layers(own, case)
    common.update(
        reference=result_key(reference),
        candidate=result_key(result),
        band_pruned=candidate.band_pruned,
        width_pruned=candidate.width_pruned,
        scorer_calls=scorer.calls,
        candidate_trace=capture_summary(own),
        reference_trace=capture_summary(ref),
    )
    observation = None
    if category == "semantic":
        quiet = band_beam_solve(
            task,
            ScriptedScorer(case.script),
            score_band_nats=case.band,
            **case.settings(),
        )
        assert result_key(quiet.beam_result) == result_key(result)
        assert quiet.band_pruned == candidate.band_pruned
        observation = {
            "id": f"observation/{case.id}",
            "expected": "equal observed/unobserved result excluding seconds",
            "actual": result_key(quiet.beam_result),
            "band_pruned": quiet.band_pruned,
            "check": "PASS",
        }
    return common, observation


def check_input(name):
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
    return {
        "id": f"input/{name}",
        "task": asdict(task),
        "typed_inputs": inputs,
        "typed_outputs": outputs,
        "expected": "typed states and explicit manual witness replay",
        "actual": "PASS",
        "check": "PASS",
    }


def check_gc(name, collecting, tau=None):
    before = gc.isenabled()
    (gc.enable if collecting else gc.disable)()
    try:
        if name == "scorer":
            try:
                band_beam_solve(
                    manual_task("T-probe"),
                    ScriptedScorer(Script(raises=True)),
                    score_band_nats=1,
                    guard="off",
                )
            except RuntimeError as error:
                if "scorer" not in str(error):
                    raise AssertionError("unexpected RuntimeError") from error
            else:
                raise AssertionError("scorer exception missing")
            id = f"exception/scorer/gc{int(collecting)}"
        else:
            try:
                band_beam_solve(object(), object(), score_band_nats=tau)
            except ValueError as error:
                if "score_band_nats" not in str(error):
                    raise AssertionError("unexpected ValueError") from error
            else:
                raise AssertionError("invalid band accepted")
            id = f"parameter/{name}/gc{int(collecting)}"
        assert gc.isenabled() == collecting
        return {
            "id": id,
            "expected": "exception and GC restoration",
            "actual": "exception and unchanged GC",
            "collecting": collecting,
            "tau": tau,
            "check": "PASS",
        }
    finally:
        (gc.enable if before else gc.disable)()


def check_arithmetic(kind):
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
    assert layer.stop == stop and layer.width_pruned == (stop == "width")
    assert layer.band_pruned == (stop == "score_band")
    assert len(layer.verdicts) == (1 if stop == "width" else 2)
    return {
        "id": f"unit/{kind}",
        "scope": "arithmetic-only/no-solver-outcome",
        "input": {"scores": scores, "width": width, "tau": 1, "typed_rows": products},
        "expected": stop,
        "actual": layer,
        "check": "PASS",
    }


def check_api():
    import warnings

    stub = SimpleNamespace(logits=lambda blocks: np.zeros((2, len(OPS))))
    mask = np.zeros((2, len(OPS)), dtype=bool)
    mask[1, OP_INDEX["0"]] = True
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        values = PolicyScorer.log_probs(stub, [None, None], mask)
    messages = {str(w.message) for w in observed}
    assert messages == {
        "divide by zero encountered in log",
        "invalid value encountered in subtract",
    }
    assert np.isnan(values[0]).all() and values[1, OP_INDEX["0"]] == 0
    assert np.isneginf(values[1, ~mask[1]]).all()
    return {
        "id": "unit/scorer-dead-and-live",
        "scope": "scorer-API-only/no-model/no-solver-outcome",
        "mask": mask.tolist(),
        "actual": values.tolist(),
        "warnings": sorted(messages),
        "expected": "all-NaN dead row and live literal0 normalized0",
        "check": "PASS",
    }


def atomic_json(path, value, limit, *, deadline=None):
    raw = (
        json.dumps(
            json_value(value), indent=2, sort_keys=True, allow_nan=False
        ).encode()
        + b"\n"
    )
    if len(raw) > limit:
        raise ValueError("output byte bound exceeded")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("NO_GATE_RESULT: cooperative300s before atomic publication")
    os.replace(temporary, path)


def run_gate(out: Path, receipt_path: Path):
    repo = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip():
        raise ValueError("gate source tree must be clean")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("verdict") != "PRE-GATE CODE PASS"
        or receipt.get("source_revision") != source
    ):
        raise ValueError("exact-source pre-gate CODE PASS required")
    review_log = Path(receipt["run_log"])
    log = review_log.read_text()
    if (
        not log.endswith("exit 0\n")
        or digest_bytes(review_log) != receipt["run_log_sha256"]
    ):
        raise ValueError("pre-gate review log incomplete or changed")
    review = json.loads(log.rsplit("\nexit 0", 1)[0])
    if review.get("is_error") or "PRE-GATE CODE PASS" not in review.get("result", ""):
        raise ValueError("review did not grant pre-gate CODE PASS")
    if (
        hashlib.sha256(review["result"].strip().encode() + b"\n").hexdigest()
        != receipt["report_sha256"]
    ):
        raise ValueError("review report hash mismatch")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("CPU/thread1 environment required")
    for module in (stack_beam,):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise ValueError("production module resolved outside source tree")
    import experiments.score_band_search as candidate_module

    if not Path(candidate_module.__file__).resolve().is_relative_to(repo):
        raise ValueError("candidate module resolved outside source tree")
    hashes = {p: digest_bytes(repo / p) for p in SOURCE_PATHS}
    if hashes["wandering_light/stack_beam.py"] != REFERENCE_SHA256:
        raise ValueError("production reference changed")
    ids = catalog_ids()
    if out.exists():
        raise ValueError("attempt output already exists; no implicit retry")
    out.mkdir(parents=True)
    # Atomic exclusive reservation BEFORE fixture data work. A crash has no final.
    atomic_json(
        out / "attempt.json",
        {
            "source_revision": source,
            "source_hashes": hashes,
            "expected_ids": ids,
            "review_receipt_sha256": digest_bytes(receipt_path),
        },
        1_000_000,
    )
    started = time.perf_counter()
    results = []
    observations = []

    def bounded(call, *args):
        if time.perf_counter() - started > 300:
            raise TimeoutError("NO_GATE_RESULT: cooperative300s")
        return call(*args)

    try:
        for category, cases in (
            ("disabled", disabled_cases()),
            ("semantic", semantic_cases()),
            ("resource", resource_cases()),
        ):
            for case in cases:
                result, observation = bounded(run_search_case, case, category)
                results.append(result)
                if observation is not None:
                    observations.append(observation)
        results.extend(observations)
        for name in TASK_TABLE:
            results.append(bounded(check_input, name))
        for name, value in PARAMETERS.items():
            for collecting in (False, True):
                results.append(bounded(check_gc, name, collecting, value))
        for collecting in (False, True):
            results.append(bounded(check_gc, "scorer", collecting))
        for kind in ARITHMETIC:
            results.append(bounded(check_arithmetic, kind))
        results.append(bounded(check_api))
        results.append(
            {
                "id": "source/production-reference",
                "expected": REFERENCE_SHA256,
                "actual": hashes["wandering_light/stack_beam.py"],
                "check": "PASS",
            }
        )
        actual = tuple(r["id"] for r in results)
        if actual != ids or not all(r["check"] == "PASS" for r in results):
            raise ValueError("incomplete or altered fixed fixture catalog")
        # Hash source again and recheck cleanliness BEFORE the atomic final.
        if {p: digest_bytes(repo / p) for p in SOURCE_PATHS} != hashes:
            raise ValueError("source changed during gate")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip():
            raise ValueError("source tree changed during gate")
        if time.perf_counter() - started > 300:
            raise TimeoutError("NO_GATE_RESULT: cooperative300s")
        manifest = {
            "verdict": "SYNTHETIC_GATE_A_PASS",
            "complete": True,
            "scope": "software QA only; no real coverage/clock/allocation claim",
            "source_revision": source,
            "source_hashes": hashes,
            "ops_order": [op.name for op in OPS],
            "expected_ids": ids,
            "actual_ids": actual,
            "count": len(results),
            "results": results,
            "attempt_sha256": digest_bytes(out / "attempt.json"),
            "review_receipt_sha256": digest_bytes(receipt_path),
            "limits": {"seconds": 300, "output_bytes": 1_000_000_000, "cpu_threads": 1},
            "model_calls": 0,
            "corpus_reads": 0,
            "checkpoint_reads": 0,
        }
        atomic_json(out / "final.json", manifest, 999_000_000, deadline=started + 300)
        print(json.dumps({"verdict": manifest["verdict"], "count": len(results)}))
    except Exception as error:
        atomic_json(
            out / "failure.json",
            {
                "verdict": "NO_GATE_RESULT",
                "complete": False,
                "completed_ids": [r["id"] for r in results],
                "exception": type(error).__name__,
                "reason": str(error),
            },
            1_000_000,
        )
        raise


def digest_bytes(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pre-gate-receipt", type=Path, required=True)
    args = parser.parse_args()
    run_gate(args.out, args.pre_gate_receipt)


if __name__ == "__main__":
    main()
