"""One fixed #180 reference diagnostic; COMPLETE review gates real entry.

No flag/limit/panel/checkpoint overrides in the CLI. Unit tests call the same
composition with fresh dummy paths and mock CPU scorers. Independent whole-run
verification is a separate AFTER-run entrypoint, never invoked here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from experiments import score_gap_provenance as provenance
from experiments.score_gap_analysis import analyze
from experiments.score_gap_inputs import (
    CHECKPOINT_FILES,
    InputSeal,
    _file_pin,
    _read_json,
    check_pins,
    hash_file,
    load_inputs,
    seal_inputs,
)
from experiments.score_gap_profiles import SEEDS
from experiments.score_gap_read import load_cell
from experiments.score_gap_recorder import CellRecorder
from experiments.score_gap_storage import (
    AttemptReceipts,
    DiagnosticProvenanceError,
    PayloadBudget,
    canonical_json,
)
from wandering_light import stack_beam
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_policy import OP_NAMES, PolicyScorer, load_policy

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = (
    Path.home() / "repos/wandering-light/checkpoints/score-gap-feasibility-20261003"
)
SPEC = REPO / provenance.REPORT_ROOT / "inputs.json"
REVIEW = CAMPAIGN / "code-review/receipt.json"
GLOBAL_SECONDS = 2700
SEARCH = {
    "beam_width": 64,
    "max_depth": 48,
    "guard": "both",
    "max_state_atoms": 16384,
    "budget": 200000,
    "budget_unit": "executions",
}
LIMITS = {
    "global_seconds": GLOBAL_SECONDS,
    "cell_seconds": 10,
    "payload_bytes": 2 * 1024**3,
    "attempts": 12000000,
    "process_memory_bytes": 4 * 1024**3,
    "external_runtime_seconds": 5400,
}
READINGS = {
    "INSUFFICIENT_SELECTION",
    "INSUFFICIENT_LINEAGE_SUPPORT",
    "NO_HEADROOM_OVER_PREFIX32",
    "NO_SELECTION_PROFILE",
    "NO_CONFIRMATION_PROFILE",
    "PASS_PROFILE_FEASIBILITY",
}


def default_loader(checkpoint):
    return PolicyScorer(load_policy(checkpoint, device="cpu"), "cpu")


def _output(path, value, budget):
    with path.open("xb") as stream:
        budget.write(stream, canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    return hash_file(path, budget)


def _checkpoint(pin, budget):
    for name in sorted(CHECKPOINT_FILES):
        if hash_file(Path(pin["path"]) / name, budget)["sha256"] != pin["hashes"][name]:
            raise DiagnosticProvenanceError(
                "checkpoint changed before/after load/search"
            )


def _scorer(scorer, previous):
    if (
        any(scorer is old or scorer.model is old.model for old in previous)
        or str(scorer.device) != "cpu"
        or scorer.encoding != "difference"
        or tuple(scorer.model.actions) != OP_NAMES
        or scorer.model.training is not False
        or any(
            str(parameter.device) != "cpu" for parameter in scorer.model.parameters()
        )
    ):
        raise DiagnosticProvenanceError(
            "scorers must be distinct eval-mode CPU difference mark28 policies"
        )


def _identity(item, seed):
    return {
        "position": item.position,
        "group": item.group,
        "half": item.half,
        "seed": seed,
        "behaviour": item.behaviour,
        "source_index": item.source_index,
    }


def _cells(panel, pins, root, budget, progress, loader):
    scorers = []
    for pin in pins["checkpoints"]:
        progress.update(phase="model_load", seed=pin["seed"])
        budget.check()
        _checkpoint(pin, budget)
        progress["model_load_attempts"] += 1
        scorer = loader(Path(pin["path"]))
        budget.check()
        _checkpoint(pin, budget)
        _scorer(scorer, scorers)
        scorers.append(scorer)
        progress["models_loaded"] += 1
    references = []
    for item in panel:
        for seed in SEEDS:
            identity = _identity(item, seed)
            directory = f"cells/g{item.position:03d}-s{seed}"
            progress.update(phase="reference_search", current=identity)
            budget.check()
            if beam_solve is not stack_beam.beam_solve:
                raise DiagnosticProvenanceError(
                    "reference must be imported production beam"
                )
            with CellRecorder(root / directory, budget, identity) as recorder:
                # Observer construction/identity IO is outside this per-cell
                # solve clock but inside the diagnostic2700s. Never min with
                # global deadline: global exhaustion propagates separately.
                deadline = time.perf_counter() + LIMITS["cell_seconds"]
                result = beam_solve(
                    item.solver,
                    scorers[seed],
                    **SEARCH,
                    deadline=deadline,
                    observer=recorder,
                )
                recorder.finish(result)
            budget.check()
            seal = hash_file(root / directory / "cell-seal.json", budget)
            references.append(
                {
                    "ordinal": len(references),
                    "identity": identity,
                    "directory": directory,
                    "seal": seal,
                    "result": asdict(result),
                }
            )
            progress["cells_sealed"] += 1
    for pin in pins["checkpoints"]:
        _checkpoint(pin, budget)
    for index, scorer in enumerate(scorers):
        _scorer(scorer, scorers[:index])
    del scorers
    return references


def _replay(panel, references, root, budget, progress):
    """No analysis/replay opens until the entire ordered144-cell set is sealed."""
    expected = [(item, seed) for item in panel for seed in SEEDS]
    if len(references) != 144 or len(expected) != 144:
        raise DiagnosticProvenanceError("all144 cell seals required before replay")
    # Validate every seal pin first; load_cell hashes its streams before decode.
    for reference in references:
        if (
            hash_file(root / reference["directory"] / "cell-seal.json", budget)
            != reference["seal"]
        ):
            raise DiagnosticProvenanceError("cell seal changed before all-cell replay")
    loaded = []
    for (item, seed), reference in zip(expected, references, strict=True):
        progress.update(phase="known_returned_replay", current=reference["identity"])
        budget.check()
        if reference["identity"] != _identity(item, seed):
            raise DiagnosticProvenanceError(
                "cell order differs from sealed original selection"
            )
        row = load_cell(
            root / reference["directory"],
            budget,
            reference["identity"],
            item.product,
            item.target,
        )
        if row.seal_sha256 != reference["seal"]["sha256"] or canonical_json(
            asdict(row.result)
        ) != canonical_json(reference["result"]):
            raise DiagnosticProvenanceError(
                "loaded cell differs from sealed production return"
            )
        reference["refusal_counts"] = row.refusal_counts
        loaded.append(row.cell)
        progress["cells_replayed"] += 1
    return tuple(loaded)


def _reserved(
    spec, repo, root, receipts, budget, source, review, runtime, progress, loader
):
    source_pin = _output(root / "source.json", source, budget)
    runtime_pin = _output(root / "runtime.json", runtime, budget)
    progress["phase"] = "shown_input_stage"
    input_seal = seal_inputs(spec, repo, receipts, budget, root / "inputs")
    manifest, panel = load_inputs(input_seal, budget)
    references, profile = [], None
    if input_seal.kind == "INSUFFICIENT_SELECTION":
        reading = input_seal.kind
        receipts.selection(
            {
                "schema": "SG-EMPTY-SELECTION1",
                "terminal": reading,
                "tau": None,
                "input_seal_sha256": input_seal.sha256,
            }
        )
    else:
        references = _cells(panel, manifest["pins"], root, budget, progress, loader)
        progress["phase"] = "all_reference_cells_sealed"
        budget.check()
        cells = _replay(panel, references, root, budget, progress)
        progress["phase"] = "profile_analysis"
        profile = analyze(
            cells,
            [item.group for item in panel],
            receipts,
            budget,
            root / "analysis.jsonl",
        )
        reading = profile["terminal"]
        if hash_file(root / "analysis.jsonl", budget) != {
            key: profile[key] for key in ("bytes", "sha256")
        }:
            raise DiagnosticProvenanceError(
                "persisted profile differs from sealed stream"
            )
    progress["phase"] = "final_provenance"
    if check_pins(manifest["pins"], budget) != manifest["checked_files"]:
        raise DiagnosticProvenanceError("original input/checkpoint pins changed")
    provenance.recheck_source(source, budget)
    if canonical_json(
        provenance.require_review(review["path"], source, budget)
    ) != canonical_json(review):
        raise DiagnosticProvenanceError(
            "complete-code review evidence changed during diagnostic"
        )
    provenance.check_runtime(runtime)
    # Reconsume typed input files to close their provenance interval.
    reread, repeated = load_inputs(input_seal, budget)
    if canonical_json(reread) != canonical_json(manifest) or repeated != panel:
        raise DiagnosticProvenanceError("selected original inputs changed")
    lock_pin = hash_file(root / "selection-lock.json", budget)
    budget.check()
    run = {
        "schema": "SG-RUN1",
        "complete": True,
        "exit_code": 0,
        "reading": reading,
        "source_revision": source["revision"],
        "source": source_pin,
        "runtime": runtime_pin,
        "review": {key: review[key] for key in ("path", "bytes", "sha256")},
        "input": {
            "directory": "inputs",
            "sha256": input_seal.sha256,
            "kind": input_seal.kind,
        },
        "cells": references,
        "profile": profile,
        "selection_lock": lock_pin,
        "limits": dict(LIMITS),
        "search": dict(SEARCH),
        "library": False,
        "restarts": False,
        "seeds": list(SEEDS),
        "ledger": {
            "bytes_before_manifest": budget.bytes_written,
            "attempts": budget.attempts_written,
        },
        "timing": {
            "global_started_perf_counter": budget.deadline - GLOBAL_SECONDS,
            "global_deadline_perf_counter": budget.deadline,
            "elapsed_before_manifest": budget.clock()
            - (budget.deadline - GLOBAL_SECONDS),
        },
        "progress": {**progress, "phase": "complete"},
        "independent_verification": "REQUIRED_AFTER_RUN_NOT_YET_PERFORMED",
        "confirmation_population": "familiar original TRAIN skeleton half",
        "scope": "same-snapshot positive-lineage protection/frontier slots",
        "coverage_clock_or_best_claim": False,
        "retry_allowed": False,
    }
    validate_run_shape(run)
    run_pin = _output(root / "run.json", run, budget)
    # The structural publication validator never calls fullattempt verification,
    # a model loader, or profile computation again.
    validate_publication(root, run, run_pin, budget)
    budget.check()
    receipts.terminal(
        "final",
        {
            "schema": "SG-FINAL1",
            "reading": reading,
            "run": run_pin,
            "payload_bytes": budget.bytes_written,
            "attempts": budget.attempts_written,
            "elapsed_seconds": budget.clock() - (budget.deadline - GLOBAL_SECONDS),
        },
    )
    budget.check()
    return run


def run_diagnostic(spec_path, repo, root, receipt_path, *, loader=default_loader):
    """Fixed resources; no real entry unless complete source/reviewer PASS.

    Tests inject freshly made dummy paths/mock loaders and mock code-only
    provenance. The actual CLI exposes none of those injections or overrides.
    """
    started = time.perf_counter()
    budget = PayloadBudget(deadline=started + GLOBAL_SECONDS)
    source = provenance.capture_source(repo, budget)
    review = provenance.require_review(receipt_path, source, budget)
    root = Path(root)
    receipts = AttemptReceipts(root)
    receipts.reserve(
        {
            "schema": "SG-ATTEMPT1",
            "source_revision": source["revision"],
            "source_sha256": hashlib.sha256(canonical_json(source)).hexdigest(),
            "review": {key: review[key] for key in ("path", "bytes", "sha256")},
            "started_pdt": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "retry_allowed": False,
        }
    )
    progress = {
        "phase": "reserved",
        "model_load_attempts": 0,
        "models_loaded": 0,
        "cells_sealed": 0,
        "cells_replayed": 0,
    }
    previous_threads = torch.get_num_threads()
    try:
        budget.check()
        runtime = provenance.configure_cpu()
        spec_path = Path(spec_path).resolve()
        relative = str(spec_path.relative_to(Path(source["repo"])))
        if hash_file(spec_path, budget) != source["files"].get(relative):
            raise DiagnosticProvenanceError(
                "input-spec bytes not covered by complete-code source review"
            )
        spec = json.loads(spec_path.read_bytes())
        if hash_file(spec_path, budget) != source["files"][relative]:
            raise DiagnosticProvenanceError("input-spec changed on read")
        run = _reserved(
            spec,
            Path(repo),
            root,
            receipts,
            budget,
            source,
            review,
            runtime,
            progress,
            loader,
        )
        print(
            json.dumps(
                {
                    "complete": True,
                    "reading": run["reading"],
                    "cells": len(run["cells"]),
                    "independent_verification": "required",
                }
            )
        )
        return 0
    except Exception as error:
        if not receipts.finished:
            receipts.terminal(
                "failure",
                {
                    "schema": "SG-FAIL1",
                    "complete": False,
                    "exception": type(error).__name__,
                    "reason": str(error)[:2000],
                    "progress": dict(progress),
                    "retry_allowed": False,
                },
            )
        print(json.dumps({"complete": False, "exception": type(error).__name__}))
        return 1
    finally:
        torch.set_num_threads(previous_threads)


def validate_run_shape(run):
    keys = {
        "schema",
        "complete",
        "exit_code",
        "reading",
        "source_revision",
        "source",
        "runtime",
        "review",
        "input",
        "cells",
        "profile",
        "selection_lock",
        "limits",
        "search",
        "library",
        "restarts",
        "seeds",
        "ledger",
        "timing",
        "progress",
        "independent_verification",
        "confirmation_population",
        "scope",
        "coverage_clock_or_best_claim",
        "retry_allowed",
    }
    if (
        type(run) is not dict
        or set(run) != keys
        or canonical_json(run.get("limits")) != canonical_json(LIMITS)
        or canonical_json(run.get("search")) != canonical_json(SEARCH)
        or canonical_json(run.get("seeds")) != canonical_json(list(SEEDS))
        or run.get("schema") != "SG-RUN1"
        or run.get("complete") is not True
        or type(run.get("exit_code")) is not int
        or run["exit_code"] != 0
        or run.get("reading") not in READINGS
        or run.get("limits") != LIMITS
        or run.get("search") != SEARCH
        or run.get("seeds") != list(SEEDS)
        or run.get("library") is not False
        or run.get("restarts") is not False
        or run.get("retry_allowed") is not False
        or run.get("coverage_clock_or_best_claim") is not False
        or run.get("independent_verification") != "REQUIRED_AFTER_RUN_NOT_YET_PERFORMED"
    ):
        raise DiagnosticProvenanceError("invalid global run contract")
    if (
        type(run["cells"]) is not list
        or set(run["input"]) != {"directory", "sha256", "kind"}
        or run["input"]["directory"] != "inputs"
    ):
        raise DiagnosticProvenanceError("invalid global input/cell schema")
    for name in ("source", "runtime", "selection_lock"):
        _file_pin(run[name])
    if (
        set(run["review"]) != {"path", "bytes", "sha256"}
        or type(run["review"]["path"]) is not str
    ):
        raise DiagnosticProvenanceError("invalid global review pointer")
    _file_pin({key: run["review"][key] for key in ("bytes", "sha256")})
    timing = run["timing"]
    if (
        set(timing)
        != {
            "global_started_perf_counter",
            "global_deadline_perf_counter",
            "elapsed_before_manifest",
        }
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in timing.values()
        )
        or timing["global_deadline_perf_counter"]
        - timing["global_started_perf_counter"]
        != GLOBAL_SECONDS
        or not 0 <= timing["elapsed_before_manifest"] < GLOBAL_SECONDS
    ):
        raise DiagnosticProvenanceError("invalid global diagnostic timing")
    if set(run["ledger"]) != {"bytes_before_manifest", "attempts"} or any(
        type(n) is not int or n < 0 for n in run["ledger"].values()
    ):
        raise DiagnosticProvenanceError("invalid global ledger schema")
    insufficient = run["reading"] == "INSUFFICIENT_SELECTION"
    if (
        len(run["cells"]) != (0 if insufficient else 144)
        or (run["profile"] is None) != insufficient
    ):
        raise DiagnosticProvenanceError("global cell/profile completeness mismatch")
    progress = run["progress"]
    if any(
        type(progress[name]) is not int
        for name in (
            "model_load_attempts",
            "models_loaded",
            "cells_sealed",
            "cells_replayed",
        )
    ):
        raise DiagnosticProvenanceError("global counters require native integers")
    if progress["phase"] != "complete" or [
        progress[name]
        for name in (
            "model_load_attempts",
            "models_loaded",
            "cells_sealed",
            "cells_replayed",
        )
    ] != ([0, 0, 0, 0] if insufficient else [3, 3, 144, 144]):
        raise DiagnosticProvenanceError("incomplete global phase counters")
    if insufficient:
        if run["input"]["kind"] != "INSUFFICIENT_SELECTION":
            raise DiagnosticProvenanceError(
                "insufficient selection without input terminal"
            )
    elif (
        run["input"]["kind"] != "selected"
        or run["profile"]["terminal"] != run["reading"]
    ):
        raise DiagnosticProvenanceError("profile/input terminal mismatch")
    for reference in run["cells"]:
        if (
            set(reference)
            != {"ordinal", "identity", "directory", "seal", "result", "refusal_counts"}
            or type(reference["ordinal"]) is not int
            or reference["ordinal"] < 0
        ):
            raise DiagnosticProvenanceError("invalid global cell reference schema")
        counts = reference["refusal_counts"]
        if set(counts) != {"success", "ordinary", "fuel", "depth"} or any(
            type(n) is not int or n < 0 for n in counts.values()
        ):
            raise DiagnosticProvenanceError("invalid global refusal histogram")


def validate_publication(root, run, run_pin, budget):
    """Strict closed outer linkage and persisted hashes; NOT full verifier."""
    validate_run_shape(run)
    if hash_file(root / "run.json", budget) != run_pin or canonical_json(
        _read_json(root / "run.json")
    ) != canonical_json(run):
        raise DiagnosticProvenanceError("published global manifest changed")
    for name in ("source", "runtime"):
        if hash_file(root / f"{name}.json", budget) != run[name]:
            raise DiagnosticProvenanceError("published source/runtime changed")
    source = _read_json(root / "source.json")
    if source["revision"] != run["source_revision"]:
        raise DiagnosticProvenanceError("global source revision linkage mismatch")
    input_manifest, panel = load_inputs(
        InputSeal(root / "inputs", run["input"]["sha256"], run["input"]["kind"]), budget
    )
    attempt = _read_json(root / "attempt.json")
    if (
        hash_file(root / "attempt.json", budget) != input_manifest["attempt"]
        or attempt["metadata"]["source_sha256"]
        != hashlib.sha256(canonical_json(source)).hexdigest()
        or attempt["metadata"]["review"] != run["review"]
        or attempt["metadata"]["source_revision"] != run["source_revision"]
    ):
        raise DiagnosticProvenanceError("attempt/input/source/review linkage mismatch")
    if hash_file(root / "selection-lock.json", budget) != run["selection_lock"]:
        raise DiagnosticProvenanceError("selection receipt changed")
    lock = _read_json(root / "selection-lock.json")
    if lock["kind"] != "selection":
        raise DiagnosticProvenanceError("invalid selection receipt kind")
    if run["profile"] is None:
        if lock["metadata"] != {
            "schema": "SG-EMPTY-SELECTION1",
            "terminal": "INSUFFICIENT_SELECTION",
            "tau": None,
            "input_seal_sha256": run["input"]["sha256"],
        }:
            raise DiagnosticProvenanceError("invalid insufficient-selection null lock")
    else:
        profile = run["profile"]
        if hash_file(root / "analysis.jsonl", budget) != {
            key: profile[key] for key in ("bytes", "sha256")
        }:
            raise DiagnosticProvenanceError("global analysis stream changed")
        if profile["tau"] != lock["metadata"]["tau"]:
            raise DiagnosticProvenanceError("global tau differs from immutable lock")
        n = lock["metadata"]["selection_prefix_bytes"]
        if type(n) is not int or not 0 < n <= profile["bytes"]:
            raise DiagnosticProvenanceError("invalid selection-prefix length")
        with (root / "analysis.jsonl").open("rb") as stream:
            prefix = stream.read(n)
        if (
            len(prefix) != n
            or hashlib.sha256(prefix).hexdigest()
            != lock["metadata"]["selection_prefix_sha256"]
        ):
            raise DiagnosticProvenanceError("locked selection-prefix hash mismatch")
        last, selection_events, terminals = None, 0, 0
        with (root / "analysis.jsonl").open("rb") as stream:
            for line in stream:
                budget.check()
                row = json.loads(line)
                if type(row) is not dict or canonical_json(row) != line:
                    raise DiagnosticProvenanceError("noncanonical analysis record")
                last = row
                if row.get("kind") == "selection_lock":
                    selection_events += 1
                    if canonical_json(
                        {key: value for key, value in row.items() if key != "kind"}
                    ) != canonical_json(lock["metadata"]):
                        raise DiagnosticProvenanceError(
                            "analysis selection event differs from immutable lock"
                        )
                if row.get("kind") == "terminal":
                    terminals += 1
        if (
            selection_events != 1
            or terminals != 1
            or last is None
            or canonical_json(last)
            != canonical_json(
                {
                    "kind": "terminal",
                    "terminal": profile["terminal"],
                    "tau": profile["tau"],
                    "confirmation_checks": profile["confirmation_checks"],
                    "confirmation_profile_opened": profile["tau"] is not None,
                }
            )
        ):
            raise DiagnosticProvenanceError(
                "analysis terminal differs from global profile"
            )
    expected = [(item, seed) for item in panel for seed in SEEDS]
    if len(expected) != len(run["cells"]):
        raise DiagnosticProvenanceError("input/cell population mismatch")
    for ordinal, ((item, seed), reference) in enumerate(
        zip(expected, run["cells"], strict=True)
    ):
        if (
            reference["ordinal"] != ordinal
            or canonical_json(reference["identity"])
            != canonical_json(_identity(item, seed))
            or reference["directory"] != f"cells/g{item.position:03d}-s{seed}"
        ):
            raise DiagnosticProvenanceError("global cell order/identity/path mismatch")
        directory = root / reference["directory"]
        if hash_file(directory / "cell-seal.json", budget) != reference["seal"]:
            raise DiagnosticProvenanceError("global cell seal changed")
        cell = _read_json(directory / "cell-seal.json")
        if sum(reference["refusal_counts"].values()) != cell["attempts"]:
            raise DiagnosticProvenanceError(
                "global refusal histogram/attempt count mismatch"
            )
        if canonical_json(cell["result"]) != canonical_json(reference["result"]):
            raise DiagnosticProvenanceError("global result differs from sealed cell")
        # Close every trace interval before final publication, without replaying
        # attempted ops or classifying reasons inside the bounded diagnostic.
        for name, pin in cell["files"].items():
            if hash_file(directory / name, budget) != pin:
                raise DiagnosticProvenanceError("trace stream changed after replay")
    expected_bytes = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
        and (
            (
                path.parent == root
                and path.name in ("source.json", "runtime.json", "analysis.jsonl")
            )
            or path.is_relative_to(root / "inputs")
            or path.is_relative_to(root / "cells")
        )
    )
    expected_attempts = sum(
        _read_json(root / ref["directory"] / "cell-seal.json")["attempts"]
        for ref in run["cells"]
    )
    if (
        run["ledger"]
        != {"bytes_before_manifest": expected_bytes, "attempts": expected_attempts}
        or budget.bytes_written != expected_bytes + run_pin["bytes"]
        or budget.attempts_written != expected_attempts
    ):
        raise DiagnosticProvenanceError("global payload/attempt ledger mismatch")
    budget.check()


def main():
    return run_diagnostic(SPEC, REPO, CAMPAIGN, REVIEW)


if __name__ == "__main__":
    raise SystemExit(main())
