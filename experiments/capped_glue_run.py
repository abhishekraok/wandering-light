"""The single bounded #178 D1+A diagnostic; requires independent pre-trial PASS.

Run from a frozen reviewed tree: python -m experiments.capped_glue_run.
Real CLI paths/limits are fixed. Tests inject synthetic paths and mock scorers.
Existing attempt markers forbid retry after any trial outcome or hard kill.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from experiments import capped_glue_loss as core
from experiments import capped_glue_metrics as metrics
from experiments import capped_glue_stage_a as stage_a
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_policy import (
    OP_NAMES,
    PolicyScorer,
    load_policy,
    node_encoder,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
MAIN = Path.home() / "repos" / "wandering-light"
CAMPAIGN = MAIN / "checkpoints" / "capped-glue-loss-20261003"
BATCH_SIZE = 32
SECONDS = 900


def require_review(receipt: Mapping, revision: str) -> None:
    if (
        receipt.get("verdict") != "PASS"
        or receipt.get("source_revision") != revision
        or receipt.get("issue") != 178
        or receipt.get("design") != "D1+AmendmentA"
        or receipt.get("reviewer_family") != "claude"
        or receipt.get("effort") != "high"
    ):
        raise core.InvalidDiagnosticError(
            "independent pre-trial PASS on exact source required"
        )
    report = receipt.get("report", {})
    if not report.get("path") or stage_a.sha256_file(
        Path(report["path"])
    ) != report.get("sha256"):
        raise core.InvalidDiagnosticError("independent pre-trial report hash required")


def default_loader(checkpoint: Path) -> PolicyScorer:
    return PolicyScorer(load_policy(checkpoint, device="cpu"), "cpu")


def check_checkpoint_pin(pin: Mapping, deadline: float) -> None:
    for name in sorted(stage_a.CHECKPOINT_FILES):
        if (
            stage_a.sha256_file(Path(pin["path"]) / name, deadline=deadline)
            != pin["hashes"][name]
        ):
            raise core.InvalidDiagnosticError(
                "checkpoint changed after provenance seal"
            )


def _blocks(
    rows: Sequence[Mapping], encoding: str, deadline: float
) -> list[np.ndarray]:
    blocks = []
    for row in rows:
        core.check_deadline(deadline)
        product = tuple(parse_stack(text, basis="mark-28") for text in row["product"])
        target = tuple(parse_stack(text, basis="mark-28") for text in row["target"])
        block = node_encoder(target, encoding)(product)
        if block.shape != (
            len(product),
            row["encoder_diagnostics"]["layouts"][encoding]["block_width"],
        ):
            raise core.InvalidDiagnosticError(
                "encoded block disagrees with sealed exposure"
            )
        blocks.append(block)
    return blocks


def score_checkpoint(
    stage: Mapping,
    pin: Mapping,
    out: Path,
    progress: dict,
    deadline: float,
    loader: Callable,
) -> dict:
    """One checkpoint, ordered sealed rows, fixed batches, durable raw vectors."""
    encoding, seed = pin["encoding"], pin["seed"]
    progress.update(
        phase="checkpoint_load", checkpoint={"encoding": encoding, "seed": seed}
    )
    core.check_deadline(deadline)
    check_checkpoint_pin(pin, deadline)
    progress["checkpoint_load_attempts"] += 1
    scorer = loader(Path(pin["path"]))
    progress["models_loaded"] += 1
    core.check_deadline(deadline)
    check_checkpoint_pin(pin, deadline)
    if (
        scorer.encoding != encoding
        or tuple(scorer.model.actions) != OP_NAMES
        or str(scorer.device) != "cpu"
    ):
        raise core.InvalidDiagnosticError(
            "loaded checkpoint encoding/actions/device mismatch"
        )
    raw_path = out / f"{encoding}-s{seed}-raw.jsonl"
    scored = []
    progress["phase"] = "checkpoint_logits"
    with raw_path.open("x", encoding="utf-8") as stream:
        for start in range(0, len(stage["rows"]), BATCH_SIZE):
            core.check_deadline(deadline)
            batch = stage["rows"][start : start + BATCH_SIZE]
            logits = np.asarray(scorer.logits(_blocks(batch, encoding, deadline)))
            core.check_deadline(deadline)
            if (
                logits.shape != (len(batch), len(OP_NAMES))
                or not np.isfinite(logits).all()
            ):
                raise core.InvalidDiagnosticError(
                    "incomplete/nonfinite checkpoint batch logits"
                )
            for ordinal, (row, vector) in enumerate(
                zip(batch, logits, strict=True), start
            ):
                core.check_deadline(deadline)
                diagnostic = metrics.loss_metrics(
                    vector, row["legality"]["mask"], row["op"]
                )
                raw = {
                    "encoding": encoding,
                    "seed": seed,
                    "ordinal": ordinal,
                    "behaviour": row["behaviour"],
                    "step": row["step"],
                    "op": row["op"],
                    "logits": vector.tolist(),
                    "metrics": diagnostic,
                }
                stream.write(stage_a.canonical_bytes(raw).decode("utf-8"))
                scored.append(raw)
                progress["logit_rows"] += 1
            stream.flush()
    del scorer
    core.check_deadline(deadline)
    progress["phase"] = "checkpoint_summary"
    summary = metrics.summarize_checkpoint(stage, scored, encoding=encoding, seed=seed)
    summary_path = out / f"{encoding}-s{seed}-summary.json"
    summary_hash = stage_a.atomic_new_json(summary_path, summary)
    raw_hash = stage_a.sha256_file(raw_path, deadline=deadline)
    progress["checkpoints_complete"] += 1
    return {
        "encoding": encoding,
        "seed": seed,
        "rows": len(scored),
        "raw": {"path": str(raw_path.resolve()), "sha256": raw_hash},
        "summary": {"path": str(summary_path.resolve()), "sha256": summary_hash},
    }


def verified_checkpoint_output(
    stage: Mapping, output: Mapping, deadline: float | None = None
) -> dict:
    """Recompute a complete checkpoint summary from persisted raw vectors."""
    for role in ("raw", "summary"):
        if (
            stage_a.sha256_file(Path(output[role]["path"]), deadline=deadline)
            != output[role]["sha256"]
        ):
            raise core.InvalidDiagnosticError("checkpoint output hash mismatch")
    scored = []
    with Path(output["raw"]["path"]).open(encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            core.check_deadline(deadline)
            row = json.loads(line)
            if ordinal >= len(stage["rows"]):
                raise core.InvalidDiagnosticError("extra raw checkpoint row")
            expected = stage["rows"][ordinal]
            if (
                row["ordinal"] != ordinal
                or (row["encoding"], row["seed"])
                != (output["encoding"], output["seed"])
                or (row["behaviour"], row["step"], row["op"])
                != (expected["behaviour"], expected["step"], expected["op"])
            ):
                raise core.InvalidDiagnosticError(
                    "raw descriptor/order/identity mismatch"
                )
            diagnostic = metrics.loss_metrics(
                row["logits"], expected["legality"]["mask"], expected["op"]
            )
            if stage_a.canonical_bytes(diagnostic) != stage_a.canonical_bytes(
                row["metrics"]
            ):
                raise core.InvalidDiagnosticError("raw metrics disagree with logits")
            scored.append(row)
    if len(scored) != len(stage["rows"]) or output["rows"] != len(scored):
        raise core.InvalidDiagnosticError(
            "persisted checkpoint row coverage incomplete"
        )
    summary = metrics.summarize_checkpoint(
        stage, scored, encoding=output["encoding"], seed=output["seed"]
    )
    saved = json.loads(Path(output["summary"]["path"]).read_text())
    if stage_a.canonical_bytes(summary) != stage_a.canonical_bytes(saved):
        raise core.InvalidDiagnosticError(
            "persisted checkpoint summary disagrees with raw logits"
        )
    core.check_deadline(deadline)
    return summary


def validate_complete_manifest(
    manifest: Mapping, *, deadline: float | None = None
) -> None:
    if manifest["complete"] is not True or manifest["exit_code"] != 0:
        raise core.InvalidDiagnosticError("not a complete diagnostic result")
    seal = stage_a.StageSeal(**manifest["stage_a"])
    stage = stage_a.read_sealed_stage_a(seal)
    if manifest["source_revision"] != stage["source_revision"]:
        raise core.InvalidDiagnosticError("terminal source differs from sealed source")
    outputs = manifest["checkpoints"]
    if stage["kind"] == "support_only":
        if (
            manifest["reading"] != "INSUFFICIENT_SUPPORT"
            or outputs
            or manifest["progress"]["models_loaded"] != 0
            or manifest["progress"]["logit_rows"] != 0
            or manifest["progress"]["checkpoint_load_attempts"] != 0
            or manifest["progress"]["checkpoints_complete"] != 0
            or manifest["result"] is not None
        ):
            raise core.InvalidDiagnosticError(
                "support-only terminal state contains model work"
            )
    else:
        if [(row["encoding"], row["seed"]) for row in outputs] != list(
            stage_a.CHECKPOINT_ORDER
        ):
            raise core.InvalidDiagnosticError("six checkpoint descriptors incomplete")
        if (
            manifest["progress"]["models_loaded"] != 6
            or manifest["progress"]["checkpoint_load_attempts"] != 6
            or manifest["progress"]["logit_rows"] != seal.rows * 6
            or manifest["progress"]["checkpoints_complete"] != 6
        ):
            raise core.InvalidDiagnosticError(
                "six checkpoint progress/counts incomplete"
            )
        summaries = [
            verified_checkpoint_output(stage, row, deadline) for row in outputs
        ]
        reading = metrics.six_checkpoint_reading(summaries)
        if (
            stage_a.canonical_bytes(reading)
            != stage_a.canonical_bytes(manifest["result"])
            or manifest["reading"] != reading["reading"]
        ):
            raise core.InvalidDiagnosticError(
                "terminal reading disagrees with complete logits"
            )
    core.check_deadline(deadline)


def run_diagnostic(
    spec_path: Path,
    repo: Path,
    out: Path,
    receipt: Mapping,
    revision: str,
    *,
    loader: Callable = default_loader,
) -> int:
    """One reserved attempt. Exceptions yield INVALID; kills yield no final file.

    Tests inject synthetic files/loaders. The actual CLI fixes all paths and has
    no retry/limit/selection override. A missing review or prior attempt refuses
    before data/model work and does not modify an earlier result.
    """
    require_review(receipt, revision)
    started = time.perf_counter()
    deadline = started + SECONDS
    out.mkdir(parents=True, exist_ok=True)
    for path in (
        out / "stage-a.json",
        out / "final.json",
        *out.glob("*-raw.jsonl"),
        *out.glob("*-summary.json"),
    ):
        if path.exists():
            raise FileExistsError(path)
    stage_a.atomic_new_json(
        out / "attempt.json",
        {"source_revision": revision, "review": dict(receipt), "retry_allowed": False},
    )
    progress = {
        "phase": "provenance",
        "checkpoint_load_attempts": 0,
        "models_loaded": 0,
        "logit_rows": 0,
        "checkpoints_complete": 0,
    }
    seal = None
    outputs = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        spec = json.loads(spec_path.read_text())
        provenance = stage_a.verify_provenance(spec, repo, deadline=deadline)
        provenance["spec_sha256"] = stage_a.sha256_file(spec_path, deadline=deadline)
        progress["phase"] = "input_load"
        tasks = stage_a.read_train(
            Path(provenance["inputs"]["train"]["path"]), deadline=deadline
        )
        programs = stage_a.read_side(
            Path(provenance["inputs"]["side"]["path"]), deadline=deadline
        )
        # Close the input read/hash interval before any selection or models.
        for pin in provenance["inputs"].values():
            if (
                stage_a.sha256_file(Path(pin["path"]), deadline=deadline)
                != pin["sha256"]
            ):
                raise core.InvalidDiagnosticError("input changed during loading")
        progress["phase"] = "stage_a"
        stage = stage_a.build_stage_a(
            tasks,
            programs,
            provenance,
            source_revision=revision,
            deadline=deadline,
            progress=progress,
        )
        core.check_deadline(deadline)
        seal = stage_a.seal_stage_a(stage, out / "stage-a.json")
        stage = stage_a.read_sealed_stage_a(seal)
        core.check_deadline(deadline)
        if seal.kind == "support_only":
            reading, result = "INSUFFICIENT_SUPPORT", None
        else:
            for pin in provenance["checkpoints"]:
                outputs.append(
                    score_checkpoint(stage, pin, out, progress, deadline, loader)
                )
            progress["phase"] = "complete_output_verification"
            summaries = [
                verified_checkpoint_output(stage, output, deadline)
                for output in outputs
            ]
            result = metrics.six_checkpoint_reading(summaries)
            reading = result["reading"]
        progress["phase"] = "final_manifest"
        manifest = {
            "source_revision": revision,
            "complete": True,
            "exit_code": 0,
            "reading": reading,
            "result": result,
            "stage_a": asdict(seal),
            "checkpoints": outputs,
            "progress": dict(progress),
            "elapsed_seconds": time.perf_counter() - started,
            "retry_allowed": False,
        }
        validate_complete_manifest(manifest, deadline=deadline)
        core.check_deadline(deadline)
        stage_a.atomic_new_json(out / "final.json", manifest)
        return 0
    except Exception as error:
        invalid = {
            "source_revision": revision,
            "complete": False,
            "exit_code": 2,
            "reading": "INVALID",
            "result": None,
            "error": {"type": type(error).__name__, "message": str(error)},
            "stage_a": asdict(seal) if seal is not None else None,
            "checkpoints": outputs,
            "progress": dict(progress),
            "elapsed_seconds": time.perf_counter() - started,
            "retry_allowed": False,
        }
        stage_a.atomic_new_json(out / "final.json", invalid)
        return 2
    finally:
        torch.set_num_threads(previous_threads)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty:
        raise core.InvalidDiagnosticError(
            "diagnostic source tree must be frozen and clean"
        )
    receipt = json.loads((CAMPAIGN / "pretrial-review" / "receipt.json").read_text())
    return run_diagnostic(
        REPO / "reports" / "capped-glue-loss-20261003" / "inputs.json",
        REPO,
        CAMPAIGN / "trial",
        receipt,
        revision,
    )


if __name__ == "__main__":
    raise SystemExit(main())
