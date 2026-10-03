"""End-to-end synthetic files and mock scorers; no existing checkpoint loading."""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import capped_glue_loss as core
from experiments import capped_glue_run as run
from experiments import capped_glue_stage_a as stage_a
from wandering_light.stack_pbe import Task
from wandering_light.stack_policy import OP_INDEX, OP_NAMES, validation_skeleton


@pytest.fixture
def inputs(tmp_path):
    def make(size=20):
        tasks = []
        index = 0
        while len(tasks) < size:
            task = Task(
                behaviour=f"fixture{index}",
                summary="synthetic",
                split="train",
                task_hash_split="train",
                inputs=("2",),
                outputs=("2",),
                held_out_inputs=("(",),
                held_out_outputs=(")",),
                program=("dup", "drop"),
                length=2,
                examples=1,
                held_out=1,
                fuel=0,
                tags=(),
                skeleton=f"group{index}",
            )
            if not validation_skeleton(task):
                tasks.append(task)
            index += 1
        train = tmp_path / "train.jsonl"
        train.write_text("".join(json.dumps(asdict(task)) + "\n" for task in tasks))
        side = tmp_path / "side.jsonl"
        side.write_text(
            "".join(
                json.dumps(
                    {
                        "behaviour": task.behaviour,
                        "program": "dup dup [ dup ] i drop drop [ drop ] i",
                    }
                )
                + "\n"
                for task in tasks
            )
        )
        spec = {"inputs": {}, "checkpoints": []}
        for role, path in (("train", train), ("side", side)):
            spec["inputs"][role] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        for encoding, seed in stage_a.CHECKPOINT_ORDER:
            path = tmp_path / f"{encoding}-s{seed}"
            path.mkdir()
            hashes = {}
            for name in stage_a.CHECKPOINT_FILES:
                data = f"mock {encoding} {seed} {name}".encode()
                (path / name).write_bytes(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
            spec["checkpoints"].append(
                {
                    "encoding": encoding,
                    "seed": seed,
                    "path": path.name,
                    "hashes": hashes,
                }
            )
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))
        receipt = {
            "verdict": "PASS",
            "source_revision": "fixture-source",
            "issue": 178,
            "design": "D1+AmendmentA",
            "reviewer_family": "claude",
            "effort": "high",
        }
        report = tmp_path / "synthetic-review.txt"
        report.write_text("Mock independent PASS for synthetic fixtures only.\n")
        receipt["report"] = {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        }
        return spec_path, tmp_path, tmp_path / "trial", receipt, "fixture-source"

    return make


def mock_loader(out, *, positive=False, failure=None):
    loads, batches = [], []

    class Scorer:
        def __init__(self, path):
            self.encoding = path.name.split("-s")[0]
            self.device = "cpu"
            self.model = SimpleNamespace(actions=OP_NAMES)
            self.offset = 0
            self.path = path

        def logits(self, blocks):
            batches.append((self.path.name, len(blocks)))
            if failure is not None:
                failure(self, blocks)
            logits = np.zeros((len(blocks), 28), dtype=np.float32)
            if positive:
                stage = json.loads((out / "stage-a.json").read_text())
                for index, row in enumerate(
                    stage["rows"][self.offset : self.offset + len(blocks)]
                ):
                    if row["side"] == "top":
                        logits[index, OP_INDEX[row["op"]]] = -1
            self.offset += len(blocks)
            return logits

    def load(path):
        # The very first load must occur after the complete seal is published.
        stage = json.loads((out / "stage-a.json").read_text())
        assert stage["complete"] and stage["kind"] == "supported"
        assert all(
            "legality" in row and "encoder_diagnostics" in row for row in stage["rows"]
        )
        loads.append(path.name)
        return Scorer(path)

    return load, loads, batches


def test_support_only_completes_without_any_checkpoint_load(inputs):
    args = inputs(19)

    def forbidden(path):
        pytest.fail("unsupported panel loaded checkpoint")

    assert run.run_diagnostic(*args, loader=forbidden) == 0
    manifest = json.loads((args[2] / "final.json").read_text())
    assert manifest["complete"] and manifest["reading"] == "INSUFFICIENT_SUPPORT"
    assert (
        manifest["progress"]["models_loaded"] == manifest["progress"]["logit_rows"] == 0
    )
    assert manifest["checkpoints"] == [] and manifest["result"] is None
    assert manifest["progress"]["stage_a_tasks_replayed"] == 19
    run.validate_complete_manifest(manifest)


@pytest.mark.parametrize(
    ("positive", "reading"),
    [(False, "NO_MATCHED_MIDSPAN_TOP_EXCESS"), (True, "TOP_LOSS_ROUTE")],
)
def test_six_once_fixed_batches_exact_rows_and_atomic_final(inputs, positive, reading):
    args = inputs()
    loader, loads, batches = mock_loader(args[2], positive=positive)
    assert run.run_diagnostic(*args, loader=loader) == 0
    manifest = json.loads((args[2] / "final.json").read_text())
    assert manifest["complete"] and manifest["reading"] == reading
    assert loads == [
        f"{encoding}-s{seed}" for encoding, seed in stage_a.CHECKPOINT_ORDER
    ]
    assert len(batches) == 48 and {size for _, size in batches} == {16, 32}
    assert (
        manifest["progress"]["models_loaded"]
        == manifest["progress"]["checkpoint_load_attempts"]
        == 6
    )
    assert manifest["progress"]["logit_rows"] == 240 * 6
    assert len(manifest["checkpoints"]) == 6 and all(
        row["rows"] == 240 for row in manifest["checkpoints"]
    )
    run.validate_complete_manifest(manifest)
    assert not list(args[2].glob(".*"))


def test_prior_attempt_is_never_overwritten_or_retried(inputs):
    args = inputs(1)
    assert run.run_diagnostic(*args) == 0
    old = {path.name: path.read_bytes() for path in args[2].iterdir()}
    with pytest.raises(FileExistsError):
        run.run_diagnostic(*args)
    assert {path.name: path.read_bytes() for path in args[2].iterdir()} == old


@pytest.mark.parametrize(
    "change",
    ["verdict", "source_revision", "issue", "design", "reviewer_family", "effort"],
)
def test_missing_or_wrong_pretrial_pass_refuses_before_trial(inputs, change):
    args = list(inputs(1))
    args[3][change] = "incorrect"
    with pytest.raises(core.InvalidDiagnosticError, match="pre-trial PASS"):
        run.run_diagnostic(*args)
    assert not args[2].exists()


def test_wrong_review_report_hash_refuses_before_trial(inputs):
    args = inputs(1)
    args[3]["report"]["sha256"] = "00" * 32
    with pytest.raises(core.InvalidDiagnosticError, match="report hash"):
        run.run_diagnostic(*args)
    assert not args[2].exists()


@pytest.mark.parametrize("change", ["nan", "width", "encoding", "actions", "device"])
def test_bad_scorer_or_batch_is_invalid_before_any_logit_rows(inputs, change):
    args = inputs()
    base, _, _ = mock_loader(args[2])

    def bad_loader(path):
        scorer = base(path)
        if change == "encoding":
            scorer.encoding = "incorrect"
        elif change == "actions":
            scorer.model.actions = (*OP_NAMES, "foreign")
        elif change == "device":
            scorer.device = "cuda"
        else:
            original = scorer.logits

            def bad_logits(blocks):
                values = original(blocks)
                if change == "nan":
                    values[0, 0] = float("nan")
                    return values
                return values[:, :-1]

            scorer.logits = bad_logits
        return scorer

    assert run.run_diagnostic(*args, loader=bad_loader) == 2
    manifest = json.loads((args[2] / "final.json").read_text())
    assert (
        manifest["reading"] == "INVALID"
        and not manifest["complete"]
        and manifest["result"] is None
    )
    assert (
        manifest["progress"]["models_loaded"] == 1
        and manifest["progress"]["logit_rows"] == 0
    )


def test_bad_input_hash_writes_invalid_with_no_effect_or_model_work(inputs):
    args = inputs()
    spec = json.loads(args[0].read_text())
    spec["inputs"]["train"]["sha256"] = "00" * 32
    args[0].write_text(json.dumps(spec))
    assert run.run_diagnostic(*args) == 2
    manifest = json.loads((args[2] / "final.json").read_text())
    assert manifest["reading"] == "INVALID" and not manifest["complete"]
    assert manifest["exit_code"] == 2 and manifest["result"] is None
    assert manifest["stage_a"] is None and manifest["progress"]["models_loaded"] == 0
    with pytest.raises(FileExistsError):
        run.run_diagnostic(*args)


def test_partial_second_checkpoint_failure_retains_progress_not_reading(inputs):
    args = inputs()

    def fail(scorer, blocks):
        if scorer.path.name == "machine-s1" and scorer.offset >= 32:
            raise core.InvalidDiagnosticError("fixture nonfinite logits")

    loader, _, _ = mock_loader(args[2], failure=fail)
    assert run.run_diagnostic(*args, loader=loader) == 2
    manifest = json.loads((args[2] / "final.json").read_text())
    assert (
        not manifest["complete"]
        and manifest["reading"] == "INVALID"
        and manifest["result"] is None
    )
    assert manifest["progress"]["models_loaded"] == 2
    assert (
        manifest["progress"]["logit_rows"] == 272
        and manifest["progress"]["checkpoints_complete"] == 1
    )
    assert (
        len(manifest["checkpoints"]) == 1
        and manifest["progress"]["phase"] == "checkpoint_logits"
    )


def test_deadline_after_backend_call_is_invalid_before_any_logits_saved(
    inputs, monkeypatch
):
    args = inputs()
    clock = {"seconds": 0.0}
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock["seconds"])

    def expire(scorer, blocks):
        clock["seconds"] = 901.0

    loader, _, _ = mock_loader(args[2], failure=expire)
    assert run.run_diagnostic(*args, loader=loader) == 2
    manifest = json.loads((args[2] / "final.json").read_text())
    assert manifest["reading"] == "INVALID" and manifest["result"] is None
    assert (
        "deadline" in manifest["error"]["message"]
        and manifest["elapsed_seconds"] == 901
    )
    assert (
        manifest["progress"]["models_loaded"] == 1
        and manifest["progress"]["logit_rows"] == 0
    )


def test_simulated_hard_kill_has_no_final_result_and_cannot_retry(inputs):
    args = inputs()

    def killed(scorer, blocks):
        raise KeyboardInterrupt("synthetic hard termination")

    loader, _, _ = mock_loader(args[2], failure=killed)
    with pytest.raises(KeyboardInterrupt):
        run.run_diagnostic(*args, loader=loader)
    assert (args[2] / "attempt.json").exists() and not (args[2] / "final.json").exists()
    with pytest.raises(FileExistsError):
        run.run_diagnostic(*args, loader=loader)


@pytest.mark.parametrize(
    "change",
    [
        "missing_row",
        "metric",
        "nonfinite",
        "summary",
        "descriptor",
        "progress",
        "source",
    ],
)
def test_final_verifier_rejects_tampering_even_with_updated_output_hash(inputs, change):
    args = inputs()
    loader, _, _ = mock_loader(args[2])
    assert run.run_diagnostic(*args, loader=loader) == 0
    manifest = json.loads((args[2] / "final.json").read_text())
    output = manifest["checkpoints"][0]
    if change in ("missing_row", "metric", "nonfinite"):
        path = Path(output["raw"]["path"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "missing_row":
            rows.pop()
        elif change == "metric":
            rows[0]["metrics"]["unmasked_nll"] += 1
        else:
            rows[0]["logits"][0] = float("nan")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output["raw"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "summary":
        path = Path(output["summary"]["path"])
        value = json.loads(path.read_text())
        value["primary"]["delta"] = 1
        path.write_text(json.dumps(value))
        output["summary"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "descriptor":
        manifest["checkpoints"][-1] = manifest["checkpoints"][0]
    elif change == "source":
        manifest["source_revision"] = "wrong-source"
    else:
        manifest["progress"]["logit_rows"] -= 1
    with pytest.raises(core.InvalidDiagnosticError):
        run.validate_complete_manifest(manifest)
