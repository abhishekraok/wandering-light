from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import evaluate_rollout_policy as evaluation
from experiments.evaluate_rollout_policy import (
    EXPECTED_TEST_COHORT_DIGEST,
    LOCKED_WIDTHS,
    EvaluationConfig,
    _assert_cohort_contract,
    _checkpoint_identity,
    _log_wandb_run,
    _sha256_file,
    _validate_config,
    aggregate_rollout_rows,
    build_selection_lock,
    consume_test_selection_lock,
    create_selection_lock,
    evaluate_rollout_records,
    load_bfs_reference,
    matched_bfs_comparison,
    run_evaluation,
    select_validation_width,
    validate_test_selection_lock,
)
from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.typed_list import TypedList


@pytest.fixture
def arithmetic_basis() -> FunctionDefSet:
    return FunctionDefSet(
        [
            FunctionDef(
                name="inc",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x + 1",
            ),
            FunctionDef(
                name="double",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x * 2",
            ),
        ]
    )


def _record(
    distance: int,
    *,
    source_index: int,
    split: str = "validation",
) -> BasisTaskRecord:
    return BasisTaskRecord.create(
        split=split,
        input_value=TypedList([source_index], int),
        output_value=TypedList([source_index + 1], int),
        witness_function_ids=["bf:inc:test"],
        witness_function_names=["inc"],
        basis_set_id="test-basis-v1",
        basis_set_digest="sha256:" + "1" * 64,
        generator="unit-test",
        seed=7,
        source_index=source_index,
        metadata={
            "certified_distance": distance,
            "root_digest": f"sha256:{source_index:064x}",
        },
    )


def _records(*, per_distance: int = 1, split: str = "validation"):
    return [
        _record(
            distance,
            source_index=distance * 100 + offset,
            split=split,
        )
        for distance in range(1, 9)
        for offset in range(per_distance)
    ]


@dataclass
class _Result:
    function_defs: FunctionDefList
    executions: int
    termination: str
    trace: tuple = ()


def _rollout_row(record, width, *, success=False, executions=None):
    return {
        "task_id": record.task_id,
        "split": record.split,
        "certified_distance": record.metadata["certified_distance"],
        "root_digest": record.metadata["root_digest"],
        "beam_width": width,
        "success": success,
        "executions": executions if executions is not None else width * 10,
        "termination": "solved" if success else "max_depth",
        "solution_length": 1 if success else None,
        "solution_function_names": ["inc"] if success else [],
        "witness_exact_match": success,
        "seconds": 0.01,
        "diagnostics": None,
    }


def test_locked_config_requires_freeze_and_exact_protocol(tmp_path):
    validation = EvaluationConfig(
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        dry_run=True,
    )
    _validate_config(validation)

    with pytest.raises(ValueError, match="selection-lock"):
        _validate_config(
            EvaluationConfig(
                checkpoint=validation.checkpoint,
                output_dir=validation.output_dir,
                split="test",
                bfs_reference=tmp_path / "bfs.json",
            )
        )
    with pytest.raises(ValueError, match="bfs-reference"):
        _validate_config(
            EvaluationConfig(
                checkpoint=validation.checkpoint,
                output_dir=validation.output_dir,
                split="test",
                selection_lock=tmp_path / "lock.json",
            )
        )
    with pytest.raises(ValueError, match="protocol changed"):
        _validate_config(
            EvaluationConfig(
                checkpoint=validation.checkpoint,
                output_dir=validation.output_dir,
                widths=(1, 8),
            )
        )
    with pytest.raises(ValueError, match="dry runs are forbidden"):
        _validate_config(
            EvaluationConfig(
                checkpoint=validation.checkpoint,
                output_dir=validation.output_dir,
                split="test",
                selection_lock=tmp_path / "lock.json",
                bfs_reference=tmp_path / "bfs.json",
                dry_run=True,
            )
        )


def test_each_width_is_an_independent_costed_search(arithmetic_basis):
    records = [_record(1, source_index=1), _record(2, source_index=2)]
    inc = arithmetic_basis.name_to_function["inc"]
    factories = []

    class Predictor:
        def __init__(self, width):
            self.width = width

        def search_with_cost(self, input_list, output_list, available_functions):
            if self.width == 8:
                return _Result(FunctionDefList([inc]), self.width + 2, "solved")
            trace = (
                SimpleNamespace(
                    depth=1,
                    retained_paths=(("double",),),
                    unique_successors=2,
                ),
            )
            return _Result(FunctionDefList(), self.width + 2, "max_depth", trace)

    def factory(scorer, width, max_depth, budget):
        factories.append((id(scorer), width, max_depth, budget))
        return Predictor(width)

    scorer = object()
    rows = evaluate_rollout_records(
        records,
        scorer=scorer,
        available_functions=arithmetic_basis,
        widths=(1, 8),
        predictor_factory=factory,
    )

    assert [row["beam_width"] for row in rows] == [1, 1, 8, 8]
    assert [row["executions"] for row in rows] == [3, 3, 10, 10]
    assert [row["success"] for row in rows] == [False, False, True, True]
    assert factories == [
        (id(scorer), 1, 8, 100_000),
        (id(scorer), 8, 8, 100_000),
    ]
    assert rows[0]["diagnostics"]["depths"][0] == {
        "depth": 1,
        "retained_paths": [["double"]],
        "unique_successors": 2,
        "witness_prefix_retained": False,
    }
    assert rows[0]["diagnostics"]["witness_first_pruned_depth"] == 1
    assert rows[-1]["diagnostics"] == {
        "trace_supported": True,
        "depths": [],
        "observed_completed_layers": 0,
        "right_censored": True,
        "censoring_reason": "solved",
        "witness_first_pruned_depth": None,
        "witness_survived_all_observed_depths": True,
    }


def test_reported_solution_is_reexecuted(arithmetic_basis):
    record = _record(1, source_index=1)

    class LyingPredictor:
        def search_with_cost(self, *args):
            return _Result(FunctionDefList(), 0, "solved")

    with pytest.raises(ValueError, match="false solution"):
        evaluate_rollout_records(
            [record],
            scorer=object(),
            available_functions=arithmetic_basis,
            widths=(1,),
            predictor_factory=lambda *args: LyingPredictor(),
        )


def test_aggregate_requires_exact_width_task_cross_product():
    records = _records()
    rows = [
        _rollout_row(
            record,
            width,
            success=width >= 10 and record.metadata["certified_distance"] <= 5,
        )
        for width in LOCKED_WIDTHS
        for record in records
    ]

    summary, per_task = aggregate_rollout_rows(rows, records=records)

    assert summary["by_width"]["10"]["5-8"]["tasks"] == 4
    assert summary["by_width"]["10"]["5-8"]["solved"] == 1
    assert summary["by_width"]["64"]["7-8"]["solved"] == 0
    assert set(per_task) == set(LOCKED_WIDTHS)
    with pytest.raises(ValueError, match="width/task cross product"):
        aggregate_rollout_rows(rows[:-1], records=records)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_rollout_rows([*rows, rows[0]], records=records)


def _write_bfs_table(path: Path, records, *, cost=lambda record: 100):
    payload = {
        "columns": list(evaluation.BFS_REFERENCE_COLUMNS),
        "data": [
            [
                record.task_id,
                record.metadata["certified_distance"],
                True,
                cost(record),
                0.1,
            ]
            for record in records
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_bfs_reference_is_hash_and_task_guarded(tmp_path, monkeypatch):
    records = _records(split="test")
    path = tmp_path / "bfs.json"
    payload = _write_bfs_table(
        path,
        records,
        cost=lambda record: 100 * record.metadata["certified_distance"],
    )
    monkeypatch.setattr(
        evaluation,
        "EXPECTED_BFS_REFERENCE_SHA256",
        _sha256_file(path),
    )

    rows, identity = load_bfs_reference(path, records)

    assert len(rows) == 8
    assert identity["aggregate"]["7-8"]["mean_executions"] == 750
    payload["data"][0][1] = 8
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        evaluation,
        "EXPECTED_BFS_REFERENCE_SHA256",
        _sha256_file(path),
    )
    with pytest.raises(ValueError, match="distance mismatch"):
        load_bfs_reference(path, records)


def test_matched_bfs_uses_per_task_capped_spend():
    records = _records()
    rows = []
    for width in LOCKED_WIDTHS:
        for record in records:
            # Width 1 solves d7 only in the 7-8 group and spends 10 on each task.
            success = width == 1 and record.metadata["certified_distance"] == 7
            rows.append(_rollout_row(record, width, success=success, executions=10))
    rollout, per_task = aggregate_rollout_rows(rows, records=records)
    bfs = [
        {
            "task_id": record.task_id,
            "certified_distance": record.metadata["certified_distance"],
            "success": True,
            "executions": 100 * record.metadata["certified_distance"],
            "seconds": 0.1,
        }
        for record in records
    ]

    matched = matched_bfs_comparison(
        rollout=rollout,
        per_task=per_task,
        bfs_rows=bfs,
        records=records,
    )

    cell = matched["7-8"]["1"]
    assert cell["rollout_solved"] == 1
    assert cell["bfs_uniform_budget_for_at_least_rollout_solve_count"] == 700
    assert cell["bfs_total_executions_for_at_least_rollout_solve_count"] == 1_400
    assert cell["bfs_to_rollout_total_execution_ratio"] == 70
    assert cell["rollout_cheaper_than_bfs_at_equal_or_higher_solve_count"] is True


def _checkpoint_summary(tmp_path):
    files = []
    for name, payload in {
        "model.safetensors": b"weights",
        "policy_config.json": b"{}",
        "training_manifest.json": b"{}",
        "trainer_state.json": b"{}",
        "wandb_run.url": b"https://wandb.ai/abhishekraok-na/project/runs/run",
    }.items():
        path = tmp_path / name
        path.write_bytes(payload)
        files.append(
            {
                "name": name,
                "bytes": len(payload),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "path": str(tmp_path),
        "artifact_files": files,
        "training_manifest_sha256": next(
            row["sha256"] for row in files if row["name"] == "training_manifest.json"
        ),
        "weights": next(row for row in files if row["name"] == "model.safetensors"),
    }


def _validation_summary(tmp_path):
    by_width = {}
    for width in LOCKED_WIDTHS:
        solved = {1: 0, 8: 2, 10: 4, 64: 3}[width]
        row = {
            "tasks": 48,
            "solved": solved,
            "solve_rate": solved / 48,
            "mean_executions": width * 10,
        }
        by_width[str(width)] = {
            "5-8": dict(row),
            "7-8": {**row, "solved": min(solved, 2)},
            "all": {**row, "tasks": 96, "solved": solved + 10},
        }
    return {
        "status": "complete",
        "corpus": {
            "split": "validation",
            "manifest_digest": "sha256:" + "a" * 64,
            "basis_set_id": "wl-core-v1",
            "basis_set_digest": "sha256:" + "b" * 64,
        },
        "cohort": {
            "selection": "distance-stratified-within-frozen-root-split",
            "seed": 20260816,
            "tasks": 96,
            "task_ids_digest": "sha256:" + "c" * 64,
            "by_certified_distance": {str(distance): 12 for distance in range(1, 9)},
        },
        "checkpoint": _checkpoint_summary(tmp_path),
        "protocol": {
            "widths": list(LOCKED_WIDTHS),
            "seed": 20260816,
            "tasks_per_distance": 12,
            "max_depth": 8,
            "execution_budget": 100_000,
            "device": "cpu",
            "wandb_entity": "abhishekraok-na",
            "wandb_project": "wandering-light-deep-corpus-v1",
            "selection_rule": (
                "Lexicographic: distance-5-8 solved, distance-7-8 solved, "
                "all-distance solved, lower distance-5-8 mean executions, then "
                "narrower beam."
            ),
            "cost_definition": evaluation.COST_DEFINITION,
        },
        "source_code": {
            "runner": {
                "path": "experiments/evaluate_rollout_policy.py",
                "sha256": _sha256_file(Path(evaluation.__file__)),
            }
        },
        "wandb": {
            "entity": "abhishekraok-na",
            "project": "wandering-light-deep-corpus-v1",
            "url": (
                "https://wandb.ai/abhishekraok-na/"
                "wandering-light-deep-corpus-v1/runs/validation"
            ),
        },
        "rollout": {"widths": list(LOCKED_WIDTHS), "by_width": by_width},
    }


def test_validation_width_selection_and_exclusive_freeze(tmp_path):
    summary = _validation_summary(tmp_path)
    assert select_validation_width(summary) == 10
    lock = build_selection_lock(
        summary,
        validation_summary_sha256="sha256:" + "d" * 64,
        locked_at_utc="2026-08-27T00:00:00+00:00",
    )
    assert lock["champion"]["beam_width"] == 10
    assert lock["discipline"]["held_out_test_policy_scored"] is False

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "selection-lock.json"
    created = create_selection_lock(summary_path, output=output)
    assert json.loads(output.read_text(encoding="utf-8")) == created
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_selection_lock(summary_path, output=output)


def test_test_lock_freezes_checkpoint_protocol_and_is_single_use(tmp_path):
    summary = _validation_summary(tmp_path)
    lock = build_selection_lock(
        summary,
        validation_summary_sha256="sha256:" + "d" * 64,
        locked_at_utc="2026-08-27T00:00:00+00:00",
    )
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    config = EvaluationConfig(
        checkpoint=tmp_path,
        output_dir=tmp_path / "test-output",
        split="test",
        selection_lock=lock_path,
        bfs_reference=tmp_path / "bfs.json",
    )
    cohort = {
        "selection": "distance-stratified-within-frozen-root-split",
        "seed": 20260816,
        "tasks": 96,
        "task_ids_digest": EXPECTED_TEST_COHORT_DIGEST,
        "by_certified_distance": {str(distance): 12 for distance in range(1, 9)},
    }
    corpus = {
        "manifest_digest": "sha256:" + "a" * 64,
        "basis_set_id": "wl-core-v1",
        "basis_set_digest": "sha256:" + "b" * 64,
    }

    access = validate_test_selection_lock(
        lock_path,
        config=config,
        checkpoint_identity=summary["checkpoint"],
        corpus_identity=corpus,
        cohort_identity=cohort,
    )
    assert access["champion_beam_width"] == 10
    receipt = consume_test_selection_lock(
        lock_path,
        access=access,
        output_dir=config.output_dir,
    )
    assert receipt["selection_lock_sha256"] == access["selection_lock_sha256"]
    with pytest.raises(RuntimeError, match="already been consumed"):
        consume_test_selection_lock(
            lock_path,
            access=access,
            output_dir=config.output_dir,
        )


def test_test_cohort_digest_is_pinned():
    cohort = {
        "seed": 20260816,
        "tasks": 96,
        "task_ids_digest": EXPECTED_TEST_COHORT_DIGEST,
        "by_certified_distance": {str(distance): 12 for distance in range(1, 9)},
    }
    _assert_cohort_contract("test", cohort)
    with pytest.raises(ValueError, match="exact #50"):
        _assert_cohort_contract(
            "test", {**cohort, "task_ids_digest": "sha256:" + "0" * 64}
        )


def test_checkpoint_identity_checks_corpus_and_basis(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    policy_config = {
        "basis_set": {
            "basis_set_id": "wl-core-v1",
            "basis_set_digest": "sha256:" + "b" * 64,
        }
    }
    manifest = {
        "status": "complete",
        "source_corpus": {
            "manifest_digest": "sha256:" + "a" * 64,
            "basis_set_id": "wl-core-v1",
            "basis_set_digest": "sha256:" + "b" * 64,
            "task_roles": {
                "test": {"role": "held-out-evaluation-only; no labels materialized"}
            },
        },
        "prefix_examples": {"discovery": 10, "validation": 2, "test": 0},
        "wandb": {
            "entity": "abhishekraok-na",
            "project": "wandering-light-deep-corpus-v1",
            "url": (
                "https://wandb.ai/abhishekraok-na/"
                "wandering-light-deep-corpus-v1/runs/training"
            ),
        },
    }
    files = {
        "model.safetensors": b"weights",
        "policy_config.json": json.dumps(policy_config).encode(),
        "trainer_state.json": b"{}",
        "wandb_run.url": (
            b"https://wandb.ai/abhishekraok-na/"
            b"wandering-light-deep-corpus-v1/runs/training"
        ),
    }
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
    manifest["artifacts"] = {name: _sha256_file(checkpoint / name) for name in files}
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    identity = _checkpoint_identity(
        checkpoint,
        corpus_manifest_digest="sha256:" + "a" * 64,
        basis_set_id="wl-core-v1",
        basis_set_digest="sha256:" + "b" * 64,
    )
    assert identity["weights"]["name"] == "model.safetensors"
    with pytest.raises(ValueError, match="different corpus"):
        _checkpoint_identity(
            checkpoint,
            corpus_manifest_digest="sha256:" + "0" * 64,
            basis_set_id="wl-core-v1",
            basis_set_digest="sha256:" + "b" * 64,
        )


def test_wandb_logging_is_personal_and_carries_task_evidence(tmp_path, monkeypatch):
    import wandb

    records = _records()
    rows = [
        _rollout_row(
            record,
            width,
            success=width == 8 and record.metadata["certified_distance"] == 5,
        )
        for width in LOCKED_WIDTHS
        for record in records
    ]
    rollout, _ = aggregate_rollout_rows(rows, records=records)
    summary = _validation_summary(tmp_path)
    summary.update(
        {
            "rollout": rollout,
            "matched_compute": None,
            "bfs_reference": None,
            "primary": {"validation_selected_beam_width": 8},
        }
    )
    for name, payload in {
        "summary.json": summary,
        "cohort.json": summary["cohort"],
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "rollout_results.jsonl.gz").write_bytes(b"rows")

    class Artifact:
        def __init__(self):
            self.files = []

        def add_file(self, path, *, name):
            self.files.append((path, name))

    class Run:
        url = (
            "https://wandb.ai/abhishekraok-na/"
            "wandering-light-deep-corpus-v1/runs/rollout-validation"
        )
        id = "rollout-validation"

        def __init__(self):
            self.logs = []
            self.summary = {}

        def log(self, value):
            self.logs.append(value)

        def log_artifact(self, artifact):
            self.artifact = artifact

        def finish(self):
            self.finished = True

    run = Run()
    init_args = {}

    def init(**kwargs):
        init_args.update(kwargs)
        return run

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Table", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb, "Artifact", lambda **kwargs: Artifact())
    config = EvaluationConfig(
        checkpoint=tmp_path,
        output_dir=tmp_path,
        wandb_run_name="rollout-validation",
    )

    url = _log_wandb_run(
        config=config,
        summary=summary,
        rows=rows,
        output_dir=tmp_path,
    )

    assert url == run.url
    assert init_args["entity"] == "abhishekraok-na"
    assert "issue-55" in init_args["tags"]
    assert run.summary["validation_selected_beam_width"] == 8
    assert run.finished is True
    assert {name for _, name in run.artifact.files} == {
        "summary.json",
        "cohort.json",
        "rollout_results.jsonl.gz",
    }
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["wandb"]["url"] == run.url


def test_validation_dry_run_never_loads_or_scores_policy(tmp_path, monkeypatch):
    records = _records(per_distance=12)
    manifest = {
        "manifest_digest": "sha256:" + "a" * 64,
        "basis_set_id": "wl-core-v1",
        "basis_set_digest": "sha256:" + "b" * 64,
    }
    basis = SimpleNamespace()
    monkeypatch.setattr(
        evaluation,
        "verify_and_load_split",
        lambda corpus_dir, split: (manifest, basis, records, {"ok": True}),
    )
    monkeypatch.setattr(
        evaluation,
        "select_balanced_cohort",
        lambda records, tasks_per_distance, seed: records,
    )
    monkeypatch.setattr(
        evaluation,
        "_corpus_identity",
        lambda corpus_dir, manifest, split: {**manifest, "split": split},
    )
    monkeypatch.setattr(
        evaluation,
        "_checkpoint_identity",
        lambda *args, **kwargs: {
            "artifact_files": [],
            "training_manifest_sha256": "sha256:" + "d" * 64,
        },
    )
    monkeypatch.setattr(
        evaluation,
        "_source_identity",
        lambda: {
            "runner": {
                "path": "experiments/evaluate_rollout_policy.py",
                "sha256": "sha256:" + "e" * 64,
            }
        },
    )
    output = tmp_path / "output"

    summary = run_evaluation(
        EvaluationConfig(
            checkpoint=tmp_path / "checkpoint",
            output_dir=output,
            corpus_dir=tmp_path / "corpus",
            dry_run=True,
        ),
        scorer_loader=lambda *args: pytest.fail("dry run loaded the policy"),
    )

    assert summary["status"] == "dry-run"
    assert summary["cohort"]["tasks"] == 96
    assert not output.exists()
