from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from experiments import create_deep_corpus_v2_scratch_report as report
from experiments import publish_deep_corpus_v2_scratch_report as publisher
from experiments.evaluate_deep_corpus_v2_scratch import (
    SUMMARY_DIGEST_FIELD,
    _seal_summary,
    _training_outcomes,
)
from wandering_light.evals.deep_corpus_protocol import (
    ARM_IDS,
    begin_test_opening,
    build_selection_lock,
    complete_test_opening,
    new_test_opening_ledger,
    record_test_arm_result,
    sha256_file,
    sha256_json,
)
from wandering_light.training.deep_corpus_scratch import (
    DEFAULT_WANDB_PROJECT,
    LR_GRID,
    PERSONAL_WANDB_ENTITY,
    select_pilot_learning_rate,
)

PROTOCOL_DIGEST = "sha256:" + "1" * 64
CORPUS_DIGEST = "sha256:" + "2" * 64
TOKENIZER_DIGEST = "sha256:" + "3" * 64
FEASIBILITY_DIGEST = "sha256:" + "4" * 64
GIT_COMMIT = "a" * 40
PROBE_SOURCE_DIGEST = "sha256:" + "b" * 64
REAL_LOAD_TEST_ARMS = report._load_test_arms


def _metric(*, solved: int, tasks: int = 12) -> dict:
    return {
        "tasks": tasks,
        "solved": solved,
        "solve_rate": solved / tasks,
        "exact_matches": solved,
        "exact_match_rate": solved / tasks,
        "total_basis_executions": tasks * 10,
        "mean_basis_executions": 10.0,
        "mean_hit_basis_executions": 5.0 if solved else 0.0,
    }


def _funnel() -> dict:
    by_k = {}
    for sample_count in (1, 8, 64):
        candidates = 96 * sample_count
        unparseable = 96
        parseable = candidates - unparseable
        executable = max(0, parseable - 48)
        correct = min(24, executable)
        by_k[str(sample_count)] = {
            "tasks": 96,
            "sample_count": sample_count,
            "candidates": candidates,
            "parseable": parseable,
            "parseable_rate": parseable / candidates,
            "unparseable": unparseable,
            "unparseable_rate": unparseable / candidates,
            "executable": executable,
            "executable_rate": executable / candidates,
            "executable_given_parseable_rate": (
                executable / parseable if parseable else 0.0
            ),
            "type_mismatch": 24,
            "type_mismatch_rate": 24 / candidates,
            "type_mismatch_given_parseable_rate": (
                24 / parseable if parseable else 0.0
            ),
            "execution_failure_other": 24,
            "execution_failure_other_rate": 24 / candidates,
            "execution_correct": correct,
            "execution_correct_rate": correct / candidates,
            "correct_given_parseable_rate": correct / parseable if parseable else 0.0,
            "correct_given_executable_rate": correct / executable
            if executable
            else 0.0,
            "exact_match": correct,
            "exact_match_rate": correct / candidates,
            "tasks_solved": min(12, correct),
            "solve_rate": min(12, correct) / 96,
            "tasks_exact_match": min(12, correct),
            "exact_solve_rate": min(12, correct) / 96,
            "parse_failure_counts": {
                "empty": 48,
                "unknown_function": 48,
                "malformed_separator": 0,
                "truncation": 0,
                "other": 0,
            },
            "parse_failure_rates": {
                "empty": 48 / candidates,
                "unknown_function": 48 / candidates,
                "malformed_separator": 0.0,
                "truncation": 0.0,
                "other": 0.0,
            },
            "top_parse_failure_reasons": [
                {
                    "category": "unknown_function",
                    "unknown_function_names": ["bogus"],
                    "count": 48,
                    "candidate_rate": 48 / candidates,
                    "fraction_of_unparseable": 0.5,
                }
            ],
            "top_unknown_function_names": [{"name": "bogus", "count": 48}],
        }
    return {"by_k": by_k}


def _evaluation_summary(
    arm_id: str,
    *,
    split: str,
    checkpoint: dict,
    training: dict,
    target_solved: int,
) -> dict:
    by_distance = {
        str(distance): _metric(
            solved=(target_solved if distance >= 5 else min(12, target_solved + 1))
        )
        for distance in range(1, 9)
    }
    overall = {
        str(sample_count): {
            **_metric(solved=target_solved * 4, tasks=96),
        }
        for sample_count in (1, 8, 64)
    }
    matched = {
        "7-8": {
            "matched": {
                "64": {
                    "tasks": 24,
                    "solved": target_solved * 2,
                    "solve_rate": target_solved / 12,
                    "exact_matches": target_solved * 2,
                    "exact_match_rate": target_solved / 12,
                    "total_executions_spent": 240,
                    "mean_executions_spent": 10.0,
                    "bfs_uniform_budget_at_or_below_model_mean_execution_spend": 10,
                    "bfs_mean_executions_at_model_mean_execution_spend": 10.0,
                    "bfs_solved_at_model_mean_execution_spend": 0,
                    "bfs_solve_rate_at_model_mean_execution_spend": 0.0,
                    "bfs_uniform_budget_for_at_least_model_solve_count": 0,
                    "bfs_solved_at_matched_rate_budget": 0,
                    "bfs_total_executions_for_at_least_model_solve_count": 0,
                    "bfs_to_model_total_execution_ratio": None,
                    "model_cheaper_than_bfs_at_equal_or_higher_solve_count": False,
                }
            }
        }
    }
    bfs_by_distance = {
        str(distance): {
            "tasks": 12,
            "solved": 12,
            "solve_rate": 1.0,
            "total_executions": 1_200,
            "mean_executions": 100.0,
        }
        for distance in range(1, 9)
    }
    return _seal_summary(
        {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "arm_id": arm_id,
            "protocol_lock_sha256": PROTOCOL_DIGEST,
            "cohort_sha256": "sha256:" + ("5" if split == "validation" else "6") * 64,
            "checkpoint": checkpoint,
            "generation": {
                "runtime": {
                    "seconds": 1.0,
                    "samples": 6_144,
                    "samples_per_second": 100.0,
                    "peak_torch_allocated_bytes": 1_000,
                    "peak_torch_reserved_bytes": 1_100,
                }
            },
            "model": {
                "overall": overall,
                "by_certified_distance_at_64": by_distance,
            },
            "candidate_funnel": _funnel(),
            "bfs": {
                "overall": {
                    "tasks": 96,
                    "solved": 96,
                    "solve_rate": 1.0,
                    "total_basis_executions": 9_600,
                    "mean_basis_executions": 100.0,
                },
                "by_certified_distance": bfs_by_distance,
            },
            "matched_compute": matched,
            "training": training,
            "artifacts": {
                "candidates": {
                    "path": "model_candidates.jsonl.gz",
                    "bytes": 1,
                    "sha256": "sha256:" + "a" * 64,
                },
                "bfs": {
                    "path": "bfs_results.jsonl.gz",
                    "bytes": 1,
                    "sha256": "sha256:" + "b" * 64,
                },
            },
        }
    )


def _study_identity(arm_id: str, architecture: dict, protocol: dict) -> str:
    return sha256_json(
        {
            "architecture": architecture,
            "corpus_manifest_digest": CORPUS_DIGEST,
            "tokenizer_artifact_digest": TOKENIZER_DIGEST,
            "retained_training_task_ids_digest": "sha256:" + "7" * 64,
            "per_device_batch_size": protocol["per_device_train_batch_size"],
            "gradient_accumulation_steps": protocol["gradient_accumulation_steps"],
            "max_length": protocol["max_length"],
            "optimizer": protocol["optimizer"],
            "precision": protocol["precision"],
            "gradient_checkpointing": protocol["gradient_checkpointing"],
            "initialization_seed": 54_001,
            "data_order_seed": 54_002,
        }
    )


def _training_manifest(
    arm_id: str,
    *,
    phase: str,
    rate: float,
    gate: dict,
    best_loss: float,
) -> dict:
    index = ARM_IDS.index(arm_id) + 1
    architecture = {"arm_id": arm_id, "hidden_size": index * 32}
    protocol = {
        "phase": phase,
        "num_train_epochs": 0.25 if phase == "pilot" else 3.0,
        "learning_rate": rate,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_length": 2048,
        "optimizer": "adamw_torch",
        "precision": "bf16",
        "gradient_checkpointing": arm_id == "smollm2-360m-scratch",
    }
    initial = "sha256:" + str(index) * 64
    study = _study_identity(arm_id, architecture, protocol)
    return {
        "status": "complete",
        "phase": phase,
        "arm_id": arm_id,
        "architecture": architecture,
        "creation": {
            "creation_method": "AutoModelForCausalLM.from_config",
            "pretrained_weights_loaded": False,
        },
        "initial_state_digest": initial,
        "initialization_seed": 54_001,
        "data_order_seed": 54_002,
        "effective_config_digest": "sha256:" + "8" * 64,
        "exact_trainable_parameters": index * 10_000_000,
        "tokenizer_digest": TOKENIZER_DIGEST,
        "retained_training_task_ids_digest": "sha256:" + "7" * 64,
        "source_data": {
            "corpus_manifest_digest": CORPUS_DIGEST,
            "task_roles": {
                "discovery": {
                    "retained_task_ids_digest": "sha256:" + "7" * 64,
                    "retained_records": 100,
                }
            },
        },
        "training_protocol": protocol,
        "learning_rate": rate,
        "num_train_epochs": 0.25 if phase == "pilot" else 3.0,
        "best_validation_loss": best_loss,
        "selection_lock": {},
        "feasibility_gate": {
            "artifact_digest": FEASIBILITY_DIGEST,
            "arm_id": arm_id,
            "arm_gate": deepcopy(gate),
            "study_identity_digest": study,
        },
        "source_code": {"dirty": False},
        "workload": {
            "unique_training_examples": 100,
            "planned_example_presentations": 25 if phase == "pilot" else 300,
            "planned_combined_training_tokens": (
                10_000 if phase == "pilot" else 120_000
            ),
            "planned_completion_loss_tokens": (5_000 if phase == "pilot" else 60_000),
        },
        "trainer_state": {
            "global_step": 10 if phase == "pilot" else 120,
            "num_input_tokens_seen": 10_000 if phase == "pilot" else 120_000,
            "train_metrics": {"train_runtime": 10.0 if phase == "pilot" else 100.0},
            "loss_curve": [
                {
                    "step": 10 if phase == "pilot" else 120,
                    "epoch": 0.25 if phase == "pilot" else 3.0,
                    "preceding_train_window_loss": best_loss - 0.1,
                    "validation_loss": best_loss,
                }
            ],
        },
        "validation_trend": {"classification": "compute-limited"},
        "execution_environment": {
            "full_invocation_elapsed_seconds": (12.0 if phase == "pilot" else 120.0),
            "full_invocation_runtime_scope": ("unit-test run through artifact writes"),
            "peak_torch_allocated_bytes": 1_000,
            "peak_torch_reserved_bytes": 1_100,
            "peak_process_gpu_memory_bytes": 1_200,
            "process_gpu_memory": {
                "measurement": "nvidia-smi PID/GPU-UUID sampling",
                "gpu_uuid": "GPU-unit-test",
            },
            "physical_vram_bytes": 2_000,
        },
        "wandb": {
            "entity": PERSONAL_WANDB_ENTITY,
            "project": DEFAULT_WANDB_PROJECT,
            "run_name": f"{phase}-{arm_id}-{rate}",
            "url": (
                f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
                f"{DEFAULT_WANDB_PROJECT}/runs/{phase}{index}{str(rate).replace('.', '')}"
            ),
        },
    }


def _receipt(arm_id: str, split: str, summary: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "published",
        "split": split,
        "arm_id": arm_id,
        "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
        "wandb_receipt_sha256": "sha256:" + "c" * 64,
        "url": (
            f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
            f"{DEFAULT_WANDB_PROJECT}/runs/{split}{ARM_IDS.index(arm_id)}"
        ),
    }


@pytest.fixture
def study(tmp_path, monkeypatch):
    corpus = {
        "manifest_digest": CORPUS_DIGEST,
        "basis_set_id": "wl-core-v1",
        "basis_set_digest": "sha256:" + "d" * 64,
        "hub": {"repo_id": "abhishekraok/wandering-light-deep-corpus-v2"},
        "statistics": {"records": 313_500, "roots": 2_400},
        "target_contract": {"profile": "deep-corpus-v2-issue-54"},
    }
    protocol = {
        "protocol_lock_sha256": PROTOCOL_DIGEST,
        "allowed_arm_ids": list(ARM_IDS),
        "corpus": {"manifest": corpus},
        "cohorts": {
            "validation": {"cohort_sha256": "sha256:" + "5" * 64},
            "test": {"cohort_sha256": "sha256:" + "6" * 64},
        },
        "generation": {
            "sampling_seed": 20_260_826,
            "sample_prefixes": [1, 8, 64],
            "samples_per_task": 64,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 64,
            "batch_size": 32,
            "max_sequence_length": 2048,
            "do_sample": True,
            "generation_sha256": "sha256:" + "e" * 64,
        },
        "bfs": {
            "budget": 20_000_000,
            "max_depth": 8,
            "bfs_sha256": "sha256:" + "f" * 64,
        },
        "tokenizer": {"metadata": {"artifact_digest": TOKENIZER_DIGEST}},
        "evaluator_sources": {
            "source_sha256": "sha256:" + "0" * 64,
            "metadata": {"git_commit": GIT_COMMIT},
            "files": [
                {
                    "path": "experiments/probe_deep_corpus_v2_scratch.py",
                    "sha256": PROBE_SOURCE_DIGEST,
                }
            ],
        },
    }
    tokenizer_stats = {
        str(distance): {
            field: {
                "count": 10,
                "total": 100,
                "min": 1,
                "max": 20,
                "mean": 10.0,
                "p50": 10,
                "p90": 18,
                "p95": 19,
                "p99": 20,
                "histogram": {"10": 10},
            }
            for field in ("prompt", "completion", "completion_with_eos", "combined")
        }
        for distance in range(1, 9)
    }
    tokenizer = {
        "artifact_digest": TOKENIZER_DIGEST,
        "tokenizer": {
            "total_vocab_size": 600,
            "base_vocab_size": 512,
            "max_length": 2048,
        },
        "selection": {
            "selected_records": 101,
            "retained_records": 100,
            "excluded_records": 1,
            "statistics": {"retained": {"by_certified_distance": tokenizer_stats}},
            "excluded": [
                {
                    "task_id": "too-long",
                    "certified_distance": 8,
                    "prompt_tokens": 2_040,
                    "completion_tokens_with_eos": 20,
                    "combined_tokens": 2_061,
                    "reason": "combined_tokens_exceed_fixed_max_length",
                }
            ],
            "token_id_digests": {"retained": {"combined": "sha256:" + "9" * 64}},
        },
    }
    gates = {
        arm_id: {
            "launch_allowed": True,
            "route": "direct",
            "runtime_hard_stop_required": True,
            "baseline": {
                "projected_pilot_hours": 1.0,
                "projected_final_hours": 5.0,
                "projected_pilot_plus_final_hours": 6.0,
            },
        }
        for arm_id in ARM_IDS
    }
    pilots_by_arm = {}
    finals = {}
    feasibility_results = []
    pilot_paths = []
    for arm_id in ARM_IDS:
        pilots = [
            _training_manifest(
                arm_id,
                phase="pilot",
                rate=rate,
                gate=gates[arm_id],
                best_loss=0.5 + index / 10,
            )
            for index, rate in enumerate(LR_GRID)
        ]
        lock = select_pilot_learning_rate(arm_id, pilots)
        final = _training_manifest(
            arm_id,
            phase="final",
            rate=float(lock["selected_learning_rate"]),
            gate=gates[arm_id],
            best_loss=0.4,
        )
        final["selection_lock"] = lock
        pilots_by_arm[arm_id] = pilots
        finals[arm_id] = final
        feasibility_results.append(
            {
                "arm_id": arm_id,
                "study_identity_digest": final["feasibility_gate"][
                    "study_identity_digest"
                ],
                "throughput": {
                    "elapsed_seconds": 500.0,
                    "peak_torch_allocated_bytes": 1_000,
                    "peak_torch_reserved_bytes": 1_100,
                },
                "longest_batch": {
                    "passed": True,
                    "peak_torch_allocated_bytes": 1_000,
                    "peak_torch_reserved_bytes": 1_100,
                },
                "gpu": {"physical_vram_bytes": 2_000},
            }
        )
        for index, pilot in enumerate(pilots):
            path = tmp_path / f"pilot-{arm_id}-{index}.json"
            path.write_text(json.dumps(pilot), encoding="utf-8")
            pilot_paths.append(path)
    feasibility = {
        "artifact_digest": FEASIBILITY_DIGEST,
        "source_code": {
            "commit": GIT_COMMIT,
            "dirty": False,
            "probe_runner": {
                "path": "experiments/probe_deep_corpus_v2_scratch.py",
                "sha256": PROBE_SOURCE_DIGEST,
            },
        },
        "source_data": {
            "corpus_manifest_digest": CORPUS_DIGEST,
            "tokenizer_artifact_digest": TOKENIZER_DIGEST,
        },
        "launch_gate": {"launch_allowed": True, "per_arm": gates},
        "arm_results": feasibility_results,
    }

    validation = {}
    selection_locks = []
    test_summaries = {}
    for arm_id in ARM_IDS:
        index = ARM_IDS.index(arm_id) + 1
        checkpoint = {
            "path": f"/live/checkpoints/{arm_id}",
            "arm_id": arm_id,
            "checkpoint_sha256": "sha256:" + str(index) * 64,
            "exact_trainable_parameters": finals[arm_id]["exact_trainable_parameters"],
        }
        validation_summary = _evaluation_summary(
            arm_id,
            split="validation",
            checkpoint=checkpoint,
            training=_training_outcomes(finals[arm_id]),
            target_solved=index - 1,
        )
        lock = build_selection_lock(
            arm_id=arm_id,
            protocol_lock_sha256=PROTOCOL_DIGEST,
            checkpoint_identity=checkpoint,
            validation_selection={
                "validation_summary_sha256": validation_summary[SUMMARY_DIGEST_FIELD]
            },
        )
        receipt = _receipt(arm_id, "validation", validation_summary)
        validation[arm_id] = {
            "summary": validation_summary,
            "selection_lock": lock,
            "checkpoint": checkpoint,
            "training_manifest": finals[arm_id],
            "wandb_receipt": receipt,
        }
        selection_locks.append(lock)
        test_summaries[arm_id] = _evaluation_summary(
            arm_id,
            split="test",
            checkpoint=checkpoint,
            training=_training_outcomes(finals[arm_id]),
            target_solved=0,
        )

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    bfs_path = test_dir / report.BFS_FILENAME
    bfs_path.write_bytes(b"shared-bfs")
    bfs_identity = {
        "path": report.BFS_FILENAME,
        "bytes": bfs_path.stat().st_size,
        "sha256": sha256_file(bfs_path),
    }
    ledger = new_test_opening_ledger(
        opening_id="issue-54-test",
        protocol_lock_sha256=PROTOCOL_DIGEST,
        selection_locks=selection_locks,
    )
    ledger = begin_test_opening(ledger)
    for arm_id in ARM_IDS:
        summary = test_summaries[arm_id]
        ledger = record_test_arm_result(
            ledger,
            arm_id=arm_id,
            result_identity={
                "evaluation_summary_sha256": summary[SUMMARY_DIGEST_FIELD],
                "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
                "candidates_sha256": summary["artifacts"]["candidates"]["sha256"],
            },
        )
    ledger = complete_test_opening(ledger, shared_bfs_identity=bfs_identity)
    (test_dir / report.TEST_LEDGER_FILENAME).write_text(
        json.dumps(ledger), encoding="utf-8"
    )
    combined_body = {
        "schema_version": 1,
        "status": "complete",
        "split": "test",
        "protocol_lock_sha256": PROTOCOL_DIGEST,
        "test_cohort_sha256": protocol["cohorts"]["test"]["cohort_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "test_access_count": 1,
        "arm_summary_sha256_by_arm": {
            arm_id: test_summaries[arm_id][SUMMARY_DIGEST_FIELD] for arm_id in ARM_IDS
        },
        "candidate_funnels": {
            "arm_ids": list(ARM_IDS),
            "sample_prefixes": [1, 8, 64],
            "arms": {
                arm_id: test_summaries[arm_id]["candidate_funnel"] for arm_id in ARM_IDS
            },
        },
        "shared_bfs": bfs_identity,
        "arms": {
            arm_id: {
                "model": test_summaries[arm_id]["model"],
                "bfs": test_summaries[arm_id]["bfs"],
                "matched_compute": test_summaries[arm_id]["matched_compute"],
                "training": test_summaries[arm_id]["training"],
                "generation_runtime": test_summaries[arm_id]["generation"]["runtime"],
            }
            for arm_id in ARM_IDS
        },
    }
    combined = _seal_summary(combined_body)
    (test_dir / report.TEST_SUMMARY_FILENAME).write_text(
        json.dumps(combined), encoding="utf-8"
    )

    corpus_path = tmp_path / "manifest.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    feasibility_path = tmp_path / "feasibility.json"
    feasibility_path.write_text("{}", encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    inputs = report.ReportInputs(
        protocol_lock=protocol_path,
        corpus_manifest=corpus_path,
        tokenizer_dir=tokenizer_dir,
        feasibility_manifest=feasibility_path,
        validation_results=tuple(tmp_path / arm_id for arm_id in ARM_IDS),
        pilot_manifests=tuple(reversed(pilot_paths)),
        test_dir=test_dir,
        output_dir=tmp_path / "report",
        repo_root=tmp_path,
    )
    test_arms = {
        arm_id: {
            "summary": test_summaries[arm_id],
            "wandb_receipt": _receipt(arm_id, "test", test_summaries[arm_id]),
            "directory": test_dir / "arms" / arm_id,
        }
        for arm_id in ARM_IDS
    }
    monkeypatch.setattr(report, "load_protocol_lock", lambda *_, **__: protocol)
    monkeypatch.setattr(report, "validate_frozen_tokenizer", lambda *_, **__: tokenizer)
    monkeypatch.setattr(report, "load_probe_manifest", lambda *_, **__: feasibility)
    monkeypatch.setattr(
        report,
        "load_validation_results",
        lambda *_, **__: (validation, selection_locks),
    )
    monkeypatch.setattr(report, "_load_test_arms", lambda *_, **__: test_arms)
    return {
        "inputs": inputs,
        "protocol": protocol,
        "validation": validation,
        "test_summaries": test_summaries,
        "test_arms": test_arms,
        "combined": combined,
        "ledger": ledger,
        "feasibility": feasibility,
        "pilots": pilots_by_arm,
        "pilot_paths": pilot_paths,
    }


def test_builds_atomic_deterministic_complete_json_and_markdown(study):
    inputs = study["inputs"]

    first_report, first_markdown = report.build_report(inputs)
    second_report, second_markdown = report.build_report(inputs)

    assert first_report == second_report
    assert first_markdown == second_markdown
    assert first_report["test_opening"]["test_access_count"] == 1
    assert first_report["decision_rules"]["all_models_zero_at_each_test_distance_5_8"]
    assert "0/12 at every distance from 5 through 8" in first_markdown
    assert (
        "one-seed bounded feasibility comparison, not a scaling law" in first_markdown
    )
    assert "Every excluded discovery row" in first_markdown
    assert "unknown_function:bogus" in first_markdown
    assert "Exhaustive BFS measured cost" in first_markdown
    assert "Matched basis-execution comparison" in first_markdown
    assert "Generation runtime and memory diagnostics" in first_markdown
    assert (
        len(
            [
                run["wandb_url"]
                for arm in first_report["training"].values()
                for run in [*arm["pilots"], arm["final"]]
            ]
        )
        == 12
    )

    result = report.create_report(inputs)
    assert result == first_report
    assert (inputs.output_dir / report.REPORT_JSON_FILENAME).is_file()
    assert (inputs.output_dir / report.REPORT_MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    ) == first_markdown
    with pytest.raises(FileExistsError, match="fresh directory"):
        report.create_report(inputs)


def test_publishes_sealed_report_to_fixed_personal_wandb_project(
    study, tmp_path, monkeypatch
):
    inputs = study["inputs"]
    report.create_report(inputs)
    observed = {"reports": []}

    class FakeMarkdownBlock:
        def __init__(self, *, text):
            self.text = text

    class FakeReport:
        def __init__(self, **kwargs):
            observed["reports"].append(kwargs)

        def save(self):
            return types.SimpleNamespace(
                url=(
                    f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
                    f"{DEFAULT_WANDB_PROJECT}/reports/issue-54--sealed"
                )
            )

    workspaces = types.ModuleType("wandb_workspaces")
    workspaces.__path__ = []
    reports = types.ModuleType("wandb_workspaces.reports")
    reports.__path__ = []
    v2 = types.ModuleType("wandb_workspaces.reports.v2")
    v2.MarkdownBlock = FakeMarkdownBlock
    v2.Report = FakeReport
    workspaces.reports = reports
    reports.v2 = v2
    monkeypatch.setitem(sys.modules, "wandb_workspaces", workspaces)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports", reports)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports.v2", v2)

    receipt_path = tmp_path / "wandb-report-receipt.json"
    receipt = publisher.publish_report(
        report_dir=inputs.output_dir,
        receipt_path=receipt_path,
    )
    assert receipt["entity"] == PERSONAL_WANDB_ENTITY
    assert receipt["project"] == DEFAULT_WANDB_PROJECT
    assert (
        receipt["report_sha256"]
        == json.loads(
            (inputs.output_dir / report.REPORT_JSON_FILENAME).read_text(
                encoding="utf-8"
            )
        )[report.REPORT_DIGEST_FIELD]
    )
    assert len(observed["reports"]) == 1
    assert observed["reports"][0]["blocks"][0].text.startswith(
        "# Deep corpus v2 scratch scaling report"
    )

    assert (
        publisher.publish_report(
            report_dir=inputs.output_dir,
            receipt_path=receipt_path,
        )
        == receipt
    )
    assert len(observed["reports"]) == 1

    class FailingReport(FakeReport):
        def save(self):
            raise ConnectionError("network failed after intent")

    v2.Report = FailingReport
    uncertain_receipt = tmp_path / "uncertain-wandb-report-receipt.json"
    with pytest.raises(ConnectionError, match="network failed"):
        publisher.publish_report(
            report_dir=inputs.output_dir,
            receipt_path=uncertain_receipt,
        )
    with pytest.raises(RuntimeError, match="may already have succeeded"):
        publisher.publish_report(
            report_dir=inputs.output_dir,
            receipt_path=uncertain_receipt,
        )
    recovered = publisher.publish_report(
        report_dir=inputs.output_dir,
        receipt_path=uncertain_receipt,
        recovered_url=(
            f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
            f"{DEFAULT_WANDB_PROJECT}/reports/issue-54--recovered"
        ),
    )
    assert recovered["status"] == "published"

    markdown_path = inputs.output_dir / report.REPORT_MARKDOWN_FILENAME
    markdown_path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown differs"):
        publisher.publish_report(
            report_dir=inputs.output_dir,
            receipt_path=receipt_path,
        )


def test_requires_exactly_nine_digest_matched_pilots(study):
    inputs = study["inputs"]
    missing = report.ReportInputs(
        **{**inputs.__dict__, "pilot_manifests": inputs.pilot_manifests[:-1]}
    )
    with pytest.raises(ValueError, match="Exactly nine"):
        report.build_report(missing)

    pilot_path = inputs.pilot_manifests[0]
    changed = json.loads(pilot_path.read_text(encoding="utf-8"))
    changed["trainer_state"]["global_step"] += 1
    pilot_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="digest differs"):
        report.build_report(inputs)


def test_rejects_training_wandb_and_feasibility_study_mismatch(study):
    inputs = study["inputs"]
    pilot_path = inputs.pilot_manifests[0]
    changed = json.loads(pilot_path.read_text(encoding="utf-8"))
    changed["wandb"]["entity"] = "wrong"
    final = study["validation"][changed["arm_id"]]["training_manifest"]
    rate = float(changed["learning_rate"])
    lock_row = next(
        row for row in final["selection_lock"]["pilots"] if row["learning_rate"] == rate
    )
    lock_row["manifest_digest"] = sha256_json(changed)
    if final["selection_lock"]["selected_learning_rate"] == rate:
        final["selection_lock"]["selected_pilot_manifest_digest"] = sha256_json(changed)
    pilot_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="training W&B identity"):
        report.build_report(inputs)

    changed["wandb"]["entity"] = PERSONAL_WANDB_ENTITY
    changed["feasibility_gate"]["study_identity_digest"] = "sha256:" + "0" * 64
    lock_row["manifest_digest"] = sha256_json(changed)
    if final["selection_lock"]["selected_learning_rate"] == rate:
        final["selection_lock"]["selected_pilot_manifest_digest"] = sha256_json(changed)
    pilot_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="study identity"):
        report.build_report(inputs)


def test_rejects_feasibility_probe_from_another_source(study):
    feasibility = study["feasibility"]
    feasibility["source_code"]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="outside the frozen commit"):
        report.build_report(study["inputs"])

    feasibility["source_code"]["commit"] = GIT_COMMIT
    feasibility["source_code"]["probe_runner"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="runner differs"):
        report.build_report(study["inputs"])


def test_load_test_arms_requires_all_receipts_and_rebuilds_combined(study, monkeypatch):
    test_dir = study["inputs"].test_dir
    combined = study["combined"]
    ledger = study["ledger"]
    validation = study["validation"]
    summaries = study["test_summaries"]
    calls = []
    monkeypatch.setattr(report, "_load_test_arms", REAL_LOAD_TEST_ARMS)
    monkeypatch.setattr(
        report,
        "_read_json",
        lambda path, **_: summaries[Path(path).parent.name],
    )
    monkeypatch.setattr(report, "validate_evaluation_summary", lambda value, **_: value)
    monkeypatch.setattr(
        report, "_validate_raw_artifacts", lambda *_args, **_kwargs: None
    )

    def receipt(root, *, summary):
        calls.append((Path(root).name, summary["arm_id"]))
        return _receipt(summary["arm_id"], "test", summary)

    monkeypatch.setattr(report, "validate_wandb_receipt", receipt)
    loaded = report._load_test_arms(
        test_dir,
        protocol=study["protocol"],
        combined=combined,
        validation=validation,
        ledger=ledger,
    )
    assert set(loaded) == set(ARM_IDS)
    assert calls == [(arm_id, arm_id) for arm_id in ARM_IDS]

    changed = deepcopy(combined)
    changed["candidate_funnels"]["arms"][ARM_IDS[0]]["by_k"]["64"]["unparseable"] += 1
    with pytest.raises(ValueError, match="candidate funnel"):
        report._load_test_arms(
            test_dir,
            protocol=study["protocol"],
            combined=changed,
            validation=validation,
            ledger=ledger,
        )


def test_report_seal_and_markdown_digest_reject_mutation(study, tmp_path):
    value, markdown = report.build_report(study["inputs"])
    changed = deepcopy(value)
    changed["decision_rules"]["seed_caveat"] = "scaling law"
    with pytest.raises(ValueError, match="changed after it was sealed"):
        report.validate_report(changed)
    with pytest.raises(ValueError, match="Markdown differs"):
        report.write_report(tmp_path / "bad", value, markdown + "changed")
