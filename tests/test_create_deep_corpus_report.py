from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest

from experiments.create_deep_corpus_report import (
    _arm_name,
    _loss_gap_rows,
    _training_manifest,
    _validate_inputs,
    _verify_checkpoint_artifacts,
    build_report_markdown,
    select_validation_champion,
)
from experiments.evaluate_deep_corpus_sft import (
    DEFAULT_WANDB_PROJECT,
    PERSONAL_WANDB_ENTITY,
)


def _metric_row(
    *,
    solve_rate: float,
    exact_match_rate: float,
    mean_executions: float,
) -> dict[str, Any]:
    tasks = 10
    solved = round(solve_rate * tasks)
    exact_matches = round(exact_match_rate * tasks)
    return {
        "solve_rate": solve_rate,
        "solved": solved,
        "tasks": tasks,
        "exact_match_rate": exact_match_rate,
        "exact_matches": exact_matches,
        "mean_executions_spent": mean_executions,
        "total_executions_spent": mean_executions * tasks,
        "bfs_mean_executions_at_model_mean_execution_spend": mean_executions / 2,
        "bfs_solve_rate_at_model_mean_execution_spend": solve_rate / 2,
        "bfs_solved_at_model_mean_execution_spend": round(solve_rate * tasks / 2),
        "bfs_solved_at_matched_rate_budget": solved,
        "bfs_uniform_budget_for_at_least_model_solve_count": 12,
        "bfs_to_model_total_execution_ratio": 2.5,
        "model_cheaper_than_bfs_at_equal_or_higher_solve_count": False,
    }


def _summary(
    *,
    checkpoint_id: str,
    min_train_distance: int,
    split: str = "validation",
    solve_64: float = 0.5,
    solve_8: float = 0.4,
    solve_1: float = 0.2,
    exact_64: float = 0.1,
    mean_executions_64: float = 30.0,
    killed: bool | None = None,
) -> dict[str, Any]:
    distance_7_8 = {
        "1": _metric_row(
            solve_rate=solve_1,
            exact_match_rate=solve_1 / 2,
            mean_executions=1.0,
        ),
        "8": _metric_row(
            solve_rate=solve_8,
            exact_match_rate=solve_8 / 2,
            mean_executions=5.0,
        ),
        "64": _metric_row(
            solve_rate=solve_64,
            exact_match_rate=exact_64,
            mean_executions=mean_executions_64,
        ),
    }
    matched_compute = {
        str(distance): {"matched": deepcopy(distance_7_8)} for distance in range(1, 9)
    }
    matched_compute["7-8"] = {"matched": deepcopy(distance_7_8)}
    model_by_distance = {
        str(distance): deepcopy(distance_7_8) for distance in range(1, 9)
    }
    bfs = {
        "by_certified_distance": {
            str(distance): {
                "solved": 10,
                "tasks": 10,
                "mean_executions": 100.0,
            }
            for distance in range(1, 9)
        }
    }
    return {
        "status": "complete",
        "completed_at_utc": "2026-08-25T07:00:00+00:00",
        "corpus": {
            "split": split,
            "manifest_digest": "sha256:" + "1" * 64,
            "basis_set_id": "wl-core-v1",
            "basis_set_digest": "sha256:" + "2" * 64,
            "hub": {"repo_id": "owner/corpus", "revision": "3" * 40},
        },
        "cohort": {
            "seed": 20260816,
            "selection": "distance-stratified-within-frozen-root-split",
            "tasks": 80,
            "by_certified_distance": {str(distance): 10 for distance in range(1, 9)},
            "task_ids_digest": "sha256:" + ("4" if split == "validation" else "6") * 64,
        },
        "checkpoint": {
            "weights": [
                {
                    "name": "model.safetensors",
                    "sha256": f"sha256:{checkpoint_id * 64}",
                }
            ],
            "training_manifest_sha256": f"sha256:{checkpoint_id * 64}",
            "artifact_files": [
                {
                    "name": "model.safetensors",
                    "sha256": f"sha256:{checkpoint_id * 64}",
                },
                {
                    "name": "tokenizer.json",
                    "sha256": "sha256:" + "e" * 64,
                },
            ],
            "training_wandb_url": (
                f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/training/runs/train"
            ),
            "training": {
                "hyperparameters": {
                    "min_train_distance": min_train_distance,
                    "seed": 42,
                    "data_seed": 42,
                    "precision": "bf16",
                    "attention_implementation": "sdpa",
                    "gradient_checkpointing": True,
                    "learning_rate": 2e-5,
                },
                "base_model": {
                    "repo_id": "facebook/opt-125m",
                    "revision": "5" * 40,
                },
                "source_corpus": {
                    "manifest_digest": "sha256:" + "1" * 64,
                    "basis_set_id": "wl-core-v1",
                    "basis_set_digest": "sha256:" + "2" * 64,
                    "hub": {"repo_id": "owner/corpus", "revision": "3" * 40},
                    "verification": {
                        "ok": True,
                        "records": 100,
                        "root_split_leaks": [],
                    },
                    "task_roles": {
                        "validation": {"ordered_task_ids_digest": "sha256:" + "d" * 64}
                    },
                },
                "tokenization": {
                    "validation": {
                        "retained_task_ids_digest": "sha256:" + "d" * 64,
                        "retained_records": 80,
                        "excluded_records": 0,
                    }
                },
                "source_code": {
                    "runner": {
                        "path": "experiments/train_deep_corpus_sft.py",
                        "sha256": "sha256:" + "c" * 64,
                    }
                },
                "trainer_state": {
                    "global_step": 100,
                    "train_metrics": {"train_runtime": 60.0, "train_loss": 1.0},
                    "final_validation_metrics": {"eval_loss": 1.1},
                },
                "wandb": {
                    "url": f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
                    f"{DEFAULT_WANDB_PROJECT}/runs/train-{checkpoint_id}"
                },
            },
        },
        "generation": {
            "batch_size": 32,
            "cost_definition": "fixed",
            "max_new_tokens": 64,
            "sample_counts": [1, 8, 64],
            "samples_per_task": 64,
            "seed": 20260816,
            "temperature": 0.8,
            "top_p": 0.95,
        },
        "source_code": {
            "runner": {"sha256": "sha256:" + "7" * 64},
        },
        "model": {
            "sample_counts": [1, 8, 64],
            "by_certified_distance": model_by_distance,
            "distance_7_8": distance_7_8,
        },
        "bfs": bfs,
        "matched_compute": matched_compute,
        "verdict": {"kill_criterion_triggered": killed},
        "wandb": {
            "url": (
                f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
                f"{DEFAULT_WANDB_PROJECT}/runs/eval-{checkpoint_id}"
            )
        },
    }


def _summary_hashes() -> dict[int, str]:
    return {1: "sha256:" + "8" * 64, 4: "sha256:" + "9" * 64}


def _selection_lock(
    validation_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    champion = select_validation_champion(validation_summaries)
    summary_hashes = _summary_hashes()

    def candidate(summary: dict[str, Any]) -> dict[str, Any]:
        arm = summary["checkpoint"]["training"]["hyperparameters"]["min_train_distance"]
        distance_7_8 = summary["model"]["distance_7_8"]
        return {
            "arm_min_train_distance": arm,
            "checkpoint_signature": deepcopy(summary["checkpoint"]["artifact_files"]),
            "checkpoint_training_manifest_sha256": summary["checkpoint"][
                "training_manifest_sha256"
            ],
            "selection_key": [
                distance_7_8["64"]["solve_rate"],
                distance_7_8["8"]["solve_rate"],
                distance_7_8["1"]["solve_rate"],
                -distance_7_8["64"]["mean_executions_spent"],
                distance_7_8["64"]["exact_match_rate"],
            ],
            "summary_sha256": summary_hashes[arm],
        }

    generation = deepcopy(validation_summaries[0]["generation"])
    generation.pop("cost_definition")
    return {
        "status": "locked-before-held-out-test",
        "locked_at_utc": "2026-08-25T06:00:00+00:00",
        "discipline": {
            "held_out_test_accessed": False,
            "selection_data": "validation summaries only",
        },
        "frozen_evaluator": {"sha256": "sha256:" + "7" * 64},
        "validation_identity": {
            "corpus_manifest_digest": "sha256:" + "1" * 64,
            "basis_set_digest": "sha256:" + "2" * 64,
            "cohort_task_ids_digest": "sha256:" + "4" * 64,
            "generation_protocol": generation,
        },
        "candidates": [candidate(summary) for summary in validation_summaries],
        "champion": candidate(champion),
    }


def _validate_fixture(
    test: dict[str, Any],
    validation_summaries: list[dict[str, Any]],
    *,
    selection_lock: dict[str, Any] | None = None,
    entity: str = PERSONAL_WANDB_ENTITY,
) -> None:
    _validate_inputs(
        test,
        validation_summaries,
        selection_lock=selection_lock or _selection_lock(validation_summaries),
        validation_summary_hashes=_summary_hashes(),
        entity=entity,
        project=DEFAULT_WANDB_PROJECT,
        require_local_artifacts=False,
    )


def test_validation_champion_uses_documented_lexicographic_order():
    cheap_but_lower_solve = _summary(
        checkpoint_id="a",
        min_train_distance=1,
        solve_64=0.50,
        exact_64=0.50,
        mean_executions_64=1.0,
    )
    higher_solve = _summary(
        checkpoint_id="b",
        min_train_distance=4,
        solve_64=0.51,
        exact_64=0.0,
        mean_executions_64=100.0,
    )

    assert (
        select_validation_champion([cheap_but_lower_solve, higher_solve])
        is higher_solve
    )

    tied_solve_but_cheaper = deepcopy(cheap_but_lower_solve)
    tied_solve_but_cheaper["model"]["distance_7_8"]["64"] = _metric_row(
        solve_rate=0.51,
        exact_match_rate=0.0,
        mean_executions=50.0,
    )
    higher_solve["model"]["distance_7_8"]["64"] = _metric_row(
        solve_rate=0.51,
        exact_match_rate=1.0,
        mean_executions=100.0,
    )
    assert (
        select_validation_champion([higher_solve, tied_solve_but_cheaper])
        is tied_solve_but_cheaper
    )


def test_validation_selection_requires_distinct_checkpoint_weights():
    first = _summary(checkpoint_id="a", min_train_distance=1)
    duplicate = _summary(checkpoint_id="a", min_train_distance=4)

    with pytest.raises(ValueError, match="distinct checkpoints"):
        select_validation_champion([first, duplicate])


def test_loss_gap_rows_hashes_trainer_state_and_flags_turn_up(tmp_path):
    summary = _summary(checkpoint_id="a", min_train_distance=1)
    state = {
        "log_history": [
            {"step": 50, "epoch": 0.5, "loss": 0.8},
            {"step": 100, "epoch": 1.0, "eval_loss": 0.9},
            {"step": 150, "epoch": 1.5, "loss": 0.7},
            {"step": 200, "epoch": 2.0, "eval_loss": 0.91},
        ]
    }
    trainer_path = tmp_path / "trainer_state.json"
    payload = json.dumps(state).encode()
    trainer_path.write_bytes(payload)
    summary["checkpoint"]["path"] = str(tmp_path)
    summary["checkpoint"]["artifact_files"].append(
        {
            "name": "trainer_state.json",
            "bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }
    )

    rows = _loss_gap_rows(summary)

    assert rows[0][4] == pytest.approx(0.1)
    assert rows[0][5] == "baseline"
    assert rows[1][4] == pytest.approx(0.21)
    assert rows[1][5] == "turned up"

    trainer_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after evaluation"):
        _loss_gap_rows(summary)


def test_training_manifest_is_rehashed_and_matches_embedded_provenance(tmp_path):
    summary = _summary(checkpoint_id="a", min_train_distance=1)
    payload = (
        json.dumps(summary["checkpoint"]["training"], sort_keys=True) + "\n"
    ).encode()
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    summary["checkpoint"]["path"] = str(tmp_path)
    summary["checkpoint"]["training_manifest_sha256"] = digest
    summary["checkpoint"]["artifact_files"].append(
        {
            "name": "training_manifest.json",
            "bytes": len(payload),
            "sha256": digest,
        }
    )

    assert (
        _training_manifest(summary, require_local_artifacts=True)
        == summary["checkpoint"]["training"]
    )

    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after evaluation"):
        _training_manifest(summary, require_local_artifacts=True)


def test_checkpoint_artifacts_are_rehashed_against_recorded_identity(tmp_path):
    payload = b"weights"
    artifact_path = tmp_path / "model.safetensors"
    artifact_path.write_bytes(payload)
    summary = _summary(checkpoint_id="a", min_train_distance=1)
    summary["checkpoint"]["path"] = str(tmp_path)
    summary["checkpoint"]["artifact_files"] = [
        {
            "name": artifact_path.name,
            "bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }
    ]

    _verify_checkpoint_artifacts(summary)

    artifact_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after evaluation"):
        _verify_checkpoint_artifacts(summary)


@pytest.mark.parametrize(
    ("minimum", "expected"),
    [(1, "all distances"), (4, "distance >= 4")],
)
def test_arm_name_is_derived_from_training_provenance(minimum, expected):
    summary = _summary(
        checkpoint_id="a",
        min_train_distance=minimum,
    )

    assert _arm_name(summary) == expected


def test_validation_selection_requires_exact_ablation_and_sample_counts():
    one = _summary(checkpoint_id="a", min_train_distance=1)
    another_one = _summary(checkpoint_id="b", min_train_distance=1)
    four = _summary(checkpoint_id="c", min_train_distance=4)

    with pytest.raises(ValueError, match="min-distance 1 and 4"):
        select_validation_champion([one, another_one])
    with pytest.raises(ValueError, match="exactly two"):
        select_validation_champion([one, four, another_one])

    four["model"]["sample_counts"] = [1, 64]
    with pytest.raises(ValueError, match="solve@1, solve@8, and solve@64"):
        select_validation_champion([one, four])

    invalid = _summary(checkpoint_id="d", min_train_distance=7)
    with pytest.raises(ValueError, match="Unexpected training ablation"):
        _arm_name(invalid)


def test_input_validation_enforces_split_basis_and_personal_wandb_identity():
    validation_one = _summary(checkpoint_id="a", min_train_distance=1)
    validation_four = _summary(checkpoint_id="b", min_train_distance=4)
    test = _summary(
        checkpoint_id="a",
        min_train_distance=1,
        split="test",
        killed=False,
    )

    _validate_fixture(test, [validation_one, validation_four])

    wrong_basis = deepcopy(validation_one)
    wrong_basis["corpus"]["basis_set_digest"] = "sha256:" + "9" * 64
    with pytest.raises(ValueError, match="one corpus and basis identity"):
        _validate_fixture(test, [wrong_basis, validation_four])

    with pytest.raises(ValueError, match="personal W&B entity"):
        _validate_fixture(test, [validation_one, validation_four], entity="ai2-llm")

    fake_host = deepcopy(validation_one)
    fake_host["wandb"]["url"] = fake_host["wandb"]["url"].replace(
        "wandb.ai", "example.com"
    )
    with pytest.raises(ValueError, match="Invalid W&B run URL"):
        _validate_fixture(test, [fake_host, validation_four])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda summary: summary["generation"].__setitem__("seed", 17), "generation"),
        (
            lambda summary: summary["generation"].__setitem__("temperature", 0.7),
            "generation",
        ),
        (
            lambda summary: summary["source_code"]["runner"].__setitem__(
                "sha256", "sha256:" + "0" * 64
            ),
            "evaluator",
        ),
        (
            lambda summary: summary["cohort"].__setitem__("seed", 17),
            "cohort protocol",
        ),
        (
            lambda summary: summary["checkpoint"]["training"][
                "hyperparameters"
            ].__setitem__("learning_rate", 1e-4),
            "training settings",
        ),
        (
            lambda summary: summary["checkpoint"]["training"]["base_model"].__setitem__(
                "revision", "0" * 40
            ),
            "base-model identity",
        ),
        (
            lambda summary: summary["checkpoint"]["training"][
                "source_corpus"
            ].__setitem__("manifest_digest", "sha256:" + "0" * 64),
            "different source corpus",
        ),
        (
            lambda summary: summary["checkpoint"]["training"]["source_code"][
                "runner"
            ].__setitem__("sha256", "sha256:" + "0" * 64),
            "training runner",
        ),
        (
            lambda summary: summary["checkpoint"]["training"]["tokenization"][
                "validation"
            ].__setitem__("retained_records", 79),
            "validation tokenization",
        ),
        (
            lambda summary: summary["checkpoint"]["artifact_files"][1].__setitem__(
                "sha256", "sha256:" + "0" * 64
            ),
            "base tokenizer identity",
        ),
    ],
)
def test_input_validation_rejects_frozen_protocol_drift(mutation, message):
    validation_one = _summary(checkpoint_id="a", min_train_distance=1)
    validation_four = _summary(checkpoint_id="b", min_train_distance=4)
    test = _summary(
        checkpoint_id="a",
        min_train_distance=1,
        split="test",
        killed=False,
    )
    mutation(test)

    with pytest.raises(ValueError, match=message):
        _validate_fixture(test, [validation_one, validation_four])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda lock: lock["discipline"].__setitem__("held_out_test_accessed", True),
            "locked-test discipline",
        ),
        (
            lambda lock: lock["frozen_evaluator"].__setitem__(
                "sha256", "sha256:" + "0" * 64
            ),
            "different frozen evaluator",
        ),
        (
            lambda lock: lock["candidates"][0].__setitem__(
                "summary_sha256", "sha256:" + "0" * 64
            ),
            "does not match arm 1",
        ),
        (
            lambda lock: lock.__setitem__("locked_at_utc", "2026-08-25T08:00:00+00:00"),
            "not locked before",
        ),
    ],
)
def test_input_validation_enforces_selection_lock(mutation, message):
    validation_one = _summary(checkpoint_id="a", min_train_distance=1)
    validation_four = _summary(checkpoint_id="b", min_train_distance=4)
    validations = [validation_one, validation_four]
    test = _summary(
        checkpoint_id="a",
        min_train_distance=1,
        split="test",
        killed=False,
    )
    lock = _selection_lock(validations)
    mutation(lock)

    with pytest.raises(ValueError, match=message):
        _validate_fixture(test, validations, selection_lock=lock)


@pytest.mark.parametrize(
    ("killed", "verdict_text"),
    [
        (True, "Kill criterion: TRIGGERED"),
        (False, "Kill criterion: not triggered"),
    ],
)
def test_report_verdict_and_provenance_are_explicit(killed, verdict_text):
    validation_one = _summary(
        checkpoint_id="a",
        min_train_distance=1,
        solve_64=0.4,
    )
    validation_four = _summary(
        checkpoint_id="b",
        min_train_distance=4,
        solve_64=0.6,
    )
    test = _summary(
        checkpoint_id="b",
        min_train_distance=4,
        split="test",
        solve_64=0.6,
        killed=killed,
    )

    markdown = build_report_markdown(test, [validation_one, validation_four])

    assert verdict_text in markdown
    assert "Headline: distance-8 solve rate against executions spent" in markdown
    assert "`facebook/opt-125m` at `" + "5" * 40 + "`" in markdown
    assert "Basis: `wl-core-v1` / `sha256:" + "2" * 64 + "`" in markdown
    assert "Central question: does the stronger base solve distance 5-8?" in markdown
    assert "Held-out solve@k and execution-vs-exact gap" in markdown
    assert "Local train/validation loss gap and overfitting" in markdown
    assert (
        "| distance >= 4 | 16/80 | 32/80 | 48/80 | 8/80 | 2/10 | 4/10 | "
        "6/10 | 30.0 | 1/10 | yes |" in markdown
    )
    if killed:
        assert "model solved 6/10" in markdown
        assert "solved 0/24" not in markdown


def test_report_rejects_test_result_for_non_champion_checkpoint():
    validation_one = _summary(
        checkpoint_id="a",
        min_train_distance=1,
        solve_64=0.7,
    )
    validation_four = _summary(
        checkpoint_id="b",
        min_train_distance=4,
        solve_64=0.6,
    )
    test = _summary(
        checkpoint_id="b",
        min_train_distance=4,
        split="test",
        killed=False,
    )

    with pytest.raises(ValueError, match="validation-selected ablation champion"):
        build_report_markdown(test, [validation_one, validation_four])
