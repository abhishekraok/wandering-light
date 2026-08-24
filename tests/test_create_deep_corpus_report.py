from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from experiments.create_deep_corpus_report import (
    _arm_name,
    _validate_inputs,
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
    return {
        "solve_rate": solve_rate,
        "exact_match_rate": exact_match_rate,
        "mean_executions_spent": mean_executions,
        "bfs_mean_executions_at_model_mean_execution_spend": mean_executions / 2,
        "bfs_solve_rate_at_model_mean_execution_spend": solve_rate / 2,
        "bfs_to_model_total_execution_ratio": 2.5,
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
    return {
        "status": "complete",
        "corpus": {
            "split": split,
            "manifest_digest": "sha256:" + "1" * 64,
            "basis_set_id": "wl-core-v1",
            "basis_set_digest": "sha256:" + "2" * 64,
            "hub": {"revision": "3" * 40},
        },
        "cohort": {"task_ids_digest": "sha256:" + "4" * 64},
        "checkpoint": {
            "weights": [
                {
                    "name": "model.safetensors",
                    "sha256": f"sha256:{checkpoint_id * 64}",
                }
            ],
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
                },
                "base_model": {
                    "repo_id": "facebook/opt-125m",
                    "revision": "5" * 40,
                },
            },
        },
        "model": {
            "sample_counts": [1, 8, 64],
            "distance_7_8": distance_7_8,
        },
        "matched_compute": matched_compute,
        "verdict": {"kill_criterion_triggered": killed},
        "wandb": {
            "url": (
                f"https://wandb.ai/{PERSONAL_WANDB_ENTITY}/"
                f"{DEFAULT_WANDB_PROJECT}/runs/eval-{checkpoint_id}"
            )
        },
    }


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
        checkpoint_id="b",
        min_train_distance=4,
        split="test",
        killed=False,
    )

    _validate_inputs(
        test,
        [validation_one, validation_four],
        entity=PERSONAL_WANDB_ENTITY,
        project=DEFAULT_WANDB_PROJECT,
    )

    wrong_basis = deepcopy(validation_one)
    wrong_basis["corpus"]["basis_set_digest"] = "sha256:" + "9" * 64
    with pytest.raises(ValueError, match="one corpus and basis identity"):
        _validate_inputs(
            test,
            [wrong_basis, validation_four],
            entity=PERSONAL_WANDB_ENTITY,
            project=DEFAULT_WANDB_PROJECT,
        )

    with pytest.raises(ValueError, match="personal W&B entity"):
        _validate_inputs(
            test,
            [validation_one, validation_four],
            entity="ai2-llm",
            project=DEFAULT_WANDB_PROJECT,
        )

    fake_host = deepcopy(validation_one)
    fake_host["wandb"]["url"] = fake_host["wandb"]["url"].replace(
        "wandb.ai", "example.com"
    )
    with pytest.raises(ValueError, match="Invalid W&B run URL"):
        _validate_inputs(
            test,
            [fake_host, validation_four],
            entity=PERSONAL_WANDB_ENTITY,
            project=DEFAULT_WANDB_PROJECT,
        )


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
    assert "| distance >= 4 | 60.0%" in markdown
    assert "| distance >= 4 | 60.0% | 10.0% | 30.0 | yes |" in markdown


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
