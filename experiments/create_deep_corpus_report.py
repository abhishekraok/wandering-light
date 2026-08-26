"""Create issue #50's final W&B report from verified evaluation summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from experiments.evaluate_deep_corpus_sft import (
    DEFAULT_SAMPLE_COUNTS,
    DEFAULT_WANDB_PROJECT,
    PERSONAL_WANDB_ENTITY,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class ReportInputs:
    test_summary: Path
    validation_summaries: tuple[Path, ...]
    selection_lock: Path
    output: Path
    entity: str = PERSONAL_WANDB_ENTITY
    project: str = DEFAULT_WANDB_PROJECT
    title: str = "Deep corpus v1: learned policy vs exhaustive search"
    operational_notes: tuple[str, ...] = ()


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read evaluation summary {path}: {error}"
        ) from error
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise ValueError(f"Evaluation summary is not complete: {path}")
    return value


def _read_selection_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read validation selection lock {path}: {error}"
        ) from error
    supported_statuses = {
        "locked-before-held-out-test",
        "locked-before-held-out-test-scoring",
    }
    if not isinstance(value, dict) or value.get("status") not in supported_statuses:
        raise ValueError(f"Validation selection lock is not complete: {path}")
    return value


def _run_url_parts(summary: Mapping[str, Any]) -> tuple[str, str, str]:
    url = summary.get("wandb", {}).get("url")
    if not isinstance(url, str):
        raise ValueError("Evaluation summary has no W&B run URL")
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() not in {"wandb.ai", "www.wandb.ai"}
        or len(parts) != 4
        or parts[2] != "runs"
        or not parts[3]
    ):
        raise ValueError(f"Invalid W&B run URL in evaluation summary: {url!r}")
    return parts[0], parts[1], parts[-1]


def _checkpoint_signature(summary: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    files = summary.get("checkpoint", {}).get("artifact_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Evaluation summary has no complete checkpoint identity")
    try:
        return tuple(sorted((row["name"], row["sha256"]) for row in files))
    except (KeyError, TypeError) as error:
        raise ValueError("Evaluation checkpoint identity is malformed") from error


def _verify_checkpoint_artifacts(summary: Mapping[str, Any]) -> None:
    checkpoint = summary.get("checkpoint", {})
    checkpoint_path = checkpoint.get("path")
    files = checkpoint.get("artifact_files")
    if not isinstance(checkpoint_path, str) or not isinstance(files, list) or not files:
        raise ValueError("Evaluation summary has no local checkpoint artifact identity")
    root = Path(checkpoint_path)
    for row in files:
        try:
            path = root / row["name"]
            expected_bytes = int(row["bytes"])
            expected_digest = row["sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Evaluation checkpoint identity is malformed") from error
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise ValueError(
                f"Could not stat checkpoint artifact {path}: {error}"
            ) from error
        if actual_bytes != expected_bytes or _sha256_file(path) != expected_digest:
            raise ValueError(
                f"Checkpoint artifact changed after evaluation: {path.name}"
            )


def _max_samples(summary: Mapping[str, Any]) -> str:
    counts = summary.get("model", {}).get("sample_counts")
    if not isinstance(counts, list) or not counts:
        raise ValueError("Evaluation summary has no sample counts")
    parsed = tuple(int(value) for value in counts)
    if parsed != DEFAULT_SAMPLE_COUNTS:
        raise ValueError(
            "Issue #50 report requires solve@1, solve@8, and solve@64 exactly"
        )
    return str(DEFAULT_SAMPLE_COUNTS[-1])


def _model_long_row(summary: Mapping[str, Any], samples: int) -> Mapping[str, Any]:
    try:
        return summary["model"]["distance_7_8"][str(samples)]
    except KeyError as error:
        raise ValueError(
            f"Evaluation summary has no distance-7/8 solve@{samples} result"
        ) from error


def _selection_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Validation-only ordering: execution solve@64/@8/@1, cost, then exact."""

    max_count = int(_max_samples(summary))
    max_row = _model_long_row(summary, max_count)
    return (
        float(_model_long_row(summary, 64)["solve_rate"]),
        float(_model_long_row(summary, 8)["solve_rate"]),
        float(_model_long_row(summary, 1)["solve_rate"]),
        -float(max_row["mean_executions_spent"]),
        float(max_row["exact_match_rate"]),
    )


def select_validation_champion(
    validation_summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if len(validation_summaries) != 2:
        raise ValueError(
            "The all-distance vs distance>=4 ablation needs exactly two validations"
        )
    minima = [_arm_min_distance(summary) for summary in validation_summaries]
    if set(minima) != {1, 4}:
        raise ValueError(
            "Validation summaries must contain the min-distance 1 and 4 arms"
        )
    signatures = [_checkpoint_signature(summary) for summary in validation_summaries]
    if len(set(signatures)) != len(signatures):
        raise ValueError("Validation summaries must represent distinct checkpoints")
    return max(validation_summaries, key=_selection_key)


def _arm_min_distance(summary: Mapping[str, Any]) -> int:
    hyperparameters = (
        summary.get("checkpoint", {}).get("training", {}).get("hyperparameters", {})
    )
    minimum = hyperparameters.get("min_train_distance")
    if minimum not in {1, 4}:
        raise ValueError(f"Unexpected training ablation minimum distance: {minimum!r}")
    return int(minimum)


def _arm_name(summary: Mapping[str, Any]) -> str:
    minimum = _arm_min_distance(summary)
    if minimum == 1:
        return "all distances"
    return "distance >= 4"


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def _aggregate_model_row(
    summary: Mapping[str, Any], samples: int
) -> dict[str, float | int]:
    try:
        rows = [
            summary["model"]["by_certified_distance"][str(distance)][str(samples)]
            for distance in range(1, 9)
        ]
        tasks = sum(int(row["tasks"]) for row in rows)
        solved = sum(int(row["solved"]) for row in rows)
        exact_matches = sum(int(row["exact_matches"]) for row in rows)
        total_executions = sum(float(row["total_executions_spent"]) for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Evaluation summary has no complete solve@{samples} distance rows"
        ) from error
    if tasks <= 0:
        raise ValueError("Evaluation summary has no model tasks")
    return {
        "tasks": tasks,
        "solved": solved,
        "exact_matches": exact_matches,
        "solve_rate": solved / tasks,
        "exact_match_rate": exact_matches / tasks,
        "mean_executions_spent": total_executions / tasks,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(
            f"Could not hash checkpoint artifact {path}: {error}"
        ) from error
    return "sha256:" + digest.hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _without_keys(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _generation_protocol(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = summary.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Evaluation summary has no generation protocol")
    return _without_keys(generation, "runtime")


def _training_protocol(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    hyperparameters = (
        summary.get("checkpoint", {}).get("training", {}).get("hyperparameters")
    )
    if not isinstance(hyperparameters, dict):
        raise ValueError("Evaluation summary has no training protocol")
    return _without_keys(hyperparameters, "min_train_distance")


def _training_manifest(
    summary: Mapping[str, Any], *, require_local_artifacts: bool
) -> Mapping[str, Any]:
    checkpoint = summary.get("checkpoint", {})
    embedded = checkpoint.get("training")
    if not isinstance(embedded, dict):
        raise ValueError("Evaluation summary has no embedded training manifest")
    if not require_local_artifacts:
        return embedded
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str):
        raise ValueError("Evaluation summary has no local checkpoint path")
    artifact_files = checkpoint.get("artifact_files")
    if not isinstance(artifact_files, list):
        raise ValueError("Evaluation summary has no checkpoint artifact list")
    artifact = next(
        (row for row in artifact_files if row.get("name") == "training_manifest.json"),
        None,
    )
    if not isinstance(artifact, dict):
        raise ValueError("Checkpoint identity has no training_manifest.json")
    manifest_path = Path(checkpoint_path) / "training_manifest.json"
    try:
        size = manifest_path.stat().st_size
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read training manifest: {error}") from error
    digest = _sha256_file(manifest_path)
    if (
        size != artifact.get("bytes")
        or digest != artifact.get("sha256")
        or digest != checkpoint.get("training_manifest_sha256")
    ):
        raise ValueError("Training manifest changed after evaluation recorded it")
    if manifest != embedded:
        raise ValueError("Embedded training provenance differs from the saved manifest")
    return embedded


def _tokenizer_signature(
    summary: Mapping[str, Any], *, require_local_artifacts: bool
) -> tuple[tuple[str, str], ...]:
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    }
    files = summary.get("checkpoint", {}).get("artifact_files", [])
    try:
        signature = tuple(
            sorted(
                (row["name"], row["sha256"])
                for row in files
                if row.get("name") in tokenizer_names
            )
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint tokenizer identity is malformed") from error
    if not signature:
        raise ValueError("Checkpoint identity has no tokenizer artifacts")
    if require_local_artifacts:
        checkpoint_path = summary.get("checkpoint", {}).get("path")
        if not isinstance(checkpoint_path, str):
            raise ValueError("Evaluation summary has no local checkpoint path")
        for name, expected_digest in signature:
            if _sha256_file(Path(checkpoint_path) / name) != expected_digest:
                raise ValueError(f"Checkpoint tokenizer artifact changed: {name}")
    return signature


def _evaluator_sha256(summary: Mapping[str, Any]) -> str:
    value = summary.get("source_code", {}).get("runner", {}).get("sha256")
    if not isinstance(value, str):
        raise ValueError("Evaluation summary has no evaluator identity")
    return value


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} has no timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} timestamp is malformed: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp must include a UTC offset")
    return parsed


def _validate_validation_summaries_for_lock(
    validation_summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Reject an unmatched ablation before selecting and freezing a champion."""

    for summary in validation_summaries:
        if summary.get("corpus", {}).get("split") != "validation":
            raise ValueError("Selection locks may use validation summaries only")
        _max_samples(summary)
    select_validation_champion(validation_summaries)
    identity_fields = ("manifest_digest", "basis_set_id", "basis_set_digest")
    identities = {
        (
            *(summary["corpus"].get(field) for field in identity_fields),
            summary["corpus"].get("hub", {}).get("repo_id"),
            summary["corpus"].get("hub", {}).get("revision"),
        )
        for summary in validation_summaries
    }
    if len(identities) != 1:
        raise ValueError("Validation inputs do not share one corpus and basis identity")
    matched_protocols = (
        ("cohort", lambda summary: summary["cohort"]),
        ("generation", _generation_protocol),
        ("training", _training_protocol),
        (
            "base model",
            lambda summary: summary["checkpoint"]["training"]["base_model"],
        ),
    )
    for label, extract in matched_protocols:
        frozen = {
            json.dumps(extract(summary), sort_keys=True)
            for summary in validation_summaries
        }
        if len(frozen) != 1:
            raise ValueError(f"Validation inputs do not share one frozen {label}")
    if len({_evaluator_sha256(summary) for summary in validation_summaries}) != 1:
        raise ValueError("Validation inputs do not share one frozen evaluator")


def build_validation_selection_lock(
    validation_summaries: Sequence[Mapping[str, Any]],
    *,
    validation_summary_hashes: Mapping[int, str],
    locked_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the validation-only artifact required before held-out test scoring."""

    _validate_validation_summaries_for_lock(validation_summaries)
    if set(validation_summary_hashes) != {1, 4}:
        raise ValueError("Selection lock requires hashes for ablation arms 1 and 4")
    champion = select_validation_champion(validation_summaries)

    def candidate(summary: Mapping[str, Any]) -> dict[str, Any]:
        arm = _arm_min_distance(summary)
        distance_7_8 = summary["model"]["distance_7_8"]
        return {
            "arm_min_train_distance": arm,
            "checkpoint_signature": [
                {"name": name, "sha256": digest}
                for name, digest in _checkpoint_signature(summary)
            ],
            "checkpoint_training_manifest_sha256": summary["checkpoint"][
                "training_manifest_sha256"
            ],
            "selection_key": list(_selection_key(summary)),
            "summary_sha256": validation_summary_hashes[arm],
            "distance_7_8": {
                str(samples): {
                    "solved": distance_7_8[str(samples)]["solved"],
                    "tasks": distance_7_8[str(samples)]["tasks"],
                    "mean_executions_spent": distance_7_8[str(samples)][
                        "mean_executions_spent"
                    ],
                }
                for samples in DEFAULT_SAMPLE_COUNTS
            },
        }

    first = validation_summaries[0]
    evaluator = first.get("source_code", {}).get("runner", {})
    evaluator_path = evaluator.get("path")
    if not isinstance(evaluator_path, str):
        raise ValueError("Validation summary has no frozen evaluator path")
    locked_at = locked_at_utc or datetime.now(UTC).isoformat()
    _parse_timestamp(locked_at, label="Validation selection lock")
    return {
        "status": "locked-before-held-out-test-scoring",
        "locked_at_utc": locked_at,
        "discipline": {
            "selection_data": "validation summaries only",
            "held_out_test_integrity_read_allowed": True,
            "held_out_test_model_scored": False,
            "held_out_test_outcomes_used_for_selection": False,
        },
        "frozen_evaluator": {
            "path": evaluator_path,
            "sha256": _evaluator_sha256(first),
        },
        "validation_identity": {
            "corpus_manifest_digest": first["corpus"]["manifest_digest"],
            "basis_set_digest": first["corpus"]["basis_set_digest"],
            "cohort_task_ids_digest": first["cohort"]["task_ids_digest"],
            "cohort_protocol": {
                key: first["cohort"][key]
                for key in ("seed", "selection", "tasks", "by_certified_distance")
            },
            "generation_protocol": _generation_protocol(first),
        },
        "candidates": [candidate(summary) for summary in validation_summaries],
        "champion": candidate(champion),
    }


def create_validation_selection_lock(
    validation_summary_paths: Sequence[Path],
    *,
    output: Path,
) -> dict[str, Any]:
    """Read two validation summaries and exclusively write their champion lock."""

    summaries = [_read_summary(path) for path in validation_summary_paths]
    hashes = {
        _arm_min_distance(summary): _sha256_file(path)
        for path, summary in zip(validation_summary_paths, summaries, strict=True)
    }
    lock = build_validation_selection_lock(
        summaries,
        validation_summary_hashes=hashes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as output_file:
            json.dump(lock, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"Selection-lock output already exists; refusing to overwrite: {output}"
        ) from error
    return lock


def _loss_gap_rows(
    summary: Mapping[str, Any],
) -> list[tuple[float, int, float, float, float, str]]:
    """Read verified epoch losses from the checkpoint recorded by an evaluation."""

    checkpoint = summary.get("checkpoint", {})
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str):
        return []
    artifact_files = checkpoint.get("artifact_files")
    if not isinstance(artifact_files, list):
        raise ValueError("Evaluation summary has no checkpoint artifact list")
    trainer_artifact = next(
        (row for row in artifact_files if row.get("name") == "trainer_state.json"),
        None,
    )
    if not isinstance(trainer_artifact, dict):
        raise ValueError("Checkpoint identity has no trainer_state.json")
    trainer_path = Path(checkpoint_path) / "trainer_state.json"
    try:
        trainer_bytes = trainer_path.stat().st_size
    except OSError as error:
        raise ValueError(
            f"Could not stat trainer state {trainer_path}: {error}"
        ) from error
    if trainer_bytes != trainer_artifact.get("bytes") or _sha256_file(
        trainer_path
    ) != trainer_artifact.get("sha256"):
        raise ValueError(
            "Trainer state changed after evaluation recorded the checkpoint"
        )
    try:
        state = json.loads(trainer_path.read_text(encoding="utf-8"))
        history = state["log_history"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Could not read trainer loss history: {error}") from error
    if not isinstance(history, list):
        raise ValueError("Trainer loss history is malformed")

    rows: list[tuple[float, int, float, float, float, str]] = []
    seen_evaluations: set[tuple[int, float]] = set()
    previous_validation: float | None = None
    for entry in history:
        if not isinstance(entry, dict) or "eval_loss" not in entry:
            continue
        try:
            step = int(entry["step"])
            validation_loss = float(entry["eval_loss"])
            epoch = float(entry["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Trainer evaluation history is malformed") from error
        evaluation_key = (step, validation_loss)
        if evaluation_key in seen_evaluations:
            continue
        seen_evaluations.add(evaluation_key)
        preceding = [
            train_entry
            for train_entry in history
            if isinstance(train_entry, dict)
            and "loss" in train_entry
            and int(train_entry.get("step", step + 1)) <= step
        ]
        if not preceding:
            raise ValueError(f"No training loss precedes epoch {epoch:g} validation")
        train_entry = max(preceding, key=lambda value: int(value["step"]))
        train_step = int(train_entry["step"])
        train_loss = float(train_entry["loss"])
        if previous_validation is None:
            trend = "baseline"
        elif validation_loss > previous_validation:
            trend = "turned up"
        elif validation_loss < previous_validation:
            trend = "improved"
        else:
            trend = "flat"
        rows.append(
            (
                epoch,
                train_step,
                train_loss,
                validation_loss,
                validation_loss - train_loss,
                trend,
            )
        )
        previous_validation = validation_loss
    if not rows:
        raise ValueError("Trainer state contains no epoch validation losses")
    return rows


def build_report_markdown(
    test_summary: Mapping[str, Any],
    validation_summaries: Sequence[Mapping[str, Any]],
    *,
    selection_lock: Mapping[str, Any] | None = None,
    selection_lock_sha256: str | None = None,
    operational_notes: Sequence[str] = (),
) -> str:
    champion = select_validation_champion(validation_summaries)
    if _checkpoint_signature(test_summary) != _checkpoint_signature(champion):
        raise ValueError(
            "The test checkpoint is not the validation-selected ablation champion"
        )
    max_samples = int(_max_samples(test_summary))
    primary = test_summary["matched_compute"]["7-8"]["matched"][str(max_samples)]
    checkpoint = test_summary["checkpoint"]["training"]
    base_model = checkpoint["base_model"]
    base_model_name = base_model["repo_id"]

    deep_rows = []
    deep_solved = 0
    deep_exact = 0
    deep_tasks = 0
    for distance in range(5, 9):
        model_row = test_summary["matched_compute"][str(distance)]["matched"][
            str(max_samples)
        ]
        bfs_row = test_summary["bfs"]["by_certified_distance"][str(distance)]
        deep_solved += int(model_row["solved"])
        deep_exact += int(model_row["exact_matches"])
        deep_tasks += int(model_row["tasks"])
        deep_rows.append(
            [
                distance,
                f"{model_row['solved']}/{model_row['tasks']}",
                f"{model_row['exact_matches']}/{model_row['tasks']}",
                f"{model_row['mean_executions_spent']:.1f}",
                f"{model_row['bfs_solved_at_model_mean_execution_spend']}/"
                f"{model_row['tasks']}",
                f"{model_row['bfs_solved_at_matched_rate_budget']}/"
                f"{model_row['tasks']} at "
                f"{model_row['bfs_uniform_budget_for_at_least_model_solve_count']}",
                f"{bfs_row['solved']}/{bfs_row['tasks']}",
                f"{bfs_row['mean_executions']:.1f}",
            ]
        )
    deep_table = _markdown_table(
        [
            "distance",
            f"execution solve@{max_samples}",
            f"exact@{max_samples}",
            "model mean executions",
            "BFS solves at model spend",
            "BFS matched-rate solves / uniform budget",
            "BFS full-cohort solves",
            "BFS full-solve mean executions",
        ],
        deep_rows,
    )
    if deep_solved > 0:
        deep_answer = (
            f"**Yes: {deep_solved}/{deep_tasks} held-out tasks at distances 5-8 "
            f"were execution-correct at k={max_samples}** ({deep_exact} exact). "
            "Run 1 solved 0/48 in this regime."
        )
    else:
        deep_answer = (
            f"**No: 0/{deep_tasks}, with 0/12 at every distance from 5 through 8.** "
            "The stronger base did not move run 1's target-regime result."
        )

    verdict = test_summary["verdict"]
    killed = verdict.get("kill_criterion_triggered")
    if killed is True:
        verdict_text = (
            "**Kill criterion: TRIGGERED.** The autoregressive solver is not cheaper "
            "than exhaustive search at equal solve rate on certified distance 7-8. "
        )
        primary_solved = int(primary["solved"])
        primary_tasks = int(primary["tasks"])
        if primary_solved == 0:
            verdict_text += (
                f"The validation-selected `{base_model_name}` model solved "
                f"0/{primary_tasks} there, so BFS matches its solve count at zero "
                "executions. There is no winning budget to report. With run 1 and "
                "this stronger-base rerun both missing the target regime, the "
                "autoregressive solver is finished on this substrate under the "
                "locked SFT protocol."
            )
        else:
            verdict_text += (
                f"The validation-selected `{base_model_name}` model solved "
                f"{primary_solved}/{primary_tasks}; matched-task BFS reached at least "
                "that solve count more cheaply."
            )
    elif killed is False:
        verdict_text = (
            "**Kill criterion: not triggered.** The learned solver is cheaper than "
            "exhaustive search at its achieved distance-7/8 solve rate."
        )
    else:
        raise ValueError("Test summary has no final kill-criterion verdict")

    distance_8_rows = []
    for samples in test_summary["model"]["sample_counts"]:
        row = test_summary["matched_compute"]["8"]["matched"][str(samples)]
        distance_8_rows.append(
            [
                samples,
                f"{row['solved']}/{row['tasks']} ({_pct(row['solve_rate'])})",
                f"{row['exact_matches']}/{row['tasks']} "
                f"({_pct(row['exact_match_rate'])})",
                f"{row['mean_executions_spent']:.1f}",
                f"{row['bfs_mean_executions_at_model_mean_execution_spend']:.1f}",
                f"{row['bfs_solved_at_model_mean_execution_spend']}/"
                f"{row['tasks']} "
                f"({_pct(row['bfs_solve_rate_at_model_mean_execution_spend'])})",
                _ratio(row["bfs_to_model_total_execution_ratio"]),
            ]
        )
    distance_8_headline = _markdown_table(
        [
            "k",
            "model execution solve",
            "exact match",
            "model mean executions",
            "BFS mean executions",
            "BFS solve at model spend",
            "BFS/model at >= model solve count",
        ],
        distance_8_rows,
    )

    combined_verdict = _markdown_table(
        [
            "test cohort",
            f"execution solve@{max_samples}",
            f"exact match@{max_samples}",
            "model mean executions",
            "BFS/model at >= model solve count",
        ],
        [
            [
                "distance 7-8",
                f"{primary['solved']}/{primary['tasks']} "
                f"({_pct(primary['solve_rate'])})",
                f"{primary['exact_matches']}/{primary['tasks']} "
                f"({_pct(primary['exact_match_rate'])})",
                f"{primary['mean_executions_spent']:.1f}",
                _ratio(primary["bfs_to_model_total_execution_ratio"]),
            ]
        ],
    )

    by_distance_rows = []
    for distance in range(1, 9):
        row = test_summary["matched_compute"][str(distance)]["matched"][
            str(max_samples)
        ]
        bfs_row = test_summary["bfs"]["by_certified_distance"][str(distance)]
        by_distance_rows.append(
            [
                distance,
                f"{row['solved']}/{row['tasks']} ({_pct(row['solve_rate'])})",
                f"{row['exact_matches']}/{row['tasks']} "
                f"({_pct(row['exact_match_rate'])})",
                f"{row['mean_executions_spent']:.1f}",
                f"{bfs_row['solved']}/{bfs_row['tasks']}",
                f"{bfs_row['mean_executions']:.1f}",
                f"{row['bfs_solved_at_model_mean_execution_spend']}/{row['tasks']}",
                _ratio(row["bfs_to_model_total_execution_ratio"]),
                "yes"
                if row["model_cheaper_than_bfs_at_equal_or_higher_solve_count"]
                else "no",
            ]
        )
    distance_table = _markdown_table(
        [
            "certified distance",
            f"model solve@{max_samples}",
            "exact match",
            "model mean exec",
            "BFS full solves",
            "BFS full mean exec",
            "BFS solve at model cost",
            "BFS/model at equal solve count",
            "model cheaper",
        ],
        by_distance_rows,
    )

    overall_solve_rows = []
    for samples in test_summary["model"]["sample_counts"]:
        row = _aggregate_model_row(test_summary, int(samples))
        tasks = int(row["tasks"])
        solved = int(row["solved"])
        exact_matches = int(row["exact_matches"])
        overall_solve_rows.append(
            [
                samples,
                f"{solved}/{tasks} ({_pct(float(row['solve_rate']))})",
                f"{exact_matches}/{tasks} ({_pct(float(row['exact_match_rate']))})",
                solved - exact_matches,
                f"{row['mean_executions_spent']:.1f}",
            ]
        )
    overall_solve_table = _markdown_table(
        [
            "k",
            "execution-correct",
            "exact match",
            "execution-only gap",
            "mean executions",
        ],
        overall_solve_rows,
    )

    solve_at_k_rows = []
    for samples in test_summary["model"]["sample_counts"]:
        row = test_summary["matched_compute"]["7-8"]["matched"][str(samples)]
        solve_at_k_rows.append(
            [
                samples,
                _pct(row["solve_rate"]),
                _pct(row["exact_match_rate"]),
                f"{row['mean_executions_spent']:.1f}",
                _pct(row["bfs_solve_rate_at_model_mean_execution_spend"]),
            ]
        )
    solve_at_k_table = _markdown_table(
        ["k", "execution solve", "exact match", "mean exec", "BFS at model cost"],
        solve_at_k_rows,
    )

    validation_rows = []
    for summary in validation_summaries:
        samples = int(_max_samples(summary))
        row = _model_long_row(summary, samples)
        overall_rows = {
            count: _aggregate_model_row(summary, count) for count in (1, 8, 64)
        }
        selected = _checkpoint_signature(summary) == _checkpoint_signature(champion)
        validation_rows.append(
            [
                _arm_name(summary),
                f"{overall_rows[1]['solved']}/{overall_rows[1]['tasks']}",
                f"{overall_rows[8]['solved']}/{overall_rows[8]['tasks']}",
                f"{overall_rows[64]['solved']}/{overall_rows[64]['tasks']}",
                f"{overall_rows[64]['exact_matches']}/{overall_rows[64]['tasks']}",
                f"{_model_long_row(summary, 1)['solved']}/{row['tasks']}",
                f"{_model_long_row(summary, 8)['solved']}/{row['tasks']}",
                f"{row['solved']}/{row['tasks']}",
                f"{row['mean_executions_spent']:.1f}",
                f"{row['exact_matches']}/{row['tasks']}",
                "yes" if selected else "no",
            ]
        )
    validation_table = _markdown_table(
        [
            "training arm",
            "overall solve@1",
            "overall solve@8",
            "overall solve@64",
            "overall exact@64",
            "d7-8 solve@1",
            "d7-8 solve@8",
            "d7-8 solve@64",
            "d7-8 mean exec",
            "d7-8 exact@64",
            "selected",
        ],
        validation_rows,
    )

    training_rows = []
    for summary in validation_summaries:
        training = summary["checkpoint"]["training"]
        trainer_state = training["trainer_state"]
        train_metrics = trainer_state["train_metrics"]
        final_validation = trainer_state["final_validation_metrics"]
        training_rows.append(
            [
                _arm_name(summary),
                trainer_state["global_step"],
                f"{train_metrics['train_runtime']:.1f} s "
                f"({train_metrics['train_runtime'] / 60:.1f} min)",
                f"{train_metrics['train_loss']:.6f}",
                f"{final_validation['eval_loss']:.6f}",
                training["wandb"]["url"],
            ]
        )
    training_table = _markdown_table(
        [
            "training arm",
            "steps",
            "trainer runtime",
            "whole-run train loss",
            "final validation loss",
            "W&B run",
        ],
        training_rows,
    )

    loss_gap_rows = []
    loss_trajectories = []
    for summary in validation_summaries:
        arm_loss_rows = _loss_gap_rows(summary)
        if arm_loss_rows:
            validation_values = [row[3] for row in arm_loss_rows]
            turn_up = any(row[5] == "turned up" for row in arm_loss_rows)
            loss_trajectories.append(
                f"- {_arm_name(summary)} validation loss: "
                + " -> ".join(f"`{value:.6f}`" for value in validation_values)
                + ("; **turned up**." if turn_up else "; no turn-up.")
            )
        for (
            epoch,
            train_step,
            train_loss,
            validation_loss,
            gap,
            trend,
        ) in arm_loss_rows:
            loss_gap_rows.append(
                [
                    _arm_name(summary),
                    f"{epoch:g}",
                    f"{train_loss:.6f} (step {train_step})",
                    f"{validation_loss:.6f}",
                    f"{gap:+.6f}",
                    trend,
                ]
            )
    if loss_gap_rows:
        loss_gap_table = _markdown_table(
            [
                "training arm",
                "epoch",
                "preceding train window",
                "validation loss",
                "validation - train",
                "validation trend",
            ],
            loss_gap_rows,
        )
    else:
        loss_gap_table = "Checkpoint loss histories were unavailable."

    corpus = test_summary["corpus"]
    generation = _generation_protocol(test_summary)
    training_hyperparameters = checkpoint["hyperparameters"]
    validation_links = "\n".join(
        f"- Validation {_arm_name(summary)}: {summary['wandb']['url']}"
        for summary in validation_summaries
    )
    lock_lines = []
    if selection_lock is not None:
        lock_lines.append(
            f"- Validation champion locked at: `{selection_lock['locked_at_utc']}`"
        )
        if selection_lock.get("status") == "locked-before-held-out-test-scoring":
            lock_lines.append(
                f"- Lock recorded held-out test model-scored: "
                f"`{selection_lock['discipline']['held_out_test_model_scored']}`"
            )
        else:
            lock_lines.append(
                "- Legacy lock field `held_out_test_accessed=false` means the test "
                "was not model-scored for selection; it does not mean the payload "
                "was unread."
            )
        lock_lines.append(
            "- Corpus integrity verification may read the test payload before "
            "selection; no test outcome is model-scored or used for selection."
        )
    if selection_lock_sha256 is not None:
        lock_lines.append(f"- Validation selection lock: `{selection_lock_sha256}`")
    controls = [
        f"- Precision: `{training_hyperparameters['precision']}`; attention: "
        f"`{training_hyperparameters['attention_implementation']}`; gradient "
        f"checkpointing: `{training_hyperparameters['gradient_checkpointing']}`.",
        "- The native tokenizer changed with the base model; the corpus, frozen "
        "splits, evaluator, seeds, sampling protocol, and test discipline did not.",
        *(f"- {note}" for note in operational_notes),
    ]
    provenance_lines = [
        f"- Corpus manifest: `{corpus['manifest_digest']}`",
        f"- Hub revision: `{corpus['hub']['revision']}`",
        f"- Basis: `{corpus['basis_set_id']}` / `{corpus['basis_set_digest']}`",
        f"- Base model: `{base_model['repo_id']}` at `{base_model['revision']}`",
        f"- Frozen evaluator SHA-256: `{_evaluator_sha256(test_summary)}`",
        f"- Training seed/data seed: `{training_hyperparameters['seed']}` / "
        f"`{training_hyperparameters['data_seed']}`",
        f"- Sampling: seed `{generation['seed']}`, k "
        f"`{generation['sample_counts']}`, temperature `{generation['temperature']}`, "
        f"top-p `{generation['top_p']}`, max new tokens "
        f"`{generation['max_new_tokens']}`, batch `{generation['batch_size']}`",
        "- Validation cohort task-ID digest: "
        f"`{validation_summaries[0]['cohort']['task_ids_digest']}`",
        f"- Test cohort task-ID digest: `{test_summary['cohort']['task_ids_digest']}`",
        *lock_lines,
        *validation_links.splitlines(),
        f"- Training run: {test_summary['checkpoint']['training_wandb_url']}",
        f"- Test evaluation run: {test_summary['wandb']['url']}",
    ]
    return "\n\n".join(
        [
            "## Verdict\n" + verdict_text,
            "## Central question: does the stronger base solve distance 5-8?\n"
            + deep_answer
            + "\n\n"
            + deep_table,
            "## Headline: distance-8 solve rate against executions spent\n"
            + distance_8_headline,
            "## Combined distance 7-8 kill criterion\n" + combined_verdict,
            "## Held-out solve@k and execution-vs-exact gap\n" + overall_solve_table,
            "## Solve rate by certified distance\n"
            "Model and BFS use the same task IDs. BFS is evaluated at the model's "
            "mean execution spend and at the budget needed to reach at least the "
            "same solve count.\n\n" + distance_table,
            "## Distance 7-8 solve@k and exact-match gap\n" + solve_at_k_table,
            "## Training outcomes and runtime\n" + training_table,
            "## Validation-only ablation selection\n"
            "The frozen lexicographic rule is distance-7/8 execution solve@64, "
            "solve@8, solve@1 (descending), mean executions@64 (ascending), then "
            "exact@64 (descending). Aggregate validation solves are context only.\n\n"
            + validation_table,
            "## Local train/validation loss gap and overfitting\n"
            "Train loss is the nearest preceding logged 50-step training window, "
            "not an epoch-average loss; the local gap is validation minus that "
            "window. A validation turn-up is flagged explicitly.\n\n" + loss_gap_table,
            "## Validation-loss trajectories\n"
            + (
                "\n".join(loss_trajectories)
                if loss_trajectories
                else "Checkpoint loss histories were unavailable."
            ),
            "## Run controls and disclosed technical mechanisms\n"
            + "\n".join(controls),
            "## Provenance\n" + "\n".join(provenance_lines),
        ]
    )


def _validate_inputs(
    test_summary: Mapping[str, Any],
    validation_summaries: Sequence[Mapping[str, Any]],
    *,
    selection_lock: Mapping[str, Any],
    selection_lock_sha256: str,
    validation_summary_hashes: Mapping[int, str],
    entity: str,
    project: str,
    require_local_artifacts: bool = True,
) -> None:
    if entity != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            f"Reports must be written to personal W&B entity {PERSONAL_WANDB_ENTITY!r}"
        )
    if not project:
        raise ValueError("W&B report project must not be empty")
    if test_summary.get("corpus", {}).get("split") != "test":
        raise ValueError("The final report requires a held-out test summary")
    if not isinstance(test_summary.get("matched_compute"), dict):
        raise ValueError("The held-out test summary must include matched-task BFS")
    kill_verdict = test_summary.get("verdict", {}).get("kill_criterion_triggered")
    if kill_verdict is not True and kill_verdict is not False:
        raise ValueError("The held-out test summary has no final kill verdict")
    for summary in validation_summaries:
        if summary.get("corpus", {}).get("split") != "validation":
            raise ValueError("Ablation inputs must be validation summaries")
    champion = select_validation_champion(validation_summaries)
    for summary in [test_summary, *validation_summaries]:
        _max_samples(summary)
    identities = {
        (
            summary["corpus"]["manifest_digest"],
            summary["corpus"]["hub"]["repo_id"],
            summary["corpus"]["hub"]["revision"],
            summary["corpus"]["basis_set_id"],
            summary["corpus"]["basis_set_digest"],
        )
        for summary in [test_summary, *validation_summaries]
    }
    if len(identities) != 1:
        raise ValueError("Report inputs do not share one corpus and basis identity")

    validation_cohort_identities = {
        json.dumps(summary["cohort"], sort_keys=True)
        for summary in validation_summaries
    }
    if len(validation_cohort_identities) != 1:
        raise ValueError("Validation inputs do not share one frozen cohort")
    cohort_fields = ("seed", "selection", "tasks", "by_certified_distance")
    if any(
        test_summary["cohort"].get(field)
        != validation_summaries[0]["cohort"].get(field)
        for field in cohort_fields
    ):
        raise ValueError("Validation and test inputs do not share the cohort protocol")

    all_summaries = [test_summary, *validation_summaries]
    if (
        len(
            {
                json.dumps(_generation_protocol(summary), sort_keys=True)
                for summary in all_summaries
            }
        )
        != 1
    ):
        raise ValueError("Report inputs do not share one frozen generation protocol")
    evaluator_hashes = {_evaluator_sha256(summary) for summary in all_summaries}
    if len(evaluator_hashes) != 1:
        raise ValueError("Report inputs do not share one frozen evaluator")
    if (
        len(
            {
                json.dumps(_training_protocol(summary), sort_keys=True)
                for summary in all_summaries
            }
        )
        != 1
    ):
        raise ValueError(
            "Report inputs changed training settings beyond the ablation arm"
        )
    if (
        len(
            {
                json.dumps(
                    summary["checkpoint"]["training"]["base_model"], sort_keys=True
                )
                for summary in all_summaries
            }
        )
        != 1
    ):
        raise ValueError(
            "Report inputs do not share one pretrained base-model identity"
        )

    if require_local_artifacts:
        verified_checkpoints: set[str] = set()
        for summary in all_summaries:
            checkpoint_path = summary["checkpoint"]["path"]
            if checkpoint_path not in verified_checkpoints:
                _verify_checkpoint_artifacts(summary)
                verified_checkpoints.add(checkpoint_path)
    training_manifests = [
        _training_manifest(summary, require_local_artifacts=require_local_artifacts)
        for summary in all_summaries
    ]
    training_corpus_identities = set()
    training_runner_identities = set()
    validation_tokenization_identities = set()
    for summary, manifest in zip(all_summaries, training_manifests, strict=True):
        source_corpus = manifest.get("source_corpus", {})
        try:
            source_identity = (
                source_corpus["manifest_digest"],
                source_corpus["hub"]["repo_id"],
                source_corpus["hub"]["revision"],
                source_corpus["basis_set_id"],
                source_corpus["basis_set_digest"],
            )
            evaluation_identity = (
                summary["corpus"]["manifest_digest"],
                summary["corpus"]["hub"]["repo_id"],
                summary["corpus"]["hub"]["revision"],
                summary["corpus"]["basis_set_id"],
                summary["corpus"]["basis_set_digest"],
            )
            source_verification = source_corpus["verification"]
            validation_tokenization = manifest["tokenization"]["validation"]
            validation_role = source_corpus["task_roles"]["validation"]
            runner = manifest["source_code"]["runner"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Training manifest corpus provenance is malformed"
            ) from error
        if source_identity != evaluation_identity:
            raise ValueError("Training manifest uses a different source corpus")
        if require_local_artifacts:
            repo_root = manifest.get("repo_root")
            runner_path = runner.get("path")
            runner_sha256 = runner.get("sha256")
            if not all(
                isinstance(value, str)
                for value in (repo_root, runner_path, runner_sha256)
            ):
                raise ValueError("Training manifest runner identity is malformed")
            if _sha256_file(Path(repo_root) / runner_path) != runner_sha256:
                raise ValueError("Frozen training runner changed after training")
        verified_records = source_verification.get("records")
        if (
            source_verification.get("ok") is not True
            or source_verification.get("root_split_leaks") != []
            or not isinstance(verified_records, int)
            or verified_records <= 0
        ):
            raise ValueError("Training manifest has no clean corpus verification")
        if (
            validation_tokenization.get("retained_task_ids_digest")
            != validation_role.get("ordered_task_ids_digest")
            or validation_tokenization.get("excluded_records") != 0
        ):
            raise ValueError("Training manifest changed the frozen validation records")
        training_corpus_identities.add(source_identity)
        training_runner_identities.add(json.dumps(runner, sort_keys=True))
        validation_tokenization_identities.add(
            json.dumps(validation_tokenization, sort_keys=True)
        )
    if len(training_corpus_identities) != 1:
        raise ValueError("Training arms do not share one source corpus")
    if len(training_runner_identities) != 1:
        raise ValueError("Training arms do not share one frozen training runner")
    if len(validation_tokenization_identities) != 1:
        raise ValueError("Training arms do not share frozen validation tokenization")
    if (
        len(
            {
                _tokenizer_signature(
                    summary, require_local_artifacts=require_local_artifacts
                )
                for summary in all_summaries
            }
        )
        != 1
    ):
        raise ValueError("Training arms do not share one base tokenizer identity")
    if _checkpoint_signature(test_summary) != _checkpoint_signature(champion):
        raise ValueError("Test checkpoint is not the validation-selected champion")

    lock_status = selection_lock.get("status")
    if lock_status not in {
        "locked-before-held-out-test",
        "locked-before-held-out-test-scoring",
    }:
        raise ValueError("Validation selection lock is not complete")
    discipline = selection_lock.get("discipline", {})
    legacy_lock = lock_status == "locked-before-held-out-test"
    valid_discipline = (
        discipline.get("held_out_test_accessed") is False
        if legacy_lock
        else discipline.get("held_out_test_model_scored") is False
        and discipline.get("held_out_test_outcomes_used_for_selection") is False
    )
    if (
        not valid_discipline
        or discipline.get("selection_data") != "validation summaries only"
    ):
        raise ValueError(
            "Validation selection lock does not prove locked-test discipline"
        )
    frozen_evaluator_identity = selection_lock.get("frozen_evaluator", {})
    frozen_evaluator = frozen_evaluator_identity.get("sha256")
    if frozen_evaluator not in evaluator_hashes:
        raise ValueError("Validation selection lock has a different frozen evaluator")
    if require_local_artifacts:
        evaluator_path = frozen_evaluator_identity.get("path")
        if not isinstance(evaluator_path, str):
            raise ValueError("Validation selection lock has no evaluator path")
        if _sha256_file(Path(evaluator_path)) != frozen_evaluator:
            raise ValueError("Frozen evaluator changed after validation selection")
    lock_identity = selection_lock.get("validation_identity", {})
    validation_corpus = validation_summaries[0]["corpus"]
    locked_generation = lock_identity.get("generation_protocol")
    current_generation = _generation_protocol(validation_summaries[0])
    if not isinstance(locked_generation, dict):
        raise ValueError("Validation selection lock has no generation protocol")
    if (
        lock_identity.get("corpus_manifest_digest")
        != validation_corpus["manifest_digest"]
        or lock_identity.get("basis_set_digest")
        != validation_corpus["basis_set_digest"]
        or lock_identity.get("cohort_task_ids_digest")
        != validation_summaries[0]["cohort"]["task_ids_digest"]
        or locked_generation
        != {key: current_generation.get(key) for key in locked_generation}
    ):
        raise ValueError(
            "Validation selection lock does not match frozen validation inputs"
        )

    candidates = selection_lock.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("Validation selection lock does not contain both candidates")
    locked_by_arm = {row.get("arm_min_train_distance"): row for row in candidates}
    if set(locked_by_arm) != {1, 4} or set(validation_summary_hashes) != {1, 4}:
        raise ValueError("Validation selection lock has the wrong ablation arms")
    for summary in validation_summaries:
        arm = _arm_min_distance(summary)
        locked = locked_by_arm[arm]
        try:
            locked_signature = tuple(
                sorted(
                    (row["name"], row["sha256"])
                    for row in locked["checkpoint_signature"]
                )
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Validation selection lock signature is malformed"
            ) from error
        if (
            locked.get("summary_sha256") != validation_summary_hashes[arm]
            or locked_signature != _checkpoint_signature(summary)
            or tuple(locked.get("selection_key", ())) != _selection_key(summary)
            or locked.get("checkpoint_training_manifest_sha256")
            != summary["checkpoint"]["training_manifest_sha256"]
        ):
            raise ValueError(f"Validation selection lock does not match arm {arm}")
    locked_champion = selection_lock.get("champion", {})
    champion_arm = _arm_min_distance(champion)
    if (
        locked_champion.get("arm_min_train_distance") != champion_arm
        or locked_champion.get("summary_sha256")
        != validation_summary_hashes[champion_arm]
        or tuple(locked_champion.get("selection_key", ())) != _selection_key(champion)
    ):
        raise ValueError("Validation selection lock names a different champion")
    locked_at = _parse_timestamp(
        selection_lock.get("locked_at_utc"), label="Validation selection lock"
    )
    test_completed = _parse_timestamp(
        test_summary.get("completed_at_utc"), label="Held-out test summary"
    )
    if legacy_lock:
        if locked_at >= test_completed:
            raise ValueError(
                "Validation champion was not locked before held-out test completion"
            )
    else:
        test_access = test_summary.get("test_access")
        if not isinstance(test_access, dict):
            raise ValueError(
                "Held-out test summary has no selection-lock access record"
            )
        if (
            test_access.get("selection_lock_sha256") != selection_lock_sha256
            or test_access.get("selection_locked_at_utc")
            != selection_lock.get("locked_at_utc")
            or test_access.get("frozen_evaluator_sha256") != frozen_evaluator
        ):
            raise ValueError(
                "Held-out test did not consume this validation selection lock"
            )
        if test_access.get("locked_validation_identity") != selection_lock.get(
            "validation_identity"
        ):
            raise ValueError("Held-out test used a different locked protocol identity")
        test_scoring_started = _parse_timestamp(
            test_access.get("test_scoring_started_at_utc"),
            label="Held-out test scoring",
        )
        if locked_at >= test_scoring_started or test_scoring_started >= test_completed:
            raise ValueError(
                "Validation champion was not locked before held-out test scoring"
            )
        receipt = test_access.get("consumption_receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("selection_lock_sha256") != selection_lock_sha256
            or receipt.get("test_scoring_started_at_utc")
            != test_access.get("test_scoring_started_at_utc")
        ):
            raise ValueError("Held-out test has no valid single-use lock receipt")
        if require_local_artifacts:
            receipt_path = receipt.get("path")
            receipt_sha256 = receipt.get("sha256")
            if (
                not isinstance(receipt_path, str)
                or not isinstance(receipt_sha256, str)
                or _sha256_file(Path(receipt_path)) != receipt_sha256
            ):
                raise ValueError("Held-out test lock receipt changed after scoring")

    for summary in [test_summary, *validation_summaries]:
        run_entity, run_project, _ = _run_url_parts(summary)
        if (run_entity, run_project) != (entity, project):
            raise ValueError(
                "Evaluation run is outside the requested personal W&B project: "
                f"{run_entity}/{run_project}"
            )


def create_report(inputs: ReportInputs) -> dict[str, Any]:
    """Validate the frozen summaries and create exactly one W&B report."""

    import wandb_workspaces.reports.v2 as wr

    test_summary = _read_summary(inputs.test_summary)
    validation_summaries = [_read_summary(path) for path in inputs.validation_summaries]
    selection_lock = _read_selection_lock(inputs.selection_lock)
    selection_lock_sha256 = _sha256_file(inputs.selection_lock)
    validation_summary_hashes = {
        _arm_min_distance(summary): _sha256_file(path)
        for path, summary in zip(
            inputs.validation_summaries, validation_summaries, strict=True
        )
    }
    _validate_inputs(
        test_summary,
        validation_summaries,
        selection_lock=selection_lock,
        selection_lock_sha256=selection_lock_sha256,
        validation_summary_hashes=validation_summary_hashes,
        entity=inputs.entity,
        project=inputs.project,
    )
    markdown = build_report_markdown(
        test_summary,
        validation_summaries,
        selection_lock=selection_lock,
        selection_lock_sha256=selection_lock_sha256,
        operational_notes=inputs.operational_notes,
    )
    runset = wr.Runset(
        entity=inputs.entity,
        project=inputs.project,
        name="Issue #50 finished runs",
        filters="Metric('state') in ['finished']",
    )
    report = wr.Report(
        entity=inputs.entity,
        project=inputs.project,
        title=inputs.title,
        description=(
            "Execution-verified comparison of "
            f"{test_summary['checkpoint']['training']['base_model']['repo_id']} and "
            "exhaustive search on deep-corpus-v1."
        ),
        blocks=[
            wr.MarkdownBlock(text=markdown),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    wr.LinePlot(
                        title="Headline: distance 8 solve rate vs mean executions",
                        x="distance_8_cost/mean_executions",
                        y=[
                            "distance_8_cost/model_solve_rate",
                            "distance_8_cost/bfs_solve_rate",
                        ],
                    ),
                    wr.LinePlot(
                        title="Training and validation loss",
                        x="Step",
                        y=["train/loss", "eval/loss"],
                    ),
                    wr.LinePlot(
                        title="Test solve rate by certified distance at k=64",
                        x="by_distance/certified_distance",
                        y=[
                            "by_distance/model_solve_rate",
                            "by_distance/bfs_solve_rate_at_model_cost",
                        ],
                    ),
                    wr.LinePlot(
                        title="Distance 7-8: execution and exact solve@k",
                        x="solve_at_k/k",
                        y=[
                            "solve_at_k/execution_rate",
                            "solve_at_k/exact_match_rate",
                        ],
                    ),
                    wr.WeavePanelSummaryTable(table_name="evaluation_aggregate"),
                ],
            ),
        ],
    )
    saved = report.save()
    url = str(saved.url)
    result = {
        "entity": inputs.entity,
        "project": inputs.project,
        "title": inputs.title,
        "url": url,
        "test_summary": str(inputs.test_summary.resolve()),
        "test_summary_sha256": _sha256_file(inputs.test_summary),
        "validation_summaries": [
            str(path.resolve()) for path in inputs.validation_summaries
        ],
        "validation_summary_sha256_by_arm": {
            str(arm): digest
            for arm, digest in sorted(validation_summary_hashes.items())
        },
        "selection_lock": str(inputs.selection_lock.resolve()),
        "selection_lock_sha256": selection_lock_sha256,
        "selection_locked_at_utc": selection_lock["locked_at_utc"],
        "validation_champion": _arm_name(
            select_validation_champion(validation_summaries)
        ),
        "frozen_evaluator_sha256": _evaluator_sha256(test_summary),
        "generation_protocol": _generation_protocol(test_summary),
        "report_markdown_sha256": _sha256_text(markdown),
        "report_builder_sha256": _sha256_file(Path(__file__)),
        "operational_notes": list(inputs.operational_notes),
    }
    inputs.output.parent.mkdir(parents=True, exist_ok=True)
    inputs.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> ReportInputs:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument(
        "--validation-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--title", default="Deep corpus v1: learned policy vs exhaustive search"
    )
    parser.add_argument(
        "--operational-note",
        action="append",
        default=[],
        help="Disclosed non-scientific operational mechanism; may be repeated",
    )
    arguments = parser.parse_args(argv)
    return ReportInputs(
        test_summary=arguments.test_summary,
        validation_summaries=tuple(arguments.validation_summary),
        selection_lock=arguments.selection_lock,
        output=arguments.output,
        entity=arguments.entity,
        project=arguments.project,
        title=arguments.title,
        operational_notes=tuple(arguments.operational_note),
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = create_report(_parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
