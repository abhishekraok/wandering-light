"""Create issue #50's final W&B report from verified evaluation summaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    output: Path
    entity: str = PERSONAL_WANDB_ENTITY
    project: str = DEFAULT_WANDB_PROJECT
    title: str = "Deep corpus v1: learned policy vs exhaustive search"


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


def build_report_markdown(
    test_summary: Mapping[str, Any],
    validation_summaries: Sequence[Mapping[str, Any]],
) -> str:
    champion = select_validation_champion(validation_summaries)
    if _checkpoint_signature(test_summary) != _checkpoint_signature(champion):
        raise ValueError(
            "The test checkpoint is not the validation-selected ablation champion"
        )
    max_samples = int(_max_samples(test_summary))
    primary = test_summary["matched_compute"]["7-8"]["matched"][str(max_samples)]
    verdict = test_summary["verdict"]
    killed = verdict.get("kill_criterion_triggered")
    if killed is True:
        verdict_text = (
            "**Kill criterion: TRIGGERED.** The autoregressive solver is not cheaper "
            "than exhaustive search at equal solve rate on certified distance 7-8. "
            "On this substrate, the autoregressive approach is finished."
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
                _pct(row["solve_rate"]),
                _pct(row["exact_match_rate"]),
                f"{row['mean_executions_spent']:.1f}",
                f"{row['bfs_mean_executions_at_model_mean_execution_spend']:.1f}",
                _pct(row["bfs_solve_rate_at_model_mean_execution_spend"]),
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
                _pct(primary["solve_rate"]),
                _pct(primary["exact_match_rate"]),
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
        by_distance_rows.append(
            [
                distance,
                _pct(row["solve_rate"]),
                _pct(row["exact_match_rate"]),
                f"{row['mean_executions_spent']:.1f}",
                _pct(row["bfs_solve_rate_at_model_mean_execution_spend"]),
                _ratio(row["bfs_to_model_total_execution_ratio"]),
            ]
        )
    distance_table = _markdown_table(
        [
            "certified distance",
            f"model solve@{max_samples}",
            "exact match",
            "model mean exec",
            "BFS solve at model cost",
            "BFS/model at equal solve count",
        ],
        by_distance_rows,
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
        selected = _checkpoint_signature(summary) == _checkpoint_signature(champion)
        validation_rows.append(
            [
                _arm_name(summary),
                _pct(row["solve_rate"]),
                _pct(row["exact_match_rate"]),
                f"{row['mean_executions_spent']:.1f}",
                "yes" if selected else "no",
            ]
        )
    validation_table = _markdown_table(
        ["training arm", "validation solve@64", "exact@64", "mean exec", "selected"],
        validation_rows,
    )

    corpus = test_summary["corpus"]
    checkpoint = test_summary["checkpoint"]["training"]
    base_model = checkpoint["base_model"]
    return "\n\n".join(
        [
            "## Verdict\n" + verdict_text,
            "## Headline: distance-8 solve rate against executions spent\n"
            + distance_8_headline,
            "## Combined distance 7-8 kill criterion\n" + combined_verdict,
            "## Solve rate by certified distance\n"
            "Model and BFS use the same task IDs. BFS is evaluated at the model's "
            "mean execution spend and at the budget needed to reach at least the "
            "same solve count.\n\n" + distance_table,
            "## Solve@k and exact-match gap\n" + solve_at_k_table,
            "## Validation-only ablation selection\n" + validation_table,
            "## Provenance\n"
            f"- Corpus manifest: `{corpus['manifest_digest']}`\n"
            f"- Hub revision: `{corpus['hub']['revision']}`\n"
            f"- Basis: `{corpus['basis_set_id']}` / `{corpus['basis_set_digest']}`\n"
            f"- Base model: `{base_model['repo_id']}` at `{base_model['revision']}`\n"
            f"- Test cohort task-ID digest: `{test_summary['cohort']['task_ids_digest']}`\n"
            f"- Training run: {test_summary['checkpoint']['training_wandb_url']}\n"
            f"- Test evaluation run: {test_summary['wandb']['url']}",
        ]
    )


def _validate_inputs(
    test_summary: Mapping[str, Any],
    validation_summaries: Sequence[Mapping[str, Any]],
    *,
    entity: str,
    project: str,
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
    select_validation_champion(validation_summaries)
    for summary in [test_summary, *validation_summaries]:
        _max_samples(summary)
    identities = {
        (
            summary["corpus"]["manifest_digest"],
            summary["corpus"]["basis_set_digest"],
        )
        for summary in [test_summary, *validation_summaries]
    }
    if len(identities) != 1:
        raise ValueError("Report inputs do not share one corpus and basis identity")
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
    _validate_inputs(
        test_summary,
        validation_summaries,
        entity=inputs.entity,
        project=inputs.project,
    )
    markdown = build_report_markdown(test_summary, validation_summaries)
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
            "Execution-verified comparison of a 125M autoregressive solver and "
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
        "validation_summaries": [
            str(path.resolve()) for path in inputs.validation_summaries
        ],
        "validation_champion": _arm_name(
            select_validation_champion(validation_summaries)
        ),
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
    parser.add_argument("--entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--title", default="Deep corpus v1: learned policy vs exhaustive search"
    )
    arguments = parser.parse_args(argv)
    return ReportInputs(
        test_summary=arguments.test_summary,
        validation_summaries=tuple(arguments.validation_summary),
        output=arguments.output,
        entity=arguments.entity,
        project=arguments.project,
        title=arguments.title,
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = create_report(_parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
