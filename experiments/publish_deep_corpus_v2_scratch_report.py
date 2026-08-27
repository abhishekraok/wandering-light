"""Publish one sealed issue #54 local report to the frozen personal W&B project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from experiments.create_deep_corpus_v2_scratch_report import (
    REPORT_DIGEST_FIELD,
    REPORT_JSON_FILENAME,
    REPORT_MARKDOWN_FILENAME,
    validate_report,
)
from wandering_light.evals.deep_corpus_protocol import sha256_file, sha256_json
from wandering_light.training.deep_corpus_scratch import (
    DEFAULT_WANDB_PROJECT,
    PERSONAL_WANDB_ENTITY,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "wandering-light-deep-corpus-v2-scratch-wandb-report-receipt"
RECEIPT_DIGEST_FIELD = "receipt_sha256"
INTENT_DIGEST_FIELD = "intent_sha256"
DEFAULT_TITLE = "Deep corpus v2: scratch scaling at 10M, 125M, and 360M"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_report_artifacts(report_dir: str | Path) -> tuple[dict[str, Any], str]:
    """Revalidate the exact local JSON/Markdown pair before any network write."""

    root = Path(report_dir).resolve()
    report_path = root / REPORT_JSON_FILENAME
    markdown_path = root / REPORT_MARKDOWN_FILENAME
    report = validate_report(_read_json(report_path, label="sealed report"))
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            f"Could not read report Markdown {markdown_path}: {error}"
        ) from error
    markdown_digest = "sha256:" + hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    if report.get("report_markdown_sha256") != markdown_digest:
        raise ValueError("Report Markdown differs from the sealed JSON identity")
    builder = Path(__file__).with_name("create_deep_corpus_v2_scratch_report.py")
    if report.get("report_builder_sha256") != sha256_file(builder):
        raise ValueError("Report was created by another report-builder source")
    return report, markdown


def _validate_report_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() not in {"wandb.ai", "www.wandb.ai"}
        or len(parts) < 4
        or parts[:3] != [PERSONAL_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, "reports"]
        or not parts[3]
    ):
        raise ValueError(f"W&B returned an unexpected report URL: {url!r}")
    return url


def _seal_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(body)
    receipt[RECEIPT_DIGEST_FIELD] = sha256_json(receipt)
    return receipt


def validate_receipt(
    receipt: Mapping[str, Any], *, report: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(receipt)
    digest = value.pop(RECEIPT_DIGEST_FIELD, None)
    if digest != sha256_json(value):
        raise ValueError("W&B report receipt changed after publication")
    value[RECEIPT_DIGEST_FIELD] = digest
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact_type") != ARTIFACT_TYPE
        or value.get("status") != "published"
        or value.get("entity") != PERSONAL_WANDB_ENTITY
        or value.get("project") != DEFAULT_WANDB_PROJECT
        or value.get("report_sha256") != report.get(REPORT_DIGEST_FIELD)
    ):
        raise ValueError("W&B report receipt does not match the frozen study report")
    _validate_report_url(str(value.get("url")))
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"W&B report receipt already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, allow_nan=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _intent_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f".{receipt_path.name}.publication-intent.json")


def _publication_intent(*, report: Mapping[str, Any], title: str) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "wandering-light-deep-corpus-v2-scratch-wandb-report-intent",
        "entity": PERSONAL_WANDB_ENTITY,
        "project": DEFAULT_WANDB_PROJECT,
        "title": title,
        "report_sha256": report[REPORT_DIGEST_FIELD],
    }
    return {**body, INTENT_DIGEST_FIELD: sha256_json(body)}


def _validate_intent(
    intent: Mapping[str, Any], *, report: Mapping[str, Any], title: str
) -> dict[str, Any]:
    value = dict(intent)
    digest = value.pop(INTENT_DIGEST_FIELD, None)
    if digest != sha256_json(value):
        raise ValueError("W&B report publication intent changed")
    value[INTENT_DIGEST_FIELD] = digest
    expected = _publication_intent(report=report, title=title)
    if value != expected:
        raise ValueError("W&B report publication intent belongs to another report")
    return value


def _build_receipt(
    *,
    report_dir: str | Path,
    report: Mapping[str, Any],
    title: str,
    url: str,
) -> dict[str, Any]:
    return _seal_receipt(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "status": "published",
            "entity": PERSONAL_WANDB_ENTITY,
            "project": DEFAULT_WANDB_PROJECT,
            "title": title,
            "url": _validate_report_url(url),
            "report_sha256": report[REPORT_DIGEST_FIELD],
            "report_json_sha256": sha256_file(
                Path(report_dir).resolve() / REPORT_JSON_FILENAME
            ),
            "report_markdown_sha256": report["report_markdown_sha256"],
            "publisher_sha256": sha256_file(Path(__file__)),
        }
    )


def publish_report(
    *,
    report_dir: str | Path,
    receipt_path: str | Path,
    title: str = DEFAULT_TITLE,
    recovered_url: str | None = None,
) -> dict[str, Any]:
    """Create one W&B Report, then atomically record its content-bound URL."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("W&B report title must be non-empty")
    report, markdown = load_report_artifacts(report_dir)
    destination = Path(receipt_path).resolve()
    intent_path = _intent_path(destination)
    if destination.exists():
        receipt = validate_receipt(
            _read_json(destination, label="W&B report receipt"), report=report
        )
        intent_path.unlink(missing_ok=True)
        return receipt

    intent = _publication_intent(report=report, title=title)
    if intent_path.exists():
        _validate_intent(
            _read_json(intent_path, label="W&B report publication intent"),
            report=report,
            title=title,
        )
        if recovered_url is None:
            raise RuntimeError(
                "A W&B publication may already have succeeded; inspect the fixed "
                "project and rerun with --recover-url instead of creating a duplicate"
            )
        receipt = _build_receipt(
            report_dir=report_dir,
            report=report,
            title=title,
            url=recovered_url,
        )
        _write_json_exclusive(destination, receipt)
        intent_path.unlink()
        return validate_receipt(
            _read_json(destination, label="W&B report receipt"), report=report
        )
    if recovered_url is not None:
        raise ValueError("--recover-url requires an existing publication intent")
    _write_json_exclusive(intent_path, intent)

    import wandb_workspaces.reports.v2 as wr

    hosted = wr.Report(
        entity=PERSONAL_WANDB_ENTITY,
        project=DEFAULT_WANDB_PROJECT,
        title=title,
        description=(
            "Execution-verified, once-opened issue #54 scratch scaling study on "
            "deep-corpus-v2."
        ),
        blocks=[wr.MarkdownBlock(text=markdown)],
    ).save()
    receipt = _build_receipt(
        report_dir=report_dir,
        report=report,
        title=title,
        url=str(hosted.url),
    )
    _write_json_exclusive(destination, receipt)
    intent_path.unlink()
    return validate_receipt(
        _read_json(destination, label="W&B report receipt"), report=report
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument(
        "--recover-url",
        help="finalize a pending intent after confirming the already-created report URL",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = publish_report(
        report_dir=args.report_dir,
        receipt_path=args.receipt,
        title=args.title,
        recovered_url=args.recover_url,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
