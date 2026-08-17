"""Re-certify committed ``shortest_v1`` datasets from fresh BFS expansions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.shortest_path_data import (
    RecertificationStatus,
    read_jsonl_gz,
    recertify_shortest_record,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_DATASETS = (
    Path("wandering_light/evals/data/random_inputs_500_shortest_v1.jsonl.gz"),
    Path("wandering_light/training/data/induction_shortest_v1.jsonl.gz"),
)


def verify_dataset(
    path: str | Path,
    *,
    max_states: int | None = None,
    max_transitions: int | None = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Return a compact summary; retain details only for non-passing records."""
    source = Path(path)
    records = read_jsonl_gz(source)
    counts = {status.value: 0 for status in RecertificationStatus}
    errors: list[dict[str, Any]] = []
    non_passing: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, record in enumerate(records):
        try:
            result = recertify_shortest_record(
                record,
                max_states=max_states,
                max_transitions=max_transitions,
            )
        except Exception as error:
            errors.append(
                {
                    "record_index": index,
                    "source_index": record.get("source_index"),
                    "error": str(error),
                }
            )
        else:
            counts[result.status.value] += 1
            if result.status is not RecertificationStatus.CERTIFIED:
                non_passing.append(
                    {
                        "record_index": index,
                        "source_index": record.get("source_index"),
                        **asdict(result),
                    }
                )
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"{source}: re-certified {index + 1}/{len(records)} records",
                file=sys.stderr,
                flush=True,
            )

    return {
        "path": str(source),
        "records": len(records),
        "outcomes": counts,
        "errors": errors,
        "non_passing": non_passing,
        "seconds": round(time.perf_counter() - started, 2),
        "ok": not errors and not non_passing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASETS),
        help="shortest_v1 .jsonl.gz files (defaults to both committed datasets)",
    )
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = [
        verify_dataset(
            path,
            max_states=args.max_states,
            max_transitions=args.max_transitions,
            progress_every=args.progress_every,
        )
        for path in args.datasets
    ]
    result = {"datasets": summaries, "ok": all(row["ok"] for row in summaries)}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
