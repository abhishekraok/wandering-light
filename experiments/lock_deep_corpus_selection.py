"""Freeze issue #50's validation-selected champion before held-out test scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.create_deep_corpus_report import create_validation_selection_lock

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> tuple[tuple[Path, ...], Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-summary",
        type=Path,
        action="append",
        required=True,
        help="Complete validation summary; pass once for each ablation arm",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summaries = tuple(arguments.validation_summary)
    if len(summaries) != 2:
        parser.error("exactly two --validation-summary arguments are required")
    return summaries, arguments.output


def main(argv: Sequence[str] | None = None) -> None:
    summaries, output = _parse_args(argv)
    lock = create_validation_selection_lock(summaries, output=output)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
