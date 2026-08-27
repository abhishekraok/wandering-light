"""Train and freeze issue #54's shared deep-corpus-v2 tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from wandering_light.training.deep_corpus_tokenizer import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_REFERENCE_CORPUS_DIR,
    DEFAULT_VOCAB_SIZE,
    FIXED_MAX_LENGTH,
    SharedTokenizerConfig,
    dry_run_summary,
    freeze_shared_tokenizer,
    load_verified_v2_discovery,
)


def run(
    *,
    corpus_dir: Path,
    output_dir: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    reference_corpus_dirs: Sequence[Path] = (DEFAULT_REFERENCE_CORPUS_DIR,),
    dry_run: bool = False,
) -> dict[str, Any]:
    source = load_verified_v2_discovery(
        corpus_dir,
        reference_corpus_dirs=reference_corpus_dirs,
    )
    if dry_run:
        return dry_run_summary(source)
    return freeze_shared_tokenizer(
        output_dir,
        rows=source.rows,
        basis=source.basis,
        source_identity=source.source_identity,
        config=SharedTokenizerConfig(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_length=FIXED_MAX_LENGTH,
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument(
        "--reference-corpus-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Pinned corpus whose roots/tasks V2 excludes; repeat for every "
            "target-contract reference (defaults to deep-corpus-v1)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    result = run(
        corpus_dir=arguments.corpus_dir,
        output_dir=arguments.output_dir,
        vocab_size=arguments.vocab_size,
        min_frequency=arguments.min_frequency,
        reference_corpus_dirs=tuple(
            arguments.reference_corpus_dir or (DEFAULT_REFERENCE_CORPUS_DIR,)
        ),
        dry_run=arguments.dry_run,
    )
    # The tracked manifest is compact; detailed token-length telemetry is saved
    # beside it for Hub publication. Stdout carries identity and headline only.
    if result.get("status") == "dry-run":
        printable = result
    else:
        printable = {
            "artifact_dir": str(arguments.output_dir.resolve()),
            "artifact_digest": result["artifact_digest"],
            "source_manifest_digest": result["source"]["manifest_digest"],
            "total_vocab_size": result["tokenizer"]["total_vocab_size"],
            "max_length": result["tokenizer"]["max_length"],
            "selected_records": result["selection"]["selected_records"],
            "retained_records": result["selection"]["retained_records"],
            "excluded_records": result["selection"]["excluded_records"],
        }
    print(json.dumps(printable, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
