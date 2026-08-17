"""Re-certify the repository's published shortest-path datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.generate_deep_corpus import verify_corpus
from wandering_light.shortest_path_data import (
    read_jsonl_gz,
    recertify_shortest_v1_record,
)
from wandering_light.typed_list import TypedList

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA = ROOT / "wandering_light/evals/data/random_inputs_500_shortest_v1.jsonl.gz"
INDUCTION_DATA = ROOT / "wandering_light/training/data/induction_shortest_v1.jsonl.gz"
DEEP_CORPUS = ROOT / "wandering_light/training/data/deep_corpus_v1"

PR_INDUCTION_SAMPLE = 96
DEEP_RECERTIFICATIONS = {"pr": 8, "full": 48}
OUTCOMES = ("certified", "inflated", "inconclusive")
MAX_EXAMPLES = 5


def _contains_zero_float(value: object) -> bool:
    if type(value) is float:
        return value == 0.0
    if isinstance(value, dict):
        return any(
            _contains_zero_float(key) or _contains_zero_float(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return any(_contains_zero_float(item) for item in value)
    return False


def _induction_stratum(record: dict[str, Any]) -> tuple[int, bool]:
    target = TypedList.from_str(record["output"])
    return int(record["relabeled_length"]), _contains_zero_float(target.items)


def _record_sort_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(record["source_index"]),
        str(record["input"]),
        str(record["output"]),
    )


def _stratified_induction_sample(
    records: Iterable[dict[str, Any]], sample_size: int = PR_INDUCTION_SAMPLE
) -> list[dict[str, Any]]:
    """Round-robin a stable ordering of every distance x risk stratum."""
    strata: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[_induction_stratum(record)].append(record)
    for pool in strata.values():
        pool.sort(key=_record_sort_key)

    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    offset = 0
    while len(selected) < sample_size:
        added = False
        for key in keys:
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                added = True
                if len(selected) == sample_size:
                    return selected
        if not added:
            break
        offset += 1
    raise ValueError(
        f"cannot draw {sample_size} records from {sum(map(len, strata.values()))} rows"
    )


def _scalar_result_fields(result: object) -> dict[str, bool | int | float | str | None]:
    if is_dataclass(result):
        values = ((field.name, getattr(result, field.name)) for field in fields(result))
    elif hasattr(result, "_asdict"):
        values = result._asdict().items()
    else:
        values = vars(result).items()
    return {
        name: value
        for name, value in values
        if name != "outcome"
        and (value is None or isinstance(value, bool | int | float | str))
    }


def _verify_shortest_v1_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "inflated": [],
        "inconclusive": [],
    }
    for record in records:
        result = recertify_shortest_v1_record(record)
        outcome = result.outcome
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown re-certification outcome: {outcome!r}")
        counts[outcome] += 1
        if outcome != "certified" and len(examples[outcome]) < MAX_EXAMPLES:
            examples[outcome].append(
                {
                    "source_index": record["source_index"],
                    **_scalar_result_fields(result),
                }
            )

    summary: dict[str, Any] = {
        "records": len(records),
        "outcomes": {outcome: counts[outcome] for outcome in OUTCOMES},
        "ok": counts["certified"] == len(records),
    }
    for outcome, rows in examples.items():
        if rows:
            summary[f"{outcome}_examples"] = rows
    return summary


def _stratum_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_induction_stratum(record) for record in records)
    return {
        f"distance={distance},zero_float_target={str(zero).lower()}": count
        for (distance, zero), count in sorted(counts.items())
    }


def _deep_summary(result: dict[str, Any]) -> dict[str, Any]:
    recertified = result["recertified"]
    inconclusive = result["recertification_inconclusive"]
    summary: dict[str, Any] = {
        "records": result["records"],
        "witness_failures": len(result["witness_failures"]),
        "split_leaks": len(result["roots_leaking_across_splits"]),
        "recertified": len(recertified),
        "inflated": len(result["recertification_failures"]),
        "inconclusive": len(inconclusive),
        "ok": bool(result["ok"]) and not inconclusive,
    }
    if result["witness_failures"]:
        summary["witness_failure_examples"] = result["witness_failures"][:MAX_EXAMPLES]
    if result["recertification_failures"]:
        summary["inflated_examples"] = result["recertification_failures"][:MAX_EXAMPLES]
    if inconclusive:
        summary["inconclusive_examples"] = inconclusive[:MAX_EXAMPLES]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=DEEP_RECERTIFICATIONS, required=True)
    args = parser.parse_args(argv)

    evaluation = read_jsonl_gz(EVAL_DATA)
    induction = read_jsonl_gz(INDUCTION_DATA)
    induction_selection = (
        induction if args.profile == "full" else _stratified_induction_sample(induction)
    )

    print(
        f"re-certifying eval shortest paths ({len(evaluation):,} records)", flush=True
    )
    eval_summary = _verify_shortest_v1_records(evaluation)
    print(
        f"re-certifying induction shortest paths ({len(induction_selection):,} records)",
        flush=True,
    )
    induction_summary = _verify_shortest_v1_records(induction_selection)
    induction_summary["strata"] = _stratum_counts(induction_selection)

    deep_count = DEEP_RECERTIFICATIONS[args.profile]
    print(
        f"verifying deep corpus ({deep_count} distance re-certifications)", flush=True
    )
    deep_summary = _deep_summary(verify_corpus(DEEP_CORPUS, recertify=deep_count))

    ok = eval_summary["ok"] and induction_summary["ok"] and deep_summary["ok"]
    summary = {
        "profile": args.profile,
        "eval": eval_summary,
        "induction": induction_summary,
        "deep": deep_summary,
        "ok": ok,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
