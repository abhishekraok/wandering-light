"""Reusable retrieval metrics, lexical baselines and executable evaluation.

Candidate ordering, seeded sampling and hidden-output grading live here so
callers can evaluate an embedding without importing the training driver.
"""

from __future__ import annotations

import hashlib
import random
import signal
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.nn import functional


def _partition(record) -> str:
    return record.provenance.get("partition", "unknown")


def _sample_examples(examples, count: int, rng: random.Random):
    return rng.sample(examples, min(count, len(examples)))


def retrieval_metrics(ranks: list[int]) -> dict[str, float | int | None]:
    """Ranks are one-based; zero denotes no positive candidate."""
    return {
        "queries": len(ranks),
        **{
            f"recall@{k}": sum(0 < rank <= k for rank in ranks) / len(ranks)
            if ranks
            else None
            for k in (1, 5, 10)
        },
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks)
        if ranks
        else None,
    }


def lexical_vectors(programs: list[str]) -> torch.Tensor:
    """Token-bag cosine baseline, with a vocabulary built from query and bank."""
    vocabulary = {
        word: i for i, word in enumerate(sorted(set(" ".join(programs).split())))
    }
    result = torch.zeros(len(programs), max(1, len(vocabulary)))
    for i, program in enumerate(programs):
        for word, count in Counter(program.split()).items():
            result[i, vocabulary[word]] = count
    return functional.normalize(result, dim=-1)


def _batched_encode(model, items, *, io: bool = False, batch_size: int = 16):
    encode = model.encode_io if io else model.encode_programs
    order = sorted(
        range(len(items)),
        key=lambda i: (
            sum(len(a) + len(b) for a, b in items[i]) if io else len(items[i])
        ),
    )
    sorted_items = [items[i] for i in order]
    vectors = torch.cat(
        [
            encode(sorted_items[i : i + batch_size]).detach().cpu()
            for i in range(0, len(items), batch_size)
        ]
    )
    inverse = torch.tensor(sorted(range(len(order)), key=order.__getitem__))
    return vectors[inverse]


def _rank(scores: torch.Tensor, positives: set[int], excluded: set[int] | None = None):
    order = scores.argsort(descending=True, stable=True).tolist()
    order = [i for i in order if i not in (excluded or set())]
    rank = next((j + 1 for j, i in enumerate(order) if i in positives), 0)
    return rank, order[:10]


def _execute_matches(program: str, examples) -> bool:
    from wandering_light.stack_machine import parse_stack, value_key
    from wandering_light.stack_pbe import BASIS, run_program

    if not examples:
        return False
    for given, wanted in examples:
        result = run_program(program.split(), parse_stack(given, basis=BASIS))
        if isinstance(result, str):
            return False
        actual, _ = result
        expected = parse_stack(wanted, basis=BASIS)
        if tuple(map(value_key, actual)) != tuple(map(value_key, expected)):
            return False
    return True


def _matches(program: str, examples, seconds: float = 1.0) -> bool | None:
    """None means incomplete execution, never an incorrect semantic result."""

    def timeout(_signum, _frame):
        raise TimeoutError("candidate execution deadline")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return _execute_matches(program, examples)
    except TimeoutError:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def reciprocal_rank_bounds(outcomes):
    """Truncated reciprocal-rank bounds; an unknown is neither pass nor fail."""
    lower = next((1 / (i + 1) for i, v in enumerate(outcomes) if v is True), 0.0)
    upper = next((1 / (i + 1) for i, v in enumerate(outcomes) if v is not False), 0.0)
    return [lower, upper]


def _known_any(values):
    return True if any(v is True for v in values) else None if None in values else False


def _both(a, b):
    return (
        False if a is False or b is False else None if a is None or b is None else True
    )


def _balanced_metrics(groups):
    rows = [retrieval_metrics(ranks) for ranks in groups if ranks]
    if not rows:
        return {**retrieval_metrics([]), "behaviours": 0}
    return {
        "queries": sum(r["queries"] for r in rows),
        "behaviours": len(rows),
        **{
            key: sum(r[key] for r in rows) / len(rows)
            for key in ("recall@1", "recall@5", "recall@10", "mrr")
        },
    }


def _length_bucket(length: int) -> str:
    for limit in (10, 40, 100):
        if length <= limit:
            return f"<={limit}"
    return ">100"


@torch.no_grad()
def evaluate(
    model,
    records,
    *,
    split: str = "validation",
    seed: int = 0,
    io_examples: int = 5,
    execution: bool = True,
    execution_examples: int = 16,
    max_eval_records: int = 128,
    global_bank: bool = False,
) -> dict[str, Any]:
    """No labels or hidden outputs enter either learned encoder."""
    model.eval()
    groups = defaultdict(list)
    for record in records:
        if record.split == split and record.examples and record.programs:
            groups[_partition(record)].append(record)
    output: dict[str, Any] = {
        "split": split,
        "evaluation_config": {
            "split": split,
            "seed": seed,
            "io_examples": io_examples,
            "execution": execution,
            "execution_examples": execution_examples,
            "max_eval_records": max_eval_records,
            "global_bank": global_bank,
        },
        "partitions": {},
        "records": [],
        "program_records": [],
    }
    reference_records = (
        records
        if split == "test"
        else [r for r in records if r.split in ("train", "validation")]
    )
    global_features = None
    for partition, full_group in sorted(groups.items()):
        group = full_group
        rng = random.Random(seed)
        full_count = len(group)
        bank_records = reference_records if global_bank else group
        bank = sorted(
            {program for record in bank_records for program in record.programs}
        )
        bank_index = {program: i for i, program in enumerate(bank)}
        if global_features is not None:
            bank_vectors = global_features
        else:
            bank_vectors = functional.normalize(_batched_encode(model, bank), dim=-1)
            if global_bank:
                global_features = bank_vectors
        if max_eval_records and len(group) > max_eval_records:
            group = sorted(rng.sample(group, max_eval_records), key=lambda r: r.key)
        inputs = [_sample_examples(r.examples, io_examples, rng) for r in group]
        io_vectors = functional.normalize(
            _batched_encode(model, inputs, io=True), dim=-1
        )
        io_scores = io_vectors @ bank_vectors.T
        lexical = lexical_vectors(bank)
        io_ranks, program_ranks, lexical_ranks = [], [], []
        style_ranks = defaultdict(list)
        buckets = defaultdict(list)
        verification = []
        program_outputs = {}
        for record in full_group:
            positives = {bank_index[p] for p in record.programs}
            cross_style_rows = []
            behaviour_program_ranks, behaviour_lexical_ranks = [], []
            for program, style in zip(record.programs, record.styles, strict=True):
                index = bank_index[program]
                equivalent_other_styles = {
                    bank_index[p]
                    for p, s in zip(record.programs, record.styles, strict=True)
                    if s != style and p != program
                }
                if not equivalent_other_styles:
                    continue
                # Same-style equivalents are neutral, not positives or negatives.
                excluded = positives - equivalent_other_styles
                p_rank, _ = _rank(
                    bank_vectors[index] @ bank_vectors.T,
                    equivalent_other_styles,
                    excluded,
                )
                l_rank, _ = _rank(
                    lexical[index] @ lexical.T, equivalent_other_styles, excluded
                )
                behaviour_program_ranks.append(p_rank)
                behaviour_lexical_ranks.append(l_rank)
                style_ranks[style].append(p_rank)
                cross_style_rows.append(
                    {
                        "style": style,
                        "program": program,
                        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                        "rank": p_rank,
                        "lexical_rank": l_rank,
                    }
                )
            program_ranks.append(behaviour_program_ranks)
            lexical_ranks.append(behaviour_lexical_ranks)
            program_outputs[record.key] = cross_style_rows
            if cross_style_rows:
                output["program_records"].append(
                    {
                        "key": record.key,
                        "partition": partition,
                        "cross_style": cross_style_rows,
                    }
                )
        for row, record in enumerate(group):
            positives = {bank_index[p] for p in record.programs}
            rank, top = _rank(io_scores[row], positives)
            io_ranks.append(rank)
            bucket = _length_bucket(min(len(p.split()) for p in record.programs))
            buckets[bucket].append(rank)
            record_output = {
                "key": record.key,
                "partition": partition,
                "family": record.family,
                "program_lengths": [len(p.split()) for p in record.programs],
                "styles": record.styles,
                "io_rank": rank,
                "top10": [bank[i] for i in top],
                "cross_style": program_outputs[record.key],
            }
            if execution:
                hidden = _sample_examples(
                    record.held_out_examples, execution_examples, rng
                )
                shown_pass = [_matches(bank[i], inputs[row]) for i in top]
                hidden_pass = (
                    [
                        False if passed is False else _matches(bank[i], hidden)
                        for i, passed in zip(top, shown_pass, strict=True)
                    ]
                    if hidden
                    else None
                )
                selected = next(
                    (i for i, passed in enumerate(shown_pass) if passed is True), None
                )
                selected_hidden = (
                    hidden_pass[selected]
                    if selected is not None and hidden_pass is not None
                    else None
                    if hidden_pass is None or None in shown_pass
                    else False
                )
                verified = {
                    "shown@1": shown_pass[0],
                    "shown@10": _known_any(shown_pass),
                    "shown_mrr@10_bounds": reciprocal_rank_bounds(shown_pass),
                    "hidden@1": _both(shown_pass[0], hidden_pass[0])
                    if hidden_pass is not None
                    else None,
                    "selected_hidden@10": selected_hidden,
                    "oracle_hidden@10": _known_any(
                        [
                            _both(a, b)
                            for a, b in zip(shown_pass, hidden_pass, strict=True)
                        ]
                    )
                    if hidden_pass is not None
                    else None,
                    "execution_timeouts": sum(x is None for x in shown_pass)
                    + sum(x is None for x in (hidden_pass or [])),
                    "shown_examples": len(inputs[row]),
                    "hidden_examples": len(hidden),
                }
                verification.append(verified)
                record_output["execution"] = verified
            output["records"].append(record_output)
        execution_summary = {}
        for name in (
            "shown@1",
            "shown@10",
            "hidden@1",
            "selected_hidden@10",
            "oracle_hidden@10",
        ):
            values = [r[name] for r in verification if r[name] is not None]
            execution_summary[name] = sum(values) / len(values) if values else None
            execution_summary[name + "_scored_queries"] = len(values)
            unknown = len(verification) - len(values)
            execution_summary[name + "_unknown_queries"] = unknown
            execution_summary[name + "_bounds"] = (
                [
                    sum(values) / len(verification),
                    (sum(values) + unknown) / len(verification),
                ]
                if verification
                else None
            )
        execution_summary["shown_mrr@10_bounds"] = (
            [
                sum(r["shown_mrr@10_bounds"][i] for r in verification)
                / len(verification)
                for i in (0, 1)
            ]
            if verification
            else None
        )
        execution_summary["shown_mrr@10_unknown_queries"] = sum(
            r["shown_mrr@10_bounds"][0] != r["shown_mrr@10_bounds"][1]
            for r in verification
        )
        output["partitions"][partition] = {
            "behaviours": len(group),
            "full_partition_behaviours": full_count,
            "query_sampling": "I/O: seeded uniform behaviour subset; cross-style programs: all eligible behaviours; complete candidate bank",
            "bank_programs": len(bank),
            "bank_scope": (
                "all train, validation and test records"
                if split == "test"
                else "all train and validation records"
            )
            if global_bank
            else "selected partition",
            "bank_coverage": 1.0,
            "bank_assumption": "All known programs of every selected held-out behaviour are in the bank; absent-program synthesis is not measured.",
            "io_to_program": retrieval_metrics(io_ranks),
            "cross_style_program": _balanced_metrics(program_ranks),
            "lexical_cross_style": _balanced_metrics(lexical_ranks),
            "by_query_style": {
                key: retrieval_metrics(value)
                for key, value in sorted(style_ranks.items())
            },
            "by_shortest_program_length": {
                key: retrieval_metrics(value) for key, value in sorted(buckets.items())
            },
            "execution": execution_summary,
            "execution_max_hidden_examples": execution_examples,
            "execution_seconds_per_candidate_set": 1.0,
            "execution_definition": "selected_hidden@10 grades the first shown-passing candidate; oracle_hidden@10 accepts any candidate using hidden outputs. Timeout outcomes are unknown, with bounds over all queries.",
            "execution_timeouts": sum(r["execution_timeouts"] for r in verification),
        }
    return output
