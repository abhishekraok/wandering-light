"""Train semantic program/I/O embeddings and measure held-out retrieval.

The program bank contains the complete selected held-out partition, including
all known equivalent strings. This measures retrieval when a correct program
exists in the bank; it does not measure synthesis of an absent program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from wandering_light.program_embedding import (
    EmbeddingConfig,
    ProgramEmbedding,
    symmetric_multi_positive_contrastive_loss,
)
from wandering_light.program_embedding_data import load_records, manifest_for


def exclude_programs(records, digests):
    """Remove audited variants after split assignment; preserve equivalence groups."""
    excluded = set(digests)
    result, count = [], 0
    for record in records:
        keep = [
            i
            for i, p in enumerate(record.programs)
            if hashlib.sha256(p.encode()).hexdigest() not in excluded
        ]
        removed = len(record.programs) - len(keep)
        count += removed
        provenance = dict(record.provenance)
        if removed:
            provenance["caveats"] = [
                *provenance.get("caveats", []),
                f"{removed} explicitly audited program variant(s) excluded after grouping/split assignment",
            ]
        result.append(
            replace(
                record,
                programs=[record.programs[i] for i in keep],
                styles=[record.styles[i] for i in keep],
                provenance=provenance,
            )
        )
    return result, count


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


def _validation_score(report) -> float:
    metrics = [p["io_to_program"]["mrr"] for p in report["partitions"].values()]
    return sum(metrics) / len(metrics) if metrics else -math.inf


def _git_metadata(repo_root: Path) -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(args) -> dict:
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.repo_root, extra_task_paths=args.extra_tasks)
    records, excluded_count = exclude_programs(records, args.exclude_program_sha256)
    manifest = manifest_for(records, args.repo_root)
    manifest["excluded_program_sha256"] = args.exclude_program_sha256
    manifest["excluded_program_variants"] = excluded_count
    config = EmbeddingConfig(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        byte_dim=args.byte_dim,
        max_program_bytes=args.max_program_bytes,
        max_example_bytes=args.max_example_bytes,
    )
    model = ProgramEmbedding(config).to(args.device)
    metadata = {
        **_git_metadata(Path(__file__).resolve().parents[1]),
        "source_root": str(Path(__file__).resolve().parents[1]),
        "asset_root": str(args.repo_root),
        "asset_git": _git_metadata(args.repo_root),
        "config": asdict(config),
        "training": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "extra_tasks"
        },
        "extra_task_paths": [str(p) for p in args.extra_tasks],
        "data": manifest,
        "seed": args.seed,
        "wandb_url": None,
    }
    metadata["data_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    if args.checkpoint:
        model = ProgramEmbedding.load(args.checkpoint, device=args.device)
        metadata["config"] = asdict(model.config)
        metadata["initial_checkpoint_metadata"] = model.metadata
        metadata["initial_checkpoint_digest"] = hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest()
    if args.eval_only or args.untrained_eval:
        metadata["checkpoint_metadata"] = model.metadata
        metadata["config"] = asdict(model.config)
        _write_json(args.out / "provenance.json", metadata)
        metadata["initialization"] = "random" if args.untrained_eval else "checkpoint"
        report = evaluate(
            model,
            records,
            split=args.eval_split,
            seed=args.seed,
            io_examples=args.io_examples,
            execution=not args.skip_execution,
            execution_examples=args.execution_examples,
            max_eval_records=args.max_eval_records,
            global_bank=args.global_bank,
        )
        _write_json(args.out / "evaluation.json", report)
        return report
    fit = [
        r for r in records if r.split == "train" and r.programs and len(r.examples) >= 4
    ]
    if args.max_train_records and len(fit) > args.max_train_records:
        fit = sorted(
            random.Random(args.seed).sample(fit, args.max_train_records),
            key=lambda r: r.key,
        )
    metadata["train_records_used"] = len(fit)
    metadata["train_keys_digest"] = hashlib.sha256(
        "\n".join(r.key for r in fit).encode()
    ).hexdigest()
    if not fit:
        raise ValueError("No fitting behaviours found")
    run = None
    try:
        import wandb

        run = wandb.init(
            project="wandering-light-program-embeddings",
            entity="abhishekraok-na",
            mode="offline" if args.offline else "online",
            config=metadata,
            dir=str(args.out),
        )
        if not args.offline:
            metadata["wandb_url"] = run.url
    except Exception as error:
        metadata["wandb_error"] = str(error)
    if metadata["wandb_url"]:
        (args.out / "wandb_run.url").write_text(metadata["wandb_url"] + "\n")
    _write_json(args.out / "provenance.json", metadata)
    initial = evaluate(
        model,
        records,
        seed=args.seed,
        io_examples=args.io_examples,
        execution=False,
        max_eval_records=args.max_eval_records,
        global_bank=args.global_bank,
    )
    initial["initialization"] = "checkpoint" if args.checkpoint else "random"
    _write_json(
        args.out / ("initial.json" if args.checkpoint else "untrained.json"), initial
    )
    if args.freeze_program_space:
        for module in (model.byte_embedding, model.projection, model.program_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    rewrite_fit = [r for r in fit if len(set(r.programs)) >= 2]
    metadata["rewrite_training_records"] = len(rewrite_fit)
    _write_json(args.out / "provenance.json", metadata)
    labels_by_key = {r.key: i for i, r in enumerate(fit)}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    best_score = -math.inf
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        rewrite_count = (
            round(args.batch_size * args.rewrite_fraction) if rewrite_fit else 0
        )
        batch = [rng.choice(rewrite_fit) for _ in range(rewrite_count)] + [
            rng.choice(fit) for _ in range(args.batch_size - rewrite_count)
        ]
        rng.shuffle(batch)
        first = [rng.choice(r.programs) for r in batch]
        second = [
            rng.choice([p for p in r.programs if p != a] or r.programs)
            for r, a in zip(batch, first, strict=True)
        ]
        io_a = [_sample_examples(r.examples, args.io_examples, rng) for r in batch]
        io_b = [_sample_examples(r.examples, args.io_examples, rng) for r in batch]
        labels = torch.tensor([labels_by_key[r.key] for r in batch], device=args.device)
        program_a = model.encode_programs(first)
        program_b = model.encode_programs(second)
        inputs_a = model.encode_io(io_a)
        inputs_b = model.encode_io(io_b)
        loss_io = (
            symmetric_multi_positive_contrastive_loss(
                inputs_a, program_a, labels=labels, temperature=args.temperature
            )
            + symmetric_multi_positive_contrastive_loss(
                inputs_b, program_b, labels=labels, temperature=args.temperature
            )
        ) / 2
        loss_program = symmetric_multi_positive_contrastive_loss(
            program_a, program_b, labels=labels, temperature=args.temperature
        )
        loss = args.io_weight * loss_io + args.invariance_weight * loss_program
        if args.io_io_weight:
            loss = loss + args.io_io_weight * symmetric_multi_positive_contrastive_loss(
                inputs_a, inputs_b, labels=labels, temperature=args.temperature
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate(
                model,
                records,
                seed=args.seed,
                io_examples=args.io_examples,
                execution=False,
                max_eval_records=args.max_eval_records,
                global_bank=args.global_bank,
            )
            score = _validation_score(validation)
            if args.io_weight == 0:
                available = [
                    p["cross_style_program"]["mrr"]
                    for p in validation["partitions"].values()
                    if p["cross_style_program"]["mrr"] is not None
                ]
                score = sum(available) / len(available) if available else -math.inf
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "io_loss": float(loss_io.detach()),
                "program_loss": float(loss_program.detach()),
                "validation_mrr": score if math.isfinite(score) else None,
                "selection_metric": "io_mrr" if args.io_weight else "cross_style_mrr",
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if run is not None:
                run.log(entry, step=step)
            if score > best_score or best_score == -math.inf:
                best_score = score
                model.save(
                    args.out / "best.pt",
                    {
                        **metadata,
                        "step": step,
                        "validation_mrr": entry["validation_mrr"],
                    },
                )
    model.save(args.out / "last.pt", {**metadata, "step": args.steps})
    _write_json(args.out / "history.json", history)
    model = ProgramEmbedding.load(args.out / "best.pt", device=args.device)
    final = evaluate(
        model,
        records,
        seed=args.seed,
        io_examples=args.io_examples,
        execution=not args.skip_execution,
        execution_examples=args.execution_examples,
        max_eval_records=args.max_eval_records,
        global_bank=args.global_bank,
    )
    _write_json(args.out / "validation.json", final)
    if run is not None:
        run.finish()
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-program-sha256", action="append", default=[])
    parser.add_argument("--extra-tasks", type=Path, action="append", default=[])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--rewrite-fraction", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--global-bank", action="store_true")
    parser.add_argument("--untrained-eval", action="store_true")
    parser.add_argument("--freeze-program-space", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--byte-dim", type=int, default=48)
    parser.add_argument(
        "--max-eval-records",
        type=int,
        default=128,
        help="Queries per partition; 0 means all. Bank always complete.",
    )
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-program-bytes", type=int, default=16384)
    parser.add_argument("--max-example-bytes", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--io-weight", type=float, default=1.0)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--io-io-weight", type=float, default=0.0)
    parser.add_argument("--io-examples", type=int, default=5)
    parser.add_argument("--execution-examples", type=int, default=16)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-split", choices=("validation", "test"), default="validation"
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.steps,
            args.batch_size,
            args.eval_every,
            args.io_examples,
            args.execution_examples,
        )
        <= 0
    ):
        parser.error(
            "steps, batch size, evaluation interval and example counts must be positive"
        )
    if not 0 <= args.rewrite_fraction <= 1:
        parser.error("rewrite-fraction must be between zero and one")
    if args.untrained_eval and args.checkpoint:
        parser.error("--untrained-eval initializes a random model; omit --checkpoint")
    if args.freeze_program_space and not (args.io_weight or args.io_io_weight):
        parser.error("Freezing the program space requires an I/O training objective")
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only requires --checkpoint")
    if args.eval_split == "test" and not (args.eval_only or args.untrained_eval):
        parser.error(
            "Final test evaluation requires --eval-only with a fixed checkpoint"
        )
    result = train(args)
    print(
        json.dumps(
            {"split": result["split"], "partitions": result["partitions"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
