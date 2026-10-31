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
from dataclasses import asdict, replace
from pathlib import Path

import torch

from wandering_light.program_embedding import (
    EmbeddingConfig,
    ProgramEmbedding,
    symmetric_multi_positive_contrastive_loss,
)
from wandering_light.program_embedding_artifacts import (
    git_metadata as _git_metadata,
)
from wandering_light.program_embedding_artifacts import (
    model_objectives,
    save_evaluation,
    save_training_history,
)
from wandering_light.program_embedding_artifacts import (
    write_json as _write_json,
)
from wandering_light.program_embedding_data import load_records, manifest_for
from wandering_light.program_embedding_eval import (
    _matches,
    _rank,
    _sample_examples,
    evaluate,
    lexical_vectors,
    retrieval_metrics,
)
from wandering_light.program_embedding_execution import ExecutionMatcher

# Preserve existing imports while evaluation now has its own reusable module.
__all__ = [
    "_matches",
    "_rank",
    "evaluate",
    "exclude_programs",
    "lexical_vectors",
    "main",
    "retrieval_metrics",
    "train",
]


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


def _validation_score(report) -> float:
    metrics = [
        p["execution"]["shown_mrr@10_bounds"][0]
        for p in report["partitions"].values()
        if p["execution"].get("shown_mrr@10_bounds") is not None
    ]
    return sum(metrics) / len(metrics) if metrics else -math.inf


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
        metadata.update(model_objectives(model, checkpoint=args.checkpoint is not None))
        metadata["checkpoint_metadata"] = model.metadata
        metadata["config"] = asdict(model.config)
        metadata["initialization"] = "random" if args.untrained_eval else "checkpoint"
        _write_json(args.out / "provenance.json", metadata)
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
        save_evaluation(
            args.out / "evaluation.json",
            report,
            metadata,
            model,
            checkpoint=args.checkpoint,
        )
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
    metadata["io_supervision"] = (
        {
            "kind": "execution-on-shown-io-v1",
            "positive": "candidate passes every shown example in the current query",
            "negative": "candidate fails at least one shown example",
            "unknown": "excluded from targets and softmax denominators",
            "candidate_scope": "in-batch",
            "seconds_per_program_example": args.execution_label_timeout,
            "cache_size": args.execution_label_cache_size,
            "selection": "shown execution MRR@10 lower bound; no hidden outputs",
        }
        if args.io_weight
        else {"kind": "disabled"}
    )
    metadata["checkpoint_selection"] = (
        "execution_io_mrr@10_lower" if args.io_weight else "cross_style_mrr"
    )
    metadata["io_io_supervision"] = (
        "positive-only cosine consistency within known records"
        if args.io_io_weight
        else "disabled"
    )
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
    save_evaluation(
        args.out / ("initial.json" if args.checkpoint else "untrained.json"),
        initial,
        metadata,
        model,
        checkpoint=args.checkpoint,
    )
    if args.freeze_program_space:
        for module in (model.byte_embedding, model.projection, model.program_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    rewrite_fit = [r for r in fit if len(set(r.programs)) >= 2]
    metadata["rewrite_training_records"] = len(rewrite_fit)
    _write_json(args.out / "provenance.json", metadata)
    labels_by_key = {r.key: i for i, r in enumerate(fit)}
    matcher = ExecutionMatcher(
        args.execution_label_timeout, args.execution_label_cache_size
    )
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
        loss_io = inputs_a.sum() * 0
        if args.io_weight:
            positives_a, known_a = matcher.masks(io_a, first, device=args.device)
            positives_b, known_b = matcher.masks(io_b, second, device=args.device)
            loss_io = (
                symmetric_multi_positive_contrastive_loss(
                    inputs_a,
                    program_a,
                    positive_mask=positives_a,
                    valid_mask=known_a,
                    temperature=args.temperature,
                )
                + symmetric_multi_positive_contrastive_loss(
                    inputs_b,
                    program_b,
                    positive_mask=positives_b,
                    valid_mask=known_b,
                    temperature=args.temperature,
                )
            ) / 2
        loss_program = symmetric_multi_positive_contrastive_loss(
            program_a, program_b, labels=labels, temperature=args.temperature
        )
        loss = args.io_weight * loss_io + args.invariance_weight * loss_program
        if args.io_io_weight:
            same_record = labels[:, None] == labels[None, :]
            loss = (
                loss
                + args.io_io_weight * (1 - (inputs_a @ inputs_b.T)[same_record]).mean()
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
                execution=bool(args.io_weight),
                execution_examples=0,
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
                "selection_metric": "execution_io_mrr@10_lower"
                if args.io_weight
                else "cross_style_mrr",
                "execution_labels": matcher.statistics(),
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
    save_training_history(args.out / "history.json", history, metadata)
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
    save_evaluation(
        args.out / "validation.json",
        final,
        metadata,
        model,
        checkpoint=args.out / "best.pt",
    )
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
    parser.add_argument("--execution-label-timeout", type=float, default=1.0)
    parser.add_argument("--execution-label-cache-size", type=int, default=65536)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip final execution grading; training labels and I/O checkpoint selection still execute shown examples.",
    )
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
    if (
        not math.isfinite(args.execution_label_timeout)
        or args.execution_label_timeout <= 0
    ):
        parser.error("execution-label-timeout must be positive and finite")
    if args.execution_label_cache_size < 0:
        parser.error("execution-label-cache-size must be nonnegative")
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
