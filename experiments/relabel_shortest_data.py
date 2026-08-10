"""Create shortest-path-labelled training and evaluation data.

The random walk supplies an upper bound. Exhaustive BFS proves paths through
depth three; unresolved length-five tasks use the length-penalized RL solver.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from wandering_light.common_functions import basic_fns
from wandering_light.evals.create_data import make_random_data
from wandering_light.function_def import FunctionDefList
from wandering_light.shortest_path_data import (
    apply_solver_candidate,
    bounded_relabel,
    read_jsonl_gz,
    write_jsonl_gz,
)
from wandering_light.solver import TrainedLLMTokenGenerator, create_token_solver
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList

TRAIN_COUNTS = {1: 100, 2: 10_000, 3: 100_000, 4: 10_000, 5: 1_000}
EVAL_SOURCE = Path("wandering_light/evals/data/random_inputs_500.py")
RL_MODEL = Path("checkpoints/saved/rl/induction_opt_125m_sft_434k_rl_6k_with_lp")
REPORT_DIR = Path("reports/shortest-data-20260810")
TRAIN_DATA = Path("wandering_light/training/data/induction_shortest_v1.jsonl.gz")
EVAL_DATA = Path("wandering_light/evals/data/random_inputs_500_shortest_v1.jsonl.gz")

_WORKER_CONTEXT = SimpleNamespace(specs=[], split="")


def training_specs() -> TrajectorySpecList:
    specs = make_random_data(basic_fns, TRAIN_COUNTS, seed=42)
    specs.shuffle(42)
    return specs


def current_function_specs(specs: TrajectorySpecList) -> TrajectorySpecList:
    """Replace embedded legacy definitions with the current deterministic palette."""
    return TrajectorySpecList(
        [
            TrajectorySpec(
                spec.input,
                FunctionDefList(
                    [
                        basic_fns.name_to_function[function.name]
                        for function in spec.function_defs
                    ]
                ),
            )
            for spec in specs
        ]
    )


def _bounded_worker(index: int) -> dict:
    return bounded_relabel(
        _WORKER_CONTEXT.specs[index], index, split=_WORKER_CONTEXT.split
    )


def bounded_records(specs: TrajectorySpecList, split: str, workers: int) -> list[dict]:
    _WORKER_CONTEXT.specs = list(specs)
    _WORKER_CONTEXT.split = split
    if workers == 1:
        return [
            _bounded_worker(index)
            for index in tqdm(range(len(specs)), desc=f"BFS {split}")
        ]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(
            tqdm(
                pool.map(_bounded_worker, range(len(specs)), chunksize=32),
                total=len(specs),
                desc=f"BFS {split}",
            )
        )


def apply_rl(records: list[dict], model_path: Path, budget: int) -> list[dict]:
    unresolved_indices = [
        index
        for index, record in enumerate(records)
        if record["output"] is not None and not record["certified"]
    ]
    if not unresolved_indices:
        return records

    problems = [
        (
            TypedList.from_str(records[index]["input"]),
            TypedList.from_str(records[index]["output"]),
        )
        for index in unresolved_indices
    ]
    generator = TrainedLLMTokenGenerator(str(model_path), temperature=0.8)
    solver = create_token_solver(generator, budget=budget)
    results = solver.solve_batch(problems, basic_fns)
    for index, result in zip(unresolved_indices, results, strict=True):
        candidate = result.trajectory.function_defs if result.success else None
        records[index] = apply_solver_candidate(
            records[index], candidate, budget=budget
        )
    return records


def released(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record["certified"]
        and record["output"] is not None
        and record["relabeled_length"] > 0
    ]


def summarize_split(records: list[dict]) -> dict:
    valid = [record for record in records if record["output"] is not None]
    released_records = released(records)
    by_length = {}
    for length in sorted({record["original_length"] for record in valid}):
        group = [record for record in valid if record["original_length"] == length]
        by_length[length] = {
            "valid": len(group),
            "certified": sum(record["certified"] for record in group),
            "released": sum(
                record["certified"] and record["relabeled_length"] > 0
                for record in group
            ),
            "shortened": sum(
                record["relabeled_length"] < record["original_length"]
                for record in group
            ),
            "rl_attempted": sum(record["rl_attempted"] for record in group),
        }
    return {
        "input_rows": len(records),
        "valid": len(valid),
        "invalid": len(records) - len(valid),
        "certified": sum(record["certified"] for record in valid),
        "identity_excluded": sum(
            record["certified"] and record["relabeled_length"] == 0 for record in valid
        ),
        "released": len(released_records),
        "unresolved": sum(not record["certified"] for record in valid),
        "shortened": sum(
            record["relabeled_length"] < record["original_length"] for record in valid
        ),
        "rl_attempted": sum(record["rl_attempted"] for record in valid),
        "rl_certified": sum(record["method"] == "rl_after_bfs" for record in valid),
        "search_states": sum(record["search_states"] for record in valid),
        "search_transitions": sum(record["search_transitions"] for record in valid),
        "by_original_length": by_length,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for record in records for key in record)
    )
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as file,
    ):
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot(train: list[dict], evaluation: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, (label, records) in zip(
        axes[:2],
        (("Train", released(train)), ("Eval", released(evaluation))),
        strict=True,
    ):
        matrix = np.zeros((6, 6), dtype=int)
        for record in records:
            matrix[record["relabeled_length"], record["original_length"]] += 1
        image = axis.imshow(matrix, origin="lower", cmap="Blues")
        text_threshold = matrix.max() * 0.45
        for shortest in range(1, 6):
            for original in range(1, 6):
                count = matrix[shortest, original]
                if count:
                    axis.text(
                        original,
                        shortest,
                        f"{count:,}",
                        ha="center",
                        va="center",
                        color="white" if count > text_threshold else "black",
                    )
        axis.set(
            title=f"{label}: certified released rows",
            xlabel="Random-walk length",
            ylabel="Certified shortest length",
            xticks=range(1, 6),
            yticks=range(1, 6),
        )
        figure.colorbar(image, ax=axis, label="Rows")

    lengths = range(1, 6)
    for label, records in (("Train", train), ("Eval", evaluation)):
        rates = []
        for length in lengths:
            group = [
                record
                for record in records
                if record["output"] is not None and record["original_length"] == length
            ]
            rates.append(
                sum(
                    record["relabeled_length"] < record["original_length"]
                    for record in group
                )
                / len(group)
                if group
                else 0
            )
        axes[2].plot(lengths, rates, "o-", label=label)
    axes[2].set(
        title="Random-walk labels shortened",
        xlabel="Random-walk length",
        ylabel="Fraction",
        xticks=list(lengths),
        ylim=(-0.03, 1.03),
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    figure.suptitle(
        "Shortest-path relabeling: exhaustive through depth 3, then RL best-of-K"
    )
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rl-budget", type=int, default=16)
    parser.add_argument("--rl-model", type=Path, default=RL_MODEL)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--resume-bounded", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_specs = training_specs()
    eval_specs = current_function_specs(TrajectorySpecList.from_py_file(str(EVAL_SOURCE)))
    bounded_dir = args.report_dir / "bounded"
    train_bounded = bounded_dir / "train.jsonl.gz"
    eval_bounded = bounded_dir / "eval.jsonl.gz"

    if args.resume_bounded and train_bounded.exists() and eval_bounded.exists():
        train_records = read_jsonl_gz(train_bounded)
        eval_records = read_jsonl_gz(eval_bounded)
    else:
        train_records = bounded_records(train_specs, "train", args.workers)
        eval_records = bounded_records(eval_specs, "eval", args.workers)
        write_jsonl_gz(train_records, train_bounded)
        write_jsonl_gz(eval_records, eval_bounded)

    train_records = apply_rl(train_records, args.rl_model, args.rl_budget)
    eval_records = apply_rl(eval_records, args.rl_model, args.rl_budget)

    write_jsonl_gz(released(train_records), TRAIN_DATA)
    write_jsonl_gz(released(eval_records), EVAL_DATA)
    raw_dir = args.report_dir / "raw"
    write_jsonl_gz(train_records, raw_dir / "train_audit.jsonl.gz")
    write_jsonl_gz(eval_records, raw_dir / "eval_audit.jsonl.gz")
    write_csv_gz(train_records, raw_dir / "train_audit.csv.gz")
    write_csv_gz(eval_records, raw_dir / "eval_audit.csv.gz")

    summary = {
        "claim_scope": (
            "exact only where certified=true; RL failures retain explicit lower/upper bounds"
        ),
        "config": {
            "train_counts": TRAIN_COUNTS,
            "train_seed": 42,
            "train_shuffle_seed": 42,
            "eval_source": str(EVAL_SOURCE),
            "function_semantics": "stable_hash_and_set_order_v1",
            "max_exact_depth": 3,
            "rl_model": str(args.rl_model),
            "rl_budget": args.rl_budget,
            "rl_temperature": 0.8,
        },
        "train": summarize_split(train_records),
        "eval": summarize_split(eval_records),
        "artifacts": {
            "train_data": str(TRAIN_DATA),
            "train_sha256": sha256(TRAIN_DATA),
            "eval_data": str(EVAL_DATA),
            "eval_sha256": sha256(EVAL_DATA),
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(train_records, eval_records, args.report_dir / "shortest_relabel.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
