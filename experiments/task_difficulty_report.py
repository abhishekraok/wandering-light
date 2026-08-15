"""Audit nominal task length against executable shortest-path evidence.

Example:
    PYTHONHASHSEED=0 uv run python experiments/task_difficulty_report.py
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wandering_light.common_functions import basic_fns
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList
from wandering_light.proposer_pilot import TrajectoryGraph
from wandering_light.trajectory import Trajectory, TrajectorySpec, TrajectorySpecList

DEFAULT_EVAL = Path("wandering_light/evals/data/random_inputs_500.py")
DEFAULT_OUTPUT = Path("reports/overnight-20260809/task_difficulty")


def execute(spec: TrajectorySpec) -> Trajectory | None:
    result = Executor(basic_fns).execute_trajectory(spec)
    return result.trajectory if result.success else None


def shortest_subsequence(proposal: Trajectory) -> tuple[int, list[str]]:
    """Find the shortest ordered subsequence with the same output."""
    functions = list(proposal.function_defs)
    for length in range(len(functions) + 1):
        for indices in combinations(range(len(functions)), length):
            selected = [functions[index] for index in indices]
            spec = TrajectorySpec(proposal.input, FunctionDefList(selected))
            result = execute(spec)
            if result is not None and result.output == proposal.output:
                return length, [fn.name for fn in selected]
    raise AssertionError("the full proposal must be one of its own subsequences")


def pathologies(proposal: Trajectory) -> tuple[int, bool]:
    current = proposal.input
    states = [current]
    no_op_steps = 0
    revisits_state = False
    executor = Executor(basic_fns)
    for fn in proposal.function_defs:
        current = executor.execute(fn, current)
        no_op_steps += int(current == states[-1])
        revisits_state |= current in states
        states.append(current)
    return no_op_steps, revisits_state


def certify_with_full_function_set(
    proposal: Trajectory,
    upper_bound: int,
    *,
    max_states: int,
    max_transitions: int,
) -> dict[str, object]:
    graph = TrajectoryGraph(functions=basic_fns)
    root = graph.add_root(proposal.input)
    expansion = graph.expand(
        root,
        max_depth=upper_bound,
        max_states=max_states,
        max_transitions=max_transitions,
    )
    evidence = {
        "exact_status": "not_reached",
        "verified_shortest_num_steps": None,
        "search_complete": expansion.complete,
        "certified_depth": expansion.certified_depth,
        "search_stop_reason": expansion.stop_reason,
        "search_states": expansion.num_reached_states,
        "search_transitions": expansion.attempted_transitions,
    }
    try:
        task = graph.task_from_proposal(proposal, expansion)
    except ValueError:
        return evidence
    if task.shortest_path_is_certified:
        evidence["exact_status"] = "certified"
        evidence["verified_shortest_num_steps"] = task.verified_shortest_num_steps
    else:
        evidence["exact_status"] = "budget_truncated"
    return evidence


def analyze(
    eval_file: Path,
    *,
    exact_samples_per_length: int,
    max_states: int,
    max_transitions: int,
) -> list[dict[str, object]]:
    specs = TrajectorySpecList.from_py_file(str(eval_file), trusted_legacy_python=True)
    selected_per_length: Counter[int] = Counter()
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        nominal = len(spec.function_defs)
        proposal = execute(spec)
        if proposal is None:
            rows.append(
                {
                    "index": index,
                    "valid": False,
                    "input_type": spec.input.item_type.__name__,
                    "proposed_num_steps": nominal,
                    "proposed_functions": spec.function_defs.to_string(),
                    "shortest_subsequence_num_steps": None,
                    "shortest_subsequence_functions": None,
                    "no_op_steps": None,
                    "revisits_state": None,
                    "exact_status": "invalid_proposal",
                }
            )
            continue

        subsequence_length, subsequence_functions = shortest_subsequence(proposal)
        no_op_steps, revisits_state = pathologies(proposal)
        row: dict[str, object] = {
            "index": index,
            "valid": True,
            "input_type": spec.input.item_type.__name__,
            "proposed_num_steps": nominal,
            "proposed_functions": spec.function_defs.to_string(),
            "shortest_subsequence_num_steps": subsequence_length,
            "shortest_subsequence_functions": ", ".join(subsequence_functions),
            "no_op_steps": no_op_steps,
            "revisits_state": revisits_state,
            "exact_status": "not_selected",
            "verified_shortest_num_steps": None,
            "search_complete": None,
            "certified_depth": None,
            "search_stop_reason": None,
            "search_states": None,
            "search_transitions": None,
        }
        if selected_per_length[nominal] < exact_samples_per_length:
            selected_per_length[nominal] += 1
            row.update(
                certify_with_full_function_set(
                    proposal,
                    subsequence_length,
                    max_states=max_states,
                    max_transitions=max_transitions,
                )
            )
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if row["valid"]]
    exact = [row for row in valid if row["exact_status"] != "not_selected"]
    exact_counts = Counter(int(row["proposed_num_steps"]) for row in exact)
    by_length: dict[int, dict[str, object]] = {}
    for length in sorted({int(row["proposed_num_steps"]) for row in valid}):
        group = [row for row in valid if row["proposed_num_steps"] == length]
        by_length[length] = {
            "count": len(group),
            "reducible_by_subsequence": sum(
                row["shortest_subsequence_num_steps"] < length for row in group
            ),
            "has_no_op": sum(row["no_op_steps"] > 0 for row in group),
            "revisits_state": sum(bool(row["revisits_state"]) for row in group),
        }
    return {
        "method": {
            "subsequence": "exhaustive over every ordered subsequence; proves redundancy but not global shortestness",
            "full_function_bfs": "breadth-first over all 118 basic functions; exact only when certified",
        },
        "num_specs": len(rows),
        "num_valid": len(valid),
        "num_invalid": len(rows) - len(valid),
        "num_reducible_by_subsequence": sum(
            row["shortest_subsequence_num_steps"] < row["proposed_num_steps"]
            for row in valid
        ),
        "num_with_no_op": sum(row["no_op_steps"] > 0 for row in valid),
        "num_revisiting_state": sum(bool(row["revisits_state"]) for row in valid),
        "exact_sample": {
            "selection": "first executable tasks at each proposed length",
            "selected_by_proposed_length": dict(sorted(exact_counts.items())),
            "selected": len(exact),
            "certified": sum(row["exact_status"] == "certified" for row in exact),
            "budget_truncated": sum(
                row["exact_status"] == "budget_truncated" for row in exact
            ),
            "not_reached": sum(row["exact_status"] == "not_reached" for row in exact),
            "full_function_shorter_than_subsequence": sum(
                row["exact_status"] == "certified"
                and row["verified_shortest_num_steps"]
                < row["shortest_subsequence_num_steps"]
                for row in exact
            ),
        },
        "by_proposed_length": by_length,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    valid = [row for row in rows if row["valid"]]
    max_length = max(int(row["proposed_num_steps"]) for row in valid)
    matrix = np.zeros((max_length + 1, max_length + 1), dtype=int)
    for row in valid:
        matrix[
            int(row["shortest_subsequence_num_steps"]),
            int(row["proposed_num_steps"]),
        ] += 1

    lengths = range(1, max_length + 1)
    grouped = defaultdict(list)
    for row in valid:
        grouped[int(row["proposed_num_steps"])].append(row)
    reducible = [
        np.mean(
            [row["shortest_subsequence_num_steps"] < length for row in grouped[length]]
        )
        for length in lengths
    ]
    no_op = [
        np.mean([row["no_op_steps"] > 0 for row in grouped[length]])
        for length in lengths
    ]
    revisits = [
        np.mean([row["revisits_state"] for row in grouped[length]])
        for length in lengths
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    image = axes[0].imshow(matrix, origin="lower", cmap="Blues")
    for shortest in range(max_length + 1):
        for proposed in lengths:
            count = matrix[shortest, proposed]
            if count:
                axes[0].text(proposed, shortest, str(count), ha="center", va="center")
    axes[0].set(
        xlabel="Proposed length",
        ylabel="Shortest ordered-subsequence length",
        title="Committed benchmark: 496 executable tasks",
        xticks=list(lengths),
        yticks=range(max_length + 1),
    )
    figure.colorbar(image, ax=axes[0], label="Tasks")

    axes[1].plot(lengths, reducible, "o-", label="Reducible")
    axes[1].plot(lengths, no_op, "o-", label="Contains no-op")
    axes[1].plot(lengths, revisits, "o-", label="Revisits state")
    axes[1].set(
        xlabel="Proposed length",
        ylabel="Fraction of tasks",
        title="Path defects increase with nominal length",
        xticks=list(lengths),
        ylim=(-0.03, 1.03),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.suptitle(
        "Random-walk length is not task difficulty\n"
        "Subsequence result is exhaustive but conservative; full-function BFS is separate",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def plot_exact_sample(rows: list[dict[str, object]], path: Path) -> None:
    exact = [row for row in rows if row["exact_status"] == "certified"]
    lengths = sorted({int(row["proposed_num_steps"]) for row in exact})
    count_per_length = Counter(int(row["proposed_num_steps"]) for row in exact)
    counts = set(count_per_length.values())
    sample_label = (
        f"n={counts.pop()} per length"
        if len(counts) == 1
        else f"n={len(exact)} stratified"
    )
    means = {
        "Proposed": [float(length) for length in lengths],
        "Shortest subsequence": [
            np.mean(
                [
                    row["shortest_subsequence_num_steps"]
                    for row in exact
                    if row["proposed_num_steps"] == length
                ]
            )
            for length in lengths
        ],
        "Verified shortest (118 fns)": [
            np.mean(
                [
                    row["verified_shortest_num_steps"]
                    for row in exact
                    if row["proposed_num_steps"] == length
                ]
            )
            for length in lengths
        ],
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for label, values in means.items():
        axes[0].plot(lengths, values, "o-", label=label)
    axes[0].set(
        xlabel="Proposed length",
        ylabel="Mean path length",
        title=f"Mean path lengths ({sample_label})",
        xticks=lengths,
        yticks=range(max(lengths) + 1),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    points = axes[1].scatter(
        [row["shortest_subsequence_num_steps"] for row in exact],
        [row["search_states"] for row in exact],
        c=[row["proposed_num_steps"] for row in exact],
        cmap="viridis",
        s=45,
    )
    axes[1].set(
        xlabel="Subsequence upper bound used for BFS",
        ylabel="Reachable states enumerated (log scale)",
        title="All 25 searches completed within budget",
        xticks=range(max(lengths) + 1),
        yscale="log",
    )
    axes[1].grid(alpha=0.25)
    figure.colorbar(points, ax=axes[1], label="Proposed length", ticks=lengths)
    figure.suptitle(
        "Full-function BFS finds additional shortcuts\n"
        "Exact only when every shallower state is exhaustively expanded",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exact-samples-per-length", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--max-transitions", type=int, default=200_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = analyze(
        args.eval_file,
        exact_samples_per_length=args.exact_samples_per_length,
        max_states=args.max_states,
        max_transitions=args.max_transitions,
    )
    summary = summarize(rows)
    write_csv(rows, args.output_dir / "tasks.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, args.output_dir / "task_difficulty.png")
    plot_exact_sample(rows, args.output_dir / "exact_pilot.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
