"""Measure what a basis execution costs and what policy inference costs.

Produces the numbers in ``reports/inference-cost-20260830/``. Two questions:

1. What is actually inside one ``basis_function_executions`` unit?
2. How does policy scoring latency scale with model size, on CPU and on GPU?

The second answers whether model capacity is expensive in the solver's inner
loop. Timing depends on architecture and batch shape, not on weight values, so
freshly initialised models of the published shape give the same latency as a
trained checkpoint.

Usage::

    uv run python experiments/inference_cost.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wandering_light.rollout_policy as rp
from wandering_light.common_functions import basic_fns
from wandering_light.executor import Executor
from wandering_light.solver import BFSPredictor
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList

# (name, byte embedding, conv channels, hidden) -- the published policy first.
SIZES = (
    ("published", rp.PolicyArchitecture(48, 96, 176)),
    ("1M", rp.PolicyArchitecture(128, 320, 640)),
    ("10M", rp.PolicyArchitecture(384, 1024, 2048)),
    ("40M", rp.PolicyArchitecture(768, 2048, 4096)),
    ("125M", rp.PolicyArchitecture(1280, 3584, 7168)),
)
BEAM = 64
REPEATS = 15


def _median(fn, repeats: int = REPEATS) -> float:
    fn()  # warm up
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def execution_breakdown() -> dict[str, Any]:
    """Split one charged execution into op, identity, and search bookkeeping."""
    functions = basic_fns
    executor = Executor(functions)
    state = TypedList([3, 1, 4, 1])
    compatible = [
        f for f in functions.functions if state.item_type is f.input_type_cls()
    ]

    def run_ops() -> None:
        for function in compatible:
            # A raising op still costs one execution, so it is timed too.
            with contextlib.suppress(Exception):
                executor.execute(function, state)

    results = []
    for function in compatible:
        with contextlib.suppress(Exception):
            results.append(executor.execute(function, state))

    per_op = _median(run_ops, 200) / len(compatible)
    per_key = _median(lambda: [r.search_key() for r in results], 200) / len(results)

    # System rate: what BFS actually achieves, bookkeeping included. Seeded,
    # and retried because a random walk can pick a program that raises.
    rng = random.Random(20260830)
    target = None
    for _ in range(50):
        spec = TrajectorySpec.create_random_walk(state, 6, functions.functions, rng=rng)
        executed = executor.execute_trajectory(spec)
        if executed.success and executed.trajectory is not None:
            target = executed.trajectory.output
            break
    if target is None:
        raise RuntimeError("could not sample an executable random walk")
    predictor = BFSPredictor(budget=200_000, max_depth=4)
    started = time.perf_counter()
    search = predictor.search_with_cost(state, target, functions)
    elapsed = time.perf_counter() - started
    system = elapsed / search.executions if search.executions else float("nan")

    return {
        "compatible_ops": len(compatible),
        "execute_us": per_op * 1e6,
        "search_key_us": per_key * 1e6,
        "system_us": system * 1e6,
        "system_per_second": 1 / system,
        "bookkeeping_us": system * 1e6 - (per_op + per_key) * 1e6,
        "op_share": per_op / system,
    }


def inference_latency() -> list[dict[str, Any]]:
    """End-to-end score_actions_batch latency by model size and device."""
    functions = basic_fns
    actions = len(rp.ordered_actions(functions))
    states = [TypedList([i, i + 1, i + 2, i + 3]) for i in range(1, 9)]
    target = TypedList([3, 6, 9, 12])
    problems = [(states[i % len(states)], target) for i in range(BEAM)]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for name, architecture in SIZES:
        row: dict[str, Any] = {
            "size": name,
            "architecture": architecture.to_config(),
        }
        for device in devices:
            model = rp.AlignedPairPolicy(
                action_count=actions, architecture=architecture
            ).eval()
            row["parameters"] = sum(p.numel() for p in model.parameters())
            policy = rp.RolloutPolicy.from_basis(model, functions, device=device)

            def call(policy=policy, device=device) -> None:
                policy.score_actions_batch(problems, functions)
                if device == "cuda":
                    torch.cuda.synchronize()

            row[f"{device}_ms"] = _median(call) * 1000
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="write results here")
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    breakdown = execution_breakdown()
    print("=== one charged execution ===")
    print(f"  execute(f, state)   {breakdown['execute_us']:8.2f} us")
    print(f"  search_key()        {breakdown['search_key_us']:8.2f} us")
    print(f"  bookkeeping         {breakdown['bookkeeping_us']:8.2f} us")
    print(
        f"  total (BFS system)  {breakdown['system_us']:8.2f} us"
        f"  -> {breakdown['system_per_second']:,.0f}/s"
        f"   op is {breakdown['op_share']:.0%}"
    )

    rows = inference_latency()
    print(f"\n=== score_actions_batch, beam {BEAM} ===")
    for row in rows:
        cuda = row.get("cuda_ms")
        best = "CPU" if cuda is None or row["cpu_ms"] <= cuda else "CUDA"
        print(
            f"  {row['size']:>10} {row['parameters']:>12,}"
            f"  cpu {row['cpu_ms']:8.2f} ms"
            f"  cuda {cuda if cuda is None else f'{cuda:8.2f}'} ms  -> {best}"
        )

    if arguments.json:
        payload = {"execution": breakdown, "inference": rows, "beam": BEAM}
        with open(arguments.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {arguments.json}")


if __name__ == "__main__":
    main()
