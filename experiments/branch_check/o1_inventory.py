"""O1-B: fixed train-only continuation catalogue; no model or solver changes.

Preregistered at issuecomment-5971674142, amended at issuecomment-5971719000.
Unequal catalogue upper bounds do not establish unequal optimal distances.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.staged_cycles import closed_product  # noqa: E402
from wandering_light.stack_beam import (  # noqa: E402
    Product,
    atoms_exceed,
    closed_cuts,
    key_of,
)
from wandering_light.stack_machine import parse_stack, step_raw  # noqa: E402
from wandering_light.stack_pbe import (  # noqa: E402
    BASIS,
    BY_NAME,
    OP_BY_NAME,
    assign_split,
)

SOURCE_SHA = "e358fca13e69880a48f0b3794d36769b1815f3dd249c765bc08b52fc7c1fca78"
PREREGISTRATION = (
    "https://github.com/abhishekraok/wandering-light/issues/164#issuecomment-5971674142"
)
AMENDMENT = (
    "https://github.com/abhishekraok/wandering-light/issues/164#issuecomment-5971719000"
)


class ProvenanceError(ValueError):
    """An archived claim disagrees with complete replay or registered metadata."""


class CensoredError(RuntimeError):
    def __init__(self, reason: str, scope: str = "task"):
        super().__init__(reason)
        self.scope = scope


@dataclass(frozen=True)
class Limits:
    task_calls: int = 50_000
    task_fuel: int = 2_000_000
    run_calls: int = 2_000_000
    run_fuel: int = 100_000_000
    call_fuel: int = 2_000
    atoms: int = 4_096
    seconds: float = 900


class Budget:
    """All requested paths replay in full; shared prefixes are charged again."""

    def __init__(self, limits: Limits | None = None):
        self.limits = limits or Limits()
        self.started = time.monotonic()
        self.calls = self.fuel = self.task_calls = self.task_fuel = 0

    def begin_task(self):
        self.task_calls = self.task_fuel = 0

    def check_time(self):
        if time.monotonic() - self.started >= self.limits.seconds:
            raise CensoredError("run_deadline", "run")

    def allocation(self) -> int:
        self.check_time()
        for scope, calls, fuel, call_cap, fuel_cap in (
            ("run", self.calls, self.fuel, self.limits.run_calls, self.limits.run_fuel),
            (
                "task",
                self.task_calls,
                self.task_fuel,
                self.limits.task_calls,
                self.limits.task_fuel,
            ),
        ):
            if calls >= call_cap:
                raise CensoredError(f"{scope}_calls", scope)
            if fuel >= fuel_cap:
                raise CensoredError(f"{scope}_fuel", scope)
        return min(
            self.limits.call_fuel,
            self.limits.task_fuel - self.task_fuel,
            self.limits.run_fuel - self.fuel,
        )

    def charge(self, fuel: int):
        self.calls += 1
        self.task_calls += 1
        self.fuel += fuel
        self.task_fuel += fuel

    def counters(self) -> dict:
        return {
            "calls": self.calls,
            "charged_fuel": self.fuel,
            "task_calls": self.task_calls,
            "task_charged_fuel": self.task_fuel,
        }


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def validate_row(row: dict):
    name = row["behaviour"]
    if (
        row["split"] != "train"
        or name not in BY_NAME
        or assign_split(BY_NAME[name].skeleton) != "train"
    ):
        raise ProvenanceError("row/live catalogue is not training-only")


def selected_rows(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows if row["example_set"] == 0]
    return sorted(
        eligible,
        key=lambda row: (digest(f"o1b-v1|{row['behaviour']}"), row["behaviour"]),
    )[:64]


def selected_records(row: dict) -> tuple[list[dict], dict]:
    unique = {}
    totals = Counter()
    for segment in row["segments"]:
        totals["raw_retained_cuts"] += len(segment["states"])
        if segment["stop"] is not None:
            totals["incomplete_segment_records"] += len(segment["states"])
            continue
        for record in segment["states"]:
            key = (segment["segment"], record["program"])
            if key in unique and unique[key] != record:
                raise ProvenanceError("inconsistent duplicate segment/program")
            unique[key] = record
    records = []
    for (segment, program), record in unique.items():
        text = f"o1b-v1|{row['behaviour']}|{row['example_set']}|{segment}|{program}"
        records.append(
            {**record, "segment": segment, "digest": digest(text), "key": text}
        )
    records.sort(key=lambda record: (record["digest"], record["key"]))
    totals["unique_completed_records"] = len(records)
    totals["selected_records"] = min(32, len(records))
    return records[:32], dict(totals)


def replay(start: Product, program: tuple[str, ...], budget: Budget) -> dict:
    """Return a full successful trace or a classified refusal/censoring trace."""
    states = [start]
    result: dict[str, Any] = {"status": "complete", "states": states}
    try:
        if atoms_exceed(start, budget.limits.atoms):
            raise CensoredError("atom_limit_at_source")
        for position, name in enumerate(program):
            moved = []
            for example, stack in enumerate(states[-1]):
                allocation = budget.allocation()
                outcome = step_raw(OP_BY_NAME[name], stack, allocation, 0)
                if isinstance(outcome, str):
                    charged = allocation if OP_BY_NAME[name].runs_code else 1
                    budget.charge(charged)
                    result.update(position=position, example=example, reason=outcome)
                    if outcome.endswith(("out of fuel", "quotation nested too deep")):
                        raise CensoredError(f"interpreter_resource: {outcome}")
                    result["status"] = "refused"
                    return result
                stack_out, fuel = outcome
                budget.charge(fuel)
                if fuel > allocation:
                    raise ProvenanceError("interpreter exceeded allocated fuel")
                moved.append(stack_out)
            states.append(tuple(moved))
            if atoms_exceed(states[-1], budget.limits.atoms):
                raise CensoredError("atom_limit")
        budget.check_time()
    except CensoredError as error:
        result.update(status="censored", reason=str(error), scope=error.scope)
    return result


def required_path(start: Product, program: tuple[str, ...], budget: Budget) -> list:
    result = replay(start, program, budget)
    if result["status"] == "censored":
        raise CensoredError(result["reason"], result["scope"])
    if result["status"] != "complete":
        raise ProvenanceError(f"required replay refused: {result['reason']}")
    return result["states"]


def graph_paths(
    states: list[Product], program: tuple[str, ...], budget: Budget
) -> dict:
    """Shortest graph continuations, with global token-tuple lexicographic ties."""
    goal = states[-1]
    edges: dict[Product, dict[str, Product]] = defaultdict(dict)
    reverse: dict[Product, set[Product]] = defaultdict(set)
    for source, op, target in zip(states[:-1], program, states[1:], strict=True):
        budget.check_time()
        if source == goal:
            continue
        if op in edges[source] and edges[source][op] != target:
            raise ProvenanceError("equal state/op has unequal successor")
        edges[source][op] = target
        reverse[target].add(source)
    distance = {goal: 0}
    pending = deque([goal])
    while pending:
        budget.check_time()
        target = pending.popleft()
        for source in reverse[target]:
            if source not in distance:
                distance[source] = distance[target] + 1
                pending.append(source)
    paths = {}
    for source in dict.fromkeys(states):
        if source == goal or not closed_product(source):
            continue
        if source not in distance:
            raise ProvenanceError("witness vertex cannot reach absorbing goal")
        current, path = source, []
        while current != goal:
            budget.check_time()
            op, target = min(
                (op, target)
                for op, target in edges[current].items()
                if distance.get(target) == distance[current] - 1
            )
            path.append(op)
            current = target
        paths[source] = tuple(path)
    return paths


def catalogue(states: list[Product], program: tuple[str, ...], budget: Budget) -> dict:
    paths = graph_paths(states, program, budget)
    for source, continuation in paths.items():
        if required_path(source, continuation, budget)[-1] != states[-1]:
            raise ProvenanceError("graph continuation does not reach goal")
    unique = set(paths.values())
    eligible = sorted(
        (path for path in unique if 0 < len(path) <= 96), key=lambda p: (len(p), p)
    )
    return {
        "programs": eligible[:32],
        "source_states": len(paths),
        "source_continuations": [
            {"product": key_of(source), "program": continuation}
            for source, continuation in paths.items()
        ],
        "unique_sequences": len(unique),
        "over_length": sum(len(path) > 96 for path in unique),
        "truncated": max(0, len(eligible) - 32),
    }


def ordered_pairs(candidates: list[dict]) -> tuple[list[dict], int]:
    """Count an unordered product pair once per task, using first admissible segment."""
    seen, pairs, repeats = set(), [], 0
    eligible = sorted(candidates, key=lambda c: (c["segment"], c["digest"]))
    for left, right in combinations(eligible, 2):
        if (
            left["segment"] != right["segment"]
            or left["prefix_ops"] != right["prefix_ops"]
            or (left["on_witness"] and right["on_witness"])
            or left["label"] is None
            or right["label"] is None
            or left["label"] == right["label"]
        ):
            continue
        products = tuple(
            sorted(
                json.dumps(c["product"], separators=(",", ":")) for c in (left, right)
            )
        )
        if products[0] == products[1]:
            raise ProvenanceError("pair contains equal products")
        if products in seen:
            repeats += 1
            continue
        seen.add(products)
        pairs.append(
            {
                "members": [left["digest"], right["digest"]],
                "labels": [left["label"], right["label"]],
                "segment": left["segment"],
                "prefix_ops": left["prefix_ops"],
                "both_off_witness": not left["on_witness"] and not right["on_witness"],
            }
        )
    return pairs, repeats


def analyze_task(row: dict, budget: Budget) -> dict:
    budget.begin_task()
    output: dict[str, Any] = {
        "behaviour": row["behaviour"],
        "example_set": row["example_set"],
        "status": "complete",
        "candidates": [],
        "pairs": [],
        "roots": [],
    }
    phase = "selection"
    try:
        validate_row(row)
        records, counts = selected_records(row)
        output.update(selected_records=records, counts=counts)
        start = tuple(parse_stack(text, basis=BASIS) for text in row["inputs"])
        goal = tuple(parse_stack(text, basis=BASIS) for text in row["outputs"])
        if not start or len(start) != len(goal):
            raise ProvenanceError("empty or mismatched example axes")
        output.update(start=key_of(start), goal=key_of(goal))
        phase = "witness"
        program = tuple(row["witness"].split())
        states = required_path(start, program, budget)
        if states[-1] != goal or closed_cuts(states, program) != row["witness_cuts"]:
            raise ProvenanceError("witness goal or saved cuts disagree")
        output.update(
            witness_program=program,
            witness_products=[key_of(state) for state in states],
        )
        phase = "catalogue"
        output["catalogue"] = catalogue(states, program, budget)
        phase = "roots"
        expanded: set[Product] = set()
        expanded_at = {}
        segments_seen = set()
        for segment in row["segments"]:
            budget.check_time()
            index = segment["segment"]
            if index in segments_seen or (
                segments_seen and index <= max(segments_seen)
            ):
                raise ProvenanceError("segment indices are not strictly ascending")
            segments_seen.add(index)
            for root in segment["roots"]:
                product = required_path(start, tuple(root["program"].split()), budget)[
                    -1
                ]
                expanded.add(product)
                output["roots"].append(
                    {
                        "segment": index,
                        "program": root["program"],
                        "product": key_of(product),
                    }
                )
            expanded_at[index] = frozenset(expanded)
        phase = "candidates"
        products_by_segment: dict[int, set[Product]] = defaultdict(set)
        witness = set(states)
        for record in records:
            prefix = tuple(record["program"].split())
            trace = required_path(start, prefix, budget)
            product = trace[-1]
            if not prefix or len(prefix) not in closed_cuts(trace, prefix):
                raise ProvenanceError(
                    "retained record is not a terminal combinator cut"
                )
            segment = record["segment"]
            if product in products_by_segment[segment]:
                raise ProvenanceError("equal products retained twice in one segment")
            products_by_segment[segment].add(product)
            candidate = {
                "digest": record["digest"],
                "segment": segment,
                "program": record["program"],
                "prefix_ops": len(prefix),
                "product": key_of(product),
                "on_witness": product in witness,
                "trials": [],
                "label": None,
            }
            output["candidates"].append(candidate)
            flags = {
                "code_bearing": not closed_product(product),
                "expanded": product in expanded_at[segment],
                "goal_equal": product == goal,
            }
            candidate["exclusions"] = [key for key, value in flags.items() if value]
            for key in candidate["exclusions"]:
                counts[key] = counts.get(key, 0) + 1
            if candidate["exclusions"]:
                continue
            counts["closed_eligible"] = counts.get("closed_eligible", 0) + 1
            phase = "matrix"
            for continuation in output["catalogue"]["programs"]:
                result = replay(product, continuation, budget)
                success = (
                    result["status"] == "complete" and result["states"][-1] == goal
                )
                trial = {key: value for key, value in result.items() if key != "states"}
                trial.update(
                    program=continuation,
                    success=success,
                    final_product=key_of(result["states"][-1]),
                )
                candidate["trials"].append(trial)
                if result["status"] == "censored":
                    raise CensoredError(result["reason"], result["scope"])
                if success and (
                    candidate["label"] is None or len(continuation) < candidate["label"]
                ):
                    candidate["label"] = len(continuation)
            phase = "candidates"
        output["pairs"], output["repeated_pair_occurrences"] = ordered_pairs(
            output["candidates"]
        )
        budget.check_time()
    except CensoredError as error:
        output.update(
            status="censored", reason=str(error), scope=error.scope, phase=phase
        )
        output["pairs"] = []
    except (ProvenanceError, KeyError, TypeError, IndexError, ValueError) as error:
        output.update(status="error", reason=str(error), phase=phase)
        output["pairs"] = []
    output["counters"] = budget.counters()
    return output


def group_rows(rows: list[dict]) -> dict[str, str]:
    """Transitive union across ALL train-row behaviour/skeleton/witness metadata."""
    parent = {row["behaviour"]: row["behaviour"] for row in rows}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    linked = {}
    for row in rows:
        name = row["behaviour"]
        for key in (
            ("skeleton", BY_NAME[name].skeleton),
            ("witness", tuple(row["witness"].split())),
        ):
            if key in linked:
                a, b = sorted((find(name), find(linked[key])))
                parent[b] = a
            else:
                linked[key] = name
    return {name: find(name) for name in sorted(parent)}


def gate(tasks: list[dict], groups: dict[str, str], errors: list) -> dict:
    pair_tasks = [
        task for task in tasks if task["status"] == "complete" and task["pairs"]
    ]
    pairs = sum(len(task["pairs"]) for task in pair_tasks)
    eligible_groups = sorted({groups[task["behaviour"]] for task in pair_tasks})
    return {
        "pairs": pairs,
        "groups": eligible_groups,
        "both_off_witness_pairs": sum(
            pair["both_off_witness"] for task in pair_tasks for pair in task["pairs"]
        ),
        "passed": not errors and pairs >= 100 and len(eligible_groups) >= 20,
    }


def inventory(path: Path) -> dict:
    budget = Budget()
    result: dict[str, Any] = {
        "schema_version": 1,
        "input": str(path.resolve()),
        "preregistration": PREREGISTRATION,
        "amendment": AMENDMENT,
        "limits": asdict(budget.limits),
        "tasks": [],
        "errors": [],
        "manifest": [],
        "unprocessed": [],
        "label": "shortest replay-confirmed continuation in the fixed tested catalogue, not optimal distance",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    rows, metadata, seen = [], [], set()
    try:
        source_hash = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1 << 20):
                budget.check_time()
                source_hash.update(block)
        result["sha256"] = source_hash.hexdigest()
        if result["sha256"] != SOURCE_SHA:
            raise ProvenanceError("source hash differs from preregistration")
        with gzip.open(path, "rt") as source:
            for line in source:
                budget.check_time()
                row = json.loads(line)
                try:
                    validate_row(row)
                    key = (row["behaviour"], row["example_set"])
                    if key in seen:
                        raise ProvenanceError("duplicate behaviour/example-set row")
                    seen.add(key)
                    metadata.append(
                        {"behaviour": row["behaviour"], "witness": row["witness"]}
                    )
                    if row["example_set"] == 0:
                        rows.append(row)
                except (ValueError, KeyError, TypeError) as error:
                    result["errors"].append({"phase": "input", "reason": str(error)})
        groups = group_rows(metadata)
        result["group_map"] = groups
        selected = selected_rows(rows)
        result["manifest"] = [
            [row["behaviour"], row["example_set"]] for row in selected
        ]
        result["unprocessed"] = result["manifest"].copy()
        for row in selected:
            task = analyze_task(row, budget)
            result["tasks"].append(task)
            result["unprocessed"].pop(0)
            if task["status"] == "error":
                result["errors"].append(
                    {"behaviour": row["behaviour"], "reason": task["reason"]}
                )
            if task.get("scope") == "run":
                result["stop"] = task["reason"]
                break
        result["gate"] = gate(result["tasks"], groups, result["errors"])
    except CensoredError as error:
        result["stop"] = str(error)
    except (ValueError, KeyError, TypeError) as error:
        result["errors"].append({"phase": "input", "reason": str(error)})
    result.setdefault("gate", {"passed": False, "pairs": 0, "groups": []})
    result["counters"] = budget.counters()
    result["elapsed_seconds"] = time.monotonic() - budget.started
    result["complete"] = (
        not result["unprocessed"]
        and not result["errors"]
        and not result.get("stop")
        and all(task["status"] == "complete" for task in result["tasks"])
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists: do not silently repeat a preregistered inventory")
    result = inventory(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("gate", "counters", "elapsed_seconds", "complete", "errors")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
