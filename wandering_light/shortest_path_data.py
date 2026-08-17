"""Create, read, and re-certify bounded shortest-path datasets."""

import gzip
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from wandering_light.common_functions import basic_fns
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.proposer_pilot import TrajectoryGraph
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ShortestPathRecertification:
    """Fresh lower-bound evidence for one recorded shortest distance."""

    outcome: Literal["certified", "inflated", "inconclusive"]
    recorded_distance: int
    shorter_distance: int | None
    search_depth: int
    certified_search_depth: int
    complete_expansion: bool
    states_searched: int
    transitions_attempted: int
    stop_reason: str | None

    @property
    def ok(self) -> bool:
        return self.outcome == "certified"


def _function_names(functions: FunctionDefList) -> list[str]:
    return [function.name for function in functions]


def _shortest_grading_equal_node(
    graph: TrajectoryGraph,
    node_depths: Mapping[int, int],
    target: TypedList,
) -> tuple[int, int] | None:
    """Return ``(depth, node_id)`` using the equality used to grade solvers."""
    target_key = target.canonical_key()
    return min(
        (
            (depth, node_id)
            for node_id, depth in node_depths.items()
            if graph.node(node_id).typed_list.canonical_key() == target_key
        ),
        default=None,
    )


def recertify_shortest_path(
    input_list: TypedList,
    target: TypedList,
    recorded_distance: int,
    *,
    available_functions: FunctionDefSet = basic_fns,
    max_states: int | None = None,
    max_transitions: int | None = None,
) -> ShortestPathRecertification:
    """Re-prove that ``target`` is unreachable below ``recorded_distance``.

    Expansion prunes states with :meth:`TypedList.search_key`, through
    :class:`TrajectoryGraph`, but reachability uses answer equality. A partial
    expansion can prove an inflated label by finding a shorter target; absence
    is evidence only when the expansion is complete.
    """
    if (
        isinstance(recorded_distance, bool)
        or not isinstance(recorded_distance, int)
        or recorded_distance < 0
    ):
        raise ValueError("recorded_distance must be a non-negative integer")
    if recorded_distance == 0 and input_list != target:
        raise ValueError("a distance-zero target must equal the input")

    search_depth = max(0, recorded_distance - 1)
    graph = TrajectoryGraph(functions=available_functions)
    root = graph.add_root(input_list)
    expansion = graph.expand(
        root,
        max_depth=search_depth,
        max_states=max_states,
        max_transitions=max_transitions,
    )
    match = _shortest_grading_equal_node(graph, expansion.node_depths, target)
    shorter_distance = (
        match[0] if match is not None and match[0] < recorded_distance else None
    )
    if shorter_distance is not None:
        outcome: Literal["certified", "inflated", "inconclusive"] = "inflated"
    elif expansion.complete:
        outcome = "certified"
    else:
        outcome = "inconclusive"

    return ShortestPathRecertification(
        outcome=outcome,
        recorded_distance=recorded_distance,
        shorter_distance=shorter_distance,
        search_depth=search_depth,
        certified_search_depth=expansion.certified_depth,
        complete_expansion=expansion.complete,
        states_searched=expansion.num_reached_states,
        transitions_attempted=expansion.attempted_transitions,
        stop_reason=expansion.stop_reason,
    )


def recertify_shortest_v1_record(
    record: dict[str, Any],
    *,
    available_functions: FunctionDefSet = basic_fns,
    max_states: int | None = None,
    max_transitions: int | None = None,
) -> ShortestPathRecertification:
    """Re-execute and re-certify one ``shortest_v1`` record."""
    if record.get("certified") is not True:
        raise ValueError("record is not marked certified")
    distance = record.get("relabeled_length")
    names = record.get("relabeled_functions")
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("relabeled_length must be a non-negative integer")
    if not isinstance(names, list) or len(names) != distance:
        raise ValueError("relabeled_functions must match relabeled_length")
    if record.get("output") is None:
        raise ValueError("record has no output")

    input_list = TypedList.from_str(record["input"])
    target = TypedList.from_str(record["output"])
    try:
        functions = [
            available_functions.name_to_function[name]
            for name in names
            if isinstance(name, str)
        ]
    except KeyError as error:
        raise ValueError(f"unknown witness function: {error.args[0]!r}") from error
    if len(functions) != distance:
        raise ValueError("relabeled_functions must contain function names")
    if not _execute_candidate(input_list, target, functions, available_functions):
        raise ValueError("stored witness does not reproduce the recorded output")

    return recertify_shortest_path(
        input_list,
        target,
        distance,
        available_functions=available_functions,
        max_states=max_states,
        max_transitions=max_transitions,
    )


def _execute_candidate(
    input_list: TypedList,
    target: TypedList,
    functions: list[FunctionDef],
    available_functions: FunctionDefSet,
) -> bool:
    result = Executor(available_functions).execute_trajectory(
        TrajectorySpec(input_list, FunctionDefList(functions))
    )
    return result.success and result.trajectory.output == target


def _shorter_subsequence(
    spec: TrajectorySpec,
    target: TypedList,
    *,
    min_length: int,
    max_length: int,
    available_functions: FunctionDefSet,
) -> FunctionDefList | None:
    original = list(spec.function_defs)
    for length in range(min_length, max_length + 1):
        for indices in combinations(range(len(original)), length):
            candidate = [original[index] for index in indices]
            if _execute_candidate(spec.input, target, candidate, available_functions):
                return FunctionDefList(candidate)
    return None


def bounded_relabel(
    spec: TrajectorySpec,
    source_index: int,
    *,
    split: str,
    available_functions: FunctionDefSet = basic_fns,
    max_exact_depth: int = 3,
    max_states: int = 100_000,
    max_transitions: int = 2_000_000,
) -> dict[str, Any]:
    """Find a shortest witness when bounded search is sufficient.

    The original random walk is a valid upper bound. Search therefore stops at
    ``min(max_exact_depth, original_length - 1)``: if no shorter state is found
    and the lower and upper bounds meet, the original witness is certified.
    """
    original_names = _function_names(spec.function_defs)
    original_length = len(original_names)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "source_index": source_index,
        "input": spec.input.to_string(),
        "output": None,
        "original_functions": original_names,
        "relabeled_functions": original_names,
        "original_length": original_length,
        "relabeled_length": original_length,
        "lower_bound": 0,
        "upper_bound": original_length,
        "certified": False,
        "method": "invalid",
        "search_depth": 0,
        "certified_search_depth": 0,
        "search_states": 0,
        "search_transitions": 0,
        "search_stop_reason": None,
        "rl_attempted": False,
        "rl_budget": 0,
    }
    execution = Executor(available_functions).execute_trajectory(spec)
    if not execution.success:
        record["error"] = execution.error_msg
        return record

    target = execution.trajectory.output
    record["output"] = target.to_string()
    if target == spec.input:
        record.update(
            relabeled_functions=[],
            relabeled_length=0,
            lower_bound=0,
            upper_bound=0,
            certified=True,
            method="identity",
        )
        return record

    search_depth = min(max_exact_depth, max(0, original_length - 1))
    record["search_depth"] = search_depth
    if search_depth > 0:
        graph = TrajectoryGraph(functions=available_functions)
        root = graph.add_root(spec.input)
        expansion = graph.expand(
            root,
            max_depth=search_depth,
            max_states=max_states,
            max_transitions=max_transitions,
        )
        record.update(
            certified_search_depth=expansion.certified_depth,
            search_states=expansion.num_reached_states,
            search_transitions=expansion.attempted_transitions,
            search_stop_reason=expansion.stop_reason,
        )
        match = _shortest_grading_equal_node(graph, expansion.node_depths, target)
        if match is not None:
            _, destination = match
            path = graph.shortest_path(root, destination)
            if path is None:
                raise AssertionError("reached destination has no graph path")
            path_length = len(path)
            if path_length <= expansion.certified_depth:
                record.update(
                    relabeled_functions=[function.name for function in path],
                    relabeled_length=path_length,
                    lower_bound=path_length,
                    upper_bound=path_length,
                    certified=True,
                    method="bfs",
                )
                return record
        lower_bound = expansion.certified_depth + 1
    else:
        lower_bound = 1

    record["lower_bound"] = lower_bound
    if lower_bound == original_length:
        record.update(certified=True, method="original_after_bfs")
        return record

    candidate = _shorter_subsequence(
        spec,
        target,
        min_length=lower_bound,
        max_length=original_length - 1,
        available_functions=available_functions,
    )
    if candidate is not None:
        candidate_length = len(candidate)
        record.update(
            relabeled_functions=_function_names(candidate),
            relabeled_length=candidate_length,
            upper_bound=candidate_length,
            certified=candidate_length == lower_bound,
            method=(
                "subsequence_after_bfs"
                if candidate_length == lower_bound
                else "unresolved"
            ),
        )
        return record

    record["method"] = "unresolved"
    return record


def apply_solver_candidate(
    record: dict[str, Any],
    candidate: FunctionDefList | None,
    *,
    available_functions: FunctionDefSet = basic_fns,
    budget: int,
) -> dict[str, Any]:
    """Validate an RL candidate and tighten a bounded relabel record."""
    updated = dict(record)
    updated.update(rl_attempted=True, rl_budget=budget)
    if candidate is None or updated["output"] is None:
        return updated

    input_list = TypedList.from_str(updated["input"])
    target = TypedList.from_str(updated["output"])
    functions = list(candidate)
    if not _execute_candidate(input_list, target, functions, available_functions):
        return updated

    candidate_length = len(candidate)
    if candidate_length < updated["lower_bound"]:
        raise ValueError(
            "solver candidate contradicts the certified BFS lower bound: "
            f"{candidate_length} < {updated['lower_bound']}"
        )
    if candidate_length >= updated["upper_bound"]:
        return updated

    updated.update(
        relabeled_functions=_function_names(candidate),
        relabeled_length=candidate_length,
        upper_bound=candidate_length,
        certified=candidate_length == updated["lower_bound"],
        method=(
            "rl_after_bfs"
            if candidate_length == updated["lower_bound"]
            else "unresolved"
        ),
    )
    return updated


def record_to_spec(
    record: dict[str, Any],
    *,
    available_functions: FunctionDefSet = basic_fns,
) -> TrajectorySpec:
    if not record["certified"]:
        raise ValueError("cannot create a shortest-path spec from uncertified evidence")
    input_list = TypedList.from_str(record["input"])
    functions = [
        available_functions.name_to_function[name]
        for name in record["relabeled_functions"]
    ]
    return TrajectorySpec(input_list, FunctionDefList(functions))


def certified_specs(
    records: list[dict[str, Any]],
    *,
    exclude_identity: bool = True,
    available_functions: FunctionDefSet = basic_fns,
) -> TrajectorySpecList:
    selected = [
        record
        for record in records
        if record["certified"]
        and record["output"] is not None
        and (not exclude_identity or record["relabeled_length"] > 0)
    ]
    return TrajectorySpecList(
        [
            record_to_spec(record, available_functions=available_functions)
            for record in selected
        ]
    )


def write_jsonl_gz(records: list[dict[str, Any]], path: Path) -> None:
    """Write deterministic gzip-compressed JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text_file,
    ):
        for record in records:
            text_file.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]
