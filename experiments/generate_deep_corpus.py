"""Forward generation of certified long-distance tasks via breadth-first expansion.

Relabelling random walks cannot produce long tasks: 84% of nominal length-five
walks admit a shorter subsequence, so the certified mass collapses to distance
four and below.  This generator works forward instead.  In a *complete*
breadth-first expansion from a root state, a node first reached at depth ``k``
is at distance exactly ``k`` by construction — the label falls out of the
traversal, with no search and no relabelling pass.

Two things follow for free, and both are recorded:

* the certified distance of every emitted task, and
* the **complete set of optimal first actions** for that task, read off the
  shortest-path DAG (action ``a`` is optimal for target ``t`` iff the depth-one
  state ``f_a(root)`` can still reach ``t`` along strictly-increasing depths).

Certified distances beyond the exhaustive depth come from a *frontier
extension*: given a complete expansion through depth ``D``, any state reached
in one more step that was **not** seen at depth ``<= D`` is at distance exactly
``D + 1``.  That certification is exact even when only a sample of the frontier
is extended; only the optimal-action set is then partial, and it is flagged as
such.

Typical use::

    python -m experiments.generate_deep_corpus --mode run \
      --roots 120 --max-depth 6 --deep-roots 12 --deep-max-depth 7 \
      --output-dir wandering_light/training/data/deep_corpus_v1

    python -m experiments.generate_deep_corpus --mode verify \
      --output-dir wandering_light/training/data/deep_corpus_v1 --recertify 24

    python -m experiments.generate_deep_corpus --mode reference-curve \
      --output-dir wandering_light/training/data/deep_corpus_v1 \
      --curve-depths 1,2,3,4,5,6 --curve-tasks-per-distance 12
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.basis_library_pilot import (
    SUPPORTED_RANDOM_TYPES,
    VALUE_STRATA_DESCRIPTION,
    _canonical_json,
    _function_identity,
    _has_multiple_output_values,
    _random_input,
    _sha256_bytes,
    _sha256_file,
    _type_name,
    _write_json,
)
from wandering_light.basis_dataset import (
    BasisTaskRecord,
    read_basis_task_records,
    write_basis_task_records,
)
from wandering_light.basis_set import (
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefList
from wandering_light.proposer_pilot.graph import TrajectoryGraph
from wandering_light.trajectory import TrajectorySpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.basis_set import BasisSet
    from wandering_light.function_def import FunctionDef, FunctionDefSet
    from wandering_light.typed_list import TypedList

PIPELINE_SCHEMA_VERSION = 1
GENERATOR_NAME = "trajectory-graph-forward-expansion-v1"
ROOT_GENERATOR_NAME = "builtin-behavior-strata-v2"
SPLIT_ORDER = ("discovery", "validation", "test")
# Roots, not tasks, are split: every task sharing a root shares its split.
DEFAULT_SPLIT_CYCLE = ("discovery",) * 7 + ("validation", "test")
DEFAULT_SEED = 20_260_816
DEFAULT_OUTPUT_DIR = Path("wandering_light/training/data/deep_corpus_v1")
DEFAULT_MAX_STATES = 250_000
DEFAULT_MAX_TRANSITIONS = 1_200_000
DEFAULT_DEEP_MAX_STATES = 1_200_000
DEFAULT_DEEP_MAX_TRANSITIONS = 6_000_000
CERTIFICATION_COMPLETE = "complete-bfs-expansion"
CERTIFICATION_FRONTIER = "frontier-extension"


@dataclass
class RootPlan:
    """One root state, its split, and the expansion budget it was given."""

    index: int
    split: str
    input_type: type[Any]
    typed_list: TypedList
    max_depth: int
    max_states: int
    max_transitions: int
    deep: bool


@dataclass
class PendingTask:
    """A certified task held before its split file position is known."""

    root_index: int
    split: str
    input_value: TypedList
    output_value: TypedList
    witness: list[FunctionDef]
    certified_distance: int
    certification: str
    expansion_certified_depth: int
    optimal_first_actions: list[FunctionDef]
    optimal_first_actions_complete: bool
    optimal_last_actions: list[FunctionDef]
    optimal_last_actions_complete: bool
    shell_size: int


@dataclass
class RootOutcome:
    """Per-root evidence retained for the manifest."""

    root_index: int
    split: str
    input_type: str
    max_depth: int
    certified_depth: int
    extended_depth: int
    complete: bool
    stop_reason: str | None
    reached_states: int
    attempted_transitions: int
    failed_transitions: int
    skipped_self_loops: int
    shell_sizes: dict[int, int]
    seconds: float
    emitted: int
    # Answer-equality classes per depth: the number of distinct *tasks* a layer
    # can supply, which is at most its state count and is what the sampler draws
    # from. Kept apart from shell_sizes so states_by_depth stays a state count.
    class_sizes: dict[int, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_index": self.root_index,
            "split": self.split,
            "input_type": self.input_type,
            "max_depth": self.max_depth,
            "certified_depth": self.certified_depth,
            "extended_depth": self.extended_depth,
            "complete": self.complete,
            "stop_reason": self.stop_reason,
            "reached_states": self.reached_states,
            "attempted_transitions": self.attempted_transitions,
            "failed_transitions": self.failed_transitions,
            "skipped_self_loops": self.skipped_self_loops,
            "shell_sizes": {str(k): v for k, v in sorted(self.shell_sizes.items())},
            "class_sizes": {str(k): v for k, v in sorted(self.class_sizes.items())},
            "seconds": round(self.seconds, 3),
            "emitted": self.emitted,
            "rejections": dict(sorted(self.rejections.items())),
        }


def build_root_plans(
    *,
    roots: int,
    seed: int,
    max_depth: int,
    deep_roots: int,
    deep_max_depth: int,
    max_states: int,
    max_transitions: int,
    deep_max_states: int,
    deep_max_transitions: int,
    deep_root_types: Sequence[str] = (),
    split_cycle: Sequence[str] = DEFAULT_SPLIT_CYCLE,
) -> list[RootPlan]:
    """Draw distinct roots, cycling input types and assigning splits by root."""
    if roots <= 0:
        raise ValueError("roots must be positive")
    if deep_roots < 0 or deep_roots > roots:
        raise ValueError("deep_roots must be between 0 and roots")
    rng = random.Random(seed)
    seen: set[Any] = set()
    plans: list[RootPlan] = []
    # Types cycle by position, so eligibility is decided on the position alone.
    # Restricting deep roots keeps the depth budget off types whose branching
    # factor makes the extra level unreachable anyway.
    eligible = [
        index
        for index in range(roots)
        if not deep_root_types
        or _type_name(SUPPORTED_RANDOM_TYPES[index % len(SUPPORTED_RANDOM_TYPES)])
        in deep_root_types
    ]
    if deep_roots > len(eligible):
        raise ValueError(
            f"deep_roots={deep_roots} exceeds the {len(eligible)} eligible root "
            f"positions for types {sorted(deep_root_types)}"
        )
    # Deep roots are spread evenly so the depth budget is not spent on one type.
    deep_positions = (
        {
            eligible[round(index * (len(eligible) - 1) / max(1, deep_roots - 1))]
            for index in range(deep_roots)
        }
        if deep_roots
        else set()
    )
    attempts = 0
    attempt_limit = 200 * roots
    # Splits are walked per input type, not per root index.  Indexing the cycle
    # by the raw index couples the split to the type whenever the two periods
    # share a factor -- with 12 types and a 9-long cycle, gcd 3, each split can
    # only ever draw four types and the rest never leave discovery.  Counting
    # within a type and offsetting by the type breaks that coupling: every type
    # walks the whole cycle, and short runs still populate the held-out splits
    # because each type starts at a different point in it.
    type_positions: Counter[Any] = Counter()
    while len(plans) < roots:
        attempts += 1
        if attempts > attempt_limit:
            raise RuntimeError(
                f"Could not draw {roots} distinct roots in {attempt_limit} attempts"
            )
        type_index = len(plans) % len(SUPPORTED_RANDOM_TYPES)
        input_type = SUPPORTED_RANDOM_TYPES[type_index]
        candidate = _random_input(input_type, rng)
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        index = len(plans)
        deep = index in deep_positions
        position = type_positions[input_type]
        type_positions[input_type] += 1
        plans.append(
            RootPlan(
                index=index,
                split=split_cycle[(position + type_index) % len(split_cycle)],
                input_type=input_type,
                typed_list=candidate,
                max_depth=deep_max_depth if deep else max_depth,
                max_states=deep_max_states if deep else max_states,
                max_transitions=deep_max_transitions if deep else max_transitions,
                deep=deep,
            )
        )
    return plans


def _grading_classes(
    graph: TrajectoryGraph, depths: Mapping[int, int]
) -> tuple[dict[Any, int], dict[Any, list[int]]]:
    """Collapse the reached states onto the relation a solver is graded by.

    Expansion deliberately separates states that no basis function may confuse,
    so ``-0.0`` and ``0.0`` explore their own successors.  A solution, though,
    is graded with ``==``, which equates them.  Distance therefore has to be
    read after collapsing back: the distance of a target is the shallowest depth
    at which *any* answer-equal state is reached.

    Measuring in the finer space instead files a target as deep as its own path,
    even when a solver may legitimately stop earlier at an equal value, and the
    label is then unreachable by the very solver it is meant to score.

    Returns the depth of each class and, for each class, the members sitting at
    that depth -- the only ones on a shortest path to it.
    """
    depth_of: dict[Any, int] = {}
    keys: dict[int, Any] = {}
    for node_id, depth in depths.items():
        key = graph.node(node_id).typed_list.canonical_key()
        keys[node_id] = key
        current = depth_of.get(key)
        if current is None or depth < current:
            depth_of[key] = depth
    members: defaultdict[Any, list[int]] = defaultdict(list)
    for node_id, depth in depths.items():
        if depth == depth_of[keys[node_id]]:
            members[keys[node_id]].append(node_id)
    for group in members.values():
        group.sort()
    return depth_of, dict(members)


def _shortest_path_dag_reach(
    graph: TrajectoryGraph,
    depths: Mapping[int, int],
    certified_depth: int,
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """Bitmask reachability from each depth-one state over the shortest-path DAG.

    ``reach[t]`` has bit ``i`` set exactly when the ``i``-th depth-one state can
    still reach ``t`` along edges that increase depth by one.  Because a
    depth-increasing path of length ``d - 1`` from a depth-one state ``u`` to a
    depth-``d`` state ``t`` exists iff ``dist(u, t) == d - 1``, that bit is
    precisely the statement "the action into ``u`` is an optimal first action
    for target ``t``".
    """
    first_states = [node_id for node_id, depth in depths.items() if depth == 1]
    bit_of = {node_id: 1 << index for index, node_id in enumerate(first_states)}
    reach: dict[int, int] = dict(bit_of)
    by_depth: defaultdict[int, list[int]] = defaultdict(list)
    for node_id, depth in depths.items():
        by_depth[depth].append(node_id)
    for depth in range(1, certified_depth):
        for parent_id in by_depth[depth]:
            mask = reach.get(parent_id, 0)
            if not mask:
                continue
            for _, child_id in graph.node(parent_id).out_edges:
                if depths.get(child_id) == depth + 1:
                    reach[child_id] = reach.get(child_id, 0) | mask
    return first_states, bit_of, reach


def _first_actions_by_state(
    graph: TrajectoryGraph, depths: Mapping[int, int], root_id: int
) -> dict[int, list[FunctionDef]]:
    """Group the root's outgoing basis functions by the state they produce."""
    grouped: defaultdict[int, list[FunctionDef]] = defaultdict(list)
    for function, child_id in graph.node(root_id).out_edges:
        if depths.get(child_id) == 1:
            grouped[child_id].append(function)
    return dict(grouped)


def _optimal_first_actions(
    mask: int,
    first_states: Sequence[int],
    bit_of: Mapping[int, int],
    first_actions: Mapping[int, list[FunctionDef]],
) -> list[FunctionDef]:
    actions: list[FunctionDef] = []
    for state_id in first_states:
        if mask & bit_of[state_id]:
            actions.extend(first_actions.get(state_id, ()))
    return actions


def _optimal_in_edges(
    graph: TrajectoryGraph, depths: Mapping[int, int], node_id: int, depth: int
) -> list[tuple[FunctionDef, int]]:
    return [
        (function, parent_id)
        for function, parent_id in graph.node(node_id).in_edges
        if depths.get(parent_id) == depth - 1
    ]


def _is_effective_identity(source: TypedList, target: TypedList) -> bool:
    """Whether a transition changed the declared item type but no value.

    States are deduplicated on ``(item_type, values)``, so ``int_to_float`` over
    ``[1, 2]`` produces a *new* state whose items still compare equal to its
    parent's.  Such an edge carries no value-level work, and a target reachable
    only through those edges is not a real task.
    """
    return list(source.items) == list(target.items)


def _witness_path(
    graph: TrajectoryGraph,
    depths: Mapping[int, int],
    node_id: int,
    depth: int,
    rng: random.Random,
) -> list[FunctionDef]:
    """Walk one uniformly chosen shortest path back to the root."""
    path: list[FunctionDef] = []
    current, current_depth = node_id, depth
    while current_depth > 0:
        candidates = _optimal_in_edges(graph, depths, current, current_depth)
        if not candidates:
            raise ValueError(f"state {current} has no depth-{current_depth - 1} parent")
        function, parent_id = rng.choice(candidates)
        path.append(function)
        current, current_depth = parent_id, current_depth - 1
    path.reverse()
    return path


def _emit_state(
    *,
    graph: TrajectoryGraph,
    plan: RootPlan,
    node_id: int,
    root_id: int,
    distance: int,
    certification: str,
    expansion_certified_depth: int,
    mask: int,
    mask_complete: bool,
    first_states: Sequence[int],
    bit_of: Mapping[int, int],
    first_actions: Mapping[int, list[FunctionDef]],
    witness: list[FunctionDef],
    in_edges: Sequence[tuple[FunctionDef, int]],
    in_edges_complete: bool,
    shell_size: int,
    allow_constant_outputs: bool,
    rejections: Counter[str],
) -> PendingTask | None:
    """Apply the corpus filters and build one pending task."""
    target = graph.node(node_id).typed_list
    root_value = plan.typed_list
    if node_id == root_id:
        rejections["self_loop_task"] += 1
        return None
    if _is_effective_identity(root_value, target):
        rejections["value_identity_task"] += 1
        return None
    if in_edges_complete and all(
        _is_effective_identity(graph.node(parent_id).typed_list, target)
        for _, parent_id in in_edges
    ):
        rejections["effective_identity_in_edges"] += 1
        return None
    if not allow_constant_outputs and not _has_multiple_output_values(target):
        rejections["constant_output_example"] += 1
        return None
    optimal_first = _optimal_first_actions(mask, first_states, bit_of, first_actions)
    if not optimal_first:
        rejections["no_optimal_first_action"] += 1
        return None
    return PendingTask(
        root_index=plan.index,
        split=plan.split,
        input_value=root_value,
        output_value=target,
        witness=witness,
        certified_distance=distance,
        certification=certification,
        expansion_certified_depth=expansion_certified_depth,
        optimal_first_actions=optimal_first,
        optimal_first_actions_complete=mask_complete,
        optimal_last_actions=[function for function, _ in in_edges],
        optimal_last_actions_complete=in_edges_complete,
        shell_size=shell_size,
    )


def expand_root(
    plan: RootPlan,
    *,
    functions: FunctionDefSet,
    seed: int,
    min_distance: int,
    tasks_per_distance: int,
    frontier_sample: int,
    allow_constant_outputs: bool,
    verify_witnesses: bool,
) -> tuple[list[PendingTask], RootOutcome]:
    """Expand one root exhaustively and emit certified tasks from the result."""
    rng = random.Random(f"{seed}:root:{plan.index}")
    graph = TrajectoryGraph(functions)
    executor = Executor(functions)
    started = time.perf_counter()
    root_id = graph.add_root(plan.typed_list)
    expansion = graph.expand(
        root_id,
        plan.max_depth,
        max_states=plan.max_states,
        max_transitions=plan.max_transitions,
    )
    depths = expansion.node_depths
    certified_depth = expansion.certified_depth
    nodes_by_depth: defaultdict[int, list[int]] = defaultdict(list)
    for node_id, depth in depths.items():
        nodes_by_depth[depth].append(node_id)

    # Tasks are one per answer-equality class, filed at the class's own depth.
    class_depth, class_members = _grading_classes(graph, depths)
    by_depth: defaultdict[int, list[Any]] = defaultdict(list)
    for key, depth in class_depth.items():
        by_depth[depth].append(key)
    for group in by_depth.values():
        # Canonical keys are not orderable across types; the smallest member
        # node id is, and it is stable for a given expansion.
        group.sort(key=lambda key: class_members[key][0])

    first_states, bit_of, reach = _shortest_path_dag_reach(
        graph, depths, certified_depth
    )
    first_actions = _first_actions_by_state(graph, depths, root_id)
    rejections: Counter[str] = Counter()
    pending: list[PendingTask] = []

    for distance in range(max(1, min_distance), certified_depth + 1):
        shell = by_depth.get(distance, [])
        if not shell:
            continue
        # Oversample: the filters below reject a minority of candidate states.
        sample_size = min(len(shell), max(tasks_per_distance * 4, tasks_per_distance))
        accepted = 0
        for key in rng.sample(shell, sample_size):
            if accepted >= tasks_per_distance:
                break
            members = class_members[key]
            representative = members[0]
            # Any answer-equal state at this depth is a correct answer, so every
            # way of reaching one contributes an optimal action.
            mask = 0
            in_edges: list[tuple[FunctionDef, int]] = []
            for member in members:
                mask |= reach.get(member, 0)
                in_edges.extend(_optimal_in_edges(graph, depths, member, distance))
            task = _emit_state(
                graph=graph,
                plan=plan,
                node_id=representative,
                root_id=root_id,
                distance=distance,
                certification=CERTIFICATION_COMPLETE,
                expansion_certified_depth=certified_depth,
                mask=mask,
                mask_complete=True,
                first_states=first_states,
                bit_of=bit_of,
                first_actions=first_actions,
                witness=_witness_path(graph, depths, representative, distance, rng),
                in_edges=in_edges,
                in_edges_complete=True,
                shell_size=len(shell),
                allow_constant_outputs=allow_constant_outputs,
                rejections=rejections,
            )
            if task is not None:
                pending.append(task)
                accepted += 1

    extended_depth = 0
    # Layers 0..certified_depth-1 were expanded exhaustively, so the shell at
    # certified_depth is itself complete and can be extended even when a budget
    # stopped the run before max_depth.
    if frontier_sample and certified_depth >= 1:
        extension, extension_rejections = _extend_frontier(
            graph=graph,
            executor=executor,
            plan=plan,
            depths=depths,
            frontier=nodes_by_depth.get(certified_depth, []),
            reached_classes=class_depth,
            certified_depth=certified_depth,
            reach=reach,
            first_states=first_states,
            bit_of=bit_of,
            first_actions=first_actions,
            rng=rng,
            frontier_sample=frontier_sample,
            tasks_per_distance=tasks_per_distance,
            allow_constant_outputs=allow_constant_outputs,
        )
        rejections.update(extension_rejections)
        if extension:
            extended_depth = certified_depth + 1
            pending.extend(extension)

    if verify_witnesses:
        for task in pending:
            execution = executor.execute_trajectory(
                TrajectorySpec(task.input_value, FunctionDefList(list(task.witness)))
            )
            if (
                not execution.success
                or execution.trajectory.output != task.output_value
            ):
                raise RuntimeError(
                    f"witness did not reproduce the target for root {plan.index} "
                    f"at distance {task.certified_distance}"
                )
            if len(task.witness) != task.certified_distance:
                raise RuntimeError(
                    f"witness length {len(task.witness)} != certified distance "
                    f"{task.certified_distance} for root {plan.index}"
                )

    outcome = RootOutcome(
        root_index=plan.index,
        split=plan.split,
        input_type=_type_name(plan.input_type),
        max_depth=plan.max_depth,
        certified_depth=certified_depth,
        extended_depth=extended_depth,
        complete=expansion.complete,
        stop_reason=expansion.stop_reason,
        reached_states=expansion.num_reached_states,
        attempted_transitions=expansion.attempted_transitions,
        failed_transitions=expansion.failed_transitions,
        skipped_self_loops=expansion.skipped_self_loops,
        # States, not answer classes: this feeds the manifest's states_by_depth,
        # which has to stay reconcilable with reached_states.
        shell_sizes={depth: len(nodes) for depth, nodes in nodes_by_depth.items()},
        class_sizes={depth: len(keys) for depth, keys in by_depth.items()},
        seconds=time.perf_counter() - started,
        emitted=len(pending),
        rejections=dict(rejections),
    )
    return pending, outcome


def _extend_frontier(
    *,
    graph: TrajectoryGraph,
    executor: Executor,
    plan: RootPlan,
    depths: Mapping[int, int],
    frontier: Sequence[int],
    reached_classes: Mapping[Any, int],
    certified_depth: int,
    reach: Mapping[int, int],
    first_states: Sequence[int],
    bit_of: Mapping[int, int],
    first_actions: Mapping[int, list[FunctionDef]],
    rng: random.Random,
    frontier_sample: int,
    tasks_per_distance: int,
    allow_constant_outputs: bool,
) -> tuple[list[PendingTask], Counter[str]]:
    """Certify distance ``certified_depth + 1`` from a sample of the frontier.

    The exhaustive layers already prove that every state at distance
    ``<= certified_depth`` is in the graph, so a state produced in one more step
    and absent from the graph is at distance exactly ``certified_depth + 1``.
    Only a sample of the frontier is expanded, so the optimal-action sets
    recovered here are partial and are flagged
    ``optimal_first_actions_complete: false``.

    When a budget stopped the expansion mid-layer the graph also holds part of
    the next shell; those states are skipped rather than mislabelled, which
    costs candidates but never certification.
    """
    rejections: Counter[str] = Counter()
    if not frontier:
        return [], rejections
    distance = certified_depth + 1
    parents = rng.sample(frontier, min(len(frontier), frontier_sample))
    candidates: dict[Any, dict[str, Any]] = {}
    for parent_id in parents:
        parent_value = graph.node(parent_id).typed_list
        parent_mask = reach.get(parent_id, 0)
        for function in functions_for(graph, parent_value):
            try:
                result = executor.execute(function, parent_value)
            except Exception:
                continue
            key = result.canonical_key()
            if key in reached_classes:
                # An answer-equal state is already reachable within
                # certified_depth, so this step does not extend the distance.
                # The test is answer equality, not search identity: a solver
                # that stops at the equal state is graded correct, so a target
                # equal to one of them is not one step further out.
                continue
            # Candidates group by answer equality for the same reason. Two
            # results a solver cannot be asked to tell apart are one task, and
            # every way of reaching either is an optimal action for it.
            entry = candidates.get(key)
            if entry is None:
                candidates[key] = {
                    "value": result,
                    "mask": parent_mask,
                    "in_edges": [(function, parent_id)],
                }
            else:
                entry["mask"] |= parent_mask
                entry["in_edges"].append((function, parent_id))

    emitted: list[PendingTask] = []
    # Insertion order is already deterministic; shuffle so the emitted sample is
    # not biased toward the first sampled frontier parents.
    keys = list(candidates)
    rng.shuffle(keys)
    for key in keys:
        if len(emitted) >= tasks_per_distance:
            break
        entry = candidates[key]
        target = entry["value"]
        witness_function, witness_parent = entry["in_edges"][0]
        if _is_effective_identity(plan.typed_list, target):
            rejections["value_identity_task"] += 1
            continue
        if all(
            _is_effective_identity(graph.node(parent_id).typed_list, target)
            for _, parent_id in entry["in_edges"]
        ):
            rejections["effective_identity_in_edges"] += 1
            continue
        if not allow_constant_outputs and not _has_multiple_output_values(target):
            rejections["constant_output_example"] += 1
            continue
        optimal_first = _optimal_first_actions(
            entry["mask"], first_states, bit_of, first_actions
        )
        if not optimal_first:
            rejections["no_optimal_first_action"] += 1
            continue
        witness = [
            *_witness_path(graph, depths, witness_parent, certified_depth, rng),
            witness_function,
        ]
        emitted.append(
            PendingTask(
                root_index=plan.index,
                split=plan.split,
                input_value=plan.typed_list,
                output_value=target,
                witness=witness,
                certified_distance=distance,
                certification=CERTIFICATION_FRONTIER,
                expansion_certified_depth=certified_depth,
                optimal_first_actions=optimal_first,
                optimal_first_actions_complete=False,
                optimal_last_actions=[f for f, _ in entry["in_edges"]],
                optimal_last_actions_complete=False,
                shell_size=len(candidates),
            )
        )
    return emitted, rejections


def functions_for(graph: TrajectoryGraph, value: TypedList) -> list[FunctionDef]:
    """Basis functions whose declared input type matches this state."""
    return [
        function
        for function in graph.functions
        if function.input_type_cls() is value.item_type
    ]


def _record_for(
    task: PendingTask,
    *,
    basis: BasisSet,
    seed: int,
    source_index: int,
) -> BasisTaskRecord:
    witness_ids: list[str] = []
    witness_names: list[str] = []
    for function in task.witness:
        stable_id, name = _function_identity(function)
        witness_ids.append(stable_id)
        witness_names.append(name)
    first_ids = sorted({_function_identity(f)[0] for f in task.optimal_first_actions})
    first_names = sorted({f.name for f in task.optimal_first_actions})
    last_ids = sorted({_function_identity(f)[0] for f in task.optimal_last_actions})
    return BasisTaskRecord.create(
        split=task.split,
        input_value=task.input_value,
        output_value=task.output_value,
        witness_function_ids=witness_ids,
        witness_function_names=witness_names,
        basis_set_id=basis.basis_set_id,
        basis_set_digest=basis.digest,
        generator=GENERATOR_NAME,
        seed=seed,
        source_index=source_index,
        metadata={
            "input_type": _type_name(task.input_value.item_type),
            "output_type": _type_name(task.output_value.item_type),
            "certified_distance": task.certified_distance,
            "certification": task.certification,
            "expansion_certified_depth": task.expansion_certified_depth,
            "optimal_first_action_ids": first_ids,
            "optimal_first_action_names": first_names,
            "optimal_first_actions_complete": task.optimal_first_actions_complete,
            "optimal_last_action_ids": last_ids,
            "optimal_last_actions_complete": task.optimal_last_actions_complete,
            "root_index": task.root_index,
            "root_digest": _sha256_bytes(task.input_value.to_string().encode("utf-8")),
            "distance_shell_size": task.shell_size,
        },
    )


def _distance_histogram(records: Sequence[BasisTaskRecord]) -> dict[str, int]:
    counts = Counter(record.metadata["certified_distance"] for record in records)
    return {str(key): counts[key] for key in sorted(counts)}


def _distance_certification(records: Sequence[BasisTaskRecord]) -> dict[str, Any]:
    """How each distance's mass was certified: exhaustive layer or extension."""
    grouped: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[record.metadata["certified_distance"]][
            record.metadata["certification"]
        ] += 1
    return {
        str(distance): dict(sorted(grouped[distance].items()))
        for distance in sorted(grouped)
    }


def _split_summary(records: Sequence[BasisTaskRecord]) -> dict[str, Any]:
    first_action_counts = [
        len(record.metadata["optimal_first_action_ids"]) for record in records
    ]
    return {
        "size": len(records),
        "by_certified_distance": _distance_histogram(records),
        "by_distance_certification": _distance_certification(records),
        "by_input_type": dict(
            sorted(Counter(r.metadata["input_type"] for r in records).items())
        ),
        "by_output_type": dict(
            sorted(Counter(r.metadata["output_type"] for r in records).items())
        ),
        "by_certification": dict(
            sorted(Counter(r.metadata["certification"] for r in records).items())
        ),
        "roots": len({r.metadata["root_index"] for r in records}),
        "root_digests": len({r.metadata["root_digest"] for r in records}),
        "optimal_first_actions": {
            "mean": (
                statistics.fmean(first_action_counts) if first_action_counts else None
            ),
            "median": (
                statistics.median(first_action_counts) if first_action_counts else None
            ),
            "max": max(first_action_counts, default=None),
            "complete_labels": sum(
                bool(r.metadata["optimal_first_actions_complete"]) for r in records
            ),
        },
    }


def generate_corpus(
    *,
    basis_set_id: str,
    output_dir: str | Path,
    roots: int,
    max_depth: int,
    deep_roots: int,
    deep_max_depth: int,
    min_distance: int,
    tasks_per_distance: int,
    frontier_sample: int,
    max_states: int,
    max_transitions: int,
    deep_max_states: int,
    deep_max_transitions: int,
    seed: int,
    deep_root_types: Sequence[str] = (),
    allow_constant_outputs: bool = False,
    verify_witnesses: bool = True,
    overwrite: bool = False,
    progress: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Expand every root, then write split files and a self-describing manifest."""
    basis = load_basis_set(basis_set_id)
    require_reproducible_basis_runtime(basis)
    functions = basis.as_function_set()
    corpus_dir = Path(output_dir)
    planned = [corpus_dir / f"{split}.jsonl.gz" for split in SPLIT_ORDER]
    planned.append(corpus_dir / "manifest.json")
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Corpus outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    plans = build_root_plans(
        roots=roots,
        seed=seed,
        max_depth=max_depth,
        deep_roots=deep_roots,
        deep_max_depth=deep_max_depth,
        max_states=max_states,
        max_transitions=max_transitions,
        deep_max_states=deep_max_states,
        deep_max_transitions=deep_max_transitions,
        deep_root_types=deep_root_types,
    )
    pending_by_split: dict[str, list[PendingTask]] = {s: [] for s in SPLIT_ORDER}
    outcomes: list[RootOutcome] = []
    started = time.perf_counter()
    for plan in plans:
        tasks, outcome = expand_root(
            plan,
            functions=functions,
            seed=seed,
            min_distance=min_distance,
            tasks_per_distance=tasks_per_distance,
            frontier_sample=frontier_sample,
            allow_constant_outputs=allow_constant_outputs,
            verify_witnesses=verify_witnesses,
        )
        pending_by_split[plan.split].extend(tasks)
        outcomes.append(outcome)
        if progress:
            print(
                f"[root {plan.index + 1}/{len(plans)}] {outcome.input_type} "
                f"split={plan.split} certified={outcome.certified_depth}"
                f"{'+1' if outcome.extended_depth else ''} "
                f"states={outcome.reached_states} tasks={outcome.emitted} "
                f"{outcome.seconds:.1f}s",
                flush=True,
            )

    # Answer equality, not task_id: task_id hashes the serialized values, and
    # `-0.0` and `0.0` serialize differently while grading as the same answer.
    # Keying on it lets two records that pose one task both through, which is
    # how a pair carrying different certified distances survived.
    seen_task_keys: set[tuple[Any, Any]] = set()
    duplicate_tasks = 0
    unserializable = 0
    split_metadata: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        tasks = pending_by_split[split]
        # Heterogeneous file order without disturbing the generation streams.
        random.Random(f"{seed}:split:{split}").shuffle(tasks)
        records: list[BasisTaskRecord] = []
        for task in tasks:
            try:
                record = _record_for(
                    task, basis=basis, seed=seed, source_index=len(records)
                )
            except (ValueError, TypeError, OverflowError):
                unserializable += 1
                continue
            task_key = (
                task.input_value.canonical_key(),
                task.output_value.canonical_key(),
            )
            if task_key in seen_task_keys:
                duplicate_tasks += 1
                continue
            seen_task_keys.add(task_key)
            records.append(record)
        path = corpus_dir / f"{split}.jsonl.gz"
        write_basis_task_records(records, path)
        summary = _split_summary(records)
        summary.update({"path": path.name, "sha256": _sha256_file(path), "seed": seed})
        split_metadata[split] = summary

    split_roots = {
        split: sorted(
            {
                outcome.root_index
                for outcome in outcomes
                if outcome.split == split and outcome.emitted
            }
        )
        for split in SPLIT_ORDER
    }
    overlapping = [
        (left, right)
        for index, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[index + 1 :]
        if set(split_roots[left]) & set(split_roots[right])
    ]
    if overlapping:
        raise RuntimeError(f"roots leak across splits: {overlapping}")

    manifest: dict[str, Any] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "generator_description": (
            "Complete breadth-first expansion from each root; a state first "
            "reached at depth k is certified at distance k. One sampled frontier "
            "layer certifies distance k+1 against the complete depth-k set. "
            "Expansion separates states no basis function may confuse, but "
            "distance is read after collapsing them onto answer equality, so a "
            "certified distance is the fewest steps to any output a solver is "
            "graded correct for."
        ),
        "distance_semantics": (
            "shortest number of basis applications from input to any output "
            "equal to the recorded one under TypedList.__eq__"
        ),
        "root_generator": {
            "name": ROOT_GENERATOR_NAME,
            "strata": VALUE_STRATA_DESCRIPTION,
            "input_types": [_type_name(t) for t in SUPPORTED_RANDOM_TYPES],
        },
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "seed": seed,
        "config": {
            "roots": roots,
            "max_depth": max_depth,
            "deep_roots": deep_roots,
            "deep_max_depth": deep_max_depth,
            "min_distance": min_distance,
            "tasks_per_distance": tasks_per_distance,
            "frontier_sample": frontier_sample,
            "max_states": max_states,
            "max_transitions": max_transitions,
            "deep_max_states": deep_max_states,
            "deep_max_transitions": deep_max_transitions,
            "deep_root_types": list(deep_root_types),
            "allow_constant_outputs": allow_constant_outputs,
            "split_cycle": list(DEFAULT_SPLIT_CYCLE),
        },
        "filters": {
            "skip_self_loops": True,
            "reject_value_identity_task": True,
            "reject_effective_identity_in_edges": True,
            "reject_constant_output_example": not allow_constant_outputs,
            "effective_identity_definition": (
                "transition whose item values compare equal to its parent's, so "
                "only the declared item type changed"
            ),
            "constant_output_definition": "fewer than two distinct output items",
        },
        "split_policy": "by root: every task from one root shares that root's split",
        "global_task_count": len(seen_task_keys),
        "global_dedupe_key": "(canonical_key(input), canonical_key(output))",
        "duplicate_tasks_rejected": duplicate_tasks,
        "unserializable_states_rejected": unserializable,
        "expansion": {
            "roots_expanded": len(outcomes),
            "roots_complete": sum(outcome.complete for outcome in outcomes),
            "reached_states": sum(outcome.reached_states for outcome in outcomes),
            "attempted_transitions": sum(
                outcome.attempted_transitions for outcome in outcomes
            ),
            "failed_transitions": sum(
                outcome.failed_transitions for outcome in outcomes
            ),
            "skipped_self_loops": sum(
                outcome.skipped_self_loops for outcome in outcomes
            ),
            "wall_seconds": round(time.perf_counter() - started, 2),
            "states_by_depth": {
                str(depth): sum(
                    outcome.shell_sizes.get(depth, 0) for outcome in outcomes
                )
                for depth in sorted(
                    {depth for outcome in outcomes for depth in outcome.shell_sizes}
                )
            },
            "certified_depth_histogram": {
                str(depth): count
                for depth, count in sorted(
                    Counter(outcome.certified_depth for outcome in outcomes).items()
                )
            },
            "stop_reasons": {
                str(reason): count
                for reason, count in sorted(
                    Counter(outcome.stop_reason for outcome in outcomes).items(),
                    key=lambda item: str(item[0]),
                )
            },
            "rejections": dict(
                sorted(
                    sum(
                        (Counter(outcome.rejections) for outcome in outcomes),
                        Counter(),
                    ).items()
                )
            ),
            "roots": [outcome.to_dict() for outcome in outcomes],
        },
        "splits": split_metadata,
        "split_roots": {split: split_roots[split] for split in SPLIT_ORDER},
    }
    manifest["manifest_digest"] = _sha256_bytes(_canonical_json(manifest).encode())
    manifest_path = _write_json(manifest, corpus_dir / "manifest.json")
    return manifest, manifest_path


def load_corpus(
    corpus_dir: str | Path,
    *,
    splits: Sequence[str] = SPLIT_ORDER,
) -> tuple[dict[str, Any], list[BasisTaskRecord]]:
    """Load a corpus, checking the manifest digest, file digests and provenance."""
    root = Path(corpus_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stored_digest = manifest.get("manifest_digest")
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    # "hub" records where the payload is published, which is not part of what
    # the corpus is; a corpus keeps its identity whether or not it was uploaded.
    # Digesting it would mean publishing a corpus invalidated its own manifest.
    payload.pop("hub", None)
    expected = _sha256_bytes(_canonical_json(payload).encode())
    if stored_digest != expected:
        raise ValueError(f"manifest digest mismatch: {stored_digest!r} != {expected!r}")
    basis = load_basis_set(manifest["basis_set_id"])
    if basis.digest != manifest["basis_set_digest"]:
        raise ValueError("manifest references a different basis-set digest")

    records: list[BasisTaskRecord] = []
    for split in splits:
        metadata = manifest["splits"][split]
        name = metadata["path"]
        if Path(name).name != name:
            raise ValueError(f"unsafe split path for {split!r}: {name!r}")
        path = root / name
        if _sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"corpus file digest mismatch: {path}")
        split_records = read_basis_task_records(
            path,
            expected_basis_set_id=basis.basis_set_id,
            expected_basis_set_digest=basis.digest,
        )
        if len(split_records) != metadata["size"]:
            raise ValueError(f"corpus record count mismatch: {path}")
        for record in split_records:
            if record.split != split:
                raise ValueError(f"record {record.task_id} has the wrong split label")
        records.extend(split_records)
    return manifest, records


def verify_corpus(
    corpus_dir: str | Path,
    *,
    recertify: int = 0,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Re-execute every witness and optionally re-prove distances from scratch.

    Re-certification is independent of the generation run: it expands a fresh
    graph from the task's input through ``distance - 1`` and requires the target
    to be absent, which is exactly the lower-bound half of the distance claim.
    """
    manifest, records = load_corpus(corpus_dir)
    basis = load_basis_set(manifest["basis_set_id"])
    functions = basis.as_function_set()
    executor = Executor(functions)
    names_by_id = {function.function_id: function.name for function in basis.functions}
    by_name = functions.name_to_function

    witness_failures: list[str] = []
    for record in records:
        for function_id, name in zip(
            record.witness_function_ids, record.witness_function_names, strict=True
        ):
            if names_by_id.get(function_id) != name:
                witness_failures.append(f"{record.task_id}: unknown id {function_id}")
        if record.witness_length != record.metadata["certified_distance"]:
            witness_failures.append(f"{record.task_id}: witness length mismatch")
            continue
        spec = TrajectorySpec(
            record.input_value,
            [by_name[name] for name in record.witness_function_names],
        )
        execution = executor.execute_trajectory(spec)
        if not execution.success or execution.trajectory.output != record.output_value:
            witness_failures.append(f"{record.task_id}: witness did not reproduce")

    root_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        root_splits[record.metadata["root_digest"]].add(record.split)
    leaking_roots = sorted(
        digest for digest, splits in root_splits.items() if len(splits) > 1
    )

    recertified: list[dict[str, Any]] = []
    if recertify:
        rng = random.Random(seed)
        by_distance: defaultdict[int, list[BasisTaskRecord]] = defaultdict(list)
        for record in records:
            by_distance[record.metadata["certified_distance"]].append(record)
        per_distance = max(1, recertify // max(1, len(by_distance)))
        for distance in sorted(by_distance):
            for record in rng.sample(
                by_distance[distance],
                min(per_distance, len(by_distance[distance])),
            ):
                graph = TrajectoryGraph(functions)
                root_id = graph.add_root(record.input_value)
                started = time.perf_counter()
                expansion = graph.expand(root_id, distance - 1)
                # Answer equality, not graph.find: find resolves through the
                # search index, so asking it whether the target is reachable
                # sooner re-uses the very identity the distance was assigned
                # under, and cannot contradict it.  The claim being checked is
                # that no state a solver would be *graded correct* for is
                # reachable sooner.
                target_key = record.output_value.canonical_key()
                shorter = any(
                    graph.node(node_id).typed_list.canonical_key() == target_key
                    for node_id in expansion.node_depths
                )
                recertified.append(
                    {
                        "task_id": record.task_id,
                        "certified_distance": distance,
                        "expanded_depth": distance - 1,
                        "complete_expansion": expansion.complete,
                        "reachable_below_distance": bool(shorter),
                        "states_searched": expansion.num_reached_states,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                )

    failed_recertification = [
        row for row in recertified if row["reachable_below_distance"]
    ]
    return {
        "corpus_dir": str(corpus_dir),
        "records": len(records),
        "witness_failures": witness_failures,
        "roots_leaking_across_splits": leaking_roots,
        "distance_histogram": _distance_histogram(records),
        "recertified": recertified,
        "recertification_failures": failed_recertification,
        "ok": not witness_failures and not leaking_roots and not failed_recertification,
    }


def reference_curve(
    corpus_dir: str | Path,
    *,
    depths: Sequence[int],
    tasks_per_distance: int,
    budget: int,
    splits: Sequence[str] = ("validation", "test"),
    seed: int = DEFAULT_SEED,
    progress: bool = True,
) -> dict[str, Any]:
    """Measure the shipped BFS solver's solve rate and wall clock per depth.

    This is the number a learned policy has to beat.  BFS is run through
    ``create_bfs_solver`` so the curve measures the solver the repository
    actually ships, including its budget accounting.
    """
    from wandering_light.solver import create_bfs_solver

    manifest, records = load_corpus(corpus_dir)
    basis = load_basis_set(manifest["basis_set_id"])
    functions = basis.as_function_set()
    rng = random.Random(seed)
    by_distance: defaultdict[int, list[BasisTaskRecord]] = defaultdict(list)
    for record in records:
        if record.split in splits:
            by_distance[record.metadata["certified_distance"]].append(record)
    sample: list[BasisTaskRecord] = []
    for distance in sorted(by_distance):
        pool = by_distance[distance]
        sample.extend(rng.sample(pool, min(tasks_per_distance, len(pool))))

    rows: list[dict[str, Any]] = []
    for depth in depths:
        solver = create_bfs_solver(
            budget=budget, max_depth=depth, track_function_usage=False
        )
        for record in sample:
            started = time.perf_counter()
            result = solver.solve(record.input_value, record.output_value, functions)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "depth": depth,
                    "task_id": record.task_id,
                    "certified_distance": record.metadata["certified_distance"],
                    "input_type": record.metadata["input_type"],
                    "success": bool(result.success),
                    "solution_length": (
                        len(result.trajectory.function_defs)
                        if result.success and result.trajectory is not None
                        else None
                    ),
                    "seconds": elapsed,
                }
            )
        if progress:
            depth_rows = [row for row in rows if row["depth"] == depth]
            solved = sum(row["success"] for row in depth_rows)
            print(
                f"[bfs depth {depth}] solved {solved}/{len(depth_rows)} "
                f"total {sum(row['seconds'] for row in depth_rows):.1f}s",
                flush=True,
            )

    curve: list[dict[str, Any]] = []
    for depth in depths:
        depth_rows = [row for row in rows if row["depth"] == depth]
        by_distance_rows: dict[str, Any] = {}
        for distance in sorted({row["certified_distance"] for row in depth_rows}):
            subset = [
                row for row in depth_rows if row["certified_distance"] == distance
            ]
            by_distance_rows[str(distance)] = {
                "tasks": len(subset),
                "solved": sum(row["success"] for row in subset),
                "solve_rate": sum(row["success"] for row in subset) / len(subset),
                "mean_ms": 1000 * statistics.fmean(row["seconds"] for row in subset),
                "max_ms": 1000 * max(row["seconds"] for row in subset),
            }
        seconds = [row["seconds"] for row in depth_rows]
        curve.append(
            {
                "depth": depth,
                "tasks": len(depth_rows),
                "solved": sum(row["success"] for row in depth_rows),
                "solve_rate": sum(row["success"] for row in depth_rows)
                / len(depth_rows),
                "mean_ms_per_task": 1000 * statistics.fmean(seconds),
                "median_ms_per_task": 1000 * statistics.median(seconds),
                "total_seconds": sum(seconds),
                "by_certified_distance": by_distance_rows,
            }
        )
    return {
        "corpus_dir": str(corpus_dir),
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "solver": "wandering_light.solver.create_bfs_solver",
        "budget": budget,
        "splits": list(splits),
        "seed": seed,
        "tasks_per_distance": tasks_per_distance,
        "sampled_tasks": len(sample),
        "sample_distance_histogram": {
            str(distance): sum(
                1
                for record in sample
                if record.metadata["certified_distance"] == distance
            )
            for distance in sorted(
                {record.metadata["certified_distance"] for record in sample}
            )
        },
        "curve": curve,
    }


def plot_summary(
    corpus_dir: str | Path,
    *,
    plot_path: str | Path,
    curve_summary_path: str | Path | None = None,
) -> Path:
    """Render the corpus distance profile and the BFS reference curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, muted = "#0b0b0b", "#52514e"
    manifest = json.loads(
        (Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    curve: dict[str, Any] | None = None
    if curve_summary_path is not None and Path(curve_summary_path).exists():
        curve = json.loads(Path(curve_summary_path).read_text(encoding="utf-8"))

    exhaustive: Counter[int] = Counter()
    extended: Counter[int] = Counter()
    for split in SPLIT_ORDER:
        summary = manifest["splits"][split]
        for distance, count in summary["by_certified_distance"].items():
            exhaustive[int(distance)] += count
        for distance, count in summary.get("by_distance_certification", {}).items():
            extended[int(distance)] += count.get(CERTIFICATION_FRONTIER, 0)
    distances = sorted(exhaustive)
    complete_counts = [exhaustive[d] - extended.get(d, 0) for d in distances]
    extended_counts = [extended.get(d, 0) for d in distances]

    panels = 3 if curve else 1
    figure, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 4.1))
    axes = axes if panels > 1 else [axes]
    figure.patch.set_facecolor("#fcfcfb")

    axis = axes[0]
    axis.bar(distances, complete_counts, color=blue, label="exhaustive expansion")
    axis.bar(
        distances,
        extended_counts,
        bottom=complete_counts,
        color=orange,
        label="frontier extension",
    )
    for distance, total in zip(
        distances,
        [
            count + extra
            for count, extra in zip(complete_counts, extended_counts, strict=True)
        ],
        strict=True,
    ):
        axis.annotate(
            f"{total:,}",
            (distance, total),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color=muted,
        )
    totals = [c + e for c, e in zip(complete_counts, extended_counts, strict=True)]
    axis.set_ylim(0, max(totals) * 1.22 if totals else 1)
    axis.set_xticks(distances)
    axis.set_title("Certified distance", color=ink, fontsize=11)
    axis.set_xlabel("certified distance", color=muted, fontsize=9)
    axis.set_ylabel("tasks", color=muted, fontsize=9)
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    if curve:
        rows = curve["curve"]
        depths = [row["depth"] for row in rows]
        buckets = (
            ("distance 1-4", blue, lambda d: d <= 4),
            ("distance 5-6", orange, lambda d: 5 <= d <= 6),
            ("distance 7-8", aqua, lambda d: d >= 7),
        )
        axis = axes[1]
        for offset, (label, color, predicate) in enumerate(buckets):
            rates: list[float] = []
            for row in rows:
                subset = [
                    stats
                    for distance, stats in row["by_certified_distance"].items()
                    if predicate(int(distance))
                ]
                solved = sum(stats["solved"] for stats in subset)
                tasks = sum(stats["tasks"] for stats in subset)
                rates.append(100 * solved / tasks if tasks else 0.0)
            axis.plot(
                depths,
                rates,
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                label=label,
            )
            # Stagger the direct labels: the deeper buckets sit on 0% together.
            axis.annotate(
                label,
                (depths[-1], rates[-1]),
                textcoords="offset points",
                xytext=(-4, 8 - 11 * offset),
                ha="right",
                fontsize=8,
                color=color,
            )
        axis.set_ylim(-14, 112)
        axis.set_xticks(depths)
        axis.set_title("BFS solve rate by search depth", color=ink, fontsize=11)
        axis.set_xlabel("BFS max depth", color=muted, fontsize=9)
        axis.set_ylabel("solved (%)", color=muted, fontsize=9)
        axis.legend(frameon=False, fontsize=8, loc="center left")

        axis = axes[2]
        milliseconds = [row["mean_ms_per_task"] for row in rows]
        axis.plot(
            depths, milliseconds, color=blue, linewidth=2, marker="o", markersize=5
        )
        for depth, value in zip(depths, milliseconds, strict=True):
            axis.annotate(
                f"{value:,.0f}",
                (depth, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=muted,
            )
        axis.set_yscale("log")
        axis.set_ylim(min(milliseconds) / 2, max(milliseconds) * 4)
        axis.set_xticks(depths)
        axis.set_title("BFS cost per task", color=ink, fontsize=11)
        axis.set_xlabel("BFS max depth", color=muted, fontsize=9)
        axis.set_ylabel("mean ms per task (log)", color=muted, fontsize=9)

    for axis in axes:
        axis.set_facecolor("#fcfcfb")
        axis.grid(axis="y", color="#d8d7d2", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#d8d7d2")
        axis.spines["bottom"].set_color("#d8d7d2")
        axis.tick_params(colors=muted, labelsize=8)

    figure.tight_layout()
    output = Path(plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def _parse_depths(value: str) -> list[int]:
    depths = [int(part) for part in value.split(",") if part.strip()]
    if not depths or any(depth < 1 for depth in depths):
        raise argparse.ArgumentTypeError("depths must be positive integers")
    return depths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", choices=("run", "verify", "reference-curve", "plot"), default="run"
    )
    parser.add_argument("--basis-set-id", default="default")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roots", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--deep-roots", type=int, default=0)
    parser.add_argument("--deep-max-depth", type=int, default=7)
    parser.add_argument("--min-distance", type=int, default=1)
    parser.add_argument("--tasks-per-distance", type=int, default=40)
    parser.add_argument("--frontier-sample", type=int, default=24)
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES)
    parser.add_argument("--max-transitions", type=int, default=DEFAULT_MAX_TRANSITIONS)
    parser.add_argument("--deep-max-states", type=int, default=DEFAULT_DEEP_MAX_STATES)
    parser.add_argument(
        "--deep-max-transitions", type=int, default=DEFAULT_DEEP_MAX_TRANSITIONS
    )
    parser.add_argument(
        "--deep-root-types",
        default="",
        help=(
            "comma-separated input types eligible for the deeper expansion "
            "budget, e.g. builtins.list,builtins.set; empty means every type"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-constant-outputs", action="store_true")
    parser.add_argument("--skip-witness-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--recertify", type=int, default=0)
    parser.add_argument("--curve-depths", type=_parse_depths, default=[1, 2, 3, 4, 5])
    parser.add_argument("--curve-tasks-per-distance", type=int, default=12)
    parser.add_argument("--curve-budget", type=int, default=5_000_000)
    parser.add_argument("--curve-splits", default="validation,test")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--curve-summary-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "run":
        manifest, manifest_path = generate_corpus(
            basis_set_id=args.basis_set_id,
            output_dir=args.output_dir,
            roots=args.roots,
            max_depth=args.max_depth,
            deep_roots=args.deep_roots,
            deep_max_depth=args.deep_max_depth,
            min_distance=args.min_distance,
            tasks_per_distance=args.tasks_per_distance,
            frontier_sample=args.frontier_sample,
            max_states=args.max_states,
            max_transitions=args.max_transitions,
            deep_max_states=args.deep_max_states,
            deep_max_transitions=args.deep_max_transitions,
            deep_root_types=tuple(
                part.strip() for part in args.deep_root_types.split(",") if part.strip()
            ),
            seed=args.seed,
            allow_constant_outputs=args.allow_constant_outputs,
            verify_witnesses=not args.skip_witness_verification,
            overwrite=args.overwrite,
            progress=not args.quiet,
        )
        print(f"wrote {manifest_path}")
        for split in SPLIT_ORDER:
            summary = manifest["splits"][split]
            print(
                f"  {split}: {summary['size']} tasks, distances "
                f"{summary['by_certified_distance']}"
            )
        return 0
    if args.mode == "verify":
        result = verify_corpus(
            args.output_dir, recertify=args.recertify, seed=args.seed
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.summary_path is not None:
            _write_json(result, args.summary_path)
        return 0 if result["ok"] else 1
    if args.mode == "plot":
        if args.plot_path is None:
            raise SystemExit("--plot-path is required for --mode plot")
        plot_path = plot_summary(
            args.output_dir,
            plot_path=args.plot_path,
            curve_summary_path=args.curve_summary_path,
        )
        print(f"wrote {plot_path}")
        return 0
    result = reference_curve(
        args.output_dir,
        depths=args.curve_depths,
        tasks_per_distance=args.curve_tasks_per_distance,
        budget=args.curve_budget,
        splits=tuple(part for part in args.curve_splits.split(",") if part),
        seed=args.seed,
        progress=not args.quiet,
    )
    print(json.dumps(result["curve"], indent=2, sort_keys=True))
    if args.summary_path is not None:
        _write_json(result, args.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
