"""Domain operations behind the explorer API.

Free of FastAPI so the interesting parts -- what a state's successors are, how
an expansion looks, what a solver's attempt cost -- can be tested directly.

Everything the browser holds is addressed by a state's *wire* form (the JSON a
``TypedList`` serialises to). That is already canonical and already what every
artifact stores, so a state the UI is pointing at, a corpus record and a graph
node all name each other without a lookup table.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefSet
from wandering_light.proposer_pilot.graph import TrajectoryGraph
from wandering_light.solver import create_bfs_solver, create_random_solver
from wandering_light.state_io import parse_typed_list

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from wandering_light.typed_list import TypedList


@dataclass(frozen=True)
class StateView:
    """One state, in every form the browser needs at once."""

    id: str
    wire: str
    label: str
    type: str
    size: int

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def state_view(value: TypedList) -> StateView:
    wire = value.to_string()
    return StateView(
        id=hashlib.sha1(wire.encode("utf-8")).hexdigest()[:16],
        wire=wire,
        label=repr(value),
        type=value.item_type.__name__,
        size=len(value),
    )


def parse_state(text: str) -> TypedList:
    """Parse a state from a repr or the JSON wire format."""
    return parse_typed_list(text)


@dataclass(frozen=True)
class Step:
    """One applied function, and what it produced."""

    function: str
    ok: bool
    state: StateView | None = None
    error: str | None = None

    def dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "ok": self.ok,
            "state": self.state.dict() if self.state else None,
            "error": self.error,
        }


def run_trajectory(
    start: TypedList, function_names: Sequence[str], functions: FunctionDefSet
) -> list[Step]:
    """Apply functions in order, stopping at the first failure.

    Editing one edge in the middle invalidates everything after it, so the UI
    re-runs the whole list rather than patching; at these lengths that is
    cheaper than tracking what stayed valid.
    """
    executor = Executor(functions)
    steps: list[Step] = []
    current = start
    for name in function_names:
        function = functions.name_to_function.get(name)
        if function is None:
            steps.append(Step(function=name, ok=False, error="not in this basis"))
            break
        try:
            current = executor.execute(function, current)
        except Exception as error:
            steps.append(
                Step(function=name, ok=False, error=f"{type(error).__name__}: {error}")
            )
            break
        steps.append(Step(function=name, ok=True, state=state_view(current)))
    return steps


@dataclass(frozen=True)
class Successor:
    """What one basis function does to one state, including doing nothing."""

    function: str
    ok: bool
    state: StateView | None = None
    error: str | None = None
    self_loop: bool = False

    def dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "ok": self.ok,
            "state": self.state.dict() if self.state else None,
            "error": self.error,
            "self_loop": self.self_loop,
        }


def successors(value: TypedList, functions: FunctionDefSet) -> list[Successor]:
    """Every type-compatible function applied to one state.

    Failures and self-loops are returned rather than dropped: which actions are
    unavailable, and which do nothing here, is exactly the question someone
    asks when judging whether the basis can express what they want.
    """
    executor = Executor(functions)
    results: list[Successor] = []
    for function in functions:
        if function.input_type_cls() is not value.item_type:
            continue
        try:
            produced = executor.execute(function, value)
        except Exception as error:
            results.append(
                Successor(
                    function=function.name,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            continue
        results.append(
            Successor(
                function=function.name,
                ok=True,
                state=state_view(produced),
                self_loop=produced.search_key() == value.search_key(),
            )
        )
    return results


@dataclass
class ExpansionView:
    """A drawable expansion: nodes with depths, edges with function labels."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    types: dict[str, int] = field(default_factory=dict)
    function_edges: dict[str, int] = field(default_factory=dict)
    idle_functions: list[str] = field(default_factory=list)

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def expand(
    root: TypedList,
    functions: FunctionDefSet,
    *,
    max_depth: int = 2,
    max_states: int | None = 200,
    max_transitions: int | None = 20_000,
) -> ExpansionView:
    """Breadth-first expansion from one root, shaped for drawing.

    Only nodes the expansion actually reached are returned, so the depth of a
    node is its certified distance from the root whenever the expansion ran to
    completion.
    """
    graph = TrajectoryGraph(functions=functions)
    root_id = graph.add_root(root)
    started = time.perf_counter()
    result = graph.expand(
        root_id,
        max_depth=max_depth,
        max_states=max_states,
        max_transitions=max_transitions,
    )
    elapsed = time.perf_counter() - started

    depths = dict(result.node_depths)
    view = ExpansionView()
    for node_id, depth in sorted(depths.items(), key=lambda item: (item[1], item[0])):
        value = graph.node(node_id).typed_list
        node = state_view(value).dict()
        node["depth"] = depth
        node["node_id"] = node_id
        view.nodes.append(node)
        view.types[value.item_type.__name__] = (
            view.types.get(value.item_type.__name__, 0) + 1
        )

    seen_edges: set[tuple[int, str, int]] = set()
    for node_id, node_depth in depths.items():
        for function, child_id in graph.node(node_id).out_edges:
            if child_id not in depths:
                continue
            key = (node_id, function.name, child_id)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            view.edges.append(
                {
                    "source": node_id,
                    "target": child_id,
                    "function": function.name,
                    "depth": node_depth,
                }
            )
            view.function_edges[function.name] = (
                view.function_edges.get(function.name, 0) + 1
            )

    view.idle_functions = sorted(
        function.name
        for function in functions
        if function.name not in view.function_edges
    )
    view.stats = {
        "root_id": root_id,
        "nodes": len(view.nodes),
        "edges": len(view.edges),
        "certified_depth": result.certified_depth,
        "attempted_transitions": result.attempted_transitions,
        "failed_transitions": result.failed_transitions,
        "skipped_self_loops": result.skipped_self_loops,
        "complete": result.complete,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": round(elapsed, 4),
        "by_depth": _depth_histogram(depths),
    }
    return view


def _depth_histogram(depths: dict[int, int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for depth in depths.values():
        histogram[str(depth)] = histogram.get(str(depth), 0) + 1
    return histogram


@dataclass(frozen=True)
class SolveAttempt:
    """What a solver did with a task, and what it cost."""

    solver: str
    success: bool
    functions: list[str]
    output: StateView | None
    error: str | None
    elapsed_seconds: float

    def dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "success": self.success,
            "functions": self.functions,
            "output": self.output.dict() if self.output else None,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


def solve(
    start: TypedList,
    target: TypedList,
    functions: FunctionDefSet,
    *,
    solver: str = "bfs",
    budget: int = 2000,
    max_depth: int = 3,
) -> SolveAttempt:
    """Run a search solver and report the attempt, timing included.

    The elapsed time is the point as much as the answer: a task BFS misses
    costs it a full sweep to miss, and that is the difficulty a learned solver
    is being asked to shortcut.
    """
    if solver == "bfs":
        instance = create_bfs_solver(budget=budget, max_depth=max_depth)
    elif solver == "random":
        instance = create_random_solver(budget=budget, path_length=max_depth)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    started = time.perf_counter()
    result = instance.solve(start, target, functions)
    elapsed = time.perf_counter() - started
    trajectory = result.trajectory
    return SolveAttempt(
        solver=f"{solver} (budget {budget}, depth {max_depth})",
        success=result.success,
        functions=[fn.name for fn in trajectory.function_defs] if trajectory else [],
        output=state_view(trajectory.output) if trajectory else None,
        error=result.error_msg,
        elapsed_seconds=round(elapsed, 4),
    )


def palette(functions: FunctionDefSet, names: Iterable[str] | None) -> FunctionDefSet:
    """Restrict a basis to the named functions, or keep all of them."""
    if names is None:
        return functions
    selected = [functions.name_to_function[name] for name in names]
    return FunctionDefSet(selected)
