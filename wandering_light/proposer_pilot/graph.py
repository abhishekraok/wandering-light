"""Graph of TypedList states connected by FunctionDef edges.

States are deduplicated structurally, so two paths that land on the same value
collapse to one node. Edges are labeled with the FunctionDef that produced
them; multiple edges between the same pair are allowed when distinct functions
yield the same result.
"""

import math
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from wandering_light.common_functions import basic_fns
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.trajectory import Trajectory, TrajectorySpec
from wandering_light.typed_list import TypedList

type StateKey = tuple[str, tuple[object, ...]]
type FunctionFingerprint = tuple[tuple[str, str, str, str], ...]


def _type_name(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _freeze(value: object) -> object:
    """Return a deterministic, hashable representation of a builtin value."""
    value_type = _type_name(type(value))
    if value is None:
        return (value_type, None)
    if isinstance(value, bool | int | str | bytes):
        return (value_type, value)
    if isinstance(value, float):
        # Signed zero is kept: ``-0.0 == 0.0`` in Python, but ``float_to_str``,
        # ``f_fraction`` and ``f_sin`` distinguish them, so merging the two
        # states would leave one set of successors unexplored and inflate the
        # certified distance of anything reachable only through them.  NaNs stay
        # collapsed -- no basis function observes their sign or payload.
        normalized = "nan" if math.isnan(value) else value.hex()
        return (value_type, normalized)
    if isinstance(value, bytearray):
        return (value_type, bytes(value))
    if isinstance(value, complex):
        return (value_type, _freeze(value.real), _freeze(value.imag))
    if isinstance(value, range):
        return (value_type, value.start, value.stop, value.step)
    if isinstance(value, list | tuple):
        return (value_type, tuple(_freeze(item) for item in value))
    if isinstance(value, set | frozenset):
        return (value_type, frozenset(_freeze(item) for item in value))
    if isinstance(value, dict):
        return (
            value_type,
            frozenset((_freeze(key), _freeze(item)) for key, item in value.items()),
        )
    if hasattr(value, "model_dump"):
        return (value_type, _freeze(value.model_dump()))
    if hasattr(value, "dict"):
        return (value_type, _freeze(value.dict()))
    try:
        hash(value)
    except TypeError:
        return (value_type, repr(value))
    return (value_type, value)


def _state_key(tl: TypedList) -> StateKey:
    return (_type_name(tl.item_type), tuple(_freeze(item) for item in tl.items))


def _fingerprint(functions: FunctionDefSet) -> FunctionFingerprint:
    return tuple((fn.name, fn.input_type, fn.output_type, fn.code) for fn in functions)


@dataclass
class Node:
    id: int
    typed_list: TypedList
    out_edges: list[tuple[FunctionDef, int]] = field(default_factory=list)
    in_edges: list[tuple[FunctionDef, int]] = field(default_factory=list)


@dataclass
class Task:
    src_id: int
    dst_id: int
    trajectory: Trajectory
    proposed_num_steps: int | None = None
    verified_shortest_num_steps: int | None = None
    certification_depth: int | None = None

    @property
    def num_steps(self) -> int:
        """Length of the witness trajectory, retained for compatibility."""
        return len(self.trajectory.function_defs)

    @property
    def shortest_path_is_certified(self) -> bool:
        return self.verified_shortest_num_steps is not None


@dataclass(frozen=True)
class ExpansionResult:
    """Evidence produced by one bounded breadth-first expansion.

    ``certified_depth`` is the largest path length for which all shallower
    reachable states were fully expanded. A task at or below that depth has a
    globally shortest witness relative to this function set.
    """

    root_id: int
    max_depth: int
    node_depths: Mapping[int, int] = field(repr=False, compare=False)
    attempted_transitions: int = 0
    failed_transitions: int = 0
    skipped_self_loops: int = 0
    edges_added: int = 0
    new_nodes: int = 0
    certified_depth: int = 0
    stop_reason: str | None = None
    _graph_id: int = field(default=0, repr=False, compare=False)
    _function_fingerprint: FunctionFingerprint = field(
        default=(), repr=False, compare=False
    )
    _parents: Mapping[int, tuple[int, FunctionDef]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def complete(self) -> bool:
        """Whether expansion was exhaustive through ``max_depth``."""
        return self.stop_reason is None

    @property
    def num_reached_states(self) -> int:
        return len(self.node_depths)


class TrajectoryGraph:
    def __init__(self, functions: FunctionDefSet | None = None):
        self.functions = functions if functions is not None else basic_fns
        # The graph may retain observational edges proposed outside the
        # expansion palette; certified paths still restrict those edges below.
        self.executor = Executor(self.functions, enforce_membership=False)
        self._nodes: dict[int, Node] = {}
        self._state_index: dict[StateKey, int] = {}
        self._roots: list[int] = []
        self._next_id = 0

    def add_root(self, tl: TypedList) -> int:
        node_id = self._get_or_create(tl)
        if node_id not in self._roots:
            self._roots.append(node_id)
        return node_id

    def apply(self, parent_id: int, fn: FunctionDef) -> int:
        if parent_id not in self._nodes:
            raise KeyError(f"unknown parent_id {parent_id}")
        parent = self._nodes[parent_id]
        result = self.executor.execute(fn, parent.typed_list)
        child_id, _ = self._record_transition(parent_id, fn, result)
        return child_id

    def apply_by_name(self, parent_id: int, fn_name: str) -> int:
        fn = self.functions.name_to_function.get(fn_name)
        if fn is None:
            raise KeyError(f"unknown function name {fn_name!r}")
        return self.apply(parent_id, fn)

    def _get_or_create(self, tl: TypedList) -> int:
        key = _state_key(tl)
        if key in self._state_index:
            return self._state_index[key]
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = Node(id=node_id, typed_list=tl)
        self._state_index[key] = node_id
        return node_id

    def _record_transition(
        self, parent_id: int, fn: FunctionDef, result: TypedList
    ) -> tuple[int, bool]:
        parent = self._nodes[parent_id]
        child_id = self._get_or_create(result)
        edge_exists = any(c == child_id and f == fn for f, c in parent.out_edges)
        if not edge_exists:
            parent.out_edges.append((fn, child_id))
            self._nodes[child_id].in_edges.append((fn, parent_id))
        return child_id, not edge_exists

    def expand(
        self,
        root_id: int,
        max_depth: int,
        *,
        max_states: int | None = None,
        max_transitions: int | None = None,
        skip_self_loops: bool = True,
    ) -> ExpansionResult:
        """Exhaustively apply every type-compatible function breadth-first.

        Runtime failures are treated as absent transitions. ``max_states``
        counts states reachable from this root, including the root;
        ``max_transitions`` counts attempted function executions. Hitting either
        budget stops the search and is recorded in the returned evidence.
        """
        if root_id not in self._nodes:
            raise KeyError(f"unknown root_id {root_id}")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if max_states is not None and max_states < 1:
            raise ValueError("max_states must be at least 1")
        if max_transitions is not None and max_transitions < 0:
            raise ValueError("max_transitions must be non-negative")

        node_depths = {root_id: 0}
        parents: dict[int, tuple[int, FunctionDef]] = {}
        queue: deque[int] = deque([root_id])
        attempted = 0
        failed = 0
        skipped_self_loops = 0
        edges_added = 0
        nodes_before = self.num_nodes()
        certified_depth = 0
        stop_reason: str | None = None

        while queue:
            layer_depth = node_depths[queue[0]]
            if layer_depth >= max_depth:
                break

            layer: list[int] = []
            while queue and node_depths[queue[0]] == layer_depth:
                layer.append(queue.popleft())

            layer_complete = True
            for parent_id in layer:
                parent = self._nodes[parent_id]
                parent_key = _state_key(parent.typed_list)
                applicable = (
                    fn
                    for fn in self.functions
                    if fn.input_type_cls() is parent.typed_list.item_type
                )
                for fn in applicable:
                    if max_transitions is not None and attempted >= max_transitions:
                        stop_reason = "max_transitions"
                        layer_complete = False
                        break

                    attempted += 1
                    try:
                        result = self.executor.execute(fn, parent.typed_list)
                    except Exception:
                        failed += 1
                        continue

                    result_key = _state_key(result)
                    if skip_self_loops and result_key == parent_key:
                        skipped_self_loops += 1
                        continue

                    child_id = self._state_index.get(result_key)
                    is_new_reachable = child_id is None or child_id not in node_depths
                    if (
                        is_new_reachable
                        and max_states is not None
                        and len(node_depths) >= max_states
                    ):
                        stop_reason = "max_states"
                        layer_complete = False
                        break

                    child_id, added = self._record_transition(parent_id, fn, result)
                    edges_added += int(added)
                    if child_id not in node_depths:
                        node_depths[child_id] = layer_depth + 1
                        parents[child_id] = (parent_id, fn)
                        queue.append(child_id)

                if not layer_complete:
                    break

            if not layer_complete:
                break
            certified_depth = layer_depth + 1

        if stop_reason is None:
            certified_depth = max_depth

        return ExpansionResult(
            root_id=root_id,
            max_depth=max_depth,
            node_depths=MappingProxyType(node_depths.copy()),
            attempted_transitions=attempted,
            failed_transitions=failed,
            skipped_self_loops=skipped_self_loops,
            edges_added=edges_added,
            new_nodes=self.num_nodes() - nodes_before,
            certified_depth=certified_depth,
            stop_reason=stop_reason,
            _graph_id=id(self),
            _function_fingerprint=_fingerprint(self.functions),
            _parents=MappingProxyType(parents.copy()),
        )

    @property
    def roots(self) -> list[int]:
        return list(self._roots)

    def node(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def nodes(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return sum(len(n.out_edges) for n in self._nodes.values())

    def find(self, tl: TypedList) -> int | None:
        return self._state_index.get(_state_key(tl))

    def shortest_path(self, src_id: int, dst_id: int) -> list[FunctionDef] | None:
        if src_id not in self._nodes or dst_id not in self._nodes:
            raise KeyError("unknown node id")
        if src_id == dst_id:
            return []
        parent_of: dict[int, tuple[int, FunctionDef]] = {}
        visited = {src_id}
        queue: deque[int] = deque([src_id])
        found = False
        while queue:
            cur = queue.popleft()
            if cur == dst_id:
                found = True
                break
            for fn, child in self._nodes[cur].out_edges:
                if child in visited:
                    continue
                visited.add(child)
                parent_of[child] = (cur, fn)
                queue.append(child)
        if not found and dst_id not in parent_of:
            return None
        path: list[FunctionDef] = []
        cur = dst_id
        while cur != src_id:
            prev, fn = parent_of[cur]
            path.append(fn)
            cur = prev
        path.reverse()
        return path

    def tasks(
        self,
        src_ids: list[int] | None = None,
        min_steps: int = 1,
        max_steps: int | None = None,
    ) -> Iterator[Task]:
        srcs = src_ids if src_ids is not None else self._roots
        for src_id in srcs:
            parent_of: dict[int, tuple[int, FunctionDef]] = {}
            depth: dict[int, int] = {src_id: 0}
            queue: deque[int] = deque([src_id])
            while queue:
                cur = queue.popleft()
                d = depth[cur]
                if max_steps is not None and d >= max_steps:
                    continue
                for fn, child in self._nodes[cur].out_edges:
                    if child in depth:
                        continue
                    depth[child] = d + 1
                    parent_of[child] = (cur, fn)
                    queue.append(child)
            for dst_id, d in depth.items():
                if dst_id == src_id or d < min_steps:
                    continue
                if max_steps is not None and d > max_steps:
                    continue
                path: list[FunctionDef] = []
                cur = dst_id
                while cur != src_id:
                    prev, fn = parent_of[cur]
                    path.append(fn)
                    cur = prev
                path.reverse()
                src_tl = self._nodes[src_id].typed_list
                dst_tl = self._nodes[dst_id].typed_list
                spec = TrajectorySpec(src_tl, FunctionDefList(path))
                yield Task(
                    src_id=src_id,
                    dst_id=dst_id,
                    trajectory=Trajectory(spec, dst_tl),
                )

    def tasks_from_expansion(
        self,
        expansion: ExpansionResult,
        *,
        min_steps: int = 1,
        max_steps: int | None = None,
        require_certified: bool = True,
    ) -> Iterator[Task]:
        """Yield tasks reached by an expansion, with certification metadata."""
        self._validate_expansion(expansion)
        for dst_id in expansion.node_depths:
            if dst_id == expansion.root_id:
                continue
            path = self._expansion_path(expansion, dst_id)
            num_steps = len(path)
            if num_steps < min_steps:
                continue
            if max_steps is not None and num_steps > max_steps:
                continue
            is_certified = num_steps <= expansion.certified_depth
            if require_certified and not is_certified:
                continue
            src_tl = self._nodes[expansion.root_id].typed_list
            dst_tl = self._nodes[dst_id].typed_list
            spec = TrajectorySpec(src_tl, FunctionDefList(path))
            yield Task(
                src_id=expansion.root_id,
                dst_id=dst_id,
                trajectory=Trajectory(spec, dst_tl),
                verified_shortest_num_steps=num_steps if is_certified else None,
                certification_depth=expansion.certified_depth,
            )

    def task_from_proposal(
        self, proposal: Trajectory, expansion: ExpansionResult
    ) -> Task:
        """Compare a proposed trajectory length with a certified shortest path."""
        self._validate_expansion(expansion)
        if _state_key(proposal.input) != _state_key(
            self._nodes[expansion.root_id].typed_list
        ):
            raise ValueError("proposal input does not match expansion root")
        dst_id = self.find(proposal.output)
        if dst_id is None or dst_id not in expansion.node_depths:
            raise ValueError("proposal output was not reached by this expansion")
        path = self._expansion_path(expansion, dst_id)
        num_steps = len(path)
        is_certified = num_steps <= expansion.certified_depth
        spec = TrajectorySpec(proposal.input, FunctionDefList(path))
        return Task(
            src_id=expansion.root_id,
            dst_id=dst_id,
            trajectory=Trajectory(spec, proposal.output),
            proposed_num_steps=len(proposal.function_defs),
            verified_shortest_num_steps=num_steps if is_certified else None,
            certification_depth=expansion.certified_depth,
        )

    def _validate_expansion(self, expansion: ExpansionResult) -> None:
        if expansion._graph_id != id(self):
            raise ValueError("expansion belongs to a different graph")
        if expansion._function_fingerprint != _fingerprint(self.functions):
            raise ValueError("function set changed after expansion")

    @staticmethod
    def _expansion_path(expansion: ExpansionResult, dst_id: int) -> list[FunctionDef]:
        if dst_id not in expansion.node_depths:
            raise ValueError("destination was not reached by this expansion")
        path: list[FunctionDef] = []
        cur = dst_id
        while cur != expansion.root_id:
            parent = expansion._parents.get(cur)
            if parent is None:
                raise ValueError("expansion evidence is missing a parent transition")
            cur, fn = parent
            path.append(fn)
        path.reverse()
        return path

    def to_networkx(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        for node in self._nodes.values():
            g.add_node(node.id, typed_list=repr(node.typed_list))
        for node in self._nodes.values():
            for fn, child_id in node.out_edges:
                g.add_edge(node.id, child_id, fn=fn.name)
        return g
