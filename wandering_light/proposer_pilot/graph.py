"""Graph of TypedList states connected by FunctionDef edges.

States are deduplicated by canonical TypedList serialization, so two paths
that land on the same value collapse to one node. Edges are labeled with
the FunctionDef that produced them; multiple edges between the same pair
are allowed when distinct functions yield the same result.
"""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from wandering_light.common_functions import basic_fns
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.typed_list import TypedList


def _state_key(tl: TypedList) -> str:
    return tl.to_string()


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
    src_tl: TypedList
    dst_tl: TypedList
    gt_path: list[FunctionDef]

    @property
    def num_steps(self) -> int:
        return len(self.gt_path)


class TrajectoryGraph:
    def __init__(self, functions: FunctionDefSet | None = None):
        self.functions = functions if functions is not None else basic_fns
        self.executor = Executor(self.functions)
        self._nodes: dict[int, Node] = {}
        self._state_index: dict[str, int] = {}
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
        child_id = self._get_or_create(result)
        if not any(c == child_id and f == fn for f, c in parent.out_edges):
            parent.out_edges.append((fn, child_id))
            self._nodes[child_id].in_edges.append((fn, parent_id))
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
                yield Task(
                    src_id=src_id,
                    dst_id=dst_id,
                    src_tl=self._nodes[src_id].typed_list,
                    dst_tl=self._nodes[dst_id].typed_list,
                    gt_path=path,
                )

    def to_networkx(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        for node in self._nodes.values():
            g.add_node(node.id, typed_list=repr(node.typed_list))
        for node in self._nodes.values():
            for fn, child_id in node.out_edges:
                g.add_edge(node.id, child_id, fn=fn.name)
        return g
