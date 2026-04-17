from dataclasses import dataclass, field
from typing import Any

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef
from wandering_light.trajectory import Trajectory
from wandering_light.typed_list import TypedList

ROOT_ID = 0


@dataclass
class TrajectoryTree:
    """Mutable tree of TypedList nodes connected by FunctionDef edges.

    The root has `applied_fn_def=None`. Every other node represents a function
    applied to its parent's TypedList. This class is intentionally free of UI
    dependencies so it can be unit-tested in isolation.
    """

    nodes: dict[int, dict[str, Any]] = field(default_factory=dict)
    _next_id: int = 0

    @classmethod
    def with_root(cls, input_tl: TypedList) -> "TrajectoryTree":
        tree = cls()
        tree.nodes[ROOT_ID] = {
            "typed_list": input_tl,
            "error": None,
            "parent": None,
            "applied_fn_def": None,
            "children": [],
        }
        tree._next_id = ROOT_ID + 1
        return tree

    @classmethod
    def from_trajectory(
        cls, traj: Trajectory, executor: Executor
    ) -> "TrajectoryTree":
        tree = cls.with_root(traj.input)
        parent_id = ROOT_ID
        for fn in traj.function_defs:
            parent_id = tree.append_child(parent_id, fn, executor)
        return tree

    def append_child(
        self, parent_id: int, fn: FunctionDef, executor: Executor
    ) -> int:
        new_id = self._next_id
        self._next_id += 1
        self.nodes[new_id] = {
            "typed_list": None,
            "error": None,
            "parent": parent_id,
            "applied_fn_def": fn,
            "children": [],
        }
        self.nodes[parent_id]["children"].append(new_id)
        self._execute_node(new_id, executor)
        return new_id

    def replace_edge(
        self, node_id: int, new_fn: FunctionDef, executor: Executor
    ) -> list[int]:
        """Change the incoming edge's function, drop any descendants, recompute
        this node. Returns the ids of deleted descendants (useful for callers
        that need to clean up external state keyed by node id)."""
        if node_id == ROOT_ID:
            raise ValueError("root has no incoming edge")
        deleted = self.delete_descendants(node_id)
        self.nodes[node_id]["applied_fn_def"] = new_fn
        self._execute_node(node_id, executor)
        return deleted

    def delete_descendants(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        deleted: list[int] = []
        for child_id in list(node["children"]):
            deleted.extend(self.delete_descendants(child_id))
            deleted.append(child_id)
            self.nodes.pop(child_id, None)
        node["children"] = []
        return deleted

    def _execute_node(self, node_id: int, executor: Executor) -> None:
        node = self.nodes[node_id]
        parent = self.nodes[node["parent"]]
        fn: FunctionDef = node["applied_fn_def"]
        parent_tl = parent["typed_list"]
        if parent_tl is None:
            node["typed_list"] = None
            node["error"] = "parent has no output"
            return
        try:
            node["typed_list"] = executor.execute(fn, parent_tl)
            node["error"] = None
        except Exception as e:
            node["typed_list"] = None
            node["error"] = f"{type(e).__name__}: {e}"
