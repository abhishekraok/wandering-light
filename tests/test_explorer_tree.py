import pytest

from wandering_light.evals.explorer_tree import ROOT_ID, TrajectoryTree
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList

inc = FunctionDef(
    name="inc",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 1",
)
double = FunctionDef(
    name="double",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x * 2",
)
to_str = FunctionDef(
    name="to_str",
    input_type="builtins.int",
    output_type="builtins.str",
    code="return str(x)",
)
str_len = FunctionDef(
    name="str_len",
    input_type="builtins.str",
    output_type="builtins.int",
    code="return len(x)",
)
div_zero = FunctionDef(
    name="div_zero",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x // 0",
)

FUNCTIONS = FunctionDefSet([inc, double, to_str, str_len, div_zero])


@pytest.fixture
def executor() -> Executor:
    return Executor(FUNCTIONS)


def _build_chain(
    executor: Executor, input_tl: TypedList, fns: list[FunctionDef]
) -> TrajectoryTree:
    tree = TrajectoryTree.with_root(input_tl)
    parent_id = ROOT_ID
    for fn in fns:
        parent_id = tree.append_child(parent_id, fn, executor)
    return tree


def _chain_ids(tree: TrajectoryTree) -> list[int]:
    """Return node ids along the (linear) chain from root to the deepest leaf."""
    ids = [ROOT_ID]
    while tree.nodes[ids[-1]]["children"]:
        ids.append(tree.nodes[ids[-1]]["children"][0])
    return ids


def test_from_trajectory_builds_chain_with_computed_outputs(executor):
    spec = TrajectorySpec(
        input_list=TypedList([1, 2, 3]),
        function_defs=FunctionDefList([inc, double, to_str]),
    )
    traj = executor.execute_trajectory(spec).trajectory

    tree = TrajectoryTree.from_trajectory(traj, executor)

    ids = _chain_ids(tree)
    assert len(ids) == 4
    assert tree.nodes[ids[0]]["applied_fn_def"] is None
    assert tree.nodes[ids[0]]["typed_list"].items == [1, 2, 3]
    assert tree.nodes[ids[1]]["applied_fn_def"].name == "inc"
    assert tree.nodes[ids[1]]["typed_list"].items == [2, 3, 4]
    assert tree.nodes[ids[2]]["applied_fn_def"].name == "double"
    assert tree.nodes[ids[2]]["typed_list"].items == [4, 6, 8]
    assert tree.nodes[ids[3]]["applied_fn_def"].name == "to_str"
    assert tree.nodes[ids[3]]["typed_list"].items == ["4", "6", "8"]
    assert all(tree.nodes[i]["error"] is None for i in ids)


def test_append_child_extends_leaf(executor):
    tree = TrajectoryTree.with_root(TypedList([10]))

    child_id = tree.append_child(ROOT_ID, inc, executor)

    assert tree.nodes[ROOT_ID]["children"] == [child_id]
    assert tree.nodes[child_id]["typed_list"].items == [11]
    assert tree.nodes[child_id]["error"] is None


def test_replace_edge_prunes_descendants_and_recomputes(executor):
    tree = _build_chain(executor, TypedList([1, 2]), [inc, inc, inc])
    chain = _chain_ids(tree)
    edited_id = chain[1]  # first `inc`
    descendants_before = chain[2:]

    deleted = tree.replace_edge(edited_id, double, executor)

    assert sorted(deleted) == sorted(descendants_before)
    assert all(d not in tree.nodes for d in descendants_before)
    assert tree.nodes[edited_id]["applied_fn_def"].name == "double"
    assert tree.nodes[edited_id]["typed_list"].items == [2, 4]
    assert tree.nodes[edited_id]["children"] == []


def test_replace_edge_on_root_raises(executor):
    tree = TrajectoryTree.with_root(TypedList([1]))
    with pytest.raises(ValueError):
        tree.replace_edge(ROOT_ID, inc, executor)


def test_replace_edge_with_incompatible_fn_drops_stale_descendants(executor):
    # Before the cascade-prune fix, editing an int->str edge in the middle of
    # an int-only chain surfaced TypeErrors on every descendant. Now descendants
    # are dropped so the edited node is just a new leaf.
    tree = _build_chain(executor, TypedList([1, 2]), [inc, inc, double])
    chain = _chain_ids(tree)
    edited_id = chain[1]

    tree.replace_edge(edited_id, to_str, executor)

    assert tree.nodes[edited_id]["typed_list"].items == ["1", "2"]
    assert tree.nodes[edited_id]["children"] == []
    # Root + edited node only; no stale descendants carrying type errors.
    assert len(tree.nodes) == 2


def test_execution_error_surfaces_on_node(executor):
    tree = TrajectoryTree.with_root(TypedList([1, 2, 3]))

    child_id = tree.append_child(ROOT_ID, div_zero, executor)

    node = tree.nodes[child_id]
    assert node["typed_list"] is None
    assert node["error"] is not None
    assert "ZeroDivisionError" in node["error"]


def test_delete_descendants_only_affects_subtree(executor):
    tree = TrajectoryTree.with_root(TypedList([5]))
    branch_a = tree.append_child(ROOT_ID, inc, executor)
    branch_b = tree.append_child(ROOT_ID, double, executor)
    grandchild = tree.append_child(branch_a, inc, executor)

    deleted = tree.delete_descendants(branch_a)

    assert deleted == [grandchild]
    assert grandchild not in tree.nodes
    assert branch_b in tree.nodes
    assert tree.nodes[branch_a]["children"] == []
    assert tree.nodes[ROOT_ID]["children"] == [branch_a, branch_b]
