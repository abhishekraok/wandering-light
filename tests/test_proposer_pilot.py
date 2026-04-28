import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.proposer_pilot import SolveRater, Task, TrajectoryGraph
from wandering_light.solver import create_bfs_solver
from wandering_light.trajectory import Trajectory, TrajectorySpec
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
add_two = FunctionDef(
    name="add_two",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x + 2",
)
to_str = FunctionDef(
    name="to_str",
    input_type="builtins.int",
    output_type="builtins.str",
    code="return str(x)",
)

FUNCTIONS = FunctionDefSet([inc, double, add_two, to_str])


def _tl(items, t=int):
    return TypedList(items, item_type=t)


class TestTrajectoryGraph:
    def test_add_root_dedupes_by_state(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        a = g.add_root(_tl([1, 2, 3]))
        b = g.add_root(_tl([1, 2, 3]))
        assert a == b
        assert g.num_nodes() == 1
        assert g.roots == [a]

    def test_apply_executes_and_dedupes_child(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        # inc then inc again -> [3]; or add_two once -> [3]. Same state.
        via_inc = g.apply(g.apply(root, inc), inc)
        via_add_two = g.apply(root, add_two)
        assert via_inc == via_add_two
        assert g.node(via_inc).typed_list == _tl([3])

    def test_apply_allows_parallel_edges_with_distinct_fns(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        c1 = g.apply(root, inc)  # [2]
        c2 = g.apply(root, double)  # [2]
        assert c1 == c2
        # two distinct edges from root to c1
        out_fns = sorted(fn.name for fn, _ in g.node(root).out_edges)
        assert out_fns == ["double", "inc"]

    def test_apply_does_not_duplicate_same_edge(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        g.apply(root, inc)
        g.apply(root, inc)
        assert len(g.node(root).out_edges) == 1
        assert g.num_edges() == 1

    def test_apply_unknown_parent_raises(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        with pytest.raises(KeyError):
            g.apply(999, inc)

    def test_apply_by_name(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        c = g.apply_by_name(root, "inc")
        assert g.node(c).typed_list == _tl([2])

    def test_apply_by_name_unknown_raises(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        with pytest.raises(KeyError):
            g.apply_by_name(root, "nope")

    def test_shortest_path_direct(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        c = g.apply(root, inc)
        path = g.shortest_path(root, c)
        assert [fn.name for fn in path] == ["inc"]

    def test_shortest_path_self(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        assert g.shortest_path(root, root) == []

    def test_shortest_path_prefers_shorter_when_paths_merge(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        # add_two: 1-step path to [3]
        # inc -> inc: 2-step path to [3]
        mid = g.apply(root, inc)
        long_dst = g.apply(mid, inc)
        short_dst = g.apply(root, add_two)
        assert long_dst == short_dst
        path = g.shortest_path(root, short_dst)
        assert [fn.name for fn in path] == ["add_two"]

    def test_shortest_path_unreachable(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        a = g.add_root(_tl([1]))
        b = g.add_root(_tl([42]))
        assert g.shortest_path(a, b) is None

    def test_find(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        assert g.find(_tl([1])) == root
        assert g.find(_tl([99])) is None

    def test_tasks_respects_min_max_steps(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        n1 = g.apply(root, inc)  # depth 1
        n2 = g.apply(n1, inc)  # depth 2
        g.apply(n2, inc)  # depth 3

        all_tasks = list(g.tasks())
        assert len(all_tasks) == 3
        assert {t.num_steps for t in all_tasks} == {1, 2, 3}

        bounded = list(g.tasks(min_steps=2, max_steps=2))
        assert len(bounded) == 1
        assert bounded[0].num_steps == 2

    def test_tasks_uses_shortest_path_under_dedup(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        mid = g.apply(root, inc)
        g.apply(mid, inc)  # creates [3] via 2 steps
        g.apply(root, add_two)  # also lands on [3] in 1 step
        tasks = [t for t in g.tasks() if t.trajectory.output == _tl([3])]
        assert len(tasks) == 1
        assert tasks[0].num_steps == 1
        assert [fn.name for fn in tasks[0].trajectory.function_defs] == ["add_two"]

    def test_tasks_from_explicit_srcs(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        other = g.add_root(_tl([10]))
        g.apply(root, inc)
        g.apply(other, double)
        tasks_from_other = list(g.tasks(src_ids=[other]))
        assert all(t.src_id == other for t in tasks_from_other)
        assert len(tasks_from_other) == 1


class TestSolveRater:
    def test_rate_solvable_with_bfs(self):
        rater = SolveRater(
            solver=create_bfs_solver(budget=50, max_depth=3), functions=FUNCTIONS
        )
        result = rater.rate(_tl([1]), _tl([3]))
        assert result.n_attempts == 1
        assert result.n_solved == 1
        assert result.rate == 1.0

    def test_rate_unsolvable_with_bfs(self):
        rater = SolveRater(
            solver=create_bfs_solver(budget=10, max_depth=2), functions=FUNCTIONS
        )
        # [1] -> [100] requires more than 2 steps with these functions
        result = rater.rate(_tl([1]), _tl([100]))
        assert result.n_solved == 0
        assert result.rate == 0.0

    def test_rate_tasks_batch(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        g.apply(root, inc)
        g.apply(g.apply(root, inc), inc)
        tasks = list(g.tasks())
        rater = SolveRater(
            solver=create_bfs_solver(budget=50, max_depth=3), functions=FUNCTIONS
        )
        results = rater.rate_tasks(tasks)
        assert len(results) == len(tasks)
        # All graph-built tasks have a known short path; BFS at depth 3 finds them.
        assert all(r.rate == 1.0 for r in results)

    def test_n_attempts_propagates(self):
        rater = SolveRater(
            solver=create_bfs_solver(budget=50, max_depth=3),
            functions=FUNCTIONS,
            n_attempts=3,
        )
        result = rater.rate(_tl([1]), _tl([2]))
        assert result.n_attempts == 3
        assert result.n_solved == 3


class TestTaskDataclass:
    def test_num_steps(self):
        spec = TrajectorySpec(_tl([1]), FunctionDefList([inc]))
        t = Task(
            src_id=0,
            dst_id=1,
            trajectory=Trajectory(spec, _tl([2])),
        )
        assert t.num_steps == 1
        assert t.trajectory.input == _tl([1])
        assert t.trajectory.output == _tl([2])
