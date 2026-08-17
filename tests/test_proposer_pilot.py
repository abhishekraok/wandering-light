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
dec = FunctionDef(
    name="dec",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x - 1",
)
identity = FunctionDef(
    name="identity",
    input_type="builtins.int",
    output_type="builtins.int",
    code="return x",
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

    def test_state_dedup_is_structural_for_nested_unordered_values(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        a = TypedList([{"a": [1, 2], "b": {3, 4}}], item_type=dict)
        b = TypedList([{"b": {4, 3}, "a": [1, 2]}], item_type=dict)
        assert g.add_root(a) == g.add_root(b)

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


class TestTrajectoryGraphExpansion:
    def test_exhaustive_expansion_certifies_shortest_paths(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))

        expansion = g.expand(root, max_depth=2)
        task = next(
            task
            for task in g.tasks_from_expansion(expansion)
            if task.trajectory.output == _tl([3])
        )

        assert expansion.complete
        assert expansion.certified_depth == 2
        assert expansion.attempted_transitions == 12
        assert task.verified_shortest_num_steps == 1
        assert task.shortest_path_is_certified
        assert [fn.name for fn in task.trajectory.function_defs] == ["add_two"]

    def test_certified_path_ignores_edges_outside_expansion_function_set(self):
        functions = FunctionDefSet([inc])
        g = TrajectoryGraph(functions=functions)
        root = g.add_root(_tl([1]))
        g.apply(root, add_two)

        expansion = g.expand(root, max_depth=2)
        task = next(
            task
            for task in g.tasks_from_expansion(expansion)
            if task.trajectory.output == _tl([3])
        )

        assert task.verified_shortest_num_steps == 2
        assert [fn.name for fn in task.trajectory.function_defs] == ["inc", "inc"]

    def test_expansion_evidence_depths_are_immutable(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        expansion = g.expand(root, max_depth=1)

        with pytest.raises(TypeError):
            expansion.node_depths[root] = 99

    def test_expansion_filters_noops_but_retains_cycles(self):
        functions = FunctionDefSet([inc, dec, identity])
        g = TrajectoryGraph(functions=functions)
        root = g.add_root(_tl([0]))

        expansion = g.expand(root, max_depth=2)

        assert expansion.complete
        assert expansion.skipped_self_loops == 3
        assert all(
            child != node.id for node in g.nodes() for _, child in node.out_edges
        )
        assert any(child == root for fn, child in g.node(g.find(_tl([1]))).out_edges)

    def test_partial_function_failure_does_not_invalidate_completeness(self):
        reciprocal = FunctionDef(
            name="reciprocal",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return 1 // x",
        )
        g = TrajectoryGraph(functions=FunctionDefSet([inc, reciprocal]))
        root = g.add_root(_tl([0, 1]))

        expansion = g.expand(root, max_depth=1)

        assert expansion.complete
        assert expansion.attempted_transitions == 2
        assert expansion.failed_transitions == 1
        assert expansion.certified_depth == 1

    def test_graph_state_key_agrees_with_typed_list_search_key(self):
        # All exhaustive expansion must delegate to the same "may these two
        # states be merged" rule as the shipped BFS solver. A divergent copy of
        # this key is precisely the bug class that inflates certified distances.
        from wandering_light.proposer_pilot.graph import _state_key

        values = [
            _tl([0.0], float),
            _tl([-0.0], float),
            _tl([0.0, 1.0], float),
            _tl([-0.0, 1.0], float),
            _tl([float("nan")], float),
            _tl([-float("nan")], float),
            _tl([1.5], float),
            _tl([{"a": [0.0]}], dict),
            _tl([{"a": [-0.0]}], dict),
            _tl([(0.0,)], tuple),
            _tl([(-0.0,)], tuple),
        ]

        for left in values:
            assert _state_key(left) == left.search_key()

    def test_signed_zero_is_not_merged_with_positive_zero(self):
        # -0.0 == 0.0 in Python, so keying states on equality merges them and
        # leaves -0.0's successors unexplored: ["-0.0"] then looks unreachable,
        # or is found later through some longer path and mislabelled.
        negate = FunctionDef(
            name="negate",
            input_type="builtins.float",
            output_type="builtins.float",
            code="return -x",
        )
        show = FunctionDef(
            name="show",
            input_type="builtins.float",
            output_type="builtins.str",
            code="return str(x)",
        )
        g = TrajectoryGraph(functions=FunctionDefSet([negate, show]))
        root = g.add_root(_tl([0.0], float))

        expansion = g.expand(root, max_depth=2)
        negative = g.find(_tl([-0.0], float))
        negative_text = g.find(_tl(["-0.0"], str))

        assert expansion.complete
        assert negative is not None and negative != root
        assert expansion.node_depths[negative] == 1
        assert negative_text is not None
        assert expansion.node_depths[negative_text] == 2
        assert g.find(_tl(["0.0"], str)) != negative_text

    def test_transition_budget_only_certifies_completed_layers(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))

        expansion = g.expand(root, max_depth=3, max_transitions=5)
        certified = list(g.tasks_from_expansion(expansion))
        all_reached = list(g.tasks_from_expansion(expansion, require_certified=False))

        assert not expansion.complete
        assert expansion.stop_reason == "max_transitions"
        assert expansion.certified_depth == 1
        assert certified
        assert all(task.num_steps == 1 for task in certified)
        assert len(all_reached) >= len(certified)
        assert all(
            task.verified_shortest_num_steps is None
            for task in all_reached
            if task.num_steps > expansion.certified_depth
        )

    def test_state_budget_before_root_layer_completion_certifies_nothing(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))

        expansion = g.expand(root, max_depth=2, max_states=2)

        assert expansion.stop_reason == "max_states"
        assert expansion.certified_depth == 0
        assert list(g.tasks_from_expansion(expansion)) == []
        reached = list(g.tasks_from_expansion(expansion, require_certified=False))
        assert len(reached) == 1
        assert reached[0].verified_shortest_num_steps is None

    def test_task_from_proposal_keeps_proposed_and_verified_lengths_separate(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        expansion = g.expand(root, max_depth=2)
        proposal_spec = TrajectorySpec(_tl([1]), FunctionDefList([inc, inc]))
        proposal = Trajectory(proposal_spec, _tl([3]))

        task = g.task_from_proposal(proposal, expansion)

        assert task.proposed_num_steps == 2
        assert task.verified_shortest_num_steps == 1
        assert task.num_steps == 1
        assert [fn.name for fn in task.trajectory.function_defs] == ["add_two"]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_depth": -1}, "max_depth"),
            ({"max_depth": 1, "max_states": 0}, "max_states"),
            ({"max_depth": 1, "max_transitions": -1}, "max_transitions"),
        ],
    )
    def test_expansion_rejects_invalid_bounds(self, kwargs, message):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        with pytest.raises(ValueError, match=message):
            g.expand(root, **kwargs)

    def test_expansion_evidence_is_graph_specific(self):
        g = TrajectoryGraph(functions=FUNCTIONS)
        root = g.add_root(_tl([1]))
        expansion = g.expand(root, max_depth=1)
        other = TrajectoryGraph(functions=FUNCTIONS)
        other.add_root(_tl([1]))

        with pytest.raises(ValueError, match="different graph"):
            list(other.tasks_from_expansion(expansion))


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
        assert t.proposed_num_steps is None
        assert t.verified_shortest_num_steps is None
        assert not t.shortest_path_is_certified
