"""Tests for forward generation of certified long-distance tasks."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from experiments import generate_deep_corpus as deep
from wandering_light.basis_set import load_basis_set
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.proposer_pilot.graph import TrajectoryGraph
from wandering_light.typed_list import TypedList

BASIS = load_basis_set("default")
FUNCTIONS = BASIS.as_function_set()


def _plan(typed_list, *, max_depth, index=0, split="discovery"):
    return deep.RootPlan(
        index=index,
        split=split,
        input_type=typed_list.item_type,
        typed_list=typed_list,
        max_depth=max_depth,
        max_states=200_000,
        max_transitions=400_000,
        deep=False,
    )


def _expand(typed_list, *, max_depth, tasks_per_distance=2, frontier_sample=0):
    return deep.expand_root(
        _plan(typed_list, max_depth=max_depth),
        functions=FUNCTIONS,
        seed=11,
        min_distance=1,
        tasks_per_distance=tasks_per_distance,
        frontier_sample=frontier_sample,
        allow_constant_outputs=False,
        verify_witnesses=True,
    )


def test_frontier_extension_keeps_signed_zero_candidates_apart():
    # Two frontier parents whose successors are equal under Python but distinct
    # to the basis. Grouping candidates on canonical_key merges them into one
    # entry whose mask and in_edges -- the optimal action labels -- would then
    # mix both parents.
    to_float = FunctionDef(
        name="to_float",
        input_type="builtins.int",
        output_type="builtins.float",
        code="return float(x)",
    )
    neg_float = FunctionDef(
        name="neg_float",
        input_type="builtins.int",
        output_type="builtins.float",
        code="return -float(x)",
    )
    mul_zero = FunctionDef(
        name="mul_zero",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return x * 0.0",
    )
    functions = FunctionDefSet([to_float, neg_float, mul_zero])
    root = TypedList([1, 2], item_type=int)

    tasks, _ = deep.expand_root(
        _plan(root, max_depth=1),
        functions=functions,
        seed=11,
        min_distance=1,
        tasks_per_distance=4,
        frontier_sample=8,
        allow_constant_outputs=True,
        verify_witnesses=True,
    )

    # float(1) * 0.0 is 0.0 and -float(1) * 0.0 is -0.0: equal, distinguishable.
    extended = [task for task in tasks if task.certified_distance == 2]
    outputs = {task.output_value.search_key() for task in extended}
    assert len(outputs) == 2, f"expected both signed zeros, got {len(outputs)}"
    assert len({task.output_value.canonical_key() for task in extended}) == 1


def _reaches_within(source, target, depth):
    """Whether an exhaustive expansion of ``depth`` steps reaches ``target``."""
    if depth < 0:
        return False
    graph = TrajectoryGraph(FUNCTIONS)
    root_id = graph.add_root(source)
    expansion = graph.expand(root_id, depth)
    found = graph.find(target)
    return found is not None and found in expansion.node_depths


def _tiny_corpus(tmp_path, **overrides):
    options = {
        "basis_set_id": "default",
        "output_dir": tmp_path / "corpus",
        "roots": 4,
        "max_depth": 2,
        "deep_roots": 0,
        "deep_max_depth": 3,
        "min_distance": 1,
        "tasks_per_distance": 3,
        "frontier_sample": 4,
        "max_states": 50_000,
        "max_transitions": 100_000,
        "deep_max_states": 50_000,
        "deep_max_transitions": 100_000,
        "seed": 5,
        "progress": False,
    }
    options.update(overrides)
    manifest, path = deep.generate_corpus(**options)
    return manifest, path.parent


def test_import_does_not_load_torch():
    """Corpus generation must not depend on the model stack."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import experiments.generate_deep_corpus; "
            "print('torch' in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_roots_are_distinct_and_split_by_root():
    plans = deep.build_root_plans(
        roots=27,
        seed=3,
        max_depth=4,
        deep_roots=0,
        deep_max_depth=5,
        max_states=1000,
        max_transitions=1000,
        deep_max_states=1000,
        deep_max_transitions=1000,
    )
    assert len({plan.typed_list.canonical_key() for plan in plans}) == 27
    assert len({plan.input_type for plan in plans}) == len(deep.SUPPORTED_RANDOM_TYPES)
    counts = Counter(plan.split for plan in plans)
    assert counts["discovery"] == 23
    assert counts["validation"] == 2
    assert counts["test"] == 2


def test_every_split_covers_every_input_type():
    # The split cycle is nine long and the types cycle twelve; indexing the
    # cycle by root index would lock each split to 12 / gcd(9, 12) = 4 types and
    # leave the other eight entirely in discovery, so the held-out sets would
    # differ from discovery by input type rather than only by root.
    plans = deep.build_root_plans(
        roots=240,
        seed=3,
        max_depth=4,
        deep_roots=0,
        deep_max_depth=5,
        max_states=1000,
        max_transitions=1000,
        deep_max_states=1000,
        deep_max_transitions=1000,
    )

    expected = set(deep.SUPPORTED_RANDOM_TYPES)
    for split in ("discovery", "validation", "test"):
        covered = {plan.input_type for plan in plans if plan.split == split}
        assert covered == expected, f"{split} covers {len(covered)} of {len(expected)}"


def test_deep_root_types_restrict_the_deeper_budget():
    plans = deep.build_root_plans(
        roots=24,
        seed=3,
        max_depth=4,
        deep_roots=2,
        deep_max_depth=6,
        max_states=1000,
        max_transitions=1000,
        deep_max_states=9000,
        deep_max_transitions=9000,
        deep_root_types=("builtins.set",),
    )
    deep_plans = [plan for plan in plans if plan.deep]
    assert len(deep_plans) == 2  # only two positions in 24 roots carry set inputs
    assert all(plan.input_type is set for plan in deep_plans)
    assert all(plan.max_depth == 6 for plan in deep_plans)
    assert all(plan.max_depth == 4 for plan in plans if not plan.deep)


def test_deep_roots_cannot_exceed_eligible_positions():
    with pytest.raises(ValueError, match="eligible root"):
        deep.build_root_plans(
            roots=12,
            seed=3,
            max_depth=4,
            deep_roots=5,
            deep_max_depth=5,
            max_states=1000,
            max_transitions=1000,
            deep_max_states=1000,
            deep_max_transitions=1000,
            deep_root_types=("builtins.bool",),
        )


def test_certified_distance_admits_no_shorter_path():
    tasks, outcome = _expand(TypedList([3, -1]), max_depth=3)
    assert tasks
    assert outcome.certified_depth == 3
    for task in tasks:
        assert len(task.witness) == task.certified_distance
        assert not _reaches_within(
            task.input_value, task.output_value, task.certified_distance - 1
        )


def test_optimal_first_actions_match_brute_force():
    """Every action labelled optimal must land on a state one step closer."""
    tasks, _ = _expand(TypedList([3, -1]), max_depth=3)
    root = TypedList([3, -1])
    graph = TrajectoryGraph(FUNCTIONS)
    graph.add_root(root)
    successors = {}
    for function in FUNCTIONS:
        if function.input_type_cls() is not root.item_type:
            continue
        try:
            successors[function.name] = graph.executor.execute(function, root)
        except Exception:
            continue

    for task in tasks:
        distance = task.certified_distance
        expected = {
            name
            for name, value in successors.items()
            if value.canonical_key() != root.canonical_key()
            and _reaches_within(value, task.output_value, distance - 1)
            and not _reaches_within(value, task.output_value, distance - 2)
        }
        assert {f.name for f in task.optimal_first_actions} == expected
        assert task.optimal_first_actions_complete


def test_frontier_extension_certifies_one_more_level():
    tasks, outcome = _expand(TypedList([3, -1]), max_depth=3, frontier_sample=6)
    extended = [
        task for task in tasks if task.certification == deep.CERTIFICATION_FRONTIER
    ]
    assert extended
    assert outcome.extended_depth == outcome.certified_depth + 1
    for task in extended:
        assert task.certified_distance == outcome.certified_depth + 1
        assert not task.optimal_first_actions_complete
        assert not _reaches_within(
            task.input_value, task.output_value, task.certified_distance - 1
        )


def test_effective_identity_is_a_type_only_change():
    assert deep._is_effective_identity(TypedList([1, 2]), TypedList([1.0, 2.0]))
    assert not deep._is_effective_identity(TypedList([1, 2]), TypedList([2, 3]))


def test_generated_tasks_are_not_value_identities():
    tasks, _ = _expand(TypedList([3, -1]), max_depth=3, tasks_per_distance=4)
    for task in tasks:
        assert list(task.input_value.items) != list(task.output_value.items)


def test_generate_and_verify_roundtrip(tmp_path):
    manifest, corpus_dir = _tiny_corpus(tmp_path)
    assert manifest["basis_set_id"] == BASIS.basis_set_id
    assert manifest["basis_set_digest"] == BASIS.digest
    assert manifest["split_policy"].startswith("by root")

    result = deep.verify_corpus(corpus_dir, recertify=3, seed=5)
    assert result["ok"], result
    assert result["records"] > 0
    assert result["witness_failures"] == []
    assert result["roots_leaking_across_splits"] == []
    assert result["recertified"]
    assert all(not row["reachable_below_distance"] for row in result["recertified"])

    _, records = deep.load_corpus(corpus_dir)
    assert len(records) == result["records"]
    for record in records:
        metadata = record.metadata
        assert metadata["certified_distance"] == record.witness_length
        assert metadata["optimal_first_action_ids"]
        assert metadata["certification"] in {
            deep.CERTIFICATION_COMPLETE,
            deep.CERTIFICATION_FRONTIER,
        }
        assert record.basis_set_id == BASIS.basis_set_id


def test_roots_do_not_leak_across_splits(tmp_path):
    _, corpus_dir = _tiny_corpus(tmp_path, roots=18, max_depth=2)
    _, records = deep.load_corpus(corpus_dir)
    splits_by_root = {}
    for record in records:
        splits_by_root.setdefault(record.metadata["root_digest"], set()).add(
            record.split
        )
    assert all(len(splits) == 1 for splits in splits_by_root.values())
    assert len({record.split for record in records}) > 1


def test_existing_corpus_is_not_overwritten_by_default(tmp_path):
    _, corpus_dir = _tiny_corpus(tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _tiny_corpus(tmp_path)
    manifest, _ = _tiny_corpus(tmp_path, overwrite=True)
    assert manifest["splits"]["discovery"]["size"] > 0
    assert (corpus_dir / "manifest.json").exists()


def test_tampered_manifest_is_rejected(tmp_path):
    _, corpus_dir = _tiny_corpus(tmp_path)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["discovery"]["size"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        deep.load_corpus(corpus_dir)


def test_tampered_split_file_is_rejected(tmp_path):
    _, corpus_dir = _tiny_corpus(tmp_path)
    path = corpus_dir / "discovery.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file digest mismatch"):
        deep.load_corpus(corpus_dir)


def test_reference_curve_reports_solve_rate_and_cost(tmp_path):
    _, corpus_dir = _tiny_corpus(
        tmp_path, roots=18, max_depth=2, tasks_per_distance=3, frontier_sample=0
    )
    result = deep.reference_curve(
        corpus_dir,
        depths=[1, 2],
        tasks_per_distance=3,
        budget=100_000,
        splits=("validation", "test"),
        seed=5,
        progress=False,
    )
    assert [row["depth"] for row in result["curve"]] == [1, 2]
    depth_one, depth_two = result["curve"]
    assert "1" in depth_one["by_certified_distance"]
    assert depth_one["by_certified_distance"]["1"]["solve_rate"] == 1.0
    assert depth_two["solved"] >= depth_one["solved"]
    assert depth_two["mean_ms_per_task"] > 0
    assert result["basis_set_digest"] == BASIS.digest
