"""Tests for forward generation of certified long-distance tasks."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import generate_deep_corpus as deep
from wandering_light.basis_dataset import BasisTaskRecord, write_basis_task_records
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


def test_shell_sizes_count_states_and_reconcile_with_reached():
    # shell_sizes feeds the manifest's states_by_depth. Tasks are drawn per
    # answer class, so it is tempting to count classes here -- but then the
    # published per-layer counts no longer sum to reached_states and the field
    # silently means something other than its name. class_sizes carries that.
    _, outcome = _expand(TypedList([1, 2, 3], item_type=int), max_depth=4)

    assert sum(outcome.shell_sizes.values()) == outcome.reached_states
    for depth, classes in outcome.class_sizes.items():
        assert classes <= outcome.shell_sizes[depth]


def test_distance_is_measured_in_answer_equality_space():
    # `neg` reaches [-0.0] in one step and `slow_pos` reaches [0.0] in three.
    # The two compare equal, so a solver stopping at [-0.0] after one step is
    # graded correct and the task is at distance 1. Filing [0.0] at 3 -- its own
    # path length -- would be a label no solver could be scored against.
    neg = FunctionDef(
        name="neg",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return -0.0 if x == 1.0 else x + 100.0",
    )
    step = FunctionDef(
        name="step",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return x + 1.0 if x in (1.0, 2.0) else x + 200.0",
    )
    settle = FunctionDef(
        name="settle",
        input_type="builtins.float",
        output_type="builtins.float",
        code="return 0.0 if x == 3.0 else x + 300.0",
    )
    functions = FunctionDefSet([neg, step, settle])
    root = TypedList([1.0, 1.0], item_type=float)

    tasks, _ = deep.expand_root(
        _plan(root, max_depth=3),
        functions=functions,
        seed=11,
        min_distance=1,
        tasks_per_distance=8,
        frontier_sample=0,
        allow_constant_outputs=True,
        verify_witnesses=True,
    )

    zero = TypedList([0.0, 0.0], item_type=float)
    matching = [task for task in tasks if task.output_value == zero]
    assert matching, "the all-zero target should be emitted"
    for task in matching:
        assert task.certified_distance == 1, (
            f"answer-equal target filed at {task.certified_distance}, not 1"
        )
    # And it is emitted once, not once per signed-zero variant.
    assert len(matching) == 1


def test_frontier_extension_merges_answer_equal_candidates():
    # Two frontier parents whose successors are 0.0 and -0.0. A solver cannot be
    # asked to tell those apart, so they are one task, reachable by either first
    # action -- and both actions belong in the optimal set. Splitting them would
    # emit the task twice, each copy naming only half of its optimal actions.
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

    # float(1) * 0.0 is 0.0 and -float(1) * 0.0 is -0.0: equal answers.
    extended = [task for task in tasks if task.certified_distance == 2]
    assert len(extended) == 1, f"expected one task, got {len(extended)}"
    assert extended[0].output_value == TypedList([0.0, 0.0], item_type=float)
    # Both first moves reach an acceptable answer in two, so both are optimal.
    assert {f.name for f in extended[0].optimal_first_actions} == {
        "to_float",
        "neg_float",
    }


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


def test_exact_split_counts_and_reference_root_exclusion():
    common = {
        "roots": 27,
        "seed": 3,
        "max_depth": 4,
        "deep_roots": 0,
        "deep_max_depth": 5,
        "max_states": 1000,
        "max_transitions": 1000,
        "deep_max_states": 1000,
        "deep_max_transitions": 1000,
    }
    baseline = deep.build_root_plans(**common)
    excluded = {plan.typed_list.canonical_key() for plan in baseline[:5]}

    plans = deep.build_root_plans(
        **common,
        split_counts={"discovery": 21, "validation": 3, "test": 3},
        excluded_root_keys=tuple(excluded),
    )

    assert Counter(plan.split for plan in plans) == {
        "discovery": 21,
        "validation": 3,
        "test": 3,
    }
    assert not ({plan.typed_list.canonical_key() for plan in plans} & excluded)
    assert len({plan.typed_list.canonical_key() for plan in plans}) == 27


def test_exact_split_counts_must_sum_to_roots():
    with pytest.raises(ValueError, match="sum to"):
        deep.build_root_plans(
            roots=4,
            seed=3,
            max_depth=2,
            deep_roots=0,
            deep_max_depth=3,
            max_states=1000,
            max_transitions=1000,
            deep_max_states=1000,
            deep_max_transitions=1000,
            split_counts={"discovery": 3, "validation": 1, "test": 1},
        )


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
    assert all(row["outcome"] == "certified" for row in result["recertified"])
    assert result["recertification_inconclusive"] == []
    assert result["provenance_failures"] == []
    assert result["manifest_summary_failures"] == []
    assert result["duplicate_task_ids"] == []
    assert result["tasks_leaking_across_splits"] == []

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


def test_reference_corpus_identities_are_excluded_and_verified(tmp_path):
    _, reference_dir = _tiny_corpus(tmp_path / "reference", roots=6, seed=13)
    reference = deep.collect_corpus_identities(reference_dir)

    manifest, corpus_dir = _tiny_corpus(
        tmp_path / "new",
        roots=6,
        seed=13,
        split_counts={"discovery": 4, "validation": 1, "test": 1},
        reference_identities=(reference,),
        target_contract={
            "root_count": 6,
            "split_root_counts": {"discovery": 4, "validation": 1, "test": 1},
            "reference_manifest_digests": [reference.manifest_digest],
        },
    )
    identities = deep.collect_corpus_identities(corpus_dir)
    result = deep.verify_corpus(corpus_dir, reference_identities=(reference,))

    assert identities.root_digests.isdisjoint(reference.root_digests)
    assert identities.root_keys.isdisjoint(reference.root_keys)
    assert identities.task_ids.isdisjoint(reference.task_ids)
    assert manifest["config"]["split_counts"] == {
        "discovery": 4,
        "validation": 1,
        "test": 1,
    }
    assert manifest["reference_overlap"] == {"root_digests": 0, "task_ids": 0}
    assert result["ok"], result


def test_process_workers_preserve_payload_bytes(tmp_path):
    serial, _ = _tiny_corpus(tmp_path / "serial", roots=6, seed=17, workers=1)
    parallel, _ = _tiny_corpus(tmp_path / "parallel", roots=6, seed=17, workers=2)

    for split in deep.SPLIT_ORDER:
        assert serial["splits"][split]["sha256"] == parallel["splits"][split]["sha256"]
    assert serial["identity"] == parallel["identity"]


def _leave_resume_after_root_failure(
    tmp_path, monkeypatch, *, seed=19, workers=1, fail_index=1
):
    resume_dir = tmp_path / "root-checkpoints"
    original = deep.expand_root

    def fail_selected_root(plan, **kwargs):
        if plan.index == fail_index:
            raise RuntimeError("injected root failure")
        return original(plan, **kwargs)

    monkeypatch.setattr(deep, "expand_root", fail_selected_root)
    with pytest.raises(RuntimeError, match="injected root failure"):
        _tiny_corpus(
            tmp_path,
            roots=6,
            seed=seed,
            resume_dir=resume_dir,
            workers=workers,
        )
    monkeypatch.setattr(deep, "expand_root", original)
    assert not (tmp_path / "corpus").exists()
    assert (resume_dir / deep.RESUME_MANIFEST_NAME).is_file()
    assert (resume_dir / "root-000000.json.gz").is_file()
    return resume_dir


def test_resumed_serial_and_parallel_payloads_are_byte_identical(tmp_path, monkeypatch):
    clean, clean_dir = _tiny_corpus(tmp_path / "clean", roots=6, seed=19)
    resume_root = tmp_path / "resumed"
    resume_dir = _leave_resume_after_root_failure(
        resume_root, monkeypatch, workers=2, fail_index=4
    )

    resumed, resumed_dir = _tiny_corpus(
        resume_root,
        roots=6,
        seed=19,
        resume_dir=resume_dir,
        workers=2,
    )
    parallel, parallel_dir = _tiny_corpus(
        tmp_path / "parallel", roots=6, seed=19, workers=2
    )

    for split in deep.SPLIT_ORDER:
        clean_payload = (clean_dir / f"{split}.jsonl.gz").read_bytes()
        assert (resumed_dir / f"{split}.jsonl.gz").read_bytes() == clean_payload
        assert (parallel_dir / f"{split}.jsonl.gz").read_bytes() == clean_payload
        assert resumed["splits"][split]["sha256"] == clean["splits"][split]["sha256"]
        assert parallel["splits"][split]["sha256"] == clean["splits"][split]["sha256"]
    assert resumed["identity"] == clean["identity"] == parallel["identity"]
    assert deep.remove_resume_directory(resume_dir)
    assert not resume_dir.exists()


def test_resume_rejects_configuration_drift(tmp_path, monkeypatch):
    resume_dir = _leave_resume_after_root_failure(tmp_path, monkeypatch, seed=23)

    with pytest.raises(ValueError, match="job fingerprint mismatch"):
        _tiny_corpus(
            tmp_path,
            roots=6,
            seed=23,
            tasks_per_distance=4,
            resume_dir=resume_dir,
        )

    assert (resume_dir / "root-000000.json.gz").is_file()
    assert not (tmp_path / "corpus").exists()


def test_resume_rejects_corrupt_root_shard(tmp_path, monkeypatch):
    resume_dir = _leave_resume_after_root_failure(tmp_path, monkeypatch, seed=29)
    shard = resume_dir / "root-000000.json.gz"
    shard.write_bytes(shard.read_bytes()[:-7])

    with pytest.raises(ValueError, match="corrupt resume shard"):
        _tiny_corpus(
            tmp_path,
            roots=6,
            seed=29,
            resume_dir=resume_dir,
        )

    assert shard.exists()
    assert not (tmp_path / "corpus").exists()


def test_verify_detects_record_provenance_mismatch(tmp_path):
    manifest, corpus_dir = _tiny_corpus(tmp_path)
    _, records = deep.load_corpus(corpus_dir, splits=("discovery",))
    tampered = records[0].to_dict()
    tampered["generator"] = "wrong-generator"
    records[0] = BasisTaskRecord.from_dict(tampered)
    split_path = corpus_dir / "discovery.jsonl.gz"
    write_basis_task_records(records, split_path)
    manifest["splits"]["discovery"]["sha256"] = deep._sha256_file(split_path)
    payload = dict(manifest)
    payload.pop("manifest_digest")
    manifest["manifest_digest"] = deep._sha256_bytes(
        deep._canonical_json(payload).encode()
    )
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = deep.verify_corpus(corpus_dir)

    assert not result["ok"]
    assert any("generator mismatch" in row for row in result["provenance_failures"])


def test_recertification_sample_balances_distance_and_zero_exposure():
    records = [
        SimpleNamespace(
            metadata={"certified_distance": distance},
            output_value=TypedList([value], item_type=float),
        )
        for distance in (1, 2)
        for value in (1.0, 0.0)
    ]

    sample = deep._recertification_sample(records, 4, seed=7)

    assert {
        (
            record.metadata["certified_distance"],
            deep._contains_zero_float(record.output_value.items),
        )
        for record in sample
    } == {(1, False), (1, True), (2, False), (2, True)}


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


def test_per_root_telemetry_is_detached_and_content_addressed(tmp_path):
    manifest, corpus_dir = _tiny_corpus(tmp_path)

    assert "roots" not in manifest["expansion"]
    identity = manifest["detail_files"]["expansion"]
    detail_path = corpus_dir / deep.EXPANSION_DETAIL_NAME
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    assert identity == {
        "path": deep.EXPANSION_DETAIL_NAME,
        "bytes": detail_path.stat().st_size,
        "sha256": deep._sha256_file(detail_path),
        "records": manifest["expansion"]["roots_expanded"],
        "contents": "per-root generation and expansion telemetry",
    }
    assert len(detail["roots"]) == manifest["expansion"]["roots_expanded"]
    assert detail["basis_set_digest"] == manifest["basis_set_digest"]


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
