"""stack_pbe_world_v1 builder (#120): rows are labelled by execution and round-trip."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wandering_light.stack_pbe import Task, load

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    path = ROOT / "experiments" / "world_behaviours" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(tmp_path: Path, program: str, macro_program: str) -> Path:
    behaviour = {
        "id": "mbpp-0",
        "source": "mbpp",
        "description": "Write a function to double a number. Then something else.",
        "tests": [{"args": [n], "out": 2 * n} for n in range(8)],
        "shown": [0, 1, 2],
        "held_out": [3, 4, 5, 6, 7],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "behaviours.jsonl").write_text(json.dumps(behaviour) + "\n")
    report = {
        "results": [
            {
                "id": "mbpp-0",
                "solved": True,
                "program": program,
                "macro_program": macro_program,
            }
        ]
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path


def test_build_writes_verified_rows_and_round_trips(tmp_path: Path) -> None:
    build_corpus = _module("build_corpus")
    probe = _probe(tmp_path / "probe", "dup add", "dup add")
    tasks = build_corpus.build(probe)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.program == ("dup", "add")
    assert task.macro_program == "dup add"
    assert task.witness_source == "translated"
    assert task.split == "train"
    assert task.inputs == ("[ 0 ]", "[ 1 ]", "[ 2 ]")
    assert task.outputs == ("[ 0 ]", "[ 2 ]", "[ 4 ]")
    assert task.held_out == 5
    assert task.summary == "Write a function to double a number."
    assert task.tags == ("mbpp",)

    corpus_dir = tmp_path / "corpus"
    build_corpus.write(tasks, corpus_dir, probe)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["split_policy"] == "train-only"
    loaded = load(corpus_dir)
    assert loaded == tasks


def test_repeated_inputs_are_not_shown_and_held_out_at_once(tmp_path: Path) -> None:
    build_corpus = _module("build_corpus")
    probe = _probe(tmp_path / "probe", "dup add", "dup add")
    behaviour = json.loads((probe / "behaviours.jsonl").read_text())
    behaviour["tests"][4] = behaviour["tests"][
        1
    ]  # a held-out input repeats a shown one
    (probe / "behaviours.jsonl").write_text(json.dumps(behaviour) + "\n")
    (task,) = build_corpus.build(probe)
    assert task.examples == 3
    assert task.held_out == 4
    assert not set(task.inputs) & set(task.held_out_inputs)


def test_build_refuses_a_witness_that_disagrees_with_the_tests(tmp_path: Path) -> None:
    build_corpus = _module("build_corpus")
    probe = _probe(tmp_path / "probe", "dup mul", "dup mul")
    with pytest.raises(Exception, match="disagrees"):
        build_corpus.build(probe)


def test_macro_program_defaults_for_older_payloads() -> None:
    task = Task(
        behaviour="b",
        summary="s",
        split="train",
        task_hash_split="train",
        inputs=("[ 1 ]",),
        outputs=("[ 2 ]",),
        held_out_inputs=(),
        held_out_outputs=(),
        program=("1", "add"),
        length=2,
        examples=1,
        held_out=0,
        fuel=2,
        tags=(),
    )
    assert task.macro_program == ""


def test_generated_examples_break_a_coincidental_invariant() -> None:
    """Three asserts with distinct elements must not restrict sort to distinct inputs.

    Codex review on #125: under the strict inferred `distinct` property the
    shortcut `dedup sort` passed every generated example of the sorting
    behaviours. Relaxed candidates must include duplicates, and the shortcut
    must fail the resulting set.
    """
    import random

    extract = _module("extract")
    verify = _module("verify")
    item = {
        "source": "mbpp",
        "id": "mbpp-sort-test",
        "name": "sort_list",
        "description": "Write a function to sort a list.",
        "solution": "def sort_list(xs):\n    return sorted(xs)\n",
        "entry": "sort_list",
        "tests": [
            {"args": [[3, 1, 2]], "out": [1, 2, 3]},
            {"args": [[9, 4, 7, 1]], "out": [1, 4, 7, 9]},
            {"args": [[5, 2]], "out": [2, 5]},
        ],
    }
    extended = extract.extend(17, item)
    assert extended is not None
    generated = [t for t in extended["tests"] if t.get("generated")]
    assert any(len(set(t["args"][0])) < len(t["args"][0]) for t in generated), (
        "no generated input contains a duplicate"
    )
    macros = verify.load_macros()
    shortcut = verify.check(verify.expand(["dedup", "sort"], macros), extended)
    assert shortcut["passed"] < shortcut["total"]
    honest = verify.check(verify.expand(["sort"], macros), extended)
    assert honest["passed"] == honest["total"]
    random.seed(0)  # keep the module-level RNG untouched for other tests


def test_migration_preserves_recorded_repairs(tmp_path: Path) -> None:
    """Codex review on #125: re-running migrate.py must not lose a repaired witness."""
    import shutil

    migrate = _module("migrate")
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "programs_1.jsonl").write_text(
        json.dumps(
            {
                "id": "mbpp-644",
                "program": "over over take reverse k1--rot drop-n cat",
                "status": "solved",
            }
        )
        + "\n"
    )
    out = tmp_path / "out"
    shutil.copytree(
        ROOT / "reports" / "world-behaviours-library-20260910" / "authors",
        out / "authors",
    )
    shutil.copy(
        ROOT / "reports" / "world-behaviours-library-20260910" / "repairs.jsonl",
        out / "repairs.jsonl",
    )
    assert migrate.main([str(probe), str(out)]) == 0
    (row,) = [json.loads(l) for l in (out / "programs_canonical.jsonl").open()]
    assert row["program"].startswith("[ dup 0 eq? ]")
    assert row["migrated_program"] == "over over take reverse -rot drop-n cat"


def test_migration_prefers_the_reports_own_alias_table(tmp_path: Path) -> None:
    """Codex review on #129: author letters repeat across waves, so `ka-complement`
    meant one thing in wave 1 and another in wave 2; a wave's aliases.txt wins."""
    migrate = _module("migrate")
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "programs_a.jsonl").write_text(
        json.dumps({"id": "x-1", "program": "ka-complement", "status": "solved"}) + "\n"
    )
    out = tmp_path / "out"
    (out / "authors").mkdir(parents=True)
    (out / "authors" / "macros_a.txt").write_text("ka-complement: neg dec\n")
    (out / "aliases.txt").write_text("ka-complement: @inline\n")
    assert migrate.main([str(probe), str(out)]) == 0
    (row,) = [json.loads(l) for l in (out / "programs_canonical.jsonl").open()]
    assert row["program"] == "neg dec"


def test_relations_to_argument_zero_are_scheduled_and_violated() -> None:
    """Codex re-review on #125: an index or membership relation to argument 0 is a relation, not a disabled property."""
    import random

    extract = _module("extract")
    given = [
        [[3, 1, 4], 1],
        [[9, 4, 7, 1], 4],
        [[5, 2], 2],
    ]  # int is an index into, and a member of, the list
    specs = extract.infer_invariants(given)
    assert specs[1]["bounded_by"] == 0
    assert specs[1]["member_of"] == 0
    holding = extract.holding_properties(specs)
    assert (1, "bounded_by") in holding
    assert (1, "member_of") in holding
    rng = random.Random(3)
    outside = [
        extract.mutate(rng, given[0], extract.relax(specs, (1, "bounded_by")))
        for _ in range(20)
    ]
    assert all(not 0 <= n <= len(xs) for xs, n in outside)
    strangers = [
        extract.mutate(rng, given[0], extract.relax(specs, (1, "member_of")))
        for _ in range(20)
    ]
    assert all(n not in xs for xs, n in strangers)


def test_excluding_eval_behaviours_leaves_the_train_payload_unchanged(
    tmp_path: Path,
) -> None:
    """world-plus-v4-policy (#134): contaminated eval behaviours are removed.

    A behaviour's split depends only on its own id, so removing eval behaviours
    must not move anything between splits: the train payload digest is the
    same with and without the exclusion, which keeps a run trained on the
    earlier revision valid against the new one.
    """
    build_corpus = _module("build_corpus")
    ids = [f"mbpp-{n}" for n in range(12)]
    probe = tmp_path / "probe"
    probe.mkdir()
    with (probe / "behaviours.jsonl").open("w") as fh:
        for behaviour_id in ids:
            fh.write(
                json.dumps(
                    {
                        "id": behaviour_id,
                        "source": "mbpp",
                        "description": "Double a number.",
                        "tests": [{"args": [n], "out": 2 * n} for n in range(8)],
                        "shown": [0, 1, 2],
                        "held_out": [3, 4, 5, 6, 7],
                    }
                )
                + "\n"
            )
    (probe / "report.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": i,
                        "solved": True,
                        "program": "dup add",
                        "macro_program": "dup add",
                    }
                    for i in ids
                ]
            }
        )
    )
    # a share between the ids' buckets, so both splits are populated
    share = sorted(build_corpus.eval_bucket(i) for i in ids)[len(ids) // 2]
    full = build_corpus.build(probe, share)
    eval_ids = frozenset(t.behaviour for t in full if t.split == "eval")
    assert eval_ids
    assert len(eval_ids) < len(ids)

    dropped = frozenset(sorted(eval_ids)[: max(1, len(eval_ids) // 2)])
    kept = build_corpus.build(probe, share, dropped)
    assert {t.behaviour for t in kept}.isdisjoint(dropped)
    assert [t for t in kept if t.split == "train"] == [
        t for t in full if t.split == "train"
    ]

    before = tmp_path / "before"
    after = tmp_path / "after"
    build_corpus.write(full, before, probe, "stack_pbe_world_test", share, 134)
    build_corpus.write(
        kept, after, probe, "stack_pbe_world_test", share, 134, dropped, "test"
    )
    digest = {
        d: json.loads((d / "manifest.json").read_text())["splits"]
        for d in (before, after)
    }
    assert digest[before]["train"]["sha256"] == digest[after]["train"]["sha256"]
    assert digest[after]["eval"]["records"] == digest[before]["eval"]["records"] - len(
        dropped
    )
    recorded = json.loads((after / "manifest.json").read_text())["config"]
    assert recorded["excluded_behaviours"] == sorted(dropped)
    assert recorded["exclusion_reason"] == "test"
