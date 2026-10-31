"""Data preparation should not execute programs or depend on a model."""

import gzip
import hashlib
import json
from dataclasses import asdict

import pytest

from wandering_light.program_embedding_data import (
    _assign_splits,
    _deduplicate,
    _merge_exact_programs,
    _new_record,
    load_records,
    main,
    manifest_for,
)
from wandering_light.stack_pbe import Task, skeleton


def _task(name="sum", program=("0", "[", "add", "]", "fold"), split="train"):
    return asdict(
        Task(
            behaviour=name,
            summary=name,
            split=split,
            task_hash_split="train",
            inputs=("[ 1 2 ]", "[ 3 4 ]", "[ 5 ]", "[ 7 ]", "[ 8 ]"),
            outputs=("3", "7", "5", "7", "8"),
            held_out_inputs=("[ 9 ]",),
            held_out_outputs=("9",),
            program=program,
            length=len(program),
            examples=5,
            held_out=1,
            fuel=100,
            tags=(),
            skeleton=skeleton(program),
        )
    )


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _repo(tmp_path, rows):
    corpus = tmp_path / "wandering_light/evals/data/stack_pbe_v4"
    _write_jsonl(corpus / "tasks.jsonl", rows)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {"splits": {"train": {"path": "tasks.jsonl", "sha256": "sha256:wrong"}}}
        )
    )
    return tmp_path


def test_aggregated_hidden_inputs_never_appear_in_visible_views(tmp_path):
    first, second = _task(), _task()
    second["held_out_inputs"] = ["[ 1 2 ]"]
    second["held_out_outputs"] = ["3"]
    root = _repo(tmp_path, [first, second])
    (record,) = load_records(root)
    assert len(record.examples) == 4
    assert set(dict(record.examples)).isdisjoint(dict(record.held_out_examples))
    assert record.provenance["shown_hidden_input_overlap_removed"] == 1
    assert any("digest mismatch" in text for text in record.provenance["caveats"])


def test_conflicting_outputs_are_excluded_from_both_views():
    record = _new_record("sum")
    record.examples = [(" [ 1 ] ", "1"), ("[ 1 ]", "2"), ("[ 2 ]", "2")]
    record.held_out_examples = [("[ 2 ]", "3"), ("[ 3 ]", "3")]
    _deduplicate(record)
    assert record.examples == []
    assert record.held_out_examples == [("[ 3 ]", "3")]


def test_sidefile_styles_and_flagged_variants(tmp_path):
    root = _repo(tmp_path, [_task(), _task("bad", ("dup",))])
    _write_jsonl(
        root / "experiments/staged_witnesses/v4.jsonl",
        [
            {
                "behaviour": "sum",
                "program": "0 swap [ add ] fold",
                "refused_program": "0 [ add ] fold",
                "flags": [],
            },
            {
                "behaviour": "bad",
                "program": "drop",
                "refused_program": "swap",
                "flags": ["precondition: nonempty"],
            },
        ],
    )
    records = {row.key: row for row in load_records(root)}
    assert records["sum"].programs == ["0 [ add ] fold", "0 swap [ add ] fold"]
    assert records["sum"].styles == ["canonical", "staged"]
    assert records["bad"].programs == ["dup"]


def test_every_witness_skeleton_connects_split_components():
    a, b, c = [_new_record(key) for key in ("a", "b", "c")]
    a.programs = ["dup", "[ 0 add ] map"]
    b.programs = ["[ 1 mul ] map", "swap"]
    c.programs = ["swap"]
    c.provenance["official_eval"] = True
    _assign_splits([a, b, c])
    assert [r.split for r in (a, b, c)] == ["test"] * 3
    assert len({r.provenance["component"] for r in (a, b, c)}) == 1
    assert a.provenance["partition"] == "official_eval_overlap"
    assert c.provenance["partition"] == "official_eval"


def test_peer_family_is_stratification_template_is_split_group(tmp_path):
    root = _repo(tmp_path, [_task("official", ("drop",), "eval")])
    extra = root / "accepted.jsonl"
    rows = []
    for index, program in enumerate((("dup",), ("swap",))):
        task = _task(f"peer-{index}", program)
        rows.append(
            {
                "status": "accepted",
                "candidate": {"family": "broad", "template_id": f"template-{index}"},
                "task": task,
                "signature": "same-signature-not-same-domain",
            }
        )
    _write_jsonl(extra, rows)
    records = {r.key: r for r in load_records(root, [extra])}
    assert records["peer-0"].family == records["peer-1"].family == "broad"
    assert (
        records["peer-0"].provenance["component"]
        != records["peer-1"].provenance["component"]
    )
    assert records["official"].provenance["partition"] == "official_eval"


def test_exact_program_duplicates_merge_aliases_and_keep_official_eval():
    a, b, c = [_new_record(key) for key in ("a", "b", "c")]
    a.programs, a.styles = ["dup"], ["peer"]
    b.programs, b.styles = ["dup", "swap"], ["sol", "opus"]
    c.programs, c.styles = ["swap"], ["canonical"]
    for record in (a, b, c):
        record.provenance["sources"] = []
    c.provenance["official_eval"] = True
    (merged,) = _merge_exact_programs([a, b, c])
    assert merged.key == "c"
    assert set(merged.programs) == {"dup", "swap"}
    assert set(merged.provenance["aliases"]) == {"a", "b"}


@pytest.mark.parametrize("first_program", ["dup drop", ""])
def test_independent_teachers_use_existing_labels_and_verification(
    tmp_path, first_program
):
    root = _repo(tmp_path, [_task()])
    base = root / "reports/distinct-programs-20260926"
    behavior = {
        "id": "dc-test-k-k1",
        "tests": [{"args": [[1]], "out": [1]}, {"args": [[2]], "out": [2]}],
        "shown": [0],
        "held_out": [1],
    }
    _write_jsonl(base / "behaviours.jsonl", [behavior])
    _write_jsonl(base / "audit.jsonl", [{"id": behavior["id"], "verdict": "keep"}])
    _write_jsonl(base / "verdicts.jsonl", [{"id": behavior["id"], "verified": True}])
    _write_jsonl(
        base / "programs_w01.jsonl", [{"id": behavior["id"], "program": first_program}]
    )
    _write_jsonl(base / "opus_check.out", [{"id": behavior["id"], "verified": True}])
    with (base / "opus_check.out").open("a") as stream:
        stream.write("1 verified\n")
    _write_jsonl(
        base / "programs_o01.jsonl", [{"id": behavior["id"], "program": "nil cat"}]
    )
    records = {r.key: r for r in load_records(root)}
    independent = records[behavior["id"]]
    assert independent.programs == [first_program, "nil cat"]
    assert independent.styles == ["sol", "opus"]
    assert independent.examples == [("[ (1) ]", "[ (1) ]")]
    assert independent.held_out_examples == [("[ (2) ]", "[ (2) ]")]


def test_gzip_task_adapter_and_manifest_cli_are_deterministic(tmp_path):
    root = _repo(tmp_path, [_task()])
    extra = root / "extra.jsonl.gz"
    with gzip.open(extra, "wt") as stream:
        stream.write(json.dumps(_task("other", ("dup",))) + "\n")
    before = load_records(root, [extra])
    after = load_records(root, [extra])
    assert before == after
    assert (
        manifest_for(before, root)["records_digest"]
        == manifest_for(after, root)["records_digest"]
    )
    out = root / "prepared"
    main(["--repo-root", str(root), "--extra-tasks", str(extra), "--out", str(out)])
    manifest = json.loads((out / "manifest.json").read_text())
    assert sum(manifest["counts"].values()) == 2
    payload = (out / "records.jsonl").read_bytes()
    assert len(payload.splitlines()) == 2
    assert manifest["records_digest"] == "sha256:" + hashlib.sha256(payload).hexdigest()
    preparation = manifest["preparation"]
    assert preparation["commit"] is not None
    assert isinstance(preparation["dirty"], bool)
    assert preparation["config"]["repo_root"] == str(root.resolve())
    assert preparation["config"]["extra_tasks"] == [str(extra.resolve())]
    assert preparation["config"]["split_policy"] == manifest["config"]["split_policy"]
    assert preparation["seed"] == manifest["seed"]
    assert preparation["wandb_run.url"] is None
    assert (
        preparation["input_records_digest"]
        == manifest_for(before, root)["records_digest"]
    )
    main(["--repo-root", str(root), "--extra-tasks", str(extra), "--out", str(out)])
    assert (out / "records.jsonl").read_bytes() == payload
    # The records file carries preparation lineage even away from its sidecar.
    copied = tmp_path / "copied-records.jsonl"
    copied.write_bytes(payload)
    (out / "manifest.json").unlink()
    for row in map(json.loads, copied.read_text().splitlines()):
        assert row["provenance"]["preparation"] == preparation
        assert row["provenance"]["source_digests"]


@pytest.mark.parametrize("program", [[], ""])
def test_accepted_identity_witness_survives_without_missing_programs(tmp_path, program):
    root = _repo(tmp_path, [_task()])
    extra = root / "accepted.jsonl"
    identity = _task("identity", ())
    identity["program"] = program
    identity["alternative_programs"] = [["dup", "drop"]]
    identity["outputs"] = identity["inputs"]
    identity["held_out_outputs"] = identity["held_out_inputs"]
    missing = {**identity, "behaviour": "missing", "alternative_programs": []}
    missing.pop("program")
    unsolved = {**missing, "behaviour": "unsolved", "program": None}
    rejected = {**identity, "behaviour": "rejected"}
    _write_jsonl(
        extra,
        [
            {"status": "accepted", "task": identity},
            {"status": "accepted", "task": missing},
            {"status": "accepted", "task": unsolved},
            {"status": "rejected", "task": rejected},
        ],
    )
    records = {record.key: record for record in load_records(root, [extra])}
    assert records["identity"].programs == ["", "dup drop"]
    assert records["identity"].styles == ["peer", "alternative"]
    assert records["missing"].programs == records["unsolved"].programs == []
    assert "rejected" not in records


@pytest.mark.parametrize("canonical", [(), ("dup", "drop")])
def test_identity_is_retained_in_canonical_and_sidefile_programs(tmp_path, canonical):
    task = _task("identity", canonical)
    task["outputs"] = task["inputs"]
    task["held_out_outputs"] = task["held_out_inputs"]
    root = _repo(tmp_path, [task])
    _write_jsonl(
        root / "experiments/staged_witnesses/v4.jsonl",
        [{"behaviour": "identity", "program": "dup drop", "refused_program": ""}],
    )
    (record,) = load_records(root)
    if canonical:
        assert record.programs == ["dup drop", ""]
        assert record.styles == ["canonical", "refused"]
    else:
        assert record.programs == ["", "dup drop"]
        assert record.styles == ["canonical", "staged"]


@pytest.mark.parametrize("primary", [{}, {"program": None}])
def test_sidefile_missing_primary_preserves_identity_alternative(tmp_path, primary):
    root = _repo(tmp_path, [_task("identity", ("dup", "drop"))])
    _write_jsonl(
        root / "experiments/staged_witnesses/v4.jsonl",
        [{"behaviour": "identity", "refused_program": "", **primary}],
    )
    (record,) = load_records(root)
    assert record.programs == ["dup drop", ""]
    assert record.styles == ["canonical", "refused"]
