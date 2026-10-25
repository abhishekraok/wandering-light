"""Campaign validation rejects bad authorship before exporting Task rows."""

import gzip
import json
from pathlib import Path

import pytest

from experiments.pbe_scale.validate import (
    PROBE_POLICY,
    SHARED_PROBES,
    bounded,
    check_reference,
    expanded_program,
    probe_bank,
    run,
    validate_candidate,
)


def candidate(**updates):
    row = {
        "id": "sum-a",
        "family": "aggregate",
        "template_id": "sum-template",
        "description": "Sum the input list.",
        "input_kind": "list",
        "program": "sum",
        "reference": "def f(xs):\n    return sum(xs)\n",
    }
    return row | updates


def test_probe_bank_is_distinct_shared_seeded_and_covers_edges():
    for kind in ("list", "list_int", "two_lists", "int", "two_ints"):
        inputs = probe_bank(kind, 4)
        assert inputs == probe_bank(kind, 4)
        assert len({json.dumps(x) for x in inputs}) == len(inputs)
        assert len(inputs) == (
            21 if kind == "int" else 441 if kind == "two_ints" else SHARED_PROBES
        )
        if "list" in kind:
            assert {len(args[0]) for args in inputs[:6]} == {0, 1, 2, 3, 8, 12}
        fresh = probe_bank(kind, 5, 16, exclude=inputs)
        assert not {json.dumps(x) for x in fresh} & {json.dumps(x) for x in inputs}


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(xs): return xs",
        "def f(xs):\n    return open('/tmp/x').read()",
        "def f(xs):\n    return xs.__class__",
        "@sum\ndef f(xs): return xs",
    ],
)
def test_reference_capabilities_are_rejected(source):
    with pytest.raises((ValueError, SyntaxError)):
        check_reference(source)


def test_reference_timeout_and_invalid_outputs():
    with pytest.raises(ValueError, match="timeout"):
        bounded("reference", "def f(xs):\n    while True: pass", [[[]]], timeout=0.1)
    for expression in ("True", "[[1]]", "[0] * 257", "10**13"):
        row = validate_candidate(
            candidate(reference=f"def f(xs): return {expression}"), 0, {}
        )
        assert row["status"] == "rejected"
        assert row["stage"] == "reference"
    row = validate_candidate(candidate(reference="def f(xs): return max(xs)"), 0, {})
    assert row["status"] == "rejected"
    assert "empty" in row["reason"]


def test_valid_witness_and_fresh_hidden_export():
    row = validate_candidate(candidate(), 42, {})
    assert row["status"] == "accepted", row
    assert row["verification_count"] == 208
    assert row["shown_count"] == 6
    assert row["hidden_count"] == 202
    task = row["task"]
    assert len(set(task["inputs"] + task["held_out_inputs"])) == 208
    assert task["macro_program"] == "sum"
    assert len(task["program"]) == task["length"]


def test_dedup_precedes_stack_verification_and_rejects_identity_constant():
    signatures = {}
    assert validate_candidate(candidate(), 4, signatures)["status"] == "accepted"
    duplicate = validate_candidate(
        candidate(id="sum-b", program="garbage"), 4, signatures
    )
    assert duplicate["stage"] == "dedup"
    assert duplicate["duplicate_of"] == "sum-a"
    for expression, reason in (("xs", "identity"), ("0", "constant")):
        row = validate_candidate(
            candidate(reference=f"def f(xs): return {expression}"), 4, {}
        )
        assert row["stage"] == "dedup"
        assert reason in row["reason"]


def test_witness_failure_does_not_occupy_signature_and_private_macros_work():
    signatures = {}
    bad = validate_candidate(candidate(program="len"), 4, signatures)
    assert bad["stage"] == "witness"
    assert "mismatch" in bad["reason"]
    assert not signatures
    good = validate_candidate(
        candidate(program="mysum", macros={"mysum": "0 [ add ] fold"}), 4, signatures
    )
    assert good["status"] == "accepted", good
    with pytest.raises(ValueError, match="recursion"):
        expanded_program(candidate(program="loop", macros={"loop": "loop"}))
    with pytest.raises(ValueError, match="shadows"):
        expanded_program(candidate(macros={"sum": "len"}))


def test_scalar_exhaustive_and_template_split():
    row = validate_candidate(
        candidate(
            id="neg", input_kind="int", program="neg", reference="def f(x): return -x"
        ),
        3,
        {},
    )
    assert row["status"] == "accepted", row
    assert row["scalar_exhaustive"]
    assert (row["shown_count"], row["hidden_count"]) == (6, 15)
    sibling = validate_candidate(
        candidate(id="len", program="len", reference="def f(xs): return len(xs)"), 3, {}
    )
    assert sibling["status"] == "accepted", sibling
    assert row["task"]["split"] == sibling["task"]["split"]


def test_atomic_resume_and_export_preserve_author_provenance(tmp_path: Path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(candidate(_provenance={"model": "author", "source_batch": "batch1"}))
        + "\n"
    )
    output = tmp_path / "output"
    first = run([source], output, seed=12)
    assert first["accepted"] == 1
    # Simulate an interruption after journaling but before snapshot replacement.
    (output / "accepted.jsonl").write_text("")
    second = run([source], output, seed=12)
    assert second["candidates"] == 1
    restored = json.loads((output / "accepted.jsonl").read_text())
    assert restored["candidate"]["_provenance"]["source_batch"] == "batch1"
    exported = []
    for split in ("train", "eval"):
        with gzip.open(output / f"tasks_{split}.jsonl.gz", "rt") as handle:
            exported.extend(json.loads(line) for line in handle)
    assert len(exported) == 1
    assert exported[0]["behaviour"] == "sum-a"
    assert second["exact_verified_pairs"] == 208


def test_shown_inputs_vary_by_candidate_and_list_lengths():
    total = validate_candidate(candidate(), 188, {})
    size = validate_candidate(
        candidate(id="length-b", program="len", reference="def f(xs): return len(xs)"),
        188,
        {},
    )
    assert total["status"] == size["status"] == "accepted"
    assert total["task"]["inputs"] != size["task"]["inputs"]
    for row in (total, size):
        task = row["task"]
        assert not set(task["inputs"]) & set(task["held_out_inputs"])
        assert len(set(task["inputs"])) == 6
        # The machine serializes flat lists as '( ... )'.
        assert len({len(text.split()) for text in task["inputs"]}) == 6
        assert row["verification_count"] == 208
    scalar_a = validate_candidate(
        candidate(
            id="neg-a", input_kind="int", program="neg", reference="def f(x): return -x"
        ),
        188,
        {},
    )
    scalar_b = validate_candidate(
        candidate(
            id="neg-b", input_kind="int", program="neg", reference="def f(x): return -x"
        ),
        188,
        {},
    )
    assert scalar_a["task"]["inputs"] != scalar_b["task"]["inputs"]
    assert scalar_a["verification_count"] == 21


def test_reference_persistent_default_state_is_rejected():
    source = "def f(xs, seen=[]):\n    result = 999999 if xs in seen else sum(xs)\n    seen.append(xs)\n    return result"
    row = validate_candidate(candidate(reference=source), 188, {})
    assert row["status"] == "rejected"
    assert row["stage"] == "reference"
    assert "defaults forbidden" in row["reason"]


def test_reference_function_attribute_state_is_rejected():
    source = "def f(xs):\n    if not xs: f.count = []\n    result = 999999 if xs in f.count else sum(xs)\n    f.count.append(xs)\n    return result"
    row = validate_candidate(candidate(reference=source), 188, {})
    assert row["status"] == "rejected"
    assert row["stage"] == "reference"
    assert "attribute" in row["reason"]


def test_resume_freezes_seed_split_and_compacts_source_provenance(tmp_path):
    first_source = tmp_path / "first.jsonl"
    second_source = tmp_path / "second.jsonl"
    first_source.write_text(json.dumps(candidate()) + "\n")
    size = candidate(
        id="length-second", program="len", reference="def f(xs): return len(xs)"
    )
    second_source.write_text(json.dumps(size) + "\n")
    output = tmp_path / "out"
    run([first_source], output, seed=12, dev_share=0)
    report = run([second_source], output, seed=999, dev_share=0.99)
    assert report["accepted"] == 2
    assert report["resume_caveats"]
    assert report["provenance"]["config"]["seed"] == 12
    assert set(report["provenance"]["source_data_digests"]) >= {
        str(first_source),
        str(second_source),
    }
    rows = [
        json.loads(line)
        for line in (output / "accepted.jsonl").read_text().splitlines()
    ]
    for row in rows:
        assert "provenance" not in row
        assert len(row["provenance_id"]) == 64
        assert len(row["source"]["sha256"]) == 64
        assert (output / "provenance" / (row["provenance_id"] + ".json")).exists()
        assert row["task"]["split"] == "train"
    expected = validate_candidate(size, 12, {})
    assert rows[-1]["task"]["inputs"] == list(expected["task"]["inputs"])


def test_resume_reads_legacy_inline_provenance(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(candidate()) + "\n")
    output = tmp_path / "out"
    run([source], output, seed=12)
    journal = next((output / "records").glob("*.json"))
    row = json.loads(journal.read_text())
    row["provenance"] = json.loads(
        (output / "provenance" / (row.pop("provenance_id") + ".json")).read_text()
    )
    journal.write_text(json.dumps(row))
    (output / "campaign_config.json").unlink()
    report = run([source], output, seed=999)
    assert report["accepted"] == 1
    assert report["resume_caveats"]
    compact = json.loads((output / "accepted.jsonl").read_text())
    assert "provenance" not in compact
    assert "provenance_id" in compact


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ('int("".join(str(abs(x) % 10) for x in xs)) if xs else 0', [0, 112]),
        ("sorted(set(xs).intersection({-1, 0, 1}))", [[], [-1, 1]]),
        ("sorted(set(xs).union({0}))", [[0], [-1, 0, 1, 2]]),
        ("sorted(set(xs).difference({1}))", [[], [-1, 2]]),
        ("sorted(set(xs).symmetric_difference({0, 1}))", [[0, 1], [-1, 0, 2]]),
    ],
)
def test_pure_string_and_set_reference_operations(expression, expected):
    source = f"def f(xs): return {expression}"
    check_reference(source)
    assert bounded("reference", source, [[[]], [[1, -1, 2]]]) == expected


def test_v3_cartesian_probes_cover_confirmed_false_merge():
    inputs = probe_bank("two_lists", 188)
    assert [[1], [1]] in inputs
    assert [[-1], [-1]] in inputs
    assert [[], [1, 1]] in inputs
    assert [[1, -1], [10]] in inputs
    reference_span = "def f(xs, ys):\n    zs = sorted(set(max(0, min(x, y)) for x, y in zip(xs, ys)))\n    return max(zs, default=0) - min(zs, default=0)"
    reference_max = "def f(xs, ys):\n    zs = sorted(set(max(0, min(x, y)) for x, y in zip(xs, ys)))\n    return max(zs, default=0)"
    span = bounded("reference", reference_span, inputs)
    maximum = bounded("reference", reference_max, inputs)
    assert span != maximum
    index = inputs.index([[1], [1]])
    assert (span[index], maximum[index]) == (0, 1)
    assert len(inputs) == 192
    assert len(probe_bank("two_ints", 188)) == 441
    assert {tuple(args) for args in probe_bank("two_ints", 188)} == {
        (a, b) for a in range(-10, 11) for b in range(-10, 11)
    }
    assert [[1], 1] in probe_bank("list_int", 188)


def test_old_output_directory_keeps_legacy_probe_policy_with_caveat(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(candidate()) + "\n")
    output = tmp_path / "old"
    output.mkdir()
    (output / "campaign_config.json").write_text(
        json.dumps({"seed": 188, "dev_share": 0})
    )
    report = run([source], output, seed=188)
    assert report["provenance"]["config"]["probe_policy"] != PROBE_POLICY
    assert report["exact_verified_pairs"] == 80
    assert any("fresh output" in caveat for caveat in report["resume_caveats"])


def test_constant_fresh_shown_gets_different_output_shared_probe(monkeypatch):
    from experiments.pbe_scale import validate as module

    original = module.probe_bank
    shared = original("list", 188)
    # Fresh inputs vary in length and are disjoint from shared probes, but
    # all produce zero. Only a positive shared edge reveals the rare branch.
    fresh = [
        [[value] * length]
        for value in range(-2, -10, -1)
        for length in range(1, 13)
        if [[value] * length] not in shared
    ][:16]

    def probes(kind, seed, count=None, exclude=None, policy=PROBE_POLICY):
        return fresh if exclude else shared

    monkeypatch.setattr(module, "probe_bank", probes)
    row = validate_candidate(
        candidate(
            id="rare-positive",
            program="[ 0 gt? ] filter len",
            reference="def f(xs): return sum(x > 0 for x in xs)",
        ),
        188,
        {},
    )
    assert row["status"] == "accepted", row
    assert row["shown_output_values"] >= 2
    assert len(set(row["task"]["outputs"])) >= 2
    assert len({len(text.split()) for text in row["task"]["inputs"]}) == 6
    assert not set(row["task"]["inputs"]) & set(row["task"]["held_out_inputs"])
