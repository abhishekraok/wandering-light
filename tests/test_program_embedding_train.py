from types import SimpleNamespace

import pytest
import torch

from wandering_light.program_embedding_train import (
    _matches,
    _rank,
    evaluate,
    lexical_vectors,
    retrieval_metrics,
)


class LookupModel:
    def __init__(self, programs):
        self.programs = programs

    def eval(self):
        return self

    def encode_programs(self, programs):
        return torch.stack([self.programs[p] for p in programs])

    def encode_io(self, examples):
        return torch.stack([self.programs[pairs[0][1]] for pairs in examples])


def test_multiple_positives_and_exact_query_exclusion():
    scores = torch.tensor([1.0, 0.9, 0.8, 0.7])
    assert _rank(scores, {0, 2}, {0})[0] == 2
    assert _rank(scores, set())[0] == 0
    assert retrieval_metrics([1, 2, 0])["mrr"] == pytest.approx(0.5)


def test_lexical_baseline_is_order_invariant():
    vectors = lexical_vectors(["dup add", "add dup", "drop"])
    assert torch.allclose(vectors[0], vectors[1])
    assert torch.dot(vectors[0], vectors[2]) == 0


def test_query_cap_preserves_complete_bank_and_hidden_pairs_are_not_encoded():
    records, vectors = [], {}
    for i in range(4):
        a, b = f"program-{i}", f"equivalent-{i}"
        vector = torch.eye(4)[i]
        vectors[a] = vectors[b] = vector
        records.append(
            SimpleNamespace(
                key=str(i),
                family="test",
                split="validation",
                programs=[a, b],
                styles=["fused", "staged"],
                examples=[("shown", a)],
                held_out_examples=[("HIDDEN MUST NOT ENTER ENCODER", "unknown")],
                provenance={"partition": "cross_style_holdout"},
            )
        )
    result = evaluate(
        LookupModel(vectors), records, execution=False, max_eval_records=2
    )
    report = result["partitions"]["cross_style_holdout"]
    assert report["behaviours"] == 2
    assert report["full_partition_behaviours"] == 4
    assert report["bank_programs"] == 8
    assert report["io_to_program"]["recall@1"] == 1
    assert report["cross_style_program"]["recall@1"] == 1
    assert len(result["records"]) == 2


def test_execution_uses_program_level_fuel_and_output_equality():
    assert _matches("dup add", [("3", "6"), ("-2", "-4")])
    assert not _matches("dup add", [("3", "7")])
    assert not _matches("invalid", [("3", "3")])
    assert not _matches("dup", [])


def test_cross_style_excludes_same_style_equivalents_and_balances_behaviours():
    vectors = {
        "a": torch.tensor([1.0, 0.0, 0.0]),
        "b": torch.tensor([1.0, 0.0, 0.0]),
        "c": torch.tensor([0.0, 0.0, 1.0]),
        "d": torch.tensor([0.9, 0.1, 0.0]),
    }
    records = [
        SimpleNamespace(
            key="one",
            family="test",
            split="validation",
            programs=["a", "b", "c"],
            styles=["fused", "fused", "staged"],
            examples=[("shown", "a")],
            held_out_examples=[],
            provenance={"partition": "cross_style_holdout"},
        ),
        SimpleNamespace(
            key="two",
            family="test",
            split="train",
            programs=["d"],
            styles=["fused"],
            examples=[("shown", "d")],
            held_out_examples=[],
            provenance={"partition": "fit"},
        ),
    ]
    report = evaluate(LookupModel(vectors), records, execution=False, global_bank=True)
    row = report["records"][0]
    assert row["cross_style"][0]["rank"] == 2
    summary = report["partitions"]["cross_style_holdout"]
    assert summary["bank_programs"] == 4
    assert summary["cross_style_program"]["behaviours"] == 1
    assert summary["cross_style_program"]["queries"] == 3


def test_all_cross_style_queries_survive_io_cap():
    vectors = {}
    records = []
    for i in range(20):
        programs = [f"a{i}", f"b{i}"] if i < 4 else [f"a{i}"]
        for p in programs:
            vectors[p] = torch.eye(20)[i]
        records.append(
            SimpleNamespace(
                key=str(i),
                family="test",
                split="validation",
                programs=programs,
                styles=["fused", "staged"][: len(programs)],
                examples=[("shown", programs[0])],
                held_out_examples=[],
                provenance={"partition": "cross_style_holdout"},
            )
        )
    result = evaluate(
        LookupModel(vectors), records, execution=False, max_eval_records=1
    )
    summary = result["partitions"]["cross_style_holdout"]
    assert summary["io_to_program"]["queries"] == 1
    assert summary["cross_style_program"]["behaviours"] == 4
    assert summary["cross_style_program"]["queries"] == 8
    assert len(result["program_records"]) == 4


def test_audited_variant_removed_after_split_assignment():
    import hashlib

    from wandering_light.program_embedding_data import BehaviorRecord
    from wandering_light.program_embedding_train import exclude_programs

    record = BehaviorRecord(
        key="stable",
        family="test",
        split="validation",
        programs=["dup add", "0 add"],
        styles=["fused", "staged"],
        examples=[("1", "2")],
        held_out_examples=[],
        provenance={"component": "stable-component"},
    )
    digest = hashlib.sha256(b"0 add").hexdigest()
    records, removed = exclude_programs([record], [digest])
    assert removed == 1
    assert records[0].programs == ["dup add"]
    assert records[0].styles == ["fused"]
    assert records[0].split == "validation"
    assert records[0].provenance["component"] == "stable-component"
    assert records[0].key == "stable"
    assert record.programs == ["dup add", "0 add"]
