"""Blind audit fixtures never expose or require real benchmark targets."""

import gzip
import json
from types import SimpleNamespace

from experiments.pbe_scale import audit_eval as audit


def benchmark(name="private-synthetic-name", reference=sum):
    def sample(rng):
        return ([rng.randint(-9, 9) for _ in range(rng.randint(1, 12))],)

    return SimpleNamespace(name=name, reference=reference, sample=sample)


def record(name, expression):
    return {
        "candidate": {
            "id": name,
            "input_kind": "list",
            "reference": f"def f(xs): return {expression}",
        },
        "candidate_digest": name * 3,
        "task": {"behaviour": name, "split": "train"},
    }


def test_sampler_domain_distinct_nonconstant_boundaries_and_small_domain():
    target = benchmark(reference=max)  # Empty input would fail: sampler is nonempty.
    banks, insufficient = audit.banks_for([target])
    assert insufficient == 0
    bank = banks[0]
    assert len(bank["inputs"]) == 64
    assert len({json.dumps(x) for x in bank["inputs"]}) == 64
    assert all(args[0] for args in bank["inputs"])
    assert {
        min(len(args[0]) for args in bank["inputs"]),
        max(len(args[0]) for args in bank["inputs"]),
    } == {1, 12}
    small = SimpleNamespace(
        name="small-private-domain",
        sample=lambda rng: (rng.randint(-10, 10),),
        reference=lambda x: -x,
    )
    banks, insufficient = audit.banks_for([small])
    assert not banks
    assert insufficient == 1


def test_prefilters_are_batched_and_full_checks_only_for_survivors(monkeypatch):
    banks, _ = audit.banks_for([benchmark(), benchmark(name="other", reference=max)])
    audit.init_worker(banks)
    calls = []
    original = audit.evaluate

    def counted(source, inputs):
        calls.append(len(inputs))
        return original(source, inputs)

    monkeypatch.setattr(audit, "evaluate", counted)
    assert audit.audit_candidate(record("duplicate", "sum(xs)")) == (True, False)
    assert len(calls) == 2
    calls.clear()
    assert audit.audit_candidate(record("different", "sum(xs) + 10000")) == (
        False,
        False,
    )
    assert len(calls) == 1
    assert audit.evaluate("def f(xs): return 1 // 0", [[[1]], [[2]]]) == [None, None]


def test_exclusion_report_and_original_task_export_are_blind(tmp_path):
    source = tmp_path / "accepted.jsonl"
    rows = [record("duplicate-id", "sum(xs)"), record("new-id", "sum(xs) + 10000")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "audit"
    filtered = tmp_path / "filtered.jsonl.gz"
    report = audit.run(
        source, output, workers=2, filtered=filtered, references=[benchmark()]
    )
    assert report["excluded_ids"] == ["duplicate-id"]
    assert report["excluded_hashes"] == ["duplicate-id" * 3]
    assert report["audited_candidates"] == 2
    assert report["covered_eval_references"] == 1
    assert "private-synthetic-name" not in (output / "exclusions.json").read_text()
    assert "inputs" not in report and "outputs" not in report
    with gzip.open(filtered, "rt") as stream:
        assert [json.loads(line) for line in stream] == [rows[1]["task"]]
