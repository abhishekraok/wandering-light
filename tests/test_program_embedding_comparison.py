"""Geometry negatives must fail execution; timeouts and passing neighbors are excluded."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_geometry_excludes_unknown_and_passing_neighbors(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "reports/program-embeddings-20261024"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "compare_corrected", folder / "compare_corrected.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = ["dup", "dup dup", "drop", "unknown", "swap"]
    outcomes = {"drop": False, "unknown": None, "swap": True}
    monkeypatch.setattr(
        module.ev, "_matches", lambda program, examples: outcomes[program]
    )
    index = SimpleNamespace(
        programs=bank,
        vectors=torch.tensor(
            [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ),
        model=SimpleNamespace(encode_io=lambda examples: torch.tensor([[1.0, 0.0]])),
    )
    record = SimpleNamespace(
        key="task",
        split="test",
        examples=[("given", "wanted")],
        programs=bank[:2],
        styles=["a", "b"],
        provenance={"partition": "cross_style_holdout"},
    )
    result = module.geometry(
        index, [record], [("cross_style_holdout", "task", record.examples)]
    )
    assert result["negative_execution_counts"] == {
        "passed": 1,
        "failed": 1,
        "unknown": 1,
    }
    assert len(result["positive_pairs"]) == 1
    assert result["positive_pairs"][0]["token_jaccard"] == 1
    assert [r["margin"] for r in result["program_margins"]] == pytest.approx([1.8, 1.6])
    assert result["io_margins"][0]["margin"] == pytest.approx(1.8)
    assert result["distributions"]["low_lexical_positive_cosine"] == {"count": 0}

    provenance = {
        "commit": "abc",
        "dirty": True,
        "seed": 0,
        "config": {"dim": 128},
        "index_config": {"batch_size": 32},
        "checkpoint_digest": "sha256:weights",
        "wandb_run.url": "https://wandb.ai/example/run",
        "data": {"records_digest": "sha256:data"},
        "checkpoint_metadata": {"large_nested_history": [1, 2, 3]},
    }
    compact = json.loads(json.dumps(module.compact_provenance(provenance)))
    assert compact["records_digest"] == "sha256:data"
    assert compact["checkpoint_digest"] == "sha256:weights"
    assert compact["wandb_run.url"] == provenance["wandb_run.url"]
    assert "checkpoint_metadata" not in compact
    assert "data" not in compact
