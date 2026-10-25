"""Regression coverage for saved evaluations, lineage and legacy plot inputs."""

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from wandering_light import program_embedding_eval as evaluator
from wandering_light import program_embedding_train as trainer
from wandering_light.program_embedding import EmbeddingConfig, ProgramEmbedding
from wandering_light.program_embedding_artifacts import save_evaluation
from wandering_light.program_embedding_data import BehaviorRecord


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    records = [
        BehaviorRecord(
            key=split,
            family="toy",
            split=split,
            programs=["dup add", "2 mul"],
            styles=["sol", "opus"],
            examples=[(str(i), str(2 * i)) for i in range(1, 7)],
            held_out_examples=[("7", "14")],
            provenance={
                "partition": "cross_style_holdout",
                "component": split,
                "component_size": 1,
                "source_digests": {"fixture": "sha256:toy-input"},
            },
        )
        for split in ("train", "validation", "test")
    ]
    monkeypatch.setattr(trainer, "load_records", lambda *a, **k: records)
    monkeypatch.setattr(
        trainer,
        "_git_metadata",
        lambda root: {"commit": "evaluator-code", "dirty": True},
    )
    run = SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
    import wandb

    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    checkpoint = tmp_path / "legacy-v2.pt"
    model = ProgramEmbedding(EmbeddingConfig(dim=6, hidden_dim=5, byte_dim=4))
    # Existing format, with an intentionally different training seed/code/data.
    model.save(
        checkpoint,
        {
            "commit": "training-code",
            "dirty": False,
            "seed": 17,
            "data_digest": "training-data",
            "wandb_url": "https://wandb.ai/personal/project/runs/original",
            "step": 23,
            "initial_checkpoint_metadata": {"commit": "parent-training-code"},
        },
    )
    return records, checkpoint


def _run(tmp_path, *options):
    out = tmp_path / "output"
    trainer.main(
        [
            "--out",
            str(out),
            "--repo-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--offline",
            "--dim",
            "8",
            "--hidden-dim",
            "8",
            "--byte-dim",
            "4",
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "11",
            "--io-examples",
            "2",
            "--execution-examples",
            "3",
            "--max-eval-records",
            "1",
            "--global-bank",
            "--skip-execution",
            *options,
        ]
    )
    return out


def _assert_provenance(report, checkpoint=None):
    provenance = report["provenance"]
    assert provenance["commit"] == "evaluator-code"
    assert provenance["dirty"] is True
    assert provenance["seed"] == 11
    assert provenance["evaluation_source"]["commit"] == "evaluator-code"
    assert provenance["training"]["max_eval_records"] == 1
    assert provenance["data"]["source_data_digests"] == {"fixture": "sha256:toy-input"}
    assert len(provenance["data"]["records_digest"]) > 60
    assert len(provenance["data_digest"]) == 64
    assert provenance["evaluation_config"] == report["evaluation_config"]
    assert provenance["evaluation_config"]["split"] == report["split"]
    assert provenance["evaluation_config"]["execution"] is False
    assert provenance["evaluation_config"]["io_examples"] == 2
    assert provenance["evaluation_config"]["global_bank"] is True
    if checkpoint is None:
        assert provenance["initialization"] == "random"
        assert provenance["checkpoint_path"] is None
        assert provenance["checkpoint_digest"] is None
        assert provenance["checkpoint_metadata"] == {}
    else:
        assert provenance["initialization"] == "checkpoint"
        assert provenance["checkpoint_path"] == str(checkpoint)
        assert (
            provenance["checkpoint_digest"]
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        model = ProgramEmbedding.load(checkpoint)
        assert provenance["config"] == asdict(model.config)
        assert provenance["checkpoint_metadata"] == model.metadata


@pytest.mark.parametrize("untrained", [False, True])
def test_eval_cli_writes_standalone_provenance(tmp_path, experiment, untrained):
    _, checkpoint = experiment
    options = (
        ["--untrained-eval"]
        if untrained
        else ["--eval-only", "--checkpoint", str(checkpoint)]
    )
    out = _run(tmp_path, *options, "--eval-split", "test")
    report = json.loads((out / "evaluation.json").read_text())
    _assert_provenance(report, None if untrained else checkpoint)
    assert report["provenance"]["evaluation_config"]["execution_examples"] == 3
    if not untrained:
        assert report["provenance"]["checkpoint_metadata"]["seed"] == 17
        assert report["provenance"]["checkpoint_metadata"]["commit"] == "training-code"
        assert report["provenance"]["config"]["dim"] == 6
        assert report["provenance"]["training"]["dim"] == 8
        assert report["provenance"]["wandb_run.url"].endswith("/original")
    sidecar = json.loads((out / "provenance.json").read_text())
    assert sidecar["initialization"] == report["provenance"]["initialization"]
    # Copying the report alone preserves all traceability, without its sidecar
    # or checkpoint being present at read time.
    standalone = tmp_path / "copied-evaluation.json"
    standalone.write_bytes((out / "evaluation.json").read_bytes())
    (out / "provenance.json").unlink()
    checkpoint.unlink()
    assert json.loads(standalone.read_text())["provenance"] == report["provenance"]


@pytest.mark.parametrize("warm_start", [False, True])
def test_training_reports_trace_initial_and_selected_checkpoint(
    tmp_path, experiment, warm_start
):
    _, checkpoint = experiment
    options = ["--checkpoint", str(checkpoint)] if warm_start else []
    out = _run(tmp_path, *options)
    initial = json.loads(
        (out / ("initial.json" if warm_start else "untrained.json")).read_text()
    )
    final = json.loads((out / "validation.json").read_text())
    _assert_provenance(initial, checkpoint if warm_start else None)
    _assert_provenance(final, out / "best.pt")
    assert final["provenance"]["checkpoint_metadata"]["step"] == 1
    assert final["provenance"]["evaluation_config"]["execution_examples"] == 3
    # Initial evaluation uses evaluate()'s default, rather than the final grader
    # argument: the artifact records the effective settings, not just CLI flags.
    assert initial["provenance"]["evaluation_config"]["execution_examples"] == 16
    assert initial["provenance"]["config"] == final["provenance"]["config"]
    if warm_start:
        parent = final["provenance"]["checkpoint_metadata"]
        assert parent["initial_checkpoint_metadata"]["commit"] == "training-code"
        assert (
            parent["initial_checkpoint_digest"]
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )


def test_saved_provenance_is_a_snapshot_and_legacy_metadata_is_optional(tmp_path):
    model = ProgramEmbedding(EmbeddingConfig(dim=4, hidden_dim=4, byte_dim=4))
    model.metadata = {"nested": {"seed": 3}}
    metadata = {"data": {"digests": ["original"]}}
    report = {"evaluation_config": {"seed": 9}, "partitions": {}}
    path = tmp_path / "report.json"
    save_evaluation(path, report, metadata, model)
    metadata["data"]["digests"].append("changed")
    model.metadata["nested"]["seed"] = 4
    report["evaluation_config"]["seed"] = 10
    assert report["provenance"] == json.loads(path.read_text())["provenance"]
    assert report["provenance"]["data"]["digests"] == ["original"]
    assert report["provenance"]["checkpoint_metadata"]["nested"]["seed"] == 3
    assert report["provenance"]["evaluation_config"]["seed"] == 9
    assert report["provenance"]["evaluation_source"]["commit"] is None


def test_evaluation_imports_remain_compatible():
    assert trainer.evaluate is evaluator.evaluate
    assert trainer.lexical_vectors is evaluator.lexical_vectors
    assert trainer.retrieval_metrics is evaluator.retrieval_metrics


@pytest.mark.parametrize("legacy", [False, True])
def test_plot_accepts_new_and_legacy_reports(tmp_path, experiment, legacy):
    _ = experiment
    out = _run(tmp_path, "--untrained-eval", "--eval-split", "test")
    report = json.loads((out / "evaluation.json").read_text())
    if legacy:
        report.pop("provenance")
        report.pop("evaluation_config")
    for name in ("test-trained.json", "test-untrained.json"):
        (out / name).write_text(json.dumps(report))
    (out / "library_manifest.json").write_text(
        json.dumps({"programs": 2, "provenance": {"commit": "library-code"}})
    )
    script = (
        Path(__file__).parents[1]
        / "reports/program-embeddings-20261024/plot_results.py"
    )
    spec = importlib.util.spec_from_file_location("embedding_plot", script)
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    inputs = {path: path.read_bytes() for path in out.glob("*.json")}
    figure = tmp_path / "figures/metrics.png"
    assert plotter.plot(out, figure) == figure
    assert figure.read_bytes().startswith(b"\x89PNG")
    summary = json.loads(figure.with_name("metrics.json").read_text())
    assert summary["test_trained"] == plotter._compact(report)
    assert summary["test_untrained"] == plotter._compact(report)
    if legacy:
        assert summary["evaluation_source"] is None
        assert summary["evaluation_provenance"] == {"Trained": None, "Random": None}
        assert any("source is unknown" in caveat for caveat in summary["caveats"])
    else:
        assert summary["evaluation_source"]["commit"] == "evaluator-code"
        assert summary["evaluation_provenance"]["Random"] == report["provenance"]
    assert all(path.read_bytes() == content for path, content in inputs.items())
