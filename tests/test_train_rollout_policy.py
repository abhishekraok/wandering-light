"""Prefix supervision and the loadable one-step policy are experiment-critical."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiments.train_rollout_policy as training
import wandering_light.rollout_policy as rollout_policy_module
from experiments.train_rollout_policy import (
    CHECKPOINT_FILES,
    POLICY_SIZE_ARMS,
    PrefixExample,
    PreparedPrefixes,
    RolloutTrainingConfig,
    build_prefix_examples,
    run_training,
    source_code_identity,
    train_policy_model,
)
from wandering_light.basis_set import load_basis_set
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDefSet
from wandering_light.rollout_policy import (
    CONFIG_FILE,
    MAX_PAIR_WIDTH,
    MODEL_FILE,
    TRAINER_STATE_FILE,
    TRAINING_MANIFEST_FILE,
    WANDB_LINK_FILE,
    AlignedPairPolicy,
    PolicyArchitecture,
    RolloutPolicy,
    assert_weights_only_checkpoint,
    pair_batch,
    policy_config,
)
from wandering_light.typed_list import TypedList


def _basis() -> FunctionDefSet:
    return load_basis_set("wl-core-v1").to_function_def_set()


def _write_corpus(path: Path, split: str, witnesses: list[list[str]]) -> Path:
    functions = _basis()
    executor = Executor(functions)
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path / f"{split}.jsonl.gz", "wt", encoding="utf-8") as file:
        for index, names in enumerate(witnesses):
            state = TypedList([2, 3], item_type=int)
            source = state.to_string()
            for name in names:
                state = executor.execute(functions.name_to_function[name], state)
            file.write(
                json.dumps(
                    {
                        "split": split,
                        "task_id": f"task-{index}",
                        "input": source,
                        "output": state.to_string(),
                        "witness_length": len(names),
                        "witness_function_names": names,
                        "metadata": {
                            "certified_distance": len(names),
                            "root_digest": f"sha256:{index:064x}",
                        },
                    }
                )
                + "\n"
            )
    return path


def test_every_proper_prefix_is_replayed_and_labelled_in_its_split(tmp_path):
    corpus = _write_corpus(
        tmp_path, "discovery", [["inc", "double", "neg"], ["square"]]
    )

    examples = build_prefix_examples(corpus, "discovery", _basis())

    assert [example.action_name for example in examples] == [
        "inc",
        "double",
        "neg",
        "square",
    ]
    assert [example.remaining_distance for example in examples] == [3, 2, 1, 1]
    assert [example.prefix_index for example in examples] == [0, 1, 2, 0]
    assert {example.split for example in examples} == {"discovery"}
    assert {example.root_digest for example in examples[:3]} == {"sha256:" + "0" * 64}

    states = [TypedList.from_str(example.state) for example in examples[:3]]
    assert states == [
        TypedList([2, 3], item_type=int),
        TypedList([3, 4], item_type=int),
        TypedList([6, 8], item_type=int),
    ]
    target = TypedList.from_str(examples[0].target)
    assert (
        Executor(_basis()).execute(_basis().name_to_function["neg"], states[-1])
        == target
    )
    assert all(
        TypedList.from_str(example.state) != TypedList.from_str(example.target)
        for example in examples
    )


def _serialized(values, item_type=int) -> str:
    return TypedList(values, item_type=item_type).to_string()


def test_pair_batch_tracks_real_items_and_the_model_is_batch_invariant():
    small = (_serialized([1]), _serialized([2]))
    large = (_serialized([1, 22, 333]), _serialized([2, 44, 666]))
    sequences, counts, mask = pair_batch([small, large])
    assert counts.tolist() == [1, 3]
    assert mask.tolist() == [[True, False, False], [True, True, True]]
    assert sequences.shape[0] == 4

    torch.manual_seed(0)
    model = AlignedPairPolicy(5).eval()
    with torch.inference_mode():
        alone = model(*pair_batch([small]))[0]
        batched = model(*pair_batch([small, large]))[0]
    assert torch.allclose(alone, batched, atol=1e-5)
    assert (
        sum(parameter.numel() for parameter in AlignedPairPolicy(118).parameters())
        == 90_374
    )


def test_policy_size_arms_have_exact_checkpoint_native_dimensions():
    allocated = {
        name: sum(
            parameter.numel()
            for parameter in AlignedPairPolicy(118, architecture).parameters()
        )
        for name, architecture in POLICY_SIZE_ARMS.items()
    }
    expected = {
        "90k": 90_374,
        "1m": 724_022,
        "10m": 6_504_310,
        "40m": 25_329_270,
    }
    assert allocated == expected
    assert {
        name: architecture.parameter_count(118)
        for name, architecture in POLICY_SIZE_ARMS.items()
    } == expected


@pytest.mark.parametrize(
    ("arm_id", "batching"),
    training.POLICY_SIZE_BATCHING.items(),
)
def test_policy_size_cli_defaults_to_the_frozen_safe_batching(
    tmp_path: Path,
    arm_id: str,
    batching: tuple[int, int, int],
):
    config = training._parse_args(
        [
            "--output-dir",
            str(tmp_path / arm_id),
            "--size-arm",
            arm_id,
            "--dry-run",
        ]
    )

    assert (
        config.train_batch_size,
        config.gradient_accumulation_steps,
        config.validation_batch_size,
    ) == batching
    assert config.corpus_dir.name == "deep_corpus_v2"


def test_fixed_width_training_batch_preserves_logits_and_gradients():
    problems = [
        (_serialized([1]), _serialized([2])),
        (_serialized([1, 22, 333]), _serialized([2, 44, 666])),
    ]
    torch.manual_seed(7)
    dynamic = AlignedPairPolicy(5)
    fixed = AlignedPairPolicy(5)
    fixed.load_state_dict(dynamic.state_dict())

    dynamic_logits = dynamic(*pair_batch(problems))
    fixed_inputs = pair_batch(problems, sequence_width=MAX_PAIR_WIDTH)
    fixed_logits = fixed(*fixed_inputs)
    dynamic_logits.sum().backward()
    fixed_logits.sum().backward()

    assert fixed_inputs[0].shape[1] == MAX_PAIR_WIDTH == 193
    assert torch.allclose(dynamic_logits, fixed_logits, atol=1e-6)
    assert all(
        torch.allclose(dynamic_parameter.grad, fixed_parameter.grad, atol=1e-6)
        for dynamic_parameter, fixed_parameter in zip(
            dynamic.parameters(), fixed.parameters(), strict=True
        )
    )


def test_loaded_scorer_returns_one_finite_raw_logit_per_pinned_action():
    functions = _basis()
    torch.manual_seed(4)
    policy = RolloutPolicy.from_basis(AlignedPairPolicy(len(functions)), functions)
    problems = [
        (TypedList([1, 2], item_type=int), TypedList([2, 3], item_type=int)),
        (TypedList(["a", "b"], item_type=str), TypedList(["A", "B"], item_type=str)),
    ]

    scores = policy.score_actions_batch(problems, functions)

    expected_names = [function.name for function in functions]
    assert len(scores) == 2
    assert all(list(row) == expected_names for row in scores)
    assert all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in scores)
    # The scorer is intentionally unmasked; RolloutPredictor owns type filtering.
    assert "upper" in scores[0] and "inc" in scores[1]
    assert policy.score_actions_batch([], functions) == []

    reordered = FunctionDefSet(list(reversed(functions.functions)))
    with pytest.raises(ValueError, match="ordered basis"):
        policy.score_actions_batch(problems, reordered)


def _manifest(url: str, policy: RolloutPolicy) -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "basis_set": {
            "basis_set_id": policy.basis_set_id,
            "basis_set_digest": policy.basis_set_digest,
            "ordered_action_names": list(policy.action_names),
        },
        "source_corpus": {"manifest_digest": "sha256:" + "1" * 64},
        "wandb": {
            "entity": "abhishekraok-na",
            "project": "wandering-light-rollout-policy",
            "run_name": "test",
            "url": url,
        },
    }


def test_checkpoint_round_trip_is_weights_only_and_basis_pinned(tmp_path):
    functions = _basis()
    torch.manual_seed(7)
    policy = RolloutPolicy.from_basis(AlignedPairPolicy(len(functions)), functions)
    url = "https://wandb.ai/abhishekraok-na/wandering-light-rollout-policy/runs/test"
    checkpoint = tmp_path / "policy"
    policy.save(
        checkpoint,
        training_manifest=_manifest(url, policy),
        trainer_state={"best_epoch": 2, "best_validation_cross_entropy": 1.25},
        wandb_url=url,
    )

    assert {path.name for path in checkpoint.iterdir()} == {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINING_MANIFEST_FILE,
        TRAINER_STATE_FILE,
        WANDB_LINK_FILE,
    }
    assert_weights_only_checkpoint(checkpoint)
    restored = RolloutPolicy.load(checkpoint)
    problem = [(TypedList([1]), TypedList([2]))]
    before = policy.score_actions_batch(problem, functions)[0]
    after = restored.score_actions_batch(problem, functions)[0]
    assert after == pytest.approx(before, abs=0.0)

    (checkpoint / "optimizer.pt").write_bytes(b"disposable trainer state")
    with pytest.raises(ValueError, match="not weights-only"):
        assert_weights_only_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="not weights-only"):
        RolloutPolicy.load(checkpoint)


def test_nondefault_architecture_round_trips_without_runtime_global_mutation(tmp_path):
    functions = _basis()
    architecture = PolicyArchitecture(32, 40, 48)
    policy = RolloutPolicy.from_basis(
        AlignedPairPolicy(len(functions), architecture), functions
    )
    url = "https://wandb.ai/abhishekraok-na/wandering-light-rollout-policy/runs/wide"
    checkpoint = tmp_path / "wide"
    policy.save(
        checkpoint,
        training_manifest=_manifest(url, policy),
        trainer_state={"best_epoch": 1},
        wandb_url=url,
    )

    restored = RolloutPolicy.load(checkpoint)

    assert restored.model.architecture == architecture
    assert restored.config["architecture"] == architecture.to_config()
    problem = [(TypedList([1]), TypedList([2]))]
    assert restored.score_actions_batch(problem, functions)[0] == pytest.approx(
        policy.score_actions_batch(problem, functions)[0]
    )


def test_checkpoint_save_cleans_staging_directory_on_failure(tmp_path, monkeypatch):
    functions = _basis()
    policy = RolloutPolicy.from_basis(AlignedPairPolicy(len(functions)), functions)
    url = "https://wandb.ai/abhishekraok-na/wandering-light-rollout-policy/runs/test"
    checkpoint = tmp_path / "policy"

    def fail_wandb_link(path, value):
        del path, value
        raise RuntimeError("injected checkpoint failure")

    monkeypatch.setattr(rollout_policy_module, "_write_wandb_url", fail_wandb_link)
    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        policy.save(
            checkpoint,
            training_manifest=_manifest(url, policy),
            trainer_state={"best_epoch": 1},
            wandb_url=url,
        )

    assert not checkpoint.exists()
    assert list(tmp_path.glob(".policy.staging-*")) == []


def test_checkpoint_load_cross_checks_manifest_and_config_basis(tmp_path):
    functions = _basis()
    policy = RolloutPolicy.from_basis(AlignedPairPolicy(len(functions)), functions)
    url = "https://wandb.ai/abhishekraok-na/wandering-light-rollout-policy/runs/test"
    checkpoint = tmp_path / "policy"
    policy.save(
        checkpoint,
        training_manifest=_manifest(url, policy),
        trainer_state={"best_epoch": 1},
        wandb_url=url,
    )
    manifest_path = checkpoint / TRAINING_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["basis_set"]["ordered_action_names"][0] = "not-the-first-action"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest basis differs"):
        RolloutPolicy.load(checkpoint)


def test_source_identity_pins_head_diff_untracked_files_and_runner():
    identity = source_code_identity(Path(__file__).resolve().parents[1])

    assert len(identity["head_commit"]) == 40
    assert identity["tracked_diff_sha256"].startswith("sha256:")
    assert identity["untracked_files_digest"].startswith("sha256:")
    assert identity["runner"]["path"] == "experiments/train_rollout_policy.py"
    assert identity["runner"]["sha256"].startswith("sha256:")


def _example(
    state: int, target: int, action: str, *, split: str = "discovery"
) -> PrefixExample:
    return PrefixExample(
        state=_serialized([state]),
        target=_serialized([target]),
        action_name=action,
        split=split,
        task_id=f"{state}-{target}-{action}",
        root_digest="sha256:" + "2" * 64,
        prefix_index=0,
        remaining_distance=1,
        certified_distance=1,
    )


def test_training_and_validation_are_deterministic_for_one_seed():
    train = [
        _example(0, 1, "inc"),
        _example(1, 0, "dec"),
        _example(2, 3, "inc"),
        _example(3, 2, "dec"),
    ]
    validation = [
        _example(4, 5, "inc", split="validation"),
        _example(5, 4, "dec", split="validation"),
    ]
    options = {
        "epochs": 2,
        "learning_rate": 2e-3,
        "train_batch_size": 2,
        "validation_batch_size": 2,
        "seed": 19,
        "device": "cpu",
    }

    first, first_state = train_policy_model(
        train, validation, ["inc", "dec"], **options
    )
    second, second_state = train_policy_model(
        train, validation, ["inc", "dec"], **options
    )

    nondeterministic = {
        "training_seconds",
        "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes",
    }
    assert {
        key: value for key, value in first_state.items() if key not in nondeterministic
    } == {
        key: value for key, value in second_state.items() if key not in nondeterministic
    }
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_gradient_accumulation_preserves_effective_optimizer_step_count():
    train = [_example(index, index + 1, "inc") for index in range(5)]
    validation = [_example(7, 8, "inc", split="validation")]

    _, state = train_policy_model(
        train,
        validation,
        ["inc"],
        epochs=2,
        learning_rate=2e-3,
        train_batch_size=1,
        gradient_accumulation_steps=2,
        validation_batch_size=1,
        seed=3,
        device="cpu",
    )

    assert state["optimizer_steps"] == 6
    assert state["training_seconds"] > 0
    assert state["peak_cuda_memory_allocated_bytes"] is None


def test_training_rejects_prefix_examples_from_the_wrong_roles():
    train = [_example(0, 1, "inc")]
    validation = [_example(1, 2, "inc", split="validation")]
    options = {
        "epochs": 1,
        "learning_rate": 2e-3,
        "train_batch_size": 1,
        "validation_batch_size": 1,
        "seed": 0,
        "device": "cpu",
    }

    with pytest.raises(ValueError, match="Training PrefixExamples"):
        train_policy_model(
            [_example(0, 1, "inc", split="validation")],
            validation,
            ["inc"],
            **options,
        )
    with pytest.raises(ValueError, match="Validation PrefixExamples"):
        train_policy_model(train, [_example(1, 2, "inc")], ["inc"], **options)


def test_run_training_publishes_and_logs_exact_checkpoint(tmp_path, monkeypatch):
    functions = _basis()
    basis = policy_config(functions)["basis_set"]
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.json").write_text('{"fixture": true}\n', encoding="utf-8")
    corpus_manifest = {
        "manifest_digest": "sha256:" + "3" * 64,
        "basis_set_id": basis["basis_set_id"],
        "basis_set_digest": basis["basis_set_digest"],
        "hub": {"dataset": "fixture"},
        "split_policy": {"kind": "fixture"},
        "splits": {
            split: {
                "size": 1,
                "roots": 1,
                "sha256": f"sha256:{index:064x}",
            }
            for index, split in enumerate(("discovery", "validation", "test"), 1)
        },
    }
    prepared = PreparedPrefixes(
        train=(_example(0, 1, "inc"),),
        validation=(_example(1, 2, "inc", split="validation"),),
        manifest=corpus_manifest,
        verification={
            "ok": True,
            "witness_failures": [],
            "roots_leaking_across_splits": [],
        },
    )
    source_identity = {
        "head_commit": "a" * 40,
        "dirty": False,
        "tracked_diff_sha256": "sha256:" + "0" * 64,
        "untracked_files": [],
        "untracked_files_digest": "sha256:" + "0" * 64,
        "runner": {
            "path": "experiments/train_rollout_policy.py",
            "sha256": "sha256:" + "4" * 64,
        },
    }
    monkeypatch.setattr(
        training, "prepare_prefixes", lambda config: (functions, prepared)
    )
    monkeypatch.setattr(
        training, "source_code_identity", lambda repo_root: source_identity
    )

    optimizer = {
        "class": "torch.optim.Adam",
        "defaults": {
            "lr": 2e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
        },
    }

    def fake_train(train, validation, action_names, **kwargs):
        assert {example.split for example in train} == {"discovery"}
        assert {example.split for example in validation} == {"validation"}
        assert kwargs["device"] == "cpu"
        return AlignedPairPolicy(len(action_names)), {
            "best_epoch": 1,
            "best_validation_cross_entropy": 0.25,
            "epochs_completed": 1,
            "history": [],
            "train_prefixes": 1,
            "validation_prefixes": 1,
            "optimizer_steps": 1,
            "training_seconds": 0.5,
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
            "optimizer": optimizer,
        }

    monkeypatch.setattr(training, "train_policy_model", fake_train)

    class FakeArtifact:
        def __init__(self, **artifact_kwargs):
            self.name = artifact_kwargs["name"]
            self.type = artifact_kwargs["type"]
            self.metadata = artifact_kwargs["metadata"]
            self.files = []

        def add_file(self, path, *, name):
            assert Path(path).is_file()
            self.files.append(name)

    class FakeRun:
        id = "fixture-run"
        entity = "abhishekraok-na"
        url = (
            "https://wandb.ai/abhishekraok-na/"
            "wandering-light-rollout-policy/runs/fixture-run"
        )

        def __init__(self):
            self.summary = {}
            self.logged_artifacts = []
            self.finished = False

        def log(self, row, *, step):
            del row, step

        def log_artifact(self, artifact):
            self.logged_artifacts.append(artifact)

        def finish(self):
            self.finished = True

    run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: run,
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    checkpoint = tmp_path / "checkpoint"
    result = run_training(
        RolloutTrainingConfig(
            output_dir=checkpoint,
            corpus_dir=corpus_dir,
            epochs=1,
            device="cpu",
            wandb_run_name="fixture-run",
        )
    )

    persisted = json.loads(
        (checkpoint / TRAINING_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert result == persisted
    assert set(result["artifacts"]) == {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINER_STATE_FILE,
        WANDB_LINK_FILE,
    }
    assert result["runtime"]["precision"] == "fp32"
    assert result["runtime"]["model"]["parameters"] == 90_374
    assert result["runtime"]["optimizer"] == optimizer
    assert (
        "test payload never fetched, opened, or materialized"
        in result["source_corpus"]["task_roles"]["test"]["role"]
    )
    assert {path.name for path in checkpoint.iterdir()} == set(CHECKPOINT_FILES)
    assert list(tmp_path.glob(".checkpoint.staging-*")) == []

    assert run.finished
    assert len(run.logged_artifacts) == 1
    artifact = run.logged_artifacts[0]
    assert artifact.type == "model"
    assert artifact.name == "rollout-policy-checkpoint-fixture-run"
    assert artifact.files == list(CHECKPOINT_FILES)
    signature = run.summary["checkpoint_file_signature"]
    assert [row["name"] for row in signature] == list(CHECKPOINT_FILES)
    assert run.summary["checkpoint_training_manifest_sha256"] == next(
        row["sha256"] for row in signature if row["name"] == TRAINING_MANIFEST_FILE
    )
