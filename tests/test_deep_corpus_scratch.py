"""Tests for the random-initialization and model-budget contract."""

from __future__ import annotations

import copy
from typing import ClassVar

import pytest
from transformers import LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM

from wandering_light.training import deep_corpus_scratch as scratch


def test_deterministic_cuda_environment_is_frozen_before_torch_cuda():
    environment: dict[str, str] = {}
    assert scratch.configure_deterministic_cuda_environment(environment) == ":16:8"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert scratch.configure_deterministic_cuda_environment(environment) == ":16:8"

    with pytest.raises(RuntimeError, match="frozen deterministic CUDA protocol"):
        scratch.configure_deterministic_cuda_environment(
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        )


def test_parameter_formulas_match_actual_tied_models():
    opt_config = OPTConfig(
        vocab_size=123,
        hidden_size=64,
        word_embed_proj_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=128,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    llama_config = LlamaConfig(
        vocab_size=123,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        head_dim=16,
        tie_word_embeddings=True,
    )
    for config, model_class in (
        (opt_config, OPTForCausalLM),
        (llama_config, LlamaForCausalLM),
    ):
        model = model_class(config)
        assert scratch.estimate_unique_trainable_parameters(config) == sum(
            parameter.numel() for parameter in model.parameters()
        )


class _FakeAutoConfig:
    configs: ClassVar = {
        "facebook/opt-125m": OPTConfig(
            vocab_size=50_272,
            hidden_size=768,
            word_embed_proj_dim=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            ffn_dim=3_072,
            max_position_embeddings=2_048,
            tie_word_embeddings=True,
        ),
        "HuggingFaceTB/SmolLM2-360M": LlamaConfig(
            vocab_size=49_152,
            hidden_size=960,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            intermediate_size=2_560,
            max_position_embeddings=8_192,
            head_dim=64,
            tie_word_embeddings=True,
        ),
    }

    @classmethod
    def from_pretrained(cls, repo_id, *, revision):
        assert (
            revision
            == scratch.ARM_SPECS[
                next(
                    key
                    for key, spec in scratch.ARM_SPECS.items()
                    if spec.config_repo == repo_id
                )
            ].config_revision
        )
        return copy.deepcopy(cls.configs[repo_id])


@pytest.mark.parametrize("arm_id", sorted(scratch.ARM_SPECS))
def test_shared_vocabulary_configs_hit_predeclared_total_budgets(arm_id):
    resolved = scratch.resolve_scratch_config(
        arm_id,
        vocab_size=4_192,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        auto_config_class=_FakeAutoConfig,
    )
    assert resolved.config.vocab_size == 4_192
    assert resolved.parameter_error / resolved.spec.target_parameters <= 0.01
    assert resolved.exact_trainable_parameters == (
        scratch.estimate_unique_trainable_parameters(resolved.config)
    )


def test_model_initialization_uses_from_config_and_hashes_pristine_state():
    resolved = scratch.resolve_scratch_config(
        "scratch-10m",
        vocab_size=512,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        auto_config_class=_FakeAutoConfig,
    )
    # Keep this unit test small while retaining the exact resolved identity.
    resolved.config.hidden_size = 32
    resolved.config.word_embed_proj_dim = 32
    resolved.config.num_hidden_layers = 1
    resolved.config.num_attention_heads = 4
    resolved.config.ffn_dim = 64
    exact = scratch.estimate_unique_trainable_parameters(resolved.config)
    resolved = scratch.ResolvedScratchConfig(
        spec=resolved.spec,
        config=resolved.config,
        upstream_config_digest=resolved.upstream_config_digest,
        effective_config_digest=scratch.digest_json(resolved.config.to_dict()),
        exact_trainable_parameters=exact,
        parameter_error=abs(exact - resolved.spec.target_parameters),
    )

    class Factory:
        from_pretrained = staticmethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("pretrained weights must never be loaded")
            )
        )

        @staticmethod
        def from_config(config, **kwargs):
            assert kwargs["attn_implementation"] == "sdpa"
            return OPTForCausalLM(config)

    first, first_provenance = scratch.initialize_model_from_config(
        resolved,
        initialization_seed=19,
        auto_model_class=Factory,
        set_seed_function=lambda seed, deterministic: __import__("torch").manual_seed(
            seed
        ),
    )
    second, second_provenance = scratch.initialize_model_from_config(
        resolved,
        initialization_seed=19,
        auto_model_class=Factory,
        set_seed_function=lambda seed, deterministic: __import__("torch").manual_seed(
            seed
        ),
    )
    assert first_provenance["creation_method"].endswith("from_config")
    assert first_provenance["pretrained_weights_loaded"] is False
    assert (
        first_provenance["initial_state_digest"]
        == second_provenance["initial_state_digest"]
    )
    scratch.assert_config_created_model(first, resolved)
    del first, second


def _pilot(rate, loss, **overrides):
    value = {
        "status": "complete",
        "phase": "pilot",
        "arm_id": "scratch-10m",
        "num_train_epochs": 0.25,
        "learning_rate": rate,
        "best_validation_loss": loss,
        "initial_state_digest": "sha256:" + "1" * 64,
        "initialization_seed": 54_001,
        "data_order_seed": 54_002,
        "tokenizer_digest": "sha256:" + "2" * 64,
        "retained_training_task_ids_digest": "sha256:" + "3" * 64,
        "effective_config_digest": "sha256:" + "4" * 64,
        "exact_trainable_parameters": 10_000_001,
    }
    value.update(overrides)
    return value


def test_pilot_selection_is_validation_only_and_final_must_restart_pristine():
    pilots = [
        _pilot(1e-4, 0.8),
        _pilot(3e-4, 0.6),
        _pilot(1e-3, 0.7),
    ]
    lock = scratch.select_pilot_learning_rate("scratch-10m", pilots)
    assert lock["selected_learning_rate"] == 3e-4
    scratch.validate_final_restart(
        selection_lock=lock,
        arm_id="scratch-10m",
        learning_rate=3e-4,
        initialization_provenance={
            "initial_state_digest": "sha256:" + "1" * 64,
            "initialization_seed": 54_001,
            "effective_config_digest": "sha256:" + "4" * 64,
            "exact_trainable_parameters": 10_000_001,
        },
        num_train_epochs=3,
    )
    with pytest.raises(ValueError, match="exactly 3 epochs"):
        scratch.validate_final_restart(
            selection_lock=lock,
            arm_id="scratch-10m",
            learning_rate=3e-4,
            initialization_provenance={
                "initial_state_digest": "sha256:" + "1" * 64,
                "initialization_seed": 54_001,
                "effective_config_digest": "sha256:" + "4" * 64,
                "exact_trainable_parameters": 10_000_001,
            },
            num_train_epochs=2.9,
        )
    with pytest.raises(ValueError, match="validation-selected"):
        scratch.validate_final_restart(
            selection_lock=lock,
            arm_id="scratch-10m",
            learning_rate=1e-3,
            initialization_provenance={},
            num_train_epochs=3,
        )


def test_pilot_selection_rejects_initialization_drift():
    pilots = [
        _pilot(1e-4, 0.8),
        _pilot(3e-4, 0.6),
        _pilot(1e-3, 0.7, initial_state_digest="sha256:" + "9" * 64),
    ]
    with pytest.raises(ValueError, match="must be identical"):
        scratch.select_pilot_learning_rate("scratch-10m", pilots)


def test_validation_trend_and_feasibility_gate_are_fail_closed():
    assert (
        scratch.classify_validation_trend(
            [{"eval_loss": 1.0}, {"eval_loss": 0.8}, {"eval_loss": 0.7}]
        )["classification"]
        == "compute-limited"
    )
    assert (
        scratch.classify_validation_trend(
            [{"eval_loss": 1.0}, {"eval_loss": 0.8}, {"eval_loss": 0.9}]
        )["classification"]
        == "overfitting"
    )

    accepted = scratch.feasibility_decision(
        measured_steps=500,
        measured_seconds=500,
        projected_total_steps=10_000,
        peak_physical_vram_bytes=20,
        physical_vram_bytes=24,
    )
    assert accepted["accepted"]
    rejected = scratch.feasibility_decision(
        measured_steps=500,
        measured_seconds=3_000,
        projected_total_steps=10_000,
        peak_physical_vram_bytes=20,
        physical_vram_bytes=24,
    )
    assert not rejected["accepted"]
