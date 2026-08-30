"""Scratch-model contracts for the deep-corpus-v2 scaling study.

This module is deliberately independent of the training loop.  It freezes the
three architecture families, derives an effective configuration for the shared
tokenizer vocabulary, proves the resulting parameter budget, constructs models
from configuration only, and validates the learning-rate pilot selection.  The
heavy Transformers and Torch imports stay inside functions so the provenance
helpers remain cheap to test and import.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 1
PERSONAL_WANDB_ENTITY = "abhishekraok-na"
DEFAULT_WANDB_PROJECT = "wandering-light-deep-corpus-v2-scratch"
TOKENIZER_MAX_LENGTH = 2_048
LR_GRID = (1e-4, 3e-4, 1e-3)
PILOT_EPOCHS = 0.25
MAX_FINAL_EPOCHS = 3.0
INITIALIZATION_SEED = 54_001
DATA_ORDER_SEED = 54_002
FEASIBILITY_STEPS = 500
FEASIBILITY_SOFT_LIMIT_HOURS = 14.0
FEASIBILITY_HARD_LIMIT_HOURS = 16.0
PARAMETER_TOLERANCE_FRACTION = 0.01
MODEL_CREATION_MARKER = "_wandering_light_config_only_provenance"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":16:8"


def configure_deterministic_cuda_environment(
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Set CuBLAS's required deterministic workspace before Torch uses CUDA."""

    target = os.environ if environ is None else environ
    configured = target.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        configured = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        target["CUBLAS_WORKSPACE_CONFIG"] = configured
    if configured != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :16:8 for the frozen deterministic "
            "CUDA protocol"
        )
    return configured


@dataclass(frozen=True)
class ScratchArmSpec:
    """One predeclared architecture family and total parameter target."""

    arm_id: str
    label: str
    family: str
    config_repo: str
    config_revision: str
    target_parameters: int
    fixed_overrides: Mapping[str, Any]
    budget_dimension: str
    budget_minimum: int
    budget_maximum: int
    budget_multiple: int = 64


ARM_SPECS: dict[str, ScratchArmSpec] = {
    "scratch-10m": ScratchArmSpec(
        arm_id="scratch-10m",
        label="10M scratch causal LM",
        family="opt",
        config_repo="facebook/opt-125m",
        config_revision="27dcfa74d334bc871f3234de431e71c6eeba5dd6",
        target_parameters=10_000_000,
        fixed_overrides={
            "hidden_size": 256,
            "word_embed_proj_dim": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
        },
        budget_dimension="ffn_dim",
        budget_minimum=256,
        budget_maximum=8_192,
    ),
    "opt-125m-scratch": ScratchArmSpec(
        arm_id="opt-125m-scratch",
        label="OPT-style 125M scratch causal LM",
        family="opt",
        config_repo="facebook/opt-125m",
        config_revision="27dcfa74d334bc871f3234de431e71c6eeba5dd6",
        target_parameters=125_000_000,
        fixed_overrides={},
        budget_dimension="ffn_dim",
        budget_minimum=1_024,
        budget_maximum=8_192,
    ),
    "smollm2-360m-scratch": ScratchArmSpec(
        arm_id="smollm2-360m-scratch",
        label="SmolLM2-style 360M scratch causal LM",
        family="llama",
        config_repo="HuggingFaceTB/SmolLM2-360M",
        config_revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        target_parameters=360_000_000,
        fixed_overrides={},
        budget_dimension="intermediate_size",
        budget_minimum=1_024,
        budget_maximum=8_192,
    ),
}


@dataclass(frozen=True)
class ResolvedScratchConfig:
    """A config-only arm after shared-vocabulary parameter retargeting."""

    spec: ScratchArmSpec
    config: Any
    upstream_config_digest: str
    effective_config_digest: str
    exact_trainable_parameters: int
    parameter_error: int

    def identity_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.spec.arm_id,
            "label": self.spec.label,
            "family": self.spec.family,
            "config_source": {
                "repo_id": self.spec.config_repo,
                "revision": self.spec.config_revision,
                "digest": self.upstream_config_digest,
            },
            "effective_config_digest": self.effective_config_digest,
            "target_trainable_parameters": self.spec.target_parameters,
            "exact_trainable_parameters": self.exact_trainable_parameters,
            "parameter_error": self.parameter_error,
            "parameter_error_fraction": (
                self.parameter_error / self.spec.target_parameters
            ),
            "budget_dimension": self.spec.budget_dimension,
            "budget_dimension_value": getattr(self.config, self.spec.budget_dimension),
            "vocab_size": self.config.vocab_size,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def digest_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def build_scratch_study_identity(
    *,
    architecture: Mapping[str, Any],
    corpus_manifest_digest: str,
    tokenizer_artifact_digest: str,
    retained_training_task_ids_digest: str,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    optimizer: str,
    precision: str,
    gradient_checkpointing: bool,
    gradient_checkpointing_kwargs: Mapping[str, Any],
    cublas_workspace_config: str,
    pythonhashseed: str,
    initialization_seed: int,
    data_order_seed: int,
) -> dict[str, Any]:
    """Build the exact feasibility identity shared by probe, train, and eval."""

    return {
        "architecture": copy.deepcopy(dict(architecture)),
        "corpus_manifest_digest": corpus_manifest_digest,
        "tokenizer_artifact_digest": tokenizer_artifact_digest,
        "retained_training_task_ids_digest": retained_training_task_ids_digest,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "optimizer": optimizer,
        "precision": precision,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": copy.deepcopy(
            dict(gradient_checkpointing_kwargs)
        ),
        "cublas_workspace_config": cublas_workspace_config,
        "pythonhashseed": pythonhashseed,
        "initialization_seed": initialization_seed,
        "data_order_seed": data_order_seed,
    }


def _config_dict(config: Any) -> dict[str, Any]:
    value = config.to_dict()
    if not isinstance(value, dict):
        raise TypeError("Transformers config.to_dict() did not return a mapping")
    return value


def estimate_unique_trainable_parameters(config: Any) -> int:
    """Return the exact tied-weight parameter count for supported families.

    Both study families tie the language-model head to the token embedding.  The
    closed forms below count that shared parameter once, matching
    ``sum(p.numel() for p in model.parameters())``.  Runtime construction checks
    the formula against the actual model before any optimizer can see it.
    """

    model_type = getattr(config, "model_type", None)
    if getattr(config, "tie_word_embeddings", None) is not True:
        raise ValueError("The scratch study requires tied token/output embeddings")
    vocabulary = int(config.vocab_size)
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    if min(vocabulary, hidden, layers) <= 0:
        raise ValueError("Config dimensions must be positive")

    if model_type == "opt":
        if int(config.word_embed_proj_dim) != hidden:
            raise ValueError("The scratch OPT arms require unprojected embeddings")
        feed_forward = int(config.ffn_dim)
        positions = int(config.max_position_embeddings)
        # tokens + learned positions + final layer norm; each decoder layer has
        # q/k/v/out projections and biases, two norms, and the two-layer MLP.
        return (
            vocabulary * hidden
            + (positions + 2) * hidden
            + 2 * hidden
            + layers
            * (
                4 * hidden * hidden
                + 4 * hidden
                + 2 * hidden
                + 2 * hidden * feed_forward
                + feed_forward
                + hidden
                + 2 * hidden
            )
        )

    if model_type == "llama":
        intermediate = int(config.intermediate_size)
        head_dim = int(
            getattr(config, "head_dim", hidden // config.num_attention_heads)
        )
        key_value_width = int(config.num_key_value_heads) * head_dim
        # Token embedding + final RMSNorm; each block has q/o at hidden width,
        # k/v at grouped-query width, three MLP matrices, and two RMSNorms.
        return (
            vocabulary * hidden
            + layers
            * (
                2 * hidden * hidden
                + 2 * hidden * key_value_width
                + 3 * hidden * intermediate
                + 2 * hidden
            )
            + hidden
        )

    raise ValueError(f"Unsupported scratch architecture family: {model_type!r}")


def _candidate_values(spec: ScratchArmSpec) -> range:
    if spec.budget_multiple <= 0:
        raise ValueError("budget_multiple must be positive")
    start = math.ceil(spec.budget_minimum / spec.budget_multiple) * spec.budget_multiple
    return range(start, spec.budget_maximum + 1, spec.budget_multiple)


def resolve_scratch_config(
    arm_id: str,
    *,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    auto_config_class: Any | None = None,
) -> ResolvedScratchConfig:
    """Load architecture metadata only and retarget it to the shared vocabulary."""

    if arm_id not in ARM_SPECS:
        raise ValueError(f"Unknown scratch arm: {arm_id!r}")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    token_ids = (bos_token_id, eos_token_id, pad_token_id)
    if any(
        type(token_id) is not int or not 0 <= token_id < vocab_size
        for token_id in token_ids
    ):
        raise ValueError("Special-token IDs must be distinct valid vocabulary IDs")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("BOS, EOS, and PAD IDs must be distinct")

    if auto_config_class is None:
        from transformers import AutoConfig

        auto_config_class = AutoConfig
    spec = ARM_SPECS[arm_id]
    upstream = auto_config_class.from_pretrained(
        spec.config_repo,
        revision=spec.config_revision,
    )
    if getattr(upstream, "model_type", None) != spec.family:
        raise ValueError(
            f"Pinned {spec.config_repo} config is {upstream.model_type!r}, "
            f"not required family {spec.family!r}"
        )
    upstream_digest = digest_json(_config_dict(upstream))
    effective = copy.deepcopy(upstream)
    for name, value in spec.fixed_overrides.items():
        setattr(effective, name, value)
    effective.vocab_size = vocab_size
    effective.bos_token_id = bos_token_id
    effective.eos_token_id = eos_token_id
    effective.pad_token_id = pad_token_id
    effective.tie_word_embeddings = True
    effective.use_cache = False

    choices: list[tuple[int, int, int]] = []
    for candidate in _candidate_values(spec):
        setattr(effective, spec.budget_dimension, candidate)
        count = estimate_unique_trainable_parameters(effective)
        choices.append((abs(count - spec.target_parameters), candidate, count))
    error, selected, exact = min(choices)
    setattr(effective, spec.budget_dimension, selected)
    if error / spec.target_parameters > PARAMETER_TOLERANCE_FRACTION:
        raise ValueError(
            f"Could not place {arm_id} within parameter tolerance: "
            f"target={spec.target_parameters:,}, closest={exact:,}"
        )
    return ResolvedScratchConfig(
        spec=spec,
        config=effective,
        upstream_config_digest=upstream_digest,
        effective_config_digest=digest_json(_config_dict(effective)),
        exact_trainable_parameters=exact,
        parameter_error=error,
    )


def exact_model_parameter_count(model: Any) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def hash_model_state(model: Any) -> str:
    """Incrementally hash the exact pristine state without a giant copy."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(canonical_json(header).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            value.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()
        )
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def initialize_model_from_config(
    resolved: ResolvedScratchConfig,
    *,
    initialization_seed: int = INITIALIZATION_SEED,
    torch_dtype: Any | None = None,
    auto_model_class: Any | None = None,
    set_seed_function: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create one pristine model without permitting an upstream weight load."""

    if auto_model_class is None or set_seed_function is None:
        from transformers import AutoModelForCausalLM, set_seed

        auto_model_class = auto_model_class or AutoModelForCausalLM
        set_seed_function = set_seed_function or set_seed
    set_seed_function(initialization_seed, deterministic=True)
    keyword_arguments: dict[str, Any] = {"attn_implementation": "sdpa"}
    if torch_dtype is not None:
        keyword_arguments["torch_dtype"] = torch_dtype
    model = auto_model_class.from_config(resolved.config, **keyword_arguments)
    exact = exact_model_parameter_count(model)
    if exact != resolved.exact_trainable_parameters:
        raise RuntimeError(
            "Instantiated parameter count differs from the frozen architecture "
            f"calculation: model={exact:,}, expected={resolved.exact_trainable_parameters:,}"
        )
    provenance = {
        **resolved.identity_dict(),
        "creation_method": "AutoModelForCausalLM.from_config",
        "pretrained_weights_loaded": False,
        "initialization_seed": initialization_seed,
        "initial_state_digest": hash_model_state(model),
    }
    setattr(model, MODEL_CREATION_MARKER, dict(provenance))
    return model, provenance


def assert_config_created_model(model: Any, resolved: ResolvedScratchConfig) -> None:
    marker = getattr(model, MODEL_CREATION_MARKER, None)
    if not isinstance(marker, dict):
        raise RuntimeError("Trainer model lacks config-only creation provenance")
    if marker.get("arm_id") != resolved.spec.arm_id:
        raise RuntimeError("Trainer model was created for a different scratch arm")
    if marker.get("effective_config_digest") != resolved.effective_config_digest:
        raise RuntimeError(
            "Trainer model config differs from the frozen effective config"
        )
    if marker.get("pretrained_weights_loaded") is not False:
        raise RuntimeError("Trainer model provenance permits pretrained weights")


def _manifest_field(manifest: Mapping[str, Any], name: str) -> Any:
    if name not in manifest:
        raise ValueError(f"Pilot manifest is missing {name!r}")
    return manifest[name]


def select_pilot_learning_rate(
    arm_id: str,
    pilot_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Lock one LR using validation loss only across the predeclared grid."""

    if arm_id not in ARM_SPECS:
        raise ValueError(f"Unknown scratch arm: {arm_id!r}")
    if len(pilot_manifests) != len(LR_GRID):
        raise ValueError(
            "Exactly one quarter-epoch pilot per predeclared LR is required"
        )
    by_rate: dict[float, Mapping[str, Any]] = {}
    shared_fields = (
        "initial_state_digest",
        "initialization_seed",
        "data_order_seed",
        "tokenizer_digest",
        "retained_training_task_ids_digest",
        "effective_config_digest",
        "exact_trainable_parameters",
    )
    expected_shared: dict[str, Any] | None = None
    for manifest in pilot_manifests:
        if _manifest_field(manifest, "status") != "complete":
            raise ValueError("Every pilot must be complete before LR selection")
        if _manifest_field(manifest, "phase") != "pilot":
            raise ValueError("LR selection accepts pilot manifests only")
        if _manifest_field(manifest, "arm_id") != arm_id:
            raise ValueError("Pilot manifests must all belong to the selected arm")
        if float(_manifest_field(manifest, "num_train_epochs")) != PILOT_EPOCHS:
            raise ValueError("Every pilot must use exactly one quarter epoch")
        rate = float(_manifest_field(manifest, "learning_rate"))
        if rate in by_rate:
            raise ValueError(f"Duplicate pilot learning rate: {rate}")
        validation_loss = _manifest_field(manifest, "best_validation_loss")
        if not isinstance(validation_loss, int | float) or not math.isfinite(
            float(validation_loss)
        ):
            raise ValueError("Pilot validation loss must be finite")
        by_rate[rate] = manifest
        shared = {field: _manifest_field(manifest, field) for field in shared_fields}
        if expected_shared is None:
            expected_shared = shared
        elif shared != expected_shared:
            raise ValueError(
                "Pilot initialization, data order, tokenizer, task set, config, "
                "and parameter count must be identical"
            )
    if set(by_rate) != set(LR_GRID):
        raise ValueError(f"Pilot learning rates must be exactly {LR_GRID}")

    selected_rate, selected = min(
        by_rate.items(),
        key=lambda item: (float(item[1]["best_validation_loss"]), item[0]),
    )
    lock: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked-before-final-training",
        "selection_metric": "minimum validation loss; lower learning rate breaks ties",
        "arm_id": arm_id,
        "learning_rate_grid": list(LR_GRID),
        "selected_learning_rate": selected_rate,
        "selected_pilot_manifest_digest": digest_json(dict(selected)),
        "pilots": [
            {
                "learning_rate": rate,
                "best_validation_loss": float(by_rate[rate]["best_validation_loss"]),
                "manifest_digest": digest_json(dict(by_rate[rate])),
            }
            for rate in LR_GRID
        ],
        **(expected_shared or {}),
    }
    lock["lock_digest"] = digest_json(lock)
    return lock


def validate_final_restart(
    *,
    selection_lock: Mapping[str, Any],
    arm_id: str,
    learning_rate: float,
    initialization_provenance: Mapping[str, Any],
    num_train_epochs: float,
) -> None:
    """Fail closed unless the final run is a pristine selected-seed restart."""

    payload = dict(selection_lock)
    stored_digest = payload.pop("lock_digest", None)
    if stored_digest != digest_json(payload):
        raise ValueError("Learning-rate selection lock digest mismatch")
    if selection_lock.get("status") != "locked-before-final-training":
        raise ValueError("Learning-rate selection is not locked")
    if selection_lock.get("arm_id") != arm_id:
        raise ValueError("Selection lock belongs to another arm")
    if float(selection_lock.get("selected_learning_rate", -1)) != learning_rate:
        raise ValueError("Final run must use the validation-selected learning rate")
    if not 0 < num_train_epochs <= MAX_FINAL_EPOCHS:
        raise ValueError(f"Final training may use at most {MAX_FINAL_EPOCHS:g} epochs")
    comparisons = {
        "initial_state_digest": initialization_provenance.get("initial_state_digest"),
        "initialization_seed": initialization_provenance.get("initialization_seed"),
        "effective_config_digest": initialization_provenance.get(
            "effective_config_digest"
        ),
        "exact_trainable_parameters": initialization_provenance.get(
            "exact_trainable_parameters"
        ),
    }
    for name, actual in comparisons.items():
        if selection_lock.get(name) != actual:
            raise ValueError(f"Final run is not a pristine pilot restart: {name}")


def classify_validation_trend(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify the observed validation curve without overstating evidence."""

    if not evaluations:
        raise ValueError("At least one validation evaluation is required")
    losses = [float(row["eval_loss"]) for row in evaluations]
    if any(not math.isfinite(loss) for loss in losses):
        raise ValueError("Validation losses must be finite")
    best_index = min(range(len(losses)), key=losses.__getitem__)
    turned_up = any(
        loss > min(losses[:index]) for index, loss in enumerate(losses) if index > 0
    )
    if turned_up:
        label = "overfitting"
    elif best_index == len(losses) - 1 and len(losses) > 1:
        label = "compute-limited"
    else:
        label = "insufficient-curve" if len(losses) == 1 else "plateaued"
    return {
        "classification": label,
        "best_evaluation_index": best_index,
        "best_validation_loss": losses[best_index],
        "validation_still_improving_at_cap": label == "compute-limited",
        "validation_turned_up": turned_up,
    }


def feasibility_decision(
    *,
    measured_steps: int,
    measured_seconds: float,
    projected_total_steps: int,
    peak_physical_vram_bytes: int,
    physical_vram_bytes: int,
) -> dict[str, Any]:
    """Apply the predeclared one-4090 runtime gate without changing protocol."""

    if measured_steps != FEASIBILITY_STEPS:
        raise ValueError(
            f"Feasibility probe must run exactly {FEASIBILITY_STEPS} steps"
        )
    if measured_seconds <= 0 or projected_total_steps <= 0:
        raise ValueError("Probe seconds and projected steps must be positive")
    if not 0 < peak_physical_vram_bytes <= physical_vram_bytes:
        raise ValueError("Peak VRAM must be positive and fit the physical device")
    seconds_per_step = measured_seconds / measured_steps
    projected_hours = seconds_per_step * projected_total_steps / 3600
    accepted = projected_hours <= FEASIBILITY_SOFT_LIMIT_HOURS
    return {
        "measured_steps": measured_steps,
        "measured_seconds": measured_seconds,
        "seconds_per_step": seconds_per_step,
        "projected_total_steps": projected_total_steps,
        "projected_pilot_plus_final_hours": projected_hours,
        "soft_limit_hours": FEASIBILITY_SOFT_LIMIT_HOURS,
        "hard_limit_hours": FEASIBILITY_HARD_LIMIT_HOURS,
        "peak_physical_vram_bytes": peak_physical_vram_bytes,
        "physical_vram_bytes": physical_vram_bytes,
        "vram_headroom_bytes": physical_vram_bytes - peak_physical_vram_bytes,
        "accepted": accepted,
        "rejection_reason": None if accepted else "projected runtime exceeds 14 hours",
    }
