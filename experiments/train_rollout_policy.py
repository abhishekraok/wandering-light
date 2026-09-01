"""Train a one-step policy on every certified witness prefix.

Each corpus record stays in its frozen root-level split.  Replaying a shortest
witness ``w_1..w_d`` yields ``d`` supervised rows
``(s_j, target) -> w_{j+1}``; the final target state is excluded because rollout
performs a goal test before asking the policy for another action.

The real run selects weights by validation cross-entropy and writes one
weights-only checkpoint: model/config, trainer metadata, corpus and basis
lineage, and a personal W&B run URL.  Optimizer, scheduler, and RNG state are
deliberately disposable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generate_deep_corpus import verify_corpus
from wandering_light.basis_set import load_basis_set
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.executor import Executor
from wandering_light.rollout_policy import (
    CONFIG_FILE,
    DEFAULT_POLICY_ARCHITECTURE,
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

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.function_def import FunctionDefSet

SCHEMA_VERSION = 1
PERSONAL_WANDB_ENTITY = "abhishekraok-na"
DEFAULT_WANDB_PROJECT = "wandering-light-rollout-policy"
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v2")
TRAIN_SPLIT = "discovery"
VALIDATION_SPLIT = "validation"
HELD_OUT_SPLIT = "test"
POLICY_SIZE_ARMS = {
    "90k": DEFAULT_POLICY_ARCHITECTURE,
    "1m": PolicyArchitecture(128, 320, 640),
    "10m": PolicyArchitecture(384, 1024, 2048),
    "40m": PolicyArchitecture(768, 2048, 4096),
}
POLICY_SIZE_BATCHING = {
    "90k": (256, 1, 512),
    "1m": (256, 1, 512),
    "10m": (256, 1, 256),
    "40m": (128, 2, 128),
}
CHECKPOINT_FILES = (
    MODEL_FILE,
    CONFIG_FILE,
    TRAINING_MANIFEST_FILE,
    TRAINER_STATE_FILE,
    WANDB_LINK_FILE,
)


@dataclass(frozen=True)
class PrefixExample:
    """One exactly on-path next-action label."""

    state: str
    target: str
    action_name: str
    split: str
    task_id: str
    root_digest: str
    prefix_index: int
    remaining_distance: int
    certified_distance: int


@dataclass(frozen=True)
class RolloutTrainingConfig:
    output_dir: Path
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    epochs: int = 12
    learning_rate: float = 2e-3
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    validation_batch_size: int = 512
    seed: int = 0
    device: str = "cuda"
    wandb_entity: str = PERSONAL_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_run_name: str | None = None
    architecture: PolicyArchitecture = DEFAULT_POLICY_ARCHITECTURE
    dry_run: bool = False


@dataclass(frozen=True)
class PreparedPrefixes:
    train: tuple[PrefixExample, ...]
    validation: tuple[PrefixExample, ...]
    manifest: Mapping[str, Any]
    verification: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def source_code_identity(repo_root: Path) -> dict[str, Any]:
    """Content-address committed, tracked-dirty, and untracked training source."""

    def run_git(*arguments: str, text: bool = True) -> str | bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
        )
        return completed.stdout

    try:
        commit = str(run_git("rev-parse", "HEAD")).strip()
        tracked_diff = str(run_git("diff", "--binary", "HEAD"))
        raw_untracked = bytes(
            run_git("ls-files", "--others", "--exclude-standard", "-z", text=False)
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not record rollout-policy source identity") from error
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"Git returned an invalid HEAD commit: {commit!r}")

    untracked_files = []
    for raw_path in sorted(filter(None, raw_untracked.split(b"\0"))):
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("Untracked git path is not valid UTF-8") from error
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Source identity supports only regular untracked files: {relative}"
            )
        untracked_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    runner = Path(__file__).resolve()
    return {
        "head_commit": commit,
        "dirty": bool(tracked_diff or untracked_files),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff.encode("utf-8")),
        "untracked_files": untracked_files,
        "untracked_files_digest": _sha256_bytes(
            _canonical_json(untracked_files).encode("utf-8")
        ),
        "runner": {
            "path": runner.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(runner),
        },
    }


def _type_name(value: TypedList) -> str:
    item_type = value.item_type
    return f"{item_type.__module__}.{item_type.__qualname__}"


def build_prefix_examples(
    corpus_dir: Path,
    split: str,
    available_functions: FunctionDefSet,
    *,
    limit: int | None = None,
) -> list[PrefixExample]:
    """Replay every witness prefix without moving any row across splits."""

    if split not in {TRAIN_SPLIT, VALIDATION_SPLIT, HELD_OUT_SPLIT}:
        raise ValueError(f"Unknown deep-corpus split: {split!r}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")

    by_name = available_functions.name_to_function
    executor = Executor(available_functions)
    examples = []
    with gzip.open(corpus_dir / f"{split}.jsonl.gz", "rt", encoding="utf-8") as file:
        for record_index, line in enumerate(file):
            if limit is not None and record_index >= limit:
                break
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError(
                    f"Corpus row {row.get('task_id')} belongs to {row.get('split')}, "
                    f"not {split}"
                )
            names = row.get("witness_function_names")
            distance = row.get("metadata", {}).get("certified_distance")
            if (
                not isinstance(names, list)
                or not names
                or type(distance) is not int
                or distance != len(names)
                or row.get("witness_length") != distance
            ):
                raise ValueError(f"Task {row.get('task_id')} has an invalid witness")
            root_digest = row.get("metadata", {}).get("root_digest")
            if not isinstance(root_digest, str) or not root_digest:
                raise ValueError(f"Task {row.get('task_id')} has no root digest")

            state = TypedList.from_str(row["input"])
            target = TypedList.from_str(row["output"])
            for prefix_index, name in enumerate(names):
                function = by_name.get(name)
                if function is None:
                    raise ValueError(
                        f"Task {row.get('task_id')} references unknown action {name!r}"
                    )
                if function.input_type != _type_name(state):
                    raise ValueError(
                        f"Task {row.get('task_id')} action {name!r} is incompatible "
                        f"with prefix {prefix_index}"
                    )
                examples.append(
                    PrefixExample(
                        state=state.to_string(),
                        target=row["output"],
                        action_name=name,
                        split=split,
                        task_id=row["task_id"],
                        root_digest=root_digest,
                        prefix_index=prefix_index,
                        remaining_distance=distance - prefix_index,
                        certified_distance=distance,
                    )
                )
                state = executor.execute(function, state)
            if state != target:
                raise ValueError(
                    f"Task {row.get('task_id')} witness replay missed its target"
                )
    return examples


def _expected_prefixes(manifest: Mapping[str, Any], split: str) -> int:
    histogram = manifest["splits"][split]["by_certified_distance"]
    return sum(int(distance) * int(count) for distance, count in histogram.items())


def ensure_training_splits(corpus_dir: Path) -> Path:
    """Materialize discovery and validation without touching held-out test."""

    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deep-corpus manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = (TRAIN_SPLIT, VALIDATION_SPLIT)
    missing = [
        corpus_dir / manifest["splits"][split]["path"]
        for split in required
        if not (corpus_dir / manifest["splits"][split]["path"]).is_file()
    ]
    if missing:
        fetch_corpus(
            manifest_path,
            destination=corpus_dir,
            splits=required,
            include_details=False,
        )
    still_missing = [path for path in missing if not path.is_file()]
    if still_missing:
        raise FileNotFoundError(
            "Corpus fetch did not materialize training splits: "
            f"{[str(path) for path in still_missing]}"
        )
    return corpus_dir


def prepare_prefixes(
    config: RolloutTrainingConfig,
) -> tuple[FunctionDefSet, PreparedPrefixes]:
    """Verify corpus/basis/root boundaries, then expose discovery and validation."""

    ensure_training_splits(config.corpus_dir)
    manifest_path = config.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    basis = load_basis_set(manifest["basis_set_id"])
    if manifest.get("basis_set_digest") != basis.digest:
        raise ValueError("Corpus manifest and loaded basis digests differ")
    available_functions = basis.to_function_def_set()

    verification = verify_corpus(
        config.corpus_dir,
        splits=(TRAIN_SPLIT, VALIDATION_SPLIT),
        recertify=0,
    )
    if not verification.get("ok"):
        raise ValueError(f"Deep corpus verification failed: {verification}")
    if verification.get("roots_leaking_across_splits"):
        raise ValueError("Deep corpus has roots leaking across splits")

    train = build_prefix_examples(config.corpus_dir, TRAIN_SPLIT, available_functions)
    validation = build_prefix_examples(
        config.corpus_dir, VALIDATION_SPLIT, available_functions
    )
    expected = {
        TRAIN_SPLIT: _expected_prefixes(manifest, TRAIN_SPLIT),
        VALIDATION_SPLIT: _expected_prefixes(manifest, VALIDATION_SPLIT),
    }
    if (
        len(train) != expected[TRAIN_SPLIT]
        or len(validation) != expected[VALIDATION_SPLIT]
    ):
        raise ValueError(
            "Materialized prefix counts differ from the frozen distance histogram: "
            f"actual={(len(train), len(validation))}, expected={expected}"
        )
    return available_functions, PreparedPrefixes(
        train=tuple(train),
        validation=tuple(validation),
        manifest=manifest,
        verification=verification,
    )


def configure_determinism(seed: int) -> None:
    """Pin model initialization, data order, and deterministic torch kernels."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _examples_batch(
    examples: Sequence[PrefixExample],
    action_index: Mapping[str, int],
    *,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    inputs = pair_batch(
        [(example.state, example.target) for example in examples],
        device=device,
        sequence_width=MAX_PAIR_WIDTH,
    )
    labels = torch.tensor(
        [action_index[example.action_name] for example in examples],
        dtype=torch.long,
        device=device,
    )
    return inputs, labels


def train_policy_model(
    train: Sequence[PrefixExample],
    validation: Sequence[PrefixExample],
    action_names: Sequence[str],
    *,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    gradient_accumulation_steps: int = 1,
    validation_batch_size: int,
    seed: int,
    device: str | torch.device,
    architecture: PolicyArchitecture = DEFAULT_POLICY_ARCHITECTURE,
    epoch_callback=None,
) -> tuple[AlignedPairPolicy, dict[str, Any]]:
    """Train deterministically and restore the lowest validation-loss weights."""

    if not train or not validation:
        raise ValueError("Training and validation prefix sets must both be non-empty")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if train_batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if len(action_names) != len(set(action_names)) or not action_names:
        raise ValueError("action_names must be a non-empty unique sequence")
    if any(example.split != TRAIN_SPLIT for example in train):
        raise ValueError("Training PrefixExamples must all belong to discovery")
    if any(example.split != VALIDATION_SPLIT for example in validation):
        raise ValueError("Validation PrefixExamples must all belong to validation")

    configure_determinism(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA rollout-policy training requested without CUDA")
    action_index = {name: index for index, name in enumerate(action_names)}
    unknown = sorted(
        {example.action_name for example in [*train, *validation]} - set(action_index)
    )
    if unknown:
        raise ValueError(f"Prefix labels reference unknown actions: {unknown}")

    model = AlignedPairPolicy(len(action_names), architecture).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_identity = {
        "class": "torch.optim.Adam",
        "defaults": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in optimizer.defaults.items()
        },
    }
    order = list(range(len(train)))
    rng = random.Random(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    optimizer_steps = 0
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        train_total = torch.zeros((), device=target_device)
        for group_start in range(0, len(order), effective_batch_size):
            group = order[group_start : group_start + effective_batch_size]
            optimizer.zero_grad()
            for start in range(0, len(group), train_batch_size):
                indices = group[start : start + train_batch_size]
                batch = [train[index] for index in indices]
                inputs, labels = _examples_batch(
                    batch, action_index, device=target_device
                )
                loss_sum = torch.nn.functional.cross_entropy(
                    model(*inputs), labels, reduction="sum"
                )
                (loss_sum / len(group)).backward()
                train_total += loss_sum.detach()
            optimizer.step()
            optimizer_steps += 1

        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for start in range(0, len(validation), validation_batch_size):
                batch = validation[start : start + validation_batch_size]
                inputs, labels = _examples_batch(
                    batch, action_index, device=target_device
                )
                validation_total += torch.nn.functional.cross_entropy(
                    model(*inputs), labels, reduction="sum"
                ).item()
        train_loss = train_total.item() / len(train)
        validation_loss = validation_total / len(validation)
        row = {
            "epoch": epoch,
            "train_cross_entropy": train_loss,
            "validation_cross_entropy": validation_loss,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(dict(row))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "epochs_completed": epochs,
        "history": history,
        "train_prefixes": len(train),
        "validation_prefixes": len(validation),
        "optimizer_steps": optimizer_steps,
        "training_seconds": training_seconds,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(target_device))
            if target_device.type == "cuda"
            else None
        ),
        "optimizer": optimizer_identity,
    }


def capture_runtime_identity(
    model: AlignedPairPolicy,
    *,
    device: str | torch.device,
    optimizer: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the exact software, accelerator, dtype, and optimizer runtime."""

    resolved = torch.device(device)
    floating_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if floating_dtypes != ["torch.float32"]:
        raise ValueError(f"Rollout policy is not entirely fp32: {floating_dtypes}")
    accelerator = None
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        resolved = torch.device("cuda", index)
        accelerator = {
            "index": index,
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    return {
        "resolved_device": str(resolved),
        "precision": "fp32",
        "floating_parameter_dtypes": floating_dtypes,
        "model": {
            "architecture": "aligned-item-pairs-untyped-v1",
            "architecture_config": model.architecture.to_config(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "optimizer": dict(optimizer),
        "accelerator": accelerator,
        "software": {
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    }


def _validate_wandb_url(url: str, *, entity: str, project: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() not in {"wandb.ai", "www.wandb.ai"}
        or len(parts) != 4
        or parts[:3] != [entity, project, "runs"]
        or not parts[3]
    ):
        raise ValueError(f"W&B returned a run outside {entity}/{project}: {url!r}")
    return url


def _validate_config(config: RolloutTrainingConfig) -> None:
    if config.wandb_entity != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            f"W&B writes are restricted to personal entity {PERSONAL_WANDB_ENTITY!r}"
        )
    if config.epochs <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if config.train_batch_size <= 0 or config.validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not isinstance(config.architecture, PolicyArchitecture):
        raise TypeError("architecture must be a PolicyArchitecture")
    if not config.wandb_project:
        raise ValueError("wandb_project must be non-empty")
    if not config.dry_run and not config.wandb_run_name:
        raise ValueError("Actual training requires --wandb-run-name")


def _corpus_provenance(
    config: RolloutTrainingConfig, prepared: PreparedPrefixes
) -> dict[str, Any]:
    manifest = prepared.manifest
    roles = {
        TRAIN_SPLIT: "training-only",
        VALIDATION_SPLIT: "model-selection-only",
        HELD_OUT_SPLIT: (
            "held-out-evaluation-only; manifest metadata only; test payload never "
            "fetched, opened, or materialized"
        ),
    }
    return {
        "root": str(config.corpus_dir.resolve()),
        "manifest_path": str((config.corpus_dir / "manifest.json").resolve()),
        "manifest_file_sha256": _sha256_file(config.corpus_dir / "manifest.json"),
        "manifest_digest": manifest["manifest_digest"],
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "hub": dict(manifest.get("hub", {})),
        "split_policy": manifest["split_policy"],
        "task_roles": {
            split: {
                "role": role,
                "records": manifest["splits"][split]["size"],
                "roots": manifest["splits"][split]["roots"],
                "sha256": manifest["splits"][split]["sha256"],
            }
            for split, role in roles.items()
        },
        "verification": {
            "ok": prepared.verification.get("ok"),
            "verified_splits": prepared.verification.get("verified_splits"),
            "full_corpus_verified": prepared.verification.get("full_corpus_verified"),
            "witness_failures": prepared.verification.get("witness_failures"),
            "roots_leaking_across_splits": prepared.verification.get(
                "roots_leaking_across_splits"
            ),
        },
    }


def build_training_manifest(
    config: RolloutTrainingConfig,
    prepared: PreparedPrefixes,
    policy: RolloutPolicy,
    trainer_state: Mapping[str, Any],
    *,
    status: str,
    started_at_utc: str,
    wandb_url: str | None,
    source_code: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat()
        if status == "complete"
        else None,
        "objective": "witness-next-action-cross-entropy",
        "teacher_forcing": "every proper witness prefix; no off-path augmentation",
        "basis_set": {
            "basis_set_id": policy.basis_set_id,
            "basis_set_digest": policy.basis_set_digest,
            "ordered_action_names": list(policy.action_names),
        },
        "source_code": dict(source_code),
        "runtime": dict(runtime_identity),
        "source_corpus": _corpus_provenance(config, prepared),
        "prefix_examples": {
            TRAIN_SPLIT: len(prepared.train),
            VALIDATION_SPLIT: len(prepared.validation),
            HELD_OUT_SPLIT: 0,
        },
        "hyperparameters": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "optimizer": "torch.optim.Adam",
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_train_batch_size": (
                config.train_batch_size * config.gradient_accumulation_steps
            ),
            "validation_batch_size": config.validation_batch_size,
            "sequence_width": MAX_PAIR_WIDTH,
            "sequence_padding": "fixed-to-maximum-for-training",
            "seed": config.seed,
            "device": config.device,
            "selection": "minimum validation cross-entropy",
            "resumption_supported": False,
            "weights_only": True,
            "architecture": config.architecture.to_config(),
        },
        "trainer_state": dict(trainer_state),
        "wandb": {
            "entity": config.wandb_entity,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
            "url": wandb_url,
        },
    }


def checkpoint_file_signature(checkpoint: Path) -> list[dict[str, Any]]:
    """Hash the exact five durable files after atomic publication."""

    assert_weights_only_checkpoint(checkpoint)
    return [
        {
            "name": name,
            "bytes": (checkpoint / name).stat().st_size,
            "sha256": _sha256_file(checkpoint / name),
        }
        for name in CHECKPOINT_FILES
    ]


def log_checkpoint_artifact(
    wandb_module: Any,
    run: Any,
    *,
    checkpoint: Path,
    signature: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> str:
    """Log the complete weights-only tree as a personal W&B model artifact."""

    run_id = getattr(run, "id", None)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("W&B run has no stable ID for the checkpoint artifact")
    artifact_name = f"rollout-policy-checkpoint-{run_id}"
    artifact = wandb_module.Artifact(
        name=artifact_name,
        type="model",
        metadata={
            "personal_entity": PERSONAL_WANDB_ENTITY,
            "corpus_manifest_digest": manifest["source_corpus"]["manifest_digest"],
            "basis_set_digest": manifest["basis_set"]["basis_set_digest"],
            "training_manifest_sha256": next(
                row["sha256"]
                for row in signature
                if row["name"] == TRAINING_MANIFEST_FILE
            ),
        },
    )
    for row in signature:
        name = str(row["name"])
        artifact.add_file(str(checkpoint / name), name=name)
    run.log_artifact(artifact)
    return artifact_name


def run_training(config: RolloutTrainingConfig) -> dict[str, Any]:
    """Verify and dry-run, or train once and persist a complete checkpoint."""

    _validate_config(config)
    started_at = datetime.now(UTC).isoformat()
    repo_root = Path(__file__).resolve().parents[1]
    source_code = source_code_identity(repo_root)
    available_functions, prepared = prepare_prefixes(config)
    model_config = policy_config(available_functions, config.architecture)
    if config.dry_run:
        policy = RolloutPolicy(
            AlignedPairPolicy(len(available_functions), config.architecture),
            model_config,
            device="cpu",
        )
        return build_training_manifest(
            config,
            prepared,
            policy,
            trainer_state={},
            status="dry-run",
            started_at_utc=started_at,
            wandb_url=None,
            source_code=source_code,
            runtime_identity={},
        )

    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")
    import wandb

    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode="online",
        tags=[config.corpus_dir.name, "one-step-policy"],
        config={
            "corpus_manifest_digest": prepared.manifest["manifest_digest"],
            "basis_set": model_config["basis_set"],
            "train_prefixes": len(prepared.train),
            "validation_prefixes": len(prepared.validation),
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_train_batch_size": (
                config.train_batch_size * config.gradient_accumulation_steps
            ),
            "validation_batch_size": config.validation_batch_size,
            "seed": config.seed,
            "architecture": config.architecture.to_config(),
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create an online run")
    try:
        if getattr(run, "entity", None) != PERSONAL_WANDB_ENTITY:
            raise RuntimeError(
                f"W&B initialized unexpected entity {getattr(run, 'entity', None)!r}"
            )
        wandb_url = _validate_wandb_url(
            str(run.url), entity=config.wandb_entity, project=config.wandb_project
        )

        def log_epoch(row: dict[str, Any]) -> None:
            run.log(row, step=row["epoch"])

        model, trainer_state = train_policy_model(
            prepared.train,
            prepared.validation,
            [function.name for function in available_functions],
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            train_batch_size=config.train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            validation_batch_size=config.validation_batch_size,
            seed=config.seed,
            device=config.device,
            architecture=config.architecture,
            epoch_callback=log_epoch,
        )
        policy = RolloutPolicy(model, model_config, device=config.device)
        runtime_identity = capture_runtime_identity(
            model,
            device=config.device,
            optimizer=trainer_state["optimizer"],
        )
        if source_code_identity(repo_root) != source_code:
            raise RuntimeError(
                "Repository source changed while rollout-policy training was running"
            )
        manifest = build_training_manifest(
            config,
            prepared,
            policy,
            trainer_state,
            status="complete",
            started_at_utc=started_at,
            wandb_url=wandb_url,
            source_code=source_code,
            runtime_identity=runtime_identity,
        )
        policy.save(
            config.output_dir,
            training_manifest=manifest,
            trainer_state=trainer_state,
            wandb_url=wandb_url,
        )
        persisted_manifest = json.loads(
            (config.output_dir / TRAINING_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        signature = checkpoint_file_signature(config.output_dir)
        manifest_sha256 = next(
            row["sha256"] for row in signature if row["name"] == TRAINING_MANIFEST_FILE
        )
        artifact_name = log_checkpoint_artifact(
            wandb,
            run,
            checkpoint=config.output_dir,
            signature=signature,
            manifest=persisted_manifest,
        )
        run.summary.update(
            {
                "checkpoint": str(config.output_dir.resolve()),
                "checkpoint_artifact": artifact_name,
                "checkpoint_file_signature": signature,
                "checkpoint_file_signature_sha256": _sha256_bytes(
                    _canonical_json(signature).encode("utf-8")
                ),
                "checkpoint_training_manifest_sha256": manifest_sha256,
                "best_epoch": trainer_state["best_epoch"],
                "best_validation_cross_entropy": trainer_state[
                    "best_validation_cross_entropy"
                ],
                "training_seconds": trainer_state["training_seconds"],
                "optimizer_steps": trainer_state["optimizer_steps"],
                "peak_cuda_memory_allocated_bytes": trainer_state[
                    "peak_cuda_memory_allocated_bytes"
                ],
                "peak_cuda_memory_reserved_bytes": trainer_state[
                    "peak_cuda_memory_reserved_bytes"
                ],
            }
        )
        return persisted_manifest
    finally:
        run.finish()


def _parse_args(argv: Sequence[str] | None = None) -> RolloutTrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--size-arm", choices=tuple(POLICY_SIZE_ARMS), default="90k")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    train_batch_size, accumulation_steps, validation_batch_size = POLICY_SIZE_BATCHING[
        arguments.size_arm
    ]
    return RolloutTrainingConfig(
        output_dir=arguments.output_dir,
        corpus_dir=arguments.corpus_dir,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        train_batch_size=(
            arguments.train_batch_size
            if arguments.train_batch_size is not None
            else train_batch_size
        ),
        gradient_accumulation_steps=(
            arguments.gradient_accumulation_steps
            if arguments.gradient_accumulation_steps is not None
            else accumulation_steps
        ),
        validation_batch_size=(
            arguments.validation_batch_size
            if arguments.validation_batch_size is not None
            else validation_batch_size
        ),
        seed=arguments.seed,
        device=arguments.device,
        wandb_entity=arguments.wandb_entity,
        wandb_project=arguments.wandb_project,
        wandb_run_name=arguments.wandb_run_name,
        architecture=POLICY_SIZE_ARMS[arguments.size_arm],
        dry_run=arguments.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = run_training(_parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
