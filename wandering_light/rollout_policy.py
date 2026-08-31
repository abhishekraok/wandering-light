"""Loadable one-step policy used by materialized-state rollout search.

The encoder is the untyped aligned-pair arm from the policy probe: each
``(state[i], target[i])`` example is encoded independently, then mean- and
max-pooled across examples.  The wrapper pins the exact ordered basis used by
the output head and exposes raw finite logits through ``ActionScorer``; rollout
search owns type masking and normalization.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from wandering_light.function_usage import FunctionUsageTracker
from wandering_light.solver import ActionScorer

if TYPE_CHECKING:
    from wandering_light.function_def import FunctionDefSet
    from wandering_light.typed_list import TypedList


SCHEMA_VERSION = 1
ARCHITECTURE = "aligned-item-pairs-untyped-v1"
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINING_MANIFEST_FILE = "training_manifest.json"
TRAINER_STATE_FILE = "trainer_state.json"
WANDB_LINK_FILE = "wandb_run.url"

ITEM_CAP = 96
MAX_ITEMS = 16
PAD = 256
SEP = 257
MAX_PAIR_WIDTH = 2 * ITEM_CAP + 1
BYTE_EMBEDDING = 48
CONV_CHANNELS = 96
CONV_KERNEL = 5
HIDDEN = 176

_FORBIDDEN_STATE_NAMES = frozenset(
    {"optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_wandb_url(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("[InternetShortcut]"):
        for line in text.splitlines():
            if line.startswith("URL="):
                return line.removeprefix("URL=").strip()
        raise ValueError(f"W&B shortcut has no URL: {path}")
    return text


def _write_wandb_url(path: Path, url: str) -> None:
    if not isinstance(url, str) or not url.startswith(
        "https://wandb.ai/abhishekraok-na/"
    ):
        raise ValueError(f"Expected a personal W&B run URL, got {url!r}")
    path.write_text(f"[InternetShortcut]\nURL={url}\n", encoding="utf-8")


@dataclass(frozen=True)
class OrderedAction:
    """One output row of the policy head, including stable basis identity."""

    function_id: str
    name: str
    input_type: str
    output_type: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        expected = {"function_id", "name", "input_type", "output_type"}
        if set(value) != expected or not all(
            isinstance(value[key], str) and value[key] for key in expected
        ):
            raise ValueError(f"Invalid ordered action identity: {value!r}")
        return cls(**{key: value[key] for key in expected})

    def to_dict(self) -> dict[str, str]:
        return {
            "function_id": self.function_id,
            "name": self.name,
            "input_type": self.input_type,
            "output_type": self.output_type,
        }


def ordered_actions(functions: FunctionDefSet) -> tuple[OrderedAction, ...]:
    """Return the exact ordered identity consumed by the output projection."""

    actions = []
    for function in functions:
        function_id = (function.metadata or {}).get("basis_function_id")
        if not isinstance(function_id, str) or not function_id:
            raise ValueError(f"Basis function {function.name!r} has no stable ID")
        actions.append(
            OrderedAction(
                function_id=function_id,
                name=function.name,
                input_type=function.input_type,
                output_type=function.output_type,
            )
        )
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise ValueError("Ordered basis contains duplicate function names")
    return tuple(actions)


def _serialized_pair_items(state: str, target: str) -> list[list[int]]:
    """Encode aligned item pairs exactly as the strongest probe arm did."""

    source = json.loads(state)
    destination = json.loads(target)
    source_items = source.get("items")
    target_items = destination.get("items")
    if not isinstance(source_items, list) or not isinstance(target_items, list):
        raise ValueError("Serialized TypedList must contain an items array")
    if len(source_items) != len(target_items):
        raise ValueError(
            "Aligned policy encoding requires equal state and target item counts"
        )

    pairs = []
    for left, right in zip(
        source_items[:MAX_ITEMS], target_items[:MAX_ITEMS], strict=True
    ):
        left_bytes = json.dumps(left, separators=(",", ":")).encode("utf-8", "replace")
        right_bytes = json.dumps(right, separators=(",", ":")).encode(
            "utf-8", "replace"
        )
        pairs.append([*left_bytes[:ITEM_CAP], SEP, *right_bytes[:ITEM_CAP]])
    return pairs or [[SEP]]


def pair_batch(
    serialized_problems: Sequence[tuple[str, str]],
    *,
    device: torch.device | str | None = None,
    sequence_width: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Batch serialized pairs, optionally padding every item to a fixed width."""

    if not serialized_problems:
        raise ValueError("Cannot construct an empty policy batch")
    encoded = [_serialized_pair_items(*problem) for problem in serialized_problems]
    natural_width = max(len(sequence) for pairs in encoded for sequence in pairs)
    if sequence_width is None:
        width = natural_width
    elif not natural_width <= sequence_width <= MAX_PAIR_WIDTH:
        raise ValueError(
            f"sequence_width must be in [{natural_width}, {MAX_PAIR_WIDTH}], "
            f"got {sequence_width}"
        )
    else:
        width = sequence_width
    sequences: list[list[int]] = []
    counts = []
    for pairs in encoded:
        counts.append(len(pairs))
        sequences.extend(
            sequence + [PAD] * (width - len(sequence)) for sequence in pairs
        )
    count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    item_mask = (
        torch.arange(int(count_tensor.max()), device=device)[None, :]
        < count_tensor[:, None]
    )
    return (
        torch.tensor(sequences, dtype=torch.long, device=device),
        count_tensor,
        item_mask,
    )


class AlignedPairPolicy(nn.Module):
    """Small conv encoder over aligned examples, pooled as an unordered set."""

    def __init__(self, action_count: int):
        super().__init__()
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        self.action_count = action_count
        self.embedding = nn.Embedding(258, BYTE_EMBEDDING)
        self.convolution = nn.Conv1d(
            BYTE_EMBEDDING,
            CONV_CHANNELS,
            kernel_size=CONV_KERNEL,
            padding=CONV_KERNEL // 2,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * CONV_CHANNELS, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_count),
        )

    def _encode_items(self, sequences: Tensor) -> Tensor:
        real = sequences != PAD
        embedded = self.embedding(sequences) * real.unsqueeze(-1)
        convolved = self.convolution(embedded.transpose(1, 2)).relu()
        convolved = convolved.masked_fill(~real.unsqueeze(1), -1e9)
        return convolved.max(dim=2).values

    def pooled_features(
        self, sequences: Tensor, counts: Tensor, item_mask: Tensor
    ) -> Tensor:
        """The pooled representation the action head reads.

        This is the policy's own view of a ``(state, target)`` pair, before any
        action is scored. The rollout debugger projects it to plot search
        progress in the space the policy actually searches in.
        """

        encoded = self._encode_items(sequences)
        dense = nn.utils.rnn.pad_sequence(
            encoded.split(counts.tolist()), batch_first=True
        )
        mask = item_mask.unsqueeze(-1)
        mean = (dense * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        peak = dense.masked_fill(~mask, -1e9).max(dim=1).values
        return torch.cat([mean, peak], dim=1)

    def forward(self, sequences: Tensor, counts: Tensor, item_mask: Tensor) -> Tensor:
        return self.head(self.pooled_features(sequences, counts, item_mask))


def policy_config(
    functions: FunctionDefSet,
) -> dict[str, Any]:
    """Create the architecture and ordered-basis identity saved with weights."""

    basis_set_id, basis_digest = FunctionUsageTracker.basis_provenance(functions)
    if basis_set_id is None or basis_digest is None:
        raise ValueError("Rollout policy checkpoints require basis provenance")
    actions = ordered_actions(functions)
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": {
            "name": ARCHITECTURE,
            "item_cap": ITEM_CAP,
            "max_items": MAX_ITEMS,
            "pad_token": PAD,
            "separator_token": SEP,
            "byte_embedding": BYTE_EMBEDDING,
            "conv_channels": CONV_CHANNELS,
            "conv_kernel": CONV_KERNEL,
            "hidden": HIDDEN,
        },
        "basis_set": {
            "basis_set_id": basis_set_id,
            "basis_set_digest": basis_digest,
            "ordered_actions": [action.to_dict() for action in actions],
        },
    }


def _parse_config(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[OrderedAction, ...]]:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported rollout policy config schema")
    expected_architecture = policy_config_architecture()
    if value.get("architecture") != expected_architecture:
        raise ValueError("Rollout policy architecture does not match this runtime")
    basis = value.get("basis_set")
    if not isinstance(basis, Mapping):
        raise ValueError("Rollout policy config has no basis identity")
    if not isinstance(basis.get("basis_set_id"), str) or not isinstance(
        basis.get("basis_set_digest"), str
    ):
        raise ValueError("Rollout policy config has invalid basis provenance")
    raw_actions = basis.get("ordered_actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Rollout policy config has no ordered actions")
    actions = tuple(OrderedAction.from_dict(action) for action in raw_actions)
    if len({action.name for action in actions}) != len(actions):
        raise ValueError("Rollout policy config contains duplicate action names")
    return dict(basis), actions


def policy_config_architecture() -> dict[str, Any]:
    return {
        "name": ARCHITECTURE,
        "item_cap": ITEM_CAP,
        "max_items": MAX_ITEMS,
        "pad_token": PAD,
        "separator_token": SEP,
        "byte_embedding": BYTE_EMBEDDING,
        "conv_channels": CONV_CHANNELS,
        "conv_kernel": CONV_KERNEL,
        "hidden": HIDDEN,
    }


def assert_weights_only_checkpoint(
    checkpoint: Path, *, require_complete: bool = True
) -> None:
    """Reject resumable trainer state and incomplete policy artifacts."""

    forbidden = []
    for path in checkpoint.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _FORBIDDEN_STATE_NAMES or path.name.startswith("rng_state_"):
            forbidden.append(path.relative_to(checkpoint).as_posix())
    intermediate = [
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("checkpoint-*")
    ]
    if forbidden or intermediate:
        raise ValueError(
            "Rollout policy artifact is not weights-only: "
            f"forbidden={sorted(forbidden)}, checkpoints={sorted(intermediate)}"
        )
    required = {
        MODEL_FILE,
        CONFIG_FILE,
        TRAINING_MANIFEST_FILE,
        TRAINER_STATE_FILE,
        WANDB_LINK_FILE,
    }
    files = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(files - required)
    if unexpected:
        raise ValueError(
            f"Rollout policy checkpoint has unexpected files: {unexpected}"
        )
    if require_complete:
        missing = sorted(name for name in required if not (checkpoint / name).is_file())
        if missing:
            raise ValueError(f"Rollout policy checkpoint is incomplete: {missing}")


class RolloutPolicy(ActionScorer):
    """A basis-pinned, batchable one-step action scorer."""

    def __init__(
        self,
        model: AlignedPairPolicy,
        config: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        basis, actions = _parse_config(config)
        if model.action_count != len(actions):
            raise ValueError("Policy head width differs from its ordered basis")
        self.model = model.to(device).eval()
        self.config = dict(config)
        self.basis_set_id = basis["basis_set_id"]
        self.basis_set_digest = basis["basis_set_digest"]
        self.actions = actions
        self.action_names = tuple(action.name for action in actions)
        self.device = torch.device(device)

    @classmethod
    def from_basis(
        cls,
        model: AlignedPairPolicy,
        functions: FunctionDefSet,
        *,
        device: torch.device | str = "cpu",
    ) -> Self:
        return cls(model, policy_config(functions), device=device)

    def _require_basis(self, available_functions: FunctionDefSet) -> None:
        basis_set_id, basis_digest = FunctionUsageTracker.basis_provenance(
            available_functions
        )
        if (basis_set_id, basis_digest) != (
            self.basis_set_id,
            self.basis_set_digest,
        ):
            raise ValueError(
                "Available basis provenance differs from policy checkpoint"
            )
        actual = ordered_actions(available_functions)
        if actual != self.actions:
            raise ValueError("Available ordered basis differs from policy checkpoint")

    def score_actions_batch(
        self,
        problems: list[tuple[TypedList, TypedList]],
        available_functions: FunctionDefSet,
    ) -> list[dict[str, float]]:
        """Return finite raw logits; rollout applies compatibility and softmax."""

        self._require_basis(available_functions)
        if not problems:
            return []
        serialized = [
            (state.to_string(), target.to_string()) for state, target in problems
        ]
        inputs = pair_batch(serialized, device=self.device)
        with torch.inference_mode():
            rows = self.model(*inputs).detach().cpu().tolist()
        scored = []
        for row in rows:
            if len(row) != len(self.action_names) or not all(
                math.isfinite(score) for score in row
            ):
                raise ValueError("Policy produced a malformed action-score row")
            scored.append(dict(zip(self.action_names, row, strict=True)))
        return scored

    def save(
        self,
        checkpoint: Path,
        *,
        training_manifest: Mapping[str, Any],
        trainer_state: Mapping[str, Any],
        wandb_url: str,
    ) -> Path:
        """Write one complete weights-only checkpoint to a new directory."""

        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{checkpoint.name}.staging-", dir=checkpoint.parent
            )
        )
        try:
            cpu_state = {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in self.model.state_dict().items()
            }
            save_file(
                cpu_state,
                str(staging / MODEL_FILE),
                metadata={"format": "pt", "architecture": ARCHITECTURE},
            )
            _write_json(staging / CONFIG_FILE, self.config)
            _write_json(staging / TRAINER_STATE_FILE, dict(trainer_state))
            _write_wandb_url(staging / WANDB_LINK_FILE, wandb_url)

            manifest = dict(training_manifest)
            manifest["artifacts"] = {
                MODEL_FILE: _sha256_file(staging / MODEL_FILE),
                CONFIG_FILE: _sha256_file(staging / CONFIG_FILE),
                TRAINER_STATE_FILE: _sha256_file(staging / TRAINER_STATE_FILE),
                WANDB_LINK_FILE: _sha256_file(staging / WANDB_LINK_FILE),
            }
            _write_json(staging / TRAINING_MANIFEST_FILE, manifest)
            assert_weights_only_checkpoint(staging)
            if checkpoint.exists():
                raise FileExistsError(f"Checkpoint appeared during save: {checkpoint}")
            staging.replace(checkpoint)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return checkpoint

    @classmethod
    def load(cls, checkpoint: Path, *, device: torch.device | str = "cpu") -> Self:
        """Load and verify one weights-only rollout-policy checkpoint."""

        checkpoint = Path(checkpoint)
        assert_weights_only_checkpoint(checkpoint)
        config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
        config_basis, actions = _parse_config(config)
        manifest = json.loads(
            (checkpoint / TRAINING_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        if manifest.get("status") != "complete":
            raise ValueError("Rollout policy training manifest is not complete")
        manifest_basis = manifest.get("basis_set")
        expected_manifest_basis = {
            "basis_set_id": config_basis["basis_set_id"],
            "basis_set_digest": config_basis["basis_set_digest"],
            "ordered_action_names": [action.name for action in actions],
        }
        if manifest_basis != expected_manifest_basis:
            raise ValueError(
                "Rollout policy training manifest basis differs from policy config"
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Rollout policy manifest has no artifact hashes")
        for name in (MODEL_FILE, CONFIG_FILE, TRAINER_STATE_FILE, WANDB_LINK_FILE):
            if artifacts.get(name) != _sha256_file(checkpoint / name):
                raise ValueError(f"Rollout policy artifact hash mismatch: {name}")
        manifest_wandb = manifest.get("wandb")
        if not isinstance(manifest_wandb, Mapping) or manifest_wandb.get(
            "url"
        ) != _read_wandb_url(checkpoint / WANDB_LINK_FILE):
            raise ValueError("Rollout policy W&B lineage does not match its link")

        model = AlignedPairPolicy(len(actions))
        model.load_state_dict(load_file(str(checkpoint / MODEL_FILE), device="cpu"))
        return cls(model, config, device=device)
