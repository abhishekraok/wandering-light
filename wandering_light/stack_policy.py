"""One-step policy over aligned mark-28 stack PBE examples."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.functional import cross_entropy

from wandering_light.rollout_policy import AlignedPairPolicy, PolicyArchitecture
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from torch import Tensor

    from wandering_light.stack_machine import Value
    from wandering_light.stack_pbe import Task

OP_NAMES = tuple(op.name for op in OPS)
OP_INDEX = {name: index for index, name in enumerate(OP_NAMES)}

PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG = range(9)
SMALL_INT = 64
_OP_START = BIG + 1
_INT_START = _OP_START + len(OP_NAMES)
_DIGIT_START = _INT_START + 2 * SMALL_INT + 1
VOCAB = _DIGIT_START + 10
STATE_CAP = 160
TARGET_CAP = 64

SCHEMA_VERSION = 1
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "training_manifest.json"
WANDB_LINK_FILE = "wandb_run.url"

type Stack = tuple[Value, ...]
type Product = tuple[Stack, ...]


def _reverse_tokens(value: Value) -> Iterator[int]:
    """Emit topmost tokens first so truncation can stop at the cap."""
    if isinstance(value, Int):
        number = value.value
        if -SMALL_INT <= number <= SMALL_INT:
            yield _INT_START + number + SMALL_INT
        else:
            magnitude = abs(number)
            while magnitude:
                magnitude, digit = divmod(magnitude, 10)
                yield _DIGIT_START + digit
            if number < 0:
                yield NEG
            yield BIG
    elif isinstance(value, Bool):
        yield TRUE_TOKEN if value.value else FALSE_TOKEN
    elif isinstance(value, Mark):
        yield MARK_TOKEN
    elif isinstance(value, Sym):
        yield _OP_START + OP_INDEX[value.name]
    elif isinstance(value, Lst):
        yield CLOSE
        for item in reversed(value.items):
            yield from _reverse_tokens(item)
        yield OPEN


def encode_stack(stack: Sequence[Value], cap: int) -> list[int]:
    """Encode the topmost tokens of a stack in bottom-to-top order."""
    tokens = (token for value in reversed(stack) for token in _reverse_tokens(value))
    encoded = list(islice(tokens, cap))
    encoded.reverse()
    return encoded


def encode_item(state: Stack, target_tokens: Sequence[int]) -> list[int]:
    return [*encode_stack(state, STATE_CAP), SEP, *target_tokens]


def encode_targets(outputs: Sequence[Stack]) -> list[list[int]]:
    return [encode_stack(stack, TARGET_CAP) for stack in outputs]


def item_array(items: Sequence[Sequence[int]]) -> np.ndarray:
    block = np.full((len(items), max(map(len, items))), PAD, dtype=np.int16)
    for index, item in enumerate(items):
        block[index, : len(item)] = item
    return block


def collate(
    blocks: Sequence[np.ndarray], device: str | torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten items with the counts and mask used by the aligned-pair model."""
    if len(blocks) == 0:
        raise ValueError("Cannot collate an empty policy batch")
    sizes = [len(block) for block in blocks]
    width = max(block.shape[1] for block in blocks)
    padded = np.full((sum(sizes), width), PAD, dtype=np.int16)
    offset = 0
    for block in blocks:
        padded[offset : offset + len(block), : block.shape[1]] = block
        offset += len(block)
    sequences = torch.as_tensor(padded, dtype=torch.long, device=device)
    counts = torch.tensor(sizes, dtype=torch.long, device=device)
    item_mask = torch.arange(max(sizes), device=device)[None, :] < counts[:, None]
    return sequences, counts, item_mask


class StackPairPolicy(AlignedPairPolicy):
    """Use the shared aligned-pair model with stack tokens and mark-28 actions."""

    def __init__(self, architecture: PolicyArchitecture):
        super().__init__(len(OP_NAMES), architecture)
        self.embedding = nn.Embedding(
            VOCAB, architecture.byte_embedding, padding_idx=PAD
        )

    def _encode_items(self, sequences: Tensor) -> Tensor:
        # The shared model pools these features across examples unchanged.
        real = sequences != PAD
        embedded = self.embedding(sequences).masked_fill(~real.unsqueeze(-1), 0)
        convolved = self.convolution(embedded.transpose(1, 2)).relu()
        return convolved.masked_fill(~real.unsqueeze(1), -1e9).max(dim=2).values

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class PolicyScorer:
    def __init__(
        self, model: StackPairPolicy, device: str | torch.device = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def log_probs(self, blocks: Sequence[np.ndarray], legal: np.ndarray) -> np.ndarray:
        mask = torch.as_tensor(legal, dtype=torch.bool, device=self.device)
        logits = self.model(*collate(blocks, self.device))
        if mask.shape != logits.shape:
            raise ValueError("Legal mask must match the policy logits")
        masked = logits.masked_fill(~mask, -torch.inf)
        return masked.log_softmax(dim=1).masked_fill(~mask, -torch.inf).cpu().numpy()


@dataclass(frozen=True, slots=True)
class Row:
    items: np.ndarray
    label: int
    task: int
    step: int
    remaining: int


def replay_witness(task: Task) -> Iterator[tuple[int, Product, str]]:
    """Yield the shown product before each witness op."""
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    spent = [0] * len(product)
    for step, name in enumerate(task.program):
        op = OP_BY_NAME.get(name)
        if op is None:
            raise ValueError(
                f"{task.behaviour} witness step {step}: unknown op {name!r}"
            )
        moved = []
        for example, stack in enumerate(product):
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT - spent[example], 0)
            if isinstance(outcome, str):
                raise ValueError(
                    f"{task.behaviour} witness step {step}, example {example}, "
                    f"{name}: {outcome}"
                )
            next_stack, fuel = outcome
            spent[example] += fuel
            moved.append(next_stack)
        yield step, product, name
        product = tuple(moved)


def witness_rows(tasks: Sequence[Task]) -> list[Row]:
    rows = []
    for index, task in enumerate(tasks):
        outputs = [parse_stack(text, basis=BASIS) for text in task.outputs]
        targets = encode_targets(outputs)
        for step, product, name in replay_witness(task):
            items = item_array(
                [
                    encode_item(state, target)
                    for state, target in zip(product, targets, strict=True)
                ]
            )
            rows.append(Row(items, OP_INDEX[name], index, step, task.length - step))
    return rows


def _batch_loss(
    model: StackPairPolicy, rows: Sequence[Row], device: str | torch.device
) -> Tensor:
    inputs = collate([row.items for row in rows], device)
    labels = torch.tensor([row.label for row in rows], dtype=torch.long, device=device)
    return cross_entropy(model(*inputs), labels, reduction="sum")


def train_policy(
    train: Sequence[Row],
    validation: Sequence[Row],
    architecture: PolicyArchitecture,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    validation_batch_size: int,
    seed: int,
    device: str | torch.device,
    epoch_callback: Callable[[dict[str, int | float]], None] | None = None,
) -> tuple[StackPairPolicy, dict]:
    """Fit by witness cross-entropy and restore the best validation epoch."""
    if not train or not validation:
        raise ValueError("Training and validation rows must both be non-empty")
    if min(epochs, batch_size, validation_batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("Epochs, batch sizes, and learning rate must be positive")

    if torch.device(device).type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    order = list(range(len(train)))
    model = StackPairPolicy(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            rng.shuffle(order)
            train_total = torch.zeros((), device=device)
            for start in range(0, len(order), batch_size):
                batch = [train[index] for index in order[start : start + batch_size]]
                optimizer.zero_grad()
                loss = _batch_loss(model, batch, device)
                (loss / len(batch)).backward()
                optimizer.step()
                train_total += loss.detach()

            model.eval()
            with torch.inference_mode():
                validation_total = sum(
                    _batch_loss(
                        model, validation[start : start + validation_batch_size], device
                    )
                    for start in range(0, len(validation), validation_batch_size)
                )
            validation_loss = validation_total.item() / len(validation)
            metrics = {
                "epoch": epoch,
                "train_cross_entropy": train_total.item() / len(train),
                "validation_cross_entropy": validation_loss,
            }
            history.append(metrics)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().to("cpu", copy=True)
                    for name, tensor in model.state_dict().items()
                }
            if epoch_callback is not None:
                epoch_callback(dict(metrics))
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    if best_state is None:
        raise RuntimeError("Training produced no finite validation loss")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "history": history,
        "parameters": model.parameter_count,
    }


def validation_skeleton(task: Task, *, out_of: int = 10) -> bool:
    if out_of <= 0:
        raise ValueError("out_of must be positive")
    digest = hashlib.sha256((task.skeleton or task.behaviour).encode()).digest()
    return int.from_bytes(digest, "big") % out_of == 0


def _write_json(path: Path, data: Mapping) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_policy(
    checkpoint: str | Path,
    model: StackPairPolicy,
    *,
    trainer_state: Mapping,
    manifest: Mapping,
    wandb_url: str | None,
) -> Path:
    """Publish a complete weights-only checkpoint under a new path."""
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{checkpoint.name}-", dir=checkpoint.parent
    ) as tmp:
        staging = Path(tmp)
        weights = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        save_file(weights, str(staging / MODEL_FILE), metadata={"format": "pt"})
        _write_json(
            staging / CONFIG_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "architecture": asdict(model.architecture),
                "vocab": VOCAB,
                "basis": BASIS,
                "ordered_actions": list(OP_NAMES),
            },
        )
        _write_json(staging / TRAINER_STATE_FILE, dict(trainer_state))
        _write_json(staging / MANIFEST_FILE, {**manifest, "wandb_url": wandb_url})
        if wandb_url:
            (staging / WANDB_LINK_FILE).write_text(
                f"[InternetShortcut]\nURL={wandb_url}\n", encoding="utf-8"
            )
        try:
            if checkpoint.exists():
                raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
            staging.rename(checkpoint)
        except OSError as error:
            if checkpoint.exists() and not isinstance(error, FileExistsError):
                raise FileExistsError(
                    f"Checkpoint already exists: {checkpoint}"
                ) from error
            raise
    return checkpoint


def load_policy(
    checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> StackPairPolicy:
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported stack policy config schema")
    if config.get("basis") != BASIS:
        raise ValueError("Unsupported stack policy basis")
    if config.get("ordered_actions") != list(OP_NAMES):
        raise ValueError("Unsupported stack policy action order")
    if config.get("vocab") != VOCAB:
        raise ValueError("Unsupported stack policy vocabulary")
    architecture = PolicyArchitecture(**config["architecture"])
    model = StackPairPolicy(architecture).to(device)
    model.load_state_dict(load_file(str(checkpoint / MODEL_FILE), device=str(device)))
    return model.eval()
