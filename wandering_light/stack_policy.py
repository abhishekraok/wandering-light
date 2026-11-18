"""One-step prediction on the shown examples of a mark-28 PBE task."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn.functional import cross_entropy

from wandering_light.rollout_policy import AlignedPairPolicy, PolicyArchitecture
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    Value,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from wandering_light.stack_pbe import Task

Stack = tuple[Value, ...]
Product = tuple[Stack, ...]

OP_NAMES = tuple(op.name for op in OPS)
OP_INDEX = {name: index for index, name in enumerate(OP_NAMES)}

PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG = range(9)
SMALL_INT = 64
OP_BASE = BIG + 1
INT_BASE = OP_BASE + len(OP_NAMES)
DIGIT_BASE = INT_BASE + 2 * SMALL_INT + 1
VOCAB = DIGIT_BASE + 10
STATE_CAP = 160
TARGET_CAP = 64

SCHEMA_VERSION = 1
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "training_manifest.json"
WANDB_LINK_FILE = "wandb_run.url"


def _reverse_tokens(stack: Stack) -> Iterator[int]:
    # Start at the top so the cap also bounds work on long quotation bodies.
    for value in reversed(stack):
        if isinstance(value, Int):
            magnitude = abs(value.value)
            if magnitude <= SMALL_INT:
                yield INT_BASE + SMALL_INT + value.value
            else:
                while magnitude:
                    magnitude, digit = divmod(magnitude, 10)
                    yield DIGIT_BASE + digit
                if value.value < 0:
                    yield NEG
                yield BIG
        elif isinstance(value, Bool):
            yield TRUE_TOKEN if value.value else FALSE_TOKEN
        elif isinstance(value, Mark):
            yield MARK_TOKEN
        elif isinstance(value, Sym):
            yield OP_BASE + OP_INDEX[value.name]
        elif isinstance(value, Lst):
            yield CLOSE
            yield from _reverse_tokens(value.items)
            yield OPEN


def encode_stack(stack: Stack, cap: int) -> list[int]:
    tokens = list(islice(_reverse_tokens(stack), cap))
    tokens.reverse()
    return tokens


def encode_item(state: Stack, target_tokens: Sequence[int]) -> list[int]:
    return [*encode_stack(state, STATE_CAP), SEP, *target_tokens]


def encode_targets(outputs: Sequence[Stack | str]) -> list[list[int]]:
    return [
        encode_stack(
            parse_stack(out, basis=BASIS) if isinstance(out, str) else out, TARGET_CAP
        )
        for out in outputs
    ]


def item_array(items: Sequence[Sequence[int]]) -> np.ndarray:
    """Store a single product compactly between training or inference batches."""
    block = np.full((len(items), max(map(len, items))), PAD, dtype=np.int16)
    for row, item in zip(block, items, strict=True):
        row[: len(item)] = item
    return block


def product_items(product: Product, targets: Sequence[Sequence[int]]) -> np.ndarray:
    return item_array(
        [
            encode_item(stack, target)
            for stack, target in zip(product, targets, strict=True)
        ]
    )


def collate(blocks: Sequence[np.ndarray], device: str | torch.device):
    counts = [len(block) for block in blocks]
    width = max(block.shape[1] for block in blocks)
    sequences = np.full((sum(counts), width), PAD, dtype=np.int16)
    start = 0
    for block in blocks:
        end = start + len(block)
        sequences[start:end, : block.shape[1]] = block
        start = end
    count_tensor = torch.tensor(counts, device=device)
    item_mask = (
        torch.arange(max(counts), device=device)[None, :] < count_tensor[:, None]
    )
    return (
        torch.as_tensor(sequences, dtype=torch.long, device=device),
        count_tensor,
        item_mask,
    )


class StackPairPolicy(AlignedPairPolicy):
    def __init__(self, architecture: PolicyArchitecture):
        super().__init__(len(OP_NAMES), architecture, vocab_size=VOCAB, padding_idx=PAD)


def parameter_count(architecture: PolicyArchitecture | StackPairPolicy) -> int:
    if isinstance(architecture, StackPairPolicy):
        return sum(p.numel() for p in architecture.parameters())
    return architecture.parameter_count(len(OP_NAMES), vocab_size=VOCAB)


class PolicyScorer:
    def __init__(self, model: StackPairPolicy, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def log_probs(self, blocks: Sequence[np.ndarray], legal: np.ndarray) -> np.ndarray:
        """Normalize each node over the ops that stepped all its shown stacks."""
        logits = self.model(*collate(blocks, self.device))
        mask = torch.as_tensor(legal, device=self.device, dtype=torch.bool)
        if mask.shape != logits.shape:
            raise ValueError("legal must have one entry per node and action")
        scores = logits.masked_fill(~mask, -torch.inf).log_softmax(dim=1)
        # A node with no legal move can still share a batch with live nodes.
        return scores.masked_fill(~mask, -torch.inf).cpu().numpy()


@dataclass(frozen=True, slots=True)
class Row:
    items: np.ndarray
    label: int
    task: int
    step: int
    remaining: int


def replay_witness(task: Task) -> Iterator[tuple[int, Product, str]]:
    product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    fuels = [0] * len(product)
    for step, name in enumerate(task.program):
        if name not in OP_BY_NAME:
            raise ValueError(f"Unknown witness op {name!r} at step {step}")
        moved = []
        for i, stack in enumerate(product):
            result = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT - fuels[i], 0)
            if isinstance(result, str):
                raise ValueError(f"Witness step {step} ({name}), example {i}: {result}")
            next_stack, spent = result
            moved.append(next_stack)
            fuels[i] += spent
        yield step, product, name
        product = tuple(moved)


def witness_rows(tasks: Sequence[Task]) -> list[Row]:
    rows = []
    for index, task in enumerate(tasks):
        targets = encode_targets(task.outputs)
        for step, product, name in replay_witness(task):
            rows.append(
                Row(
                    product_items(product, targets),
                    OP_INDEX[name],
                    index,
                    step,
                    task.length - step,
                )
            )
    return rows


def validation_skeleton(task: Task, *, out_of: int = 10) -> bool:
    if out_of < 1:
        raise ValueError("out_of must be positive")
    digest = hashlib.sha256((task.skeleton or task.behaviour).encode()).digest()
    return int.from_bytes(digest[:4], "big") % out_of == 0


def _batch_loss(model, rows, device):
    inputs = collate([row.items for row in rows], device)
    labels = torch.tensor([row.label for row in rows], device=device)
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
    epoch_callback: Callable[[dict], None] | None = None,
) -> tuple[StackPairPolicy, dict]:
    if not train or not validation:
        raise ValueError("Training and validation rows must both be non-empty")
    if min(epochs, batch_size, validation_batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch sizes, and learning rate must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = StackPairPolicy(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    order = list(range(len(train)))
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        train_total = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            rows = [train[i] for i in order[start : start + batch_size]]
            optimizer.zero_grad()
            loss = _batch_loss(model, rows, device)
            (loss / len(rows)).backward()
            optimizer.step()
            train_total += loss.detach()

        model.eval()
        with torch.inference_mode():
            validation_total = torch.zeros((), device=device)
            for start in range(0, len(validation), validation_batch_size):
                rows = validation[start : start + validation_batch_size]
                validation_total += _batch_loss(model, rows, device)
        validation_loss = validation_total.item() / len(validation)
        metrics = {
            "epoch": epoch,
            "train_cross_entropy": train_total.item() / len(train),
            "validation_cross_entropy": validation_loss,
        }
        history.append(metrics)
        if epoch_callback is not None:
            epoch_callback(dict(metrics))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_weights is None:
        raise ValueError("Training produced no finite validation loss")
    model.load_state_dict(best_weights)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "history": history,
        "parameters": parameter_count(model),
    }


def save_policy(
    checkpoint: Path,
    model: StackPairPolicy,
    *,
    trainer_state: dict,
    manifest: dict,
    wandb_url: str | None,
) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "architecture": asdict(model.architecture),
        "vocab": VOCAB,
        "small_int": SMALL_INT,
        "state_cap": STATE_CAP,
        "target_cap": TARGET_CAP,
        "basis": BASIS,
        "ordered_actions": OP_NAMES,
    }
    with tempfile.TemporaryDirectory(dir=checkpoint.parent) as staging:
        directory = Path(staging)
        save_file(
            {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
            directory / MODEL_FILE,
        )
        for filename, data in (
            (CONFIG_FILE, config),
            (TRAINER_STATE_FILE, trainer_state),
            (MANIFEST_FILE, {**manifest, "wandb_url": wandb_url}),
        ):
            (directory / filename).write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        if wandb_url:
            (directory / WANDB_LINK_FILE).write_text(
                f"[InternetShortcut]\nURL={wandb_url}\n", encoding="utf-8"
            )
        try:
            directory.rename(checkpoint)
        except OSError as error:
            if checkpoint.exists():
                raise FileExistsError(
                    f"Checkpoint already exists: {checkpoint}"
                ) from error
            raise
    return checkpoint


def load_policy(
    checkpoint: Path, *, device: str | torch.device = "cpu"
) -> StackPairPolicy:
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported stack policy schema version")
    if config.get("basis") != BASIS:
        raise ValueError("Unsupported stack policy basis")
    if config.get("ordered_actions") != list(OP_NAMES):
        raise ValueError("Unsupported stack policy action order")
    if config.get("vocab") != VOCAB:
        raise ValueError("Unsupported stack policy vocabulary")
    model = StackPairPolicy(PolicyArchitecture(**config["architecture"]))
    model.load_state_dict(load_file(checkpoint / MODEL_FILE))
    return model.to(device).eval()
