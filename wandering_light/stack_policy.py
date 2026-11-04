"""A one-step policy over ``mark-28`` product states, for ``stack_pbe_v4``.

The search state of a PBE task is the *product* of its shown examples: one
stack per example, every one stepped by the same op. The policy reads that
state against the shown outputs and scores the next op.

An *item* is one shown example: its current stack and its expected output,
each serialized as op/value tokens rather than bytes. Token serialization is
what lets an open quotation's body fit -- a body is a run of op symbols, one
token each, where a byte encoding would spend five or six per symbol. A task
with ``k`` shown examples is ``k`` items, encoded independently and pooled as
an unordered set by the aligned-pair encoder of
``wandering_light/rollout_policy.py``, reused unchanged apart from the
embedding vocabulary and the padding token.

Training reads witness prefixes: every prefix of a task's witness program on
its shown inputs is one row, labelled with the witness's next op.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from wandering_light.rollout_policy import (
    CONV_KERNEL,
    AlignedPairPolicy,
    PolicyArchitecture,
)
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
    from wandering_light.stack_pbe import Task

# ---------------------------------------------------------------------------
# Actions and vocabulary
# ---------------------------------------------------------------------------

OP_NAMES: tuple[str, ...] = tuple(op.name for op in OPS)
OP_INDEX: dict[str, int] = {name: index for index, name in enumerate(OP_NAMES)}

PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG = range(9)
SMALL_INT = 64
SYMBOL_BASE = 9
INT_BASE = SYMBOL_BASE + len(OP_NAMES)
DIGIT_BASE = INT_BASE + 2 * SMALL_INT + 1
VOCAB = DIGIT_BASE + 10

STATE_CAP = 160
TARGET_CAP = 64

SCHEMA_VERSION = 1
ARCHITECTURE = "stack-token-pairs-v1"
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "training_manifest.json"
WANDB_LINK_FILE = "wandb_run.url"

Product = tuple[tuple[Value, ...], ...]


def _encode_value(value: Value, out: list[int]) -> None:
    if isinstance(value, Bool):  # before Int: keep the two apart
        out.append(TRUE_TOKEN if value.value else FALSE_TOKEN)
    elif isinstance(value, Int):
        number = value.value
        if -SMALL_INT <= number <= SMALL_INT:
            out.append(INT_BASE + number + SMALL_INT)
        else:
            out.append(BIG)
            if number < 0:
                out.append(NEG)
            out.extend(DIGIT_BASE + int(digit) for digit in str(abs(number)))
    elif isinstance(value, Mark):
        out.append(MARK_TOKEN)
    elif isinstance(value, Sym):
        index = OP_INDEX.get(value.name)
        if index is None:
            raise ValueError(f"symbol {value.name!r} is not a {BASIS} op")
        out.append(SYMBOL_BASE + index)
    elif isinstance(value, Lst):
        out.append(OPEN)
        for item in value.items:
            _encode_value(item, out)
        out.append(CLOSE)
    else:
        raise ValueError(f"cannot encode {value!r}")


def encode_stack(stack: Sequence[Value], cap: int) -> list[int]:
    """Tokens bottom to top; over ``cap``, keep the top ``cap`` tokens."""

    out: list[int] = []
    for value in stack:
        _encode_value(value, out)
    return out[-cap:] if len(out) > cap else out


def encode_item(state: Sequence[Value], target_tokens: Sequence[int]) -> list[int]:
    return [*encode_stack(state, STATE_CAP), SEP, *target_tokens]


def encode_targets(outputs: Sequence[Sequence[Value]]) -> list[list[int]]:
    return [encode_stack(output, TARGET_CAP) for output in outputs]


def item_array(items: Sequence[Sequence[int]]) -> np.ndarray:
    """One node's items as a ``PAD``-padded ``(items, width)`` int16 block."""

    if not items:
        raise ValueError("a node needs at least one item")
    width = max(len(item) for item in items)
    block = np.full((len(items), width), PAD, dtype=np.int16)
    for row, item in enumerate(items):
        block[row, : len(item)] = item
    return block


def collate(
    blocks: Sequence[np.ndarray], device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor, Tensor]:
    """``(sequences, counts, item_mask)``: the contract of ``pair_batch``."""

    if not blocks:
        raise ValueError("Cannot construct an empty policy batch")
    width = max(block.shape[1] for block in blocks)
    total = sum(block.shape[0] for block in blocks)
    sequences = np.full((total, width), PAD, dtype=np.int64)
    row = 0
    for block in blocks:
        sequences[row : row + block.shape[0], : block.shape[1]] = block
        row += block.shape[0]
    counts = torch.tensor([block.shape[0] for block in blocks], dtype=torch.long)
    item_mask = torch.arange(int(counts.max()))[None, :] < counts[:, None]
    return (
        torch.from_numpy(sequences).to(device),
        counts.to(device),
        item_mask.to(device),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class StackPairPolicy(AlignedPairPolicy):
    """The aligned-pair encoder over the stack-token vocabulary."""

    def __init__(self, architecture: PolicyArchitecture):
        super().__init__(len(OP_NAMES), architecture)
        self.embedding = nn.Embedding(
            VOCAB, architecture.byte_embedding, padding_idx=PAD
        )

    def _encode_items(self, sequences: Tensor) -> Tensor:
        real = sequences != PAD
        embedded = self.embedding(sequences) * real.unsqueeze(-1)
        convolved = self.convolution(embedded.transpose(1, 2)).relu()
        convolved = convolved.masked_fill(~real.unsqueeze(1), -1e9)
        return convolved.max(dim=2).values

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class PolicyScorer:
    """Log-probabilities over the legal ops of a batch of nodes."""

    def __init__(self, model: StackPairPolicy, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def log_probs(self, blocks: Sequence[np.ndarray], legal: np.ndarray) -> np.ndarray:
        legal = np.asarray(legal, dtype=bool)
        if legal.shape != (len(blocks), len(OP_NAMES)):
            raise ValueError(
                f"legal must be ({len(blocks)}, {len(OP_NAMES)}), got {legal.shape}"
            )
        if not blocks:
            return np.zeros((0, len(OP_NAMES)))
        with torch.inference_mode():
            logits = self.model(*collate(blocks, self.device)).float().cpu()
        mask = torch.from_numpy(legal)
        masked = logits.masked_fill(~mask, -math.inf)
        result = torch.log_softmax(masked, dim=1)
        # A row with nothing legal is all -inf, not NaN.
        result = result.masked_fill(~mask, -math.inf)
        return result.numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    items: tuple[tuple[int, ...], ...]
    label: int
    task: int
    step: int
    remaining: int


def task_product(task: Task) -> tuple[Product, Product]:
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    return inputs, outputs


def replay_witness(task: Task) -> Iterator[tuple[int, Product, int]]:
    """``(step, product, next op index)`` along the witness on shown inputs."""

    product, _outputs = task_product(task)
    for step, name in enumerate(task.program):
        if name not in OP_INDEX:
            raise ValueError(f"{task.behaviour}: unknown op {name!r}")
        yield step, product, OP_INDEX[name]
        op = OP_BY_NAME[name]
        moved = []
        for stack in product:
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                raise ValueError(
                    f"{task.behaviour}: witness step {step} ({name}) refused: {outcome}"
                )
            moved.append(outcome[0])
        product = tuple(moved)


def witness_rows(tasks: Sequence[Task]) -> list[Row]:
    rows = []
    for task_index, task in enumerate(tasks):
        _inputs, outputs = task_product(task)
        targets = encode_targets(outputs)
        for step, product, label in replay_witness(task):
            items = tuple(
                tuple(encode_item(state, target))
                for state, target in zip(product, targets, strict=True)
            )
            rows.append(
                Row(
                    items=items,
                    label=label,
                    task=task_index,
                    step=step,
                    remaining=task.length - step,
                )
            )
    return rows


def validation_skeleton(task: Task, *, out_of: int = 10) -> bool:
    """Whether the task's skeleton is held out, about one in ``out_of``."""

    key = task.skeleton or task.behaviour
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % out_of == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _cross_entropy(
    model: StackPairPolicy,
    blocks: Sequence[np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(blocks), batch_size):
            batch = blocks[start : start + batch_size]
            target = torch.from_numpy(labels[start : start + batch_size]).to(device)
            logits = model(*collate(batch, device))
            total += float(nn.functional.cross_entropy(logits, target, reduction="sum"))
    return total / len(blocks)


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
    device: torch.device | str,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[StackPairPolicy, dict[str, Any]]:
    """Adam at a constant LR; returns the lowest-validation epoch's weights."""

    if not train or not validation:
        raise ValueError("train_policy needs non-empty train and validation rows")
    if epochs <= 0 or batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = StackPairPolicy(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_blocks = [item_array(row.items) for row in train]
    train_labels = np.array([row.label for row in train], dtype=np.int64)
    validation_blocks = [item_array(row.items) for row in validation]
    validation_labels = np.array([row.label for row in validation], dtype=np.int64)
    order_rng = np.random.default_rng(seed)

    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_loss = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(len(train_blocks))
        seen = 0
        total = 0.0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            inputs = collate([train_blocks[i] for i in chosen], device)
            target = torch.from_numpy(train_labels[chosen]).to(device)
            loss = nn.functional.cross_entropy(model(*inputs), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chosen)
            seen += len(chosen)
        validation_loss = _cross_entropy(
            model, validation_blocks, validation_labels, validation_batch_size, device
        )
        record = {
            "epoch": epoch,
            "train_cross_entropy": total / seen,
            "validation_cross_entropy": validation_loss,
        }
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        if epoch_callback is not None:
            epoch_callback(dict(record))

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    trainer_state = {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "history": history,
        "parameters": model.parameter_count(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }
    return model, trainer_state


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _vocabulary() -> dict[str, Any]:
    return {
        "size": VOCAB,
        "small_int": SMALL_INT,
        "state_cap": STATE_CAP,
        "target_cap": TARGET_CAP,
        "pad": PAD,
        "separator": SEP,
    }


def policy_config(model: StackPairPolicy) -> dict[str, Any]:
    architecture = model.architecture
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": {
            "name": ARCHITECTURE,
            "byte_embedding": architecture.byte_embedding,
            "conv_channels": architecture.conv_channels,
            "conv_kernel": CONV_KERNEL,
            "hidden": architecture.hidden,
        },
        "vocabulary": _vocabulary(),
        "basis": BASIS,
        "actions": list(OP_NAMES),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_policy(
    checkpoint: Path,
    model: StackPairPolicy,
    *,
    trainer_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wandb_url: str | None,
) -> Path:
    """Write one weights-only checkpoint to a directory that must not exist."""

    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{checkpoint.name}.staging-", dir=checkpoint.parent)
    )
    try:
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        save_file(
            state,
            str(staging / MODEL_FILE),
            metadata={"format": "pt", "architecture": ARCHITECTURE},
        )
        _write_json(staging / CONFIG_FILE, policy_config(model))
        _write_json(staging / TRAINER_STATE_FILE, dict(trainer_state))
        _write_json(staging / MANIFEST_FILE, {**manifest, "wandb_url": wandb_url})
        if wandb_url:
            (staging / WANDB_LINK_FILE).write_text(
                f"[InternetShortcut]\nURL={wandb_url}\n", encoding="utf-8"
            )
        if checkpoint.exists():
            raise FileExistsError(f"Checkpoint appeared during save: {checkpoint}")
        staging.replace(checkpoint)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return checkpoint


def load_policy(
    checkpoint: Path, *, device: torch.device | str = "cpu"
) -> StackPairPolicy:
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported stack policy schema: {config.get('schema_version')!r}"
        )
    if config.get("basis") != BASIS:
        raise ValueError(f"Unsupported stack policy basis: {config.get('basis')!r}")
    if config.get("actions") != list(OP_NAMES):
        raise ValueError("Stack policy action order differs from the mark-28 basis")
    if config.get("vocabulary") != _vocabulary():
        raise ValueError("Stack policy vocabulary differs from this module")
    dims = config.get("architecture")
    if (
        not isinstance(dims, Mapping)
        or dims.get("name") != ARCHITECTURE
        or dims.get("conv_kernel") != CONV_KERNEL
    ):
        raise ValueError(f"Unsupported stack policy architecture: {dims!r}")
    architecture = PolicyArchitecture(
        byte_embedding=dims["byte_embedding"],
        conv_channels=dims["conv_channels"],
        hidden=dims["hidden"],
    )
    model = StackPairPolicy(architecture)
    model.load_state_dict(load_file(str(checkpoint / MODEL_FILE), device="cpu"))
    return model.to(device).eval()
