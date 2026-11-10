"""One-step policy over ``mark-28`` product states, for ``stack_pbe_v4``.

A node of a PBE search is a *product*: one machine stack per shown example,
all moved by the same program. The policy reads that node as a set of
**items**, one per shown example -- the example's current stack and its
expected output -- and scores the 28 ops of ``mark-28`` as the next step.

Items are op/value **tokens**, not bytes. Under an open ``[`` the machine
captures every op as a symbol, so a quotation body is a run of op names; one
token per name keeps a twenty-op body at twenty tokens where its printed form
would be well over a hundred bytes. Small integers are one token each, larger
ones are spelled in decimal digits, and lists are bracketed by ``OPEN`` /
``CLOSE``. A stack longer than ``STATE_CAP`` tokens keeps its *top*, which is
where every op acts.

The encoder is the aligned-pair arm of ``wandering_light/rollout_policy.py``
reused unchanged: each item through an embedding and a 1-D convolution, max
over positions, then mean and max over the item set. Only the vocabulary
differs, so ``StackPairPolicy`` swaps the embedding and the padding token and
nothing else.

Training rows come from replaying each task's authored witness over its shown
inputs: one row per witness prefix, labelled with the witness's next op.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import time
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
# Actions
# ---------------------------------------------------------------------------

#: The policy head's output order: ``MARK_OPS``, exactly as ``stack_pbe.OPS``.
OP_NAMES: tuple[str, ...] = tuple(op.name for op in OPS)
OP_INDEX: dict[str, int] = {name: index for index, name in enumerate(OP_NAMES)}

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG = range(9)
SPECIAL_NAMES: tuple[str, ...] = (
    "PAD",
    "SEP",
    "OPEN",
    "CLOSE",
    "MARK",
    "TRUE",
    "FALSE",
    "NEG",
    "BIG",
)
SYMBOL_BASE = len(SPECIAL_NAMES)
SMALL_INT = 64
INT_BASE = SYMBOL_BASE + len(OP_NAMES)
DIGIT_BASE = INT_BASE + 2 * SMALL_INT + 1
VOCAB = DIGIT_BASE + 10

STATE_CAP = 160
TARGET_CAP = 64

SYMBOL_TOKEN: dict[str, int] = {
    name: SYMBOL_BASE + index for index, name in enumerate(OP_NAMES)
}


def _int_tokens(value: int) -> list[int]:
    if -SMALL_INT <= value <= SMALL_INT:
        return [INT_BASE + value + SMALL_INT]
    sign = [NEG] if value < 0 else []
    return [BIG, *sign, *(DIGIT_BASE + int(digit) for digit in str(abs(value)))]


def _value_tokens(value: Value, out: list[int]) -> None:
    if isinstance(value, Int):
        out.extend(_int_tokens(value.value))
    elif isinstance(value, Bool):
        out.append(TRUE_TOKEN if value.value else FALSE_TOKEN)
    elif isinstance(value, Mark):
        out.append(MARK_TOKEN)
    elif isinstance(value, Sym):
        token = SYMBOL_TOKEN.get(value.name)
        if token is None:
            raise ValueError(f"symbol {value.name!r} is not a {BASIS} op")
        out.append(token)
    elif isinstance(value, Lst):
        out.append(OPEN)
        for item in value.items:
            _value_tokens(item, out)
        out.append(CLOSE)
    else:
        raise ValueError(f"cannot tokenize {value!r}")


def encode_stack(stack: Sequence[Value], cap: int) -> list[int]:
    """Tokens for a stack, bottom to top, keeping the last ``cap`` (its top)."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    tokens: list[int] = []
    for value in stack:
        _value_tokens(value, tokens)
    return tokens[-cap:] if len(tokens) > cap else tokens


def encode_item(state: Sequence[Value], target_tokens: Sequence[int]) -> list[int]:
    """One shown example at one node: its current stack, ``SEP``, its target."""
    return [*encode_stack(state, STATE_CAP), SEP, *target_tokens]


def encode_targets(outputs: Sequence[Sequence[Value]]) -> list[list[int]]:
    """Each shown output, encoded once per task at ``TARGET_CAP``."""
    return [encode_stack(output, TARGET_CAP) for output in outputs]


def item_array(items: Sequence[Sequence[int]]) -> np.ndarray:
    """One node's items, right-padded with ``PAD`` into an int16 block."""
    if not items:
        raise ValueError("a node needs at least one item")
    width = max(len(item) for item in items)
    block = np.full((len(items), width), PAD, dtype=np.int16)
    for row, item in enumerate(items):
        block[row, : len(item)] = item
    return block


def collate(
    blocks: Sequence[np.ndarray], device: torch.device | str | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """``(sequences, counts, item_mask)`` -- the contract of ``pair_batch``.

    Blocks are padded to a common width; the encoder masks ``PAD`` out, so the
    width a node is padded to never changes its logits.
    """
    if not blocks:
        raise ValueError("cannot collate an empty batch")
    width = max(block.shape[1] for block in blocks)
    total = sum(block.shape[0] for block in blocks)
    sequences = np.full((total, width), PAD, dtype=np.int64)
    counts = []
    row = 0
    for block in blocks:
        rows, columns = block.shape
        sequences[row : row + rows, :columns] = block
        row += rows
        counts.append(rows)
    count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    item_mask = (
        torch.arange(int(count_tensor.max()), device=device)[None, :]
        < count_tensor[:, None]
    )
    return torch.from_numpy(sequences).to(device), count_tensor, item_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

ARCHITECTURE = "stack-token-aligned-pairs-v1"
SCHEMA_VERSION = 1


def parameter_count(architecture: PolicyArchitecture) -> int:
    """Trainable parameters of a ``StackPairPolicy`` with these dimensions."""
    embedding = VOCAB * architecture.byte_embedding
    convolution = (
        architecture.conv_channels * architecture.byte_embedding * CONV_KERNEL
        + architecture.conv_channels
    )
    hidden = 2 * architecture.conv_channels * architecture.hidden + architecture.hidden
    actions = architecture.hidden * len(OP_NAMES) + len(OP_NAMES)
    return embedding + convolution + hidden + actions


class StackPairPolicy(AlignedPairPolicy):
    """The aligned-pair encoder over the stack token vocabulary.

    Identical to ``AlignedPairPolicy`` but for the embedding table (``VOCAB``
    rows, ``PAD`` as its padding index) and which token counts as absent.
    """

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
    """Batched, legality-masked log-probabilities from a ``StackPairPolicy``."""

    def __init__(self, model: StackPairPolicy, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def log_probs(
        self, blocks: Sequence[np.ndarray], legal: np.ndarray | Sequence
    ) -> np.ndarray:
        """Row-wise log-softmax over the legal ops only; illegal ops are -inf.

        ``legal`` is ``(len(blocks), len(OP_NAMES))`` booleans. A row with no
        legal op is all -inf.
        """
        mask = np.asarray(legal, dtype=bool)
        if mask.shape != (len(blocks), len(OP_NAMES)):
            raise ValueError(
                f"legal mask must be {(len(blocks), len(OP_NAMES))}, got {mask.shape}"
            )
        if not blocks:
            return np.zeros((0, len(OP_NAMES)), dtype=np.float64)
        with torch.inference_mode():
            logits = self.model(*collate(blocks, self.device)).double().cpu()
        legal_tensor = torch.from_numpy(mask)
        masked = logits.masked_fill(~legal_tensor, float("-inf"))
        result = torch.log_softmax(masked, dim=1)
        # A row with nothing legal softmaxes to NaN; it has no distribution.
        result = result.masked_fill(~legal_tensor, float("-inf"))
        return result.numpy()


# ---------------------------------------------------------------------------
# Witness rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Row:
    """One supervised step: a node's item block and the witness's next op.

    ``items`` is the node's ``item_array`` block (read-only), one row per shown
    example. Compared by identity, since the block is an array.
    """

    items: np.ndarray
    label: int
    task: int
    step: int
    remaining: int


Product = tuple[tuple[Value, ...], ...]


def _shown(task: Task) -> tuple[Product, Product]:
    inputs = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    outputs = tuple(parse_stack(text, basis=BASIS) for text in task.outputs)
    return inputs, outputs


def node_block(product: Product, targets: Sequence[Sequence[int]]) -> np.ndarray:
    """The item block for one search node."""
    return item_array(
        [
            encode_item(stack, target)
            for stack, target in zip(product, targets, strict=True)
        ]
    )


def replay_witness(task: Task) -> Iterator[tuple[int, Product, str]]:
    """``(step, product, next op)`` along the witness, on the shown inputs.

    Every op is applied, including the last, under the default fuel limit per
    op (the same limit ``stack_beam.step_all`` uses), so a witness the machine
    refuses anywhere raises ``ValueError``.
    """
    product, _ = _shown(task)
    for step, name in enumerate(task.program):
        yield step, product, name
        op = OP_BY_NAME.get(name)
        if op is None:
            raise ValueError(f"{task.behaviour}: unknown op {name!r} at step {step}")
        moved = []
        for stack in product:
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                raise ValueError(
                    f"{task.behaviour}: witness refused at step {step} ({name}): "
                    f"{outcome}"
                )
            moved.append(outcome[0])
        product = tuple(moved)


def witness_rows(tasks: Sequence[Task]) -> list[Row]:
    """One row per witness prefix of every task, items one per shown example."""
    rows: list[Row] = []
    for task_index, task in enumerate(tasks):
        _, outputs = _shown(task)
        targets = encode_targets(outputs)
        for step, product, name in replay_witness(task):
            block = node_block(product, targets)
            block.setflags(write=False)
            rows.append(
                Row(
                    items=block,
                    label=OP_INDEX[name],
                    task=task_index,
                    step=step,
                    remaining=task.length - step,
                )
            )
    return rows


def validation_skeleton(task: Task, *, out_of: int = 10) -> bool:
    """Whether a task's skeleton is held out for validation (~1 in ``out_of``).

    Keyed on the skeleton, falling back to the behaviour for a corpus that
    predates skeletons, so no program shape is in both train and validation.
    """
    if type(out_of) is not int or out_of <= 0:
        raise ValueError("out_of must be a positive integer")
    key = task.skeleton or task.behaviour
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % out_of == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _rows_batch(
    rows: Sequence[Row], device: torch.device
) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
    labels = torch.tensor([row.label for row in rows], dtype=torch.long, device=device)
    return collate([row.items for row in rows], device), labels


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
    """Train deterministically; return the lowest-validation-loss weights.

    Adam at a constant learning rate on summed-then-averaged cross-entropy,
    one validation pass per epoch. Initialisation and data order follow
    ``seed``, and torch's deterministic kernels are on for the duration.
    """
    if not train or not validation:
        raise ValueError("training and validation rows must both be non-empty")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested without CUDA")

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = StackPairPolicy(architecture).to(target)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        order = list(range(len(train)))
        rng = random.Random(seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        history: list[dict[str, Any]] = []
        steps = 0
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            rng.shuffle(order)
            train_total = 0.0
            for start in range(0, len(order), batch_size):
                batch = [train[index] for index in order[start : start + batch_size]]
                inputs, labels = _rows_batch(batch, target)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(*inputs), labels)
                loss.backward()
                optimizer.step()
                steps += 1
                train_total += loss.item() * len(batch)

            model.eval()
            validation_total = 0.0
            with torch.inference_mode():
                for start in range(0, len(validation), validation_batch_size):
                    batch = validation[start : start + validation_batch_size]
                    inputs, labels = _rows_batch(batch, target)
                    validation_total += nn.functional.cross_entropy(
                        model(*inputs), labels, reduction="sum"
                    ).item()
            row = {
                "epoch": epoch,
                "train_cross_entropy": train_total / len(train),
                "validation_cross_entropy": validation_total / len(validation),
            }
            history.append(row)
            if epoch_callback is not None:
                epoch_callback(dict(row))
            if row["validation_cross_entropy"] < best_loss:
                best_loss = row["validation_cross_entropy"]
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        seconds = time.perf_counter() - started
    finally:
        torch.use_deterministic_algorithms(was_deterministic)

    if best_state is None:
        raise RuntimeError("training produced no finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "epochs_completed": epochs,
        "history": history,
        "parameters": model.parameter_count(),
        "architecture": architecture_config(architecture),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "optimizer": "torch.optim.Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer_steps": steps,
        "seed": seed,
        "training_seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "training_manifest.json"
WANDB_LINK_FILE = "wandb_run.url"


def architecture_config(architecture: PolicyArchitecture) -> dict[str, Any]:
    return {
        "name": ARCHITECTURE,
        "byte_embedding": architecture.byte_embedding,
        "conv_channels": architecture.conv_channels,
        "conv_kernel": CONV_KERNEL,
        "hidden": architecture.hidden,
    }


def vocabulary_config() -> dict[str, Any]:
    return {
        "size": VOCAB,
        "specials": list(SPECIAL_NAMES),
        "symbols": list(OP_NAMES),
        "small_int": SMALL_INT,
        "digits": 10,
        "state_cap": STATE_CAP,
        "target_cap": TARGET_CAP,
    }


def policy_config(architecture: PolicyArchitecture) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture_config(architecture),
        "vocabulary": vocabulary_config(),
        "basis": BASIS,
        "ordered_actions": list(OP_NAMES),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_policy(
    checkpoint: Path | str,
    model: StackPairPolicy,
    *,
    trainer_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    wandb_url: str | None,
) -> Path:
    """Write a weights-only checkpoint into a new directory.

    Staged beside the target and renamed into place, so a failed save leaves
    nothing behind. The W&B link file is written only when there is a URL;
    the manifest records the URL (or ``None``) either way.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint}")
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
        _write_json(staging / CONFIG_FILE, policy_config(model.architecture))
        _write_json(staging / TRAINER_STATE_FILE, dict(trainer_state))
        _write_json(staging / MANIFEST_FILE, {**dict(manifest), "wandb_url": wandb_url})
        if wandb_url:
            (staging / WANDB_LINK_FILE).write_text(
                f"[InternetShortcut]\nURL={wandb_url}\n", encoding="utf-8"
            )
        if checkpoint.exists():
            raise FileExistsError(f"checkpoint appeared during save: {checkpoint}")
        staging.replace(checkpoint)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return checkpoint


def load_policy(
    checkpoint: Path | str, *, device: torch.device | str = "cpu"
) -> StackPairPolicy:
    """Rebuild a saved policy, in eval mode on ``device``."""
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported stack policy schema {config.get('schema_version')!r}"
        )
    if config.get("basis") != BASIS:
        raise ValueError(f"unsupported basis {config.get('basis')!r}, want {BASIS}")
    if config.get("ordered_actions") != list(OP_NAMES):
        raise ValueError("checkpoint action order differs from mark-28")
    if config.get("vocabulary") != vocabulary_config():
        raise ValueError("checkpoint vocabulary differs from this module's")
    dims = config.get("architecture")
    if (
        not isinstance(dims, Mapping)
        or dims.get("name") != ARCHITECTURE
        or dims.get("conv_kernel") != CONV_KERNEL
    ):
        raise ValueError(f"unsupported stack policy architecture {dims!r}")
    architecture = PolicyArchitecture(
        byte_embedding=dims["byte_embedding"],
        conv_channels=dims["conv_channels"],
        hidden=dims["hidden"],
    )
    model = StackPairPolicy(architecture)
    model.load_state_dict(load_file(str(checkpoint / MODEL_FILE), device="cpu"))
    return model.to(device).eval()
