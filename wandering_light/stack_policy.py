"""A one-step policy over ``mark-28`` product states, and what trains it.

The rollout policy in ``wandering_light/rollout_policy.py`` reads each example
as JSON *bytes*, 96 per side. That unit does not suit this machine: under an
open ``[`` the stack is a ``MARK`` followed by the body written so far, one
symbol per op, and a long body does not fit in 96 bytes. This module keeps that
encoder -- embed, convolve, max over positions, then mean- and max-pool over
the examples as an unordered set -- and changes only what a position *is*: one
op or one value, so a forty-op body is forty tokens.

An **item** is one shown example of a task: its current stack, a separator,
and the output that example must reach. A task with ``k`` shown examples is
``k`` items. The head has one output per op of the basis, in basis order.

Nothing here searches. ``wandering_light/stack_beam.py`` spends the scores.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, OPS, Task

SCHEMA_VERSION = 1
ARCHITECTURE = "aligned-item-pairs-stack-tokens-v1"

# The head's output order. Everything that turns a name into a column, or a
# column back into a name, goes through these two.
OP_NAMES: tuple[str, ...] = tuple(op.name for op in OPS)
OP_INDEX: dict[str, int] = {name: index for index, name in enumerate(OP_NAMES)}

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG = range(9)
SMALL_INT = 64
SYMBOL_BASE = 9  # one token per op name, in OP_NAMES order
INT_BASE = SYMBOL_BASE + len(OP_NAMES)  # one token per integer in [-64, 64]
DIGIT_BASE = INT_BASE + 2 * SMALL_INT + 1  # ten digits, for anything larger
VOCAB = DIGIT_BASE + 10

STATE_CAP = 160
TARGET_CAP = 64

Stack = Sequence[Value]
Product = tuple[tuple[Value, ...], ...]

_END = object()


def _decimal_digits(magnitude: int, cap: int) -> str:
    """The digits of ``magnitude``, or only its low ``cap`` when it has more.

    ``dup mul`` squares a number per pair of ops, so a search can hold an
    integer past the interpreter's int-to-str limit. Reducing first keeps the
    conversion at ``cap`` digits, and loses nothing: an integer wider than
    ``cap`` cannot keep its high digits through the truncation anyway.
    """
    limit = 10**cap
    if magnitude < limit:
        return str(magnitude)
    return str(magnitude % limit).zfill(cap)


def _scalar_tokens(value: Value, cap: int) -> list[int]:
    if isinstance(value, Int):
        number = value.value
        if -SMALL_INT <= number <= SMALL_INT:
            return [INT_BASE + SMALL_INT + number]
        digits = [
            DIGIT_BASE + int(digit) for digit in _decimal_digits(abs(number), cap)
        ]
        return [BIG, NEG, *digits] if number < 0 else [BIG, *digits]
    if isinstance(value, Bool):
        return [TRUE_TOKEN if value.value else FALSE_TOKEN]
    if isinstance(value, Mark):
        return [MARK_TOKEN]
    if isinstance(value, Sym):
        index = OP_INDEX.get(value.name)
        if index is None:
            raise ValueError(f"symbol {value.name!r} names no {BASIS} op")
        return [SYMBOL_BASE + index]
    raise TypeError(f"not a stack machine value: {value!r}")


def encode_stack(stack: Stack, cap: int) -> list[int]:
    """A stack as tokens, bottom to top, keeping the top ``cap`` when longer.

    The top is what the next op reads, and under an open ``[`` it is the end
    of the body being written, so that is the end worth keeping.

    The walk runs from the top down and stops once ``cap`` tokens are in hand,
    so the cost is bounded by ``cap`` rather than by the stack: ``dup cat``
    doubles a list per pair of ops and a search will find that.
    """
    if cap < 0:
        raise ValueError("cap must not be negative")
    tokens: list[int] = []  # top first; reversed on the way out
    frames = [reversed(stack)]
    while frames and len(tokens) < cap:
        value = next(frames[-1], _END)
        if value is _END:
            frames.pop()
            if frames:  # the frame that just ended was a list's items
                tokens.append(OPEN)
        elif isinstance(value, Lst):
            tokens.append(CLOSE)
            frames.append(reversed(value.items))
        else:
            tokens.extend(reversed(_scalar_tokens(value, cap)))
    del tokens[cap:]
    tokens.reverse()
    return tokens


def encode_targets(outputs: Sequence[Stack]) -> list[list[int]]:
    """Each expected output stack, encoded once per task rather than per node."""
    return [encode_stack(output, TARGET_CAP) for output in outputs]


def encode_item(state: Stack, target_tokens: Sequence[int]) -> list[int]:
    """One shown example: where its stack is, and where it has to end up."""
    return [*encode_stack(state, STATE_CAP), SEP, *target_tokens]


def item_array(items: Sequence[Sequence[int]]) -> np.ndarray:
    """One node's items as a ``(items, width)`` int16 block, padded with PAD.

    int16 because a training set holds one block per witness prefix, and the
    vocabulary fits with room to spare.
    """
    if not items:
        raise ValueError("a node needs at least one item")
    block = np.full((len(items), max(len(item) for item in items)), PAD, np.int16)
    for row, item in enumerate(items):
        block[row, : len(item)] = item
    return block


def collate(
    blocks: Sequence[np.ndarray], device: torch.device | str | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Batch blocks into ``(sequences, counts, item_mask)``, as ``pair_batch`` does.

    ``sequences`` stacks every item of every block, padded to the widest;
    ``counts[i]`` is how many of those rows belong to block ``i``; and
    ``item_mask[i, j]`` says whether block ``i`` has a ``j``-th item.
    """
    if not len(blocks):
        raise ValueError("Cannot construct an empty policy batch")
    counts = [block.shape[0] for block in blocks]
    if min(counts) < 1:
        raise ValueError("every block needs at least one item")
    width = max(block.shape[1] for block in blocks)
    sequences = np.full((sum(counts), width), PAD, dtype=np.int64)
    start = 0
    for block, count in zip(blocks, counts, strict=True):
        sequences[start : start + count, : block.shape[1]] = block
        start += count
    count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    item_mask = (
        torch.arange(max(counts), device=device)[None, :] < count_tensor[:, None]
    )
    return torch.from_numpy(sequences).to(device), count_tensor, item_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class StackPairPolicy(AlignedPairPolicy):
    """``AlignedPairPolicy`` over stack tokens, with one output per basis op.

    Two things differ from the parent, both forced by the vocabulary: the
    embedding table has ``VOCAB`` rows rather than 258, and the token that
    means "nothing here" is this module's ``PAD``, not the byte encoder's.
    The convolution, both poolings and the head are the parent's.
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

    @property
    def parameter_count(self) -> int:
        """Every weight in the model, the padding row of the embedding included."""
        return sum(parameter.numel() for parameter in self.parameters())


class PolicyScorer:
    """A model held ready for search: eval mode, on one device, numpy out."""

    def __init__(self, model: StackPairPolicy, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def log_probs(self, blocks: Sequence[np.ndarray], legal: Any) -> np.ndarray:
        """Log-probability of each op at each node, renormalised over the legal ones.

        ``legal`` is ``(nodes, ops)`` booleans in ``OP_NAMES`` order. Illegal
        entries come back ``-inf``, and so does a whole row with nothing legal,
        rather than the NaN a softmax over nothing would give.
        """
        mask = np.asarray(legal, dtype=bool)
        if mask.shape != (len(blocks), len(OP_NAMES)):
            raise ValueError(
                f"legal must be {(len(blocks), len(OP_NAMES))}, got {mask.shape}"
            )
        scores = np.full(mask.shape, -np.inf)
        if not len(blocks):
            return scores
        with torch.inference_mode():
            logits = self.model(*collate(blocks, self.device))
        logits = logits.double().cpu().numpy()
        if not np.isfinite(logits[mask]).all():
            raise ValueError("Policy produced a non-finite score for a legal op")
        rows = mask.any(axis=1)
        masked = np.where(mask[rows], logits[rows], -np.inf)
        peak = masked.max(axis=1, keepdims=True)
        total = peak + np.log(np.exp(masked - peak).sum(axis=1, keepdims=True))
        scores[rows] = masked - total
        return scores


# ---------------------------------------------------------------------------
# Supervision: one row per witness prefix
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Row:
    """One supervised step: the state a witness was in, and the op it took.

    ``items`` is the ``item_array`` block for that state, one item per shown
    example. ``task`` indexes the sequence ``witness_rows`` was given, ``step``
    counts the ops already applied, and ``remaining`` is how many the witness
    still has to go, this one included.
    """

    items: np.ndarray
    label: int
    task: int
    step: int
    remaining: int


def replay_witness(task: Task) -> Iterator[tuple[int, Product, str]]:
    """``(step, product, next op name)`` along a task's witness.

    The product is the shown inputs after ``step`` ops, which is exactly the
    state a search would be holding when it has to choose the next one. Each
    op gets the default fuel limit to itself, as it does in search.
    """
    product: Product = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    for step, name in enumerate(task.program):
        op = OP_BY_NAME.get(name)
        if op is None:
            raise ValueError(f"{task.behaviour}: step {step} names no op: {name!r}")
        moved = []
        for stack in product:
            outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                raise ValueError(
                    f"{task.behaviour}: witness refused at step {step} "
                    f"({name}): {outcome}"
                )
            moved.append(outcome[0])
        # Stepped before it is yielded, so a caller never holds a transition
        # the machine went on to refuse.
        yield step, product, name
        product = tuple(moved)


def witness_rows(tasks: Sequence[Task]) -> list[Row]:
    """Every prefix of every task's witness, labelled with the op that follows."""
    rows = []
    for index, task in enumerate(tasks):
        targets = encode_targets(
            [parse_stack(text, basis=BASIS) for text in task.outputs]
        )
        for step, product, name in replay_witness(task):
            items = item_array(
                [
                    encode_item(stack, target)
                    for stack, target in zip(product, targets, strict=True)
                ]
            )
            rows.append(
                Row(
                    items=items,
                    label=OP_INDEX[name],
                    task=index,
                    step=step,
                    remaining=task.length - step,
                )
            )
    return rows


def validation_skeleton(task: Task, *, out_of: int = 10) -> bool:
    """Whether a task's skeleton is one of the roughly 1 in ``out_of`` held out.

    Validation is carved from the training split by *skeleton* for the reason
    the corpus is split that way: two behaviours of one shape are the same
    problem with different leaves, so holding out by task would validate on
    shapes the model has already fitted. A task without a skeleton falls back
    to its behaviour, the next coarsest unit.
    """
    if type(out_of) is not int or out_of < 1:
        raise ValueError("out_of must be a positive integer")
    key = task.skeleton or task.behaviour
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16) % out_of == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _batch(rows: Sequence[Row], device: torch.device) -> tuple[tuple, Tensor]:
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
    """Train on witness rows and return the lowest-validation-loss weights.

    Adam at a constant learning rate on cross-entropy against the witness's
    next op. Initialisation and batch order both follow ``seed``, and the
    deterministic kernels are switched on for the duration and put back
    afterwards. The returned state carries no wall-clock or memory figures, so
    it is a function of the arguments alone; ``epoch_callback`` receives each
    epoch's history row as it lands, for a caller that wants to time or log.
    """
    if not train or not validation:
        raise ValueError("Training and validation rows must both be non-empty")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA stack-policy training requested without CUDA")

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_benchmark = torch.backends.cudnn.benchmark
    torch.manual_seed(seed)
    if target.type == "cuda":
        # cuBLAS needs this before it will run deterministically at all.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        model = StackPairPolicy(architecture).to(target)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        order = list(range(len(train)))
        rng = random.Random(seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        history: list[dict[str, Any]] = []
        optimizer_steps = 0

        for epoch in range(1, epochs + 1):
            model.train()
            rng.shuffle(order)
            train_total = torch.zeros((), dtype=torch.float64, device=target)
            for start in range(0, len(order), batch_size):
                rows = [train[index] for index in order[start : start + batch_size]]
                inputs, labels = _batch(rows, target)
                loss_sum = nn.functional.cross_entropy(
                    model(*inputs), labels, reduction="sum"
                )
                optimizer.zero_grad()
                (loss_sum / len(rows)).backward()
                optimizer.step()
                optimizer_steps += 1
                train_total += loss_sum.detach()

            model.eval()
            validation_total = 0.0
            with torch.inference_mode():
                for start in range(0, len(validation), validation_batch_size):
                    rows = validation[start : start + validation_batch_size]
                    inputs, labels = _batch(rows, target)
                    validation_total += nn.functional.cross_entropy(
                        model(*inputs), labels, reduction="sum"
                    ).item()
            validation_loss = validation_total / len(validation)
            row = {
                "epoch": epoch,
                "train_cross_entropy": train_total.item() / len(train),
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
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
        torch.backends.cudnn.benchmark = was_benchmark

    if best_state is None:
        raise RuntimeError("No epoch produced a finite validation cross-entropy")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "epochs_completed": epochs,
        "history": history,
        "parameters": model.parameter_count,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "optimizer_steps": optimizer_steps,
        "optimizer": "torch.optim.Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Checkpoints: weights only, plus what is needed to read them back
# ---------------------------------------------------------------------------

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "training_manifest.json"
WANDB_LINK_FILE = "wandb_run.url"


def _vocabulary_config() -> dict[str, int]:
    return {
        "size": VOCAB,
        "small_int": SMALL_INT,
        "symbol_base": SYMBOL_BASE,
        "int_base": INT_BASE,
        "digit_base": DIGIT_BASE,
        "state_cap": STATE_CAP,
        "target_cap": TARGET_CAP,
    }


def policy_config(architecture: PolicyArchitecture) -> dict[str, Any]:
    """What a set of weights means: the sizes, the tokens and the output order."""
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": {
            "name": ARCHITECTURE,
            "byte_embedding": architecture.byte_embedding,
            "conv_channels": architecture.conv_channels,
            "conv_kernel": CONV_KERNEL,
            "hidden": architecture.hidden,
        },
        "vocabulary": _vocabulary_config(),
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
    """Write one weights-only checkpoint to a directory that does not exist yet.

    The W&B URL is recorded in the manifest either way and as ``wandb_run.url``
    when there is one. It is not checked against any entity: lineage here is
    recorded, not enforced.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Built beside the destination and renamed into place, so a save that
    # fails halfway leaves nothing behind to trip the check above.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{checkpoint.name}.staging-", dir=checkpoint.parent)
    )
    try:
        weights = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        save_file(
            weights,
            str(staging / MODEL_FILE),
            metadata={"format": "pt", "architecture": ARCHITECTURE},
        )
        _write_json(staging / CONFIG_FILE, policy_config(model.architecture))
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
    checkpoint: Path | str, *, device: torch.device | str = "cpu"
) -> StackPairPolicy:
    """Rebuild a saved policy, in eval mode on ``device``.

    Refuses a checkpoint whose tokens or output columns this module would read
    differently, since those weights would load cleanly and score nonsense.
    """
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Stack policy config is not an object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported stack policy schema: {config.get('schema_version')!r}"
        )
    if config.get("basis") != BASIS:
        raise ValueError(f"Unsupported stack policy basis: {config.get('basis')!r}")
    if config.get("ordered_actions") != list(OP_NAMES):
        raise ValueError(f"Stack policy action order differs from {BASIS}'s")
    if config.get("vocabulary") != _vocabulary_config():
        raise ValueError("Stack policy vocabulary differs from this module's")
    dims = config.get("architecture")
    if (
        not isinstance(dims, Mapping)
        or dims.get("name") != ARCHITECTURE
        or dims.get("conv_kernel") != CONV_KERNEL
    ):
        raise ValueError("Unsupported stack policy architecture")
    architecture = PolicyArchitecture(
        byte_embedding=dims.get("byte_embedding"),
        conv_channels=dims.get("conv_channels"),
        hidden=dims.get("hidden"),
    )
    model = StackPairPolicy(architecture)
    model.load_state_dict(load_file(str(checkpoint / MODEL_FILE), device="cpu"))
    return model.to(device).eval()
