"""A one-step policy over `mark-28` product states, for PBE beam search.

The model is the aligned-pair encoder from `rollout_policy` unchanged: each
item is one ``(state, target)`` pair, items are pooled as an unordered set. What
differs is what an *item* is. On the typed-list side an item was one list
element and its counterpart in the target; here an item is one **example** of
the PBE task -- the current stack for that example's input, and that example's
expected output. A task with k shown examples is k items, so the set pooling is
also the answer to "how do several examples enter the policy".

Items are token streams, not bytes. Along v4 witness replays 44% of states are
wider than the byte encoder's 96-byte cap, because 80% of them sit inside an
open ``[`` with a long body on top. One token per op symbol, bracket, boolean
or small integer keeps a state under ``STATE_CAP`` in nearly every case, and a
state that still overflows keeps its *top*, which is what the next op reads.

Training rows replay each task's authored witness on its shown examples and
label every prefix with the witness's next op. The witness is an upper bound,
not a certified shortest program; the label is "what the author did here".
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from wandering_light.rollout_policy import AlignedPairPolicy, PolicyArchitecture
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    MARK_OPS,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    Value,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from wandering_light.stack_pbe import Task

SCHEMA_VERSION = 1
ARCHITECTURE = "aligned-example-pairs-stack-tokens-v1"
OP_NAMES: tuple[str, ...] = tuple(op.name for op in MARK_OPS)
OP_INDEX: dict[str, int] = {name: index for index, name in enumerate(OP_NAMES)}

# Vocabulary. Structural tokens first, then one token per op symbol, then the
# small integers as single tokens, then the digits a large integer spells out.
PAD, SEP, OPEN, CLOSE, MARK_TOKEN, TRUE_TOKEN, FALSE_TOKEN, NEG, BIG = range(9)
SYM_BASE = 9
INT_BASE = SYM_BASE + len(OP_NAMES)
SMALL_INT = 64
DIGIT_BASE = INT_BASE + 2 * SMALL_INT + 1
VOCAB = DIGIT_BASE + 10
STATE_CAP = 160
TARGET_CAP = 64
ITEM_WIDTH = STATE_CAP + 1 + TARGET_CAP

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "policy_config.json"
TRAINER_STATE_FILE = "trainer_state.json"
MANIFEST_FILE = "training_manifest.json"
WANDB_LINK_FILE = "wandb_run.url"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _encode_value(value: Value, out: list[int]) -> None:
    if isinstance(value, Int):
        if abs(value.value) <= SMALL_INT:
            out.append(INT_BASE + value.value + SMALL_INT)
            return
        out.append(BIG)
        if value.value < 0:
            out.append(NEG)
        out.extend(DIGIT_BASE + int(digit) for digit in str(abs(value.value)))
    elif isinstance(value, Bool):
        out.append(TRUE_TOKEN if value.value else FALSE_TOKEN)
    elif isinstance(value, Sym):
        out.append(SYM_BASE + OP_INDEX[value.name])
    elif isinstance(value, Mark):
        out.append(MARK_TOKEN)
    elif isinstance(value, Lst):
        out.append(OPEN)
        for item in value.items:
            _encode_value(item, out)
        out.append(CLOSE)
    else:  # pragma: no cover - the Value union is closed
        raise TypeError(f"cannot encode {value!r}")


def encode_stack(stack: Sequence[Value], cap: int) -> list[int]:
    """Bottom-to-top tokens; an overflowing stack keeps its top ``cap`` tokens."""
    out: list[int] = []
    for value in stack:
        _encode_value(value, out)
    return out[-cap:] if len(out) > cap else out


def encode_item(state: Sequence[Value], target_tokens: Sequence[int]) -> list[int]:
    return [*encode_stack(state, STATE_CAP), SEP, *target_tokens]


def encode_targets(outputs: Sequence[Sequence[Value]]) -> list[list[int]]:
    """Target tokens are fixed for a task; encode them once per search."""
    return [encode_stack(output, TARGET_CAP) for output in outputs]


def item_array(items: Sequence[Sequence[int]]) -> np.ndarray:
    """One node's items as a padded int16 block, the unit ``collate`` consumes."""
    width = max(len(item) for item in items)
    block = np.full((len(items), width), PAD, dtype=np.int16)
    for row, item in enumerate(items):
        block[row, : len(item)] = item
    return block


def collate(
    blocks: Sequence[np.ndarray], device: torch.device | str
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a batch of item blocks to a common width; same contract as `pair_batch`."""
    if not blocks:
        raise ValueError("cannot collate an empty batch")
    width = max(block.shape[1] for block in blocks)
    counts = [block.shape[0] for block in blocks]
    dense = np.full((sum(counts), width), PAD, dtype=np.int64)
    position = 0
    for block in blocks:
        rows, cols = block.shape
        dense[position : position + rows, :cols] = block
        position += rows
    count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    item_mask = (
        torch.arange(max(counts), device=device)[None, :] < count_tensor[:, None]
    )
    return torch.from_numpy(dense).to(device), count_tensor, item_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class StackPairPolicy(AlignedPairPolicy):
    """The aligned-pair encoder over stack tokens instead of bytes.

    ``actions`` is the action space the head scores. It defaults to the 28
    `mark-28` ops, which is what every published checkpoint was trained with and
    what `load_policy` reconstructs for them, so the default path is unchanged.
    A wider space (macro-action-space, #162, adds the 133 authoring macros) must
    keep the base ops as its prefix, so a `mark-28` head is a prefix of a macro
    head and the two orderings cannot silently disagree. The **encoder** is
    untouched either way: a macro is an action and never appears in a stack, so
    the token vocabulary does not grow.
    """

    def __init__(
        self,
        architecture: PolicyArchitecture,
        actions: Sequence[str] = OP_NAMES,
    ) -> None:
        actions = tuple(actions)
        if actions[: len(OP_NAMES)] != OP_NAMES:
            raise ValueError("an action space must begin with the mark-28 ops")
        self.actions = actions
        super().__init__(len(actions), architecture)
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
        return sum(parameter.numel() for parameter in self.parameters())


class PolicyScorer:
    """Masked log-probabilities over the 28 ops for a batch of product states."""

    def __init__(self, model: StackPairPolicy, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        # The pooled rows behind the last ``logits`` call. ``forward`` is
        # ``head(pooled_features(...))``, so keeping the intermediate costs
        # nothing and lets an observer lay the beam out in the policy's own
        # space without a second pass over every node.
        self.last_features: np.ndarray | None = None

    def logits(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        """The head's raw rows, one per node, over all 28 ops; nothing masked."""
        with torch.inference_mode():
            pooled = self.model.pooled_features(*collate(blocks, self.device))
            logits = self.model.head(pooled)
        self.last_features = pooled.float().cpu().numpy()
        return logits.float().cpu().numpy()

    def pooled_features(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        """What the head reads for each node: the pooled ``(state, target)`` set."""
        with torch.inference_mode():
            features = self.model.pooled_features(*collate(blocks, self.device))
        return features.float().cpu().numpy()

    def log_probs(self, blocks: Sequence[np.ndarray], legal: np.ndarray) -> np.ndarray:
        """Row-wise log-softmax over the legal ops; illegal entries are -inf."""
        logits = self.logits(blocks)
        logits = np.where(legal, logits, -np.inf)
        peak = logits.max(axis=1, keepdims=True)
        peak = np.where(np.isfinite(peak), peak, 0.0)
        shifted = logits - peak
        norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return shifted - norm


# ---------------------------------------------------------------------------
# Training rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One witness prefix on one task: its k example items and the next op."""

    items: np.ndarray
    label: int
    task: int
    step: int
    remaining: int


def replay_witness(
    task: Task,
) -> Iterable[tuple[int, tuple[tuple[Value, ...], ...], str]]:
    """Yield ``(step, product state, next op)`` along the task's witness."""
    states = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
    for step, name in enumerate(task.program):
        yield step, states, name
        moved = []
        for stack in states:
            outcome = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                raise ValueError(
                    f"{task.behaviour}: witness refused at {name}: {outcome}"
                )
            moved.append(outcome[0])
        states = tuple(moved)


def witness_rows(tasks: Sequence[Task]) -> list[Row]:
    rows = []
    for index, task in enumerate(tasks):
        targets = encode_targets(
            [parse_stack(text, basis=BASIS) for text in task.outputs]
        )
        for step, states, name in replay_witness(task):
            items = [
                encode_item(state, target)
                for state, target in zip(states, targets, strict=True)
            ]
            rows.append(
                Row(item_array(items), OP_INDEX[name], index, step, task.length - step)
            )
    return rows


def validation_skeleton(task: Task, *, out_of: int = 10) -> bool:
    """Hold out roughly one skeleton in ``out_of`` for model selection."""
    key = task.skeleton or task.behaviour
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % out_of == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _batch(
    rows: Sequence[Row], device: torch.device
) -> tuple[tuple[Tensor, ...], Tensor]:
    inputs = collate([row.items for row in rows], device)
    labels = torch.tensor([row.label for row in rows], dtype=torch.long, device=device)
    return inputs, labels


def _mean_cross_entropy(
    model: StackPairPolicy, rows: Sequence[Row], batch_size: int, device: torch.device
) -> float:
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            inputs, labels = _batch(rows[start : start + batch_size], device)
            total += nn.functional.cross_entropy(
                model(*inputs), labels, reduction="sum"
            ).item()
    return total / len(rows)


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
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    epoch_end: Callable[[int, StackPairPolicy, dict[str, Any]], None] | None = None,
) -> tuple[StackPairPolicy, dict[str, Any]]:
    """Plain Adam at a constant LR; the lowest-validation-CE weights come back.

    ``epoch_end(epoch, model, row)`` runs after each epoch's validation, with
    the model in eval mode holding that epoch's weights, so a caller can save
    them. Selecting the best-validation weights alone confounds a comparison
    between runs with how long the selected weights trained (#134).
    """
    if not train or not validation:
        raise ValueError("train and validation rows must both be non-empty")
    seed_everything(seed)
    target = torch.device(device)
    model = StackPairPolicy(architecture).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    order = list(range(len(train)))
    rng = random.Random(seed)
    best = (float("inf"), 0, None)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        train_total = torch.zeros((), device=target)
        for start in range(0, len(order), batch_size):
            inputs, labels = _batch(
                [train[i] for i in order[start : start + batch_size]], target
            )
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(*inputs), labels, reduction="sum")
            (loss / len(labels)).backward()
            optimizer.step()
            train_total += loss.detach()
        model.eval()
        validation_loss = _mean_cross_entropy(
            model, validation, validation_batch_size, target
        )
        row = {
            "epoch": epoch,
            "train_cross_entropy": train_total.item() / len(train),
            "validation_cross_entropy": validation_loss,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(dict(row))
        if epoch_end is not None:
            epoch_end(epoch, model, dict(row))
        if validation_loss < best[0]:
            best = (
                validation_loss,
                epoch,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
    model.load_state_dict(best[2])
    model.eval()
    return model, {
        "best_epoch": best[1],
        "best_validation_cross_entropy": best[0],
        "epochs_completed": epochs,
        "history": history,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "training_seconds": time.perf_counter() - started,
        "optimizer": {"class": "torch.optim.Adam", "learning_rate": learning_rate},
        "batch_size": batch_size,
        "seed": seed,
        "device": str(target),
        "parameters": model.parameter_count,
    }


# ---------------------------------------------------------------------------
# Checkpoints: weights-only, provenance recorded beside them
# ---------------------------------------------------------------------------


def policy_config(
    architecture: PolicyArchitecture, actions: Sequence[str] = OP_NAMES
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": {
            "name": ARCHITECTURE,
            "byte_embedding": architecture.byte_embedding,
            "conv_channels": architecture.conv_channels,
            "hidden": architecture.hidden,
        },
        "vocabulary": {
            "size": VOCAB,
            "small_int": SMALL_INT,
            "state_cap": STATE_CAP,
            "target_cap": TARGET_CAP,
        },
        "basis": BASIS,
        "ordered_actions": list(actions),
    }


def save_policy(
    checkpoint: Path,
    model: StackPairPolicy,
    *,
    trainer_state: dict[str, Any],
    manifest: dict[str, Any],
    wandb_url: str | None,
) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(checkpoint / MODEL_FILE), metadata={"format": "pt"})
    (checkpoint / CONFIG_FILE).write_text(
        json.dumps(
            policy_config(model.architecture, getattr(model, "actions", OP_NAMES)),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (checkpoint / TRAINER_STATE_FILE).write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True) + "\n"
    )
    (checkpoint / MANIFEST_FILE).write_text(
        json.dumps({**manifest, "wandb_url": wandb_url}, indent=2, sort_keys=True)
        + "\n"
    )
    if wandb_url:
        (checkpoint / WANDB_LINK_FILE).write_text(
            f"[InternetShortcut]\nURL={wandb_url}\n"
        )
    return checkpoint


def load_policy(
    checkpoint: Path, *, device: torch.device | str = "cpu"
) -> StackPairPolicy:
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text())
    if config.get("schema_version") != SCHEMA_VERSION or config.get("basis") != BASIS:
        raise ValueError(f"unsupported stack policy checkpoint: {checkpoint}")
    actions = tuple(config.get("ordered_actions", ()))
    # The check that mattered was that a checkpoint's head is not read under a
    # different ordering than it was trained with. That is still the check; it
    # is made against the checkpoint's own declared space rather than a module
    # constant, so a `mark-28` checkpoint reconstructs exactly as before and a
    # wider space (#162) is loadable instead of refused.
    if actions[: len(OP_NAMES)] != OP_NAMES:
        raise ValueError("checkpoint action order differs from mark-28")
    dims = config["architecture"]
    architecture = PolicyArchitecture(
        dims["byte_embedding"], dims["conv_channels"], dims["hidden"]
    )
    model = StackPairPolicy(architecture, actions)
    model.load_state_dict(load_file(str(checkpoint / MODEL_FILE), device="cpu"))
    return model.to(device).eval()
