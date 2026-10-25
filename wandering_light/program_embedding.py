"""Learned program/I/O embeddings; semantics come from paired training data.

Only program tokens and example values enter the encoders. Neither program names
nor behaviour labels are features. Encoding never executes a program. Long inputs
raise rather than silently losing their suffix. Checkpoints contain weights,
configuration and JSON metadata, with no optimizer or training-resumption state.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from wandering_light.stack_machine import MARK_OPS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

Program = str | tuple[str, ...]
Examples = list[tuple[str, str]]
_PAD, _START, _END, _INPUT, _OUTPUT = 0, 257, 258, 259, 260
_TOKEN_START, _TOKEN_END = 261, 262
OP_NAMES = tuple(op.name for op in MARK_OPS)
_OP_IDS = {name: 263 + index for index, name in enumerate(OP_NAMES)}
_VOCAB_SIZE = 263 + len(OP_NAMES)


@dataclass(frozen=True)
class EmbeddingConfig:
    dim: int = 128
    hidden_dim: int = 128
    byte_dim: int = 48
    num_layers: int = 1
    max_program_bytes: int = 16384
    max_example_bytes: int = 16384

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _bytes(text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("Program tokens and example fields must be strings")
    return [byte + 1 for byte in text.encode("utf-8")]


class ProgramEmbedding(nn.Module):
    """Bidirectional token/byte GRUs and an invariant mean of I/O pair vectors.

    Known mark-28 operations get individual tokens; unknown symbols fall back to
    delimited UTF-8 bytes. Whitespace is canonicalized in program text. Each GRU
    pools masked mean/max states and both terminal states, preserving signals
    throughout long sequences rather than relying on the final forward state.

    A shared projection puts both modalities on the same unit sphere. Empty
    programs represent identity; an empty batch returns shape ``(0, dim)``.
    An I/O set must contain at least one pair. ``device`` optionally checks the
    model's placement; move the module with ``model.to(device)`` before encoding.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        super().__init__()
        self.config = config or EmbeddingConfig()
        self.byte_embedding = nn.Embedding(
            _VOCAB_SIZE, self.config.byte_dim, padding_idx=_PAD
        )
        self.program_encoder = nn.GRU(
            self.config.byte_dim,
            self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.io_encoder = nn.GRU(
            self.config.byte_dim,
            self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(6 * self.config.hidden_dim),
            nn.Linear(6 * self.config.hidden_dim, self.config.dim),
        )
        self.metadata: dict = {}

    @property
    def device(self) -> torch.device:
        return self.byte_embedding.weight.device

    def _device(self, device: str | torch.device | None) -> torch.device:
        requested = self.device if device is None else torch.device(device)
        if requested.type != self.device.type or (
            requested.index is not None and requested.index != self.device.index
        ):
            raise ValueError("device must match the model; call model.to(device) first")
        return self.device

    def _encode(self, rows: list[list[int]], encoder: nn.GRU) -> Tensor:
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
        tokens = pad_sequence(
            [torch.tensor(row, device=self.device) for row in rows], batch_first=True
        )
        packed = pack_padded_sequence(
            self.byte_embedding(tokens), lengths, batch_first=True, enforce_sorted=False
        )
        packed_states, hidden = encoder(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True)
        positions = torch.arange(states.shape[1], device=self.device)
        valid = positions[None, :] < lengths.to(self.device)[:, None]
        mean = (states * valid.unsqueeze(-1)).sum(dim=1) / lengths.to(
            device=self.device, dtype=states.dtype
        ).unsqueeze(-1)
        maximum = states.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(dim=1)
        terminal = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return torch.cat((mean, maximum, terminal), dim=1)

    def encode_programs(
        self, programs: list[Program], device: str | torch.device | None = None
    ) -> Tensor:
        self._device(device)
        rows = []
        for program in programs:
            if isinstance(program, tuple):
                if not all(isinstance(token, str) for token in program):
                    raise TypeError("Program tokens must be strings")
                text = " ".join(program)
            elif isinstance(program, str):
                text = program
            else:
                raise TypeError("A program must be a string or tuple of tokens")
            payload = _bytes(text)
            if len(payload) > self.config.max_program_bytes:
                raise ValueError(
                    f"Program has {len(payload)} bytes; limit is "
                    f"{self.config.max_program_bytes} (no truncation)"
                )
            tokens = [_START]
            for symbol in text.split():
                if symbol in _OP_IDS:
                    tokens.append(_OP_IDS[symbol])
                else:
                    tokens.extend([_TOKEN_START, *_bytes(symbol), _TOKEN_END])
            rows.append([*tokens, _END])
        if not rows:
            return self.byte_embedding.weight.new_empty((0, self.config.dim))
        return functional.normalize(
            self.projection(self._encode(rows, self.program_encoder)), dim=1
        )

    def encode_io(
        self,
        examples: list[Examples],
        device: str | torch.device | None = None,
    ) -> Tensor:
        self._device(device)
        if not examples:
            return self.byte_embedding.weight.new_empty((0, self.config.dim))
        rows, counts = [], []
        for pairs in examples:
            if not pairs:
                raise ValueError("Each I/O example set must contain at least one pair")
            counts.append(len(pairs))
            for pair in pairs:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(
                        "Each I/O pair must contain input and output strings"
                    )
                source, target = map(_bytes, pair)
                if len(source) + len(target) > self.config.max_example_bytes:
                    raise ValueError(
                        f"I/O pair has {len(source) + len(target)} bytes; limit is "
                        f"{self.config.max_example_bytes} (no truncation)"
                    )
                # Markers lie outside byte IDs, including literal NUL/newline bytes.
                rows.append([_START, _INPUT, *source, _OUTPUT, *target, _END])
        pair_vectors = self._encode(rows, self.io_encoder)
        sets = torch.stack([part.mean(dim=0) for part in pair_vectors.split(counts)])
        return functional.normalize(self.projection(sets), dim=1)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save JSON provenance plus weights; callers supply run/data provenance."""
        details = json.loads(
            json.dumps(self.metadata if metadata is None else metadata)
        )
        torch.save(
            {
                "format_version": 2,
                "config": asdict(self.config),
                "metadata": details,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.state_dict().items()
                },
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> ProgramEmbedding:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 2:
            raise ValueError("Unsupported program embedding checkpoint format")
        model = cls(EmbeddingConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.metadata = payload["metadata"]
        return model.to(device).eval()


def symmetric_multi_positive_contrastive_loss(
    left: Tensor,
    right: Tensor,
    labels: Sequence[str | int] | Tensor | None = None,
    positive_mask: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    """Average both directional supervised contrastive losses.

    Labels identify equivalent rows in aligned batches. Additional positives can
    be supplied by a boolean mask (rectangular batches are supported with a mask
    alone). The union prevents same-label pairs from becoming false negatives.
    Without either, aligned diagonal pairs are positives. Each positive gets
    equal weight within its anchor, so every positive receives a training signal.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Embeddings must be matrices with matching feature dimensions")
    if left.shape[0] == 0 or right.shape[0] == 0:
        raise ValueError("Contrastive batches must be nonempty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    shape = (left.shape[0], right.shape[0])
    mask = torch.zeros(shape, device=left.device, dtype=torch.bool)
    if positive_mask is not None:
        if positive_mask.shape != shape or positive_mask.dtype != torch.bool:
            raise ValueError(
                "positive_mask must be a boolean matrix matching batch sizes"
            )
        mask |= positive_mask.to(left.device)
    if labels is not None:
        if shape[0] != shape[1] or len(labels) != shape[0]:
            raise ValueError("labels require aligned batches of the same size")
        if isinstance(labels, Tensor):
            if labels.ndim != 1:
                raise ValueError("labels must be one-dimensional")
            label_values = labels.detach().cpu().tolist()
        else:
            label_values = list(labels)
        mask |= torch.tensor(
            [[first == second for second in label_values] for first in label_values],
            device=left.device,
            dtype=torch.bool,
        )
    elif positive_mask is None:
        if shape[0] != shape[1]:
            raise ValueError("Unaligned batches require a positive_mask")
        mask.fill_diagonal_(True)
    if not mask.any(dim=1).all() or not mask.any(dim=0).all():
        raise ValueError("Every row and column must have at least one positive")
    logits = (
        functional.normalize(left, dim=1)
        @ functional.normalize(right, dim=1).T
        / temperature
    )

    def directional(scores: Tensor, positives: Tensor) -> Tensor:
        log_prob = functional.log_softmax(scores, dim=1)
        return -(
            log_prob.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)
        ).mean()

    return (directional(logits, mask) + directional(logits.T, mask.T)) / 2


class ProgramIndex:
    """An in-memory cosine index over program embeddings, with both query modes.

    Refit after changing model weights. Queries use evaluation mode temporarily,
    preserving the caller's training mode. Returned indices refer to ``programs``.
    """

    def __init__(self, model: ProgramEmbedding):
        self.model = model
        self.programs: list[Program] = []
        self.vectors: Tensor | None = None

    def fit(self, programs: list[Program], batch_size: int = 128) -> ProgramIndex:
        if not programs:
            raise ValueError("Cannot fit an empty program index")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                vectors = torch.cat(
                    [
                        self.model.encode_programs(programs[start : start + batch_size])
                        for start in range(0, len(programs), batch_size)
                    ]
                )
        finally:
            self.model.train(was_training)
        self.programs = list(programs)
        self.vectors = vectors
        return self

    def _search(
        self, query: Program | Examples, k: int, *, io: bool
    ) -> tuple[Tensor, Tensor]:
        if self.vectors is None:
            raise ValueError("Fit the program index before searching")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                encoded = (
                    self.model.encode_io([query])
                    if io
                    else self.model.encode_programs([query])
                )
                scores = (encoded @ self.vectors.to(encoded.device).T).squeeze(0)
                best_scores, indices = scores.topk(min(k, len(self.programs)))
        finally:
            self.model.train(was_training)
        return indices, best_scores

    def search_program(self, program: Program, k: int = 10) -> tuple[Tensor, Tensor]:
        return self._search(program, k, io=False)

    def search_io(self, examples: Examples, k: int = 10) -> tuple[Tensor, Tensor]:
        return self._search(examples, k, io=True)
