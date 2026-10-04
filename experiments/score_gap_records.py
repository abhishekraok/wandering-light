"""Compact #180 attempt codec; not a recorder, runner or result verifier.

Reasons and moved products are reconstructed from typed source parents and
pinned low-level machine operations. In particular there is no reason dictionary.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ATTEMPT_STRUCT = struct.Struct("<BBBQQ")
MAX_ATTEMPTS = 12_000_000
WORST_ATTEMPTS = 144 * (28 + 28 * 28 + 46 * 64 * 28)
WORST_ATTEMPT_BYTES = WORST_ATTEMPTS * ATTEMPT_STRUCT.size


@dataclass(frozen=True, slots=True)
class Attempt:
    depth: int  # zero-based source layer, 0..47
    node: int  # index in that layer's source beam, 0..63
    op: int  # index in pinned OPS, 0..27
    fuel: int  # successful shown-example fuel only, as production charges it
    stepped: int  # successes before refusal, or full source-product length

    def __post_init__(self) -> None:
        for name, maximum in (
            ("depth", 47),
            ("node", 63),
            ("op", 27),
            ("fuel", (1 << 64) - 1),
            ("stepped", (1 << 64) - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"invalid attempt {name}")

    def encode(self) -> bytes:
        return ATTEMPT_STRUCT.pack(
            self.depth, self.node, self.op, self.fuel, self.stepped
        )


def decode_attempts(payload: bytes) -> Iterator[Attempt]:
    """Decode complete fixed records in order; reject invalid or truncated data.

    The caller still owes cell/layer boundaries, actual parent/op identities,
    complete callback order, low-level replay, counters and all global bounds.
    """
    if not isinstance(payload, bytes):
        raise ValueError("attempt payload must be bytes")
    if len(payload) % ATTEMPT_STRUCT.size:
        raise ValueError("truncated attempt record")
    if len(payload) // ATTEMPT_STRUCT.size > MAX_ATTEMPTS:
        raise ValueError("too many attempt records")
    for fields in ATTEMPT_STRUCT.iter_unpack(payload):
        yield Attempt(*fields)
