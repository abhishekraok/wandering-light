"""novelty-reduction (#171): canonical keys, truncation and the cheap predictor's features.

- **Permutation-canonical keys** (F4). A state's examples, or a pair's aligned
  ``(s_i, t_i)`` examples, sorted by value key: two pairs that differ only in
  example order collide.
- **Truncation** (F8). The policy encoder keeps a state's top 160 tokens and a
  target's top 64; a pair with any example over either cap is truncated.
- **Encoded collisions** (F8). Distinct pairs whose encoded blocks are equal
  are indistinguishable to the heads.
- **Cheap features** (F3). d_ops, k, atoms, depth, top-item type and the
  path's last combinator. The combinator is path information the heads never
  see, so this predictor is a deliberately hard bar, not a like-for-like one.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from wandering_light.stack_machine import Bool, Int, Lst, value_key
from wandering_light.stack_policy import STATE_CAP, TARGET_CAP, encode_stack

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_beam import Product

COMBINATORS = ("map", "filter", "fold", "dip", "i", "ifte", "other")
TYPES = ("int", "bool", "list", "empty", "other")


def stack_key(stack) -> tuple:
    return tuple(value_key(value) for value in stack)


def state_key(state: Product) -> tuple:
    return tuple(sorted(stack_key(stack) for stack in state))


def pair_key(s: Product, t: Product) -> tuple:
    return tuple(
        sorted((stack_key(a), stack_key(b)) for a, b in zip(s, t, strict=True))
    )


def _tokens(stack) -> int:
    return len(encode_stack(stack, 1 << 30))


def truncated(s: Product, t: Product) -> bool:
    """Does any example exceed the encoder's state (160) or target (64) token cap?"""
    return any(_tokens(a) > STATE_CAP for a in s) or any(
        _tokens(b) > TARGET_CAP for b in t
    )


def block_digest(block: np.ndarray) -> bytes:
    return hashlib.blake2b(
        block.tobytes() + str(block.shape).encode(), digest_size=16
    ).digest()


def encoded_collisions(blocks: Sequence[np.ndarray], keys: Sequence) -> np.ndarray:
    """Per pair: does another pair with a different key share its encoded block?"""
    owners: dict[bytes, set] = {}
    digests = [block_digest(block) for block in blocks]
    for digest, key in zip(digests, keys, strict=True):
        owners.setdefault(digest, set()).add(key)
    return np.array([len(owners[d]) > 1 for d in digests], dtype=bool)


def _atoms(value) -> int:
    if isinstance(value, Lst):
        return 1 + sum(_atoms(item) for item in value.items)
    return 1


def _top_type(stack) -> str:
    if not stack:
        return "empty"
    top = stack[-1]
    if isinstance(top, Int):
        return "int"
    if isinstance(top, Bool):
        return "bool"
    if isinstance(top, Lst):
        return "list"
    return "other"


def _majority(stacks) -> str:
    kinds = [_top_type(stack) for stack in stacks]
    return max(TYPES, key=kinds.count)


def cheap_features(s: Product, t: Product, path: Sequence[str]) -> list[float]:
    atoms_s = sum(_atoms(v) for stack in s for v in stack) / len(s)
    atoms_t = sum(_atoms(v) for stack in t for v in stack) / len(t)
    last = next((op for op in reversed(path) if op in COMBINATORS), "other")
    return [
        float(len(path)),
        float(len(s)),
        atoms_s,
        atoms_t,
        atoms_t - atoms_s,
        float(np.mean([len(stack) for stack in s])),
        float(np.mean([len(stack) for stack in t])),
        *(float(_majority(s) == kind) for kind in TYPES),
        *(float(_majority(t) == kind) for kind in TYPES),
        *(float(last == name) for name in COMBINATORS),
    ]
