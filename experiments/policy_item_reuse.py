"""Observe exact padded policy items; never serve cached model outputs (#184)."""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from fractions import Fraction

import numpy as np

from wandering_light.stack_policy import PAD, PolicyScorer

CAPACITY = 4096


def empty_totals():
    return {
        "calls": 0,
        "items": 0,
        "positions": 0,
        "hits": 0,
        "hit_positions": 0,
        "within_call_hits": 0,
        "cross_call_hits": 0,
    }


def valid_digest(key):
    return (
        isinstance(key, str)
        and len(key) == 64
        and all(c in "0123456789abcdef" for c in key)
    )


class ItemRecorder:
    """A single task's row-ordered LRU plus the ordered counter records."""

    def __init__(self, capacity=CAPACITY, digest=None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("positive integer capacity required")
        self.capacity = capacity
        self.digest = digest or (lambda key: hashlib.sha256(key).hexdigest())
        self.cache = OrderedDict()
        self.active_digests = {}
        self.calls = []
        self.totals = empty_totals()

    def observe(self, blocks):
        if not blocks or any(
            block.ndim != 2
            or not block.shape[0]
            or not block.shape[1]
            or not np.issubdtype(block.dtype, np.integer)
            for block in blocks
        ):
            raise ValueError("nonempty integer item blocks required")
        width = max(block.shape[1] for block in blocks)
        count = sum(block.shape[0] for block in blocks)
        call = self.totals["calls"]
        keys = []
        for block in blocks:
            for row in block:
                padded = np.full(width, PAD, dtype="<i8")
                padded[: len(row)] = row
                exact = struct.pack("<Q", width) + padded.tobytes()
                digest = self.digest(exact)
                if not valid_digest(digest):
                    raise ValueError("invalid item digest")
                if (
                    digest in self.active_digests
                    and self.active_digests[digest] != exact
                ):
                    raise ValueError("distinct active item keys share a digest")
                keys.append(digest)
                if exact in self.cache:
                    _, previous_call = self.cache.pop(exact)
                    kind = (
                        "within_call_hits"
                        if previous_call == call
                        else "cross_call_hits"
                    )
                    self.totals[kind] += 1
                    self.totals["hits"] += 1
                    self.totals["hit_positions"] += width
                elif len(self.cache) == self.capacity:
                    old, (old_digest, _) = self.cache.popitem(last=False)
                    assert self.active_digests.pop(old_digest) == old
                self.cache[exact] = (digest, call)
                self.active_digests[digest] = exact
        self.calls.append({"n": count, "width": width, "keys": keys})
        self.totals["calls"] += 1
        self.totals["items"] += count
        self.totals["positions"] += count * width


class RecordingScorer(PolicyScorer):
    """The ordinary scorer, with observation before its unchanged logits call."""

    recorder: ItemRecorder | None = None

    def logits(self, blocks):
        if self.recorder is None:
            raise ValueError("start the task recorder before scoring")
        self.recorder.observe(blocks)
        return super().logits(blocks)


def replay_counts(records, capacity=CAPACITY):
    """Recompute arithmetic from the saved digest order, not ItemRecorder state."""
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("positive integer capacity required")
    recent = OrderedDict()
    totals = empty_totals()
    for call, record in enumerate(records):
        if set(record) != {"n", "width", "keys"}:
            raise ValueError("malformed call record")
        n, width, keys = record["n"], record["width"], record["keys"]
        if (
            type(n) is not int
            or n <= 0
            or type(width) is not int
            or width <= 0
            or not isinstance(keys, list)
            or len(keys) != n
        ):
            raise ValueError("invalid item count/width")
        for key in keys:
            if not valid_digest(key):
                raise ValueError("invalid item digest")
            if key in recent:
                prior, prior_width = recent.pop(key)
                if prior_width != width:
                    raise ValueError("active digest reused at a different padded width")
                totals["hits"] += 1
                totals["hit_positions"] += width
                totals["within_call_hits" if prior == call else "cross_call_hits"] += 1
            elif len(recent) >= capacity:
                recent.popitem(last=False)
            recent[key] = (call, width)
        totals["calls"] += 1
        totals["items"] += n
        totals["positions"] += n * width
    return totals


def eligibility(totals):
    if len(totals) != 3 or any(
        type(t["positions"]) is not int
        or type(t["hit_positions"]) is not int
        or t["positions"] <= 0
        or not 0 <= t["hit_positions"] <= t["positions"]
        for t in totals
    ):
        raise ValueError("three nonempty complete repeats required")
    fractions = [Fraction(t["hit_positions"], t["positions"]) for t in totals]
    enough = all(f >= Fraction(1, 4) for f in fractions)
    stable = max(fractions) - min(fractions) < Fraction(1, 20)
    return {
        "reading": "REUSE_OPPORTUNITY" if enough and stable else "NOT_JUSTIFIED",
        "fractions_exact": [[f.numerator, f.denominator] for f in fractions],
        "fractions": [float(f) for f in fractions],
        "range_exact": [
            (max(fractions) - min(fractions)).numerator,
            (max(fractions) - min(fractions)).denominator,
        ],
        "every_fraction_at_least_quarter": enough,
        "range_below_five_points": stable,
    }
