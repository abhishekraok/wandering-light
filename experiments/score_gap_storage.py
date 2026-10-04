"""Synthetic-testable #180 storage primitives; no trial runner or observer.

Typed beams are encoded once at the root and each retention. Later start events
must reference the prior retention. Receipts are small control records, separate
from the shared trace-payload ledger; no reservation is made on import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.score_gap_records import MAX_ATTEMPTS
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym
from wandering_light.stack_pbe import OPS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import BinaryIO

    from wandering_light.stack_beam import Node

U64 = struct.Struct("<Q")
OP_INDEX = {op.name: index for index, op in enumerate(OPS)}
RECEIPT_CAP = 65_536


class DiagnosticResourceError(RuntimeError):
    """Global diagnostic exhaustion; never a production search terminal."""


class DiagnosticProvenanceError(ValueError):
    """Invalid/incomplete evidence invalidates the run, not just one label."""


@dataclass
class PayloadBudget:
    deadline: float
    max_bytes: int = 2 * 1024**3
    max_attempts: int = MAX_ATTEMPTS
    clock: Callable[[], float] = time.perf_counter
    bytes_written: int = 0
    attempts_written: int = 0

    def __post_init__(self):
        if not math.isfinite(self.deadline):
            raise ValueError("invalid global deadline")
        if any(
            type(n) is not int or n < 0 for n in (self.max_bytes, self.max_attempts)
        ):
            raise ValueError("invalid global budget")
        if self.bytes_written or self.attempts_written:
            raise ValueError("budget must start empty")

    def check(self, size: int = 0, attempts: int = 0):
        if any(type(n) is not int or n < 0 for n in (size, attempts)):
            raise ValueError("invalid record charge")
        now = self.clock()
        if not math.isfinite(now):
            raise DiagnosticProvenanceError("invalid global clock")
        if now >= self.deadline:
            raise DiagnosticResourceError("global deadline")
        if self.bytes_written + size > self.max_bytes:
            raise DiagnosticResourceError("global trace bytes")
        if self.attempts_written + attempts > self.max_attempts:
            raise DiagnosticResourceError("global attempts")

    def write(self, stream: BinaryIO, payload: bytes, *, attempts: int = 0):
        if not isinstance(payload, bytes):
            raise ValueError("record must be bytes")
        self.check(len(payload), attempts)
        written = stream.write(payload)
        if written != len(payload):
            raise OSError("short trace write; no complete result")
        self.bytes_written += written
        self.attempts_written += attempts
        self.check()


def _count(n: int) -> bytes:
    if type(n) is not int or not 0 <= n < 1 << 64:
        raise DiagnosticProvenanceError("invalid typed count")
    return U64.pack(n)


def _values(values: Sequence) -> Iterator[bytes]:
    """Iterative traversal avoids duplicating nested typed state structures."""
    pending = [iter(values)]
    exhausted = object()
    while pending:
        value = next(pending[-1], exhausted)
        if value is exhausted:
            pending.pop()
        elif type(value) is Int:
            magnitude = abs(value.value)
            size = (magnitude.bit_length() + 7) // 8
            yield b"i" + bytes([value.value < 0]) + _count(size)
            yield magnitude.to_bytes(size, "little")
        elif type(value) is Bool:
            yield b"b" + bytes([value.value])
        elif type(value) is Sym:
            text = value.name.encode("utf-8", errors="surrogatepass")
            yield b"s" + _count(len(text))
            yield text
        elif type(value) is Mark:
            yield b"m"
        elif type(value) is Lst:
            if type(value.items) is not tuple:
                raise DiagnosticProvenanceError("list items must be tuple")
            yield b"l" + _count(len(value.items))
            pending.append(iter(value.items))
        else:
            raise DiagnosticProvenanceError("unknown typed value")


def beam_chunks(beam: Sequence[Node]) -> Iterator[bytes]:
    """V1: count, products/stacks/typed values, op-index program, scalar bits."""
    if not 0 <= len(beam) <= 64:
        raise DiagnosticProvenanceError("invalid beam size")
    yield b"SG-BEAM1" + bytes([len(beam)])
    for product, program, score in beam:
        if not product or len(program) > 48:
            raise DiagnosticProvenanceError("invalid product/program")
        yield _count(len(product))
        for stack in product:
            yield _count(len(stack))
            yield from _values(stack)
        try:
            code = bytes(OP_INDEX[name] for name in program)
        except KeyError as error:
            raise DiagnosticProvenanceError("unknown program op") from error
        yield bytes([len(code)]) + code
        if type(score) is float:
            kind, scalar = 0, np.asarray(score, dtype="<f8")
        elif type(score) is np.float32:
            kind, scalar = 1, np.asarray(score, dtype="<f4")
        elif type(score) is np.float64:
            kind, scalar = 2, np.asarray(score, dtype="<f8")
        else:
            raise DiagnosticProvenanceError("invalid score scalar type")
        if np.isnan(scalar) or np.isposinf(scalar):
            raise DiagnosticProvenanceError("invalid cumulative score")
        yield bytes([kind]) + scalar.tobytes()


class _Reader:
    def __init__(self, payload: bytes):
        self.payload = memoryview(payload)
        self.position = 0

    def take(self, n: int) -> bytes:
        end = self.position + n
        if end > len(self.payload):
            raise DiagnosticProvenanceError("truncated typed beam")
        result = bytes(self.payload[self.position : end])
        self.position = end
        return result

    def count(self, *, minimum_bytes: int = 1) -> int:
        n = U64.unpack(self.take(8))[0]
        if n > (len(self.payload) - self.position) // minimum_bytes:
            raise DiagnosticProvenanceError("impossible typed count")
        return n

    def value(self):
        frames = []
        while True:
            tag = self.take(1)
            if tag == b"l":
                n = self.count()
                if n:
                    frames.append((n, []))
                    continue
                value = Lst()
            elif tag == b"i":
                negative = self.take(1)[0]
                raw = self.take(self.count())
                if negative > 1 or (raw and raw[-1] == 0) or (negative and not raw):
                    raise DiagnosticProvenanceError("noncanonical integer")
                n = int.from_bytes(raw, "little")
                value = Int(-n if negative else n)
            elif tag == b"b":
                flag = self.take(1)[0]
                if flag > 1:
                    raise DiagnosticProvenanceError("invalid boolean")
                value = Bool(bool(flag))
            elif tag == b"s":
                value = Sym(
                    self.take(self.count()).decode("utf-8", errors="surrogatepass")
                )
            elif tag == b"m":
                value = Mark()
            else:
                raise DiagnosticProvenanceError("unknown typed tag")
            while frames:
                n, items = frames[-1]
                items.append(value)
                if len(items) < n:
                    break
                frames.pop()
                value = Lst(tuple(items))
            else:
                return value


def decode_beam(payload: bytes) -> tuple[Node, ...]:
    """Companion decoder only; an independent verifier must not reuse it."""
    reader = _Reader(payload)
    if reader.take(8) != b"SG-BEAM1":
        raise DiagnosticProvenanceError("wrong typed beam version")
    count = reader.take(1)[0]
    if count > 64:
        raise DiagnosticProvenanceError("invalid beam size")
    beam = []
    for _ in range(count):
        examples = reader.count(minimum_bytes=8)
        if not examples:
            raise DiagnosticProvenanceError("empty product")
        product = []
        for _ in range(examples):
            product.append(tuple(reader.value() for _ in range(reader.count())))
        length = reader.take(1)[0]
        indices = reader.take(length)
        if length > 48 or any(index >= len(OPS) for index in indices):
            raise DiagnosticProvenanceError("invalid program indices")
        program = tuple(OPS[index].name for index in indices)
        kind = reader.take(1)[0]
        if kind not in (0, 1, 2):
            raise DiagnosticProvenanceError("invalid score kind")
        scalar = np.frombuffer(
            reader.take(4 if kind == 1 else 8), dtype="<f4" if kind == 1 else "<f8"
        )[0]
        if np.isnan(scalar) or np.isposinf(scalar):
            raise DiagnosticProvenanceError("invalid cumulative score")
        beam.append((tuple(product), program, float(scalar) if kind == 0 else scalar))
    if reader.position != len(payload):
        raise DiagnosticProvenanceError("trailing typed beam bytes")
    return tuple(beam)


def write_chunks(stream: BinaryIO, chunks: Iterable[bytes], budget: PayloadBudget):
    """Return exact offset/size/hash only after every chunk completes."""
    offset, size, digest = stream.tell(), 0, hashlib.sha256()
    for chunk in chunks:
        budget.write(stream, chunk)
        size += len(chunk)
        digest.update(chunk)
    return {"offset": offset, "bytes": size, "sha256": digest.hexdigest()}


def array_chunks(array: np.ndarray) -> Iterator[bytes]:
    """V1 exact C-order float32/float64/bool arrays, including illegal NaNs.

    This codec does not establish score validity. The observer must validate
    cumulative scores for all executable non-fit successors separately.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise DiagnosticProvenanceError("invalid scored array")
    kinds = {np.dtype("<f4"): 0, np.dtype("<f8"): 1, np.dtype("bool"): 2}
    if array.dtype not in kinds:
        raise DiagnosticProvenanceError("unsupported scored dtype")
    if not 1 <= array.shape[0] <= 64 or array.shape[1] != 28:
        raise DiagnosticProvenanceError("invalid scored shape")
    yield b"SG-ARRAY1" + bytes([kinds[array.dtype], array.shape[0], 28])
    yield array.tobytes(order="C")


def decode_array(payload: bytes) -> np.ndarray:
    """Companion decoder, not independent verification."""
    reader = _Reader(payload)
    if reader.take(9) != b"SG-ARRAY1":
        raise DiagnosticProvenanceError("wrong array version")
    kind, rows, columns = reader.take(3)
    if kind not in (0, 1, 2) or not 1 <= rows <= 64 or columns != 28:
        raise DiagnosticProvenanceError("invalid scored header")
    dtype = np.dtype(("<f4", "<f8", "bool")[kind])
    raw = reader.take(rows * columns * dtype.itemsize)
    if reader.position != len(payload):
        raise DiagnosticProvenanceError("trailing scored bytes")
    if kind == 2 and any(value > 1 for value in raw):
        raise DiagnosticProvenanceError("noncanonical legal mask")
    return np.frombuffer(raw, dtype=dtype).reshape(rows, columns).copy()


def validate_scored(beam: Sequence[Node], scores: np.ndarray, legal: np.ndarray):
    """Run-wide provenance failure for any executable non-fit invalid score.

    Scoring happens only after all expanded successors failed shown fit. Check
    before retention, including later guarded/duplicate/unread successors.
    Ignore illegal matrix positions; do not change production accumulation.
    """
    expected = (len(beam), len(OPS))
    if scores.shape != expected or legal.shape != expected or legal.dtype != np.bool_:
        raise DiagnosticProvenanceError("scored shape/legal dtype mismatch")
    if scores.dtype not in (np.dtype("<f4"), np.dtype("<f8")):
        raise DiagnosticProvenanceError("invalid scored dtype")
    for parent, (_, _, cumulative) in enumerate(beam):
        for op in np.flatnonzero(legal[parent]):
            value = cumulative + scores[parent, op]
            if np.isnan(value) or np.isposinf(value):
                raise DiagnosticProvenanceError("invalid executable cumulative score")


def refusal_class(reason: str | None) -> str:
    """Exact reason suffix, including all nested op prefixes from step_raw."""
    if reason is None:
        return "success"
    if not isinstance(reason, str):
        raise DiagnosticProvenanceError("invalid replayed refusal")
    if reason == "out of fuel" or reason.endswith(": out of fuel"):
        return "fuel"
    if reason == "quotation nested too deep" or reason.endswith(
        ": quotation nested too deep"
    ):
        return "depth"
    return "ordinary"


def canonical_json(value) -> bytes:
    pending = [value]
    containers = set()
    while pending:
        item = pending.pop()
        if type(item) in (dict, list, tuple):
            if id(item) in containers:
                continue
            containers.add(id(item))
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise DiagnosticProvenanceError("JSON keys must be strings")
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise DiagnosticProvenanceError("unsupported JSON value")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish(root: Path, name: str, value):
    """Atomic no-replace publish, file plus directory fsync; never overwrite."""
    payload = canonical_json(value)
    if len(payload) > RECEIPT_CAP:
        raise DiagnosticProvenanceError("control receipt too large")
    temporary = root / (".receipt-" + uuid.uuid4().hex)
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            if stream.write(payload) != len(payload):
                raise OSError("short receipt write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, root / name)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class AttemptReceipts:
    """One unique attempt in an existing root, with one immutable terminal.

    The caller must reserve before reading any real input. A terminal.json
    tagged final/failure is one atomic mutually exclusive final/failure record.
    It is not itself a complete result manifest or authorization to read data.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.reserved = False
        self.selected = False
        self.finished = False

    def reserve(self, metadata):
        if self.reserved:
            raise DiagnosticProvenanceError("already reserved")
        self.root.mkdir(parents=True, exist_ok=True)
        _publish(self.root, "attempt.json", {"kind": "attempt", "metadata": metadata})
        self.reserved = True

    def selection(self, metadata):
        self._active()
        if self.selected:
            raise DiagnosticProvenanceError("selection already locked")
        _publish(
            self.root,
            "selection-lock.json",
            {"kind": "selection", "metadata": metadata},
        )
        self.selected = True

    def terminal(self, kind: str, metadata):
        self._active()
        if kind not in ("final", "failure"):
            raise DiagnosticProvenanceError("invalid terminal kind")
        if kind == "final" and not self.selected:
            raise DiagnosticProvenanceError("final without selection lock")
        _publish(self.root, "terminal.json", {"kind": kind, "metadata": metadata})
        self.finished = True

    def _active(self):
        if not self.reserved or self.finished:
            raise DiagnosticProvenanceError("attempt not active")
