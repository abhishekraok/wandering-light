"""Only manually typed synthetic storage/receipt/resource tests; no data attempt."""

import hashlib
import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.score_gap_lineage import SyntheticCapture, same_node
from experiments.score_gap_storage import (
    AttemptReceipts,
    DiagnosticProvenanceError,
    DiagnosticResourceError,
    PayloadBudget,
    array_chunks,
    beam_chunks,
    canonical_json,
    decode_array,
    decode_beam,
    refusal_class,
    validate_scored,
    write_chunks,
)
from tests.test_score_gap_lineage import ScriptedScorer, task
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    step_raw,
)
from wandering_light.stack_pbe import OP_BY_NAME


@pytest.mark.parametrize(
    "score", [0.0, -0.0, np.float32(-0.0), np.float64(-0.0), -np.inf]
)
def test_typed_values_programs_scalar_class_dtype_and_bits(score):
    product = (
        (Int(1), Bool(True), Mark(), Sym("a\ud800π"), Lst((Int(-1), Lst()))),
        (Int(0), Bool(False), Int(1 << 20_000)),
        (),
    )
    original = ((product, ("dup", "[", "mul"), score),)
    payload = b"".join(beam_chunks(original))
    restored = decode_beam(payload)
    assert type(restored[0][2]) is type(score)
    assert same_node(restored[0], original[0])
    assert type(restored[0][0][0][0]) is Int
    assert type(restored[0][0][0][1]) is Bool
    assert b"".join(beam_chunks(restored)) == payload


def test_empty_beam_and_deep_nested_lists_without_recursive_codec():
    assert decode_beam(b"".join(beam_chunks(()))) == ()
    value = Int(7)
    for _ in range(2000):
        value = Lst((value,))
    payload = b"".join(beam_chunks(((((value,),), (), 0.0),)))
    assert b"".join(beam_chunks(decode_beam(payload))) == payload


@pytest.mark.parametrize("score", [True, 1, np.float32(np.nan), np.inf])
def test_malformed_score_rejected(score):
    with pytest.raises(DiagnosticProvenanceError, match="score"):
        b"".join(beam_chunks(((((Int(1),),), (), score),)))


@pytest.mark.parametrize("value", [None, 1, "x", Lst([Int(1)])])
def test_untyped_values_and_non_tuple_list_items_rejected(value):
    with pytest.raises(DiagnosticProvenanceError):
        b"".join(beam_chunks(((((value,),), (), 0.0),)))


def test_every_truncated_prefix_and_trailing_bytes_rejected():
    payload = b"".join(beam_chunks(((((Int(-257),),), ("dup",), np.float32(-0.0)),)))
    for length in range(len(payload)):
        with pytest.raises(DiagnosticProvenanceError):
            decode_beam(payload[:length])
    with pytest.raises(DiagnosticProvenanceError, match="trailing"):
        decode_beam(payload + b"x")


@pytest.mark.parametrize(
    "encoded",
    [
        b"i\x01" + struct.pack("<Q", 0),  # negative zero
        b"i\x00" + struct.pack("<Q", 1) + b"\x00",  # redundant zero byte
        b"b\x02",
        b"z",
        b"l" + struct.pack("<Q", 1 << 63),
    ],
)
def test_invalid_typed_encodings_rejected(encoded):
    header = b"SG-BEAM1\x01" + struct.pack("<QQ", 1, 1)
    tail = b"\x00\x00" + struct.pack("<d", 0.0)
    with pytest.raises(DiagnosticProvenanceError):
        decode_beam(header + encoded + tail)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_])
def test_array_full_bits_and_illegal_nan_retained(dtype):
    source = np.zeros((2, 28), dtype=dtype)
    if dtype is np.bool_:
        source[0, 3] = True
    else:
        source[0, :4] = [-0.0, -np.inf, np.nan, -1.25]
        # Preserve a non-default NaN payload, rather than merely isNaN equality.
        source.view(np.uint32 if dtype is np.float32 else np.uint64)[0, 2] = (
            0x7FC00009 if dtype is np.float32 else 0x7FF8000000000009
        )
    payload = b"".join(array_chunks(source))
    restored = decode_array(payload)
    assert restored.dtype == source.dtype
    assert restored.shape == source.shape
    assert restored.tobytes() == source.tobytes()
    for length in (0, 9, len(payload) - 1):
        with pytest.raises(DiagnosticProvenanceError):
            decode_array(payload[:length])
    with pytest.raises(DiagnosticProvenanceError, match="trailing"):
        decode_array(payload + b"x")


def test_noncanonical_legal_byte_and_invalid_array_shape_dtype():
    payload = b"SG-ARRAY1\x02\x01\x1c" + b"\x02" + bytes(27)
    with pytest.raises(DiagnosticProvenanceError, match="legal mask"):
        decode_array(payload)
    for source in (
        np.zeros((65, 28)),
        np.zeros((2, 27)),
        np.zeros((2, 28), dtype=np.int32),
    ):
        with pytest.raises(DiagnosticProvenanceError):
            b"".join(array_chunks(source))


def budget(**kwargs):
    return PayloadBudget(deadline=10.0, clock=lambda: 0.0, **kwargs)


def test_one_global_ledger_across_files_and_prewrite_caps():
    ledger = budget(max_bytes=4, max_attempts=2)
    a, b = io.BytesIO(), io.BytesIO()
    ledger.write(a, b"ab", attempts=1)
    ledger.write(b, b"cd", attempts=1)
    assert (ledger.bytes_written, ledger.attempts_written) == (4, 2)
    with pytest.raises(DiagnosticResourceError, match="bytes"):
        ledger.write(a, b"e")
    with pytest.raises(DiagnosticResourceError, match="attempts"):
        ledger.write(a, b"", attempts=1)
    assert (a.getvalue(), b.getvalue()) == (b"ab", b"cd")


def test_chunk_offsets_hashes_and_partial_write_never_seal():
    stream = io.BytesIO(b"head")
    stream.seek(4)
    record = write_chunks(stream, [b"ab", b"c"], budget())
    assert record == {
        "offset": 4,
        "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    limited = budget(max_bytes=2)
    partial = io.BytesIO()
    with pytest.raises(DiagnosticResourceError, match="bytes"):
        write_chunks(partial, [b"ab", b"c"], limited)
    assert partial.getvalue() == b"ab"


def test_global_deadline_distinct_from_production_wall_clock_cap_and_gc_restored():
    class LimitedCapture(SyntheticCapture):
        def __init__(self):
            super().__init__()
            self.ledger = PayloadBudget(deadline=0.0, clock=lambda: 0.0)

        def expansion_attempted(self, *args):
            self.ledger.write(io.BytesIO(), b"record", attempts=1)

    capture = LimitedCapture()
    import gc

    collecting = gc.isenabled()
    with pytest.raises(DiagnosticResourceError, match="global deadline"):
        beam_solve(task(), ScriptedScorer(), observer=capture)
    assert capture.terminal is None
    assert gc.isenabled() == collecting


def test_postwrite_deadline_is_no_complete_result():
    times = iter((0.0, 10.0))
    ledger = PayloadBudget(deadline=10.0, clock=lambda: next(times))
    stream = io.BytesIO()
    with pytest.raises(DiagnosticResourceError, match="deadline"):
        ledger.write(stream, b"partial")
    assert ledger.bytes_written == 7


def test_existing_review_root_and_atomic_immutable_receipts(tmp_path):
    (tmp_path / "design-review").mkdir()
    receipt = AttemptReceipts(tmp_path)
    receipt.reserve({"synthetic": True})
    original = (tmp_path / "attempt.json").read_bytes()
    with pytest.raises(FileExistsError):
        AttemptReceipts(tmp_path).reserve({"retry": True})
    assert (tmp_path / "attempt.json").read_bytes() == original
    with pytest.raises(DiagnosticProvenanceError, match="without selection"):
        receipt.terminal("final", {})
    receipt.selection({"tau": None, "reason": "synthetic-support"})
    locked = (tmp_path / "selection-lock.json").read_bytes()
    with pytest.raises(DiagnosticProvenanceError, match="already locked"):
        receipt.selection({"tau": 32})
    assert (tmp_path / "selection-lock.json").read_bytes() == locked
    receipt.terminal("final", {"complete": True})
    with pytest.raises(DiagnosticProvenanceError, match="not active"):
        receipt.terminal("failure", {})
    assert json.loads((tmp_path / "terminal.json").read_text())["kind"] == "final"
    assert not list(tmp_path.glob(".receipt-*"))


def test_racing_attempt_reservations_allow_exactly_one(tmp_path):
    def reserve(_):
        try:
            AttemptReceipts(tmp_path).reserve({"synthetic": True})
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(reserve, range(4)))
    assert sum(outcomes) == 1
    assert json.loads((tmp_path / "attempt.json").read_text())["kind"] == "attempt"
    assert not list(tmp_path.glob(".receipt-*"))


def test_failure_terminal_allowed_without_selection(tmp_path):
    receipt = AttemptReceipts(tmp_path)
    receipt.reserve({})
    receipt.terminal("failure", {"reason": "synthetic-global-bytes"})
    assert not (tmp_path / "selection-lock.json").exists()
    assert json.loads((tmp_path / "terminal.json").read_text())["kind"] == "failure"


@pytest.mark.parametrize(
    "metadata",
    [{1: "ambiguous-key"}, {"x": np.float32(0)}, {"x": np.nan}, {"x": "x" * 65_536}],
)
def test_receipt_invalidity_leaves_no_published_or_temporary_file(tmp_path, metadata):
    with pytest.raises((ValueError, DiagnosticProvenanceError)):
        AttemptReceipts(tmp_path).reserve(metadata)
    assert not list(tmp_path.iterdir())


def test_canonical_json_stable_and_nonfinite_rejected():
    assert canonical_json({"z": [1, True], "a": None}) == b'{"a":null,"z":[1,true]}\n'
    with pytest.raises(ValueError):
        canonical_json({"bad": np.inf})


def test_cyclic_metadata_rejected_and_shared_metadata_permitted():
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="Circular reference"):
        canonical_json(cycle)
    shared = {"x": [1]}
    assert json.loads(canonical_json([shared, shared])) == [shared, shared]


def test_all_executable_scores_checked_including_unread_tail_illegal_ignored():
    beam = ((((),), (), np.float32(-0.0)),)
    scores = np.full((1, 28), np.nan, dtype=np.float32)
    legal = np.zeros((1, 28), dtype=bool)
    scores[0, 27], legal[0, 27] = -np.inf, True
    original = scores.tobytes()
    validate_scored(beam, scores, legal)
    assert scores.tobytes() == original
    for bad in (np.nan, np.inf):
        scores[0, 27] = bad
        with pytest.raises(DiagnosticProvenanceError, match="executable cumulative"):
            validate_scored(beam, scores, legal)
    # Valid log-probabilities plus an invalid source accumulation are invalid.
    scores[0, 27] = 0.0
    with pytest.raises(DiagnosticProvenanceError):
        validate_scored(((((),), (), np.nan),), scores, legal)


def test_full_default_fuel_and_depth_refusals_classified_after_nested_prefixes():
    body = Lst((Sym("swap"),) * DEFAULT_FUEL_LIMIT)
    outcome = step_raw(OP_BY_NAME["i"], (Int(1), Int(2), body), DEFAULT_FUEL_LIMIT, 0)
    assert isinstance(outcome, str)
    assert refusal_class(outcome) == "fuel"
    assert refusal_class("example 1: i: " + outcome) == "fuel"
    nested = Lst()
    for _ in range(40):
        nested = Lst((nested, Sym("i")))
    outcome = step_raw(OP_BY_NAME["i"], (nested,), DEFAULT_FUEL_LIMIT, 0)
    assert isinstance(outcome, str)
    assert refusal_class(outcome) == "depth"
    assert refusal_class("example 0: " + outcome) == "depth"
    unknown = step_raw(
        OP_BY_NAME["i"], (Lst((Sym("out of fuel"),)),), DEFAULT_FUEL_LIMIT, 0
    )
    assert isinstance(unknown, str)
    assert refusal_class(unknown) == "ordinary"
    assert refusal_class(None) == "success"


def test_short_write_and_invalid_clock_are_never_completed():
    class ShortStream(io.BytesIO):
        def write(self, payload):
            return super().write(payload[:1])

    ledger = budget()
    with pytest.raises(OSError, match="short trace write"):
        ledger.write(ShortStream(), b"more")
    assert ledger.bytes_written == 0
    broken_clock = PayloadBudget(deadline=10.0, clock=lambda: np.nan)
    with pytest.raises(DiagnosticProvenanceError, match="global clock"):
        broken_clock.check()


def test_failed_atomic_link_cleans_temp_and_never_publishes(tmp_path, monkeypatch):
    def fail_link(*args):
        raise OSError("synthetic link failure")

    monkeypatch.setattr("experiments.score_gap_storage.os.link", fail_link)
    with pytest.raises(OSError, match="synthetic link failure"):
        AttemptReceipts(tmp_path).reserve({})
    assert not list(tmp_path.iterdir())


def test_temp_name_collision_preserves_file_not_created_by_receipt(
    tmp_path, monkeypatch
):
    unrelated = tmp_path / ".receipt-collision"
    unrelated.write_bytes(b"preserve")
    monkeypatch.setattr(
        "experiments.score_gap_storage.uuid.uuid4",
        lambda: SimpleNamespace(hex="collision"),
    )
    with pytest.raises(FileExistsError):
        AttemptReceipts(tmp_path).reserve({})
    assert unrelated.read_bytes() == b"preserve"
    assert not (tmp_path / "attempt.json").exists()
