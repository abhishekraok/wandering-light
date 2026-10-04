"""Synthetic compact records, including real production observer callbacks."""

import struct

import pytest

from experiments.score_gap_lineage import SyntheticCapture
from experiments.score_gap_records import (
    ATTEMPT_STRUCT,
    MAX_ATTEMPTS,
    WORST_ATTEMPT_BYTES,
    WORST_ATTEMPTS,
    Attempt,
    decode_attempts,
)
from tests.test_score_gap_lineage import ScriptedScorer, task
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, step_raw
from wandering_light.stack_pbe import OPS


def test_literal_layout_and_bound():
    row = Attempt(47, 63, 27, 0x0102030405060708, 2)
    assert row.encode() == bytes.fromhex("2f3f1b08070605040302010200000000000000")
    assert ATTEMPT_STRUCT.size == 19
    assert WORST_ATTEMPTS == 11_987_136 < MAX_ATTEMPTS
    assert WORST_ATTEMPT_BYTES == 227_755_584
    assert list(decode_attempts(row.encode())) == [row]


@pytest.mark.parametrize("field", ["depth", "node", "op", "fuel", "stepped"])
@pytest.mark.parametrize("value", [-1, True, 1.0, None, 1 << 64])
def test_out_of_range_and_noninteger_never_wrap(field, value):
    fields = {"depth": 0, "node": 0, "op": 0, "fuel": 0, "stepped": 0}
    fields[field] = value
    with pytest.raises(ValueError, match=f"invalid attempt {field}"):
        Attempt(**fields)


@pytest.mark.parametrize("bad", [b"\x00", bytes(18), bytes(20)])
def test_truncated_payload_rejected(bad):
    with pytest.raises(ValueError, match="truncated"):
        list(decode_attempts(bad))


def test_invalid_encoded_op_rejected():
    with pytest.raises(ValueError, match="invalid attempt op"):
        list(decode_attempts(struct.pack("<BBBQQ", 0, 0, 255, 0, 0)))


class AttemptCapture(SyntheticCapture):
    def __init__(self):
        super().__init__()
        self.payload = bytearray()
        self.callbacks = []

    def expansion_attempted(self, depth, node, op, moved, reason, fuel, stepped):
        self.payload.extend(Attempt(depth, node, op, fuel, stepped).encode())
        self.callbacks.append((moved, reason))


def test_every_record_replays_exact_product_reason_and_partial_charge():
    capture = AttemptCapture()
    # mul succeeds on example0 but refuses on example1; production charges only
    # example0. A later scored frontier and the unexpanded last layer also occur.
    manual = task(inputs=("2 3", "true"), outputs=("999", "999"))
    result = beam_solve(
        manual,
        ScriptedScorer(),
        beam_width=8,
        max_depth=2,
        guard="off",
        observer=capture,
    )
    rows = list(decode_attempts(bytes(capture.payload)))
    assert len(rows) == len(capture.callbacks)
    assert [(r.depth, r.node, r.op) for r in rows] == [
        (depth, node, op)
        for depth, beam in capture.starts
        for node in range(len(beam))
        for op in range(len(OPS))
    ]
    partial_refusal = False
    for row, (saved_moved, saved_reason) in zip(rows, capture.callbacks, strict=True):
        parent = capture.starts[row.depth][1][row.node][0]
        moved, fuel, reason = [], 0, None
        for stepped, stack in enumerate(parent):
            outcome = step_raw(OPS[row.op], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                reason = f"example {stepped}: {outcome}"
                break
            moved.append(outcome[0])
            fuel += outcome[1]
        else:
            stepped = len(parent)
        assert (row.fuel, row.stepped) == (fuel, stepped)
        assert saved_reason == reason
        assert saved_moved == (tuple(moved) if reason is None else None)
        partial_refusal |= reason is not None and stepped == 1
    assert partial_refusal
    assert sum(row.fuel for row in rows) == result.fuel
    assert sum(row.stepped for row in rows) == result.executions
    assert result.expansions == sum(len(beam) for _, beam in capture.starts)
