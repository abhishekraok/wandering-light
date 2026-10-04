"""Sealed-cell structural reader and shown-lineage bridge; not a full verifier.

Only returned programs/suffixes are replayed here. Full attempted-op, guard,
ranking and seen reconstruction belongs to the AFTER-run independent verifier.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.score_gap_lineage import Retained, reconstruct_lineage, same_node
from experiments.score_gap_profiles import TERMINATIONS, Cell
from experiments.score_gap_records import ATTEMPT_STRUCT, decode_attempts
from experiments.score_gap_storage import (
    DiagnosticProvenanceError,
    canonical_json,
    decode_array,
    decode_beam,
    validate_scored,
)
from wandering_light.stack_beam import BeamResult
from wandering_light.stack_pbe import OPS

FILES = ("events.jsonl", "attempts.bin", "beams.bin", "scores.bin")


@dataclass
class TraceData:
    starts: list = field(default_factory=list)
    scored: dict = field(default_factory=dict)
    retained: list = field(default_factory=list)
    terminal: tuple | None = None


@dataclass(frozen=True)
class LoadedCell:
    cell: Cell
    result: BeamResult
    refusal_counts: dict
    seal_sha256: str


def _fail(message):
    raise DiagnosticProvenanceError(message)


def _int(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        _fail("invalid sealed integer")
    return value


def _json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        _fail("nonfinite JSON constant")

    try:
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
        if type(value) is not dict:
            _fail("sealed JSON record must be an object")
        if canonical_json(value) != payload:
            _fail("noncanonical sealed JSON")
        return value
    except (ValueError, UnicodeError) as error:
        raise DiagnosticProvenanceError("invalid sealed JSON") from error


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        _fail("sealed schema keys mismatch")


def _result(raw):
    _keys(
        raw,
        (
            "program",
            "termination",
            "seconds",
            "executions",
            "fuel",
            "expansions",
            "depth",
            "pruned",
        ),
    )
    for name in ("executions", "fuel", "expansions", "depth"):
        _int(raw[name])
    if (
        type(raw["seconds"]) is not float
        or not np.isfinite(raw["seconds"])
        or raw["seconds"] < 0
        or type(raw["pruned"]) is not bool
    ):
        _fail("invalid sealed result scalar")
    program = raw["program"]
    if program is not None and (
        type(program) is not list or any(type(op) is not str for op in program)
    ):
        _fail("invalid sealed program")
    if raw["termination"] not in TERMINATIONS or (program is not None) != (
        raw["termination"] == "fit"
    ):
        _fail("invalid sealed fit/terminal")
    if program is not None and (
        len(program) != raw["depth"]
        or any(name not in {op.name for op in OPS} for name in program)
    ):
        _fail("invalid sealed program depth/op")
    return BeamResult(**{**raw, "program": None if program is None else tuple(program)})


def load_cell(
    directory: Path, budget, expected_identity: dict, inputs, targets
) -> LoadedCell:
    """Hash all streams BEFORE decoding, consume their exact framed bytes.

    The caller owes pinned input selection and run-wide source/model provenance.
    Structural validation and known-positive replay do not verify other moves.
    """
    directory = Path(directory)
    budget.check()
    seal_bytes = (directory / "cell-seal.json").read_bytes()
    if len(seal_bytes) > 65_536:
        _fail("oversized cell seal")
    seal = _json(seal_bytes)
    _keys(seal, ("schema", "complete", "files", "attempts", "result"))
    if seal["schema"] != "SG-CELL1" or seal["complete"] is not True:
        _fail("unsuccessful/incomplete cell")
    _keys(seal["files"], FILES)
    _int(seal["attempts"], 12_000_000)
    result = _result(seal["result"])
    total = 0
    with ExitStack() as context:
        streams, hashes = {}, {}
        for name in FILES:
            record = seal["files"][name]
            _keys(record, ("bytes", "sha256"))
            total += _int(record["bytes"], budget.max_bytes)
            if total > budget.max_bytes:
                _fail("sealed payload exceeds global byte bound")
            stream = context.enter_context((directory / name).open("rb"))
            digest, size = hashlib.sha256(), 0
            while payload := stream.read(1024 * 1024):
                budget.check()
                size += len(payload)
                if size > record["bytes"]:
                    _fail("sealed file size mismatch")
                digest.update(payload)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                _fail("sealed file hash/size mismatch")
            stream.seek(0)
            streams[name], hashes[name] = stream, hashlib.sha256()
        if (
            seal["files"]["attempts.bin"]["bytes"]
            != seal["attempts"] * ATTEMPT_STRUCT.size
        ):
            _fail("attempt byte/count mismatch")

        def blob(name, reference):
            _keys(reference, ("offset", "bytes", "sha256"))
            stream = streams[name]
            if _int(reference["offset"]) != stream.tell():
                _fail("noncontiguous or reused payload frame")
            size = _int(reference["bytes"], seal["files"][name]["bytes"])
            if stream.tell() + size > seal["files"][name]["bytes"]:
                _fail("payload frame outside sealed file")
            budget.check()
            payload = stream.read(size)
            if (
                len(payload) != size
                or hashlib.sha256(payload).hexdigest() != reference["sha256"]
            ):
                _fail("payload frame hash/size mismatch")
            hashes[name].update(payload)
            return payload

        trace = TraceData()
        phase, depth, beam, retained_reference = "header", -1, (), None
        count = executions = fuel = successes = 0
        legal = None
        terminal_row = None

        def attempts(n):
            nonlocal count, executions, fuel, successes, legal
            n = _int(n, len(beam) * 28)
            payload = streams["attempts.bin"].read(n * 19)
            if len(payload) != n * 19:
                _fail("truncated framed attempts")
            hashes["attempts.bin"].update(payload)
            legal = np.zeros((len(beam), 28), dtype=bool)
            for i, attempt in enumerate(decode_attempts(payload)):
                budget.check()
                if (attempt.depth, attempt.node, attempt.op) != (
                    depth,
                    i // 28,
                    i % 28,
                ):
                    _fail("attempt stream not exact source prefix")
                examples = len(beam[attempt.node][0])
                if attempt.stepped > examples:
                    _fail("attempt charge exceeds shown examples")
                moved = attempt.stepped == examples
                legal[attempt.node, attempt.op] = moved
                successes += moved
                executions += attempt.stepped
                fuel += attempt.fuel
            count += n

        for line in streams["events.jsonl"]:
            budget.check()
            hashes["events.jsonl"].update(line)
            row = _json(line)
            kind = row.get("kind")
            if kind == "cell" and phase == "header":
                _keys(row, ("kind", "schema", "identity"))
                if row["schema"] != "SG-CELL1" or canonical_json(
                    row["identity"]
                ) != canonical_json(expected_identity):
                    _fail("cell identity mismatch")
                phase = "initial"
            elif kind == "started" and phase in ("initial", "retained"):
                _keys(
                    row,
                    ("kind", "depth", "state", "source_retention", "attempt_offset"),
                )
                if (
                    _int(row["depth"], 47) != depth + 1
                    or _int(row["attempt_offset"]) != streams["attempts.bin"].tell()
                ):
                    _fail("started callback depth/attempt offset mismatch")
                depth = row["depth"]
                if depth == 0:
                    if row["source_retention"] is not None:
                        _fail("root has source retention")
                    beam = decode_beam(blob("beams.bin", row["state"]))
                    if len(beam) != 1 or not same_node(beam[0], (inputs, (), 0.0)):
                        _fail("typed root differs from pinned shown input")
                elif _int(row["source_retention"]) != depth - 1 or canonical_json(
                    row["state"]
                ) != canonical_json(retained_reference):
                    _fail("start does not reference previous retained state")
                if not beam:
                    _fail("empty source beam")
                trace.starts.append((depth, beam))
                phase = "started"
            elif kind == "scored" and phase == "started":
                _keys(row, ("kind", "depth", "scores", "legal", "attempts"))
                if (
                    _int(row["depth"]) != depth
                    or _int(row["attempts"]) != len(beam) * 28
                ):
                    _fail("scored callback depth/attempt mismatch")
                attempts(row["attempts"])
                scores = decode_array(blob("scores.bin", row["scores"]))
                masks = decode_array(blob("scores.bin", row["legal"]))
                if not np.array_equal(masks, legal):
                    _fail("saved legal mask differs from attempts")
                validate_scored(beam, scores, masks)
                trace.scored[depth] = (scores, masks)
                phase = "scored"
            elif kind == "retained" and phase == "scored":
                _keys(row, ("kind", "depth", "state", "parents", "verdicts"))
                if _int(row["depth"]) != depth:
                    _fail("retained depth mismatch")
                children = decode_beam(blob("beams.bin", row["state"]))
                parents = tuple(tuple(pair) for pair in row["parents"])
                if len(parents) != len(children) or len(set(parents)) != len(parents):
                    _fail("retained parent count/identity mismatch")
                verdicts = {}
                for node, op, why in row["verdicts"]:
                    _int(node, len(beam) - 1)
                    _int(op, 27)
                    if (
                        (node, op) in verdicts
                        or not legal[node, op]
                        or why
                        not in (
                            "retained",
                            "duplicate",
                            "guard:close",
                            "guard:type",
                            "guard:size",
                        )
                    ):
                        _fail("invalid recorded verdict")
                    verdicts[node, op] = why
                if row["verdicts"] != [
                    [*pair, verdicts[pair]] for pair in sorted(verdicts)
                ] or set(parents) != {
                    key for key, why in verdicts.items() if why == "retained"
                }:
                    _fail("retained verdict/order mismatch")
                for child, pair in zip(children, parents, strict=True):
                    if len(pair) != 2:
                        _fail("invalid parent arity")
                    node, op = pair
                    _int(node, len(beam) - 1)
                    _int(op, 27)
                    parent = beam[node]
                    expected_child = (
                        child[0],
                        (*parent[1], OPS[op].name),
                        parent[2] + trace.scored[depth][0][node, op],
                    )
                    if len(child[0]) != len(parent[0]) or not same_node(
                        child, expected_child
                    ):
                        _fail("retained program/product-count/score mismatch")
                trace.retained.append(Retained(depth, children, parents))
                beam, retained_reference, phase = children, row["state"], "retained"
            elif kind == "terminal" and phase in ("started", "retained"):
                _keys(
                    row,
                    (
                        "kind",
                        "terminal",
                        "unscored_layer",
                        "partial_expansion",
                        "last_layer_attempts",
                        "total_attempts",
                        "refusal_counts",
                    ),
                )
                if row["unscored_layer"] is not (phase == "started"):
                    _fail("unscored terminal phase mismatch")
                n = _int(row["last_layer_attempts"], len(trace.starts[-1][1]) * 28)
                if row["partial_expansion"] is not (n < len(trace.starts[-1][1]) * 28):
                    _fail("partial expansion flag mismatch")
                if phase == "started":
                    attempts(n)
                elif n != len(trace.starts[-1][1]) * 28:
                    _fail("retained layer has incomplete attempts")
                terminal = tuple(row["terminal"])
                if len(terminal) != 5:
                    _fail("invalid terminal arity")
                for value in terminal[1:]:
                    _int(value)
                if phase == "retained":
                    if (
                        terminal[0] not in ("max_depth", "frontier_exhausted")
                        or terminal[1] != depth + 1
                    ):
                        _fail("post-retention stop mismatch")
                else:
                    expected_depth = (
                        depth if terminal[0] == "frontier_exhausted" else depth + 1
                    )
                    if (
                        terminal[0]
                        not in ("fit", "budget", "wall_clock_cap", "frontier_exhausted")
                        or terminal[1] != expected_depth
                    ):
                        _fail("partial/unscored stop mismatch")
                    if terminal[0] == "frontier_exhausted" and (
                        n != len(beam) * 28 or np.any(legal)
                    ):
                        _fail("unscored exhaustion has executable successors")
                expected = (
                    result.termination,
                    result.depth,
                    result.executions,
                    result.fuel,
                    result.expansions,
                )
                if (
                    terminal != expected
                    or (executions, fuel) != (result.executions, result.fuel)
                    or _int(row["total_attempts"]) != count
                ):
                    _fail("terminal result/charge mismatch")
                _keys(row["refusal_counts"], ("success", "ordinary", "fuel", "depth"))
                if (
                    sum(_int(n) for n in row["refusal_counts"].values()) != count
                    or row["refusal_counts"]["success"] != successes
                ):
                    _fail("refusal classification count mismatch")
                trace.terminal, terminal_row, phase = terminal, row, "terminal"
            elif kind == "result" and phase == "terminal":
                _keys(row, ("kind", "result"))
                if canonical_json(row["result"]) != canonical_json(seal["result"]):
                    _fail("result event differs from seal")
                phase = "complete"
            else:
                _fail("unexpected or missing callback order")
        if phase != "complete" or count != seal["attempts"]:
            _fail("incomplete sealed callback stream")
        for name in FILES:
            if (
                streams[name].tell() != seal["files"][name]["bytes"]
                or hashes[name].hexdigest() != seal["files"][name]["sha256"]
            ):
                _fail("unconsumed or changed sealed stream")
        try:
            layers = reconstruct_lineage(
                trace, inputs, targets, result, check=budget.check
            )
        except (ValueError, KeyError, IndexError) as error:
            raise DiagnosticProvenanceError(
                f"invalid returned lineage: {error}"
            ) from error
        cell = Cell(
            expected_identity["group"],
            expected_identity["half"],
            expected_identity["seed"],
            result.termination,
            layers,
        )
        return LoadedCell(
            cell,
            result,
            terminal_row["refusal_counts"],
            hashlib.sha256(seal_bytes).hexdigest(),
        )
