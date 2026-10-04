"""Bounded per-cell production observer; no task/model loader or data runner.

All invocation in tests is synthetic. Actual use still requires complete-code
review, prior attempt reservation, frozen inputs and independent verification.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.score_gap_lineage import same_node
from experiments.score_gap_profiles import TERMINATIONS
from experiments.score_gap_records import Attempt
from experiments.score_gap_storage import (
    DiagnosticProvenanceError,
    _publish,
    array_chunks,
    beam_chunks,
    canonical_json,
    validate_scored,
    write_chunks,
)
from wandering_light.stack_beam import BeamObserver
from wandering_light.stack_pbe import OPS


class CellRecorder(BeamObserver):
    """Four bounded streams plus one final cell seal, no reason dictionary.

    events.jsonl frames cell/layer callbacks and points into beams.bin and
    scores.bin; attempts.bin is the fixed19-byte ordered stream. A root or
    retained beam is stored once, including the final unexpanded retention.
    The next start references that same state record, rather than copying it.
    A resource exception leaves unsealed partial files and no search terminal.
    """

    def __init__(self, directory: Path, budget, identity):
        self.directory = Path(directory)
        self.budget = budget
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = {}
        self.phase = "initial"
        self.depth = -1
        self.beam = ()
        self.state = None
        self.retained_beam = ()
        self.terminal = None
        self.closed = False
        self.finished = False
        self.attempts = self.layer_attempts = self.executions = self.fuel = 0
        try:
            for name in ("events.jsonl", "attempts.bin", "beams.bin", "scores.bin"):
                self.files[name] = (self.directory / name).open("xb")
            self._event({"kind": "cell", "schema": "SG-CELL1", "identity": identity})
        except BaseException:
            self.close()
            raise

    def _event(self, value):
        self.budget.write(self.files["events.jsonl"], canonical_json(value))

    def _expect(self, *phases):
        self.budget.check()
        if self.closed or self.phase not in phases:
            raise DiagnosticProvenanceError("invalid callback phase/order")

    def layer_started(self, depth, beam):
        self._expect("initial", "retained")
        if type(depth) is not int or depth != self.depth + 1 or depth > 47:
            raise DiagnosticProvenanceError("invalid started depth")
        if not 1 <= len(beam) <= 64:
            raise DiagnosticProvenanceError("invalid source beam")
        if depth == 0:
            if (
                len(beam) != 1
                or beam[0][1] != ()
                or type(beam[0][2]) is not float
                or beam[0][2] != 0.0
            ):
                raise DiagnosticProvenanceError("invalid root beam")
            self.state = write_chunks(
                self.files["beams.bin"], beam_chunks(beam), self.budget
            )
        elif len(beam) != len(self.retained_beam) or any(
            not same_node(a, b) for a, b in zip(beam, self.retained_beam, strict=True)
        ):
            raise DiagnosticProvenanceError("next start differs from retained beam")
        self.depth, self.beam = depth, tuple(beam)
        self.phase, self.layer_attempts = "started", 0
        self.legal = np.zeros((len(beam), len(OPS)), dtype=bool)
        self._event(
            {
                "kind": "started",
                "depth": depth,
                "state": self.state,
                "source_retention": depth - 1 if depth else None,
                "attempt_offset": self.files["attempts.bin"].tell(),
            }
        )

    def expansion_attempted(
        self, depth, node, op_index, successor, reason, fuel, stepped
    ):
        self._expect("started")
        if (depth, node, op_index) != (
            self.depth,
            self.layer_attempts // 28,
            self.layer_attempts % 28,
        ):
            raise DiagnosticProvenanceError(
                "attempts are not exact lexicographic prefix"
            )
        if not 0 <= node < len(self.beam):
            raise DiagnosticProvenanceError("attempt outside source beam")
        attempt = Attempt(depth, node, op_index, fuel, stepped)
        examples = len(self.beam[node][0])
        if successor is None:
            if not isinstance(reason, str) or not 0 <= stepped < examples:
                raise DiagnosticProvenanceError("invalid refusal callback")
        elif reason is not None or stepped != examples or len(successor) != examples:
            raise DiagnosticProvenanceError("invalid successful callback")
        self.budget.write(self.files["attempts.bin"], attempt.encode(), attempts=1)
        self.legal[node, op_index] = successor is not None
        self.layer_attempts += 1
        self.attempts += 1
        self.executions += stepped
        self.fuel += fuel

    def layer_scored(self, depth, log_probs, legal):
        self._expect("started")
        if depth != self.depth or self.layer_attempts != len(self.beam) * 28:
            raise DiagnosticProvenanceError("scoring incomplete expansion layer")
        if not np.array_equal(legal, self.legal):
            raise DiagnosticProvenanceError("legal mask differs from attempts")
        validate_scored(self.beam, log_probs, legal)
        self.scores = np.array(log_probs, copy=True)
        scores = write_chunks(
            self.files["scores.bin"], array_chunks(log_probs), self.budget
        )
        masks = write_chunks(self.files["scores.bin"], array_chunks(legal), self.budget)
        self._event(
            {
                "kind": "scored",
                "depth": depth,
                "scores": scores,
                "legal": masks,
                "attempts": self.layer_attempts,
            }
        )
        self.phase = "scored"

    def layer_retained(self, depth, beam, verdicts, parents):
        self._expect("scored")
        if depth != self.depth or len(parents) != len(beam) or len(beam) > 64:
            raise DiagnosticProvenanceError("invalid retained callback")
        for (node, op), why in verdicts.items():
            if (
                type(node) is not int
                or type(op) is not int
                or not 0 <= node < len(self.beam)
                or not 0 <= op < 28
            ):
                raise DiagnosticProvenanceError("invalid verdict identity")
            if not self.legal[node, op] or why not in (
                "retained",
                "duplicate",
                "guard:close",
                "guard:type",
                "guard:size",
            ):
                raise DiagnosticProvenanceError("invalid retention verdict")
        identities = set()
        for child, (node, op) in zip(beam, parents, strict=True):
            if (
                type(node) is not int
                or type(op) is not int
                or (node, op) in identities
                or verdicts.get((node, op)) != "retained"
            ):
                raise DiagnosticProvenanceError("invalid retained parent identity")
            identities.add((node, op))
            parent = self.beam[node]
            expected = (
                child[0],
                (*parent[1], OPS[op].name),
                parent[2] + self.scores[node, op],
            )
            if not same_node(child, expected) or len(child[0]) != len(parent[0]):
                raise DiagnosticProvenanceError("retained program/score mismatch")
        if identities != {key for key, why in verdicts.items() if why == "retained"}:
            raise DiagnosticProvenanceError("retained verdict/parent mismatch")
        self.state = write_chunks(
            self.files["beams.bin"], beam_chunks(beam), self.budget
        )
        self._event(
            {
                "kind": "retained",
                "depth": depth,
                "state": self.state,
                "parents": [list(pair) for pair in parents],
                "verdicts": [[*pair, verdicts[pair]] for pair in sorted(verdicts)],
            }
        )
        self.retained_beam, self.phase = tuple(beam), "retained"

    def search_terminated(self, termination, depth, executions, fuel, expansions):
        self._expect("started", "retained")
        if termination not in TERMINATIONS or (executions, fuel) != (
            self.executions,
            self.fuel,
        ):
            raise DiagnosticProvenanceError("invalid terminal/counter callback")
        if any(
            type(n) is not int or n < 0 for n in (depth, executions, fuel, expansions)
        ):
            raise DiagnosticProvenanceError("invalid terminal integers")
        if self.phase == "retained":
            if (
                termination not in ("max_depth", "frontier_exhausted")
                or depth != self.depth + 1
            ):
                raise DiagnosticProvenanceError("invalid post-retention terminal")
        else:
            expected_depth = (
                self.depth if termination == "frontier_exhausted" else self.depth + 1
            )
            if depth != expected_depth or termination == "max_depth":
                raise DiagnosticProvenanceError("invalid partial/unscored terminal")
        self.terminal = (termination, depth, executions, fuel, expansions)
        self._event(
            {
                "kind": "terminal",
                "terminal": list(self.terminal),
                "unscored_layer": self.phase == "started",
                "partial_expansion": self.layer_attempts < len(self.beam) * 28,
                "last_layer_attempts": self.layer_attempts,
                "total_attempts": self.attempts,
            }
        )
        self.phase = "terminal"

    def finish(self, result):
        self._expect("terminal")
        if self.terminal != (
            result.termination,
            result.depth,
            result.executions,
            result.fuel,
            result.expansions,
        ):
            raise DiagnosticProvenanceError("BeamResult differs from terminal callback")
        if not math.isfinite(result.seconds) or result.seconds < 0:
            raise DiagnosticProvenanceError("invalid result seconds")
        if (result.program is not None) != (result.termination == "fit"):
            raise DiagnosticProvenanceError("invalid fit/program result")
        if result.program is not None and (
            len(result.program) != result.depth
            or any(name not in {op.name for op in OPS} for name in result.program)
        ):
            raise DiagnosticProvenanceError("invalid fit program")
        self._event({"kind": "result", "result": asdict(result)})
        self.close()
        manifest = {}
        for name in self.files:
            digest, size = hashlib.sha256(), 0
            with (self.directory / name).open("rb") as stream:
                while payload := stream.read(1024 * 1024):
                    self.budget.check()
                    digest.update(payload)
                    size += len(payload)
            manifest[name] = {"bytes": size, "sha256": digest.hexdigest()}
        seal = {
            "schema": "SG-CELL1",
            "complete": True,
            "files": manifest,
            "attempts": self.attempts,
            "result": asdict(result),
        }
        size = len(canonical_json(seal))
        self.budget.check(size)
        _publish(self.directory, "cell-seal.json", seal)
        self.budget.bytes_written += size
        self.budget.check()
        self.finished = True
        return seal

    def close(self):
        if not self.closed:
            try:
                for stream in self.files.values():
                    if not stream.closed:
                        stream.flush()
                        os.fsync(stream.fileno())
            finally:
                for stream in self.files.values():
                    stream.close()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
