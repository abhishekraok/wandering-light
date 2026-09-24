"""novelty-reduction (#171): the loop's own closed-state archive.

Nodes are canonical states (the product itself, so k is part of the key).
Every archived pair ``(s, t)`` keeps its shortest known path, ``d_ops`` and
``d_cuts`` (lineage nodes passed, ``t`` included), where it came from, and
its attempt count. A relaxation is a later path between an archived pair in
fewer ops: the path is overwritten and the event logged.

``visits[x]`` is v(x): attempts with x as start or target, plus walker paths
through x. ``novelty`` is the count table's N(x) = 1 / sqrt(1 + v(x)).

The machine is deterministic, so a path is valid from ``s`` wherever ``s``
was met; every node also keeps one origin (a root's input texts and a program
from them), so an archive can be rebuilt from text.
"""

from __future__ import annotations

import gzip
import math
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wandering_light.stack_beam import closed_cuts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from wandering_light.stack_beam import Product

PROPOSAL = "proposal"
ROLLOUT = "rollout"


@dataclass(slots=True)
class PairRecord:
    """``path``/``d_ops``/``d_cuts`` are the shortest known and may relax; the
    ``witness_*`` fields are the path that first archived the pair and never
    change, so strata and distance features keep one provenance (#171 F5)."""

    s: int
    t: int
    path: tuple[str, ...]
    d_ops: int
    d_cuts: int
    origin: str
    cycle: int
    attempts: int = 0
    witness_path: tuple[str, ...] = ()
    witness_d_ops: int = 0
    witness_d_cuts: int = 0

    def __post_init__(self) -> None:
        if not self.witness_path:
            self.witness_path = self.path
            self.witness_d_ops = self.d_ops
            self.witness_d_cuts = self.d_cuts


@dataclass(slots=True)
class LoopArchive:
    states: list = field(default_factory=list)
    index: dict = field(default_factory=dict)  # product -> node id
    visits: list[int] = field(default_factory=list)
    # node id -> (inputs texts, program from them), the first route seen
    origins: list = field(default_factory=list)
    pairs: dict[tuple[int, int], PairRecord] = field(default_factory=dict)
    relaxations: list[dict] = field(default_factory=list)

    def node(self, product: Product, origin=None) -> int:
        found = self.index.get(product)
        if found is not None:
            return found
        self.index[product] = len(self.states)
        self.states.append(product)
        self.visits.append(0)
        self.origins.append(origin)
        return len(self.states) - 1

    def novelty(self, node: int) -> float:
        return 1 / math.sqrt(1 + self.visits[node])

    def add_path(
        self,
        states: Sequence,
        ops: Sequence[str],
        origin: str,
        cycle: int,
        *,
        start_origin=None,
    ) -> tuple[int, int]:
        """Archive a path's nodes and every (ancestor, descendant) pair on it.

        Nodes are the start, every closed cut, and any state already archived.
        ``start_origin`` is the start's route from text, when it is new; each
        new node's route extends it. Returns ``(new pairs, relaxations)``.
        """
        ops = tuple(ops)
        positions = sorted(
            {0, *closed_cuts(states, ops)}
            | {i for i, state in enumerate(states) if state in self.index}
        )
        ids = []
        for position in positions:
            route = None
            if start_origin is not None:
                route = (start_origin[0], (*start_origin[1], *ops[:position]))
            ids.append(self.node(states[position], route))
        added = relaxed = 0
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                s, t = ids[a], ids[b]
                if s == t:
                    continue
                i, j = positions[a], positions[b]
                d_cuts = b - a
                seen = self.pairs.get((s, t))
                if seen is None:
                    self.pairs[(s, t)] = PairRecord(
                        s, t, ops[i:j], j - i, d_cuts, origin, cycle
                    )
                    added += 1
                elif j - i < seen.d_ops:
                    self.relaxations.append(
                        {
                            "cycle": cycle,
                            "s": s,
                            "t": t,
                            "old_d_ops": seen.d_ops,
                            "new_d_ops": j - i,
                            "by": origin,
                        }
                    )
                    seen.path, seen.d_ops, seen.d_cuts = ops[i:j], j - i, d_cuts
                    relaxed += 1
        return added, relaxed

    def origin_of(self, node: int):
        return self.origins[node]

    def lineage(self, node: int) -> tuple:
        """The input set a node was first reached from: its root's texts."""
        origin = self.origins[node]
        return origin[0] if origin is not None else ("node", node)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> LoopArchive:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)

    def summary(self) -> dict:
        by_origin: dict[str, int] = {}
        for record in self.pairs.values():
            by_origin[record.origin] = by_origin.get(record.origin, 0) + 1
        return {
            "nodes": len(self.states),
            "pairs": len(self.pairs),
            "pairs_by_origin": by_origin,
            "relaxations": len(self.relaxations),
        }
