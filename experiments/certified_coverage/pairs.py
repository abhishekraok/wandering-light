"""certified-coverage (#170): judged (s, t) pairs on closed-state lineages.

A node is the product (one stack per shown example) a program prefix leaves on
a task's ``inputs``. The nodes on a lineage are the inputs, every closed cut
(`closed_cuts`, short-segment-campaign #168) and the final state. A pair is an
ancestor/descendant pair of nodes in one task, so ``t`` is produced from ``s``
by a known program:

- ``d_ops``, the connecting program's length;
- ``d_cuts``, the lineage nodes passed on the way, ``t`` included.

Both are upper bounds on the true distance. Pairs are deduplicated on
``(s, t)``; ``multiplicity`` counts the lineages that gave each one, and the
distances kept are the shortest seen.

Sources:

- ``archive``: the #168 train-task archive. A task's nodes are the inputs, its
  kept closed states and its fit's final state; a pair's two ends are both
  archive nodes (never an unkept state in between).
- ``F1``: example sets 4 and up of the capped train behaviours, on their
  `train_capped.jsonl` witnesses (fresh inputs, seen programs).
- ``F2``: the v4 train behaviours outside `train_capped`, on their corpus
  programs (fresh programs).
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wandering_light.stack_beam import Product, closed_cuts, step_all_quiet
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, OP_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

ARCHIVE = "archive"
FRESH_INPUTS = "F1"
FRESH_PROGRAMS = "F2"
ARCHIVE_SETS = 4  # archive_train's --sets: example sets 0-3 are archived


@dataclass(frozen=True, slots=True)
class Node:
    program: tuple[str, ...]
    product: Product
    # Lineage-node positions along ``program``: its closed cuts and its end.
    marks: frozenset[int]


@dataclass(slots=True)
class Pair:
    source: str
    behaviour: str
    example_set: int
    inputs: tuple[str, ...]
    s_program: tuple[str, ...]
    t_program: tuple[str, ...]
    d_ops: int
    d_cuts: int
    s: Product
    t: Product
    multiplicity: int = 1

    @property
    def k(self) -> int:
        return len(self.s)

    @property
    def key(self) -> tuple[Product, Product]:
        return self.s, self.t

    def row(self) -> dict:
        """The pair without its products; replay the programs on ``inputs`` to rebuild them."""
        return {
            "source": self.source,
            "behaviour": self.behaviour,
            "example_set": self.example_set,
            "k": self.k,
            "inputs": list(self.inputs),
            "s_program": " ".join(self.s_program),
            "t_program": " ".join(self.t_program),
            "d_ops": self.d_ops,
            "d_cuts": self.d_cuts,
            "multiplicity": self.multiplicity,
        }


def parse_inputs(inputs: Sequence[str]) -> Product:
    return tuple(parse_stack(text, basis=BASIS) for text in inputs)


def replay(inputs: Product, program: Sequence[str]) -> list[Product] | None:
    """The products along ``program``, inputs first; None if the machine refuses a step."""
    states = [inputs]
    for name in program:
        moved, _fuel, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None:
            return None
        states.append(moved)
    return states


def node(inputs: Product, program: Sequence[str]) -> Node | None:
    program = tuple(program)
    states = replay(inputs, program)
    if states is None:
        return None
    return Node(
        program, states[-1], frozenset((*closed_cuts(states, program), len(program)))
    )


def lineage_nodes(inputs: Product, program: Sequence[str]) -> list[Node] | None:
    """Every lineage node along one program: inputs, closed cuts, final state."""
    program = tuple(program)
    states = replay(inputs, program)
    if states is None:
        return None
    cuts = {*closed_cuts(states, program), len(program)}
    return [
        Node(program[:m], states[m], frozenset(c for c in cuts if c < m) | {m})
        for m in sorted({0, *cuts})
    ]


def tree_pairs(
    nodes: Iterable[Node],
    *,
    source: str,
    behaviour: str,
    example_set: int,
    inputs: Sequence[str],
    max_cuts: int | None = None,
) -> Iterator[Pair]:
    """Every ancestor/descendant pair among one task's nodes, ``s != t``."""
    by_program = {n.program: n for n in nodes}
    for t in by_program.values():
        for length in range(len(t.program)):
            s = by_program.get(t.program[:length])
            if s is None or s.product == t.product:
                continue
            d_cuts = sum(1 for m in t.marks if length < m <= len(t.program))
            if max_cuts is not None and d_cuts > max_cuts:
                continue
            yield Pair(
                source,
                behaviour,
                example_set,
                tuple(inputs),
                s.program,
                t.program,
                len(t.program) - length,
                d_cuts,
                s.product,
                t.product,
            )


def dedupe(pairs: Iterable[Pair]) -> list[Pair]:
    """One pair per ``(s, t)``, first seen kept; shortest distances; multiplicity counted."""
    kept: dict[tuple[Product, Product], Pair] = {}
    for pair in pairs:
        seen = kept.get(pair.key)
        if seen is None:
            kept[pair.key] = dataclasses.replace(pair, multiplicity=1)
            continue
        seen.multiplicity += 1
        seen.d_ops = min(seen.d_ops, pair.d_ops)
        seen.d_cuts = min(seen.d_cuts, pair.d_cuts)
    return list(kept.values())


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def archive_records(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as lines:
        for line in lines:
            yield json.loads(line)


def archive_task_nodes(record: dict) -> list[Node]:
    """A train-archive task's nodes: the inputs, every kept closed state, the fit's end."""
    inputs = parse_inputs(record["inputs"])
    programs = {()}
    for segment in record["segments"]:
        for root in segment["roots"]:
            programs.add(tuple(root["program"].split()))
        for state in segment["states"]:
            programs.add(tuple(state["program"].split()))
    if record.get("program") is not None:
        programs.add(tuple(record["program"].split()))
    nodes = []
    for program in programs:
        made = node(inputs, program)
        if made is not None:  # the archive only holds replayable programs
            nodes.append(made)
    return nodes


def archive_pairs(path: Path, *, max_cuts: int | None = None) -> list[Pair]:
    """Deduplicated pairs over every task of an archive snapshot."""
    out = []
    for record in archive_records(path):
        out.extend(
            tree_pairs(
                archive_task_nodes(record),
                source=ARCHIVE,
                behaviour=record["behaviour"],
                example_set=record["example_set"],
                inputs=record["inputs"],
                max_cuts=max_cuts,
            )
        )
    return dedupe(out)


def witness_pairs(
    tasks: Iterable[tuple[str, int, Sequence[str], Sequence[str]]],
    *,
    source: str,
    max_cuts: int | None = None,
) -> list[Pair]:
    """Deduplicated pairs along witness lineages: ``(behaviour, example set, inputs, program)``."""
    out = []
    for behaviour, example_set, inputs, program in tasks:
        nodes = lineage_nodes(parse_inputs(inputs), program)
        if nodes is None:
            continue
        out.extend(
            tree_pairs(
                nodes,
                source=source,
                behaviour=behaviour,
                example_set=example_set,
                inputs=inputs,
                max_cuts=max_cuts,
            )
        )
    return dedupe(out)


def fresh_tasks(
    v4: dict[str, list],
    capped: dict[str, str],
) -> dict[str, list[tuple[str, int, tuple[str, ...], tuple[str, ...]]]]:
    """F1 and F2 witness tasks from the v4 train split; eval behaviours never enter.

    ``v4`` is behaviour -> tasks in corpus order (`staged.v4_tasks`), ``capped``
    behaviour -> its `train_capped.jsonl` program. Example sets are numbered as
    `archive_train` numbers them, so F1's sets 4 and up are the unarchived ones.
    """
    f1, f2 = [], []
    for behaviour, tasks in sorted(v4.items()):
        train = [t for t in tasks if t.split == "train"]
        for number, task in enumerate(train):
            if behaviour in capped:
                if number >= ARCHIVE_SETS:
                    program = tuple(capped[behaviour].split())
                    f1.append((behaviour, number, tuple(task.inputs), program))
            elif task.program:
                f2.append((behaviour, number, tuple(task.inputs), tuple(task.program)))
    return {FRESH_INPUTS: f1, FRESH_PROGRAMS: f2}


def draw(pairs: Sequence[Pair], quota: int | None, seed: int) -> list[Pair]:
    """Up to ``quota`` pairs per behaviour, uniform within it, reproducibly.

    Behaviours with fewer pairs give all they have. Pairs are put in a
    canonical order first, because their build order follows set iteration,
    which string hashing randomises per process.
    """
    by_behaviour: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_behaviour.setdefault(pair.behaviour, []).append(pair)
    out = []
    for behaviour, group in sorted(by_behaviour.items()):
        group.sort(key=lambda p: (p.example_set, p.s_program, p.t_program))
        if quota is None or quota >= len(group):
            out.extend(group)
        else:
            out.extend(random.Random(f"{seed}:{behaviour}").sample(group, quota))
    return out
