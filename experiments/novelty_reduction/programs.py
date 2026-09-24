"""novelty-reduction (#171): random capped-style programs, inputs and fresh (F3) pairs.

A proposal from a state ``s`` is 1-3 random segments. A segment is 0-3 plain
ops, then ``[`` + a flat body of 1-4 plain atoms + ``]``, then one of
map/filter/fold/dip/i: at most 8 ops, closed at its end. A candidate segment
is rejected unless it runs on every example, ends closed, changes the state,
revisits no state on the path and stays under the atom cap. M2 restricts the
state graph this way (no ifte, no nesting); widening it is later work.

Inputs come from the v4 train behaviours' samplers only (eval samplers never),
with k uniform in 3-6. F3 is the same generator on inputs from seeds the loop
never uses, never archived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from experiments.certified_coverage import pairs as sampler
from wandering_light.stack_beam import atoms_exceed, closed_cuts, step_all_quiet
from wandering_light.stack_pbe import BY_NAME, OP_BY_NAME, build_task, draw_task

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from wandering_light.stack_beam import Product

PLAIN = (
    *("0", "1", "nil", "true", "false"),
    *("dup", "swap", "drop", "over", "cons", "uncons", "null?"),
    *("add", "sub", "mul", "div", "eq?", "lt?", "quote", "cat"),
)
COMBINATORS = ("map", "filter", "fold", "dip", "i")
RUNS = frozenset((*COMBINATORS, "ifte"))
SEGMENT_OPS = 8
BODY_ATOMS = 4
PREFIX_OPS = 3
MAX_STATE_ATOMS = 16_384
FRESH = "F3"


@dataclass(frozen=True, slots=True)
class Grammar:
    """The grammar level (max body atoms, max segments, nesting) and rejection limits.

    M2 is fixed at level (4, 3, none). A later run can start smaller, e.g.
    (2, 1, none), and widen as certification grows; the widening rule is not
    built. Nesting is not implemented, so only ``nesting=False`` is accepted.
    """

    max_body: int = BODY_ATOMS
    max_segments: int = 3
    nesting: bool = False
    tries: int = 200  # candidate segments per step before the proposal ends
    max_state_atoms: int = MAX_STATE_ATOMS

    def __post_init__(self) -> None:
        if self.nesting:
            raise NotImplementedError("nested bodies are not in the M2 grammar")
        if not 1 <= self.max_body <= BODY_ATOMS or self.max_segments < 1:
            raise ValueError(f"bad grammar level {self.level}")

    @property
    def level(self) -> str:
        return f"{self.max_body},{self.max_segments},{'nested' if self.nesting else 'none'}"

    @staticmethod
    def parse(level: str) -> Grammar:
        body, segments, nesting = level.split(",")
        if nesting not in ("none", "nested"):
            raise ValueError(f"nesting must be none or nested, not {nesting!r}")
        return Grammar(int(body), int(segments), nesting == "nested")


GRAMMAR = Grammar()


def body_atoms(path) -> int:
    """The most atoms in any top-level quotation of ``path``, nested atoms counted."""
    best = depth = count = 0
    for name in path:
        if name == "[":
            depth += 1
            if depth == 1:
                count = 0
            continue
        if name == "]":
            depth -= 1
            if depth == 0:
                best = max(best, count)
            continue
        if depth >= 1:
            count += 1
    return best


def path_segments(path) -> int:
    """Top-level combinator applications along ``path``."""
    depth = segments = 0
    for name in path:
        depth += (name == "[") - (name == "]")
        segments += depth == 0 and name in RUNS
    return segments


def body_stratum(atoms: int) -> str:
    return {0: "0", 1: "1", 2: "2", 3: "3-4", 4: "3-4"}.get(atoms, "5+")


@dataclass(slots=True)
class Proposal:
    """A random program from ``states[0]``; ``cuts`` are its closed op counts."""

    ops: list[str]
    states: list
    cuts: list[int]


def random_segment(rng: random.Random, grammar: Grammar = GRAMMAR) -> tuple[str, ...]:
    body = rng.randint(1, grammar.max_body)
    prefix = rng.randint(0, min(PREFIX_OPS, SEGMENT_OPS - 3 - body))
    return (
        *rng.choices(PLAIN, k=prefix),
        "[",
        *rng.choices(PLAIN, k=body),
        "]",
        rng.choice(COMBINATORS),
    )


def run_segment(
    product: Product, segment: Sequence[str], max_state_atoms: int
) -> list | None:
    """The products along ``segment`` if it is admissible from ``product``, else None."""
    states = [product]
    for name in segment:
        moved, _fuel, _stepped = step_all_quiet(states[-1], OP_BY_NAME[name])
        if moved is None or atoms_exceed(moved, max_state_atoms):
            return None
        states.append(moved)
    if len(segment) not in closed_cuts(states, segment):
        return None
    return states


def propose(
    product: Product, rng: random.Random, grammar: Grammar = GRAMMAR
) -> Proposal:
    """Up to a random number of admissible segments from ``product``; possibly none."""
    target = rng.randint(1, grammar.max_segments)
    out = Proposal([], [product], [])
    seen = {product}
    for _ in range(target):
        for _ in range(grammar.tries):
            segment = random_segment(rng, grammar)
            states = run_segment(out.states[-1], segment, grammar.max_state_atoms)
            if states is None or states[-1] in seen:
                continue
            seen.add(states[-1])
            out.ops.extend(segment)
            out.states.extend(states[1:])
            out.cuts.append(len(out.ops))
            break
        else:
            break
    return out


def train_behaviours() -> list[str]:
    """The v4 train behaviours, whose input samplers are the loop's only source."""
    from experiments.staged_witnesses import check as staged

    tasks = staged.v4_tasks()
    return sorted(
        b for b, group in tasks.items() if any(t.split == "train" for t in group)
    )


def draw_inputs(
    rng: random.Random, behaviours: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """One fresh input set: a train behaviour's sampler, k uniform in 3-6."""
    name = rng.choice(behaviours)
    task = build_task(
        BY_NAME[name], draw_task(BY_NAME[name], rng, rng.randint(3, 6), 1)
    )
    return name, tuple(task.inputs)


def fresh_pairs(
    count: int,
    rng: random.Random,
    behaviours: Sequence[str],
    *,
    max_cuts: int = 1,
    grammar: Grammar = GRAMMAR,
    attempts: int = 20,
) -> list[sampler.Pair]:
    """``count`` F3 pairs, one per fresh input set, uniform over its pairs at d_cuts <= max_cuts."""
    out: list[sampler.Pair] = []
    seen = set()
    for _ in range(count * attempts):
        if len(out) == count:
            break
        behaviour, inputs = draw_inputs(rng, behaviours)
        proposal = propose(sampler.parse_inputs(inputs), rng, grammar)
        if not proposal.cuts:
            continue
        nodes = sampler.lineage_nodes(sampler.parse_inputs(inputs), proposal.ops)
        found = [
            p
            for p in sampler.tree_pairs(
                nodes,
                source=FRESH,
                behaviour=behaviour,
                example_set=-1,
                inputs=inputs,
                max_cuts=max_cuts,
            )
            if p.key not in seen
        ]
        if found:
            pick = rng.choice(found)
            seen.add(pick.key)
            out.append(pick)
    return out
