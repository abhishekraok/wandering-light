"""Mine candidate basis words from solved programs, and price each one twice.

Under a concatenative reading of the corpus, any contiguous run of a solved
program is itself a program, so growing the basis is a substring operation: cut
the run, name it.  That makes Sequitur (Nevill-Manning and Witten, JAIR 1997)
the natural candidate generator -- every rule it forms is a repeated substring
that a hierarchy already found worth naming, and its two constraints are the
two things a miner needs: digram uniqueness is deduplication, and rule utility
is a description-length test.

The two prices are the point of this script.

*Description length.*  Naming a substring of length ``L`` occurring ``F`` times,
with ``N`` total corpus ops and basis size ``B``, writing the definition once::

    before = N * log2(B)
    after  = (N - F * (L - 1)) * log2(B + 1) + L * log2(B)

Accept iff ``after < before``.  Equivalently, a word pays iff naming it removes
more than ``(log2(B+1) - log2(B)) / log2(B+1)`` of the total corpus ops -- that
is 0.177% at ``B = 118``.  The ``log2(fuel)`` term of Kt cancels exactly here,
because a named word executes precisely the ops it abbreviates, so this is the
whole criterion rather than an approximation of one.

*Branching.*  MDL does not price search at all.  A new word is one more edge out
of every state whose type it accepts, so it raises the effective branching
factor and multiplies the cost of a depth-``d`` search by ``(b_new/b_old)**d``.
A word that need only compress 0.177% can easily cost several percent of
search.  Both numbers are reported per candidate so that the trade is visible
instead of one objective being optimised blind.

Applicability follows ``TrajectoryGraph.expand``: a function is legal at a state
iff ``fn.input_type_cls() is state.item_type``.  ``Executor.execute`` stamps its
result with the declared ``output_type`` of the function, so the type of every
state along a witness is fixed by the declared signatures alone, and the
branching model needs no execution.

Acceptance is greedy rather than a batch.  Each accepted word raises ``B``,
which raises the bar for everything after it, and overlapping candidates
cannibalise each other counts, so the corpus is rewritten and Sequitur re-run
after every acceptance.

Run it with::

    uv run python experiments/mine_basis_words.py --report-dir reports/DIR
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

CORPUS_DIR = "wandering_light/training/data/deep_corpus_v1"
SPLITS = ("discovery", "validation", "test")
BASIS_SET_ID = "wl-core-v1"
SEARCH_DEPTH = 6

# Ops that map a state to a pseudorandom one of the same type. They manufacture
# reachable states cheaply, which is exactly what a frequency-based miner
# rewards and what a basis should not grow toward. On the stack machine the
# same failure mode wore the name `quote quote`.
DIGEST_OPS = frozenset({"str_hash", "set_hash"})

SEPARATOR_PREFIX = "\x00sep"


# --------------------------------------------------------------------------
# Sequitur
# --------------------------------------------------------------------------


class _Symbol:
    """One cell of the doubly linked list a Sequitur grammar is made of.

    A symbol is terminal (``rule is None``, ``value`` set) or non-terminal
    (``rule`` set).  Every rule owns one guard symbol that serves as both head
    and tail sentinel of its body; guards never take part in digrams.
    """

    __slots__ = ("is_guard", "nxt", "prev", "rule", "value")

    def __init__(
        self,
        value: str | None = None,
        rule: _Rule | None = None,
        *,
        is_guard: bool = False,
    ) -> None:
        self.value = value
        self.rule = rule
        self.is_guard = is_guard
        self.prev: _Symbol | None = None
        self.nxt: _Symbol | None = None

    @property
    def key(self) -> Any:
        """What makes two symbols interchangeable for digram matching."""
        return self.value if self.rule is None else self.rule

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.is_guard:
            return "<guard>"
        label = self.value if self.rule is None else f"R{self.rule.index}"
        return f"<{label}>"


class _Rule:
    __slots__ = ("guard", "index", "ref_count")

    def __init__(self, index: int) -> None:
        self.index = index
        self.ref_count = 0
        self.guard = _Symbol(rule=self, is_guard=True)
        self.guard.prev = self.guard
        self.guard.nxt = self.guard

    @property
    def first(self) -> _Symbol:
        return self.guard.nxt

    @property
    def last(self) -> _Symbol:
        return self.guard.prev

    def body(self) -> Iterator[_Symbol]:
        symbol = self.guard.nxt
        while not symbol.is_guard:
            yield symbol
            symbol = symbol.nxt


class Sequitur:
    """Linear-time grammar induction under digram uniqueness and rule utility.

    ``feed`` appends one terminal to the start rule and restores both
    constraints.  Sequences are separated by sentinels unique to each boundary,
    so no rule can straddle two programs: such a rule would not be a substring
    of any single program, and so not a nameable word.
    """

    def __init__(self) -> None:
        self._digrams: dict[tuple[Any, Any], _Symbol] = {}
        self._next_index = 0
        self.rules: list[_Rule] = []
        self.start = self._new_rule()

    # -- structure -------------------------------------------------------

    def _new_rule(self) -> _Rule:
        rule = _Rule(self._next_index)
        self._next_index += 1
        self.rules.append(rule)
        return rule

    def _digram_key(self, symbol: _Symbol) -> tuple[Any, Any] | None:
        nxt = symbol.nxt
        if symbol.is_guard or nxt is None or nxt.is_guard:
            return None
        return (symbol.key, nxt.key)

    def _forget_digram(self, symbol: _Symbol) -> None:
        key = self._digram_key(symbol)
        if key is not None and self._digrams.get(key) is symbol:
            del self._digrams[key]

    def _remember_digram(self, symbol: _Symbol) -> None:
        key = self._digram_key(symbol)
        if key is not None:
            self._digrams[key] = symbol

    def _join(self, left: _Symbol, right: _Symbol) -> None:
        if left.nxt is not None:
            self._forget_digram(left)
            # A run of three equal symbols holds two overlapping copies of one
            # digram but only one table slot. Breaking the recorded copy has to
            # hand the slot to the copy that survives, or the occurrence is
            # lost and the grammar stops being unique in its digrams.
            if (
                right.prev is not None
                and right.nxt is not None
                and right.key == right.prev.key == right.nxt.key
            ):
                self._remember_digram(right)
            if (
                left.prev is not None
                and left.nxt is not None
                and left.key == left.nxt.key == left.prev.key
            ):
                self._remember_digram(left.prev)
        left.nxt = right
        right.prev = left

    def _insert_after(self, anchor: _Symbol, symbol: _Symbol) -> None:
        self._join(symbol, anchor.nxt)
        self._join(anchor, symbol)

    def _unlink(self, symbol: _Symbol) -> None:
        self._join(symbol.prev, symbol.nxt)
        if not symbol.is_guard:
            self._forget_digram(symbol)
            if symbol.rule is not None:
                symbol.rule.ref_count -= 1

    # -- the two constraints ---------------------------------------------

    def _check(self, symbol: _Symbol) -> bool:
        """Enforce digram uniqueness at ``symbol``; True if it acted."""
        key = self._digram_key(symbol)
        if key is None:
            return False
        seen = self._digrams.get(key)
        if seen is None:
            self._digrams[key] = symbol
            return False
        if seen.nxt is not symbol:
            self._match(symbol, seen)
        return True

    def _match(self, here: _Symbol, seen: _Symbol) -> None:
        if seen.prev.is_guard and seen.nxt.nxt.is_guard:
            # The earlier occurrence is the entire body of an existing rule.
            # Reuse that rule instead of minting a second name for it.
            rule = seen.prev.rule
            self._substitute(here, rule)
        else:
            rule = self._new_rule()
            self._insert_after(rule.last, self._copy(here))
            self._insert_after(rule.last, self._copy(here.nxt))
            self._substitute(seen, rule)
            self._substitute(here, rule)
            self._remember_digram(rule.first)

        first = rule.first
        if first.rule is not None and first.rule.ref_count == 1:
            self._expand(first)

    def _copy(self, symbol: _Symbol) -> _Symbol:
        clone = _Symbol(value=symbol.value, rule=symbol.rule)
        if clone.rule is not None:
            clone.rule.ref_count += 1
        return clone

    def _substitute(self, symbol: _Symbol, rule: _Rule) -> None:
        anchor = symbol.prev
        self._unlink(anchor.nxt)
        self._unlink(anchor.nxt)
        replacement = _Symbol(rule=rule)
        rule.ref_count += 1
        self._insert_after(anchor, replacement)
        if not self._check(anchor):
            self._check(anchor.nxt)

    def _expand(self, symbol: _Symbol) -> None:
        """Rule utility: a rule referenced once is inlined and retired.

        Only ever reached on the first symbol of a rule body, so the left
        neighbour is that guard and the digram it forms is not a real one.
        """
        rule = symbol.rule
        left, right = symbol.prev, symbol.nxt
        first, last = rule.first, rule.last
        self._forget_digram(symbol)
        self.rules.remove(rule)
        self._join(left, first)
        self._join(last, right)
        self._remember_digram(last)

    # -- driving ---------------------------------------------------------

    def feed(self, token: str) -> None:
        self._insert_after(self.start.last, _Symbol(value=token))
        self._check(self.start.last.prev)

    def feed_sequences(self, sequences: Iterable[Sequence[str]]) -> Sequitur:
        for index, sequence in enumerate(sequences):
            for token in sequence:
                self.feed(token)
            # A sentinel unique to this boundary can never repeat, so no digram
            # containing it can ever become a rule.
            self.feed(f"{SEPARATOR_PREFIX}{index}")
        return self

    # -- reading the result ----------------------------------------------

    def expansion(self, rule: _Rule) -> tuple[str, ...]:
        """The terminal string a rule stands for."""
        out: list[str] = []
        stack: list[_Symbol] = list(rule.body())[::-1]
        while stack:
            symbol = stack.pop()
            if symbol.rule is None:
                out.append(symbol.value)
            else:
                stack.extend(list(symbol.rule.body())[::-1])
        return tuple(out)

    def start_string(self) -> list[str]:
        """The start rule as a flat list, non-terminals rendered as ``R<n>``."""
        return [
            symbol.value if symbol.rule is None else f"R{symbol.rule.index}"
            for symbol in self.start.body()
        ]

    def digram_multiset(self) -> Counter[tuple[Any, Any]]:
        """Every digram in every rule body, for checking digram uniqueness."""
        found: Counter[tuple[Any, Any]] = Counter()
        for rule in self.rules:
            body = list(rule.body())
            for left, right in itertools.pairwise(body):
                found[(left.key, right.key)] += 1
        return found

    def words(self, min_length: int = 2) -> list[tuple[str, ...]]:
        """Every rule terminal expansion, longest first, sentinels dropped."""
        found: list[tuple[str, ...]] = []
        for rule in self.rules:
            if rule is self.start:
                continue
            word = self.expansion(rule)
            if len(word) < min_length:
                continue
            if any(token.startswith(SEPARATOR_PREFIX) for token in word):
                continue
            found.append(word)
        seen: set[tuple[str, ...]] = set()
        unique: list[tuple[str, ...]] = []
        for word in sorted(found, key=lambda w: (-len(w), w)):
            if word not in seen:
                seen.add(word)
                unique.append(word)
        return unique


def sequitur_words(
    sequences: Iterable[Sequence[str]], min_length: int = 2
) -> list[tuple[str, ...]]:
    """Run Sequitur over a corpus of sequences and return its rules as words."""
    return Sequitur().feed_sequences(sequences).words(min_length=min_length)


# --------------------------------------------------------------------------
# The description-length criterion
# --------------------------------------------------------------------------


def compression_bar(basis_size: int) -> float:
    """Fraction of total ops a word must remove to pay for its slot.

    Scale free: independent of ``N`` and of the length of the word itself.
    """
    if basis_size < 1:
        raise ValueError("basis_size must be at least 1")
    widened = math.log2(basis_size + 1)
    return (widened - math.log2(basis_size)) / widened


def mdl_delta_bits(
    *, total_ops: int, basis_size: int, occurrences: int, word_length: int
) -> float:
    """Bits saved by naming a word. Positive means the word pays for itself."""
    if word_length < 2:
        raise ValueError("a word must abbreviate at least two ops")
    if occurrences < 0:
        raise ValueError("occurrences must be non-negative")
    ops_removed = occurrences * (word_length - 1)
    if ops_removed > total_ops:
        raise ValueError("a word cannot remove more ops than the corpus has")
    before = total_ops * math.log2(basis_size)
    after = (total_ops - ops_removed) * math.log2(basis_size + 1) + (
        word_length * math.log2(basis_size)
    )
    return before - after


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Program:
    ops: tuple[str, ...]
    input_type: str
    root_digest: str
    split: str
    certified_distance: int


def load_programs(
    corpus_dir: str = CORPUS_DIR, splits: Sequence[str] = SPLITS
) -> list[Program]:
    """Read ``witness_function_names`` plus the provenance a review needs."""
    programs: list[Program] = []
    for split in splits:
        path = os.path.join(corpus_dir, f"{split}.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                metadata = record.get("metadata", {})
                programs.append(
                    Program(
                        ops=tuple(record["witness_function_names"]),
                        input_type=metadata.get("input_type", ""),
                        root_digest=metadata.get("root_digest", ""),
                        split=split,
                        certified_distance=int(metadata.get("certified_distance", 0)),
                    )
                )
    return programs


# --------------------------------------------------------------------------
# Occurrence counting
# --------------------------------------------------------------------------


def count_occurrences(
    sequences: Sequence[Sequence[str]], words: Sequence[tuple[str, ...]]
) -> dict[tuple[str, ...], list[tuple[int, int]]]:
    """Non-overlapping occurrences of every word, greedily left to right.

    Only non-overlapping copies can all be rewritten, so counting overlapping
    matches would overstate what naming a word actually removes.  One pass over
    the corpus per distinct word length finds every match; a greedy sweep then
    thins the list of each word.
    """
    by_length: dict[int, set[tuple[str, ...]]] = {}
    for word in words:
        by_length.setdefault(len(word), set()).add(word)
    lengths = sorted(by_length)

    raw: dict[tuple[str, ...], list[tuple[int, int]]] = {word: [] for word in words}
    for index, sequence in enumerate(sequences):
        tokens = tuple(sequence)
        span = len(tokens)
        for length in lengths:
            if length > span:
                break
            wanted = by_length[length]
            for start in range(span - length + 1):
                piece = tokens[start : start + length]
                if piece in wanted:
                    raw[piece].append((index, start))

    thinned: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for word, hits in raw.items():
        length = len(word)
        kept: list[tuple[int, int]] = []
        last_index, last_end = -1, -1
        for sequence_index, start in hits:
            if sequence_index != last_index or start >= last_end:
                kept.append((sequence_index, start))
                last_index, last_end = sequence_index, start + length
        thinned[word] = kept
    return thinned


def rewrite(
    sequences: Sequence[Sequence[str]],
    word: tuple[str, ...],
    name: str,
    hits: Sequence[tuple[int, int]],
) -> list[tuple[str, ...]]:
    """Replace the chosen occurrences of ``word`` with a single token."""
    starts: dict[int, set[int]] = {}
    for index, start in hits:
        starts.setdefault(index, set()).add(start)
    length = len(word)
    out: list[tuple[str, ...]] = []
    for index, sequence in enumerate(sequences):
        wanted = starts.get(index)
        if not wanted:
            out.append(tuple(sequence))
            continue
        tokens = tuple(sequence)
        rewritten: list[str] = []
        cursor = 0
        while cursor < len(tokens):
            if cursor in wanted:
                rewritten.append(name)
                cursor += length
            else:
                rewritten.append(tokens[cursor])
                cursor += 1
        out.append(tuple(rewritten))
    return out


# --------------------------------------------------------------------------
# Branching
# --------------------------------------------------------------------------


@dataclass
class BranchingModel:
    """Effective branching under ``TrajectoryGraph``-style type compatibility.

    ``in_degree`` counts, per state type, how many basis functions accept it.
    ``state_types`` is the distribution over states a search actually expands,
    taken from every state along every corpus witness.  Adding a word adds one
    legal move at exactly the states whose type its first op accepts, so its
    branching cost is below one edge per state whenever that type is not
    universal -- which is the part a flat ``b + 1`` estimate gets wrong.
    """

    signatures: dict[str, tuple[str, str]]
    in_degree: Counter[str] = field(default_factory=Counter)
    state_types: Counter[str] = field(default_factory=Counter)
    extra: list[tuple[str, str]] = field(default_factory=list)
    growth_cache: dict[tuple[str, str] | None, float] = field(default_factory=dict)

    @classmethod
    def from_basis(
        cls, signatures: dict[str, tuple[str, str]], programs: Sequence[Program]
    ) -> BranchingModel:
        model = cls(signatures=dict(signatures))
        for input_type, _ in signatures.values():
            model.in_degree[input_type] += 1
        for program in programs:
            current = program.input_type
            model.state_types[current] += 1
            for op in program.ops:
                current = signatures[op][1]
                model.state_types[current] += 1
        return model

    @property
    def num_states(self) -> int:
        return sum(self.state_types.values())

    def type_fraction(self, type_name: str) -> float:
        total = self.num_states
        return self.state_types[type_name] / total if total else 0.0

    def b_eff(self, added: tuple[str, str] | None = None) -> float:
        """Mean number of legal functions per visited state."""
        total = self.num_states
        if not total:
            return 0.0
        degree = Counter(self.in_degree)
        for input_type, _ in self.extra:
            degree[input_type] += 1
        weighted = sum(
            count * degree[name] for name, count in self.state_types.items()
        )
        base = weighted / total
        if added is None:
            return base
        return base + self.type_fraction(added[0])

    def accept(self, signature: tuple[str, str]) -> None:
        self.extra.append(signature)
        self.growth_cache.clear()

    # -- typed growth rate ------------------------------------------------

    def growth_rate(self, added: tuple[str, str] | None = None) -> float:
        """Per-step growth of the typed search tree.

        ``b_eff`` treats every state as if it drew successors from one pool.  A
        typed basis does not: a ``bool`` state has five legal moves and a
        ``str`` state twenty-nine, so the real growth of the tree is set by how
        the types feed each other.  The dominant eigenvalue of the
        type-transition count matrix is that rate, and it is the quantity
        ``b**d`` was standing in for.
        """
        if added in self.growth_cache:
            return self.growth_cache[added]
        types = sorted(
            set(self.state_types)
            | {t for t, _ in self.signatures.values()}
            | {t for _, t in self.signatures.values()}
        )
        position = {name: i for i, name in enumerate(types)}
        size = len(types)
        matrix = [[0.0] * size for _ in range(size)]
        edges = list(self.signatures.values()) + list(self.extra)
        if added is not None:
            edges.append(added)
        for input_type, output_type in edges:
            matrix[position[input_type]][position[output_type]] += 1.0

        vector = [1.0] * size
        rate = 0.0
        for _ in range(400):
            nxt = [
                sum(matrix[i][j] * vector[j] for j in range(size)) for i in range(size)
            ]
            scale = max(nxt)
            if scale <= 0:
                rate = 0.0
                break
            nxt = [value / scale for value in nxt]
            settled = all(abs(a - b) < 1e-12 for a, b in zip(nxt, vector, strict=True))
            vector, rate = nxt, scale
            if settled:
                break
        self.growth_cache[added] = rate
        return rate


def search_multiplier(b_old: float, b_new: float, depth: int = SEARCH_DEPTH) -> float:
    """How much more a depth-``d`` search costs once the word is legal."""
    if b_old <= 0:
        raise ValueError("b_old must be positive")
    return (b_new / b_old) ** depth


# --------------------------------------------------------------------------
# Degeneracy diagnostics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostics:
    """Signals that a frequent word is not compositional structure.

    None of these is decisive on its own; together they are enough to hold a
    candidate back for a human read, which is the only filter that caught
    ``quote quote`` on the stack machine.
    """

    distinct_followers: int
    distinct_predecessors: int
    terminal_fraction: float
    initial_fraction: float
    distinct_roots: int
    distinct_input_types: int
    single_op_repeat: bool
    contains_digest: bool

    @property
    def flags(self) -> list[str]:
        found: list[str] = []
        if self.single_op_repeat:
            found.append("one-op-repeat")
        if self.contains_digest:
            found.append("digest-projection")
        if self.terminal_fraction >= 0.9:
            found.append("terminal-sink")
        elif self.distinct_followers <= 1:
            found.append("single-follower")
        if self.distinct_roots <= 2:
            found.append("root-concentrated")
        return found


def diagnose(
    word: tuple[str, ...],
    hits: Sequence[tuple[int, int]],
    sequences: Sequence[Sequence[str]],
    programs: Sequence[Program],
    primitives: Sequence[str],
) -> Diagnostics:
    followers: Counter[str] = Counter()
    predecessors: Counter[str] = Counter()
    roots: set[str] = set()
    input_types: set[str] = set()
    terminal = 0
    initial = 0
    length = len(word)
    for index, start in hits:
        tokens = sequences[index]
        end = start + length
        if end >= len(tokens):
            terminal += 1
        else:
            followers[tokens[end]] += 1
        if start == 0:
            initial += 1
        else:
            predecessors[tokens[start - 1]] += 1
        roots.add(programs[index].root_digest)
        input_types.add(programs[index].input_type)
    total = max(len(hits), 1)
    return Diagnostics(
        distinct_followers=len(followers),
        distinct_predecessors=len(predecessors),
        terminal_fraction=terminal / total,
        initial_fraction=initial / total,
        distinct_roots=len(roots),
        distinct_input_types=len(input_types),
        single_op_repeat=len(set(primitives)) == 1,
        contains_digest=any(op in DIGEST_OPS for op in primitives),
    )


# --------------------------------------------------------------------------
# Greedy mining
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    round_index: int
    name: str
    word: tuple[str, ...]
    primitives: tuple[str, ...]
    length: int
    primitive_length: int
    occurrences: int
    ops_removed: int
    removed_fraction: float
    mdl_delta: float
    total_ops_before: int
    basis_size_before: int
    bar: float
    b_old: float
    b_new: float
    multiplier: float
    growth_old: float
    growth_new: float
    growth_multiplier: float
    mean_depth: float
    net_multiplier: float
    signature: tuple[str, str]
    diagnostics: Diagnostics

    @property
    def b_delta(self) -> float:
        return self.b_new - self.b_old

    def as_row(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "name": self.name,
            "word": list(self.word),
            "primitives": list(self.primitives),
            "length": self.length,
            "primitive_length": self.primitive_length,
            "occurrences": self.occurrences,
            "ops_removed": self.ops_removed,
            "removed_fraction": self.removed_fraction,
            "mdl_delta_bits": self.mdl_delta,
            "total_ops_before": self.total_ops_before,
            "basis_size_before": self.basis_size_before,
            "basis_size_after": self.basis_size_before + 1,
            "bar": self.bar,
            "b_eff_before": self.b_old,
            "b_eff_after": self.b_new,
            "b_eff_delta": self.b_delta,
            "search_multiplier_d6": self.multiplier,
            "growth_rate_before": self.growth_old,
            "growth_rate_after": self.growth_new,
            "growth_multiplier_d6": self.growth_multiplier,
            "mean_depth_d6": self.mean_depth,
            "net_multiplier_d6": self.net_multiplier,
            "input_type": self.signature[0],
            "output_type": self.signature[1],
            "flags": self.diagnostics.flags,
            "distinct_followers": self.diagnostics.distinct_followers,
            "distinct_predecessors": self.diagnostics.distinct_predecessors,
            "terminal_fraction": self.diagnostics.terminal_fraction,
            "initial_fraction": self.diagnostics.initial_fraction,
            "distinct_roots": self.diagnostics.distinct_roots,
            "distinct_input_types": self.diagnostics.distinct_input_types,
        }


@dataclass
class MiningResult:
    accepted: list[Candidate]
    first_round: list[Candidate]
    first_round_rules: int
    total_ops_before: int
    total_ops_after: int
    basis_size_before: int
    model: BranchingModel


def _expand_word(
    word: Sequence[str], defined: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    out: list[str] = []
    for token in word:
        out.extend(defined.get(token, (token,)))
    return tuple(out)


def _net_multiplier(
    sequences: Sequence[Sequence[str]],
    programs: Sequence[Program],
    hits: Sequence[tuple[int, int]],
    word_length: int,
    b_old: float,
    b_new: float,
    depth: int,
) -> tuple[float, float]:
    """Mean depth-``d`` witness length after the word, and the search cost.

    ``(b_new/b_old)**d`` holds depth fixed, which is the pessimistic half of the
    trade: the word also shortens the programs that use it.  For the depth-``d``
    tasks the honest comparison is ``b_new**d_new / b_old**d``, where ``d_new``
    is their mean length once the word is available.
    """
    saved: Counter[int] = Counter()
    for index, _ in hits:
        saved[index] += word_length - 1
    lengths = [
        len(sequences[i]) - saved.get(i, 0)
        for i, program in enumerate(programs)
        if program.certified_distance == depth
    ]
    if not lengths:
        return float(depth), search_multiplier(b_old, b_new, depth)
    mean_depth = sum(lengths) / len(lengths)
    return mean_depth, math.exp(
        mean_depth * math.log(b_new) - depth * math.log(b_old)
    )


def mine(
    programs: Sequence[Program],
    signatures: dict[str, tuple[str, str]],
    *,
    max_rounds: int = 400,
    min_length: int = 2,
    depth: int = SEARCH_DEPTH,
    verbose: bool = False,
) -> MiningResult:
    """Greedy accept: score, take the best, rewrite, re-run Sequitur, repeat.

    The first round is kept whole so the report can show what independent batch
    scoring would have claimed -- the comparison that makes cannibalisation and
    the moving bar visible.
    """
    sequences: list[tuple[str, ...]] = [program.ops for program in programs]
    total_ops_before = sum(len(sequence) for sequence in sequences)
    defined: dict[str, tuple[str, ...]] = {}
    signature_of = dict(signatures)
    basis_size = len(signatures)
    basis_size_before = basis_size
    model = BranchingModel.from_basis(signatures, programs)

    accepted: list[Candidate] = []
    first_round: list[Candidate] = []
    first_round_rules = 0

    for round_index in range(max_rounds):
        total_ops = sum(len(sequence) for sequence in sequences)
        words = sequitur_words(sequences, min_length=min_length)
        if round_index == 0:
            first_round_rules = len(words)
        if not words:
            break
        hits_by_word = count_occurrences(sequences, words)
        bar = compression_bar(basis_size)
        b_old = model.b_eff()
        growth_old = model.growth_rate()

        scored: list[Candidate] = []
        for word in words:
            hits = hits_by_word[word]
            occurrences = len(hits)
            length = len(word)
            if occurrences < 2:
                continue
            delta = mdl_delta_bits(
                total_ops=total_ops,
                basis_size=basis_size,
                occurrences=occurrences,
                word_length=length,
            )
            if delta <= 0:
                continue
            primitives = _expand_word(word, defined)
            signature = (signature_of[word[0]][0], signature_of[word[-1]][1])
            b_new = model.b_eff(added=signature)
            growth_new = model.growth_rate(added=signature)
            ops_removed = occurrences * (length - 1)
            mean_depth, net_multiplier = _net_multiplier(
                sequences, programs, hits, length, b_old, b_new, depth
            )
            scored.append(
                Candidate(
                    round_index=round_index,
                    name=f"w{len(accepted) + 1:03d}",
                    word=word,
                    primitives=primitives,
                    length=length,
                    primitive_length=len(primitives),
                    occurrences=occurrences,
                    ops_removed=ops_removed,
                    removed_fraction=ops_removed / total_ops,
                    mdl_delta=delta,
                    total_ops_before=total_ops,
                    basis_size_before=basis_size,
                    bar=bar,
                    b_old=b_old,
                    b_new=b_new,
                    multiplier=search_multiplier(b_old, b_new, depth),
                    growth_old=growth_old,
                    growth_new=growth_new,
                    growth_multiplier=(growth_new / growth_old) ** depth,
                    mean_depth=mean_depth,
                    net_multiplier=net_multiplier,
                    signature=signature,
                    diagnostics=diagnose(word, hits, sequences, programs, primitives),
                )
            )

        if round_index == 0:
            first_round = sorted(scored, key=lambda c: -c.mdl_delta)
        if not scored:
            break

        best = max(scored, key=lambda c: c.mdl_delta)
        accepted.append(best)
        sequences = rewrite(sequences, best.word, best.name, hits_by_word[best.word])
        defined[best.name] = best.primitives
        signature_of[best.name] = best.signature
        model.accept(best.signature)
        basis_size += 1
        if verbose:
            body = " ".join(best.primitives)
            print(
                f"round {round_index:3d}  {best.name}  {body:58.58s} "
                f"F={best.occurrences:5d} dMDL={best.mdl_delta:9.0f} "
                f"b={best.b_new:.4f}",
                flush=True,
            )

    return MiningResult(
        accepted=accepted,
        first_round=first_round,
        first_round_rules=first_round_rules,
        total_ops_before=total_ops_before,
        total_ops_after=sum(len(sequence) for sequence in sequences),
        basis_size_before=basis_size_before,
        model=model,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_TABLE_HEADER = (
    "| # | candidate | L | occurrences | ops cut | dMDL (bits) | B after "
    "| b_eff after | d b_eff | search x at d=6 | flags |"
)
_TABLE_RULE = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"


def render_table(candidates: Sequence[Candidate], limit: int | None = None) -> str:
    rows = list(candidates)[: limit if limit is not None else len(candidates)]
    lines = [_TABLE_HEADER, _TABLE_RULE]
    for index, candidate in enumerate(rows, start=1):
        flags = ", ".join(candidate.diagnostics.flags)
        body = " ".join(candidate.primitives)
        lines.append(
            f"| {index} | `{body}` "
            f"| {candidate.primitive_length} | {candidate.occurrences:,} "
            f"| {candidate.ops_removed:,} | {candidate.mdl_delta:,.0f} "
            f"| {candidate.basis_size_before + 1} | {candidate.b_new:.3f} "
            f"| +{candidate.b_delta:.3f} | {candidate.multiplier:.3f} | {flags} |"
        )
    return "\n".join(lines)


def cost_curve(result: MiningResult, depth: int = SEARCH_DEPTH) -> list[dict[str, Any]]:
    """Cumulative search cost after each acceptance, relative to the bare basis.

    A word buys shorter programs and sells more branching, so the honest
    accounting is ``b_k ** d_k`` against ``b_0 ** d``, where ``d_k`` is the mean
    length of the depth-``d`` witnesses once the first ``k`` words exist.  If
    the trade has an interior optimum, it shows up here as a minimum.
    """
    if not result.accepted:
        return []
    b_start = result.accepted[0].b_old
    rows = [
        {
            "words": 0,
            "b_eff": b_start,
            "mean_depth": float(depth),
            "cost_ratio": 1.0,
            "ops_removed_fraction": 0.0,
        }
    ]
    removed = 0
    for index, candidate in enumerate(result.accepted, start=1):
        removed += candidate.ops_removed
        rows.append(
            {
                "words": index,
                "b_eff": candidate.b_new,
                "mean_depth": candidate.mean_depth,
                "cost_ratio": math.exp(
                    candidate.mean_depth * math.log(candidate.b_new)
                    - depth * math.log(b_start)
                ),
                "ops_removed_fraction": removed / result.total_ops_before,
            }
        )
    return rows


def summarise(result: MiningResult, programs: Sequence[Program]) -> dict[str, Any]:
    b_start = result.accepted[0].b_old if result.accepted else result.model.b_eff()
    b_end = result.model.b_eff()
    growth_start = (
        result.accepted[0].growth_old if result.accepted else result.model.growth_rate()
    )
    growth_end = result.model.growth_rate()
    flagged = [c for c in result.accepted if c.diagnostics.flags]
    curve = cost_curve(result)
    cheapest = min(curve, key=lambda row: row["cost_ratio"]) if curve else None
    return {
        "programs": len(programs),
        "corpus_ops": result.total_ops_before,
        "basis_set_id": BASIS_SET_ID,
        "basis_size": result.basis_size_before,
        "mdl_bar_fraction": compression_bar(result.basis_size_before),
        "sequitur_rules_round_1": result.first_round_rules,
        "round_1_clearing_mdl": len(result.first_round),
        "accepted": len(result.accepted),
        "accepted_flagged": len(flagged),
        "corpus_ops_after": result.total_ops_after,
        "ops_removed_fraction": 1 - result.total_ops_after / result.total_ops_before,
        "basis_size_after": result.basis_size_before + len(result.accepted),
        "b_eff_before": b_start,
        "b_eff_after": b_end,
        "search_multiplier_d6": search_multiplier(b_start, b_end),
        "growth_rate_before": growth_start,
        "growth_rate_after": growth_end,
        "growth_multiplier_d6": (growth_end / growth_start) ** SEARCH_DEPTH,
        "cost_curve": curve,
        "cheapest_search_at_words": cheapest["words"] if cheapest else 0,
        "cheapest_search_cost_ratio": cheapest["cost_ratio"] if cheapest else 1.0,
        "state_type_distribution": dict(result.model.state_types.most_common()),
        "in_degree_by_type": dict(result.model.in_degree.most_common()),
    }


def write_artifacts(
    report_dir: str, result: MiningResult, programs: Sequence[Program]
) -> None:
    os.makedirs(report_dir, exist_ok=True)
    summary = summarise(result, programs)
    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, sort_keys=True)
        out.write("\n")
    payload = {
        "first_round": [c.as_row() for c in result.first_round],
        "accepted": [c.as_row() for c in result.accepted],
    }
    candidates_path = os.path.join(report_dir, "candidates.json")
    with open(candidates_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")
    tables = [
        "## Round 1, ranked by MDL delta, scored independently",
        "",
        render_table(result.first_round, limit=40),
        "",
        "## Greedily accepted, in order",
        "",
        render_table(result.accepted),
        "",
    ]
    with open(os.path.join(report_dir, "tables.md"), "w", encoding="utf-8") as out:
        out.write("\n".join(tables))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def basis_signatures(basis_set_id: str = BASIS_SET_ID) -> dict[str, tuple[str, str]]:
    from wandering_light.basis_set import load_basis_set

    functions = load_basis_set(basis_set_id).to_function_def_set()
    return {fn.name: (fn.input_type, fn.output_type) for fn in functions}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine and price candidate basis words from a solved corpus."
    )
    parser.add_argument("--corpus-dir", default=CORPUS_DIR)
    parser.add_argument("--basis-set-id", default=BASIS_SET_ID)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--max-rounds", type=int, default=400)
    parser.add_argument("--depth", type=int, default=SEARCH_DEPTH)
    args = parser.parse_args(argv)

    programs = load_programs(args.corpus_dir)
    signatures = basis_signatures(args.basis_set_id)
    result = mine(
        programs,
        signatures,
        max_rounds=args.max_rounds,
        depth=args.depth,
        verbose=True,
    )
    print(json.dumps(summarise(result, programs), indent=2, sort_keys=True))
    if args.report_dir:
        write_artifacts(args.report_dir, result, programs)
        print(f"wrote artifacts to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
