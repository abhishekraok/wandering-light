"""Pure synthetic-testable #180 selection and snapshot profiles; no data runner."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SEEDS = (0, 1, 2)
HALVES = ("selection", "confirmation")
TERMINATIONS = ("fit", "max_depth", "budget", "wall_clock_cap", "frontier_exhausted")


def digest(prefix: str, name: str) -> str:
    return hashlib.sha256((prefix + name).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShownTask:
    source_index: int
    behaviour: str
    skeleton: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def group(self) -> str:
        return self.skeleton or self.behaviour


def project_shown(raw: Mapping, source_index: int) -> ShownTask:
    """Read only solver-facing whitelisted fields; never touch hidden values."""
    if raw["split"] != "train":
        raise ValueError("TRAIN projection received another split")
    behaviour, skeleton = raw["behaviour"], raw.get("skeleton", "")
    inputs, outputs = raw["inputs"], raw["outputs"]
    if not isinstance(behaviour, str) or not behaviour:
        raise ValueError("invalid behaviour")
    if not isinstance(skeleton, str):
        raise ValueError("invalid original skeleton")
    for values in (inputs, outputs):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("empty or invalid shown examples")
        if any(not isinstance(value, str) for value in values):
            raise ValueError("shown examples must be text")
    if len(inputs) != len(outputs):
        raise ValueError("unequal shown pairs")
    if type(source_index) is not int or source_index < 0:
        raise ValueError("invalid source index")
    return ShownTask(source_index, behaviour, skeleton, tuple(inputs), tuple(outputs))


@dataclass(frozen=True)
class Selected:
    position: int
    half: str
    task: ShownTask


def select_panel(
    tasks: Iterable[ShownTask], eligible_behaviours: frozenset[str], *, size: int = 48
) -> tuple[Selected, ...]:
    """Small sizes are for synthetic tests; the real design pins size48."""
    if type(size) is not int or size <= 0 or size % 2:
        raise ValueError("panel size must be a positive even integer")
    groups: dict[str, dict[str, ShownTask]] = defaultdict(dict)
    behaviour_groups: dict[str, str] = {}
    source_indices = set()
    for task in tasks:
        if task.source_index in source_indices:
            raise ValueError("duplicate source index")
        source_indices.add(task.source_index)
        previous = behaviour_groups.setdefault(task.behaviour, task.group)
        if previous != task.group:
            raise ValueError("inconsistent original behaviour skeleton")
        validation = (
            int(hashlib.sha256(task.group.encode("utf-8")).hexdigest(), 16) % 10 == 0
        )
        if task.behaviour not in eligible_behaviours or validation:
            continue
        old = groups[task.group].get(task.behaviour)
        if old is None or task.source_index < old.source_index:
            groups[task.group][task.behaviour] = task
    if len(groups) < size:
        raise ValueError("INSUFFICIENT_SELECTION")
    ordered = sorted(groups, key=lambda g: (digest("score-gap-D1-group:", g), g))[:size]
    selected = []
    for position, group in enumerate(ordered):
        task = min(
            groups[group].values(),
            key=lambda t: (digest("score-gap-D1-behaviour:", t.behaviour), t.behaviour),
        )
        selected.append(Selected(position, HALVES[position % 2], task))
    return tuple(selected)


@dataclass(frozen=True)
class Snapshot:
    scores: tuple[float, ...]
    ancestor_rank: int | None = None  # one-based rank in the actual retained C

    def __post_init__(self) -> None:
        if any(isinstance(s, bool) or not isinstance(s, Real) for s in self.scores):
            raise ValueError("invalid recorded score scalar")
        scores = tuple(float(value) for value in self.scores)
        if len(scores) > 64 or any(math.isnan(s) or s == math.inf for s in scores):
            raise ValueError("invalid retained cumulative score")
        if any(left < right for left, right in pairwise(scores)):
            raise ValueError("retained scores are not in stable descending order")
        if self.ancestor_rank is not None and (
            type(self.ancestor_rank) is not int
            or not 1 <= self.ancestor_rank <= len(scores)
        ):
            raise ValueError("invalid ancestor rank")


def band_size(layer: Snapshot, tau: float) -> int:
    if isinstance(tau, bool) or not math.isfinite(tau) or tau < 0:
        raise ValueError("invalid finite band")
    if not layer.scores:
        return 0
    # Python floats are float64. Convert saved cumulative scores only HERE:
    # upstream accumulation must retain its original production dtype/bits.
    anchor = float(layer.scores[0])
    if anchor == -math.inf:
        return len(layer.scores)
    threshold = anchor - float(tau)
    return sum(float(score) >= threshold for score in layer.scores)


@dataclass(frozen=True)
class Cell:
    group: str
    half: str
    seed: int
    termination: str
    layers: tuple[Snapshot, ...]

    def __post_init__(self) -> None:
        if (
            not self.group
            or self.half not in HALVES
            or type(self.seed) is not int
            or self.seed not in SEEDS
        ):
            raise ValueError("invalid cell identity")
        if self.termination not in TERMINATIONS:
            raise ValueError("invalid reference terminal")
        for layer in self.layers:
            if (layer.ancestor_rank is not None) != (self.termination == "fit"):
                raise ValueError("fit ancestry missing or nonfit ancestry invented")

    @property
    def informative(self) -> bool:
        return (
            self.termination == "fit"
            and len(self.layers) >= 2
            and any(len(layer.scores) > 32 for layer in self.layers)
        )


def profile_sizes(cell: Cell, tau: float | None, *, width: int = 32) -> tuple[int, ...]:
    if type(width) is not int or not 1 <= width <= 64:
        raise ValueError("invalid descriptive prefix width")
    return tuple(
        min(width, len(layer.scores)) if tau is None else band_size(layer, tau)
        for layer in cell.layers
    )


def slot_fraction(cell: Cell, sizes: Sequence[int]) -> Fraction:
    if len(sizes) != len(cell.layers) or any(
        type(size) is not int or not 0 <= size <= len(layer.scores)
        for layer, size in zip(cell.layers, sizes, strict=True)
    ):
        raise ValueError("invalid slot profile")
    total = sum(len(layer.scores) for layer in cell.layers)
    return Fraction(sum(sizes), total) if total else Fraction(1)


def protected(cell: Cell, sizes: Sequence[int]) -> bool:
    # Validate shape/counters even for a cell without a labelled continuation.
    slot_fraction(cell, sizes)
    return cell.informative and all(
        layer.ancestor_rank <= size
        for layer, size in zip(cell.layers, sizes, strict=True)
    )


@dataclass(frozen=True)
class SeedStats:
    groups: int
    informative: int
    band_protected: int
    prefix_protected: int
    band_slots: Fraction
    prefix_slots: Fraction

    def __post_init__(self) -> None:
        counts = (
            self.groups,
            self.informative,
            self.band_protected,
            self.prefix_protected,
        )
        if any(type(n) is not int or n < 0 for n in counts) or not self.groups:
            raise ValueError("invalid profile counts")
        if (
            not 0 <= self.band_protected <= self.informative <= self.groups
            or not 0 <= self.prefix_protected <= self.informative
        ):
            raise ValueError("profile counts exceed their denominators")
        if any(
            not isinstance(s, Fraction) or not 0 <= s <= 1
            for s in (self.band_slots, self.prefix_slots)
        ):
            raise ValueError("slot means must be exact bounded fractions")

    @property
    def band_protection(self) -> Fraction | None:
        return (
            Fraction(self.band_protected, self.informative)
            if self.informative
            else None
        )

    @property
    def prefix_protection(self) -> Fraction | None:
        return (
            Fraction(self.prefix_protected, self.informative)
            if self.informative
            else None
        )


def summarize(cells: Sequence[Cell], tau: float) -> SeedStats:
    if not cells or len({cell.group for cell in cells}) != len(cells):
        raise ValueError("empty or duplicate group population")
    if len({(cell.half, cell.seed) for cell in cells}) != 1:
        raise ValueError("mixed half or seed population")
    band = [profile_sizes(cell, tau) for cell in cells]
    prefix = [profile_sizes(cell, None) for cell in cells]
    return SeedStats(
        len(cells),
        sum(cell.informative for cell in cells),
        sum(protected(cell, sizes) for cell, sizes in zip(cells, band, strict=True)),
        sum(protected(cell, sizes) for cell, sizes in zip(cells, prefix, strict=True)),
        sum(
            (
                slot_fraction(cell, sizes)
                for cell, sizes in zip(cells, band, strict=True)
            ),
            Fraction(),
        )
        / len(cells),
        sum(
            (
                slot_fraction(cell, sizes)
                for cell, sizes in zip(cells, prefix, strict=True)
            ),
            Fraction(),
        )
        / len(cells),
    )


def gate_components(stats: SeedStats) -> dict[str, bool]:
    band, prefix = stats.band_protection, stats.prefix_protection
    return {
        "support": stats.informative >= 12,
        "protection": band is not None and band >= Fraction(19, 20),
        "gain": band is not None
        and prefix is not None
        and band - prefix >= Fraction(1, 10),
        "slots": stats.band_slots <= stats.prefix_slots,
    }


def by_seed(cells: Sequence[Cell], half: str) -> dict[int, tuple[Cell, ...]]:
    if half not in HALVES or any(cell.half != half for cell in cells):
        raise ValueError("wrong-half profile access")
    grouped = {
        seed: tuple(cell for cell in cells if cell.seed == seed) for seed in SEEDS
    }
    populations = [{cell.group for cell in rows} for rows in grouped.values()]
    if any(
        len(rows) != 24 or len(groups) != 24
        for rows, groups in zip(grouped.values(), populations, strict=True)
    ):
        raise ValueError("incomplete or duplicate half/seed panel")
    if any(groups != populations[0] for groups in populations[1:]):
        raise ValueError("seed group population differs")
    return grouped


@dataclass(frozen=True)
class Choice:
    terminal: str
    tau: float | None


def choose_selection(cells: Sequence[Cell]) -> Choice:
    """Selection-only pure decision; caller must seal receipt before confirmation."""
    grouped = by_seed(cells, "selection")
    initial = {seed: summarize(rows, GRID[0]) for seed, rows in grouped.items()}
    if any(stats.informative < 12 for stats in initial.values()):
        return Choice("INSUFFICIENT_LINEAGE_SUPPORT", None)
    if any(stats.prefix_protection > Fraction(9, 10) for stats in initial.values()):
        return Choice("NO_HEADROOM_OVER_PREFIX32", None)
    for tau in GRID:
        if all(
            all(gate_components(summarize(rows, tau)).values())
            for rows in grouped.values()
        ):
            return Choice("SELECTION_PROFILE_LOCKED", tau)
    return Choice("NO_SELECTION_PROFILE", None)


def confirm(
    cells: Sequence[Cell], choice: Choice
) -> tuple[str, dict[int, dict[str, bool]]]:
    if (
        choice.terminal != "SELECTION_PROFILE_LOCKED"
        or isinstance(choice.tau, bool)
        or choice.tau not in GRID
    ):
        raise ValueError("no locked selection band")
    grouped = by_seed(cells, "confirmation")
    checks = {
        seed: gate_components(summarize(rows, choice.tau))
        for seed, rows in grouped.items()
    }
    if any(not row["support"] for row in checks.values()):
        return "INSUFFICIENT_LINEAGE_SUPPORT", checks
    terminal = (
        "PASS_PROFILE_FEASIBILITY"
        if all(all(row.values()) for row in checks.values())
        else "NO_CONFIRMATION_PROFILE"
    )
    return terminal, checks
