"""novelty-reduction (#171): selector v1 for a calibration phase.

Of ``size`` picks, from archived pairs within the curriculum's ``max_cuts``:

- ``reattempt``: 10% from pairs already attempted (cycle 2 on), so measured
  progress (the change in |V_hat - V_emp| on the same pair) exists;
- ``novel``: the rest, uniform over pairs whose target's count novelty is
  above ``novel_threshold``; ``novel_fill`` tops up by lowest v(t) when too
  few are novel.

The predicted-progress ranking (largest drop in predicted E between the last
two calibration phases) is computed and logged, never used to collect: a
falling E is not demonstrated progress (#171 F11). Every pick carries the
rule that chose it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

    from experiments.novelty_reduction.archive import LoopArchive

REATTEMPT = "reattempt"
NOVEL = "novel"
NOVEL_FILL = "novel_fill"
E_RANKED = "e_ranked"


@dataclass(frozen=True, slots=True)
class Selector:
    size: int = 1000
    reattempt: float = 0.10
    novel_threshold: float = 0.5  # N(t) > 0.5 is v(t) < 3
    max_cuts: int = 1


SELECTOR = Selector()


def select(
    archive: LoopArchive,
    cycle: int,
    rng: random.Random,
    config: Selector = SELECTOR,
) -> list[tuple[tuple[int, int], str]]:
    """Eligibility is on the witness d_cuts, which relaxation never changes."""
    candidates = sorted(
        key
        for key, record in archive.pairs.items()
        if record.witness_d_cuts <= config.max_cuts
    )
    picked: list[tuple[tuple[int, int], str]] = []
    taken: set[tuple[int, int]] = set()

    def take(keys, rule):
        for key in keys:
            if key not in taken:
                taken.add(key)
                picked.append((key, rule))

    attempted = [key for key in candidates if archive.pairs[key].attempts > 0]
    if cycle >= 2 and attempted:
        count = min(round(config.size * config.reattempt), len(attempted))
        take(rng.sample(attempted, count), REATTEMPT)
    rest = config.size - len(picked)
    novel = [
        key
        for key in candidates
        if key not in taken and archive.novelty(key[1]) > config.novel_threshold
    ]
    take(rng.sample(novel, min(rest, len(novel))), NOVEL)
    rest = config.size - len(picked)
    if rest > 0:
        left = [key for key in candidates if key not in taken]
        left.sort(key=lambda k: (archive.visits[k[1]], k))
        take(left[:rest], NOVEL_FILL)
    return picked


def select_ranked(
    archive: LoopArchive,
    cycle: int,
    rng: random.Random,
    ranked: dict[tuple[int, int], float],
    config: Selector = SELECTOR,
) -> list[tuple[tuple[int, int], str]]:
    """M3's E-ranked arm: the same re-attempt share, then highest E_hat first.

    ``ranked`` holds E_hat for uncertified eligible pairs (the previous cycle's
    E on its own trunk); pairs outside it are reached only by the novel
    rules, if the ranking runs out.
    """
    candidates = sorted(
        key
        for key, record in archive.pairs.items()
        if record.witness_d_cuts <= config.max_cuts
    )
    attempted = [key for key in candidates if archive.pairs[key].attempts > 0]
    taken: set[tuple[int, int]] = set()
    picked = []
    if cycle >= 2 and attempted:
        count = min(round(config.size * config.reattempt), len(attempted))
        for key in rng.sample(attempted, count):
            taken.add(key)
            picked.append((key, REATTEMPT))
    order = sorted(
        (k for k in ranked if k not in taken and k in archive.pairs),
        key=lambda k: (-ranked[k], k),
    )
    for key in order[: config.size - len(picked)]:
        taken.add(key)
        picked.append((key, E_RANKED))
    rest = config.size - len(picked)
    if rest > 0:  # the ranking ran out: the uniform arm's novel rules fill
        fill = select(
            archive,
            0,  # no re-attempts in the fill
            rng,
            Selector(len(candidates), 0.0, config.novel_threshold, config.max_cuts),
        )
        picked += [(k, rule) for k, rule in fill if k not in taken][:rest]
    return picked


def predicted_progress(
    e_history: dict[int, dict[tuple[int, int], float]], cycle: int, limit: int
) -> list[dict]:
    """The top ``limit`` pairs by predicted E drop from cycle - 2 to cycle - 1; logged only."""
    previous, latest = e_history.get(cycle - 2), e_history.get(cycle - 1)
    if previous is None or latest is None:
        return []
    both = sorted(
        (k for k in latest if k in previous),
        key=lambda k: (-(previous[k] - latest[k]), k),
    )
    return [
        {"s": k[0], "t": k[1], "e_drop": previous[k] - latest[k]} for k in both[:limit]
    ]
