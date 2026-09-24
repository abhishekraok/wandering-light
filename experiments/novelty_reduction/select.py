"""novelty-reduction (#171): selector v1 for a calibration phase.

Of ``size`` picks, from archived pairs within the curriculum's ``max_cuts``:

- ``reattempt``: 10% from pairs already attempted (cycle 2 on), so measured
  progress (the change in |V_hat - V_emp| on the same pair) exists;
- ``progress``: half the rest, the largest *predicted* drop in E between the
  last two calibration phases, among tracked pairs E was predicted on in both;
  with fewer than two phases behind it the share goes to the novelty rule;
- ``novel``: the rest, uniform over pairs whose target's count novelty is
  above ``novel_threshold``; ``novel_fill`` tops up by lowest v(t) when too
  few are novel.

Every pick carries the rule that chose it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

    from experiments.novelty_reduction.archive import LoopArchive

REATTEMPT = "reattempt"
PROGRESS = "progress"
NOVEL = "novel"
NOVEL_FILL = "novel_fill"


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
    e_history: dict[int, dict[tuple[int, int], float]],
    rng: random.Random,
    config: Selector = SELECTOR,
) -> list[tuple[tuple[int, int], str]]:
    candidates = sorted(
        key for key, record in archive.pairs.items() if record.d_cuts <= config.max_cuts
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
    previous, latest = e_history.get(cycle - 2), e_history.get(cycle - 1)
    if previous is not None and latest is not None:
        both = [k for k in latest if k in previous and k not in taken]
        both.sort(key=lambda k: (-(previous[k] - latest[k]), k))
        take(both[: rest // 2], PROGRESS)
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
