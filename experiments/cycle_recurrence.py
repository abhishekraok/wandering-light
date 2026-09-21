"""mine-v4-library (#148): how often do whole cycles recur across the staged v4 witnesses?

Counts `cycle_deficit.cycle_spans` cycles over `experiments/staged_witnesses/v4.jsonl`
and the length compression from the top-K as single actions, in sample and held
out 5-fold by behaviour hash. No corpus, no policy.

    uv run python experiments/cycle_recurrence.py --out reports/cycle-deficit-20260920
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.cycle_deficit import cycle_spans, library  # noqa: E402
from experiments.staged_arms import SIDE_FILE  # noqa: E402

FOLDS = 5


def fold_of(behaviour: str) -> int:
    return int(hashlib.sha256(behaviour.encode("utf-8")).hexdigest(), 16) % FOLDS


def compressed(program: tuple[str, ...], lib: set[tuple[str, ...]]) -> int:
    return len(program) - sum(
        b - a - 1 for a, b in cycle_spans(program) if program[a:b] in lib
    )


def measure(programs: dict[str, tuple[str, ...]], sizes: tuple[int, ...]) -> dict:
    total = sum(len(p) for p in programs.values())
    count: Counter[tuple[str, ...]] = Counter()
    where: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for behaviour, program in programs.items():
        for a, b in cycle_spans(program):
            count[program[a:b]] += 1
            where[program[a:b]].add(behaviour)
    shared = {c: n for c, n in count.items() if len(where[c]) >= 2}
    out = {
        "witnesses": len(programs),
        "ops": total,
        "cycles": sum(count.values()),
        "distinct": len(count),
        "shared": len(shared),
        "shared_share_of_ops": sum(len(c) * n for c, n in shared.items()) / total,
        "top": [
            {"cycle": " ".join(c), "uses": count[c], "behaviours": len(where[c])}
            for c in library(list(programs.values()), 15)
        ],
        "compression": {},
    }
    for size in sizes:
        lib = set(library(list(programs.values()), size))
        held = 0
        for fold in range(FOLDS):
            fit = [p for b, p in programs.items() if fold_of(b) != fold]
            fold_lib = set(library(fit, size))
            held += sum(
                compressed(p, fold_lib)
                for b, p in programs.items()
                if fold_of(b) == fold
            )
        out["compression"][size] = {
            "in_sample": total / sum(compressed(p, lib) for p in programs.values()),
            "held_out": total / held,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in SIDE_FILE.read_text().splitlines()]
    staged = [r for r in rows if r["verdict"] == "staged"]
    honest = [
        r for r in staged if not any(f.startswith("contrived") for f in r["flags"])
    ]
    result = {
        name: measure(
            {r["behaviour"]: tuple(r["program"].split()) for r in kept}, (20, 35, 50)
        )
        for name, kept in (("all_staged", staged), ("without_contrived", honest))
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "recurrence.json").write_text(json.dumps(result, indent=1))
    for name, cell in result.items():
        print(name, {k: v for k, v in cell.items() if k != "top"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
