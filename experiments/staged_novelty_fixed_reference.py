"""Staged-arms (#155) novelty recounted against fixed references.

staged-arms counted a verified behaviour as novel when one of its verified
solving programs occurs nowhere in *that arm's own* training text
(`experiments/distinct_programs/containment.py`). The reference then moves
with the arm: a program both arms write counts as novel for fused and as
retrieved for staged. This recounts every arm against fixed texts too: the
fused (v4 witness) text, and the union of the fused, re-fused and staged
texts. Then it lists behaviours solved by an identical string in fused and
staged, and how many seeds of each arm verify each behaviour that differs.

    uv run python experiments/staged_novelty_fixed_reference.py \\
        --run-dir checkpoints/staged-arms-20260920 \\
        --verdicts reports/staged-arms-20260920/verified_solves.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.staged_arms as sa  # noqa: E402
from experiments.distinct_programs.containment import (  # noqa: E402
    contained,
    train_text,
)

ARMS = ("fused", "refused", "staged", "both")


def solving_programs(run_dir: Path, verdicts: dict) -> dict:
    out: dict[tuple[str, int], dict[str, set[str]]] = {}
    for path in sorted((run_dir / "timing").glob("*.json")):
        cell = json.loads(path.read_text())
        policy = f"{cell['arm']}-s{cell['seed']}-e{cell['policy_epoch']}"
        manifest = run_dir / "policies" / policy / "training_manifest.json"
        assert not json.loads(manifest.read_text())["keep_contrived"], policy
        solving: dict[str, set[str]] = defaultdict(set)
        for r in cell["rows"]:
            text = " ".join(r["program"] or ())
            if r["solved"] and verdicts.get((r["behaviour"], text)) in sa.VERIFIED:
                solving[r["behaviour"]].add(text)
        out[cell["arm"], cell["seed"]] = solving
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    args = parser.parse_args(argv)
    verdicts = {
        (row["behaviour"], row["program"]): row["verdict"]
        for row in json.loads(args.verdicts.read_text())["programs"]
    }
    solving = solving_programs(args.run_dir, verdicts)
    seeds = sorted({seed for _, seed in solving})
    tasks = co.corpus()
    texts = {arm: train_text(arm, tasks) for arm in ARMS}
    union = texts["fused"] | texts["refused"] | texts["staged"]

    print("| arm | verified | novel vs own text | vs fused text | vs union of texts |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        cells = [solving[arm, s] for s in seeds]
        cols = [[len(c) for c in cells]]
        for text in (texts[arm], texts["fused"], union):
            cols.append(
                [
                    sum(any(not contained(p, text) for p in ps) for ps in c.values())
                    for c in cells
                ]
            )
        print(
            f"| {arm} | "
            + " | ".join(f"{mean(n):.2f} ({' / '.join(map(str, n))})" for n in cols)
            + " |"
        )

    print(
        "\nSolved by an identical string in fused and staged (in fused text / staged text):"
    )
    behaviours = sorted({b for c in solving.values() for b in c})
    for b in behaviours:
        fused = set().union(*(solving["fused", s].get(b, set()) for s in seeds))
        staged = set().union(*(solving["staged", s].get(b, set()) for s in seeds))
        for p in sorted(fused & staged):
            print(
                f"- {b}: `{p}` ({contained(p, texts['fused'])} / "
                f"{contained(p, texts['staged'])})"
            )

    print("\nSeeds verifying each behaviour, where arms differ:\n")
    print("| behaviour | " + " | ".join(ARMS) + " |")
    print("|---|" + "---:|" * len(ARMS))
    freq = {arm: Counter(b for s in seeds for b in solving[arm, s]) for arm in ARMS}
    for b in behaviours:
        row = [freq[arm][b] for arm in ARMS]
        if len(set(row)) > 1:
            print(f"| {b} | " + " | ".join(map(str, row)) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
