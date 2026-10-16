"""Shown-pair count as a natural experiment on memorisation.

`stack_pbe.generate` draws each task's shown-pair count as
``rng.randint(min_examples, max_examples)`` (3 to 6 on v4), independently of
the behaviour, and always holds out 8 pairs. So the shown count is randomised
within every behaviour, and a solver's verdicts can be split by it: a
memorisation fits every shown pair and fails the held-out ones, and more shown
pairs leave fewer wrong programs that fit.

Reads a solver's eval candidates (default: astra low over the full v4 eval
split, the #123 phase-2 shards) against the eval payload pinned by the corpus
manifest, fetched and checked against its digest. Prints verdicts by shown
count, then a within-behaviour comparison (3-4 against 5-6 shown) with an
exact two-sided sign test, and a permutation check that the shown count is
independent of behaviour.

    uv run python experiments/shown_count_memorisation.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wandering_light.corpus_hub import corpus_split_files, fetch_corpus  # noqa: E402

MANIFEST = REPO / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
REPORT = REPO / "reports/stack-pbe-v4-codex-exec-20260910"
PATTERN = "astra_low_v4_full_shard*.json"


def eval_rows(manifest: Path) -> list[dict]:
    directory = fetch_corpus(manifest, splits=["eval"], include_details=False)
    meta = json.loads((directory / "manifest.json").read_text())
    [(_, name)] = corpus_split_files(meta, splits=["eval"])
    with gzip.open(directory / name, "rt") as handle:
        return [json.loads(line) for line in handle]


def candidates(report: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(report.glob(pattern)):
        data = json.loads(path.read_text())
        assert not data.get("partial"), path
        rows += data["candidates"]
    return sorted(rows, key=lambda c: c["task_index"])


def sign_test(up: int, down: int) -> float:
    n, k = up + down, min(up, down)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def chi2(behaviours: list[str], shown: list[int]) -> tuple[float, int]:
    table = Counter(zip(behaviours, shown, strict=True))
    rows, cols, n = Counter(behaviours), Counter(shown), len(shown)
    stat = sum(
        (table[b, k] - rows[b] * cols[k] / n) ** 2 / (rows[b] * cols[k] / n)
        for b in rows
        for k in cols
    )
    return stat, (len(rows) - 1) * (len(cols) - 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--pattern", default=PATTERN)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args(argv)

    rows = eval_rows(args.manifest)
    cands = candidates(args.report, args.pattern)
    assert [c["task_index"] for c in cands] == list(range(len(rows)))
    assert all(rows[c["task_index"]]["behaviour"] == c["behaviour"] for c in cands)
    shown = [rows[c["task_index"]]["examples"] for c in cands]
    behaviours = [c["behaviour"] for c in cands]

    print(f"{len(cands)} tasks, {len(set(behaviours))} behaviours, {args.pattern}\n")
    print(
        "| shown pairs | tasks | solved | memorised | failed | memorised of shown fits |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    by_k: dict[int, Counter] = defaultdict(Counter)
    for c, k in zip(cands, shown, strict=True):
        by_k[k][c["verdict"]] += 1
    for k in sorted(by_k):
        v, n = by_k[k], sum(by_k[k].values())
        fit = v["solved"] + v["memorised"]
        print(
            f"| {k} | {n} | {100 * v['solved'] / n:.1f}% | "
            f"{100 * v['memorised'] / n:.1f}% | {100 * v['failed'] / n:.1f}% | "
            f"{100 * v['memorised'] / fit:.1f}% |"
        )

    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"lo": [], "hi": []})
    for c, k in zip(cands, shown, strict=True):
        groups[c["behaviour"]]["lo" if k <= 4 else "hi"].append(c["verdict"])
    print("\nWithin behaviour, rate at 5-6 shown against 3-4 shown:\n")
    for verdict in ("memorised", "solved"):
        up = down = 0
        for g in groups.values():
            if not g["lo"] or not g["hi"]:
                continue
            lo = g["lo"].count(verdict) / len(g["lo"])
            hi = g["hi"].count(verdict) / len(g["hi"])
            up += hi > lo
            down += hi < lo
        print(
            f"- {verdict}: higher in {up} behaviours, lower in {down}, "
            f"two-sided sign test p = {sign_test(up, down):.1g}"
        )

    stat, df = chi2(behaviours, shown)
    rng = random.Random(0)
    permuted, at_least = list(shown), 0
    for _ in range(args.permutations):
        rng.shuffle(permuted)
        at_least += chi2(behaviours, permuted)[0] >= stat
    print(
        f"\nShown count against behaviour: chi2 {stat:.1f} on {df} df, "
        f"permutation p = {(at_least + 1) / (args.permutations + 1):.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
