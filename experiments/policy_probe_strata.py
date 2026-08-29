"""Cut a depth-probe result by how much of the target the budget showed.

Comparing the encoders probe by probe confounds visibility with which end is
being predicted: the starved probes are front k=0 plus all three back rows, and
the back has no visibility variation at all across depth. Within a probe, end
and mask distribution are held fixed and only visibility varies.

Two cuts, and the second is the one to read. Splitting by how much of the target
the 256-byte budget showed is easy to interpret but needs transfer to be common
across strata -- and the blind stratum is exactly the long-input tasks, which is
where only the wider arm could have learned anything, so that assumption is
least safe precisely where the effect is read. Splitting instead by how much the
wider window *newly reveals* estimates transfer directly in the bin where it
reveals nothing, and the visibility effect is the slope above it.

Effects are differenced **pair by pair across seeds**, never pooled over tasks.
The replicate is the seed: two models that landed in different basins move every
task in a bin together, so seed variation is a shared shock and pooling n tasks
x 3 seeds would treat one observation as thousands.

    uv run python experiments/policy_probe_strata.py <corpus> <depth*.json.gz>
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy_probe as pp
import policy_probe_depth as ppd
from policy_probe import resolve_corpus

STRATA = (
    ("target entirely unseen", lambda v: v == 0.0),
    ("target partly seen", lambda v: 0.0 < v < 1.0),
    ("target fully seen (floor)", lambda v: v == 1.0),
)
# How much target the wider window newly reveals, per task. The zero bin is
# transfer measured directly -- `wide` sees nothing there that `flat` did not --
# so the effect of visibility is the slope above it, and no assumption about
# transfer being common across strata is needed.
DOSE_BINS = (
    ("nothing new (anchor)", 0.0, 0.0),
    ("0-25% newly seen", 0.0, 0.25),
    ("25-50%", 0.25, 0.50),
    ("50-75%", 0.50, 0.75),
    ("75-100%", 0.75, 1.0),
)


def peeled_test(corpus: Path, depth: int):
    """The same test split the probe scored, in the same order."""
    from wandering_light.basis_set import load_basis_set
    from wandering_light.executor import Executor

    fs = load_basis_set("wl-core-v1").to_function_def_set()
    by_name = {f.name: f for f in fs.functions}
    return ppd.peel(corpus, "test", by_name, Executor(fs), depth, 3)


def spread(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var**0.5


def arm_series(rows, probe, idx):
    """Per-seed mean rank and top-1 on `idx`, one entry per seed, arm by arm.

    Kept as a series rather than collapsed, because the replicate here is the
    seed and not the task. Two models that landed in different basins move every
    task in a bin together, so seed variation is a shared shock, not independent
    noise -- pooling 853 tasks x 3 seeds as n = 2,559 would understate the
    uncertainty by treating one observation as thousands.
    """
    out = {}
    for arm in ("flat", "wide", "pairs"):
        got = sorted(
            (
                r
                for r in rows
                if r["probe"] == probe and r["predictor"] == f"learned ({arm})"
            ),
            key=lambda r: r["seed"],
        )
        out[arm] = (
            [sum(r["ranks"][i] for i in idx) / len(idx) for r in got],
            [sum(r["ranks"][i] == 1 for i in idx) / len(idx) for r in got],
        )
    return out


def gap(series, a, b):
    """`b` minus `a`, paired seed by seed: three estimates, not one pool.

    Seed `i` of both arms starts from the same initialisation -- identical
    architecture, same `torch.manual_seed` -- so the pairing removes the shared
    draw and what is left is the effect. The spread across the three pairs is
    the model-level uncertainty, and it is the one that decides whether an
    effect is real; a sign disagreement among the three is itself the finding.
    """
    rank = [x - y for x, y in zip(series[a][0], series[b][0], strict=True)]
    top1 = [y - x for x, y in zip(series[a][1], series[b][1], strict=True)]
    mr, sr = spread(rank)
    mt, st = spread(top1)
    signs = "" if len({v > 0 for v in rank}) == 1 else " [signs differ]"
    flag = "" if abs(mr) > 2 * sr else " (ns)"
    return f"{mr:+.2f} ±{sr:.2f}{flag} / {mt:+.1%} ±{st:.1%}{signs}"


def show(series, a):
    (mr, sr), (mt, st) = spread(series[a][0]), spread(series[a][1])
    return f"{mr:.2f} ±{sr:.2f} / {mt:.1%} ±{st:.1%}"


def read(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def main():
    argv = sys.argv[1:]
    corpus = resolve_corpus(["", argv[0]])
    rows = [r for path in argv[1:] for r in read(path)["rows"]]

    for depth in (0, 1, 2):
        front, back = peeled_test(corpus, depth)
        for end, tasks in (("front", front), ("back", back)):
            probe = f"{end} k={depth}"
            if not any(r["probe"] == probe for r in rows):
                continue
            seen = [pp.visible_target(t, pp.MAX_LEN) for t in tasks]
            new = [
                pp.visible_target(t, pp.WIDE_LEN) - v
                for t, v in zip(tasks, seen, strict=True)
            ]
            print(f"\n### {probe}  ({len(tasks)} test tasks)\n")
            print("Cut by how much of the target the 256-byte budget showed.\n")
            print("| stratum | n | flat | wide | pairs | wide-flat | pairs-wide |")
            print("|---|---|---|---|---|---|---|")
            for name, belongs in STRATA:
                idx = [i for i, v in enumerate(seen) if belongs(v)]
                if not idx:
                    continue
                cell = arm_series(rows, probe, idx)
                print(
                    f"| {name} | {len(idx)} | {show(cell, 'flat')} | {show(cell, 'wide')} | "
                    f"{show(cell, 'pairs')} | {gap(cell, 'flat', 'wide')} | "
                    f"{gap(cell, 'wide', 'pairs')} |"
                )
            print("\nCut by how much the wider window newly reveals. The anchor bin")
            print("is transfer alone; visibility is the slope above it.\n")
            print("| newly revealed | n | flat | wide | wide-flat |")
            print("|---|---|---|---|---|")
            for name, lo, hi in DOSE_BINS:
                if lo == hi == 0.0:
                    idx = [i for i, v in enumerate(new) if v == 0.0]
                else:
                    idx = [i for i, v in enumerate(new) if lo < v <= hi]
                if not idx:
                    continue
                cell = arm_series(rows, probe, idx)
                print(
                    f"| {name} | {len(idx)} | {show(cell, 'flat')} | "
                    f"{show(cell, 'wide')} | {gap(cell, 'flat', 'wide')} |"
                )


if __name__ == "__main__":
    main()
