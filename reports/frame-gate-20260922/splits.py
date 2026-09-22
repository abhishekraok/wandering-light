"""The E2 gate's ratios on subsets of ops: depth, clipping, one witness (#160).

Every ratio is of calibrated deficit totals over the subset, arm over arm in the
same (seed, epoch) cell, median [min, max] over the 18 cells. ``quotients`` are
per-cell ratios of two frames/world ratios, e.g. defined bodies over top-level
ops. An op is *clipped* in an encoding when any shown example's state, in that
encoding, is wider than ``STATE_CAP`` tokens; *levels* is the number of open
levels. Reads the per-op rows under ``checkpoints/frame-arm-20260922/measure``,
which are too large to commit; writes ``splits.json``.

    uv run python reports/frame-gate-20260922/splits.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.frame_gate import _token_starts, eval_witnesses
from experiments.frame_rows import example_walks, frame_state
from wandering_light.stack_policy import STATE_CAP

MEASURE = Path("checkpoints/frame-arm-20260922/measure")
ARMS = ("world", "tags", "frames")
CELLS = [(s, e) for s in range(3) for e in range(1, 7)]
# The v4 witness whose nested min/max idiom sits under dip-dip(-dip)-fold-ifte
# chains that are rare in fit (144 and 36 steps); a fifth of the world arm's v4
# defined total.
OUTLIER = "bounds_cover_list_flag"


def _defined(op):
    return op["cls"] == "defined"


SUBSETS = {
    "defined": _defined,
    "defined, not fold": lambda op: _defined(op) and op["tag"] != "fold",
    f"defined, without {OUTLIER}": lambda op: _defined(op) and op["name"] != OUTLIER,
    "defined, 1 level": lambda op: _defined(op) and op["levels"] == 1,
    "defined, 2+ levels": lambda op: _defined(op) and op["levels"] >= 2,
    "defined, 1 level, not fold": lambda op: (
        _defined(op) and op["levels"] == 1 and op["tag"] != "fold"
    ),
    "defined, 2+ levels, not fold": lambda op: (
        _defined(op) and op["levels"] >= 2 and op["tag"] != "fold"
    ),
    f"defined, 1 level, without {OUTLIER}": lambda op: (
        _defined(op) and op["levels"] == 1 and op["name"] != OUTLIER
    ),
    f"defined, 2+ levels, without {OUTLIER}": lambda op: (
        _defined(op) and op["levels"] >= 2 and op["name"] != OUTLIER
    ),
    "defined, tags clipped": lambda op: _defined(op) and op["tags_clip"],
    "defined, tags unclipped": lambda op: _defined(op) and not op["tags_clip"],
    "defined, 1 level, tags unclipped": lambda op: (
        _defined(op) and op["levels"] == 1 and not op["tags_clip"]
    ),
    "defined, 2+ levels, tags unclipped": lambda op: (
        _defined(op) and op["levels"] >= 2 and not op["tags_clip"]
    ),
    "reencoded": lambda op: op["cls"] == "reencoded",
    "reencoded, tags unclipped": lambda op: (
        op["cls"] == "reencoded" and not op["tags_clip"]
    ),
    "untouched": lambda op: op["cls"] == "untouched",
    "top": lambda op: op["cls"] == "top",
    "top, direct": lambda op: op["cls"] == "top" and op["kind"] == "direct",
    "top, open and run": lambda op: op["cls"] == "top" and op["kind"] != "direct",
    "all body": lambda op: op["cls"] != "top",
}
for _consumer in ("dip", "filter", "fold", "ifte", "map"):
    SUBSETS[f"defined {_consumer}, without {OUTLIER}"] = lambda op, c=_consumer: (
        _defined(op) and op["tag"] == c and op["name"] != OUTLIER
    )
QUOTIENTS = {
    "defined / top": ("defined", "top"),
    "defined / top direct": ("defined", "top, direct"),
    "defined / untouched": ("defined", "untouched"),
    f"defined / top, without {OUTLIER}": (f"defined, without {OUTLIER}", "top"),
}


def step_flags(corpus: str) -> dict[str, list[dict]]:
    """Per witness, per step: open levels, and whether tags / frames clip."""
    out = {}
    for task in eval_witnesses(corpus):
        walks = example_walks(task)
        flags = []
        for step in range(len(task.program)):
            per = {"levels": 0, "tags_clip": False, "frames_clip": False}
            for states, views in walks:
                view = views.get(step, ())
                per["levels"] = max(per["levels"], len(view))
                for encoding in ("tags", "frames"):
                    shown = frame_state(states[step], view, encoding)
                    if _token_starts(shown)[1] > STATE_CAP:
                        per[f"{encoding}_clip"] = True
            flags.append(per)
        out[task.behaviour] = flags
    return out


def cell_totals(corpus, flags, arm, seed, epoch) -> tuple[dict, dict]:
    cell = json.loads((MEASURE / f"{arm}-s{seed}-e{epoch}.json").read_text())["cells"]
    t = cell["calibration"]["temperature"]
    totals = dict.fromkeys(SUBSETS, 0.0)
    ops = dict.fromkeys(SUBSETS, 0)
    for row in cell[corpus]["machine_mask"]["rows"]:
        for step, (d, cls, kind, tag) in enumerate(
            zip(row["deficits"], row["classes"], row["kinds"], row["tags"], strict=True)
        ):
            if d is None:
                continue
            op = {
                **flags[row["behaviour"]][step],
                "cls": cls,
                "kind": kind,
                "tag": tag,
                "name": row["behaviour"],
            }
            for name, keep in SUBSETS.items():
                if keep(op):
                    totals[name] += d / t
                    ops[name] += 1
    return totals, ops


def summary(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [
        round(statistics.median(values), 3),
        round(min(values), 2),
        round(max(values), 2),
    ]


def split(corpus: str) -> dict:
    flags = step_flags(corpus)
    per = {
        arm: [cell_totals(corpus, flags, arm, s, e)[0] for s, e in CELLS]
        for arm in ARMS
    }
    ops = cell_totals(corpus, flags, "world", *CELLS[0])[1]

    def ratios(num, den, name):
        return [
            a[name] / b[name]
            for a, b in zip(per[num], per[den], strict=True)
            if b[name] > 0
        ]

    subsets = {
        name: {
            "ops": ops[name],
            **{
                f"{num}/{den}": summary(ratios(num, den, name))
                for num, den in (
                    ("frames", "world"),
                    ("tags", "world"),
                    ("frames", "tags"),
                )
            },
        }
        for name in SUBSETS
    }
    quotients = {}
    for label, (top, bottom) in QUOTIENTS.items():
        for num in ("frames", "tags"):
            q = [
                (a[top] / b[top]) / (a[bottom] / b[bottom])
                for a, b in zip(per[num], per["world"], strict=True)
                if b[top] > 0 and b[bottom] > 0 and a[bottom] > 0
            ]
            quotients[f"{num}: {label}"] = summary(q)
    return {"subsets": subsets, "quotients": quotients}


def main() -> None:
    out = {corpus: split(corpus) for corpus in ("stack_pbe_v4", "stack_pbe_world_v3")}
    Path(__file__).with_name("splits.json").write_text(json.dumps(out, indent=1))
    for corpus, result in out.items():
        print(corpus)
        for name, entry in result["subsets"].items():
            print(f"  {name:48s} {json.dumps(entry)}")
        for name, entry in result["quotients"].items():
            print(f"  {name:48s} {entry}")


if __name__ == "__main__":
    main()
