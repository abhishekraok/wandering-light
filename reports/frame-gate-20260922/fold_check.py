"""Which witnesses carry the defined `fold` body deficit, per arm (#160 E2).

Reads the per-op rows under ``checkpoints/frame-arm-20260922/measure``, which
are too large to commit. Medians over the 18 cells of each arm, calibrated.

    uv run python reports/frame-gate-20260922/fold_check.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

MEASURE = Path("checkpoints/frame-arm-20260922/measure")
ARMS = ("world", "tags", "frames")
CELLS = [(s, e) for s in range(3) for e in range(1, 7)]


def fold_by_witness(corpus: str) -> dict[str, dict[str, list[float]]]:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for arm in ARMS:
        for seed, epoch in CELLS:
            path = MEASURE / f"{arm}-s{seed}-e{epoch}.json"
            cell = json.loads(path.read_text())["cells"]
            t = cell["calibration"]["temperature"]
            for row in cell[corpus]["machine_mask"]["rows"]:
                steps = [
                    d
                    for d, cls, tag in zip(
                        row["deficits"], row["classes"], row["tags"], strict=True
                    )
                    if cls == "defined" and tag == "fold"
                ]
                if steps:
                    per[row["behaviour"]]["ops"] = [len(steps)]
                    per[row["behaviour"]][arm].append(sum(steps) / t)
    return per


def main() -> None:
    for corpus in ("stack_pbe_v4", "stack_pbe_world_v3"):
        per = fold_by_witness(corpus)
        rows = sorted(
            (
                statistics.median(d["tags"]) - statistics.median(d["world"]),
                name,
                d,
            )
            for name, d in per.items()
        )[::-1]
        print(f"{corpus}: {len(per)} witnesses with defined fold ops")
        for excess, name, d in rows[:8]:
            medians = "  ".join(f"{a} {statistics.median(d[a]):6.1f}" for a in ARMS)
            print(
                f"  {name:36s} ops {d['ops'][0]:3d}  {medians}  tags-world {excess:+6.1f}"
            )
        print(f"  summed tags-world excess {sum(r[0] for r in rows):+.1f}")
        for arm in ("tags", "frames"):
            rest = [
                abs(statistics.median(d[arm]) - statistics.median(d["world"]))
                for _excess, name, d in rows[1:]
            ]
            print(f"  largest |{arm}-world| on the other witnesses {max(rest):.2f}")


if __name__ == "__main__":
    main()
