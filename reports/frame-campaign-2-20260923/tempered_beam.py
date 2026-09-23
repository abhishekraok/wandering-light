"""Does a beam that ranks on calibrated log-probabilities solve more? (#160 campaign 2)

The campaign's clears, like the first gate's, count calibrated deficits: each
step's gap to the best legal op divided by the checkpoint's fitted temperature.
That predicts a beam ranking on ``log_softmax(logits / T)``. The solver ranks on
raw log-probabilities, which segment-composite (#160) counts, so the two
predictors describe two different beams. Path ranking is not
temperature-invariant: the per-state normaliser differs across paths.

This runs `solve.BEST`'s search (beam 64 x depth 48, guard `both`, library on,
state cap 16,384) with the world arm's `1m-s0-e5` on the #101 cohort, once at
T = 1 and once at the checkpoint's fitted T, and counts graded and verified
behaviours with best-system-command (#158)'s driver and verifier.

Predicted before running (the advisor session's prediction, which I share):
little or no change, within one verified behaviour, because softening acts like
widening the tolerance, and the width ladder was flat. If coverage does not
move, the campaign's clears are relative across arms, not coverage forecasts.

    uv run python reports/frame-campaign-2-20260923/tempered_beam.py run --cell t1
    uv run python reports/frame-campaign-2-20260923/tempered_beam.py run --cell tcal
    uv run python reports/frame-campaign-2-20260923/tempered_beam.py report
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import experiments.best_system_cells as bsc  # noqa: E402
from wandering_light import solve  # noqa: E402
from wandering_light.stack_policy import PolicyScorer  # noqa: E402

RUN_DIR = REPO / "checkpoints" / "tempered-beam-20260923"
COHORT_DIR = REPO / "checkpoints" / "best-system-20260922"
CHECKPOINT = REPO / "checkpoints" / "world-arm-20260922" / "policies" / "1m-s0-e5"
FITTED_T = json.loads((HERE / "cells.json").read_text())["world-s0-e5"]["calibration"][
    "temperature"
]
TEMPERATURES = {"t1": 1.0, "tcal": FITTED_T}


class TemperedScorer(PolicyScorer):
    """`PolicyScorer` with its logits divided by a temperature before masking."""

    def __init__(self, model, device="cpu", *, temperature: float = 1.0):
        super().__init__(model, device)
        self.temperature = temperature

    def logits(self, blocks):
        return super().logits(blocks) / self.temperature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--cell", choices=tuple(TEMPERATURES))
    args = parser.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        print(line, flush=True)
        with open(RUN_DIR / "log.txt", "a") as handle:
            handle.write(line + "\n")

    # The #101 cohort as #158 wrote it, and #158's draw-test verdicts, so
    # programs both runs find are not drawn again.
    for name in ("cohort.jsonl.gz", "cohort_index.json", "verified.json"):
        if not (RUN_DIR / name).exists():
            shutil.copy(COHORT_DIR / name, RUN_DIR / name)
    if args.stage == "run":
        t = TEMPERATURES[args.cell]
        log(f"cell {args.cell}: T = {t}, checkpoint {CHECKPOINT}")
        bsc.CELLS[args.cell] = lambda best: best
        solve.PolicyScorer = functools.partial(TemperedScorer, temperature=t)
        bsc.run(RUN_DIR, [args.cell], CHECKPOINT, log)
    else:
        bsc.report(RUN_DIR, HERE / "tempered-beam", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
