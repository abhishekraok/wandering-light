#!/usr/bin/env bash
# floored-path-score: charge each step at most tau nats in the ranking key.
#
# An adversarial review refuted this as "~0.30 nats per step, below a width step
# already recorded as buying nothing". Beam 256 has since measured +2 non-leaky
# behaviours, and a x4 width step is ln 4 = 1.386 nats IN TOTAL, while 0.30 nats
# PER STEP over a 16-op witness is 4.8 nats in total. The refutation compared a
# per-step quantity against a total one, so the cell is worth running.
#
# tau = inf is beam_solve exactly and is the equivalence check: it must
# reproduce off-s0-e4's 94 tasks and 71,700,695 executions.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=checkpoints/cache-20260919
LOG=$OUT/floored.log
RUN=checkpoints/hindsight-20260919
mkdir -p "$OUT"

while ! grep -q "width sweep done" "$OUT/width.log" 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) width sweep done; starting floored-path-score" | tee -a "$LOG"

for tau in inf 3.0 1.5 0.7; do
  name="floor-tau$tau-s0-e4"
  [ -f "$OUT/$name.json" ] && continue
  echo "$(date +%H:%M:%S) START $name" | tee -a "$LOG"
  uv run python experiments/floored_beam.py \
    --policy "$RUN/policies/witness-s0-e4" --tau "$tau" --out "$OUT/$name.json" 2>&1 | tee -a "$LOG"
done
echo "$(date +%H:%M:%S) floored beam done" | tee -a "$LOG"
exec bash experiments/run_deficit_curve_after_width.sh
