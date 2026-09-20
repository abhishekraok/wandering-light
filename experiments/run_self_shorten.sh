#!/usr/bin/env bash
# The README's loop, aimed at the quantity that binds.
#
# 37.5% of verified solves on disk are strictly shorter than the authored
# witness (44% in the 1-5 bucket saving a median 2 ops, 51% in 11-20 saving 1),
# and nothing has ever fed them back. Shortening L raises the per-step budget
# ln W / L for every step at once: a 5-op witness solved in 3 is a 67% larger
# allowance, where a x4 beam is ln 4 = 1.386 nats in total.
#
# Harvest on the train split only, accept a program only when it matches the
# authored witness on every shown input of every train task of that behaviour,
# retrain on the shortened labels, and search the cohort.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=checkpoints/self-shorten-20260920
CELLS=checkpoints/cache-20260919
RUN=checkpoints/hindsight-20260919
LOG=$OUT/run.log
mkdir -p "$OUT"

while ! grep -q "floored beam done" "$CELLS/floored.log" 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) floored beam done; starting self-shorten" | tee -a "$LOG"

uv run python experiments/self_shorten.py harvest \
  --policy "$RUN/policies/witness-s0-e4" --out "$OUT/harvest.json" --tasks 1200 2>&1 | tee -a "$LOG"

uv run python experiments/self_shorten.py train \
  --harvest "$OUT/harvest.json" --run-dir "$OUT" --seeds 0,1,2 2>&1 | tee -a "$LOG"

for seed in 0 1 2; do
  for epoch in 4 3 5; do
    name="off-shortened-s$seed-e$epoch"
    [ -f "$CELLS/$name.json" ] && continue
    [ -d "$OUT/policies/shortened-s$seed-e$epoch" ] || continue
    echo "$(date +%H:%M:%S) START $name" | tee -a "$LOG"
    uv run python experiments/solve_with_cache.py \
      --policy "$OUT/policies/shortened-s$seed-e$epoch" \
      --out "$CELLS/$name.json" --cache off 2>&1 | tee -a "$LOG"
  done
done
echo "$(date +%H:%M:%S) self-shorten done" | tee -a "$LOG"
exec bash experiments/run_analysis.sh --now
