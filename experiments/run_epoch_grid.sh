#!/usr/bin/env bash
# The full (arm x seed x epoch) grid on the 300-task cohort, cache off.
#
# Why it is affordable now: a cell was 941 s in the recorded campaign and is
# ~200 s here, so 54 cells is under three hours instead of fourteen. Why it is
# worth it: hindsight-subtargets (#152) compared at one epoch inherits exactly
# the confound world-plus-v4-policy (#134) was burned by -- solves track the
# epoch as much as the data -- and epoch-selection-in-behaviours (#154) needs a
# validation set bigger than one arm's 18 cells.
#
# Resumable: every cell skips when its json exists. One cell at a time, because
# wall-clock is the metric.
set -u
cd "$(dirname "$0")/.." || exit 1

RUN=checkpoints/hindsight-20260919
OUT=checkpoints/cache-20260919
LOG=$OUT/grid.log
mkdir -p "$OUT"

# Wait for the cache campaign to finish so no two cells are ever timed together.
while ! grep -q "campaign done" "$OUT/campaign.log" 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) cache campaign done; starting the epoch grid" | tee -a "$LOG"

for epoch in 4 3 5 2 6 1; do          # the epochs that matter first
  for arm in witness hindsight hindsight-all; do
    for seed in 0 1 2; do
      name="off-$arm-s$seed-e$epoch"
      policy="$RUN/policies/$arm-s$seed-e$epoch"
      [ -f "$OUT/$name.json" ] && continue
      [ -d "$policy" ] || { echo "$(date +%H:%M:%S) SKIP $name (no checkpoint)" | tee -a "$LOG"; continue; }
      echo "$(date +%H:%M:%S) START $name" | tee -a "$LOG"
      uv run python experiments/solve_with_cache.py \
        --policy "$policy" --out "$OUT/$name.json" --cache off 2>&1 | tee -a "$LOG"
    done
  done
done
echo "$(date +%H:%M:%S) grid done" | tee -a "$LOG"
