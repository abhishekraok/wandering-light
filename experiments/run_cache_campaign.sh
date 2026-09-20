#!/usr/bin/env bash
# Timing cells for the solved-program cache and the hindsight arms.
#
# Every cell is a wall-clock measurement, so they run ONE AT A TIME and nothing
# else heavy may run beside them. The script waits for the training PID first.
set -u
cd "$(dirname "$0")/.." || exit 1

RUN=checkpoints/hindsight-20260919
OUT=checkpoints/cache-20260919
LOG=$OUT/campaign.log
mkdir -p "$OUT"

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "$(date +%H:%M:%S) waiting for training pid $WAIT_PID" | tee -a "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "$(date +%H:%M:%S) training finished" | tee -a "$LOG"
fi

cell() {  # cell <name> <policy> <extra args...>
  local name=$1; shift
  local policy=$1; shift
  if [ ! -d "$policy" ]; then
    echo "$(date +%H:%M:%S) SKIP $name (no checkpoint $policy)" | tee -a "$LOG"
    return
  fi
  echo "$(date +%H:%M:%S) START $name" | tee -a "$LOG"
  uv run python experiments/solve_with_cache.py \
    --policy "$policy" --out "$OUT/$name.json" "$@" 2>&1 | tee -a "$LOG"
}

# --- the headline A/B, on the checkpoint that reproduces #134's 94/300 -------
cell off-s0-e4            "$RUN/policies/witness-s0-e4" --cache off
cell on-s0-e4             "$RUN/policies/witness-s0-e4" --cache on
cell on-s0-e4-order1      "$RUN/policies/witness-s0-e4" --cache on --order-seed 1

# --- Control: replay only programs found on a DIFFERENT behaviour. Separates --
# --- cross-behaviour reuse from the within-behaviour recurrence the cohort's --
# --- 43-behaviours-behind-300-tasks construction supplies. Needs behaviour ----
# --- labels, so it is a diagnostic and never a solver. -----------------------
cell disjoint-s0-e4       "$RUN/policies/witness-s0-e4" --cache disjoint

# --- the three training arms at a matched epoch, cache off (#152) -------------
for arm in witness hindsight hindsight-all; do
  for seed in 0 1 2; do
    [ "$arm-$seed" = "witness-0" ] && continue   # off-s0-e4 above is this cell
    cell "off-$arm-s$seed-e4" "$RUN/policies/$arm-s$seed-e4" --cache off
  done
done

# --- the rest of the search-shape sweep --------------------------------------
cell on-s0-e4-b256        "$RUN/policies/witness-s0-e4" --cache on --beam 256
cell on-s0-e4-d20         "$RUN/policies/witness-s0-e4" --cache on --depth 20
cell on-s0-e4-order2      "$RUN/policies/witness-s0-e4" --cache on --order-seed 2
cell off-s0-e4-order1     "$RUN/policies/witness-s0-e4" --cache off --order-seed 1

echo "$(date +%H:%M:%S) campaign done" | tee -a "$LOG"
