#!/usr/bin/env bash
# Everything that turns the night's cells into numbers. CPU only, runs last so
# it never overlaps a timing cell.
set -u
cd "$(dirname "$0")/.." || exit 1
CELLS=checkpoints/cache-20260919
RUN=checkpoints/hindsight-20260919
REP=reports/solved-program-cache-20260919
PROXY=$CELLS/proxy
LOG=$REP/analysis.log
mkdir -p "$REP" "$PROXY"

if [ "${1:-}" != "--now" ]; then
  while ! grep -q "deficit curve training done" checkpoints/deficit-curve-20260919/train.log 2>/dev/null; do sleep 60; done
fi
echo "$(date +%H:%M:%S) starting analysis" | tee -a "$LOG"

# 1. the cache over 100 exact task orders
uv run python experiments/cache_order_sim.py \
  --cell "$CELLS/off-s0-e4.json" --out "$REP/order_sim.json" --orders 100 2>&1 | tee -a "$LOG"

# 2. the search proxy on every witness checkpoint with a known cohort outcome
for seed in 0 1 2; do
  for epoch in 1 2 3 4 5 6; do
    p="$RUN/policies/witness-s$seed-e$epoch"
    [ -d "$p" ] || continue
    uv run python experiments/search_proxy.py score \
      --policy "$p" --out "$PROXY/witness-s$seed-e$epoch.json" --device cuda 2>&1 | tee -a "$LOG"
  done
done
uv run python experiments/search_proxy.py correlate \
  --proxy "$PROXY/witness-s*.json" --out "$REP/proxy_correlation.json" 2>&1 | tee -a "$LOG"

# 3. the budget law, per behaviour, on the cohort (analysis only -- sees eval)
uv run python experiments/search_proxy.py score \
  --policy "$RUN/policies/witness-s0-e4" --out "$PROXY/cohort-s0-e4.json" \
  --sets cohort --device cuda 2>&1 | tee -a "$LOG"
uv run python experiments/deficit_law.py \
  --proxy "$PROXY/cohort-s0-e4.json" --cell "$CELLS/off-s0-e4.json" \
  --out "$REP/deficit_law.json" 2>&1 | tee -a "$LOG"

# 4. the report
uv run python experiments/cache_report.py \
  --run-dir "$CELLS" --sim "$REP/order_sim.json" --out "$REP" 2>&1 | tee -a "$LOG"
echo "$(date +%H:%M:%S) analysis done" | tee -a "$LOG"

# The deficit curve is training, not timing, so it runs last: if the night is
# cut short the report survives and only the curve is lost.
exec bash experiments/run_deficit_curve_after_width.sh
