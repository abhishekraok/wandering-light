#!/usr/bin/env bash
# Where does beam width saturate, and at what cost?
#
# The ln W budget says a width step of x4 is worth ln 4 = 1.386 nats. That was
# read as "0.33 extra depth at 4.2 nats/depth, therefore a dead zone", which was
# wrong: the sibling substrate's 4.2 nats/depth was measured on 5-8 op witnesses
# where one depth is one op. On a 16-op witness the same 1.386 nats raises the
# allowed per-step deficit from 0.260 to 0.347, a third more, and beam 256
# measured +2 non-leaky behaviours (12 -> 14) with the gain concentrated in the
# 11-20 bucket.
#
# So sweep it. Cache OFF, to separate width from reuse; then two cache-on cells
# so the cost-to-clear pair is reported at the width that wins on coverage.
set -u
cd "$(dirname "$0")/.." || exit 1

RUN=checkpoints/hindsight-20260919
OUT=checkpoints/cache-20260919
LOG=$OUT/width.log
mkdir -p "$OUT"

while ! grep -q "grid done" "$OUT/grid.log" 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) grid done; starting the width sweep" | tee -a "$LOG"

cell() {
  local name=$1; shift
  [ -f "$OUT/$name.json" ] && { echo "$(date +%H:%M:%S) skip $name" | tee -a "$LOG"; return; }
  echo "$(date +%H:%M:%S) START $name" | tee -a "$LOG"
  uv run python experiments/solve_with_cache.py \
    --policy "$RUN/policies/witness-s0-e4" --out "$OUT/$name.json" "$@" 2>&1 | tee -a "$LOG"
}

# width isolated from reuse
for w in 128 256 512; do
  cell "off-s0-e4-b$w" --cache off --beam "$w"
done
# the cost-to-clear pair at the widths that matter
cell on-s0-e4-b128 --cache on --beam 128
cell on-s0-e4-b512 --cache on --beam 512
echo "$(date +%H:%M:%S) width sweep done" | tee -a "$LOG"

# then the deficit curve, which no longer waits on the grid marker
exec bash experiments/run_floored_beam.sh
