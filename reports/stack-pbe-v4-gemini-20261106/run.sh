#!/usr/bin/env bash
# gemini-4-argon and gemini-3.8-flash at thinking levels low / medium / high
# (the only levels either accepts) on the seed-17 40-task eval sample of
# stack_pbe_v4, through the Gemini REST API. Same prompt, parser and held-out
# grader as ../stack-pbe-v4-effort-sweep-20260927. Each arm runs as 4 shards
# (calls take minutes); analyse.py joins them. Output goes to OUT (the main
# checkout) so it survives the worktree. Key: GEMINI_API_KEY_PUBLIC.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-$HOME/repos/wandering-light/checkpoints/gemini-sweep-20261106}
mkdir -p "$OUT"
source ~/.secrets.sh
export GOOGLE_API_KEY=$GEMINI_API_KEY_PUBLIC GEMINI_API_KEY=$GEMINI_API_KEY_PUBLIC
git rev-parse HEAD > "$OUT/commit.txt"; git status --porcelain >> "$OUT/commit.txt"

arm() {  # model effort tag shard
  local stem="$OUT/${3}_v4_sample40_shard${4}"
  echo "RUN STARTED $(date)" >> "$stem.log"
  # Grading executes generated stack programs: cap memory (#172).
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- \
    .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider gemini \
    --model "$1" --effort "$2" --tasks 40 --split eval --seed 17 --shard "$4/4" \
    --out "$stem.json" >> "$stem.log" 2>&1
  local status=$?
  echo "RUN FINISHED $(date) (exit $status)" >> "$stem.log"
  [ "$status" -eq 0 ] || echo "ARM FAILED: $1 $2 shard $4"
}
export -f arm; export OUT

{
  for m in "gemini-4-argon argon4" "gemini-3.8-flash flash38"; do
    set -- $m
    for e in low medium high; do for k in 0 1 2 3; do echo "$1 $e $2_$e $k"; done; done
  done
} | xargs -P 24 -L 1 bash -c 'arm "$@"' _
echo "ALL DONE $(date)"
