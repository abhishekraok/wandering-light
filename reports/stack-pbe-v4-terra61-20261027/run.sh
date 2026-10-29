#!/usr/bin/env bash
# Same fixed sample and untooled Codex protocol as sol61-sample40 (#177).
# Default efforts follow owner message 1155. Max additionally authorised by
# owner message 2210 (2026-10-27), selected explicitly with EFFORTS=max.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-.venv/bin/python}
OUT=${OUT:-reports/stack-pbe-v4-terra61-20261027}
export RUST_LOG=${RUST_LOG:-codex_api::sse::responses=trace,tungstenite::protocol=trace}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

for effort in ${EFFORTS:-low medium high xhigh}; do
  tag="terra61_${effort}_v4_sample40"
  if [ -e "$OUT/$tag.json" ] || [ -e "$OUT/$tag.log" ]; then
    echo "Refusing to overwrite an existing arm: $tag" >&2
    exit 1
  fi
  printf 'RUN STARTED %s codex %s\n' "$(date -Is)" "$(codex --version)" >> "$OUT/$tag.log"
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
    "$PYTHON" -u reports/stack-pbe-v4-terra61-20261027/capture.py \
    --backend llm --provider codex --model gpt-6.1-terra --effort "$effort" \
    --tasks 40 --split eval --seed 17 --out "$OUT/$tag.json" >> "$OUT/$tag.log" 2>&1
  status=$?
  printf 'RUN FINISHED %s (exit %s)\n' "$(date -Is)" "$status" >> "$OUT/$tag.log"
  if [ "$status" -ne 0 ]; then
    echo "INCOMPLETE: $effort arm failed" >&2
    exit "$status"
  fi
done
printf 'REQUESTED ARMS DONE %s\n' "$(date -Is)"
