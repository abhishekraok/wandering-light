#!/usr/bin/env bash
# Same fixed sample and untooled Claude protocol as the Haiku 5.5 report.
# Low effort only, explicitly authorized by owner message 4946 (2026-11-10).
set -uo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-.venv/bin/python}
OUT=${OUT:-reports/stack-pbe-v4-fable52-20261110}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
tag=fable52_low_v4_sample40
if [ -e "$OUT/$tag.json" ] || [ -e "$OUT/$tag.log" ]; then
  echo "Refusing to overwrite an existing arm: $tag" >&2
  exit 1
fi
printf 'RUN STARTED %s claude %s\n' "$(date -Is)" "$(claude --version)" >> "$OUT/$tag.log"
"$PYTHON" -u reports/stack-pbe-v4-fable52-20261110/capture.py \
  --backend llm --provider claude --model claude-fable-5-2 --effort low \
  --tasks 40 --split eval --seed 17 --out "$OUT/$tag.json" >> "$OUT/$tag.log" 2>&1
status=$?
printf 'RUN FINISHED %s (exit %s)\n' "$(date -Is)" "$status" >> "$OUT/$tag.log"
if [ "$status" -ne 0 ]; then
  echo 'INCOMPLETE: low arm failed' >&2
  exit "$status"
fi
printf 'REQUESTED ARM DONE %s\n' "$(date -Is)"
