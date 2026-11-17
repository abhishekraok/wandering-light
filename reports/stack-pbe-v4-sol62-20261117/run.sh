#!/usr/bin/env bash
# One arm of the fixed seed-17 sample, with a separate low-effort preflight.
set -euo pipefail
cd "$(dirname "$0")/../.."
report=reports/stack-pbe-v4-sol62-20261117
prior=reports/stack-pbe-v4-terra61-20261027
PYTHON=${PYTHON:-.venv/bin/python}
OUT=${OUT:-$report}
effort=${1:?Specify low, medium, high or xhigh}
case "$effort" in
  low|medium|high|xhigh) ;;
  *) printf 'Unsupported effort: %s\n' "$effort" >&2; exit 2 ;;
esac
export RUST_LOG=${RUST_LOG:-codex_api::sse::responses=trace,tungstenite::protocol=trace}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset OPENAI_API_KEY CODEX_API_KEY OPENAI_BASE_URL
mkdir -p "$OUT"
tag="sol62_${effort}_v4_sample40"
preflight="$OUT/preflight-before-$effort.json"
if [[ -e "$OUT/$tag.json" || -e "$OUT/$tag.log" ]]; then
  printf 'Existing arm output: %s\n' "$tag" >&2
  exit 1
fi
if [[ ! -f "$preflight" ]]; then
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
    "$PYTHON" "$report/preflight.py" --out "$preflight"
fi
"$PYTHON" - "$preflight" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

proof = json.loads(Path(sys.argv[1]).read_text())
version = subprocess.check_output(['codex', '--version'], text=True).strip()
if (not proof['passed'] or proof['served_models'] != ['gpt-6.2-sol']
    or proof['effort'] != 'low' or proof['cli_version'] != version):
    raise SystemExit('Current Codex needs a successful low-effort serving-model check')
PY
printf 'RUN STARTED %s %s\n' "$(date -Is)" "$(codex --version)" >> "$OUT/$tag.log"
status=0
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  "$PYTHON" -u "$prior/capture.py" \
  --backend llm --provider codex --model gpt-6.2-sol --effort "$effort" \
  --tasks 40 --split eval --seed 17 --out "$OUT/$tag.json" >> "$OUT/$tag.log" 2>&1 || status=$?
printf 'RUN FINISHED %s (exit %s)\n' "$(date -Is)" "$status" >> "$OUT/$tag.log"
exit "$status"
