#!/usr/bin/env bash
# Same protocol as reports/stack-pbe-v4-baseline-20260901 qwen3_v4_sample40: seed 17, 40-task eval sample, 300 s deadline.
# Plus --ollama-think: gpt-oss reasoning effort high (the --effort default), matching luna's effort setting.
# Capped to end 01:47 (GPU window agreed with the #80 campaign owner); SIGTERM makes the script write a partial file.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-local-arm-20260902
echo "RUN STARTED $(date)" | tee -a $R/gptoss20b_high_v4_sample40.log
OLLAMA_BASE_URL=http://localhost:11434 \
timeout -s TERM -k 60 5916 .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider ollama \
  --model gpt-oss:20b --ollama-think --effort high --tasks 40 --split eval \
  --out $R/gptoss20b_high_v4_sample40.json 2>&1 | tee -a $R/gptoss20b_high_v4_sample40.log
ollama stop gpt-oss:20b
echo "RUN FINISHED $(date)" | tee -a $R/gptoss20b_high_v4_sample40.log
