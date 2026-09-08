#!/usr/bin/env bash
# Same protocol as reports/stack-pbe-v4-baseline-20260901 qwen3_v4_sample40: seed 17, 40-task eval sample, 300 s deadline.
# Capped at 2 h wall clock (GPU window agreed with the #80 campaign owner); SIGTERM makes the script write a partial file.
# Direct .venv python, not `uv run`: uv swallows the SIGTERM and the child writes nothing.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-local-arm-20260902
echo "RUN STARTED $(date)" | tee -a $R/qwen3_thinking2507_v4_sample40.log
OLLAMA_BASE_URL=http://localhost:11434 \
timeout -s TERM -k 60 6500 .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider ollama \
  --model qwen3:30b-a3b-thinking-2507-q4_K_M --tasks 40 --split eval \
  --out $R/qwen3_thinking2507_v4_sample40.json 2>&1 | tee -a $R/qwen3_thinking2507_v4_sample40.log
ollama stop qwen3:30b-a3b-thinking-2507-q4_K_M
echo "RUN FINISHED $(date)" | tee -a $R/qwen3_thinking2507_v4_sample40.log
