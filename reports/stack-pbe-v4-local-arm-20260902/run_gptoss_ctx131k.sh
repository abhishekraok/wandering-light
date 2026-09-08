#!/usr/bin/env bash
# gpt-oss:20b, effort high, context window raised to the model's 131,072-token limit so the 300 s deadline is the only bound.
# Otherwise the protocol of run_gptoss.sh: seed 17, 40-task v4 eval sample, 300 s deadline. No wall-clock cap (GPU free, campaign complete).
set -uo pipefail
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-local-arm-20260902
echo "RUN STARTED $(date)" | tee -a $R/gptoss20b_high_ctx131k_v4_sample40.log
OLLAMA_BASE_URL=http://localhost:11434 \
.venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider ollama \
  --model gpt-oss:20b --ollama-think --effort high --ollama-num-ctx 131072 --tasks 40 --split eval \
  --out $R/gptoss20b_high_ctx131k_v4_sample40.json 2>&1 | tee -a $R/gptoss20b_high_ctx131k_v4_sample40.log
ollama stop gpt-oss:20b
echo "RUN FINISHED $(date)" | tee -a $R/gptoss20b_high_ctx131k_v4_sample40.log
