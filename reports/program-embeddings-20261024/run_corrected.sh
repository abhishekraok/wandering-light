#!/usr/bin/env bash
# Matched seed-0 recipe; outputs stay in the artifact checkout.
set -euo pipefail
asset_root=${1:-/home/abhishekrao/repos/wandering-light}
run_root=${2:-$asset_root/checkpoints/semantic-embeddings-20261031-corrected}
old_root=$asset_root/checkpoints/semantic-embeddings-20261024
mkdir -p "$run_root"
common=(
  --repo-root "$asset_root"
  --extra-tasks "$old_root/data/validated-v2.jsonl"
  --extra-tasks "$old_root/data/author-validation-v2.jsonl"
  --exclude-program-sha256 0f80c1f45e2d148534defa33e2b6efce5fddaa9dc2bf6837ce97e62fad537a43
  --seed 0 --batch-size 64 --threads 4 --device cuda
  --execution-label-workers 4
  --global-bank --max-eval-records 128
  --dim 128 --hidden-dim 128 --byte-dim 48
  --max-program-bytes 16384 --max-example-bytes 16384
  --temperature .07 --weight-decay .01
  --io-examples 5 --execution-examples 16 --io-io-weight 0
)
run_stage() {
  local stage=$1
  shift
  date -Iseconds
  printf 'Starting %s\n' "$stage"
  uv run python -m wandering_light.program_embedding_train "${common[@]}" \
    --out "$run_root/$stage" "$@" > "$run_root/$stage.log" 2>&1
  printf 'Completed %s\n' "$stage"
}
run_stage joint-s0 --steps 1500 --eval-every 300 \
  --lr .001 --rewrite-fraction .5 --io-weight 1 --invariance-weight 1
run_stage program-s0 --steps 500 --eval-every 250 \
  --lr .0002 --rewrite-fraction 1 --io-weight 0 --invariance-weight 1 \
  --checkpoint "$run_root/joint-s0/best.pt" --skip-execution
run_stage aligned-s0 --steps 500 --eval-every 250 \
  --lr .001 --rewrite-fraction .5 --io-weight 1 --invariance-weight 0 \
  --checkpoint "$run_root/program-s0/best.pt" --freeze-program-space
run_stage test-trained --eval-only --eval-split test \
  --checkpoint "$run_root/aligned-s0/best.pt"
if [[ ! -f "$run_root/historical-execution.json" ]]; then
  uv run python reports/program-embeddings-20261024/replay_execution_rankings.py \
    --repo-root "$asset_root" --input "$old_root/aligned-s0/test-trained.json" \
    --out "$run_root/historical-execution.json"
fi
uv run python reports/program-embeddings-20261024/compare_corrected.py \
  --run-root "$run_root" --repo-root "$asset_root" \
  --historical "$old_root/aligned-s0" --device cuda \
  > "$run_root/comparison.log" 2>&1
printf 'Completed corrected model comparison and index\n'
