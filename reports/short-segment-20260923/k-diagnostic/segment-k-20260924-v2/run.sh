#!/usr/bin/env bash
# K diagnostic with the variant columns (#168), per-arm depth. Code: claude/campaign-code 6bd7036.
cd /home/abhishekrao/repos/wandering-light/.claude/worktrees/campaign-code
k() { uv run python experiments/segment_k_diagnostic.py --checkpoint $1 --set $2 --segment-depth $3 --limit 40 --device cuda --out /home/abhishekrao/repos/wandering-light/checkpoints/segment-k-20260924-v2/$4 >> /home/abhishekrao/repos/wandering-light/checkpoints/segment-k-20260924-v2/$4-$2.out 2>&1; }
(k best staged-train 16 best; k best yardstick-168 16 best) &
(k /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-difference-20260923/policies/capped-s0-e3 capped-train 8 capped-difference; k /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-difference-20260923/policies/capped-s0-e3 yardstick-168 8 capped-difference) &
wait
touch /home/abhishekrao/repos/wandering-light/checkpoints/segment-k-20260924-v2/DONE
